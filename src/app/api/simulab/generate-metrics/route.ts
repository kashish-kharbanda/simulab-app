import { NextRequest, NextResponse } from "next/server"
import { getScenariosByProteinTarget, findScenarioBySmiles, findScenarioByScaffold, SheetScenario } from "../sheets/data"
import { evaluateMolecules, isDevMode, isAgentexConfigured, AGENTS } from "@/lib/agent-client"

/**
 * SimuLab Generate Metrics API Route (Simulator Agent)
 * 
 * FULL INTEGRATION: Calls the deployed Simulator agent via SGP/Agentex API.
 * Falls back to local LLM calls if agent is unavailable.
 * 
 * Flow:
 * 1. If Agentex configured: Call deployed Simulator agent's /evaluate_molecules endpoint
 * 2. Agent does LLM + database cross-check with tracing
 * 3. Returns metrics with confidence level
 * 
 * Fallback (if agent unavailable):
 * 1. Call LLM directly (no tracing)
 * 2. Cross-check with local database
 */

export const maxDuration = 240; // 4 minutes for agent processing
export const dynamic = "force-dynamic"

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || process.env.NEXT_PUBLIC_OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";

interface Scenario {
  scenario_id: string;
  scaffold?: string;
  smiles?: string;
}

interface GeneratedMetrics {
  docking: {
    binding_affinity_kcal_per_mol: number;
    potency_pass: boolean;
  };
  admet: {
    toxicity_risk: "LOW" | "MED" | "HIGH";
    toxicity_prob: number;
    herg_flag: boolean;
    is_safe: boolean;
  };
  synthesis: {
    sa_score: number;
    num_steps: number;
    estimated_cost_usd: number;
  };
}

interface ScenarioResult {
  scenario_id: string;
  smiles: string;
  scaffold: string;
  metrics: GeneratedMetrics;
  is_winner: boolean;
  rejection_reason?: string;
  data_source: "agent" | "llm" | "llm_validated";
  confidence: "high" | "medium";
}

interface GenerateMetricsRequest {
  scenarios: Scenario[];
  protein_target: string;
  goal?: string;
  constraints?: string[];
  decision_criteria?: {
    docking?: { idealMin?: number; idealMax?: number; hardFailThreshold?: number };
    admet?: { idealMin?: number; idealMax?: number; hardFailHERG?: boolean };
    synthesis?: { idealSaMax?: number; idealStepsMax?: number; hardFailSa?: number; hardFailSteps?: number };
  };
}

/**
 * Call LLM to calculate metrics for a single molecule - FALLBACK only
 */
async function calculateMetricsWithLLM(
  smiles: string, 
  scaffold: string, 
  proteinTarget: string,
  decisionCriteria: Record<string, unknown>
): Promise<GeneratedMetrics> {
  if (!OPENAI_API_KEY) {
    throw new Error("OpenAI API key not configured - LLM is required for fallback");
  }

  const systemPrompt = `You are a PhD-level computational chemist with 20+ years of experience in molecular docking, ADMET prediction, and synthetic chemistry.

You are the Simulator agent for a multi-agent virtual drug discovery lab. Your job is to evaluate molecular candidates and predict their properties.

TASK: Given a molecule (SMILES) and target protein, calculate:

1. BINDING AFFINITY (ΔG in kcal/mol):
   - Analyze molecular features for target binding
   - Typical range: -6 to -12 kcal/mol (more negative = stronger binding)
   - Potency threshold: < -7 kcal/mol passes

2. ADMET PROFILE:
   - Toxicity risk: LOW/MED/HIGH based on structural alerts
   - hERG flag: true if cardiac toxicity risk
   - is_safe: true only if LOW toxicity AND no hERG flag

3. SYNTHESIS FEASIBILITY:
   - SA Score (1-10): synthetic accessibility (lower = easier)
   - num_steps: estimated synthesis steps (2-12)
   - estimated_cost_usd: synthesis cost for 1g ($500-5000)

Return ONLY valid JSON:
{
  "binding_affinity_kcal_per_mol": -9.5,
  "potency_pass": true,
  "toxicity_risk": "LOW",
  "toxicity_prob": 0.15,
  "herg_flag": false,
  "is_safe": true,
  "sa_score": 3.5,
  "num_steps": 4,
  "estimated_cost_usd": 1200
}`;

  const dockingCriteria = decisionCriteria?.docking as Record<string, unknown> | undefined;
  const admetCriteria = decisionCriteria?.admet as Record<string, unknown> | undefined;
  const synthesisCriteria = decisionCriteria?.synthesis as Record<string, unknown> | undefined;

  const userPrompt = `Evaluate this molecule:

SMILES: ${smiles}
SCAFFOLD CLASS: ${scaffold}
TARGET PROTEIN: ${proteinTarget}

Decision Criteria:
- Potency threshold: ΔG < ${dockingCriteria?.hardFailThreshold || -7} kcal/mol
- hERG veto: ${admetCriteria?.hardFailHERG !== false ? "Yes" : "No"}
- SA hard fail: > ${synthesisCriteria?.hardFailSa || 6}

Calculate all metrics based on the molecular structure. Be scientifically rigorous.`;

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.2,
      max_tokens: 500,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`LLM call failed (${response.status}): ${error}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content || "";

  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    throw new Error("LLM did not return valid JSON");
  }

  const parsed = JSON.parse(jsonMatch[0]);

  return {
    docking: {
      binding_affinity_kcal_per_mol: parsed.binding_affinity_kcal_per_mol || -7.0,
      potency_pass: parsed.potency_pass ?? (parsed.binding_affinity_kcal_per_mol < -7),
    },
    admet: {
      toxicity_risk: parsed.toxicity_risk || "MED",
      toxicity_prob: parsed.toxicity_prob || 0.3,
      herg_flag: parsed.herg_flag || false,
      is_safe: parsed.is_safe ?? (!parsed.herg_flag && parsed.toxicity_risk === "LOW"),
    },
    synthesis: {
      sa_score: parsed.sa_score || 4.0,
      num_steps: parsed.num_steps || 5,
      estimated_cost_usd: parsed.estimated_cost_usd || 1500,
    },
  };
}

/**
 * Cross-check LLM metrics with database and override if match found
 */
function crossCheckWithDatabase(
  scenario: Scenario,
  llmMetrics: GeneratedMetrics,
  decisionCriteria: Record<string, unknown>
): { metrics: GeneratedMetrics; wasOverridden: boolean; dbMatch: SheetScenario | null } {
  
  let dbMatch: SheetScenario | null = null;
  
  if (scenario.smiles) {
    dbMatch = findScenarioBySmiles(scenario.smiles);
  }
  if (!dbMatch && scenario.scaffold) {
    dbMatch = findScenarioByScaffold(scenario.scaffold);
  }

  if (!dbMatch) {
    return { metrics: llmMetrics, wasOverridden: false, dbMatch: null };
  }

  console.log(`[Simulator] ✓ Database match found for ${scenario.scenario_id}`);

  const dockingCriteria = decisionCriteria?.docking as Record<string, unknown> | undefined;
  const hardFailThreshold = (dockingCriteria?.hardFailThreshold as number) ?? -7;
  const bindingAffinity = dbMatch.reference_binding_affinity ?? llmMetrics.docking.binding_affinity_kcal_per_mol;
  const hergFlag = dbMatch.reference_herg_flag ?? llmMetrics.admet.herg_flag;
  const saScore = dbMatch.reference_sa_score ?? llmMetrics.synthesis.sa_score;

  const overriddenMetrics: GeneratedMetrics = {
    docking: {
      binding_affinity_kcal_per_mol: bindingAffinity,
      potency_pass: bindingAffinity < hardFailThreshold,
    },
    admet: {
      toxicity_risk: hergFlag ? "HIGH" : "LOW",
      toxicity_prob: hergFlag ? 0.8 : 0.1,
      herg_flag: hergFlag,
      is_safe: !hergFlag,
    },
    synthesis: {
      sa_score: saScore,
      num_steps: Math.floor(saScore) + 2,
      estimated_cost_usd: Math.floor(500 + saScore * 300),
    },
  };

  return { metrics: overriddenMetrics, wasOverridden: true, dbMatch };
}

/**
 * Determine winner status based on metrics and decision criteria
 */
function determineWinnerStatus(
  metrics: GeneratedMetrics,
  decisionCriteria: Record<string, unknown>,
  dbMatch: SheetScenario | null
): { isWinner: boolean; rejectionReason?: string } {
  
  if (dbMatch?.target_result) {
    const targetResult = dbMatch.target_result.toUpperCase();
    const isWinner = targetResult === "WINNER";
    
    if (!isWinner) {
      return {
        isWinner: false,
        rejectionReason: `${dbMatch.result_category || "Rejected"}`,
      };
    }
    return { isWinner: true };
  }

  const dockingCriteria = decisionCriteria?.docking as Record<string, unknown> | undefined;
  const synthesisCriteria = decisionCriteria?.synthesis as Record<string, unknown> | undefined;
  const admetCriteria = decisionCriteria?.admet as Record<string, unknown> | undefined;
  
  const hardFailThreshold = (dockingCriteria?.hardFailThreshold as number) ?? -7;
  const hardFailSa = (synthesisCriteria?.hardFailSa as number) ?? 6;
  const hardFailHERG = admetCriteria?.hardFailHERG !== false;

  const reasons: string[] = [];

  if (!metrics.docking.potency_pass) {
    reasons.push(`Weak binding (ΔG ${metrics.docking.binding_affinity_kcal_per_mol} > ${hardFailThreshold} kcal/mol)`);
  }
  if (hardFailHERG && metrics.admet.herg_flag) {
    reasons.push("hERG cardiac toxicity flag");
  }
  if (metrics.synthesis.sa_score > hardFailSa) {
    reasons.push(`Poor synthetic accessibility (SA ${metrics.synthesis.sa_score} > ${hardFailSa})`);
  }

  if (reasons.length > 0) {
    return { isWinner: false, rejectionReason: reasons.join("; ") };
  }

  return { isWinner: true };
}

export async function POST(request: NextRequest) {
  try {
    const body: GenerateMetricsRequest = await request.json();
    const { 
      scenarios = [], 
      protein_target = "Unknown",
      decision_criteria = {}
    } = body;

    console.log(`[Simulator] ========================================`);
    console.log(`[Simulator] Evaluating ${scenarios.length} scenarios for ${protein_target}`);
    console.log(`[Simulator] Mode: ${isDevMode() ? 'DEV (local agent)' : 'PROD (deployed agent)'}`);

    if (!scenarios.length) {
      return NextResponse.json({ error: "No scenarios provided" }, { status: 400 });
    }

    const experimentId = `exp-${Date.now()}`;

    // =========================================================================
    // TRY DEPLOYED AGENT FIRST (Full Integration)
    // =========================================================================
    if (isDevMode() || isAgentexConfigured()) {
      console.log(`[Simulator] Calling ${AGENTS.SIMULATOR} agent...`);
      
      const agentResult = await evaluateMolecules({
        experiment_id: experimentId,
        protein_target,
        scenarios: scenarios.map(s => ({
          scenario_id: s.scenario_id,
          smiles: s.smiles || "",
          scaffold: s.scaffold || "Unknown",
        })),
      });

      if (agentResult.success && agentResult.data) {
        console.log(`[Simulator] Agent returned ${agentResult.data.results?.length || 0} results`);
        
        // Map agent response to our format
        const results: ScenarioResult[] = (agentResult.data.results || []).map(r => ({
          scenario_id: r.scenario_id,
          smiles: r.smiles,
          scaffold: r.scaffold || "Unknown",
          metrics: {
            docking: {
              binding_affinity_kcal_per_mol: r.docking.binding_affinity_kcal_per_mol,
              potency_pass: r.docking.potency_pass,
            },
            admet: {
              toxicity_risk: (r.admet.toxicity_risk as "LOW" | "MED" | "HIGH") || "MED",
              toxicity_prob: r.admet.herg_flag ? 0.8 : 0.2,
              herg_flag: r.admet.herg_flag,
              is_safe: r.admet.is_safe,
            },
            synthesis: {
              sa_score: r.synthesis.sa_score,
              num_steps: Math.floor(r.synthesis.sa_score) + 2,
              estimated_cost_usd: r.synthesis.estimated_cost_usd,
            },
          },
          is_winner: false, // Will be determined by Judge
          data_source: "agent" as const,
          confidence: (r.confidence === "HIGH" ? "high" : "medium") as "high" | "medium",
        }));

        console.log(`[Simulator] ========================================`);
        
        return NextResponse.json({
          results,
          source: "agent",
          confidence: "high",
          _via: "deployed_agent",
        });
      }

      console.log(`[Simulator] Agent call failed: ${agentResult.error}, falling back to local LLM`);
    } else {
      console.log(`[Simulator] Agentex not configured, using local LLM fallback`);
    }

    // =========================================================================
    // FALLBACK: Local LLM processing (no tracing)
    // =========================================================================
    const results: ScenarioResult[] = [];

    for (const scenario of scenarios) {
      console.log(`[Simulator] Processing ${scenario.scenario_id}: ${scenario.scaffold}`);

      let llmMetrics: GeneratedMetrics;
      try {
        llmMetrics = await calculateMetricsWithLLM(
          scenario.smiles || "",
          scenario.scaffold || "",
          protein_target,
          decision_criteria
        );
        console.log(`[Simulator] LLM calculated: ΔG=${llmMetrics.docking.binding_affinity_kcal_per_mol}, hERG=${llmMetrics.admet.herg_flag}, SA=${llmMetrics.synthesis.sa_score}`);
      } catch (llmError) {
        console.error(`[Simulator] LLM error for ${scenario.scenario_id}:`, llmError);
        return NextResponse.json({
          error: `Simulator LLM failed for ${scenario.scenario_id}: ${llmError instanceof Error ? llmError.message : "Unknown"}`,
        }, { status: 500 });
      }

      const { metrics, wasOverridden, dbMatch } = crossCheckWithDatabase(
        scenario,
        llmMetrics,
        decision_criteria
      );

      if (wasOverridden) {
        console.log(`[Simulator] Overridden with DB: ΔG=${metrics.docking.binding_affinity_kcal_per_mol}, hERG=${metrics.admet.herg_flag}, SA=${metrics.synthesis.sa_score}`);
      }

      const { isWinner, rejectionReason } = determineWinnerStatus(metrics, decision_criteria, dbMatch);

      results.push({
        scenario_id: scenario.scenario_id,
        smiles: scenario.smiles || "",
        scaffold: scenario.scaffold || "Unknown",
        metrics,
        is_winner: isWinner,
        rejection_reason: rejectionReason,
        data_source: wasOverridden ? "llm_validated" : "llm",
        confidence: wasOverridden ? "high" : "medium",
      });
    }

    const validatedCount = results.filter(r => r.data_source === "llm_validated").length;
    console.log(`[Simulator] Complete: ${results.length} scenarios, ${validatedCount} validated against database`);
    console.log(`[Simulator] ========================================`);

    return NextResponse.json({
      results,
      source: validatedCount > 0 ? "llm_validated" : "llm",
      confidence: validatedCount === results.length ? "high" : "medium",
      _via: "local_llm_fallback",
    });

  } catch (error: unknown) {
    console.error("[Simulator] Unexpected error:", error);
    return NextResponse.json({
      error: `Simulator failed: ${error instanceof Error ? error.message : "Unknown"}`,
    }, { status: 500 });
  }
}
