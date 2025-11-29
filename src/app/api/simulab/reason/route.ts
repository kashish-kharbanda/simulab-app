import { NextRequest, NextResponse } from "next/server"
import { getScenariosByProteinTarget, findScenarioBySmiles, findScenarioByScaffold, SheetScenario } from "../sheets/data"
import { generateVerdict, isDevMode, isAgentexConfigured, AGENTS, traceReportFeedback } from "@/lib/agent-client"

/**
 * SimuLab Reason API Route (Judge Agent)
 * 
 * FULL INTEGRATION: Calls the deployed Judge agent via SGP/Agentex API.
 * Falls back to local LLM calls if agent is unavailable.
 * 
 * Flow:
 * 1. If Agentex configured: Call deployed Judge agent's /generate_verdict endpoint
 * 2. Agent does multi-objective analysis with tracing
 * 3. Returns verdict with confidence level
 * 
 * Fallback (if agent unavailable):
 * 1. Call LLM directly (limited tracing)
 * 2. Cross-check with local database
 */

export const maxDuration = 240; // 4 minutes for agent processing
export const dynamic = "force-dynamic"

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || process.env.NEXT_PUBLIC_OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";

interface ScenarioMetrics {
  docking?: {
    binding_affinity_kcal_per_mol?: number;
    potency_pass?: boolean;
  };
  admet?: {
    toxicity_risk?: string;
    herg_flag?: boolean;
    is_safe?: boolean;
  };
  synthesis?: {
    sa_score?: number;
    estimated_cost_usd?: number;
    num_steps?: number;
  };
}

interface ReasonRequest {
  scenarios?: Array<{ 
    scenario_id: string; 
    smiles?: string; 
    scaffold?: string;
  }>;
  scenarioMetrics?: Record<string, ScenarioMetrics>;
  context?: {
    protein_target?: string;
    goal?: string;
    constraints?: string[];
  };
  decisionCriteria?: {
    docking?: { idealMin?: number; idealMax?: number; hardFailThreshold?: number };
    admet?: { idealMin?: number; idealMax?: number; hardFailHERG?: boolean };
    synthesis?: { idealSaMax?: number; idealStepsMax?: number; hardFailSa?: number; hardFailSteps?: number };
  };
}

/**
 * Call LLM to generate verdict and rationale - FALLBACK only
 */
async function generateVerdictWithLLM(
  scenarios: Array<{ scenario_id: string; scaffold?: string; smiles?: string }>,
  scenarioMetrics: Record<string, ScenarioMetrics>,
  context: { protein_target?: string; goal?: string },
  decisionCriteria: Record<string, unknown>
): Promise<Record<string, unknown>> {
  if (!OPENAI_API_KEY) {
    throw new Error("OpenAI API key not configured - LLM is required for fallback");
  }

  const dockingCriteria = decisionCriteria?.docking as Record<string, unknown> | undefined;
  const admetCriteria = decisionCriteria?.admet as Record<string, unknown> | undefined;
  const synthesisCriteria = decisionCriteria?.synthesis as Record<string, unknown> | undefined;

  const criteriaText = [
    `Docking: ideal ΔG ${dockingCriteria?.idealMin || -12} to ${dockingCriteria?.idealMax || -8} kcal/mol; hard fail if > ${dockingCriteria?.hardFailThreshold || -7} kcal/mol`,
    `ADMET: ${admetCriteria?.hardFailHERG !== false ? "hERG flag triggers veto" : "hERG informational"}`,
    `Synthesis: ideal SA ≤ ${synthesisCriteria?.idealSaMax || 4}; hard fail if SA > ${synthesisCriteria?.hardFailSa || 6}`,
  ].join("\n");

  const systemPrompt = `You are a PhD-level medicinal chemist with 20+ years of experience in drug discovery.

You are the Judge agent for a multi-agent virtual drug discovery lab. Categorize molecules into:
1. **WINNER** - Best molecule from selected pool (or null if none pass)
2. **SELECTED** - Pass all criteria but aren't winner
3. **REJECTED** - Fail one or more criteria

DECISION CRITERIA:
${criteriaText}

Return ONLY valid JSON:
{
  "executive_summary": "2-3 sentence overview",
  "winner": { "scenario_id": "...", "scaffold": "...", "binding_affinity": -9.5, "herg_flag": false, "sa_score": 3.5, "cost_usd": 1200, "rationale": "..." } or null,
  "selected": [...],
  "rejected": [...],
  "comparative_analysis": "Detailed comparison with specific values",
  "recommendation": "Next steps"
}`;

  const scenarioDetails = scenarios.map(s => {
    const metrics = scenarioMetrics[s.scenario_id] || {};
    return {
      scenario_id: s.scenario_id,
      scaffold: s.scaffold,
      smiles: s.smiles,
      binding_affinity: metrics.docking?.binding_affinity_kcal_per_mol,
      herg_flag: metrics.admet?.herg_flag,
      sa_score: metrics.synthesis?.sa_score,
      estimated_cost_usd: metrics.synthesis?.estimated_cost_usd,
    };
  });

  const userPrompt = `Analyze these candidates for ${context?.protein_target || "Unknown"}:

${JSON.stringify(scenarioDetails, null, 2)}

Apply decision criteria strictly.`;

  console.log(`[Judge] Calling LLM (${OPENAI_MODEL}) for verdict...`);

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
      max_tokens: 2500,
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

  return JSON.parse(jsonMatch[0]);
}

/**
 * Cross-check LLM verdict with database - respects user's decision criteria
 */
function crossCheckVerdictWithDatabase(
  llmVerdict: Record<string, unknown>,
  scenarios: Array<{ scenario_id: string; smiles?: string; scaffold?: string }>,
  proteinTarget: string,
  decisionCriteria?: Record<string, unknown>
): { verdict: Record<string, unknown>; wasOverridden: boolean; corrections: string[] } {
  
  const corrections: string[] = [];
  let wasOverridden = false;

  llmVerdict.selected = llmVerdict.selected || [];

  const dbScenarios = getScenariosByProteinTarget(proteinTarget);
  
  if (dbScenarios.length === 0) {
    console.log(`[Judge] No database validation available for ${proteinTarget}`);
    return { verdict: llmVerdict, wasOverridden: false, corrections: [] };
  }

  console.log(`[Judge] Cross-checking against ${dbScenarios.length} database entries`);

  const dbDataMap = new Map<string, SheetScenario>();
  
  for (const scenario of scenarios) {
    let dbMatch: SheetScenario | null = null;
    
    if (scenario.smiles) {
      dbMatch = findScenarioBySmiles(scenario.smiles);
    }
    if (!dbMatch && scenario.scaffold) {
      dbMatch = findScenarioByScaffold(scenario.scaffold);
    }
    
    if (dbMatch) {
      dbDataMap.set(scenario.scenario_id, dbMatch);
    }
  }

  if (dbDataMap.size === 0) {
    return { verdict: llmVerdict, wasOverridden: false, corrections: [] };
  }

  const admetCriteria = decisionCriteria?.admet as Record<string, unknown> | undefined;
  const dockingCriteria = decisionCriteria?.docking as Record<string, unknown> | undefined;
  const synthesisCriteria = decisionCriteria?.synthesis as Record<string, unknown> | undefined;

  const hergVetoEnabled = admetCriteria?.hardFailHERG !== false;
  const potencyThreshold = (dockingCriteria?.hardFailThreshold as number) ?? -7;
  const saThreshold = (synthesisCriteria?.hardFailSa as number) ?? 6;

  const newRejected: Array<Record<string, unknown>> = [];
  const passing: Array<Record<string, unknown>> = [];

  for (const [scenarioId, db] of dbDataMap) {
    const bindingAffinity = db.reference_binding_affinity ?? -7;
    const hergFlag = db.reference_herg_flag ?? false;
    const saScore = db.reference_sa_score ?? 4;

    let isRejected = false;
    let rejectionReason = "";

    if (bindingAffinity > potencyThreshold) {
      isRejected = true;
      rejectionReason = `Potency Fail (ΔG ${bindingAffinity} > ${potencyThreshold} kcal/mol)`;
    }

    if (!isRejected && hergVetoEnabled && hergFlag) {
      isRejected = true;
      rejectionReason = `Safety Veto (hERG cardiac toxicity flag)`;
    }

    if (!isRejected && saScore > saThreshold) {
      isRejected = true;
      rejectionReason = `Cost Veto (SA Score ${saScore} > ${saThreshold})`;
    }

    const scenarioData = {
      scenario_id: scenarioId,
      scaffold: db.scaffold_hypothesis,
      smiles: db.smiles,
      binding_affinity: bindingAffinity,
      toxicity_risk: hergFlag ? "HIGH" : "LOW",
      herg_flag: hergFlag,
      sa_score: saScore,
      cost_usd: Math.floor(500 + saScore * 300),
    };

    if (isRejected) {
      newRejected.push({ ...scenarioData, rejection_reason: rejectionReason });
      console.log(`[Judge] ${scenarioId}: REJECTED - ${rejectionReason}`);
    } else {
      passing.push(scenarioData);
      console.log(`[Judge] ${scenarioId}: PASSES all criteria`);
    }
  }

  let finalWinner: Record<string, unknown> | null = null;
  const newSelected: Array<Record<string, unknown>> = [];
  
  if (passing.length > 0) {
    passing.sort((a, b) => (a.binding_affinity as number) - (b.binding_affinity as number));
    finalWinner = {
      ...passing[0],
      rationale: `Best binding affinity (ΔG ${passing[0].binding_affinity} kcal/mol) among passing candidates.`,
    };
    
    for (let i = 1; i < passing.length; i++) {
      newSelected.push({
        ...passing[i],
        selection_reason: `Passes all criteria. Viable backup candidate.`,
      });
    }
  }

  const originalWinner = llmVerdict.winner as Record<string, unknown> | null;
  if (originalWinner?.scenario_id !== finalWinner?.scenario_id) {
    corrections.push(`Winner changed based on user criteria`);
    wasOverridden = true;
  }

  llmVerdict.winner = finalWinner;
  llmVerdict.selected = newSelected;
  llmVerdict.rejected = newRejected;

  if (wasOverridden) {
    llmVerdict.executive_summary = `Re-evaluated ${dbDataMap.size} candidates. ${passing.length} passed, ${newRejected.length} rejected.${finalWinner ? ` ${finalWinner.scenario_id} selected as winner.` : ""}`;
  }

  return { verdict: llmVerdict, wasOverridden, corrections };
}

export async function POST(request: NextRequest) {
  try {
    const body: ReasonRequest = await request.json();
    const { 
      scenarios = [], 
      scenarioMetrics = {}, 
      context = {}, 
      decisionCriteria = {} 
    } = body;

    const proteinTarget = context?.protein_target || "Unknown";
    const experimentId = `judge-${Date.now()}`;

    console.log(`[Judge] ========================================`);
    console.log(`[Judge] Analyzing ${scenarios.length} scenarios for ${proteinTarget}`);
    console.log(`[Judge] Mode: ${isDevMode() ? 'DEV (local agent)' : 'PROD (deployed agent)'}`);

    if (!scenarios.length) {
      return NextResponse.json({ error: "No scenarios to judge" }, { status: 400 });
    }

    // =========================================================================
    // TRY DEPLOYED AGENT FIRST (Full Integration)
    // =========================================================================
    if (isDevMode() || isAgentexConfigured()) {
      console.log(`[Judge] Calling ${AGENTS.JUDGE} agent...`);
      
      const admetCriteria = decisionCriteria?.admet as Record<string, unknown> | undefined;
      const dockingCriteria = decisionCriteria?.docking as Record<string, unknown> | undefined;
      const synthesisCriteria = decisionCriteria?.synthesis as Record<string, unknown> | undefined;

      // Build scenarios with metrics for the agent
      const scenariosWithMetrics = scenarios.map(s => {
        const metrics = scenarioMetrics[s.scenario_id] || {};
        return {
          scenario_id: s.scenario_id,
          scaffold: s.scaffold || "Unknown",
          smiles: s.smiles || "",
          binding_affinity: metrics.docking?.binding_affinity_kcal_per_mol || -7,
          herg_flag: metrics.admet?.herg_flag || false,
          sa_score: metrics.synthesis?.sa_score || 4,
          cost_usd: metrics.synthesis?.estimated_cost_usd || 1500,
        };
      });

      const agentResult = await generateVerdict({
        experiment_id: experimentId,
        protein_target: proteinTarget,
        scenarios: scenariosWithMetrics,
        decision_criteria: {
          herg_veto: admetCriteria?.hardFailHERG !== false,
          potency_threshold: (dockingCriteria?.hardFailThreshold as number) ?? -7,
          sa_threshold: (synthesisCriteria?.hardFailSa as number) ?? 6,
        },
        goal: context?.goal,
        constraints: context?.constraints,
      });

      if (agentResult.success && agentResult.data) {
        const winner = agentResult.data.verdict?.winner as Record<string, unknown> | null;
        console.log(`[Judge] Agent returned verdict: winner=${winner?.scenario_id || "none"}`);
        console.log(`[Judge] ========================================`);
        
        return NextResponse.json({
          reason: "Verdict generated successfully",
          structured: {
            ...agentResult.data.verdict,
            executive_summary: agentResult.data.executive_summary,
            comparative_analysis: agentResult.data.comparative_analysis,
          },
          data_source: "agent",
          confidence: "high",
          _via: "deployed_agent",
        });
      }

      console.log(`[Judge] Agent call failed: ${agentResult.error}, falling back to local LLM`);
    } else {
      console.log(`[Judge] Agentex not configured, using local LLM fallback`);
    }

    // =========================================================================
    // FALLBACK: Local LLM processing (limited tracing)
    // =========================================================================
    let llmVerdict: Record<string, unknown>;
    try {
      llmVerdict = await generateVerdictWithLLM(
        scenarios,
        scenarioMetrics,
        context,
        decisionCriteria
      );
      const winner = llmVerdict.winner as Record<string, unknown> | null;
      console.log(`[Judge] LLM verdict: winner=${winner?.scenario_id || "none"}`);
    } catch (llmError) {
      console.error(`[Judge] LLM error:`, llmError);
      return NextResponse.json({
        error: `Judge LLM failed: ${llmError instanceof Error ? llmError.message : "Unknown"}`,
      }, { status: 500 });
    }

    const { verdict, wasOverridden, corrections } = crossCheckVerdictWithDatabase(
      llmVerdict,
      scenarios,
      proteinTarget,
      decisionCriteria
    );

    if (wasOverridden) {
      console.log(`[Judge] Verdict corrected based on database validation`);
      corrections.forEach(c => console.log(`[Judge]   - ${c}`));
    }

    const finalWinner = verdict.winner as Record<string, unknown> | null;
    console.log(`[Judge] Final verdict: winner=${finalWinner?.scenario_id || "none"}`);
    console.log(`[Judge] ========================================`);

    return NextResponse.json({
      reason: "Verdict generated successfully",
      structured: verdict,
      data_source: wasOverridden ? "llm_validated" : "llm",
      confidence: wasOverridden ? "high" : "medium",
      validation_notes: corrections,
      _via: "local_llm_fallback",
    });

  } catch (error: unknown) {
    console.error("[Judge] Unexpected error:", error);
    return NextResponse.json({
      error: `Judge failed: ${error instanceof Error ? error.message : "Unknown"}`,
    }, { status: 500 });
  }
}
