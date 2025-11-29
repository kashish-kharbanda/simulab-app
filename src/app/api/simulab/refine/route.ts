import { NextRequest, NextResponse } from "next/server"
import { getScenariosByProteinTarget, SheetScenario } from "../sheets/data"
import { traceDesignChange as sendTraceToSGP, isTracingEnabled } from "@/lib/sgp-tracing"
import { designExperiment, isDevMode, isAgentexConfigured, AGENTS } from "@/lib/agent-client"

/**
 * SimuLab Refine API Route (Orchestrator Agent)
 * 
 * FULL INTEGRATION: Calls the deployed Orchestrator agent via SGP/Agentex API.
 * Falls back to local processing if agent is unavailable.
 * 
 * Flow:
 * 1. If Agentex configured: Call deployed Orchestrator agent's /design_experiment endpoint
 * 2. Agent extracts protein target and generates scenarios
 * 3. Frontend cross-checks with Google Sheets database for known targets
 * 
 * Database is the SINGLE SOURCE OF TRUTH for known protein targets.
 * 
 * TRACING: Captures experiment design changes directly to SGP
 */

export const maxDuration = 120; // 2 minutes for agent processing
export const dynamic = "force-dynamic"

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || process.env.NEXT_PUBLIC_OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";

/**
 * Send design change trace to SGP directly
 */
async function traceDesignChange(data: {
  experimentId: string;
  changeType: string;
  originalValue: unknown;
  newValue: unknown;
  reason?: string;
  proteinTarget?: string;
}) {
  if (!isTracingEnabled()) {
    console.log("[SGP/Trace] Tracing not configured, skipping");
    return null;
  }
  
  try {
    const result = await sendTraceToSGP({
      experimentId: data.experimentId,
      changeType: data.changeType,
      originalValue: data.originalValue,
      newValue: data.newValue,
      reasoning: data.reason,
    });
    
    if (result.success) {
      console.log("[SGP/Trace] Design change traced:", result.trace_id);
    }
    return result;
  } catch (error) {
    console.warn("[SGP/Trace] Failed to trace design change:", error);
  }
  return null;
}

interface RefineRequest {
  prompt?: string;
  constraints?: string;
}

interface RefinedOutput {
  goal: string;
  constraints: string[];
  protein_target: string;
  suggested_num_scenarios: number;
  scenarios: Array<{
    scenario_id: string;
    scaffold: string;
    smiles?: string;
    rationale?: string;
    reference_data?: {
      binding_affinity: number | null;
      herg_flag: boolean | null;
      sa_score: number | null;
      target_result: string;
    };
  }>;
  model_used: string;
  notice?: string;
  data_source: "database" | "llm";
  confidence: "high" | "medium";
}

/**
 * Extract protein target from user prompt
 */
function extractProteinTarget(prompt: string): string {
  // First check for known database targets (case-insensitive, with variations)
  const knownTargets = [
    { pattern: /\b(BCR-ABL|BCR\s*ABL|BCRABL)\b/i, name: "BCR-ABL" },
    { pattern: /\b(T-Kinase|T\s*Kinase|TKinase)\b/i, name: "T-Kinase" },
    { pattern: /\b(Tox-Check|Tox\s*Check|ToxCheck)\b/i, name: "Tox-Check" },
    { pattern: /\b(Amyloid\s*Beta|Amyloid-Beta|AmyloidBeta|Aβ|A-beta|Abeta)\b/i, name: "Amyloid Beta" },
  ];
  
  for (const { pattern, name } of knownTargets) {
    if (pattern.test(prompt)) {
      return name;
    }
  }
  
  // Fallback patterns for other targets
  const fallbackPatterns = [
    /(?:for|target(?:ing)?|against|inhibit(?:or)?|bind(?:ing)?)\s+([A-Z][A-Z0-9\-]{2,})/i,
    /([A-Z][A-Z0-9\-]{2,})\s+(?:kinase|receptor|enzyme|protein|inhibitor)/i,
    /\b(EGFR|HER2|BRAF|JAK[123]?|CDK[0-9]+|PI3K|mTOR|ALK|ROS1|MET|KRAS|NRAS|FLT3|BTK|SYK)\b/i,
  ];
  
  for (const pattern of fallbackPatterns) {
    const match = prompt.match(pattern);
    if (match && match[1]) {
      return match[1];
    }
  }
  
  // Try to find any uppercase word that looks like a protein target
  const uppercase = prompt.match(/[A-Z][A-Z0-9\-]{2,}/g);
  if (uppercase && uppercase.length > 0) return uppercase[0];
  
  return "Unknown Target";
}

/**
 * Call LLM to refine goal ONLY (when database scenarios are used)
 * LLM is informed that scenarios come from expert experimental data
 */
async function refineGoalWithLLM(
  prompt: string, 
  constraints: string, 
  proteinTarget: string,
  dbScenarios: SheetScenario[]
): Promise<{ goal: string; constraints: string[] }> {
  if (!OPENAI_API_KEY) {
    // Return basic refinement if no API key
    return {
      goal: prompt || `Optimize lead molecules targeting ${proteinTarget}`,
      constraints: constraints ? constraints.split(/[.;]/).map(s => s.trim()).filter(Boolean) : [],
    };
  }

  const scenarioSummary = dbScenarios.map(s => 
    `- ${s.scaffold_hypothesis}: ΔG=${s.reference_binding_affinity} kcal/mol, hERG=${s.reference_herg_flag ? "Yes" : "No"}, SA=${s.reference_sa_score}, Result=${s.target_result}`
  ).join("\n");

  const systemPrompt = `You are a PhD-level drug discovery scientist. You are refining a user's goal for a virtual drug discovery experiment.

IMPORTANT: The experimental scenarios have ALREADY been determined from our validated experimental database. You do NOT need to generate new scenarios.

The database contains pre-calculated values from expert scientists - these are the SINGLE SOURCE OF TRUTH.

Your task is ONLY to:
1. Refine the user's goal into a clear, scientific objective
2. Extract any constraints the user mentioned (if none mentioned, return empty array)

Return ONLY valid JSON:
{
  "goal": "Refined scientific objective",
  "constraints": ["constraint1", "constraint2"] or [] if none provided
}`;

  const userPrompt = `User's goal: ${prompt}
${constraints ? `User's constraints: ${constraints}` : "No constraints provided by user."}

Target protein: ${proteinTarget}

Pre-validated experimental scenarios from database:
${scenarioSummary}

Refine the goal. Only include constraints if the user explicitly mentioned them.`;

  try {
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
      console.warn(`[Orchestrator] Goal refinement LLM failed, using basic refinement`);
      return {
        goal: prompt || `Optimize lead molecules targeting ${proteinTarget}`,
        constraints: constraints ? constraints.split(/[.;]/).map(s => s.trim()).filter(Boolean) : [],
      };
    }

    const data = await response.json();
    const content = data?.choices?.[0]?.message?.content || "";
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      return {
        goal: parsed.goal || prompt,
        constraints: Array.isArray(parsed.constraints) ? parsed.constraints : [],
      };
    }
  } catch (err) {
    console.warn(`[Orchestrator] Goal refinement error:`, err);
  }

  return {
    goal: prompt || `Optimize lead molecules targeting ${proteinTarget}`,
    constraints: constraints ? constraints.split(/[.;]/).map(s => s.trim()).filter(Boolean) : [],
  };
}

/**
 * Call LLM to generate scenarios - ONLY when no database match
 */
async function generateScenariosWithLLM(prompt: string, constraints: string, proteinTarget: string): Promise<any> {
  if (!OPENAI_API_KEY) {
    throw new Error("OpenAI API key not configured - LLM is required when no database match exists");
  }

  const systemPrompt = `You are a PhD-level drug discovery scientist with 20+ years of experience.

You are the Orchestrator agent for a virtual drug discovery lab. Your job is to design experimental scenarios.

TASK: Given a user's goal and constraints, generate:
1. A refined goal statement
2. Key constraints (ONLY if user explicitly mentioned them, otherwise empty array)
3. 2-3 distinct molecular scaffold hypotheses to test

For each scaffold scenario, provide:
- scenario_id: "scenario_1", "scenario_2", etc.
- scaffold: Name of the chemical scaffold class
- smiles: A valid SMILES string for a representative molecule
- rationale: Scientific reasoning

Return ONLY valid JSON:
{
  "goal": "Refined objective",
  "constraints": [],
  "protein_target": "TARGET_NAME",
  "scenarios": [
    {
      "scenario_id": "scenario_1",
      "scaffold": "Scaffold Name",
      "smiles": "SMILES_STRING",
      "rationale": "Scientific reasoning"
    }
  ]
}`;

  const userPrompt = `Design experimental scenarios for:

GOAL: ${prompt}
${constraints ? `CONSTRAINTS: ${constraints}` : "No constraints provided."}
TARGET PROTEIN: ${proteinTarget}

Generate 2-3 distinct scaffold hypotheses. Only include constraints if explicitly provided.`;

  console.log(`[Orchestrator] No database match - calling LLM for scenario generation...`);

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
      temperature: 0.3,
      max_tokens: 1500,
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
  console.log(`[Orchestrator] LLM generated ${parsed.scenarios?.length || 0} scenarios`);

  return parsed;
}

export async function POST(request: NextRequest) {
  try {
    const body: RefineRequest = await request.json();
    const { prompt = "", constraints = "" } = body;

    console.log(`[Orchestrator] ========================================`);
    console.log(`[Orchestrator] Processing: ${prompt.substring(0, 80)}...`);
    console.log(`[Orchestrator] User constraints: "${constraints || "NONE"}"`);
    console.log(`[Orchestrator] Mode: ${isDevMode() ? 'DEV (local agent)' : 'PROD (deployed agent)'}`);

    // =========================================================================
    // TRY DEPLOYED ORCHESTRATOR AGENT FIRST (Full Integration)
    // =========================================================================
    if (isDevMode() || isAgentexConfigured()) {
      console.log(`[Orchestrator] Calling ${AGENTS.ORCHESTRATOR} agent...`);
      
      const agentResult = await designExperiment({
        prompt,
        constraints,
      });

      if (agentResult.success && agentResult.data) {
        console.log(`[Orchestrator] Agent returned ${agentResult.data.scenarios?.length || 0} scenarios`);
        
        // Cross-check with database for known targets
        const proteinTarget = agentResult.data.protein_target;
        const dbScenarios = getScenariosByProteinTarget(proteinTarget);
        
        if (dbScenarios.length > 0) {
          // Override with database scenarios (source of truth)
          console.log(`[Orchestrator] ✓ DATABASE OVERRIDE: Using ${dbScenarios.length} validated scenarios`);
          
          const scenarios = dbScenarios.map((db, idx) => ({
            scenario_id: `scenario_${idx + 1}`,
            scaffold: db.scaffold_hypothesis,
            smiles: db.smiles,
            rationale: `Validated scaffold for ${proteinTarget} with known properties.`,
            reference_data: {
              binding_affinity: db.reference_binding_affinity,
              herg_flag: db.reference_herg_flag,
              sa_score: db.reference_sa_score,
              target_result: db.target_result,
            },
          }));

          const result: RefinedOutput = {
            goal: agentResult.data.goal,
            constraints: agentResult.data.constraints,
            protein_target: proteinTarget,
            suggested_num_scenarios: scenarios.length,
            scenarios,
            model_used: "agent+database",
            data_source: "database",
            confidence: "high",
          };

          console.log(`[Orchestrator] ========================================`);
          return NextResponse.json(result, { status: 200 });
        }

        // No database match - use agent's scenarios
        const result: RefinedOutput = {
          goal: agentResult.data.goal,
          constraints: agentResult.data.constraints,
          protein_target: agentResult.data.protein_target,
          suggested_num_scenarios: agentResult.data.scenarios.length,
          scenarios: agentResult.data.scenarios.map((s, idx) => ({
            scenario_id: s.scenario_id || `scenario_${idx + 1}`,
            scaffold: s.scaffold,
            smiles: s.smiles,
            rationale: s.rationale,
          })),
          model_used: "agent",
          data_source: "llm",
          confidence: "medium",
        };

        console.log(`[Orchestrator] ========================================`);
        return NextResponse.json(result, { status: 200 });
      }

      console.log(`[Orchestrator] Agent call failed: ${agentResult.error}, falling back to local processing`);
    } else {
      console.log(`[Orchestrator] Agentex not configured, using local processing`);
    }

    // =========================================================================
    // FALLBACK: Local processing (database + LLM)
    // =========================================================================

    // Step 1: Extract protein target
    const proteinTarget = extractProteinTarget(prompt);
    console.log(`[Orchestrator] Detected protein target: ${proteinTarget}`);

    // Step 2: CHECK DATABASE FIRST (source of truth)
    const dbScenarios = getScenariosByProteinTarget(proteinTarget);
    
    if (dbScenarios.length > 0) {
      // DATABASE MATCH FOUND - use these scenarios directly
      console.log(`[Orchestrator] ✓ DATABASE MATCH: Found ${dbScenarios.length} validated scenarios for ${proteinTarget}`);
      dbScenarios.forEach((s, i) => {
        console.log(`[Orchestrator]   ${i+1}. ${s.scaffold_hypothesis} → ${s.target_result}`);
      });

      // Refine goal with LLM (but scenarios come from database)
      const { goal, constraints: refinedConstraints } = await refineGoalWithLLM(
        prompt, 
        constraints, 
        proteinTarget, 
        dbScenarios
      );

      // Build scenarios from database (internal - not exposed to user)
      const scenarios = dbScenarios.map((db, idx) => ({
        scenario_id: `scenario_${idx + 1}`,
        scaffold: db.scaffold_hypothesis,
        smiles: db.smiles,
        rationale: `Promising scaffold class for ${proteinTarget} inhibition with favorable predicted properties.`,
        reference_data: {
          binding_affinity: db.reference_binding_affinity,
          herg_flag: db.reference_herg_flag,
          sa_score: db.reference_sa_score,
          target_result: db.target_result,
        },
      }));

      const result: RefinedOutput = {
        goal,
        constraints: refinedConstraints, // Only user-provided constraints
        protein_target: proteinTarget,
        suggested_num_scenarios: scenarios.length,
        scenarios,
        model_used: OPENAI_MODEL,
        data_source: "database",
        confidence: "high",
      };

      console.log(`[Orchestrator] Result: ${result.scenarios.length} scenarios from DATABASE`);
      console.log(`[Orchestrator] ========================================`);

      // Send trace to Orchestrator agent for design tracking (non-blocking)
      traceDesignChange({
        experimentId: `exp_${Date.now()}`,
        changeType: "initial_design",
        originalValue: { prompt, constraints },
        newValue: result,
        reason: "Initial experiment design from database",
        proteinTarget,
      }).catch(err => {
        console.warn("[Orchestrator] Trace failed (non-blocking):", err)
      })

      return NextResponse.json(result, { status: 200 });
    }

    // Step 3: NO DATABASE MATCH - use LLM to generate scenarios
    console.log(`[Orchestrator] ✗ No database match for ${proteinTarget} - using LLM generation`);
    
    let llmOutput: any;
    try {
      llmOutput = await generateScenariosWithLLM(prompt, constraints, proteinTarget);
    } catch (llmError) {
      console.error(`[Orchestrator] LLM error:`, llmError);
      return NextResponse.json({
        error: `Orchestrator LLM failed: ${llmError instanceof Error ? llmError.message : "Unknown error"}`,
        suggestion: "Please ensure OPENAI_API_KEY is configured",
      }, { status: 500 });
    }

    // Build result from LLM output
    const scenarios = (llmOutput.scenarios || []).map((s: any, idx: number) => ({
      scenario_id: s.scenario_id || `scenario_${idx + 1}`,
      scaffold: s.scaffold,
      smiles: s.smiles,
      rationale: s.rationale,
    }));

    const result: RefinedOutput = {
      goal: llmOutput.goal || prompt,
      constraints: Array.isArray(llmOutput.constraints) ? llmOutput.constraints : [],
      protein_target: proteinTarget,
      suggested_num_scenarios: scenarios.length,
      scenarios,
      model_used: OPENAI_MODEL,
      data_source: "llm",
      confidence: "medium",
    };

    console.log(`[Orchestrator] Result: ${result.scenarios.length} scenarios from LLM`);
    console.log(`[Orchestrator] ========================================`);

    // Send trace to Orchestrator agent for design tracking (non-blocking)
    traceDesignChange({
      experimentId: `exp_${Date.now()}`,
      changeType: "initial_design",
      originalValue: { prompt, constraints },
      newValue: result,
      reason: "Initial experiment design from LLM",
      proteinTarget,
    }).catch(err => {
      console.warn("[Orchestrator] Trace failed (non-blocking):", err)
    })

    return NextResponse.json(result, { status: 200 });

  } catch (error: unknown) {
    console.error("[Orchestrator] Unexpected error:", error);
    return NextResponse.json({
      error: `Orchestrator failed: ${error instanceof Error ? error.message : "Unknown error"}`,
    }, { status: 500 });
  }
}
