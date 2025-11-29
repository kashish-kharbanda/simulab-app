import { NextRequest, NextResponse } from "next/server"

/**
 * SimuLab Generate Report API Route
 * 
 * Uses LLM to generate a structured report with all data extracted into JSON format.
 * This allows the frontend to populate existing UI components dynamically.
 */

export const maxDuration = 120;
export const dynamic = "force-dynamic"

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || process.env.NEXT_PUBLIC_OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";

interface ScenarioAnalysis {
  scenario_id: string;
  smiles: string;
  scaffold: string;
  status: "winner" | "selected" | "rejected" | "considered";
  metrics: {
    binding_affinity_kcal_per_mol: number;
    potency_pass: boolean;
    toxicity_risk: "LOW" | "MED" | "HIGH";
    toxicity_prob: number;
    herg_flag: boolean;
    is_safe: boolean;
    sa_score: number;
    num_steps: number;
    estimated_cost_usd: number;
  };
  pros: string[];
  cons: string[];
  rejection_reason?: string;
  selection_reason?: string;
}

interface StructuredReport {
  executive_summary: string;
  target_protein: string;
  goal_achieved: boolean;
  goal_summary: string;
  scenarios: ScenarioAnalysis[];
  comparative_analysis: string;
  winner_justification: string;
  recommendations: string[];
  next_steps: string[];
}

interface GenerateReportRequest {
  scenarios: Array<{
    scenario_id: string;
    smiles?: string;
    scaffold?: string;
    metadata?: { scaffold?: string };
  }>;
  scenarioMetrics: Record<string, {
    docking?: { binding_affinity_kcal_per_mol?: number; potency_pass?: boolean };
    admet?: { toxicity_risk?: string; toxicity_prob?: number; herg_flag?: boolean; is_safe?: boolean };
    synthesis?: { sa_score?: number; num_steps?: number; estimated_cost_usd?: number };
  }>;
  winners: Array<{ scenario_id: string; smiles?: string }>;
  rejected: Array<{ scenario_id: string; smiles?: string; veto_reason?: string }>;
  context: {
    protein_target: string;
    goal: string;
    constraints: string[];
  };
  decisionCriteria: {
    docking?: { idealMin?: number; idealMax?: number; hardFailThreshold?: number };
    admet?: { idealMin?: number; idealMax?: number; hardFailHERG?: boolean };
    synthesis?: { idealSaMax?: number; idealStepsMax?: number; hardFailSa?: number; hardFailSteps?: number };
  };
}

export async function POST(request: NextRequest) {
  try {
    const body: GenerateReportRequest = await request.json()
    const { 
      scenarios = [], 
      scenarioMetrics = {},
      winners = [],
      rejected = [],
      context = { protein_target: "Unknown", goal: "", constraints: [] },
      decisionCriteria = {}
    } = body

    console.log(`[SimuLab/GenerateReport] Generating structured report for ${scenarios.length} scenarios`);

    // Build enriched scenario data
    const enrichedScenarios: ScenarioAnalysis[] = scenarios.map((s) => {
      const metrics = scenarioMetrics[s.scenario_id] || {};
      const isWinner = winners.some(w => w.scenario_id === s.scenario_id);
      const isRejected = rejected.some(r => r.scenario_id === s.scenario_id);
      const rejectionInfo = rejected.find(r => r.scenario_id === s.scenario_id);
      
      // Determine status - winner, selected (passed but not winner), rejected, or considered
      let status: "winner" | "selected" | "rejected" | "considered" = "considered";
      if (isWinner) {
        status = "winner";
      } else if (isRejected) {
        status = "rejected";
      } else {
        // If not winner and not rejected, it's selected (passed criteria)
        status = "selected";
      }
      
      return {
        scenario_id: s.scenario_id,
        smiles: s.smiles || "",
        scaffold: s.scaffold || s.metadata?.scaffold || "Unknown scaffold",
        status,
        metrics: {
          binding_affinity_kcal_per_mol: metrics.docking?.binding_affinity_kcal_per_mol ?? 0,
          potency_pass: metrics.docking?.potency_pass ?? false,
          toxicity_risk: (metrics.admet?.toxicity_risk as "LOW" | "MED" | "HIGH") || "LOW",
          toxicity_prob: metrics.admet?.toxicity_prob ?? 0,
          herg_flag: metrics.admet?.herg_flag ?? false,
          is_safe: metrics.admet?.is_safe ?? true,
          sa_score: metrics.synthesis?.sa_score ?? 0,
          num_steps: metrics.synthesis?.num_steps ?? 0,
          estimated_cost_usd: metrics.synthesis?.estimated_cost_usd ?? 0,
        },
        pros: [],
        cons: [],
        rejection_reason: rejectionInfo?.veto_reason,
      };
    });

    if (!OPENAI_API_KEY) {
      console.warn('[SimuLab/GenerateReport] No OpenAI API key, using fallback structured report');
      return NextResponse.json({ 
        report: generateFallbackReport(enrichedScenarios, context, winners, rejected),
        source: "fallback"
      });
    }

    const systemPrompt = `You are a senior medicinal chemist generating a structured drug discovery report.
Your response must be ONLY valid JSON matching this exact schema:

{
  "executive_summary": "2-3 sentence professional summary of the screening experiment results",
  "target_protein": "Name of the protein target",
  "goal_achieved": true/false,
  "goal_summary": "Brief statement on how well the goal was met",
  "scenarios": [
    {
      "scenario_id": "scenario_1",
      "smiles": "SMILES string",
      "scaffold": "Scaffold name",
      "status": "winner" | "selected" | "rejected",
      "pros": ["Pro 1", "Pro 2"],
      "cons": ["Con 1", "Con 2"],
      "rejection_reason": "Only for rejected molecules - specific reason",
      "selection_reason": "Only for selected (non-winner passing) molecules - why it passed"
    }
  ],
  "comparative_analysis": "Professional paragraph comparing all candidates, discussing trade-offs",
  "winner_justification": "Scientific justification for the winning molecule selection",
  "recommendations": ["Recommendation 1", "Recommendation 2"],
  "next_steps": ["Next step 1", "Next step 2"]
}

STATUS MEANINGS:
- "winner": The best molecule that passes all criteria
- "selected": Molecules that pass all criteria but aren't the best (viable backup candidates)
- "rejected": Molecules that fail one or more hard criteria (safety veto, potency fail, cost veto)

IMPORTANT:
- Use professional scientific language
- Be specific with data and metrics
- Do NOT include markdown formatting
- Response must be valid JSON only`;

    const userPrompt = `Generate a structured report for this drug discovery screening:

Target Protein: ${context.protein_target}
Goal: ${context.goal}
Constraints: ${context.constraints.join("; ") || "Standard drug-like properties"}

Scenarios evaluated:
${JSON.stringify(enrichedScenarios, null, 2)}

Decision Criteria:
- Docking: ideal ΔG ${decisionCriteria.docking?.idealMin || -12} to ${decisionCriteria.docking?.idealMax || -8} kcal/mol
- ADMET: ideal toxicity 0-0.3, hERG veto: ${decisionCriteria.admet?.hardFailHERG ? "Yes" : "No"}
- Synthesis: ideal SA ≤ ${decisionCriteria.synthesis?.idealSaMax || 4}

Winners: ${winners.map(w => w.scenario_id).join(", ") || "None"}
Rejected: ${rejected.map(r => `${r.scenario_id} (${r.veto_reason || "criteria not met"})`).join(", ") || "None"}

Generate the structured JSON report with professional analysis.`;

    console.log(`[SimuLab/GenerateReport] Calling OpenAI ${OPENAI_MODEL}...`);

    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
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
        max_tokens: 2000,
        response_format: { type: "json_object" },
      }),
    });

    if (!resp.ok) {
      const errorText = await resp.text();
      console.error(`[SimuLab/GenerateReport] OpenAI error (${resp.status}):`, errorText);
      return NextResponse.json({ 
        report: generateFallbackReport(enrichedScenarios, context, winners, rejected),
        source: "fallback",
        error: `LLM error: ${resp.status}`
      });
    }

    const data = await resp.json();
    const content = data?.choices?.[0]?.message?.content || "";

    try {
      const report = JSON.parse(content) as StructuredReport;
      
      // Merge LLM analysis with actual metrics data
      const mergedScenarios = enrichedScenarios.map(s => {
        const llmScenario = report.scenarios?.find(rs => rs.scenario_id === s.scenario_id);
        return {
          ...s,
          pros: llmScenario?.pros || generateDefaultPros(s),
          cons: llmScenario?.cons || generateDefaultCons(s),
          rejection_reason: s.rejection_reason || llmScenario?.rejection_reason,
        };
      });

      const finalReport: StructuredReport = {
        ...report,
        scenarios: mergedScenarios,
      };

      console.log(`[SimuLab/GenerateReport] Report generated successfully`);
      return NextResponse.json({ report: finalReport, source: "llm" });

    } catch (parseError) {
      console.error('[SimuLab/GenerateReport] Failed to parse LLM response:', parseError);
      return NextResponse.json({ 
        report: generateFallbackReport(enrichedScenarios, context, winners, rejected),
        source: "fallback",
        parseError: String(parseError)
      });
    }

  } catch (error: unknown) {
    console.error('[SimuLab/GenerateReport] Unexpected error:', error);
    return NextResponse.json(
      { error: `Failed to generate report: ${error instanceof Error ? error.message : "Unknown"}` },
      { status: 500 }
    );
  }
}

function generateFallbackReport(
  scenarios: any[],
  context: { protein_target: string; goal: string; constraints: string[] },
  winners: any[],
  rejected: any[]
): StructuredReport {
  const winnerScenarios = scenarios.filter(s => s.status === "winner");
  const rejectedScenarios = scenarios.filter(s => s.status === "rejected");

  return {
    executive_summary: `The virtual screening campaign for ${context.protein_target} evaluated ${scenarios.length} molecular candidate(s). ${winners.length} molecule(s) passed all screening criteria and ${rejected.length} were rejected based on the defined thresholds.`,
    target_protein: context.protein_target,
    goal_achieved: winners.length > 0,
    goal_summary: winners.length > 0 
      ? `Successfully identified ${winners.length} lead candidate(s) meeting the optimization criteria.`
      : `No candidates met all screening criteria. Further optimization may be required.`,
    scenarios: scenarios.map(s => ({
      ...s,
      pros: generateDefaultPros(s),
      cons: generateDefaultCons(s),
    })),
    comparative_analysis: generateComparativeAnalysis(scenarios),
    winner_justification: winnerScenarios.length > 0
      ? `${winnerScenarios[0].scenario_id} was selected as the lead candidate based on its optimal balance of binding affinity (${winnerScenarios[0].metrics.binding_affinity_kcal_per_mol} kcal/mol), acceptable safety profile, and favorable synthetic accessibility (SA ${winnerScenarios[0].metrics.sa_score}).`
      : "No winner identified in this screening round.",
    recommendations: [
      "Validate lead candidate(s) with additional in vitro assays",
      "Conduct selectivity profiling against related kinases",
      "Perform metabolic stability assessment",
    ],
    next_steps: [
      "Proceed to hit-to-lead optimization",
      "Scale up synthesis for further testing",
      "Initiate ADMET profiling studies",
    ],
  };
}

function generateDefaultPros(scenario: any): string[] {
  const pros: string[] = [];
  const m = scenario.metrics;
  
  if (m.binding_affinity_kcal_per_mol < -8) {
    pros.push(`Strong binding affinity (${m.binding_affinity_kcal_per_mol} kcal/mol)`);
  }
  if (m.potency_pass) {
    pros.push("Meets potency threshold");
  }
  if (m.is_safe) {
    pros.push("Acceptable safety profile");
  }
  if (!m.herg_flag) {
    pros.push("No hERG liability");
  }
  if (m.sa_score < 4) {
    pros.push(`Good synthetic accessibility (SA ${m.sa_score})`);
  }
  if (m.estimated_cost_usd < 2000) {
    pros.push(`Cost-effective synthesis ($${m.estimated_cost_usd})`);
  }
  
  return pros.length > 0 ? pros : ["Candidate under evaluation"];
}

function generateDefaultCons(scenario: any): string[] {
  const cons: string[] = [];
  const m = scenario.metrics;
  
  if (m.binding_affinity_kcal_per_mol > -7) {
    cons.push(`Weak binding affinity (${m.binding_affinity_kcal_per_mol} kcal/mol)`);
  }
  if (!m.potency_pass) {
    cons.push("Below potency threshold");
  }
  if (m.herg_flag) {
    cons.push("hERG cardiac toxicity flag");
  }
  if (m.toxicity_risk === "HIGH") {
    cons.push("High toxicity risk");
  } else if (m.toxicity_risk === "MED") {
    cons.push("Moderate toxicity risk");
  }
  if (m.sa_score > 5) {
    cons.push(`Poor synthetic accessibility (SA ${m.sa_score})`);
  }
  if (m.estimated_cost_usd > 3000) {
    cons.push(`High synthesis cost ($${m.estimated_cost_usd})`);
  }
  
  return cons.length > 0 ? cons : ["No significant concerns identified"];
}

function generateComparativeAnalysis(scenarios: any[]): string {
  if (scenarios.length === 0) return "No candidates to compare.";
  if (scenarios.length === 1) {
    return `Single candidate (${scenarios[0].scenario_id}) was evaluated. ${scenarios[0].status === "winner" ? "The molecule meets all screening criteria." : "The molecule did not meet all screening criteria."}`;
  }
  
  const winners = scenarios.filter(s => s.status === "winner");
  const rejected = scenarios.filter(s => s.status === "rejected");
  
  let analysis = `Comparative analysis of ${scenarios.length} candidates reveals `;
  
  if (winners.length > 0 && rejected.length > 0) {
    analysis += `a clear differentiation between passing and failing molecules. `;
    analysis += `Winner(s) demonstrated superior balance of potency, safety, and manufacturability. `;
    analysis += `Rejected candidate(s) failed primarily due to ${rejected[0].rejection_reason || "not meeting threshold criteria"}.`;
  } else if (winners.length > 0) {
    analysis += `all candidates met the screening criteria with varying degrees of optimization potential.`;
  } else {
    analysis += `no candidates fully met the screening criteria. Further scaffold exploration may be warranted.`;
  }
  
  return analysis;
}


