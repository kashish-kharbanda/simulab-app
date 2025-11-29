import { NextRequest, NextResponse } from "next/server"
import { processReportEdit, isDevMode, isAgentexConfigured, AGENTS, traceReportEdit } from "@/lib/agent-client"

/**
 * SimuLab Edit Report API Route
 * 
 * FULL INTEGRATION: Calls the deployed Simulator agent via SGP/Agentex API.
 * Falls back to local LLM calls if agent is unavailable.
 * 
 * Flow:
 * 1. If Agentex configured: Call deployed Simulator agent's /process_edit endpoint
 * 2. Agent processes edit with LLM and captures tracing
 * 3. Returns updated report
 * 
 * Fallback (if agent unavailable):
 * 1. Call LLM directly (limited tracing)
 */

export const maxDuration = 240; // 4 minutes for agent processing
export const dynamic = "force-dynamic"

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || process.env.NEXT_PUBLIC_OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o";

interface ScenarioData {
  scenario_id: string;
  scaffold: string;
  binding_affinity: number;
  toxicity_risk: string;
  herg_flag: boolean;
  sa_score: number;
  cost_usd: number;
  rationale?: string;
  rejection_reason?: string;
  smiles?: string;
}

interface StructuredReport {
  executive_summary?: string;
  winner?: ScenarioData;
  selected?: ScenarioData[];
  rejected?: ScenarioData[];
  comparative_analysis?: string;
  recommendation?: string;
}

interface EditReportRequest {
  structuredReport: StructuredReport;
  editInstruction: string;
  context?: {
    protein_target?: string;
    goal?: string;
    taskId?: string;
  };
}

/**
 * Process edit locally using LLM - FALLBACK only
 */
async function processEditWithLLM(
  structuredReport: StructuredReport,
  editInstruction: string
): Promise<StructuredReport> {
  if (!OPENAI_API_KEY) {
    throw new Error("OpenAI API key not configured");
  }

  const currentWinner = structuredReport.winner;
  const currentSelected = structuredReport.selected || [];
  const currentRejected = structuredReport.rejected || [];
  const totalScenarios = (currentWinner ? 1 : 0) + currentSelected.length + currentRejected.length;

  const systemPrompt = `You are an expert drug discovery scientist. Edit the report based on user instructions.

CRITICAL: Preserve ALL ${totalScenarios} scenarios. Every scenario must appear in winner, selected, or rejected.

Current state:
- Winner: ${currentWinner?.scenario_id || "none"}
- Selected: ${currentSelected.map(s => s.scenario_id).join(", ") || "none"}
- Rejected: ${currentRejected.map(r => r.scenario_id).join(", ") || "none"}

After editing:
1. Apply the requested change
2. Re-evaluate categories based on decision criteria
3. Update executive_summary, comparative_analysis, rationale/rejection_reason
4. Ensure all ${totalScenarios} scenarios are present

Return ONLY valid JSON:
{
  "executive_summary": "...",
  "winner": {...} or null,
  "selected": [...],
  "rejected": [...],
  "comparative_analysis": "...",
  "recommendation": "..."
}`;

  const userPrompt = `Edit instruction: "${editInstruction}"

Current report:
${JSON.stringify(structuredReport, null, 2)}

Apply the edit and return complete updated JSON.`;

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
        { role: "user", content: userPrompt }
      ],
      temperature: 0.1,
      max_tokens: 3000,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`LLM error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content;

  if (!content) {
    throw new Error("No content in LLM response");
  }

  let jsonStr = content.trim();
  
  if (jsonStr.startsWith("```json")) {
    jsonStr = jsonStr.slice(7);
  } else if (jsonStr.startsWith("```")) {
    jsonStr = jsonStr.slice(3);
  }
  if (jsonStr.endsWith("```")) {
    jsonStr = jsonStr.slice(0, -3);
  }
  jsonStr = jsonStr.trim();
  
  const jsonMatch = jsonStr.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    jsonStr = jsonMatch[0];
  }
  
  return JSON.parse(jsonStr);
}

/**
 * Generate summary of changes between reports
 */
function generateChangeSummary(
  oldReport: StructuredReport,
  newReport: StructuredReport
): string[] {
  const changes: string[] = [];
  
  const oldWinner = oldReport.winner;
  const newWinner = newReport.winner;
  
  if (oldWinner && newWinner) {
    if (oldWinner.binding_affinity !== newWinner.binding_affinity) {
      changes.push(`${newWinner.scenario_id} binding affinity: ${oldWinner.binding_affinity} → ${newWinner.binding_affinity} kcal/mol`);
    }
    if (oldWinner.sa_score !== newWinner.sa_score) {
      changes.push(`${newWinner.scenario_id} SA score: ${oldWinner.sa_score} → ${newWinner.sa_score}`);
    }
    if (oldWinner.herg_flag !== newWinner.herg_flag) {
      changes.push(`${newWinner.scenario_id} hERG flag: ${oldWinner.herg_flag} → ${newWinner.herg_flag}`);
    }
    if (oldWinner.cost_usd !== newWinner.cost_usd) {
      changes.push(`${newWinner.scenario_id} cost: $${oldWinner.cost_usd} → $${newWinner.cost_usd}`);
    }
  }
  
  if (oldWinner?.scenario_id !== newWinner?.scenario_id) {
    if (newWinner) {
      changes.push(`Winner changed to ${newWinner.scenario_id}`);
    } else {
      changes.push(`${oldWinner?.scenario_id} moved to rejected`);
    }
  }

  const oldRejected = oldReport.rejected || [];
  const newRejected = newReport.rejected || [];
  
  for (const newR of newRejected) {
    const oldR = oldRejected.find(r => r.scenario_id === newR.scenario_id);
    if (oldR) {
      if (oldR.sa_score !== newR.sa_score) {
        changes.push(`${newR.scenario_id} SA score: ${oldR.sa_score} → ${newR.sa_score}`);
      }
      if (oldR.binding_affinity !== newR.binding_affinity) {
        changes.push(`${newR.scenario_id} binding affinity: ${oldR.binding_affinity} → ${newR.binding_affinity} kcal/mol`);
      }
    }
  }
  
  return changes;
}

export async function POST(request: NextRequest) {
  try {
    const body: EditReportRequest = await request.json()
    const { structuredReport, editInstruction, context } = body

    if (!editInstruction?.trim()) {
      return NextResponse.json(
        { error: "Edit instruction is required" },
        { status: 400 }
      )
    }

    if (!structuredReport) {
      return NextResponse.json({
        error: "No report to edit",
        updatedReport: null,
        summary: "⚠️ No report data available to edit."
      }, { status: 200 })
    }

    console.log("[EditReport] ========================================")
    console.log("[EditReport] Processing edit:", editInstruction)
    console.log(`[EditReport] Mode: ${isDevMode() ? 'DEV (local agent)' : 'PROD (deployed agent)'}`)
    
    const experimentId = context?.taskId || `edit-${Date.now()}`;

    // =========================================================================
    // TRY DEPLOYED AGENT FIRST (Full Integration)
    // =========================================================================
    if (isDevMode() || isAgentexConfigured()) {
      console.log(`[EditReport] Calling ${AGENTS.SIMULATOR} agent...`);
      
      const agentResult = await processReportEdit({
        experiment_id: experimentId,
        edit_instruction: editInstruction,
        current_report: structuredReport as unknown as Record<string, unknown>,
      });

      if (agentResult.success && agentResult.data) {
        console.log(`[EditReport] Agent processed edit successfully`);
        console.log("[EditReport] ========================================")
        
        const updatedReport = agentResult.data.updated_report as StructuredReport;
        const changes = generateChangeSummary(structuredReport, updatedReport);
        const summary = changes.length > 0 
          ? `✓ ${changes.join("; ")}`
          : agentResult.data.summary || "✓ Report updated successfully.";

        return NextResponse.json({
          updatedReport,
          summary,
          _via: "deployed_agent",
        });
      }

      console.log(`[EditReport] Agent call failed: ${agentResult.error}, falling back to local LLM`);
      console.log(`[EditReport] Agent error details:`, JSON.stringify(agentResult.data || agentResult));
      // Include agent error in response for debugging
      var agentError = agentResult.error;
      var agentErrorDetails = agentResult.data;
    } else {
      console.log(`[EditReport] Agentex not configured, using local LLM fallback`);
    }

    // =========================================================================
    // FALLBACK: Local LLM processing (limited tracing)
    // =========================================================================
    if (!OPENAI_API_KEY) {
      return NextResponse.json({
        error: "LLM not configured. Please set OPENAI_API_KEY.",
        updatedReport: null,
        summary: "⚠️ Cannot process edit - LLM not configured."
      }, { status: 200 })
    }

    let updatedReport: StructuredReport;
    try {
      updatedReport = await processEditWithLLM(structuredReport, editInstruction);
      console.log("[EditReport] LLM processed edit successfully");
    } catch (parseError) {
      console.error("[EditReport] Failed to process edit:", parseError);
      return NextResponse.json({
        error: "Failed to process edit",
        updatedReport: null,
        summary: "⚠️ Error processing edit. Please try again."
      }, { status: 200 })
    }

    if (!updatedReport.executive_summary && !updatedReport.winner && !updatedReport.rejected) {
      return NextResponse.json({
        error: "Invalid report structure",
        updatedReport: null,
        summary: "⚠️ LLM returned invalid report structure."
      }, { status: 200 })
    }

    const changes = generateChangeSummary(structuredReport, updatedReport);
    const summary = changes.length > 0 
      ? `✓ ${changes.join("; ")}`
      : "✓ Report updated successfully.";

    console.log(`[EditReport] Edit applied. Summary: ${summary}`);
    console.log("[EditReport] ========================================")

    // Trace the edit (non-blocking)
    traceReportEdit(
      experimentId,
      editInstruction,
      structuredReport as unknown as Record<string, unknown>,
      updatedReport as unknown as Record<string, unknown>,
      changes
    ).catch(err => {
      console.warn("[EditReport] Trace failed (non-blocking):", err);
    });

    return NextResponse.json({
      updatedReport,
      summary,
      _via: "local_llm_fallback",
      _agent_error: agentError || null,
      _agent_error_details: agentErrorDetails || null,
    });

  } catch (error: unknown) {
    console.error("[EditReport] Unexpected error:", error);
    return NextResponse.json({
      error: error instanceof Error ? error.message : "Internal error",
      updatedReport: null,
      summary: "⚠️ Unexpected error occurred."
    }, { status: 200 })
  }
}
