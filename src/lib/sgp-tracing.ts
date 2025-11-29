/**
 * SGP Tracing Utility for SimuLab Frontend
 * 
 * Calls deployed agents through the Agentex API proxy to capture traces.
 * The agents are registered with SGP and handle tracing internally via adk.tracing.span.
 */

const SGP_API_KEY = process.env.SGP_API_KEY || "";
const SGP_ACCOUNT_ID = process.env.SGP_ACCOUNT_ID || "6887f093600ecd59bbbd3095";

// Use internal API proxy to call agents through SGP
// In Vercel, this will be the deployed URL; locally it's localhost
const getAgentProxyUrl = () => {
  if (typeof window !== 'undefined') {
    // Client-side
    return `${window.location.origin}/api/agentex`;
  }
  // Server-side - use environment variable or default
  return process.env.VERCEL_URL 
    ? `https://${process.env.VERCEL_URL}/api/agentex`
    : process.env.NEXT_PUBLIC_APP_URL 
      ? `${process.env.NEXT_PUBLIC_APP_URL}/api/agentex`
      : "http://localhost:3000/api/agentex";
};

interface TraceSpan {
  trace_id: string;
  name: string;
  input: Record<string, unknown>;
  output?: Record<string, unknown>;
  parent_id?: string;
  start_time?: string;
  end_time?: string;
}

interface TraceResult {
  success: boolean;
  trace_id: string;
  error?: string;
}

/**
 * Check if SGP tracing is configured
 */
export function isTracingEnabled(): boolean {
  return Boolean(SGP_API_KEY && SGP_ACCOUNT_ID);
}

/**
 * Generate a unique trace ID
 */
export function generateTraceId(prefix: string): string {
  const timestamp = new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14);
  const random = Math.random().toString(36).substring(2, 8);
  return `${prefix}_${timestamp}_${random}`;
}

/**
 * Call an agent's trace endpoint through the Agentex proxy
 * 
 * This calls the deployed agent's REST endpoint, which uses adk.tracing.span internally.
 */
async function callAgentTraceEndpoint(
  agentName: string,
  endpoint: string,
  payload: Record<string, unknown>
): Promise<TraceResult> {
  if (!isTracingEnabled()) {
    console.log("[SGP/Trace] Tracing not configured (missing SGP_API_KEY)");
    return { success: false, trace_id: "", error: "Tracing not configured" };
  }

  try {
    // Call the agent directly through the Agentex API
    // The proxy route handles SGP authentication
    const agentUrl = `${getAgentProxyUrl()}/agents/name/${agentName}${endpoint}`;
    
    console.log(`[SGP/Trace] Calling agent: ${agentName}${endpoint}`);

    const response = await fetch(agentUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      const result = await response.json();
      console.log(`[SGP/Trace] ✅ Agent trace captured: ${agentName}${endpoint}`);
      return { success: true, trace_id: result.trace_id || generateTraceId(agentName) };
    } else {
      const errorText = await response.text();
      console.warn(`[SGP/Trace] ❌ Agent call failed: ${response.status} - ${errorText}`);
      return { success: false, trace_id: "", error: errorText };
    }
  } catch (error) {
    console.error("[SGP/Trace] Error calling agent:", error);
    return { success: false, trace_id: "", error: String(error) };
  }
}

/**
 * Trace a report edit (Simulator agent trace)
 * 
 * Calls the Simulator agent's /trace/report_edit endpoint through SGP
 */
export async function traceReportEdit(data: {
  experimentId: string;
  editInstruction: string;
  originalReport: Record<string, unknown>;
  editedReport: Record<string, unknown>;
  changedFields?: string[];
}): Promise<TraceResult> {
  return callAgentTraceEndpoint("simulab-simulator", "/trace/report_edit", {
    experiment_id: data.experimentId,
    edit_instruction: data.editInstruction,
    original_report: data.originalReport,
    updated_report: data.editedReport,
    edit_summary: data.changedFields?.join(", ") || "",
  });
}

/**
 * Trace a design change (Simulator agent trace - per user spec)
 * 
 * Calls the Simulator agent's /trace/design_change endpoint through SGP
 */
export async function traceDesignChange(data: {
  experimentId: string;
  changeType: string;
  originalValue: unknown;
  newValue: unknown;
  reasoning?: string;
}): Promise<TraceResult> {
  return callAgentTraceEndpoint("simulab-simulator", "/trace/design_change", {
    experiment_id: data.experimentId,
    change_type: data.changeType,
    original_value: data.originalValue,
    new_value: data.newValue,
    reasoning: data.reasoning || "",
  });
}

/**
 * Trace a report feedback/edit (Judge agent trace)
 * 
 * Calls the Judge agent's /trace/report_feedback endpoint through SGP
 */
export async function traceReportFeedback(data: {
  experimentId: string;
  feedbackInstruction: string;
  originalReport: Record<string, unknown>;
  updatedReport: Record<string, unknown>;
}): Promise<TraceResult> {
  return callAgentTraceEndpoint("simulab-judge", "/trace/report_feedback", {
    experiment_id: data.experimentId,
    feedback_instruction: data.feedbackInstruction,
    original_report: data.originalReport,
    updated_report: data.updatedReport,
  });
}

/**
 * Trace a criteria re-evaluation (Judge agent trace)
 * 
 * Calls the Judge agent's /trace/reevaluation endpoint through SGP
 */
export async function traceCriteriaReevaluation(data: {
  experimentId: string;
  originalCriteria: Record<string, unknown>;
  newCriteria: Record<string, unknown>;
  originalVerdict: Record<string, unknown>;
  newVerdict: Record<string, unknown>;
}): Promise<TraceResult> {
  return callAgentTraceEndpoint("simulab-judge", "/trace/reevaluation", {
    experiment_id: data.experimentId,
    original_criteria: data.originalCriteria,
    new_criteria: data.newCriteria,
    original_verdict: data.originalVerdict,
    new_verdict: data.newVerdict,
  });
}

/**
 * Helper to summarize a report for tracing (avoid sending full report data)
 */
function summarizeReport(report: Record<string, unknown>): Record<string, unknown> {
  return {
    has_winner: Boolean(report.winner),
    num_selected: Array.isArray(report.selected) ? report.selected.length : 0,
    num_rejected: Array.isArray(report.rejected) ? report.rejected.length : 0,
    has_executive_summary: Boolean(report.executive_summary),
  };
}

