import { NextRequest, NextResponse } from "next/server"
import { isDevMode, AGENTS } from "@/lib/agent-client"

/**
 * SimuLab Trace Design Change API Route
 * 
 * Traces design changes made in the Review & Confirm screen.
 * Calls the deployed Simulator agent's /trace/design_change endpoint.
 * 
 * NOTE: Requires agent redeployment to have this endpoint available.
 */

export const maxDuration = 60;
export const dynamic = "force-dynamic"

// Get environment variables (server-side only)
function getApiKey(): string | undefined {
  return process.env.AGENTEX_SDK_API_KEY || process.env.SGP_API_KEY;
}

function getAccountId(): string | undefined {
  return process.env.NEXT_PUBLIC_ACCOUNT_ID || process.env.SGP_ACCOUNT_ID;
}

function getBaseUrl(): string {
  return process.env.NEXT_PUBLIC_AGENTEX_API_BASE_URL || 
         process.env.AGENTEX_BASE_URL || 
         'https://agentex.agentex.azure.workspace.egp.scale.com';
}

interface TraceDesignChangeRequest {
  experiment_id: string;
  change_type: string;
  original_value?: unknown;
  new_value?: unknown;
  reasoning: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: TraceDesignChangeRequest = await request.json();
    const { experiment_id, change_type, original_value, new_value, reasoning } = body;

    if (!experiment_id || !change_type) {
      return NextResponse.json(
        { success: false, error: "experiment_id and change_type are required" },
        { status: 400 }
      );
    }

    console.log(`[TraceDesignChange] ========================================`);
    console.log(`[TraceDesignChange] Tracing: ${change_type} for ${experiment_id}`);
    console.log(`[TraceDesignChange] Reasoning: ${reasoning?.substring(0, 100)}...`);
    console.log(`[TraceDesignChange] Mode: ${isDevMode() ? 'DEV' : 'PROD'}`);

    const apiKey = getApiKey();
    const accountId = getAccountId();
    const baseUrl = getBaseUrl();

    if (!apiKey || !accountId) {
      console.log(`[TraceDesignChange] AgentEx not configured, skipping trace`);
      return NextResponse.json({
        success: false,
        error: "AgentEx configuration is missing",
        _via: "skipped",
      });
    }

    // Call the deployed Simulator agent's /trace/design_change endpoint
    const agentUrl = `${baseUrl}/agents/forward/name/${AGENTS.SIMULATOR}/trace/design_change`;
    
    console.log(`[TraceDesignChange] Calling: ${agentUrl}`);

    const response = await fetch(agentUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'x-account-id': accountId,
      },
      body: JSON.stringify({
        experiment_id,
        change_type,
        reasoning,
        // Note: original_value and new_value are omitted as they cause issues with deployed agent
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[TraceDesignChange] Agent error: ${response.status} - ${errorText}`);
      return NextResponse.json({
        success: false,
        error: `Agent error: ${response.status}`,
        error_details: errorText,
        agent_url: agentUrl,
        request_body: { experiment_id, change_type, reasoning },
        _via: "agent_error",
      });
    }

    const result = await response.json();
    console.log(`[TraceDesignChange] Success: trace_id=${result.trace_id}`);
    console.log(`[TraceDesignChange] ========================================`);

    return NextResponse.json({
      success: true,
      trace_id: result.trace_id,
      _via: "deployed_agent",
    });

  } catch (error: unknown) {
    console.error("[TraceDesignChange] Unexpected error:", error);
    return NextResponse.json({
      success: false,
      error: `Trace failed: ${error instanceof Error ? error.message : "Unknown"}`,
    }, { status: 500 });
  }
}

