import { NextRequest, NextResponse } from 'next/server';

/**
 * SimuLab Task Creation API Route
 * 
 * Creates a new virtual lab task via the AgentEx backend orchestrator.
 * Accepts protein target, seed molecule, and optional user-edited lab design.
 */

// Allow longer execution for task creation
export const maxDuration = 60;

// Environment configuration
const AGENTEX_BASE_URL = process.env.AGENTEX_BASE_URL || 'http://localhost:5003';
const ORCH_AGENT_NAME = process.env.SIMULAB_ORCH_AGENT_NAME || 'simulab-orchestrator';

/** Request body interface for task creation */
interface CreateTaskRequest {
  protein_target: string;
  seed_molecule?: string;
  num_scenarios?: number;
  name?: string;
  goal?: string;
  constraints?: string[];
  scenarios?: Array<{
    scenario_id: string;
    scaffold?: string;
    smiles?: string;
    rationale?: string;
  }>;
}

/** Response from AgentEx RPC task/create */
interface CreateTaskResponse {
  result?: { id: string };
  id?: string;
  error?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: CreateTaskRequest = await request.json();
    const { 
      protein_target, 
      seed_molecule, 
      num_scenarios = 3, 
      name,
      goal,
      constraints,
      scenarios,
    } = body;

    // Validate required fields
    if (!protein_target) {
      console.warn('[SimuLab/Create] Missing required field: protein_target');
      return NextResponse.json(
        { error: 'protein_target is required' }, 
        { status: 400 }
      );
    }

    console.log(`[SimuLab/Create] Creating task for target: ${protein_target}`);
    console.log(`[SimuLab/Create] Orchestrator: ${ORCH_AGENT_NAME}`);
    console.log(`[SimuLab/Create] Scenarios: ${scenarios?.length || num_scenarios}`);

    // Build RPC payload
    const rpcPayload = {
      method: 'task/create',
      params: {
        name: name || `SimuLab ${protein_target}`,
        params: {
          protein_target,
          seed_molecule,
          num_scenarios,
          goal,
          constraints,
          scenarios,
        },
      },
    };

    const rpcUrl = `${AGENTEX_BASE_URL}/agents/name/${encodeURIComponent(ORCH_AGENT_NAME)}/rpc`;
    console.log(`[SimuLab/Create] RPC URL: ${rpcUrl}`);

    const createRes = await fetch(rpcUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(rpcPayload),
    });

    if (!createRes.ok) {
      const errorText = await createRes.text();
      console.error(`[SimuLab/Create] RPC error (${createRes.status}):`, errorText);
      
      let errorDetails: unknown;
      try {
        errorDetails = JSON.parse(errorText);
      } catch {
        errorDetails = errorText;
      }
      
      return NextResponse.json(
        { error: `Failed to create task: ${createRes.status}`, details: errorDetails },
        { status: createRes.status }
      );
    }

    const created: CreateTaskResponse = await createRes.json();
    const taskId = created?.result?.id || created?.id;
    
    if (!taskId) {
      console.error('[SimuLab/Create] No task ID in response:', created);
      return NextResponse.json(
        { error: 'No task ID returned from orchestrator' },
        { status: 500 }
      );
    }

    console.log(`[SimuLab/Create] Task created successfully: ${taskId}`);
    return NextResponse.json({ id: taskId });
    
  } catch (error: unknown) {
    console.error('[SimuLab/Create] Unexpected error:', error);
    
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Internal server error',
        details: error instanceof Error ? error.stack : undefined,
      }, 
      { status: 500 }
    );
  }
}


