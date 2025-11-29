/**
 * SimuLab Agent Client
 * 
 * Connects frontend to deployed SimuLab agents via SGP/Agentex API.
 * Uses the EXACT same pattern as Subrogation demo.
 * 
 * Agents:
 * - simulab-orchestrator: Designs experiments, coordinates workflow
 * - simulab-simulator: Evaluates molecules (docking, ADMET, synthesis)
 * - simulab-judge: Multi-objective analysis and verdict generation
 * 
 * Environment variables (set in deployment, same as Subrogation):
 * - AGENT_MODE: 'dev' for local, 'prod' for deployed (default: prod)
 * - AGENTEX_SDK_API_KEY: API key for SGP/Agentex
 * - NEXT_PUBLIC_ACCOUNT_ID: Account ID
 * - NEXT_PUBLIC_AGENTEX_API_BASE_URL: Agentex API base URL
 */

// Agent names as deployed in SGP
export const AGENTS = {
  ORCHESTRATOR: 'simulab-orchestrator',
  SIMULATOR: 'simulab-simulator',
  JUDGE: 'simulab-judge',
} as const;

export type AgentName = typeof AGENTS[keyof typeof AGENTS];

// Local development ports (same pattern as Subrogation)
const AGENT_PORTS: Record<AgentName, string> = {
  [AGENTS.ORCHESTRATOR]: process.env.ORCHESTRATOR_PORT || '8003',
  [AGENTS.SIMULATOR]: process.env.SIMULATOR_PORT || '8001',
  [AGENTS.JUDGE]: process.env.JUDGE_PORT || '8002',
};

/**
 * Check if we're in development mode
 */
export function isDevMode(): boolean {
  return process.env.AGENT_MODE === 'dev';
}

/**
 * Get API key - supports both naming conventions
 */
function getApiKey(): string | undefined {
  return process.env.AGENTEX_SDK_API_KEY || process.env.SGP_API_KEY;
}

/**
 * Get Account ID - supports both naming conventions
 */
function getAccountId(): string | undefined {
  return process.env.NEXT_PUBLIC_ACCOUNT_ID || process.env.SGP_ACCOUNT_ID;
}

/**
 * Get Base URL - with default fallback
 * Note: The correct URL is agentex.agentex (NOT api.agentex)
 */
function getBaseUrl(): string {
  return process.env.NEXT_PUBLIC_AGENTEX_API_BASE_URL || 
         process.env.AGENTEX_BASE_URL || 
         'https://agentex.agentex.azure.workspace.egp.scale.com';
}

/**
 * Check if Agentex is properly configured for production
 */
export function isAgentexConfigured(): boolean {
  const apiKey = getApiKey();
  const accountId = getAccountId();
  return Boolean(apiKey && accountId);
}

/**
 * Get the URL and headers for an agent endpoint
 * Follows exact same pattern as Subrogation demo
 */
function getAgentConfig(agent: AgentName, endpoint: string): {
  url: string;
  headers: Record<string, string>;
} {
  const agentMode = process.env.AGENT_MODE || 'prod';
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };

  if (agentMode === 'dev') {
    // Development mode - connect directly to local agent
    const port = AGENT_PORTS[agent];
    console.log(`[${agent}] Dev mode: Connecting to local agent on port ${port}`);
    return {
      url: `http://localhost:${port}${endpoint}`,
      headers,
    };
  } else {
    // Production mode - use deployed agent via Forward API
    const apiKey = getApiKey();
    const accountId = getAccountId();
    const baseURL = getBaseUrl();

    if (apiKey) headers['x-api-key'] = apiKey;
    if (accountId) headers['x-account-id'] = accountId;

    console.log(`[${agent}] Prod mode: Base URL: ${baseURL}`);
    return {
      url: `${baseURL}/agents/forward/name/${agent}${endpoint}`,
      headers,
    };
  }
}

/**
 * Call an agent endpoint directly (for synchronous operations)
 * Same pattern as Subrogation demo
 */
export async function callAgent<T = unknown>(
  agent: AgentName,
  endpoint: string,
  payload?: Record<string, unknown>,
  method: 'GET' | 'POST' = 'POST'
): Promise<{ success: boolean; data?: T; error?: string }> {
  const agentMode = process.env.AGENT_MODE || 'prod';
  
  // Check configuration in prod mode
  if (agentMode !== 'dev' && !isAgentexConfigured()) {
    console.error(`[${agent}] AgentEx configuration is missing`);
    return { success: false, error: 'AgentEx configuration is missing' };
  }

  const { url, headers } = getAgentConfig(agent, endpoint);
  
  console.log(`[${agent}] ${method} ${url}`);

  try {
    const response = await fetch(url, {
      method,
      headers,
      body: method === 'POST' && payload ? JSON.stringify(payload) : undefined,
    });

    console.log(`[${agent}] Response status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[${agent}] Agent error (${response.status}):`, errorText);
      
      let errorDetails;
      try {
        errorDetails = JSON.parse(errorText);
      } catch {
        errorDetails = errorText;
      }
      
      return { 
        success: false, 
        error: `Agent error: ${response.status}`,
        data: errorDetails as T,
      };
    }

    const data = await response.json();
    return { success: true, data: data as T };
  } catch (error) {
    console.error(`[${agent}] Error:`, error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    };
  }
}

/**
 * Call an agent with polling for long-running operations
 * Same pattern as Subrogation demo (start + poll status)
 */
export async function callAgentWithPolling<T = unknown>(
  agent: AgentName,
  startEndpoint: string,
  statusEndpointBase: string,
  payload: Record<string, unknown>,
  idField: string = 'id',
  maxAttempts: number = 60,
  pollIntervalMs: number = 3000
): Promise<{ success: boolean; data?: T; error?: string }> {
  const agentMode = process.env.AGENT_MODE || 'prod';
  
  // Check configuration in prod mode
  if (agentMode !== 'dev' && !isAgentexConfigured()) {
    return { success: false, error: 'AgentEx configuration is missing' };
  }

  // Step 1: Start processing
  const startConfig = getAgentConfig(agent, startEndpoint);
  console.log(`[${agent}] Starting: ${startConfig.url}`);

  try {
    const startResponse = await fetch(startConfig.url, {
      method: 'POST',
      headers: startConfig.headers,
      body: JSON.stringify(payload),
    });

    console.log(`[${agent}] Start response status: ${startResponse.status}`);

    if (!startResponse.ok) {
      const errorText = await startResponse.text();
      console.error(`[${agent}] Agent error (${startResponse.status}):`, errorText);
      return { success: false, error: `Failed to start: ${startResponse.status}` };
    }

    const startResult = await startResponse.json();
    const taskId = startResult[idField] || payload[idField];
    
    if (!taskId) {
      // Synchronous response - return directly
      console.log(`[${agent}] Synchronous response received`);
      return { success: true, data: startResult as T };
    }

    console.log(`[${agent}] Processing started! ID: ${taskId}`);

    // Step 2: Poll for completion
    const statusConfig = getAgentConfig(agent, `${statusEndpointBase}/${taskId}`);
    let attempts = 0;

    while (attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
      attempts++;

      console.log(`[${agent}] Polling status (attempt ${attempts}/${maxAttempts})...`);

      // For status polling, use GET with appropriate headers
      const statusHeaders: Record<string, string> = {};
      if (agentMode !== 'dev') {
        const apiKey = getApiKey();
        const accountId = getAccountId();
        if (apiKey) statusHeaders['x-api-key'] = apiKey;
        if (accountId) statusHeaders['x-account-id'] = accountId;
      }

      const statusResponse = await fetch(statusConfig.url, {
        method: 'GET',
        headers: statusHeaders,
      });

      if (!statusResponse.ok) {
        console.error(`[${agent}] Status check failed: ${statusResponse.status}`);
        continue;
      }

      const statusData = await statusResponse.json();
      console.log(`[${agent}] Status: ${statusData.status}, Progress: ${statusData.progress}%, Message: ${statusData.message}`);

      if (statusData.status === 'completed') {
        console.log(`[${agent}] Processing complete!`);
        return { success: true, data: statusData.data as T };
      }

      if (statusData.status === 'error') {
        console.error(`[${agent}] Processing failed: ${statusData.message}`);
        return { success: false, error: statusData.error || statusData.message };
      }
    }

    // Timeout
    console.error(`[${agent}] Polling timeout after ${maxAttempts} attempts`);
    return { success: false, error: 'Processing timeout - please try again later' };

  } catch (error) {
    console.error(`[${agent}] Error:`, error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    };
  }
}

// ============================================================================
// SIMULATOR AGENT FUNCTIONS
// ============================================================================

export interface EvaluateMoleculesRequest {
  experiment_id: string;
  protein_target: string;
  scenarios: Array<{
    scenario_id: string;
    smiles: string;
    scaffold: string;
  }>;
}

export interface EvaluateMoleculesResponse {
  success: boolean;
  experiment_id: string;
  results: Array<{
    scenario_id: string;
    smiles: string;
    scaffold: string;
    docking: {
      binding_affinity_kcal_per_mol: number;
      potency_pass: boolean;
    };
    admet: {
      toxicity_risk: string;
      herg_flag: boolean;
      is_safe: boolean;
    };
    synthesis: {
      sa_score: number;
      estimated_cost_usd: number;
    };
    data_source: string;
    confidence: string;
  }>;
}

/**
 * Call Simulator agent to evaluate multiple molecules
 * Uses the /evaluate_molecule endpoint (singular) for each scenario in parallel
 */
export async function evaluateMolecules(
  request: EvaluateMoleculesRequest
): Promise<{ success: boolean; data?: EvaluateMoleculesResponse; error?: string }> {
  console.log(`[Simulator] Evaluating ${request.scenarios.length} molecules in parallel...`);
  
  // Call /evaluate_molecule for each scenario in parallel
  const evaluationPromises = request.scenarios.map(async (scenario) => {
    const result = await callAgent<{
      success: boolean;
      smiles: string;
      protein_target: string;
      metrics: {
        docking: { binding_affinity_kcal_per_mol: number; potency_pass: boolean };
        admet: { toxicity_risk: string; toxicity_prob: number; herg_flag: boolean; is_safe: boolean };
        synthesis: { sa_score: number; num_steps: number; estimated_cost_usd: number };
        _source?: string;
      };
    }>(
      AGENTS.SIMULATOR,
      '/evaluate_molecule',
      {
        smiles: scenario.smiles,
        scaffold: scenario.scaffold,
        protein_target: request.protein_target,
      }
    );

    if (!result.success || !result.data) {
      console.error(`[Simulator] Failed to evaluate ${scenario.scenario_id}: ${result.error}`);
      return null;
    }

    const metrics = result.data.metrics;
    return {
      scenario_id: scenario.scenario_id,
      smiles: scenario.smiles,
      scaffold: scenario.scaffold,
      docking: metrics.docking,
      admet: {
        toxicity_risk: metrics.admet.toxicity_risk,
        herg_flag: metrics.admet.herg_flag,
        is_safe: metrics.admet.is_safe,
      },
      synthesis: {
        sa_score: metrics.synthesis.sa_score,
        estimated_cost_usd: metrics.synthesis.estimated_cost_usd,
      },
      data_source: metrics._source || 'agent',
      confidence: 'HIGH',
    };
  });

  const results = await Promise.all(evaluationPromises);
  const validResults = results.filter((r): r is NonNullable<typeof r> => r !== null);

  if (validResults.length === 0) {
    return { success: false, error: 'All molecule evaluations failed' };
  }

  return {
    success: true,
    data: {
      success: true,
      experiment_id: request.experiment_id,
      results: validResults,
    },
  };
}

export interface ProcessEditRequest {
  experiment_id: string;
  edit_instruction: string;
  current_report: Record<string, unknown>;
}

export interface ProcessEditResponse {
  success: boolean;
  updated_report: Record<string, unknown>;
  summary: string;
  changes: string[];
}

/**
 * Call Simulator agent to process a report edit
 */
export async function processReportEdit(
  request: ProcessEditRequest
): Promise<{ success: boolean; data?: ProcessEditResponse; error?: string }> {
  return callAgent<ProcessEditResponse>(
    AGENTS.SIMULATOR,
    '/process_edit',
    request as unknown as Record<string, unknown>
  );
}

// ============================================================================
// ORCHESTRATOR AGENT FUNCTIONS
// ============================================================================

export interface DesignExperimentRequest {
  prompt: string;
  constraints?: string;
}

export interface DesignExperimentResponse {
  success: boolean;
  goal: string;
  protein_target: string;
  constraints: string[];
  scenarios: Array<{
    scenario_id: string;
    scaffold: string;
    smiles: string;
    rationale?: string;
  }>;
  suggested_num_scenarios: number;
  data_source: string;
  confidence: string;
}

/**
 * Call Orchestrator agent to design an experiment
 * Uses polling pattern for long-running operations
 */
export async function designExperiment(
  request: DesignExperimentRequest
): Promise<{ success: boolean; data?: DesignExperimentResponse; error?: string }> {
  // Try polling pattern first
  const startResult = await callAgent<{ job_id: string; status: string }>(
    AGENTS.ORCHESTRATOR,
    '/start_design',
    request as unknown as Record<string, unknown>
  );

  if (startResult.success && startResult.data?.job_id) {
    // Poll for completion
    return pollForResult<DesignExperimentResponse>(
      AGENTS.ORCHESTRATOR,
      `/design_status/${startResult.data.job_id}`,
      60, // max attempts
      2000 // poll interval ms
    );
  }

  // Fallback to synchronous call (may timeout)
  console.log('[Orchestrator] Polling not available, trying synchronous call');
  return callAgent<DesignExperimentResponse>(
    AGENTS.ORCHESTRATOR,
    '/design_experiment',
    request as unknown as Record<string, unknown>
  );
}

/**
 * Generic polling function for long-running agent operations
 */
async function pollForResult<T>(
  agent: AgentName,
  statusEndpoint: string,
  maxAttempts: number = 60,
  pollIntervalMs: number = 2000
): Promise<{ success: boolean; data?: T; error?: string }> {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
    
    const statusResult = await callAgent<{
      status: string;
      progress: number;
      message: string;
      data: T | null;
      error: string | null;
    }>(agent, statusEndpoint, undefined, 'GET');

    if (!statusResult.success) {
      console.log(`[${agent}] Status check failed, retrying...`);
      continue;
    }

    const status = statusResult.data;
    if (!status) continue;

    console.log(`[${agent}] Status: ${status.status}, Progress: ${status.progress}%`);

    if (status.status === 'completed' && status.data) {
      return { success: true, data: status.data };
    }

    if (status.status === 'error') {
      return { success: false, error: status.error || status.message };
    }
  }

  return { success: false, error: 'Polling timeout' };
}

// ============================================================================
// JUDGE AGENT FUNCTIONS
// ============================================================================

export interface GenerateVerdictRequest {
  experiment_id: string;
  protein_target: string;
  scenarios: Array<{
    scenario_id: string;
    scaffold: string;
    smiles: string;
    binding_affinity: number;
    herg_flag: boolean;
    sa_score: number;
    cost_usd: number;
  }>;
  decision_criteria: {
    herg_veto: boolean;
    potency_threshold: number;
    sa_threshold: number;
  };
  goal?: string;
  constraints?: string[];
}

export interface GenerateVerdictResponse {
  success: boolean;
  verdict: {
    winner: Record<string, unknown> | null;
    selected: Array<Record<string, unknown>>;
    rejected: Array<Record<string, unknown>>;
  };
  executive_summary: string;
  comparative_analysis: string;
  data_source: string;
  confidence: string;
}

/**
 * Call Judge agent to generate verdict
 * Uses polling pattern for long-running operations
 */
export async function generateVerdict(
  request: GenerateVerdictRequest
): Promise<{ success: boolean; data?: GenerateVerdictResponse; error?: string }> {
  // Try polling pattern first
  const startResult = await callAgent<{ job_id: string; status: string }>(
    AGENTS.JUDGE,
    '/start_verdict',
    request as unknown as Record<string, unknown>
  );

  if (startResult.success && startResult.data?.job_id) {
    // Poll for completion
    return pollForResult<GenerateVerdictResponse>(
      AGENTS.JUDGE,
      `/verdict_status/${startResult.data.job_id}`,
      60, // max attempts
      2000 // poll interval ms
    );
  }

  // Fallback to synchronous call (may timeout)
  console.log('[Judge] Polling not available, trying synchronous call');
  return callAgent<GenerateVerdictResponse>(
    AGENTS.JUDGE,
    '/generate_verdict',
    request as unknown as Record<string, unknown>
  );
}

/**
 * Call Judge agent to re-evaluate with new criteria
 */
export async function reevaluateWithCriteria(
  experimentId: string,
  scenarios: Array<Record<string, unknown>>,
  decisionCriteria: Record<string, unknown>,
  proteinTarget?: string,
  goal?: string,
  constraints?: string[]
): Promise<{ success: boolean; data?: GenerateVerdictResponse; error?: string }> {
  return callAgent<GenerateVerdictResponse>(
    AGENTS.JUDGE,
    '/reevaluate',
    {
      experiment_id: experimentId,
      protein_target: proteinTarget,
      scenarios,
      decision_criteria: decisionCriteria,
      goal,
      constraints,
    }
  );
}

// ============================================================================
// TRACING FUNCTIONS (for audit logging)
// ============================================================================

/**
 * Trace a design change (Review & Confirm screen)
 */
export async function traceDesignChange(
  experimentId: string,
  changeType: string,
  originalValue: unknown,
  newValue: unknown,
  reasoning: string
): Promise<{ success: boolean; trace_id?: string; error?: string }> {
  return callAgent(
    AGENTS.SIMULATOR,
    '/trace/design_change',
    {
      experiment_id: experimentId,
      change_type: changeType,
      original_value: originalValue,
      new_value: newValue,
      reasoning,
    }
  );
}

/**
 * Trace a report edit (Results page chat)
 */
export async function traceReportEdit(
  experimentId: string,
  editInstruction: string,
  originalReport: Record<string, unknown>,
  updatedReport: Record<string, unknown>,
  changedFields?: string[]
): Promise<{ success: boolean; trace_id?: string; error?: string }> {
  return callAgent(
    AGENTS.SIMULATOR,
    '/trace/report_edit',
    {
      experiment_id: experimentId,
      edit_instruction: editInstruction,
      original_report: originalReport,
      updated_report: updatedReport,
      changed_fields: changedFields,
    }
  );
}

/**
 * Trace report feedback (Results page chat - traced under Judge)
 */
export async function traceReportFeedback(
  experimentId: string,
  feedbackInstruction: string,
  originalReport: Record<string, unknown>,
  updatedReport: Record<string, unknown>,
  feedbackSummary: string
): Promise<{ success: boolean; trace_id?: string; error?: string }> {
  return callAgent(
    AGENTS.JUDGE,
    '/trace/report_feedback',
    {
      experiment_id: experimentId,
      feedback_instruction: feedbackInstruction,
      original_report: originalReport,
      updated_report: updatedReport,
      feedback_summary: feedbackSummary,
    }
  );
}
