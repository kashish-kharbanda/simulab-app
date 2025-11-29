import { NextResponse } from 'next/server';

/**
 * SimuLab Agents Health Check API Route
 * 
 * Checks the health status of all SimuLab agents via the AgentEx backend.
 * Returns registry status and ACP health for each agent.
 */

// Disable caching for real-time health checks
export const revalidate = 0;
export const dynamic = 'force-dynamic';

// Environment configuration
const AGENTEX_BASE_URL = process.env.AGENTEX_BASE_URL || 'http://localhost:5003';

// SimuLab agent names
const SIMULAB_AGENTS = [
  'simulab-orchestrator',
  'simulab-simulator',
  'simu-docking',
  'simu-admet',
  'simu-synthesis',
  'simulab-judge',
] as const;

type AgentName = typeof SIMULAB_AGENTS[number];

/** Agent health status */
interface AgentHealthStatus {
  name: string;
  status?: string;
  acp_url?: string;
  acp_healthy?: boolean;
  error?: string;
}

/** Registry agent entry */
interface RegistryAgent {
  name?: string;
  status?: string;
  health?: string;
  state?: string;
  acp_url?: string;
  acp?: { url?: string };
  url?: string;
}

/**
 * Fetch JSON with timeout
 */
async function fetchJson<T = unknown>(
  url: string, 
  init?: RequestInit, 
  timeoutMs = 1500
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const res = await fetch(url, { 
      ...init, 
      signal: controller.signal,
      cache: 'no-store',
    });
    
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    
    return await res.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Check ACP health endpoint
 */
async function checkAcpHealth(acpUrl: string): Promise<boolean> {
  try {
    const healthUrl = acpUrl.replace(/\/+$/, '') + '/healthz';
    const res = await fetch(healthUrl, { 
      method: 'GET', 
      cache: 'no-store',
      signal: AbortSignal.timeout(1000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function GET() {
  console.log('[SimuLab/Health] Checking agent health status...');
  
  try {
    // Fetch agent registry
    let registryAgents: RegistryAgent[] = [];
    try {
      const registry = await fetchJson<{ agents?: RegistryAgent[] } | RegistryAgent[]>(
        `${AGENTEX_BASE_URL}/agents`
      );
      registryAgents = Array.isArray(registry) 
        ? registry 
        : (Array.isArray(registry?.agents) ? registry.agents : []);
    } catch (err) {
      console.warn('[SimuLab/Health] Failed to fetch registry:', err);
    }
    
    // Build lookup map
    const agentsByName = new Map<string, RegistryAgent>();
    for (const agent of registryAgents) {
      const key = String(agent?.name || '').toLowerCase();
      if (key) agentsByName.set(key, agent);
    }

    // Check each SimuLab agent
    const results: AgentHealthStatus[] = [];
    
    for (const name of SIMULAB_AGENTS) {
      const status: AgentHealthStatus = { name };
      const key = name.toLowerCase();
      
      let agent = agentsByName.get(key);
      
      // Fallback: try individual agent endpoint
      if (!agent) {
        try {
          const response = await fetchJson<{ result?: RegistryAgent } | RegistryAgent>(
            `${AGENTEX_BASE_URL}/agents/name/${encodeURIComponent(name)}`
          );
          agent = (response as { result?: RegistryAgent })?.result || response as RegistryAgent;
        } catch (err) {
          status.error = `registry: ${err instanceof Error ? err.message : 'failed'}`;
        }
      }
      
      if (agent) {
        status.status = agent.status || agent.health || agent.state;
        status.acp_url = agent.acp_url || agent.acp?.url || agent.url;
      }
      
      // Check ACP health if URL available
      if (status.acp_url) {
        status.acp_healthy = await checkAcpHealth(status.acp_url);
      }
      
      results.push(status);
    }
    
    const healthyCount = results.filter(r => r.acp_healthy).length;
    console.log(`[SimuLab/Health] ${healthyCount}/${results.length} agents healthy`);
    
    return NextResponse.json({ agents: results }, { status: 200 });
    
  } catch (error: unknown) {
    console.error('[SimuLab/Health] Unexpected error:', error);
    
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' }, 
      { status: 500 }
    );
  }
}


