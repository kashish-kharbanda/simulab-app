import { NextResponse } from 'next/server';
import { isDevMode, isAgentexConfigured, AGENTS } from '@/lib/agent-client';

/**
 * Debug endpoint to check environment configuration
 * DELETE THIS AFTER DEBUGGING
 */
export const dynamic = 'force-dynamic';

export async function GET() {
  const config = {
    // Check which mode we're in
    agent_mode: process.env.AGENT_MODE || '(not set, defaults to prod)',
    is_dev_mode: isDevMode(),
    is_agentex_configured: isAgentexConfigured(),
    
    // Check env vars (masked for security)
    env_vars: {
      AGENTEX_SDK_API_KEY: process.env.AGENTEX_SDK_API_KEY ? `${process.env.AGENTEX_SDK_API_KEY.substring(0, 8)}...` : '(not set)',
      SGP_API_KEY: process.env.SGP_API_KEY ? `${process.env.SGP_API_KEY.substring(0, 8)}...` : '(not set)',
      NEXT_PUBLIC_ACCOUNT_ID: process.env.NEXT_PUBLIC_ACCOUNT_ID || '(not set)',
      SGP_ACCOUNT_ID: process.env.SGP_ACCOUNT_ID || '(not set)',
      NEXT_PUBLIC_AGENTEX_API_BASE_URL: process.env.NEXT_PUBLIC_AGENTEX_API_BASE_URL || '(not set)',
      AGENTEX_BASE_URL: process.env.AGENTEX_BASE_URL || '(not set)',
      OPENAI_API_KEY: process.env.OPENAI_API_KEY ? 'set' : '(not set)',
    },
    
    // Agent names
    agents: AGENTS,
    
    // What URL would be called
    example_url: isAgentexConfigured() 
      ? `${process.env.NEXT_PUBLIC_AGENTEX_API_BASE_URL || 'https://agentex.agentex.azure.workspace.egp.scale.com'}/agents/forward/name/${AGENTS.SIMULATOR}/process_edit`
      : 'Would use local LLM fallback',
  };

  return NextResponse.json(config);
}

