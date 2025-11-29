import { NextRequest, NextResponse } from 'next/server';

/**
 * SimuLab Messages API Route
 * 
 * Proxies message requests to the central AgentEx backend.
 * Returns all messages/events for a given task ID.
 */

// Disable caching for real-time message polling
export const revalidate = 0;
export const dynamic = 'force-dynamic';

// Environment configuration
const AGENTEX_BASE_URL = process.env.AGENTEX_BASE_URL || 'http://localhost:5003';

/** Message entry from AgentEx backend */
interface MessageEntry {
  id?: string;
  task_id: string;
  type: string;
  content?: unknown;
  timestamp?: string;
  agent_name?: string;
}

export async function GET(request: NextRequest) {
  const taskId = request.nextUrl.searchParams.get('task_id');
  
  if (!taskId) {
    console.warn('[SimuLab/Messages] Missing required parameter: task_id');
    return NextResponse.json(
      { error: 'task_id is required' }, 
      { status: 400 }
    );
  }

  try {
    const messagesUrl = `${AGENTEX_BASE_URL}/messages?task_id=${encodeURIComponent(taskId)}`;
    
    const res = await fetch(messagesUrl, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      cache: 'no-store',
    });
    
    if (!res.ok) {
      const errorText = await res.text();
      console.error(`[SimuLab/Messages] Backend error (${res.status}):`, errorText);
      
      let errorDetails: unknown;
      try {
        errorDetails = JSON.parse(errorText);
      } catch {
        errorDetails = errorText;
      }
      
      return NextResponse.json(
        { error: `Failed to fetch messages: ${res.status}`, details: errorDetails },
        { status: res.status }
      );
    }
    
    const messages: MessageEntry[] = await res.json();
    return NextResponse.json(messages);
    
  } catch (error: unknown) {
    console.error('[SimuLab/Messages] Unexpected error:', error);
    
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Internal server error',
      }, 
      { status: 500 }
    );
  }
}


