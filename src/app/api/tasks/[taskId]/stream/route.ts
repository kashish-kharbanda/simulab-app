import { NextRequest } from 'next/server';

const BASE_URL = process.env.AGENTEX_BASE_URL || 'http://localhost:5003';

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ taskId: string }> }
) {
  const { taskId } = await context.params;

  console.log(`SSE stream requested for task: ${taskId}`);

  try {
    const response = await fetch(`${BASE_URL}/tasks/${taskId}/stream`, {
      headers: {
        'Accept': 'text/event-stream',
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to connect to stream: ${response.statusText}`);
    }

    return new Response(response.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    });
  } catch (error) {
    console.error('Error setting up stream:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to connect to stream' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}

export const dynamic = 'force-dynamic';
export const dynamicParams = true;
export const runtime = 'nodejs';
export const preferredRegion = 'auto'; 