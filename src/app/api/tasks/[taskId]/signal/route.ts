import { NextRequest } from 'next/server';

const BASE_URL = process.env.AGENTEX_BASE_URL || 'http://localhost:5003';

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ taskId: string }> }
) {
  const { taskId } = await context.params;
  const body = await request.json();

  try {
    const response = await fetch(`${BASE_URL}/tasks/${taskId}/signal`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`Failed to send signal: ${response.statusText}`);
    }

    const data = await response.json();
    return new Response(JSON.stringify(data), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error sending signal:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to send signal' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
} 