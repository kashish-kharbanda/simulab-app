import { NextRequest, NextResponse } from 'next/server';

const BASE_URL = process.env.AGENTEX_BASE_URL || 'http://localhost:5003';

export async function GET(
  _: NextRequest,
  context: { params: Promise<{ taskId: string }> }
) {
  const { taskId } = await context.params;

  if (!taskId) {
    return NextResponse.json({ error: 'Task ID is required' }, { status: 400 });
  }

  try {
    const response = await fetch(`${BASE_URL}/messages?task_id=${taskId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch messages for task ${taskId}`);
    }
    const messages = await response.json();
    return NextResponse.json(messages);
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}

export const revalidate = 0; 