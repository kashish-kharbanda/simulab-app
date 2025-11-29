import { NextRequest, NextResponse } from 'next/server';

/**
 * API route to fetch spans data from the backend service
 */
export const runtime = 'edge';
export const preferredRegion = ['auto'];
export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const traceId = searchParams.get('trace_id');

  if (!traceId) {
    console.error('Missing required trace_id parameter');
    return NextResponse.json({ error: 'Missing required trace_id parameter' }, { status: 400 });
  }

  // Log the trace ID being fetched
  console.log(`Fetching spans for trace_id: ${traceId}`);

  // Construct the API URL from environment variables, defaulting to localhost
  const apiUrl = process.env.AGENTEX_BASE_URL || 'http://localhost:5003';

  try {
    // Fetch spans from backend API
    const response = await fetch(`${apiUrl}/spans?trace_id=${traceId}`, {
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        Pragma: 'no-cache',
        Expires: '0',
      },
      // Ensure we're bypassing any cache
      cache: 'no-store',
    });

    // Log the response status
    console.log(`API response status: ${response.status}`);

    if (!response.ok) {
      if (response.status === 404) {
        return NextResponse.json(
          { error: 'No spans found for the specified trace ID' },
          { status: 404 },
        );
      }

      const errorText = await response.text();
      console.error(`Error fetching spans: ${errorText}`);

      return NextResponse.json(
        { error: `Failed to fetch spans: ${response.statusText}` },
        { status: response.status },
      );
    }

    // Parse and return the spans data
    const spansData = await response.json();
    return NextResponse.json(spansData, {
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        Pragma: 'no-cache',
        Expires: '0',
      },
    });
  } catch (error) {
    console.error('Error fetching spans:', error);
    return NextResponse.json(
      { error: 'Failed to fetch spans data from backend service' },
      { status: 500 },
    );
  }
}

// Next.js 14 route segment config
export const dynamicParams = true;
export const revalidate = 0; // Disable cache completely 