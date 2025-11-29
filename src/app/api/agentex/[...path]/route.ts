import { NextRequest, NextResponse } from "next/server";

/**
 * Agentex API Proxy
 * 
 * Proxies requests to the Agentex/SGP API with proper authentication.
 * This allows the Vercel frontend to call deployed agents through SGP.
 * 
 * Usage:
 *   POST /api/agentex/agents/name/simulab-simulator/rpc
 *   → Proxies to https://api.agentex.azure.workspace.egp.scale.com/agents/name/simulab-simulator/rpc
 */

const AGENTEX_BASE_URL = process.env.AGENTEX_BASE_URL || "https://api.agentex.azure.workspace.egp.scale.com";
const SGP_API_KEY = process.env.SGP_API_KEY || "";
const SGP_ACCOUNT_ID = process.env.SGP_ACCOUNT_ID || "";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxyRequest(request, await params);
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxyRequest(request, await params);
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxyRequest(request, await params);
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxyRequest(request, await params);
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxyRequest(request, await params);
}

async function proxyRequest(request: NextRequest, params: { path: string[] }) {
  const startTime = Date.now();

  try {
    const { path } = params;

    if (!SGP_API_KEY) {
      console.error("[Agentex Proxy] SGP_API_KEY not configured");
      return NextResponse.json(
        { error: "SGP authentication not configured" },
        { status: 500 }
      );
    }

    if (!SGP_ACCOUNT_ID) {
      console.error("[Agentex Proxy] SGP_ACCOUNT_ID not configured");
      return NextResponse.json(
        { error: "SGP account not configured" },
        { status: 500 }
      );
    }

    // Construct the target URL
    const targetPath = path.join("/");
    const url = new URL(targetPath, AGENTEX_BASE_URL);

    // Forward query parameters
    request.nextUrl.searchParams.forEach((value, key) => {
      url.searchParams.append(key, value);
    });

    // Prepare headers
    const headers: HeadersInit = {
      "x-api-key": SGP_API_KEY,
      "x-selected-account-id": SGP_ACCOUNT_ID,
    };

    // Forward relevant headers from original request
    const headersToForward = ["content-type", "accept"];
    for (const header of headersToForward) {
      const value = request.headers.get(header);
      if (value) {
        headers[header] = value;
      }
    }

    // Get request body if present
    let body: string | null = null;
    if (request.method !== "GET" && request.method !== "HEAD") {
      try {
        body = await request.text();
      } catch {
        // No body
      }
    }

    console.log(`[Agentex Proxy] ${request.method} ${url.toString()}`);

    // Make the proxied request
    const response = await fetch(url.toString(), {
      method: request.method,
      headers,
      body: body || undefined,
    });

    const duration = Date.now() - startTime;

    // Get response body
    const responseText = await response.text();

    console.log(`[Agentex Proxy] Response: ${response.status} (${duration}ms)`);

    // Try to parse as JSON, otherwise return as text
    try {
      const responseJson = JSON.parse(responseText);
      return NextResponse.json(responseJson, { status: response.status });
    } catch {
      return new NextResponse(responseText, {
        status: response.status,
        headers: { "Content-Type": response.headers.get("content-type") || "text/plain" },
      });
    }
  } catch (error) {
    console.error("[Agentex Proxy] Error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Proxy error" },
      { status: 500 }
    );
  }
}

