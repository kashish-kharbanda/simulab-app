import { NextRequest, NextResponse } from 'next/server';

const SGP_API_KEY = process.env.SGP_API_KEY;
const SGP_ACCOUNT_ID = process.env.SGP_ACCOUNT_ID;
const SGP_API_URL = process.env.SGP_API_URL;

export async function POST(req: NextRequest) {
  if (!SGP_API_KEY || !SGP_ACCOUNT_ID) {
    return NextResponse.json({ error: 'Missing API key or account ID' }, { status: 500 });
  }

  try {
    const formData = await req.formData();

    const response = await fetch(`${SGP_API_URL}/v5/files`, {
      method: 'POST',
      headers: {
        'x-api-key': SGP_API_KEY,
        'x-selected-account-id': SGP_ACCOUNT_ID,
      },
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Scale API error: ${response.status}, ${errorText}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Error uploading file:', error);
    return NextResponse.json({ error: error.message || 'File upload failed' }, { status: 500 });
  }
} 