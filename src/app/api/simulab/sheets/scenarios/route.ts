import { NextRequest, NextResponse } from "next/server"

/**
 * SimuLab Google Sheets API Route
 * 
 * Accesses Google Sheets to get scenario data.
 * This is the source of truth for validated experimental data.
 * 
 * Supports two modes:
 * 1. Service account auth (GOOGLE_APPLICATION_CREDENTIALS_JSON)
 * 2. API key for public sheets (GOOGLE_SHEETS_API_KEY)
 */

export const maxDuration = 30;
export const dynamic = "force-dynamic"

// Google Sheets configuration
// Public SimuLab sheet: https://docs.google.com/spreadsheets/d/17bd4GhtN66ekoWxff1qaGNa_SWuYPThR6zgEu215ZWQ/edit
const SHEET_ID = process.env.SIMULAB_GOOGLE_SHEET_ID || "17bd4GhtN66ekoWxff1qaGNa_SWuYPThR6zgEu215ZWQ";
const GOOGLE_CREDENTIALS_JSON = process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON;
const GOOGLE_API_KEY = process.env.GOOGLE_SHEETS_API_KEY;

interface SheetScenario {
  scenario_id: string;
  protein_target: string;
  scaffold_hypothesis: string;
  smiles: string;
  pdb_id: string;
  reference_binding_affinity: number | null;
  reference_herg_flag: boolean | null;
  reference_sa_score: number | null;
  target_result: string;
  result_category: string;
}

/**
 * Parse a float value from sheet data
 */
function parseFloatSafe(value: any): number | null {
  if (value === null || value === undefined || value === "") return null;
  const num = Number(value);
  return isNaN(num) ? null : num;
}

/**
 * Parse a boolean value from sheet data (Yes/No)
 */
function parseBoolSafe(value: any): boolean | null {
  if (value === null || value === undefined || value === "") return null;
  const str = String(value).toLowerCase().trim();
  if (str === "yes" || str === "true" || str === "1") return true;
  if (str === "no" || str === "false" || str === "0") return false;
  return null;
}

/**
 * Create a JWT for Google API authentication
 */
async function createGoogleJWT(credentials: any): Promise<string> {
  const header = {
    alg: "RS256",
    typ: "JWT",
  };

  const now = Math.floor(Date.now() / 1000);
  const payload = {
    iss: credentials.client_email,
    scope: "https://www.googleapis.com/auth/spreadsheets.readonly",
    aud: "https://oauth2.googleapis.com/token",
    iat: now,
    exp: now + 3600, // 1 hour
  };

  // Base64url encode
  const base64url = (obj: object) => {
    const json = JSON.stringify(obj);
    const base64 = Buffer.from(json).toString("base64");
    return base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");
  };

  const headerEncoded = base64url(header);
  const payloadEncoded = base64url(payload);
  const signatureInput = `${headerEncoded}.${payloadEncoded}`;

  // Sign with private key using Node.js crypto
  const crypto = await import("crypto");
  const sign = crypto.createSign("RSA-SHA256");
  sign.update(signatureInput);
  const signature = sign.sign(credentials.private_key, "base64");
  const signatureEncoded = signature.replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");

  return `${headerEncoded}.${payloadEncoded}.${signatureEncoded}`;
}

/**
 * Get access token from Google OAuth using service account
 */
async function getGoogleAccessToken(credentials: any): Promise<string> {
  const jwt = await createGoogleJWT(credentials);

  const response = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
      assertion: jwt,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to get access token: ${error}`);
  }

  const data = await response.json();
  return data.access_token;
}

/**
 * Parse sheet rows into scenarios
 */
function parseRows(rows: any[][]): SheetScenario[] {
  if (!rows || rows.length < 2) {
    console.log("[SimuLab/Sheets] No data found in sheet");
    return [];
  }

  // First row is headers
  const headers = rows[0] as string[];
  const dataRows = rows.slice(1);

  console.log(`[SimuLab/Sheets] Found ${dataRows.length} rows with headers: ${headers.join(", ")}`);

  // Map column indices (case-insensitive search)
  const findCol = (patterns: string[]) => {
    return headers.findIndex((h: string) => {
      const hLower = h.toLowerCase();
      return patterns.some(p => hLower.includes(p.toLowerCase()));
    });
  };

  const colIndex = {
    scenario_id: findCol(["scenario"]),
    protein_target: findCol(["protein"]),
    scaffold_hypothesis: findCol(["scaffold"]),
    smiles: findCol(["smiles"]),
    pdb_id: findCol(["pdb"]),
    binding_affinity: findCol(["affinity", "ΔG"]),
    herg_flag: findCol(["herg"]),
    sa_score: findCol(["sa score", "cost"]),
    target_result: findCol(["final result", "target final"]),
    result_category: findCol(["category"]),
  };

  console.log(`[SimuLab/Sheets] Column indices:`, colIndex);

  const scenarios: SheetScenario[] = dataRows
    .filter((row: any[]) => row.length > 0 && row.some(cell => cell)) // Skip empty rows
    .map((row: any[], idx: number) => ({
      scenario_id: row[colIndex.scenario_id] || `Scenario_${idx + 1}`,
      protein_target: row[colIndex.protein_target] || "",
      scaffold_hypothesis: row[colIndex.scaffold_hypothesis] || "",
      smiles: row[colIndex.smiles] || "",
      pdb_id: row[colIndex.pdb_id] || "",
      reference_binding_affinity: parseFloatSafe(row[colIndex.binding_affinity]),
      reference_herg_flag: parseBoolSafe(row[colIndex.herg_flag]),
      reference_sa_score: parseFloatSafe(row[colIndex.sa_score]),
      target_result: row[colIndex.target_result] || "",
      result_category: row[colIndex.result_category] || "",
    }));

  console.log(`[SimuLab/Sheets] Parsed ${scenarios.length} scenarios`);
  console.log(`[SimuLab/Sheets] Protein targets found:`, [...new Set(scenarios.map(s => s.protein_target))]);

  return scenarios;
}

/**
 * Get all scenarios from Google Sheets using service account
 */
async function getAllScenariosWithServiceAccount(): Promise<SheetScenario[]> {
  if (!GOOGLE_CREDENTIALS_JSON) {
    return [];
  }

  try {
    const credentials = JSON.parse(GOOGLE_CREDENTIALS_JSON);
    console.log(`[SimuLab/Sheets] Using service account: ${credentials.client_email}`);

    const accessToken = await getGoogleAccessToken(credentials);
    console.log("[SimuLab/Sheets] Got access token");

    const url = `https://sheets.googleapis.com/v4/spreadsheets/${SHEET_ID}/values/Sheet1!A:J`;
    const response = await fetch(url, {
      headers: { Authorization: `Bearer ${accessToken}` },
    });

    if (!response.ok) {
      const error = await response.text();
      console.error(`[SimuLab/Sheets] Failed to read sheet: ${error}`);
      return [];
    }

    const data = await response.json();
    return parseRows(data.values);

  } catch (error) {
    console.error("[SimuLab/Sheets] Service account error:", error);
    return [];
  }
}

/**
 * Get all scenarios from Google Sheets using API key (for public sheets)
 */
async function getAllScenariosWithApiKey(): Promise<SheetScenario[]> {
  if (!GOOGLE_API_KEY) {
    return [];
  }

  try {
    console.log("[SimuLab/Sheets] Using API key for public sheet access");

    const url = `https://sheets.googleapis.com/v4/spreadsheets/${SHEET_ID}/values/Sheet1!A:J?key=${GOOGLE_API_KEY}`;
    const response = await fetch(url);

    if (!response.ok) {
      const error = await response.text();
      console.error(`[SimuLab/Sheets] Failed to read sheet with API key: ${error}`);
      return [];
    }

    const data = await response.json();
    return parseRows(data.values);

  } catch (error) {
    console.error("[SimuLab/Sheets] API key error:", error);
    return [];
  }
}

/**
 * Get scenarios from public Google Sheet using CSV export (no auth required)
 */
async function getAllScenariosPublicCSV(): Promise<SheetScenario[]> {
  try {
    console.log("[SimuLab/Sheets] Attempting public CSV export (no auth)...");
    
    // Google Sheets public CSV export URL
    const csvUrl = `https://docs.google.com/spreadsheets/d/${SHEET_ID}/export?format=csv&gid=0`;
    
    console.log(`[SimuLab/Sheets] Fetching: ${csvUrl}`);
    
    // Use redirect: 'follow' explicitly to handle Google's redirects
    const response = await fetch(csvUrl, {
      method: 'GET',
      redirect: 'follow',
      headers: {
        'Accept': 'text/csv,text/plain,*/*',
        'User-Agent': 'SimuLab/1.0',
      },
    });

    console.log(`[SimuLab/Sheets] Response status: ${response.status}, type: ${response.type}`);

    if (!response.ok) {
      console.error(`[SimuLab/Sheets] CSV export failed: ${response.status} ${response.statusText}`);
      const text = await response.text();
      console.error(`[SimuLab/Sheets] Response body: ${text.substring(0, 500)}`);
      return [];
    }

    const csvText = await response.text();
    console.log(`[SimuLab/Sheets] Got CSV data (${csvText.length} chars)`);
    
    // Check if we got HTML instead of CSV (redirect issue)
    if (csvText.trim().startsWith('<')) {
      console.error("[SimuLab/Sheets] Got HTML instead of CSV - redirect not followed");
      console.error(`[SimuLab/Sheets] First 200 chars: ${csvText.substring(0, 200)}`);
      return [];
    }
    
    // Parse CSV
    const lines = csvText.split('\n').filter(line => line.trim());
    if (lines.length < 2) {
      console.log("[SimuLab/Sheets] No data rows in CSV");
      return [];
    }

    // Parse header row
    const headers = parseCSVLine(lines[0]);
    console.log(`[SimuLab/Sheets] Headers: ${headers.join(', ')}`);

    // Find column indices
    const findCol = (patterns: string[]) => {
      return headers.findIndex((h: string) => {
        const hLower = h.toLowerCase();
        return patterns.some(p => hLower.includes(p.toLowerCase()));
      });
    };

    const colIndex = {
      scenario_id: findCol(["scenario"]),
      protein_target: findCol(["protein"]),
      scaffold_hypothesis: findCol(["scaffold", "hypothesis"]),
      smiles: findCol(["smiles"]),
      pdb_id: findCol(["pdb"]),
      binding_affinity: findCol(["affinity", "ΔG", "δg"]),
      herg_flag: findCol(["herg"]),
      sa_score: findCol(["sa score", "cost"]),
      target_result: findCol(["final result", "target final"]),
      result_category: findCol(["category"]),
    };

    console.log(`[SimuLab/Sheets] Column indices:`, colIndex);

    // Parse data rows
    const scenarios: SheetScenario[] = [];
    for (let i = 1; i < lines.length; i++) {
      const row = parseCSVLine(lines[i]);
      if (row.length === 0 || !row.some(cell => cell.trim())) continue;

      const scenario: SheetScenario = {
        scenario_id: row[colIndex.scenario_id] || `Scenario_${i}`,
        protein_target: row[colIndex.protein_target] || "",
        scaffold_hypothesis: row[colIndex.scaffold_hypothesis] || "",
        smiles: row[colIndex.smiles] || "",
        pdb_id: row[colIndex.pdb_id] || "",
        reference_binding_affinity: parseFloatSafe(row[colIndex.binding_affinity]),
        reference_herg_flag: parseBoolSafe(row[colIndex.herg_flag]),
        reference_sa_score: parseFloatSafe(row[colIndex.sa_score]),
        target_result: row[colIndex.target_result] || "",
        result_category: row[colIndex.result_category] || "",
      };

      // Only add if has meaningful data
      if (scenario.protein_target || scenario.smiles) {
        scenarios.push(scenario);
      }
    }

    console.log(`[SimuLab/Sheets] Parsed ${scenarios.length} scenarios from CSV`);
    console.log(`[SimuLab/Sheets] Protein targets:`, [...new Set(scenarios.map(s => s.protein_target))]);
    
    return scenarios;

  } catch (error) {
    console.error("[SimuLab/Sheets] CSV export error:", error);
    return [];
  }
}

/**
 * Parse a CSV line handling quoted values
 */
function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  
  result.push(current.trim());
  return result;
}

/**
 * Hardcoded fallback data from the public sheet
 * This ensures data is always available even if Google Sheets API fails
 */
function getHardcodedScenarios(): SheetScenario[] {
  // Data from: https://docs.google.com/spreadsheets/d/17bd4GhtN66ekoWxff1qaGNa_SWuYPThR6zgEu215ZWQ
  return [
    {
      scenario_id: "1",
      protein_target: "BCR-ABL",
      scaffold_hypothesis: "Pyrrolo-pyrimidine",
      smiles: "CC(=O)C1=C(N)N=C(C)N=1",
      pdb_id: "1M17",
      reference_binding_affinity: -11.5,
      reference_herg_flag: true,
      reference_sa_score: 4.5,
      target_result: "Rejected",
      result_category: "Safety Veto",
    },
    {
      scenario_id: "1",
      protein_target: "BCR-ABL",
      scaffold_hypothesis: "Pyrazolo-pyridine",
      smiles: "CC1=C(C)N=C(C)C=C1N",
      pdb_id: "1M17",
      reference_binding_affinity: -8.8,
      reference_herg_flag: false,
      reference_sa_score: 3.1,
      target_result: "WINNER",
      result_category: "Winner",
    },
    {
      scenario_id: "2",
      protein_target: "T-Kinase",
      scaffold_hypothesis: "Thiazole-Thiophene",
      smiles: "C1=CC(=CN=C1)C2=CSC=C2",
      pdb_id: "2H9T",
      reference_binding_affinity: -9.1,
      reference_herg_flag: false,
      reference_sa_score: 4.1,
      target_result: "Accepted",
      result_category: "Non-Winner",
    },
    {
      scenario_id: "2",
      protein_target: "T-Kinase",
      scaffold_hypothesis: "Triazole-Pyridine",
      smiles: "CC1=NC=NC=C1C(C)C",
      pdb_id: "2H9T",
      reference_binding_affinity: -9.5,
      reference_herg_flag: false,
      reference_sa_score: 6.5,
      target_result: "Rejected",
      result_category: "Cost Veto",
    },
    {
      scenario_id: "2",
      protein_target: "T-Kinase",
      scaffold_hypothesis: "Pyrimidine-Amide",
      smiles: "C1=NC(=CN=C1)CC(=O)N",
      pdb_id: "2H9T",
      reference_binding_affinity: -10.2,
      reference_herg_flag: false,
      reference_sa_score: 3.8,
      target_result: "WINNER",
      result_category: "Winner",
    },
    {
      scenario_id: "3",
      protein_target: "Tox-Check",
      scaffold_hypothesis: "Complex X-Ring",
      smiles: "C1C(N(C)C)C(=O)C2=CC=C12",
      pdb_id: "3V03",
      reference_binding_affinity: -7.5,
      reference_herg_flag: true,
      reference_sa_score: 7.2,
      target_result: "Rejected",
      result_category: "Safety Veto",
    },
    {
      scenario_id: "3",
      protein_target: "Tox-Check",
      scaffold_hypothesis: "Simple Y-Chain",
      smiles: "CCCC(N)C(=O)O",
      pdb_id: "3V03",
      reference_binding_affinity: -6.8,
      reference_herg_flag: false,
      reference_sa_score: 1.9,
      target_result: "Rejected",
      result_category: "Potency Fail",
    },
    {
      scenario_id: "3",
      protein_target: "Tox-Check",
      scaffold_hypothesis: "Mid-Range Z-Ring",
      smiles: "C1C(N)C(=O)C=C1",
      pdb_id: "3V03",
      reference_binding_affinity: -8.1,
      reference_herg_flag: false,
      reference_sa_score: 5.5,
      target_result: "Rejected",
      result_category: "Cost/Risk Fail",
    },
  ];
}

/**
 * Try to get scenarios using available methods (public CSV first, then fallback to hardcoded)
 */
async function getAllScenarios(): Promise<SheetScenario[]> {
  // Try public CSV export first (no auth required for public sheets)
  console.log("[SimuLab/Sheets] Attempting public CSV export...");
  const csvScenarios = await getAllScenariosPublicCSV();
  if (csvScenarios.length > 0) {
    console.log(`[SimuLab/Sheets] ✓ Got ${csvScenarios.length} scenarios from public CSV`);
    return csvScenarios;
  }

  // Try service account if CSV fails
  if (GOOGLE_CREDENTIALS_JSON) {
    console.log("[SimuLab/Sheets] Attempting service account authentication...");
    const scenarios = await getAllScenariosWithServiceAccount();
    if (scenarios.length > 0) return scenarios;
  }

  // Try API key for public sheets
  if (GOOGLE_API_KEY) {
    console.log("[SimuLab/Sheets] Attempting API key authentication...");
    const scenarios = await getAllScenariosWithApiKey();
    if (scenarios.length > 0) return scenarios;
  }

  // Use hardcoded data as last resort (ensures demo always works)
  console.log("[SimuLab/Sheets] Using hardcoded scenario data (from public sheet snapshot)");
  return getHardcodedScenarios();
}

/**
 * GET /api/simulab/sheets/scenarios
 * Returns all scenarios from the Google Sheet
 * 
 * Query params:
 * - protein_target: Filter by protein target (case-insensitive)
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const proteinTarget = searchParams.get("protein_target");

    console.log(`[SimuLab/Sheets] GET request, protein_target: ${proteinTarget || "ALL"}`);

    const allScenarios = await getAllScenarios();

    if (allScenarios.length === 0) {
      console.warn("[SimuLab/Sheets] No scenarios loaded from sheet");
      return NextResponse.json({
        success: false,
        error: "Could not load scenarios from Google Sheets. Configure GOOGLE_APPLICATION_CREDENTIALS_JSON or GOOGLE_SHEETS_API_KEY.",
        count: 0,
        scenarios: [],
        debug: {
          hasCredentials: !!GOOGLE_CREDENTIALS_JSON,
          hasApiKey: !!GOOGLE_API_KEY,
          sheetId: SHEET_ID,
        }
      });
    }

    if (proteinTarget) {
      // Filter by protein target (case-insensitive, partial match)
      const targetLower = proteinTarget.toLowerCase().trim();
      const filtered = allScenarios.filter(
        s => s.protein_target.toLowerCase().includes(targetLower)
      );

      console.log(`[SimuLab/Sheets] Found ${filtered.length} scenarios for protein target: ${proteinTarget}`);

      return NextResponse.json({
        success: true,
        protein_target: proteinTarget,
        count: filtered.length,
        scenarios: filtered,
      });
    }

    return NextResponse.json({
      success: true,
      count: allScenarios.length,
      scenarios: allScenarios,
    });

  } catch (error) {
    console.error("[SimuLab/Sheets] API error:", error);
    return NextResponse.json({
      success: false,
      error: String(error),
      scenarios: [],
    }, { status: 500 });
  }
}
