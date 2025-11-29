/**
 * SimuLab Sheets Data Module
 * 
 * Shared data access for Google Sheets scenarios.
 * Used by both the sheets API route and the refine route directly.
 * 
 * Source of truth: https://docs.google.com/spreadsheets/d/17bd4GhtN66ekoWxff1qaGNa_SWuYPThR6zgEu215ZWQ
 */

export interface SheetScenario {
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
 * Hardcoded data from the public Google Sheet
 * This ensures data is always available even if Google Sheets API fails
 * 
 * Last updated: Matches public sheet at
 * https://docs.google.com/spreadsheets/d/17bd4GhtN66ekoWxff1qaGNa_SWuYPThR6zgEu215ZWQ
 */
export const SHEET_DATA: SheetScenario[] = [
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
    reference_sa_score: 6.2,
    target_result: "Rejected",
    result_category: "Cost/Risk Fail",
  },
  // Scenario 4: Amyloid Beta (6 scenarios) - Updated from Google Sheet
  {
    scenario_id: "4",
    protein_target: "Amyloid Beta",
    scaffold_hypothesis: "Lipophilic Diamine Core",
    smiles: "CC1=NC(=CS1)C2=CC=C(C=C2)N(C)C",
    pdb_id: "4B5S",
    reference_binding_affinity: -11.2,
    reference_herg_flag: true,
    reference_sa_score: 4.8,
    target_result: "Rejected",
    result_category: "Safety Veto",
  },
  {
    scenario_id: "4",
    protein_target: "Amyloid Beta",
    scaffold_hypothesis: "Multi-Chiral Macrocycle",
    smiles: "C1=C(C=C(C=C1)C2=C(C)C(=O)O2)C",
    pdb_id: "4B5S",
    reference_binding_affinity: -10.9,
    reference_herg_flag: false,
    reference_sa_score: 6.9,
    target_result: "Rejected",
    result_category: "Cost Veto",
  },
  {
    scenario_id: "4",
    protein_target: "Amyloid Beta",
    scaffold_hypothesis: "Simple Benzimidazole",
    smiles: "C1=CC=C(C=C1)C(C)C(=O)O",
    pdb_id: "4B5S",
    reference_binding_affinity: -6.9,
    reference_herg_flag: false,
    reference_sa_score: 2.5,
    target_result: "Rejected",
    result_category: "Potency Fail",
  },
  {
    scenario_id: "4",
    protein_target: "Amyloid Beta",
    scaffold_hypothesis: "Piperazine Amide Analogue",
    smiles: "C1=CC(=C(C=C1)OC)C(=O)N(C)C",
    pdb_id: "4B5S",
    reference_binding_affinity: -10.7,
    reference_herg_flag: false,
    reference_sa_score: 2.2,
    target_result: "WINNER",
    result_category: "Winner",
  },
  {
    scenario_id: "4",
    protein_target: "Amyloid Beta",
    scaffold_hypothesis: "Fluorinated Poly-Aromatic Ring",
    smiles: "C1=C(C=C(C=C1)C2=NC=C(C=C2)C)C",
    pdb_id: "4B5S",
    reference_binding_affinity: -9.5,
    reference_herg_flag: false,
    reference_sa_score: 5.5,
    target_result: "Accepted",
    result_category: "Non-Winner",
  },
  {
    scenario_id: "4",
    protein_target: "Amyloid Beta",
    scaffold_hypothesis: "Small Heterocycle Analogue",
    smiles: "CC(=O)C1=CC=C(C=C1)C",
    pdb_id: "4B5S",
    reference_binding_affinity: -8.1,
    reference_herg_flag: false,
    reference_sa_score: 3.9,
    target_result: "Accepted",
    result_category: "Non-Winner",
  },
];

/**
 * Get all scenarios from the database
 */
export function getAllScenarios(): SheetScenario[] {
  return SHEET_DATA;
}

/**
 * Get scenarios filtered by protein target (case-insensitive)
 */
export function getScenariosByProteinTarget(proteinTarget: string): SheetScenario[] {
  const targetLower = proteinTarget.toLowerCase().trim();
  return SHEET_DATA.filter(s => 
    s.protein_target.toLowerCase().includes(targetLower) ||
    targetLower.includes(s.protein_target.toLowerCase())
  );
}

/**
 * Get unique protein targets
 */
export function getUniqueProteinTargets(): string[] {
  return Array.from(new Set(SHEET_DATA.map(s => s.protein_target)));
}

/**
 * Find scenario by SMILES string
 */
export function findScenarioBySmiles(smiles: string): SheetScenario | null {
  return SHEET_DATA.find(s => s.smiles.trim() === smiles.trim()) || null;
}

/**
 * Find scenario by scaffold name (partial match)
 */
export function findScenarioByScaffold(scaffold: string): SheetScenario | null {
  const scaffoldLower = scaffold.toLowerCase();
  return SHEET_DATA.find(s => {
    const dbScaffold = s.scaffold_hypothesis.toLowerCase();
    return scaffoldLower.includes(dbScaffold) || dbScaffold.includes(scaffoldLower);
  }) || null;
}

