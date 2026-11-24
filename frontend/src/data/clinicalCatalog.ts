import type { ColormapName, PatientProfile, WindowPreset } from "../types/clinical";

export type PageId = "overview" | "studio" | "consensus" | "report" | "triage" | "records";

export const NAV_PAGES: Array<{
  id: PageId;
  num: string;
  keyHint: string;
  label: string;
  shortDesc: string;
}> = [
  { id: "overview", num: "01", keyHint: "1", label: "Overview", shortDesc: "Start & samples" },
  { id: "studio", num: "02", keyHint: "2", label: "Scan Studio", shortDesc: "X-ray & overlays" },
  { id: "consensus", num: "03", keyHint: "3", label: "Consensus & CTR", shortDesc: "Model breakdown" },
  { id: "report", num: "04", keyHint: "4", label: "Report & Copilot", shortDesc: "ACR & AI chat" },
  { id: "triage", num: "05", keyHint: "5", label: "Batch Triage", shortDesc: "Multi-scan queue" },
  { id: "records", num: "06", keyHint: "6", label: "Records & Evals", shortDesc: "Patients & metrics" },
];

export const PATHOLOGY_CLINICAL_NOTE: Record<string, { plain: string; guideline?: string }> = {
  Atelectasis: { plain: "Partial lung collapse or volume loss", guideline: "ATS" },
  Cardiomegaly: { plain: "Enlarged heart silhouette (CTR > 0.50)", guideline: "ACC/AHA" },
  Effusion: { plain: "Fluid buildup in the pleural space", guideline: "BTS Pleural" },
  Infiltration: { plain: "Patchy alveolar or interstitial opacity", guideline: "ATS/IDSA" },
  Mass: { plain: "Discrete pulmonary lesion > 3 cm", guideline: "Fleischner 2017" },
  Nodule: { plain: "Rounded pulmonary opacity ≤ 3 cm", guideline: "Fleischner 2017" },
  Pneumonia: { plain: "Infectious airspace consolidation", guideline: "ATS/IDSA CAP" },
  Pneumothorax: { plain: "Collapsed lung air leak — urgent", guideline: "ACCP STAT" },
  Consolidation: { plain: "Dense alveolar inflammatory exudate", guideline: "ATS/IDSA" },
  Edema: { plain: "Pulmonary vascular fluid congestion", guideline: "ACC/AHA HF" },
  Emphysema: { plain: "Air trapping & parenchymal loss", guideline: "GOLD COPD" },
  Fibrosis: { plain: "Chronic interstitial lung scarring", guideline: "ATS ILD" },
  Pleural_Thickening: { plain: "Thickened pleural lining", guideline: "BTS" },
  Hernia: { plain: "Diaphragmatic visceral protrusion" },
};

export const PATIENT_PERSONAS: Record<string, { label: string; profile: PatientProfile }> = {
  "chf-cardio": {
    label: "68F · Inpatient Cardiology (HFrEF, Dyspnea)",
    profile: {
      age: 68,
      sex: "Female",
      setting: "Inpatient Cardiology Service",
      indication: "Progressive dyspnea on exertion, 3-pillow orthopnea, bilateral pitting ankle edema",
      comorbidities: "Congestive Heart Failure (HFrEF, EF 32%), Chronic Kidney Disease",
      smoking: "Never smoker",
    },
  },
  "smoker-nodule": {
    label: "64M · Pulmonology Clinic (35 Pack-Year Smoker)",
    profile: {
      age: 64,
      sex: "Male",
      setting: "Pulmonology Outpatient Clinic",
      indication: "Chronic dry cough, 14 lb unintentional weight loss over 3 months, mild hemoptysis",
      comorbidities: "COPD (GOLD Stage II)",
      smoking: "Active smoker (35 pack-years)",
    },
  },
  "elderly-cap": {
    label: "72M · Emergency Dept (High Fever, Productive Cough)",
    profile: {
      age: 72,
      sex: "Male",
      setting: "Emergency Department (ED)",
      indication: "High fever (39.1°C), productive cough with purulent sputum, pleuritic right chest pain",
      comorbidities: "Type 2 Diabetes, Hypertension",
      smoking: "Former smoker (quit 12y ago)",
    },
  },
  "preop-normal": {
    label: "45F · Outpatient (Routine Pre-Op Clearance)",
    profile: {
      age: 45,
      sex: "Female",
      setting: "Ambulatory / Outpatient",
      indication: "Routine preoperative cardiopulmonary clearance",
      comorbidities: "None documented",
      smoking: "Never smoker",
    },
  },
};

export const WINDOW_PRESETS: Array<{
  id: WindowPreset;
  label: string;
  brightness: number;
  contrast: number;
  invert: boolean;
}> = [
  { id: "standard", label: "Standard", brightness: 100, contrast: 105, invert: false },
  { id: "lung", label: "Lung", brightness: 108, contrast: 132, invert: false },
  { id: "bone", label: "Bone", brightness: 88, contrast: 150, invert: false },
  { id: "inverted", label: "Invert", brightness: 102, contrast: 114, invert: true },
];

export const COLORMAP_OPTIONS: Array<{ id: ColormapName; label: string }> = [
  { id: "turbo", label: "Turbo" },
  { id: "inferno", label: "Inferno" },
  { id: "clinical", label: "Clinical" },
  { id: "viridis", label: "Viridis" },
];

export function formatPathology(name: string): string {
  return name.replace(/_/g, " ");
}

export function cleanSampleTitle(name: string): string {
  return name.replace(/^(NIH Case:|Clinical Case:)\s*/i, "");
}

/** Short, crisp, never-truncated study labels for pill bars & cards */
export function shortSampleTitle(name: string): string {
  const clean = cleanSampleTitle(name);
  if (clean.toLowerCase().includes("cardiomegaly")) return "Cardiomegaly";
  if (clean.toLowerCase().includes("nodule")) return "Lung Nodule";
  if (clean.toLowerCase().includes("mass")) return "Thoracic Mass";
  if (clean.toLowerCase().includes("atelectasis")) return "Atelectasis";
  if (clean.toLowerCase().includes("pneumonia")) return "Pneumonia";
  return clean.length > 22 ? `${clean.slice(0, 20)}…` : clean;
}

/** Human-friendly urgency label without raw ALL_CAPS_UNDERSCORES */
export function formatTriageLabel(level: string | null | undefined): string {
  if (!level) return "Routine";
  const u = level.toUpperCase();
  if (u.includes("STAT")) return "STAT Critical";
  if (u.includes("SEMI")) return "Semi-Urgent";
  if (u.includes("URGENT")) return "Urgent";
  if (u.includes("ROUTINE")) return "Routine";
  return level.replace(/_/g, " ");
}

/** Prevents redundant or truncated subtitles on sample study cards. */
export function formatSampleSubtitle(name: string, subtitle: string, findings: string[]): string {
  const title = shortSampleTitle(name).toLowerCase();
  if (title === "cardiomegaly") return "Cardiac silhouette";
  const joined = findings.map(formatPathology).join(" · ");
  if (joined && joined.toLowerCase() !== title) {
    return joined.length > 22 ? `${joined.slice(0, 20)}…` : joined;
  }
  if (subtitle) {
    return subtitle.length > 22 ? `${subtitle.slice(0, 20)}…` : subtitle;
  }
  return "NIH Benchmark";
}
