import type { ColormapName, PatientProfile, WindowPreset } from "../types/clinical";

export type PageId = "overview" | "studio" | "consensus" | "report" | "triage" | "records";

export const NAV_PAGES: Array<{
  id: PageId;
  num: string;
  keyHint: string;
  label: string;
  shortDesc: string;
}> = [
  { id: "overview", num: "01", keyHint: "1", label: "Overview", shortDesc: "Intake & specimen" },
  { id: "studio", num: "02", keyHint: "2", label: "Scan Studio", shortDesc: "Lightbox & Grad-CAM++" },
  { id: "consensus", num: "03", keyHint: "3", label: "Consensus & CTR", shortDesc: "4-model biometry" },
  { id: "report", num: "04", keyHint: "4", label: "Report & Copilot", shortDesc: "ACR & AI dictation" },
  { id: "triage", num: "05", keyHint: "5", label: "Batch Triage", shortDesc: "STAT worklist" },
  { id: "records", num: "06", keyHint: "6", label: "Records & Evals", shortDesc: "User DB & metrics" },
];

export const ARCHITECTURE_PILLARS = [
  {
    code: "ENSEMBLE",
    metric: "3 GPU CNNs",
    detail: "DenseNet-121 All · NIH-14 · ResNet-50 512px",
  },
  {
    code: "BIOMETRY",
    metric: "PSPNet CTR",
    detail: "Automated cardiothoracic ratio & lung symmetry",
  },
  {
    code: "DECISION AI",
    metric: "0 Out Tokens",
    detail: "Gemini Decision 1 calibrated log-odds fusion",
  },
  {
    code: "BENCHMARK",
    metric: "92.9% Acc",
    detail: "97.3% specificity · sub-second RTX CUDA vision",
  },
];

export const PATHOLOGY_CLINICAL_NOTE: Record<string, { plain: string; guideline?: string }> = {
  Atelectasis: { plain: "Partial alveolar collapse or volume loss", guideline: "ATS" },
  Cardiomegaly: { plain: "Cardiac silhouette enlargement (CTR > 0.50)", guideline: "ACC/AHA" },
  Effusion: { plain: "Fluid accumulation in pleural space", guideline: "BTS Pleural" },
  Infiltration: { plain: "Patchy alveolar or interstitial opacity", guideline: "ATS/IDSA" },
  Mass: { plain: "Discrete pulmonary lesion > 3 cm", guideline: "Fleischner 2017" },
  Nodule: { plain: "Discrete rounded pulmonary opacity ≤ 3 cm", guideline: "Fleischner 2017" },
  Pneumonia: { plain: "Infectious airspace consolidation", guideline: "ATS/IDSA CAP" },
  Pneumothorax: { plain: "Pleural air leak — urgent evaluation", guideline: "ACCP STAT" },
  Consolidation: { plain: "Dense alveolar inflammatory exudate", guideline: "ATS/IDSA" },
  Edema: { plain: "Pulmonary vascular fluid congestion", guideline: "ACC/AHA HF" },
  Emphysema: { plain: "Hyperinflation & parenchymal attenuation", guideline: "GOLD COPD" },
  Fibrosis: { plain: "Chronic reticular interstitial scarring", guideline: "ATS ILD" },
  Pleural_Thickening: { plain: "Fibrous pleural lining thickening", guideline: "BTS" },
  Hernia: { plain: "Hiatal or diaphragmatic visceral protrusion" },
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
  { id: "lung", label: "Lung Field", brightness: 108, contrast: 132, invert: false },
  { id: "bone", label: "Bone Detail", brightness: 88, contrast: 150, invert: false },
  { id: "inverted", label: "Inverted", brightness: 102, contrast: 114, invert: true },
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

/** Prevents redundant "Cardiomegaly / Cardiomegaly" subtitle duplication on cards. */
export function formatSampleSubtitle(name: string, subtitle: string, findings: string[]): string {
  const title = cleanSampleTitle(name).toLowerCase();
  const joined = findings.map(formatPathology).join(" · ");
  if (joined && joined.toLowerCase() !== title) {
    return joined;
  }
  if (subtitle) {
    return subtitle.length > 34 ? `${subtitle.slice(0, 32)}…` : subtitle;
  }
  return "Verified NIH Benchmark";
}
