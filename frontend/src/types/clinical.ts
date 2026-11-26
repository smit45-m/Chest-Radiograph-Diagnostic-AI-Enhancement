export type ThemeMode = "dark" | "light";
export type ColormapName = "turbo" | "inferno" | "clinical" | "viridis" | "jet";
export type WindowPreset = "standard" | "lung" | "bone" | "inverted";
export type AnalysisStage = "idle" | "vision" | "consensus" | "report";

export interface PatientProfile {
  age: number | null;
  sex: string;
  setting: string;
  indication: string;
  comorbidities: string;
  smoking: string;
}

export interface PathologyPrediction {
  name: string;
  score: number;
  severity: "mild" | "moderate" | "severe";
  review_flag?: "positive" | "borderline" | "uncertain" | "clear";
  conformal_decision?: "AUTO_POSITIVE" | "AUTO_NEGATIVE" | "DEFER_RADIOLOGIST" | null;
  calibrated_risk?: number | null;
  uncertainty?: number;
  agreement?: number;
  ci_low?: number | null;
  ci_high?: number | null;
  cnn_score?: number | null;
  gemini_decision_score?: number | null;
  backbones?: Record<string, number>;
  anatomical_zone?: string | null;
}

export interface HeatmapData {
  pathology: string;
  image_base64: string;
  raw_heatmap_base64?: string | null;
  method?: string;
  colormap?: string;
  peak_x?: number | null;
  peak_y?: number | null;
  peak_intensity?: number | null;
  bbox?: number[] | null;
  coverage_pct?: number | null;
  anatomical_zone?: string | null;
  laterality?: string | null;
}

export interface AnatomyData {
  ctr: number | null;
  ctr_status: string;
  ctr_cutoff?: number | null;
  projection_view?: string | null;
  ctr_coords?: {
    thorax_x1: number;
    thorax_x2: number;
    heart_x1: number;
    heart_x2: number;
    y: number;
  } | null;
  left_lung_pct: number | null;
  right_lung_pct: number | null;
  lung_symmetry_ratio: number | null;
  overlay_base64?: string | null;
  segmentation_ms?: number;
  organs_detected?: string[];
}

export interface DecisionSummary {
  active: boolean;
  status: string;
  model: string;
  transport?: string | null;
  latency_ms: number;
  output_tokens?: number;
  is_valid_cxr?: boolean | null;
  valid_cxr_probability?: number | null;
  triage_acuity?: string | null;
  triage_probabilities?: Record<string, number>;
  projection_view?: string | null;
  image_quality_score?: number | null;
  message?: string | null;
}

export interface ReportData {
  findings: string;
  impression: string;
  dictation_transcript?: string;
  recommendations?: string;
  prescriptions?: string;
  patient_summary?: string;
  executive_summary?: string[];
  differential_diagnoses?: string[];
  triage_level?: string;
  critical_alert?: boolean;
  compartments?: Record<string, string>;
  full_text: string;
  source: string;
}

export interface PredictionResponse {
  study_id?: string | null;
  user_id?: string | null;
  predictions: PathologyPrediction[];
  heatmaps: HeatmapData[];
  anatomy?: AnatomyData | null;
  decision_summary?: DecisionSummary | null;
  ensemble_info?: {
    mode?: string;
    active_backbones?: string[];
    tta_enabled?: boolean;
    segmentation_enabled?: boolean;
    gemini_decision_active?: boolean;
    raddino_active?: boolean;
    meta_calibrator_active?: boolean;
    device?: string;
  };
  timings?: {
    vision_ms?: number;
    gradcam_ms?: number;
    core_total_ms?: number;
    decision_api_ms?: number;
    report_ms?: number;
    total_ms?: number;
  };
  dicom_metadata?: Record<string, string | number> | null;
  report: ReportData;
  patient?: PatientProfile | null;
  cache_hit?: boolean;
  inference_time_ms: number;
  vision_latency_ms?: number | null;
}

export interface HealthStatus {
  status: string;
  device: string;
  gpu_name?: string | null;
  model_loaded: boolean;
  active_backbones: string[];
  segmentation_ready: boolean;
  gemini_configured: boolean;
  gemini_flash_model: string;
  gemini_decision_enabled: boolean;
  batcher_active: boolean;
  uptime_seconds: number;
}

export interface SampleCase {
  id: string;
  name: string;
  subtitle: string;
  filename: string;
  primary_findings: string[];
  suggested_patient?: PatientProfile | null;
}

export interface AIConfigStatus {
  gemini_configured: boolean;
  gemini_key_hint: string;
  gemini_flash_model: string;
  gemini_decision_model: string;
  gemini_decision_enabled: boolean;
  gemini_flash_enabled: boolean;
  decision_fusion_weight: number;
  groq_configured: boolean;
  groq_model: string;
  ensemble_mode: "full" | "fast";
  available_flash_models: string[];
}

export interface BatchStudyItem {
  filename: string;
  triage_level: string;
  critical_alert: boolean;
  top_findings: PathologyPrediction[];
  ctr?: number | null;
  vision_latency_ms: number;
  error?: string | null;
}

export interface BatchPredictionResponse {
  studies: BatchStudyItem[];
  total_studies: number;
  total_latency_ms: number;
  avg_latency_ms: number;
  stat_count: number;
}

export interface CopilotMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  provider?: string;
  latencyMs?: number;
  timestamp: string;
}

export interface CaliperMeasurement {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  mm: number;
}

/* Persistent Per-User Clinical Database Types */

export interface UserRecord {
  id: string;
  mrn: string;
  full_name: string;
  email: string;
  role: string;
  age: number | null;
  sex: string;
  setting: string;
  indication: string;
  comorbidities: string;
  smoking: string;
  preferred_theme?: string;
  preferred_threshold?: number;
  preferred_colormap?: string;
  notes: string;
  created_at: string;
  updated_at: string;
  study_count?: number;
  last_study_at?: string | null;
  latest_triage?: string | null;
  latest_ctr?: number | null;
}

export interface ArchivedStudySummary {
  id: string;
  user_id: string;
  study_name: string;
  filename: string;
  sample_id?: string | null;
  triage_level: string;
  critical_alert: boolean;
  top_pathology?: string | null;
  top_score?: number | null;
  ctr?: number | null;
  ctr_status?: string | null;
  report_source: string;
  dictation_transcript: string;
  physician_name: string;
  physician_creds: string;
  physician_notes: string;
  signed_at?: string | null;
  vision_latency_ms: number;
  total_latency_ms: number;
  created_at: string;
  updated_at: string;
}

export interface ArchivedStudyFull extends ArchivedStudySummary {
  predictions: PathologyPrediction[];
  anatomy?: AnatomyData | null;
  decision_summary?: DecisionSummary | null;
  report: ReportData;
  ensemble_info?: PredictionResponse["ensemble_info"];
  timings?: PredictionResponse["timings"];
  copilot_messages?: Array<{
    id: string;
    role: "user" | "assistant";
    content: string;
    provider?: string;
    interaction_id?: string;
    latency_ms?: number;
    created_at: string;
  }>;
}

export interface UserDetail extends UserRecord {
  studies: ArchivedStudySummary[];
  ctr_history: Array<{
    study_id: string;
    study_name: string;
    ctr: number;
    created_at: string;
  }>;
}

export interface BackboneEvalMetrics {
  accuracy: number;
  micro_roc_auc: number | null;
  macro_roc_auc: number | null;
  micro_pr_auc: number | null;
  sensitivity: number;
  specificity: number;
  precision: number;
  f1_score: number;
  brier_score: number;
  ece: number;
}

export interface NihStudyMetricRow {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  micro_roc_auc: number;
  macro_roc_auc: number;
  micro_pr_auc: number;
  macro_pr_auc: number;
  brier: number;
}

export interface NihGeneralizationStudy {
  generated_at?: string;
  cohort: {
    oof_train_val_images: number;
    oof_train_val_patients: number;
    held_out_test_images: number;
    held_out_test_patients: number;
    excluded_benchmark_patients?: number[];
  };
  multilabel_full_coverage_test: Record<string, NihStudyMetricRow>;
  multilabel_conformal_selective_test?: Record<
    string,
    {
      tau_low: number;
      tau_high: number;
      test_coverage: number;
      test_deferred_pct: number;
      test_accuracy: number;
      test_precision: number;
      test_recall: number;
      test_npv: number;
      test_f1: number;
    }
  >;
  binary_abnormal_triage_test?: {
    full_coverage: {
      roc_auc: number;
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
    };
    selective_conformal_90: {
      tau_low: number;
      tau_high: number;
      test_coverage: number;
      test_accuracy: number;
      test_precision: number;
      test_recall: number;
      test_npv: number;
      test_f1: number;
    };
  };
  per_class_test_5vision?: Array<{
    pathology: string;
    support: number;
    v9_auc: number;
    v10_5vision_auc: number;
    v10_5vision_pr: number;
    meta_blend_alpha: number;
    oof_f1_threshold: number;
  }>;
}

export interface EvaluationReport {
  status?: string;
  generated_at?: string;
  device?: string;
  gpu_name?: string;
  threshold?: number;
  cases_evaluated?: number;
  total_pathology_judgments?: number;
  headline_metrics?: BackboneEvalMetrics & {
    confusion?: { tp: number; tn: number; fp: number; fn: number; total: number };
    per_class?: Array<{
      pathology: string;
      roc_auc: number | null;
      pr_auc: number | null;
      sensitivity: number;
      specificity: number;
      support: number;
    }>;
  };
  operating_profiles?: Record<
    string,
    {
      threshold: number;
      accuracy: number;
      sensitivity: number;
      specificity: number;
      precision: number;
      f1_score: number;
    }
  >;
  backbone_comparison?: Record<string, BackboneEvalMetrics>;
  nih_generalization_study?: NihGeneralizationStudy;
  latency_sla?: {
    avg_vision_ms: number;
    avg_gradcam_ms: number;
    avg_core_gpu_ms: number;
    avg_decision_1_ms: number;
    avg_report_synthesis_ms: number;
  };
  ai_verification?: {
    gemini_decision_active_cases: number;
    live_gemini_transcripts: number;
    total_cases: number;
    flash_model: string;
    decision_model: string;
  };
  cases?: Array<{
    case_id: string;
    name: string;
    filename: string;
    ground_truth: string[];
    detected: string[];
    match: string;
    ctr: number | null;
    ctr_status?: string | null;
    gradcam_zone?: string | null;
    decision_1_active: boolean;
    decision_1_triage?: string | null;
    decision_1_view?: string | null;
    decision_1_quality?: number | null;
    report_source: string;
    triage_level?: string | null;
    impression: string;
    dictation_transcript: string;
    recommendations?: string;
    timings?: Record<string, number>;
  }>;
}
