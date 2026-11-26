import type {
  AIConfigStatus,
  ArchivedStudyFull,
  BatchPredictionResponse,
  ColormapName,
  EvaluationReport,
  HealthStatus,
  HeatmapData,
  PatientProfile,
  PredictionResponse,
  ReportData,
  SampleCase,
  UserDetail,
  UserRecord,
} from "../types/clinical";

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  getHealth: () => jsonFetch<HealthStatus>("/api/v1/health"),
  getSamples: () => jsonFetch<SampleCase[]>("/api/v1/samples"),
  getSampleImageUrl: (filename: string) => `/api/v1/samples/${encodeURIComponent(filename)}`,

  async fetchSampleAsFile(sample: SampleCase): Promise<File> {
    const res = await fetch(api.getSampleImageUrl(sample.filename));
    if (!res.ok) throw new Error(`Failed to fetch sample ${sample.filename}`);
    const blob = await res.blob();
    return new File([blob], sample.filename, { type: blob.type || "image/png" });
  },

  async predict(params: {
    file: File;
    threshold: number;
    tta: boolean;
    useGeminiDecision: boolean;
    useLlm: boolean;
    colormap: ColormapName;
    patient?: PatientProfile;
    userId?: string | null;
    studyName?: string;
    sampleId?: string | null;
  }): Promise<PredictionResponse> {
    const fd = new FormData();
    fd.append("file", params.file);
    fd.append("threshold", String(params.threshold));
    fd.append("tta", String(params.tta));
    fd.append("use_gemini_decision", String(params.useGeminiDecision));
    fd.append("use_llm", String(params.useLlm));
    fd.append("colormap", params.colormap);
    fd.append("run_segmentation", "true");
    if (params.userId) fd.append("user_id", params.userId);
    if (params.studyName) fd.append("study_name", params.studyName);
    if (params.sampleId) fd.append("sample_id", params.sampleId);
    if (params.patient) {
      fd.append("patient_context", JSON.stringify(params.patient));
    }
    return jsonFetch<PredictionResponse>("/api/v1/predict", {
      method: "POST",
      body: fd,
    });
  },

  async predictStream(
    params: {
      file: File;
      threshold: number;
      tta: boolean;
      useGeminiDecision: boolean;
      useLlm: boolean;
      colormap: ColormapName;
      patient?: PatientProfile;
      userId?: string | null;
      studyName?: string;
      sampleId?: string | null;
    },
    onStage?: (event: "vision" | "consensus", payload: Partial<PredictionResponse>) => void
  ): Promise<PredictionResponse> {
    const fd = new FormData();
    fd.append("file", params.file);
    fd.append("threshold", String(params.threshold));
    fd.append("tta", String(params.tta));
    fd.append("use_gemini_decision", String(params.useGeminiDecision));
    fd.append("use_llm", String(params.useLlm));
    fd.append("colormap", params.colormap);
    fd.append("run_segmentation", "true");
    if (params.userId) fd.append("user_id", params.userId);
    if (params.studyName) fd.append("study_name", params.studyName);
    if (params.sampleId) fd.append("sample_id", params.sampleId);
    if (params.patient) {
      fd.append("patient_context", JSON.stringify(params.patient));
    }

    const res = await fetch("/api/v1/predict/stream", {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch {
        // ignore
      }
      throw new Error(detail);
    }

    if (!res.body) {
      return api.predict(params);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalResp: PredictionResponse | null = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) continue;
        const parsed = JSON.parse(line) as {
          event: "vision" | "consensus" | "complete" | "error";
          data?: PredictionResponse;
          detail?: string;
        };
        if (parsed.event === "error") {
          throw new Error(parsed.detail || "Streaming diagnosis failed.");
        }
        if (parsed.event === "vision" || parsed.event === "consensus") {
          if (parsed.data && onStage) onStage(parsed.event, parsed.data);
        } else if (parsed.event === "complete" && parsed.data) {
          finalResp = parsed.data;
        }
      }
    }

    const tail = buffer.trim();
    if (tail) {
      const parsed = JSON.parse(tail) as {
        event: string;
        data?: PredictionResponse;
        detail?: string;
      };
      if (parsed.event === "error") {
        throw new Error(parsed.detail || "Streaming diagnosis failed.");
      }
      if (parsed.event === "complete" && parsed.data) {
        finalResp = parsed.data;
      }
    }

    if (!finalResp) {
      throw new Error("Prediction stream closed before final response.");
    }
    return finalResp;
  },

  async predictBatch(files: File[], threshold: number): Promise<BatchPredictionResponse> {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    fd.append("threshold", String(threshold));
    fd.append("tta", "false");
    return jsonFetch<BatchPredictionResponse>("/api/v1/predict/batch", {
      method: "POST",
      body: fd,
    });
  },

  async getSingleHeatmap(file: File, pathology: string, colormap: ColormapName): Promise<HeatmapData> {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("pathology", pathology);
    fd.append("colormap", colormap);
    return jsonFetch<HeatmapData>("/api/v1/predict/heatmap", {
      method: "POST",
      body: fd,
    });
  },

  async synthesizeReport(params: {
    scores: Record<string, number>;
    threshold: number;
    patient?: PatientProfile;
    anatomy?: Record<string, unknown> | null;
    hotspots?: HeatmapData[];
  }): Promise<ReportData> {
    return jsonFetch<ReportData>("/api/v1/report/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scores: params.scores,
        threshold: params.threshold,
        patient: params.patient,
        anatomy: params.anatomy,
        hotspots: params.hotspots,
        use_llm: true,
      }),
    });
  },

  async copilotChat(params: {
    question: string;
    study_context: Record<string, unknown>;
    history: Array<{ role: string; content: string }>;
    previous_interaction_id?: string | null;
    user_id?: string | null;
    study_id?: string | null;
  }): Promise<{ answer: string; provider: string; interaction_id?: string | null; latency_ms: number }> {
    return jsonFetch("/api/v1/copilot/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
  },

  getAIConfig: () => jsonFetch<AIConfigStatus>("/api/v1/ai/config"),

  updateAIConfig: (payload: Record<string, unknown>) =>
    jsonFetch<AIConfigStatus>("/api/v1/ai/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  /* Per-User SQLite Database & Longitudinal Study Endpoints */
  listUsers: () => jsonFetch<UserRecord[]>("/api/v1/users"),
  getUser: (userId: string) => jsonFetch<UserDetail>(`/api/v1/users/${encodeURIComponent(userId)}`),

  createUser: (payload: Partial<UserRecord> & { full_name: string }) =>
    jsonFetch<UserDetail>("/api/v1/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  updateUser: (userId: string, patch: Partial<UserRecord>) =>
    jsonFetch<UserDetail>(`/api/v1/users/${encodeURIComponent(userId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),

  deleteUser: (userId: string) =>
    jsonFetch<{ deleted: boolean; user_id: string }>(`/api/v1/users/${encodeURIComponent(userId)}`, {
      method: "DELETE",
    }),

  getStudy: (studyId: string) =>
    jsonFetch<ArchivedStudyFull>(`/api/v1/studies/${encodeURIComponent(studyId)}`),

  updateStudy: (studyId: string, patch: Record<string, unknown>) =>
    jsonFetch<ArchivedStudyFull>(`/api/v1/studies/${encodeURIComponent(studyId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),

  deleteStudy: (studyId: string) =>
    jsonFetch<{ deleted: boolean; study_id: string }>(`/api/v1/studies/${encodeURIComponent(studyId)}`, {
      method: "DELETE",
    }),

  /* Live Verified ML Evaluation Metrics */
  getEvaluations: () => jsonFetch<EvaluationReport>("/api/v1/evaluations"),
};
