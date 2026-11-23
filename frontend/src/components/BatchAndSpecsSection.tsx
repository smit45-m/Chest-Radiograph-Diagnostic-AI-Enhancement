import React, { useRef } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  ArrowUpRight,
  Cpu,
  FolderUp,
  Layers,
  Play,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import { formatPathology } from "../data/clinicalCatalog";
import type {
  AIConfigStatus,
  BatchPredictionResponse,
  HealthStatus,
  SampleCase,
} from "../types/clinical";
import { AnimatedNumber } from "./AnimatedNumber";
import { SpotlightCard } from "./SpotlightCard";

interface BatchAndSpecsSectionProps {
  samples: SampleCase[];
  batchResult: BatchPredictionResponse | null;
  batchLoading: boolean;
  health: HealthStatus | null;
  aiConfig: AIConfigStatus | null;
  onRunBenchmarkBatch: () => void;
  onUploadBatchFiles: (files: File[]) => void;
  onInspectSampleByFilename: (filename: string) => void;
}

export const BatchAndSpecsSection: React.FC<BatchAndSpecsSectionProps> = ({
  samples,
  batchResult,
  batchLoading,
  health,
  aiConfig,
  onRunBenchmarkBatch,
  onUploadBatchFiles,
  onInspectSampleByFilename,
}) => {
  const multiInputRef = useRef<HTMLInputElement | null>(null);

  const handleMultiFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    if (files.length > 0) onUploadBatchFiles(files);
    e.target.value = "";
  };

  const sampleFilenames = new Set(samples.map((s) => s.filename));

  return (
    <div className="mx-auto max-w-[1360px] px-5 py-7 pb-24 md:px-8 md:py-9 lg:pb-12 space-y-8">
      {/* Page Header */}
      <div className="flex flex-col justify-between gap-5 md:flex-row md:items-end">
        <div className="space-y-1.5">
          <div
            className="font-mono-tech text-[10px] font-medium tracking-[0.2em] uppercase"
            style={{ color: "var(--accent-clinical)" }}
          >
            05 · MULTI-STUDY STAT TRIAGE WORKLIST
          </div>
          <h1 className="font-display text-3xl font-bold sm:text-4xl">
            Prioritize emergency queues,{" "}
            <span
              className="font-serif-editorial font-normal"
              style={{ color: "var(--accent-clinical)" }}
            >
              automatically.
            </span>
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <input
            ref={multiInputRef}
            type="file"
            multiple
            accept="image/*,.dcm"
            onChange={handleMultiFiles}
            className="hidden"
          />
          <button
            type="button"
            disabled={batchLoading}
            onClick={onRunBenchmarkBatch}
            className="flex cursor-pointer items-center gap-2 rounded-full px-5 py-2.5 text-xs font-semibold transition-transform active:scale-95 disabled:opacity-50"
            style={{
              backgroundColor: "var(--accent-clinical)",
              color: "#ffffff",
              boxShadow: "0 8px 24px -8px var(--accent-clinical-glow)",
            }}
          >
            <Play className="h-3.5 w-3.5" />
            <span>
              {batchLoading
                ? "Triaging Queue..."
                : `Triage All ${samples.length || 5} Benchmark Studies`}
            </span>
          </button>

          <button
            type="button"
            disabled={batchLoading}
            onClick={() => multiInputRef.current?.click()}
            className="flex cursor-pointer items-center gap-2 rounded-full border px-5 py-2.5 text-xs font-semibold transition-opacity hover:opacity-85 disabled:opacity-50"
            style={{
              borderColor: "var(--border-strong)",
              backgroundColor: "var(--bg-elevated)",
            }}
          >
            <FolderUp className="h-3.5 w-3.5" />
            <span>Upload Multiple Scans</span>
          </button>
        </div>
      </div>

      {/* Main Batch Worklist Panel */}
      <SpotlightCard strong className="rounded-[26px] p-6 sm:p-7 space-y-6">
        {batchResult ? (
          <>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              {[
                { label: "Total Studies", val: batchResult.total_studies, suffix: "" },
                {
                  label: "STAT Critical",
                  val: batchResult.stat_count,
                  suffix: "",
                  alert: batchResult.stat_count > 0,
                },
                { label: "Avg Vision Latency", val: batchResult.avg_latency_ms, suffix: " ms" },
                { label: "Queue Wall Time", val: batchResult.total_latency_ms, suffix: " ms" },
              ].map((kpi) => (
                <div
                  key={kpi.label}
                  className="rounded-2xl border p-4"
                  style={{
                    borderColor: "var(--border-subtle)",
                    backgroundColor: "var(--bg-secondary)",
                  }}
                >
                  <div
                    className="font-mono-tech text-[10px] tracking-[0.14em] uppercase"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {kpi.label}
                  </div>
                  <div
                    className="mt-1.5 font-mono-tech text-2xl font-semibold"
                    style={{
                      color: kpi.alert ? "var(--accent-alert)" : "var(--text-primary)",
                    }}
                  >
                    <AnimatedNumber value={kpi.val} decimals={0} suffix={kpi.suffix} />
                  </div>
                </div>
              ))}
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr
                    className="border-b font-mono-tech text-[10px] tracking-[0.14em] uppercase"
                    style={{
                      borderColor: "var(--border-subtle)",
                      color: "var(--text-muted)",
                    }}
                  >
                    <th className="py-3 pr-4 font-medium">Priority</th>
                    <th className="px-3 py-3 font-medium">Study File</th>
                    <th className="px-3 py-3 font-medium">Lead Findings</th>
                    <th className="px-3 py-3 text-right font-medium">CTR</th>
                    <th className="px-3 py-3 text-right font-medium">Vision Latency</th>
                    <th className="py-3 pl-3 text-right font-medium">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                  {batchResult.studies.map((st, idx) => {
                    const isStat =
                      st.critical_alert || st.triage_level.toUpperCase().includes("STAT");
                    const isUrgent = st.triage_level.toUpperCase().includes("URGENT");
                    const canInspect = sampleFilenames.has(st.filename);

                    return (
                      <motion.tr
                        key={`${st.filename}-${idx}`}
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.04, duration: 0.28 }}
                        style={{ borderColor: "var(--border-subtle)" }}
                        className="transition-colors hover:bg-teal-500/[0.035]"
                      >
                        <td className="py-3.5 pr-4">
                          <span
                            className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono-tech text-[10px] font-medium uppercase"
                            style={{
                              borderColor: isStat
                                ? "var(--accent-alert)"
                                : isUrgent
                                ? "var(--accent-amber)"
                                : "var(--accent-clinical)",
                              backgroundColor: isStat
                                ? "var(--accent-alert-dim)"
                                : isUrgent
                                ? "var(--accent-amber-dim)"
                                : "var(--accent-clinical-dim)",
                              color: isStat
                                ? "var(--accent-alert)"
                                : isUrgent
                                ? "var(--accent-amber)"
                                : "var(--accent-clinical)",
                            }}
                          >
                            {isStat && <AlertTriangle className="h-3 w-3" />}
                            {st.triage_level}
                          </span>
                        </td>
                        <td className="px-3 py-3.5 font-mono-tech text-[11px] font-medium">
                          {st.filename}
                        </td>
                        <td className="px-3 py-3.5">
                          {st.error ? (
                            <span style={{ color: "var(--accent-alert)" }}>{st.error}</span>
                          ) : st.top_findings.length > 0 ? (
                            <div className="flex flex-wrap gap-1.5">
                              {st.top_findings.slice(0, 3).map((f) => (
                                <span
                                  key={f.name}
                                  className="rounded-full border px-2.5 py-0.5 font-mono-tech text-[10px]"
                                  style={{
                                    borderColor: "var(--border-subtle)",
                                    color: "var(--text-secondary)",
                                  }}
                                >
                                  {formatPathology(f.name)} {(f.score * 100).toFixed(0)}%
                                </span>
                              ))}
                            </div>
                          ) : (
                            <span style={{ color: "var(--text-muted)" }}>No acute findings</span>
                          )}
                        </td>
                        <td className="px-3 py-3.5 text-right font-mono-tech">
                          {st.ctr != null ? st.ctr.toFixed(3) : "—"}
                        </td>
                        <td
                          className="px-3 py-3.5 text-right font-mono-tech"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {Math.round(st.vision_latency_ms)} ms
                        </td>
                        <td className="py-3.5 pl-3 text-right">
                          {canInspect ? (
                            <button
                              type="button"
                              onClick={() => onInspectSampleByFilename(st.filename)}
                              className="inline-flex cursor-pointer items-center gap-1 rounded-full border px-2.5 py-1 font-mono-tech text-[10px] transition-colors hover:border-teal-500/60"
                              style={{
                                borderColor: "var(--border-subtle)",
                                color: "var(--accent-clinical)",
                              }}
                            >
                              <span>Inspect</span>
                              <ArrowUpRight className="h-3 w-3" />
                            </button>
                          ) : (
                            <span className="font-mono-tech text-[10px]" style={{ color: "var(--text-muted)" }}>
                              Batch
                            </span>
                          )}
                        </td>
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center gap-3 py-14 text-center">
            <div
              className="flex h-12 w-12 items-center justify-center rounded-2xl border"
              style={{
                borderColor: "var(--border-strong)",
                backgroundColor: "var(--bg-secondary)",
              }}
            >
              <Layers className="h-5 w-5" style={{ color: "var(--accent-clinical)" }} />
            </div>
            <div className="font-display text-lg font-bold">Adaptive Multi-Study Triage Queue</div>
            <p className="max-w-md text-sm" style={{ color: "var(--text-secondary)" }}>
              Execute concurrent RTX CUDA ensemble inference across multiple radiographs to elevate
              STAT critical studies (Pneumothorax, severe Edema, large Effusion) automatically.
            </p>
          </div>
        )}
      </SpotlightCard>

      {/* Hardware, Gemini Engine & Clinical Governance Deck */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SpotlightCard className="rounded-2xl p-5 space-y-2">
          <div
            className="flex items-center gap-2 font-mono-tech text-[10px] tracking-[0.16em] uppercase"
            style={{ color: "var(--accent-optical)" }}
          >
            <Cpu className="h-3.5 w-3.5" />
            <span>COMPUTE ENGINE</span>
          </div>
          <div className="font-display text-sm font-bold">
            {health?.gpu_name || (health?.device || "CUDA").toUpperCase()}
          </div>
          <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
            PyTorch FP16 ensemble across 3 medical CNN backbones + ChestX-Det PSPNet 512px
            segmentation.
          </p>
        </SpotlightCard>

        <SpotlightCard className="rounded-2xl p-5 space-y-2">
          <div
            className="flex items-center gap-2 font-mono-tech text-[10px] tracking-[0.16em] uppercase"
            style={{ color: "var(--accent-clinical)" }}
          >
            <Sparkles className="h-3.5 w-3.5" />
            <span>GEMINI CLINICAL STACK</span>
          </div>
          <div className="font-display text-sm font-bold">
            Decision 1 + {aiConfig?.gemini_flash_model || health?.gemini_flash_model || "gemini-3.5-flash-lite"}
          </div>
          <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
            Zero-output-token Decisions API calibration paired with stateful Interactions API ACR
            synthesis.
          </p>
        </SpotlightCard>

        <SpotlightCard className="rounded-2xl p-5 space-y-2">
          <div
            className="flex items-center gap-2 font-mono-tech text-[10px] tracking-[0.16em] uppercase"
            style={{ color: "var(--accent-titanium)" }}
          >
            <ShieldAlert className="h-3.5 w-3.5" />
            <span>CLINICAL GOVERNANCE</span>
          </div>
          <div className="font-display text-sm font-bold">92.9% Acc · 97.3% Specificity</div>
          <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
            Decision-support &amp; academic research workstation. All findings require board-certified
            radiologist verification.
          </p>
        </SpotlightCard>
      </div>
    </div>
  );
};
