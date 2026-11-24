import React, { useMemo, useRef } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  ArrowUpRight,
  Cpu,
  FolderUp,
  Layers,
  Play,
  Sparkles,
} from "lucide-react";
import {
  formatPathology,
  formatTriageLabel,
  shortSampleTitle,
} from "../data/clinicalCatalog";
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

  const sampleByFilename = useMemo(() => {
    const map = new Map<string, SampleCase>();
    samples.forEach((s) => map.set(s.filename, s));
    return map;
  }, [samples]);

  return (
    <div className="mx-auto max-w-[1280px] px-5 py-6 pb-24 md:px-8 md:py-7 lg:pb-10 space-y-6">
      {/* Compact Header */}
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <h1 className="font-display text-2xl font-bold sm:text-3xl">
            Batch{" "}
            <span
              className="font-serif-editorial font-normal"
              style={{ color: "var(--accent-clinical)" }}
            >
              Triage
            </span>
          </h1>
          <p className="mt-0.5 text-xs" style={{ color: "var(--text-muted)" }}>
            Screen multiple chest X-rays at once and sort urgent studies automatically.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
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
            className="flex cursor-pointer items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold transition-all hover:brightness-110 active:scale-[0.98] disabled:opacity-50"
            style={{
              backgroundColor: "var(--accent-clinical)",
              color: "#ffffff",
            }}
          >
            <Play className="h-3.5 w-3.5" />
            <span>
              {batchLoading
                ? "Triaging Queue..."
                : `Triage All ${samples.length || 5} Samples`}
            </span>
          </button>

          <button
            type="button"
            disabled={batchLoading}
            onClick={() => multiInputRef.current?.click()}
            className="flex cursor-pointer items-center gap-2 rounded-full border px-4 py-2 text-xs font-medium transition-colors hover:border-teal-500/40 disabled:opacity-50"
            style={{
              borderColor: "var(--border-strong)",
              backgroundColor: "var(--bg-elevated)",
            }}
          >
            <FolderUp className="h-3.5 w-3.5" />
            <span>Upload Multiple</span>
          </button>
        </div>
      </div>

      {/* Main Worklist Table */}
      <SpotlightCard strong className="rounded-[24px] p-5 sm:p-6 space-y-5">
        {batchResult ? (
          <>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              {[
                { label: "Total Scans", val: batchResult.total_studies, suffix: "" },
                {
                  label: "STAT Critical",
                  val: batchResult.stat_count,
                  suffix: "",
                  alert: batchResult.stat_count > 0,
                },
                { label: "Avg per Scan", val: batchResult.avg_latency_ms, suffix: " ms" },
                { label: "Total Queue Time", val: batchResult.total_latency_ms, suffix: " ms" },
              ].map((kpi) => (
                <div
                  key={kpi.label}
                  className="rounded-2xl border p-3.5"
                  style={{
                    borderColor: "var(--border-subtle)",
                    backgroundColor: "var(--bg-secondary)",
                  }}
                >
                  <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                    {kpi.label}
                  </div>
                  <div
                    className="mt-1 font-mono-tech text-xl font-semibold"
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
                    className="border-b text-[11px]"
                    style={{
                      borderColor: "var(--border-subtle)",
                      color: "var(--text-muted)",
                    }}
                  >
                    <th className="py-3 pr-4 font-medium">Priority</th>
                    <th className="px-3 py-3 font-medium">Study</th>
                    <th className="px-3 py-3 font-medium">Top Findings</th>
                    <th className="px-3 py-3 text-right font-medium">CTR</th>
                    <th className="px-3 py-3 text-right font-medium">Speed</th>
                    <th className="py-3 pl-3 text-right font-medium">Open</th>
                  </tr>
                </thead>
                <tbody className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                  {batchResult.studies.map((st, idx) => {
                    const isStat =
                      st.critical_alert || st.triage_level.toUpperCase().includes("STAT");
                    const isUrgent = st.triage_level.toUpperCase().includes("URGENT");
                    const matched = sampleByFilename.get(st.filename);

                    return (
                      <motion.tr
                        key={`${st.filename}-${idx}`}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.035, duration: 0.22 }}
                        style={{ borderColor: "var(--border-subtle)" }}
                        className="transition-colors hover:bg-teal-500/[0.04]"
                      >
                        <td className="py-3.5 pr-4">
                          <span
                            className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold"
                            style={{
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
                            {formatTriageLabel(st.triage_level)}
                          </span>
                        </td>
                        <td className="px-3 py-3.5">
                          <div className="text-[13px] font-semibold">
                            {matched ? shortSampleTitle(matched.name) : st.filename}
                          </div>
                          {matched && (
                            <div
                              className="font-mono-tech text-[10px]"
                              style={{ color: "var(--text-muted)" }}
                            >
                              {st.filename}
                            </div>
                          )}
                        </td>
                        <td className="px-3 py-3.5">
                          {st.error ? (
                            <span style={{ color: "var(--accent-alert)" }}>{st.error}</span>
                          ) : st.top_findings.length > 0 ? (
                            <div className="flex flex-wrap gap-1.5">
                              {st.top_findings.slice(0, 3).map((f) => (
                                <span
                                  key={f.name}
                                  className="rounded-full border px-2.5 py-0.5 text-[11px]"
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
                          {matched ? (
                            <button
                              type="button"
                              onClick={() => onInspectSampleByFilename(st.filename)}
                              className="inline-flex cursor-pointer items-center gap-1 rounded-full border px-3 py-1 text-xs font-medium transition-colors hover:border-teal-500/50"
                              style={{
                                borderColor: "var(--border-subtle)",
                                color: "var(--accent-clinical)",
                              }}
                            >
                              <span>Inspect</span>
                              <ArrowUpRight className="h-3 w-3" />
                            </button>
                          ) : (
                            <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                              Uploaded
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
                borderColor: "var(--border-subtle)",
                backgroundColor: "var(--bg-secondary)",
              }}
            >
              <Layers className="h-5 w-5" style={{ color: "var(--accent-clinical)" }} />
            </div>
            <div className="font-display text-base font-bold">Ready to Triage a Batch</div>
            <p className="max-w-md text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
              Click <strong>Triage All {samples.length || 5} Samples</strong> above or upload
              multiple chest X-rays to automatically rank critical studies to the top.
            </p>
          </div>
        )}
      </SpotlightCard>

      {/* Quiet Single-Line System Info Strip */}
      <div
        className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border px-4 py-3 text-xs"
        style={{
          borderColor: "var(--border-subtle)",
          backgroundColor: "var(--bg-elevated)",
          color: "var(--text-muted)",
        }}
      >
        <span className="flex items-center gap-2">
          <Cpu className="h-3.5 w-3.5" style={{ color: "var(--accent-optical)" }} />
          <span>{health?.gpu_name || (health?.device || "CPU").toUpperCase()}</span>
        </span>
        <span className="flex items-center gap-2">
          <Sparkles className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
          <span>
            Decision 1 + {aiConfig?.gemini_flash_model || health?.gemini_flash_model || "gemini-3.5-flash-lite"}
          </span>
        </span>
      </div>
    </div>
  );
};
