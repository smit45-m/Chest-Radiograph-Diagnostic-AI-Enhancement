import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, ChevronDown, ChevronUp, Cpu, Layers, Sparkles, Timer } from "lucide-react";
import { formatPathology, type PageId } from "../data/clinicalCatalog";
import type { PredictionResponse } from "../types/clinical";
import { AnimatedNumber } from "./AnimatedNumber";
import { SpotlightCard } from "./SpotlightCard";

interface ConsensusArchitectureProps {
  predictionData: PredictionResponse | null;
  threshold: number;
  onInspectPathology: (pathology: string) => void;
  onOpenGemini: () => void;
  onNavigatePage: (page: PageId) => void;
}

const BACKBONE_KEYS = [
  {
    key: "densenet121_all",
    alt: "densenet121-all",
    label: "DN121-All",
    title: "DenseNet-121 · Multi-Cohort",
    spec: "500K+ CXRs · 224 px",
  },
  {
    key: "densenet121_nih",
    alt: "densenet121-nih",
    label: "DN121-NIH",
    title: "DenseNet-121 · NIH-14",
    spec: "112K NIH Specialist",
  },
  {
    key: "resnet50_512",
    alt: "resnet50-512",
    label: "RN50-512",
    title: "ResNet-50 · High-Res",
    spec: "512 px Fine Detail",
  },
] as const;

function getBackboneScore(
  map: Record<string, number> | undefined,
  primary: string,
  alt: string
): number | undefined {
  if (!map) return undefined;
  return map[primary] ?? map[alt];
}

export const ConsensusArchitecture: React.FC<ConsensusArchitectureProps> = ({
  predictionData,
  threshold,
  onInspectPathology,
  onOpenGemini,
  onNavigatePage,
}) => {
  const [showAllRows, setShowAllRows] = useState(false);

  const predictions = predictionData?.predictions || [];
  const anatomy = predictionData?.anatomy || null;
  const decision = predictionData?.decision_summary || null;
  const timings = predictionData?.timings || null;
  const activeBackbones = predictionData?.ensemble_info?.active_backbones || [];

  const rows = useMemo(() => {
    if (showAllRows) return predictions;
    return predictions.slice(0, 7);
  }, [predictions, showAllRows]);

  const ctr = anatomy?.ctr ?? 0.46;
  const heartRx = Math.min(44, Math.max(22, ctr * 68));
  const heartAbnormal = anatomy?.ctr != null && anatomy.ctr > 0.5;

  const timingBars = [
    {
      label: "3-Backbone Vision + PSPNet",
      ms: timings?.vision_ms ?? predictionData?.vision_latency_ms ?? 0,
      tint: "var(--accent-clinical)",
    },
    {
      label: "Grad-CAM++ Localization",
      ms: timings?.gradcam_ms ?? 0,
      tint: "var(--accent-optical)",
    },
    {
      label: "Gemini Decision 1 (Parallel)",
      ms: timings?.decision_api_ms ?? decision?.latency_ms ?? 0,
      tint: "var(--accent-clinical)",
    },
    {
      label: "Gemini Interactions ACR",
      ms: timings?.report_ms ?? 0,
      tint: "var(--accent-titanium)",
    },
  ];
  const maxTiming = Math.max(100, ...timingBars.map((t) => t.ms));

  return (
    <div className="mx-auto max-w-[1360px] px-5 py-7 pb-24 md:px-8 md:py-9 lg:pb-12 space-y-8">
      {/* Page Header */}
      <div className="flex flex-col justify-between gap-5 md:flex-row md:items-end">
        <div className="space-y-1.5">
          <div
            className="font-mono-tech text-[10px] font-medium tracking-[0.2em] uppercase"
            style={{ color: "var(--accent-clinical)" }}
          >
            03 · MULTI-MODEL CONSENSUS &amp; QUANTITATIVE BIOMETRY
          </div>
          <h1 className="font-display text-3xl font-bold sm:text-4xl">
            Four calibrated opinions,{" "}
            <span
              className="font-serif-editorial font-normal"
              style={{ color: "var(--accent-clinical)" }}
            >
              one fused verdict.
            </span>
          </h1>
        </div>

        <button
          type="button"
          onClick={() => onNavigatePage("report")}
          className="group flex shrink-0 cursor-pointer items-center gap-2 self-start rounded-full border px-4 py-2 text-xs font-semibold transition-colors md:self-auto"
          style={{
            borderColor: "var(--border-strong)",
            backgroundColor: "var(--bg-elevated)",
          }}
        >
          <span>Continue to ACR Report</span>
          <ArrowRight
            className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5"
            style={{ color: "var(--accent-clinical)" }}
          />
        </button>
      </div>

      {/* 4 Calibrated Model Cards with Micro Probability Bars */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {BACKBONE_KEYS.map((b, idx) => {
          const loaded =
            activeBackbones.some((ab) => ab === b.key || ab === b.alt) ||
            activeBackbones.length > idx;
          const leadVal = getBackboneScore(predictions[0]?.backbones, b.key, b.alt);
          const leadPct = leadVal != null ? leadVal * 100 : 0;

          return (
            <motion.div
              key={b.key}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, delay: idx * 0.05, ease: [0.16, 1, 0.3, 1] }}
            >
              <SpotlightCard className="rounded-2xl p-5 space-y-3.5">
                <div className="flex items-center justify-between">
                  <span
                    className="font-mono-tech text-[10px] tracking-[0.16em] uppercase"
                    style={{ color: "var(--text-muted)" }}
                  >
                    BACKBONE 0{idx + 1}
                  </span>
                  <span className="flex items-center gap-1.5 font-mono-tech text-[10px]">
                    <span
                      className="h-1.5 w-1.5 rounded-full"
                      style={{
                        backgroundColor: loaded ? "var(--accent-clinical)" : "var(--text-muted)",
                      }}
                    />
                    <span style={{ color: loaded ? "var(--accent-clinical)" : "var(--text-muted)" }}>
                      {loaded ? "ACTIVE" : "STANDBY"}
                    </span>
                  </span>
                </div>

                <div>
                  <div className="font-display text-[15px] font-bold">{b.title}</div>
                  <div className="mt-0.5 text-xs" style={{ color: "var(--text-secondary)" }}>
                    {b.spec}
                  </div>
                </div>

                <div
                  className="space-y-2 border-t pt-3"
                  style={{ borderColor: "var(--border-subtle)" }}
                >
                  <div className="flex items-center justify-between font-mono-tech text-[11px]">
                    <span style={{ color: "var(--text-muted)" }}>
                      {predictions[0] ? formatPathology(predictions[0].name) : "Lead Score"}
                    </span>
                    <AnimatedNumber
                      value={leadVal != null ? leadPct : null}
                      decimals={1}
                      suffix="%"
                      className="font-semibold"
                    />
                  </div>
                  <div
                    className="h-1 w-full overflow-hidden rounded-full"
                    style={{ backgroundColor: "var(--border-subtle)" }}
                  >
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(100, leadPct)}%` }}
                      transition={{ duration: 0.65, ease: [0.16, 1, 0.3, 1] }}
                      className="h-full rounded-full"
                      style={{ backgroundColor: "var(--accent-optical)" }}
                    />
                  </div>
                </div>
              </SpotlightCard>
            </motion.div>
          );
        })}

        {/* 4th Card: Google Gemini Decision 1 */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.16, ease: [0.16, 1, 0.3, 1] }}
        >
          <SpotlightCard
            strong
            className="rounded-2xl p-5 space-y-3.5"
            style={{
              borderColor: decision?.active ? "var(--accent-clinical)" : "var(--border-strong)",
            }}
          >
            <div className="flex items-center justify-between">
              <span
                className="flex items-center gap-1.5 font-mono-tech text-[10px] font-medium tracking-[0.16em] uppercase"
                style={{ color: "var(--accent-clinical)" }}
              >
                <Sparkles className="h-3 w-3" />
                DECISION ENGINE
              </span>
              <span
                className="font-mono-tech text-[10px]"
                style={{
                  color: decision?.active ? "var(--accent-clinical)" : "var(--text-muted)",
                }}
              >
                {decision?.active ? "0 OUT TOK" : "OPTIONAL"}
              </span>
            </div>

            <div>
              <div className="font-display text-[15px] font-bold">Gemini Decision 1</div>
              <div className="mt-0.5 text-xs" style={{ color: "var(--text-secondary)" }}>
                {decision?.active
                  ? `${(decision.projection_view || "PA_UPRIGHT").replace(/_/g, " ")} · ${
                      decision.triage_acuity || "TRIAGED"
                    }`
                  : "Calibrated v1beta/decisions vision fusion"}
              </div>
            </div>

            <div
              className="space-y-2 border-t pt-3"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              {decision?.active ? (
                <>
                  <div className="flex items-center justify-between font-mono-tech text-[11px]">
                    <span style={{ color: "var(--text-muted)" }}>
                      Quality{" "}
                      {decision.image_quality_score != null
                        ? `${decision.image_quality_score.toFixed(1)}/5`
                        : "Verified"}
                    </span>
                    <span className="font-semibold" style={{ color: "var(--accent-clinical)" }}>
                      {predictions[0]?.gemini_decision_score != null ? (
                        <AnimatedNumber
                          value={predictions[0].gemini_decision_score * 100}
                          decimals={1}
                          suffix="%"
                        />
                      ) : (
                        `${Math.round(decision.latency_ms)} ms`
                      )}
                    </span>
                  </div>
                  <div
                    className="h-1 w-full overflow-hidden rounded-full"
                    style={{ backgroundColor: "var(--border-subtle)" }}
                  >
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{
                        width: `${Math.min(
                          100,
                          (predictions[0]?.gemini_decision_score ?? 0) * 100
                        )}%`,
                      }}
                      transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
                      className="h-full rounded-full"
                      style={{ backgroundColor: "var(--accent-clinical)" }}
                    />
                  </div>
                </>
              ) : (
                <button
                  type="button"
                  onClick={onOpenGemini}
                  className="cursor-pointer font-mono-tech text-[10.5px] underline underline-offset-4"
                  style={{ color: "var(--accent-clinical)" }}
                >
                  Configure Gemini Key →
                </button>
              )}
            </div>
          </SpotlightCard>
        </motion.div>
      </div>

      {/* Lower 2-Column Showcase: Living Thoracic SVG Schematic + Calibration Matrix */}
      <div className="grid grid-cols-1 gap-7 lg:grid-cols-12">
        {/* LEFT 5 COLS: Living Breathing Thoracic Geometry + Inference Latency */}
        <div className="lg:col-span-5 space-y-6">
          <SpotlightCard className="rounded-[26px] p-6 space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <div
                  className="font-mono-tech text-[10px] tracking-[0.18em] uppercase"
                  style={{ color: "var(--text-muted)" }}
                >
                  PSPNET 512 PX BIOMETRY
                </div>
                <h2 className="mt-1 font-display text-lg font-bold">
                  Thoracic Geometry Schematic
                </h2>
              </div>
              <span
                className="rounded-full border px-3 py-1 font-mono-tech text-[10.5px] font-medium"
                style={{
                  borderColor: heartAbnormal ? "var(--accent-alert)" : "var(--accent-clinical)",
                  backgroundColor: heartAbnormal
                    ? "var(--accent-alert-dim)"
                    : "var(--accent-clinical-dim)",
                  color: heartAbnormal ? "var(--accent-alert)" : "var(--accent-clinical)",
                }}
              >
                CTR <AnimatedNumber value={anatomy?.ctr ?? null} decimals={3} />
              </span>
            </div>

            {/* Living Breathing & Systolic SVG Thoracic Diagram */}
            <div
              className="relative flex items-center justify-center overflow-hidden rounded-2xl border py-5"
              style={{
                borderColor: "var(--border-subtle)",
                backgroundColor: "var(--bg-secondary)",
              }}
            >
              <svg
                viewBox="0 0 280 214"
                className="h-52 w-full max-w-[310px]"
                fill="none"
                aria-label="Quantitative thoracic biometry schematic"
              >
                <line
                  x1="140"
                  y1="10"
                  x2="140"
                  y2="196"
                  stroke="var(--border-strong)"
                  strokeDasharray="3 4"
                />
                <line
                  x1="22"
                  y1="128"
                  x2="258"
                  y2="128"
                  stroke="var(--border-subtle)"
                  strokeDasharray="2 4"
                />

                {/* Outer Thoracic Cage */}
                <motion.path
                  d="M84 30 C52 50, 38 112, 44 166 C82 178, 198 178, 236 166 C242 112, 228 50, 196 30"
                  stroke="var(--border-strong)"
                  strokeWidth="1.25"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
                />

                {/* Right Lung Field (gentle 5s respiratory fill modulation) */}
                <motion.path
                  d="M126 38 C98 44, 62 78, 56 154 C80 162, 114 156, 126 142 Z"
                  fill="var(--accent-clinical-dim)"
                  stroke="var(--accent-clinical)"
                  strokeWidth="1.35"
                  animate={{ fillOpacity: [0.55, 1, 0.55] }}
                  transition={{ duration: 5.2, repeat: Infinity, ease: "easeInOut" }}
                />

                {/* Left Lung Field */}
                <motion.path
                  d="M154 38 C182 44, 218 78, 224 154 C200 162, 166 156, 154 142 Z"
                  fill="var(--accent-clinical-dim)"
                  stroke="var(--accent-clinical)"
                  strokeWidth="1.35"
                  animate={{ fillOpacity: [0.55, 1, 0.55] }}
                  transition={{ duration: 5.2, repeat: Infinity, ease: "easeInOut" }}
                />

                {/* Cardiac Silhouette with subtle systolic pulse & dynamic CTR scaling */}
                <motion.ellipse
                  cx="148"
                  cy="128"
                  rx={heartRx}
                  ry="34"
                  fill={heartAbnormal ? "var(--accent-alert-dim)" : "var(--accent-optical-dim)"}
                  stroke={heartAbnormal ? "var(--accent-alert)" : "var(--accent-optical)"}
                  strokeWidth="1.7"
                  animate={{
                    rx: [heartRx, heartRx + 1.4, heartRx],
                    ry: [34, 34.9, 34],
                  }}
                  transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
                />

                {/* Cardiac Diameter Caliper (C) */}
                <line
                  x1={148 - heartRx}
                  y1="128"
                  x2={148 + heartRx}
                  y2="128"
                  stroke={heartAbnormal ? "var(--accent-alert)" : "var(--text-primary)"}
                  strokeWidth="1.6"
                />
                <circle
                  cx={148 - heartRx}
                  cy="128"
                  r="2.4"
                  fill={heartAbnormal ? "var(--accent-alert)" : "var(--text-primary)"}
                />
                <circle
                  cx={148 + heartRx}
                  cy="128"
                  r="2.4"
                  fill={heartAbnormal ? "var(--accent-alert)" : "var(--text-primary)"}
                />

                {/* Thoracic Width Caliper (T) */}
                <line x1="56" y1="184" x2="224" y2="184" stroke="var(--accent-titanium)" strokeWidth="1.25" />
                <line x1="56" y1="178" x2="56" y2="190" stroke="var(--accent-titanium)" strokeWidth="1.25" />
                <line x1="224" y1="178" x2="224" y2="190" stroke="var(--accent-titanium)" strokeWidth="1.25" />

                <text
                  x="140"
                  y="204"
                  textAnchor="middle"
                  fill="var(--text-muted)"
                  fontSize="9"
                  fontFamily="IBM Plex Mono"
                >
                  THORAX SPAN (T) · CTR = C / T
                </text>
              </svg>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "R. Lung Area", val: anatomy?.right_lung_pct ?? null, suffix: "%", dec: 1 },
                { label: "L. Lung Area", val: anatomy?.left_lung_pct ?? null, suffix: "%", dec: 1 },
                { label: "Symmetry", val: anatomy?.lung_symmetry_ratio ?? null, suffix: "", dec: 2 },
              ].map((m) => (
                <div
                  key={m.label}
                  className="rounded-xl border p-3 text-center"
                  style={{ borderColor: "var(--border-subtle)" }}
                >
                  <div
                    className="font-mono-tech text-[9.5px] tracking-[0.12em] uppercase"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {m.label}
                  </div>
                  <div className="mt-1 font-mono-tech text-sm font-semibold">
                    <AnimatedNumber value={m.val} decimals={m.dec} suffix={m.suffix} />
                  </div>
                </div>
              ))}
            </div>
          </SpotlightCard>

          {/* Inference Latency Telemetry */}
          <SpotlightCard className="rounded-[26px] p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 font-display text-sm font-bold">
                <Timer className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                Inference Latency Profile
              </span>
              <span
                className="flex items-center gap-1.5 font-mono-tech text-[10px]"
                style={{ color: "var(--text-muted)" }}
              >
                <Cpu className="h-3 w-3" style={{ color: "var(--accent-optical)" }} />
                {(predictionData?.ensemble_info?.device || "CUDA").toUpperCase()}
              </span>
            </div>

            <div className="space-y-3">
              {timingBars.map((t) => {
                const pct = Math.min(100, (t.ms / maxTiming) * 100);
                return (
                  <div key={t.label} className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span style={{ color: "var(--text-secondary)" }}>{t.label}</span>
                      <span className="font-mono-tech text-[11px] font-medium">
                        {t.ms > 0 ? `${Math.round(t.ms)} ms` : "—"}
                      </span>
                    </div>
                    <div
                      className="h-1.5 w-full overflow-hidden rounded-full"
                      style={{ backgroundColor: "var(--border-subtle)" }}
                    >
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ duration: 0.75, ease: [0.16, 1, 0.3, 1] }}
                        className="h-full rounded-full"
                        style={{ backgroundColor: t.tint }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </SpotlightCard>
        </div>

        {/* RIGHT 7 COLS: Backbone-to-Consensus Calibration Matrix */}
        <div className="lg:col-span-7">
          <SpotlightCard strong className="rounded-[26px] p-6 space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                <h2 className="font-display text-lg font-bold">
                  Backbone-to-Consensus Calibration Matrix
                </h2>
              </div>
              <span
                className="font-mono-tech text-[10px] uppercase"
                style={{ color: "var(--text-muted)" }}
              >
                Click any row to localize in Studio
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr
                    className="border-b font-mono-tech text-[10px] tracking-[0.12em] uppercase"
                    style={{
                      borderColor: "var(--border-subtle)",
                      color: "var(--text-muted)",
                    }}
                  >
                    <th className="py-3 pr-3 font-medium">Pathology</th>
                    <th className="px-2 py-3 text-right font-medium">DN121-All</th>
                    <th className="px-2 py-3 text-right font-medium">DN121-NIH</th>
                    <th className="px-2 py-3 text-right font-medium">RN50-512</th>
                    <th className="px-2 py-3 text-right font-medium">Decision 1</th>
                    <th className="px-2 py-3 text-right font-medium">Fused (95% CI)</th>
                    <th className="py-3 pl-2 text-right font-medium">Agree</th>
                  </tr>
                </thead>
                <tbody className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                  {rows.map((p) => {
                    const isPos = p.score >= threshold;
                    const dnAll = getBackboneScore(p.backbones, "densenet121_all", "densenet121-all");
                    const dnNih = getBackboneScore(p.backbones, "densenet121_nih", "densenet121-nih");
                    const rn50 = getBackboneScore(p.backbones, "resnet50_512", "resnet50-512");
                    const dec = p.gemini_decision_score;
                    const agreePct = p.agreement != null ? `${Math.round(p.agreement * 100)}%` : "—";
                    const ciLabel =
                      p.ci_low != null && p.ci_high != null
                        ? `${Math.round(p.ci_low * 100)}–${Math.round(p.ci_high * 100)}%`
                        : null;

                    return (
                      <tr
                        key={p.name}
                        onClick={() => onInspectPathology(p.name)}
                        className="group cursor-pointer transition-colors hover:bg-teal-500/[0.04]"
                        style={{ borderColor: "var(--border-subtle)" }}
                      >
                        <td className="py-3.5 pr-3 font-medium">
                          <div className="flex items-center gap-2">
                            <span
                              className="h-1.5 w-1.5 rounded-full"
                              style={{
                                backgroundColor: isPos
                                  ? "var(--accent-alert)"
                                  : "var(--border-strong)",
                              }}
                            />
                            <span className="group-hover:underline">
                              {formatPathology(p.name)}
                            </span>
                          </div>
                        </td>
                        <td
                          className="px-2 py-3.5 text-right font-mono-tech"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {dnAll != null ? `${(dnAll * 100).toFixed(0)}%` : "—"}
                        </td>
                        <td
                          className="px-2 py-3.5 text-right font-mono-tech"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {dnNih != null ? `${(dnNih * 100).toFixed(0)}%` : "—"}
                        </td>
                        <td
                          className="px-2 py-3.5 text-right font-mono-tech"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {rn50 != null ? `${(rn50 * 100).toFixed(0)}%` : "—"}
                        </td>
                        <td
                          className="px-2 py-3.5 text-right font-mono-tech"
                          style={{
                            color: dec != null ? "var(--accent-clinical)" : "var(--text-muted)",
                          }}
                        >
                          {dec != null ? `${(dec * 100).toFixed(0)}%` : "—"}
                        </td>
                        <td className="px-2 py-3.5 text-right">
                          <div
                            className="font-mono-tech font-semibold"
                            style={{
                              color: isPos ? "var(--accent-alert)" : "var(--text-primary)",
                            }}
                          >
                            {(p.score * 100).toFixed(1)}%
                          </div>
                          {ciLabel && (
                            <div
                              className="font-mono-tech text-[9.5px]"
                              style={{ color: "var(--text-muted)" }}
                            >
                              {ciLabel}
                            </div>
                          )}
                        </td>
                        <td
                          className="py-3.5 pl-2 text-right font-mono-tech"
                          style={{ color: "var(--text-muted)" }}
                        >
                          {agreePct}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {predictions.length > 7 && (
              <button
                type="button"
                onClick={() => setShowAllRows((v) => !v)}
                className="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-xl border py-2.5 font-mono-tech text-[10px] tracking-[0.14em] uppercase transition-colors"
                style={{
                  borderColor: "var(--border-subtle)",
                  color: "var(--text-secondary)",
                }}
              >
                {showAllRows ? (
                  <>
                    <span>Collapse to Top 7</span>
                    <ChevronUp className="h-3.5 w-3.5" />
                  </>
                ) : (
                  <>
                    <span>Show All {predictions.length} Pathologies</span>
                    <ChevronDown className="h-3.5 w-3.5" />
                  </>
                )}
              </button>
            )}
          </SpotlightCard>
        </div>
      </div>
    </div>
  );
};
