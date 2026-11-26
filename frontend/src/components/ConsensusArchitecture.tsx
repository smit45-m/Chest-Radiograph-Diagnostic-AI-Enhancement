import React, { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Cpu,
  Layers,
  Sparkles,
  Timer,
} from "lucide-react";
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
    short: "Multi-Cohort",
    title: "Multi-Cohort CNN",
    spec: "DenseNet-121 · 500K+ scans",
  },
  {
    key: "densenet121_nih",
    alt: "densenet121-nih",
    short: "NIH-14",
    title: "NIH-14 Specialist",
    spec: "DenseNet-121 · 112K scans",
  },
  {
    key: "densenet121_pc",
    alt: "densenet121-pc",
    short: "PadChest",
    title: "PadChest 160k",
    spec: "DenseNet-121 · EU hospital",
  },
  {
    key: "resnet50_512",
    alt: "resnet50-512",
    short: "High-Res",
    title: "High-Res 512px",
    spec: "ResNet-50 · fine texture",
  },
] as const;

const RADDINO_SPEC = {
  key: "raddino_vit_518",
  alt: "raddino-vit-518",
  short: "RAD-DINO",
  title: "RAD-DINO ViT",
  spec: "Microsoft ViT-B 518px · 838k CXRs",
} as const;

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
  const [showTimings, setShowTimings] = useState(false);

  const predictions = predictionData?.predictions || [];
  const anatomy = predictionData?.anatomy || null;
  const decision = predictionData?.decision_summary || null;
  const timings = predictionData?.timings || null;
  const activeBackbones = predictionData?.ensemble_info?.active_backbones || [];
  const hasRadDino =
    Boolean(predictionData?.ensemble_info?.raddino_active) ||
    activeBackbones.includes("raddino_vit_518") ||
    predictions.some((p) => p.backbones?.raddino_vit_518 != null);
  const displayedBackbones = useMemo(
    () => (hasRadDino ? [...BACKBONE_KEYS, RADDINO_SPEC] : [...BACKBONE_KEYS]),
    [hasRadDino]
  );
  const totalModels = displayedBackbones.length + 1;

  const rows = useMemo(() => {
    if (showAllRows) return predictions;
    return predictions.slice(0, 6);
  }, [predictions, showAllRows]);

  const ctr = anatomy?.ctr ?? 0.46;
  const heartRx = Math.min(44, Math.max(22, ctr * 68));
  const heartAbnormal = anatomy?.ctr != null && anatomy.ctr > 0.5;

  const timingBars = [
    {
      label: "Vision Ensemble & Anatomy",
      ms: timings?.vision_ms ?? predictionData?.vision_latency_ms ?? 0,
      tint: "var(--accent-clinical)",
    },
    {
      label: "Grad-CAM++ Heatmaps",
      ms: timings?.gradcam_ms ?? 0,
      tint: "var(--accent-optical)",
    },
    {
      label: "Gemini Decision 1",
      ms: timings?.decision_api_ms ?? decision?.latency_ms ?? 0,
      tint: "var(--accent-clinical)",
    },
    {
      label: "Report Synthesis",
      ms: timings?.report_ms ?? 0,
      tint: "var(--accent-titanium)",
    },
  ];
  const maxTiming = Math.max(100, ...timingBars.map((t) => t.ms));

  return (
    <div className="mx-auto max-w-[1280px] px-5 py-6 pb-24 md:px-8 md:py-7 lg:pb-10 space-y-6">
      {/* Compact Header */}
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <h1 className="font-display text-2xl font-bold sm:text-3xl">
            Model Consensus &amp;{" "}
            <span
              className="font-serif-editorial font-normal"
              style={{ color: "var(--accent-clinical)" }}
            >
              Biometry
            </span>
          </h1>
          <p className="mt-0.5 text-xs" style={{ color: "var(--text-muted)" }}>
            See how all {totalModels} models agree and inspect automated heart &amp; lung geometry.
          </p>
        </div>

        <button
          type="button"
          onClick={() => onNavigatePage("report")}
          className="group flex shrink-0 cursor-pointer items-center gap-2 self-start rounded-full border px-4 py-2 text-xs font-semibold transition-colors hover:border-teal-500/40 sm:self-auto"
          style={{
            borderColor: "var(--border-strong)",
            backgroundColor: "var(--bg-elevated)",
          }}
        >
          <span>Open Report &amp; Copilot</span>
          <ArrowRight
            className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5"
            style={{ color: "var(--accent-clinical)" }}
          />
        </button>
      </div>

      {/* Clean Model Opinion Cards (5 on CPU, 6 with RAD-DINO ViT on GPU) */}
      <div
        className={`grid grid-cols-1 gap-3.5 sm:grid-cols-2 lg:grid-cols-3 ${
          hasRadDino ? "xl:grid-cols-6" : "xl:grid-cols-5"
        }`}
      >
        {displayedBackbones.map((b, idx) => {
          const loaded =
            activeBackbones.some((ab) => ab === b.key || ab === b.alt) ||
            activeBackbones.length > idx;
          const leadVal = getBackboneScore(predictions[0]?.backbones, b.key, b.alt);
          const leadPct = leadVal != null ? leadVal * 100 : 0;

          return (
            <SpotlightCard key={b.key} className="rounded-2xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="font-display text-sm font-semibold">{b.title}</div>
                <span
                  className="h-2 w-2 rounded-full"
                  title={loaded ? "Active" : "Standby"}
                  style={{
                    backgroundColor: loaded ? "var(--accent-clinical)" : "var(--text-muted)",
                  }}
                />
              </div>

              <div className="text-[11.5px]" style={{ color: "var(--text-muted)" }}>
                {b.spec}
              </div>

              <div className="space-y-1.5 pt-1">
                <div className="flex items-center justify-between text-xs">
                  <span style={{ color: "var(--text-secondary)" }}>
                    {predictions[0] ? formatPathology(predictions[0].name) : "Lead score"}
                  </span>
                  <AnimatedNumber
                    value={leadVal != null ? leadPct : null}
                    decimals={1}
                    suffix="%"
                    className="font-mono-tech font-semibold"
                  />
                </div>
                <div
                  className="h-1 w-full overflow-hidden rounded-full"
                  style={{ backgroundColor: "var(--border-subtle)" }}
                >
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, leadPct)}%` }}
                    transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
                    className="h-full rounded-full"
                    style={{ backgroundColor: "var(--accent-optical)" }}
                  />
                </div>
              </div>
            </SpotlightCard>
          );
        })}

        {/* 4th Card: Gemini Decision 1 */}
        <SpotlightCard
          strong
          className="rounded-2xl p-4 space-y-3"
          style={{
            borderColor: decision?.active ? "var(--accent-clinical)" : "var(--border-subtle)",
          }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 font-display text-sm font-semibold">
              <Sparkles className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
              <span>Gemini Decision 1</span>
            </div>
            <span
              className="rounded-full px-2 py-0.5 font-mono-tech text-[9.5px] font-medium"
              style={{
                backgroundColor: decision?.active
                  ? "var(--accent-clinical-dim)"
                  : "var(--bg-secondary)",
                color: decision?.active ? "var(--accent-clinical)" : "var(--text-muted)",
              }}
            >
              {decision?.active ? "Active" : "Optional"}
            </span>
          </div>

          <div className="text-[11.5px]" style={{ color: "var(--text-muted)" }}>
            {decision?.active
              ? `Quality ${decision.image_quality_score != null ? `${decision.image_quality_score.toFixed(1)}/5` : "Verified"} · Calibrated fusion`
              : "Multimodal vision prior fusion"}
          </div>

          <div className="space-y-1.5 pt-1">
            {decision?.active ? (
              <>
                <div className="flex items-center justify-between text-xs">
                  <span style={{ color: "var(--text-secondary)" }}>
                    {predictions[0] ? formatPathology(predictions[0].name) : "Fusion score"}
                  </span>
                  <span className="font-mono-tech font-semibold" style={{ color: "var(--accent-clinical)" }}>
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
                    transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                    className="h-full rounded-full"
                    style={{ backgroundColor: "var(--accent-clinical)" }}
                  />
                </div>
              </>
            ) : (
              <button
                type="button"
                onClick={onOpenGemini}
                className="cursor-pointer text-xs font-medium underline underline-offset-4"
                style={{ color: "var(--accent-clinical)" }}
              >
                Connect Gemini Key →
              </button>
            )}
          </div>
        </SpotlightCard>
      </div>

      {/* Main 2-Column Area: Schematic + Clean Consensus Table */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* LEFT 5 COLS: Thoracic Geometry + Collapsible Timings */}
        <div className="lg:col-span-5 space-y-4">
          <SpotlightCard className="rounded-[24px] p-5 sm:p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="font-display text-base font-bold">Thoracic Geometry</h2>
                <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                  Automated heart &amp; lung segmentation
                </p>
              </div>
              <span
                className="rounded-full px-3 py-1 font-mono-tech text-xs font-semibold"
                style={{
                  backgroundColor: heartAbnormal
                    ? "var(--accent-alert-dim)"
                    : "var(--accent-clinical-dim)",
                  color: heartAbnormal ? "var(--accent-alert)" : "var(--accent-clinical)",
                }}
              >
                CTR <AnimatedNumber value={anatomy?.ctr ?? null} decimals={3} />
              </span>
            </div>

            {/* Breathing & Systolic SVG Thoracic Diagram */}
            <div
              className="relative flex items-center justify-center overflow-hidden rounded-2xl border py-4"
              style={{
                borderColor: "var(--border-subtle)",
                backgroundColor: "var(--bg-secondary)",
              }}
            >
              <svg
                viewBox="0 0 280 210"
                className="h-48 w-full max-w-[290px]"
                fill="none"
                aria-label="Quantitative thoracic biometry schematic"
              >
                <line
                  x1="140"
                  y1="10"
                  x2="140"
                  y2="192"
                  stroke="var(--border-strong)"
                  strokeDasharray="3 4"
                />
                <motion.path
                  d="M84 30 C52 50, 38 112, 44 164 C82 176, 198 176, 236 164 C242 112, 228 50, 196 30"
                  stroke="var(--border-strong)"
                  strokeWidth="1.2"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.75, ease: [0.22, 1, 0.36, 1] }}
                />

                <motion.path
                  d="M126 38 C98 44, 62 78, 56 152 C80 160, 114 154, 126 140 Z"
                  fill="var(--accent-clinical-dim)"
                  stroke="var(--accent-clinical)"
                  strokeWidth="1.3"
                  animate={{ fillOpacity: [0.55, 0.95, 0.55] }}
                  transition={{ duration: 5.2, repeat: Infinity, ease: "easeInOut" }}
                />

                <motion.path
                  d="M154 38 C182 44, 218 78, 224 152 C200 160, 166 154, 154 140 Z"
                  fill="var(--accent-clinical-dim)"
                  stroke="var(--accent-clinical)"
                  strokeWidth="1.3"
                  animate={{ fillOpacity: [0.55, 0.95, 0.55] }}
                  transition={{ duration: 5.2, repeat: Infinity, ease: "easeInOut" }}
                />

                <motion.ellipse
                  cx="148"
                  cy="126"
                  rx={heartRx}
                  ry="33"
                  fill={heartAbnormal ? "var(--accent-alert-dim)" : "var(--accent-optical-dim)"}
                  stroke={heartAbnormal ? "var(--accent-alert)" : "var(--accent-optical)"}
                  strokeWidth="1.6"
                  animate={{
                    rx: [heartRx, heartRx + 1.2, heartRx],
                    ry: [33, 33.8, 33],
                  }}
                  transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
                />

                <line
                  x1={148 - heartRx}
                  y1="126"
                  x2={148 + heartRx}
                  y2="126"
                  stroke={heartAbnormal ? "var(--accent-alert)" : "var(--text-primary)"}
                  strokeWidth="1.5"
                />
                <line x1="56" y1="182" x2="224" y2="182" stroke="var(--accent-titanium)" strokeWidth="1.2" />
                <text
                  x="140"
                  y="201"
                  textAnchor="middle"
                  fill="var(--text-muted)"
                  fontSize="9"
                  fontFamily="IBM Plex Mono"
                >
                  HEART WIDTH / THORAX WIDTH
                </text>
              </svg>
            </div>

            <div className="grid grid-cols-3 gap-2.5">
              {[
                { label: "Right Lung", val: anatomy?.right_lung_pct ?? null, suffix: "%", dec: 1 },
                { label: "Left Lung", val: anatomy?.left_lung_pct ?? null, suffix: "%", dec: 1 },
                { label: "Symmetry", val: anatomy?.lung_symmetry_ratio ?? null, suffix: "", dec: 2 },
              ].map((m) => (
                <div
                  key={m.label}
                  className="rounded-xl border p-3 text-center"
                  style={{ borderColor: "var(--border-subtle)" }}
                >
                  <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                    {m.label}
                  </div>
                  <div className="mt-1 font-mono-tech text-sm font-semibold">
                    <AnimatedNumber value={m.val} decimals={m.dec} suffix={m.suffix} />
                  </div>
                </div>
              ))}
            </div>

            {/* Collapsible Pipeline Timings */}
            <div className="border-t pt-3" style={{ borderColor: "var(--border-subtle)" }}>
              <button
                type="button"
                onClick={() => setShowTimings((v) => !v)}
                className="flex w-full cursor-pointer items-center justify-between text-xs font-medium"
                style={{ color: "var(--text-secondary)" }}
              >
                <span className="flex items-center gap-2">
                  <Timer className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
                  <span>Pipeline Latency Breakdown</span>
                </span>
                <span className="flex items-center gap-1.5 font-mono-tech text-[11px]" style={{ color: "var(--text-muted)" }}>
                  <Cpu className="h-3 w-3" />
                  {(predictionData?.ensemble_info?.device || "CUDA").toUpperCase()}
                  {showTimings ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                </span>
              </button>

              <AnimatePresence initial={false}>
                {showTimings && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="mt-3.5 space-y-2.5">
                      {timingBars.map((t) => {
                        const pct = Math.min(100, (t.ms / maxTiming) * 100);
                        return (
                          <div key={t.label} className="space-y-1">
                            <div className="flex items-center justify-between text-xs">
                              <span style={{ color: "var(--text-secondary)" }}>{t.label}</span>
                              <span className="font-mono-tech text-[11px]">
                                {t.ms > 0 ? `${Math.round(t.ms)} ms` : "—"}
                              </span>
                            </div>
                            <div
                              className="h-1 w-full overflow-hidden rounded-full"
                              style={{ backgroundColor: "var(--border-subtle)" }}
                            >
                              <div
                                className="h-full rounded-full transition-all duration-500"
                                style={{ width: `${pct}%`, backgroundColor: t.tint }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </SpotlightCard>
        </div>

        {/* RIGHT 7 COLS: Humanized Consensus Breakdown Table */}
        <div className="lg:col-span-7">
          <SpotlightCard strong className="rounded-[24px] p-5 sm:p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                <h2 className="font-display text-base font-bold">Model Agreement Matrix</h2>
              </div>
              <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                Click any row to inspect heatmap
              </span>
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
                    <th className="py-3 pr-3 font-medium">Condition</th>
                    <th className="px-2 py-3 text-right font-medium">Multi-Cohort</th>
                    <th className="px-2 py-3 text-right font-medium">NIH-14</th>
                    <th className="px-2 py-3 text-right font-medium">PadChest</th>
                    <th className="px-2 py-3 text-right font-medium">High-Res</th>
                    {hasRadDino && (
                      <th className="px-2 py-3 text-right font-medium">RAD-DINO</th>
                    )}
                    <th className="px-2 py-3 text-right font-medium">Gemini AI</th>
                    <th className="py-3 pl-2 text-right font-medium">Consensus</th>
                  </tr>
                </thead>
                <tbody className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                  {rows.map((p) => {
                    const isPos = p.score >= threshold;
                    const dnAll = getBackboneScore(p.backbones, "densenet121_all", "densenet121-all");
                    const dnNih = getBackboneScore(p.backbones, "densenet121_nih", "densenet121-nih");
                    const dnPc = getBackboneScore(p.backbones, "densenet121_pc", "densenet121-pc");
                    const rn50 = getBackboneScore(p.backbones, "resnet50_512", "resnet50-512");
                    const rad = getBackboneScore(p.backbones, "raddino_vit_518", "raddino-vit-518");
                    const dec = p.gemini_decision_score;

                    return (
                      <tr
                        key={p.name}
                        onClick={() => onInspectPathology(p.name)}
                        className="group cursor-pointer transition-colors hover:bg-teal-500/[0.045]"
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
                            <span className="text-[13px] group-hover:underline">
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
                          {dnPc != null ? `${(dnPc * 100).toFixed(0)}%` : "—"}
                        </td>
                        <td
                          className="px-2 py-3.5 text-right font-mono-tech"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {rn50 != null ? `${(rn50 * 100).toFixed(0)}%` : "—"}
                        </td>
                        {hasRadDino && (
                          <td
                            className="px-2 py-3.5 text-right font-mono-tech"
                            style={{ color: "var(--accent-optical)" }}
                          >
                            {rad != null ? `${(rad * 100).toFixed(0)}%` : "—"}
                          </td>
                        )}
                        <td
                          className="px-2 py-3.5 text-right font-mono-tech"
                          style={{
                            color: dec != null ? "var(--accent-clinical)" : "var(--text-muted)",
                          }}
                        >
                          {dec != null ? `${(dec * 100).toFixed(0)}%` : "—"}
                        </td>
                        <td className="py-3.5 pl-2 text-right">
                          <span
                            className="font-mono-tech text-[13px] font-semibold"
                            style={{
                              color: isPos ? "var(--accent-alert)" : "var(--text-primary)",
                            }}
                          >
                            {(p.score * 100).toFixed(1)}%
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {predictions.length > 6 && (
              <button
                type="button"
                onClick={() => setShowAllRows((v) => !v)}
                className="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-xl border py-2 text-xs font-medium transition-colors hover:border-teal-500/35"
                style={{
                  borderColor: "var(--border-subtle)",
                  color: "var(--text-secondary)",
                }}
              >
                {showAllRows ? (
                  <>
                    <span>Show Top 6 Only</span>
                    <ChevronUp className="h-3.5 w-3.5" />
                  </>
                ) : (
                  <>
                    <span>Show All {predictions.length} Conditions</span>
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
