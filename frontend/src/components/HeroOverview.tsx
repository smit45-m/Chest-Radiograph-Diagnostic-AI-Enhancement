import React, { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { ArrowRight, HeartPulse, ShieldCheck, Sparkles, Upload } from "lucide-react";
import {
  ARCHITECTURE_PILLARS,
  cleanSampleTitle,
  formatPathology,
  formatSampleSubtitle,
} from "../data/clinicalCatalog";
import { api } from "../services/api";
import type { HealthStatus, PredictionResponse, SampleCase, ThemeMode } from "../types/clinical";
import { AnimatedNumber } from "./AnimatedNumber";
import { SpotlightCard } from "./SpotlightCard";

interface HeroOverviewProps {
  theme: ThemeMode;
  health: HealthStatus | null;
  samples: SampleCase[];
  activeStudyId: string | null;
  imageUrl: string | null;
  predictionData: PredictionResponse | null;
  threshold: number;
  isAnalyzing: boolean;
  onSelectSample: (sample: SampleCase, openStudio?: boolean) => void;
  onUploadClick: () => void;
  onOpenStudio: () => void;
  onOpenConsensus: () => void;
}

export const HeroOverview: React.FC<HeroOverviewProps> = ({
  theme,
  health,
  samples,
  activeStudyId,
  imageUrl,
  predictionData,
  threshold,
  isAnalyzing,
  onSelectSample,
  onUploadClick,
  onOpenStudio,
  onOpenConsensus,
}) => {
  const waveCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const specimenRef = useRef<HTMLDivElement | null>(null);

  // Subtle living ICU PQRST cardiac sweep with luminous phosphor head + respiratory curve
  useEffect(() => {
    const canvas = waveCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let rafId = 0;
    let phase = 0;
    let sweepX = 0;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const ecgShape = (u: number) => {
      const m = ((u % 1) + 1) % 1;
      if (m > 0.14 && m < 0.22) return 0.14 * Math.sin(((m - 0.14) / 0.08) * Math.PI);
      if (m > 0.28 && m < 0.31) return -0.18 * Math.sin(((m - 0.28) / 0.03) * Math.PI);
      if (m >= 0.31 && m < 0.36) return 1.0 * Math.sin(((m - 0.31) / 0.05) * Math.PI);
      if (m >= 0.36 && m < 0.40) return -0.34 * Math.sin(((m - 0.36) / 0.04) * Math.PI);
      if (m > 0.48 && m < 0.62) return 0.25 * Math.sin(((m - 0.48) / 0.14) * Math.PI);
      return 0;
    };

    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      if (!prefersReduced) {
        phase += 0.0032;
        sweepX = (sweepX + 1.65) % Math.max(w, 1);
      }

      const isDark = theme === "dark";
      const teal = isDark ? "20, 184, 166" : "13, 148, 136";
      const ice = isDark ? "56, 189, 248" : "2, 132, 199";

      // 1. Soft respiratory volume curve
      ctx.beginPath();
      for (let x = 0; x <= w; x += 4) {
        const env = Math.sin((x / w) * Math.PI);
        const y = h * 0.68 + Math.sin(x * 0.0052 + phase * 3.8) * 18 * env;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `rgba(${ice}, ${isDark ? 0.14 : 0.16})`;
      ctx.lineWidth = 1.1;
      ctx.stroke();

      // 2. Crisp PQRST cardiac trace with traveling phosphor glow
      ctx.beginPath();
      let headY = h * 0.56;
      for (let x = 0; x <= w; x += 3) {
        const env = Math.sin((x / w) * Math.PI);
        const u = x / 320 - phase * 1.15;
        const y = h * 0.56 - ecgShape(u) * 32 * env;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        if (Math.abs(x - sweepX) < 3) headY = y;
      }
      ctx.strokeStyle = `rgba(${teal}, ${isDark ? 0.22 : 0.2})`;
      ctx.lineWidth = 1.35;
      ctx.stroke();

      // Traveling phosphor head dot
      if (!prefersReduced && w > 0) {
        const glow = ctx.createRadialGradient(sweepX, headY, 0, sweepX, headY, 22);
        glow.addColorStop(0, `rgba(${teal}, ${isDark ? 0.55 : 0.45})`);
        glow.addColorStop(1, `rgba(${teal}, 0)`);
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(sweepX, headY, 22, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = `rgba(${teal}, 0.92)`;
        ctx.beginPath();
        ctx.arc(sweepX, headY, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }

      if (!prefersReduced) rafId = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", resize);
    };
  }, [theme]);

  // Subtle tactile optical tilt on the live specimen film plate
  const handleSpecimenMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = specimenRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const rx = ((e.clientY - rect.top) / rect.height - 0.5) * -4.2;
    const ry = ((e.clientX - rect.left) / rect.width - 0.5) * 4.2;
    el.style.transform = `perspective(1000px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg)`;
  };

  const handleSpecimenLeave = () => {
    const el = specimenRef.current;
    if (el) el.style.transform = "perspective(1000px) rotateX(0deg) rotateY(0deg)";
  };

  const topPositive = predictionData?.predictions?.find((p) => p.score >= threshold);
  const leadPred = topPositive || predictionData?.predictions?.[0] || null;
  const ctr = predictionData?.anatomy?.ctr ?? null;

  return (
    <div className="relative mx-auto max-w-[1360px] px-5 py-7 pb-24 md:px-8 md:py-10 lg:pb-12">
      {/* Subtle Dual-Channel ECG & Pulmonary Wave Field behind Top Hero Band */}
      <div className="pointer-events-none absolute top-0 right-0 left-0 h-[480px] overflow-hidden">
        <canvas ref={waveCanvasRef} aria-hidden="true" className="h-full w-full opacity-90" />
      </div>

      <div className="relative z-10 space-y-10">
        {/* TOP HERO SPLIT: Clinical Editorial Intro + Interactive DICOM Specimen Plate */}
        <div className="grid grid-cols-1 items-center gap-10 lg:grid-cols-12 lg:gap-12">
          {/* Left 7 Cols */}
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65, ease: [0.16, 1, 0.3, 1] }}
            className="lg:col-span-7 space-y-6"
          >
            {/* Authentic Clinical Telemetry Eyebrow */}
            <div
              className="inline-flex items-center gap-2.5 rounded-full border px-3.5 py-1.5 font-mono-tech text-[10px] tracking-[0.15em] uppercase"
              style={{
                borderColor: "var(--border-strong)",
                backgroundColor: "var(--bg-elevated)",
                color: "var(--text-secondary)",
              }}
            >
              <span className="relative flex h-2 w-2 items-center justify-center">
                <span
                  className="clinical-ripple absolute inset-0 rounded-full"
                  style={{ backgroundColor: "var(--accent-clinical)" }}
                />
                <span
                  className="relative h-1.5 w-1.5 rounded-full"
                  style={{ backgroundColor: "var(--accent-clinical)" }}
                />
              </span>
              <span>
                {health?.gpu_name
                  ? `${health.gpu_name.replace(/NVIDIA GeForce\s*/i, "")} · 4-Model Thoracic Consensus`
                  : "Multi-Backbone Thoracic Diagnostic Console"}
              </span>
            </div>

            <h1 className="font-display text-4xl leading-[1.06] font-bold sm:text-5xl xl:text-[58px]">
              Thoracic radiography, interpreted with{" "}
              <span
                className="font-serif-editorial font-normal"
                style={{ color: "var(--accent-clinical)" }}
              >
                clinical precision.
              </span>
            </h1>

            <p
              className="max-w-xl text-[15.5px] leading-relaxed"
              style={{ color: "var(--text-secondary)" }}
            >
              Drop any frontal chest X-ray or DICOM radiograph. Three calibrated GPU vision
              backbones, automated PSPNet cardiac biometry, and Google Gemini synthesize Grad-CAM++
              spatial evidence and an ACR-structured clinical report in seconds.
            </p>

            {/* Primary Actions */}
            <div className="flex flex-wrap items-center gap-3 pt-1">
              <button
                type="button"
                onClick={onUploadClick}
                className="flex cursor-pointer items-center gap-2.5 rounded-full px-6 py-3.5 text-sm font-semibold transition-transform active:scale-95"
                style={{
                  backgroundColor: "var(--accent-clinical)",
                  color: "#ffffff",
                  boxShadow: "0 10px 30px -10px var(--accent-clinical-glow)",
                }}
              >
                <Upload className="h-4 w-4" />
                <span>Upload Chest X-Ray</span>
              </button>

              <button
                type="button"
                onClick={onOpenStudio}
                className="group flex cursor-pointer items-center gap-2 rounded-full border px-6 py-3.5 text-sm font-semibold transition-all hover:opacity-90"
                style={{
                  borderColor: "var(--border-strong)",
                  backgroundColor: "var(--bg-elevated)",
                  color: "var(--text-primary)",
                }}
              >
                <span>Open Scan Studio</span>
                <ArrowRight
                  className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-0.5"
                  style={{ color: "var(--accent-clinical)" }}
                />
              </button>

              <button
                type="button"
                onClick={onOpenConsensus}
                className="cursor-pointer rounded-full px-4 py-3 text-xs font-medium transition-colors hover:underline"
                style={{ color: "var(--text-secondary)" }}
              >
                Inspect 4-Model Biometry →
              </button>
            </div>

            {/* Clean 3-Stage Diagnostic Flow */}
            <div
              className="grid grid-cols-1 gap-4 border-t pt-6 sm:grid-cols-3"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              {[
                {
                  num: "01",
                  title: "Ingest & Calibrate",
                  desc: "DICOM, PNG, JPEG or verified NIH benchmark radiographs.",
                },
                {
                  num: "02",
                  title: "Consensus & CTR",
                  desc: "DenseNet-121, ResNet-50 512px, PSPNet & Gemini Decision 1.",
                },
                {
                  num: "03",
                  title: "Localize & Dictate",
                  desc: "Grad-CAM++ zones, ACR report & grounded Gemini Copilot.",
                },
              ].map((step) => (
                <div key={step.num} className="space-y-1">
                  <div
                    className="font-mono-tech text-[10px] font-medium tracking-[0.18em]"
                    style={{ color: "var(--accent-clinical)" }}
                  >
                    STAGE {step.num}
                  </div>
                  <div className="font-display text-sm font-semibold">{step.title}</div>
                  <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                    {step.desc}
                  </p>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Right 5 Cols: Interactive Optical DICOM Specimen Plate */}
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.72, delay: 0.08, ease: [0.16, 1, 0.3, 1] }}
            className="lg:col-span-5"
          >
            <div
              ref={specimenRef}
              onMouseMove={handleSpecimenMove}
              onMouseLeave={handleSpecimenLeave}
              onClick={onOpenStudio}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && onOpenStudio()}
              style={{ transition: "transform 0.35s cubic-bezier(0.16, 1, 0.3, 1)" }}
              className="atelier-card-strong group cursor-pointer overflow-hidden rounded-[28px] p-3.5"
            >
              <div
                className="relative aspect-square w-full overflow-hidden rounded-[20px]"
                style={{ backgroundColor: "var(--bg-film)" }}
              >
                {imageUrl ? (
                  <img
                    src={imageUrl}
                    alt="Active chest radiograph"
                    className="h-full w-full object-cover transition-transform duration-700 group-hover:scale-[1.03]"
                  />
                ) : (
                  <div className="flex h-full items-center justify-center font-mono-tech text-xs text-slate-500">
                    LOADING BENCHMARK RADIOGRAPH...
                  </div>
                )}

                {/* Grad-CAM++ RGBA Overlay Preview */}
                {predictionData?.heatmaps?.[0]?.raw_heatmap_base64 && (
                  <img
                    src={`data:image/png;base64,${predictionData.heatmaps[0].raw_heatmap_base64}`}
                    alt="Grad-CAM++ Preview"
                    className="pointer-events-none absolute inset-0 h-full w-full object-cover opacity-58 mix-blend-screen"
                  />
                )}

                {/* Whisper-Subtle Optical Scanline Sweep */}
                <div
                  aria-hidden="true"
                  className="idle-scanline pointer-events-none absolute right-0 left-0 h-16"
                  style={{
                    background:
                      "linear-gradient(180deg, transparent, rgba(20, 184, 166, 0.14), transparent)",
                  }}
                />

                {/* Authentic DICOM Corner Calibration L-Fiducials */}
                <div className="pointer-events-none absolute inset-3">
                  <span className="absolute top-0 left-0 h-3 w-3 border-t border-l border-teal-400/50" />
                  <span className="absolute top-0 right-0 h-3 w-3 border-t border-r border-teal-400/50" />
                  <span className="absolute bottom-0 left-0 h-3 w-3 border-b border-l border-teal-400/50" />
                  <span className="absolute right-0 bottom-0 h-3 w-3 border-r border-b border-teal-400/50" />
                </div>

                {/* Etched Right-Edge Millimeter Optical Scale Ticks */}
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute top-16 right-2.5 bottom-24 flex flex-col justify-between opacity-45"
                >
                  {Array.from({ length: 11 }).map((_, idx) => (
                    <span
                      key={idx}
                      className="block h-[1px] bg-white"
                      style={{ width: idx % 5 === 0 ? 9 : 4, marginLeft: "auto" }}
                    />
                  ))}
                </div>

                {/* Radiographic R / L Orientation Markers */}
                <span className="pointer-events-none absolute top-1/2 left-3.5 -translate-y-1/2 font-mono-tech text-[10px] font-semibold text-white/45">
                  R
                </span>
                <span className="pointer-events-none absolute top-1/2 right-5 -translate-y-1/2 font-mono-tech text-[10px] font-semibold text-white/45">
                  L
                </span>

                {isAnalyzing && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/55 backdrop-blur-[2px]">
                    <div className="flex items-center gap-2.5 rounded-full border border-teal-400/35 bg-black/85 px-4 py-2 font-mono-tech text-[11px] text-white">
                      <span className="h-2 w-2 rounded-full bg-teal-400 clinical-led" />
                      <span>Analyzing on RTX CUDA...</span>
                    </div>
                  </div>
                )}

                {/* Top Floating Specimen Telemetry */}
                <div className="absolute top-4 right-4 left-4 flex items-center justify-between">
                  <span className="flex items-center gap-1.5 rounded-full border border-white/15 bg-black/72 px-3 py-1 font-mono-tech text-[9.5px] tracking-[0.14em] text-slate-200 uppercase backdrop-blur-md">
                    <span className="h-1.5 w-1.5 rounded-full bg-teal-400" />
                    ACTIVE SPECIMEN
                  </span>
                  {ctr != null && (
                    <span className="flex items-center gap-1.5 rounded-full border border-white/15 bg-black/72 px-3 py-1 font-mono-tech text-[10px] text-white backdrop-blur-md">
                      <HeartPulse className="h-3 w-3 text-teal-400" />
                      CTR{" "}
                      <AnimatedNumber value={ctr} decimals={3} className="font-semibold" />
                    </span>
                  )}
                </div>

                {/* Bottom Verdict Overlay Plate */}
                <div className="absolute right-4 bottom-4 left-4 flex items-end justify-between rounded-2xl border border-white/12 bg-black/78 px-4 py-3 backdrop-blur-md transition-colors group-hover:border-teal-400/40">
                  <div className="min-w-0">
                    <div className="font-mono-tech text-[9px] tracking-[0.18em] text-slate-400 uppercase">
                      PRIMARY DIAGNOSTIC VERDICT
                    </div>
                    <div className="mt-0.5 truncate font-display text-base font-semibold text-white">
                      {leadPred
                        ? topPositive
                          ? formatPathology(leadPred.name)
                          : "Unremarkable / Clear"
                        : "Ready for Analysis"}
                    </div>
                  </div>
                  <div className="shrink-0 pl-3 text-right">
                    <div
                      className="font-mono-tech text-xl font-semibold"
                      style={{ color: topPositive ? "#f43f5e" : "#14b8a6" }}
                    >
                      {leadPred ? (
                        <AnimatedNumber value={leadPred.score * 100} decimals={1} suffix="%" />
                      ) : (
                        "—"
                      )}
                    </div>
                    <div className="flex items-center justify-end gap-1 font-mono-tech text-[9.5px] text-slate-400">
                      <Sparkles className="h-2.5 w-2.5 text-teal-400" />
                      <span>
                        {predictionData?.decision_summary?.active
                          ? "Decision 1 Fused"
                          : "3-Backbone Ensemble"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* VERIFIED CLINICAL RADIOGRAPH GALLERY (No duplicate Cardiomegaly/Cardiomegaly!) */}
        {samples.length > 0 && (
          <div className="space-y-3.5">
            <div className="flex items-center justify-between">
              <span
                className="font-mono-tech text-[10px] font-medium tracking-[0.2em] uppercase"
                style={{ color: "var(--text-muted)" }}
              >
                SELECT A VERIFIED CLINICAL RADIOGRAPH TO INSPECT
              </span>
              <span
                className="font-mono-tech text-[10px]"
                style={{ color: "var(--text-muted)" }}
              >
                {samples.length} BENCHMARK STUDIES · CLICK TO OPEN STUDIO
              </span>
            </div>

            <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 xl:grid-cols-5">
              {samples.map((s, idx) => {
                const isActive = activeStudyId === s.id;
                const cleanSub = formatSampleSubtitle(s.name, s.subtitle, s.primary_findings);

                return (
                  <motion.div
                    key={s.id}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{
                      duration: 0.45,
                      delay: 0.08 + idx * 0.045,
                      ease: [0.16, 1, 0.3, 1],
                    }}
                  >
                    <SpotlightCard
                      onClick={() => !isAnalyzing && onSelectSample(s, true)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => e.key === "Enter" && !isAnalyzing && onSelectSample(s, true)}
                      className={`group flex cursor-pointer items-center gap-3 rounded-2xl p-3 text-left transition-all hover:-translate-y-0.5 ${
                        isAnalyzing ? "pointer-events-none opacity-60" : ""
                      }`}
                      style={{
                        borderColor: isActive ? "var(--accent-clinical)" : "var(--border-subtle)",
                      }}
                    >
                      <div className="relative h-12 w-12 shrink-0 overflow-hidden rounded-xl border border-white/10 bg-black">
                        <img
                          src={api.getSampleImageUrl(s.filename)}
                          alt={s.name}
                          loading="lazy"
                          className="h-full w-full object-cover opacity-90 transition-transform duration-500 group-hover:scale-110"
                        />
                        {isActive && (
                          <span
                            className="absolute right-1 bottom-1 h-2 w-2 rounded-full ring-2 ring-black"
                            style={{ backgroundColor: "var(--accent-clinical)" }}
                          />
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-[13px] font-semibold">
                          {cleanSampleTitle(s.name)}
                        </div>
                        <div
                          className="mt-0.5 truncate font-mono-tech text-[10px]"
                          style={{
                            color: isActive ? "var(--accent-clinical)" : "var(--text-muted)",
                          }}
                        >
                          {cleanSub}
                        </div>
                      </div>
                    </SpotlightCard>
                  </motion.div>
                );
              })}
            </div>
          </div>
        )}

        {/* CALIBRATED INSTRUMENT CAPABILITIES STRIP (Bespoke Medical Replacement for KŌDA Marquee) */}
        <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 xl:grid-cols-4">
          {ARCHITECTURE_PILLARS.map((pillar, idx) => (
            <SpotlightCard
              key={pillar.code}
              className="rounded-2xl p-4"
              style={{
                borderColor: "var(--border-subtle)",
              }}
            >
              <div className="flex items-center justify-between">
                <span
                  className="font-mono-tech text-[9.5px] font-medium tracking-[0.18em] uppercase"
                  style={{ color: "var(--accent-clinical)" }}
                >
                  {pillar.code}
                </span>
                <span
                  className="font-mono-tech text-[9.5px]"
                  style={{ color: "var(--text-muted)" }}
                >
                  0{idx + 1}
                </span>
              </div>
              <div className="mt-2 font-display text-lg font-bold">{pillar.metric}</div>
              <p className="mt-1 text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                {pillar.detail}
              </p>
            </SpotlightCard>
          ))}
        </div>

        {/* Quiet Clinical Research Attestation Bar */}
        <div
          className="flex flex-wrap items-center justify-between gap-3 border-t pt-5 text-[11px]"
          style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
        >
          <span className="flex items-center gap-2">
            <ShieldCheck className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
            <span>
              Clinical decision-support &amp; research instrument — not a standalone FDA-cleared
              medical device.
            </span>
          </span>
          <span className="font-mono-tech text-[10px]">
            Press <kbd className="rounded border px-1.5 py-0.5" style={{ borderColor: "var(--border-subtle)" }}>1</kbd>–
            <kbd className="rounded border px-1.5 py-0.5" style={{ borderColor: "var(--border-subtle)" }}>6</kbd> to switch pages ·{" "}
            <kbd className="rounded border px-1.5 py-0.5" style={{ borderColor: "var(--border-subtle)" }}>Ctrl+K</kbd> command search
          </span>
        </div>
      </div>
    </div>
  );
};
