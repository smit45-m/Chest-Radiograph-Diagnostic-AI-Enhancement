import React, { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { ArrowRight, HeartPulse, ShieldCheck, Sparkles, Upload } from "lucide-react";
import {
  formatPathology,
  formatSampleSubtitle,
  shortSampleTitle,
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
}) => {
  const waveCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const specimenRef = useRef<HTMLDivElement | null>(null);

  // Subtle living PQRST cardiac pulse behind top hero (automatically pauses when tab hidden)
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
      if (document.hidden) {
        rafId = requestAnimationFrame(draw);
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      if (!prefersReduced) {
        phase += 0.0026;
        sweepX = (sweepX + 1.35) % Math.max(w, 1);
      }

      const isDark = theme === "dark";
      const teal = isDark ? "20, 184, 166" : "13, 148, 136";

      ctx.beginPath();
      let headY = h * 0.56;
      for (let x = 0; x <= w; x += 3) {
        const env = Math.sin((x / w) * Math.PI);
        const u = x / 340 - phase * 1.1;
        const y = h * 0.56 - ecgShape(u) * 26 * env;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        if (Math.abs(x - sweepX) < 3) headY = y;
      }
      ctx.strokeStyle = `rgba(${teal}, ${isDark ? 0.14 : 0.13})`;
      ctx.lineWidth = 1.2;
      ctx.stroke();

      if (!prefersReduced && w > 0) {
        const glow = ctx.createRadialGradient(sweepX, headY, 0, sweepX, headY, 18);
        glow.addColorStop(0, `rgba(${teal}, ${isDark ? 0.42 : 0.32})`);
        glow.addColorStop(1, `rgba(${teal}, 0)`);
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(sweepX, headY, 18, 0, Math.PI * 2);
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

  const handleSpecimenMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = specimenRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const rx = ((e.clientY - rect.top) / rect.height - 0.5) * -3.2;
    const ry = ((e.clientX - rect.left) / rect.width - 0.5) * 3.2;
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
    <div className="relative mx-auto max-w-[1280px] px-5 py-7 pb-24 md:px-8 md:py-10 lg:pb-10">
      {/* Subtle Living ECG Backdrop */}
      <div className="pointer-events-none absolute top-0 right-0 left-0 h-[420px] overflow-hidden">
        <canvas ref={waveCanvasRef} aria-hidden="true" className="h-full w-full opacity-85" />
      </div>

      <div className="relative z-10 space-y-10">
        {/* CALM 2-COLUMN HERO: Immediate Action + Interactive Live Scan Preview */}
        <div className="grid grid-cols-1 items-center gap-10 lg:grid-cols-12 lg:gap-12">
          {/* Left 7 Cols */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
            className="lg:col-span-7 space-y-6"
          >
            <div
              className="inline-flex items-center gap-2 rounded-full border px-3 py-1 font-mono-tech text-[10.5px]"
              style={{
                borderColor: "var(--border-subtle)",
                backgroundColor: "var(--bg-elevated)",
                color: "var(--text-secondary)",
              }}
            >
              <span
                className="h-1.5 w-1.5 rounded-full clinical-led"
                style={{ backgroundColor: "var(--accent-clinical)" }}
              />
              <span>
                {health?.gpu_name
                  ? `${health.gpu_name.replace(/NVIDIA GeForce\s*/i, "")} · 14-Pathology AI`
                  : "5-Model Thoracic Diagnostic AI"}
              </span>
            </div>

            <h1 className="font-display text-4xl leading-[1.08] font-bold sm:text-5xl xl:text-[54px]">
              Chest X-ray analysis,{" "}
              <span
                className="font-serif-editorial font-normal"
                style={{ color: "var(--accent-clinical)" }}
              >
                clear in seconds.
              </span>
            </h1>

            <p
              className="max-w-lg text-[15px] leading-relaxed"
              style={{ color: "var(--text-secondary)" }}
            >
              Upload any frontal chest radiograph or pick a clinical sample below to view Grad-CAM++
              heatmaps, automated heart sizing, and a structured ACR report.
            </p>

            {/* Interactive Dropzone + Open Studio */}
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center pt-1">
              <button
                type="button"
                onClick={onUploadClick}
                className="flex cursor-pointer items-center justify-center gap-2.5 rounded-full px-6 py-3.5 text-sm font-semibold transition-all hover:brightness-110 active:scale-[0.98]"
                style={{
                  backgroundColor: "var(--accent-clinical)",
                  color: "#ffffff",
                  boxShadow: "0 10px 28px -10px var(--accent-clinical-glow)",
                }}
              >
                <Upload className="h-4 w-4" />
                <span>Upload Chest X-Ray</span>
              </button>

              <button
                type="button"
                onClick={onOpenStudio}
                className="group flex cursor-pointer items-center justify-center gap-2 rounded-full border px-6 py-3.5 text-sm font-semibold transition-all hover:border-teal-500/40"
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
            </div>

            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              Supports DICOM, PNG &amp; JPEG · Drag &amp; drop anywhere or press{" "}
              <kbd
                className="rounded border px-1.5 py-0.5 font-mono-tech text-[10px]"
                style={{ borderColor: "var(--border-subtle)" }}
              >
                Ctrl+V
              </kbd>{" "}
              to paste
            </p>
          </motion.div>

          {/* Right 5 Cols: Clean Interactive Specimen Preview */}
          <motion.div
            initial={{ opacity: 0, y: 14, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.06, ease: [0.22, 1, 0.36, 1] }}
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
              style={{ transition: "transform 0.3s cubic-bezier(0.22, 1, 0.36, 1)" }}
              className="atelier-card-strong group cursor-pointer overflow-hidden rounded-[26px] p-3"
            >
              <div
                className="relative aspect-square w-full overflow-hidden rounded-[18px]"
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
                    Loading sample X-ray...
                  </div>
                )}

                {predictionData?.heatmaps?.[0]?.raw_heatmap_base64 && (
                  <img
                    src={`data:image/png;base64,${predictionData.heatmaps[0].raw_heatmap_base64}`}
                    alt="Grad-CAM++ localization preview"
                    className="pointer-events-none absolute inset-0 h-full w-full object-cover opacity-55 mix-blend-screen"
                  />
                )}

                {/* Whisper-Subtle Optical Scanline */}
                <div
                  aria-hidden="true"
                  className="idle-scanline pointer-events-none absolute right-0 left-0 h-14"
                  style={{
                    background:
                      "linear-gradient(180deg, transparent, rgba(20, 184, 166, 0.12), transparent)",
                  }}
                />

                {isAnalyzing && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/55 backdrop-blur-[2px]">
                    <div className="flex items-center gap-2 rounded-full border border-teal-400/35 bg-black/85 px-4 py-2 font-mono-tech text-[11px] text-white">
                      <span className="h-2 w-2 rounded-full bg-teal-400 clinical-led" />
                      <span>Analyzing scan...</span>
                    </div>
                  </div>
                )}

                {/* Minimal Top Bar */}
                <div className="absolute top-3.5 right-3.5 left-3.5 flex items-center justify-between">
                  <span className="rounded-full border border-white/12 bg-black/70 px-3 py-1 font-mono-tech text-[10px] text-slate-200 backdrop-blur-md">
                    Active Study
                  </span>
                  {ctr != null && (
                    <span className="flex items-center gap-1.5 rounded-full border border-white/12 bg-black/70 px-3 py-1 font-mono-tech text-[10px] text-white backdrop-blur-md">
                      <HeartPulse className="h-3 w-3 text-teal-400" />
                      CTR <AnimatedNumber value={ctr} decimals={3} className="font-semibold" />
                    </span>
                  )}
                </div>

                {/* Clean Verdict Footer */}
                <div className="absolute right-3.5 bottom-3.5 left-3.5 flex items-center justify-between rounded-2xl border border-white/12 bg-black/78 px-4 py-3 backdrop-blur-md transition-colors group-hover:border-teal-400/40">
                  <div className="min-w-0">
                    <div className="text-[11px] text-slate-400">Primary Finding</div>
                    <div className="truncate font-display text-base font-semibold text-white">
                      {leadPred
                        ? topPositive
                          ? formatPathology(leadPred.name)
                          : "No Acute Findings"
                        : "Ready"}
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
                    <div className="flex items-center justify-end gap-1 text-[10px] text-teal-300 opacity-85 transition-opacity group-hover:opacity-100">
                      <Sparkles className="h-2.5 w-2.5" />
                      <span>Inspect in Studio →</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* SAMPLE RADIOGRAPHS GALLERY — Short, Untruncated, One-Click Launch */}
        {samples.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="font-display text-sm font-semibold">
                Try a verified sample X-ray
              </h2>
              <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                Click any scan to open in Studio
              </span>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
              {samples.map((s, idx) => {
                const isActive = activeStudyId === s.id;
                const cleanSub = formatSampleSubtitle(s.name, s.subtitle, s.primary_findings);

                return (
                  <motion.div
                    key={s.id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{
                      duration: 0.35,
                      delay: 0.05 + idx * 0.035,
                      ease: [0.22, 1, 0.36, 1],
                    }}
                  >
                    <SpotlightCard
                      onClick={() => !isAnalyzing && onSelectSample(s, true)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => e.key === "Enter" && !isAnalyzing && onSelectSample(s, true)}
                      className={`group flex cursor-pointer items-center gap-3 rounded-2xl p-3 text-left transition-all hover:-translate-y-0.5 hover:border-teal-500/40 ${
                        isAnalyzing ? "pointer-events-none opacity-60" : ""
                      }`}
                      style={{
                        borderColor: isActive ? "var(--accent-clinical)" : "var(--border-subtle)",
                      }}
                    >
                      <div className="relative h-12 w-12 shrink-0 overflow-hidden rounded-xl border border-white/10 bg-black">
                        <img
                          src={api.getSampleImageUrl(s.filename)}
                          alt={shortSampleTitle(s.name)}
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
                        <div className="truncate text-[13.5px] font-semibold">
                          {shortSampleTitle(s.name)}
                        </div>
                        <div
                          className="mt-0.5 truncate text-[11px]"
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

        {/* Quiet Single-Line Clinical Notice */}
        <div
          className="flex flex-wrap items-center justify-between gap-3 border-t pt-4 text-xs"
          style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
        >
          <span className="flex items-center gap-2">
            <ShieldCheck className="h-3.5 w-3.5 shrink-0" style={{ color: "var(--accent-clinical)" }} />
            <span>Clinical decision-support research tool — requires physician verification.</span>
          </span>
          <span className="hidden font-mono-tech text-[10.5px] sm:inline">
            Shortcuts: Keys 1–6 navigate · Ctrl+K quick search
          </span>
        </div>
      </div>
    </div>
  );
};
