import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Columns2,
  Eye,
  HeartPulse,
  Image as ImageIcon,
  Layers,
  MoveHorizontal,
  RotateCcw,
  Ruler,
  Search,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload,
  UserRound,
} from "lucide-react";
import {
  COLORMAP_OPTIONS,
  PATHOLOGY_CLINICAL_NOTE,
  WINDOW_PRESETS,
  formatPathology,
  formatTriageLabel,
  shortSampleTitle,
  type PageId,
} from "../data/clinicalCatalog";
import type {
  AnalysisStage,
  CaliperMeasurement,
  ColormapName,
  HeatmapData,
  PatientProfile,
  PredictionResponse,
  SampleCase,
  WindowPreset,
} from "../types/clinical";
import { AnimatedNumber } from "./AnimatedNumber";
import { SpotlightCard } from "./SpotlightCard";

interface DiagnosticStudioProps {
  samples: SampleCase[];
  activeStudyId: string | null;
  activeStudyName: string;
  imageUrl: string | null;
  predictionData: PredictionResponse | null;
  activePathology: string | null;
  activeHeatmap: HeatmapData | null;
  heatmapLoading: boolean;
  threshold: number;
  colormap: ColormapName;
  patient: PatientProfile;
  isAnalyzing: boolean;
  analysisStage?: AnalysisStage;
  holdingSpace: boolean;
  onSelectSample: (sample: SampleCase) => void;
  onFileUpload: (file: File) => void;
  onUploadClick: () => void;
  onSelectPathology: (pathology: string) => void;
  onThresholdChange: (value: number) => void;
  onColormapChange: (cmap: ColormapName) => void;
  onOpenPatientDrawer: () => void;
  onNavigatePage: (page: PageId) => void;
}

type ViewMode = "heatmap" | "anatomy" | "compare" | "original";

const CLINICAL_MODES = [
  { id: "screening", label: "Screening", value: 0.35, hint: "High Recall" },
  { id: "balanced", label: "Balanced", value: 0.5, hint: "Optimal F1" },
  { id: "precision", label: "High-Spec", value: 0.62, hint: "Rule-In" },
] as const;

function loadImg(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

export const DiagnosticStudio: React.FC<DiagnosticStudioProps> = ({
  samples,
  activeStudyId,
  imageUrl,
  predictionData,
  activePathology,
  activeHeatmap,
  heatmapLoading,
  threshold,
  colormap,
  patient,
  isAnalyzing,
  analysisStage = "idle",
  holdingSpace,
  onSelectSample,
  onFileUpload,
  onUploadClick,
  onSelectPathology,
  onThresholdChange,
  onColormapChange,
  onOpenPatientDrawer,
  onNavigatePage,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [baseImg, setBaseImg] = useState<HTMLImageElement | null>(null);
  const [heatImg, setHeatImg] = useState<HTMLImageElement | null>(null);
  const [anatImg, setAnatImg] = useState<HTMLImageElement | null>(null);

  // Clean 4-Mode Segmented Viewer State + Collapsible Precision Tools
  const [viewMode, setViewMode] = useState<ViewMode>("heatmap");
  const [toolsOpen, setToolsOpen] = useState(false);
  const [cutoffOpen, setCutoffOpen] = useState(false);
  const [splitPos, setSplitPos] = useState(0.5);
  const [loupeActive, setLoupeActive] = useState(false);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.68);
  const [windowPreset, setWindowPreset] = useState<WindowPreset>("standard");

  // Zoom / Pan / Caliper / Pointer State
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [caliperMode, setCaliperMode] = useState(false);
  const [calipers, setCalipers] = useState<CaliperMeasurement[]>([]);
  const [draftCaliper, setDraftCaliper] = useState<{
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  } | null>(null);
  const [pointer, setPointer] = useState<{
    x: number;
    y: number;
    nx: number;
    ny: number;
    inside: boolean;
  }>({ x: 0, y: 0, nx: 0.5, ny: 0.5, inside: false });
  const [isDraggingPan, setIsDraggingPan] = useState(false);
  const [isDraggingSplit, setIsDraggingSplit] = useState(false);
  const [dragOrigin, setDragOrigin] = useState({ x: 0, y: 0 });
  const [isDragOver, setIsDragOver] = useState(false);
  const [showAllPathologies, setShowAllPathologies] = useState(false);

  const showHeatmap = viewMode === "heatmap" || viewMode === "compare";
  const showAnatomy = viewMode === "anatomy";
  const splitCompare = viewMode === "compare";

  useEffect(() => {
    if (!imageUrl) {
      setBaseImg(null);
      return;
    }
    let cancelled = false;
    loadImg(imageUrl)
      .then((img) => {
        if (!cancelled) {
          setBaseImg(img);
          setZoom(1);
          setPan({ x: 0, y: 0 });
          setCalipers([]);
        }
      })
      .catch(() => !cancelled && setBaseImg(null));
    return () => {
      cancelled = true;
    };
  }, [imageUrl]);

  useEffect(() => {
    const b64 = activeHeatmap?.raw_heatmap_base64 || activeHeatmap?.image_base64;
    if (!b64) {
      setHeatImg(null);
      return;
    }
    let cancelled = false;
    loadImg(`data:image/png;base64,${b64}`)
      .then((img) => !cancelled && setHeatImg(img))
      .catch(() => !cancelled && setHeatImg(null));
    return () => {
      cancelled = true;
    };
  }, [activeHeatmap]);

  useEffect(() => {
    const b64 = predictionData?.anatomy?.overlay_base64;
    if (!b64) {
      setAnatImg(null);
      return;
    }
    let cancelled = false;
    loadImg(`data:image/png;base64,${b64}`)
      .then((img) => !cancelled && setAnatImg(img))
      .catch(() => !cancelled && setAnatImg(null));
    return () => {
      cancelled = true;
    };
  }, [predictionData?.anatomy?.overlay_base64]);

  const currentWindow = useMemo(
    () => WINDOW_PRESETS.find((p) => p.id === windowPreset) || WINDOW_PRESETS[0],
    [windowPreset]
  );

  const getDrawRect = useCallback((w: number, h: number, img: HTMLImageElement) => {
    const side = Math.min(w, h) * 0.94;
    const dx = (w - side) / 2;
    const dy = (h - side) / 2;
    const srcSide = Math.min(img.naturalWidth || img.width, img.naturalHeight || img.height);
    const sx = ((img.naturalWidth || img.width) - srcSide) / 2;
    const sy = ((img.naturalHeight || img.height) - srcSide) / 2;
    return { sx, sy, srcSide, dx, dy, side };
  }, []);

  // Radiograph Canvas Compositor
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(320, Math.floor(rect.width));
    const h = Math.max(320, Math.floor(rect.height));

    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#020406";
    ctx.fillRect(0, 0, w, h);

    if (!baseImg) return;

    const { sx, sy, srcSide, dx, dy, side } = getDrawRect(w, h, baseImg);
    const filterStr = `brightness(${currentWindow.brightness}%) contrast(${currentWindow.contrast}%)${
      currentWindow.invert ? " invert(100%)" : ""
    }`;

    const drawScene = (withOverlays: boolean) => {
      ctx.save();
      ctx.translate(w / 2 + pan.x, h / 2 + pan.y);
      ctx.scale(zoom, zoom);
      ctx.translate(-w / 2, -h / 2);

      ctx.filter = filterStr;
      ctx.drawImage(baseImg, sx, sy, srcSide, srcSide, dx, dy, side, side);
      ctx.filter = "none";

      if (withOverlays && !holdingSpace && viewMode !== "original") {
        if (showAnatomy && anatImg) {
          ctx.save();
          ctx.globalAlpha = 0.86;
          ctx.drawImage(anatImg, dx, dy, side, side);
          ctx.restore();
        }

        if (showHeatmap && heatImg) {
          ctx.save();
          ctx.globalAlpha = activeHeatmap?.raw_heatmap_base64
            ? heatmapOpacity
            : heatmapOpacity * 0.72;
          ctx.drawImage(heatImg, dx, dy, side, side);
          ctx.restore();

          if (activeHeatmap?.bbox && activeHeatmap.bbox.length === 4) {
            const [bx1, by1, bx2, by2] = activeHeatmap.bbox;
            const rx = dx + bx1 * side;
            const ry = dy + by1 * side;
            const rw = (bx2 - bx1) * side;
            const rh = (by2 - by1) * side;
            ctx.save();
            ctx.strokeStyle = "rgba(20, 184, 166, 0.75)";
            ctx.lineWidth = 1.2 / zoom;
            ctx.setLineDash([5 / zoom, 4 / zoom]);
            ctx.strokeRect(rx, ry, rw, rh);
            ctx.restore();
          }

          if (activeHeatmap?.peak_x != null && activeHeatmap?.peak_y != null) {
            const px = dx + activeHeatmap.peak_x * side;
            const py = dy + activeHeatmap.peak_y * side;
            const r = 8 / zoom;
            ctx.save();
            ctx.strokeStyle = "#f1f5f8";
            ctx.lineWidth = 1.3 / zoom;
            ctx.beginPath();
            ctx.arc(px, py, r, 0, Math.PI * 2);
            ctx.moveTo(px - r * 1.7, py);
            ctx.lineTo(px - r * 0.65, py);
            ctx.moveTo(px + r * 0.65, py);
            ctx.lineTo(px + r * 1.7, py);
            ctx.moveTo(px, py - r * 1.7);
            ctx.lineTo(px, py - r * 0.65);
            ctx.moveTo(px, py + r * 0.65);
            ctx.lineTo(px, py + r * 1.7);
            ctx.stroke();
            ctx.restore();
          }
        }
      }

      // Caliper Measurements
      const allCalipers = draftCaliper
        ? [
            ...calipers,
            {
              id: "draft",
              ...draftCaliper,
              mm:
                Math.hypot(draftCaliper.x2 - draftCaliper.x1, draftCaliper.y2 - draftCaliper.y1) *
                350,
            },
          ]
        : calipers;

      allCalipers.forEach((c) => {
        const cx1 = dx + c.x1 * side;
        const cy1 = dy + c.y1 * side;
        const cx2 = dx + c.x2 * side;
        const cy2 = dy + c.y2 * side;
        ctx.save();
        ctx.strokeStyle = "#14b8a6";
        ctx.fillStyle = "#14b8a6";
        ctx.lineWidth = 1.6 / zoom;
        ctx.beginPath();
        ctx.moveTo(cx1, cy1);
        ctx.lineTo(cx2, cy2);
        ctx.stroke();

        [
          [cx1, cy1],
          [cx2, cy2],
        ].forEach(([ex, ey]) => {
          ctx.beginPath();
          ctx.arc(ex, ey, 3 / zoom, 0, Math.PI * 2);
          ctx.fill();
        });

        const mx = (cx1 + cx2) / 2;
        const my = (cy1 + cy2) / 2 - 11 / zoom;
        const label = `${c.mm.toFixed(1)} mm`;
        ctx.font = `${11 / zoom}px "IBM Plex Mono", monospace`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(6, 9, 12, 0.88)";
        ctx.fillRect(mx - tw / 2 - 5 / zoom, my - 10 / zoom, tw + 10 / zoom, 15 / zoom);
        ctx.fillStyle = "#f1f5f8";
        ctx.textAlign = "center";
        ctx.fillText(label, mx, my + 1 / zoom);
        ctx.restore();
      });

      ctx.restore();
    };

    if (splitCompare && !holdingSpace) {
      const splitX = w * splitPos;
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, splitX, h);
      ctx.clip();
      drawScene(false);
      ctx.restore();

      ctx.save();
      ctx.beginPath();
      ctx.rect(splitX, 0, w - splitX, h);
      ctx.clip();
      drawScene(true);
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = "rgba(20, 184, 166, 0.85)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(splitX, 0);
      ctx.lineTo(splitX, h);
      ctx.stroke();
      ctx.restore();
    } else {
      drawScene(true);
    }

    // Subtle Left-Edge Scale Ruler
    ctx.save();
    const rulerTop = h * 0.22;
    const rulerBot = h * 0.78;
    const ticks = 16;
    ctx.strokeStyle = "rgba(241, 245, 248, 0.22)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(14, rulerTop);
    ctx.lineTo(14, rulerBot);
    for (let i = 0; i <= ticks; i++) {
      const ty = rulerTop + ((rulerBot - rulerTop) * i) / ticks;
      const len = i % 4 === 0 ? 7 : 3.5;
      ctx.moveTo(14, ty);
      ctx.lineTo(14 + len, ty);
    }
    ctx.stroke();
    ctx.restore();

    // 2.5× Optical Magnifier Loupe
    if (loupeActive && pointer.inside && !caliperMode && !isDraggingSplit) {
      const radius = 82;
      const mag = 2.5;
      const lx = Math.min(Math.max(pointer.x, radius + 14), w - radius - 14);
      const ly = Math.min(Math.max(pointer.y, radius + 14), h - radius - 14);

      ctx.save();
      ctx.beginPath();
      ctx.arc(lx, ly, radius, 0, Math.PI * 2);
      ctx.clip();
      ctx.fillStyle = "#020406";
      ctx.fillRect(lx - radius, ly - radius, radius * 2, radius * 2);
      ctx.translate(lx - pointer.x * mag, ly - pointer.y * mag);
      ctx.scale(mag, mag);
      drawScene(true);
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = "rgba(20, 184, 166, 0.92)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(lx, ly, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }, [
    baseImg,
    heatImg,
    anatImg,
    viewMode,
    showHeatmap,
    showAnatomy,
    heatmapOpacity,
    currentWindow,
    zoom,
    pan,
    splitCompare,
    splitPos,
    loupeActive,
    pointer,
    holdingSpace,
    activeHeatmap,
    calipers,
    draftCaliper,
    caliperMode,
    isDraggingSplit,
    getDrawRect,
  ]);

  const toNormCoords = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas || !baseImg) return { nx: 0.5, ny: 0.5, x: 0, y: 0, w: 1, h: 1 };
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const w = rect.width;
    const h = rect.height;
    const { dx, dy, side } = getDrawRect(w, h, baseImg);
    const worldX = (x - w / 2 - pan.x) / zoom + w / 2;
    const worldY = (y - h / 2 - pan.y) / zoom + h / 2;
    const nx = Math.min(1, Math.max(0, (worldX - dx) / side));
    const ny = Math.min(1, Math.max(0, (worldY - dy) / side));
    return { nx, ny, x, y, w, h };
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const { nx, ny, x, y, w } = toNormCoords(e.clientX, e.clientY);
    if (splitCompare && Math.abs(x - w * splitPos) < 28) {
      setIsDraggingSplit(true);
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }
    if (caliperMode) {
      setDraftCaliper({ x1: nx, y1: ny, x2: nx, y2: ny });
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }
    if (zoom > 1) {
      setIsDraggingPan(true);
      setDragOrigin({ x: x - pan.x, y: y - pan.y });
      e.currentTarget.setPointerCapture(e.pointerId);
    }
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const { nx, ny, x, y, w } = toNormCoords(e.clientX, e.clientY);
    setPointer({ x, y, nx, ny, inside: true });
    if (isDraggingSplit) {
      setSplitPos(Math.min(0.92, Math.max(0.08, x / w)));
      return;
    }
    if (draftCaliper) {
      setDraftCaliper((prev) => (prev ? { ...prev, x2: nx, y2: ny } : null));
      return;
    }
    if (isDraggingPan) {
      setPan({ x: x - dragOrigin.x, y: y - dragOrigin.y });
    }
  };

  const handlePointerUp = () => {
    if (draftCaliper) {
      const dist = Math.hypot(draftCaliper.x2 - draftCaliper.x1, draftCaliper.y2 - draftCaliper.y1);
      if (dist > 0.015) {
        setCalipers((prev) => [
          ...prev,
          { id: String(Date.now()), ...draftCaliper, mm: dist * 350 },
        ]);
      }
      setDraftCaliper(null);
    }
    setIsDraggingPan(false);
    setIsDraggingSplit(false);
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    if (e.ctrlKey || zoom > 1) {
      e.preventDefault();
      const next = Math.min(3.5, Math.max(1, zoom - e.deltaY * 0.002));
      setZoom(next);
      if (next === 1) setPan({ x: 0, y: 0 });
    }
  };

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setWindowPreset("standard");
    setViewMode("heatmap");
    setLoupeActive(false);
    setCaliperMode(false);
    setCalipers([]);
  }, []);

  // Radiologist PACS Keyboard Shortcuts (H/A/C/O view modes, I invert, L loupe, M measure, R reset)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target;
      if (t instanceof HTMLElement) {
        const tag = t.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable) return;
      }
      const k = e.key.toLowerCase();
      if (k === "h") setViewMode("heatmap");
      else if (k === "a") setViewMode("anatomy");
      else if (k === "c") setViewMode("compare");
      else if (k === "o") setViewMode("original");
      else if (k === "i") setWindowPreset((prev) => (prev === "inverted" ? "standard" : "inverted"));
      else if (k === "l") {
        setLoupeActive((v) => !v);
        setCaliperMode(false);
      } else if (k === "m") {
        setCaliperMode((v) => !v);
        setLoupeActive(false);
      } else if (k === "r") {
        resetView();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [resetView]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onFileUpload(file);
  };

  const predictions = predictionData?.predictions || [];
  const positives = useMemo(
    () => predictions.filter((p) => p.score >= threshold),
    [predictions, threshold]
  );
  const visiblePredictions = useMemo(() => {
    if (showAllPathologies) return predictions;
    const topFive = predictions.slice(0, 5);
    const extraPositives = positives.filter((p) => !topFive.some((t) => t.name === p.name));
    return [...topFive, ...extraPositives];
  }, [predictions, positives, showAllPathologies]);

  const leadPositive = positives[0] || null;
  const anatomy = predictionData?.anatomy || null;
  const decision = predictionData?.decision_summary || null;
  const rawTriage =
    predictionData?.report?.triage_level || (positives.length ? "URGENT" : "ROUTINE");
  const triageLabel = formatTriageLabel(rawTriage);
  const isCritical = Boolean(
    predictionData?.report?.critical_alert || rawTriage.toUpperCase().includes("STAT")
  );
  const ctr = anatomy?.ctr ?? null;
  const isApView = Boolean(
    (anatomy?.projection_view || decision?.projection_view || "").toUpperCase().includes("AP")
  );
  const ctrNormalCutoff = isApView ? 0.55 : 0.5;
  const ctrEnlargedCutoff = anatomy?.ctr_cutoff ?? (isApView ? 0.58 : 0.53);
  const ctrPct = ctr != null ? Math.min(100, Math.max(0, ((ctr - 0.3) / 0.4) * 100)) : 0;
  const ctrMarkerPct = ((ctrNormalCutoff - 0.3) / 0.4) * 100;
  const ctrIsAlert = ctr != null && ctr >= ctrNormalCutoff;

  return (
    <div className="mx-auto max-w-[1280px] px-5 py-6 pb-24 md:px-8 md:py-7 lg:pb-10">
      {/* Compact Single-Line Studio Header + Non-Wrapping Study Pill Bar */}
      <div className="mb-6 flex flex-col justify-between gap-4 xl:flex-row xl:items-center">
        <div>
          <h1 className="font-display text-2xl font-bold sm:text-3xl">
            Scan{" "}
            <span
              className="font-serif-editorial font-normal"
              style={{ color: "var(--accent-clinical)" }}
            >
              Studio
            </span>
          </h1>
          <p className="mt-0.5 text-xs" style={{ color: "var(--text-muted)" }}>
            Click any pathology on the right to localize its Grad-CAM++ heatmap.
          </p>
        </div>

        <div className="no-scrollbar flex items-center gap-1.5 overflow-x-auto pb-1 xl:pb-0">
          {samples.map((s) => {
            const active = activeStudyId === s.id;
            return (
              <button
                key={s.id}
                type="button"
                disabled={isAnalyzing}
                onClick={() => onSelectSample(s)}
                className="shrink-0 cursor-pointer rounded-full border px-3 py-1.5 text-xs font-medium whitespace-nowrap transition-all hover:border-teal-500/40 disabled:opacity-50"
                style={{
                  borderColor: active ? "var(--accent-clinical)" : "var(--border-subtle)",
                  backgroundColor: active ? "var(--accent-clinical-dim)" : "var(--bg-elevated)",
                  color: active ? "var(--text-primary)" : "var(--text-secondary)",
                }}
              >
                {shortSampleTitle(s.name)}
              </button>
            );
          })}
          <button
            type="button"
            onClick={onUploadClick}
            className="flex shrink-0 cursor-pointer items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-semibold whitespace-nowrap transition-colors hover:border-teal-500/50"
            style={{
              borderColor: "var(--border-strong)",
              backgroundColor: "var(--bg-elevated)",
              color: "var(--text-primary)",
            }}
          >
            <Upload className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
            <span>Upload</span>
          </button>
        </div>
      </div>

      {/* Main 2-Column Workspace */}
      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        {/* LEFT 7 COLS: Calm Radiograph Lightbox */}
        <div className="lg:col-span-7 space-y-3">
          {/* Unified 4-Mode Segmented Control + Progressive Disclosure Tools Button */}
          <div className="atelier-card flex flex-wrap items-center justify-between gap-2 rounded-2xl px-2.5 py-2">
            <div
              className="inline-flex items-center gap-0.5 rounded-full p-0.5"
              style={{ backgroundColor: "var(--bg-secondary)" }}
            >
              {(
                [
                  { id: "heatmap", label: "Heatmap", icon: Eye },
                  { id: "anatomy", label: "Anatomy", icon: Layers },
                  { id: "compare", label: "Compare", icon: Columns2 },
                  { id: "original", label: "Original", icon: ImageIcon },
                ] as const
              ).map((m) => {
                const Icon = m.icon;
                const active = viewMode === m.id;
                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setViewMode(m.id)}
                    className="relative z-10 flex cursor-pointer items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium whitespace-nowrap transition-colors"
                    style={{
                      color: active ? "var(--text-primary)" : "var(--text-secondary)",
                    }}
                  >
                    {active && (
                      <motion.span
                        layoutId="studio-view-mode"
                        transition={{ type: "spring", stiffness: 420, damping: 34 }}
                        className="absolute inset-0 -z-10 rounded-full border"
                        style={{
                          backgroundColor: "var(--bg-elevated)",
                          borderColor: "var(--border-strong)",
                        }}
                      />
                    )}
                    <Icon
                      className="h-3.5 w-3.5"
                      style={{ color: active ? "var(--accent-clinical)" : "var(--text-muted)" }}
                    />
                    <span>{m.label}</span>
                  </button>
                );
              })}
            </div>

            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => setToolsOpen((v) => !v)}
                className="flex cursor-pointer items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors hover:border-teal-500/40"
                style={{
                  borderColor:
                    toolsOpen || loupeActive || caliperMode || windowPreset !== "standard"
                      ? "var(--accent-clinical)"
                      : "var(--border-subtle)",
                  backgroundColor: toolsOpen ? "var(--accent-clinical-dim)" : "transparent",
                  color: toolsOpen ? "var(--text-primary)" : "var(--text-secondary)",
                }}
              >
                <SlidersHorizontal className="h-3.5 w-3.5" />
                <span>Adjust</span>
              </button>

              {(zoom > 1 || windowPreset !== "standard" || calipers.length > 0 || loupeActive || caliperMode) && (
                <button
                  type="button"
                  onClick={resetView}
                  title="Reset view"
                  aria-label="Reset viewer"
                  className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full border transition-colors hover:border-teal-500/40"
                  style={{ borderColor: "var(--border-subtle)", color: "var(--text-secondary)" }}
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          </div>

          {/* Smooth Collapsible Precision Image Controls (Progressive Disclosure) */}
          <AnimatePresence initial={false}>
            {toolsOpen && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                className="overflow-hidden"
              >
                <div className="atelier-card flex flex-wrap items-center justify-between gap-3 rounded-2xl px-4 py-3">
                  {/* Window/Level Presets */}
                  <div className="flex flex-wrap items-center gap-1">
                    {WINDOW_PRESETS.map((wp) => {
                      const active = windowPreset === wp.id;
                      return (
                        <button
                          key={wp.id}
                          type="button"
                          onClick={() => setWindowPreset(wp.id)}
                          className="cursor-pointer rounded-full px-2.5 py-1 text-[11.5px] font-medium transition-colors"
                          style={{
                            backgroundColor: active ? "var(--text-primary)" : "var(--bg-secondary)",
                            color: active ? "var(--bg-primary)" : "var(--text-secondary)",
                          }}
                        >
                          {wp.label}
                        </button>
                      );
                    })}
                  </div>

                  {/* Opacity + Colormap + Loupe + Caliper */}
                  <div className="flex flex-wrap items-center gap-2.5">
                    {showHeatmap && (
                      <div className="flex items-center gap-2">
                        <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                          Opacity {Math.round(heatmapOpacity * 100)}%
                        </span>
                        <input
                          type="range"
                          min={0.15}
                          max={1}
                          step={0.05}
                          value={heatmapOpacity}
                          onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                          aria-label="Heatmap opacity"
                          className="clinical-slider w-16"
                        />
                      </div>
                    )}

                    <select
                      aria-label="Heatmap Colormap"
                      value={colormap}
                      onChange={(e) => onColormapChange(e.target.value as ColormapName)}
                      className="cursor-pointer rounded-full border px-2.5 py-1 text-[11.5px] focus:outline-none"
                      style={{
                        borderColor: "var(--border-subtle)",
                        backgroundColor: "var(--bg-secondary)",
                        color: "var(--text-primary)",
                      }}
                    >
                      {COLORMAP_OPTIONS.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.label} Map
                        </option>
                      ))}
                    </select>

                    <button
                      type="button"
                      onClick={() => {
                        setLoupeActive((v) => !v);
                        setCaliperMode(false);
                      }}
                      className="flex cursor-pointer items-center gap-1 rounded-full border px-2.5 py-1 text-[11.5px] font-medium"
                      style={{
                        borderColor: loupeActive ? "var(--accent-clinical)" : "var(--border-subtle)",
                        backgroundColor: loupeActive ? "var(--accent-clinical-dim)" : "transparent",
                        color: loupeActive ? "var(--text-primary)" : "var(--text-secondary)",
                      }}
                    >
                      <Search className="h-3 w-3" />
                      <span>2.5× Loupe</span>
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        setCaliperMode((v) => !v);
                        setLoupeActive(false);
                      }}
                      className="flex cursor-pointer items-center gap-1 rounded-full border px-2.5 py-1 text-[11.5px] font-medium"
                      style={{
                        borderColor: caliperMode ? "var(--accent-clinical)" : "var(--border-subtle)",
                        backgroundColor: caliperMode ? "var(--accent-clinical-dim)" : "transparent",
                        color: caliperMode ? "var(--text-primary)" : "var(--text-secondary)",
                      }}
                    >
                      <Ruler className="h-3 w-3" />
                      <span>Measure</span>
                    </button>

                    {calipers.length > 0 && (
                      <button
                        type="button"
                        onClick={() => setCalipers([])}
                        title="Clear measurements"
                        className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-full border"
                        style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Clean Carbon Film Lightbox */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            className="relative aspect-square w-full overflow-hidden rounded-[24px] border transition-all"
            style={{
              backgroundColor: "var(--bg-film)",
              borderColor: isDragOver ? "var(--accent-clinical)" : "var(--border-strong)",
              boxShadow: "var(--shadow-card)",
            }}
          >
            <canvas
              ref={canvasRef}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerLeave={() => setPointer((p) => ({ ...p, inside: false }))}
              onPointerUp={handlePointerUp}
              onWheel={handleWheel}
              className="h-full w-full touch-none"
              style={{
                cursor: caliperMode
                  ? "crosshair"
                  : splitCompare
                  ? "ew-resize"
                  : zoom > 1
                  ? "grab"
                  : "default",
              }}
            />

            {/* Draggable Split-Compare Handle */}
            {splitCompare && !holdingSpace && (
              <div
                aria-hidden="true"
                className="pointer-events-none absolute top-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-teal-400/75 bg-black/82 text-teal-300 shadow-xl backdrop-blur-md"
                style={{ left: `${splitPos * 100}%` }}
              >
                <MoveHorizontal className="h-4 w-4" />
              </div>
            )}

            {/* Single Clean Localization Badge (No Redundant Top-Left Badges) */}
            {activePathology && viewMode !== "original" && !holdingSpace && (
              <div className="pointer-events-none absolute top-3.5 right-3.5 flex items-center gap-2 rounded-full border border-white/14 bg-black/78 px-3 py-1.5 backdrop-blur-md">
                <span className="h-1.5 w-1.5 rounded-full bg-teal-400" />
                <span className="text-[11px] font-medium text-white">
                  {formatPathology(activePathology)}
                </span>
                {activeHeatmap?.anatomical_zone && (
                  <span className="hidden border-l border-white/15 pl-2 text-[11px] text-slate-300 sm:inline">
                    {activeHeatmap.anatomical_zone}
                  </span>
                )}
              </div>
            )}

            {holdingSpace && (
              <div className="pointer-events-none absolute top-3.5 left-3.5 rounded-full border border-teal-400/60 bg-black/85 px-3 py-1 font-mono-tech text-[10px] text-teal-300">
                Original X-Ray (Space)
              </div>
            )}

            {/* Minimal Bottom Telemetry (Only when relevant) */}
            <div className="pointer-events-none absolute right-3.5 bottom-3.5 left-3.5 flex items-center justify-between">
              <div>
                {splitCompare && !holdingSpace ? (
                  <span className="rounded-full border border-white/12 bg-black/75 px-2.5 py-1 font-mono-tech text-[10px] text-slate-200">
                    Original
                  </span>
                ) : (
                  pointer.inside &&
                  zoom > 1 && (
                    <span className="rounded-full border border-white/12 bg-black/75 px-2.5 py-1 font-mono-tech text-[10px] text-slate-200">
                      {zoom.toFixed(1)}× Zoom
                    </span>
                  )
                )}
              </div>

              <div className="flex items-center gap-2">
                {splitCompare && !holdingSpace && (
                  <span className="rounded-full border border-teal-400/40 bg-black/75 px-2.5 py-1 font-mono-tech text-[10px] text-teal-300">
                    AI Overlay
                  </span>
                )}
                {heatmapLoading && !isAnalyzing && (
                  <span className="rounded-full border border-white/15 bg-black/80 px-3 py-1 font-mono-tech text-[10px] text-slate-200">
                    Updating heatmap...
                  </span>
                )}
              </div>
            </div>

            {caliperMode && (
              <div className="pointer-events-none absolute bottom-12 left-1/2 -translate-x-1/2 rounded-full border border-teal-400/45 bg-black/85 px-3.5 py-1.5 text-[11px] text-teal-300 backdrop-blur-md">
                Click &amp; drag across the image to measure in mm
              </div>
            )}

            {isAnalyzing && analysisStage === "vision" && (
              <div className="absolute inset-0 overflow-hidden bg-black/55 backdrop-blur-[2px]">
                <div
                  className="scan-beam h-24 w-full"
                  style={{
                    background:
                      "linear-gradient(180deg, transparent, rgba(20, 184, 166, 0.28), transparent)",
                  }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="flex items-center gap-2.5 rounded-full border border-white/15 bg-black/88 px-4 py-2 text-xs font-medium text-white shadow-2xl">
                    <span className="h-2 w-2 rounded-full bg-teal-400 clinical-led" />
                    <span>Running 4-CNN ensemble &amp; Grad-CAM++...</span>
                  </div>
                </div>
              </div>
            )}

            {/* Non-blocking Progressive Background Refinement Pill (Lightbox Already Interactive!) */}
            {isAnalyzing && analysisStage !== "vision" && (
              <div className="pointer-events-none absolute top-3.5 left-3.5 flex items-center gap-2 rounded-full border border-teal-400/45 bg-black/82 px-3 py-1.5 text-[11px] font-medium text-teal-200 backdrop-blur-md">
                <span className="h-1.5 w-1.5 rounded-full bg-teal-400 clinical-led" />
                <span>
                  {analysisStage === "consensus"
                    ? "Fusing Gemini Decision 1..."
                    : "Synthesizing ACR report..."}
                </span>
              </div>
            )}
          </div>

          <div
            className="flex flex-wrap items-center justify-between gap-2 px-1 text-[11px]"
            style={{ color: "var(--text-muted)" }}
          >
            <span>Space: peek original · H/A/C/O: views · I: invert · M: measure</span>
            <span className="font-mono-tech">
              {predictionData?.cache_hit
                ? `Instant Cache · ${Math.max(1, Math.round(predictionData.inference_time_ms))} ms`
                : predictionData?.vision_latency_ms
                ? `Vision · ${Math.round(predictionData.vision_latency_ms)} ms`
                : "Ctrl + Scroll to zoom"}
            </span>
          </div>
        </div>

        {/* RIGHT 5 COLS: Clear Diagnostic Summary + Scannable Findings */}
        <div className="lg:col-span-5 space-y-5">
          {/* Primary Diagnostic Verdict */}
          <SpotlightCard strong className="rounded-[24px] p-5 sm:p-6 space-y-5">
            <div className="flex items-center justify-between gap-2">
              <span
                className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold"
                style={{
                  backgroundColor:
                    isCritical || leadPositive
                      ? "var(--accent-alert-dim)"
                      : "var(--accent-clinical-dim)",
                  color:
                    isCritical || leadPositive ? "var(--accent-alert)" : "var(--accent-clinical)",
                }}
              >
                <span
                  className="h-1.5 w-1.5 rounded-full"
                  style={{
                    backgroundColor:
                      isCritical || leadPositive
                        ? "var(--accent-alert)"
                        : "var(--accent-clinical)",
                  }}
                />
                {triageLabel}
              </span>

              {decision?.active && (
                <span
                  className="inline-flex items-center gap-1 text-[11px]"
                  style={{ color: "var(--text-muted)" }}
                >
                  <Sparkles className="h-3 w-3" style={{ color: "var(--accent-clinical)" }} />
                  <span>Gemini Fused</span>
                </span>
              )}
            </div>

            <div className="space-y-1">
              <div className="flex items-baseline justify-between gap-4">
                <h2 className="font-display text-2xl font-bold sm:text-[26px]">
                  {leadPositive
                    ? formatPathology(leadPositive.name)
                    : predictionData
                    ? "No Acute Findings"
                    : "Loading..."}
                </h2>
                {leadPositive && (
                  <AnimatedNumber
                    value={leadPositive.score * 100}
                    decimals={1}
                    suffix="%"
                    className="font-mono-tech text-2xl font-semibold"
                    style={{ color: "var(--accent-alert)" }}
                  />
                )}
              </div>
              <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                {leadPositive
                  ? PATHOLOGY_CLINICAL_NOTE[leadPositive.name]?.plain ||
                    "Abnormality detected above threshold."
                  : predictionData
                  ? "All 14 evaluated conditions are below the active threshold."
                  : "Upload an X-ray or choose a sample study above."}
              </p>
            </div>

            {/* Clean Heart Size (CTR) Bar */}
            <div
              className="rounded-2xl border p-3.5 space-y-2"
              style={{
                borderColor: "var(--border-subtle)",
                backgroundColor: "var(--bg-secondary)",
              }}
            >
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-1.5 font-medium">
                  <HeartPulse className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
                  <span>Heart Size (CTR)</span>
                  {isApView && (
                    <span
                      className="rounded-full px-1.5 py-0.5 font-mono-tech text-[9.5px]"
                      style={{
                        backgroundColor: "var(--accent-optical-dim)",
                        color: "var(--accent-optical)",
                      }}
                    >
                      AP Adjusted
                    </span>
                  )}
                </span>
                <div className="flex items-center gap-2 font-mono-tech">
                  <AnimatedNumber
                    value={ctr}
                    decimals={3}
                    className="font-semibold"
                    style={{
                      color: ctrIsAlert ? "var(--accent-alert)" : "var(--accent-clinical)",
                    }}
                  />
                  <span className="text-[10.5px]" style={{ color: "var(--text-muted)" }}>
                    {ctr != null
                      ? ctr >= ctrEnlargedCutoff
                        ? "Enlarged"
                        : ctr >= ctrNormalCutoff
                        ? "Borderline"
                        : "Normal"
                      : "—"}
                  </span>
                </div>
              </div>

              <div
                className="relative h-1.5 w-full rounded-full"
                style={{ backgroundColor: "var(--border-subtle)" }}
              >
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${ctrPct}%` }}
                  transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                  className="h-full rounded-full"
                  style={{
                    backgroundColor: ctrIsAlert ? "var(--accent-alert)" : "var(--accent-clinical)",
                  }}
                />
                <div
                  className="absolute top-1/2 h-3 w-[1.5px] -translate-x-1/2 -translate-y-1/2"
                  style={{
                    left: `${ctrMarkerPct}%`,
                    backgroundColor: "var(--text-primary)",
                  }}
                  title={`${ctrNormalCutoff.toFixed(2)} clinical cutoff`}
                />
              </div>

              <div
                className="flex items-center justify-between font-mono-tech text-[10px]"
                style={{ color: "var(--text-muted)" }}
              >
                <span>0.30</span>
                <span>
                  {ctrNormalCutoff.toFixed(2)} {isApView ? "AP portable limit" : "PA normal limit"}
                </span>
                <span>0.70</span>
              </div>
            </div>

            {/* Clear Primary Next Step + Secondary Link */}
            <div className="flex items-center gap-2.5">
              <button
                type="button"
                onClick={() => onNavigatePage("report")}
                className="group flex flex-1 cursor-pointer items-center justify-center gap-2 rounded-full py-2.5 text-xs font-semibold transition-all hover:brightness-110 active:scale-[0.98]"
                style={{
                  backgroundColor: "var(--accent-clinical)",
                  color: "#ffffff",
                }}
              >
                <span>Open Report &amp; AI Chat</span>
                <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
              </button>
              <button
                type="button"
                onClick={() => onNavigatePage("consensus")}
                className="cursor-pointer rounded-full border px-3.5 py-2.5 text-xs font-medium transition-colors hover:border-teal-500/40"
                style={{
                  borderColor: "var(--border-strong)",
                  color: "var(--text-primary)",
                }}
              >
                {predictionData?.ensemble_info?.raddino_active ? "6-Model View" : "5-Model View"}
              </button>
            </div>
          </SpotlightCard>

          {/* Ranked Findings Card (Top 5 Clean Default + Collapsible Cutoff) */}
          <SpotlightCard className="rounded-[24px] p-5 sm:p-6 space-y-4">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                <h3 className="font-display text-sm font-bold">Top Findings</h3>
              </div>

              <button
                type="button"
                onClick={() => setCutoffOpen((v) => !v)}
                className="flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono-tech text-[10.5px] transition-colors hover:border-teal-500/40"
                style={{
                  borderColor: cutoffOpen ? "var(--accent-clinical)" : "var(--border-subtle)",
                  color: "var(--text-secondary)",
                }}
              >
                <SlidersHorizontal className="h-3 w-3" />
                <span>Threshold {Math.round(threshold * 100)}%</span>
              </button>
            </div>

            <AnimatePresence initial={false}>
              {cutoffOpen && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.18 }}
                  className="overflow-hidden"
                >
                  <div
                    className="space-y-2.5 rounded-xl border p-3 text-xs"
                    style={{
                      borderColor: "var(--border-subtle)",
                      backgroundColor: "var(--bg-secondary)",
                    }}
                  >
                    <div className="grid grid-cols-3 gap-1.5">
                      {CLINICAL_MODES.map((m) => {
                        const active = Math.abs(threshold - m.value) < 0.025;
                        return (
                          <button
                            key={m.id}
                            type="button"
                            onClick={() => onThresholdChange(m.value)}
                            className="cursor-pointer rounded-lg border px-2 py-1.5 text-center transition-colors"
                            style={{
                              borderColor: active ? "var(--accent-clinical)" : "var(--border-subtle)",
                              backgroundColor: active ? "var(--accent-clinical-dim)" : "var(--bg-elevated)",
                              color: active ? "var(--text-primary)" : "var(--text-secondary)",
                            }}
                          >
                            <div className="text-[11px] font-semibold">{m.label}</div>
                            <div
                              className="font-mono-tech text-[9.5px]"
                              style={{ color: active ? "var(--accent-clinical)" : "var(--text-muted)" }}
                            >
                              {Math.round(m.value * 100)}% · {m.hint}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                    <div className="flex items-center justify-between gap-3 pt-0.5">
                      <span style={{ color: "var(--text-secondary)" }}>Fine cutoff</span>
                      <input
                        type="range"
                        min={0.25}
                        max={0.75}
                        step={0.01}
                        value={threshold}
                        onChange={(e) => onThresholdChange(Number(e.target.value))}
                        aria-label="Diagnostic cutoff threshold"
                        className="clinical-slider w-28"
                      />
                      <span className="font-mono-tech font-semibold">
                        {Math.round(threshold * 100)}%
                      </span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="space-y-1.5">
              {visiblePredictions.map((pred) => {
                const isPos = pred.score >= threshold;
                const isReview =
                  !isPos &&
                  (pred.review_flag === "borderline" ||
                    pred.review_flag === "uncertain" ||
                    pred.score >= 0.3);
                const isSelected = activePathology === pred.name;
                const note = PATHOLOGY_CLINICAL_NOTE[pred.name];

                return (
                  <button
                    key={pred.name}
                    type="button"
                    onClick={() => onSelectPathology(pred.name)}
                    className="group w-full cursor-pointer rounded-2xl border px-3.5 py-2.5 text-left transition-all hover:border-teal-500/35"
                    style={{
                      borderColor: isSelected
                        ? "var(--accent-clinical)"
                        : isPos
                        ? "var(--accent-alert-dim)"
                        : "transparent",
                      backgroundColor: isSelected
                        ? "var(--accent-clinical-dim)"
                        : isPos
                        ? "var(--accent-alert-dim)"
                        : "transparent",
                    }}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex min-w-0 items-center gap-2">
                        <span
                          className="h-1.5 w-1.5 shrink-0 rounded-full"
                          style={{
                            backgroundColor: isPos
                              ? "var(--accent-alert)"
                              : isReview
                              ? "var(--accent-amber)"
                              : isSelected
                              ? "var(--accent-clinical)"
                              : "var(--border-strong)",
                          }}
                        />
                        <span className="truncate text-[13.5px] font-medium">
                          {formatPathology(pred.name)}
                        </span>
                        {isPos && pred.conformal_decision === "AUTO_POSITIVE" && (
                          <span
                            title="Split-conformal high-agreement call (low inter-model variance)"
                            className="shrink-0 rounded-full px-1.5 py-0.5 font-mono-tech text-[9.5px] font-medium"
                            style={{
                              backgroundColor: "var(--accent-clinical-dim)",
                              color: "var(--accent-clinical)",
                            }}
                          >
                            High-Conf
                          </span>
                        )}
                        {isReview && (
                          <span
                            title="Sub-threshold borderline or inter-model uncertainty — verify clinically"
                            className="shrink-0 rounded-full px-1.5 py-0.5 font-mono-tech text-[9.5px] font-medium"
                            style={{
                              backgroundColor: "var(--accent-amber-dim)",
                              color: "var(--accent-amber)",
                            }}
                          >
                            Verify
                          </span>
                        )}
                      </div>
                      <span
                        className="font-mono-tech text-xs font-semibold"
                        style={{
                          color: isPos
                            ? "var(--accent-alert)"
                            : isSelected
                            ? "var(--accent-clinical)"
                            : "var(--text-secondary)",
                        }}
                      >
                        {(pred.score * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div
                      className="relative mt-2 h-1 w-full rounded-full"
                      style={{ backgroundColor: "var(--border-subtle)" }}
                    >
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${Math.min(100, Math.max(2, pred.score * 100))}%`,
                          backgroundColor: isPos
                            ? "var(--accent-alert)"
                            : isSelected
                            ? "var(--accent-clinical)"
                            : "var(--accent-titanium)",
                        }}
                      />
                    </div>

                    {isSelected && note && (
                      <div
                        className="mt-1.5 flex items-center justify-between text-[11px]"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        <span className="truncate">{note.plain}</span>
                        {pred.anatomical_zone && (
                          <span
                            className="ml-2 shrink-0 font-mono-tech text-[10px]"
                            style={{ color: "var(--accent-clinical)" }}
                          >
                            {pred.anatomical_zone}
                          </span>
                        )}
                      </div>
                    )}
                  </button>
                );
              })}
            </div>

            {predictions.length > 5 && (
              <button
                type="button"
                onClick={() => setShowAllPathologies((v) => !v)}
                className="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-xl border py-2 text-xs font-medium transition-colors hover:border-teal-500/35"
                style={{
                  borderColor: "var(--border-subtle)",
                  color: "var(--text-secondary)",
                }}
              >
                {showAllPathologies ? (
                  <>
                    <span>Show Top 5 Only</span>
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

            {/* Quiet Patient Context Bar */}
            <div
              className="flex items-center justify-between gap-3 border-t pt-3.5"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              <div className="flex min-w-0 items-center gap-2.5">
                <UserRound className="h-4 w-4 shrink-0" style={{ color: "var(--accent-clinical)" }} />
                <div className="min-w-0">
                  <div className="truncate text-xs font-medium">
                    {patient.age ? `${patient.age}y ` : ""}
                    {patient.sex} · {patient.setting}
                  </div>
                </div>
              </div>
              <button
                type="button"
                onClick={onOpenPatientDrawer}
                className="shrink-0 cursor-pointer text-xs font-medium transition-colors hover:underline"
                style={{ color: "var(--accent-clinical)" }}
              >
                Edit Context
              </button>
            </div>
          </SpotlightCard>
        </div>
      </div>
    </div>
  );
};
