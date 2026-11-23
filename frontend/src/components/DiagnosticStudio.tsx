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
  cleanSampleTitle,
  formatPathology,
  type PageId,
} from "../data/clinicalCatalog";
import type {
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
  activeStudyName,
  imageUrl,
  predictionData,
  activePathology,
  activeHeatmap,
  heatmapLoading,
  threshold,
  colormap,
  patient,
  isAnalyzing,
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

  // Core 4 Imaging Overlays
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showAnatomy, setShowAnatomy] = useState(false);
  const [splitCompare, setSplitCompare] = useState(false);
  const [splitPos, setSplitPos] = useState(0.5);
  const [loupeActive, setLoupeActive] = useState(false);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.68);
  const [windowPreset, setWindowPreset] = useState<WindowPreset>("standard");

  // Zoom / Pan / Caliper / Cursor Telemetry
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

  // Precision Radiograph Canvas Compositor
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

      // 1. Base Chest Radiograph
      ctx.filter = filterStr;
      ctx.drawImage(baseImg, sx, sy, srcSide, srcSide, dx, dy, side, side);
      ctx.filter = "none";

      if (withOverlays && !holdingSpace) {
        // 2. PSPNet Anatomy & CTR Contour Overlay
        if (showAnatomy && anatImg) {
          ctx.save();
          ctx.globalAlpha = 0.86;
          ctx.drawImage(anatImg, dx, dy, side, side);
          ctx.restore();
        }

        // 3. Grad-CAM++ Pure RGBA Spatial Heatmap + Reticle
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
            ctx.strokeStyle = "rgba(20, 184, 166, 0.78)";
            ctx.lineWidth = 1.25 / zoom;
            ctx.setLineDash([5 / zoom, 4 / zoom]);
            ctx.strokeRect(rx, ry, rw, rh);
            ctx.restore();
          }

          if (activeHeatmap?.peak_x != null && activeHeatmap?.peak_y != null) {
            const px = dx + activeHeatmap.peak_x * side;
            const py = dy + activeHeatmap.peak_y * side;
            const r = 8.5 / zoom;
            ctx.save();
            ctx.strokeStyle = "#f1f5f8";
            ctx.lineWidth = 1.4 / zoom;
            ctx.beginPath();
            ctx.arc(px, py, r, 0, Math.PI * 2);
            ctx.moveTo(px - r * 1.75, py);
            ctx.lineTo(px - r * 0.65, py);
            ctx.moveTo(px + r * 0.65, py);
            ctx.lineTo(px + r * 1.75, py);
            ctx.moveTo(px, py - r * 1.75);
            ctx.lineTo(px, py - r * 0.65);
            ctx.moveTo(px, py + r * 0.65);
            ctx.lineTo(px, py + r * 1.75);
            ctx.stroke();
            ctx.restore();
          }
        }
      }

      // 4. Caliper Measurements
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
        ctx.lineWidth = 1.65 / zoom;
        ctx.beginPath();
        ctx.moveTo(cx1, cy1);
        ctx.lineTo(cx2, cy2);
        ctx.stroke();

        [
          [cx1, cy1],
          [cx2, cy2],
        ].forEach(([ex, ey]) => {
          ctx.beginPath();
          ctx.arc(ex, ey, 3.2 / zoom, 0, Math.PI * 2);
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
      ctx.strokeStyle = "rgba(20, 184, 166, 0.82)";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(splitX, 0);
      ctx.lineTo(splitX, h);
      ctx.stroke();
      ctx.restore();
    } else {
      drawScene(true);
    }

    // 5. Etched DICOM Left-Edge Millimeter Scale Ruler (Authentic Radiology Craft)
    ctx.save();
    const rulerTop = h * 0.2;
    const rulerBot = h * 0.8;
    const ticks = 20;
    ctx.strokeStyle = "rgba(241, 245, 248, 0.28)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(14, rulerTop);
    ctx.lineTo(14, rulerBot);
    for (let i = 0; i <= ticks; i++) {
      const ty = rulerTop + ((rulerBot - rulerTop) * i) / ticks;
      const len = i % 5 === 0 ? 8 : 4;
      ctx.moveTo(14, ty);
      ctx.lineTo(14 + len, ty);
    }
    ctx.stroke();
    ctx.fillStyle = "rgba(241, 245, 248, 0.36)";
    ctx.font = '9px "IBM Plex Mono", monospace';
    ctx.fillText("200mm", 12, rulerBot + 14);
    ctx.restore();

    // 6. 2.5× Optical Magnifier Loupe
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

      ctx.strokeStyle = "rgba(241, 245, 248, 0.38)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(lx - 11, ly);
      ctx.lineTo(lx + 11, ly);
      ctx.moveTo(lx, ly - 11);
      ctx.lineTo(lx, ly + 11);
      ctx.stroke();
      ctx.restore();
    }
  }, [
    baseImg,
    heatImg,
    anatImg,
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

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setWindowPreset("standard");
    setSplitCompare(false);
    setLoupeActive(false);
    setCaliperMode(false);
    setCalipers([]);
  };

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
    const topSix = predictions.slice(0, 6);
    const extraPositives = positives.filter((p) => !topSix.some((t) => t.name === p.name));
    return [...topSix, ...extraPositives];
  }, [predictions, positives, showAllPathologies]);

  const leadPositive = positives[0] || null;
  const anatomy = predictionData?.anatomy || null;
  const decision = predictionData?.decision_summary || null;
  const triageLevel =
    predictionData?.report?.triage_level || (positives.length ? "URGENT" : "ROUTINE");
  const isCritical = Boolean(
    predictionData?.report?.critical_alert || triageLevel.toUpperCase().includes("STAT")
  );
  const ctr = anatomy?.ctr ?? null;
  const ctrPct = ctr != null ? Math.min(100, Math.max(0, ((ctr - 0.3) / 0.4) * 100)) : 0;

  return (
    <div className="mx-auto max-w-[1360px] px-5 py-7 pb-24 md:px-8 md:py-9 lg:pb-12">
      {/* Page Header + Fast Study Switcher */}
      <div className="mb-7 flex flex-col justify-between gap-5 xl:flex-row xl:items-end">
        <div className="space-y-1.5">
          <div
            className="font-mono-tech text-[10px] font-medium tracking-[0.2em] uppercase"
            style={{ color: "var(--accent-clinical)" }}
          >
            02 · DIAGNOSTIC SCAN STUDIO
          </div>
          <h1 className="font-display text-3xl font-bold sm:text-4xl">
            Inspect film &amp;{" "}
            <span className="font-serif-editorial font-normal" style={{ color: "var(--accent-clinical)" }}>
              localized evidence.
            </span>
          </h1>
        </div>

        {/* Clean Benchmark Case Switcher Strip */}
        <div className="flex flex-wrap items-center gap-1.5">
          {samples.map((s) => {
            const active = activeStudyId === s.id;
            return (
              <button
                key={s.id}
                type="button"
                disabled={isAnalyzing}
                onClick={() => onSelectSample(s)}
                className="cursor-pointer rounded-full border px-3 py-1.5 text-xs font-medium transition-all disabled:opacity-50"
                style={{
                  borderColor: active ? "var(--accent-clinical)" : "var(--border-subtle)",
                  backgroundColor: active ? "var(--accent-clinical-dim)" : "var(--bg-elevated)",
                  color: active ? "var(--text-primary)" : "var(--text-secondary)",
                }}
              >
                {cleanSampleTitle(s.name)}
              </button>
            );
          })}
          <button
            type="button"
            onClick={onUploadClick}
            className="flex cursor-pointer items-center gap-1.5 rounded-full px-3.5 py-1.5 text-xs font-semibold transition-transform active:scale-95"
            style={{
              backgroundColor: "var(--text-primary)",
              color: "var(--bg-primary)",
            }}
          >
            <Upload className="h-3.5 w-3.5" />
            <span>Upload Film</span>
          </button>
        </div>
      </div>

      {/* Main 2-Column Studio Grid */}
      <div className="grid grid-cols-1 items-start gap-7 lg:grid-cols-12">
        {/* LEFT 7 COLS: Precision DICOM Lightbox */}
        <div className="lg:col-span-7 space-y-3.5">
          {/* Top 4-Mode Overlay Strip — Crisp Single Horizontal Row even at 1280px */}
          <div className="atelier-card flex flex-wrap items-center justify-between gap-2 rounded-2xl px-3 py-2 xl:flex-nowrap">
            <div className="flex flex-wrap items-center gap-1 xl:flex-nowrap">
              {[
                {
                  label: "Grad-CAM++",
                  icon: Eye,
                  active: showHeatmap,
                  onClick: () => setShowHeatmap((v) => !v),
                },
                {
                  label: "Anatomy & CTR",
                  icon: Layers,
                  active: showAnatomy,
                  onClick: () => setShowAnatomy((v) => !v),
                },
                {
                  label: "Split Compare",
                  icon: Columns2,
                  active: splitCompare,
                  onClick: () => setSplitCompare((v) => !v),
                },
                {
                  label: "2.5× Loupe",
                  icon: Search,
                  active: loupeActive,
                  onClick: () => {
                    setLoupeActive((v) => !v);
                    setCaliperMode(false);
                  },
                },
              ].map((ctrl) => {
                const Icon = ctrl.icon;
                return (
                  <button
                    key={ctrl.label}
                    type="button"
                    onClick={ctrl.onClick}
                    className="flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-1 text-[11px] font-medium transition-all"
                    style={{
                      borderColor: ctrl.active ? "var(--accent-clinical)" : "var(--border-subtle)",
                      backgroundColor: ctrl.active ? "var(--accent-clinical-dim)" : "transparent",
                      color: ctrl.active ? "var(--text-primary)" : "var(--text-secondary)",
                    }}
                  >
                    <Icon
                      className="h-3 w-3 shrink-0"
                      style={{ color: ctrl.active ? "var(--accent-clinical)" : "var(--text-muted)" }}
                    />
                    <span>{ctrl.label}</span>
                  </button>
                );
              })}
            </div>

            {showHeatmap && (
              <div className="flex shrink-0 items-center gap-1.5 pl-1">
                <span
                  className="font-mono-tech text-[9.5px] uppercase whitespace-nowrap"
                  style={{ color: "var(--text-muted)" }}
                >
                  {Math.round(heatmapOpacity * 100)}%
                </span>
                <input
                  type="range"
                  min={0.15}
                  max={1}
                  step={0.05}
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                  aria-label="Heatmap opacity"
                  className="clinical-slider w-14"
                />
              </div>
            )}
          </div>

          {/* Pure Carbon Film Plate */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            className="relative aspect-square w-full overflow-hidden rounded-[26px] border transition-all"
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

            {/* Tangible Frosted Glass Split-Compare Grip Handle */}
            {splitCompare && !holdingSpace && (
              <div
                aria-hidden="true"
                className="pointer-events-none absolute top-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-teal-400/70 bg-black/80 text-teal-300 shadow-xl backdrop-blur-md"
                style={{ left: `${splitPos * 100}%` }}
              >
                <MoveHorizontal className="h-4 w-4" />
              </div>
            )}

            {/* Top Left DICOM Film Metadata */}
            <div className="pointer-events-none absolute top-4 left-4 flex flex-wrap items-center gap-2">
              <span className="rounded-full border border-white/15 bg-black/78 px-3 py-1 font-mono-tech text-[10px] tracking-[0.14em] text-slate-200 uppercase backdrop-blur-md">
                {activeStudyName}
              </span>
              {decision?.projection_view && (
                <span className="rounded-full border border-white/12 bg-black/72 px-2.5 py-1 font-mono-tech text-[10px] text-slate-300 backdrop-blur-md">
                  {decision.projection_view.replace(/_/g, " ")}
                </span>
              )}
              {holdingSpace && (
                <span className="rounded-full border border-teal-400/60 bg-black/88 px-2.5 py-1 font-mono-tech text-[10px] text-teal-400">
                  RAW FILM (SPACE)
                </span>
              )}
            </div>

            {/* Top Right Active Localization Pill */}
            {activePathology && (
              <div className="pointer-events-none absolute top-4 right-4 flex items-center gap-2 rounded-full border border-white/15 bg-black/80 px-3.5 py-1.5 backdrop-blur-md">
                <span className="h-1.5 w-1.5 rounded-full bg-teal-400" />
                <span className="font-mono-tech text-[10.5px] font-medium text-white">
                  {formatPathology(activePathology)}
                </span>
                {activeHeatmap?.anatomical_zone && (
                  <span className="hidden border-l border-white/15 pl-2 font-mono-tech text-[10px] text-slate-300 sm:inline">
                    {activeHeatmap.anatomical_zone}
                  </span>
                )}
              </div>
            )}

            {/* Radiographic R / L Orientation Markers */}
            <span className="pointer-events-none absolute top-14 left-4 font-mono-tech text-[10px] font-semibold text-white/40">
              R
            </span>
            <span className="pointer-events-none absolute top-14 right-4 font-mono-tech text-[10px] font-semibold text-white/40">
              L
            </span>

            {/* Bottom Live Optical Telemetry Bar (Coordinates & Zoom) */}
            <div className="pointer-events-none absolute right-4 bottom-3.5 left-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                {splitCompare && !holdingSpace ? (
                  <span className="rounded-full border border-white/12 bg-black/78 px-2.5 py-1 font-mono-tech text-[9.5px] tracking-[0.14em] text-slate-300 uppercase">
                    RAW FILM
                  </span>
                ) : (
                  pointer.inside && (
                    <span className="rounded-full border border-white/12 bg-black/75 px-2.5 py-1 font-mono-tech text-[9.5px] text-slate-300 backdrop-blur-md">
                      X {pointer.nx.toFixed(2)} · Y {pointer.ny.toFixed(2)} · {zoom.toFixed(1)}×
                    </span>
                  )
                )}
              </div>

              <div className="flex items-center gap-2">
                {splitCompare && !holdingSpace && (
                  <span className="rounded-full border border-teal-400/40 bg-black/78 px-2.5 py-1 font-mono-tech text-[9.5px] tracking-[0.14em] text-teal-400 uppercase">
                    AI OVERLAY
                  </span>
                )}
                {heatmapLoading && !isAnalyzing && (
                  <span className="rounded-full border border-white/15 bg-black/82 px-3 py-1 font-mono-tech text-[10px] text-slate-200 backdrop-blur-md">
                    Computing Grad-CAM++...
                  </span>
                )}
              </div>
            </div>

            {caliperMode && (
              <div className="pointer-events-none absolute bottom-12 left-1/2 -translate-x-1/2 rounded-full border border-teal-400/45 bg-black/85 px-3.5 py-1.5 font-mono-tech text-[10px] text-teal-400 backdrop-blur-md">
                Click &amp; drag across anatomy to measure (mm)
              </div>
            )}

            {isDragOver && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/82 backdrop-blur-sm">
                <Upload className="h-8 w-8 text-teal-400" />
                <div className="font-display text-lg font-semibold text-white">
                  Drop Chest Radiograph or DICOM
                </div>
              </div>
            )}

            {isAnalyzing && (
              <div className="absolute inset-0 overflow-hidden bg-black/55 backdrop-blur-[2px]">
                <div
                  className="scan-beam h-24 w-full"
                  style={{
                    background:
                      "linear-gradient(180deg, transparent, rgba(20, 184, 166, 0.3), transparent)",
                  }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="flex items-center gap-3 rounded-full border border-white/15 bg-black/88 px-5 py-2.5 font-mono-tech text-xs text-white shadow-2xl">
                    <span className="h-2 w-2 rounded-full bg-teal-400 clinical-led" />
                    <span>Running 3-Backbone Ensemble + Gemini Decision 1...</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Bottom Window/Level + Colormap + Caliper Bar */}
          <div className="atelier-card flex flex-wrap items-center justify-between gap-3 rounded-2xl px-4 py-2.5">
            <div className="flex flex-wrap items-center gap-1">
              {WINDOW_PRESETS.map((wp) => {
                const active = windowPreset === wp.id;
                return (
                  <button
                    key={wp.id}
                    type="button"
                    onClick={() => setWindowPreset(wp.id)}
                    className="cursor-pointer rounded-full px-3 py-1 text-xs font-medium transition-colors"
                    style={{
                      backgroundColor: active ? "var(--text-primary)" : "transparent",
                      color: active ? "var(--bg-primary)" : "var(--text-secondary)",
                    }}
                  >
                    {wp.label}
                  </button>
                );
              })}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <div
                className="flex items-center gap-1.5 rounded-full border px-2.5 py-1"
                style={{ borderColor: "var(--border-subtle)" }}
              >
                <span
                  className="font-mono-tech text-[9.5px] tracking-[0.12em] uppercase"
                  style={{ color: "var(--text-muted)" }}
                >
                  Map
                </span>
                <select
                  aria-label="Heatmap Colormap"
                  value={colormap}
                  onChange={(e) => onColormapChange(e.target.value as ColormapName)}
                  className="cursor-pointer bg-transparent font-mono-tech text-[11px] focus:outline-none"
                  style={{ color: "var(--text-primary)" }}
                >
                  {COLORMAP_OPTIONS.map((c) => (
                    <option
                      key={c.id}
                      value={c.id}
                      style={{
                        backgroundColor: "var(--bg-elevated)",
                        color: "var(--text-primary)",
                      }}
                    >
                      {c.label}
                    </option>
                  ))}
                </select>
              </div>

              <button
                type="button"
                onClick={() => {
                  setCaliperMode((v) => !v);
                  setLoupeActive(false);
                }}
                className="flex cursor-pointer items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors"
                style={{
                  borderColor: caliperMode ? "var(--accent-clinical)" : "var(--border-subtle)",
                  backgroundColor: caliperMode ? "var(--accent-clinical-dim)" : "transparent",
                  color: caliperMode ? "var(--text-primary)" : "var(--text-secondary)",
                }}
              >
                <Ruler className="h-3.5 w-3.5" />
                <span>Caliper</span>
              </button>

              {calipers.length > 0 && (
                <button
                  type="button"
                  onClick={() => setCalipers([])}
                  title="Clear caliper measurements"
                  className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-full border transition-opacity hover:opacity-80"
                  style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              )}

              <button
                type="button"
                onClick={resetView}
                title="Reset viewer"
                className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-full border transition-opacity hover:opacity-80"
                style={{ borderColor: "var(--border-subtle)", color: "var(--text-secondary)" }}
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          <div
            className="flex flex-wrap items-center justify-between px-2 font-mono-tech text-[10px]"
            style={{ color: "var(--text-muted)" }}
          >
            <span>Hold Space = Peek raw radiograph · Ctrl + Scroll = Zoom</span>
            <span>Ctrl + V = Paste radiograph from clipboard</span>
          </div>
        </div>

        {/* RIGHT 5 COLS: Crystal-Clear Clinical Verdict Panel */}
        <div className="lg:col-span-5 space-y-5">
          <SpotlightCard strong className="rounded-[26px] p-6 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span
                className="inline-flex items-center gap-2 rounded-full border px-3 py-1 font-mono-tech text-[10px] font-medium tracking-[0.14em] uppercase"
                style={{
                  borderColor:
                    isCritical || leadPositive ? "var(--accent-alert)" : "var(--accent-clinical)",
                  backgroundColor:
                    isCritical || leadPositive
                      ? "var(--accent-alert-dim)"
                      : "var(--accent-clinical-dim)",
                  color:
                    isCritical || leadPositive ? "var(--accent-alert)" : "var(--accent-clinical)",
                }}
              >
                <span className="relative flex h-2 w-2 items-center justify-center">
                  {(isCritical || leadPositive) && (
                    <span
                      className="clinical-ripple absolute inset-0 rounded-full"
                      style={{ backgroundColor: "var(--accent-alert)" }}
                    />
                  )}
                  <span
                    className="relative h-1.5 w-1.5 rounded-full"
                    style={{
                      backgroundColor:
                        isCritical || leadPositive
                          ? "var(--accent-alert)"
                          : "var(--accent-clinical)",
                    }}
                  />
                </span>
                {triageLevel}
              </span>

              {decision?.active && (
                <span
                  className="inline-flex items-center gap-1.5 font-mono-tech text-[10px]"
                  style={{ color: "var(--text-secondary)" }}
                >
                  <Sparkles className="h-3 w-3" style={{ color: "var(--accent-clinical)" }} />
                  <span>
                    Decision 1 · Q{" "}
                    {decision.image_quality_score != null
                      ? `${decision.image_quality_score.toFixed(1)}/5`
                      : "OK"}
                  </span>
                </span>
              )}
            </div>

            <div className="space-y-1.5">
              <div
                className="font-mono-tech text-[10px] tracking-[0.16em] uppercase"
                style={{ color: "var(--text-muted)" }}
              >
                PRIMARY DIAGNOSTIC VERDICT
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <h2 className="font-display text-2xl font-bold sm:text-3xl">
                  {leadPositive
                    ? formatPathology(leadPositive.name)
                    : predictionData
                    ? "No Acute Findings"
                    : "Awaiting Study"}
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
              <p className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                {leadPositive
                  ? PATHOLOGY_CLINICAL_NOTE[leadPositive.name]?.plain ||
                    "Localized thoracic abnormality detected above operating cutoff."
                  : predictionData
                  ? "All 14 evaluated thoracic pathologies fall below the active clinical decision threshold."
                  : "Upload a frontal chest radiograph or choose a benchmark case above."}
              </p>
            </div>

            {/* Precision Calibrated CTR Gauge with Micro-Ticks */}
            <div
              className="rounded-2xl border p-4 space-y-2.5"
              style={{
                borderColor: "var(--border-subtle)",
                backgroundColor: "var(--bg-secondary)",
              }}
            >
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-xs font-semibold">
                  <HeartPulse className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
                  <span>Cardiothoracic Ratio (CTR)</span>
                </span>
                <div className="flex items-center gap-2 font-mono-tech text-xs">
                  <AnimatedNumber
                    value={ctr}
                    decimals={3}
                    className="font-semibold"
                    style={{
                      color:
                        ctr != null && ctr > 0.5
                          ? "var(--accent-alert)"
                          : "var(--accent-clinical)",
                    }}
                  />
                  <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                    {anatomy?.ctr_status || "Pending"}
                  </span>
                </div>
              </div>

              <div
                className="relative h-2 w-full rounded-full"
                style={{ backgroundColor: "var(--border-subtle)" }}
              >
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${ctrPct}%` }}
                  transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
                  className="h-full rounded-full"
                  style={{
                    backgroundColor:
                      ctr != null && ctr > 0.5 ? "var(--accent-alert)" : "var(--accent-clinical)",
                  }}
                />
                {/* Minor scale ticks */}
                {[25, 50, 75].map((pct) => (
                  <div
                    key={pct}
                    className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
                    style={{
                      left: `${pct}%`,
                      width: pct === 50 ? 2 : 1,
                      height: pct === 50 ? 14 : 8,
                      backgroundColor: pct === 50 ? "var(--text-primary)" : "var(--border-strong)",
                    }}
                  />
                ))}
              </div>

              <div
                className="flex items-center justify-between font-mono-tech text-[10px]"
                style={{ color: "var(--text-muted)" }}
              >
                <span>0.30</span>
                <span>0.50 Cutoff</span>
                <span>
                  L {anatomy?.left_lung_pct != null ? `${anatomy.left_lung_pct.toFixed(1)}%` : "—"} ·
                  R{" "}
                  {anatomy?.right_lung_pct != null ? `${anatomy.right_lung_pct.toFixed(1)}%` : "—"}
                </span>
              </div>
            </div>

            {/* Direct Page Navigation Buttons */}
            <div className="grid grid-cols-2 gap-2.5">
              <button
                type="button"
                onClick={() => onNavigatePage("consensus")}
                className="group flex cursor-pointer items-center justify-between rounded-xl border px-3.5 py-2.5 text-xs font-semibold transition-colors"
                style={{
                  borderColor: "var(--border-subtle)",
                  backgroundColor: "var(--bg-secondary)",
                }}
              >
                <span>Consensus &amp; CTR</span>
                <ArrowRight
                  className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5"
                  style={{ color: "var(--accent-clinical)" }}
                />
              </button>
              <button
                type="button"
                onClick={() => onNavigatePage("report")}
                className="group flex cursor-pointer items-center justify-between rounded-xl border px-3.5 py-2.5 text-xs font-semibold transition-colors"
                style={{
                  borderColor: "var(--border-subtle)",
                  backgroundColor: "var(--bg-secondary)",
                }}
              >
                <span>ACR Report &amp; AI</span>
                <ArrowRight
                  className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5"
                  style={{ color: "var(--accent-clinical)" }}
                />
              </button>
            </div>
          </SpotlightCard>

          {/* Ranked Pathology Confidence Card */}
          <SpotlightCard className="rounded-[26px] p-6 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="flex items-center gap-2">
                  <Activity className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
                  <h3 className="font-display text-sm font-bold">Pathology Confidence</h3>
                </div>
                <p className="mt-0.5 text-[11px]" style={{ color: "var(--text-muted)" }}>
                  Select any condition to render its Grad-CAM++ map.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <SlidersHorizontal className="h-3 w-3" style={{ color: "var(--text-muted)" }} />
                <span
                  className="font-mono-tech text-[10px]"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Cutoff {Math.round(threshold * 100)}%
                </span>
                <input
                  type="range"
                  min={0.25}
                  max={0.75}
                  step={0.05}
                  value={threshold}
                  onChange={(e) => onThresholdChange(Number(e.target.value))}
                  aria-label="Diagnostic cutoff threshold"
                  className="clinical-slider w-16"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <AnimatePresence initial={false}>
                {visiblePredictions.map((pred) => {
                  const isPos = pred.score >= threshold;
                  const isSelected = activePathology === pred.name;
                  const note = PATHOLOGY_CLINICAL_NOTE[pred.name];

                  return (
                    <motion.button
                      layout
                      key={pred.name}
                      type="button"
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.22 }}
                      onClick={() => onSelectPathology(pred.name)}
                      className="group w-full cursor-pointer rounded-2xl border px-3.5 py-2.5 text-left transition-colors"
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
                                : isSelected
                                ? "var(--accent-clinical)"
                                : "var(--border-strong)",
                            }}
                          />
                          <span className="truncate text-sm font-medium">
                            {formatPathology(pred.name)}
                          </span>
                          {isPos && note?.guideline && (
                            <span
                              className="hidden rounded-full border px-2 py-0.5 font-mono-tech text-[9px] sm:inline"
                              style={{
                                borderColor: "var(--border-subtle)",
                                color: "var(--text-secondary)",
                              }}
                            >
                              {note.guideline}
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
                        <div
                          className="absolute top-1/2 h-2.5 w-[1px] -translate-y-1/2"
                          style={{
                            left: `${threshold * 100}%`,
                            backgroundColor: "var(--text-muted)",
                          }}
                        />
                      </div>

                      {(isSelected || isPos) && note && (
                        <div
                          className="mt-1.5 flex items-center justify-between text-[11px]"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          <span className="truncate">{note.plain}</span>
                          {pred.anatomical_zone && (
                            <span
                              className="ml-2 shrink-0 font-mono-tech text-[9.5px]"
                              style={{ color: "var(--accent-clinical)" }}
                            >
                              {pred.anatomical_zone}
                            </span>
                          )}
                        </div>
                      )}
                    </motion.button>
                  );
                })}
              </AnimatePresence>
            </div>

            {predictions.length > 6 && (
              <button
                type="button"
                onClick={() => setShowAllPathologies((v) => !v)}
                className="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-xl border py-2 font-mono-tech text-[10px] tracking-[0.14em] uppercase transition-colors"
                style={{
                  borderColor: "var(--border-subtle)",
                  color: "var(--text-secondary)",
                }}
              >
                {showAllPathologies ? (
                  <>
                    <span>Show Top 6 Only</span>
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

            {/* Quiet Patient Profile Footer */}
            <div
              className="flex items-center justify-between gap-3 border-t pt-4"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              <div className="flex min-w-0 items-center gap-2.5">
                <div
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border"
                  style={{
                    borderColor: "var(--border-subtle)",
                    backgroundColor: "var(--bg-secondary)",
                  }}
                >
                  <UserRound className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
                </div>
                <div className="min-w-0">
                  <div className="truncate text-xs font-semibold">
                    {patient.age ? `${patient.age}y ` : ""}
                    {patient.sex} · {patient.setting}
                  </div>
                  <div className="truncate text-[11px]" style={{ color: "var(--text-muted)" }}>
                    {patient.indication || "No clinical history specified"}
                  </div>
                </div>
              </div>
              <button
                type="button"
                onClick={onOpenPatientDrawer}
                className="shrink-0 cursor-pointer rounded-full border px-3 py-1.5 text-xs font-medium transition-colors"
                style={{
                  borderColor: "var(--border-strong)",
                  color: "var(--text-primary)",
                }}
              >
                Edit Profile
              </button>
            </div>
          </SpotlightCard>
        </div>
      </div>
    </div>
  );
};
