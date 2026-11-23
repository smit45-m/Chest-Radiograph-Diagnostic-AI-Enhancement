import React from "react";
import { motion } from "framer-motion";
import { Command, HeartPulse, Moon, Sparkles, Sun, Upload } from "lucide-react";
import { NAV_PAGES, formatPathology, type PageId } from "../data/clinicalCatalog";
import type { AIConfigStatus, HealthStatus, PredictionResponse, ThemeMode } from "../types/clinical";

interface TopHeaderProps {
  activePage: PageId;
  sidebarCollapsed: boolean;
  theme: ThemeMode;
  onToggleTheme: () => void;
  health: HealthStatus | null;
  aiConfig: AIConfigStatus | null;
  activeStudyName: string;
  imageUrl: string | null;
  predictionData: PredictionResponse | null;
  threshold: number;
  isAnalyzing: boolean;
  onSelectPage: (page: PageId) => void;
  onOpenCommand: () => void;
  onOpenGemini: () => void;
  onUploadClick: () => void;
}

export const TopHeader: React.FC<TopHeaderProps> = ({
  activePage,
  sidebarCollapsed,
  theme,
  onToggleTheme,
  health,
  aiConfig,
  activeStudyName,
  imageUrl,
  predictionData,
  threshold,
  isAnalyzing,
  onSelectPage,
  onOpenCommand,
  onOpenGemini,
  onUploadClick,
}) => {
  const pageMeta = NAV_PAGES.find((p) => p.id === activePage) || NAV_PAGES[0];
  const geminiReady = Boolean(aiConfig?.gemini_configured || health?.gemini_configured);
  const topPos = predictionData?.predictions?.find((p) => p.score >= threshold);
  const lead = topPos || predictionData?.predictions?.[0] || null;
  const ctr = predictionData?.anatomy?.ctr ?? null;

  return (
    <header
      role="banner"
      className="no-print sticky top-0 z-30 flex h-16 items-center justify-between gap-4 border-b px-5 transition-colors md:px-8"
      style={{
        backgroundColor: "var(--bg-glass)",
        borderColor: "var(--border-subtle)",
        backdropFilter: "blur(22px) saturate(165%)",
        WebkitBackdropFilter: "blur(22px) saturate(165%)",
      }}
    >
      {/* Left: Page Breadcrumb + Mobile Brand */}
      <div className="flex min-w-0 items-center gap-3">
        <button
          type="button"
          onClick={() => onSelectPage("overview")}
          className="font-display text-sm font-bold tracking-[0.06em] lg:hidden"
        >
          CXR·DIAGNOSTIC
        </button>

        <div className="hidden items-center gap-2.5 lg:flex">
          <span
            className="font-mono-tech text-[11px] font-medium"
            style={{ color: "var(--accent-clinical)" }}
          >
            {pageMeta.num}
          </span>
          <span style={{ color: "var(--border-strong)" }}>/</span>
          <span className="font-display text-sm font-semibold">{pageMeta.label}</span>
          <span
            className="hidden rounded-full border px-2.5 py-0.5 font-mono-tech text-[10px] xl:inline"
            style={{
              borderColor: "var(--border-subtle)",
              color: "var(--text-muted)",
            }}
          >
            {pageMeta.shortDesc}
          </span>
        </div>

        {isAnalyzing && (
          <div
            className="flex items-center gap-2 rounded-full border px-3 py-1 font-mono-tech text-[10px]"
            style={{
              borderColor: "var(--accent-clinical)",
              backgroundColor: "var(--accent-clinical-dim)",
              color: "var(--accent-clinical)",
            }}
          >
            <span
              className="h-1.5 w-1.5 rounded-full clinical-led"
              style={{ backgroundColor: "var(--accent-clinical)" }}
            />
            <span>Inferring on RTX CUDA...</span>
          </div>
        )}
      </div>

      {/* Center: Persistent Active Specimen Telemetry Pill (Visible across all 5 pages!) */}
      {predictionData && !isAnalyzing && (
        <button
          type="button"
          onClick={() => onSelectPage("studio")}
          title="Jump to active radiograph in Scan Studio"
          className={`hidden cursor-pointer items-center gap-3 rounded-full border py-1 pr-3.5 pl-1.5 transition-all hover:opacity-95 ${
            sidebarCollapsed ? "md:flex" : "xl:flex"
          }`}
          style={{
            borderColor: "var(--border-strong)",
            backgroundColor: "var(--bg-elevated)",
          }}
        >
          {imageUrl && (
            <img
              src={imageUrl}
              alt="Active study thumbnail"
              className="h-7 w-7 rounded-full border border-white/10 bg-black object-cover"
            />
          )}
          <div className="flex items-center gap-2.5 text-xs">
            <span className="max-w-[140px] truncate font-medium">{activeStudyName}</span>
            <span style={{ color: "var(--border-strong)" }}>·</span>
            <span
              className="font-mono-tech text-[11px] font-semibold"
              style={{ color: topPos ? "var(--accent-alert)" : "var(--accent-clinical)" }}
            >
              {topPos
                ? formatPathology(topPos.name).toLowerCase() === activeStudyName.toLowerCase()
                  ? `${(topPos.score * 100).toFixed(1)}%`
                  : `${formatPathology(topPos.name)} ${(topPos.score * 100).toFixed(1)}%`
                : lead
                ? `Clear (${(lead.score * 100).toFixed(0)}%)`
                : "Ready"}
            </span>
            {ctr != null && (
              <>
                <span style={{ color: "var(--border-strong)" }}>·</span>
                <span
                  className="flex items-center gap-1 font-mono-tech text-[10.5px]"
                  style={{ color: "var(--text-secondary)" }}
                >
                  <HeartPulse className="h-3 w-3" style={{ color: "var(--accent-clinical)" }} />
                  CTR {ctr.toFixed(3)}
                </span>
              </>
            )}
          </div>
        </button>
      )}

      {/* Right: Instrument Controls */}
      <div className="flex shrink-0 items-center gap-2">
        <button
          type="button"
          onClick={onOpenGemini}
          className="flex cursor-pointer items-center gap-1.5 rounded-full border px-3 py-1.5 font-mono-tech text-[10px] tracking-[0.1em] uppercase whitespace-nowrap transition-opacity hover:opacity-90"
          style={{
            borderColor: geminiReady ? "var(--accent-clinical)" : "var(--border-subtle)",
            backgroundColor: geminiReady ? "var(--accent-clinical-dim)" : "var(--bg-elevated)",
            color: "var(--text-primary)",
          }}
          title="Configure Gemini Decision 1 & Interactions API"
        >
          <Sparkles
            className="h-3 w-3"
            style={{ color: geminiReady ? "var(--accent-clinical)" : "var(--text-muted)" }}
          />
          <span className="hidden sm:inline">{geminiReady ? "Gemini Live" : "Gemini AI"}</span>
        </button>

        <button
          type="button"
          onClick={onToggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "Clinical Alabaster light" : "Radiology Darkroom dark"} theme`}
          title={theme === "dark" ? "Switch to Clinical Alabaster Light" : "Switch to Radiology Darkroom"}
          className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full border transition-colors hover:opacity-85"
          style={{
            borderColor: "var(--border-subtle)",
            backgroundColor: "var(--bg-elevated)",
            color: "var(--text-primary)",
          }}
        >
          <motion.span
            key={theme}
            initial={{ rotate: -45, opacity: 0, scale: 0.7 }}
            animate={{ rotate: 0, opacity: 1, scale: 1 }}
            transition={{ duration: 0.25 }}
            className="flex items-center justify-center"
          >
            {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
          </motion.span>
        </button>

        <button
          type="button"
          onClick={onOpenCommand}
          aria-label="Open Command Palette"
          title="Command Palette (Ctrl+K)"
          className="hidden h-8 cursor-pointer items-center gap-1 rounded-full border px-2.5 font-mono-tech text-[10px] transition-opacity hover:opacity-85 sm:flex"
          style={{
            borderColor: "var(--border-subtle)",
            backgroundColor: "var(--bg-elevated)",
            color: "var(--text-secondary)",
          }}
        >
          <Command className="h-3 w-3" />
          <span>K</span>
        </button>

        <button
          type="button"
          onClick={onUploadClick}
          className="flex h-8 cursor-pointer items-center gap-1.5 rounded-full px-3.5 text-xs font-semibold whitespace-nowrap transition-transform active:scale-95"
          style={{
            backgroundColor: "var(--accent-clinical)",
            color: "#ffffff",
            boxShadow: "0 4px 18px -4px var(--accent-clinical-glow)",
          }}
        >
          <Upload className="h-3.5 w-3.5" />
          <span>Upload Scan</span>
        </button>
      </div>
    </header>
  );
};
