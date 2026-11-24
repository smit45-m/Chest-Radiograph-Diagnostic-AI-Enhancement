import React from "react";
import { motion } from "framer-motion";
import { HeartPulse, Moon, Search, Sparkles, Sun, Upload } from "lucide-react";
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
      className="no-print sticky top-0 z-30 flex h-14 items-center justify-between gap-4 border-b px-5 transition-colors md:px-8"
      style={{
        backgroundColor: "var(--bg-glass)",
        borderColor: "var(--border-subtle)",
        backdropFilter: "blur(22px) saturate(165%)",
        WebkitBackdropFilter: "blur(22px) saturate(165%)",
      }}
    >
      {/* Subtle Top Indeterminate Progress Bar while Analyzing */}
      {isAnalyzing && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute right-0 bottom-0 left-0 h-[2px] overflow-hidden"
        >
          <div
            className="top-progress-bar h-full w-1/3 rounded-full"
            style={{
              background:
                "linear-gradient(90deg, transparent, var(--accent-clinical), transparent)",
            }}
          />
        </div>
      )}

      {/* Left: Clean Page Title */}
      <div className="flex min-w-0 items-center gap-3">
        <button
          type="button"
          onClick={() => onSelectPage("overview")}
          className="font-display text-sm font-bold tracking-[0.02em] lg:hidden"
        >
          CXR·Diagnostic
        </button>

        <div className="hidden items-center gap-2 lg:flex">
          <span className="font-display text-sm font-semibold">{pageMeta.label}</span>
          {isAnalyzing && (
            <span
              className="ml-1 flex items-center gap-1.5 font-mono-tech text-[11px]"
              style={{ color: "var(--accent-clinical)" }}
            >
              <span
                className="h-1.5 w-1.5 rounded-full clinical-led"
                style={{ backgroundColor: "var(--accent-clinical)" }}
              />
              Analyzing...
            </span>
          )}
        </div>
      </div>

      {/* Center: Clean Active Study Status Pill */}
      {predictionData && !isAnalyzing && (
        <button
          type="button"
          onClick={() => onSelectPage("studio")}
          title="Open active X-ray in Scan Studio"
          className={`hidden cursor-pointer items-center gap-2.5 rounded-full border py-1 pr-3.5 pl-1.5 transition-all hover:border-teal-500/40 ${
            sidebarCollapsed ? "md:flex" : "lg:flex"
          }`}
          style={{
            borderColor: "var(--border-subtle)",
            backgroundColor: "var(--bg-elevated)",
          }}
        >
          {imageUrl && (
            <img
              src={imageUrl}
              alt="Active study thumbnail"
              className="h-6 w-6 rounded-full border border-white/10 bg-black object-cover"
            />
          )}
          <div className="flex items-center gap-2 text-xs">
            <span className="max-w-[130px] truncate font-medium">{activeStudyName}</span>
            <span style={{ color: "var(--border-strong)" }}>·</span>
            <span
              className="font-mono-tech text-[11px] font-semibold"
              style={{ color: topPos ? "var(--accent-alert)" : "var(--accent-clinical)" }}
            >
              {topPos
                ? formatPathology(topPos.name).toLowerCase() === activeStudyName.toLowerCase()
                  ? `${(topPos.score * 100).toFixed(0)}%`
                  : `${formatPathology(topPos.name)} ${(topPos.score * 100).toFixed(0)}%`
                : lead
                ? "Clear"
                : "Ready"}
            </span>
            {ctr != null && (
              <>
                <span className="hidden sm:inline" style={{ color: "var(--border-strong)" }}>
                  ·
                </span>
                <span
                  className="hidden items-center gap-1 font-mono-tech text-[10.5px] sm:flex"
                  style={{ color: "var(--text-secondary)" }}
                >
                  <HeartPulse className="h-3 w-3" style={{ color: "var(--accent-clinical)" }} />
                  {ctr.toFixed(2)}
                </span>
              </>
            )}
          </div>
        </button>
      )}

      {/* Right: 3 Clean Utility Actions (No Clutter) */}
      <div className="flex shrink-0 items-center gap-2">
        {/* Mobile-only Gemini button since sidebar is hidden on mobile */}
        <button
          type="button"
          onClick={onOpenGemini}
          aria-label="Configure Gemini AI"
          title="AI Engine Settings"
          className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full border lg:hidden"
          style={{
            borderColor: geminiReady ? "var(--accent-clinical)" : "var(--border-subtle)",
            backgroundColor: "var(--bg-elevated)",
            color: geminiReady ? "var(--accent-clinical)" : "var(--text-muted)",
          }}
        >
          <Sparkles className="h-3.5 w-3.5" />
        </button>

        <button
          type="button"
          onClick={onOpenCommand}
          aria-label="Search commands"
          title="Quick Search (Ctrl+K)"
          className="hidden h-8 cursor-pointer items-center gap-2 rounded-full border px-3 text-xs transition-colors hover:border-teal-500/40 sm:flex"
          style={{
            borderColor: "var(--border-subtle)",
            backgroundColor: "var(--bg-elevated)",
            color: "var(--text-muted)",
          }}
        >
          <Search className="h-3.5 w-3.5" />
          <span className="hidden md:inline">Search</span>
          <kbd className="font-mono-tech text-[10px] opacity-75">⌘K</kbd>
        </button>

        <button
          type="button"
          onClick={onToggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
          className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full border transition-colors hover:border-teal-500/40"
          style={{
            borderColor: "var(--border-subtle)",
            backgroundColor: "var(--bg-elevated)",
            color: "var(--text-secondary)",
          }}
        >
          <motion.span
            key={theme}
            initial={{ rotate: -35, opacity: 0, scale: 0.75 }}
            animate={{ rotate: 0, opacity: 1, scale: 1 }}
            transition={{ duration: 0.22 }}
            className="flex items-center justify-center"
          >
            {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
          </motion.span>
        </button>

        <button
          type="button"
          onClick={onUploadClick}
          className="flex h-8 cursor-pointer items-center gap-1.5 rounded-full px-3.5 text-xs font-semibold whitespace-nowrap transition-all hover:brightness-110 active:scale-[0.97]"
          style={{
            backgroundColor: "var(--accent-clinical)",
            color: "#ffffff",
            boxShadow: "0 4px 16px -4px var(--accent-clinical-glow)",
          }}
        >
          <Upload className="h-3.5 w-3.5" />
          <span>Upload X-Ray</span>
        </button>
      </div>
    </header>
  );
};
