import React, { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowRight,
  Moon,
  Printer,
  Search,
  Sparkles,
  Sun,
  Upload,
  UserRound,
} from "lucide-react";
import {
  NAV_PAGES,
  cleanSampleTitle,
  formatPathology,
  type PageId,
} from "../data/clinicalCatalog";
import type { PathologyPrediction, SampleCase, ThemeMode } from "../types/clinical";

interface CommandPaletteProps {
  open: boolean;
  theme: ThemeMode;
  samples: SampleCase[];
  predictions: PathologyPrediction[];
  onClose: () => void;
  onNavigatePage: (page: PageId) => void;
  onSelectSample: (sample: SampleCase) => void;
  onSelectPathology: (name: string) => void;
  onToggleTheme: () => void;
  onOpenGemini: () => void;
  onOpenPatient: () => void;
  onUploadClick: () => void;
  onPrintPdf: () => void;
}

interface CommandItem {
  id: string;
  group: string;
  label: string;
  hint?: string;
  action: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  open,
  theme,
  samples,
  predictions,
  onClose,
  onNavigatePage,
  onSelectSample,
  onSelectPathology,
  onToggleTheme,
  onOpenGemini,
  onOpenPatient,
  onUploadClick,
  onPrintPdf,
}) => {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (open) {
      setQuery("");
      setTimeout(() => inputRef.current?.focus(), 40);
    }
  }, [open]);

  const items = useMemo<CommandItem[]>(() => {
    const list: CommandItem[] = [
      ...NAV_PAGES.map((pg) => ({
        id: `pg-${pg.id}`,
        group: "Workspace Pages",
        label: `${pg.num} · ${pg.label}`,
        hint: `Key ${pg.keyHint}`,
        action: () => onNavigatePage(pg.id),
      })),
      {
        id: "act-upload",
        group: "Actions",
        label: "Upload Chest X-Ray or DICOM",
        hint: "File",
        action: onUploadClick,
      },
      {
        id: "act-theme",
        group: "Actions",
        label:
          theme === "dark"
            ? "Switch to Clinical Alabaster Light Theme"
            : "Switch to Radiology Darkroom Theme",
        hint: "Theme",
        action: onToggleTheme,
      },
      {
        id: "act-gemini",
        group: "Actions",
        label: "Configure Gemini Decision 1 & Interactions API",
        hint: "AI Engine",
        action: onOpenGemini,
      },
      {
        id: "act-patient",
        group: "Actions",
        label: "Edit Patient Clinical Profile",
        hint: "Context",
        action: onOpenPatient,
      },
      {
        id: "act-pdf",
        group: "Actions",
        label: "Print Hospital PDF Consultation Sheet",
        hint: "Export",
        action: onPrintPdf,
      },
      ...samples.map((s) => ({
        id: `sample-${s.id}`,
        group: "Verified Benchmark Studies",
        label: cleanSampleTitle(s.name),
        hint: s.primary_findings.map(formatPathology).join(", "),
        action: () => onSelectSample(s),
      })),
      ...predictions.slice(0, 8).map((p) => ({
        id: `path-${p.name}`,
        group: "Localize Pathology (Grad-CAM++)",
        label: formatPathology(p.name),
        hint: `${(p.score * 100).toFixed(1)}%`,
        action: () => onSelectPathology(p.name),
      })),
    ];

    if (!query.trim()) return list;
    const q = query.toLowerCase();
    return list.filter(
      (i) =>
        i.label.toLowerCase().includes(q) ||
        i.group.toLowerCase().includes(q) ||
        (i.hint && i.hint.toLowerCase().includes(q))
    );
  }, [
    query,
    theme,
    samples,
    predictions,
    onNavigatePage,
    onUploadClick,
    onToggleTheme,
    onOpenGemini,
    onOpenPatient,
    onPrintPdf,
    onSelectSample,
    onSelectPathology,
  ]);

  const run = (item: CommandItem) => {
    onClose();
    item.action();
  };

  return (
    <AnimatePresence>
      {open && (
        <div className="no-print fixed inset-0 z-50 flex items-start justify-center px-4 pt-24">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/65 backdrop-blur-sm"
          />

          <motion.div
            role="dialog"
            aria-label="Command Palette"
            initial={{ opacity: 0, y: -12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -12, scale: 0.98 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="atelier-card-strong relative z-10 w-full max-w-xl overflow-hidden rounded-3xl"
          >
            <div
              className="flex items-center gap-3 border-b px-5 py-4"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              <Search className="h-4 w-4 shrink-0" style={{ color: "var(--accent-clinical)" }} />
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Escape") onClose();
                  if (e.key === "Enter" && items[0]) run(items[0]);
                }}
                placeholder="Jump to page, benchmark study, pathology, or action..."
                className="w-full bg-transparent text-sm focus:outline-none"
              />
              <span
                className="rounded-md border px-2 py-0.5 font-mono-tech text-[10px]"
                style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
              >
                ESC
              </span>
            </div>

            <div className="max-h-96 overflow-y-auto p-2.5">
              {items.length === 0 ? (
                <div className="py-10 text-center text-xs" style={{ color: "var(--text-muted)" }}>
                  No matching clinical commands.
                </div>
              ) : (
                items.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => run(item)}
                    className="flex w-full cursor-pointer items-center justify-between rounded-2xl px-3.5 py-2.5 text-left text-xs transition-colors hover:bg-teal-500/[0.06]"
                  >
                    <div className="flex items-center gap-2.5">
                      {item.id === "act-theme" ? (
                        theme === "dark" ? (
                          <Sun className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
                        ) : (
                          <Moon className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
                        )
                      ) : item.id === "act-gemini" ? (
                        <Sparkles className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
                      ) : item.id === "act-upload" ? (
                        <Upload className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
                      ) : item.id === "act-patient" ? (
                        <UserRound className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
                      ) : item.id === "act-pdf" ? (
                        <Printer className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
                      ) : (
                        <ArrowRight className="h-3.5 w-3.5" style={{ color: "var(--text-muted)" }} />
                      )}
                      <div>
                        <div className="font-medium">{item.label}</div>
                        <div
                          className="font-mono-tech text-[9.5px]"
                          style={{ color: "var(--text-muted)" }}
                        >
                          {item.group}
                        </div>
                      </div>
                    </div>
                    {item.hint && (
                      <span
                        className="font-mono-tech text-[10px]"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {item.hint}
                      </span>
                    )}
                  </button>
                ))
              )}
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
