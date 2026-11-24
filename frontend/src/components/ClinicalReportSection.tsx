import React, { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  ChevronDown,
  ChevronUp,
  Copy,
  Database,
  FileJson,
  FileText,
  Mic,
  Printer,
  RefreshCw,
  RotateCcw,
  Save,
  Send,
  ShieldCheck,
  Square,
  Stethoscope,
  Volume2,
} from "lucide-react";
import { formatTriageLabel } from "../data/clinicalCatalog";
import type { CopilotMessage, PredictionResponse, ReportData } from "../types/clinical";
import { SpotlightCard } from "./SpotlightCard";

interface ClinicalReportSectionProps {
  activeStudyName: string;
  activeStudyRecordId: string | null;
  activeUserName: string | null;
  predictionData: PredictionResponse | null;
  report: ReportData | null;
  reportLoading: boolean;
  copilotMessages: CopilotMessage[];
  copilotLoading: boolean;
  physicianName: string;
  physicianCreds: string;
  physicianNotes: string;
  signedAt: string | null;
  onPhysicianNameChange: (v: string) => void;
  onPhysicianCredsChange: (v: string) => void;
  onPhysicianNotesChange: (v: string) => void;
  onSignReport: () => void;
  onSaveTranscript: (nextTranscript: string) => Promise<void>;
  onResynthesize: () => void;
  onSendCopilot: (question: string) => void;
  onClearCopilot: () => void;
  onExportFhir: () => void;
  onPrintPdf: () => void;
}

type ReportTab = "findings" | "guidelines" | "patient" | "signoff";

const COPILOT_SUGGESTIONS = [
  "Summarize the primary findings and urgency.",
  "Explain the CTR biometry and cardiac risk.",
  "What follow-up imaging or labs are recommended?",
  "Draft a clear patient-friendly explanation.",
];

const QUICK_FOLLOWUPS = [
  { label: "Explain CTR risk", q: "Explain the CTR biometry and cardiac risk." },
  { label: "Next steps?", q: "What follow-up imaging or labs are recommended?" },
  { label: "Patient summary", q: "Draft a clear patient-friendly explanation." },
];

/** Parses **bold**, *italic*, and `inline code` cleanly with zero raw markdown artifacts */
function renderInlineRichText(text: string): React.ReactNode {
  const cleaned = text.replace(/\$([^$]+)\$/g, "$1");
  const tokens = cleaned.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);
  return tokens.map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={idx} className="font-semibold" style={{ color: "var(--text-primary)" }}>
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return (
        <em key={idx} className="italic" style={{ color: "var(--text-primary)" }}>
          {part.slice(1, -1)}
        </em>
      );
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code
          key={idx}
          className="rounded px-1.5 py-0.5 font-mono-tech text-[11.5px]"
          style={{
            backgroundColor: "var(--bg-elevated)",
            color: "var(--accent-clinical)",
          }}
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    return <React.Fragment key={idx}>{part}</React.Fragment>;
  });
}

function renderClinicalMarkdown(text: string): React.ReactNode {
  const lines = text
    .split(/\r?\n/)
    .filter((l) => l.trim().length > 0)
    // Filter out noisy fallback setup tips so clinicians see clean medical guidance only
    .filter((l) => !/Add a .*GEMINI_API_KEY.*in AI Engine Settings/i.test(l));

  return (
    <div className="space-y-2">
      {lines.map((raw, idx) => {
        const trimmed = raw.trim();
        if (/^#{1,4}\s+/.test(trimmed)) {
          return (
            <div
              key={idx}
              className="pt-1 text-xs font-semibold tracking-wide"
              style={{ color: "var(--accent-clinical)" }}
            >
              {trimmed.replace(/^#{1,4}\s+/, "")}
            </div>
          );
        }
        if (/^[-*•]\s+/.test(trimmed)) {
          return (
            <div key={idx} className="flex items-start gap-2.5 text-[13px] leading-relaxed">
              <span
                className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full"
                style={{ backgroundColor: "var(--accent-clinical)" }}
              />
              <span style={{ color: "var(--text-secondary)" }}>
                {renderInlineRichText(trimmed.replace(/^[-*•]\s+/, ""))}
              </span>
            </div>
          );
        }
        return (
          <p
            key={idx}
            className="text-[13px] leading-relaxed"
            style={{ color: "var(--text-secondary)" }}
          >
            {renderInlineRichText(trimmed)}
          </p>
        );
      })}
    </div>
  );
}

export const ClinicalReportSection: React.FC<ClinicalReportSectionProps> = ({
  activeStudyName,
  activeStudyRecordId,
  activeUserName,
  predictionData,
  report,
  reportLoading,
  copilotMessages,
  copilotLoading,
  physicianName,
  physicianCreds,
  physicianNotes,
  signedAt,
  onPhysicianNameChange,
  onPhysicianCredsChange,
  onPhysicianNotesChange,
  onSignReport,
  onSaveTranscript,
  onResynthesize,
  onSendCopilot,
  onClearCopilot,
  onExportFhir,
  onPrintPdf,
}) => {
  const [tab, setTab] = useState<ReportTab>("findings");
  const [copied, setCopied] = useState(false);
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [editingTranscript, setEditingTranscript] = useState(false);
  const [transcriptDraft, setTranscriptDraft] = useState("");
  const [showCompartments, setShowCompartments] = useState(false);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  const liveTranscript = report?.dictation_transcript || report?.impression || "";

  useEffect(() => {
    setTranscriptDraft(liveTranscript);
    setEditingTranscript(false);
  }, [liveTranscript]);

  useEffect(() => {
    return () => {
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const toggleSpeakTranscript = () => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }
    if (!liveTranscript) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(liveTranscript);
    utter.rate = 0.98;
    utter.pitch = 1.0;
    utter.onend = () => setIsSpeaking(false);
    utter.onerror = () => setIsSpeaking(false);
    setIsSpeaking(true);
    window.speechSynthesis.speak(utter);
  };

  useEffect(() => {
    if (copilotMessages.length === 0 && !copilotLoading) return;
    const el = chatScrollRef.current;
    if (el) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  }, [copilotMessages.length, copilotLoading]);

  const handleCopy = async () => {
    if (!report?.full_text) return;
    try {
      await navigator.clipboard.writeText(report.full_text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // ignore
    }
  };

  const handleCopyMsg = async (id: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMsgId(id);
      setTimeout(() => setCopiedMsgId((prev) => (prev === id ? null : prev)), 1500);
    } catch {
      // ignore
    }
  };

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    const q = chatInput.trim();
    if (!q || copilotLoading) return;
    setChatInput("");
    onSendCopilot(q);
  };

  const compartments = report?.compartments || {};
  const compartmentEntries = Object.entries(compartments).filter(([, v]) => Boolean(v));
  const execSummary = report?.executive_summary || [];
  const differentials = report?.differential_diagnoses || [];
  const isCritical = Boolean(report?.critical_alert);

  return (
    <div className="mx-auto max-w-[1280px] px-5 py-6 pb-24 md:px-8 md:py-7 lg:pb-10 space-y-6">
      {/* Compact Header + Clean Action Row */}
      <div className="flex flex-col justify-between gap-4 xl:flex-row xl:items-center">
        <div>
          <h1 className="font-display text-2xl font-bold sm:text-3xl">
            Clinical Report &amp;{" "}
            <span
              className="font-serif-editorial font-normal"
              style={{ color: "var(--accent-clinical)" }}
            >
              Copilot
            </span>
          </h1>
          <p className="mt-0.5 text-xs" style={{ color: "var(--text-muted)" }}>
            Review AI dictation, listen aloud, export PDF/FHIR, or ask follow-up questions.
          </p>
        </div>

        <div className="flex shrink-0 flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={!predictionData || reportLoading}
            onClick={onResynthesize}
            className="flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-full border px-3.5 py-2 text-xs font-medium transition-colors hover:border-teal-500/40 disabled:opacity-50"
            style={{
              borderColor: "var(--border-subtle)",
              backgroundColor: "var(--bg-elevated)",
            }}
          >
            <RefreshCw className={`h-3.5 w-3.5 ${reportLoading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
          <button
            type="button"
            disabled={!report}
            onClick={handleCopy}
            className="flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-full border px-3.5 py-2 text-xs font-medium transition-colors hover:border-teal-500/40 disabled:opacity-50"
            style={{
              borderColor: "var(--border-subtle)",
              backgroundColor: "var(--bg-elevated)",
            }}
          >
            {copied ? (
              <Check className="h-3.5 w-3.5" style={{ color: "var(--accent-clinical)" }} />
            ) : (
              <Copy className="h-3.5 w-3.5" />
            )}
            <span>{copied ? "Copied" : "Copy"}</span>
          </button>
          <button
            type="button"
            disabled={!predictionData}
            onClick={onExportFhir}
            className="flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-full border px-3.5 py-2 text-xs font-medium transition-colors hover:border-teal-500/40 disabled:opacity-50"
            style={{
              borderColor: "var(--border-subtle)",
              backgroundColor: "var(--bg-elevated)",
            }}
          >
            <FileJson className="h-3.5 w-3.5" />
            <span>FHIR</span>
          </button>
          <button
            type="button"
            disabled={!predictionData}
            onClick={onPrintPdf}
            className="flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-full px-4 py-2 text-xs font-semibold transition-all hover:brightness-110 active:scale-[0.98] disabled:opacity-50"
            style={{
              backgroundColor: "var(--accent-clinical)",
              color: "#ffffff",
            }}
          >
            <Printer className="h-3.5 w-3.5" />
            <span>Print PDF</span>
          </button>
        </div>
      </div>

      {/* Main 2-Column Workspace */}
      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        {/* LEFT 7 COLS: Editorial ACR Document (No Nested Boxes-Inside-Boxes) */}
        <div className="lg:col-span-7">
          <SpotlightCard strong className="rounded-[24px] p-5 sm:p-7 space-y-6">
            {/* Document Meta Header */}
            <div
              className="flex flex-wrap items-center justify-between gap-3 border-b pb-4"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                  <h2 className="font-display text-base font-bold">
                    {activeStudyName} Report
                  </h2>
                </div>
                <div
                  className="flex flex-wrap items-center gap-2 text-[11px]"
                  style={{ color: "var(--text-muted)" }}
                >
                  <span>{signedAt ? "Signed Final" : "AI-Assisted Draft"}</span>
                  {activeStudyRecordId && (
                    <>
                      <span>·</span>
                      <span
                        className="inline-flex items-center gap-1"
                        style={{ color: "var(--accent-clinical)" }}
                      >
                        <Database className="h-3 w-3" />
                        Saved to {activeUserName || "Database"}
                      </span>
                    </>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-2">
                {report?.triage_level && (
                  <span
                    className="rounded-full px-2.5 py-1 text-[11px] font-semibold"
                    style={{
                      color: isCritical ? "var(--accent-alert)" : "var(--accent-clinical)",
                      backgroundColor: isCritical
                        ? "var(--accent-alert-dim)"
                        : "var(--accent-clinical-dim)",
                    }}
                  >
                    {formatTriageLabel(report.triage_level)}
                  </span>
                )}
              </div>
            </div>

            {/* Clean Segmented Tabs */}
            <div
              className="inline-flex flex-wrap gap-0.5 rounded-full p-1"
              style={{ backgroundColor: "var(--bg-secondary)" }}
            >
              {(
                [
                  { id: "findings", label: "Findings" },
                  { id: "guidelines", label: "Guidelines" },
                  { id: "patient", label: "Patient View" },
                  { id: "signoff", label: signedAt ? "Signed ✓" : "Sign-Off" },
                ] as const
              ).map((item) => {
                const active = tab === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setTab(item.id)}
                    className="relative z-10 cursor-pointer rounded-full px-3.5 py-1.5 text-xs font-medium transition-colors"
                    style={{
                      color: active ? "var(--text-primary)" : "var(--text-secondary)",
                    }}
                  >
                    {active && (
                      <motion.span
                        layoutId="report-tab-pill"
                        transition={{ type: "spring", stiffness: 420, damping: 34 }}
                        className="absolute inset-0 -z-10 rounded-full border"
                        style={{
                          backgroundColor: "var(--bg-elevated)",
                          borderColor: "var(--border-strong)",
                        }}
                      />
                    )}
                    {item.label}
                  </button>
                );
              })}
            </div>

            {/* Tab Body */}
            {!report ? (
              <div className="py-14 text-center text-sm" style={{ color: "var(--text-muted)" }}>
                Analyze a chest X-ray to generate an ACR clinical report.
              </div>
            ) : (
              <AnimatePresence mode="wait">
                <motion.div
                  key={tab}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
                >
                  {tab === "findings" && (
                    <div className="space-y-6">
                      {/* Refined AI Dictation Audio Banner */}
                      {liveTranscript && (
                        <div
                          className="rounded-2xl border p-4 sm:p-5 space-y-3"
                          style={{
                            borderColor: "var(--border-strong)",
                            backgroundColor: "var(--bg-secondary)",
                          }}
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div
                              className="flex items-center gap-2 text-xs font-semibold"
                              style={{ color: "var(--accent-clinical)" }}
                            >
                              <Mic className="h-3.5 w-3.5" />
                              <span>AI Dictation Transcript</span>
                              {isSpeaking && (
                                <span className="ml-1 flex items-end gap-0.5 h-3">
                                  {[0, 0.14, 0.28, 0.42].map((d, i) => (
                                    <motion.span
                                      key={i}
                                      className="w-0.5 rounded-full"
                                      style={{ backgroundColor: "var(--accent-clinical)" }}
                                      animate={{ height: ["3px", "12px", "4px"] }}
                                      transition={{ duration: 0.7, repeat: Infinity, delay: d }}
                                    />
                                  ))}
                                </span>
                              )}
                            </div>

                            <div className="flex items-center gap-1.5">
                              <button
                                type="button"
                                onClick={toggleSpeakTranscript}
                                className="flex cursor-pointer items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-medium transition-colors hover:border-teal-500/50"
                                style={{
                                  borderColor: isSpeaking
                                    ? "var(--accent-alert)"
                                    : "var(--border-strong)",
                                  backgroundColor: "var(--bg-elevated)",
                                  color: "var(--text-primary)",
                                }}
                              >
                                {isSpeaking ? (
                                  <>
                                    <Square className="h-2.5 w-2.5" style={{ color: "var(--accent-alert)" }} />
                                    <span>Stop</span>
                                  </>
                                ) : (
                                  <>
                                    <Volume2 className="h-3 w-3" style={{ color: "var(--accent-clinical)" }} />
                                    <span>Listen</span>
                                  </>
                                )}
                              </button>

                              {!editingTranscript ? (
                                <button
                                  type="button"
                                  onClick={() => setEditingTranscript(true)}
                                  className="cursor-pointer rounded-full border px-2.5 py-1 text-[11px] transition-colors hover:border-teal-500/40"
                                  style={{
                                    borderColor: "var(--border-subtle)",
                                    backgroundColor: "var(--bg-elevated)",
                                    color: "var(--text-secondary)",
                                  }}
                                >
                                  Edit
                                </button>
                              ) : (
                                <button
                                  type="button"
                                  onClick={async () => {
                                    await onSaveTranscript(transcriptDraft);
                                    setEditingTranscript(false);
                                  }}
                                  className="flex cursor-pointer items-center gap-1 rounded-full px-3 py-1 text-[11px] font-semibold"
                                  style={{
                                    backgroundColor: "var(--accent-clinical)",
                                    color: "#ffffff",
                                  }}
                                >
                                  <Save className="h-3 w-3" />
                                  <span>Save</span>
                                </button>
                              )}
                            </div>
                          </div>

                          {editingTranscript ? (
                            <textarea
                              rows={3}
                              value={transcriptDraft}
                              onChange={(e) => setTranscriptDraft(e.target.value)}
                              className="w-full rounded-xl border p-3 text-sm leading-relaxed focus:outline-none"
                              style={{
                                borderColor: "var(--border-strong)",
                                backgroundColor: "var(--bg-elevated)",
                                color: "var(--text-primary)",
                              }}
                            />
                          ) : (
                            <p
                              className="text-[14px] leading-relaxed"
                              style={{ color: "var(--text-primary)" }}
                            >
                              “{liveTranscript}”
                            </p>
                          )}
                        </div>
                      )}

                      {/* Clean Editorial Impression */}
                      <div className="space-y-1.5">
                        <div className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>
                          Impression
                        </div>
                        <p
                          className="whitespace-pre-line text-[15px] leading-relaxed font-semibold"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {report.impression}
                        </p>
                      </div>

                      {/* Clean Clinical Highlights (Hairline separated — no heavy nested box) */}
                      {execSummary.length > 0 && (
                        <div
                          className="border-t pt-5 space-y-2"
                          style={{ borderColor: "var(--border-subtle)" }}
                        >
                          <div className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>
                            Key Highlights
                          </div>
                          <ul className="space-y-1.5 text-[13.5px] leading-relaxed">
                            {execSummary.map((line, i) => (
                              <li key={i} className="flex items-start gap-2.5">
                                <span
                                  className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full"
                                  style={{ backgroundColor: "var(--accent-clinical)" }}
                                />
                                <span style={{ color: "var(--text-secondary)" }}>{line}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {differentials.length > 0 && (
                        <div className="space-y-2">
                          <div className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>
                            Differential Considerations
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {differentials.map((dx, i) => (
                              <span
                                key={i}
                                className="rounded-full border px-3 py-1 text-xs"
                                style={{
                                  borderColor: "var(--border-subtle)",
                                  backgroundColor: "var(--bg-secondary)",
                                  color: "var(--text-secondary)",
                                }}
                              >
                                {dx}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Detailed Findings */}
                      <div
                        className="border-t pt-5 space-y-1.5"
                        style={{ borderColor: "var(--border-subtle)" }}
                      >
                        <div className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>
                          Detailed Findings
                        </div>
                        <p
                          className="whitespace-pre-line text-[13.5px] leading-relaxed"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {report.findings}
                        </p>
                      </div>

                      {/* Collapsible Anatomical Breakdown (Progressive Disclosure) */}
                      {compartmentEntries.length > 0 && (
                        <div
                          className="border-t pt-4"
                          style={{ borderColor: "var(--border-subtle)" }}
                        >
                          <button
                            type="button"
                            onClick={() => setShowCompartments((v) => !v)}
                            className="flex w-full cursor-pointer items-center justify-between text-xs font-medium"
                            style={{ color: "var(--text-secondary)" }}
                          >
                            <span>Anatomical Compartments ({compartmentEntries.length})</span>
                            {showCompartments ? (
                              <ChevronUp className="h-4 w-4" />
                            ) : (
                              <ChevronDown className="h-4 w-4" />
                            )}
                          </button>

                          <AnimatePresence initial={false}>
                            {showCompartments && (
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: "auto" }}
                                exit={{ opacity: 0, height: 0 }}
                                transition={{ duration: 0.2 }}
                                className="overflow-hidden"
                              >
                                <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                                  {compartmentEntries.map(([comp, desc]) => (
                                    <div
                                      key={comp}
                                      className="rounded-xl border p-3.5 space-y-1"
                                      style={{
                                        borderColor: "var(--border-subtle)",
                                        backgroundColor: "var(--bg-secondary)",
                                      }}
                                    >
                                      <div
                                        className="text-[11px] font-semibold capitalize"
                                        style={{ color: "var(--accent-optical)" }}
                                      >
                                        {comp.replace(/_/g, " ")}
                                      </div>
                                      <p
                                        className="text-xs leading-relaxed"
                                        style={{ color: "var(--text-secondary)" }}
                                      >
                                        {desc}
                                      </p>
                                    </div>
                                  ))}
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      )}
                    </div>
                  )}

                  {tab === "guidelines" && (
                    <div className="space-y-5">
                      <div className="space-y-1.5">
                        <div className="text-xs font-semibold" style={{ color: "var(--accent-clinical)" }}>
                          Evidence-Based Recommendations
                        </div>
                        <p className="whitespace-pre-line text-sm leading-relaxed">
                          {report.recommendations ||
                            "Correlate radiologic findings with clinical presentation and laboratory markers."}
                        </p>
                      </div>

                      <div
                        className="border-t pt-5 space-y-1.5"
                        style={{ borderColor: "var(--border-subtle)" }}
                      >
                        <div className="text-xs font-semibold" style={{ color: "var(--accent-optical)" }}>
                          Management Considerations
                        </div>
                        <p
                          className="whitespace-pre-line text-sm leading-relaxed"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {report.prescriptions ||
                            "No automated pharmacologic order suggested. Attending clinician should direct therapy."}
                        </p>
                      </div>
                    </div>
                  )}

                  {tab === "patient" && (
                    <div className="space-y-4">
                      <div
                        className="rounded-2xl border p-5 space-y-2.5"
                        style={{
                          borderColor: "var(--border-subtle)",
                          backgroundColor: "var(--bg-secondary)",
                        }}
                      >
                        <div className="text-xs font-semibold" style={{ color: "var(--accent-clinical)" }}>
                          Plain-Language Summary
                        </div>
                        <p className="whitespace-pre-line text-[14.5px] leading-relaxed">
                          {report.patient_summary ||
                            "Your chest X-ray has been reviewed with AI diagnostic support. Please discuss these results directly with your doctor."}
                        </p>
                      </div>
                      <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                        Written for patient communication; verify with the attending radiologist before sharing.
                      </p>
                    </div>
                  )}

                  {tab === "signoff" && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2">
                        <label className="space-y-1">
                          <span className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                            Attending Physician
                          </span>
                          <input
                            type="text"
                            value={physicianName}
                            onChange={(e) => onPhysicianNameChange(e.target.value)}
                            className="w-full rounded-xl border bg-transparent px-3.5 py-2 text-sm focus:border-teal-500/60 focus:outline-none"
                            style={{ borderColor: "var(--border-strong)" }}
                          />
                        </label>
                        <label className="space-y-1">
                          <span className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                            Credentials
                          </span>
                          <input
                            type="text"
                            value={physicianCreds}
                            onChange={(e) => onPhysicianCredsChange(e.target.value)}
                            className="w-full rounded-xl border bg-transparent px-3.5 py-2 text-sm focus:border-teal-500/60 focus:outline-none"
                            style={{ borderColor: "var(--border-strong)" }}
                          />
                        </label>
                      </div>

                      <label className="block space-y-1">
                        <span className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                          Clinical Notes / Addendum
                        </span>
                        <textarea
                          rows={3}
                          value={physicianNotes}
                          onChange={(e) => onPhysicianNotesChange(e.target.value)}
                          placeholder="Optional comparison with prior radiographs..."
                          className="w-full rounded-xl border bg-transparent px-3.5 py-2.5 text-sm focus:border-teal-500/60 focus:outline-none"
                          style={{ borderColor: "var(--border-strong)" }}
                        />
                      </label>

                      <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                        {signedAt ? (
                          <div
                            className="flex items-center gap-1.5 text-xs font-medium"
                            style={{ color: "var(--accent-clinical)" }}
                          >
                            <ShieldCheck className="h-4 w-4" />
                            <span>Signed · {signedAt}</span>
                          </div>
                        ) : (
                          <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                            Saves signature to the database and PDF sheet.
                          </span>
                        )}
                        <button
                          type="button"
                          onClick={onSignReport}
                          className="cursor-pointer rounded-full px-5 py-2 text-xs font-semibold transition-all hover:brightness-110 active:scale-[0.98]"
                          style={{
                            backgroundColor: "var(--accent-clinical)",
                            color: "#ffffff",
                          }}
                        >
                          {signedAt ? "Update Signature" : "Sign & Attest"}
                        </button>
                      </div>
                    </div>
                  )}
                </motion.div>
              </AnimatePresence>
            )}
          </SpotlightCard>
        </div>

        {/* RIGHT 5 COLS: Clean Grounded Radiology Copilot */}
        <div className="lg:col-span-5">
          <div className="atelier-card flex h-[640px] flex-col overflow-hidden rounded-[24px]">
            {/* Header */}
            <div
              className="flex items-center justify-between border-b px-5 py-3.5"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              <div className="flex items-center gap-2.5">
                <div
                  className="flex h-8 w-8 items-center justify-center rounded-full"
                  style={{ backgroundColor: "var(--accent-clinical-dim)" }}
                >
                  <Stethoscope className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                </div>
                <div>
                  <div className="font-display text-sm font-bold">Radiology Copilot</div>
                  <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                    Grounded in this scan &amp; patient record
                  </div>
                </div>
              </div>

              {copilotMessages.length > 0 && (
                <button
                  type="button"
                  onClick={onClearCopilot}
                  title="Clear chat"
                  className="flex cursor-pointer items-center gap-1 rounded-full border px-2.5 py-1 text-[11px] transition-colors hover:border-teal-500/40"
                  style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
                >
                  <RotateCcw className="h-3 w-3" />
                  <span>Clear</span>
                </button>
              )}
            </div>

            {/* Conversation Transcript */}
            <div ref={chatScrollRef} className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
              {copilotMessages.length === 0 ? (
                <div className="space-y-4 pt-3">
                  <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                    Ask anything about this radiograph — all 14 pathology scores, heart size, and
                    patient history are attached automatically.
                  </p>
                  <div className="flex flex-col gap-2">
                    {COPILOT_SUGGESTIONS.map((q) => (
                      <button
                        key={q}
                        type="button"
                        disabled={!predictionData || copilotLoading}
                        onClick={() => onSendCopilot(q)}
                        className="cursor-pointer rounded-2xl border px-3.5 py-2.5 text-left text-xs font-medium transition-all hover:border-teal-500/45 disabled:opacity-50"
                        style={{
                          borderColor: "var(--border-subtle)",
                          backgroundColor: "var(--bg-secondary)",
                          color: "var(--text-primary)",
                        }}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                copilotMessages.map((msg) => (
                  <motion.div
                    key={msg.id}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
                    className={`group flex flex-col ${
                      msg.role === "user" ? "items-end" : "items-start"
                    }`}
                  >
                    <div
                      className="relative max-w-[92%] rounded-2xl border px-4 py-3 text-[13px] leading-relaxed"
                      style={{
                        borderColor:
                          msg.role === "user" ? "var(--accent-clinical)" : "var(--border-subtle)",
                        backgroundColor:
                          msg.role === "user" ? "var(--accent-clinical-dim)" : "var(--bg-secondary)",
                        color: "var(--text-primary)",
                      }}
                    >
                      {msg.role === "assistant"
                        ? renderClinicalMarkdown(msg.content)
                        : msg.content}
                    </div>
                    <div
                      className="mt-1 flex items-center gap-2 px-1 text-[10px]"
                      style={{ color: "var(--text-muted)" }}
                    >
                      <span>{msg.role === "user" ? "You" : "AI Copilot"}</span>
                      {msg.role === "assistant" && (
                        <button
                          type="button"
                          onClick={() => handleCopyMsg(msg.id, msg.content)}
                          className="cursor-pointer opacity-0 transition-opacity group-hover:opacity-100"
                          style={{ color: "var(--accent-clinical)" }}
                        >
                          {copiedMsgId === msg.id ? "Copied ✓" : "Copy"}
                        </button>
                      )}
                    </div>
                  </motion.div>
                ))
              )}

              {copilotLoading && (
                <div
                  className="inline-flex items-center gap-2.5 rounded-2xl border px-4 py-2.5 text-xs"
                  style={{
                    borderColor: "var(--border-subtle)",
                    backgroundColor: "var(--bg-secondary)",
                    color: "var(--text-secondary)",
                  }}
                >
                  <span className="flex items-end gap-0.5 h-3">
                    {[0, 0.16, 0.32].map((delay, idx) => (
                      <motion.span
                        key={idx}
                        className="w-0.5 rounded-full"
                        style={{ backgroundColor: "var(--accent-clinical)" }}
                        animate={{ height: ["4px", "12px", "4px"] }}
                        transition={{ duration: 0.85, repeat: Infinity, delay, ease: "easeInOut" }}
                      />
                    ))}
                  </span>
                  <span>Thinking...</span>
                </div>
              )}
            </div>

            {/* Quick Follow-Up Chips when chat is active */}
            {copilotMessages.length > 0 && !copilotLoading && (
              <div className="no-scrollbar flex items-center gap-1.5 overflow-x-auto px-4 pb-2">
                {QUICK_FOLLOWUPS.map((chip) => (
                  <button
                    key={chip.label}
                    type="button"
                    disabled={!predictionData}
                    onClick={() => onSendCopilot(chip.q)}
                    className="shrink-0 cursor-pointer rounded-full border px-2.5 py-1 text-[11px] whitespace-nowrap transition-colors hover:border-teal-500/45"
                    style={{
                      borderColor: "var(--border-subtle)",
                      backgroundColor: "var(--bg-secondary)",
                      color: "var(--text-secondary)",
                    }}
                  >
                    {chip.label}
                  </button>
                ))}
              </div>
            )}

            {/* Input Form */}
            <form
              onSubmit={handleSend}
              className="flex items-center gap-2 border-t p-3.5"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={!predictionData || copilotLoading}
                placeholder="Ask a follow-up question..."
                className="flex-1 rounded-full border bg-transparent px-4 py-2 text-xs focus:border-teal-500/60 focus:outline-none disabled:opacity-50"
                style={{ borderColor: "var(--border-strong)" }}
              />
              <button
                type="submit"
                disabled={!predictionData || !chatInput.trim() || copilotLoading}
                aria-label="Send message to Radiology Copilot"
                className="flex h-8.5 w-8.5 shrink-0 cursor-pointer items-center justify-center rounded-full transition-all hover:brightness-110 active:scale-95 disabled:opacity-40"
                style={{
                  backgroundColor: "var(--accent-clinical)",
                  color: "#ffffff",
                }}
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};
