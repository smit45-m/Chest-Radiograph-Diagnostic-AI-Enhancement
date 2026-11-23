import React, { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
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
  Sparkles,
  Square,
  Stethoscope,
  Volume2,
} from "lucide-react";
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

function renderInlineBold(text: string): React.ReactNode {
  const cleaned = text.replace(/\$([^$]+)\$/g, "$1");
  const parts = cleaned.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={idx} className="font-semibold" style={{ color: "var(--text-primary)" }}>
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <React.Fragment key={idx}>{part}</React.Fragment>;
  });
}

function renderClinicalMarkdown(text: string): React.ReactNode {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  return (
    <div className="space-y-2">
      {lines.map((raw, idx) => {
        const trimmed = raw.trim();
        if (/^#{1,4}\s+/.test(trimmed)) {
          return (
            <div
              key={idx}
              className="pt-1 font-mono-tech text-[10.5px] font-semibold tracking-[0.14em] uppercase"
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
                {renderInlineBold(trimmed.replace(/^[-*•]\s+/, ""))}
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
            {renderInlineBold(trimmed)}
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
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  const liveTranscript =
    report?.dictation_transcript ||
    report?.impression ||
    "";

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
    <div className="mx-auto max-w-[1360px] px-5 py-7 pb-24 md:px-8 md:py-9 lg:pb-12 space-y-8">
      {/* Page Header + Single-Line Export Actions */}
      <div className="flex flex-col justify-between gap-5 xl:flex-row xl:items-end">
        <div className="space-y-1.5">
          <div
            className="font-mono-tech text-[10px] font-medium tracking-[0.2em] uppercase"
            style={{ color: "var(--accent-clinical)" }}
          >
            04 · CLINICAL MONOGRAPH &amp; RADIOLOGY COPILOT
          </div>
          <h1 className="font-display text-3xl font-bold sm:text-4xl">
            Structured ACR dictation &amp;{" "}
            <span
              className="font-serif-editorial font-normal"
              style={{ color: "var(--accent-clinical)" }}
            >
              grounded dialogue.
            </span>
          </h1>
        </div>

        <div className="flex shrink-0 flex-wrap items-center gap-2 lg:flex-nowrap">
          <button
            type="button"
            disabled={!predictionData || reportLoading}
            onClick={onResynthesize}
            className="flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-full border px-3.5 py-2 text-xs font-medium transition-opacity hover:opacity-85 disabled:opacity-50"
            style={{
              borderColor: "var(--border-strong)",
              backgroundColor: "var(--bg-elevated)",
            }}
          >
            <RefreshCw className={`h-3.5 w-3.5 ${reportLoading ? "animate-spin" : ""}`} />
            <span>Re-Synthesize</span>
          </button>
          <button
            type="button"
            disabled={!report}
            onClick={handleCopy}
            className="flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-full border px-3.5 py-2 text-xs font-medium transition-opacity hover:opacity-85 disabled:opacity-50"
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
            <span>{copied ? "Copied" : "Copy Text"}</span>
          </button>
          <button
            type="button"
            disabled={!predictionData}
            onClick={onExportFhir}
            className="flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-full border px-3.5 py-2 text-xs font-medium transition-opacity hover:opacity-85 disabled:opacity-50"
            style={{
              borderColor: "var(--border-subtle)",
              backgroundColor: "var(--bg-elevated)",
            }}
          >
            <FileJson className="h-3.5 w-3.5" />
            <span>FHIR JSON</span>
          </button>
          <button
            type="button"
            disabled={!predictionData}
            onClick={onPrintPdf}
            className="flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-full px-4 py-2 text-xs font-semibold transition-transform active:scale-95 disabled:opacity-50"
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

      {/* Main 2-Column Layout */}
      <div className="grid grid-cols-1 items-start gap-7 lg:grid-cols-12">
        {/* LEFT 7 COLS: Structured ACR Clinical Monograph */}
        <div className="lg:col-span-7">
          <SpotlightCard strong className="rounded-[26px] p-6 sm:p-7 space-y-6">
            {/* Accession & Provider Bar */}
            <div
              className="flex flex-wrap items-center justify-between gap-3 border-b pb-4"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                  <h2 className="font-display text-base font-bold">
                    ACR Thoracic Radiology Monograph
                  </h2>
                </div>
                <div
                  className="flex flex-wrap items-center gap-2 font-mono-tech text-[10px] uppercase"
                  style={{ color: "var(--text-muted)" }}
                >
                  <span>
                    STUDY: {activeStudyName} · {signedAt ? "ATTESTED FINAL" : "AI-ASSISTED DRAFT"}
                  </span>
                  {activeStudyRecordId && (
                    <span
                      className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[9px]"
                      style={{
                        borderColor: "var(--accent-clinical)",
                        color: "var(--accent-clinical)",
                      }}
                    >
                      <Database className="h-2.5 w-2.5" />
                      Synced · {activeUserName || "User DB"}
                    </span>
                  )}
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {report?.source && (
                  <span
                    className="rounded-full border px-2.5 py-1 font-mono-tech text-[10px]"
                    style={{
                      borderColor: "var(--border-subtle)",
                      color: "var(--text-secondary)",
                    }}
                  >
                    {report.source}
                  </span>
                )}
                {report?.triage_level && (
                  <span
                    className="rounded-full border px-2.5 py-1 font-mono-tech text-[10px] font-medium uppercase"
                    style={{
                      borderColor: isCritical ? "var(--accent-alert)" : "var(--accent-clinical)",
                      color: isCritical ? "var(--accent-alert)" : "var(--accent-clinical)",
                      backgroundColor: isCritical
                        ? "var(--accent-alert-dim)"
                        : "var(--accent-clinical-dim)",
                    }}
                  >
                    {report.triage_level}
                  </span>
                )}
              </div>
            </div>

            {/* Clean 4-Tab Pill Switcher */}
            <div
              className="inline-flex flex-wrap gap-1 rounded-full border p-1"
              style={{
                borderColor: "var(--border-subtle)",
                backgroundColor: "var(--bg-secondary)",
              }}
            >
              {(
                [
                  { id: "findings", label: "Findings" },
                  { id: "guidelines", label: "Guidelines & Rx" },
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
                        transition={{ type: "spring", stiffness: 400, damping: 32 }}
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

            {/* Smooth Cross-Fading Tab Panels */}
            {!report ? (
              <div className="py-16 text-center text-sm" style={{ color: "var(--text-muted)" }}>
                Analyze a radiograph to synthesize an ACR-compliant clinical monograph.
              </div>
            ) : (
              <AnimatePresence mode="wait">
                <motion.div
                  key={tab}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                >
                  {tab === "findings" && (
                    <div className="space-y-6">
                      {/* Real AI Attending Dictation Transcript with Audio Readout & DB Save */}
                      {liveTranscript && (
                        <div
                          className="rounded-2xl border p-5 space-y-3"
                          style={{
                            borderColor: "var(--accent-clinical)",
                            backgroundColor: "var(--accent-clinical-dim)",
                          }}
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div
                              className="flex items-center gap-2 font-mono-tech text-[10px] font-semibold tracking-[0.16em] uppercase"
                              style={{ color: "var(--accent-clinical)" }}
                            >
                              <Mic className="h-3.5 w-3.5" />
                              <span>AI ATTENDING DICTATION TRANSCRIPT</span>
                            </div>

                            <div className="flex items-center gap-1.5">
                              <button
                                type="button"
                                onClick={toggleSpeakTranscript}
                                className="flex cursor-pointer items-center gap-1.5 rounded-full border px-3 py-1 font-mono-tech text-[10px] font-medium transition-opacity hover:opacity-85"
                                style={{
                                  borderColor: "var(--accent-clinical)",
                                  backgroundColor: "var(--bg-elevated)",
                                  color: "var(--text-primary)",
                                }}
                              >
                                {isSpeaking ? (
                                  <>
                                    <Square className="h-2.5 w-2.5" style={{ color: "var(--accent-alert)" }} />
                                    <span>Stop Audio</span>
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
                                  className="cursor-pointer rounded-full border px-2.5 py-1 font-mono-tech text-[10px]"
                                  style={{
                                    borderColor: "var(--border-strong)",
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
                                  className="flex cursor-pointer items-center gap-1 rounded-full px-3 py-1 font-mono-tech text-[10px] font-semibold"
                                  style={{
                                    backgroundColor: "var(--accent-clinical)",
                                    color: "#ffffff",
                                  }}
                                >
                                  <Save className="h-3 w-3" />
                                  <span>Save to DB</span>
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

                      {execSummary.length > 0 && (
                        <div
                          className="rounded-2xl border p-5 space-y-2.5"
                          style={{
                            borderColor: "var(--border-subtle)",
                            backgroundColor: "var(--bg-secondary)",
                          }}
                        >
                          <div
                            className="font-mono-tech text-[10px] font-medium tracking-[0.18em] uppercase"
                            style={{ color: "var(--accent-clinical)" }}
                          >
                            EXECUTIVE CLINICAL HANDOFF
                          </div>
                          <ul className="space-y-1.5 text-sm leading-relaxed">
                            {execSummary.map((line, i) => (
                              <li key={i} className="flex items-start gap-2.5">
                                <span
                                  className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full"
                                  style={{ backgroundColor: "var(--accent-clinical)" }}
                                />
                                <span>{line}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      <div className="space-y-2">
                        <div
                          className="font-mono-tech text-[10px] tracking-[0.18em] uppercase"
                          style={{ color: "var(--text-muted)" }}
                        >
                          IMPRESSION
                        </div>
                        <p
                          className="whitespace-pre-line text-[15px] leading-relaxed font-semibold"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {report.impression}
                        </p>
                      </div>

                      {differentials.length > 0 && (
                        <div className="space-y-2">
                          <div
                            className="font-mono-tech text-[10px] tracking-[0.18em] uppercase"
                            style={{ color: "var(--text-muted)" }}
                          >
                            DIFFERENTIAL CONSIDERATIONS
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {differentials.map((dx, i) => (
                              <span
                                key={i}
                                className="rounded-full border px-3 py-1 text-xs"
                                style={{
                                  borderColor: "var(--border-strong)",
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

                      {compartmentEntries.length > 0 && (
                        <div className="space-y-2.5">
                          <div
                            className="font-mono-tech text-[10px] tracking-[0.18em] uppercase"
                            style={{ color: "var(--text-muted)" }}
                          >
                            ANATOMICAL COMPARTMENTS
                          </div>
                          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                            {compartmentEntries.map(([comp, desc]) => (
                              <div
                                key={comp}
                                className="rounded-2xl border p-4 space-y-1"
                                style={{ borderColor: "var(--border-subtle)" }}
                              >
                                <div
                                  className="font-mono-tech text-[10px] tracking-[0.14em] uppercase"
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
                        </div>
                      )}

                      <div
                        className="border-t pt-5 space-y-2"
                        style={{ borderColor: "var(--border-subtle)" }}
                      >
                        <div
                          className="font-mono-tech text-[10px] tracking-[0.18em] uppercase"
                          style={{ color: "var(--text-muted)" }}
                        >
                          DETAILED RADIOLOGIC FINDINGS
                        </div>
                        <p
                          className="whitespace-pre-line text-sm leading-relaxed"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {report.findings}
                        </p>
                      </div>
                    </div>
                  )}

                  {tab === "guidelines" && (
                    <div className="space-y-6">
                      <div className="space-y-2">
                        <div
                          className="font-mono-tech text-[10px] tracking-[0.18em] uppercase"
                          style={{ color: "var(--accent-clinical)" }}
                        >
                          EVIDENCE-BASED RECOMMENDATIONS
                        </div>
                        <p className="whitespace-pre-line text-sm leading-relaxed">
                          {report.recommendations ||
                            "Correlate radiologic findings with clinical presentation and laboratory markers."}
                        </p>
                      </div>

                      <div
                        className="rounded-2xl border p-5 space-y-2"
                        style={{
                          borderColor: "var(--border-subtle)",
                          backgroundColor: "var(--bg-secondary)",
                        }}
                      >
                        <div
                          className="font-mono-tech text-[10px] tracking-[0.18em] uppercase"
                          style={{ color: "var(--accent-optical)" }}
                        >
                          THERAPEUTIC &amp; MANAGEMENT CONSIDERATIONS
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
                        className="rounded-2xl border p-6 space-y-3"
                        style={{
                          borderColor: "var(--border-subtle)",
                          backgroundColor: "var(--bg-secondary)",
                        }}
                      >
                        <div
                          className="font-mono-tech text-[10px] tracking-[0.18em] uppercase"
                          style={{ color: "var(--accent-clinical)" }}
                        >
                          PLAIN-LANGUAGE PATIENT EXPLANATION
                        </div>
                        <p className="whitespace-pre-line text-[15px] leading-relaxed">
                          {report.patient_summary ||
                            "Your chest X-ray has been reviewed by an AI diagnostic support system. Please discuss these results directly with your physician."}
                        </p>
                      </div>
                      <p className="text-xs leading-relaxed" style={{ color: "var(--text-muted)" }}>
                        Designed for clear shared communication; verify with the attending
                        radiologist before releasing to the patient portal.
                      </p>
                    </div>
                  )}

                  {tab === "signoff" && (
                    <div className="space-y-5">
                      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                        <label className="space-y-1.5">
                          <span
                            className="font-mono-tech text-[10px] tracking-[0.14em] uppercase"
                            style={{ color: "var(--text-muted)" }}
                          >
                            Attending Physician
                          </span>
                          <input
                            type="text"
                            value={physicianName}
                            onChange={(e) => onPhysicianNameChange(e.target.value)}
                            className="w-full rounded-xl border bg-transparent px-3.5 py-2.5 text-sm focus:outline-none"
                            style={{ borderColor: "var(--border-strong)" }}
                          />
                        </label>
                        <label className="space-y-1.5">
                          <span
                            className="font-mono-tech text-[10px] tracking-[0.14em] uppercase"
                            style={{ color: "var(--text-muted)" }}
                          >
                            Credentials / Board
                          </span>
                          <input
                            type="text"
                            value={physicianCreds}
                            onChange={(e) => onPhysicianCredsChange(e.target.value)}
                            className="w-full rounded-xl border bg-transparent px-3.5 py-2.5 text-sm focus:outline-none"
                            style={{ borderColor: "var(--border-strong)" }}
                          />
                        </label>
                      </div>

                      <label className="block space-y-1.5">
                        <span
                          className="font-mono-tech text-[10px] tracking-[0.14em] uppercase"
                          style={{ color: "var(--text-muted)" }}
                        >
                          Clinical Addendum / Correlation Notes
                        </span>
                        <textarea
                          rows={3}
                          value={physicianNotes}
                          onChange={(e) => onPhysicianNotesChange(e.target.value)}
                          placeholder="Optional comparison with prior radiographs or bedside exam..."
                          className="w-full rounded-xl border bg-transparent px-3.5 py-2.5 text-sm focus:outline-none"
                          style={{ borderColor: "var(--border-strong)" }}
                        />
                      </label>

                      <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                        {signedAt ? (
                          <div
                            className="flex items-center gap-2 font-mono-tech text-xs"
                            style={{ color: "var(--accent-clinical)" }}
                          >
                            <ShieldCheck className="h-4 w-4" />
                            <span>Digitally attested · {signedAt}</span>
                          </div>
                        ) : (
                          <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                            Attestation stamps the printed PDF consultation sheet.
                          </span>
                        )}
                        <button
                          type="button"
                          onClick={onSignReport}
                          className="cursor-pointer rounded-full px-5 py-2.5 text-xs font-semibold transition-transform active:scale-95"
                          style={{
                            backgroundColor: "var(--accent-clinical)",
                            color: "#ffffff",
                          }}
                        >
                          {signedAt ? "Update Signature" : "Sign & Attest Report"}
                        </button>
                      </div>
                    </div>
                  )}
                </motion.div>
              </AnimatePresence>
            )}
          </SpotlightCard>
        </div>

        {/* RIGHT 5 COLS: Grounded Gemini Radiology Copilot */}
        <div className="lg:col-span-5">
          <div className="atelier-card flex h-[660px] flex-col overflow-hidden rounded-[26px]">
            {/* Copilot Header */}
            <div
              className="flex items-center justify-between border-b px-6 py-4"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              <div className="flex items-center gap-2.5">
                <div
                  className="flex h-8 w-8 items-center justify-center rounded-full border"
                  style={{
                    borderColor: "var(--accent-clinical)",
                    backgroundColor: "var(--accent-clinical-dim)",
                  }}
                >
                  <Stethoscope className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                </div>
                <div>
                  <div className="font-display text-sm font-bold">Gemini Radiology Copilot</div>
                  <div
                    className="font-mono-tech text-[10px]"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Grounded in live scores, CTR &amp; Grad-CAM++
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {copilotMessages.length > 0 && (
                  <button
                    type="button"
                    onClick={onClearCopilot}
                    title="Clear conversation"
                    className="flex cursor-pointer items-center gap-1 rounded-full border px-2.5 py-1 font-mono-tech text-[10px] transition-opacity hover:opacity-80"
                    style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
                  >
                    <RotateCcw className="h-3 w-3" />
                    <span>Clear</span>
                  </button>
                )}
                <Sparkles className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
              </div>
            </div>

            {/* Transcript */}
            <div ref={chatScrollRef} className="flex-1 space-y-4 overflow-y-auto px-6 py-5">
              {copilotMessages.length === 0 ? (
                <div className="space-y-5 pt-4">
                  <p
                    className="text-sm leading-relaxed"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Ask clinical questions about the active chest radiograph. Context from all 14
                    pathology scores, cardiac biometry, and patient history is attached
                    automatically.
                  </p>
                  <div className="space-y-2">
                    <div
                      className="font-mono-tech text-[10px] tracking-[0.16em] uppercase"
                      style={{ color: "var(--text-muted)" }}
                    >
                      SUGGESTED CLINICAL PROMPTS
                    </div>
                    <div className="flex flex-col gap-2">
                      {COPILOT_SUGGESTIONS.map((q) => (
                        <button
                          key={q}
                          type="button"
                          disabled={!predictionData || copilotLoading}
                          onClick={() => onSendCopilot(q)}
                          className="cursor-pointer rounded-2xl border px-4 py-2.5 text-left text-xs font-medium transition-all hover:border-teal-500/40 disabled:opacity-50"
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
                </div>
              ) : (
                copilotMessages.map((msg) => (
                  <motion.div
                    key={msg.id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                    className={`group flex flex-col ${
                      msg.role === "user" ? "items-end" : "items-start"
                    }`}
                  >
                    <div
                      className="relative max-w-[92%] rounded-2xl border px-4 py-3 text-sm leading-relaxed"
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
                      className="mt-1 flex items-center gap-2 px-1 font-mono-tech text-[9.5px]"
                      style={{ color: "var(--text-muted)" }}
                    >
                      <span>{msg.role === "user" ? "Clinician" : msg.provider || "Gemini"}</span>
                      {msg.latencyMs != null && <span>· {Math.round(msg.latencyMs)} ms</span>}
                      {msg.role === "assistant" && (
                        <button
                          type="button"
                          onClick={() => handleCopyMsg(msg.id, msg.content)}
                          className="ml-1 cursor-pointer opacity-0 transition-opacity group-hover:opacity-100"
                          style={{ color: "var(--accent-clinical)" }}
                        >
                          {copiedMsgId === msg.id ? "Copied ✓" : "Copy"}
                        </button>
                      )}
                    </div>
                  </motion.div>
                ))
              )}

              {/* Animated 3-Bar Clinical Waveform Typing Indicator */}
              {copilotLoading && (
                <div
                  className="inline-flex items-center gap-2.5 rounded-2xl border px-4 py-2.5 font-mono-tech text-xs"
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
                        transition={{
                          duration: 0.9,
                          repeat: Infinity,
                          delay,
                          ease: "easeInOut",
                        }}
                      />
                    ))}
                  </span>
                  <span>Synthesizing clinical guidance...</span>
                </div>
              )}
            </div>

            {/* Clean Input Form */}
            <form
              onSubmit={handleSend}
              className="flex items-center gap-2 border-t p-4"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={!predictionData || copilotLoading}
                placeholder="Ask about findings, guidelines, differentials..."
                className="flex-1 rounded-full border bg-transparent px-4 py-2.5 text-xs focus:outline-none disabled:opacity-50"
                style={{ borderColor: "var(--border-strong)" }}
              />
              <button
                type="submit"
                disabled={!predictionData || !chatInput.trim() || copilotLoading}
                aria-label="Send message to Radiology Copilot"
                className="flex h-9 w-9 shrink-0 cursor-pointer items-center justify-center rounded-full transition-transform active:scale-95 disabled:opacity-40"
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
