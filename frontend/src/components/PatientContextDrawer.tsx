import React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Sparkles, UserRound, X } from "lucide-react";
import { PATIENT_PERSONAS } from "../data/clinicalCatalog";
import type { PatientProfile } from "../types/clinical";

interface PatientContextDrawerProps {
  open: boolean;
  patient: PatientProfile;
  onChange: (next: PatientProfile) => void;
  onApplyAndResynthesize: () => void;
  onClose: () => void;
}

export const PatientContextDrawer: React.FC<PatientContextDrawerProps> = ({
  open,
  patient,
  onChange,
  onApplyAndResynthesize,
  onClose,
}) => {
  return (
    <AnimatePresence>
      {open && (
        <div className="no-print fixed inset-0 z-50 flex justify-end">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          />

          <motion.aside
            role="dialog"
            aria-label="Patient Clinical Profile"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 320, damping: 32 }}
            className="relative z-10 flex h-full w-full max-w-md flex-col border-l p-6 shadow-2xl"
            style={{
              backgroundColor: "var(--bg-elevated)",
              borderColor: "var(--border-strong)",
            }}
          >
            {/* Drawer Header */}
            <div
              className="flex items-center justify-between border-b pb-4"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              <div className="flex items-center gap-2.5">
                <UserRound className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                <div>
                  <h3 className="font-display text-base font-semibold">Patient Clinical Profile</h3>
                  <p className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                    Contextualizes the ACR report and Gemini Copilot
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close drawer"
                className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full border"
                style={{ borderColor: "var(--border-subtle)", color: "var(--text-secondary)" }}
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Body */}
            <div className="flex-1 space-y-5 overflow-y-auto py-5">
              {/* Quick Clinical Personas */}
              <div className="space-y-2">
                <div
                  className="font-mono-tech text-[10px] tracking-[0.16em] uppercase"
                  style={{ color: "var(--text-muted)" }}
                >
                  CLINICAL PERSONA PRESETS
                </div>
                <div className="grid grid-cols-1 gap-2">
                  {Object.entries(PATIENT_PERSONAS).map(([key, item]) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => onChange(item.profile)}
                      className="cursor-pointer rounded-xl border px-3.5 py-2 text-left text-xs transition-colors"
                      style={{
                        borderColor: "var(--border-subtle)",
                        backgroundColor: "var(--bg-secondary)",
                      }}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Structured Form */}
              <div className="grid grid-cols-2 gap-3">
                <label className="space-y-1">
                  <span
                    className="font-mono-tech text-[10px] uppercase"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Age
                  </span>
                  <input
                    type="number"
                    min={1}
                    max={115}
                    value={patient.age ?? ""}
                    onChange={(e) =>
                      onChange({
                        ...patient,
                        age: e.target.value ? Number(e.target.value) : null,
                      })
                    }
                    className="w-full rounded-xl border bg-transparent px-3 py-2 text-sm focus:outline-none"
                    style={{ borderColor: "var(--border-strong)" }}
                  />
                </label>

                <label className="space-y-1">
                  <span
                    className="font-mono-tech text-[10px] uppercase"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Sex
                  </span>
                  <select
                    value={patient.sex}
                    onChange={(e) => onChange({ ...patient, sex: e.target.value })}
                    className="w-full rounded-xl border px-3 py-2 text-sm focus:outline-none"
                    style={{
                      borderColor: "var(--border-strong)",
                      backgroundColor: "var(--bg-elevated)",
                      color: "var(--text-primary)",
                    }}
                  >
                    <option value="Female">Female</option>
                    <option value="Male">Male</option>
                    <option value="Unspecified">Unspecified</option>
                  </select>
                </label>
              </div>

              <label className="block space-y-1">
                <span
                  className="font-mono-tech text-[10px] uppercase"
                  style={{ color: "var(--text-muted)" }}
                >
                  Care Setting
                </span>
                <input
                  type="text"
                  value={patient.setting}
                  onChange={(e) => onChange({ ...patient, setting: e.target.value })}
                  className="w-full rounded-xl border bg-transparent px-3 py-2 text-sm focus:outline-none"
                  style={{ borderColor: "var(--border-strong)" }}
                />
              </label>

              <label className="block space-y-1">
                <span
                  className="font-mono-tech text-[10px] uppercase"
                  style={{ color: "var(--text-muted)" }}
                >
                  Clinical Indication / Symptoms
                </span>
                <textarea
                  rows={2}
                  value={patient.indication}
                  onChange={(e) => onChange({ ...patient, indication: e.target.value })}
                  className="w-full rounded-xl border bg-transparent px-3 py-2 text-sm focus:outline-none"
                  style={{ borderColor: "var(--border-strong)" }}
                />
              </label>

              <label className="block space-y-1">
                <span
                  className="font-mono-tech text-[10px] uppercase"
                  style={{ color: "var(--text-muted)" }}
                >
                  Comorbidities
                </span>
                <input
                  type="text"
                  value={patient.comorbidities}
                  onChange={(e) => onChange({ ...patient, comorbidities: e.target.value })}
                  className="w-full rounded-xl border bg-transparent px-3 py-2 text-sm focus:outline-none"
                  style={{ borderColor: "var(--border-strong)" }}
                />
              </label>

              <label className="block space-y-1">
                <span
                  className="font-mono-tech text-[10px] uppercase"
                  style={{ color: "var(--text-muted)" }}
                >
                  Smoking History
                </span>
                <input
                  type="text"
                  value={patient.smoking}
                  onChange={(e) => onChange({ ...patient, smoking: e.target.value })}
                  className="w-full rounded-xl border bg-transparent px-3 py-2 text-sm focus:outline-none"
                  style={{ borderColor: "var(--border-strong)" }}
                />
              </label>
            </div>

            {/* Footer Action */}
            <div className="border-t pt-4" style={{ borderColor: "var(--border-subtle)" }}>
              <button
                type="button"
                onClick={onApplyAndResynthesize}
                className="flex w-full cursor-pointer items-center justify-center gap-2 rounded-full py-3 text-xs font-medium transition-transform active:scale-95"
                style={{
                  backgroundColor: "var(--accent-clinical)",
                  color: "#ffffff",
                }}
              >
                <Sparkles className="h-3.5 w-3.5" />
                <span>Apply Profile &amp; Re-Synthesize Report</span>
              </button>
            </div>
          </motion.aside>
        </div>
      )}
    </AnimatePresence>
  );
};
