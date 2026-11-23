import React, { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, KeyRound, Sparkles, X } from "lucide-react";
import type { AIConfigStatus } from "../types/clinical";

interface GeminiEngineModalProps {
  open: boolean;
  config: AIConfigStatus | null;
  saving: boolean;
  onSave: (payload: Record<string, unknown>) => Promise<void>;
  onClose: () => void;
}

export const GeminiEngineModal: React.FC<GeminiEngineModalProps> = ({
  open,
  config,
  saving,
  onSave,
  onClose,
}) => {
  const [apiKey, setApiKey] = useState("");
  const [flashModel, setFlashModel] = useState("gemini-3.5-flash-lite");
  const [decisionEnabled, setDecisionEnabled] = useState(true);
  const [flashEnabled, setFlashEnabled] = useState(true);
  const [weight, setWeight] = useState(0.3);
  const [ensembleMode, setEnsembleMode] = useState<"full" | "fast">("full");

  useEffect(() => {
    if (!config) return;
    setFlashModel(config.gemini_flash_model || "gemini-3.5-flash-lite");
    setDecisionEnabled(config.gemini_decision_enabled);
    setFlashEnabled(config.gemini_flash_enabled);
    setWeight(config.decision_fusion_weight);
    setEnsembleMode(config.ensemble_mode || "full");
  }, [config, open]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const payload: Record<string, unknown> = {
      gemini_flash_model: flashModel,
      gemini_decision_enabled: decisionEnabled,
      gemini_flash_enabled: flashEnabled,
      decision_fusion_weight: weight,
      ensemble_mode: ensembleMode,
    };
    if (apiKey.trim()) {
      payload.gemini_api_key = apiKey.trim();
    }
    await onSave(payload);
    setApiKey("");
  };

  const models = config?.available_flash_models?.length
    ? config.available_flash_models
    : ["gemini-3.5-flash-lite", "gemini-4-hydrogen", "gemini-3.1-flash-lite", "gemini-4-argon"];

  return (
    <AnimatePresence>
      {open && (
        <div className="no-print fixed inset-0 z-50 flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/65 backdrop-blur-sm"
          />

          <motion.div
            role="dialog"
            aria-label="Gemini AI Engine Configuration"
            initial={{ opacity: 0, scale: 0.96, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 12 }}
            transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
            className="atelier-card-strong relative z-10 w-full max-w-lg rounded-3xl p-6 sm:p-7"
          >
            {/* Header */}
            <div
              className="flex items-start justify-between border-b pb-4"
              style={{ borderColor: "var(--border-subtle)" }}
            >
              <div className="space-y-1">
                <div
                  className="flex items-center gap-1.5 font-mono-tech text-[10px] tracking-[0.18em] uppercase"
                  style={{ color: "var(--accent-clinical)" }}
                >
                  <Sparkles className="h-3 w-3" />
                  <span>GOOGLE GEMINI CLINICAL ENGINE</span>
                </div>
                <h3 className="font-display text-xl font-semibold">
                  Decision 1 &amp; Interactions API
                </h3>
              </div>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close modal"
                className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full border"
                style={{ borderColor: "var(--border-subtle)", color: "var(--text-secondary)" }}
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <form onSubmit={handleSubmit} className="mt-5 space-y-5">
              {/* Live Connection Status */}
              <div
                className="flex items-center justify-between rounded-2xl border px-4 py-3 text-xs"
                style={{
                  borderColor: config?.gemini_configured
                    ? "var(--accent-clinical)"
                    : "var(--border-subtle)",
                  backgroundColor: config?.gemini_configured
                    ? "var(--accent-clinical-dim)"
                    : "var(--bg-secondary)",
                }}
              >
                <div className="flex items-center gap-2">
                  <CheckCircle2
                    className="h-4 w-4"
                    style={{
                      color: config?.gemini_configured
                        ? "var(--accent-clinical)"
                        : "var(--text-muted)",
                    }}
                  />
                  <span className="font-medium">
                    {config?.gemini_configured
                      ? `Connected (${config.gemini_key_hint || "Key Active"})`
                      : "Gemini API key not configured"}
                  </span>
                </div>
                <span className="font-mono-tech text-[10px]" style={{ color: "var(--text-muted)" }}>
                  v1beta
                </span>
              </div>

              {/* Optional API Key Update */}
              <label className="block space-y-1.5">
                <span
                  className="flex items-center gap-1.5 font-mono-tech text-[10px] tracking-[0.14em] uppercase"
                  style={{ color: "var(--text-muted)" }}
                >
                  <KeyRound className="h-3 w-3" />
                  Replace Gemini API Key (optional)
                </span>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Leave blank to keep active local .env key"
                  className="w-full rounded-xl border bg-transparent px-3.5 py-2.5 font-mono-tech text-xs focus:outline-none"
                  style={{ borderColor: "var(--border-strong)" }}
                />
              </label>

              {/* Interactions API Model Selector */}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <label className="space-y-1.5">
                  <span
                    className="font-mono-tech text-[10px] tracking-[0.14em] uppercase"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Interactions Report Model
                  </span>
                  <select
                    value={flashModel}
                    onChange={(e) => setFlashModel(e.target.value)}
                    className="w-full rounded-xl border px-3 py-2.5 font-mono-tech text-xs focus:outline-none"
                    style={{
                      borderColor: "var(--border-strong)",
                      backgroundColor: "var(--bg-elevated)",
                      color: "var(--text-primary)",
                    }}
                  >
                    {models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="space-y-1.5">
                  <span
                    className="font-mono-tech text-[10px] tracking-[0.14em] uppercase"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Vision Backbone Mode
                  </span>
                  <select
                    value={ensembleMode}
                    onChange={(e) => setEnsembleMode(e.target.value as "full" | "fast")}
                    className="w-full rounded-xl border px-3 py-2.5 font-mono-tech text-xs focus:outline-none"
                    style={{
                      borderColor: "var(--border-strong)",
                      backgroundColor: "var(--bg-elevated)",
                      color: "var(--text-primary)",
                    }}
                  >
                    <option value="full">Full 3-Backbone Ensemble</option>
                    <option value="fast">Fast Single-Backbone</option>
                  </select>
                </label>
              </div>

              {/* Clean Toggles */}
              <div className="space-y-2.5">
                <label
                  className="flex cursor-pointer items-center justify-between rounded-2xl border px-4 py-3"
                  style={{ borderColor: "var(--border-subtle)" }}
                >
                  <div>
                    <div className="text-xs font-semibold">Gemini Decision 1 Fusion</div>
                    <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                      Zero output tokens · calibrated 18-question diagnostic priors
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={decisionEnabled}
                    onChange={(e) => setDecisionEnabled(e.target.checked)}
                    className="h-4 w-4 accent-[#10b99a]"
                  />
                </label>

                <label
                  className="flex cursor-pointer items-center justify-between rounded-2xl border px-4 py-3"
                  style={{ borderColor: "var(--border-subtle)" }}
                >
                  <div>
                    <div className="text-xs font-semibold">Interactions API ACR Synthesis</div>
                    <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                      Structured JSON reports &amp; stateful multi-turn Radiology Copilot
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={flashEnabled}
                    onChange={(e) => setFlashEnabled(e.target.checked)}
                    className="h-4 w-4 accent-[#10b99a]"
                  />
                </label>
              </div>

              {/* Fusion Weight Slider */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span style={{ color: "var(--text-secondary)" }}>Decision 1 Fusion Weight</span>
                  <span className="font-mono-tech font-semibold" style={{ color: "var(--accent-clinical)" }}>
                    {Math.round(weight * 100)}% Gemini / {Math.round((1 - weight) * 100)}% CNN
                  </span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={0.5}
                  step={0.05}
                  value={weight}
                  onChange={(e) => setWeight(Number(e.target.value))}
                  className="clinical-slider w-full"
                />
              </div>

              {/* Submit Row */}
              <div className="flex items-center justify-end gap-2.5 pt-2">
                <button
                  type="button"
                  onClick={onClose}
                  className="cursor-pointer rounded-full border px-4 py-2 text-xs font-medium"
                  style={{ borderColor: "var(--border-subtle)", color: "var(--text-secondary)" }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={saving}
                  className="cursor-pointer rounded-full px-5 py-2 text-xs font-medium transition-transform active:scale-95 disabled:opacity-50"
                  style={{
                    backgroundColor: "var(--accent-clinical)",
                    color: "#ffffff",
                  }}
                >
                  {saving ? "Applying..." : "Save Configuration"}
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
