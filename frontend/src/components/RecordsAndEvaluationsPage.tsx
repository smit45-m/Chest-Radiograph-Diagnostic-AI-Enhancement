import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Clock,
  Database,
  FileAudio,
  HeartPulse,
  Plus,
  Save,
  Search,
  Sparkles,
  Trash2,
  UserCheck,
  UserPlus,
} from "lucide-react";
import { formatPathology } from "../data/clinicalCatalog";
import type {
  ArchivedStudySummary,
  EvaluationReport,
  UserDetail,
  UserRecord,
} from "../types/clinical";
import { AnimatedNumber } from "./AnimatedNumber";
import { SpotlightCard } from "./SpotlightCard";

interface RecordsAndEvaluationsPageProps {
  users: UserRecord[];
  activeUserId: string | null;
  activeUserDetail: UserDetail | null;
  evaluations: EvaluationReport | null;
  savingUser: boolean;
  onSelectUser: (userId: string) => void;
  onCreateUser: (payload: Partial<UserRecord> & { full_name: string }) => Promise<void>;
  onUpdateUser: (userId: string, patch: Partial<UserRecord>) => Promise<void>;
  onDeleteUser: (userId: string) => Promise<void>;
  onLoadStudy: (studyId: string) => Promise<void>;
  onDeleteStudy: (studyId: string) => Promise<void>;
  onOpenStudio: () => void;
}

type SubTab = "records" | "evaluations";

const BACKBONE_LABELS: Record<string, string> = {
  densenet121_all: "DenseNet-121 Multi-Cohort",
  densenet121_nih: "DenseNet-121 NIH-14",
  resnet50_512: "ResNet-50 High-Res 512px",
  cnn_ensemble_3x: "3-Backbone CNN Ensemble",
  gemini_decision_1: "Gemini Decision 1 (0 Tok)",
  fused_consensus: "Fused Clinical Consensus",
};

export const RecordsAndEvaluationsPage: React.FC<RecordsAndEvaluationsPageProps> = ({
  users,
  activeUserId,
  activeUserDetail,
  evaluations,
  savingUser,
  onSelectUser,
  onCreateUser,
  onUpdateUser,
  onDeleteUser,
  onLoadStudy,
  onDeleteStudy,
  onOpenStudio,
}) => {
  const [subTab, setSubTab] = useState<SubTab>("records");
  const [search, setSearch] = useState("");
  const [showNewForm, setShowNewForm] = useState(false);

  // Editable user form synced to activeUserDetail
  const [editForm, setEditForm] = useState({
    full_name: "",
    mrn: "",
    email: "",
    age: "",
    sex: "Unspecified",
    setting: "",
    indication: "",
    comorbidities: "",
    smoking: "",
    notes: "",
  });

  const [newForm, setNewForm] = useState({
    full_name: "",
    mrn: "",
    age: "55",
    sex: "Female",
    setting: "Ambulatory / Outpatient",
    indication: "Routine thoracic radiograph evaluation",
    comorbidities: "",
    smoking: "Never smoker",
  });

  useEffect(() => {
    if (!activeUserDetail) return;
    setEditForm({
      full_name: activeUserDetail.full_name || "",
      mrn: activeUserDetail.mrn || "",
      email: activeUserDetail.email || "",
      age: activeUserDetail.age != null ? String(activeUserDetail.age) : "",
      sex: activeUserDetail.sex || "Unspecified",
      setting: activeUserDetail.setting || "",
      indication: activeUserDetail.indication || "",
      comorbidities: activeUserDetail.comorbidities || "",
      smoking: activeUserDetail.smoking || "",
      notes: activeUserDetail.notes || "",
    });
  }, [activeUserDetail]);

  const filteredUsers = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return users;
    return users.filter(
      (u) =>
        u.full_name.toLowerCase().includes(q) ||
        u.mrn.toLowerCase().includes(q) ||
        (u.setting || "").toLowerCase().includes(q)
    );
  }, [users, search]);

  const handleSaveUser = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeUserId) return;
    await onUpdateUser(activeUserId, {
      full_name: editForm.full_name.trim(),
      mrn: editForm.mrn.trim(),
      email: editForm.email.trim(),
      age: editForm.age ? Number(editForm.age) : null,
      sex: editForm.sex,
      setting: editForm.setting.trim(),
      indication: editForm.indication.trim(),
      comorbidities: editForm.comorbidities.trim(),
      smoking: editForm.smoking.trim(),
      notes: editForm.notes.trim(),
    });
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newForm.full_name.trim()) return;
    await onCreateUser({
      full_name: newForm.full_name.trim(),
      mrn: newForm.mrn.trim() || undefined,
      age: newForm.age ? Number(newForm.age) : null,
      sex: newForm.sex,
      setting: newForm.setting,
      indication: newForm.indication,
      comorbidities: newForm.comorbidities,
      smoking: newForm.smoking,
    });
    setShowNewForm(false);
    setNewForm({
      full_name: "",
      mrn: "",
      age: "55",
      sex: "Female",
      setting: "Ambulatory / Outpatient",
      indication: "Routine thoracic radiograph evaluation",
      comorbidities: "",
      smoking: "Never smoker",
    });
  };

  const ctrHistory = activeUserDetail?.ctr_history || [];
  const headline = evaluations?.headline_metrics;
  const backboneEntries = Object.entries(evaluations?.backbone_comparison || {});

  return (
    <div className="mx-auto max-w-[1360px] px-5 py-7 pb-24 md:px-8 md:py-9 lg:pb-12 space-y-8">
      {/* Header + 2-Tab Pill Switcher */}
      <div className="flex flex-col justify-between gap-5 md:flex-row md:items-end">
        <div className="space-y-1.5">
          <div
            className="font-mono-tech text-[10px] font-medium tracking-[0.2em] uppercase"
            style={{ color: "var(--accent-clinical)" }}
          >
            06 · PATIENT DATABASE &amp; MODEL EVALUATIONS
          </div>
          <h1 className="font-display text-3xl font-bold sm:text-4xl">
            Longitudinal records &amp;{" "}
            <span
              className="font-serif-editorial font-normal"
              style={{ color: "var(--accent-clinical)" }}
            >
              verified metrics.
            </span>
          </h1>
        </div>

        <div
          className="inline-flex self-start rounded-full border p-1 md:self-auto"
          style={{
            borderColor: "var(--border-subtle)",
            backgroundColor: "var(--bg-secondary)",
          }}
        >
          {(
            [
              { id: "records", label: "User & Study Database", icon: Database },
              { id: "evaluations", label: "Evaluation Metrics", icon: BarChart3 },
            ] as const
          ).map((t) => {
            const Icon = t.icon;
            const active = subTab === t.id;
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => setSubTab(t.id)}
                className="relative z-10 flex cursor-pointer items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold transition-colors"
                style={{
                  color: active ? "var(--text-primary)" : "var(--text-secondary)",
                }}
              >
                {active && (
                  <motion.span
                    layoutId="records-eval-tab"
                    transition={{ type: "spring", stiffness: 400, damping: 32 }}
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
                <span>{t.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <AnimatePresence mode="wait">
        {subTab === "records" ? (
          <motion.div
            key="records"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.24 }}
            className="grid grid-cols-1 items-start gap-7 lg:grid-cols-12"
          >
            {/* LEFT 4 COLS: Persisted SQLite User / Patient Directory */}
            <div className="lg:col-span-4 space-y-4">
              <SpotlightCard strong className="rounded-[26px] p-5 space-y-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <Database className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                    <h2 className="font-display text-sm font-bold">
                      Stored Users ({users.length})
                    </h2>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowNewForm((v) => !v)}
                    className="flex cursor-pointer items-center gap-1 rounded-full px-3 py-1.5 text-[11px] font-semibold transition-transform active:scale-95"
                    style={{
                      backgroundColor: "var(--accent-clinical)",
                      color: "#ffffff",
                    }}
                  >
                    <Plus className="h-3.5 w-3.5" />
                    <span>New User</span>
                  </button>
                </div>

                {/* Inline New User Creator */}
                <AnimatePresence>
                  {showNewForm && (
                    <motion.form
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      onSubmit={handleCreate}
                      className="overflow-hidden rounded-2xl border p-4 space-y-3"
                      style={{
                        borderColor: "var(--accent-clinical)",
                        backgroundColor: "var(--bg-secondary)",
                      }}
                    >
                      <div className="flex items-center gap-1.5 font-mono-tech text-[10px] uppercase" style={{ color: "var(--accent-clinical)" }}>
                        <UserPlus className="h-3.5 w-3.5" />
                        <span>Create Persistent Record</span>
                      </div>
                      <input
                        type="text"
                        required
                        placeholder="Full Name (e.g. Dr. Clara Diaz)"
                        value={newForm.full_name}
                        onChange={(e) => setNewForm({ ...newForm, full_name: e.target.value })}
                        className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:outline-none"
                        style={{ borderColor: "var(--border-strong)" }}
                      />
                      <div className="grid grid-cols-2 gap-2">
                        <input
                          type="text"
                          placeholder="MRN (auto if blank)"
                          value={newForm.mrn}
                          onChange={(e) => setNewForm({ ...newForm, mrn: e.target.value })}
                          className="rounded-xl border bg-transparent px-3 py-2 font-mono-tech text-xs focus:outline-none"
                          style={{ borderColor: "var(--border-strong)" }}
                        />
                        <input
                          type="number"
                          placeholder="Age"
                          value={newForm.age}
                          onChange={(e) => setNewForm({ ...newForm, age: e.target.value })}
                          className="rounded-xl border bg-transparent px-3 py-2 text-xs focus:outline-none"
                          style={{ borderColor: "var(--border-strong)" }}
                        />
                      </div>
                      <input
                        type="text"
                        placeholder="Clinical Indication"
                        value={newForm.indication}
                        onChange={(e) => setNewForm({ ...newForm, indication: e.target.value })}
                        className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:outline-none"
                        style={{ borderColor: "var(--border-strong)" }}
                      />
                      <div className="flex justify-end gap-2">
                        <button
                          type="button"
                          onClick={() => setShowNewForm(false)}
                          className="cursor-pointer rounded-full border px-3 py-1 text-[11px]"
                          style={{ borderColor: "var(--border-subtle)" }}
                        >
                          Cancel
                        </button>
                        <button
                          type="submit"
                          disabled={savingUser}
                          className="cursor-pointer rounded-full px-3.5 py-1 text-[11px] font-semibold"
                          style={{ backgroundColor: "var(--accent-clinical)", color: "#fff" }}
                        >
                          Save User
                        </button>
                      </div>
                    </motion.form>
                  )}
                </AnimatePresence>

                {/* Search Input */}
                <div
                  className="flex items-center gap-2 rounded-xl border px-3 py-2"
                  style={{
                    borderColor: "var(--border-subtle)",
                    backgroundColor: "var(--bg-secondary)",
                  }}
                >
                  <Search className="h-3.5 w-3.5" style={{ color: "var(--text-muted)" }} />
                  <input
                    type="text"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder="Filter by name, MRN, or unit..."
                    className="w-full bg-transparent text-xs focus:outline-none"
                  />
                </div>

                {/* User Cards List */}
                <div className="max-h-[540px] space-y-2 overflow-y-auto pr-1">
                  {filteredUsers.map((u) => {
                    const isActive = u.id === activeUserId;
                    return (
                      <button
                        key={u.id}
                        type="button"
                        onClick={() => onSelectUser(u.id)}
                        className="w-full cursor-pointer rounded-2xl border p-3.5 text-left transition-all"
                        style={{
                          borderColor: isActive ? "var(--accent-clinical)" : "var(--border-subtle)",
                          backgroundColor: isActive
                            ? "var(--accent-clinical-dim)"
                            : "var(--bg-secondary)",
                        }}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="flex items-center gap-1.5">
                              <span className="truncate text-sm font-semibold">{u.full_name}</span>
                              {isActive && (
                                <UserCheck
                                  className="h-3.5 w-3.5 shrink-0"
                                  style={{ color: "var(--accent-clinical)" }}
                                />
                              )}
                            </div>
                            <div
                              className="mt-0.5 font-mono-tech text-[10px]"
                              style={{ color: "var(--text-muted)" }}
                            >
                              {u.mrn} · {u.age ? `${u.age}y ` : ""}
                              {u.sex}
                            </div>
                          </div>
                          <span
                            className="shrink-0 rounded-full border px-2 py-0.5 font-mono-tech text-[9.5px]"
                            style={{
                              borderColor: "var(--border-strong)",
                              color: "var(--text-secondary)",
                            }}
                          >
                            {u.study_count || 0} studies
                          </span>
                        </div>
                        <div
                          className="mt-2 flex items-center justify-between text-[11px]"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          <span className="truncate">{u.setting}</span>
                          {u.latest_ctr != null && (
                            <span
                              className="ml-2 shrink-0 font-mono-tech text-[10px]"
                              style={{
                                color:
                                  u.latest_ctr > 0.5
                                    ? "var(--accent-alert)"
                                    : "var(--accent-clinical)",
                              }}
                            >
                              CTR {u.latest_ctr.toFixed(3)}
                            </span>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </SpotlightCard>
            </div>

            {/* RIGHT 8 COLS: Selected User Profile Editor + CTR Trend + Study & AI Transcript History */}
            <div className="lg:col-span-8 space-y-6">
              {activeUserDetail && (
                <>
                  {/* Editable SQLite User Record */}
                  <SpotlightCard strong className="rounded-[26px] p-6 space-y-5">
                    <div
                      className="flex flex-wrap items-center justify-between gap-3 border-b pb-4"
                      style={{ borderColor: "var(--border-subtle)" }}
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <h2 className="font-display text-lg font-bold">
                            {activeUserDetail.full_name}
                          </h2>
                          <span
                            className="rounded-full border px-2.5 py-0.5 font-mono-tech text-[10px]"
                            style={{
                              borderColor: "var(--accent-clinical)",
                              backgroundColor: "var(--accent-clinical-dim)",
                              color: "var(--accent-clinical)",
                            }}
                          >
                            {activeUserDetail.mrn}
                          </span>
                        </div>
                        <p className="mt-0.5 font-mono-tech text-[10px]" style={{ color: "var(--text-muted)" }}>
                          SQLITE PERSISTENT RECORD · UPDATED{" "}
                          {new Date(activeUserDetail.updated_at).toLocaleString()}
                        </p>
                      </div>

                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={onOpenStudio}
                          className="cursor-pointer rounded-full border px-3.5 py-1.5 text-xs font-semibold"
                          style={{
                            borderColor: "var(--border-strong)",
                            backgroundColor: "var(--bg-secondary)",
                          }}
                        >
                          Scan for this User →
                        </button>
                        {users.length > 1 && (
                          <button
                            type="button"
                            onClick={() => onDeleteUser(activeUserDetail.id)}
                            title="Delete user record"
                            className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full border transition-colors hover:border-rose-500/60"
                            style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>
                    </div>

                    <form onSubmit={handleSaveUser} className="space-y-4">
                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
                        <label className="sm:col-span-2 space-y-1">
                          <span className="font-mono-tech text-[10px] uppercase" style={{ color: "var(--text-muted)" }}>
                            Full Name
                          </span>
                          <input
                            type="text"
                            value={editForm.full_name}
                            onChange={(e) => setEditForm({ ...editForm, full_name: e.target.value })}
                            className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:outline-none"
                            style={{ borderColor: "var(--border-strong)" }}
                          />
                        </label>
                        <label className="space-y-1">
                          <span className="font-mono-tech text-[10px] uppercase" style={{ color: "var(--text-muted)" }}>
                            Age
                          </span>
                          <input
                            type="number"
                            value={editForm.age}
                            onChange={(e) => setEditForm({ ...editForm, age: e.target.value })}
                            className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:outline-none"
                            style={{ borderColor: "var(--border-strong)" }}
                          />
                        </label>
                        <label className="space-y-1">
                          <span className="font-mono-tech text-[10px] uppercase" style={{ color: "var(--text-muted)" }}>
                            Sex
                          </span>
                          <select
                            value={editForm.sex}
                            onChange={(e) => setEditForm({ ...editForm, sex: e.target.value })}
                            className="w-full rounded-xl border px-3 py-2 text-xs focus:outline-none"
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

                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                        <label className="space-y-1">
                          <span className="font-mono-tech text-[10px] uppercase" style={{ color: "var(--text-muted)" }}>
                            Care Setting
                          </span>
                          <input
                            type="text"
                            value={editForm.setting}
                            onChange={(e) => setEditForm({ ...editForm, setting: e.target.value })}
                            className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:outline-none"
                            style={{ borderColor: "var(--border-strong)" }}
                          />
                        </label>
                        <label className="space-y-1">
                          <span className="font-mono-tech text-[10px] uppercase" style={{ color: "var(--text-muted)" }}>
                            Smoking History
                          </span>
                          <input
                            type="text"
                            value={editForm.smoking}
                            onChange={(e) => setEditForm({ ...editForm, smoking: e.target.value })}
                            className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:outline-none"
                            style={{ borderColor: "var(--border-strong)" }}
                          />
                        </label>
                      </div>

                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                        <label className="space-y-1">
                          <span className="font-mono-tech text-[10px] uppercase" style={{ color: "var(--text-muted)" }}>
                            Clinical Indication
                          </span>
                          <input
                            type="text"
                            value={editForm.indication}
                            onChange={(e) => setEditForm({ ...editForm, indication: e.target.value })}
                            className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:outline-none"
                            style={{ borderColor: "var(--border-strong)" }}
                          />
                        </label>
                        <label className="space-y-1">
                          <span className="font-mono-tech text-[10px] uppercase" style={{ color: "var(--text-muted)" }}>
                            Comorbidities
                          </span>
                          <input
                            type="text"
                            value={editForm.comorbidities}
                            onChange={(e) => setEditForm({ ...editForm, comorbidities: e.target.value })}
                            className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:outline-none"
                            style={{ borderColor: "var(--border-strong)" }}
                          />
                        </label>
                      </div>

                      <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                        <input
                          type="text"
                          value={editForm.notes}
                          onChange={(e) => setEditForm({ ...editForm, notes: e.target.value })}
                          placeholder="Longitudinal chart notes..."
                          className="flex-1 rounded-xl border bg-transparent px-3 py-2 text-xs focus:outline-none"
                          style={{ borderColor: "var(--border-subtle)" }}
                        />
                        <button
                          type="submit"
                          disabled={savingUser}
                          className="flex cursor-pointer items-center gap-1.5 rounded-full px-5 py-2 text-xs font-semibold transition-transform active:scale-95 disabled:opacity-50"
                          style={{
                            backgroundColor: "var(--accent-clinical)",
                            color: "#ffffff",
                          }}
                        >
                          <Save className="h-3.5 w-3.5" />
                          <span>{savingUser ? "Updating..." : "Update User Record"}</span>
                        </button>
                      </div>
                    </form>
                  </SpotlightCard>

                  {/* Longitudinal CTR Trend Sparkline (when >= 1 CTR study exists) */}
                  {ctrHistory.length > 0 && (
                    <SpotlightCard className="rounded-[24px] p-5 space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <HeartPulse className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                          <h3 className="font-display text-sm font-bold">
                            Longitudinal Cardiothoracic Ratio (CTR) Trend
                          </h3>
                        </div>
                        <span className="font-mono-tech text-[10px]" style={{ color: "var(--text-muted)" }}>
                          {ctrHistory.length} RECORDED MEASUREMENT{ctrHistory.length > 1 ? "S" : ""}
                        </span>
                      </div>
                      <div
                        className="rounded-xl border px-4 py-3"
                        style={{
                          borderColor: "var(--border-subtle)",
                          backgroundColor: "var(--bg-secondary)",
                        }}
                      >
                        <svg viewBox="0 0 520 86" className="h-20 w-full" fill="none">
                          {/* 0.50 Reference Cutoff Line */}
                          <line
                            x1="12"
                            y1="43"
                            x2="508"
                            y2="43"
                            stroke="var(--accent-alert)"
                            strokeOpacity="0.45"
                            strokeDasharray="4 4"
                          />
                          <text
                            x="472"
                            y="38"
                            fill="var(--accent-alert)"
                            fontSize="9"
                            fontFamily="IBM Plex Mono"
                          >
                            0.50
                          </text>
                          {ctrHistory.length > 1 && (
                            <polyline
                              fill="none"
                              stroke="var(--accent-clinical)"
                              strokeWidth="2"
                              points={ctrHistory
                                .map((pt, idx) => {
                                  const x =
                                    24 + (idx / Math.max(ctrHistory.length - 1, 1)) * 452;
                                  const y = 74 - Math.min(1, Math.max(0, (pt.ctr - 0.35) / 0.3)) * 62;
                                  return `${x},${y}`;
                                })
                                .join(" ")}
                            />
                          )}
                          {ctrHistory.map((pt, idx) => {
                            const x =
                              ctrHistory.length === 1
                                ? 260
                                : 24 + (idx / Math.max(ctrHistory.length - 1, 1)) * 452;
                            const y = 74 - Math.min(1, Math.max(0, (pt.ctr - 0.35) / 0.3)) * 62;
                            return (
                              <g key={pt.study_id}>
                                <circle
                                  cx={x}
                                  cy={y}
                                  r="4.2"
                                  fill={pt.ctr > 0.5 ? "var(--accent-alert)" : "var(--accent-clinical)"}
                                />
                                <text
                                  x={x}
                                  y={Math.max(12, y - 9)}
                                  textAnchor="middle"
                                  fill="var(--text-primary)"
                                  fontSize="9.5"
                                  fontFamily="IBM Plex Mono"
                                >
                                  {pt.ctr.toFixed(3)}
                                </text>
                              </g>
                            );
                          })}
                        </svg>
                      </div>
                    </SpotlightCard>
                  )}

                  {/* Archived Studies & Real AI Dictation Transcripts */}
                  <SpotlightCard className="rounded-[26px] p-6 space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <FileAudio className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                        <h3 className="font-display text-base font-bold">
                          Archived Diagnostic Studies &amp; AI Transcripts ({activeUserDetail.studies.length})
                        </h3>
                      </div>
                      <span className="font-mono-tech text-[10px]" style={{ color: "var(--text-muted)" }}>
                        AUTO-SAVED ON EVERY SCAN
                      </span>
                    </div>

                    {activeUserDetail.studies.length === 0 ? (
                      <div
                        className="rounded-2xl border py-10 text-center text-xs"
                        style={{
                          borderColor: "var(--border-subtle)",
                          color: "var(--text-muted)",
                        }}
                      >
                        No diagnostic studies archived for {activeUserDetail.full_name} yet. Upload
                        or run a scan in Scan Studio to archive automatically.
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {activeUserDetail.studies.map((st: ArchivedStudySummary) => (
                          <div
                            key={st.id}
                            className="rounded-2xl border p-4 space-y-2.5 transition-colors hover:border-teal-500/40"
                            style={{
                              borderColor: "var(--border-subtle)",
                              backgroundColor: "var(--bg-secondary)",
                            }}
                          >
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <div className="flex flex-wrap items-center gap-2">
                                <span className="font-display text-sm font-bold">
                                  {st.study_name}
                                </span>
                                <span
                                  className="rounded-full border px-2 py-0.5 font-mono-tech text-[9.5px] uppercase"
                                  style={{
                                    borderColor: st.critical_alert
                                      ? "var(--accent-alert)"
                                      : "var(--accent-clinical)",
                                    color: st.critical_alert
                                      ? "var(--accent-alert)"
                                      : "var(--accent-clinical)",
                                  }}
                                >
                                  {st.triage_level}
                                </span>
                                <span
                                  className="rounded-full border px-2 py-0.5 font-mono-tech text-[9.5px]"
                                  style={{
                                    borderColor: "var(--border-subtle)",
                                    color: "var(--text-secondary)",
                                  }}
                                >
                                  {st.report_source}
                                </span>
                                {st.signed_at && (
                                  <span
                                    className="inline-flex items-center gap-1 font-mono-tech text-[9.5px]"
                                    style={{ color: "var(--accent-clinical)" }}
                                  >
                                    <CheckCircle2 className="h-3 w-3" />
                                    Signed
                                  </span>
                                )}
                              </div>

                              <div className="flex items-center gap-2">
                                <span
                                  className="flex items-center gap-1 font-mono-tech text-[10px]"
                                  style={{ color: "var(--text-muted)" }}
                                >
                                  <Clock className="h-3 w-3" />
                                  {new Date(st.created_at).toLocaleString()}
                                </span>
                                <button
                                  type="button"
                                  onClick={() => onLoadStudy(st.id)}
                                  className="cursor-pointer rounded-full border px-3 py-1 font-mono-tech text-[10px] font-medium transition-colors hover:border-teal-400"
                                  style={{
                                    borderColor: "var(--border-strong)",
                                    color: "var(--accent-clinical)",
                                  }}
                                >
                                  Load Transcript
                                </button>
                                <button
                                  type="button"
                                  onClick={() => onDeleteStudy(st.id)}
                                  aria-label="Delete archived study"
                                  className="cursor-pointer p-1 opacity-60 hover:opacity-100"
                                  style={{ color: "var(--text-muted)" }}
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                </button>
                              </div>
                            </div>

                            <div className="flex flex-wrap gap-3 font-mono-tech text-[10.5px]" style={{ color: "var(--text-secondary)" }}>
                              {st.top_pathology && (
                                <span>
                                  Lead: <strong>{formatPathology(st.top_pathology)}</strong>{" "}
                                  {st.top_score != null ? `(${(st.top_score * 100).toFixed(1)}%)` : ""}
                                </span>
                              )}
                              {st.ctr != null && <span>· CTR {st.ctr.toFixed(3)}</span>}
                              {st.vision_latency_ms > 0 && (
                                <span>· Vision {Math.round(st.vision_latency_ms)} ms</span>
                              )}
                            </div>

                            {st.dictation_transcript && (
                              <p
                                className="line-clamp-2 border-l-2 pl-3 text-xs leading-relaxed italic"
                                style={{
                                  borderColor: "var(--accent-clinical)",
                                  color: "var(--text-secondary)",
                                }}
                              >
                                “{st.dictation_transcript}”
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </SpotlightCard>
                </>
              )}
            </div>
          </motion.div>
        ) : (
          /* TAB 2: LIVE VERIFIED ML EVALUATION & BENCHMARK OBSERVATORY */
          <motion.div
            key="evaluations"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.24 }}
            className="space-y-7"
          >
            {!headline ? (
              <SpotlightCard className="rounded-[26px] p-12 text-center text-sm" style={{ color: "var(--text-muted)" }}>
                Evaluation report is generating or not yet saved. Run{" "}
                <code>python scripts/run_full_evaluation.py</code>.
              </SpotlightCard>
            ) : (
              <>
                {/* 6 Headline KPI Metric Cards */}
                <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
                  {[
                    { label: "Overall Accuracy", val: headline.accuracy * 100, dec: 1, suf: "%" },
                    { label: "Micro ROC-AUC", val: headline.micro_roc_auc, dec: 3, suf: "" },
                    { label: "Macro ROC-AUC", val: headline.macro_roc_auc, dec: 3, suf: "" },
                    { label: "Specificity", val: headline.specificity * 100, dec: 1, suf: "%" },
                    { label: "F1-Score", val: headline.f1_score, dec: 3, suf: "" },
                    { label: "Brier Error (ECE)", val: headline.brier_score, dec: 3, suf: "" },
                  ].map((k) => (
                    <SpotlightCard key={k.label} className="rounded-2xl p-4">
                      <div
                        className="font-mono-tech text-[9.5px] tracking-[0.14em] uppercase"
                        style={{ color: "var(--text-muted)" }}
                      >
                        {k.label}
                      </div>
                      <div
                        className="mt-2 font-mono-tech text-2xl font-semibold"
                        style={{ color: "var(--accent-clinical)" }}
                      >
                        <AnimatedNumber value={k.val} decimals={k.dec} suffix={k.suf} />
                      </div>
                    </SpotlightCard>
                  ))}
                </div>

                {/* Backbone Ablation Comparison + Latency & Live AI Verification */}
                <div className="grid grid-cols-1 gap-7 lg:grid-cols-12">
                  <SpotlightCard strong className="lg:col-span-7 rounded-[26px] p-6 space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Activity className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                        <h2 className="font-display text-base font-bold">
                          Backbone Ablation &amp; Calibration Comparison
                        </h2>
                      </div>
                      <span className="font-mono-tech text-[10px]" style={{ color: "var(--text-muted)" }}>
                        {evaluations?.total_pathology_judgments} JUDGMENTS
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr
                            className="border-b font-mono-tech text-[10px] uppercase"
                            style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
                          >
                            <th className="py-2.5 pr-3">Model / Pipeline</th>
                            <th className="px-2 py-2.5 text-right">Accuracy</th>
                            <th className="px-2 py-2.5 text-right">Micro AUC</th>
                            <th className="px-2 py-2.5 text-right">Spec.</th>
                            <th className="px-2 py-2.5 text-right">F1</th>
                            <th className="py-2.5 pl-2 text-right">Brier</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                          {backboneEntries.map(([key, m]) => {
                            const isFused = key === "fused_consensus";
                            return (
                              <tr
                                key={key}
                                style={{
                                  backgroundColor: isFused ? "var(--accent-clinical-dim)" : "transparent",
                                  borderColor: "var(--border-subtle)",
                                }}
                              >
                                <td className="py-3 pr-3 font-semibold">
                                  {BACKBONE_LABELS[key] || key}
                                </td>
                                <td className="px-2 py-3 text-right font-mono-tech font-semibold">
                                  {(m.accuracy * 100).toFixed(1)}%
                                </td>
                                <td
                                  className="px-2 py-3 text-right font-mono-tech"
                                  style={{ color: "var(--accent-clinical)" }}
                                >
                                  {m.micro_roc_auc != null ? m.micro_roc_auc.toFixed(3) : "—"}
                                </td>
                                <td className="px-2 py-3 text-right font-mono-tech">
                                  {(m.specificity * 100).toFixed(1)}%
                                </td>
                                <td className="px-2 py-3 text-right font-mono-tech">
                                  {m.f1_score.toFixed(3)}
                                </td>
                                <td
                                  className="py-3 pl-2 text-right font-mono-tech"
                                  style={{ color: "var(--text-secondary)" }}
                                >
                                  {m.brier_score.toFixed(3)}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </SpotlightCard>

                  <SpotlightCard className="lg:col-span-5 rounded-[26px] p-6 space-y-5">
                    <div className="flex items-center gap-2">
                      <Sparkles className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                      <h2 className="font-display text-base font-bold">
                        Live AI &amp; Latency SLA Audit
                      </h2>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      {[
                        {
                          label: "Live Gemini Reports",
                          val: `${evaluations?.ai_verification?.live_gemini_transcripts ?? 0} / ${evaluations?.ai_verification?.total_cases ?? 6}`,
                        },
                        {
                          label: "Decision 1 Active",
                          val: `${evaluations?.ai_verification?.gemini_decision_active_cases ?? 0} / ${evaluations?.ai_verification?.total_cases ?? 6}`,
                        },
                        {
                          label: "Avg Core GPU Vision",
                          val: `${Math.round(evaluations?.latency_sla?.avg_core_gpu_ms ?? 0)} ms`,
                        },
                        {
                          label: "Avg Decision 1 API",
                          val: `${Math.round(evaluations?.latency_sla?.avg_decision_1_ms ?? 0)} ms`,
                        },
                      ].map((item) => (
                        <div
                          key={item.label}
                          className="rounded-2xl border p-3.5"
                          style={{
                            borderColor: "var(--border-subtle)",
                            backgroundColor: "var(--bg-secondary)",
                          }}
                        >
                          <div className="font-mono-tech text-[9.5px] uppercase" style={{ color: "var(--text-muted)" }}>
                            {item.label}
                          </div>
                          <div className="mt-1 font-mono-tech text-lg font-semibold">
                            {item.val}
                          </div>
                        </div>
                      ))}
                    </div>

                    {headline.confusion && (
                      <div
                        className="rounded-2xl border p-4 font-mono-tech text-xs"
                        style={{
                          borderColor: "var(--border-subtle)",
                          backgroundColor: "var(--bg-secondary)",
                        }}
                      >
                        <div className="mb-2 text-[10px] uppercase" style={{ color: "var(--text-muted)" }}>
                          CONFUSION MATRIX (THRESHOLD 0.50)
                        </div>
                        <div className="grid grid-cols-4 gap-2 text-center">
                          <div>
                            <div style={{ color: "var(--accent-clinical)" }} className="text-base font-bold">
                              {headline.confusion.tp}
                            </div>
                            <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>TP</div>
                          </div>
                          <div>
                            <div className="text-base font-bold">{headline.confusion.tn}</div>
                            <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>TN</div>
                          </div>
                          <div>
                            <div style={{ color: "var(--accent-alert)" }} className="text-base font-bold">
                              {headline.confusion.fp}
                            </div>
                            <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>FP</div>
                          </div>
                          <div>
                            <div style={{ color: "var(--accent-amber)" }} className="text-base font-bold">
                              {headline.confusion.fn}
                            </div>
                            <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>FN</div>
                          </div>
                        </div>
                      </div>
                    )}
                  </SpotlightCard>
                </div>

                {/* Per-Case Verified Real AI Transcript Matrix */}
                <SpotlightCard strong className="rounded-[26px] p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <h2 className="font-display text-base font-bold">
                      Per-Case Benchmark &amp; Verified Real Gemini Transcripts
                    </h2>
                    <span className="font-mono-tech text-[10px]" style={{ color: "var(--text-muted)" }}>
                      {evaluations?.gpu_name} · {evaluations?.ai_verification?.flash_model}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    {(evaluations?.cases || []).map((c) => (
                      <div
                        key={c.case_id}
                        className="rounded-2xl border p-4 space-y-2.5"
                        style={{
                          borderColor: "var(--border-subtle)",
                          backgroundColor: "var(--bg-secondary)",
                        }}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div>
                            <span className="font-mono-tech text-[10px]" style={{ color: "var(--accent-clinical)" }}>
                              {c.case_id}
                            </span>
                            <div className="font-display text-sm font-bold">{c.name}</div>
                          </div>
                          <span
                            className="rounded-full border px-2.5 py-0.5 font-mono-tech text-[9.5px]"
                            style={{
                              borderColor: "var(--accent-clinical)",
                              backgroundColor: "var(--accent-clinical-dim)",
                              color: "var(--accent-clinical)",
                            }}
                          >
                            {c.report_source}
                          </span>
                        </div>

                        <div className="flex flex-wrap gap-2 font-mono-tech text-[10px]" style={{ color: "var(--text-secondary)" }}>
                          <span>GT: {c.ground_truth.length ? c.ground_truth.join(", ") : "Normal"}</span>
                          <span>·</span>
                          <span>AI: {c.detected.length ? c.detected.join(", ") : "Clear"}</span>
                          {c.ctr != null && (
                            <>
                              <span>·</span>
                              <span>CTR {c.ctr.toFixed(3)}</span>
                            </>
                          )}
                        </div>

                        <p
                          className="border-l-2 pl-3 text-xs leading-relaxed"
                          style={{
                            borderColor: "var(--accent-clinical)",
                            color: "var(--text-secondary)",
                          }}
                        >
                          {c.dictation_transcript || c.impression}
                        </p>
                      </div>
                    ))}
                  </div>
                </SpotlightCard>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
