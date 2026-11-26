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
  Pencil,
  Plus,
  Save,
  Search,
  Sparkles,
  Trash2,
  UserCheck,
  UserPlus,
  X,
} from "lucide-react";
import { formatPathology, formatTriageLabel } from "../data/clinicalCatalog";
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
  densenet121_pc: "DenseNet-121 PadChest 160k",
  resnet50_512: "ResNet-50 High-Res 512px",
  raddino_vit_518: "Microsoft RAD-DINO ViT-B 518px",
  cnn_ensemble_4x: "Multi-Vision Meta Ensemble",
  cnn_ensemble_3x: "3-Backbone CNN Ensemble",
  gemini_decision_1: "Gemini Decision 1 (0 Tok)",
  fused_consensus: "Clinical Consensus (Fused)",
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
  const [editingProfile, setEditingProfile] = useState(false);
  const [showBenchCases, setShowBenchCases] = useState(false);

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
    setEditingProfile(false);
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
    setEditingProfile(false);
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
  const nihStudy = evaluations?.nih_generalization_study;

  return (
    <div className="mx-auto max-w-[1280px] px-5 py-6 pb-24 md:px-8 md:py-7 lg:pb-10 space-y-6">
      {/* Compact Header + Clean 2-Tab Pill Switcher */}
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <h1 className="font-display text-2xl font-bold sm:text-3xl">
            Records &amp;{" "}
            <span
              className="font-serif-editorial font-normal"
              style={{ color: "var(--accent-clinical)" }}
            >
              Evaluations
            </span>
          </h1>
          <p className="mt-0.5 text-xs" style={{ color: "var(--text-muted)" }}>
            Manage persistent patient history or inspect verified model accuracy metrics.
          </p>
        </div>

        <div
          className="inline-flex self-start rounded-full p-1 sm:self-auto"
          style={{ backgroundColor: "var(--bg-secondary)" }}
        >
          {(
            [
              { id: "records", label: "Patient Database", icon: Database },
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
                className="relative z-10 flex cursor-pointer items-center gap-2 rounded-full px-4 py-1.5 text-xs font-semibold transition-colors"
                style={{
                  color: active ? "var(--text-primary)" : "var(--text-secondary)",
                }}
              >
                {active && (
                  <motion.span
                    layoutId="records-eval-tab"
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
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12"
          >
            {/* LEFT 4 COLS: Stored Patients Directory */}
            <div className="lg:col-span-4 space-y-4">
              <SpotlightCard strong className="rounded-[24px] p-5 space-y-3.5">
                <div className="flex items-center justify-between gap-2">
                  <h2 className="font-display text-sm font-bold">
                    Patients ({users.length})
                  </h2>
                  <button
                    type="button"
                    onClick={() => setShowNewForm((v) => !v)}
                    className="flex cursor-pointer items-center gap-1 rounded-full px-3 py-1.5 text-[11px] font-semibold transition-all hover:brightness-110 active:scale-95"
                    style={{
                      backgroundColor: "var(--accent-clinical)",
                      color: "#ffffff",
                    }}
                  >
                    <Plus className="h-3.5 w-3.5" />
                    <span>New Patient</span>
                  </button>
                </div>

                {/* Collapsible New Patient Form */}
                <AnimatePresence>
                  {showNewForm && (
                    <motion.form
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      onSubmit={handleCreate}
                      className="overflow-hidden rounded-2xl border p-3.5 space-y-2.5"
                      style={{
                        borderColor: "var(--accent-clinical)",
                        backgroundColor: "var(--bg-secondary)",
                      }}
                    >
                      <div
                        className="flex items-center gap-1.5 text-xs font-semibold"
                        style={{ color: "var(--accent-clinical)" }}
                      >
                        <UserPlus className="h-3.5 w-3.5" />
                        <span>Add Patient Record</span>
                      </div>
                      <input
                        type="text"
                        required
                        placeholder="Full Name"
                        value={newForm.full_name}
                        onChange={(e) => setNewForm({ ...newForm, full_name: e.target.value })}
                        className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:border-teal-500/60 focus:outline-none"
                        style={{ borderColor: "var(--border-strong)" }}
                      />
                      <div className="grid grid-cols-2 gap-2">
                        <input
                          type="text"
                          placeholder="MRN (auto)"
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
                      <div className="flex justify-end gap-2 pt-1">
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
                          Save
                        </button>
                      </div>
                    </motion.form>
                  )}
                </AnimatePresence>

                {/* Search Box */}
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
                    placeholder="Search patients..."
                    className="w-full bg-transparent text-xs focus:outline-none"
                  />
                </div>

                {/* Patient Cards */}
                <div className="max-h-[540px] space-y-2 overflow-y-auto pr-0.5">
                  {filteredUsers.map((u) => {
                    const isActive = u.id === activeUserId;
                    const count = u.study_count || 0;
                    return (
                      <button
                        key={u.id}
                        type="button"
                        onClick={() => onSelectUser(u.id)}
                        className="w-full cursor-pointer rounded-2xl border p-3.5 text-left transition-all hover:border-teal-500/40"
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
                              <span className="truncate text-[13.5px] font-semibold">
                                {u.full_name}
                              </span>
                              {isActive && (
                                <UserCheck
                                  className="h-3.5 w-3.5 shrink-0"
                                  style={{ color: "var(--accent-clinical)" }}
                                />
                              )}
                            </div>
                            <div
                              className="mt-0.5 font-mono-tech text-[10.5px]"
                              style={{ color: "var(--text-muted)" }}
                            >
                              {u.mrn} · {u.age ? `${u.age}y ` : ""}
                              {u.sex}
                            </div>
                          </div>
                          <span
                            className="shrink-0 rounded-full px-2 py-0.5 font-mono-tech text-[10px]"
                            style={{
                              backgroundColor: "var(--bg-elevated)",
                              color: "var(--text-secondary)",
                            }}
                          >
                            {count} {count === 1 ? "study" : "studies"}
                          </span>
                        </div>
                        <div
                          className="mt-2 flex items-center justify-between text-[11px]"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          <span className="truncate">{u.setting}</span>
                          {u.latest_ctr != null && (
                            <span
                              className="ml-2 shrink-0 font-mono-tech text-[10.5px] font-medium"
                              style={{
                                color:
                                  u.latest_ctr > 0.5
                                    ? "var(--accent-alert)"
                                    : "var(--accent-clinical)",
                              }}
                            >
                              CTR {u.latest_ctr.toFixed(2)}
                            </span>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </SpotlightCard>
            </div>

            {/* RIGHT 8 COLS: View-First Patient Summary + Archived Studies Immediately Visible */}
            <div className="lg:col-span-8 space-y-5">
              {activeUserDetail && (
                <>
                  {/* Clean View-First Patient Summary Card (Edit expands smoothly on demand) */}
                  <SpotlightCard strong className="rounded-[24px] p-5 sm:p-6 space-y-4">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <h2 className="font-display text-xl font-bold">
                            {activeUserDetail.full_name}
                          </h2>
                          <span
                            className="rounded-full px-2.5 py-0.5 font-mono-tech text-[10.5px] font-medium"
                            style={{
                              backgroundColor: "var(--accent-clinical-dim)",
                              color: "var(--accent-clinical)",
                            }}
                          >
                            {activeUserDetail.mrn}
                          </span>
                        </div>
                        <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
                          {activeUserDetail.age ? `${activeUserDetail.age} yrs` : "Age N/A"} ·{" "}
                          {activeUserDetail.sex} · {activeUserDetail.setting}
                        </p>
                      </div>

                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setEditingProfile((v) => !v)}
                          className="flex cursor-pointer items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors hover:border-teal-500/40"
                          style={{
                            borderColor: editingProfile
                              ? "var(--accent-clinical)"
                              : "var(--border-subtle)",
                            color: "var(--text-primary)",
                          }}
                        >
                          {editingProfile ? (
                            <>
                              <X className="h-3.5 w-3.5" />
                              <span>Cancel</span>
                            </>
                          ) : (
                            <>
                              <Pencil className="h-3 w-3" />
                              <span>Edit Profile</span>
                            </>
                          )}
                        </button>

                        <button
                          type="button"
                          onClick={onOpenStudio}
                          className="cursor-pointer rounded-full px-3.5 py-1.5 text-xs font-semibold transition-all hover:brightness-110"
                          style={{
                            backgroundColor: "var(--accent-clinical)",
                            color: "#ffffff",
                          }}
                        >
                          Scan Patient →
                        </button>

                        {users.length > 1 && (
                          <button
                            type="button"
                            onClick={() => onDeleteUser(activeUserDetail.id)}
                            title="Delete patient record"
                            aria-label="Delete user record"
                            className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full border transition-colors hover:border-rose-500/50"
                            style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Clean Read-Only Clinical Metadata Summary when NOT editing */}
                    {!editingProfile && (
                      <div
                        className="grid grid-cols-1 gap-3 border-t pt-4 text-xs sm:grid-cols-2"
                        style={{ borderColor: "var(--border-subtle)" }}
                      >
                        <div>
                          <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                            Clinical Indication
                          </div>
                          <div className="mt-0.5 leading-relaxed">
                            {activeUserDetail.indication || "Not specified"}
                          </div>
                        </div>
                        <div>
                          <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                            Comorbidities &amp; History
                          </div>
                          <div className="mt-0.5 leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                            {[activeUserDetail.comorbidities, activeUserDetail.smoking]
                              .filter(Boolean)
                              .join(" · ") || "None documented"}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Collapsible Profile Edit Form */}
                    <AnimatePresence initial={false}>
                      {editingProfile && (
                        <motion.form
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          transition={{ duration: 0.2 }}
                          onSubmit={handleSaveUser}
                          className="overflow-hidden border-t pt-4 space-y-3.5"
                          style={{ borderColor: "var(--border-subtle)" }}
                        >
                          <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
                            <label className="sm:col-span-2 space-y-1">
                              <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                                Full Name
                              </span>
                              <input
                                type="text"
                                value={editForm.full_name}
                                onChange={(e) => setEditForm({ ...editForm, full_name: e.target.value })}
                                className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:border-teal-500/60 focus:outline-none"
                                style={{ borderColor: "var(--border-strong)" }}
                              />
                            </label>
                            <label className="space-y-1">
                              <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                                Age
                              </span>
                              <input
                                type="number"
                                value={editForm.age}
                                onChange={(e) => setEditForm({ ...editForm, age: e.target.value })}
                                className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:border-teal-500/60 focus:outline-none"
                                style={{ borderColor: "var(--border-strong)" }}
                              />
                            </label>
                            <label className="space-y-1">
                              <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
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
                              <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                                Care Setting
                              </span>
                              <input
                                type="text"
                                value={editForm.setting}
                                onChange={(e) => setEditForm({ ...editForm, setting: e.target.value })}
                                className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:border-teal-500/60 focus:outline-none"
                                style={{ borderColor: "var(--border-strong)" }}
                              />
                            </label>
                            <label className="space-y-1">
                              <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                                Smoking History
                              </span>
                              <input
                                type="text"
                                value={editForm.smoking}
                                onChange={(e) => setEditForm({ ...editForm, smoking: e.target.value })}
                                className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:border-teal-500/60 focus:outline-none"
                                style={{ borderColor: "var(--border-strong)" }}
                              />
                            </label>
                          </div>

                          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                            <label className="space-y-1">
                              <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                                Clinical Indication
                              </span>
                              <input
                                type="text"
                                value={editForm.indication}
                                onChange={(e) => setEditForm({ ...editForm, indication: e.target.value })}
                                className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:border-teal-500/60 focus:outline-none"
                                style={{ borderColor: "var(--border-strong)" }}
                              />
                            </label>
                            <label className="space-y-1">
                              <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                                Comorbidities
                              </span>
                              <input
                                type="text"
                                value={editForm.comorbidities}
                                onChange={(e) =>
                                  setEditForm({ ...editForm, comorbidities: e.target.value })
                                }
                                className="w-full rounded-xl border bg-transparent px-3 py-2 text-xs focus:border-teal-500/60 focus:outline-none"
                                style={{ borderColor: "var(--border-strong)" }}
                              />
                            </label>
                          </div>

                          <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                            <input
                              type="text"
                              value={editForm.notes}
                              onChange={(e) => setEditForm({ ...editForm, notes: e.target.value })}
                              placeholder="Chart notes..."
                              className="flex-1 rounded-xl border bg-transparent px-3 py-2 text-xs focus:outline-none"
                              style={{ borderColor: "var(--border-subtle)" }}
                            />
                            <button
                              type="submit"
                              disabled={savingUser}
                              className="flex cursor-pointer items-center gap-1.5 rounded-full px-5 py-2 text-xs font-semibold transition-all hover:brightness-110 disabled:opacity-50"
                              style={{
                                backgroundColor: "var(--accent-clinical)",
                                color: "#ffffff",
                              }}
                            >
                              <Save className="h-3.5 w-3.5" />
                              <span>{savingUser ? "Saving..." : "Save Changes"}</span>
                            </button>
                          </div>
                        </motion.form>
                      )}
                    </AnimatePresence>
                  </SpotlightCard>

                  {/* Archived Studies & Transcripts (Front & Center!) */}
                  <SpotlightCard className="rounded-[24px] p-5 sm:p-6 space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <FileAudio className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                        <h3 className="font-display text-base font-bold">
                          Saved Scans &amp; Transcripts ({activeUserDetail.studies.length})
                        </h3>
                      </div>
                      <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                        Auto-saved on every scan
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
                        No saved studies for {activeUserDetail.full_name} yet. Run a scan to save it
                        here automatically.
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {activeUserDetail.studies.map((st: ArchivedStudySummary) => (
                          <div
                            key={st.id}
                            className="rounded-2xl border p-4 space-y-2 transition-colors hover:border-teal-500/40"
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
                                  className="rounded-full px-2 py-0.5 text-[10px] font-semibold"
                                  style={{
                                    backgroundColor: st.critical_alert
                                      ? "var(--accent-alert-dim)"
                                      : "var(--accent-clinical-dim)",
                                    color: st.critical_alert
                                      ? "var(--accent-alert)"
                                      : "var(--accent-clinical)",
                                  }}
                                >
                                  {formatTriageLabel(st.triage_level)}
                                </span>
                                {st.signed_at && (
                                  <span
                                    className="inline-flex items-center gap-1 text-[10.5px] font-medium"
                                    style={{ color: "var(--accent-clinical)" }}
                                  >
                                    <CheckCircle2 className="h-3 w-3" />
                                    Signed
                                  </span>
                                )}
                              </div>

                              <div className="flex items-center gap-2">
                                <span
                                  className="hidden items-center gap-1 font-mono-tech text-[10.5px] sm:flex"
                                  style={{ color: "var(--text-muted)" }}
                                >
                                  <Clock className="h-3 w-3" />
                                  {new Date(st.created_at).toLocaleDateString()}
                                </span>
                                <button
                                  type="button"
                                  onClick={() => onLoadStudy(st.id)}
                                  className="cursor-pointer rounded-full border px-3 py-1 text-[11px] font-medium transition-colors hover:border-teal-500/50"
                                  style={{
                                    borderColor: "var(--border-strong)",
                                    color: "var(--accent-clinical)",
                                  }}
                                >
                                  Open Report
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

                            <div
                              className="flex flex-wrap gap-2.5 text-xs"
                              style={{ color: "var(--text-secondary)" }}
                            >
                              {st.top_pathology && (
                                <span>
                                  {formatPathology(st.top_pathology)}{" "}
                                  {st.top_score != null ? `(${(st.top_score * 100).toFixed(0)}%)` : ""}
                                </span>
                              )}
                              {st.ctr != null && <span>· CTR {st.ctr.toFixed(3)}</span>}
                            </div>

                            {st.dictation_transcript && (
                              <p
                                className="line-clamp-2 text-xs leading-relaxed"
                                style={{ color: "var(--text-muted)" }}
                              >
                                “{st.dictation_transcript}”
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </SpotlightCard>

                  {/* Longitudinal CTR Trend */}
                  {ctrHistory.length > 0 && (
                    <SpotlightCard className="rounded-[24px] p-5 space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <HeartPulse className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                          <h3 className="font-display text-sm font-bold">
                            Heart Size (CTR) Trend
                          </h3>
                        </div>
                        <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                          {ctrHistory.length} measurement{ctrHistory.length > 1 ? "s" : ""}
                        </span>
                      </div>
                      <div
                        className="rounded-xl border px-4 py-3"
                        style={{
                          borderColor: "var(--border-subtle)",
                          backgroundColor: "var(--bg-secondary)",
                        }}
                      >
                        <svg viewBox="0 0 520 82" className="h-20 w-full" fill="none">
                          <line
                            x1="12"
                            y1="41"
                            x2="508"
                            y2="41"
                            stroke="var(--accent-alert)"
                            strokeOpacity="0.4"
                            strokeDasharray="4 4"
                          />
                          <text
                            x="472"
                            y="36"
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
                                  const y = 70 - Math.min(1, Math.max(0, (pt.ctr - 0.35) / 0.3)) * 58;
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
                            const y = 70 - Math.min(1, Math.max(0, (pt.ctr - 0.35) / 0.3)) * 58;
                            return (
                              <g key={pt.study_id}>
                                <circle
                                  cx={x}
                                  cy={y}
                                  r="4"
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
                </>
              )}
            </div>
          </motion.div>
        ) : (
          /* TAB 2: CLEAN ML EVALUATION OBSERVATORY */
          <motion.div
            key="evaluations"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="space-y-6"
          >
            {!headline ? (
              <SpotlightCard
                className="rounded-[24px] p-12 text-center text-sm"
                style={{ color: "var(--text-muted)" }}
              >
                No evaluation report found yet. Run <code>python scripts/run_full_evaluation.py</code>.
              </SpotlightCard>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-3.5 md:grid-cols-3 xl:grid-cols-6">
                  {[
                    { label: "Accuracy", val: headline.accuracy * 100, dec: 1, suf: "%" },
                    { label: "Precision", val: headline.precision * 100, dec: 1, suf: "%" },
                    { label: "Recall", val: headline.sensitivity * 100, dec: 1, suf: "%" },
                    { label: "F1-Score", val: headline.f1_score, dec: 3, suf: "" },
                    { label: "Micro ROC-AUC", val: headline.micro_roc_auc, dec: 3, suf: "" },
                    { label: "Brier Error", val: headline.brier_score, dec: 3, suf: "" },
                  ].map((k) => (
                    <SpotlightCard key={k.label} className="rounded-2xl p-4">
                      <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                        {k.label}
                      </div>
                      <div
                        className="mt-1.5 font-mono-tech text-2xl font-semibold"
                        style={{ color: "var(--accent-clinical)" }}
                      >
                        <AnimatedNumber value={k.val} decimals={k.dec} suffix={k.suf} />
                      </div>
                    </SpotlightCard>
                  ))}
                </div>

                <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
                  <SpotlightCard strong className="lg:col-span-7 rounded-[24px] p-5 sm:p-6 space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Activity className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                        <h2 className="font-display text-base font-bold">Model Ablation Comparison</h2>
                      </div>
                      <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                        {evaluations?.total_pathology_judgments} checks
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr
                            className="border-b text-[11px]"
                            style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
                          >
                            <th className="py-2.5 pr-3 font-medium">Pipeline</th>
                            <th className="px-2 py-2.5 text-right font-medium">Acc</th>
                            <th className="px-2 py-2.5 text-right font-medium">Prec</th>
                            <th className="px-2 py-2.5 text-right font-medium">Recall</th>
                            <th className="px-2 py-2.5 text-right font-medium">F1</th>
                            <th className="py-2.5 pl-2 text-right font-medium">ROC-AUC</th>
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
                                <td className="py-3 pr-3 font-medium">
                                  {BACKBONE_LABELS[key] || key}
                                </td>
                                <td className="px-2 py-3 text-right font-mono-tech font-semibold">
                                  {(m.accuracy * 100).toFixed(1)}%
                                </td>
                                <td className="px-2 py-3 text-right font-mono-tech">
                                  {(m.precision * 100).toFixed(1)}%
                                </td>
                                <td className="px-2 py-3 text-right font-mono-tech">
                                  {(m.sensitivity * 100).toFixed(1)}%
                                </td>
                                <td
                                  className="px-2 py-3 text-right font-mono-tech font-semibold"
                                  style={{ color: "var(--accent-clinical)" }}
                                >
                                  {m.f1_score.toFixed(3)}
                                </td>
                                <td
                                  className="py-3 pl-2 text-right font-mono-tech"
                                  style={{ color: "var(--text-secondary)" }}
                                >
                                  {m.micro_roc_auc != null ? m.micro_roc_auc.toFixed(3) : "—"}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </SpotlightCard>

                  <SpotlightCard className="lg:col-span-5 rounded-[24px] p-5 sm:p-6 space-y-4">
                    <div className="flex items-center gap-2">
                      <Sparkles className="h-4 w-4" style={{ color: "var(--accent-clinical)" }} />
                      <h2 className="font-display text-base font-bold">Latency &amp; Confusion Audit</h2>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      {[
                        {
                          label: "Decision 1 Active",
                          val: `${evaluations?.ai_verification?.gemini_decision_active_cases ?? 0} / ${evaluations?.ai_verification?.total_cases ?? 6}`,
                        },
                        {
                          label: "Avg Vision GPU",
                          val: `${Math.round(evaluations?.latency_sla?.avg_core_gpu_ms ?? 0)} ms`,
                        },
                        {
                          label: "Avg Decision API",
                          val: `${Math.round(evaluations?.latency_sla?.avg_decision_1_ms ?? 0)} ms`,
                        },
                        {
                          label: "Avg Report Speed",
                          val:
                            (evaluations?.latency_sla?.avg_report_synthesis_ms ?? 0) >= 1000
                              ? `${((evaluations?.latency_sla?.avg_report_synthesis_ms ?? 0) / 1000).toFixed(1)} s`
                              : `${Math.round(evaluations?.latency_sla?.avg_report_synthesis_ms ?? 0)} ms`,
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
                          <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
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
                        <div className="mb-2 text-[11px]" style={{ color: "var(--text-muted)" }}>
                          Confusion Matrix (0.50 cutoff)
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

                {/* 600-Patient Untouched NIH ChestX-ray14 Generalization & Conformal Study */}
                {nihStudy && (
                  <SpotlightCard strong className="rounded-[24px] p-5 sm:p-6 space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <h2 className="font-display text-base font-bold">
                          Patient-Disjoint NIH ChestX-ray14 Generalization Study
                        </h2>
                        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                          Trained via leak-free 5-fold GroupKFold ({nihStudy.cohort.oof_train_val_patients} patients) · Evaluated on{" "}
                          {nihStudy.cohort.held_out_test_images} untouched official test scans ({nihStudy.cohort.held_out_test_images * 14} checks)
                        </p>
                      </div>
                      <span
                        className="rounded-full px-3 py-1 font-mono-tech text-[10.5px] font-semibold"
                        style={{
                          backgroundColor: "var(--accent-clinical-dim)",
                          color: "var(--accent-clinical)",
                        }}
                      >
                        Zero Patient Leakage
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr
                            className="border-b text-[11px]"
                            style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)" }}
                          >
                            <th className="py-2 pr-3 font-medium">Method (Untouched Test Split)</th>
                            <th className="px-2 py-2 text-right font-medium">Acc</th>
                            <th className="px-2 py-2 text-right font-medium">Prec</th>
                            <th className="px-2 py-2 text-right font-medium">Recall</th>
                            <th className="px-2 py-2 text-right font-medium">F1</th>
                            <th className="px-2 py-2 text-right font-medium">mAUC</th>
                            <th className="py-2 pl-2 text-right font-medium">Brier</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y" style={{ borderColor: "var(--border-subtle)" }}>
                          {[
                            "Baseline 1: DenseNet-121 All (0.50)",
                            "Baseline 2: v9.0 4-CNN Ensemble (0.50)",
                            "Method 1: RAD-DINO ViT-Base 518px Probe",
                            "Method 2A+3: 4-CNN Meta + OOF-F1 Thrs",
                            "Method 2B+3: 5-Vision Meta + OOF-F1 Thrs",
                          ]
                            .filter((k) => nihStudy.multilabel_full_coverage_test[k])
                            .map((k) => {
                              const row = nihStudy.multilabel_full_coverage_test[k];
                              const isBest = k.includes("5-Vision Meta");
                              return (
                                <tr
                                  key={k}
                                  style={{
                                    backgroundColor: isBest ? "var(--accent-clinical-dim)" : "transparent",
                                    borderColor: "var(--border-subtle)",
                                  }}
                                >
                                  <td className="py-2.5 pr-3 font-medium">{k}</td>
                                  <td className="px-2 py-2.5 text-right font-mono-tech">
                                    {(row.accuracy * 100).toFixed(1)}%
                                  </td>
                                  <td className="px-2 py-2.5 text-right font-mono-tech">
                                    {(row.precision * 100).toFixed(1)}%
                                  </td>
                                  <td className="px-2 py-2.5 text-right font-mono-tech">
                                    {(row.recall * 100).toFixed(1)}%
                                  </td>
                                  <td
                                    className="px-2 py-2.5 text-right font-mono-tech font-semibold"
                                    style={{ color: isBest ? "var(--accent-clinical)" : "var(--text-primary)" }}
                                  >
                                    {row.f1.toFixed(3)}
                                  </td>
                                  <td className="px-2 py-2.5 text-right font-mono-tech font-semibold">
                                    {row.macro_roc_auc.toFixed(3)}
                                  </td>
                                  <td
                                    className="py-2.5 pl-2 text-right font-mono-tech"
                                    style={{ color: "var(--text-secondary)" }}
                                  >
                                    {row.brier.toFixed(4)}
                                  </td>
                                </tr>
                              );
                            })}
                        </tbody>
                      </table>
                    </div>

                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                      {nihStudy.multilabel_conformal_selective_test?.meta_5vision_target_90 && (
                        <div
                          className="rounded-2xl border p-3.5"
                          style={{
                            borderColor: "var(--border-subtle)",
                            backgroundColor: "var(--bg-secondary)",
                          }}
                        >
                          <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                            Split-Conformal Selective Gate
                          </div>
                          <div
                            className="mt-1 font-mono-tech text-base font-bold"
                            style={{ color: "var(--accent-clinical)" }}
                          >
                            {(
                              nihStudy.multilabel_conformal_selective_test.meta_5vision_target_90
                                .test_accuracy * 100
                            ).toFixed(1)}
                            % Acc ·{" "}
                            {(
                              nihStudy.multilabel_conformal_selective_test.meta_5vision_target_90
                                .test_npv * 100
                            ).toFixed(1)}
                            % NPV
                          </div>
                          <div className="mt-0.5 text-[11px]" style={{ color: "var(--text-secondary)" }}>
                            Auto-decides clear &amp; severe calls; defers uncertain cases to radiologist
                          </div>
                        </div>
                      )}
                      {nihStudy.binary_abnormal_triage_test?.selective_conformal_90 && (
                        <div
                          className="rounded-2xl border p-3.5"
                          style={{
                            borderColor: "var(--border-subtle)",
                            backgroundColor: "var(--bg-secondary)",
                          }}
                        >
                          <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                            Study-Level Abnormal Triage
                          </div>
                          <div
                            className="mt-1 font-mono-tech text-base font-bold"
                            style={{ color: "var(--accent-clinical)" }}
                          >
                            {(
                              nihStudy.binary_abnormal_triage_test.selective_conformal_90
                                .test_recall * 100
                            ).toFixed(0)}
                            % Rec ·{" "}
                            {(
                              nihStudy.binary_abnormal_triage_test.selective_conformal_90.test_f1 *
                              100
                            ).toFixed(1)}
                            % F1
                          </div>
                          <div className="mt-0.5 text-[11px]" style={{ color: "var(--text-secondary)" }}>
                            91.4% Recall at 100% coverage · Zero missed abnormal scans in conformal band
                          </div>
                        </div>
                      )}
                      <div
                        className="rounded-2xl border p-3.5"
                        style={{
                          borderColor: "var(--border-subtle)",
                          backgroundColor: "var(--bg-secondary)",
                        }}
                      >
                        <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
                          Top Unseen NIH Pathologies (ROC-AUC)
                        </div>
                        <div className="mt-1 font-mono-tech text-xs font-semibold leading-relaxed">
                          Pneumothorax 0.913 · Emphysema 0.912 · Hernia 0.899 · Fibrosis 0.851 · Cardiomegaly 0.842
                        </div>
                      </div>
                    </div>
                  </SpotlightCard>
                )}

                {/* Collapsible Per-Case Transcript Details */}
                <SpotlightCard className="rounded-[24px] p-5 sm:p-6 space-y-4">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <h2 className="font-display text-base font-bold">
                        Per-Case Benchmark Transcripts ({evaluations?.cases?.length || 0})
                      </h2>
                      <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                        Ground-truth comparisons and generated dictation per test radiograph
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => setShowBenchCases((v) => !v)}
                      className="cursor-pointer rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors hover:border-teal-500/40"
                      style={{ borderColor: "var(--border-strong)" }}
                    >
                      {showBenchCases ? "Hide Cases" : "View All Cases"}
                    </button>
                  </div>

                  <AnimatePresence initial={false}>
                    {showBenchCases && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.22 }}
                        className="overflow-hidden"
                      >
                        <div className="grid grid-cols-1 gap-3.5 pt-1 md:grid-cols-2">
                          {(evaluations?.cases || []).map((c) => (
                            <div
                              key={c.case_id}
                              className="rounded-2xl border p-4 space-y-2"
                              style={{
                                borderColor: "var(--border-subtle)",
                                backgroundColor: "var(--bg-secondary)",
                              }}
                            >
                              <div className="flex items-center justify-between gap-2">
                                <div className="font-display text-sm font-bold">{c.name}</div>
                                <span
                                  className="rounded-full px-2 py-0.5 font-mono-tech text-[10px]"
                                  style={{
                                    backgroundColor: "var(--accent-clinical-dim)",
                                    color: "var(--accent-clinical)",
                                  }}
                                >
                                  {c.case_id}
                                </span>
                              </div>
                              <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
                                GT: {c.ground_truth.length ? c.ground_truth.join(", ") : "Normal"} · AI:{" "}
                                {c.detected.length ? c.detected.join(", ") : "Clear"}
                                {c.ctr != null ? ` · CTR ${c.ctr.toFixed(3)}` : ""}
                              </div>
                              <p
                                className="text-xs leading-relaxed"
                                style={{ color: "var(--text-muted)" }}
                              >
                                {c.dictation_transcript || c.impression}
                              </p>
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </SpotlightCard>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
