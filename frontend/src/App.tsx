import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, Upload } from "lucide-react";
import { AmbientRadiologyField } from "./components/AmbientRadiologyField";
import { BatchAndSpecsSection } from "./components/BatchAndSpecsSection";
import { ClinicalReportSection } from "./components/ClinicalReportSection";
import { CommandPalette } from "./components/CommandPalette";
import { ConsensusArchitecture } from "./components/ConsensusArchitecture";
import { DiagnosticStudio } from "./components/DiagnosticStudio";
import { GeminiEngineModal } from "./components/GeminiEngineModal";
import { HeroOverview } from "./components/HeroOverview";
import { PatientContextDrawer } from "./components/PatientContextDrawer";
import { PrintConsultationSheet } from "./components/PrintConsultationSheet";
import { RecordsAndEvaluationsPage } from "./components/RecordsAndEvaluationsPage";
import { SidebarNav } from "./components/SidebarNav";
import { TopHeader } from "./components/TopHeader";
import {
  NAV_PAGES,
  PATIENT_PERSONAS,
  cleanSampleTitle,
  type PageId,
} from "./data/clinicalCatalog";
import { api } from "./services/api";
import type {
  AIConfigStatus,
  BatchPredictionResponse,
  ColormapName,
  CopilotMessage,
  EvaluationReport,
  HealthStatus,
  HeatmapData,
  PatientProfile,
  PredictionResponse,
  ReportData,
  SampleCase,
  ThemeMode,
  UserDetail,
  UserRecord,
} from "./types/clinical";

const VALID_PAGES = new Set<PageId>([
  "overview",
  "studio",
  "consensus",
  "report",
  "triage",
  "records",
]);

function getInitialPage(): PageId {
  if (typeof window === "undefined") return "overview";
  const raw = window.location.hash.replace(/^#\/?/, "").toLowerCase() as PageId;
  return VALID_PAGES.has(raw) ? raw : "overview";
}

function userToPatientProfile(u: UserRecord): PatientProfile {
  return {
    age: u.age,
    sex: u.sex || "Unspecified",
    setting: u.setting || "Ambulatory / Outpatient",
    indication: u.indication || "Thoracic diagnostic evaluation",
    comorbidities: u.comorbidities || "",
    smoking: u.smoking || "",
  };
}

export default function App() {
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const saved = typeof window !== "undefined" ? localStorage.getItem("cxr-theme") : null;
    return saved === "light" ? "light" : "dark";
  });

  const [activePage, setActivePage] = useState<PageId>(getInitialPage);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Live Backend Telemetry & Benchmarks
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [aiConfig, setAiConfig] = useState<AIConfigStatus | null>(null);
  const [samples, setSamples] = useState<SampleCase[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationReport | null>(null);

  // Persistent Per-User SQLite Database State
  const [users, setUsers] = useState<UserRecord[]>([]);
  const [activeUserId, setActiveUserId] = useState<string | null>(null);
  const [activeUserDetail, setActiveUserDetail] = useState<UserDetail | null>(null);
  const [savingUser, setSavingUser] = useState(false);

  // Active Study State
  const [activeStudyId, setActiveStudyId] = useState<string | null>(null);
  const [activeStudyRecordId, setActiveStudyRecordId] = useState<string | null>(null);
  const [activeStudyName, setActiveStudyName] = useState<string>("Initializing Study...");
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [predictionData, setPredictionData] = useState<PredictionResponse | null>(null);
  const [report, setReport] = useState<ReportData | null>(null);

  // Interactive Controls
  const [threshold, setThreshold] = useState<number>(0.5);
  const [colormap, setColormap] = useState<ColormapName>("turbo");
  const [activePathology, setActivePathology] = useState<string | null>(null);
  const [heatmapCache, setHeatmapCache] = useState<Record<string, HeatmapData>>({});
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [holdingSpace, setHoldingSpace] = useState(false);

  // Patient & Physician Attestation State
  const [patient, setPatient] = useState<PatientProfile>(PATIENT_PERSONAS["chf-cardio"].profile);
  const [physicianName, setPhysicianName] = useState("Dr. S. Patel, MD");
  const [physicianCreds, setPhysicianCreds] = useState("Board-Certified Thoracic Radiologist");
  const [physicianNotes, setPhysicianNotes] = useState("");
  const [signedAt, setSignedAt] = useState<string | null>(null);

  // Grounded Copilot State (Stateful Interactions API + SQLite Persisted)
  const [copilotMessages, setCopilotMessages] = useState<CopilotMessage[]>([]);
  const [copilotLoading, setCopilotLoading] = useState(false);
  const [lastInteractionId, setLastInteractionId] = useState<string | null>(null);

  // Batch STAT Triage State
  const [batchResult, setBatchResult] = useState<BatchPredictionResponse | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);

  // Modals / Drawers / Toast / Global Drag-and-Drop
  const [patientDrawerOpen, setPatientDrawerOpen] = useState(false);
  const [geminiModalOpen, setGeminiModalOpen] = useState(false);
  const [geminiSaving, setGeminiSaving] = useState(false);
  const [commandOpen, setCommandOpen] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [globalDragOver, setGlobalDragOver] = useState(false);
  const dragDepthRef = useRef(0);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const autoLoadedRef = useRef(false);
  const aiConfigRef = useRef<AIConfigStatus | null>(null);
  const activeUserIdRef = useRef<string | null>(null);

  useEffect(() => {
    aiConfigRef.current = aiConfig;
  }, [aiConfig]);

  useEffect(() => {
    activeUserIdRef.current = activeUserId;
  }, [activeUserId]);

  const notify = useCallback((msg: string) => {
    setToast(msg);
    window.setTimeout(() => {
      setToast((prev) => (prev === msg ? null : prev));
    }, 3400);
  }, []);

  // Refresh User Directory + Active User Detail from SQLite
  const refreshUserDatabase = useCallback(async (targetUserId?: string | null) => {
    try {
      const list = await api.listUsers();
      setUsers(list);
      const chosenId = targetUserId || activeUserIdRef.current || list[0]?.id || null;
      if (chosenId) {
        setActiveUserId(chosenId);
        activeUserIdRef.current = chosenId;
        const detail = await api.getUser(chosenId).catch(() => null);
        if (detail) {
          setActiveUserDetail(detail);
        }
      }
    } catch {
      // non-fatal
    }
  }, []);

  // Select Active User from Sidebar or Records Page
  const handleSelectUser = useCallback(
    async (userId: string) => {
      setActiveUserId(userId);
      activeUserIdRef.current = userId;
      try {
        const detail = await api.getUser(userId);
        setActiveUserDetail(detail);
        setPatient(userToPatientProfile(detail));
        if (detail.preferred_threshold) setThreshold(detail.preferred_threshold);
        notify(`Switched active record to ${detail.full_name} (${detail.mrn})`);
      } catch {
        notify("Could not load selected user record.");
      }
    },
    [notify]
  );

  // Sync theme to document root + optionally persist to active user
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("cxr-theme", theme);
  }, [theme]);

  // Smooth Page Navigation with URL Hash History
  const navigatePage = useCallback((page: PageId) => {
    setActivePage(page);
    if (window.location.hash !== `#${page}`) {
      window.history.pushState(null, "", `#${page}`);
    }
    window.scrollTo({ top: 0, left: 0, behavior: "smooth" });
  }, []);

  useEffect(() => {
    const onHash = () => setActivePage(getInitialPage());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // Core Inference Runner (Auto-Archives to Active User's SQLite Database!)
  const runStudyAnalysis = useCallback(
    async (
      file: File,
      studyId: string | null,
      studyLabel: string,
      patientCtx: PatientProfile,
      cmap: ColormapName,
      cutoff: number,
      explicitUserId?: string | null
    ) => {
      setIsAnalyzing(true);
      setActiveStudyId(studyId);
      setActiveStudyRecordId(null);
      setActiveStudyName(studyLabel);
      setCurrentFile(file);
      setSignedAt(null);
      setCopilotMessages([]);
      setLastInteractionId(null);

      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
      const nextUrl = URL.createObjectURL(file);
      objectUrlRef.current = nextUrl;
      setImageUrl(nextUrl);

      try {
        const cfg = aiConfigRef.current;
        const ownerUserId = explicitUserId !== undefined ? explicitUserId : activeUserIdRef.current;
        const res = await api.predict({
          file,
          threshold: cutoff,
          tta: false,
          useGeminiDecision: cfg?.gemini_decision_enabled ?? true,
          useLlm: cfg?.gemini_flash_enabled ?? true,
          colormap: cmap,
          patient: patientCtx,
          userId: ownerUserId,
          studyName: studyLabel,
          sampleId: studyId,
        });

        setPredictionData(res);
        setReport(res.report);
        if (res.study_id) {
          setActiveStudyRecordId(res.study_id);
        }

        const nextCache: Record<string, HeatmapData> = {};
        (res.heatmaps || []).forEach((hm) => {
          nextCache[hm.pathology] = hm;
        });
        setHeatmapCache(nextCache);

        const lead =
          res.predictions.find((p) => p.score >= cutoff)?.name ||
          res.heatmaps?.[0]?.pathology ||
          res.predictions?.[0]?.name ||
          null;
        setActivePathology(lead);

        if (ownerUserId) {
          refreshUserDatabase(ownerUserId);
        }
      } catch (err) {
        notify(err instanceof Error ? err.message : "Failed to analyze chest radiograph.");
      } finally {
        setIsAnalyzing(false);
      }
    },
    [notify, refreshUserDatabase]
  );

  // Select Benchmark Sample
  const handleSelectSample = useCallback(
    async (sample: SampleCase, openStudio = false) => {
      const nextPatient = sample.suggested_patient || patient;
      if (sample.suggested_patient) {
        setPatient(sample.suggested_patient);
      }
      if (openStudio) {
        navigatePage("studio");
      }
      try {
        const file = await api.fetchSampleAsFile(sample);
        await runStudyAnalysis(
          file,
          sample.id,
          cleanSampleTitle(sample.name),
          nextPatient,
          colormap,
          threshold
        );
      } catch (err) {
        notify(err instanceof Error ? err.message : "Could not load sample radiograph.");
      }
    },
    [colormap, navigatePage, notify, patient, runStudyAnalysis, threshold]
  );

  // Upload Local Film or DICOM
  const handleFileUpload = useCallback(
    async (file: File) => {
      navigatePage("studio");
      await runStudyAnalysis(
        file,
        null,
        file.name.replace(/\.[^.]+$/, ""),
        patient,
        colormap,
        threshold
      );
      notify(`Analyzed & saved ${file.name} to user record`);
    },
    [colormap, navigatePage, notify, patient, runStudyAnalysis, threshold]
  );

  // Initial Bootstrap: Health, AI Config, Samples, Evaluations, SQLite Users & Auto-Load First Study
  useEffect(() => {
    if (autoLoadedRef.current) return;
    autoLoadedRef.current = true;

    Promise.all([
      api.getHealth().catch(() => null),
      api.getAIConfig().catch(() => null),
      api.getSamples().catch(() => [] as SampleCase[]),
      api.listUsers().catch(() => [] as UserRecord[]),
      api.getEvaluations().catch(() => null),
    ]).then(async ([h, cfg, sampleList, userList, evalReport]) => {
      if (h) setHealth(h);
      if (cfg) {
        aiConfigRef.current = cfg;
        setAiConfig(cfg);
      }
      setSamples(sampleList);
      setUsers(userList);
      if (evalReport) setEvaluations(evalReport);

      const defaultUser = userList[0] || null;
      let defaultUserId: string | null = null;
      if (defaultUser) {
        defaultUserId = defaultUser.id;
        setActiveUserId(defaultUser.id);
        activeUserIdRef.current = defaultUser.id;
        api
          .getUser(defaultUser.id)
          .then((d) => setActiveUserDetail(d))
          .catch(() => {});
      }

      if (sampleList.length > 0) {
        const first = sampleList[0];
        const firstPatient =
          defaultUser
            ? userToPatientProfile(defaultUser)
            : first.suggested_patient || PATIENT_PERSONAS["chf-cardio"].profile;
        setPatient(firstPatient);
        api
          .fetchSampleAsFile(first)
          .then((file) => {
            runStudyAnalysis(
              file,
              first.id,
              cleanSampleTitle(first.name),
              firstPatient,
              "turbo",
              0.5,
              defaultUserId
            );
          })
          .catch(() => {});
      }
    });
  }, [runStudyAnalysis]);

  // Global Shortcuts: Ctrl/Cmd+K, 1-6 pages, Space (raw peek), Ctrl+V (paste radiograph)
  useEffect(() => {
    const isTyping = (target: EventTarget | null) => {
      if (!(target instanceof HTMLElement)) return false;
      const tag = target.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCommandOpen((v) => !v);
        return;
      }

      if (!isTyping(e.target) && !commandOpen && !geminiModalOpen && !patientDrawerOpen) {
        if (e.code === "Space") {
          e.preventDefault();
          setHoldingSpace(true);
          return;
        }
        if (!e.metaKey && !e.ctrlKey && !e.altKey) {
          const match = NAV_PAGES.find((p) => p.keyHint === e.key);
          if (match) {
            e.preventDefault();
            navigatePage(match.id);
          }
        }
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") setHoldingSpace(false);
    };

    const onPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        if (it.type.startsWith("image/")) {
          const blob = it.getAsFile();
          if (blob) {
            handleFileUpload(new File([blob], "pasted-radiograph.png", { type: blob.type }));
            break;
          }
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("paste", onPaste);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("paste", onPaste);
    };
  }, [commandOpen, geminiModalOpen, handleFileUpload, navigatePage, patientDrawerOpen]);

  // On-Demand Grad-CAM++ Localization
  const handleSelectPathology = useCallback(
    async (pathology: string) => {
      setActivePathology(pathology);
      if (heatmapCache[pathology] || !currentFile) return;

      setHeatmapLoading(true);
      try {
        const hm = await api.getSingleHeatmap(currentFile, pathology, colormap);
        setHeatmapCache((prev) => ({ ...prev, [pathology]: hm }));
      } catch {
        notify(`Could not render Grad-CAM++ for ${pathology}`);
      } finally {
        setHeatmapLoading(false);
      }
    },
    [colormap, currentFile, heatmapCache, notify]
  );

  // Colormap Update
  const handleColormapChange = useCallback(
    async (nextCmap: ColormapName) => {
      setColormap(nextCmap);
      setHeatmapCache({});
      if (!currentFile || !activePathology) return;
      setHeatmapLoading(true);
      try {
        const hm = await api.getSingleHeatmap(currentFile, activePathology, nextCmap);
        setHeatmapCache({ [activePathology]: hm });
      } catch {
        // ignore
      } finally {
        setHeatmapLoading(false);
      }
    },
    [activePathology, currentFile]
  );

  // Re-Synthesize ACR Clinical Report + Persist Updated Transcript to SQLite
  const handleResynthesize = useCallback(
    async (overridePatient?: PatientProfile) => {
      if (!predictionData) return;
      setReportLoading(true);
      try {
        const scores: Record<string, number> = {};
        predictionData.predictions.forEach((p) => {
          scores[p.name] = p.score;
        });
        const nextReport = await api.synthesizeReport({
          scores,
          threshold,
          patient: overridePatient || patient,
          anatomy: (predictionData.anatomy as unknown as Record<string, unknown>) || null,
          hotspots: Object.values(heatmapCache),
        });
        setReport(nextReport);

        if (activeStudyRecordId) {
          await api
            .updateStudy(activeStudyRecordId, {
              report: nextReport as unknown as Record<string, unknown>,
            })
            .catch(() => {});
          refreshUserDatabase(activeUserIdRef.current);
        }
        notify(`Synthesized & saved report via ${nextReport.source}`);
      } catch (err) {
        notify(err instanceof Error ? err.message : "Report synthesis failed.");
      } finally {
        setReportLoading(false);
      }
    },
    [activeStudyRecordId, heatmapCache, notify, patient, predictionData, refreshUserDatabase, threshold]
  );

  // Save Edited Dictation Transcript to SQLite
  const handleSaveTranscript = useCallback(
    async (nextTranscript: string) => {
      setReport((prev) => (prev ? { ...prev, dictation_transcript: nextTranscript } : prev));
      if (activeStudyRecordId) {
        try {
          await api.updateStudy(activeStudyRecordId, {
            dictation_transcript: nextTranscript,
          });
          await refreshUserDatabase(activeUserIdRef.current);
          notify("Dictation transcript updated in SQLite database");
        } catch {
          notify("Could not update transcript in database.");
        }
      }
    },
    [activeStudyRecordId, notify, refreshUserDatabase]
  );

  // Sign & Attest Report -> Persists to SQLite
  const handleSignReport = useCallback(async () => {
    const stamp = new Date().toLocaleString();
    setSignedAt(stamp);
    if (activeStudyRecordId) {
      await api
        .updateStudy(activeStudyRecordId, {
          physician_name: physicianName,
          physician_creds: physicianCreds,
          physician_notes: physicianNotes,
          signed_at: stamp,
        })
        .catch(() => {});
      refreshUserDatabase(activeUserIdRef.current);
    }
    notify(`Report attested & saved by ${physicianName}`);
  }, [activeStudyRecordId, notify, physicianCreds, physicianName, physicianNotes, refreshUserDatabase]);

  // Grounded Gemini Radiology Copilot (Auto-Archives Q&A to SQLite)
  const handleSendCopilot = useCallback(
    async (question: string) => {
      if (!predictionData) return;
      const userMsg: CopilotMessage = {
        id: `u-${Date.now()}`,
        role: "user",
        content: question,
        timestamp: new Date().toLocaleTimeString(),
      };
      setCopilotMessages((prev) => [...prev, userMsg]);
      setCopilotLoading(true);

      try {
        const cleanAnatomy = predictionData.anatomy
          ? {
              ctr: predictionData.anatomy.ctr,
              ctr_status: predictionData.anatomy.ctr_status,
              left_lung_pct: predictionData.anatomy.left_lung_pct,
              right_lung_pct: predictionData.anatomy.right_lung_pct,
              lung_symmetry_ratio: predictionData.anatomy.lung_symmetry_ratio,
              organs_detected: predictionData.anatomy.organs_detected,
            }
          : null;

        const res = await api.copilotChat({
          question,
          study_context: {
            study_name: activeStudyName,
            predictions: predictionData.predictions.slice(0, 8),
            anatomy: cleanAnatomy,
            decision_summary: predictionData.decision_summary,
            impression: (report || predictionData.report)?.impression,
            triage_level: (report || predictionData.report)?.triage_level,
            patient,
            threshold,
          },
          history: copilotMessages.map((m) => ({ role: m.role, content: m.content })),
          previous_interaction_id: lastInteractionId,
          user_id: activeUserIdRef.current,
          study_id: activeStudyRecordId,
        });
        if (res.interaction_id) setLastInteractionId(res.interaction_id);
        setCopilotMessages((prev) => [
          ...prev,
          {
            id: `a-${Date.now()}`,
            role: "assistant",
            content: res.answer,
            provider: res.provider,
            latencyMs: res.latency_ms,
            timestamp: new Date().toLocaleTimeString(),
          },
        ]);
      } catch (err) {
        notify(err instanceof Error ? err.message : "Copilot request failed.");
      } finally {
        setCopilotLoading(false);
      }
    },
    [
      activeStudyName,
      activeStudyRecordId,
      copilotMessages,
      lastInteractionId,
      notify,
      patient,
      predictionData,
      report,
      threshold,
    ]
  );

  // User CRUD Handlers for SQLite Database Page
  const handleCreateUser = useCallback(
    async (payload: Partial<UserRecord> & { full_name: string }) => {
      setSavingUser(true);
      try {
        const created = await api.createUser(payload);
        await refreshUserDatabase(created.id);
        setPatient(userToPatientProfile(created));
        notify(`Created user record ${created.full_name} (${created.mrn})`);
      } catch (err) {
        notify(err instanceof Error ? err.message : "Failed to create user record.");
      } finally {
        setSavingUser(false);
      }
    },
    [notify, refreshUserDatabase]
  );

  const handleUpdateUser = useCallback(
    async (userId: string, patch: Partial<UserRecord>) => {
      setSavingUser(true);
      try {
        const updated = await api.updateUser(userId, patch);
        setActiveUserDetail(updated);
        setPatient(userToPatientProfile(updated));
        await refreshUserDatabase(userId);
        notify(`Updated ${updated.full_name} in SQLite database`);
      } catch (err) {
        notify(err instanceof Error ? err.message : "Failed to update user record.");
      } finally {
        setSavingUser(false);
      }
    },
    [notify, refreshUserDatabase]
  );

  const handleDeleteUser = useCallback(
    async (userId: string) => {
      try {
        await api.deleteUser(userId);
        const list = await api.listUsers();
        setUsers(list);
        const nextId = list[0]?.id || null;
        setActiveUserId(nextId);
        activeUserIdRef.current = nextId;
        if (nextId) {
          const detail = await api.getUser(nextId);
          setActiveUserDetail(detail);
          setPatient(userToPatientProfile(detail));
        } else {
          setActiveUserDetail(null);
        }
        notify("Deleted user record from database");
      } catch (err) {
        notify(err instanceof Error ? err.message : "Failed to delete user.");
      }
    },
    [notify]
  );

  // Load Archived Study & Real AI Transcript from SQLite back into Active Workspace!
  const handleLoadArchivedStudy = useCallback(
    async (studyId: string) => {
      try {
        const st = await api.getStudy(studyId);
        setActiveStudyRecordId(st.id);
        setActiveStudyName(st.study_name);
        setActiveStudyId(st.sample_id || null);
        setReport(st.report);
        setPhysicianName(st.physician_name || "Dr. S. Patel, MD");
        setPhysicianCreds(st.physician_creds || "Board-Certified Thoracic Radiologist");
        setPhysicianNotes(st.physician_notes || "");
        setSignedAt(st.signed_at || null);
        setPredictionData({
          study_id: st.id,
          user_id: st.user_id,
          predictions: st.predictions || [],
          heatmaps: [],
          anatomy: st.anatomy || null,
          decision_summary: st.decision_summary || null,
          ensemble_info: st.ensemble_info || {},
          timings: st.timings || {},
          report: st.report,
          patient,
          inference_time_ms: st.total_latency_ms || 0,
          vision_latency_ms: st.vision_latency_ms || 0,
        });
        setCopilotMessages(
          (st.copilot_messages || []).map((m) => ({
            id: m.id,
            role: m.role,
            content: m.content,
            provider: m.provider,
            latencyMs: m.latency_ms,
            timestamp: new Date(m.created_at).toLocaleTimeString(),
          }))
        );
        // If archived study maps to a sample image, restore its film preview cleanly too
        const matchedSample = samples.find(
          (s) => s.id === st.sample_id || s.filename === st.filename
        );
        if (matchedSample) {
          setImageUrl(api.getSampleImageUrl(matchedSample.filename));
        }
        navigatePage("report");
        notify(`Loaded archived study & transcript: ${st.study_name}`);
      } catch {
        notify("Could not load archived study.");
      }
    },
    [navigatePage, notify, patient, samples]
  );

  const handleDeleteStudy = useCallback(
    async (studyId: string) => {
      try {
        await api.deleteStudy(studyId);
        await refreshUserDatabase(activeUserIdRef.current);
        notify("Deleted archived study from database");
      } catch {
        notify("Could not delete study.");
      }
    },
    [notify, refreshUserDatabase]
  );

  // Batch Multi-Study STAT Triage
  const handleRunBenchmarkBatch = useCallback(async () => {
    if (!samples.length) return;
    setBatchLoading(true);
    try {
      const files = await Promise.all(samples.map((s) => api.fetchSampleAsFile(s)));
      const res = await api.predictBatch(files, threshold);
      setBatchResult(res);
      notify(
        `Triaged ${res.total_studies} benchmark studies in ${Math.round(res.total_latency_ms)} ms`
      );
    } catch (err) {
      notify(err instanceof Error ? err.message : "Batch triage failed.");
    } finally {
      setBatchLoading(false);
    }
  }, [notify, samples, threshold]);

  const handleUploadBatchFiles = useCallback(
    async (files: File[]) => {
      setBatchLoading(true);
      try {
        const res = await api.predictBatch(files, threshold);
        setBatchResult(res);
        notify(`Triaged ${res.total_studies} uploaded scans`);
      } catch (err) {
        notify(err instanceof Error ? err.message : "Batch upload failed.");
      } finally {
        setBatchLoading(false);
      }
    },
    [notify, threshold]
  );

  const handleInspectSampleByFilename = useCallback(
    (filename: string) => {
      const found = samples.find((s) => s.filename === filename);
      if (found) {
        handleSelectSample(found, true);
      }
    },
    [handleSelectSample, samples]
  );

  // Save Gemini Engine Config
  const handleSaveGemini = useCallback(
    async (payload: Record<string, unknown>) => {
      setGeminiSaving(true);
      try {
        const updated = await api.updateAIConfig(payload);
        setAiConfig(updated);
        const h = await api.getHealth().catch(() => null);
        if (h) setHealth(h);
        setGeminiModalOpen(false);
        notify(`Gemini engine updated (${updated.gemini_flash_model})`);
      } catch (err) {
        notify(err instanceof Error ? err.message : "Failed to update Gemini configuration.");
      } finally {
        setGeminiSaving(false);
      }
    },
    [notify]
  );

  // Export HL7 FHIR DiagnosticReport JSON
  const handleExportFhir = useCallback(() => {
    if (!predictionData || !report) return;
    const fhir = {
      resourceType: "DiagnosticReport",
      status: signedAt ? "final" : "preliminary",
      subject: activeUserDetail
        ? { display: activeUserDetail.full_name, identifier: { value: activeUserDetail.mrn } }
        : undefined,
      code: { text: "Chest Radiograph Multi-Pathology AI Assessment (CXR-Diagnostic)" },
      issued: new Date().toISOString(),
      conclusion: report.impression,
      extension: [
        { url: "https://cxr-diagnostic.local/ctr", valueDecimal: predictionData.anatomy?.ctr },
        { url: "https://cxr-diagnostic.local/provider", valueString: report.source },
        {
          url: "https://cxr-diagnostic.local/dictation_transcript",
          valueString: report.dictation_transcript || "",
        },
      ],
      result: predictionData.predictions.map((p) => ({
        display: p.name,
        probability: Number(p.score.toFixed(4)),
        positive: p.score >= threshold,
      })),
    };
    const blob = new Blob([JSON.stringify(fhir, null, 2)], { type: "application/fhir+json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cxr-diagnostic-fhir-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    notify("Exported HL7 FHIR DiagnosticReport JSON");
  }, [activeUserDetail, notify, predictionData, report, signedAt, threshold]);

  const activeHeatmap = useMemo(
    () => (activePathology ? heatmapCache[activePathology] || null : null),
    [activePathology, heatmapCache]
  );

  const positiveCount = useMemo(
    () => (predictionData?.predictions || []).filter((p) => p.score >= threshold).length,
    [predictionData, threshold]
  );

  return (
    <div
      className="relative min-h-screen"
      onDragEnter={(e) => {
        if (e.dataTransfer.types.includes("Files")) {
          dragDepthRef.current += 1;
          setGlobalDragOver(true);
        }
      }}
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes("Files")) {
          e.preventDefault();
        }
      }}
      onDragLeave={() => {
        dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
        if (dragDepthRef.current === 0) setGlobalDragOver(false);
      }}
      onDrop={(e) => {
        if (e.dataTransfer.files?.length) {
          e.preventDefault();
          dragDepthRef.current = 0;
          setGlobalDragOver(false);
          handleFileUpload(e.dataTransfer.files[0]);
        }
      }}
    >
      <AmbientRadiologyField theme={theme} />
      <div className="film-grain no-print" aria-hidden="true" />

      <AnimatePresence>
        {globalDragOver && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.16 }}
            className="no-print pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-black/65 backdrop-blur-md"
          >
            <div
              className="flex flex-col items-center gap-3 rounded-3xl border-2 border-dashed px-12 py-10 text-center"
              style={{
                borderColor: "var(--accent-clinical)",
                backgroundColor: "var(--bg-elevated)",
              }}
            >
              <Upload className="h-9 w-9" style={{ color: "var(--accent-clinical)" }} />
              <div className="font-display text-xl font-semibold">
                Drop Chest X-Ray or DICOM anywhere
              </div>
              <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
                Release to run instant multi-model analysis &amp; Grad-CAM++ localization
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,.dcm"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFileUpload(f);
          e.target.value = "";
        }}
      />

      <SidebarNav
        activePage={activePage}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((v) => !v)}
        onSelectPage={navigatePage}
        health={health}
        aiConfig={aiConfig}
        users={users}
        activeUserId={activeUserId}
        onSelectUser={handleSelectUser}
        positiveCount={positiveCount}
        decisionActive={Boolean(predictionData?.decision_summary?.active)}
        hasReport={Boolean(report)}
        batchStatCount={batchResult?.stat_count ?? null}
        totalUserStudies={activeUserDetail?.studies?.length || 0}
        onOpenGemini={() => setGeminiModalOpen(true)}
      />

      <div
        className={`no-print relative z-10 flex min-h-screen flex-col transition-[padding-left] duration-300 ease-out ${
          sidebarCollapsed ? "lg:pl-[72px]" : "lg:pl-[228px]"
        }`}
      >
        <TopHeader
          activePage={activePage}
          sidebarCollapsed={sidebarCollapsed}
          theme={theme}
          onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          health={health}
          aiConfig={aiConfig}
          activeStudyName={activeStudyName}
          imageUrl={imageUrl}
          predictionData={predictionData}
          threshold={threshold}
          isAnalyzing={isAnalyzing}
          onSelectPage={navigatePage}
          onOpenCommand={() => setCommandOpen(true)}
          onOpenGemini={() => setGeminiModalOpen(true)}
          onUploadClick={() => fileInputRef.current?.click()}
        />

        <main className="flex-1 min-w-0 overflow-x-clip">
          <AnimatePresence mode="wait">
            <motion.div
              key={activePage}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            >
              {activePage === "overview" && (
                <HeroOverview
                  theme={theme}
                  health={health}
                  samples={samples}
                  activeStudyId={activeStudyId}
                  imageUrl={imageUrl}
                  predictionData={predictionData}
                  threshold={threshold}
                  isAnalyzing={isAnalyzing}
                  onSelectSample={(s, open) => handleSelectSample(s, open)}
                  onUploadClick={() => fileInputRef.current?.click()}
                  onOpenStudio={() => navigatePage("studio")}
                  onOpenConsensus={() => navigatePage("consensus")}
                />
              )}

              {activePage === "studio" && (
                <DiagnosticStudio
                  samples={samples}
                  activeStudyId={activeStudyId}
                  activeStudyName={activeStudyName}
                  imageUrl={imageUrl}
                  predictionData={predictionData}
                  activePathology={activePathology}
                  activeHeatmap={activeHeatmap}
                  heatmapLoading={heatmapLoading}
                  threshold={threshold}
                  colormap={colormap}
                  patient={patient}
                  isAnalyzing={isAnalyzing}
                  holdingSpace={holdingSpace}
                  onSelectSample={(s) => handleSelectSample(s, false)}
                  onFileUpload={handleFileUpload}
                  onUploadClick={() => fileInputRef.current?.click()}
                  onSelectPathology={handleSelectPathology}
                  onThresholdChange={setThreshold}
                  onColormapChange={handleColormapChange}
                  onOpenPatientDrawer={() => setPatientDrawerOpen(true)}
                  onNavigatePage={navigatePage}
                />
              )}

              {activePage === "consensus" && (
                <ConsensusArchitecture
                  predictionData={predictionData}
                  threshold={threshold}
                  onInspectPathology={(name) => {
                    handleSelectPathology(name);
                    navigatePage("studio");
                  }}
                  onOpenGemini={() => setGeminiModalOpen(true)}
                  onNavigatePage={navigatePage}
                />
              )}

              {activePage === "report" && (
                <ClinicalReportSection
                  activeStudyName={activeStudyName}
                  activeStudyRecordId={activeStudyRecordId}
                  activeUserName={activeUserDetail?.full_name || null}
                  predictionData={predictionData}
                  report={report}
                  reportLoading={reportLoading}
                  copilotMessages={copilotMessages}
                  copilotLoading={copilotLoading}
                  physicianName={physicianName}
                  physicianCreds={physicianCreds}
                  physicianNotes={physicianNotes}
                  signedAt={signedAt}
                  onPhysicianNameChange={setPhysicianName}
                  onPhysicianCredsChange={setPhysicianCreds}
                  onPhysicianNotesChange={setPhysicianNotes}
                  onSignReport={handleSignReport}
                  onSaveTranscript={handleSaveTranscript}
                  onResynthesize={() => handleResynthesize()}
                  onSendCopilot={handleSendCopilot}
                  onClearCopilot={() => {
                    setCopilotMessages([]);
                    setLastInteractionId(null);
                  }}
                  onExportFhir={handleExportFhir}
                  onPrintPdf={() => window.print()}
                />
              )}

              {activePage === "triage" && (
                <BatchAndSpecsSection
                  samples={samples}
                  batchResult={batchResult}
                  batchLoading={batchLoading}
                  health={health}
                  aiConfig={aiConfig}
                  onRunBenchmarkBatch={handleRunBenchmarkBatch}
                  onUploadBatchFiles={handleUploadBatchFiles}
                  onInspectSampleByFilename={handleInspectSampleByFilename}
                />
              )}

              {activePage === "records" && (
                <RecordsAndEvaluationsPage
                  users={users}
                  activeUserId={activeUserId}
                  activeUserDetail={activeUserDetail}
                  evaluations={evaluations}
                  savingUser={savingUser}
                  onSelectUser={handleSelectUser}
                  onCreateUser={handleCreateUser}
                  onUpdateUser={handleUpdateUser}
                  onDeleteUser={handleDeleteUser}
                  onLoadStudy={handleLoadArchivedStudy}
                  onDeleteStudy={handleDeleteStudy}
                  onOpenStudio={() => navigatePage("studio")}
                />
              )}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>

      {/* Patient Profile Drawer -> Updates SQLite User Record + Re-Synthesizes AI Report */}
      <PatientContextDrawer
        open={patientDrawerOpen}
        patient={patient}
        onChange={setPatient}
        onApplyAndResynthesize={async () => {
          setPatientDrawerOpen(false);
          if (activeUserIdRef.current) {
            await handleUpdateUser(activeUserIdRef.current, {
              age: patient.age,
              sex: patient.sex,
              setting: patient.setting,
              indication: patient.indication,
              comorbidities: patient.comorbidities,
              smoking: patient.smoking,
            });
          }
          handleResynthesize(patient);
        }}
        onClose={() => setPatientDrawerOpen(false)}
      />

      <GeminiEngineModal
        open={geminiModalOpen}
        config={aiConfig}
        saving={geminiSaving}
        onSave={handleSaveGemini}
        onClose={() => setGeminiModalOpen(false)}
      />

      <CommandPalette
        open={commandOpen}
        theme={theme}
        samples={samples}
        predictions={predictionData?.predictions || []}
        onClose={() => setCommandOpen(false)}
        onNavigatePage={navigatePage}
        onSelectSample={(s) => handleSelectSample(s, true)}
        onSelectPathology={(name) => {
          handleSelectPathology(name);
          navigatePage("studio");
        }}
        onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
        onOpenGemini={() => setGeminiModalOpen(true)}
        onOpenPatient={() => setPatientDrawerOpen(true)}
        onUploadClick={() => fileInputRef.current?.click()}
        onPrintPdf={() => window.print()}
      />

      <PrintConsultationSheet
        studyName={activeStudyName}
        imageUrl={imageUrl}
        predictionData={predictionData}
        activeHeatmap={activeHeatmap}
        report={report}
        patient={patient}
        threshold={threshold}
        physicianName={physicianName}
        physicianCreds={physicianCreds}
        physicianNotes={physicianNotes}
        signedAt={signedAt}
      />

      <AnimatePresence>
        {toast && (
          <motion.div
            initial={{ opacity: 0, y: 14, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.96 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="no-print fixed right-6 bottom-6 z-50 flex items-center gap-2 rounded-full border px-4 py-2 text-xs font-medium shadow-xl backdrop-blur-xl"
            style={{
              backgroundColor: "var(--bg-elevated)",
              borderColor: "var(--border-strong)",
              color: "var(--text-primary)",
            }}
          >
            <CheckCircle2 className="h-3.5 w-3.5 shrink-0" style={{ color: "var(--accent-clinical)" }} />
            <span>{toast}</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
