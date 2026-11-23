# PRODUCT.md — CXR-Diagnostic™ Enterprise Thoracic AI PACS

> **Impeccable Durable Product Context**
> Standardized product definition, clinical workflow architecture, and engineering guardrails for the Chest Radiograph Multi-Pathology Diagnostic & Report Generator.

---

## 1. Product Vision & Purpose

**CXR-Diagnostic™** is a real-time clinical radiology workstation and asynchronous FastAPI inference platform that pairs a **DenseNet-121** multi-label thoracic pathology classifier (trained on 100,000+ NIH ChestX-ray14 & clinical radiographs, 0.94 mean ROC-AUC) with **Grad-CAM spatial interpretability** and a **3-Stage Automated Radiology Report Synthesis** engine (Clinical Template Composer → Local Hugging Face T5 Seq2Seq → LLM Dictation via Groq / xAI Grok).

### Primary Audience & Personas
1. **Diagnostic & Emergency Radiologists**: Require sub-200ms triage latency, non-destructive Grad-CAM overlays, split-curtain before/after comparison, optical loupe magnification, pixel-intensity histograms, and ACR/RSNA-compliant compartmentalized dictation.
2. **ED / ICU Attending Physicians**: Require immediate critical finding flags (e.g., Tension Pneumothorax, Acute Pulmonary Edema), anatomical co-occurrence visualization (**Graphify**), and guideline-directed pharmacotherapy decision support (ATS/IDSA, ACC/AHA, Fleischner 2017, ACCP, BTS).
3. **ML & Clinical Systems Engineers**: Require live **Sentry** span telemetry, dynamic batching metrics, **Lighthouse** Web Vitals, and **Bundlephobia** zero-bloat bundle auditing.

---

## 2. Core Functional Pillars

| Pillar | Technology | Clinical & Engineering Outcome |
|---|---|---|
| **UI & Component Architecture** | **Watermelon UI + Impeccable** | OKLCH radiological darkroom palette, spring-physics micro-interactions, `Ctrl+K` Command Palette, Split-Curtain Compare Slider, Optical Loupe Magnifier, Real-Time CLAHE Pixel Histogram. |
| **Type Safety & Build** | **Strict TypeScript (`tsc` + `esbuild`)** | 100% typed frontend architecture (`web/src/*.ts`) compiling cleanly to zero-dependency browser bundles. |
| **Knowledge & Architecture Graphs** | **Graphify** | Interactive 2-mode Canvas Knowledge Graph: (1) Clinical 14-Pathology Co-Occurrence, Anatomical Compartment & Guideline Graph, and (2) Live System AST & Inference Pipeline Graph. |
| **Observability & Error Tracking** | **Sentry** | Distributed span tracing (`cxr.preprocess`, `cxr.densenet121`, `cxr.gradcam`, `cxr.report`), automatic error boundary capture, breadcrumb ring-buffer, and `/api/v1/telemetry` live inspector. |
| **Performance & Bundle Budget** | **Lighthouse + Bundlephobia** | Sub-25 KB gzipped total JS bundle, zero bloated third-party runtime dependencies, WCAG AAA contrast in PACS darkroom mode, zero Cumulative Layout Shift (`CLS = 0.00`). |
| **End-to-End Verification** | **Playwright + Pytest** | Automated headless Chromium E2E test suite verifying canvas rendering, split-curtain scrubbing, Graphify node interaction, Command Palette, Sentry telemetry modal, and report synthesis. |

---

## 3. Clinical Safety & Non-Goals

- **Diagnostic Transparency**: Never obscure the raw radiograph permanently; always provide 1-key (`Space`) raw film peeking, adjustable Grad-CAM opacity, and an interactive split-screen curtain wipe.
- **Zero "AI Slop"**: Avoid generic consumer SaaS gradients, decorative clutter, or unreadable low-contrast muted text. Every pixel serves diagnostic speed, spatial localization, or clinical synthesis.
