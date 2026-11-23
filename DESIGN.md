# DESIGN.md — Impeccable + Watermelon UI Design System Specification

> **Design System & Aesthetic Direction**: *Precision Clinical Darkroom (BARCO Coronis / Sectra PACS meets Watermelon UI Micro-Interactions)*

---

## 1. Anti-Slop Design Commitments (Impeccable Principles)

1. **No Generic Consumer Gradients**: Eliminate arbitrary purple/indigo SaaS gradients. Use calibrated **OKLCH** clinical darkroom surfaces (`oklch(0.11 0.015 245)` to `oklch(0.19 0.022 245)`) engineered to preserve scotopic adaptation while reading grayscale radiographs.
2. **Typography Hierarchy with Purpose**:
   - **Display / Headers**: `'Instrument Sans'` / `'Plus Jakarta Sans'` — tight tracking (`-0.02em`), crisp geometric authority.
   - **Clinical UI Body**: `'Plus Jakarta Sans'` — high x-height for effortless scanning of clinical indications and pharmacotherapy regimens.
   - **Telemetry, Scores & DICOM HUD**: `'JetBrains Mono'` with `font-variant-numeric: tabular-nums` so probabilities, millimeter caliper readouts, and millisecond latencies never jitter during updates.
3. **High-Craft Dark UI**: Borders use directional luminance hairlines (`1px solid oklch(1 0 0 / 0.08)` with top-edge specular highlight `oklch(1 0 0 / 0.14)`), subtle ambient elevation, and semantic status tokens.

---

## 2. OKLCH Color Token System

| Token | OKLCH Value | Hex Fallback | Clinical Role |
|---|---|---|---|
| `--surface-void` | `oklch(0.08 0.012 248)` | `#03060c` | Lightbox canvas surround (maximizes dynamic range of X-ray film) |
| `--surface-app` | `oklch(0.11 0.015 248)` | `#060a12` | Primary PACS workstation background |
| `--surface-panel` | `oklch(0.145 0.018 248)` | `#0b111e` | Elevated diagnostic cards & drawers |
| `--surface-card` | `oklch(0.18 0.022 248)` | `#111a2e` | Interactive component cells & pathology rows |
| `--accent-cyan` | `oklch(0.78 0.155 215)` | `#00d4ff` | Primary PACS focus, active telemetry, caliper vector |
| `--accent-emerald` | `oklch(0.75 0.17 162)` | `#10b981` | Normal / unremarkable findings, healthy SLA latency |
| `--accent-amber` | `oklch(0.80 0.16 78)` | `#f59e0b` | Moderate severity (0.50–0.69), TTA active indicator |
| `--accent-crimson` | `oklch(0.66 0.22 18)` | `#f43f5e` | Severe pathology (≥ 0.70), Critical Pneumothorax/Edema alert |
| `--accent-violet` | `oklch(0.72 0.16 290)` | `#a855f7` | Graphify ontology edge & LLM synthesis indicator |

---

## 3. Watermelon UI Interactive Primitives

1. **Split-Curtain Radiograph Compare (`SplitCurtainCompare`)**:
   - Interactive draggable vertical divider across the 512×512 canvas allowing radiologists to scrub continuously between the **Raw CLAHE Radiograph** (left) and **Grad-CAM Activation Overlay** (right).
2. **Optical Diagnostic Loupe (`LoupeMagnifier`)**:
   - Circular 2.5× high-definition magnifying lens with real-time Hounsfield/pixel intensity readout (`HU / Luma`) and crosshair reticle.
3. **Real-Time Luminance Histogram (`HistogramInspector`)**:
   - 64-bin real-time canvas luminance distribution bar chart showing lung parenchyma air peak vs. mediastinal/osseous density.
4. **Command Palette (`Ctrl+K` / `Cmd+K`)**:
   - Instant keyboard-first launcher to switch benchmark studies, toggle Grad-CAM overlays, switch patient personas, open Graphify maps, or trigger Lighthouse/Bundlephobia audits.
5. **Graphify Interactive Canvas (`PathologyNetworkGraph`)**:
   - Physics-stabilized interactive node-link graph visualizing either the **14-Pathology Clinical Co-Occurrence & Guideline Ontology** or the **System AST & Microservice Latency Pipeline**.
6. **Observability & Audit Drawer (`TelemetryAuditModal`)**:
   - Unified **Sentry** Trace & Error Inspector + **Lighthouse / Bundlephobia** Live Performance & Bundle Budget scorecard.
