import { expect, test } from "@playwright/test";

const ARTIFACT_DIR = "C:/Users/smitp/.gemini/antigravity/brain/567b18a3-2df0-4ae5-a3f8-9c06f664fddb";

test.describe("CXR-Diagnostic v7.0 Complete Multi-Page Suite (User DB + Real AI Transcripts + Evaluations)", () => {
  test.use({ viewport: { width: 1520, height: 940 } });

  test("verifies all 6 pages, SQLite user CRUD & study archive, real Gemini dictation transcripts, live eval metrics & zero errors", async ({
    page,
  }) => {
    const pageErrors: string[] = [];
    page.on("pageerror", (err) => pageErrors.push(err.message));

    await page.goto("/", { waitUntil: "networkidle" });
    await expect(page).toHaveTitle(/CXR-Diagnostic/);

    // Wait for initial live RTX CUDA + Gemini analysis of Study #1 to complete & archive to SQLite
    await expect(page.getByText(/CTR 0\./i).first()).toBeVisible({ timeout: 65000 });
    await page.waitForTimeout(800);

    // 1. Page 01 — Overview (Darkroom)
    await page.screenshot({ path: `${ARTIFACT_DIR}/v7_01_overview_dark.png` });

    // 2. Page 02 — Scan Studio (Single-Row Overlay Bar + DICOM Lightbox)
    await page
      .getByRole("navigation", { name: /Workspace pages/i })
      .getByRole("button", { name: /Scan Studio/i })
      .click();
    await page.waitForTimeout(650);
    await expect(page.getByText(/02 · DIAGNOSTIC SCAN STUDIO/i)).toBeVisible();
    await page.screenshot({ path: `${ARTIFACT_DIR}/v7_02_studio_dark.png` });

    // Toggle Anatomy & Split Compare in Alabaster Light Theme
    await page.getByRole("button", { name: /Anatomy & CTR/i }).click();
    await page.getByRole("button", { name: /Split Compare/i }).click();
    await page.getByRole("button", { name: /Switch to Clinical Alabaster light theme/i }).click();
    await page.waitForTimeout(600);
    await page.screenshot({ path: `${ARTIFACT_DIR}/v7_02_studio_light.png` });

    // Return to Darkroom Theme
    await page.getByRole("button", { name: /Switch to Radiology Darkroom dark theme/i }).click();
    await page.waitForTimeout(400);

    // 3. Page 03 — Consensus & CTR Biometry
    await page
      .getByRole("navigation", { name: /Workspace pages/i })
      .getByRole("button", { name: /Consensus & CTR/i })
      .click();
    await page.waitForTimeout(650);
    await expect(page.getByText(/03 · MULTI-MODEL CONSENSUS/i)).toBeVisible();
    await page.screenshot({ path: `${ARTIFACT_DIR}/v7_03_consensus.png` });

    // 4. Page 04 — ACR Monograph, Live AI Dictation Transcript & Grounded Gemini Copilot
    await page
      .getByRole("navigation", { name: /Workspace pages/i })
      .getByRole("button", { name: /Report & Copilot/i })
      .click();
    await page.waitForTimeout(600);
    await expect(page.getByText(/AI ATTENDING DICTATION TRANSCRIPT/i)).toBeVisible();
    await page.getByRole("button", { name: /Explain the CTR biometry/i }).click();
    await expect(page.getByText(/Synthesizing clinical guidance/i)).toBeHidden({ timeout: 35000 });
    await page.waitForTimeout(450);
    await page.screenshot({ path: `${ARTIFACT_DIR}/v7_04_report_transcript_copilot.png` });

    // 5. Page 05 — Batch STAT Triage Worklist
    await page
      .getByRole("navigation", { name: /Workspace pages/i })
      .getByRole("button", { name: /Batch Triage/i })
      .click();
    await page.waitForTimeout(500);
    await page.getByRole("button", { name: /Triage All .* Benchmark Studies/i }).click();
    await expect(page.getByText(/Avg Vision Latency/i)).toBeVisible({ timeout: 35000 });
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${ARTIFACT_DIR}/v7_05_batch_triage.png` });

    // 6. Page 06 — Persistent Per-User SQLite Database & Live Evaluation Observatory
    await page
      .getByRole("navigation", { name: /Workspace pages/i })
      .getByRole("button", { name: /Records & Evals/i })
      .click();
    await page.waitForTimeout(600);
    await expect(page.getByText(/06 · PATIENT DATABASE & MODEL EVALUATIONS/i)).toBeVisible();
    await expect(page.getByText(/Archived Diagnostic Studies/i)).toBeVisible();
    await page.screenshot({ path: `${ARTIFACT_DIR}/v7_06_user_database.png` });

    // Test Updating the Active User Record live in SQLite!
    await page.getByRole("button", { name: /Update User Record/i }).click();
    await expect(page.getByText(/in SQLite database/i)).toBeVisible({ timeout: 8000 });

    // Switch to Evaluation Metrics Observatory Tab
    await page.getByRole("button", { name: /Evaluation Metrics/i }).click();
    await page.waitForTimeout(600);
    await expect(page.getByText(/Backbone Ablation & Calibration Comparison/i)).toBeVisible();
    await page.screenshot({ path: `${ARTIFACT_DIR}/v7_06_evaluations_metrics.png` });

    expect(pageErrors).toEqual([]);
  });
});
