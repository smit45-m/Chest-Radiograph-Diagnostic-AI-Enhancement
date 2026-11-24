import { expect, test } from "@playwright/test";

const SCREENSHOT_DIR = process.env.SCREENSHOT_DIR || "";

async function snap(page: import("@playwright/test").Page, filename: string) {
  if (!SCREENSHOT_DIR) return;
  await page.screenshot({ path: `${SCREENSHOT_DIR}/${filename}` });
}

test.describe("CXR-Diagnostic v8.0 Calm Precision Multi-Page Suite", () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test("verifies all 6 refined pages, progressive disclosure, SQLite updates, copilot, batch triage & zero errors", async ({
    page,
  }) => {
    const pageErrors: string[] = [];
    page.on("pageerror", (err) => pageErrors.push(err.message));

    await page.goto("/", { waitUntil: "networkidle" });
    await expect(page).toHaveTitle(/CXR-Diagnostic/);

    // Wait for initial live inference & SQLite archive of Study #1
    await expect(page.getByText(/CTR 0\./i).first()).toBeVisible({ timeout: 65000 });
    await page.waitForTimeout(600);

    // 1. Page 01 — Overview (Darkroom)
    await expect(page.getByRole("heading", { name: /Chest X-ray analysis/i })).toBeVisible();
    await snap(page, "v8_01_overview_dark.png");

    // 2. Page 02 — Scan Studio (Unified 4-Mode Segmented Lightbox)
    await page
      .getByRole("navigation", { name: /Workspace pages/i })
      .getByRole("button", { name: /Scan Studio/i })
      .click();
    await page.waitForTimeout(500);
    await expect(page.getByRole("heading", { name: /Scan Studio/i })).toBeVisible();
    await snap(page, "v8_02_studio_dark.png");

    // Test Compare mode + collapsible Adjust drawer + Light theme
    await page.getByRole("button", { name: /^Compare$/i }).click();
    await page.getByRole("button", { name: /^Adjust$/i }).click();
    await expect(page.getByRole("button", { name: /^Lung$/i })).toBeVisible();
    await page.getByRole("button", { name: /Switch to light theme/i }).click();
    await page.waitForTimeout(450);
    await snap(page, "v8_02_studio_light.png");

    // Close Adjust & return to Darkroom Heatmap
    await page.getByRole("button", { name: /^Adjust$/i }).click();
    await page.getByRole("button", { name: /^Heatmap$/i }).click();
    await page.getByRole("button", { name: /Switch to dark theme/i }).click();
    await page.waitForTimeout(350);

    // 3. Page 03 — Model Consensus & Biometry
    await page
      .getByRole("navigation", { name: /Workspace pages/i })
      .getByRole("button", { name: /Consensus & CTR/i })
      .click();
    await page.waitForTimeout(500);
    await expect(page.getByText(/Model Agreement Matrix/i)).toBeVisible();
    await snap(page, "v8_03_consensus.png");

    // 4. Page 04 — Clinical Report & Copilot
    await page
      .getByRole("navigation", { name: /Workspace pages/i })
      .getByRole("button", { name: /Report & Copilot/i })
      .click();
    await page.waitForTimeout(500);
    await expect(page.getByText(/AI Dictation Transcript/i)).toBeVisible();
    await page.getByRole("button", { name: /Explain the CTR biometry/i }).click();
    await expect(page.getByText(/Thinking\.\.\./i)).toBeHidden({ timeout: 35000 });
    await page.waitForTimeout(450);
    await snap(page, "v8_04_report_copilot.png");

    // 5. Page 05 — Batch Triage
    await page
      .getByRole("navigation", { name: /Workspace pages/i })
      .getByRole("button", { name: /Batch Triage/i })
      .click();
    await page.waitForTimeout(450);
    await page.getByRole("button", { name: /Triage All .* Samples/i }).click();
    await expect(page.getByText(/Avg per Scan/i)).toBeVisible({ timeout: 35000 });
    await page.waitForTimeout(450);
    await snap(page, "v8_05_batch_triage.png");

    // 6. Page 06 — Records & Evaluations (View-First Profile + Collapsible Edit + No X-Drift)
    await page
      .getByRole("navigation", { name: /Workspace pages/i })
      .getByRole("button", { name: /Records & Evals/i })
      .click();
    await page.waitForTimeout(500);
    await expect(page.getByText(/Saved Scans & Transcripts/i)).toBeVisible();
    await snap(page, "v8_06_records.png");

    // Expand Edit Profile -> Save Changes -> Persists to SQLite & collapses cleanly
    await page.getByRole("button", { name: /Edit Profile/i }).click();
    await page.getByRole("button", { name: /Save Changes/i }).click();
    await expect(page.getByText(/in SQLite database/i)).toBeVisible({ timeout: 8000 });

    // Switch to Evaluation Metrics Observatory Tab
    await page.getByRole("button", { name: /Evaluation Metrics/i }).click();
    await page.waitForTimeout(450);
    await expect(page.getByText(/Model Ablation Comparison/i)).toBeVisible();
    const scrollX = await page.evaluate(() => window.scrollX);
    expect(scrollX).toBe(0);
    await snap(page, "v8_06_evaluations.png");

    expect(pageErrors).toEqual([]);
  });
});
