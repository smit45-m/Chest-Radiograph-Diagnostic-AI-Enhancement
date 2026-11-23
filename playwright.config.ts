import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 120000,
  fullyParallel: false,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:8000",
    trace: "off",
    headless: true,
  },
  webServer: {
    command: "python -m uvicorn api.main:app --host 127.0.0.1 --port 8000",
    url: "http://127.0.0.1:8000/api/v1/health",
    reuseExistingServer: true,
    timeout: 90000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], channel: "chrome" },
    },
  ],
});
