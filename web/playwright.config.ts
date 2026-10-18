import { defineConfig, devices } from "@playwright/test";

// Local runs use a system Chromium if CHROMIUM is set (no browser download); CI installs Playwright's
// browsers and sets ALL_BROWSERS=1 to add Firefox and WebKit. Headless browsers here have no real GPU, so
// the tests use the WebGL2 back-end (?backend=webgl2); WebGPU is checked natively by crates/gpu-test and by
// hand in real browsers.
const chromium = process.env.CHROMIUM;
const all = !!process.env.ALL_BROWSERS;

export default defineConfig({
  testDir: "tests",
  timeout: 60_000,
  workers: 2, // each page starts a WASM thread pool; more at once only adds contention
  reporter: [["list"]],
  use: { baseURL: "http://localhost:4173", trace: "retain-on-failure" },
  webServer: { command: "pnpm preview --strictPort", url: "http://localhost:4173", reuseExistingServer: true },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: { executablePath: chromium, args: ["--enable-unsafe-webgpu"] },
      },
    },
    ...(all
      ? [
          { name: "firefox", use: { ...devices["Desktop Firefox"] } },
          { name: "webkit", use: { ...devices["Desktop Safari"] } },
        ]
      : []),
  ],
});
