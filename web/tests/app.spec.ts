import { expect, test, type Page } from "@playwright/test";

const iter = async (page: Page) => Number((await page.locator(".readout span").first().innerText()).replace(/[^\d]/g, ""));

async function open(page: Page, query = "") {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`/?backend=webgl2${query}`);
  await expect(page.locator(".problem .name")).toHaveText("Rosenbrock", { timeout: 20_000 });
  await expect.poll(() => iter(page), { timeout: 20_000 }).toBeGreaterThan(5);
  return errors;
}

async function info(page: Page) {
  await page.getByTitle("Browser capabilities and performance").click();
  return page.locator(".info").innerText();
}

test("boots, renders and runs the default problem", async ({ page }) => {
  const errors = await open(page);
  const text = await info(page);
  expect(text).toContain("WebGL2");
  expect(text).toMatch(/render worker|main thread/);
  await expect(page.locator(".error")).toHaveCount(0);
  expect(errors).toEqual([]);
  await page.screenshot({ path: "test-results/app.png" });
});

test("uses shared memory and threads when cross-origin isolated", async ({ page, browserName }) => {
  await open(page);
  const isolated = await page.evaluate(() => crossOriginIsolated);
  expect(isolated).toBe(true);
  const text = await info(page);
  expect(text).toContain("triple buffer");
  if (browserName === "chromium") expect(text).toMatch(/\d+ threads/);
});

test("falls back to message passing and one thread without SharedArrayBuffer", async ({ page }) => {
  await open(page, "&shared=0");
  const text = await info(page);
  expect(text).toContain("message passing");
  expect(text).toContain("single thread");
});

test("falls back to rendering on the main thread", async ({ page }) => {
  await open(page, "&render=main");
  expect(await info(page)).toContain("main thread");
});

test("pause and single step", async ({ page }) => {
  await open(page);
  await page.getByTitle("Pause (space)").click();
  await page.waitForTimeout(300);
  const k = await iter(page);
  await page.waitForTimeout(300);
  expect(await iter(page)).toBe(k);
  await page.getByTitle("One step (→)").click();
  await expect.poll(() => iter(page)).toBe(k + 1);
});

test("switches problem and optimizer, and shows sourced pseudocode", async ({ page }) => {
  await open(page);
  await page.getByTitle("Choose the problem").click();
  await page.locator(".menu button", { hasText: "Himmelblau" }).click();
  await expect(page.locator(".problem .name")).toHaveText("Himmelblau");
  await page.locator(".runs button", { hasText: "Momentum" }).click();
  await page.getByText("Show the algorithm").click();
  await expect(page.locator(".code")).toContainText("Polyak (1964)");
  await expect(page.locator(".code")).toContainText("Where the implementation differs");
});
