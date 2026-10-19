import { expect, test, type Page } from "@playwright/test";

const iter = async (page: Page) => Number((await page.locator(".readout span").first().innerText()).split("/")[0].replace(/[^\d]/g, ""));
const share = (s: object) => Buffer.from(JSON.stringify(s)).toString("base64url");

async function open(page: Page, query = "", hash = "") {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`/?backend=webgl2${query}${hash ? `#${hash}` : ""}`);
  await expect(page.locator(".problem .name")).not.toHaveText("loading…", { timeout: 20_000 });
  await expect.poll(() => iter(page), { timeout: 20_000 }).toBeGreaterThan(5);
  return errors;
}

async function info(page: Page) {
  await page.getByTitle("Browser capabilities and performance").click();
  return page.locator(".info").innerText();
}

test("boots, renders and races two runs on the default problem", async ({ page }) => {
  const errors = await open(page);
  await expect(page.locator(".problem .name")).toHaveText("Rosenbrock");
  await expect(page.locator(".run")).toHaveCount(2);
  expect(await info(page)).toContain("WebGL2");
  await expect(page.locator(".error")).toHaveCount(0);
  expect(errors).toEqual([]);
  await page.screenshot({ path: "test-results/app.png" });
});

test("uses shared memory and threads when cross-origin isolated", async ({ page, browserName }) => {
  await open(page);
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(true);
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

test("pause, single step, and scrub back on the timeline", async ({ page }) => {
  await open(page);
  await page.getByTitle("Pause (space)").click();
  await page.waitForTimeout(300);
  const k = await iter(page);
  await page.waitForTimeout(300);
  expect(await iter(page)).toBe(k);
  await page.getByTitle("One iteration (.)").click();
  await expect.poll(() => iter(page)).toBe(k + 1);
  await page.getByLabel("Timeline").fill(String(Math.floor(k / 2)));
  await expect.poll(() => iter(page)).toBe(Math.floor(k / 2));
});

test("adds, switches and removes runs; shows sourced pseudocode", async ({ page }) => {
  await open(page);
  await page.getByTitle("Race another optimizer on the same problem and start").click();
  await page.locator(".menu button", { hasText: "L-BFGS" }).click();
  await expect(page.locator(".run")).toHaveCount(3);
  await page.locator(".run .pick", { hasText: "Momentum" }).click();
  await page.getByText("Show the algorithm").click();
  await expect(page.locator(".code")).toContainText("Polyak (1964)");
  await expect(page.locator(".code")).toContainText("Where the implementation differs");
  await page.locator(".run").nth(2).getByTitle("Remove this run").click();
  await expect(page.locator(".run")).toHaveCount(2);
});

test("problem coefficients can be changed", async ({ page }) => {
  await open(page);
  await page.getByTitle("Choose the problem and its coefficients").click();
  await page.getByRole("button", { name: /^Himmelblau/ }).click();
  await expect(page.locator(".problem .name")).toContainText("Himmelblau");
  await page.getByTitle("Choose the problem and its coefficients").click();
  await page.locator(".coeffs .val").first().click();
  await page.locator(".coeffs .edit").fill("9");
  await page.keyboard.press("Enter");
  await expect(page.locator(".problem .name")).toContainText("custom");
});

test("a too-large step size stops with a diagnosis, and the suggested fix converges", async ({ page }) => {
  await open(page, "", share({ p: "bowl", r: [{ o: "gd", p: { lr: 0.11 } }] })).catch(() => {});
  const notice = page.locator(".notice");
  await expect(notice).toContainText("diverging", { timeout: 20_000 });
  await expect(notice).toContainText("2/L");
  await notice.locator(".primary").click();
  await expect(notice).toHaveCount(0);
  await expect(page.locator(".run .warn")).toHaveCount(0, { timeout: 10_000 });
});

test("a saddle is reported as unbounded below, not as a bad step size", async ({ page }) => {
  await page.goto(`/?backend=webgl2#${share({ p: "saddle", r: [{ o: "gd", p: { lr: 0.05 } }] })}`);
  await expect(page.locator(".notice")).toContainText("no bottom", { timeout: 30_000 });
  await page.getByRole("button", { name: "Zoom out to follow" }).first().click();
  await expect(page.getByRole("button", { name: "Reset view" })).toBeVisible();
});

test("stopping points: iteration and pseudocode line; line-by-line stepping", async ({ page }) => {
  await open(page);
  await page.getByTitle("Back to the start (r)").click();
  await page.getByTitle("Stopping points").click();
  await page.locator(".stops .num").first().fill("40");
  await page.locator(".stops .num").first().press("Enter");
  await page.getByTitle("Back to the start (r)").click();
  await expect(page.locator(".notice")).toContainText("iteration reached", { timeout: 20_000 });
  expect(await iter(page)).toBe(40);
  await page.keyboard.press(",");
  await page.keyboard.press(",");
  await expect(page.locator(".code li.lit")).toHaveCount(1);
});

test("arrow keys switch to the free camera and Esc returns to orbit", async ({ page }) => {
  await open(page);
  await page.keyboard.down("ArrowUp");
  await page.waitForTimeout(500);
  await page.keyboard.up("ArrowUp");
  await expect(page.locator(".chip", { hasText: "Free camera" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.locator(".chip", { hasText: "Free camera" })).toHaveCount(0);
});

test("contour view and a share link round trip", async ({ page }) => {
  const hash = share({ p: "camel", r: [{ o: "nelder-mead" }, { o: "newton" }], v: "contour" });
  await open(page, "", hash);
  await expect(page.locator(".problem .name")).toHaveText("Six-hump camel");
  await expect(page.getByRole("button", { name: "Contour" })).toHaveClass(/sel/);
  await expect(page.locator(".run")).toHaveCount(2);
  await page.screenshot({ path: "test-results/contour.png" });
});

test("several random starting points and random restarts", async ({ page }) => {
  await open(page, "", share({ p: "rastrigin", r: [{ o: "gd", p: { lr: 0.003 } }] }));
  await page.getByRole("group", { name: "Starting points" }).getByRole("button", { name: "16" }).click();
  await page.getByRole("group", { name: "Random restart" }).getByRole("button", { name: "both" }).click();
  await expect(page.locator(".walk")).toContainText("16 points", { timeout: 10_000 });
  await expect.poll(async () => Number((await page.locator(".walk").first().innerText()).match(/(\d+) restart/)?.[1] ?? 0), { timeout: 20_000 }).toBeGreaterThan(3);
  // the setting travels in a share link
  await page.getByTitle("Copy a link to this exact setup").click();
  await page.reload();
  await expect(page.locator(".walk")).toContainText("16 points", { timeout: 20_000 });
});

test("trust region: pseudocode shows ρ and the radius update", async ({ page }) => {
  await open(page, "", share({ p: "rosenbrock", r: [{ o: "trust-region", p: { solver: 1, delta0: 0.4 } }], v: "contour" }));
  await page.getByTitle("Pause (space)").click();
  await page.getByRole("button", { name: "Show the algorithm" }).click();
  await page.getByTitle("One iteration (.)").click();
  await expect(page.locator(".code")).toContainText("Nocedal & Wright (2006)");
  await expect(page.locator(".code li.lit")).toHaveCount(4); // subproblem, ρ, one radius rule, one x rule
  await expect(page.locator(".code")).toContainText("ρ");
  await page.getByRole("group", { name: "how the step inside the region is chosen" }).getByRole("button", { name: "exact" }).click();
  await page.getByTitle("One iteration (.)").click();
  await expect(page.locator(".code li.lit")).toHaveCount(4);
});
