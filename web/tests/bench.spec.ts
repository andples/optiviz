import { expect, test } from "@playwright/test";

// The WASM benchmark page; run with `pnpm e2e --grep @bench`. Prints the JSON for docs/perf.
test("@bench wasm benchmarks complete", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/bench.html");
  await expect(page.locator("body[data-done]")).toHaveCount(1, { timeout: 170_000 });
  const out = await page.locator("#out").innerText();
  console.log(`BENCH ${out.replace(/\s+/g, " ")}`);
  expect(await page.locator("#state").innerText()).toBe("done");
});
