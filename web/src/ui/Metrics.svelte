<script lang="ts">
  import { app } from "./state.svelte";
  import { RUN_COLORS } from "./colors";
  import Chart, { type Line } from "./Chart.svelte";

  /** Convergence charts from the engine's metric rows (iter, f, ‖∇f‖, step length): one small chart of f − f*
   *  (or f when f* is unknown), expandable to all three quantities. */
  const fmin = $derived(app.info?.manifest.f_min ?? null);
  function lines(col: number, shift = 0): Line[] {
    return app.series.map((s, i) => {
      const n = s.length / 4;
      const xs = new Float64Array(n), ys = new Float64Array(n);
      for (let k = 0; k < n; k++) {
        xs[k] = s[4 * k];
        ys[k] = s[4 * k + col] - shift;
      }
      return { xs, ys, color: RUN_COLORS[i % RUN_COLORS.length].css, bold: i === app.selected };
    });
  }
  const fLines = $derived(lines(1, fmin ?? 0));
  const fLog = $derived(fmin !== null);
  const best = $derived(app.multistart.starts > 1 || app.multistart.onConverged || app.multistart.onFailed);
  const at = $derived(app.status && app.status.iter < app.status.head ? app.status.iter : null);
</script>

<div class="panel metrics" class:big={app.bigChart}>
  <button class="expand" onclick={() => (app.bigChart = !app.bigChart)} title={app.bigChart ? "Smaller" : "All metrics"}>{app.bigChart ? "–" : "+"}</button>
  <Chart lines={fLines} log={fLog} {at} label={(best ? "best " : "") + (fLog ? "f(xₖ) − f*" : "f(xₖ)")} height={app.bigChart ? 150 : 96} />
  {#if app.bigChart}
    <Chart lines={lines(2)} log {at} label="‖∇f(xₖ)‖" height={130} />
    <Chart lines={lines(3)} log {at} label="step length ‖xₖ₊₁ − xₖ‖" height={130} />
  {/if}
</div>

<style>
  .metrics { position: absolute; right: 16px; bottom: 76px; width: 280px; padding: 6px 8px 4px; }
  .metrics.big { width: min(520px, calc(100vw - 32px)); }
  .expand { position: absolute; right: 6px; top: 3px; color: var(--dim); width: 18px; height: 18px; border-radius: 4px; z-index: 1; }
  .expand:hover { background: var(--track); color: var(--ink); }
  @media (max-width: 720px) { .metrics { display: none; } }
</style>
