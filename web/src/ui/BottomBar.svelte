<script lang="ts">
  import { app, SPEEDS, selectedRun } from "./state.svelte";
  import Tex from "./Tex.svelte";
  import { count, sci } from "./format";
  import { RUN_COLORS } from "./colors";

  const st = $derived(app.status);
  const sel = $derived(selectedRun());
  const activeStops = $derived(
    (app.stops.at_iter !== undefined ? 1 : 0) + (app.stops.every !== undefined ? 1 : 0) + (app.stops.f_below !== undefined ? 1 : 0) +
      (app.stops.grad_below !== undefined ? 1 : 0) + (app.stops.line !== undefined ? 1 : 0),
  );
  // scrubbing replays from the start: throttle while dragging over long runs
  let lastSeek = 0;
  function scrub(k: number, final: boolean) {
    const now = performance.now();
    const head = st?.head ?? 0;
    if (final || head < 200_000 || now - lastSeek > 120) {
      app.seek(k);
      lastSeek = now;
    }
  }
</script>

<footer class="panel bottom">
  <div class="controls">
    <button class="icon" onclick={() => app.toggle()} title={app.running ? "Pause (space)" : "Play (space)"}>{app.running ? "❚❚" : "▶"}</button>
    <button class="icon" onclick={() => app.step()} title="One iteration (.)">⏵|</button>
    <button class="icon" onclick={() => app.restart()} title="Back to the start (r)">↺</button>
  </div>
  <div class="speeds" role="group" aria-label="Speed">
    {#each SPEEDS as s}
      <button class:sel={app.speed === s.v} onclick={() => app.setSpeed(s.v)}>{s.label}</button>
    {/each}
  </div>
  <input
    class="timeline"
    type="range"
    min="0"
    max={Math.max(1, st?.head ?? 0)}
    value={st?.iter ?? 0}
    oninput={(e) => scrub(+e.currentTarget.value, false)}
    onchange={(e) => scrub(+e.currentTarget.value, true)}
    title="Timeline: drag back to any earlier iteration"
    aria-label="Timeline"
  />
  <div class="readout">
    {#if st}
      <span><Tex tex="k" /> = {count(st.iter)}{#if st.iter < st.head}<span class="dim"> / {count(st.head)}</span>{/if}</span>
      {#if sel.run}
        <span class="who"><i class="dot" style:background={RUN_COLORS[app.selected % RUN_COLORS.length].css}></i><Tex tex={"f(x_k)"} /> = {sci(sel.run.f)}</span>
        <span><Tex tex={"\\lVert\\nabla f\\rVert"} /> = {sci(sel.run.gradNorm)}</span>
      {/if}
    {/if}
  </div>
  <button class="stops" class:on={app.panel === "stops"} onclick={() => (app.panel = app.panel === "stops" ? "" : "stops")} title="Stopping points">
    Stops{#if activeStops}<span class="badge">{activeStops}</span>{/if}
  </button>
</footer>

<style>
  .bottom { position: absolute; left: 50%; bottom: 16px; transform: translateX(-50%); display: flex; align-items: center; gap: 12px; padding: 6px 12px; width: min(980px, calc(100vw - 32px)); flex-wrap: wrap; }
  .controls { display: flex; gap: 2px; }
  .icon { width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center; color: var(--dim); font-size: 14px; }
  .icon:hover { background: var(--track); color: var(--ink); }
  .speeds { display: flex; background: var(--track); border-radius: 7px; padding: 2px; }
  .speeds button { padding: 2px 8px; border-radius: 5px; color: var(--dim); font-size: 12px; }
  .speeds button.sel { background: var(--panel-solid); color: var(--ink); }
  .timeline { flex: 1; min-width: 120px; accent-color: var(--ink); }
  .readout { display: flex; gap: 14px; white-space: nowrap; }
  .who { display: flex; align-items: center; gap: 5px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dim { color: var(--faint); }
  .stops { color: var(--dim); font-size: 12px; padding: 4px 8px; border-radius: 6px; display: flex; gap: 5px; align-items: center; }
  .stops:hover, .stops.on { background: var(--track); color: var(--ink); }
  .badge { background: var(--accent); color: #fff; border-radius: 8px; font-size: 10px; padding: 0 5px; }
  @media (max-width: 720px) { .readout { width: 100%; } }
</style>
