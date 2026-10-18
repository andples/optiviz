<script lang="ts">
  import { app, selectedRun } from "./state.svelte";
  import { RUN_COLORS } from "./colors";
  import ParamSlider from "./ParamSlider.svelte";
  import Tex from "./Tex.svelte";
  import { sciTex } from "./format";

  let adding = $state(false);
  const sel = $derived(selectedRun());
  const runs = $derived(app.status?.runs ?? []);
  const name = (id: string) => app.optimizers.find((o) => o.id === id)?.name ?? id;
  const RESTART = [
    { label: "never", c: false, f: false },
    { label: "converged", c: true, f: false },
    { label: "failed", c: false, f: true },
    { label: "both", c: true, f: true },
  ];
  const ms = $derived(app.multistart);
  const multi = $derived(ms.starts > 1 || ms.onConverged || ms.onFailed);
  const healthNote: Record<string, string> = {
    "out-of-view": "outside the view",
    diverging: "diverging",
    unbounded: "unbounded below",
    diverged: "diverged",
  };
</script>

<aside class="panel runs-panel">
  <div class="multi">
    <div class="row" title="Every run starts from the clicked point, plus random points in the view">
      <span class="lbl">Starting points</span>
      <div class="seg" role="group" aria-label="Starting points">
        {#each [1, 4, 16, 64] as n}
          <button class:sel={ms.starts === n} onclick={() => app.setMultistart({ starts: n })}>{n}</button>
        {/each}
      </div>
    </div>
    <div class="row" title="A point that has converged or failed jumps to a random point and starts again; the best point found is kept">
      <span class="lbl">Random restart</span>
      <div class="seg" role="group" aria-label="Random restart">
        {#each RESTART as o}
          <button class:sel={ms.onConverged === o.c && ms.onFailed === o.f} onclick={() => app.setMultistart({ onConverged: o.c, onFailed: o.f })}>{o.label}</button>
        {/each}
      </div>
    </div>
  </div>
  <div class="runs">
    {#each runs as r, i}
      <div class="run" class:sel={i === app.selected}>
        <button class="pick" onclick={() => app.select(i)} title="Select this run">
          <i class="dot" style:background={RUN_COLORS[i % RUN_COLORS.length].css}></i>
          <span>{name(r.opt)}</span>
          {#if r.health !== "ok"}<span class="warn" class:soft={r.health === "out-of-view"}>{healthNote[r.health]}</span>{/if}
        </button>
        {#if runs.length > 1}<button class="x" title="Remove this run" onclick={() => app.removeRun(i)}>×</button>{/if}
      </div>
      {#if multi}
        <div class="walk">
          {#if r.walkers > 1}{r.walkers} points · {/if}{r.restarts} restart{r.restarts === 1 ? "" : "s"}{#if r.failed} · {r.failed} failing{/if}
          · best <Tex tex={`f = ${sciTex(r.bestF)}`} />
        </div>
      {/if}
    {/each}
    {#if runs.length < 8}
      <button class="add" onclick={() => (adding = !adding)} title="Race another optimizer on the same problem and start">+ add a run</button>
    {/if}
  </div>
  {#if adding}
    <div class="menu">
      {#each app.optimizers as o}
        <button onclick={() => { app.addRun(o.id); adding = false; }}>
          <span>{o.name}</span><span class="fam">{o.family}</span>
        </button>
      {/each}
    </div>
  {/if}

  {#if sel.run && sel.manifest}
    {@const m = sel.manifest}
    <div class="inspector">
      <div class="head">
        <select value={m.id} onchange={(e) => app.setOptimizer(app.selected, e.currentTarget.value)} aria-label="Optimizer">
          {#each app.optimizers as o}<option value={o.id}>{o.name}</option>{/each}
        </select>
        <span class="fam">{m.family}</span>
      </div>
      <div class="rule"><Tex tex={m.tex} /></div>
      {#each m.params as s (m.id + s.key)}
        <ParamSlider spec={s} value={sel.run.params[s.key] ?? s.default} onchange={(v) => app.setParam(s.key, v)} />
      {/each}
      <div class="foot">
        <button class="toggle" onclick={() => (app.showCode = !app.showCode)}>{app.showCode ? "Hide" : "Show"} the algorithm</button>
        <span class="evals" title="objective and gradient evaluations so far">{sel.run.evals.toLocaleString()} evaluations</span>
      </div>
    </div>
  {/if}
</aside>

<style>
  .runs-panel { position: absolute; top: 60px; right: 16px; width: 310px; padding: 8px 10px; display: flex; flex-direction: column; gap: 8px; max-height: calc(100vh - 170px); overflow: auto; }
  .multi { display: flex; flex-direction: column; gap: 5px; padding-bottom: 8px; border-bottom: 1px solid var(--rule); }
  .row { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  .lbl { color: var(--dim); font-size: 12px; white-space: nowrap; }
  .seg { display: flex; background: var(--track); border-radius: 7px; padding: 2px; }
  .seg button { padding: 1px 7px; border-radius: 5px; color: var(--dim); font-size: 11.5px; }
  .seg button.sel { background: var(--panel-solid); color: var(--ink); }
  .walk { color: var(--faint); font-size: 11px; padding: 0 6px 3px 22px; }
  .runs { display: flex; flex-direction: column; gap: 2px; }
  .run { display: flex; align-items: center; border-radius: 6px; }
  .run.sel { background: var(--track); }
  .pick { flex: 1; display: flex; align-items: center; gap: 7px; padding: 4px 6px; text-align: left; min-width: 0; }
  .pick span:first-of-type { white-space: nowrap; }
  .dot { width: 9px; height: 9px; border-radius: 50%; flex: none; box-shadow: 0 0 0 2px var(--panel-solid); }
  .warn { margin-left: auto; color: var(--accent); font-size: 11px; white-space: nowrap; }
  .warn.soft { color: var(--dim); }
  .x { color: var(--faint); padding: 0 6px; font-size: 15px; }
  .x:hover { color: var(--ink); }
  .add { align-self: flex-start; color: var(--dim); font-size: 12px; padding: 3px 6px; }
  .add:hover { color: var(--ink); }
  .menu { display: flex; flex-direction: column; border-top: 1px solid var(--rule); padding-top: 4px; }
  .menu button { display: flex; justify-content: space-between; padding: 4px 6px; border-radius: 5px; }
  .menu button:hover { background: var(--track); }
  .fam { color: var(--faint); font-size: 11px; }
  .inspector { display: flex; flex-direction: column; gap: 8px; border-top: 1px solid var(--rule); padding-top: 8px; }
  .head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
  select { font: inherit; font-weight: 600; color: var(--ink); background: transparent; border: 0; padding: 0; cursor: pointer; }
  .rule { color: var(--dim); font-size: 11.5px; overflow-x: auto; padding-bottom: 4px; border-bottom: 1px solid var(--rule); }
  .foot { display: flex; justify-content: space-between; align-items: center; }
  .toggle { color: var(--dim); font-size: 12px; }
  .toggle:hover { color: var(--ink); }
  .evals { color: var(--faint); font-size: 11px; }
  @media (max-width: 720px) { .runs-panel { top: auto; bottom: 130px; left: 16px; width: auto; max-height: 40vh; } }
</style>
