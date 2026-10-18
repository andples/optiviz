<script lang="ts">
  import { app } from "./state.svelte";
  import Prose from "./Prose.svelte";
  import Tex from "./Tex.svelte";
  import { countTex, sciTex } from "./format";
  import { RUN_COLORS } from "./colors";

  /** What stopped the run, and for divergence: why, and what to do about it (DESIGN.md §4.3). */
  const stop = $derived(app.stop);
  const runs = $derived(app.status?.runs ?? []);
  const name = (i: number) => {
    const id = runs[i]?.opt;
    return app.optimizers.find((o) => o.id === id)?.name ?? "run";
  };
  const symbolOf = (i: number, key: string) => {
    const m = app.optimizers.find((o) => o.id === runs[i]?.opt);
    return m?.params.find((p) => p.key === key)?.symbol ?? key;
  };
  const titles: Record<string, string> = {
    diverging: "is diverging",
    unbounded: "has found no bottom",
    diverged: "diverged",
    "out-of-view": "left the view",
  };
</script>

{#if stop}
  <div class="panel notice" role="status">
    {#if stop.kind === "health"}
      {@const d = runs[stop.run]?.diagnosis}
      <div class="title"><i class="dot" style:background={RUN_COLORS[stop.run % RUN_COLORS.length].css}></i>{name(stop.run)} {titles[stop.health]} at <Tex tex={`k = ${countTex(app.status?.iter ?? 0)}`} /></div>
      {#if d}<p><Prose text={d.text} /></p>{/if}
      <div class="actions">
        {#if d?.suggest}
          <button class="primary" onclick={() => app.applySuggestion(stop.run, d.suggest![0], d.suggest![1])}>
            Restart with <Tex tex={`${symbolOf(stop.run, d.suggest[0])} = ${sciTex(d.suggest[1], 2)}`} />
          </button>
        {/if}
        <button onclick={() => app.engine({ type: "fitView" })} title="Enlarge the drawn area to take in every trail">Zoom out to follow</button>
        {#if stop.health !== "diverged"}<button onclick={() => app.play()}>Continue anyway</button>{/if}
        <button class="quiet" onclick={() => (app.stop = null)}>Dismiss</button>
      </div>
    {:else}
      <div class="title">
        Stopped at <Tex tex={`k = ${countTex(app.status?.iter ?? 0)}`} />:
        {#if stop.kind === "iteration" || stop.kind === "every"}iteration reached
        {:else if stop.kind === "f-below"}{name(stop.run)} reached <Tex tex={`f = ${sciTex(stop.f)}`} />
        {:else if stop.kind === "grad-below"}{name(stop.run)} reached <Tex tex={`\\lVert\\nabla f\\rVert = ${sciTex(stop.grad)}`} />
        {:else if stop.kind === "line"}{name(stop.run)} executed line {stop.line + 1}{/if}
      </div>
      <div class="actions">
        <button class="primary" onclick={() => app.play()}>Continue</button>
        <button onclick={() => app.step()}>One step</button>
        <button class="quiet" onclick={() => (app.stop = null)}>Dismiss</button>
      </div>
    {/if}
  </div>
{/if}

<style>
  .notice { position: absolute; left: 50%; bottom: 76px; transform: translateX(-50%); width: min(560px, calc(100vw - 32px)); padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; }
  .title { font-weight: 600; display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
  .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  p { margin: 0; line-height: 1.5; color: var(--ink); font-size: 12.5px; }
  .actions { display: flex; flex-wrap: wrap; gap: 6px; }
  .actions button { padding: 4px 10px; border-radius: 6px; border: 1px solid var(--rule); font-size: 12.5px; }
  .actions button:hover { background: var(--track); }
  .actions .primary { background: var(--accent); color: #fff; border-color: transparent; }
  .actions .primary:hover { background: color-mix(in srgb, var(--accent) 85%, black); }
  .actions .quiet { border-color: transparent; color: var(--dim); }
</style>
