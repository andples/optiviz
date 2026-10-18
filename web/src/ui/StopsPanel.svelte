<script lang="ts">
  import { app } from "./state.svelte";
  import Tex from "./Tex.svelte";

  /** Stopping points (DESIGN.md §4.2). Values are typed; empty means off. */
  const num = (s: string) => {
    const v = Number(s.replace("−", "-"));
    return s.trim() === "" || !Number.isFinite(v) ? undefined : v;
  };
  const lineName = $derived.by(() => {
    const l = app.stops.line;
    return l === undefined ? "" : `line ${l + 1}`;
  });
</script>

<div class="panel stops">
  <div class="title">Stop the run when…</div>
  <label class="row">
    <input type="checkbox" checked={app.stops.on_diverge} onchange={(e) => app.setStops({ on_diverge: e.currentTarget.checked })} />
    <span>a run starts to diverge or turns out unbounded</span>
  </label>
  <label class="row">
    <span>iteration reaches <Tex tex="k =" /></span>
    <input class="num" placeholder="off" value={app.stops.at_iter ?? ""} onchange={(e) => app.setStops({ at_iter: num(e.currentTarget.value) })} />
  </label>
  <label class="row">
    <span>every</span>
    <input class="num" placeholder="off" value={app.stops.every ?? ""} onchange={(e) => app.setStops({ every: num(e.currentTarget.value) })} />
    <span>iterations</span>
  </label>
  <label class="row">
    <span><Tex tex="f(x_k) <" /></span>
    <input class="num" placeholder="off" value={app.stops.f_below ?? ""} onchange={(e) => app.setStops({ f_below: num(e.currentTarget.value) })} />
  </label>
  <label class="row">
    <span><Tex tex={"\\lVert \\nabla f(x_k) \\rVert <"} /></span>
    <input class="num" placeholder="off" value={app.stops.grad_below ?? ""} onchange={(e) => app.setStops({ grad_below: num(e.currentTarget.value) })} />
  </label>
  <div class="row">
    {#if lineName}
      <span>the selected run executes <b>{lineName}</b> of its pseudocode</span>
      <button class="link" onclick={() => app.setStops({ line: undefined })}>clear</button>
    {:else}
      <span class="dim">To stop on a pseudocode line, click its number in the algorithm panel.</span>
    {/if}
  </div>
</div>

<style>
  .stops { position: absolute; bottom: 76px; left: 50%; transform: translateX(-50%); width: 360px; padding: 10px 12px; display: flex; flex-direction: column; gap: 7px; font-size: 12.5px; }
  .title { font-weight: 600; }
  .row { display: flex; align-items: center; gap: 8px; }
  .num { width: 80px; font: inherit; padding: 2px 6px; border: 1px solid var(--rule); border-radius: 5px; background: var(--panel-solid); color: var(--ink); }
  input[type="checkbox"] { accent-color: var(--accent); }
  .link { color: var(--accent); font-size: 12px; }
  .dim { color: var(--dim); }
</style>
