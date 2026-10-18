<script lang="ts">
  import { app } from "./state.svelte";
  import ParamSlider from "./ParamSlider.svelte";
  import Tex from "./Tex.svelte";

  const current = $derived(app.info?.manifest);
  const coeff = (key: string, dflt: number) => app.info?.coeffs[key] ?? dflt;
  const customized = $derived(!!app.info && Object.keys(app.info.coeffs).length > 0);
</script>

<div class="panel pp">
  <div class="list">
    {#each app.problems as p}
      <button class:sel={p.id === current?.id} onclick={() => app.pickProblem(p.id)}>
        <span class="name">{p.name}</span>
        <span class="blurb">{p.blurb}</span>
      </button>
    {/each}
  </div>
  {#if current}
    <div class="coeffs">
      <div class="head">
        <span class="name">{current.name}</span>
        {#if customized}<button class="link" onclick={() => app.resetCoeffs()}>textbook values</button>{/if}
      </div>
      <div class="formula"><Tex tex={current.tex} display /></div>
      {#if current.params.length}
        {#each current.params as s (current.id + s.key)}
          <ParamSlider spec={s} value={coeff(s.key, s.default)} onchange={(v) => app.setCoeff(s.key, v)} />
        {/each}
      {:else}
        <p class="dim">This problem has no coefficients to change.</p>
      {/if}
      <p class="dim small">
        {#if current.minima.length}
          Minimum {current.minima.length > 1 ? `(one of ${current.minima.length})` : ""} at
          <Tex tex={`(${current.minima[0].map((v) => v.toFixed(4)).join(",\\ ")})`} />{#if current.f_min !== null}, <Tex tex={`f^* = ${current.f_min.toPrecision(6)}`} />{/if}.
        {:else}
          No minimum in view{current.f_min === null ? ": f is unbounded below or the minima are elsewhere" : ""}.
        {/if}
        Click the surface to start from another point.
      </p>
    </div>
  {/if}
</div>

<style>
  .pp { position: absolute; top: 76px; left: 16px; display: grid; grid-template-columns: 260px 300px; max-height: calc(100vh - 170px); overflow: hidden; }
  .list { overflow: auto; padding: 6px; border-right: 1px solid var(--rule); }
  .list button { display: flex; flex-direction: column; align-items: flex-start; gap: 1px; padding: 7px 9px; border-radius: 7px; text-align: left; width: 100%; }
  .list button:hover, .list button.sel { background: var(--track); }
  .name { font-weight: 600; }
  .blurb { color: var(--dim); font-size: 11.5px; line-height: 1.35; }
  .coeffs { padding: 10px 12px; display: flex; flex-direction: column; gap: 9px; overflow: auto; }
  .head { display: flex; justify-content: space-between; align-items: baseline; }
  .formula { overflow-x: auto; font-size: 12px; }
  .link { color: var(--accent); font-size: 12px; }
  .dim { color: var(--dim); margin: 0; }
  .small { font-size: 11.5px; line-height: 1.5; }
  @media (max-width: 720px) { .pp { grid-template-columns: 1fr; right: 16px; } .list { max-height: 30vh; border-right: 0; border-bottom: 1px solid var(--rule); } }
</style>
