<script lang="ts">
  import type { ParamSpec } from "../engine/protocol";
  import Tex from "./Tex.svelte";
  import { sci } from "./format";

  /** One parameter: a slider (log or linear) with an exact-value field, or a segmented choice. */
  let { spec, value, onchange }: { spec: ParamSpec; value: number; onchange: (v: number) => void } = $props();

  const toSlider = (v: number) =>
    spec.log ? (1000 * Math.log(v / spec.min)) / Math.log(spec.max / spec.min) : (1000 * (v - spec.min)) / (spec.max - spec.min);
  const fromSlider = (t: number) => (spec.log ? spec.min * (spec.max / spec.min) ** (t / 1000) : spec.min + ((spec.max - spec.min) * t) / 1000);
  let editing = $state(false);
  let text = $state("");

  function commit() {
    editing = false;
    const v = Number(text.replace("·10^", "e").replace("−", "-").replace(",", "."));
    if (Number.isFinite(v)) onchange(Math.min(spec.max, Math.max(spec.min, v)));
  }
</script>

{#if spec.choices.length}
  <div class="param choice" title={spec.description}>
    <span class="sym"><Tex tex={spec.symbol} /></span>
    <div class="seg" role="group" aria-label={spec.description}>
      {#each spec.choices as label, i}
        <button class:sel={Math.round(value) === i} onclick={() => onchange(i)}>{label}</button>
      {/each}
    </div>
  </div>
{:else}
  <label class="param" title={spec.description}>
    <span class="sym"><Tex tex={spec.symbol} /></span>
    <input type="range" min="0" max="1000" step="1" value={toSlider(value)} oninput={(e) => onchange(fromSlider(+e.currentTarget.value))} />
    {#if editing}
      <!-- svelte-ignore a11y_autofocus -->
      <input class="val edit" bind:value={text} autofocus onblur={commit} onkeydown={(e) => { if (e.key === "Enter") commit(); if (e.key === "Escape") editing = false; }} />
    {:else}
      <button class="val" title="Type an exact value" onclick={() => { text = String(Number(value.toPrecision(6))); editing = true; }}>{sci(value, 2)}</button>
    {/if}
  </label>
{/if}

<style>
  .param { display: grid; grid-template-columns: 30px 1fr 64px; align-items: center; gap: 8px; }
  .choice { grid-template-columns: auto 1fr; }
  .choice .sym { text-align: left; font-size: 12px; color: var(--dim); }
  .sym { text-align: center; }
  .val { text-align: right; color: var(--dim); font-size: 12px; font: inherit; font-size: 12px; padding: 1px 2px; border-radius: 4px; }
  button.val:hover { color: var(--ink); background: var(--track); }
  .edit { width: 64px; border: 1px solid var(--rule); background: var(--panel-solid); color: var(--ink); text-align: right; }
  input[type="range"] { width: 100%; accent-color: var(--accent); height: 18px; }
  .seg { display: flex; flex-wrap: wrap; gap: 2px; background: var(--track); border-radius: 7px; padding: 2px; }
  .seg button { flex: 1; padding: 2px 6px; border-radius: 5px; color: var(--dim); font-size: 11.5px; white-space: nowrap; }
  .seg button.sel { background: var(--panel-solid); color: var(--ink); }
</style>
