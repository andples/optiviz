<script lang="ts">
  import type { Pseudocode } from "../engine/protocol";
  import Prose from "./Prose.svelte";
  import Tex from "./Tex.svelte";

  /** `pulse` changes whenever a step has run; the lines a step executes flash. */
  let { code, pulse, running }: { code: Pseudocode; pulse: number; running: boolean } = $props();
  let flash = $state(false);
  let hover = $state<number | null>(null);
  $effect(() => {
    pulse;
    if (running) return;
    flash = true;
    const t = setTimeout(() => (flash = false), 450);
    return () => clearTimeout(t);
  });
  const deviated = $derived(new Set(code.deviations.map((d) => d.line).filter((l) => l !== null)));
</script>

<div class="pc">
  <ol>
    {#each code.lines as line, i}
      <li
        class:step={line.step}
        class:lit={line.step && (running || flash)}
        class:dev={deviated.has(i)}
        class:hover={hover === i}
        style:padding-left="{line.indent * 1.4 + 0.3}em"
        onmouseenter={() => (hover = deviated.has(i) ? i : null)}
        onmouseleave={() => (hover = null)}
      >
        <Tex tex={line.tex} />
        {#if deviated.has(i)}<span class="mark" title="The implementation differs here">Δ</span>{/if}
      </li>
    {/each}
  </ol>
  <div class="source">{code.source}</div>
  {#if code.deviations.length}
    <div class="devs">
      <div class="head">Where the implementation differs</div>
      {#each code.deviations as d}
        <div class="d" class:hover={hover !== null && hover === d.line}>
          {#if d.line !== null}<span class="ln">line {d.line + 1}</span>{/if}
          <div><Prose text={d.what} /></div>
          <div class="why"><Prose text={d.why} /></div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .pc { display: flex; flex-direction: column; gap: 8px; font-size: 12px; }
  ol { margin: 0; padding: 0 0 0 2.2em; display: flex; flex-direction: column; gap: 1px; }
  li { padding: 2px 4px; border-radius: 4px; position: relative; transition: background 0.35s; color: var(--dim); }
  li::marker { color: var(--faint); font-size: 10.5px; }
  li.step { color: var(--ink); }
  li.lit { background: color-mix(in srgb, var(--accent) 13%, transparent); transition: background 0.05s; }
  li.hover { outline: 1px solid color-mix(in srgb, var(--accent) 45%, transparent); }
  .mark { position: absolute; right: 4px; top: 2px; color: var(--accent); font-size: 11px; cursor: help; }
  .source { color: var(--dim); font-style: italic; font-size: 11.5px; }
  .devs { border-top: 1px solid var(--rule); padding-top: 6px; display: flex; flex-direction: column; gap: 8px; }
  .head { font-weight: 600; font-size: 11.5px; }
  .d { padding: 4px 6px; border-radius: 5px; line-height: 1.45; }
  .d.hover { background: var(--track); }
  .ln { color: var(--accent); font-size: 11px; margin-right: 4px; }
  .why { color: var(--dim); margin-top: 2px; }
</style>
