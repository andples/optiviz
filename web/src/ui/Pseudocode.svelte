<script lang="ts">
  import type { Pseudocode, TraceEntry } from "../engine/protocol";
  import { app } from "./state.svelte";
  import Prose from "./Prose.svelte";
  import Tex from "./Tex.svelte";
  import { sciTex } from "./format";

  /**
   * The algorithm as its source states it (DESIGN.md §4.1), with the step debugger on top (§4.2): the lines the
   * last step executed are lit and show the values they computed; the gutter sets a stopping point on a line.
   */
  let { code, entries, running }: { code: Pseudocode; entries: TraceEntry[]; running: boolean } = $props();
  let hover = $state<number | null>(null);
  const deviated = $derived(new Set(code.deviations.map((d) => d.line).filter((l) => l !== null)));
  /** per line: how many times the last step ran it, and the values from the last time */
  const byLine = $derived.by(() => {
    const m = new Map<number, { n: number; vars: [string, number[]][] }>();
    for (const e of entries) {
      const cur = m.get(e.line);
      m.set(e.line, { n: (cur?.n ?? 0) + 1, vars: e.vars });
    }
    return m;
  });

  function fmt(v: number[]): string {
    if (v.length === 1) return sciTex(v[0]);
    if (v.length === 4) return `\\begin{pmatrix} ${sciTex(v[0])} & ${sciTex(v[1])} \\\\ ${sciTex(v[2])} & ${sciTex(v[3])} \\end{pmatrix}`;
    return `(${v.map((x) => sciTex(x)).join(",\\ ")})`;
  }
</script>

<div class="pc">
  <ol>
    {#each code.lines as line, i}
      {@const t = byLine.get(i)}
      <li
        class:step={line.step}
        class:lit={!!t && !running}
        class:dev={deviated.has(i)}
        class:hover={hover === i}
        class:stop={app.stops.line === i}
        onmouseenter={() => (hover = deviated.has(i) ? i : null)}
        onmouseleave={() => (hover = null)}
      >
        <button
          class="gutter"
          disabled={!line.step}
          title={line.step ? (app.stops.line === i ? "Remove the stopping point" : "Stop whenever the selected run executes this line") : ""}
          onclick={() => app.toggleLineStop(i)}
        >{i + 1}</button>
        <div class="body" style:padding-left="{line.indent * 1.3}em">
          <Tex tex={line.tex} />
          {#if t && !running}
            <div class="vals">
              {#if t.n > 1}<span class="times">×{t.n}</span>{/if}
              {#each t.vars as [name, v]}
                <span class="var"><Tex tex={`${name} = ${fmt(v)}`} /></span>
              {/each}
            </div>
          {/if}
        </div>
        {#if deviated.has(i)}<span class="mark" title="The implementation differs here; see below">Δ</span>{/if}
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
  ol { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 1px; }
  li { display: grid; grid-template-columns: 24px 1fr 12px; align-items: start; border-radius: 4px; transition: background 0.35s; color: var(--dim); }
  li.step { color: var(--ink); }
  li.lit { background: color-mix(in srgb, var(--accent) 10%, transparent); }
  li.hover { outline: 1px solid color-mix(in srgb, var(--accent) 45%, transparent); }
  .gutter { color: var(--faint); font-size: 10.5px; text-align: right; padding: 3px 5px 0 0; border-radius: 4px; position: relative; }
  .gutter:not(:disabled):hover { color: var(--accent); }
  .gutter:disabled { cursor: default; }
  li.stop .gutter { color: #fff; background: var(--accent); }
  .body { padding: 2px 4px; min-width: 0; overflow-x: auto; }
  .vals { display: flex; flex-wrap: wrap; gap: 2px 12px; margin: 2px 0 2px; color: var(--accent); font-size: 11px; }
  .times { color: var(--dim); }
  .mark { color: var(--accent); font-size: 11px; cursor: help; padding-top: 2px; }
  .source { color: var(--dim); font-style: italic; font-size: 11.5px; }
  .devs { border-top: 1px solid var(--rule); padding-top: 6px; display: flex; flex-direction: column; gap: 8px; }
  .head { font-weight: 600; font-size: 11.5px; }
  .d { padding: 4px 6px; border-radius: 5px; line-height: 1.45; }
  .d.hover { background: var(--track); }
  .ln { color: var(--accent); font-size: 11px; margin-right: 4px; }
  .why { color: var(--dim); margin-top: 2px; }
</style>
