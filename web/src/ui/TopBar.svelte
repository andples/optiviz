<script lang="ts">
  import { app } from "./state.svelte";
  import Tex from "./Tex.svelte";

  const problem = $derived(app.info?.manifest);
  const outside = $derived(app.status?.runs.some((r) => r.health !== "ok") ?? false);
  const zoomedOut = $derived.by(() => {
    const i = app.info;
    if (!i) return false;
    return i.view[0][0] !== i.manifest.bounds[0][0] || i.view[0][1] !== i.manifest.bounds[0][1] || i.view[1][0] !== i.manifest.bounds[1][0] || i.view[1][1] !== i.manifest.bounds[1][1];
  });
</script>

<header class="top">
  <button class="problem" onclick={() => (app.panel = app.panel === "problems" ? "" : "problems")} title="Choose the problem and its coefficients">
    {#if problem}
      <span class="name">{problem.name}{#if app.info && Object.keys(app.info.coeffs).length}<span class="custom"> · custom</span>{/if}</span>
      <span class="formula"><Tex tex={problem.tex} /></span>
    {:else}
      <span class="name">loading…</span>
    {/if}
  </button>
  <div class="right">
    {#if zoomedOut}
      <button class="chip" onclick={() => app.engine({ type: "resetView" })} title="Back to the problem's own box">Reset view</button>
    {:else if outside}
      <button class="chip warn" onclick={() => app.engine({ type: "fitView" })} title="A run left the drawn area: enlarge it to follow">Zoom out to follow</button>
    {/if}
    <div class="seg" role="group" aria-label="View">
      <button class:sel={app.view === "surface"} onclick={() => app.setView("surface")} title="Lit 3D surface">Surface</button>
      <button class:sel={app.view === "contour"} onclick={() => app.setView("contour")} title="Contour map from above">Contour</button>
    </div>
    {#if app.camMode === "free"}
      <button class="chip" onclick={() => app.setCam("orbit")} title="Back to the orbit camera (Esc)">Free camera · ↑↓←→ move · PgUp/PgDn rise · <b>orbit</b></button>
    {:else if app.view === "surface"}
      <button class="icon" onclick={() => app.setCam("free")} title="Free camera (arrow keys)">✥</button>
    {/if}
    <button class="icon" onclick={() => app.share()} title="Copy a link to this exact setup">{app.copied ? "✓" : "⤴"}</button>
    <button class="icon" onclick={() => app.setTheme(app.theme === "light" ? "dark" : "light")} title="Light / dark">{app.theme === "light" ? "◐" : "◑"}</button>
    <button class="icon" class:on={app.panel === "info"} onclick={() => (app.panel = app.panel === "info" ? "" : "info")} title="Browser capabilities and performance">i</button>
  </div>
</header>

<style>
  .top { position: absolute; top: 14px; left: 16px; right: 16px; display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; pointer-events: none; }
  .top > * { pointer-events: auto; }
  .problem { display: flex; flex-direction: column; align-items: flex-start; gap: 2px; text-align: left; padding: 4px 6px; border-radius: 8px; max-width: 50vw; }
  .problem:hover { background: var(--panel); }
  .name { font-weight: 600; font-size: 14px; }
  .custom { color: var(--accent); font-weight: 500; font-size: 12px; }
  .formula { color: var(--dim); overflow: hidden; max-width: 100%; }
  .right { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
  .icon { width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center; color: var(--dim); font-size: 14px; }
  .icon:hover, .icon.on { background: var(--panel); color: var(--ink); }
  .chip { height: 28px; padding: 0 10px; border-radius: 8px; font-size: 12px; color: var(--dim); background: var(--panel); border: 1px solid var(--rule); }
  .chip:hover { color: var(--ink); }
  .chip b { color: var(--accent); font-weight: 600; }
  .chip.warn { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 40%, transparent); }
  .seg { display: flex; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 2px; }
  .seg button { padding: 3px 9px; border-radius: 6px; color: var(--dim); font-size: 12px; }
  .seg button.sel { background: var(--track); color: var(--ink); }
</style>
