<script lang="ts">
  import { onMount } from "svelte";
  import { boot } from "../app";
  import { app, selectedRun, THEMES } from "./state.svelte";
  import BottomBar from "./BottomBar.svelte";
  import InfoPanel from "./InfoPanel.svelte";
  import Metrics from "./Metrics.svelte";
  import Notice from "./Notice.svelte";
  import ProblemPanel from "./ProblemPanel.svelte";
  import Pseudocode from "./Pseudocode.svelte";
  import RunsPanel from "./RunsPanel.svelte";
  import StopsPanel from "./StopsPanel.svelte";
  import TopBar from "./TopBar.svelte";

  let host: HTMLDivElement;
  let canvas = $state<HTMLCanvasElement>();
  const sel = $derived(selectedRun());
  const traceFor = $derived(app.trace && app.trace.i === app.selected ? app.trace.entries : []);
  // the descent arrow shows while paused
  $effect(() => app.render({ type: "arrows", on: !app.running }));

  onMount(() => {
    (async () => {
      try {
        const b = await boot(canvas!, THEMES[app.theme], {
          engine: app.onEngine,
          render: app.onRender,
          freshCanvas: () => {
            const c = document.createElement("canvas");
            c.className = "view";
            canvas!.replaceWith(c);
            canvas = c;
            attachInput(c);
            return c;
          },
        });
        app.session = b.session;
        app.tier = b.tier;
        app.caps = b.caps;
        app.start();
        const stress = Number(new URLSearchParams(location.search).get("stress") ?? 0);
        if (stress > 0) app.engine({ type: "stress", points: stress });
      } catch (e) {
        app.error = String(e);
      }
    })();
    attachInput(canvas!);
    const ro = new ResizeObserver(() => {
      const r = host.getBoundingClientRect();
      app.render({ type: "resize", width: r.width, height: r.height, dpr: Math.min(devicePixelRatio || 1, 2) });
    });
    ro.observe(host);
    return () => ro.disconnect();
  });

  /** Drag orbits (or pans the contour map), the wheel and pinch zoom, a click without dragging sets the start. */
  function attachInput(el: HTMLCanvasElement) {
    const pts = new Map<number, { x: number; y: number }>();
    let pinch = 0;
    let down = { x: 0, y: 0, t: 0, moved: 0 };
    el.addEventListener("pointerdown", (e) => {
      el.setPointerCapture(e.pointerId);
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pts.size === 1) down = { x: e.clientX, y: e.clientY, t: performance.now(), moved: 0 };
      app.panel = "";
    });
    el.addEventListener("pointermove", (e) => {
      const p = pts.get(e.pointerId);
      if (!p) return;
      down.moved += Math.abs(e.clientX - p.x) + Math.abs(e.clientY - p.y);
      if (pts.size === 1) app.render({ type: "orbit", dx: e.clientX - p.x, dy: e.clientY - p.y });
      p.x = e.clientX;
      p.y = e.clientY;
      if (pts.size === 2) {
        const [a, b] = [...pts.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch) app.render({ type: "zoom", factor: pinch / d });
        pinch = d;
      }
    });
    const up = (e: PointerEvent) => {
      const wasSingle = pts.size === 1;
      pts.delete(e.pointerId);
      pinch = 0;
      if (e.type === "pointerup" && wasSingle && down.moved < 5 && performance.now() - down.t < 500) {
        const r = el.getBoundingClientRect();
        app.render({ type: "pick", x: e.clientX - r.left, y: e.clientY - r.top });
      }
    };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      app.render({ type: "zoom", factor: Math.exp(e.deltaY * 0.0012) });
    }, { passive: false });
  }

  const keys = { fwd: false, back: false, left: false, right: false, up: false, down: false, fast: false };
  const KEYMAP: Record<string, keyof typeof keys> = {
    ArrowUp: "fwd", ArrowDown: "back", ArrowLeft: "left", ArrowRight: "right", PageUp: "up", PageDown: "down",
  };
  const sendKeys = () => app.render({ type: "keys", keys: { ...keys } });

  function onKey(e: KeyboardEvent) {
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    const move = KEYMAP[e.key];
    if (move) {
      if (app.view !== "surface") return;
      // the arrow keys fly the free camera; the first press switches to it
      e.preventDefault();
      if (app.camMode !== "free") app.setCam("free");
      if (!keys[move] || keys.fast !== e.shiftKey) {
        keys[move] = true;
        keys.fast = e.shiftKey;
        sendKeys();
      }
      return;
    }
    if (e.key === " ") { e.preventDefault(); app.toggle(); }
    else if (e.key === ".") app.step();
    else if (e.key === ",") app.stepLine();
    else if (e.key === "r") app.restart();
    else if (e.key === "Escape") {
      if (app.camMode === "free") app.setCam("orbit");
      app.panel = "";
      app.stop = null;
    }
  }
  function onKeyUp(e: KeyboardEvent) {
    const move = KEYMAP[e.key];
    if (move && keys[move]) {
      keys[move] = false;
      keys.fast = e.shiftKey;
      sendKeys();
    }
  }
  function releaseKeys() {
    for (const k of Object.keys(keys) as (keyof typeof keys)[]) keys[k] = false;
    sendKeys();
  }
</script>

<svelte:window onkeydown={onKey} onkeyup={onKeyUp} onblur={releaseKeys} />

<div class="stage" bind:this={host}>
  <canvas class="view" bind:this={canvas}></canvas>
  <TopBar />
  {#if app.panel === "problems"}<ProblemPanel />{/if}
  {#if app.panel === "info"}<InfoPanel />{/if}
  <RunsPanel />
  {#if app.showCode && sel.manifest}
    <aside class="code panel">
      <Pseudocode code={sel.manifest.pseudocode} entries={traceFor} running={app.running} cursor={app.lineCursor} />
    </aside>
  {/if}
  <Metrics />
  {#if app.panel === "stops"}<StopsPanel />{/if}
  <Notice />
  <BottomBar />
  {#if app.error}<button class="error panel" onclick={() => (app.error = "")} title="Dismiss">{app.error}</button>{/if}
</div>

<style>
  .stage { position: fixed; inset: 0; }
  :global(canvas.view) { position: absolute; inset: 0; width: 100%; height: 100%; display: block; touch-action: none; cursor: grab; }
  :global(canvas.view:active) { cursor: grabbing; }
  :global(.panel) {
    background: var(--panel);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--rule);
    border-radius: 10px;
    box-shadow: var(--shadow);
  }
  :global(button) { background: none; border: 0; cursor: pointer; padding: 0; }
  .code { position: absolute; top: 76px; left: 16px; width: 440px; max-height: calc(100vh - 180px); overflow: auto; padding: 10px 12px; }
  .error { position: absolute; left: 16px; bottom: 76px; padding: 10px 12px; color: var(--accent); max-width: 60ch; text-align: left; }
  @media (max-width: 720px) { .code { display: none; } }
</style>
