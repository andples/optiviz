<script lang="ts">
  import { onMount } from "svelte";
  import { boot, type Caps, type Session, type Tier } from "../app";
  import type { EngineMsg, EngineStatus, OptimizerManifest, ProblemManifest, RenderMsg, RenderStats, Theme } from "../engine/protocol";
  import Pseudocode from "./Pseudocode.svelte";
  import Tex from "./Tex.svelte";
  import { count, sci } from "./format";

  const THEMES: Record<"light" | "dark", Theme> = {
    light: { bg: [0.98, 0.976, 0.965], dark: false },
    dark: { bg: [0.086, 0.094, 0.11], dark: true },
  };
  const SPEEDS = [
    { v: 10, label: "10/s" },
    { v: 60, label: "60/s" },
    { v: 600, label: "600/s" },
    { v: 0, label: "max" },
  ];

  let host: HTMLDivElement;
  let canvas = $state<HTMLCanvasElement>();
  let session: Session | null = null;
  let tier = $state<Tier | null>(null);
  let caps = $state<Caps | null>(null);
  let error = $state("");

  let problems = $state<ProblemManifest[]>([]);
  let optimizers = $state<OptimizerManifest[]>([]);
  let problemId = $state("rosenbrock");
  let optId = $state("adam");
  let params = $state<Record<string, Record<string, number>>>({});
  let status = $state<EngineStatus | null>(null);
  let stats = $state<RenderStats | null>(null);
  let running = $state(false);
  let speed = $state(60);
  let theme = $state<"light" | "dark">("light");
  let showInfo = $state(false);
  let showProblems = $state(false);
  let showCode = $state(false);
  let camMode = $state<"orbit" | "free">("orbit");
  const keys = { fwd: false, back: false, left: false, right: false, up: false, down: false, fast: false };
  const KEYMAP: Record<string, keyof typeof keys> = {
    ArrowUp: "fwd", ArrowDown: "back", ArrowLeft: "left", ArrowRight: "right", PageUp: "up", PageDown: "down",
  };

  const problem = $derived(problems.find((p) => p.id === problemId));
  const opt = $derived(optimizers.find((o) => o.id === optId));

  function onEngine(m: EngineMsg) {
    if (m.type === "ready") {
      problems = m.problems;
      optimizers = m.optimizers;
      params = Object.fromEntries(m.optimizers.map((o) => [o.id, Object.fromEntries(o.params.map((p) => [p.key, p.default]))]));
    } else if (m.type === "status") status = m.status;
    else if (m.type === "problem") session?.render({ type: "problem", manifest: m.manifest, zrange: m.zrange });
    else if (m.type === "error") error = m.message;
  }
  function onRender(m: RenderMsg) {
    if (m.type === "stats") stats = m.stats;
    else if (m.type === "error") error = m.message;
  }

  function startRun() {
    session?.engine({ type: "run", problem: problemId, optimizer: optId, seed: 1, params: $state.snapshot(params[optId]) });
    if (running) session?.engine({ type: "play" });
  }

  onMount(() => {
    (async () => {
      try {
        const b = await boot(canvas!, THEMES[theme], {
          engine: onEngine,
          render: onRender,
          freshCanvas: () => {
            const c = document.createElement("canvas");
            c.className = "view";
            canvas!.replaceWith(c);
            canvas = c;
            attachInput(c);
            return c;
          },
        });
        session = b.session;
        tier = b.tier;
        caps = b.caps;
        session.engine({ type: "speed", speed });
        startRun();
        play();
        const q = new URLSearchParams(location.search);
        const stress = Number(q.get("stress") ?? 0);
        if (stress > 0) session.engine({ type: "stress", points: stress });
      } catch (e) {
        error = String(e);
      }
    })();
    attachInput(canvas!);
    const ro = new ResizeObserver(() => {
      const r = host.getBoundingClientRect();
      session?.render({ type: "resize", width: r.width, height: r.height, dpr: Math.min(devicePixelRatio || 1, 2) });
    });
    ro.observe(host);
    return () => ro.disconnect();
  });

  function attachInput(el: HTMLCanvasElement) {
    const pts = new Map<number, { x: number; y: number }>();
    let pinch = 0;
    el.addEventListener("pointerdown", (e) => {
      el.setPointerCapture(e.pointerId);
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    });
    el.addEventListener("pointermove", (e) => {
      const p = pts.get(e.pointerId);
      if (!p) return;
      if (pts.size === 1) session?.render({ type: "orbit", dx: e.clientX - p.x, dy: e.clientY - p.y });
      p.x = e.clientX;
      p.y = e.clientY;
      if (pts.size === 2) {
        const [a, b] = [...pts.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch) session?.render({ type: "zoom", factor: pinch / d });
        pinch = d;
      }
    });
    const up = (e: PointerEvent) => {
      pts.delete(e.pointerId);
      pinch = 0;
    };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      session?.render({ type: "zoom", factor: Math.exp(e.deltaY * 0.0012) });
    }, { passive: false });
  }

  function play() {
    running = true;
    session?.engine({ type: "play" });
  }
  function pause() {
    running = false;
    session?.engine({ type: "pause" });
  }
  function toggle() {
    running ? pause() : play();
  }
  function step() {
    pause();
    session?.engine({ type: "step", n: 1 });
  }
  function restart() {
    session?.engine({ type: "restart" });
  }
  function setSpeed(v: number) {
    speed = v;
    session?.engine({ type: "speed", speed: v });
  }
  function pickProblem(id: string) {
    problemId = id;
    showProblems = false;
    startRun();
  }
  function pickOpt(id: string) {
    optId = id;
    startRun();
  }
  function setTheme(t: "light" | "dark") {
    theme = t;
    document.documentElement.dataset.theme = t;
    session?.render({ type: "theme", theme: THEMES[t] });
  }

  // parameter sliders: 0..1000 mapped linearly or logarithmically onto [min, max]
  const toSlider = (s: { min: number; max: number; log: boolean }, v: number) =>
    s.log ? (1000 * Math.log(v / s.min)) / Math.log(s.max / s.min) : (1000 * (v - s.min)) / (s.max - s.min);
  const fromSlider = (s: { min: number; max: number; log: boolean }, t: number) =>
    s.log ? s.min * (s.max / s.min) ** (t / 1000) : s.min + ((s.max - s.min) * t) / 1000;
  const paramValue = (key: string, dflt: number) => params[optId]?.[key] ?? dflt;
  function setParam(key: string, v: number) {
    params[optId] = { ...(params[optId] ?? {}), [key]: v };
    session?.engine({ type: "param", key, value: v });
  }

  function setCam(mode: "orbit" | "free") {
    camMode = mode;
    session?.render({ type: "camera", mode });
  }
  function sendKeys() {
    session?.render({ type: "keys", keys: { ...keys } });
  }
  function onKey(e: KeyboardEvent) {
    if ((e.target as HTMLElement)?.tagName === "INPUT") return;
    const move = KEYMAP[e.key];
    if (move) {
      // the arrow keys fly the free camera; the first press switches to it
      e.preventDefault();
      if (camMode !== "free") setCam("free");
      if (!keys[move] || keys.fast !== e.shiftKey) {
        keys[move] = true;
        keys.fast = e.shiftKey;
        sendKeys();
      }
      return;
    }
    if (e.key === " ") { e.preventDefault(); toggle(); }
    else if (e.key === ".") step();
    else if (e.key === "r") restart();
    else if (e.key === "Escape") {
      if (camMode === "free") setCam("orbit");
      showProblems = showInfo = false;
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

  <header class="top">
    <button class="problem" onclick={() => (showProblems = !showProblems)} title="Choose the problem">
      {#if problem}
        <span class="name">{problem.name}</span>
        <span class="formula"><Tex tex={problem.tex} /></span>
      {:else}
        <span class="name">loading…</span>
      {/if}
    </button>
    <div class="right">
      {#if camMode === "free"}
        <button class="cam" onclick={() => setCam("orbit")} title="Back to the orbit camera (Esc)">
          Free camera · ↑↓←→ move · PgUp/PgDn rise · drag to look · <b>orbit</b>
        </button>
      {:else}
        <button class="icon" onclick={() => setCam("free")} title="Free camera (arrow keys)">✥</button>
      {/if}
      <button class="icon" onclick={() => setTheme(theme === "light" ? "dark" : "light")} title="Light / dark">
        {theme === "light" ? "◐" : "◑"}
      </button>
      <button class="icon" class:on={showInfo} onclick={() => (showInfo = !showInfo)} title="Browser capabilities and performance">i</button>
    </div>
  </header>

  {#if showProblems}
    <div class="menu panel">
      {#each problems as p}
        <button class:sel={p.id === problemId} onclick={() => pickProblem(p.id)}>
          <span class="name">{p.name}</span>
          <span class="formula"><Tex tex={p.tex} /></span>
        </button>
      {/each}
    </div>
  {/if}

  {#if opt}
    <aside class="inspector panel">
      <div class="runs">
        {#each optimizers as o}
          <button class:sel={o.id === optId} onclick={() => pickOpt(o.id)}><i class="dot"></i>{o.name}</button>
        {/each}
      </div>
      <div class="rule"><Tex tex={opt.tex} /></div>
      {#each opt.params as s (optId + s.key)}
        {@const v = paramValue(s.key, s.default)}
        <label class="param" title={s.description}>
          <span class="sym"><Tex tex={s.symbol} /></span>
          <input type="range" min="0" max="1000" value={toSlider(s, v)} oninput={(e) => setParam(s.key, fromSlider(s, +e.currentTarget.value))} />
          <span class="val">{sci(v, 2)}</span>
        </label>
      {/each}
      <button class="toggle" onclick={() => (showCode = !showCode)}>{showCode ? "Hide" : "Show"} the algorithm</button>
    </aside>
    {#if showCode}
      <aside class="code panel">
        <Pseudocode code={opt.pseudocode} pulse={status?.iter ?? 0} {running} />
      </aside>
    {/if}
  {/if}

  {#if showInfo && tier && caps}
    <aside class="info panel">
      <div><b>Rendering</b> {tier.backend === "webgpu" ? "WebGPU" : "WebGL2"} in the {tier.render === "worker" ? "render worker" : "main thread"}</div>
      <div class="dim">{tier.adapter}</div>
      <div><b>Engine</b> WebAssembly {caps.wasmSimd ? "SIMD" : ""}, {tier.threads ? `${tier.threadCount} threads` : "single thread"}</div>
      <div><b>Frames</b> {tier.transport === "shared" ? "shared memory (triple buffer)" : "message passing"}</div>
      {#if stats}
        <div><b>Display</b> {stats.fps.toFixed(0)} fps · {stats.frameMs.toFixed(2)} ms/frame · frame age {stats.latencyMs.toFixed(1)} ms</div>
      {/if}
      {#if status}
        <div><b>Engine rate</b> {sci(status.stepsPerSec, 2)} steps/s · {status.framesPerSec.toFixed(0)} frames/s · publish {status.publishMs.toFixed(2)} ms</div>
        {#if status.stressPoints}<div><b>Stress</b> {count(status.stressPoints)} points per frame</div>{/if}
      {/if}
      {#if !caps.webgpu}<div class="note">WebGPU isn't available in this browser; drawing with WebGL2.</div>{/if}
      {#if !caps.isolated}<div class="note">This page isn't cross-origin isolated, so the engine runs on one thread.</div>{/if}
    </aside>
  {/if}

  <footer class="bottom panel">
    <div class="controls">
      <button class="icon" onclick={toggle} title={running ? "Pause (space)" : "Play (space)"}>{running ? "❚❚" : "▶"}</button>
      <button class="icon" onclick={step} title="One step (.)">⏵|</button>
      <button class="icon" onclick={restart} title="Restart (r)">↺</button>
    </div>
    <div class="speeds" role="group" aria-label="Speed">
      {#each SPEEDS as s}
        <button class:sel={speed === s.v} onclick={() => setSpeed(s.v)}>{s.label}</button>
      {/each}
    </div>
    <div class="readout">
      {#if status}
        <span><Tex tex="k" /> = {count(status.iter)}</span>
        <span><Tex tex={"f(x_k)"} /> = {sci(status.f)}</span>
        <span><Tex tex={"\\lVert\\nabla f\\rVert"} /> = {sci(status.gradNorm)}</span>
        {#if status.diverged}<span class="warn">diverged</span>{/if}
      {/if}
    </div>
  </footer>

  {#if error}<div class="error panel">{error}</div>{/if}
</div>

<style>
  .stage { position: fixed; inset: 0; }
  :global(canvas.view) { position: absolute; inset: 0; width: 100%; height: 100%; display: block; touch-action: none; cursor: grab; }
  :global(canvas.view:active) { cursor: grabbing; }

  .panel {
    background: var(--panel);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--rule);
    border-radius: 10px;
    box-shadow: var(--shadow);
  }
  button { background: none; border: 0; cursor: pointer; padding: 0; }

  .top { position: absolute; top: 14px; left: 16px; right: 16px; display: flex; justify-content: space-between; align-items: flex-start; pointer-events: none; }
  .top > * { pointer-events: auto; }
  .problem { display: flex; flex-direction: column; align-items: flex-start; gap: 2px; text-align: left; padding: 4px 6px; border-radius: 8px; }
  .problem:hover { background: var(--panel); }
  .problem .name { font-weight: 600; font-size: 14px; }
  .problem .formula { color: var(--dim); }
  .right { display: flex; gap: 6px; }
  .cam { height: 30px; padding: 0 10px; border-radius: 8px; font-size: 12px; color: var(--dim); background: var(--panel); border: 1px solid var(--rule); }
  .cam b { color: var(--accent); font-weight: 600; }
  .icon { width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center; color: var(--dim); font-size: 14px; }
  .icon:hover, .icon.on { background: var(--panel); color: var(--ink); }

  .menu { position: absolute; top: 72px; left: 16px; padding: 6px; display: flex; flex-direction: column; min-width: 340px; }
  .menu button { display: flex; flex-direction: column; align-items: flex-start; gap: 2px; padding: 8px 10px; border-radius: 7px; text-align: left; }
  .menu button:hover { background: var(--track); }
  .menu button.sel { background: var(--track); }
  .menu .formula { color: var(--dim); font-size: 12px; }

  .inspector { position: absolute; top: 60px; right: 16px; width: 260px; padding: 10px 12px; display: flex; flex-direction: column; gap: 8px; }
  .runs { display: flex; flex-wrap: wrap; gap: 4px; }
  .runs button { padding: 3px 8px; border-radius: 6px; color: var(--dim); display: flex; align-items: center; gap: 6px; }
  .runs button.sel { color: var(--ink); background: var(--track); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--faint); display: inline-block; }
  .runs button.sel .dot { background: var(--accent); }
  .rule { color: var(--dim); font-size: 11.5px; overflow-x: auto; padding: 2px 0 4px; border-bottom: 1px solid var(--rule); }
  .param { display: grid; grid-template-columns: 26px 1fr 58px; align-items: center; gap: 8px; }
  .param .sym { text-align: center; }
  .param .val { text-align: right; color: var(--dim); font-size: 12px; }
  input[type="range"] { width: 100%; accent-color: var(--accent); height: 18px; }

  .toggle { align-self: flex-start; color: var(--dim); font-size: 12px; padding: 2px 0; }
  .toggle:hover { color: var(--ink); }
  .code { position: absolute; top: 96px; left: 16px; width: 430px; max-height: calc(100vh - 190px); overflow: auto; padding: 10px 12px; }
  .info { position: absolute; top: 60px; right: 290px; width: 340px; padding: 10px 12px; display: flex; flex-direction: column; gap: 4px; font-size: 12px; }
  .info .dim { color: var(--dim); margin-top: -2px; word-break: break-word; }
  .info .note { color: var(--dim); font-style: italic; }

  .bottom { position: absolute; left: 50%; bottom: 16px; transform: translateX(-50%); display: flex; align-items: center; gap: 14px; padding: 6px 12px; max-width: calc(100vw - 32px); flex-wrap: wrap; }
  .controls { display: flex; gap: 2px; }
  .speeds { display: flex; background: var(--track); border-radius: 7px; padding: 2px; }
  .speeds button { padding: 2px 8px; border-radius: 5px; color: var(--dim); font-size: 12px; }
  .speeds button.sel { background: var(--panel-solid); color: var(--ink); }
  .readout { display: flex; gap: 16px; min-width: 330px; }
  .warn { color: var(--accent); }
  .error { position: absolute; left: 16px; bottom: 76px; padding: 10px 12px; color: var(--accent); max-width: 60ch; }

  @media (max-width: 720px) {
    .inspector { top: auto; bottom: 120px; right: 16px; left: 16px; width: auto; }
    .info { right: 16px; left: 16px; width: auto; }
    .code { display: none; }
    .readout { min-width: 0; }
  }
</style>
