/**
 * The UI's state and actions, shared by every component. The engine worker is the source of truth for runs and
 * the timeline; this mirrors what it reports (status ~10×/s, metrics ~5×/s) and sends commands. Shareable
 * state (problem, coefficients, start, runs, view) round-trips through the URL hash (`share`).
 */

import type { Caps, Session, Tier } from "../app";
import type {
  EngineMsg,
  EngineStatus,
  OptimizerManifest,
  ProblemInfo,
  ProblemManifest,
  RenderMsg,
  RenderStats,
  Stop,
  Stops,
  Theme,
  TraceEntry,
  ViewMode,
} from "../engine/protocol";

export const THEMES: Record<"light" | "dark", Theme> = {
  light: { bg: [0.98, 0.976, 0.965], dark: false },
  dark: { bg: [0.086, 0.094, 0.11], dark: true },
};

export const SPEEDS = [
  { v: 10, label: "10/s" },
  { v: 60, label: "60/s" },
  { v: 600, label: "600/s" },
  { v: 0, label: "max" },
];

/** What a share link carries. */
interface Shared {
  p: string;
  c?: Record<string, number>;
  s?: number[];
  r: { o: string; p?: Record<string, number> }[];
  v?: ViewMode;
}

const DEFAULT: Shared = { p: "rosenbrock", r: [{ o: "adam" }, { o: "momentum" }] };

class AppState {
  session: Session | null = null;
  tier = $state<Tier | null>(null);
  caps = $state<Caps | null>(null);
  error = $state("");

  problems = $state<ProblemManifest[]>([]);
  optimizers = $state<OptimizerManifest[]>([]);
  info = $state<ProblemInfo | null>(null);
  status = $state<EngineStatus | null>(null);
  stats = $state<RenderStats | null>(null);
  series = $state.raw<Float64Array[]>([]);
  trace = $state<{ i: number; iter: number; entries: TraceEntry[] } | null>(null);
  stop = $state<Stop | null>(null);
  stops = $state<Stops>({ on_diverge: true });

  selected = $state(0);
  running = $state(false);
  speed = $state(60);
  view = $state<ViewMode>("surface");
  theme = $state<"light" | "dark">("light");
  camMode = $state<"orbit" | "free">("orbit");
  panel = $state<"" | "problems" | "stops" | "info">("");
  showCode = $state(false);
  bigChart = $state(false);
  copied = $state(false);


  /* ---- plumbing ---- */

  engine(cmd: Parameters<Session["engine"]>[0]) {
    this.session?.engine(cmd);
  }
  render(cmd: Parameters<Session["render"]>[0]) {
    this.session?.render(cmd);
  }

  onEngine = (m: EngineMsg) => {
    switch (m.type) {
      case "ready":
        this.problems = m.problems;
        this.optimizers = m.optimizers;
        break;
      case "problem":
        this.info = m.info;
        this.render({ type: "problem", info: m.info });
        break;
      case "status":
        this.status = m.status;
        this.running = m.status.running;
        if (this.selected >= m.status.runs.length) this.selected = Math.max(0, m.status.runs.length - 1);
        break;
      case "metrics":
        this.series = m.series;
        break;
      case "stop":
        this.stop = m.stop;
        this.running = false;
        if (m.stop.kind === "health" || m.stop.kind === "line") this.select(m.stop.run);
        if (m.stop.kind === "line") this.showCode = true;
        break;
      case "trace":
        this.trace = { i: m.i, iter: m.iter, entries: m.entries };
        break;
      case "error":
        this.error = m.message;
        break;
    }
  };

  onRender = (m: RenderMsg) => {
    if (m.type === "stats") this.stats = m.stats;
    else if (m.type === "error") this.error = m.message;
    else if (m.type === "picked") {
      if (m.uv) this.engine({ type: "startUv", u: m.uv[0], v: m.uv[1] });
      else if (m.ray) this.engine({ type: "pickStart", o: m.ray.o, d: m.ray.d, hs: m.ray.hs });
    }
  };

  /** After boot: restore a shared session from the URL, or the default one. */
  start() {
    let shared = DEFAULT;
    try {
      const h = location.hash.slice(1);
      if (h) shared = JSON.parse(decodeURIComponent(escape(atob(h.replace(/-/g, "+").replace(/_/g, "/")))));
    } catch {
      shared = DEFAULT;
    }
    this.engine({ type: "speed", speed: this.speed });
    this.engine({ type: "stops", stops: $state.snapshot(this.stops) });
    this.engine({ type: "problem", id: shared.p, coeffs: shared.c ?? {}, keepStart: false });
    for (const r of shared.r.slice(0, 8)) this.engine({ type: "addRun", opt: r.o, params: r.p ?? {} });
    if (shared.s?.length === 2) this.engine({ type: "start", x: shared.s[0], y: shared.s[1] });
    if (shared.v) this.setView(shared.v);
    this.play();
  }

  /* ---- run control ---- */

  play() {
    this.stop = null;
    this.running = true;
    this.engine({ type: "play" });
  }
  pause() {
    this.running = false;
    this.engine({ type: "pause" });
  }
  toggle() {
    if (this.running) this.pause();
    else this.play();
  }
  step(n = 1) {
    this.stop = null;
    this.running = false;
    this.engine({ type: "step", n });
  }
  restart() {
    this.stop = null;
    this.engine({ type: "restart" });
  }
  seek(k: number) {
    this.running = false;
    this.engine({ type: "seek", k });
  }
  setSpeed(v: number) {
    this.speed = v;
    this.engine({ type: "speed", speed: v });
  }
  setStops(patch: Partial<Stops>) {
    this.stops = { ...this.stops, ...patch };
    for (const k of Object.keys(this.stops) as (keyof Stops)[]) if (this.stops[k] === undefined) delete this.stops[k];
    this.engine({ type: "stops", stops: $state.snapshot(this.stops) });
  }
  toggleLineStop(line: number) {
    this.setStops({ line: this.stops.line === line ? undefined : line });
  }

  /* ---- problem and runs ---- */

  pickProblem(id: string) {
    this.engine({ type: "problem", id, coeffs: {}, keepStart: false });
    this.panel = "";
  }
  setCoeff(key: string, value: number) {
    if (!this.info) return;
    const coeffs = { ...$state.snapshot(this.info.coeffs), [key]: value };
    this.engine({ type: "problem", id: this.info.manifest.id, coeffs, keepStart: true });
  }
  resetCoeffs() {
    if (this.info) this.engine({ type: "problem", id: this.info.manifest.id, coeffs: {}, keepStart: true });
  }
  addRun(opt: string) {
    this.engine({ type: "addRun", opt });
    this.select(this.status?.runs.length ?? 0);
  }
  removeRun(i: number) {
    this.engine({ type: "removeRun", i });
    if (this.selected >= i && this.selected > 0) this.select(this.selected - 1);
  }
  setOptimizer(i: number, opt: string) {
    this.engine({ type: "setOptimizer", i, opt });
  }
  setParam(key: string, value: number) {
    this.engine({ type: "param", i: this.selected, key, value });
  }
  select(i: number) {
    this.selected = i;
    this.engine({ type: "select", i });
    this.render({ type: "select", i });
  }

  /* ---- view ---- */

  setView(v: ViewMode) {
    this.view = v;
    if (v === "contour" && this.camMode === "free") this.setCam("orbit");
    this.render({ type: "view", mode: v });
  }
  setCam(mode: "orbit" | "free") {
    this.camMode = mode;
    this.render({ type: "camera", mode });
  }
  setTheme(t: "light" | "dark") {
    this.theme = t;
    document.documentElement.dataset.theme = t;
    this.render({ type: "theme", theme: THEMES[t] });
  }

  /** Applies the diagnosis' suggestion to the stopped run and restarts from the start. */
  applySuggestion(run: number, key: string, value: number) {
    this.select(run);
    this.engine({ type: "restart" });
    this.engine({ type: "param", i: run, key, value });
    this.stop = null;
    this.play();
  }

  /* ---- sharing ---- */

  shareLink(): string {
    const st = this.status;
    const info = this.info;
    if (!st || !info) return location.href;
    const s: Shared = {
      p: info.manifest.id,
      c: Object.keys(info.coeffs).length ? $state.snapshot(info.coeffs) : undefined,
      s: info.start,
      r: st.runs.map((r) => ({ o: r.opt, p: $state.snapshot(r.params) })),
      v: this.view === "surface" ? undefined : this.view,
    };
    const h = btoa(unescape(encodeURIComponent(JSON.stringify(s)))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
    return `${location.origin}${location.pathname}${location.search}#${h}`;
  }
  async share() {
    const link = this.shareLink();
    history.replaceState(null, "", link);
    try {
      await navigator.clipboard.writeText(link);
    } catch {
      /* the address bar has it */
    }
    this.copied = true;
    setTimeout(() => (this.copied = false), 1600);
  }
}

export const app = new AppState();

/** The selected run's status and manifest, for components. */
export function selectedRun() {
  const r = app.status?.runs[app.selected];
  const m = r ? app.optimizers.find((o) => o.id === r.opt) : undefined;
  return { run: r, manifest: m };
}
