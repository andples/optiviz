<script module lang="ts">
  export interface Line {
    xs: Float64Array | number[];
    ys: Float64Array | number[];
    color: string;
    bold?: boolean;
  }
</script>

<script lang="ts">
  import { onMount } from "svelte";

  /**
   * A small line chart on a canvas: one line per run (own x values, so a halted run simply ends), log or linear
   * y, a few ticks in scientific notation, and the current iteration as a vertical rule.
   */
  let { lines, log = true, at = null, label = "", height = 110 }: { lines: Line[]; log?: boolean; at?: number | null; label?: string; height?: number } = $props();
  let canvas: HTMLCanvasElement;
  let width = $state(260);
  let wrap: HTMLDivElement;

  const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  const supNum = (n: number) => (n < 0 ? "⁻" : "") + [...String(Math.abs(n))].map((d) => SUP[+d]).join("");

  function draw() {
    if (!canvas) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = width, h = height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const g = canvas.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const css = getComputedStyle(canvas);
    const ink = css.getPropertyValue("--dim") || "#666";
    const rule = css.getPropertyValue("--rule") || "#ddd";
    const tf = (y: number) => (log ? Math.log10(y) : y);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const l of lines)
      for (let i = 0; i < l.xs.length; i++) {
        const y = l.ys[i];
        if (!Number.isFinite(y) || (log && y <= 0)) continue;
        x0 = Math.min(x0, l.xs[i]);
        x1 = Math.max(x1, l.xs[i]);
        y0 = Math.min(y0, tf(y));
        y1 = Math.max(y1, tf(y));
      }
    const pad = { l: 38, r: 6, t: 14, b: 16 };
    g.font = "10px system-ui, sans-serif";
    g.fillStyle = ink;
    if (label) g.fillText(label, pad.l, 10);
    if (!Number.isFinite(x0) || x1 <= x0) {
      x0 = 0;
      x1 = Math.max(1, x1);
    }
    if (!Number.isFinite(y0)) return;
    if (y1 - y0 < 1e-12) {
      y0 -= 0.5;
      y1 += 0.5;
    }
    const X = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * (w - pad.l - pad.r);
    const Y = (y: number) => pad.t + (1 - (y - y0) / (y1 - y0)) * (h - pad.t - pad.b);
    // y ticks: decades on log scales, ~3 round values on linear ones
    g.strokeStyle = rule;
    g.lineWidth = 1;
    g.textAlign = "right";
    g.textBaseline = "middle";
    const ticks: [number, string][] = [];
    if (log) {
      const step = Math.max(1, Math.ceil((Math.floor(y1) - Math.ceil(y0) + 1) / 4));
      for (let e = Math.ceil(y0); e <= Math.floor(y1); e += step) ticks.push([e, `10${supNum(e)}`]);
    } else {
      const span = y1 - y0, mag = 10 ** Math.floor(Math.log10(span / 3)), st = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= 4)!;
      for (let v = Math.ceil(y0 / st) * st; v <= y1; v += st) ticks.push([v, Number(v.toPrecision(3)).toString().replace("-", "−")]);
    }
    for (const [v, t] of ticks) {
      g.beginPath();
      g.moveTo(pad.l, Y(v) + 0.5);
      g.lineTo(w - pad.r, Y(v) + 0.5);
      g.stroke();
      g.fillText(t, pad.l - 4, Y(v));
    }
    g.textAlign = "left";
    g.textBaseline = "alphabetic";
    g.fillText("0", pad.l, h - 4);
    g.textAlign = "right";
    g.fillText(Math.round(x1).toLocaleString(), w - pad.r, h - 4);
    // lines
    for (const l of lines) {
      g.strokeStyle = l.color;
      g.lineWidth = l.bold ? 1.8 : 1.2;
      g.beginPath();
      let pen = false;
      for (let i = 0; i < l.xs.length; i++) {
        const y = l.ys[i];
        if (!Number.isFinite(y) || (log && y <= 0)) {
          pen = false;
          continue;
        }
        const px = X(l.xs[i]), py = Y(tf(y));
        if (pen) g.lineTo(px, py);
        else g.moveTo(px, py);
        pen = true;
      }
      g.stroke();
    }
    if (at !== null && at >= x0 && at <= x1) {
      g.strokeStyle = ink;
      g.setLineDash([2, 3]);
      g.beginPath();
      g.moveTo(X(at) + 0.5, pad.t);
      g.lineTo(X(at) + 0.5, h - pad.b);
      g.stroke();
      g.setLineDash([]);
    }
  }

  $effect(() => {
    lines;
    log;
    at;
    width;
    height;
    draw();
  });
  onMount(() => {
    const ro = new ResizeObserver(() => (width = wrap.clientWidth || 260));
    ro.observe(wrap);
    return () => ro.disconnect();
  });
</script>

<div class="chart" bind:this={wrap} style:height="{height}px"><canvas bind:this={canvas} style:width="{width}px" style:height="{height}px"></canvas></div>

<style>
  .chart { width: 100%; }
  canvas { display: block; }
</style>
