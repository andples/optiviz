/** WebGL2 back-end: the same scene through GLSL ES 3.00, for browsers without WebGPU. Same uniform layout
 *  (std140 blocks), same instanced trail and sprites. */

import { LAYER_CAP, type Backend, type Layer } from "./backend";
import { DRAW_FLOATS, SCENE_FLOATS, glslPoints, glslSurface, glslTrail } from "./shaders";

const LAYERS: Layer[] = ["trail", "points", "marker", "minima"];

export function createWebGL2(canvas: OffscreenCanvas | HTMLCanvasElement): Backend | null {
  const gl = canvas.getContext("webgl2", {
    antialias: true,
    alpha: false,
    depth: true,
    powerPreference: "high-performance",
  }) as WebGL2RenderingContext | null;
  if (!gl) return null;
  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  const name = dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : String(gl.getParameter(gl.RENDERER));
  return new WebGL2Backend(gl, canvas, name);
}

class WebGL2Backend implements Backend {
  readonly kind = "webgl2";
  private sceneUbo: WebGLBuffer;
  private drawUbos = {} as Record<Layer, WebGLBuffer>;
  private dataBufs = {} as Record<Layer, WebGLBuffer>;
  private counts: Record<Layer, number> = { trail: 0, points: 0, marker: 0, minima: 0 };
  private progs: { surface?: WebGLProgram; trail?: WebGLProgram; points?: WebGLProgram } = {};
  private surfaceVao: WebGLVertexArrayObject;
  private trailVao: WebGLVertexArrayObject;
  private pointVaos = {} as Record<Layer, WebGLVertexArrayObject>;
  private indexBuf: WebGLBuffer;
  private indexCount = 0;

  constructor(
    private gl: WebGL2RenderingContext,
    private canvas: OffscreenCanvas | HTMLCanvasElement,
    readonly adapter: string,
  ) {
    this.sceneUbo = this.ubo(SCENE_FLOATS);
    for (const l of LAYERS) {
      this.drawUbos[l] = this.ubo(DRAW_FLOATS);
      const b = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, LAYER_CAP[l] * 8 + 8, gl.DYNAMIC_DRAW);
      this.dataBufs[l] = b;
    }
    this.indexBuf = gl.createBuffer()!;
    this.surfaceVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.surfaceVao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuf);

    this.trailVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.trailVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dataBufs.trail);
    for (const [loc, off] of [[0, 0], [1, 8]]) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 8, off);
      gl.vertexAttribDivisor(loc, 1);
    }
    for (const l of ["points", "marker", "minima"] as const) {
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.dataBufs[l]);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
      gl.vertexAttribDivisor(0, 1);
      this.pointVaos[l] = vao;
    }
    gl.bindVertexArray(null);
    gl.enable(gl.DEPTH_TEST);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  private ubo(floats: number) {
    const gl = this.gl;
    const b = gl.createBuffer()!;
    gl.bindBuffer(gl.UNIFORM_BUFFER, b);
    gl.bufferData(gl.UNIFORM_BUFFER, floats * 4, gl.DYNAMIC_DRAW);
    return b;
  }

  private program(src: { vs: string; fs: string }) {
    const gl = this.gl;
    const sh = (type: number, code: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, code);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`GLSL: ${gl.getShaderInfoLog(s)}`);
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, sh(gl.VERTEX_SHADER, src.vs));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, src.fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`GLSL link: ${gl.getProgramInfoLog(p)}`);
    const si = gl.getUniformBlockIndex(p, "Scene");
    if (si !== gl.INVALID_INDEX) gl.uniformBlockBinding(p, si, 0);
    const di = gl.getUniformBlockIndex(p, "Draw");
    if (di !== gl.INVALID_INDEX) gl.uniformBlockBinding(p, di, 1);
    return p;
  }

  setProblem(expr: string, n: number) {
    const gl = this.gl;
    for (const p of Object.values(this.progs)) if (p) gl.deleteProgram(p);
    this.progs = {
      surface: this.program(glslSurface(expr, n)),
      trail: this.program(glslTrail(expr)),
      points: this.program(glslPoints(expr)),
    };
    const idx = new Uint32Array((n - 1) * (n - 1) * 6);
    let k = 0;
    for (let j = 0; j < n - 1; j++)
      for (let i = 0; i < n - 1; i++) {
        const a = j * n + i, b = a + 1, c = a + n, e = c + 1;
        idx.set([a, b, e, a, e, c], k);
        k += 6;
      }
    gl.bindVertexArray(this.surfaceVao);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    this.indexCount = idx.length;
  }

  resize(w: number, h: number) {
    this.canvas.width = w;
    this.canvas.height = h;
    this.gl.viewport(0, 0, w, h);
  }

  upload(layer: Layer, data: Float32Array, count: number) {
    const n = Math.min(count, LAYER_CAP[layer]);
    this.counts[layer] = n;
    if (!n) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dataBufs[layer]);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data, 0, n * 2);
  }

  draw(scene: Float32Array, draws: Record<Layer, Float32Array>, bg: [number, number, number]) {
    const gl = this.gl;
    const { surface, trail, points } = this.progs;
    if (!surface || !trail || !points) return;
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.sceneUbo);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, scene, 0, SCENE_FLOATS);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this.sceneUbo);
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const useDraw = (l: Layer) => {
      gl.bindBuffer(gl.UNIFORM_BUFFER, this.drawUbos[l]);
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, draws[l], 0, DRAW_FLOATS);
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 1, this.drawUbos[l]);
    };

    gl.disable(gl.BLEND);
    gl.depthMask(true);
    gl.useProgram(surface);
    useDraw("trail");
    gl.bindVertexArray(this.surfaceVao);
    gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_INT, 0);

    gl.enable(gl.BLEND);
    gl.depthMask(false);
    if (this.counts.trail > 1) {
      gl.useProgram(trail);
      useDraw("trail");
      gl.bindVertexArray(this.trailVao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.counts.trail - 1);
    }
    gl.useProgram(points);
    for (const l of ["points", "minima", "marker"] as const) {
      if (!this.counts[l]) continue;
      useDraw(l);
      gl.bindVertexArray(this.pointVaos[l]);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.counts[l]);
    }
    gl.bindVertexArray(null);
    gl.depthMask(true);
  }
}
