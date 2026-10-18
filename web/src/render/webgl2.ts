/** WebGL2 back-end: the same scene through GLSL ES 3.00, for browsers without WebGPU. Same uniform layout
 *  (std140 blocks), same draw list; point buffers are bound at each draw's offset via vertexAttribPointer. */

import { BUFFER_CAP, type Backend, type BufferId, type DrawItem } from "./backend";
import { DRAW_FLOATS, SCENE_FLOATS, glslPoints, glslSurface, glslTrail } from "./shaders";
import { gridIndices } from "./webgpu";

const BUFFERS: BufferId[] = ["data", "marks", "stress"];

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
  private drawUbo: WebGLBuffer;
  private data = {} as Record<BufferId, WebGLBuffer>;
  private progs: { surface?: WebGLProgram; trail?: WebGLProgram; points?: WebGLProgram } = {};
  private surfaceVao: WebGLVertexArrayObject;
  private vao: WebGLVertexArrayObject;
  private indexBuf: WebGLBuffer;
  private indexCount = 0;

  constructor(
    private gl: WebGL2RenderingContext,
    private canvas: OffscreenCanvas | HTMLCanvasElement,
    readonly adapter: string,
  ) {
    this.sceneUbo = this.ubo(SCENE_FLOATS);
    this.drawUbo = this.ubo(DRAW_FLOATS);
    for (const b of BUFFERS) {
      const buf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, BUFFER_CAP[b] * 8 + 16, gl.DYNAMIC_DRAW);
      this.data[b] = buf;
    }
    this.indexBuf = gl.createBuffer()!;
    this.surfaceVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.surfaceVao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuf);
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    for (const loc of [0, 1]) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribDivisor(loc, 1);
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
    const idx = gridIndices(n);
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

  upload(buffer: BufferId, data: Float32Array, count: number) {
    const n = Math.min(count, BUFFER_CAP[buffer]);
    if (!n) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.data[buffer]);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data, 0, n * 2);
  }

  draw(scene: Float32Array, items: DrawItem[], bg: [number, number, number]) {
    const gl = this.gl;
    const { surface, trail, points } = this.progs;
    if (!surface || !trail || !points) return;
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.sceneUbo);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, scene, 0, SCENE_FLOATS);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this.sceneUbo);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 1, this.drawUbo);
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.disable(gl.BLEND);
    gl.depthMask(true);
    gl.useProgram(surface);
    gl.bindVertexArray(this.surfaceVao);
    gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_INT, 0);

    gl.enable(gl.BLEND);
    gl.depthMask(false);
    gl.bindVertexArray(this.vao);
    for (const it of items) {
      const isTrail = it.kind === "trail";
      if (it.count < (isTrail ? 2 : 1)) continue;
      gl.useProgram(isTrail ? trail : points);
      gl.bindBuffer(gl.UNIFORM_BUFFER, this.drawUbo);
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, it.uniforms, 0, DRAW_FLOATS);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.data[it.buffer]);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, it.first * 8);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 8, it.first * 8 + (isTrail ? 8 : 0));
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, isTrail ? it.count - 1 : it.count);
    }
    gl.bindVertexArray(null);
    gl.depthMask(true);
  }
}
