/** WebGPU back-end: 4× MSAA, one uniform buffer for the scene and one per layer, instanced trail segments and
 *  point sprites. Pipelines are rebuilt when the problem changes (they embed f). */

import { LAYER_CAP, type Backend, type Layer } from "./backend";
import { DRAW_FLOATS, SCENE_FLOATS, wgslPoints, wgslSurface, wgslTrail } from "./shaders";

const LAYERS: Layer[] = ["trail", "points", "marker", "minima"];
const SAMPLES = 4;

export async function createWebGPU(canvas: OffscreenCanvas | HTMLCanvasElement, onError: (m: string) => void): Promise<Backend | null> {
  if (!("gpu" in navigator) || !navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) => onError(`WebGPU: ${(e as GPUUncapturedErrorEvent).error.message}`));
  device.lost.then((i) => onError(`WebGPU device lost: ${i.message}`));
  const ctx = canvas.getContext("webgpu") as GPUCanvasContext | null;
  if (!ctx) return null;
  const info = adapter.info;
  const name = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ") || "unknown";
  return new WebGPUBackend(device, ctx, canvas, name);
}

class WebGPUBackend implements Backend {
  readonly kind = "webgpu";
  private format: GPUTextureFormat;
  private sceneBuf: GPUBuffer;
  private drawBufs = {} as Record<Layer, GPUBuffer>;
  private dataBufs = {} as Record<Layer, GPUBuffer>;
  private counts: Record<Layer, number> = { trail: 0, points: 0, marker: 0, minima: 0 };
  private bgl: GPUBindGroupLayout;
  private groups = {} as Record<Layer | "surface", GPUBindGroup>;
  private surface?: GPURenderPipeline;
  private trail?: GPURenderPipeline;
  private points?: GPURenderPipeline;
  private index?: GPUBuffer;
  private indexCount = 0;
  private msaa?: GPUTexture;
  private depth?: GPUTexture;

  constructor(
    private device: GPUDevice,
    private ctx: GPUCanvasContext,
    private canvas: OffscreenCanvas | HTMLCanvasElement,
    readonly adapter: string,
  ) {
    this.format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format: this.format, alphaMode: "opaque" });
    this.sceneBuf = device.createBuffer({ size: SCENE_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bgl = device.createBindGroupLayout({
      entries: [0, 1].map((binding) => ({
        binding,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" as const },
      })),
    });
    for (const l of LAYERS) {
      this.drawBufs[l] = device.createBuffer({ size: DRAW_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.dataBufs[l] = device.createBuffer({ size: LAYER_CAP[l] * 8 + 8, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      this.groups[l] = this.group(this.drawBufs[l]);
    }
    this.groups.surface = this.group(this.drawBufs.trail);
  }

  private group(draw: GPUBuffer) {
    return this.device.createBindGroup({
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.sceneBuf } },
        { binding: 1, resource: { buffer: draw } },
      ],
    });
  }

  setProblem(expr: string, n: number) {
    const d = this.device;
    const layout = d.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    const depthStencil: GPUDepthStencilState = { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" };
    const multisample = { count: SAMPLES };
    const blend: GPUBlendState = {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    };
    const mod = (code: string) => d.createShaderModule({ code });

    const sm = mod(wgslSurface(expr, n));
    this.surface = d.createRenderPipeline({
      layout,
      vertex: { module: sm, entryPoint: "vs" },
      fragment: { module: sm, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil,
      multisample,
    });
    const point2 = (location: number): GPUVertexBufferLayout => ({
      arrayStride: 8,
      stepMode: "instance",
      attributes: [{ shaderLocation: location, offset: 0, format: "float32x2" }],
    });
    const tm = mod(wgslTrail(expr));
    this.trail = d.createRenderPipeline({
      layout,
      vertex: { module: tm, entryPoint: "vs", buffers: [point2(0), point2(1)] },
      fragment: { module: tm, entryPoint: "fs", targets: [{ format: this.format, blend }] },
      primitive: { topology: "triangle-list" },
      depthStencil: { ...depthStencil, depthWriteEnabled: false },
      multisample,
    });
    const pm = mod(wgslPoints(expr));
    this.points = d.createRenderPipeline({
      layout,
      vertex: { module: pm, entryPoint: "vs", buffers: [point2(0)] },
      fragment: { module: pm, entryPoint: "fs", targets: [{ format: this.format, blend }] },
      primitive: { topology: "triangle-list" },
      depthStencil: { ...depthStencil, depthWriteEnabled: false },
      multisample,
    });

    // grid indices: two triangles per cell
    const idx = new Uint32Array((n - 1) * (n - 1) * 6);
    let k = 0;
    for (let j = 0; j < n - 1; j++)
      for (let i = 0; i < n - 1; i++) {
        const a = j * n + i, b = a + 1, c = a + n, e = c + 1;
        idx.set([a, b, e, a, e, c], k);
        k += 6;
      }
    this.index?.destroy();
    this.index = d.createBuffer({ size: idx.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.index, 0, idx);
    this.indexCount = idx.length;
  }

  resize(w: number, h: number) {
    this.canvas.width = w;
    this.canvas.height = h;
    this.msaa?.destroy();
    this.depth?.destroy();
    const size = { width: w, height: h };
    this.msaa = this.device.createTexture({ size, sampleCount: SAMPLES, format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depth = this.device.createTexture({ size, sampleCount: SAMPLES, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }

  upload(layer: Layer, data: Float32Array, count: number) {
    const n = Math.min(count, LAYER_CAP[layer]);
    this.counts[layer] = n;
    if (n > 0) this.device.queue.writeBuffer(this.dataBufs[layer], 0, data.buffer, data.byteOffset, n * 8);
  }

  draw(scene: Float32Array, draws: Record<Layer, Float32Array>, bg: [number, number, number]) {
    if (!this.surface || !this.trail || !this.points || !this.index || !this.msaa || !this.depth) return;
    const q = this.device.queue;
    q.writeBuffer(this.sceneBuf, 0, scene.buffer, scene.byteOffset, SCENE_FLOATS * 4);
    for (const l of LAYERS) q.writeBuffer(this.drawBufs[l], 0, draws[l].buffer, draws[l].byteOffset, DRAW_FLOATS * 4);

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.msaa.createView(),
        resolveTarget: this.ctx.getCurrentTexture().createView(),
        clearValue: { r: bg[0], g: bg[1], b: bg[2], a: 1 },
        loadOp: "clear",
        storeOp: "discard",
      }],
      depthStencilAttachment: { view: this.depth.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "discard" },
    });
    pass.setPipeline(this.surface);
    pass.setBindGroup(0, this.groups.surface);
    pass.setIndexBuffer(this.index, "uint32");
    pass.drawIndexed(this.indexCount);

    if (this.counts.trail > 1) {
      pass.setPipeline(this.trail);
      pass.setBindGroup(0, this.groups.trail);
      pass.setVertexBuffer(0, this.dataBufs.trail, 0);
      pass.setVertexBuffer(1, this.dataBufs.trail, 8);
      pass.draw(6, this.counts.trail - 1);
    }
    pass.setPipeline(this.points);
    for (const l of ["points", "minima", "marker"] as const) {
      if (!this.counts[l]) continue;
      pass.setBindGroup(0, this.groups[l]);
      pass.setVertexBuffer(0, this.dataBufs[l], 0);
      pass.draw(6, this.counts[l]);
    }
    pass.end();
    q.submit([enc.finish()]);
  }
}
