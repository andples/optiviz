/** WebGPU back-end: 4× MSAA, the surface plus a list of instanced trail and point-sprite draws. Each draw has its
 *  own small uniform buffer (a fixed pool); point buffers are bound at the draw's offset, so instance indices
 *  start at 0 for every item. Pipelines are rebuilt when the problem changes (they embed f). */

import { BUFFER_CAP, MAX_ITEMS, type Backend, type BufferId, type DrawItem } from "./backend";
import { DRAW_FLOATS, SCENE_FLOATS, wgslPoints, wgslSurface, wgslTrail } from "./shaders";

const SAMPLES = 4;
const BUFFERS: BufferId[] = ["data", "marks", "stress"];

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
  private slots: { buf: GPUBuffer; group: GPUBindGroup }[] = [];
  private data = {} as Record<BufferId, GPUBuffer>;
  private bgl: GPUBindGroupLayout;
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
    for (let i = 0; i < MAX_ITEMS + 1; i++) {
      const buf = device.createBuffer({ size: DRAW_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const group = device.createBindGroup({
        layout: this.bgl,
        entries: [
          { binding: 0, resource: { buffer: this.sceneBuf } },
          { binding: 1, resource: { buffer: buf } },
        ],
      });
      this.slots.push({ buf, group });
    }
    for (const b of BUFFERS) {
      this.data[b] = device.createBuffer({ size: BUFFER_CAP[b] * 8 + 16, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
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
    const idx = gridIndices(n);
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

  upload(buffer: BufferId, data: Float32Array, count: number) {
    const n = Math.min(count, BUFFER_CAP[buffer]);
    if (n > 0) this.device.queue.writeBuffer(this.data[buffer], 0, data.buffer, data.byteOffset, n * 8);
  }

  draw(scene: Float32Array, items: DrawItem[], bg: [number, number, number]) {
    if (!this.surface || !this.trail || !this.points || !this.index || !this.msaa || !this.depth) return;
    const q = this.device.queue;
    q.writeBuffer(this.sceneBuf, 0, scene.buffer, scene.byteOffset, SCENE_FLOATS * 4);
    const n = Math.min(items.length, MAX_ITEMS);
    for (let i = 0; i < n; i++) q.writeBuffer(this.slots[i + 1].buf, 0, items[i].uniforms.buffer, items[i].uniforms.byteOffset, DRAW_FLOATS * 4);

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
    pass.setBindGroup(0, this.slots[0].group);
    pass.setIndexBuffer(this.index, "uint32");
    pass.drawIndexed(this.indexCount);
    for (let i = 0; i < n; i++) {
      const it = items[i];
      const buf = this.data[it.buffer];
      if (it.kind === "trail") {
        if (it.count < 2) continue;
        pass.setPipeline(this.trail);
        pass.setBindGroup(0, this.slots[i + 1].group);
        pass.setVertexBuffer(0, buf, it.first * 8);
        pass.setVertexBuffer(1, buf, it.first * 8 + 8);
        pass.draw(6, it.count - 1);
      } else {
        if (it.count < 1) continue;
        pass.setPipeline(this.points);
        pass.setBindGroup(0, this.slots[i + 1].group);
        pass.setVertexBuffer(0, buf, it.first * 8);
        pass.draw(6, it.count);
      }
    }
    pass.end();
    q.submit([enc.finish()]);
  }
}

/** Two triangles per grid cell. */
export function gridIndices(n: number) {
  const idx = new Uint32Array((n - 1) * (n - 1) * 6);
  let k = 0;
  for (let j = 0; j < n - 1; j++)
    for (let i = 0; i < n - 1; i++) {
      const a = j * n + i, b = a + 1, c = a + n, e = c + 1;
      idx.set([a, b, e, a, e, c], k);
      k += 6;
    }
  return idx;
}
