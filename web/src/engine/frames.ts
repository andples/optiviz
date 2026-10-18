/**
 * Frames: how the engine worker hands state to the render worker without either waiting for the other.
 *
 * A frame is a Float64 header plus a Float32 data area:
 * - header: H.* scalars, then from H.index the engine's frame index (crates/wasm `build_frame`: run count,
 *   minima offset/count, then RUN_STRIDE values per run: trail offset/count, shape offset/count, marker u v,
 *   start u v, health, halted)
 * - data: the u, v pairs the index points into (trails, shapes, minima), then the stress-test points
 *
 * Two transports implement the same writer/reader interface:
 * - TripleBuffer (preferred): three frame slots in one SharedArrayBuffer. The writer always has a slot of its
 *   own to fill, the reader always has one to read, and the third is the hand-off. Publishing and picking up
 *   are one Atomics.exchange each: lock-free, and the reader always sees the newest complete frame. Needs a
 *   cross-origin isolated page.
 * - PostFrames (fallback): the same frames as transferable ArrayBuffers over a MessagePort, recycled through a
 *   small pool. Used when SharedArrayBuffer is unavailable.
 */

export const H = {
  seq: 0,
  /** performance.now() when the writer published (both workers share the page's time origin) */
  publishedAt: 1,
  iter: 2,
  head: 3,
  pointsCount: 4,
  indexLen: 5,
  /** where the engine's frame index starts */
  index: 16,
} as const;
export const HEADER_F64 = 128;
export const INDEX_HEAD = 8;
export const RUN_STRIDE = 12;
export const R = { trailOff: 0, trailN: 1, shapeOff: 2, shapeN: 3, mu: 4, mv: 5, su: 6, sv: 7, health: 8, halted: 9, du: 10, dv: 11 } as const;

export interface Layout {
  dataCap: number; // points in the data area (trails, shapes, minima)
  pointsCap: number; // stress-test points
}

export function slotBytes(l: Layout) {
  return HEADER_F64 * 8 + 4 * 2 * (l.dataCap + l.pointsCap);
}

export interface Frame {
  header: Float64Array;
  data: Float32Array; // 2 * dataCap
  points: Float32Array; // 2 * pointsCap
}

function frameOn(buf: ArrayBufferLike, offset: number, l: Layout): Frame {
  return {
    header: new Float64Array(buf, offset, HEADER_F64),
    data: new Float32Array(buf, offset + HEADER_F64 * 8, 2 * l.dataCap),
    points: new Float32Array(buf, offset + HEADER_F64 * 8 + 8 * l.dataCap, 2 * l.pointsCap),
  };
}

export interface FrameWriter {
  /** The frame to fill; valid until publish() */
  frame(): Frame;
  publish(): void;
}

export interface FrameReader {
  /** The newest published frame (or the last one again, with fresh = false). Null before the first publish. */
  latest(): { frame: Frame; fresh: boolean } | null;
}

/* ---------- triple buffer over SharedArrayBuffer ---------- */

const FRESH = 4; // bit set in the shared state when the hand-off slot holds an unread frame
const IDX = 3;
const CTRL_BYTES = 16; // Int32 state (+ padding so the slots stay 8-byte aligned)

export interface TripleInit {
  kind: "triple";
  sab: SharedArrayBuffer;
  layout: Layout;
}

export function createTriple(layout: Layout): TripleInit {
  const sab = new SharedArrayBuffer(CTRL_BYTES + 3 * slotBytes(layout));
  new Int32Array(sab, 0, 1)[0] = 1; // hand-off slot is 1; writer starts on 0, reader on 2
  return { kind: "triple", sab, layout };
}

export class TripleWriter implements FrameWriter {
  private ctrl: Int32Array;
  private slots: Frame[];
  private w = 0;
  private seq = 0;
  constructor(init: TripleInit) {
    this.ctrl = new Int32Array(init.sab, 0, 1);
    const sb = slotBytes(init.layout);
    this.slots = [0, 1, 2].map((i) => frameOn(init.sab, CTRL_BYTES + i * sb, init.layout));
  }
  frame() {
    return this.slots[this.w];
  }
  publish() {
    const f = this.slots[this.w];
    f.header[H.seq] = ++this.seq;
    f.header[H.publishedAt] = performance.now();
    this.w = Atomics.exchange(this.ctrl, 0, this.w | FRESH) & IDX;
  }
}

export class TripleReader implements FrameReader {
  private ctrl: Int32Array;
  private slots: Frame[];
  private r = 2;
  private any = false;
  constructor(init: TripleInit) {
    this.ctrl = new Int32Array(init.sab, 0, 1);
    const sb = slotBytes(init.layout);
    this.slots = [0, 1, 2].map((i) => frameOn(init.sab, CTRL_BYTES + i * sb, init.layout));
  }
  latest() {
    if (Atomics.load(this.ctrl, 0) & FRESH) {
      this.r = Atomics.exchange(this.ctrl, 0, this.r) & IDX;
      this.any = true;
      return { frame: this.slots[this.r], fresh: true };
    }
    return this.any ? { frame: this.slots[this.r], fresh: false } : null;
  }
}

/* ---------- fallback: transferable buffers over a MessagePort ---------- */

export interface PostInit {
  kind: "post";
  port: MessagePort;
  layout: Layout;
}

export class PostWriter implements FrameWriter {
  private pool: ArrayBuffer[] = [];
  private cur: { buf: ArrayBuffer; frame: Frame };
  private seq = 0;
  constructor(private init: PostInit) {
    init.port.onmessage = (e) => this.pool.push(e.data as ArrayBuffer); // the reader returns used buffers
    this.cur = this.fresh();
  }
  private fresh() {
    const buf = this.pool.pop() ?? new ArrayBuffer(slotBytes(this.init.layout));
    return { buf, frame: frameOn(buf, 0, this.init.layout) };
  }
  frame() {
    return this.cur.frame;
  }
  publish() {
    const { buf, frame } = this.cur;
    frame.header[H.seq] = ++this.seq;
    frame.header[H.publishedAt] = performance.now();
    this.init.port.postMessage(buf, [buf]);
    this.cur = this.fresh();
  }
}

export class PostReader implements FrameReader {
  private last: { buf: ArrayBuffer; frame: Frame } | null = null;
  private fresh = false;
  constructor(init: PostInit) {
    init.port.onmessage = (e) => {
      const buf = e.data as ArrayBuffer;
      if (this.last) init.port.postMessage(this.last.buf, [this.last.buf]);
      this.last = { buf, frame: frameOn(buf, 0, init.layout) };
      this.fresh = true;
    };
  }
  latest() {
    if (!this.last) return null;
    const fresh = this.fresh;
    this.fresh = false;
    return { frame: this.last.frame, fresh };
  }
}

export type TransportInit = TripleInit | PostInit;

export function writerFor(t: TransportInit): FrameWriter {
  return t.kind === "triple" ? new TripleWriter(t) : new PostWriter(t);
}
export function readerFor(t: TransportInit): FrameReader {
  return t.kind === "triple" ? new TripleReader(t) : new PostReader(t);
}
