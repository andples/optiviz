/// <reference lib="webworker" />
/** The render worker: a thin shell around RenderHost with an OffscreenCanvas. */

import type { RenderCmd, RenderMsg } from "../engine/protocol";
import { RenderHost } from "./host";

const host = new RenderHost((m: RenderMsg) => (self as unknown as Worker).postMessage(m), "worker");
self.onmessage = (e: MessageEvent<RenderCmd>) => void host.handle(e.data);
