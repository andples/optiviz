<script lang="ts">
  import { app } from "./state.svelte";
  import { count, sci } from "./format";
  const tier = $derived(app.tier);
  const caps = $derived(app.caps);
</script>

{#if tier && caps}
  <aside class="panel info">
    <div><b>Rendering</b> {tier.backend === "webgpu" ? "WebGPU" : "WebGL2"} in the {tier.render === "worker" ? "render worker" : "main thread"}</div>
    <div class="dim">{tier.adapter}</div>
    <div><b>Engine</b> WebAssembly {caps.wasmSimd ? "SIMD" : ""}, {tier.threads ? `${tier.threadCount} threads` : "single thread"}</div>
    <div><b>Frames</b> {tier.transport === "shared" ? "shared memory (triple buffer)" : "message passing"}</div>
    {#if app.stats}
      <div><b>Display</b> {app.stats.fps.toFixed(0)} fps · {app.stats.frameMs.toFixed(2)} ms/frame · frame age {app.stats.latencyMs.toFixed(1)} ms</div>
    {/if}
    {#if app.status}
      <div><b>Engine rate</b> {sci(app.status.stepsPerSec, 2)} iterations/s · {app.status.framesPerSec.toFixed(0)} frames/s · publish {app.status.publishMs.toFixed(2)} ms</div>
      {#if app.status.stressPoints}<div><b>Stress</b> {count(app.status.stressPoints)} points per frame</div>{/if}
    {/if}
    {#if !caps.webgpu}<div class="note">WebGPU isn't available in this browser; drawing with WebGL2.</div>{/if}
    {#if !caps.isolated}<div class="note">This page isn't cross-origin isolated, so the engine runs on one thread.</div>{/if}
    <div class="note">Keys: space play/pause · . one step · r restart · ←↑→↓ free camera · Esc back</div>
  </aside>
{/if}

<style>
  .info { position: absolute; top: 60px; right: 310px; width: 340px; padding: 10px 12px; display: flex; flex-direction: column; gap: 4px; font-size: 12px; }
  .dim { color: var(--dim); margin-top: -2px; word-break: break-word; }
  .note { color: var(--dim); font-style: italic; }
  @media (max-width: 720px) { .info { right: 16px; left: 16px; width: auto; } }
</style>
