// Test kernel: evaluates the problem's f (from common.wgsl) at each point, for comparison with the Rust eval.
@group(1) @binding(0) var<storage, read> pts: array<vec2f>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&pts)) { return; }
  out[i] = F(pts[i].x, pts[i].y);
}
