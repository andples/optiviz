// Shared by every WGSL shader: uniforms, the problem's f (spliced in for {{F}}), height mapping, colour map.
// The same layout and functions exist in common.glsl for the WebGL2 back-end.
const PI: f32 = 3.14159265358979;
struct Scene {
  viewProj: mat4x4f, eye: vec4f, bounds: vec4f, zinfo: vec4f,
  light: vec4f, viewport: vec4f, bg: vec4f, style: vec4f,
};
struct Draw { color: vec4f, misc: vec4f };
@group(0) @binding(0) var<uniform> S: Scene;
@group(0) @binding(1) var<uniform> D: Draw;

fn F(x: f32, y: f32) -> f32 { return {{F}}; }
fn height01(uv: vec2f) -> f32 {
  let p = mix(S.bounds.xz, S.bounds.yw, uv * 0.5 + 0.5);
  var f = F(p.x, p.y);
  if (S.zinfo.w > 0.5) { f = log(1.0 + max(f, 0.0)); }
  return (f - S.zinfo.x) / max(S.zinfo.y - S.zinfo.x, 1e-20);
}
fn worldPos(uv: vec2f) -> vec3f { return vec3f(uv.x, (height01(uv) - 0.5) * S.zinfo.z, -uv.y); }
fn viridis(t: f32) -> vec3f {
  let c0 = vec3f(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  let c1 = vec3f(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  let c2 = vec3f(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  let c3 = vec3f(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  let c4 = vec3f(6.228269936347081, 14.17993336680509, 56.69055260068105);
  let c5 = vec3f(4.776384997670288, -13.74514537774601, -65.35303263337234);
  let c6 = vec3f(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
