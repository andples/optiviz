/**
 * Shaders for both back-ends, WGSL (WebGPU) and GLSL ES 3.00 (WebGL2), kept side by side so they stay in
 * step. The problem's f(x, y) is spliced in from its manifest (`shader_expr`, written to be valid in both
 * languages; crates/gpu-test checks it against the Rust reference).
 *
 * The surface mesh only gives the silhouette: height, normals, colour and contour lines are all evaluated per
 * pixel from f itself, so the shading stays smooth at any zoom and any mesh resolution.
 *
 * Uniforms (identical layout in WGSL and std140):
 *   Scene { viewProj: mat4; eye; bounds (x0, x1, y0, y1); zinfo (zmin, zmax, height scale, log flag);
 *           light (dir xyz, ambient); viewport (w, h, dpr, time); bg (rgb, dark); style (levels, contour
 *           strength, fog, grid) }  = 44 floats
 *   Draw  { color; misc (size px, lift, mode, count) } = 8 floats
 */

export const SCENE_FLOATS = 44;
export const DRAW_FLOATS = 8;

/* ---------------- WGSL ---------------- */

const wgslCommon = (expr: string) => /* wgsl */ `
const PI: f32 = 3.14159265358979;
struct Scene {
  viewProj: mat4x4f, eye: vec4f, bounds: vec4f, zinfo: vec4f,
  light: vec4f, viewport: vec4f, bg: vec4f, style: vec4f,
};
struct Draw { color: vec4f, misc: vec4f };
@group(0) @binding(0) var<uniform> S: Scene;
@group(0) @binding(1) var<uniform> D: Draw;

fn F(x: f32, y: f32) -> f32 { return ${expr}; }
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
`;

export const wgslSurface = (expr: string, n: number) => /* wgsl */ `
${wgslCommon(expr)}
struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) wp: vec3f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let N = ${n}u;
  let uv = vec2f(f32(i % N), f32(i / N)) / f32(N - 1u) * 2.0 - 1.0;
  let wp = worldPos(uv);
  var o: VOut;
  o.pos = S.viewProj * vec4f(wp, 1.0);
  o.uv = uv;
  o.wp = wp;
  return o;
}
@fragment fn fs(v: VOut) -> @location(0) vec4f {
  let t = height01(v.uv);
  let e = 0.0015;
  let hs = S.zinfo.z;
  let hu = (height01(v.uv + vec2f(e, 0.0)) - height01(v.uv - vec2f(e, 0.0))) * hs / (2.0 * e);
  let hv = (height01(v.uv + vec2f(0.0, e)) - height01(v.uv - vec2f(0.0, e))) * hs / (2.0 * e);
  var n = normalize(vec3f(-hu, 1.0, hv));
  let V = normalize(S.eye.xyz - v.wp);
  if (dot(n, V) < 0.0) { n = -n; } // seen from below
  let L = normalize(S.light.xyz);
  let diff = max(dot(n, L), 0.0);
  let hemi = mix(0.6, 1.0, n.y * 0.5 + 0.5);
  let H = normalize(L + V);
  let spec = pow(max(dot(n, H), 0.0), 60.0) * 0.16;
  var col = viridis(clamp(t, 0.0, 1.0)) * (S.light.w * hemi + (1.0 - S.light.w) * diff) + spec;
  let c = t * S.style.x;
  let w = fwidth(c);
  let line = 1.0 - smoothstep(0.0, w * 1.3, abs(fract(c + 0.5) - 0.5));
  col = mix(col, col * 0.5, line * S.style.y);
  let d = length(S.eye.xyz - v.wp);
  let fog = 1.0 - exp(-max(d - 2.2, 0.0) * S.style.z);
  col = mix(col, S.bg.rgb, fog);
  return vec4f(col, 1.0);
}
`;

const wgslCorner = /* wgsl */ `
fn corner(i: u32) -> vec2f {
  var c = array<vec2f, 6>(vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  return c[i];
}
`;

export const wgslTrail = (expr: string) => /* wgsl */ `
${wgslCommon(expr)}
${wgslCorner}
struct VOut { @builtin(position) pos: vec4f, @location(0) a: f32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32,
              @location(0) p0: vec2f, @location(1) p1: vec2f) -> VOut {
  let k = corner(vi);
  let lift = vec3f(0.0, D.misc.y, 0.0);
  let ca = S.viewProj * vec4f(worldPos(p0) + lift, 1.0);
  let cb = S.viewProj * vec4f(worldPos(p1) + lift, 1.0);
  var dir = (cb.xy / cb.w - ca.xy / ca.w) * S.viewport.xy;
  if (length(dir) < 1e-6) { dir = vec2f(1.0, 0.0); }
  dir = normalize(dir);
  let nrm = vec2f(-dir.y, dir.x);
  var c = ca;
  if (k.x > 0.5) { c = cb; }
  let off = nrm * k.y * D.misc.x * S.viewport.z / S.viewport.xy;
  var o: VOut;
  o.pos = c + vec4f(off * c.w, 0.0, 0.0);
  o.a = mix(0.25, 1.0, pow((f32(ii) + k.x) / max(D.misc.w - 1.0, 1.0), 0.6));
  return o;
}
@fragment fn fs(v: VOut) -> @location(0) vec4f { return vec4f(D.color.rgb, D.color.a * v.a); }
`;

export const wgslPoints = (expr: string) => /* wgsl */ `
${wgslCommon(expr)}
struct VOut { @builtin(position) pos: vec4f, @location(0) q: vec2f };
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) p: vec2f) -> VOut {
  var c = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let q = c[vi];
  let cp = S.viewProj * vec4f(worldPos(p) + vec3f(0.0, D.misc.y, 0.0), 1.0);
  var o: VOut;
  o.pos = cp + vec4f(q * D.misc.x * S.viewport.z * 2.0 / S.viewport.xy * cp.w, 0.0, 0.0);
  o.q = q;
  return o;
}
@fragment fn fs(v: VOut) -> @location(0) vec4f {
  let r = length(v.q);
  let aa = fwidth(r) * 1.5;
  let disc = 1.0 - smoothstep(1.0 - aa, 1.0, r);
  if (disc <= 0.0) { discard; }
  var col = D.color.rgb;
  if (D.misc.z > 0.5) {            // filled with a light rim
    col = mix(col, vec3f(1.0), smoothstep(0.62 - aa, 0.62, r));
  }
  if (D.misc.z > 1.5) {            // ring only
    let ring = smoothstep(0.55 - aa, 0.55, r);
    return vec4f(D.color.rgb, disc * ring * D.color.a);
  }
  return vec4f(col, disc * D.color.a);
}
`;

/* ---------------- GLSL ES 3.00 ---------------- */

const glslCommon = (expr: string) => /* glsl */ `
precision highp float;
const float PI = 3.14159265358979;
layout(std140) uniform Scene {
  mat4 viewProj; vec4 eye; vec4 bounds; vec4 zinfo; vec4 light; vec4 viewport; vec4 bg; vec4 style;
} S;
layout(std140) uniform Draw { vec4 color; vec4 misc; } D;

float F(float x, float y) { return ${expr}; }
float height01(vec2 uv) {
  vec2 p = mix(S.bounds.xz, S.bounds.yw, uv * 0.5 + 0.5);
  float f = F(p.x, p.y);
  if (S.zinfo.w > 0.5) f = log(1.0 + max(f, 0.0));
  return (f - S.zinfo.x) / max(S.zinfo.y - S.zinfo.x, 1e-20);
}
vec3 worldPos(vec2 uv) { return vec3(uv.x, (height01(uv) - 0.5) * S.zinfo.z, -uv.y); }
vec3 viridis(float t) {
  const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
  const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
  const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
// our projection maps depth to [0, 1] (WebGPU); GL wants [-1, 1]
vec4 toGL(vec4 p) { return vec4(p.xy, p.z * 2.0 - p.w, p.w); }
`;

export const glslSurface = (expr: string, n: number) => ({
  vs: /* glsl */ `#version 300 es
${glslCommon(expr)}
out vec2 vUv; out vec3 vWp;
void main() {
  int N = ${n};
  vec2 uv = vec2(float(gl_VertexID % N), float(gl_VertexID / N)) / float(N - 1) * 2.0 - 1.0;
  vec3 wp = worldPos(uv);
  vUv = uv; vWp = wp;
  gl_Position = toGL(S.viewProj * vec4(wp, 1.0));
}`,
  fs: /* glsl */ `#version 300 es
${glslCommon(expr)}
in vec2 vUv; in vec3 vWp; out vec4 outColor;
void main() {
  float t = height01(vUv);
  float e = 0.0015;
  float hs = S.zinfo.z;
  float hu = (height01(vUv + vec2(e, 0.0)) - height01(vUv - vec2(e, 0.0))) * hs / (2.0 * e);
  float hv = (height01(vUv + vec2(0.0, e)) - height01(vUv - vec2(0.0, e))) * hs / (2.0 * e);
  vec3 n = normalize(vec3(-hu, 1.0, hv));
  vec3 V = normalize(S.eye.xyz - vWp);
  if (dot(n, V) < 0.0) n = -n;
  vec3 L = normalize(S.light.xyz);
  float diff = max(dot(n, L), 0.0);
  float hemi = mix(0.6, 1.0, n.y * 0.5 + 0.5);
  vec3 H = normalize(L + V);
  float spec = pow(max(dot(n, H), 0.0), 60.0) * 0.16;
  vec3 col = viridis(clamp(t, 0.0, 1.0)) * (S.light.w * hemi + (1.0 - S.light.w) * diff) + spec;
  float c = t * S.style.x;
  float w = fwidth(c);
  float line = 1.0 - smoothstep(0.0, w * 1.3, abs(fract(c + 0.5) - 0.5));
  col = mix(col, col * 0.5, line * S.style.y);
  float d = length(S.eye.xyz - vWp);
  float fog = 1.0 - exp(-max(d - 2.2, 0.0) * S.style.z);
  outColor = vec4(mix(col, S.bg.rgb, fog), 1.0);
}`,
});

export const glslTrail = (expr: string) => ({
  vs: /* glsl */ `#version 300 es
${glslCommon(expr)}
layout(location = 0) in vec2 p0; layout(location = 1) in vec2 p1;
out float vA;
const vec2 C[6] = vec2[6](vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(0.0, -1.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
void main() {
  vec2 k = C[gl_VertexID];
  vec3 lift = vec3(0.0, D.misc.y, 0.0);
  vec4 ca = S.viewProj * vec4(worldPos(p0) + lift, 1.0);
  vec4 cb = S.viewProj * vec4(worldPos(p1) + lift, 1.0);
  vec2 dir = (cb.xy / cb.w - ca.xy / ca.w) * S.viewport.xy;
  if (length(dir) < 1e-6) dir = vec2(1.0, 0.0);
  dir = normalize(dir);
  vec2 nrm = vec2(-dir.y, dir.x);
  vec4 c = k.x > 0.5 ? cb : ca;
  vec2 off = nrm * k.y * D.misc.x * S.viewport.z / S.viewport.xy;
  vA = mix(0.25, 1.0, pow((float(gl_InstanceID) + k.x) / max(D.misc.w - 1.0, 1.0), 0.6));
  gl_Position = toGL(c + vec4(off * c.w, 0.0, 0.0));
}`,
  fs: /* glsl */ `#version 300 es
${glslCommon(expr)}
in float vA; out vec4 outColor;
void main() { outColor = vec4(D.color.rgb, D.color.a * vA); }`,
});

export const glslPoints = (expr: string) => ({
  vs: /* glsl */ `#version 300 es
${glslCommon(expr)}
layout(location = 0) in vec2 p;
out vec2 vQ;
const vec2 C[6] = vec2[6](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
void main() {
  vec2 q = C[gl_VertexID];
  vec4 cp = S.viewProj * vec4(worldPos(p) + vec3(0.0, D.misc.y, 0.0), 1.0);
  vQ = q;
  gl_Position = toGL(cp + vec4(q * D.misc.x * S.viewport.z * 2.0 / S.viewport.xy * cp.w, 0.0, 0.0));
}`,
  fs: /* glsl */ `#version 300 es
${glslCommon(expr)}
in vec2 vQ; out vec4 outColor;
void main() {
  float r = length(vQ);
  float aa = fwidth(r) * 1.5;
  float disc = 1.0 - smoothstep(1.0 - aa, 1.0, r);
  if (disc <= 0.0) discard;
  vec3 col = D.color.rgb;
  if (D.misc.z > 0.5) col = mix(col, vec3(1.0), smoothstep(0.62 - aa, 0.62, r));
  if (D.misc.z > 1.5) { outColor = vec4(D.color.rgb, disc * smoothstep(0.55 - aa, 0.55, r) * D.color.a); return; }
  outColor = vec4(col, disc * D.color.a);
}`,
});
