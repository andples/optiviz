// GLSL ES 3.00 twin of common.wgsl (std140 blocks with the same layout).
precision highp float;
const float PI = 3.14159265358979;
layout(std140) uniform Scene {
  mat4 viewProj; vec4 eye; vec4 bounds; vec4 zinfo; vec4 light; vec4 viewport; vec4 bg; vec4 style;
} S;
layout(std140) uniform Draw { vec4 color; vec4 misc; } D;

float F(float x, float y) { return {{F}}; }
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
