// The surface: the mesh gives the silhouette; height, normal, colour and contours are evaluated per pixel.
struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) wp: vec3f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let N = {{N}}u;
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
