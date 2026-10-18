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
}
