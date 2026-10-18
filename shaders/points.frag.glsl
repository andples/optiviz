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
}
