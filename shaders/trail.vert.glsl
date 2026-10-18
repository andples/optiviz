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
}
