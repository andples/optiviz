layout(location = 0) in vec2 p;
out vec2 vQ;
const vec2 C[6] = vec2[6](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
void main() {
  vec2 q = C[gl_VertexID];
  vec4 cp = S.viewProj * vec4(worldPos(p) + vec3(0.0, D.misc.y, 0.0), 1.0);
  vQ = q;
  gl_Position = toGL(cp + vec4(q * D.misc.x * S.viewport.z * 2.0 / S.viewport.xy * cp.w, 0.0, 0.0));
}
