out vec2 vUv; out vec3 vWp;
void main() {
  int N = {{N}};
  vec2 uv = vec2(float(gl_VertexID % N), float(gl_VertexID / N)) / float(N - 1) * 2.0 - 1.0;
  vec3 wp = worldPos(uv);
  vUv = uv; vWp = wp;
  gl_Position = toGL(S.viewProj * vec4(wp, 1.0));
}
