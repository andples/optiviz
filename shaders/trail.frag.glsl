in float vA; out vec4 outColor;
void main() { outColor = vec4(D.color.rgb, D.color.a * vA); }
