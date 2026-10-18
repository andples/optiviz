// Point sprites: marker, minima rings and the stress-test swarm (instanced quads, anti-aliased discs).
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
