// Trail: one screen-space quad per segment (instanced), fading towards the start (unless misc.z > 0.5, used for
// the halo pass and closed shapes). Segments with an end outside the view box are dropped: a run that leaves the
// box shows as a marker pinned to the edge (host.ts), not as lines across the floor.
fn corner(i: u32) -> vec2f {
  var c = array<vec2f, 6>(vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  return c[i];
}
struct VOut { @builtin(position) pos: vec4f, @location(0) a: f32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32,
              @location(0) p0: vec2f, @location(1) p1: vec2f) -> VOut {
  let k = corner(vi);
  let inside = all(abs(p0) <= vec2f(1.0005)) && all(abs(p1) <= vec2f(1.0005));
  let lift = vec3f(0.0, D.misc.y, 0.0);
  let ca = S.viewProj * vec4f(worldPos(clamp(p0, vec2f(-1.0), vec2f(1.0))) + lift, 1.0);
  let cb = S.viewProj * vec4f(worldPos(clamp(p1, vec2f(-1.0), vec2f(1.0))) + lift, 1.0);
  var dir = (cb.xy / cb.w - ca.xy / ca.w) * S.viewport.xy;
  if (length(dir) < 1e-6) { dir = vec2f(1.0, 0.0); }
  dir = normalize(dir);
  let nrm = vec2f(-dir.y, dir.x);
  var c = ca;
  if (k.x > 0.5) { c = cb; }
  let off = nrm * k.y * D.misc.x * S.viewport.z / S.viewport.xy;
  var o: VOut;
  o.pos = c + vec4f(off * c.w, 0.0, 0.0);
  if (!inside) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); } // outside the clip volume: dropped
  o.a = mix(0.25, 1.0, pow((f32(ii) + k.x) / max(D.misc.w - 1.0, 1.0), 0.6));
  if (D.misc.z > 0.5) { o.a = 1.0; }
  return o;
}
@fragment fn fs(v: VOut) -> @location(0) vec4f { return vec4f(D.color.rgb, D.color.a * v.a); }
