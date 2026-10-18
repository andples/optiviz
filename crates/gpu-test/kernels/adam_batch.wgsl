// Spike for the basins map (DESIGN.md §4): Adam from many starts at once, one invocation per start, `steps`
// iterations per dispatch with the state kept in storage between dispatches. Gradients are central differences
// of f in f32 (the browser has no autodiff); the test measures how often the result matches the Rust f64 run.
struct Params { lr: f32, beta1: f32, beta2: f32, eps: f32, h: f32, steps: u32, t0: u32, n: u32 };
@group(1) @binding(0) var<uniform> P: Params;
@group(1) @binding(1) var<storage, read_write> x: array<vec2f>;
@group(1) @binding(2) var<storage, read_write> m: array<vec2f>;
@group(1) @binding(3) var<storage, read_write> v: array<vec2f>;

fn grad(p: vec2f) -> vec2f {
  let h = P.h;
  return vec2f(F(p.x + h, p.y) - F(p.x - h, p.y), F(p.x, p.y + h) - F(p.x, p.y - h)) / (2.0 * h);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.n) { return; }
  var xi = x[i];
  var mi = m[i];
  var vi = v[i];
  for (var k = 0u; k < P.steps; k++) {
    let t = f32(P.t0 + k + 1u);
    let g = grad(xi);
    mi = P.beta1 * mi + (1.0 - P.beta1) * g;
    vi = P.beta2 * vi + (1.0 - P.beta2) * g * g;
    let mh = mi / (1.0 - pow(P.beta1, t));
    let vh = vi / (1.0 - pow(P.beta2, t));
    xi = xi - P.lr * mh / (sqrt(vh) + P.eps);
  }
  x[i] = xi;
  m[i] = mi;
  v[i] = vi;
}
