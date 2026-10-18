//! Native harness for the shaders in /shaders: the very WGSL the browser runs, compiled and executed with
//! wgpu (Vulkan/Metal/DX12) so it can be validated, checked against the Rust f64 reference and timed on a real
//! GPU. Used by tests/kernels.rs.

use std::time::Instant;
use wgpu::util::DeviceExt;

pub const COMMON: &str = include_str!("../../../shaders/common.wgsl");
pub const SURFACE: &str = include_str!("../../../shaders/surface.wgsl");
pub const TRAIL: &str = include_str!("../../../shaders/trail.wgsl");
pub const POINTS: &str = include_str!("../../../shaders/points.wgsl");
pub const EVAL_KERNEL: &str = include_str!("../kernels/eval.wgsl");
pub const ADAM_KERNEL: &str = include_str!("../kernels/adam_batch.wgsl");

pub const SCENE_FLOATS: usize = 44;
pub const DRAW_FLOATS: usize = 8;

/// Splices f and the mesh size in, as web/src/render/shaders.ts does.
pub fn fill(src: &str, expr: &str, n: u32) -> String {
    format!("{COMMON}{src}").replace("{{F}}", expr).replace("{{N}}", &n.to_string())
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub name: String,
}

impl Gpu {
    /// The high-performance adapter, or None when the machine has no usable GPU.
    pub fn new() -> Option<Gpu> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .ok()?;
            let info = adapter.get_info();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("optiviz-gpu-test"),
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .ok()?;
            Some(Gpu { device, queue, name: format!("{} ({:?}, {})", info.name, info.backend, info.driver) })
        })
    }

    /// Compiles WGSL, returning the validation error text on failure.
    pub fn compile(&self, code: &str) -> Result<wgpu::ShaderModule, String> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        match pollster::block_on(scope.pop()) {
            Some(e) => Err(e.to_string()),
            None => Ok(m),
        }
    }

    /// Runs `f` inside a validation error scope.
    pub fn validated<T>(&self, f: impl FnOnce() -> T) -> Result<T, String> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let v = f();
        match pollster::block_on(scope.pop()) {
            Some(e) => Err(e.to_string()),
            None => Ok(v),
        }
    }

    pub fn wait(&self) {
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }

    pub fn storage(&self, data: &[u8], writable: bool) -> wgpu::Buffer {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        if !writable {
            usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        }
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: data, usage })
    }

    pub fn uniform(&self, data: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Copies a buffer back to the CPU.
    pub fn read(&self, buf: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.wait();
        let out = slice.get_mapped_range().expect("mapped range").to_vec();
        staging.unmap();
        out
    }
}

/* ---------- the render worker's camera, ported (web/src/render/camera.ts) ---------- */

pub fn view_proj(yaw: f32, pitch: f32, dist: f32, aspect: f32) -> [f32; 16] {
    let eye = eye(yaw, pitch, dist);
    mul(&perspective(32f32.to_radians(), aspect, 0.05, 50.0), &look_at(eye, [0.0, -0.05, 0.0]))
}

pub fn eye(yaw: f32, pitch: f32, dist: f32) -> [f32; 3] {
    [dist * pitch.cos() * yaw.sin(), dist * pitch.sin(), dist * pitch.cos() * yaw.cos()]
}

fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fovy / 2.0).tan();
    let mut m = [0.0; 16];
    m[0] = f / aspect;
    m[5] = f;
    m[10] = far / (near - far);
    m[11] = -1.0;
    m[14] = near * far / (near - far);
    m
}

fn look_at(eye: [f32; 3], target: [f32; 3]) -> [f32; 16] {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross =
        |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let norm = |a: [f32; 3]| {
        let l = dot(a, a).sqrt();
        [a[0] / l, a[1] / l, a[2] / l]
    };
    let z = norm(sub(eye, target));
    let x = norm(cross([0.0, 1.0, 0.0], z));
    let y = cross(z, x);
    [x[0], y[0], z[0], 0.0, x[1], y[1], z[1], 0.0, x[2], y[2], z[2], 0.0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1.0]
}

fn mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut o = [0.0; 16];
    for c in 0..4 {
        for r in 0..4 {
            o[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    o
}

/// Scene uniforms for a problem, laid out as in shaders.ts / host.ts.
pub fn scene(m: &optiviz_core::ProblemManifest, zrange: (f64, f64), w: u32, h: u32) -> [f32; SCENE_FLOATS] {
    let mut s = [0f32; SCENE_FLOATS];
    s[..16].copy_from_slice(&view_proj(-0.75, 0.62, 4.3, w as f32 / h as f32));
    let e = eye(-0.75, 0.62, 4.3);
    s[16..20].copy_from_slice(&[e[0], e[1], e[2], 1.0]);
    let b = &m.bounds;
    s[20..24].copy_from_slice(&[b[0].0 as f32, b[0].1 as f32, b[1].0 as f32, b[1].1 as f32]);
    s[24..28].copy_from_slice(&[zrange.0 as f32, zrange.1 as f32, 0.62, if m.log_height { 1.0 } else { 0.0 }]);
    s[28..32].copy_from_slice(&[-0.45, 0.85, 0.35, 0.42]);
    s[32..36].copy_from_slice(&[w as f32, h as f32, 1.0, 0.0]);
    s[36..40].copy_from_slice(&[0.98, 0.976, 0.965, 0.0]);
    s[40..44].copy_from_slice(&[24.0, 0.28, 0.22, 0.0]);
    s
}

/// [min, max] of the drawn height over a grid, as crates/wasm `surface_range` computes it.
pub fn zrange(p: &dyn optiviz_core::Problem, n: usize) -> (f64, f64) {
    let m = p.manifest();
    let (bx, by) = (m.bounds[0], m.bounds[1]);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        for j in 0..n {
            let x = bx.0 + (bx.1 - bx.0) * i as f64 / (n - 1) as f64;
            let y = by.0 + (by.1 - by.0) * j as f64 / (n - 1) as f64;
            let f = p.eval(&[x, y]);
            let h = if m.log_height { f.max(0.0).ln_1p() } else { f };
            lo = lo.min(h);
            hi = hi.max(h);
        }
    }
    (lo, hi)
}

/// Seconds since `t`.
pub fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

/// Writes an RGBA8 image as binary PPM (no image dependency needed to look at it).
pub fn write_ppm(path: &std::path::Path, w: u32, h: u32, rgba: &[u8], row_pitch: usize) {
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = y * row_pitch + x * 4;
            out.extend_from_slice(&rgba[i..i + 3]);
        }
    }
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, out).unwrap();
}
