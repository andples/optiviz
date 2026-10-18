//! The shaders in /shaders, run natively. Without a GPU every test prints a note and passes, unless
//! OPTIVIZ_REQUIRE_GPU=1 (CI with a software Vulkan sets it). Performance lines start with "PERF " (JSON).

use optiviz_core::problems;
use optiviz_gpu_test::*;
use std::time::Instant;

/// Timing sections are skipped with OPTIVIZ_PERF=0 (CI's software Vulkan would take minutes).
fn perf() -> bool {
    std::env::var("OPTIVIZ_PERF").as_deref() != Ok("0")
}

fn gpu() -> Option<Gpu> {
    let g = Gpu::new();
    match &g {
        Some(g) => eprintln!("GPU: {}", g.name),
        None if std::env::var("OPTIVIZ_REQUIRE_GPU").is_ok() => panic!("no GPU adapter"),
        None => eprintln!("no GPU adapter: skipping"),
    }
    g
}

fn compute(g: &Gpu, code: &str) -> wgpu::ComputePipeline {
    let module = g.compile(code).unwrap_or_else(|e| panic!("WGSL: {e}"));
    g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Each problem with its textbook coefficients, and with some changed (their values go into the shader as
/// literals, so both paths need checking).
fn cases() -> Vec<(&'static str, Vec<(&'static str, f64)>)> {
    let mut v: Vec<_> = problems::IDS.iter().map(|&id| (id, vec![])).collect();
    v.extend([
        ("rosenbrock", vec![("a", -0.5), ("b", 20.0)]),
        ("himmelblau", vec![("a", 9.0), ("b", 5.5)]),
        ("ackley", vec![("a", 10.0), ("b", 0.5), ("c", 3.0)]),
        ("bowl", vec![("kappa", 50.0), ("theta", -30.0)]),
        ("styblinski-tang", vec![("a", 10.0), ("b", -3.0)]),
        ("saddle", vec![("c", -0.5)]),
        ("hills", vec![("seed", 7.0), ("count", 4.0), ("width", 1.2)]),
    ]);
    v
}

#[test]
fn shader_expressions_match_the_rust_reference() {
    let Some(g) = gpu() else { return };
    for (id, coeffs) in cases() {
        let p = problems::build(id, 2, &coeffs).unwrap();
        let m = p.manifest();
        let expr = m.shader_expr.clone().unwrap();
        let pipe = compute(&g, &fill(EVAL_KERNEL, &expr, 0));
        // a grid over the view box plus the minima
        let n = 64;
        let mut pts: Vec<[f32; 2]> = vec![];
        for i in 0..n {
            for j in 0..n {
                let x = m.bounds[0].0 + (m.bounds[0].1 - m.bounds[0].0) * i as f64 / (n - 1) as f64;
                let y = m.bounds[1].0 + (m.bounds[1].1 - m.bounds[1].0) * j as f64 / (n - 1) as f64;
                pts.push([x as f32, y as f32]);
            }
        }
        pts.extend(m.minima.iter().map(|v| [v[0] as f32, v[1] as f32]));
        let input = g.storage(bytemuck::cast_slice(&pts), false);
        let output = g.storage(&vec![0u8; pts.len() * 4], true);
        let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            ],
        });
        let mut enc = g.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(1, &bg, &[]);
            pass.dispatch_workgroups((pts.len() as u32).div_ceil(64), 1, 1);
        }
        g.queue.submit([enc.finish()]);
        let got: Vec<f32> = bytemuck::cast_slice(&g.read(&output, (pts.len() * 4) as u64)).to_vec();
        let mut worst = 0f64;
        for (pt, &gv) in pts.iter().zip(&got) {
            // the reference at the same (f32-rounded) point
            let want = p.eval(&[pt[0] as f64, pt[1] as f64]);
            let err = (gv as f64 - want).abs() / (1.0 + want.abs());
            worst = worst.max(err);
            assert!(err < 5e-4, "{id} {coeffs:?} at {pt:?}: GPU {gv} vs Rust {want}");
        }
        eprintln!("{id} {coeffs:?}: worst relative error {worst:.2e} over {} points", pts.len());
    }
}

/// Builds the three render pipelines exactly as web/src/render/webgpu.ts does.
fn render_pipelines(g: &Gpu, expr: &str, n: u32, format: wgpu::TextureFormat) -> [wgpu::RenderPipeline; 3] {
    let bgl = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[0, 1].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }),
    });
    let layout = g.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let depth = Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24Plus,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: Default::default(),
        bias: Default::default(),
    });
    let no_write = depth.clone().map(|mut d| {
        d.depth_write_enabled = Some(false);
        d
    });
    let ms = wgpu::MultisampleState { count: 4, ..Default::default() };
    let point2 = |loc: u32| wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: Box::leak(Box::new([wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: loc,
        }])),
    };
    let blend = Some(wgpu::BlendState::ALPHA_BLENDING);
    let make = |code: String,
                buffers: &[Option<wgpu::VertexBufferLayout>],
                blend: Option<wgpu::BlendState>,
                depth: Option<wgpu::DepthStencilState>| {
        let module = g.compile(&code).unwrap_or_else(|e| panic!("WGSL: {e}"));
        g.validated(|| {
            g.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: depth,
                multisample: ms,
                multiview_mask: None,
                cache: None,
            })
        })
        .unwrap_or_else(|e| panic!("pipeline: {e}"))
    };
    [
        make(fill(SURFACE, expr, n), &[], None, depth.clone()),
        make(fill(TRAIL, expr, 0), &[Some(point2(0)), Some(point2(1))], blend, no_write.clone()),
        make(fill(POINTS, expr, 0), &[Some(point2(0))], blend, no_write),
    ]
}

#[test]
fn render_shaders_compile_and_link_for_every_problem() {
    let Some(g) = gpu() else { return };
    for (id, coeffs) in cases() {
        let expr = problems::build(id, 2, &coeffs).unwrap().manifest().shader_expr.unwrap();
        render_pipelines(&g, &expr, 384, wgpu::TextureFormat::Rgba8Unorm);
        eprintln!("{id}: surface, trail and points pipelines OK");
    }
}

/// Renders a problem's surface offscreen; returns (RGBA8 bytes, row pitch, seconds per frame over `frames`).
fn render_surface(g: &Gpu, id: &str, w: u32, h: u32, frames: u32) -> (Vec<u8>, usize, f64) {
    let p = problems::by_id(id, 2).unwrap();
    let m = p.manifest();
    let n = 384u32;
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let [surface, ..] = render_pipelines(g, m.shader_expr.as_deref().unwrap(), n, format);
    let scene = scene(&m, zrange(p.as_ref(), 257), w, h);
    let scene_buf = g.uniform(bytemuck::cast_slice(&scene));
    let draw_buf = g.uniform(&[0u8; DRAW_FLOATS * 4]);
    let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &surface.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scene_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: draw_buf.as_entire_binding() },
        ],
    });
    let mut idx: Vec<u32> = vec![];
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let (a, b, c) = (j * n + i, j * n + i + 1, (j + 1) * n + i);
            idx.extend([a, b, c + 1, a, c + 1, c]);
        }
    }
    let index = wgpu::util::DeviceExt::create_buffer_init(
        &g.device,
        &wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&idx),
            usage: wgpu::BufferUsages::INDEX,
        },
    );
    let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    let tex = |samples: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
        g.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let msaa = tex(4, format, wgpu::TextureUsages::RENDER_ATTACHMENT);
    let depth = tex(4, wgpu::TextureFormat::Depth24Plus, wgpu::TextureUsages::RENDER_ATTACHMENT);
    let target = tex(1, format, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
    let (mv, dv, tv) = (
        msaa.create_view(&Default::default()),
        depth.create_view(&Default::default()),
        target.create_view(&Default::default()),
    );
    let frame = || {
        let mut enc = g.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &mv,
                    depth_slice: None,
                    resolve_target: Some(&tv),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.98, g: 0.976, b: 0.965, a: 1.0 }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &dv,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&surface);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..idx.len() as u32, 0, 0..1);
        }
        g.queue.submit([enc.finish()]);
    };
    frame();
    g.wait(); // warm up
    let t = Instant::now();
    for _ in 0..frames {
        frame();
    }
    g.wait();
    let per_frame = secs(t) / frames as f64;

    // read the image back
    let row_pitch = (w as usize * 4).div_ceil(256) * 256;
    let out = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (row_pitch * h as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &out,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_pitch as u32),
                rows_per_image: None,
            },
        },
        size,
    );
    g.queue.submit([enc.finish()]);
    let bytes = g.read(&out, (row_pitch * h as usize) as u64);
    (bytes, row_pitch, per_frame)
}

#[test]
fn surfaces_render_and_report_frame_time() {
    let Some(g) = gpu() else { return };
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/gpu-test");
    for &id in problems::IDS {
        let (w, h) = (960, 600);
        let (img, pitch, _) = render_surface(&g, id, w, h, 1);
        // the surface covers a good part of the frame, and isn't one flat colour
        let mut covered = 0usize;
        let mut lum = vec![];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = y * pitch + x * 4;
                let (r, gg, b) = (img[i] as i32, img[i + 1] as i32, img[i + 2] as i32);
                if (r - 250).abs() + (gg - 249).abs() + (b - 246).abs() > 12 {
                    covered += 1;
                    lum.push(r + gg + b);
                }
            }
        }
        let frac = covered as f64 / (w * h) as f64;
        let spread = lum.iter().max().unwrap_or(&0) - lum.iter().min().unwrap_or(&0);
        assert!(frac > 0.15, "{id}: surface covers only {:.0}% of the frame", frac * 100.0);
        assert!(spread > 100, "{id}: surface looks flat (luminance spread {spread})");
        write_ppm(&dir.join(format!("{id}.ppm")), w, h, &img, pitch);
        eprintln!("{id}: covers {:.0}% of the frame, written to target/gpu-test/{id}.ppm", frac * 100.0);
    }
    if !perf() {
        return;
    }
    for (w, h) in [(1920, 1080), (3840, 2160)] {
        let (_, _, t) = render_surface(&g, "rosenbrock", w, h, 60);
        println!(
            "PERF {}",
            serde_json::json!({"test": "surface_frame", "gpu": g.name, "width": w, "height": h, "msaa": 4, "mesh": 384, "ms_per_frame": t * 1e3, "fps": 1.0 / t})
        );
    }
}

/// Runs the Adam batch kernel from `starts` for `steps` steps, `chunk` steps per dispatch. Returns final points.
fn adam_batch(g: &Gpu, expr: &str, starts: &[[f32; 2]], steps: u32, chunk: u32, lr: f32) -> (Vec<[f32; 2]>, f64) {
    let pipe = compute(g, &fill(ADAM_KERNEL, expr, 0));
    let n = starts.len();
    let x = g.storage(bytemuck::cast_slice(starts), true);
    let zeros = vec![0u8; n * 8];
    let (m, v) = (g.storage(&zeros, true), g.storage(&zeros, true));
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        h: f32,
        steps: u32,
        t0: u32,
        n: u32,
    }
    let mut params = Params { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, h: 1e-3, steps: chunk, t0: 0, n: n as u32 };
    let ubuf = g.uniform(bytemuck::bytes_of(&params));
    let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: m.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v.as_entire_binding() },
        ],
    });
    let t = Instant::now();
    let mut done = 0;
    while done < steps {
        params.steps = chunk.min(steps - done);
        params.t0 = done;
        g.queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&params));
        let mut enc = g.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(1, &bg, &[]);
            let groups = (n as u32).div_ceil(64);
            pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
        }
        g.queue.submit([enc.finish()]);
        done += params.steps;
    }
    g.wait();
    let elapsed = secs(t);
    let out: Vec<[f32; 2]> = bytemuck::cast_slice(&g.read(&x, (n * 8) as u64)).to_vec();
    (out, elapsed)
}

#[test]
fn adam_batch_kernel_agrees_with_rust_and_scales() {
    let Some(g) = gpu() else { return };
    let p = problems::by_id("himmelblau", 2).unwrap();
    let m = p.manifest();
    let expr = m.shader_expr.clone().unwrap();
    let expr = expr.as_str();
    let grid = |n: usize| -> Vec<[f32; 2]> {
        let (bx, by) = (m.bounds[0], m.bounds[1]);
        (0..n * n)
            .map(|k| {
                let (i, j) = (k % n, k / n);
                [
                    (bx.0 + (bx.1 - bx.0) * (i as f64 + 0.5) / n as f64) as f32,
                    (by.0 + (by.1 - by.0) * (j as f64 + 0.5) / n as f64) as f32,
                ]
            })
            .collect()
    };
    let nearest = |x: f64, y: f64| -> usize {
        (0..m.minima.len())
            .min_by(|&a, &b| {
                let da = (x - m.minima[a][0]).hypot(y - m.minima[a][1]);
                let db = (x - m.minima[b][0]).hypot(y - m.minima[b][1]);
                da.total_cmp(&db)
            })
            .unwrap()
    };
    // agreement: which minimum each of 64² starts ends in, GPU (f32, finite differences) vs Rust (f64, analytic)
    let starts = grid(64);
    let (gpu_end, _) = adam_batch(&g, expr, &starts, 400, 100, 0.05);
    let mut same = 0;
    for (s, e) in starts.iter().zip(&gpu_end) {
        let mut o = optiviz_core::optimizers::by_id("adam").unwrap();
        o.set_param("lr", 0.05);
        o.init(&[s[0] as f64, s[1] as f64]);
        let mut rng = optiviz_core::Rng::new(0);
        for _ in 0..400 {
            o.step(p.as_ref(), &mut rng, None);
        }
        let x = o.x();
        if nearest(x[0], x[1]) == nearest(e[0] as f64, e[1] as f64) {
            same += 1;
        }
    }
    let agree = same as f64 / starts.len() as f64;
    eprintln!("basin agreement GPU vs Rust: {:.2}%", agree * 100.0);
    assert!(agree > 0.97, "only {:.1}% of starts end in the same basin", agree * 100.0);

    // scale: the basins-map workload from DESIGN.md §8 (2048² starts × 500 Adam steps)
    if !perf() {
        return;
    }
    let big = grid(2048);
    let (_, t) = adam_batch(&g, expr, &big, 500, 50, 0.05);
    println!(
        "PERF {}",
        serde_json::json!({"test": "basins_adam", "gpu": g.name, "starts": big.len(), "steps": 500, "seconds": t, "start_steps_per_s": big.len() as f64 * 500.0 / t})
    );
}
