/**
 * Shader assembly. The sources live in /shaders (shared with crates/gpu-test, which compiles the same WGSL with
 * native wgpu and checks it against the Rust reference). Each shader is `common` + its own file, with the
 * problem's f(x, y) spliced in for {{F}} and the mesh resolution for {{N}}. `shader_expr` is written to be
 * valid in both WGSL and GLSL.
 *
 * Uniforms (identical layout in WGSL and std140):
 *   Scene { viewProj: mat4; eye; bounds (x0, x1, y0, y1); zinfo (zmin, zmax, height scale, log flag);
 *           light (dir xyz, ambient); viewport (w, h, dpr, time); bg (rgb, dark); style (levels, contour
 *           strength, fog, grid) }  = 44 floats
 *   Draw  { color; misc (size px, lift, mode, count) } = 8 floats
 */

import commonWgsl from "../../../shaders/common.wgsl?raw";
import surfaceWgsl from "../../../shaders/surface.wgsl?raw";
import trailWgsl from "../../../shaders/trail.wgsl?raw";
import pointsWgsl from "../../../shaders/points.wgsl?raw";
import commonGlsl from "../../../shaders/common.glsl?raw";
import surfaceVert from "../../../shaders/surface.vert.glsl?raw";
import surfaceFrag from "../../../shaders/surface.frag.glsl?raw";
import trailVert from "../../../shaders/trail.vert.glsl?raw";
import trailFrag from "../../../shaders/trail.frag.glsl?raw";
import pointsVert from "../../../shaders/points.vert.glsl?raw";
import pointsFrag from "../../../shaders/points.frag.glsl?raw";

export const SCENE_FLOATS = 44;
export const DRAW_FLOATS = 8;

const fill = (src: string, expr: string, n = 0) => src.replaceAll("{{F}}", expr).replaceAll("{{N}}", String(n));

export const wgslSurface = (expr: string, n: number) => fill(commonWgsl + surfaceWgsl, expr, n);
export const wgslTrail = (expr: string) => fill(commonWgsl + trailWgsl, expr);
export const wgslPoints = (expr: string) => fill(commonWgsl + pointsWgsl, expr);

const glsl = (body: string, expr: string, n = 0) => "#version 300 es\n" + fill(commonGlsl + body, expr, n);
export const glslSurface = (expr: string, n: number) => ({ vs: glsl(surfaceVert, expr, n), fs: glsl(surfaceFrag, expr, n) });
export const glslTrail = (expr: string) => ({ vs: glsl(trailVert, expr), fs: glsl(trailFrag, expr) });
export const glslPoints = (expr: string) => ({ vs: glsl(pointsVert, expr), fs: glsl(pointsFrag, expr) });
