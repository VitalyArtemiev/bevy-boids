// Shared library for the formation-crowd experiment (see src/crowd.rs):
// footprint metrics, hashes, noise, the box swell, and the "fluid
// frontline" curve. crowd.wgsl and crowd_dust.wgsl import it. The
// soldier/trace structs live in crowd.wgsl — they are private to the
// opaque pass; only CrowdOut crosses a module boundary.
//
// Space: everything lives in the box mesh's local coordinates — x runs
// along the front (centred on 0), z is depth with the ENEMY at -z (the
// fluid leading edge wobbles around z = front_z(x, t), solid ranks extend
// to z = DEPTH_M), y is up from the box floor.
//
// Two naga-oil gotchas (both silent: nothing draws, no error logged —
// well, a two-line `pipeline_cache: failed to process shader error` that
// is easy to grep past; see SKILL.md):
// 1. namespace imports (`bevy_boids::crowd_common`) resolve only against
//    LOADED shader assets — src/crowd.rs preloads this file
//    (`CrowdCommonShader`) and gates spawning on the load;
// 2. types from an imported module must be fully qualified in EVERY
//    signature (`crowd_common::CrowdOut`, never bare `CrowdOut` — WGSL
//    has no type aliases and no implicit module resolution).
#define_import_path bevy_boids::crowd_common

// Footprint metrics — must match the meshes built by `crowd_box`.
const LEN_M: f32 = 140.0;
const DEPTH_M: f32 = 36.0;
// Rays may clip a little past the floor because of the swell.
const FLOOR_PAD_M: f32 = 0.7;

// Shared vertex-stage output: the fragment needs a ray in the box's local
// space. `crowd_position` is the undeformed local position (the ray's
// anchor on the surface) and `camera_local` is the camera transformed
// into local space — a per-entity constant, so interpolating it across a
// triangle is exact.
struct CrowdOut {
    @builtin(position) position: vec4<f32>,
    @location(0) crowd_position: vec3<f32>,
    @location(1) camera_local: vec3<f32>,
};

// --- hashes -----------------------------------------------------------------

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(hash21(p), hash21(p + vec2<f32>(37.319, 11.87)));
}

// Smooth value noise + fbm for the dust volume.
fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash21(i), hash21(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash21(i + vec2<f32>(0.0, 1.0)), hash21(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y,
    );
}

fn fbm2(p: vec2<f32>) -> f32 {
    return 0.6 * vnoise(p) + 0.3 * vnoise(p * 2.3 + 17.0) + 0.1 * vnoise(p * 5.1 - 9.0);
}

// --- the deforming box ------------------------------------------------------

// Slow vertical swell shared by the vertex stage (displaces the surface)
// and the field (soldier feet ride it). Keeping one function is what makes
// the crowd stick to the deforming geometry instead of sliding through it.
// Amplitude is capped well below the lid-to-helmet headroom: a taller
// swell opens a window between the lid and the soldiers' heads through
// which tilted view rays slip over the whole rank file and hit bare floor
// (the crowd vanishes from oblique views).
fn swell_y(xz: vec2<f32>, t: f32) -> f32 {
    return 0.14 * sin(xz.x * 0.16 + t * 0.37) * sin(xz.y * 0.13 - t * 0.29)
         + 0.05 * sin(xz.x * 0.41 - t * 0.61 + 1.3) * sin(xz.y * 0.47 + t * 0.43);
}

// --- the fluid frontline ----------------------------------------------------

// Depth coordinate of the leading edge at x: the army fills z in
// [front_z, DEPTH_M]. Two wave frequencies plus a slow surge make the edge
// churn like a liquid interface instead of a rigid outline. The base sits
// NEAR THE BOX FRONT: the box is the formation's bounding volume, so most
// of its depth must hold ranks — a deep empty front strip renders as dark
// bare floor and makes the whole mass read as a hollow slab from oblique
// views.
fn front_z(x: f32, t: f32) -> f32 {
    return DEPTH_M * (0.15
        + 0.055 * sin(x * 0.09 + t * 0.42)
        + 0.035 * sin(x * 0.21 - t * 0.71 + 1.7)
        + 0.018 * sin(x * 0.47 + t * 0.23));
}

// Occupancy by depth behind the leading edge: a loose skirmish screen up
// front, solid ranks behind — the leading edge reads as fluid partly
// because it is sparse and ragged. Never below ~0.6 though: the front
// rank is what tilted views see first, and gaps there let rays slip over
// the ranks to the bare floor.
fn row_density(depth_rows: f32) -> f32 {
    return mix(0.62, 0.98, smoothstep(0.0, 7.0, depth_rows));
}

// Centre of a soldier grid cell, in box-local xz.
fn cell_center(cell: vec2<i32>, spacing: f32) -> vec2<f32> {
    return vec2<f32>((f32(cell.x) + 0.5) * spacing - LEN_M / 2.0, (f32(cell.y) + 0.5) * spacing);
}
