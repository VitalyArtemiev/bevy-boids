// Shared machinery for the formation-crowd experiment (see src/crowd.rs):
// the "fluid frontline" occupancy field, per-cell soldier generation, and
// the 2-D DDA that walks a view ray through the crowd. Both crowd.wgsl
// (opaque soldiers) and crowd_dust.wgsl (translucent haze) import this.
//
// Space: everything lives in the box mesh's local coordinates — x runs
// along the front (centred on 0), z is depth with the ENEMY at -z (the
// fluid leading edge wobbles around z = front_z(x, t), solid ranks extend
// to z = DEPTH_M), y is up from the box floor. The vertex stage deforms
// y by `swell_y` so the box reads as a fluid mass; the field rides the
// same swell, so the crowd pattern stays glued to the deforming geometry
// with no extra vertex attributes.
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
fn swell_y(xz: vec2<f32>, t: f32) -> f32 {
    return 0.30 * sin(xz.x * 0.16 + t * 0.37) * sin(xz.y * 0.13 - t * 0.29)
         + 0.12 * sin(xz.x * 0.41 - t * 0.61 + 1.3) * sin(xz.y * 0.47 + t * 0.43);
}

// --- the fluid frontline ----------------------------------------------------

// Depth coordinate of the leading edge at x: the army fills z in
// [front_z, DEPTH_M]. Two wave frequencies plus a slow surge make the edge
// churn like a liquid interface instead of a rigid outline.
fn front_z(x: f32, t: f32) -> f32 {
    return DEPTH_M * (0.68
        + 0.055 * sin(x * 0.09 + t * 0.42)
        + 0.035 * sin(x * 0.21 - t * 0.71 + 1.7)
        + 0.018 * sin(x * 0.47 + t * 0.23));
}

// Occupancy by depth behind the leading edge: a loose skirmish screen up
// front, solid ranks behind — the leading edge reads as fluid partly
// because it is sparse and ragged.
fn row_density(depth_rows: f32) -> f32 {
    return mix(0.45, 0.98, smoothstep(0.0, 7.0, depth_rows));
}

// --- per-cell soldiers ------------------------------------------------------

struct Soldier {
    base: vec3<f32>, // feet centre, riding the swell
    height: f32,     // 1.55..1.95 m
    radius: f32,     // body cylinder radius
    head_r: f32,     // helmet sphere radius
    tone: f32,       // albedo variation 0..1
    metal: f32,      // armour amount 0..1
    phase: f32,      // sway + glint phase 0..1
};

// Field + light parameters bundled for the trace (the entry shaders fill
// it from their material uniforms).
struct FieldParams {
    spacing: f32,
    team_seed: f32,
    density: f32,
    glint_rate: f32,
    glint_strength: f32,
    height: f32, // box height the trace should clip to
    sun_l: vec3<f32>, // direction TO the sun, local space
}

fn cell_center(cell: vec2<i32>, spacing: f32) -> vec2<f32> {
    return vec2<f32>((f32(cell.x) + 0.5) * spacing - LEN_M / 2.0, (f32(cell.y) + 0.5) * spacing);
}

fn soldier_exists(cell: vec2<i32>, fp: FieldParams, t: f32) -> bool {
    // Cell hashes stay small so the sin-hash keeps its bits on every GPU.
    let q = vec2<f32>(f32(cell.x % 512), f32(cell.y % 512));
    let c = cell_center(cell, fp.spacing);
    let fz = front_z(c.x, t);
    if (c.y < fz) {
        return false;
    }
    let depth_rows = (c.y - fz) / fp.spacing;
    let r = hash21(q + vec2<f32>(fp.team_seed, -fp.team_seed * 0.5));
    return r < row_density(depth_rows) * fp.density;
}

fn make_soldier(cell: vec2<i32>, fp: FieldParams, t: f32) -> Soldier {
    let q = vec2<f32>(f32(cell.x % 512), f32(cell.y % 512));
    let seed = q * 1.618 + vec2<f32>(fp.team_seed, fp.team_seed * 1.1);
    let h = hash22(seed);
    let h2 = hash22(seed + vec2<f32>(5.2, 7.9));
    let c = cell_center(cell, fp.spacing);

    let height = mix(1.55, 1.95, h.x);
    let radius = mix(0.16, 0.22, h.y);
    // Jitter plus the marching sway must keep the body disk inside its
    // cell: then per-cell DDA tests are exact with no neighbour probing.
    let jitter_max = 0.5 * fp.spacing - radius - 0.08;
    let jitter = (h2 - vec2<f32>(0.5)) * 2.0 * jitter_max;
    let phase = fract(h.x * 7.31 + h.y * 3.17);
    let phase_ang = phase * 6.28318;

    var s = Soldier(
        vec3<f32>(
            c.x + jitter.x + 0.05 * sin(t * 1.1 + phase_ang),
            swell_y(c, t) + 0.03 * sin(t * 2.2 + phase * 12.566),
            c.y + jitter.y + 0.05 * cos(t * 0.9 + phase_ang),
        ),
        height,
        radius,
        radius * 1.15,
        fract(h2.x + h2.y),
        clamp(h2.y * 1.4 - 0.2, 0.0, 1.0),
        phase,
    );
    return s;
}

// --- ray / soldier intersection ----------------------------------------------
// Body = vertical cylinder, head = sphere capping it. The design camera
// cone is within 30° of nadir, so parametrizing the ray by y turns the
// cylinder into a 2-D circle test: dx(y) = A + B*y, dz(y) = C + E*y.
//
// Top-down rays see the shoulder disk (cylinder's flat top minus the
// helmet); grazing rays at the cone's rim see the cylinder walls — that
// contrast is what gives the crowd its silhouettes.

struct Hit {
    t: f32, // < 0 when missed
    normal: vec3<f32>,
    hit_y: f32, // height above the soldier's feet
    is_head: bool,
}

fn hit_soldier(ro: vec3<f32>, rd: vec3<f32>, s: Soldier, t_min: f32) -> Hit {
    let body_top = s.height - s.head_r;
    let body_lo = 0.25; // feet/legs are occluded sludge
    let inv_dy = 1.0 / rd.y;
    let A = ro.x - s.base.x - ro.y * inv_dy * rd.x;
    let B = inv_dy * rd.x;
    let C = ro.z - s.base.z - ro.y * inv_dy * rd.z;
    let E = inv_dy * rd.z;

    var t = -1.0;
    var y = 0.0;
    var n = vec3<f32>(0.0, 1.0, 0.0);
    var head = false;

    let a = B * B + E * E;
    let b = 2.0 * (A * B + C * E);
    let c0 = A * A + C * C - s.radius * s.radius;
    if (a < 1e-7) {
        // Near-vertical ray: constant horizontal offset over y. Inside the
        // footprint means it lands on the shoulder disk.
        if (c0 < 0.0) {
            t = (s.base.y + body_top - ro.y) * inv_dy;
            y = body_top;
            n = vec3<f32>(0.0, 1.0, 0.0);
        }
    } else {
        let disc = b * b - 4.0 * a * c0;
        if (disc >= 0.0) {
            let dx_top = A + B * body_top;
            let dz_top = C + E * body_top;
            if (dx_top * dx_top + dz_top * dz_top < s.radius * s.radius) {
                // Passes over the shoulder rim: lands on the shoulders.
                t = (s.base.y + body_top - ro.y) * inv_dy;
                y = body_top;
                n = vec3<f32>(0.0, 1.0, 0.0);
            } else {
                // First wall crossing from above is the larger y root
                // (rd.y < 0 for downward rays in the design cone).
                let y_enter = (-b + sqrt(disc)) / (2.0 * a);
                if (y_enter >= body_lo && y_enter <= body_top) {
                    t = (s.base.y + y_enter - ro.y) * inv_dy;
                    y = y_enter;
                    n = normalize(vec3<f32>(A + B * y_enter, 0.0, C + E * y_enter));
                }
            }
        }
    }

    // Helmet sphere caps the cylinder; the nearer of the two wins.
    let c_head = vec3<f32>(s.base.x, s.base.y + s.height - s.head_r, s.base.z);
    let oc = ro - c_head;
    let qb = dot(oc, rd);
    let qc = dot(oc, oc) - s.head_r * s.head_r;
    let disc_h = qb * qb - qc; // rd is unit
    if (disc_h >= 0.0) {
        let t_head = -qb - sqrt(disc_h);
        if (t_head > 0.0 && (t < 0.0 || t_head < t)) {
            t = t_head;
            head = true;
            let p = ro + rd * t_head;
            n = normalize(p - c_head);
            y = p.y - s.base.y;
        }
    }

    if (t < t_min || t <= 0.0) {
        t = -1.0;
    }
    return Hit(t, n, y, head);
}

// --- the crowd trace ----------------------------------------------------------

struct Trace {
    t: f32, // < 0 when the ray threads the gaps
    normal: vec3<f32>,
    hit_y: f32,
    is_head: bool,
    soldier: Soldier,
    spark: f32, // spear-tip mirror flash strength, survives gap rays
    exit_xz: vec2<f32>, // where the ray leaves through the floor
}

fn default_trace() -> Trace {
    let s = Soldier(vec3<f32>(0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    return Trace(-1.0, vec3<f32>(0.0, 1.0, 0.0), 0.0, false, s, 0.0, vec2<f32>(0.0));
}

// Walk the ray through the soldier grid cell by cell (2-D DDA over x/z),
// keeping the nearest body hit and any spear-tip flash in front of it.
fn crowd_trace(ro: vec3<f32>, rd: vec3<f32>, t: f32, fp: FieldParams) -> Trace {
    var best = default_trace();
    // Horizontal rays are outside the design cone and would blow up the
    // y-parametrized tests; the RTS camera never gets that shallow (72°
    // off nadir at the closest zoom).
    if (abs(rd.y) < 0.15) {
        return best;
    }

    // Clip to the box, with swell headroom in y.
    let box_min = vec3<f32>(-LEN_M / 2.0, -FLOOR_PAD_M, 0.0);
    let box_max = vec3<f32>(LEN_M / 2.0, fp.height + FLOOR_PAD_M, DEPTH_M);
    var t0 = 0.0;
    var t1 = 1e9;
    for (var i = 0; i < 3; i = i + 1) {
        let d = select(rd[i], select(-1e-9, 1e-9, rd[i] >= 0.0), abs(rd[i]) < 1e-9);
        let ta = (box_min[i] - ro[i]) / d;
        let tb = (box_max[i] - ro[i]) / d;
        t0 = max(t0, min(ta, tb));
        t1 = min(t1, max(ta, tb));
    }
    if (t0 > t1) {
        return best;
    }

    let s = fp.spacing;
    let p0 = ro + rd * t0;
    var cell = vec2<i32>(
        i32(floor((p0.x + LEN_M / 2.0) / s)),
        i32(floor(p0.z / s)),
    );
    let step = vec2<i32>(select(-1, 1, rd.x >= 0.0), select(-1, 1, rd.z >= 0.0));
    var tdx = ((f32(cell.x) + select(0.0, 1.0, rd.x >= 0.0)) * s - LEN_M / 2.0 - p0.x) / rd.x;
    var tdz = ((f32(cell.y) + select(0.0, 1.0, rd.z >= 0.0)) * s - p0.z) / rd.z;
    let dtx = abs(s / rd.x);
    let dtz = abs(s / rd.z);
    let cells_x = i32(LEN_M / s) + 2;
    let cells_z = i32(DEPTH_M / s) + 2;

    // The cone geometry bounds the through-box path to a handful of cells;
    // the cap only bites on degenerate near-horizontal rays (guarded above).
    for (var n = 0; n < 48; n = n + 1) {
        if (cell.x >= 0 && cell.x < cells_x && cell.y >= 0 && cell.y < cells_z) {
            if (soldier_exists(cell, fp, t)) {
                let sol = make_soldier(cell, fp, t);
                let h = hit_soldier(ro, rd, sol, select(1e9, best.t, best.t > 0.0));
                if (h.t > 0.0) {
                    best.t = h.t;
                    best.normal = h.normal;
                    best.hit_y = h.hit_y;
                    best.is_head = h.is_head;
                    best.soldier = sol;
                }
                // Spear tip: a point flash above the helmet, ahead of the
                // ranks. The tip's tilt wobbles with the sway, so each spear
                // periodically sweeps through the mirror direction — the
                // classic field-of-pikes shimmer.
                let tip = vec3<f32>(sol.base.x, sol.base.y + sol.height + 0.16, sol.base.z - 0.18);
                let t_tip = (tip.y - ro.y) / rd.y;
                if (t_tip > t0 && t_tip < t1 && t_tip > 0.0 && (best.t < 0.0 || t_tip < best.t)) {
                    let hp = ro + rd * t_tip;
                    if (distance(hp.xz, tip.xz) < 0.08) {
                        let n_tip = normalize(vec3<f32>(
                            0.55 * sin(t * 0.9 + sol.phase * 6.28318),
                            1.0,
                            0.55 * cos(t * 0.8 + sol.phase * 5.1),
                        ));
                        let hv = normalize(-rd + fp.sun_l);
                        let flash = pow(max(dot(n_tip, hv), 0.0), 60.0);
                        best.spark = max(best.spark, flash * fp.glint_strength);
                    }
                }
            }
        }
        // Advance to the next cell boundary.
        if (tdx < tdz) {
            tdx = tdx + dtx;
            cell.x = cell.x + step.x;
        } else {
            tdz = tdz + dtz;
            cell.y = cell.y + step.y;
        }
        if (min(tdx, tdz) > t1) {
            break;
        }
    }

    let t_floor = clamp((0.0 - ro.y) / rd.y, t0, t1);
    best.exit_xz = (ro + rd * t_floor).xz;
    return best;
}
