// Crowd shell, opaque pass: renders a mass of soldiers inside the
// deformed formation box (see crowd_common.wgsl for the shared metrics,
// field functions, and local-space conventions). The vertex stage swells
// the box; the fragment stage walks the view ray through the soldier grid
// (2-D DDA in surge-warped coordinates — the whole column breathes, no
// soldier ever pops) and shades the nearest hit: shoulder disks and
// helmets from nadir, body-wall silhouettes at the cone rim, periodic
// armour and spear-tip glints. Rays beneath occupied cells shade dark
// crowd shadow; every other ray is discarded (alpha mask), so the box
// itself is invisible against the terrain.
//
// The soldier/trace structs live here rather than in crowd_common: they
// are private to this pass. Types imported from a module must be fully
// qualified everywhere (crowd_common::CrowdOut).

#import bevy_boids::crowd_common

#import bevy_pbr::{
    mesh_functions,
    forward_io::Vertex,
    view_transformations::position_world_to_clip,
}
#import bevy_pbr::mesh_view_bindings::{view, globals}
#import bevy_render::maths

// Team colour (rgb) + per-team hash seed.
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> team: vec4<f32>;
// x: occupancy density scale, y: glint rate (rad/s), z: glint strength,
// w: unused (the trace y range moved to crowd_common::terrain_c.zw with terrain follow).
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> params: vec4<f32>;
// xyz: direction TO the sun in local space, w: soldier spacing (m).
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> sun_spacing: vec4<f32>;

@vertex
fn vertex(v: Vertex) -> bevy_boids::crowd_common::CrowdOut {
    let world_from_local = mesh_functions::get_world_from_local(v.instance_index);
    var local = v.position;
    // The box deforms to ride the terrain (plus the swell); the same
    // displacement is re-derived per fragment for the ray anchor, so the
    // two agree to within the heightmap's texel resolution.
    local.y += crowd_common::terrain_h_rel(local.xz)
        + crowd_common::swell_y(local.xz, globals.time);
    let world = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(local, 1.0),
    );
    // Rigid (rotation + translation) inverse: rotate the camera into local
    // space with the transposed basis. Entity transforms here are rigid —
    // the crowd is anchored to the box's local frame and follows it.
    let basis = maths::mat4x4_to_mat3x3(world_from_local);
    let cam_local = transpose(basis) * (view.world_position - world_from_local[3].xyz);

    var out: crowd_common::CrowdOut;
    out.position = position_world_to_clip(world.xyz);
    out.crowd_position = v.position;
    out.camera_local = cam_local;
    return out;
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

// Field + light parameters for the trace, filled from material uniforms.
struct FieldParams {
    spacing: f32,
    team_seed: f32,
    density: f32,
    glint_rate: f32,
    glint_strength: f32,
    y_min: f32, // local y range the trace clips to — low enough for the
    y_max: f32, // lowest ground and high enough for the highest ground
    sun_l: vec3<f32>, // direction TO the sun, local space
};

// Static existence: every roll is time-independent, so no soldier ever
// pops in or out. The frontline motion is applied separately, as a whole-
// column z warp in crowd_trace — ranks advance and retreat bodily while
// keeping their identity.
fn soldier_exists(cell: vec2<i32>, fp: FieldParams) -> bool {
    // Cell hashes stay small so the sin-hash keeps its bits on every GPU.
    let q = vec2<f32>(f32(cell.x % 512), f32(cell.y % 512));
    let c = crowd_common::cell_center(cell, fp.spacing);
    let fz0 = crowd_common::front_z(c.x, 0.0);
    if (c.y < fz0) {
        return false;
    }
    let depth_rows = (c.y - fz0) / fp.spacing;
    let r = crowd_common::hash21(q + vec2<f32>(fp.team_seed, -fp.team_seed * 0.5));
    // Rank drumming: odd rows pack 12% looser, so from above the mass
    // shows faint horizontal banding — the rank-and-file read that makes
    // a distant crowd legible. The flank and rear edges stay straight:
    // this is a formed body of troops, and the alpha mask removes the
    // box everywhere there are no ranks to show.
    let row_gate = select(0.93, 1.0, cell.y % 2 == 0);
    return r < crowd_common::row_density(depth_rows) * fp.density * row_gate;
}

fn make_soldier(cell: vec2<i32>, fp: FieldParams, t: f32) -> Soldier {
    let q = vec2<f32>(f32(cell.x % 512), f32(cell.y % 512));
    let seed = q * 1.618 + vec2<f32>(fp.team_seed, fp.team_seed * 1.1);
    let h = crowd_common::hash22(seed);
    let h2 = crowd_common::hash22(seed + vec2<f32>(5.2, 7.9));
    let c = crowd_common::cell_center(cell, fp.spacing);

    var s: Soldier;
    // Jitter plus the marching sway must keep the body disk inside its
    // cell: then per-cell DDA tests are exact with no neighbour probing.
    // The z component stays at 30%: rows keep their alignment, which is
    // what reads as rank structure from above.
    s.radius = mix(0.21, 0.28, h.y);
    let jitter_max = 0.5 * fp.spacing - s.radius - 0.08;
    let jitter = vec2<f32>(
        (h2.x - 0.5) * 2.0 * jitter_max,
        (h2.y - 0.5) * 2.0 * jitter_max * 0.3,
    );
    s.phase = fract(h.x * 7.31 + h.y * 3.17);
    let phase_ang = s.phase * 6.28318;
    s.height = mix(1.55, 1.95, h.x);
    s.base = vec3<f32>(
        c.x + jitter.x + 0.05 * sin(t * 1.1 + phase_ang),
        // Feet ride the terrain (height relative to the box's seat) under
        // the cell, plus the swell and a small step-bob on top.
        crowd_common::terrain_h_rel(c) + crowd_common::swell_y(c, t)
            + 0.03 * sin(t * 2.2 + s.phase * 12.566),
        c.y + jitter.y + 0.05 * cos(t * 0.9 + phase_ang),
    );
    // Helmet smaller than the shoulders: from above the team-coloured
    // shoulder disk must ring the steel, or every soldier reads as a
    // white ball and the army loses its colour.
    s.head_r = s.radius * 0.62;
    s.tone = fract(h2.x + h2.y);
    s.metal = clamp(h2.y * 1.4 - 0.2, 0.0, 1.0);
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
};

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

    var hit: Hit;
    // Keep the nearest hit: t_min is the best distance so far (1e9
    // sentinel before any hit).
    hit.t = select(-1.0, t, t <= t_min && t > 0.0);
    hit.normal = n;
    hit.hit_y = y;
    hit.is_head = head;
    return hit;
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
    under_ranks: bool, // passed beneath occupied cells (crowd shadow)
};

fn default_trace() -> Trace {
    var dummy: Soldier;
    dummy.base = vec3<f32>(0.0);
    dummy.height = 1.7;
    dummy.radius = 0.19;
    dummy.head_r = 0.22;
    dummy.tone = 0.5;
    dummy.metal = 0.5;
    dummy.phase = 0.0;

    var tr: Trace;
    tr.t = -1.0;
    tr.normal = vec3<f32>(0.0, 1.0, 0.0);
    tr.hit_y = 0.0;
    tr.is_head = false;
    tr.soldier = dummy;
    tr.spark = 0.0;
    tr.exit_xz = vec2<f32>(0.0);
    tr.under_ranks = false;
    return tr;
}

// Walk the ray through the soldier grid cell by cell (2-D DDA over x/z),
// keeping the nearest body hit and any spear-tip flash in front of it.
//
// The frontline surge is applied as a WARP of the walk's z coordinate
// (z' = z − surge): cell contents are fully static, so no soldier ever
// pops in or out — the whole column translates back and forth instead,
// and each cell's soldier is always found by the rays that visit its
// warped cell.
fn crowd_trace(ro: vec3<f32>, rd: vec3<f32>, t: f32, fp: FieldParams) -> Trace {
    var best = default_trace();
    // Horizontal rays are outside the design cone and would blow up the
    // y-parametrized tests; the RTS camera never gets that shallow (72°
    // off nadir at the closest zoom).
    if (abs(rd.y) < 0.15) {
        return best;
    }

    // Clip to the box. The y range spans the terrain rise across the
    // footprint plus the box height (uniforms, computed on the CPU from
    // the heightfield), so sloped ground stays inside the trace.
    let box_min = vec3<f32>(-crowd_common::LEN_M / 2.0, fp.y_min, 0.0);
    let box_max = vec3<f32>(crowd_common::LEN_M / 2.0, fp.y_max, crowd_common::DEPTH_M);
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

    // Column surge, evaluated once at entry: inside the 30° design cone
    // the ray's x drift through the box is ~1 m, far less than the surge
    // wavelength.
    let s = fp.spacing;
    let p0 = ro + rd * t0;
    let surge = crowd_common::front_z(p0.x, t) - crowd_common::front_z(p0.x, 0.0);
    let z0 = p0.z - surge;

    var cell = vec2<i32>(
        i32(floor((p0.x + crowd_common::LEN_M / 2.0) / s)),
        i32(floor(z0 / s)),
    );
    let step = vec2<i32>(select(-1, 1, rd.x >= 0.0), select(-1, 1, rd.z >= 0.0));
    var tdx = ((f32(cell.x) + select(0.0, 1.0, rd.x >= 0.0)) * s - crowd_common::LEN_M / 2.0 - p0.x) / rd.x;
    var tdz = ((f32(cell.y) + select(0.0, 1.0, rd.z >= 0.0)) * s - z0) / rd.z;
    let dtx = abs(s / rd.x);
    let dtz = abs(s / rd.z);
    let cells_x = i32(crowd_common::LEN_M / s) + 2;
    let cells_z = i32(crowd_common::DEPTH_M / s) + 2;

    // The cone geometry bounds the through-box path to a handful of cells;
    // the cap only bites on degenerate near-horizontal rays (guarded above).
    for (var n = 0; n < 48; n = n + 1) {
        if (cell.x >= 0 && cell.x < cells_x && cell.y >= 0 && cell.y < cells_z) {
            if (soldier_exists(cell, fp)) {
                best.under_ranks = true;
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

    // Exit through the terrain under the ray (approximate — exit_xz only
    // feeds the shadow floor's hash mottling).
    let t_floor = clamp((crowd_common::terrain_h_rel(ro.xz) - ro.y) / rd.y, t0, t1);
    best.exit_xz = (ro + rd * t_floor).xz;
    return best;
}

@fragment
fn fragment(in: crowd_common::CrowdOut) -> @location(0) vec4<f32> {
    // Re-derive the same displacement the vertex stage applied, so the
    // ray anchors on the rendered (terrain-riding) surface.
    let lift = crowd_common::terrain_h_rel(in.crowd_position.xz)
        + crowd_common::swell_y(in.crowd_position.xz, globals.time);
    let ro = in.crowd_position + vec3<f32>(0.0, lift, 0.0);
    let rd = normalize(ro - in.camera_local);
    var fp: FieldParams;
    fp.spacing = sun_spacing.w;
    fp.team_seed = team.w;
    fp.density = params.x;
    fp.glint_rate = params.y;
    fp.glint_strength = params.z;
    fp.y_min = crowd_common::terrain_c.z;
    fp.y_max = crowd_common::terrain_c.w;
    fp.sun_l = sun_spacing.xyz;
    let tr = crowd_trace(ro, rd, globals.time, fp);

    // The box itself is invisible: rays that neither hit a soldier nor
    // passed beneath occupied cells fall through to the real terrain.
    // Alpha-masked cutout, so the crowd still draws in the opaque pass.
    if (tr.t < 0.0 && !tr.under_ranks && tr.spark <= 0.0) {
        discard;
    }

    // Light rig: warm sun, blue sky bounce, warm ground bounce. All in
    // local space — dots with the local-space ray/sun are invariant.
    let sun_col = vec3<f32>(1.0, 0.94, 0.86) * 3.0;
    let sky_amb = vec3<f32>(0.55, 0.63, 0.74);
    let ground_amb = vec3<f32>(0.34, 0.30, 0.24);
    let sun_l = fp.sun_l;
    let hv = normalize(-rd + sun_l);

    var color = vec3<f32>(0.0);
    if (tr.t > 0.0) {
        let s = tr.soldier;
        // Dress: team tunic strongly varied per soldier — at pixel scales
        // shading averages out, and per-soldier albedo is the variation
        // that survives; mixed toward leather; helmets read as steel.
        let tunic = team.rgb * (0.55 + 0.9 * s.tone);
        let leather = vec3<f32>(0.30, 0.22, 0.15) * (0.7 + 0.6 * s.tone);
        var albedo = mix(tunic, leather, 0.35 * s.metal);
        if (tr.is_head) {
            albedo = vec3<f32>(0.62, 0.65, 0.70) * (0.9 + 0.3 * s.tone);
        }

        let n = tr.normal;
        // Wrap diffuse: bodies in a packed rank never face a bare void —
        // neighbours fill them — so hard-term diffuse goes fully black on
        // the backlot azimuths and erases figure structure from oblique
        // views. The 0.25 wrap keeps walls readable when backlit.
        let diffuse = max(dot(n, sun_l) * 0.75 + 0.25, 0.0);
        let amb = mix(ground_amb, sky_amb, 0.5 + 0.5 * n.y);
        // Crowds occlude themselves: helmets and shoulders catch the sky,
        // legs sink into the mass — the per-soldier vertical gradient.
        let ao = mix(0.35, 1.15, smoothstep(0.0, 1.2, tr.hit_y))
            * select(0.95, 1.1, tr.is_head);
        color = albedo * (sun_col * diffuse + amb * 2.6 * ao);

        // Armour glint. The physical specular term only fires where the
        // half-vector can align (walls at some azimuths, helmet domes);
        // from near-nadir it never does, so a periodic per-soldier
        // emissive flash carries the shimmer at every angle in the cone —
        // each soldier sweeps through "catching the sun" on their own
        // phase. Spear tips add their own mirror spark on top.
        let metal = select(0.3, 1.0, tr.is_head) * (0.4 + 0.6 * s.metal);
        let flash = pow(
            max(sin(globals.time * fp.glint_rate + s.phase * 6.28318), 0.0),
            4.0,
        );
        let spec = pow(max(dot(n, hv), 0.0), 28.0) * metal;
        color = color + sun_col * spec * (0.5 + 7.0 * flash) * fp.glint_strength;
        let glint_amp = select(0.3 + 0.9 * s.metal, 1.6, tr.is_head);
        color = color + sun_col * flash * glint_amp * 1.2 * fp.glint_strength;
    } else {
        // Through the ranks: the trampled floor in the crowd's own shadow.
        // No direct sun here — rays that reach it passed under layers of
        // bodies (at oblique angles a ray crosses a whole rank height per
        // cell, so floor hits are normal even in dense ranks). Lit floor
        // here would read as bare ground between clumps instead of a mass
        // of men over dark interior. Faintly team-tinted, hash-textured.
        let g = crowd_common::hash21(tr.exit_xz * 3.7);
        let ground = mix(vec3<f32>(0.20, 0.16, 0.12), vec3<f32>(0.32, 0.26, 0.18), g);
        let tinted = mix(ground, team.rgb, 0.45);
        color = tinted * ground_amb * (0.55 + 0.5 * g);
    }
    // The spear-tip spark survives even on gap rays (it floats above the
    // ranks), so it composites last.
    color = color + sun_col * tr.spark * 10.0;

    return vec4<f32>(color, 1.0);
}