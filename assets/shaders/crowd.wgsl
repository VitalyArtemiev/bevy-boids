// Crowd shell, opaque pass: renders a mass of soldiers inside the
// deformed formation box (see crowd_common.wgsl for the field and trace).
// Vertex stage swells the box; fragment stage traces the view ray through
// the soldier grid and shades the nearest hit — helmet/shoulder disks from
// nadir, cylinder walls at the cone rim — plus periodic armour and
// spear-tip glints, and a dark trampled floor through the gaps.

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
// w: box height the trace clips to.
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> params: vec4<f32>;
// xyz: direction TO the sun in local space, w: soldier spacing (m).
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> sun_spacing: vec4<f32>;

@vertex
fn vertex(v: Vertex) -> bevy_boids::crowd_common::CrowdOut {
    let world_from_local = mesh_functions::get_world_from_local(v.instance_index);
    var local = v.position;
    local.y += crowd_common::swell_y(local.xz, globals.time);
    let world = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(local, 1.0),
    );
    // Rigid (rotation + translation) inverse: rotate the camera into local
    // space with the transposed basis. Entity transforms here are rigid.
    let basis = maths::mat4x4_to_mat3x3(world_from_local);
    let cam_local = transpose(basis) * (view.world_position - world_from_local[3].xyz);

    var out: crowd_common::CrowdOut;
    out.position = position_world_to_clip(world.xyz);
    out.crowd_position = v.position;
    out.camera_local = cam_local;
    return out;
}

@fragment
fn fragment(in: CrowdOut) -> @location(0) vec4<f32> {
    let rd = normalize(in.crowd_position - in.camera_local);
    var fp = crowd_common::FieldParams(
        sun_spacing.w,
        team.w,
        params.x,
        params.y,
        params.z,
        params.w,
        sun_spacing.xyz,
    );
    let tr = crowd_common::crowd_trace(in.crowd_position, rd, globals.time, fp);

    // Light rig: warm sun, blue sky bounce, warm ground bounce. All in
    // local space — dots with the local-space ray/sun are invariant.
    let sun_col = vec3<f32>(1.0, 0.94, 0.86) * 2.6;
    let sky_amb = vec3<f32>(0.55, 0.63, 0.74);
    let ground_amb = vec3<f32>(0.34, 0.30, 0.24);
    let sun_l = fp.sun_l;
    let hv = normalize(-rd + sun_l);

    var color = vec3<f32>(0.0);
    if (tr.t > 0.0) {
        let s = tr.soldier;
        // Dress: team tunic darkened/lightened per soldier, mixed toward
        // leather; helmets read as steel.
        let tunic = team.rgb * (0.85 + 0.5 * s.tone);
        let leather = vec3<f32>(0.30, 0.22, 0.15) * (0.9 + 0.4 * s.tone);
        var albedo = mix(tunic, leather, 0.35 * s.metal);
        if (tr.is_head) {
            albedo = vec3<f32>(0.48, 0.50, 0.54) * (0.85 + 0.4 * s.tone);
        }

        let n = tr.normal;
        let diffuse = max(dot(n, sun_l), 0.0);
        let amb = mix(ground_amb, sky_amb, 0.5 + 0.5 * n.y);
        // Crowds occlude themselves: helmets and shoulders catch the sky,
        // bodies sink into the mass.
        let ao = mix(0.45, 1.0, smoothstep(0.0, 1.2, tr.hit_y))
            * select(0.9, 1.0, tr.is_head);
        color = albedo * (sun_col * diffuse + amb * 2.2 * ao);

        // Armour glint: broad specular on metal, punched up by a periodic
        // per-soldier flash as their pose sweeps through the mirror
        // direction. Spear tips add their own mirror spark on top.
        let metal = select(0.3, 1.0, tr.is_head) * (0.4 + 0.6 * s.metal);
        let flash = pow(
            max(sin(globals.time * fp.glint_rate + s.phase * 6.28318), 0.0),
            12.0,
        );
        let spec = pow(max(dot(n, hv), 0.0), 28.0) * metal;
        color = color + sun_col * spec * (0.5 + 5.0 * flash) * fp.glint_strength;
    } else {
        // Through the ranks: the trampled, deeply shadowed floor.
        let g = crowd_common::hash21(tr.exit_xz * 3.7);
        let ground = mix(vec3<f32>(0.16, 0.14, 0.11), vec3<f32>(0.24, 0.20, 0.13), g);
        color = ground * (ground_amb * 1.6 + sun_col * max(sun_l.y, 0.0) * 0.12);
    }
    // The spear-tip spark survives even on gap rays (it floats above the
    // ranks), so it composites last.
    color = color + sun_col * tr.spark * 7.0;

    return vec4<f32>(color, 1.0);
}
