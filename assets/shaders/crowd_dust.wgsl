// Crowd shell, translucent pass: the dust volume riding above each
// formation box. Same local space as crowd.wgsl; the box's own footprint
// with a lower, taller profile. Density is kicked up along the fluid
// front (where the ranks churn) and settles with height; light forward-
// scatters, so dust between the camera and the sun glows. Alpha-blended,
// no depth write, back-to-front by the transparent pass.

#import bevy_boids::crowd_common

#import bevy_pbr::{
    mesh_functions,
    forward_io::Vertex,
    view_transformations::position_world_to_clip,
}
#import bevy_pbr::mesh_view_bindings::{view, globals}
#import bevy_render::maths

// x: density scale, y: dust box height (m), z: unused, w: unused.
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: vec4<f32>;
// xyz: direction TO the sun in local space, w: unused.
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> sun: vec4<f32>;

@vertex
fn vertex(v: Vertex) -> bevy_boids::crowd_common::CrowdOut {
    let world_from_local = mesh_functions::get_world_from_local(v.instance_index);
    let world = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(v.position, 1.0),
    );
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
    let ro = in.crowd_position;
    let t = globals.time;

    // Clip the ray to the dust box.
    let box_min = vec3<f32>(-crowd_common::LEN_M / 2.0, 0.0, 0.0);
    let box_max = vec3<f32>(crowd_common::LEN_M / 2.0, params.y, crowd_common::DEPTH_M);
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
        return vec4<f32>(vec3<f32>(0.0), 0.0);
    }

    // A few samples along the through-box segment: agitation peaks along
    // the fluid front line, density settles exponentially with height and
    // breaks up with drifting fbm.
    let wind = vec2<f32>(t * 0.35, t * 0.22);
    var acc = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let u = (f32(i) + 0.5) / 5.0;
        let p = ro + rd * mix(t0, t1, u);
        let agitation = exp(-abs(p.z - crowd_common::front_z(p.x, t)) / 5.0);
        let settle = exp(-p.y / 0.9);
        let drift = 0.35 + 0.65 * crowd_common::fbm2(p.xz * 0.09 + wind);
        acc = acc + agitation * settle * drift;
    }
    let density = params.x * acc / 5.0;
    let alpha = clamp(1.0 - exp(-density * (t1 - t0) * 1.6), 0.0, 0.5);

    // Forward scattering: dust on the camera-sun line glows warm.
    let fs = pow(max(dot(rd, sun.xyz), 0.0), 3.0);
    let dust_col = vec3<f32>(0.74, 0.66, 0.54) * (0.30 + 0.9 * fs) * 1.6;
    return vec4<f32>(dust_col, alpha);
}
