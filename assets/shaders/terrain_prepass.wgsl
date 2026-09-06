// Terrain prepass vertex stage: same displacement as terrain.wgsl so
// the depth/normal prepass — and therefore shadows — sees the displaced
// surface, not the flat shared grid. The terrain is static, so the
// motion-vector output uses the displaced position for both frames.
#import bevy_boids::terrain_common

#import bevy_pbr::{
    mesh_bindings::mesh,
    mesh_functions,
    prepass_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
}

@vertex
fn vertex(vertex_no_morph: Vertex) -> VertexOutput {
    var out: VertexOutput;

    let terrain = bevy_boids::terrain_common::terrain_displace(
        vertex_no_morph.position,
    );

    let mesh_world_from_local =
        mesh_functions::get_world_from_local(vertex_no_morph.instance_index);

    out.world_position = mesh_functions::mesh_position_local_to_world(
        mesh_world_from_local,
        terrain.world_position,
    );
    out.position = position_world_to_clip(out.world_position.xyz);

#ifdef VERTEX_UVS_A
    out.uv = vertex_no_morph.uv;
#endif

#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        terrain.world_normal,
        vertex_no_morph.instance_index,
    );
#endif
#endif

#ifdef VERTEX_COLORS
    out.color = terrain.color;
#endif

#ifdef MOTION_VECTOR_PREPASS
    // Static terrain: previous and current positions coincide.
    out.previous_world_position = out.world_position;
#endif

    return out;
}
