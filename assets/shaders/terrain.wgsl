// Terrain forward vertex stage: the standard bevy_pbr mesh vertex
// shader with the position, normal and color displaced onto the baked
// tile surface (see terrain_common.wgsl). Everything else — lighting,
// shadows via the prepass variant, fog, bloom — stays StandardMaterial.
#import bevy_boids::terrain_common

#import bevy_pbr::{
    mesh_bindings::mesh,
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
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

#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        terrain.world_normal,
        vertex_no_morph.instance_index,
    );
#endif

#ifdef VERTEX_POSITIONS
    out.world_position = mesh_functions::mesh_position_local_to_world(
        mesh_world_from_local,
        terrain.world_position,
    );
    out.position = position_world_to_clip(out.world_position.xyz);
#endif

#ifdef VERTEX_UVS_A
    out.uv = vertex_no_morph.uv;
#endif

#ifdef VERTEX_COLORS
    out.color = terrain.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex_no_morph.instance_index;
#endif

    return out;
}
