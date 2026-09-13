// Impostor normal bake: outputs the surface normal in the baking camera's
// view space (x right, y up, z toward the camera), so a runtime billboard
// shader lights the quad by transforming the light into that same basis.
// RGB = normal * 0.5 + 0.5; alpha = 1 on geometry (the coverage mask —
// the background clears to alpha 0 like the albedo bake).

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal_view = normalize(
        (view.view_from_world * vec4<f32>(normalize(in.world_normal), 0.0)).xyz,
    );
    return vec4<f32>(normal_view * 0.5 + 0.5, 1.0);
}
