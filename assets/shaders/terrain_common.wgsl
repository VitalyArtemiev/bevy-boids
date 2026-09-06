// Shared terrain displacement: bindings + the per-vertex transform from
// the shared flat grid to the baked tile surface. Imported as a module
// by terrain.wgsl and terrain_prepass.wgsl.
#define_import_path bevy_boids::terrain_common

// Per-tile material bindings. Indices 200+: they share the material
// bind group with StandardMaterial, whose own bindings stop well below
// (same convention as bevy_pbr's forward decal extension).
@group(#{MATERIAL_BIND_GROUP}) @binding(200) var<uniform> tile: TileUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(201) var height_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(202) var slope_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(203) var color_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(204) var coarse_texture: texture_2d<f32>;

struct TileUniform {
    // Tile origin (xy) and size/cell in metres (zw).
    origin_size: vec4<f32>,
    // Hole rectangle in texel units (min_x, max_x, min_z, max_z); an
    // empty rect is encoded as huge bounds (clamp is the identity).
    hole: vec4<f32>,
    // (level, skirt depth in metres, fade, unused). fade: 1 = this
    // level's geometry, 0 = the coarse representation (LOD morph).
    shading: vec4<f32>,
};

// Grid resolution of the shared mesh (33x33 verts over 32x32 quads).
const TILE_VERTS: i32 = 33;

struct TerrainVertex {
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    color: vec4<f32>,
};

// Displace one vertex of the shared grid onto the baked tile surface.
//
// `position` is tile-local: xz are texel coordinates, y = -1 flags a
// skirt vert (y = 0 is the grid). Heights, slopes and tints come from
// the bake textures; the LOD morph blends the fine height with the
// baked coarse downsample; the clipmap hole rectangle is projected away
// by clamping interior vertices onto its edge, which degenerates the
// interior quads the CPU bake used to cut from the index buffer (verts
// stay on their own texel's height, so the boundary strip keeps the
// right surface).
fn terrain_displace(position: vec3<f32>) -> TerrainVertex {
    let origin = tile.origin_size.xy;
    let cell = tile.origin_size.w;
    let skirt_depth = tile.shading.y;
    let fade = tile.shading.z;

    let texel = clamp(
        vec2<i32>(position.xz),
        vec2<i32>(0),
        vec2<i32>(TILE_VERTS - 1),
    );
    let height_fine = textureLoad(height_texture, texel, 0).r;
    let height_coarse = textureLoad(coarse_texture, texel, 0).r;
    var height = mix(height_coarse, height_fine, fade);

    let is_skirt = position.y < -0.5;
    if is_skirt {
        height = height - skirt_depth;
    }

    // Hole projection (identity for the empty rect's huge bounds).
    var local = position.xz;
    local.x = clamp(local.x, tile.hole.x, tile.hole.y);
    local.y = clamp(local.y, tile.hole.z, tile.hole.w);

    var world_normal = vec3<f32>(0.0, 1.0, 0.0);
    if !is_skirt {
        let slope = textureLoad(slope_texture, texel, 0).xy;
        world_normal = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
    }

    return TerrainVertex(
        vec4<f32>(origin.x + local.x * cell, height, origin.y + local.y * cell, 1.0),
        world_normal,
        textureLoad(color_texture, texel, 0),
    );
}
