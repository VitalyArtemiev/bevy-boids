//! GPU-side terrain rendering: the UDLOD-style displaced-mesh path.
//!
//! All tiles share ONE mesh — a flat 33×33 grid plus skirt ring in
//! tile-local texel coordinates — and every per-tile fact lives in small
//! textures baked on the CPU ([`crate::terrain::tiles::bake_tile`]):
//! final vertex heights (level drop and rim stitching included, so the
//! displaced geometry is what `tile_mesh` used to build), the 3×3-blurred
//! analytic slopes (normals), the per-vertex ground tint (the palette
//! logic, ring-edge morph included), and a coarse downsample for the
//! LOD-transition morph. The vertex shader displaces, projects the
//! clipmap hole rectangle away (interior vertices snap onto the
//! rectangle's edge, degenerating exactly the quads the CPU mesh used to
//! cut from the index buffer), and morphs between the fine and coarse
//! heights while a tile fades in or out — the cross-fade that replaces
//! both the LOD pop and the despawn hole.
//!
//! The material is an [`ExtendedMaterial`] over [`StandardMaterial`]:
//! lighting, shadows, the prepass and bloom all stay stock; only the
//! vertex stage (forward AND prepass — the prepass must see the same
//! displaced geometry or shadows render the flat grid) is ours.
//!
//! Gameplay never reads these textures: boid grounding, camera
//! clearance and cursor picking sample the analytic [`HeightField`]
//! (see `crate::terrain`), which the bake itself uses — render and
//! simulation agree by construction, as they always have.

use bevy::asset::{Asset, Handle};
use bevy::math::Vec4;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy::prelude::*;
use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};
use bevy::render::render_resource::{
    AsBindGroup, AsBindGroupShaderType, Extent3d, ShaderType, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages,
};
use bevy::shader::ShaderRef;

use super::tiles::{TILE_QUADS, TILE_VERTS};

/// The terrain material: `StandardMaterial` (white, fully lit) extended
/// with the per-tile textures and placement uniform.
pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExtension>;

/// Per-tile data bound straight into the vertex shaders. Bind indices
/// 200+: they share the material bind group with `StandardMaterial`,
/// whose own bindings stop well below (the in-tree forward decal uses
/// the same 200+ convention). The indices must match `terrain_common.wgsl`
/// exactly: uniform 200, textures 201-204.
#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct TerrainExtension {
    #[texture(201, dimension = "2d")]
    pub(crate) height: Handle<Image>,
    #[texture(202, dimension = "2d")]
    pub(crate) slope: Handle<Image>,
    #[texture(203, dimension = "2d")]
    pub(crate) color: Handle<Image>,
    #[texture(204, dimension = "2d")]
    pub(crate) coarse: Handle<Image>,
    #[uniform(200, TileUniform)]
    pub(crate) tile: TileUniform,
}

/// Everything the vertex shader needs to place, cut and morph one tile.
/// `hole` is in tile-local TEXEL units; an empty rectangle (`min >
/// max`) means no cut. `fade` is the LOD morph: 1 = fully this level's
/// geometry, 0 = fully the coarse representation.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct TileUniform {
    /// Tile origin (xy) and size/cell in metres (zw).
    pub origin_size: Vec4,
    /// Hole rectangle in texel units: (min_x, max_x, min_z, max_z).
    pub hole: Vec4,
    /// (level, skirt depth in metres, fade, unused).
    pub shading: Vec4,
}

/// The uniform the CPU struct converts to (identity today, but the
/// `AsBindGroupShaderType` impl is where any packing would live).
impl AsBindGroupShaderType<TileUniform> for TerrainExtension {
    fn as_bind_group_shader_type(&self, _images: &bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>) -> TileUniform {
        self.tile
    }
}

/// The displacement shaders: the forward and prepass vertex stages get
/// ours (the prepass must see the same displaced geometry or shadows
/// render the flat grid); the fragment stage stays `StandardMaterial`'s
/// stock lighting, tinted by the per-vertex color our vertex stage
/// writes under `VERTEX_COLORS`.
impl MaterialExtension for TerrainExtension {
    fn vertex_shader() -> ShaderRef {
        "shaders/terrain.wgsl".into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        "shaders/terrain_prepass.wgsl".into()
    }
}

/// Keeps the shared displacement module loaded. `terrain.wgsl` and
/// `terrain_prepass.wgsl` import it BY NAMESPACE (`#import
/// bevy_boids::terrain_common`), and naga-oil resolves namespace imports
/// only against shader assets that are actually loaded — no path
/// references this file, so without a load the terrain pipelines silently
/// never specialize and no tile ever draws.
#[derive(Resource)]
pub(crate) struct TerrainCommonShader(pub(crate) Handle<Shader>);

/// Registers the terrain material's plugin.
#[derive(Default)]
pub struct TerrainRenderPlugin;

impl Plugin for TerrainRenderPlugin {
    fn build(&self, app: &mut App) {
        let common: Handle<Shader> = app
            .world()
            .resource::<AssetServer>()
            .load("shaders/terrain_common.wgsl");
        app.insert_resource(TerrainCommonShader(common))
            .add_plugins(MaterialPlugin::<TerrainMaterial>::default());
    }
}

/// The one mesh every tile renders: a flat 33×33 grid over texel
/// coordinates, plus a duplicated border ring one unit below (the skirt
/// flag — grid verts carry y = 0, skirt verts y = -1; the shader drops
/// skirt verts by the tile's skirt depth). Normals and colors here are
/// placeholders purely to keep the standard pipeline's vertex layout;
/// the shader replaces both per tile. `COLOR_0` exists to arm the
/// `VERTEX_COLORS` define so `StandardMaterial` multiplies in the
/// per-vertex tint the shader writes.
#[derive(Resource)]
pub struct SharedTileMesh(pub Handle<Mesh>);

impl FromWorld for SharedTileMesh {
    fn from_world(world: &mut World) -> Self {
        let mut mesh = Mesh::new(
            bevy::render::render_resource::PrimitiveTopology::TriangleList,
            Default::default(),
        );

        // Grid verts: position = (ix, 0, iz) in texel units; the shader
        // turns that into world XZ via the tile uniform.
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(TILE_VERTS * TILE_VERTS + 4 * TILE_VERTS);
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(positions.capacity());
        for iz in 0..TILE_VERTS {
            for ix in 0..TILE_VERTS {
                positions.push([ix as f32, 0.0, iz as f32]);
                uvs.push([ix as f32, iz as f32]);
            }
        }

        // Skirt ring, mirroring `tile_mesh`'s border walk: y = -1 flags
        // the vert; the shader samples the same texel as its border
        // partner and sinks it by the skirt depth.
        let border: Vec<usize> = (0..TILE_VERTS)
            .chain((1..TILE_VERTS).map(|i| i * TILE_VERTS + TILE_QUADS))
            .chain((0..TILE_QUADS).rev().map(|i| TILE_VERTS * TILE_QUADS + i))
            .chain((1..TILE_QUADS).rev().map(|i| i * TILE_VERTS))
            .collect();
        for &v in &border {
            let [x, _, z] = positions[v];
            positions.push([x, -1.0, z]);
            uvs.push(uvs[v]);
        }

        let mut indices: Vec<u32> =
            Vec::with_capacity(TILE_QUADS * TILE_QUADS * 6 + border.len() * 6);
        for iz in 0..TILE_QUADS {
            for ix in 0..TILE_QUADS {
                let v0 = (iz * TILE_VERTS + ix) as u32;
                // Winding so faces point UP (+y): the +z corner comes
                // second (see `tile_mesh`).
                indices.extend_from_slice(&[v0, v0 + TILE_VERTS as u32, v0 + 1]);
                indices.extend_from_slice(&[
                    v0 + 1,
                    v0 + TILE_VERTS as u32,
                    v0 + TILE_VERTS as u32 + 1,
                ]);
            }
        }
        let n = border.len();
        let skirt_start = (TILE_VERTS * TILE_VERTS) as u32;
        for i in 0..n {
            let top_a = border[i] as u32;
            let top_b = border[(i + 1) % n] as u32;
            let bot_a = skirt_start + i as u32;
            let bot_b = skirt_start + ((i + 1) % n) as u32;
            // Skirt walls face outward from the tile.
            indices.extend_from_slice(&[top_a, top_b, bot_a]);
            indices.extend_from_slice(&[top_b, bot_b, bot_a]);
        }

        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(positions),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            VertexAttributeValues::Float32x2(uvs),
        );
        // Placeholder attributes: layout only, replaced in-shader.
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(vec![[0.0, 1.0, 0.0]; skirt_start as usize + n]),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_COLOR,
            VertexAttributeValues::Float32x4(vec![[1.0, 1.0, 1.0, 1.0]; skirt_start as usize + n]),
        );
        mesh.insert_indices(Indices::U32(indices));

        let handle = world.resource_mut::<Assets<Mesh>>().add(mesh);
        SharedTileMesh(handle)
    }
}

// (No per-tile `Aabb`: the shared mesh is displaced in-shader, so its
// CPU-side AABB is one flat cell at the origin and every tile would
// cull with the same wrong box. Tiles spawn with `NoFrustumCulling`
// instead — see `tiles::stream_terrain_tiles`.)

/// Make a single-channel f32 "texture" of baked values. `TEXTURE_BINDING`
/// suffices — the shaders `textureLoad` (no filtering, no sampler).
pub fn bake_texture_f32(values: &[f32]) -> Image {
    let dimension = (values.len() as f32).sqrt() as u32;
    let mut data = Vec::with_capacity(values.len() * 4);
    for &v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    Image {
        data: Some(data),
        texture_descriptor: bevy::render::render_resource::TextureDescriptor {
            label: Some("terrain_bake_f32"),
            size: Extent3d {
                width: dimension,
                height: dimension,
                depth_or_array_layers: 1,
            },
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        ..Default::default()
    }
}

/// Two-channel f32 bake (slopes).
pub fn bake_texture_rg32f(values: &[[f32; 2]]) -> Image {
    let dimension = (values.len() as f32).sqrt() as u32;
    let mut data = Vec::with_capacity(values.len() * 8);
    for v in values {
        data.extend_from_slice(&v[0].to_le_bytes());
        data.extend_from_slice(&v[1].to_le_bytes());
    }
    Image {
        data: Some(data),
        texture_descriptor: bevy::render::render_resource::TextureDescriptor {
            label: Some("terrain_bake_rg32f"),
            size: Extent3d {
                width: dimension,
                height: dimension,
                depth_or_array_layers: 1,
            },
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg32Float,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        ..Default::default()
    }
}

/// Linear RGBA8 bake (vertex tints).
pub fn bake_texture_rgba8(values: &[[f32; 4]]) -> Image {
    let dimension = (values.len() as f32).sqrt() as u32;
    let mut data = Vec::with_capacity(values.len() * 4);
    for v in values {
        for c in v {
            data.push((c * 255.0).clamp(0.0, 255.0) as u8);
        }
    }
    Image {
        data: Some(data),
        texture_descriptor: TextureDescriptor {
            label: Some("terrain_bake_rgba8"),
            size: Extent3d {
                width: dimension,
                height: dimension,
                depth_or_array_layers: 1,
            },
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        ..Default::default()
    }
}

/// LOD-transition state per tile entity: the material's `fade` uniform
/// animates toward `target` (1 = this level's geometry, 0 = the coarse
/// representation). Spawned tiles fade in (the new surface slides up
/// from where the coarser ring was); dropped tiles fade out before
/// retiring (the surface settles onto the cover that lands beneath).
/// [`super::tiles::stream_terrain_tiles`] refuses to retire a fading
/// tile while its fade is still up.
#[derive(Component, Debug)]
pub struct TileFade {
    pub target: f32,
}

/// How long one LOD transition takes.
pub const TILE_FADE_SECS: f32 = 0.25;

/// Step every fading tile's material toward its target and drop the
/// marker once it arrives. Mutates material assets (not components), so
/// it touches nothing the streaming system holds — fades and streaming
/// interleave freely.
pub fn animate_tile_fades(
    tiles: Query<(Entity, &MeshMaterial3d<TerrainMaterial>, &TileFade)>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    time: Res<Time>,
    mut commands: Commands,
) {
    let step = time.delta_secs() / TILE_FADE_SECS;
    for (entity, material, fade) in &tiles {
        let Some(mut material) = materials.get_mut(&material.0) else {
            continue;
        };
        let current = &mut material.extension.tile.shading.z;
        let delta = fade.target - *current;
        if delta.abs() <= step {
            *current = fade.target;
            commands.entity(entity).remove::<TileFade>();
        } else {
            *current += delta.signum() * step;
        }
    }
}
