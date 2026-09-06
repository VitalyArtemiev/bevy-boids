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
/// exactly: uniform 200, textures 201-204. The textures are the shared
/// [`TileAtlas`] arrays (one layer per live or cached tile); the layer
/// index rides the uniform (`shading.w`).
#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct TerrainExtension {
    #[texture(201, dimension = "2d_array")]
    pub(crate) height: Handle<Image>,
    #[texture(202, dimension = "2d_array")]
    pub(crate) slope: Handle<Image>,
    #[texture(203, dimension = "2d_array")]
    pub(crate) color: Handle<Image>,
    #[uniform(200, TileUniform)]
    pub(crate) tile: TileUniform,
}

/// Everything the vertex shader needs to place and cut one tile.
/// `hole` is in tile-local TEXEL units; an empty rectangle (`min >
/// max`) means no cut.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct TileUniform {
    /// Tile origin (xy) and size/cell in metres (zw).
    pub origin_size: Vec4,
    /// Hole rectangle in texel units: (min_x, max_x, min_z, max_z).
    pub hole: Vec4,
    /// (level, skirt depth in metres, unused, atlas layer).
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

/// How many tile layers each atlas array holds. Live tiles, tiles
/// waiting out the retire rule, and the [`super::tiles::TileRenderCache`]
/// LRU all hold layers — during a zoom gesture the retiring and incoming
/// sets coexist, so the bound must cover roughly twice the desired set
/// plus the cache. `alloc` still frees cache entries under pressure
/// before giving up.
pub(crate) const ATLAS_LAYERS: u32 = 2048;

/// The four bake textures of every tile, packed layer-wise into four
/// shared `Texture2DArray`s — one material bind group's worth of texture
/// bindings no matter how many tiles are on screen, instead of four
/// dedicated images per tile.
///
/// Layers are handed out by [`TileAtlas::alloc`] when a tile bakes and
/// returned by [`TileAtlas::release`] when its render is evicted from
/// the cache. Writing a bake patches the layer's texels inside the
/// array `Image`s; the asset change re-uploads through `COPY_DST`
/// (batched once per frame per array by the renderer, so a wave of
/// spawns costs one upload, not one per tile).
#[derive(Resource)]
pub struct TileAtlas {
    /// R32Float — final vertex heights (drop + stitching baked in).
    pub(crate) height: Handle<Image>,
    /// Rg32Float — 3×3-blurred analytic slopes.
    pub(crate) slope: Handle<Image>,
    /// Rgba8Unorm — per-vertex ground tints.
    pub(crate) color: Handle<Image>,
    /// Returned layers, most-freed-first.
    free: Vec<u32>,
    /// Next never-allocated layer.
    next: u32,
}

impl FromWorld for TileAtlas {
    fn from_world(world: &mut World) -> Self {
        let mut images = world.resource_mut::<Assets<Image>>();
        let mut array = |format: TextureFormat, texel: usize| {
            let dimension = TILE_VERTS as u32;
            images.add(Image {
                // `write_texture` takes unpadded rows; layers are
                // consecutive images in one buffer.
                data: Some(vec![0u8; ATLAS_LAYERS as usize * dimension as usize * dimension as usize * texel]),
                texture_descriptor: TextureDescriptor {
                    label: Some("terrain_tile_atlas"),
                    size: Extent3d {
                        width: dimension,
                        height: dimension,
                        depth_or_array_layers: ATLAS_LAYERS,
                    },
                    dimension: TextureDimension::D2,
                    format,
                    mip_level_count: 1,
                    sample_count: 1,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                ..Default::default()
            })
        };
        TileAtlas {
            height: array(TextureFormat::R32Float, 4),
            slope: array(TextureFormat::Rg32Float, 8),
            color: array(TextureFormat::Rgba8Unorm, 4),
            free: Vec::new(),
            next: 0,
        }
    }
}

impl TileAtlas {
    /// Layers claimable without evicting anything.
    pub(crate) fn available(&self) -> u32 {
        ATLAS_LAYERS - self.next + self.free.len() as u32
    }

    /// Hand out a layer. Reuses freed layers first; fresh layers come
    /// from the never-allocated tail. When the tail runs out the caller
    /// is expected to have evicted cache renders (see
    /// [`TileRenderCache::evict_oldest`]) — allocating past the bound is
    /// a bug and asserts.
    pub(crate) fn alloc(&mut self) -> u32 {
        if let Some(slot) = self.free.pop() {
            return slot;
        }
        assert!(
            self.next < ATLAS_LAYERS,
            "tile atlas exhausted: live tiles exceed {ATLAS_LAYERS} layers"
        );
        let slot = self.next;
        self.next += 1;
        slot
    }

    /// Take a layer back. The texels stay until the slot is rewritten.
    pub(crate) fn release(&mut self, slot: u32) {
        self.free.push(slot);
    }

    /// Layers ever handed out — the high-water mark. Constant across
    /// world rebuilds if (and only if) layers are actually recycled;
    /// pinned by the tile-atlas test.
    pub(crate) fn high_water(&self) -> u32 {
        self.next
    }

    /// Patch one tile's four bakes into its layer. `bake` data layouts
    /// match the arrays texel-for-texel (`bake_texture_*` in this
    /// module), so the splice is a plain byte copy per array.
    pub(crate) fn write(
        &self,
        images: &mut Assets<Image>,
        slot: u32,
        bake: &super::tiles::BakedTile,
    ) {
        let dimension = TILE_VERTS as usize;
        let layer = dimension * dimension;
        let mut splice = |handle: &Handle<Image>, texels: &Image| {
            let image = &mut *images.get_mut(handle).expect("atlas array asset");
            let format = image.texture_descriptor.format;
            let bytes = image.data.as_mut().expect("atlas array data");
            let src = texels.data.as_ref().expect("bake data");
            assert_eq!(src.len(), layer * bytes_per_texel(&format));
            let at = slot as usize * layer * bytes_per_texel(&format);
            bytes[at..at + src.len()].copy_from_slice(src);
        };
        splice(&self.height, &bake.height);
        splice(&self.slope, &bake.slope);
        splice(&self.color, &bake.color);
    }
}

/// Texel size in bytes of the atlas formats (compressed formats need
/// not apply).
fn bytes_per_texel(format: &TextureFormat) -> usize {
    match format {
        TextureFormat::R32Float | TextureFormat::Rgba8Unorm => 4,
        TextureFormat::Rg32Float => 8,
        _ => unreachable!("unexpected atlas texture format {format:?}"),
    }
}

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
/// coordinates — no skirt ring: rim stitching sews each fine rim onto
/// the coarse chord with an overlap bias (`STITCH_BIAS_CELL_FRACTION`),
/// so level seams are watertight by construction and hanging walls
/// would only show as coincident z-fighting polygons. Normals and
/// colors here are placeholders purely to keep the standard pipeline's
/// vertex layout; the shader replaces both per tile. `COLOR_0` exists
/// to arm the `VERTEX_COLORS` define so `StandardMaterial` multiplies
/// in the per-vertex tint the shader writes.
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
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(TILE_VERTS * TILE_VERTS);
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(positions.capacity());
        for iz in 0..TILE_VERTS {
            for ix in 0..TILE_VERTS {
                positions.push([ix as f32, 0.0, iz as f32]);
                uvs.push([ix as f32, iz as f32]);
            }
        }

        let mut indices: Vec<u32> = Vec::with_capacity(TILE_QUADS * TILE_QUADS * 6);
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
            VertexAttributeValues::Float32x3(vec![[0.0, 1.0, 0.0]; TILE_VERTS * TILE_VERTS]),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_COLOR,
            VertexAttributeValues::Float32x4(vec![[1.0, 1.0, 1.0, 1.0]; TILE_VERTS * TILE_VERTS]),
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

