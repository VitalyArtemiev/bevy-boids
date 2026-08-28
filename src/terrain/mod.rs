//! Terrain subsystem: the procedural voxel world and everything that
//! interacts with it — generation ([`noise`]), unit grounding
//! ([`grounding`]), camera clearance ([`camera`]), deformation brushes
//! ([`brush`]), and walkability derivation ([`walkability`]).

mod brush;
mod camera;
mod far;
mod grounding;
mod noise;
pub mod walkability;

pub use brush::{TerrainBrush, terrain_brush_system};
pub use camera::{CameraClearance, camera_terrain_clearance};
pub use far::{FarTerrain, far_terrain_stream};
pub use grounding::{GroundY, ground_boids};
pub use noise::{TerrainNoise, terrain_height};

use bevy::prelude::*;
use bevy_rts_camera::Ground;
use bevy_voxel_world::custom_meshing::CHUNK_SIZE_F;
use bevy_voxel_world::prelude::{
    LodLevel, TextureIndexMapperFn, VoxelLookupDelegate, VoxelWorldConfig, WorldVoxel,
};
use kinematics::{HardCollision, TrackedByTree};
use noise::{MATERIAL_BEDROCK, surface_material};
use std::collections::HashMap;
use std::sync::Arc;

use crate::kinematics;

/// Padded chunk dimensions per LOD level (interior voxels 32, 16, 8, 4, 2 m).
const LOD_PADDED_SHAPES: [u32; 5] = [34, 18, 10, 6, 4];

// ---------------------------------------------------------------------------
// bevy_voxel_world configuration
// ---------------------------------------------------------------------------

/// Marker config for our main (and currently only) voxel world.
#[derive(Resource, Default, Clone)]
pub struct MainWorld;

impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;
    /// Mark every chunk as camera ground so bevy_rts_camera raycasts (pan
    /// focus, drag) hit the actual terrain instead of a flat plane.
    type ChunkUserBundle = (Ground,);

    /// Chunks are 32 voxels; 40 chunks = 1280 m radius around the camera.
    /// Affordable only because distant chunks run at fat-voxel LODs (see
    /// `chunk_lod`/`chunk_data_shape`): only ~100 chunks near the camera
    /// are full-resolution, the rest shrink cubically with level.
    fn spawning_distance(&self) -> u32 {
        40
    }

    /// Keep a core radius spawned even when looking away (viewport culling
    /// otherwise despawns everything behind the camera between frames).
    fn min_despawn_distance(&self) -> u32 {
        8
    }

    /// Distance-based voxel LOD. The chunk grid stays 32 m regardless of
    /// level; higher levels store fewer, fatter voxels per chunk (see
    /// `chunk_data_shape`), so cost drops cubically while the near field
    /// reaches 1.3 km instead of 512 m — that is what pushes the
    /// heightfield far-field seam past the horizon of the blocky detail.
    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        _previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        const LOD0_RADIUS_M: f32 = 256.0;
        let centre = (chunk_position.as_vec3() + 0.5) * CHUNK_SIZE_F;
        let dist = centre.distance(camera_position);
        match dist {
            d if d < LOD0_RADIUS_M => 0,
            d if d < LOD0_RADIUS_M * 2.0 => 1,
            d if d < LOD0_RADIUS_M * 4.0 => 2,
            d if d < LOD0_RADIUS_M * 8.0 => 3,
            _ => 4,
        }
    }

    /// Padded voxel dimensions per LOD level: interior voxels halve per
    /// level (32, 16, 8, 4, 2 m voxels), so a level-k chunk costs (1/2^k)³
    /// of a full one. Must stay in sync with `chunk_meshing_shape`.
    fn chunk_data_shape(&self, lod_level: LodLevel) -> UVec3 {
        UVec3::splat(LOD_PADDED_SHAPES[lod_level as usize])
    }

    fn chunk_meshing_shape(&self, lod_level: LodLevel) -> UVec3 {
        UVec3::splat(LOD_PADDED_SHAPES[lod_level as usize])
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_chunk_pos, _lod, _previous| {
            let noise = TerrainNoise::default();
            // Column heights are pure in (x, z): cache per chunk so the
            // noise cascade runs once per column, not once per voxel.
            let mut height_cache: HashMap<(i32, i32), i32> = HashMap::new();
            Box::new(move |pos: IVec3, _previous| {
                let h = *height_cache
                    .entry((pos.x, pos.z))
                    .or_insert_with(|| noise.height(pos.x as f32, pos.z as f32).floor() as i32);
                if pos.y <= noise::BEDROCK_TOP_Y {
                    WorldVoxel::Solid(MATERIAL_BEDROCK)
                } else if pos.y < h {
                    WorldVoxel::Solid(surface_material(h))
                } else {
                    WorldVoxel::Air
                }
            })
        })
    }

    fn texture_index_mapper(&self) -> TextureIndexMapperFn<Self::MaterialIndex> {
        // The bundled fallback texture is a 4x4 palette; map materials onto
        // the first row until real terrain texturing lands.
        Arc::new(|material| [(material % 4) as u32; 3])
    }
}

// ---------------------------------------------------------------------------
// Obstacles
// ---------------------------------------------------------------------------

#[derive(Component, Default)]
pub struct Obstacle {
    pub(crate) normal: Vec3,
}

#[derive(Bundle, Default)]
pub struct ObstacleBundle {
    obstacle: Obstacle,
    mesh: Mesh3d,
    material: MeshMaterial3d<StandardMaterial>,
    transform: Transform,
    hard_collision: HardCollision,
    tracked: TrackedByTree,
}

impl ObstacleBundle {
    pub(crate) fn new(
        mesh: Handle<Mesh>,
        material: Handle<StandardMaterial>,
        normal: Vec3,
        pos: Vec3,
    ) -> Self {
        ObstacleBundle {
            obstacle: Obstacle { normal },
            hard_collision: Default::default(),
            tracked: Default::default(),
            mesh: Mesh3d(mesh),
            material: MeshMaterial3d(material),
            transform: Transform::from_xyz(pos.x, pos.y, pos.z),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_lod_rises_with_distance_from_the_camera() {
        let world = MainWorld;
        let camera = Vec3::ZERO;
        // Chunk (0,0,0) spans 0..32 m: its centre is 16 m from the origin.
        assert_eq!(world.chunk_lod(IVec3::ZERO, None, camera), 0);
        // ~350 m away (chunk centre at 11 * 32 + 16).
        assert_eq!(world.chunk_lod(IVec3::new(11, 0, 0), None, camera), 1);
        // ~1.4 km out.
        assert_eq!(world.chunk_lod(IVec3::new(44, 0, 0), None, camera), 3);
        // Beyond 2 km: coarsest.
        assert_eq!(world.chunk_lod(IVec3::new(80, 0, 0), None, camera), 4);
    }

    #[test]
    fn lod_shapes_shrink_and_pad_is_two() {
        let world = MainWorld;
        let mut previous = u32::MAX;
        for lod in 0..5u8 {
            let shape = world.chunk_data_shape(lod).x;
            assert_eq!(world.chunk_meshing_shape(lod).x, shape);
            assert_eq!(world.chunk_data_shape(lod).y, shape);
            assert_eq!(world.chunk_data_shape(lod).z, shape);
            assert!(shape < previous, "shapes must strictly shrink with LOD");
            assert!((shape - 2).is_power_of_two(), "interior must halve cleanly");
            previous = shape;
        }
    }
}
