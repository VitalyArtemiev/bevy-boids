//! Terrain subsystem: the procedural voxel world and everything that
//! interacts with it — generation ([`noise`]), unit grounding
//! ([`grounding`]), camera clearance ([`camera`]), deformation brushes
//! ([`brush`]), and walkability derivation ([`walkability`]).

mod brush;
mod camera;
mod grounding;
mod noise;
pub mod walkability;

pub use brush::{TerrainBrush, terrain_brush_system};
pub use camera::{CameraClearance, camera_terrain_clearance};
pub use grounding::{GroundY, ground_boids};
pub use noise::{TerrainNoise, terrain_height};

use bevy::prelude::*;
use bevy_rts_camera::Ground;
use bevy_voxel_world::prelude::{
    TextureIndexMapperFn, VoxelLookupDelegate, VoxelWorldConfig, WorldVoxel,
};
use kinematics::{HardCollision, TrackedByTree};
use noise::{MATERIAL_BEDROCK, surface_material};
use std::collections::HashMap;
use std::sync::Arc;

use crate::kinematics;

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

    /// Chunks are 32 voxels; 16 chunks = 512 m radius around the camera.
    fn spawning_distance(&self) -> u32 {
        16
    }

    /// Keep a core radius spawned even when looking away (viewport culling
    /// otherwise despawns everything behind the camera between frames).
    fn min_despawn_distance(&self) -> u32 {
        4
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
