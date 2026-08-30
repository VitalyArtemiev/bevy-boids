//! Terrain subsystem: the heightfield is THE terrain.
//!
//! A single LOD system of streamed heightfield tiles covers everything
//! from 1 m cells at the camera to 65 km cells at continental distance
//! ([`tiles`]); [`noise`] generates the heights; [`grounding`] and
//! [`camera`] integrate boids and the RTS camera with the field.
//! Sparse voxel detail volumes (forts, overhangs) will attach on top in a
//! later milestone — the heightfield stays the LOD spine everywhere else.

mod camera;
mod grounding;
mod noise;
mod tiles;

pub use camera::{CameraClearance, camera_terrain_clearance};
pub use grounding::{GroundY, ground_boids, reset_ground_caches};
pub use noise::{TerrainNoise, TerrainTuning};
pub use tiles::{TerrainTiles, stream_terrain_tiles};

use bevy::prelude::*;
use std::sync::Arc;

/// The authoritative terrain height function. Everything samples this one
/// field — tile meshes, grounding, camera clearance, cursor rays — so the
/// world stays self-consistent no matter which subsystem asks. The edit
/// layer (later milestone) will wrap the base sampler with sparse height
/// deltas instead of replacing it.
///
/// Backed by a boxed closure so tests (and later, edited worlds) can
/// substitute synthetic fields. Rebuilt from [`TerrainTuning`] by
/// [`rebuild_height_field`] whenever the tuning changes, which cascades
/// into tile rebuilds and cache resets via change detection.
#[derive(Resource, Clone)]
pub struct HeightField {
    sampler: Arc<dyn Fn(f32, f32) -> f32 + Send + Sync>,
}

impl HeightField {
    pub fn from_noise(noise: TerrainNoise) -> Self {
        HeightField {
            sampler: Arc::new(move |x, z| noise.height(x, z)),
        }
    }

    /// Height in metres at world (x, z). Cheap: one noise-cascade sample.
    pub fn height(&self, x: f32, z: f32) -> f32 {
        (self.sampler)(x, z)
    }
}

impl Default for HeightField {
    fn default() -> Self {
        HeightField::from_noise(TerrainNoise::default())
    }
}

impl HeightField {
    /// Test helper: a synthetic field from a plain function.
    pub fn from_fn(f: impl Fn(f32, f32) -> f32 + Send + Sync + 'static) -> Self {
        HeightField {
            sampler: Arc::new(f),
        }
    }
}

/// Rebuild the height field when the tuning resource changes. Running
/// before `stream_terrain_tiles`, its change tick then cascades: tiles
/// despawn/respawn from the new field, grounding caches reset, obstacles
/// re-project.
pub fn rebuild_height_field(
    tuning: Res<TerrainTuning>,
    mut field: ResMut<HeightField>,
) {
    if !tuning.is_changed() {
        return;
    }
    *field = HeightField::from_noise(TerrainNoise::from_tuning(&tuning));
}

// ---------------------------------------------------------------------------
// Obstacles
// ---------------------------------------------------------------------------

use crate::kinematics::{HardCollision, TrackedByTree};

/// Cuboid obstacles are 1 m; their centre rides half a metre above the
/// surface. Shared with `project_obstacles_onto_field`.
pub(crate) const OBSTACLE_HALF_HEIGHT: f32 = 0.5;

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

/// Re-seat obstacles on the terrain after the field changed (tuning edits).
/// Runs every frame but pays nothing unless the field's change tick moved.
pub fn project_obstacles_onto_field(
    mut query: Query<(&mut Transform, &Obstacle)>,
    field: Res<HeightField>,
) {
    if !field.is_changed() {
        return;
    }
    for (mut transform, _obstacle) in &mut query {
        let t = transform.translation;
        transform.translation.y = field.height(t.x, t.z) + OBSTACLE_HALF_HEIGHT;
    }
}
