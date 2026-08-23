//! Boid grounding: keeping units on the terrain surface.

use super::MainWorld;
use crate::boid::Boid;
use crate::kinematics::Velocity;
use bevy::prelude::*;
use bevy_voxel_world::prelude::{VoxelWorld, WorldVoxel};

/// Terrain grounding state per boid: the surface height under the boid and
/// the voxel column it was computed for. The surface is rescanned only when
/// the boid enters a new voxel column; `bob` renders from it every frame.
#[derive(Component, Debug)]
pub struct GroundY {
    /// World-space height of the terrain surface (top face of the top solid
    /// voxel) under the boid's current column.
    pub surface: f32,
    /// Column the surface was computed for; sentinel `i32::MIN` forces a
    /// wide first scan because the spawn height can be far off terrain.
    col: (i32, i32),
}

impl Default for GroundY {
    fn default() -> Self {
        GroundY {
            surface: 0.0,
            col: (i32::MIN, i32::MIN),
        }
    }
}

/// Scan a voxel column downward for the first solid voxel with air above it.
/// Returns the world-space surface height. `Unset` counts as air: chunks
/// stream in over several frames, so the world under a boid may legitimately
/// not exist yet.
pub(crate) fn find_surface<I>(
    get_voxel: &dyn Fn(IVec3) -> WorldVoxel<I>,
    x: i32,
    z: i32,
    top_y: i32,
    bottom_y: i32,
) -> Option<f32> {
    let mut y = top_y;
    while y >= bottom_y {
        let below = get_voxel(IVec3::new(x, y, z));
        if !matches!(below, WorldVoxel::Air | WorldVoxel::Unset) {
            let above = get_voxel(IVec3::new(x, y + 1, z));
            if matches!(above, WorldVoxel::Air | WorldVoxel::Unset) {
                return Some((y + 1) as f32);
            }
        }
        y -= 1;
    }
    None
}

/// Keep every boid on the terrain surface. Runs after `move_step` (fresh
/// positions) and before `bob` (which renders the y). Vertical velocity is
/// cancelled: steering targets the y boids spawned at, and `bob` overwrites
/// the rendered y anyway — without the reset both would accumulate against
/// the terrain-following height.
pub fn ground_boids(
    mut query: Query<(&Transform, &mut GroundY, &mut Velocity), With<Boid>>,
    voxel_world: VoxelWorld<MainWorld>,
) {
    let get_voxel = voxel_world.get_voxel_fn();
    query.par_iter_mut().for_each(|(transform, mut ground, mut vel)| {
        let t = transform.translation;
        let col = (t.x.floor() as i32, t.z.floor() as i32);
        if col != ground.col {
            let (top, bottom) = if ground.col.0 == i32::MIN {
                ((t.y + 64.0) as i32, (t.y - 64.0) as i32)
            } else {
                ((t.y + 8.0) as i32, (t.y - 16.0) as i32)
            };
            if let Some(surface) = find_surface(&*get_voxel, col.0, col.1, top, bottom) {
                ground.surface = surface;
            }
            // Keep the last surface when nothing was found: the chunk is
            // probably still generating, better to walk on stale ground
            // than to fall.
            ground.col = col;
        }
        vel.v.y = 0.0;
        vel.a.y = 0.0;
    });
}
