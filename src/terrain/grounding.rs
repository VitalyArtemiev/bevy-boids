//! Boid grounding: keeping units on the terrain surface.

use super::HeightField;
use crate::boid::Boid;
use crate::kinematics::Velocity;
use bevy::prelude::*;

/// Terrain grounding state per boid: the surface height under the boid and
/// the voxel-metre column it was computed for. The surface is resampled
/// only when the boid crosses into a new metre column (a noise-cascade
/// sample is cheap, but not free for 10k boids every frame); `bob` renders
/// from it every frame.
#[derive(Component, Debug)]
pub struct GroundY {
    /// World-space terrain height under the boid's current column.
    pub surface: f32,
    /// Column the surface was computed for; sentinel `i32::MIN` forces a
    /// first sample.
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

/// Keep every boid on the terrain surface. Runs after `move_step` (fresh
/// positions) and before `bob` (which renders the y). Vertical velocity is
/// cancelled: steering targets the y boids spawned at, and `bob` overwrites
/// the rendered y anyway — without the reset both would accumulate against
/// the terrain-following height.
pub fn ground_boids(
    mut query: Query<(&Transform, &mut GroundY, &mut Velocity), With<Boid>>,
    field: Res<HeightField>,
) {
    query.par_iter_mut().for_each(|(transform, mut ground, mut vel)| {
        let t = transform.translation;
        let col = (t.x.floor() as i32, t.z.floor() as i32);
        if col != ground.col {
            ground.surface = field.height(t.x, t.z);
            ground.col = col;
        }
        vel.v.y = 0.0;
        vel.a.y = 0.0;
    });
}
