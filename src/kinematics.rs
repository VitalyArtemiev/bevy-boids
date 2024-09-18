use bevy::math::Vec3;
use bevy::prelude::*;
use bevy_spatial::kdtree::KDTree3;

#[derive(Component, Default)]
pub struct Velocity {
    pub v: Vec3,
    pub a: Vec3,
    ///Acceleration due to collisions
    pub push: Vec3,
    pub(crate) target_v: f32,
}

pub const MAX_VELOCITY: f32 = 20.0;
pub const BROWNIAN_VELOCITY: f32 = 0.02;
pub const MAX_ACCELERATION: f32 = 5.0;
pub const DECELERATION_TIME_SEC: f32 = 1.0;
pub const DECELERATION_TIME_SEC_SQUARED: f32 = DECELERATION_TIME_SEC * DECELERATION_TIME_SEC;

const DECELERATION_DISTANCE: f32 =
    MAX_VELOCITY * DECELERATION_TIME_SEC - MAX_ACCELERATION * (DECELERATION_TIME_SEC_SQUARED);

pub fn move_step(mut query: Query<(&mut Transform, &mut Velocity)>, time: Res<Time>) {
    for (mut transform, mut vel) in &mut query {
        let delta_t = time.delta_seconds();
        //search for HardCollision
        vel.v = (vel.v + vel.a * delta_t).clamp_length_max(vel.target_v + BROWNIAN_VELOCITY);
        vel.v = (vel.v + vel.push * delta_t).clamp_length_max(MAX_VELOCITY);
        transform.translation += vel.v * delta_t;
    }
}

pub type NNTree = KDTree3<TrackedByTree>;

#[derive(Component, Default)]
pub struct TrackedByTree;

#[derive(Component, Default)]
pub struct SoftCollision {
    tracked: TrackedByTree,
}

#[derive(Component, Default)]
pub struct HardCollision {
    tracked: TrackedByTree,
}
