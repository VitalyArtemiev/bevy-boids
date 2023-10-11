use bevy::math::Vec3;
use bevy::prelude::*;

#[derive(Component,Default)]
pub struct Velocity {
    pub v: Vec3,
    pub a: Vec3,
    ///Acceleration due to collisions
    pub push: Vec3,
    max_v: f32,
}

pub const MAX_VELOCITY: f32 = 20.0;
pub const MAX_ACCELERATION: f32 = 5.0;
pub const DECELERATION_TIME_SEC: f32 = 1.0;
pub const DECELERATION_TIME_SEC_SQUARED: f32 = DECELERATION_TIME_SEC * DECELERATION_TIME_SEC;

const DECELERATION_DISTANCE: f32 = MAX_VELOCITY * DECELERATION_TIME_SEC - MAX_ACCELERATION * (DECELERATION_TIME_SEC_SQUARED);

pub fn move_step(mut query: Query<(&mut Transform, &mut Velocity)>, time: Res<Time>) {
    for (mut transform, mut vel) in &mut query {
        let delta_t = time.delta_seconds();
        //search for HardCollision
        vel.v = (vel.v + (vel.a + vel.push*100.0) * delta_t).clamp_length_max(MAX_VELOCITY);
        transform.translation += vel.v * delta_t;
    }
}

#[derive(Component)]
pub struct SoftCollision {}

#[derive(Component)]
pub struct HardCollision {}