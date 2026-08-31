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

/// Runtime-tunable movement limits, exposed as sliders by the debug UI.
/// Defaults mirror the consts above, which stay authoritative for spawn
/// paths and tests.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct KinematicsTuning {
    /// Hard cap on boid speed, m/s (`MAX_VELOCITY`).
    pub max_velocity_mps: f32,
    /// Slack over `target_v` the velocity clamp allows (Brownian jitter), m/s.
    pub brownian_velocity_mps: f32,
    /// Hard cap on steering acceleration, m/s² (`MAX_ACCELERATION`).
    pub max_acceleration_mpss: f32,
    /// Time budget for decelerating onto a target, seconds.
    pub deceleration_time_sec: f32,
}

impl Default for KinematicsTuning {
    fn default() -> Self {
        Self {
            max_velocity_mps: MAX_VELOCITY,
            brownian_velocity_mps: BROWNIAN_VELOCITY,
            max_acceleration_mpss: MAX_ACCELERATION,
            deceleration_time_sec: DECELERATION_TIME_SEC,
        }
    }
}

pub fn move_step(
    mut query: Query<(&mut Transform, &mut Velocity)>,
    time: Res<Time>,
    tuning: Res<KinematicsTuning>,
) {
    for (mut transform, mut vel) in &mut query {
        let delta_t = time.delta_secs();
        //search for HardCollision
        vel.v =
            (vel.v + vel.a * delta_t).clamp_length_max(vel.target_v + tuning.brownian_velocity_mps);
        vel.v = (vel.v + vel.push * delta_t).clamp_length_max(tuning.max_velocity_mps);
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
