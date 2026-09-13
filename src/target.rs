use crate::kinematics::{KinematicsTuning, Velocity};
use bevy::math::Vec3;
use bevy::prelude::{Component, Query, Res, Transform};

#[derive(Component)]
// #[require(Velocity)]
pub struct Target {
    pub pos: Vec3,
    pub dir: Vec3,
    /// Multiplier on the tuning max velocity — how fast this entity closes
    /// on its target (radial-menu Walk < Run; 1.0 = full speed).
    pub speed_scale: f32,
}

impl Default for Target {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            dir: Vec3::ZERO,
            speed_scale: 1.0,
        }
    }
}

///Add force in target direction
pub fn follow_target(
    mut query: Query<(&Transform, &Target, &mut Velocity)>,
    tuning: Res<KinematicsTuning>,
) {
    let t = tuning.deceleration_time_sec;
    for (transform, target, mut vel) in &mut query {
        let dir: Vec3 = target.pos - transform.translation;
        let v_sign = dir.dot(vel.v).signum();
        let l = dir.length();
        let v = vel.v.length() * v_sign;
        //we always wanna be there in DECELERATION_TIME_SEC
        //a = (l-vt)/t2
        let a = (l - v * t) / (t * t);
        let cap = tuning.max_velocity_mps * target.speed_scale.max(0.0);
        vel.target_v = 0.99 * (l / t).clamp(0., cap);
        vel.a = (dir.normalize_or_zero() * a).clamp_length_max(tuning.max_acceleration_mpss);
    }
}
