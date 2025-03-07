use crate::kinematics::{
    Velocity, DECELERATION_TIME_SEC, DECELERATION_TIME_SEC_SQUARED, MAX_ACCELERATION, MAX_VELOCITY,
};
use bevy::math::Vec3;
use bevy::prelude::{Component, Query, Transform};

#[derive(Component, Default)]
// #[require(Velocity)]
pub struct Target {
    pub pos: Vec3,
    pub dir: Vec3,
}

///Add force in target direction
pub fn follow_target(mut query: Query<(&Transform, &Target, &mut Velocity)>) {
    for (transform, target, mut vel) in &mut query {
        let dir: Vec3 = target.pos - transform.translation;
        let v_sign = dir.dot(vel.v).signum();
        let l = dir.length();
        let v = vel.v.length() * v_sign;
        //we always wanna be there in DECELERATION_TIME_SEC
        //a = (l-vt)/t2
        let a = (l - v * DECELERATION_TIME_SEC) / DECELERATION_TIME_SEC_SQUARED;
        vel.target_v = 0.99 * (l / DECELERATION_TIME_SEC).clamp(0., MAX_VELOCITY);
        vel.a = (dir.normalize() * a).clamp_length_max(MAX_ACCELERATION);
    }
}
