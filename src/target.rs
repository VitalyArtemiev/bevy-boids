use crate::kinematics::{KinematicsTuning, Velocity, arrival_authority, arrival_plan};
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
    /// Steering-authority fade near the target: inside this radius the
    /// acceleration clamp ramps down quadratically (0 at the target, full
    /// at the radius), so local forces — the PBD solver's anticipation and
    /// contact — outmuscle slot-keeping when a unit is roughly where it
    /// should be. 0 keeps crisp, full-authority arrival (the default for
    /// every target except formation slots, which set it from
    /// `FormationTuning::slot_soft_radius_m`).
    pub soft_arrival_m: f32,
}

impl Default for Target {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            dir: Vec3::ZERO,
            speed_scale: 1.0,
            soft_arrival_m: 0.0,
        }
    }
}

/// Steer toward [`Target`] as desired-velocity control: brake toward the
/// arrival-damped, misalignment-slowed preferred velocity (see
/// [`arrival_plan`]) instead of accelerating along the target bearing.
///
/// The old bearing-only math never braked the velocity component
/// perpendicular to the bearing, so a fast flyby of a perpendicular target
/// settled into a stable orbit at the turn radius v²/a. Steering toward a
/// preferred velocity kills that tangential component, while the
/// acceleration clamp keeps every turn at a physical radius — the momentum
/// and turn limits live in the clamp, not in the plan.
pub fn follow_target(
    mut query: Query<(&Transform, &Target, &mut Velocity)>,
    tuning: Res<KinematicsTuning>,
) {
    for (transform, target, mut vel) in &mut query {
        let cap = tuning.max_velocity_mps * target.speed_scale.max(0.0);
        let (desired_v, speed) =
            arrival_plan(transform.translation, target.pos, vel.v, cap, &tuning);
        vel.target_v = speed;
        // Near a soft target the steering authority fades out, leaving
        // avoidance in charge of the last stretch (see `Target::soft_arrival_m`).
        let authority = arrival_authority(
            transform.translation.distance(target.pos),
            target.soft_arrival_m,
        );
        vel.a = ((desired_v - vel.v) / tuning.steer_response_sec)
            .clamp_length_max(tuning.max_acceleration_mpss * authority);
    }
}
