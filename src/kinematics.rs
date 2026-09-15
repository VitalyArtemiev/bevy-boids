use crate::terrain::HeightField;
use bevy::math::{Vec2, Vec3};
use bevy::prelude::*;
use bevy_spatial::kdtree::KDTree3;

#[derive(Component)]
pub struct Velocity {
    pub v: Vec3,
    pub a: Vec3,
    ///Acceleration due to collisions
    pub push: Vec3,
    pub(crate) target_v: f32,
    /// Uphill terrain gradient (dh/dx, dh/dz) under the entity, lazily
    /// sampled once per metre column (`slope_col`) — a gradient is three
    /// height-field taps, too costly for every boid every frame.
    pub(crate) slope: Vec2,
    slope_col: (i32, i32),
}

impl Default for Velocity {
    fn default() -> Self {
        Velocity {
            v: Vec3::ZERO,
            a: Vec3::ZERO,
            push: Vec3::ZERO,
            target_v: 0.0,
            slope: Vec2::ZERO,
            // Sentinel column: forces a first sample in `move_step`.
            slope_col: (i32::MIN, i32::MIN),
        }
    }
}

pub const MAX_VELOCITY: f32 = 20.0;
pub const BROWNIAN_VELOCITY: f32 = 0.02;
pub const MAX_ACCELERATION: f32 = 5.0;
pub const DECELERATION_TIME_SEC: f32 = 1.0;
/// Time constant for relaxing velocity onto the planned velocity, seconds.
pub const STEER_RESPONSE_SEC: f32 = 0.5;
/// How strongly a misaligned target (perpendicular to dead-behind) slows
/// the preferred speed: 0 = ignore alignment, 1 = full damping.
pub const MISALIGN_SLOWDOWN: f32 = 0.5;
pub const GRAVITY_MPSS: f32 = 9.81;
/// Uphill grade (tan of the slope angle) at which steering thrust bottoms
/// out at `SLOPE_ACCEL_SCALE_MIN`; unitless.
pub const SLOPE_ACCEL_COEF: f32 = 1.0;
/// Downhill grade at which the speed cap tops out at
/// `SLOPE_CAP_SCALE_MAX`; unitless.
pub const SLOPE_CAP_COEF: f32 = 1.0;
/// Floor for the uphill thrust multiplier — units never lose *all* drive,
/// even on a cliff wall.
pub const SLOPE_ACCEL_SCALE_MIN: f32 = 0.2;
/// Ceiling for the downhill cap relaxation.
pub const SLOPE_CAP_SCALE_MAX: f32 = 1.5;
/// Forward-difference step for the terrain gradient sample, metres.
pub const SLOPE_SAMPLE_STEP_M: f32 = 0.5;

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
    /// Time constant for relaxing velocity onto the planned velocity, s.
    pub steer_response_sec: f32,
    /// Strength of the misalignment slowdown (0 = off, 1 = full).
    pub misalign_slowdown: f32,
    /// Gravity pull along the terrain surface, m/s².
    pub gravity_mpss: f32,
    /// Uphill thrust loss per unit of grade (see `SLOPE_ACCEL_COEF`).
    pub slope_accel_coef: f32,
    /// Downhill speed-cap gain per unit of grade (see `SLOPE_CAP_COEF`).
    pub slope_cap_coef: f32,
}

impl Default for KinematicsTuning {
    fn default() -> Self {
        Self {
            max_velocity_mps: MAX_VELOCITY,
            brownian_velocity_mps: BROWNIAN_VELOCITY,
            max_acceleration_mpss: MAX_ACCELERATION,
            deceleration_time_sec: DECELERATION_TIME_SEC,
            steer_response_sec: STEER_RESPONSE_SEC,
            misalign_slowdown: MISALIGN_SLOWDOWN,
            gravity_mpss: GRAVITY_MPSS,
            slope_accel_coef: SLOPE_ACCEL_COEF,
            slope_cap_coef: SLOPE_CAP_COEF,
        }
    }
}

/// Steering plan toward a point target: the arrival-damped velocity the
/// entity wants, plus the preferred speed that goes with it. `speed_cap`
/// is the entity's own ceiling (tuning max velocity × `Target::speed_scale`).
///
/// Arrival asks to close the remaining distance in
/// `deceleration_time_sec`, so the preferred speed decays linearly onto
/// the target. On top of that, the misalignment slowdown damps the
/// preferred speed by how far the current heading points away from the
/// target bearing — dead ahead keeps the full arrival speed, perpendicular
/// keeps 0.75×, dead behind 0.5× (at the default `misalign_slowdown`). A
/// perpendicular or rear target therefore asks the entity to *brake while
/// turning* instead of carving an arc around the target: with steering
/// acceleration capped, a fast flyby at full preferred speed settles into
/// a stable orbit at the turn radius v²/a (80 m at the default tuning),
/// which the damping breaks.
pub fn arrival_plan(
    pos: Vec3,
    target_pos: Vec3,
    vel: Vec3,
    speed_cap: f32,
    tuning: &KinematicsTuning,
) -> (Vec3, f32) {
    let to_target = target_pos - pos;
    let l = to_target.length();
    let dir = if l > 1e-6 { to_target / l } else { Vec3::ZERO };
    let arrival = (l / tuning.deceleration_time_sec).clamp(0.0, speed_cap.max(0.0));
    // Heading-target alignment remapped to [0, 1]; a stationary entity has
    // no heading and takes the neutral 0.5.
    let align = (vel.normalize_or_zero().dot(dir) * 0.5 + 0.5).clamp(0.0, 1.0);
    let damp = 1.0 - tuning.misalign_slowdown * (1.0 - align);
    let speed = arrival * damp;
    (dir * speed, speed)
}

/// Steering-authority multiplier for the distance left to a target with a
/// `soft_arrival_m` fade: 1.0 at or beyond the radius, `(l/r)²` inside it,
/// and always 1.0 when the radius is 0 (crisp arrival). The ramp is
/// floored at [`SOFT_AUTHORITY_FLOOR`]: near the target slot-keeping keeps
/// just enough authority to settle onto the slot and brake the final
/// approach, while staying well below the PBD solver's local forces —
/// that balance is what lets avoidance win the last stretch without
/// letting arrivals drift.
pub const SOFT_AUTHORITY_FLOOR: f32 = 0.25;

pub fn arrival_authority(distance_m: f32, soft_radius_m: f32) -> f32 {
    if soft_radius_m <= 0.0 {
        return 1.0;
    }
    let ratio = (distance_m / soft_radius_m).clamp(0.0, 1.0);
    (ratio * ratio).max(SOFT_AUTHORITY_FLOOR)
}

/// Gravity's pull along the terrain surface, projected onto the ground
/// plane: `g·sin(θ)·cos(θ)` pointing downhill, m/s². Peaks at half of
/// `gravity_mpss` on a 45° grade and falls off beyond it as the horizontal
/// projection shrinks.
pub fn slope_gravity(grad: Vec2, gravity_mpss: f32) -> Vec3 {
    let d = 1.0 + grad.length_squared();
    Vec3::new(-grad.x, 0.0, -grad.y) * (gravity_mpss / d)
}

/// Steering-thrust multiplier for the grade being climbed (`signed_grade`
/// is the terrain gradient dotted with the heading, positive uphill):
/// reduced uphill — the engine fights gravity — and 1.0 downhill or at
/// rest.
pub fn slope_accel_scale(signed_grade: f32, coef: f32) -> f32 {
    (1.0 - coef * signed_grade.max(0.0)).clamp(SLOPE_ACCEL_SCALE_MIN, 1.0)
}

/// Speed-cap multiplier for the grade being traversed: relaxed downhill,
/// 1.0 uphill or at rest. The cap never *tightens* uphill — climbs are
/// slowed by reduced thrust and gravity, not by a cap snapping excess
/// speed away (that would read as lost momentum).
pub fn slope_cap_scale(signed_grade: f32, coef: f32) -> f32 {
    (1.0 + coef * (-signed_grade).max(0.0)).clamp(1.0, SLOPE_CAP_SCALE_MAX)
}

/// Uphill terrain gradient at a position, by forward differences (three
/// height taps; central differences would cost a fourth for accuracy the
/// metre-smooth erosion field does not need).
fn slope_at(field: &HeightField, pos: Vec3) -> Vec2 {
    let h = field.height(pos.x, pos.z);
    let e = SLOPE_SAMPLE_STEP_M;
    Vec2::new(
        (field.height(pos.x + e, pos.z) - h) / e,
        (field.height(pos.x, pos.z + e) - h) / e,
    )
}

pub fn move_step(
    mut query: Query<(&mut Transform, &mut Velocity)>,
    time: Res<Time>,
    tuning: Res<KinematicsTuning>,
    field: Res<HeightField>,
) {
    let dt = time.delta_secs();
    let gravity = tuning.gravity_mpss;
    query.par_iter_mut().for_each(|(mut transform, mut vel)| {
        let pos = transform.translation;
        let col = (pos.x.floor() as i32, pos.z.floor() as i32);
        let slope = if col != vel.slope_col {
            vel.slope = slope_at(&field, pos);
            vel.slope_col = col;
            vel.slope
        } else {
            vel.slope
        };
        // Downhill pull first, so the heading the slope scales are computed
        // from already includes this step's gravity.
        let a = vel.a;
        vel.v += slope_gravity(slope, gravity) * dt;
        let heading = vel.v.normalize_or_zero().xz().normalize_or_zero();
        let grade = slope.dot(heading);
        vel.v += a * dt * slope_accel_scale(grade, tuning.slope_accel_coef);
        let cap = vel.target_v * slope_cap_scale(grade, tuning.slope_cap_coef)
            + tuning.brownian_velocity_mps;
        let len = vel.v.length();
        if len > cap {
            // Shed excess speed with bounded deceleration instead of
            // clamping it away — a snap would teleport away momentum.
            vel.v *= ((len - tuning.max_acceleration_mpss * dt).max(cap)) / len;
        }
        let v = vel.v + vel.push * dt;
        vel.v = v.clamp_length_max(tuning.max_velocity_mps);
        transform.translation += v.clamp_length_max(tuning.max_velocity_mps) * dt;
    });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::{Target, follow_target};
    use bevy::time::{TimePlugin, TimeUpdateStrategy};
    use std::time::Duration;

    #[test]
    fn arrival_plan_damps_misaligned_targets() {
        let tuning = KinematicsTuning::default();
        let pos = Vec3::ZERO;
        let target = Vec3::new(20.0, 0.0, 0.0);
        let cap = tuning.max_velocity_mps;

        // Dead ahead: full arrival speed, clamped by the speed cap.
        let (_, ahead) = arrival_plan(pos, target, Vec3::new(5.0, 0.0, 0.0), cap, &tuning);
        assert_eq!(ahead, cap);

        // On target: no preferred speed.
        let (_, arrived) = arrival_plan(target, target, Vec3::ZERO, cap, &tuning);
        assert_eq!(arrived, 0.0);

        // Perpendicular, opposite and stationary headings damp the arrival
        // speed (0.75x / 0.5x / 0.75x at the default misalign 0.5).
        let close = Vec3::new(0.0, 0.0, 4.0);
        let arrival = 4.0 / tuning.deceleration_time_sec;
        let (_, perp) = arrival_plan(pos, close, Vec3::new(5.0, 0.0, 0.0), cap, &tuning);
        assert!((perp - arrival * 0.75).abs() < 1e-5);
        let (_, behind) = arrival_plan(pos, close, Vec3::new(0.0, 0.0, -5.0), cap, &tuning);
        assert!((behind - arrival * 0.5).abs() < 1e-5);
        let (_, still) = arrival_plan(pos, close, Vec3::ZERO, cap, &tuning);
        assert!((still - arrival * 0.75).abs() < 1e-5);
    }

    /// The orbit regression: a fast flyby of a perpendicular target must
    /// converge instead of circling at the turn radius (v²/a = 80 m at the
    /// default tuning — the failure mode of bearing-only steering), and
    /// the heading may never turn faster than the acceleration clamp
    /// allows (momentum and turn radius stay physical).
    #[test]
    fn desired_velocity_steering_converges_on_a_perpendicular_target() {
        let tuning = KinematicsTuning::default();
        let dt = 1.0 / 60.0;
        let target = Vec3::new(0.0, 0.0, 10.0);
        let mut pos = Vec3::ZERO;
        let mut vel = Vec3::new(15.0, 0.0, 0.0);
        let mut max_heading_rate = 0.0_f32;

        for _ in 0..3600 {
            let (desired_v, speed) =
                arrival_plan(pos, target, vel, tuning.max_velocity_mps, &tuning);
            let a = ((desired_v - vel) / tuning.steer_response_sec)
                .clamp_length_max(tuning.max_acceleration_mpss);
            let prev_len = vel.length();
            let prev_heading = vel.normalize_or_zero();
            vel += a * dt;
            // Mirror move_step's cap: shed excess speed with bounded
            // deceleration, never a snap.
            let len = vel.length();
            let cap = speed + tuning.brownian_velocity_mps;
            if len > cap {
                vel *= ((len - tuning.max_acceleration_mpss * dt).max(cap)) / len;
            }
            pos += vel * dt;
            // The turn-rate bound only means anything with speed on the
            // wheels; below 3 m/s the same acceleration legitimately turns
            // a tighter circle.
            let heading = vel.normalize_or_zero();
            if prev_len > 3.0 && len > 3.0 {
                let rate = prev_heading.cross(heading).length() / dt;
                max_heading_rate = max_heading_rate.max(rate);
            }
        }

        assert!(
            pos.distance(target) < 1.0,
            "perpendicular flyby should converge, ended {pos:?} ({})",
            pos.distance(target)
        );
        assert!(
            max_heading_rate <= tuning.max_acceleration_mpss / 3.0 + 0.2,
            "heading snapped at {max_heading_rate} rad/s"
        );
    }

    #[test]
    fn arrival_authority_fades_inside_the_soft_radius() {
        // Crisp target (radius 0): full authority everywhere.
        assert_eq!(arrival_authority(0.0, 0.0), 1.0);
        assert_eq!(arrival_authority(0.5, 0.0), 1.0);
        // Soft target: quadratic ramp with a settle floor — floored at the
        // target, quarter at half the radius, full at and beyond.
        assert_eq!(arrival_authority(0.0, 6.0), SOFT_AUTHORITY_FLOOR);
        assert!((arrival_authority(3.0, 6.0) - 0.25).abs() < 1e-5);
        assert_eq!(arrival_authority(6.0, 6.0), 1.0);
        assert_eq!(arrival_authority(20.0, 6.0), 1.0);
    }

    /// Soft arrival must hand the last stretch to avoidance: with the raw
    /// steering demand saturating the clamp (a unit moving away fast), the
    /// same situation steers with markedly less acceleration under a soft
    /// radius than a crisp one.
    #[test]
    fn soft_arrival_reduces_steering_authority_near_the_target() {
        let tuning = KinematicsTuning::default();
        let pos = Vec3::ZERO;
        let near = Vec3::new(3.0, 0.0, 0.0);
        let vel = Vec3::new(-5.0, 0.0, 0.0); // moving away: raw demand 16 m/s²

        let (_, speed) = arrival_plan(pos, near, vel, tuning.max_velocity_mps, &tuning);
        let raw = (Vec3::new(speed, 0.0, 0.0) - vel) / tuning.steer_response_sec;
        let crisp_a = raw.clamp_length_max(tuning.max_acceleration_mpss * arrival_authority(3.0, 0.0));
        let soft_a = raw.clamp_length_max(tuning.max_acceleration_mpss * arrival_authority(3.0, 6.0));

        assert!(soft_a.length() < crisp_a.length() * 0.4);
    }

    #[test]
    fn slope_gravity_points_downhill_and_peaks_at_half_g() {
        // Gradient +x uphill => downhill pull is -x.
        let g = slope_gravity(Vec2::new(0.5, 0.0), 9.81);
        assert!(g.x < 0.0 && g.z == 0.0);
        assert!((g.length() - 9.81 * 0.5 / 1.25).abs() < 1e-4);
        // 45° grade (|grad| = 1) maximises the pull at g/2; steeper falls
        // back off as the horizontal projection shrinks.
        let peak = slope_gravity(Vec2::new(1.0, 0.0), 9.81).length();
        let steeper = slope_gravity(Vec2::new(3.0, 0.0), 9.81).length();
        assert!((peak - 9.81 / 2.0).abs() < 1e-4);
        assert!(steeper < peak);
        assert_eq!(slope_gravity(Vec2::ZERO, 9.81), Vec3::ZERO);
    }

    #[test]
    fn slope_scales_clamp_thrust_and_cap() {
        // Uphill: thrust reduced (floored), cap untouched.
        assert!((slope_accel_scale(0.3, 1.0) - 0.7).abs() < 1e-5);
        assert!((slope_accel_scale(5.0, 1.0) - SLOPE_ACCEL_SCALE_MIN).abs() < 1e-5);
        assert_eq!(slope_cap_scale(0.3, 1.0), 1.0);
        // Downhill: thrust untouched, cap relaxed (ceilinged).
        assert_eq!(slope_accel_scale(-0.3, 1.0), 1.0);
        assert!((slope_cap_scale(-0.3, 1.0) - 1.3).abs() < 1e-5);
        assert!((slope_cap_scale(-5.0, 1.0) - SLOPE_CAP_SCALE_MAX).abs() < 1e-5);
        // At rest: neither.
        assert_eq!(slope_accel_scale(0.0, 1.0), 1.0);
        assert_eq!(slope_cap_scale(0.0, 1.0), 1.0);
    }

    /// End-to-end: marching flat, uphill and downhill legs of the same
    /// grade — the climb must be clearly slower than the flat leg and the
    /// descent clearly faster than the climb.
    #[test]
    fn slopes_make_uphill_much_harder_than_downhill() {
        fn march(grade: f32) -> f32 {
            let mut app = App::new();
            app.add_plugins(TimePlugin)
                .init_resource::<KinematicsTuning>()
                .insert_resource(HeightField::from_fn(move |x, _| grade * x))
                .add_systems(Update, (follow_target, move_step).chain());
            // Burn the first update (records first_update, runs no time).
            app.update();
            app.world_mut().spawn((
                Transform::from_translation(Vec3::ZERO),
                Velocity::default(),
                Target {
                    pos: Vec3::new(1_000.0, 0.0, 0.0),
                    ..Default::default()
                },
            ));
            let dt = 1.0 / 60.0;
            for _ in 0..600 {
                app.insert_resource(TimeUpdateStrategy::ManualDuration(
                    Duration::from_secs_f32(dt),
                ));
                app.update();
            }
            let mut vel = app.world_mut().query::<&Velocity>();
            vel.single(app.world()).unwrap().v.length()
        }

        let flat = march(0.0);
        let uphill = march(0.3);
        let downhill = march(-0.3);
        assert!(uphill < flat - 4.0, "uphill {uphill} vs flat {flat}");
        assert!(
            downhill > uphill + 6.0,
            "downhill {downhill} vs uphill {uphill}"
        );
    }
}
