//! Free-fly camera: the debug alternative to the RTS camera, toggled from
//! the F3 panel.
//!
//! [`CameraMode`] is the selection resource; [`apply_camera_mode`] performs
//! the swap whenever it changes. Freecam activation is *component
//! presence*: taking over removes `RtsCamera` + `RtsCameraControls` (every
//! crate system is a query over those components, so the plugin stands
//! down without being uninstalled) and parks both, unchanged, on
//! [`Freecam::saved`] for restoration — focus, zoom and settings survive
//! the round trip. The camera keeps its `Transform`, so freecam starts
//! exactly where the RTS camera left it (and `camera_terrain_clearance`
//! keeps lifting it over the ground).
//!
//! Controls while playing (and while egui doesn't hold the input): WASD
//! flies along the view axes, Q/E descends/climbs, RMB-drag looks around,
//! Shift sprints, the scroll wheel retunes the base speed. With the RTS
//! components gone their bindings — WASD pan, Q/E rotate, RMB drag-pan,
//! wheel zoom — are all inert.

use bevy::input::mouse::{AccumulatedMouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy_rts_camera::{RtsCamera, RtsCameraControls};

/// Which controller drives the camera: the `bevy_rts_camera` setup spawned
/// in `main::setup` (default), or the freecam. Toggled from the F3 panel;
/// [`apply_camera_mode`] performs the component swap.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum CameraMode {
    #[default]
    Rts,
    Free,
}

/// Freecam state. Presence on the camera entity is the "freecam is active"
/// bit (see module docs); the fields below survive a pause or a round trip
/// back to RTS mode.
#[derive(Component)]
pub struct Freecam {
    /// The RTS camera and controls as they were when freecam took over;
    /// `apply_camera_mode` restores them when switching back.
    saved: Option<(RtsCamera, RtsCameraControls)>,
    /// Base fly speed in m/s; the scroll wheel retunes it.
    speed_mps: f32,
}

impl Default for Freecam {
    fn default() -> Self {
        Freecam {
            saved: None,
            speed_mps: FREECAM_BASE_SPEED_MPS,
        }
    }
}

/// Base fly speed, m/s — quick enough to cross the boid field, with the
/// wheel growing it to continental scales.
const FREECAM_BASE_SPEED_MPS: f32 = 50.0;
/// Wheel speed clamp: walking pace to a hypersonic continental sweep.
const FREECAM_MIN_SPEED_MPS: f32 = 1.0;
const FREECAM_MAX_SPEED_MPS: f32 = 200_000.0;
/// Speed multiplier per wheel line notch.
const FREECAM_WHEEL_SPEED_STEP: f32 = 1.5;
/// Shift multiplies the base speed.
const FREECAM_SPRINT_MULT: f32 = 5.0;
/// Look sensitivity, radians per pixel of RMB drag.
const FREECAM_LOOK_RAD_PER_PIXEL: f32 = 0.003;
/// Pitch clamp so a drag can't flip past vertical and invert the yaw axis.
const FREECAM_MAX_PITCH_RAD: f32 = 89.0f32.to_radians();
/// Pixel-unit wheel deltas are ~1000x finer (same scaling as
/// `height_scaled_zoom`).
const WHEEL_PIXEL_UNIT_SCALE: f32 = 0.001;

/// Swap the camera between the RTS controller and freecam whenever
/// [`CameraMode`] changes. To freecam: remove the RTS components (standing
/// the crate's systems down) parked on [`Freecam::saved`]. Back to RTS:
/// re-insert them as saved — the crate snaps to their targets and resumes
/// at the pre-freecam focus and zoom.
pub fn apply_camera_mode(
    mode: Res<CameraMode>,
    mut commands: Commands,
    cameras: Query<
        (
            Entity,
            Option<&RtsCamera>,
            Option<&RtsCameraControls>,
            Option<&Freecam>,
        ),
        With<Camera3d>,
    >,
) {
    for (entity, rts, controls, freecam) in &cameras {
        match *mode {
            CameraMode::Rts => {
                let Some(freecam) = freecam else {
                    continue; // already RTS-driven
                };
                let (rts, controls) = freecam
                    .saved
                    .clone()
                    .unwrap_or((RtsCamera::default(), RtsCameraControls::default()));
                commands
                    .entity(entity)
                    .remove::<Freecam>()
                    .insert((rts, controls));
            }
            CameraMode::Free => {
                if freecam.is_some() {
                    continue; // already free; keep the parked state and speed
                }
                let saved = rts.copied().zip(controls.cloned());
                commands
                    .entity(entity)
                    .remove::<RtsCamera>()
                    .remove::<RtsCameraControls>()
                    .insert(Freecam {
                        saved,
                        ..default()
                    });
            }
        }
    }
}

/// Fly the freecam (see module docs for the controls). No-op without a
/// [`Freecam`] camera, so it is cheap to leave running in RTS mode.
pub fn freecam_move(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    mut cameras: Query<(&mut Transform, &mut Freecam), With<Camera3d>>,
) {
    let wheel: f32 = mouse_wheel
        .read()
        .map(|message| match message.unit {
            MouseScrollUnit::Line => message.y,
            MouseScrollUnit::Pixel => message.y * WHEEL_PIXEL_UNIT_SCALE,
        })
        .sum();
    for (mut transform, mut freecam) in &mut cameras {
        if wheel != 0.0 {
            freecam.speed_mps = (freecam.speed_mps * FREECAM_WHEEL_SPEED_STEP.powf(wheel))
                .clamp(FREECAM_MIN_SPEED_MPS, FREECAM_MAX_SPEED_MPS);
        }
        if mouse.pressed(MouseButton::Right) {
            // Extract-adjust-rebuild from the transform: no stored
            // yaw/pitch, so no state to drift out of sync with the RTS
            // takeover pose. Dragging down looks down.
            let (mut yaw, mut pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
            yaw -= mouse_motion.delta.x * FREECAM_LOOK_RAD_PER_PIXEL;
            pitch -= mouse_motion.delta.y * FREECAM_LOOK_RAD_PER_PIXEL;
            pitch = pitch.clamp(-FREECAM_MAX_PITCH_RAD, FREECAM_MAX_PITCH_RAD);
            transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
        }
        let mut dir = Vec3::ZERO;
        let forward = *transform.forward();
        let right = *transform.right();
        if keys.pressed(KeyCode::KeyW) {
            dir += forward;
        }
        if keys.pressed(KeyCode::KeyS) {
            dir -= forward;
        }
        if keys.pressed(KeyCode::KeyD) {
            dir += right;
        }
        if keys.pressed(KeyCode::KeyA) {
            dir -= right;
        }
        if keys.pressed(KeyCode::KeyE) {
            dir += Vec3::Y;
        }
        if keys.pressed(KeyCode::KeyQ) {
            dir -= Vec3::Y;
        }
        let sprint = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
        let speed = freecam.speed_mps * if sprint { FREECAM_SPRINT_MULT } else { 1.0 };
        transform.translation += dir.normalize_or_zero() * speed * time.delta_secs();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// App with `apply_camera_mode` registered the way `main.rs` does. The
    /// camera starts RTS-driven with a recognizable zoom and pan speed.
    fn mode_app() -> (App, Entity) {
        let mut app = App::new();
        app.init_resource::<CameraMode>().add_systems(
            Update,
            apply_camera_mode.run_if(resource_changed::<CameraMode>),
        );
        let camera = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                RtsCamera {
                    target_zoom: 0.4,
                    ..Default::default()
                },
                RtsCameraControls {
                    pan_speed: 42.0,
                    ..Default::default()
                },
            ))
            .id();
        // The init counts as a change; applying Rts over an RTS camera is
        // a no-op.
        app.update();
        (app, camera)
    }

    fn set_mode(app: &mut App, mode: CameraMode) {
        // Inserting the resource counts as a change.
        app.world_mut().insert_resource(mode);
        app.update();
    }

    #[test]
    fn freecam_round_trip_restores_the_rts_camera() {
        let (mut app, camera) = mode_app();

        set_mode(&mut app, CameraMode::Free);
        let world = app.world_mut();
        assert!(world.get::<Freecam>(camera).is_some(), "freecam not active");
        assert!(
            world.get::<RtsCamera>(camera).is_none(),
            "RTS camera still drives the entity"
        );
        assert!(world.get::<RtsCameraControls>(camera).is_none());
        let freecam = world.get::<Freecam>(camera).unwrap();
        assert_eq!(
            freecam.saved.as_ref().map(|(rts, _)| rts.target_zoom),
            Some(0.4),
            "RTS state not parked on the freecam"
        );

        set_mode(&mut app, CameraMode::Rts);
        let world = app.world_mut();
        assert!(world.get::<Freecam>(camera).is_none(), "freecam not lifted");
        assert_eq!(
            world.get::<RtsCamera>(camera).unwrap().target_zoom,
            0.4,
            "saved zoom lost"
        );
        assert_eq!(
            world.get::<RtsCameraControls>(camera).unwrap().pan_speed,
            42.0,
            "saved controls lost"
        );
    }

    #[test]
    fn switching_to_free_twice_keeps_the_parked_state() {
        // Already-free cameras keep their parked RTS state (the second
        // takeover would otherwise park defaults over the real ones).
        let (mut app, camera) = mode_app();
        set_mode(&mut app, CameraMode::Free);
        set_mode(&mut app, CameraMode::Free);
        let freecam = app.world().get::<Freecam>(camera).unwrap();
        assert_eq!(
            freecam.saved.as_ref().map(|(rts, _)| rts.target_zoom),
            Some(0.4)
        );
    }

    /// App with `freecam_move` and manual input resources (no InputPlugin:
    /// bare `ButtonInput`/`AccumulatedMouseMotion` are enough).
    fn fly_app() -> (App, Entity) {
        let mut app = App::new();
        app.insert_resource(Time::<()>::default())
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .insert_resource(AccumulatedMouseMotion::default())
            .add_message::<MouseWheel>()
            .add_systems(Update, freecam_move);
        let camera = app
            .world_mut()
            .spawn((Camera3d::default(), Freecam::default()))
            .id();
        (app, camera)
    }

    #[test]
    fn freecam_flies_along_its_view_direction() {
        let (mut app, camera) = fly_app();
        // Default orientation looks down -Z; W flies that way for 1 s.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyW);
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_secs_f32(1.0));
        app.update();

        let translation = app
            .world()
            .get::<Transform>(camera)
            .unwrap()
            .translation;
        assert!(
            (translation.z + FREECAM_BASE_SPEED_MPS).abs() < 1e-3,
            "W did not fly down -Z: {translation:?}"
        );
        assert!(
            translation.x.abs() < 1e-3 && translation.y.abs() < 1e-3,
            "stray drift: {translation:?}"
        );
    }

    #[test]
    fn rmb_drag_looks_and_pitch_clamps_short_of_vertical() {
        let (mut app, camera) = fly_app();
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Right);
        // A huge downward drag: looking straight down must clamp just shy
        // of -90° so yaw never inverts.
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::new(0.0, 1e6);
        app.update();

        let (_, pitch, _) = app
            .world()
            .get::<Transform>(camera)
            .unwrap()
            .rotation
            .to_euler(EulerRot::YXZ);
        assert!(
            (pitch + FREECAM_MAX_PITCH_RAD).abs() < 1e-3,
            "pitch {pitch} not clamped"
        );
    }
}
