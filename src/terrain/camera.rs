//! Camera-terrain interaction: keeping the RTS camera out of the ground.

use super::HeightField;
use bevy::prelude::*;
use bevy_rts_camera::RtsCamera;

/// Minimum altitude kept between the camera and the highest terrain within
/// `CAMERA_CLEARANCE_RADIUS_M` of its XZ position.
const CAMERA_TERRAIN_MARGIN_M: f32 = 1.0;
/// Horizontal clearance radius. Clearing only the point directly below is
/// wrong on slopes: the camera slides along a mountainside with the ground
/// *beside* it higher, clipping into it. The camera must clear the highest
/// surface in this disc (the 8 neighbours plus a partial second ring at
/// 1 m sampling).
const CAMERA_CLEARANCE_RADIUS_M: f32 = 2.0;
/// How fast the clearance lift relaxes when terrain no longer demands it.
/// Rising is instant (never clip); falling is eased so the camera doesn't
/// staircase down rough ground in metre steps.
const CAMERA_CLEARANCE_RELAX_M_PER_SEC: f32 = 8.0;

/// Smoothed clearance lift per camera. Recomputed from scratch against the
/// plugin's placement each frame; this component carries only the relax
/// smoothing between frames.
#[derive(Component, Default)]
pub struct CameraClearance {
    offset: f32,
}

/// Terrain-follow the camera focus with one analytic height sample.
///
/// The stock crate `follow_ground` raycast every `Ground`-marked mesh, but
/// `MeshRayCast` AABB-culls the whole mesh world first — ~5 ms/frame with
/// 10k boid meshes, every frame, even with a still camera. The heightfield
/// is the authoritative terrain anyway, so sample it directly. Runs before
/// `RtsCameraSystemSet`: the plugin's smoothing then eases `focus` onto the
/// new focus height.
pub fn focus_camera_on_ground(mut cameras: Query<&mut RtsCamera>, field: Res<HeightField>) {
    for mut camera in &mut cameras {
        let focus = camera.target_focus.translation;
        camera.target_focus.translation.y = field.height(focus.x, focus.z);
    }
}

/// Keep the RTS camera above the terrain. Focus-following (above) tracks
/// only the *focus* point, so zooming in on a mountain still lowers the
/// camera body into the peak and surrounding slopes. Sample the highest
/// field height in a small disc around the camera's XZ position and keep
/// the camera above it. Runs after `RtsCameraSystemSet`; only the height
/// is touched — the plugin owns XZ and rotation.
pub fn camera_terrain_clearance(
    mut cameras: Query<(&mut Transform, &mut CameraClearance), With<Camera3d>>,
    field: Res<HeightField>,
    time: Res<Time>,
) {
    for (mut transform, mut clearance) in &mut cameras {
        let pos = transform.translation;
        let cx = pos.x.floor() as i32;
        let cz = pos.z.floor() as i32;

        let r = CAMERA_CLEARANCE_RADIUS_M.ceil() as i32;
        let mut max_surface = f32::NEG_INFINITY;
        for dx in -r..=r {
            for dz in -r..=r {
                if (dx * dx + dz * dz) as f32 > CAMERA_CLEARANCE_RADIUS_M.powi(2) {
                    continue;
                }
                max_surface = max_surface.max(field.height((cx + dx) as f32, (cz + dz) as f32));
            }
        }

        let min_y = max_surface + CAMERA_TERRAIN_MARGIN_M;
        let needed = (min_y - pos.y).max(0.0);
        if needed > clearance.offset {
            clearance.offset = needed;
        } else {
            clearance.offset = (clearance.offset
                - CAMERA_CLEARANCE_RELAX_M_PER_SEC * time.delta_secs())
            .max(needed);
        }
        if clearance.offset > 0.0 {
            transform.translation.y = pos.y + clearance.offset;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Worst-case terrain: a vertical 30 m wall for x >= 0 against a low
    /// plain — the steepest a heightfield can be.
    fn cliff_field() -> HeightField {
        HeightField::from_fn(|x, _| if x >= 0.0 { 30.0 } else { 1.0 })
    }

    fn app_with(field: HeightField, camera_at: Vec3) -> (App, Entity) {
        let mut app = App::new();
        app.insert_resource(field)
            .insert_resource(Time::<()>::default())
            .add_systems(Update, camera_terrain_clearance);
        let camera = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                CameraClearance::default(),
                Transform::from_translation(camera_at),
            ))
            .id();
        app.update();
        (app, camera)
    }

    fn camera_y(app: &App, camera: Entity) -> f32 {
        app.world().get::<Transform>(camera).unwrap().translation.y
    }

    #[test]
    fn camera_focus_samples_the_field_at_its_own_xz() {
        let mut app = App::new();
        app.insert_resource(HeightField::from_fn(|x, z| 10.0 + x + 2.0 * z))
            .add_systems(Update, focus_camera_on_ground);
        let camera = app
            .world_mut()
            .spawn(RtsCamera {
                target_focus: Transform::from_translation(Vec3::new(3.0, 999.0, -4.0)),
                ..default()
            })
            .id();
        app.update();

        let focus = app
            .world()
            .get::<RtsCamera>(camera)
            .unwrap()
            .target_focus
            .translation;
        assert_eq!(focus.x, 3.0);
        assert_eq!(focus.z, -4.0);
        assert_eq!(focus.y, 5.0);
    }

    #[test]
    fn camera_beside_a_cliff_is_lifted_above_the_wall() {
        // Camera parked at plain level just off the wall: the disc samples
        // the 30 m columns beside it, so it must be lifted above 30 + m.
        let (app, camera) = app_with(cliff_field(), Vec3::new(-0.8, 3.0, 4.0));
        assert!(
            camera_y(&app, camera) >= 30.0 + CAMERA_TERRAIN_MARGIN_M - 1e-3,
            "camera not lifted clear of the wall"
        );
    }

    #[test]
    fn camera_high_above_flat_ground_is_untouched() {
        let (app, camera) = app_with(cliff_field(), Vec3::new(-20.0, 100.0, 0.0));
        assert_eq!(camera_y(&app, camera), 100.0);
    }

    #[test]
    fn relax_lowers_the_lift_smoothly_but_never_below_needed() {
        let mut app = App::new();
        app.insert_resource(cliff_field())
            .insert_resource(Time::<()>::default())
            .add_systems(Update, camera_terrain_clearance);
        let camera = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                CameraClearance::default(),
                Transform::from_translation(Vec3::new(-0.8, 3.0, 4.0)),
            ))
            .id();
        app.update();
        let lifted = camera_y(&app, camera);
        assert!(lifted >= 30.0);

        // Move the camera far from the wall and advance time: the offset
        // relaxes gradually, never instantly.
        app.world_mut()
            .get_entity_mut(camera)
            .unwrap()
            .get_mut::<Transform>()
            .unwrap()
            .translation = Vec3::new(-100.0, 3.0, 4.0);
        let mut time = app.world_mut().resource_mut::<Time>();
        time.advance_by(std::time::Duration::from_secs_f32(0.1));
        app.update();
        let after = camera_y(&app, camera);
        assert!(
            after > 4.0 && after < lifted,
            "offset must decay smoothly, got {after} (from {lifted})"
        );
    }
}
