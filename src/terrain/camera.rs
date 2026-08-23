//! Camera-terrain interaction: keeping the RTS camera out of the voxel world.

use super::grounding::find_surface;
use bevy::prelude::*;
use bevy_rts_camera::RtsCamera;
use bevy_voxel_world::prelude::{VoxelWorld, VoxelWorldConfig};

/// Minimum altitude kept between the camera and the highest terrain column
/// within `CAMERA_CLEARANCE_RADIUS_M` of its XZ position.
const CAMERA_TERRAIN_MARGIN_M: f32 = 1.0;
/// Horizontal clearance radius. Clearing only the column directly below is
/// wrong on slopes: the camera slides along a mountainside with the columns
/// *beside* it higher, clipping into their sides. The camera must clear the
/// highest surface in this disc (the 8 neighbours plus a partial second
/// ring at 1 m voxels), which is what makes a zoomed, rotated view from a
/// mountaintop possible.
const CAMERA_CLEARANCE_RADIUS_M: f32 = 2.0;
/// How fast the clearance lift relaxes when terrain no longer demands it.
/// Rising is instant (never clip); falling is eased so the camera doesn't
/// staircase down rough ground in 1 m voxel steps.
const CAMERA_CLEARANCE_RELAX_M_PER_SEC: f32 = 8.0;
/// Column scan floor below the camera: valley surfaces far underneath
/// cannot clip the camera, so scans stop here.
const CAMERA_SCAN_BELOW_M: i32 = 64;

/// Smoothed clearance lift per camera. The plugin re-derives the camera
/// transform every frame, so the clamp is recomputed from scratch each
/// frame against the plugin's placement; this component carries only the
/// relax smoothing between frames.
#[derive(Component, Default)]
pub struct CameraClearance {
    offset: f32,
}

/// Keep the RTS camera out of the terrain. bevy_rts_camera terrain-follows
/// only the *focus* point (`follow_ground`), so zooming in on a mountain
/// lowers the camera into the peak and its surrounding columns. A
/// focus->camera ray does not work (the ~20 degree RTS angle makes it graze
/// terrain immediately); instead, sample the highest surface in a small
/// disc around the camera's XZ position and keep the camera above it.
/// Runs after `RtsCameraSystemSet`; only the height is touched — the
/// plugin owns XZ and rotation. Generic over the world so tests can run
/// it against deterministic synthetic terrain.
pub fn camera_terrain_clearance<C: VoxelWorldConfig>(
    mut cameras: Query<(&mut Transform, &RtsCamera, &mut CameraClearance), With<Camera3d>>,
    voxel_world: VoxelWorld<C>,
    time: Res<Time>,
) {
    let get_voxel = voxel_world.get_voxel_fn();
    for (mut transform, rts, mut clearance) in &mut cameras {
        let pos = transform.translation;
        let cx = pos.x.floor() as i32;
        let cz = pos.z.floor() as i32;

        // Scan from above anything the plugin can place under us (so a
        // buried camera, or a cliff wall beside it, still finds the top
        // surface) down to a floor well below: deep surfaces can't clip.
        let top = pos.y as i32 + rts.height_max as i32;
        let bottom = pos.y as i32 - CAMERA_SCAN_BELOW_M;
        let r = CAMERA_CLEARANCE_RADIUS_M.ceil() as i32;
        let mut max_surface = f32::NEG_INFINITY;
        for dx in -r..=r {
            for dz in -r..=r {
                if (dx * dx + dz * dz) as f32 > CAMERA_CLEARANCE_RADIUS_M.powi(2) {
                    continue;
                }
                if let Some(h) = find_surface(&*get_voxel, cx + dx, cz + dz, top, bottom) {
                    max_surface = max_surface.max(h);
                }
            }
        }
        if max_surface == f32::NEG_INFINITY {
            continue; // no terrain loaded around the camera yet
        }

        let min_y = max_surface + CAMERA_TERRAIN_MARGIN_M;
        let needed = (min_y - pos.y).max(0.0);
        if needed > clearance.offset {
            clearance.offset = needed;
        } else {
            clearance.offset =
                (clearance.offset - CAMERA_CLEARANCE_RELAX_M_PER_SEC * time.delta_secs())
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
    use bevy::MinimalPlugins;
    use bevy::app::App;
    use bevy::ecs::system::RunSystemOnce;
    use bevy::input::mouse::{MouseMotion, MouseWheel};
    use bevy::time::{TimePlugin, TimeUpdateStrategy};
    use bevy::transform::TransformPlugin;
    use bevy_rts_camera::{RtsCameraPlugin, RtsCameraSystemSet};
    use bevy_voxel_world::prelude::{
        ChunkDespawnStrategy, ChunkSpawnStrategy, VoxelLookupDelegate, VoxelWorldCamera,
        VoxelWorldPlugin, WorldVoxel,
    };
    use bevy_rts_camera::Ground;
    use std::time::Duration;

    /// Synthetic worst-case terrain: a `TEST_CLIFF_HEIGHT` wall for x >= 0
    /// against a low plain for x < 0 — a vertical cliff, the steepest a
    /// voxel world can be. Everything downstream (clearance, later
    /// walkability) runs against this instead of the fBm world so the
    /// geometry under test is deterministic.
    const TEST_CLIFF_HEIGHT: i32 = 30;

    #[derive(Resource, Default, Clone)]
    struct TestWorld;

    impl VoxelWorldConfig for TestWorld {
        type MaterialIndex = u8;
        type ChunkUserBundle = (Ground,);

        fn spawning_distance(&self) -> u32 {
            // Generous enough that terrain generates under the camera at
            // every zoom level (height_max = 300 m is ~10 chunks up).
            12
        }
        fn min_despawn_distance(&self) -> u32 {
            4
        }
        /// Viewport-based culling needs a real window; FarAway despawn plus
        /// Close spawn keeps chunk lifecycle purely distance-based, which a
        /// headless test can drive deterministically.
        fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
            ChunkDespawnStrategy::FarAway
        }

        fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
            ChunkSpawnStrategy::Close
        }

        fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
            Box::new(|_, _, _| {
                Box::new(|pos: IVec3, _| {
                    let surface = if pos.x >= 0 {
                        TEST_CLIFF_HEIGHT
                    } else {
                        1
                    };
                    if pos.y <= 0 {
                        WorldVoxel::Solid(0)
                    } else if pos.y < surface {
                        WorldVoxel::Solid(1)
                    } else {
                        WorldVoxel::Air
                    }
                })
            })
        }
    }

    /// Per-tick observation of the camera against the voxel world, written
    /// by `probe_camera` so the assertions can run outside the schedule.
    #[derive(Resource, Default)]
    struct CamProbe {
        camera: Vec3,
        /// Any solid voxel within one voxel of the camera position in every
        /// axis (a 3x3x3 box, the conservative reading of "at least 1 m
        /// from terrain geometry").
        solid_within_1m: bool,
        /// Surface height in the camera's own column, once terrain there is
        /// generated.
        surface_below: Option<f32>,
    }

    fn probe_camera(
        mut probe: ResMut<CamProbe>,
        cameras: Query<&Transform, With<Camera3d>>,
        voxel_world: VoxelWorld<TestWorld>,
    ) {
        let Ok(transform) = cameras.single() else {
            return;
        };
        let pos = transform.translation;
        let get_voxel = voxel_world.get_voxel_fn();
        let base = pos.floor().as_ivec3();
        let mut solid = false;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let v = get_voxel(base + IVec3::new(dx, dy, dz));
                    if !matches!(v, WorldVoxel::Air | WorldVoxel::Unset) {
                        solid = true;
                    }
                }
            }
        }
        probe.camera = pos;
        probe.solid_within_1m = solid;
        // Deep scan: at far zoom the camera is up to height_max above the
        // ground, and the interesting assertion is still "clear of the
        // surface far below".
        probe.surface_below = find_surface(
            &*get_voxel,
            base.x,
            base.z,
            (pos.y + 2.0) as i32,
            (pos.y - 400.0) as i32,
        );
    }

    /// Headless camera rig: task pool (voxel world generates chunks on it),
    /// manual time, transform propagation for the plugin's ground raycast,
    /// the RTS camera plugin, and the voxel test world with the clearance
    /// clamp and probe chained after the plugin's camera update.
    fn camera_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin))
            // RtsCameraPlugin bundles the input controller, whose systems
            // require input resources; stub them empty so those systems
            // no-op and the camera chain runs undisturbed.
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_message::<MouseMotion>()
            .add_message::<MouseWheel>()
            // MeshRayCast (follow_ground, grab_pan) needs the mesh store;
            // the voxel plugin runs meshless here, so provide it empty.
            .init_resource::<Assets<Mesh>>()
            .add_plugins(RtsCameraPlugin)
            // Custom-material mode: the chunk-task polling system is gated
            // on `LoadingTexture::is_loaded`, which only the custom-material
            // path sets synchronously (`minimal()` never loads a texture,
            // so generated voxel data would never land in the chunk map).
            // A plain StandardMaterial keeps everything headless: no render
            // plugins, but real chunk generation, meshing, and our Ground
            // bundles. The asset stores below are pure storage the plugin's
            // mesh/material bookkeeping expects to exist.
            .init_resource::<Assets<Shader>>()
            .init_resource::<Assets<StandardMaterial>>()
            .add_plugins(
                VoxelWorldPlugin::with_config(TestWorld)
                    .with_material(StandardMaterial::default()),
            )
            .init_resource::<CamProbe>()
            .add_systems(
                Update,
                (
                    camera_terrain_clearance::<TestWorld>.after(RtsCameraSystemSet),
                    probe_camera.after(camera_terrain_clearance::<TestWorld>),
                ),
            );
        // Burn the first update: it only records the initial clock instant.
        app.update();
        app
    }

    fn tick(app: &mut App, dt: f32) {
        app.insert_resource(TimeUpdateStrategy::ManualDuration(
            Duration::from_secs_f32(dt),
        ));
        app.update();
    }

    fn spawn_camera(app: &mut App, focus: Vec3) -> Entity {
        app.world_mut()
            .spawn((
                Camera3d::default(),
                VoxelWorldCamera::<TestWorld>::default(),
                CameraClearance::default(),
                RtsCamera {
                    // height 2..300, fixed 20 degree angle (dynamic angle
                    // depends on zoom smoothing on real time; keep the test
                    // deterministic).
                    height_min: 2.0,
                    height_max: 300.0,
                    angle: 20.0f32.to_radians(),
                    target_angle: 20.0f32.to_radians(),
                    min_angle: 20.0f32.to_radians(),
                    dynamic_angle: false,
                    smoothness: 0.3,
                    focus: Transform::from_translation(focus),
                    target_focus: Transform::from_translation(focus),
                    zoom: 0.0,
                    target_zoom: 0.0,
                    snap: false,
                    ..Default::default()
                },
            ))
            .id()
    }

    /// Teleport the camera (focus snapped with a facing, zoom applied
    /// immediately — the plugin's own smoothing runs on real time, which a
    /// headless test cannot advance) and reset the clearance relax state so
    /// it cannot mask a failure with a leftover lift from the previous
    /// position. `face` is the focus forward direction: the plugin pushes
    /// the camera backwards along it, which is what puts the camera beside
    /// (rather than on top of) terrain features.
    fn place_camera(app: &mut App, camera: Entity, focus: Vec3, face: Vec3, zoom: f32) {
        let orientation =
            Transform::from_translation(focus).looking_at(focus + face, Vec3::Y);
        let mut entity = app.world_mut().get_entity_mut(camera).unwrap();
        let mut rts = entity.get_mut::<RtsCamera>().unwrap();
        rts.focus = orientation;
        rts.target_focus = orientation;
        // zoom 1.0 is height_min (closest), 0.0 is height_max (far).
        rts.zoom = zoom;
        rts.target_zoom = zoom;
        drop(rts);
        entity.get_mut::<CameraClearance>().unwrap().offset = 0.0;
    }

    /// Chunk generation is async on the task pool: tick until the terrain
    /// under the camera exists, bounded so a generation failure fails the
    /// test instead of hanging it.
    fn wait_for_terrain(app: &mut App) {
        for _ in 0..600 {
            tick(app, 1.0 / 60.0);
            if app.world().resource::<CamProbe>().surface_below.is_some() {
                return;
            }
        }
        let camera = app.world().resource::<CamProbe>().camera;
        let any_solid = app
            .world_mut()
            .run_system_once(scan_for_solid_near_camera)
            .ok()
            .flatten();
        let (zoom, target_zoom, focus) = app
            .world_mut()
            .query::<&RtsCamera>()
            .single(app.world())
            .map(|rts| (rts.zoom, rts.target_zoom, rts.focus.translation))
            .unwrap_or((f32::NAN, f32::NAN, Vec3::ZERO));
        panic!(
            "terrain under the camera never generated: camera at {camera:?}, \
             zoom {zoom} target {target_zoom} focus {focus:?}, \
             any solid within 16 columns: {any_solid:?}"
        );
    }

    /// Diagnostic for wait_for_terrain failures: is there any solid voxel
    /// in a coarse box around the camera?
    fn scan_for_solid_near_camera(
        cameras: Query<&Transform, With<Camera3d>>,
        voxel_world: VoxelWorld<TestWorld>,
    ) -> Option<bool> {
        let pos = cameras.single().ok()?.translation;
        let get_voxel = voxel_world.get_voxel_fn();
        let base = pos.floor().as_ivec3();
        for dx in -16..=16 {
            for dz in -16..=16 {
                for dy in -400..=40 {
                    if !matches!(
                        get_voxel(base + IVec3::new(dx, dy, dz)),
                        WorldVoxel::Air | WorldVoxel::Unset
                    ) {
                        return Some(true);
                    }
                }
            }
        }
        Some(false)
    }

    #[test]
    fn camera_keeps_clear_of_steep_terrain_across_zoom_levels() {
        let mut app = camera_test_app();
        let camera = spawn_camera(&mut app, Vec3::new(-20.0, 1.0, 0.0));

        // Lowest and highest ground, the cliff crest, and — the case that
        // needs neighbour sampling — parked just off the wall on the plain
        // side, facing along +X so the plugin's backwards camera offset
        // pushes it over the plain beside the 30 m cliff face.
        let forward = Vec3::Z;
        let scenarios: [(Vec3, Vec3, &str); 4] = [
            (Vec3::new(-20.0, 1.0, 0.0), forward, "valley floor"),
            (
                Vec3::new(1.0, TEST_CLIFF_HEIGHT as f32, 4.0),
                forward,
                "cliff crest",
            ),
            (
                Vec3::new(10.0, TEST_CLIFF_HEIGHT as f32, 0.0),
                forward,
                "ridge top",
            ),
            (
                Vec3::new(-1.5, 1.0, 4.0),
                Vec3::NEG_X,
                "beside the cliff wall",
            ),
        ];

        for (focus, face, name) in scenarios {
            // Closest zoom first (worst case: camera_height = height_min),
            // then mid and far.
            for zoom in [1.0, 0.5, 0.0] {
                place_camera(&mut app, camera, focus, face, zoom);
                wait_for_terrain(&mut app);
                // A few more ticks so the clamp reacts to freshly loaded
                // terrain, not just the camera placement.
                for _ in 0..5 {
                    tick(&mut app, 1.0 / 60.0);
                }
                let probe = app.world().resource::<CamProbe>();
                assert!(
                    !probe.solid_within_1m,
                    "{name} @ zoom {zoom}: camera {:?} clips terrain",
                    probe.camera
                );
                if let Some(surface) = probe.surface_below {
                    assert!(
                        probe.camera.y >= surface + CAMERA_TERRAIN_MARGIN_M - 1e-3,
                        "{name} @ zoom {zoom}: camera {:?} below surface {surface} + margin",
                        probe.camera
                    );
                }
            }
        }
    }
}
