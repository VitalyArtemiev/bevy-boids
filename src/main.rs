mod billboard;
mod boid;
mod crowd;
mod formations;
mod freecam;
mod horse;
mod kinematics;
mod launch;
mod player;
mod preprocess;
mod resources;
mod scene;
mod sky;
mod target;
mod terrain;
mod ui;
mod util;

use crate::boid::*;
use crate::billboard::{BillboardPlugin, force_render, swap_boid_lod, update_billboard_yaw};
use crate::crowd::CrowdPlugin;
use crate::ui::debug::{DebugConfig, DebugUiPlugin};
use crate::ui::input::InputPlugin;
use crate::formations::{
    FormationTuning, LODGuard, assign_slots, dispatch_formation_goals, init_formation_speed,
    plan_formation_goals, propagate_formation_targets, transition_formation_orders,
};
use crate::freecam::{CameraMode, apply_camera_mode, freecam_move};
use crate::kinematics::*;
use crate::launch::{BenchPlugin, LaunchConfig, launch_terrain_enabled, parse_launch_args};
use crate::player::{
    FormationSelectionGizmo, Player, SelectionGizmo, draw_cursor, frontage_position_system,
    height_scaled_zoom, mouse_click_system, quick_group_system, selection_indicator_face,
};
use crate::ui::radial::{RadialPlugin, radial_closed};
use crate::resources::{Materials, Meshes};
use crate::scene::{ScenePlugin, TestScene};
use crate::sky::{ENVIRONMENT_MAP_SIZE_PX, SkyPlugin, SkyTuning};
use crate::target::{Target, follow_target};
use crate::terrain::{
    CameraClearance, DemoTerrain, ErosionDemoPlugin, HeightField, ObstacleBundle, TerrainMesh,
    camera_terrain_clearance,
    focus_camera_on_ground, ground_boids, project_obstacles_onto_field, reset_ground_caches,
    spawn_ground,
};
use crate::ui::{GameState, UiPlugin};
use bevy::gizmos::config::{DefaultGizmoConfigGroup, GizmoConfigStore};
use bevy::light::{AtmosphereEnvironmentMapLight, GlobalAmbientLight};
use bevy::math::bounding::Aabb2d;
use bevy::pbr::AtmosphereSettings;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings};
use bevy::window::{PresentMode, WindowPlugin};
use bevy_egui::input::{egui_wants_any_keyboard_input, egui_wants_any_pointer_input};
use bevy_rts_camera::{RtsCamera, RtsCameraControls, RtsCameraPlugin, RtsCameraSystemSet};
use bevy_spatial::{AutomaticUpdate, TransformMode};
use rand::Rng;
use std::time::Duration;

fn main() {
    let launch = match parse_launch_args(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(launch) => launch,
        Err(error) => {
            eprintln!("{error}");
            return;
        }
    };

    // Bake impostor atlases and exit instead of launching the game.
    if launch.preprocess {
        preprocess::run();
        return;
    }

    // Read out the scalar flags the plugin chain needs before `launch`
    // moves into the resource.
    let (bench, shadows, flat_scene, forced_render, no_vsync, scene_active, billboard_scene, crowd_scene) = (
        launch.bench,
        launch.shadows,
        launch.flat,
        launch.force_meshes || launch.force_billboards,
        launch.no_vsync,
        launch.scene.is_some(),
        launch.in_scene(TestScene::Billboard),
        launch.in_scene(TestScene::Crowd),
    );
    let mut sky = SkyTuning {
        shadows,
        ..default()
    };
    if let Some(azimuth) = launch.sun_azimuth_deg {
        sky.sun_azimuth_deg = azimuth;
    }
    if let Some(elevation) = launch.sun_elevation_deg {
        sky.sun_elevation_deg = elevation;
    }

    let mut wgpu_settings = WgpuSettings::default();
    // Browsers have no Vulkan; let wgpu pick (WebGL2/WebGPU) on wasm
    #[cfg(not(target_arch = "wasm32"))]
    {
        wgpu_settings.backends = Some(Backends::VULKAN);
    }

    let mut default_plugins = DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(RenderPlugin {
            render_creation: RenderCreation::Automatic(Box::new(wgpu_settings)),
            ..default()
        });
    if no_vsync {
        // Benches on light scenes would otherwise pin to the refresh rate
        // and compare nothing.
        default_plugins = default_plugins.set(WindowPlugin {
            primary_window: Some(Window {
                present_mode: PresentMode::Immediate,
                ..default()
            }),
            ..default()
        });
    }

    let mut app = App::new();
    app.init_resource::<Materials>()
        .init_resource::<Meshes>()
        .init_resource::<Player>()
        .init_resource::<LODGuard>()
        .init_resource::<HeightField>()
        // Boid identity counter: spawn order today, persisted formation
        // state once boids stream with LOD.
        .init_resource::<BoidIds>()
        .init_resource::<CameraMode>()
        .insert_resource(launch)
        .insert_resource(sky)
        .add_plugins(default_plugins)
        .add_plugins(RtsCameraPlugin)
        .add_plugins(SkyPlugin)
        // Formation crowd-shell experiment (`--scene=crowd`).
        .add_plugins(CrowdPlugin)
        // Per-boid impostor LOD: mesh <-> baked-atlas billboard by camera
        // distance (builds the shared BoidVariations catalog).
        .add_plugins(BillboardPlugin)
        // Standalone test scenes (`--scene <name>`); no-op without one.
        .add_plugins(ScenePlugin)
        // Before UiPlugin: UiPlugin's SettingsPlugin scans the registry at
        // build time, so BindingsSettings must be registered first.
        .add_plugins(InputPlugin)
        .add_plugins(UiPlugin)
        .add_plugins(RadialPlugin)
        // No-op without --bench; skips the main menu when enabled.
        .add_plugins(BenchPlugin(bench))
        .add_plugins(DebugUiPlugin)
        // --force-meshes/--force-billboards bench overrides and the
        // standalone scenes: stand the distance swap down while a render
        // path is fixed by hand (`billboard::force_render` and the scene
        // twins do their own one-shot conversions) — a scene's boids stay
        // on exactly the render path the scene gave them.
        .insert_resource(DebugConfig {
            impostor_lod: !forced_render && !scene_active,
            ..default()
        })
        // Runtime-tunable values exposed by the debug panel (F1).
        .init_resource::<KinematicsTuning>()
        .init_resource::<BoidTuning>()
        .init_resource::<FormationTuning>()
        .insert_resource(ClearColor(Color::srgb(0.38, 0.62, 0.86)))
        .init_resource::<SelectionGizmo>()
        .init_resource::<FormationSelectionGizmo>()
        .add_plugins(
            AutomaticUpdate::<TrackedByTree>::new()
                .with_frequency(Duration::from_secs_f32(1.0))
                .with_transform(TransformMode::Transform),
        )
        .add_systems(Startup, setup);

    if flat_scene || billboard_scene {
        // Flat ground for `--flat` render-path benches and the billboard
        // comparison scene: a plain plane, no water/obstacles, so nothing
        // but the units cost GPU time. The terrain settings resource
        // still exists so the F3 panel's `Res` is valid; without the
        // plugin it just has nothing to rebuild.
        app.add_systems(Startup, scene::spawn_flat_ground)
            .init_resource::<DemoTerrain>();
    } else if crowd_scene {
        // The crowd scene's rolling-hill ground and armies are
        // CrowdPlugin's own, gated on the scene; the terrain settings
        // resource still exists for the F3 panel (nothing to rebuild).
        app.init_resource::<DemoTerrain>();
    } else {
        // The 1 km² ground mesh (needs Assets<Mesh> from the plugins)
        // and the erosion-demo settings/water/rebuild wiring.
        app.init_resource::<TerrainMesh>()
            .add_plugins(ErosionDemoPlugin)
            .add_systems(Startup, spawn_ground.run_if(launch_terrain_enabled));
    }

    app
        .add_systems(
            Update,
            (
                soft_collisions,
                move_step,
                ground_boids.after(move_step).before(bob),
                bob,
                draw_cursor,
                // Input-consuming systems stand down while egui has the
                // pointer/keyboard (an open menu or text field) and while
                // the radial menu owns the pointer.
                mouse_click_system.run_if(
                    radial_closed
                        .and_then(not(egui_wants_any_pointer_input)),
                ),
                quick_group_system.run_if(not(egui_wants_any_keyboard_input)),
                frontage_position_system.run_if(
                    not(egui_wants_any_pointer_input).and_then(not(egui_wants_any_keyboard_input)),
                ),
                height_scaled_zoom.run_if(
                    radial_closed.and_then(not(egui_wants_any_pointer_input)),
                ),
                selection_indicator_face,
                reset_ground_caches,
                project_obstacles_onto_field,
                focus_camera_on_ground.before(RtsCameraSystemSet),
                camera_terrain_clearance.after(RtsCameraSystemSet),
                // Impostor LOD rides the settled camera position; yaw
                // refresh follows any fresh billboards; the one-shot
                // --force-billboards conversion trails both.
                swap_boid_lod.after(RtsCameraSystemSet),
                update_billboard_yaw,
                force_render,
                // Freecam (F3 toggle) takes the camera over by component
                // swap when the mode resource changes; the move system is
                // inert without a Freecam camera.
                apply_camera_mode.run_if(resource_changed::<CameraMode>),
                freecam_move.run_if(
                    not(egui_wants_any_pointer_input)
                        .and_then(not(egui_wants_any_keyboard_input)),
                ),
                hard_collisions.after(soft_collisions),
            )
                .run_if(in_state(GameState::Playing)),
        )
        // The whole executor pipeline runs on the fixed timestep, chained:
        // speed init -> LOD state -> order transitions -> slot re-mapping ->
        // goal planning -> goal dispatch -> member steering. The chain's
        // auto sync points make each stage's commands visible to the next,
        // replacing the former single-system ParamSet passes.
        .add_systems(
            FixedUpdate,
            (
                init_formation_speed,
                propagate_formation_targets,
                transition_formation_orders,
                assign_slots,
                plan_formation_goals,
                dispatch_formation_goals,
                follow_target,
            )
                .chain()
                .run_if(in_state(GameState::Playing)),
        )
        .run();
}

/// Camera draw distance. Sized for the 30 km ceiling: at that altitude the
/// shallow-angle horizon rays meet ground hundreds of km out, and anything
/// closer reads as an abrupt void edge over the hazy far terrain.
const CAMERA_FAR_PLANE_M: f32 = 500_000.0;

/// The game's RTS camera, as the normal launch sets it up. Shared with
/// the crowd test scene (`--scene=crowd`), which is viewed through the
/// same camera instead of a scene-specific one.
pub(crate) fn spawn_rts_camera(commands: &mut Commands) -> Entity {
    commands
        .spawn((
            Camera3d::default(),
            // Smoothed terrain-clearance lift state (see camera_terrain_clearance).
            CameraClearance::default(),
            Projection::Perspective(PerspectiveProjection {
                far: CAMERA_FAR_PLANE_M,
                ..default()
            }),
            RtsCamera {
                // 30 km ceiling: regional view over the LOD tiles (which
                // cover a continent). height_scaled_zoom keeps
                // low-altitude zooming at the legacy feel and accelerates
                // with altitude.
                bounds: Aabb2d::new(Vec2::ZERO, Vec2::new(2_000_000.0, 2_000_000.0)),
                height_min: 2.0,
                height_max: 30_000.0,
                angle: 20.0f32.to_radians(),
                target_angle: 20.0f32.to_radians(),
                min_angle: 20.0f32.to_radians(),
                dynamic_angle: true,
                smoothness: 0.3,
                focus: Transform::IDENTITY,
                target_focus: Transform::IDENTITY,
                zoom: 0.0,
                target_zoom: 0.0,
                snap: false,
            },
            RtsCameraControls {
                key_up: KeyCode::KeyW,
                key_down: KeyCode::KeyS,
                key_left: KeyCode::KeyA,
                key_right: KeyCode::KeyD,
                button_rotate: MouseButton::Middle,
                key_rotate_left: KeyCode::KeyQ,
                key_rotate_right: KeyCode::KeyE,
                key_rotate_speed: 0.5,
                lock_on_rotate: false,
                // RMB is camera drag-pan when nothing is selected;
                // frontage_position_system disables it while a selection
                // exists.
                button_drag: Option::from(MouseButton::Right),
                lock_on_drag: false,
                edge_pan_width: 0.00,
                edge_pan_restrict_to_viewport: false,
                pan_speed: 15.0,
                // Neutralized: zoom input is ours (height_scaled_zoom),
                // whose step is anchored in height metres and grows with
                // altitude.
                zoom_sensitivity: 0.0,
                enabled: true,
            },
        ))
        .id()
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut mesh_list: ResMut<Meshes>,
    mut mat_list: ResMut<Materials>,
    variations: Res<BoidVariations>,
    mut ids: ResMut<BoidIds>,
    field: Res<HeightField>,
    launch: Res<LaunchConfig>,
    mut ambient_light: ResMut<GlobalAmbientLight>,
    mut gizmo_store: ResMut<GizmoConfigStore>,
) {
    // While a `--scene` micro-scene is active it owns the camera, boids
    // and obstacles — the grid spawn below stands down (see scene.rs).
    let scene_active = launch.scene.is_some();
    // Debug gizmos (cursor, selection box, frontage line) must read through
    // the terrain: pull them to the near plane so hills can't bury them.
    let (gizmo_config, _) = gizmo_store.config_mut::<DefaultGizmoConfigGroup>();
    gizmo_config.depth_bias = -1.0;

    mat_list.black = materials.add(StandardMaterial::from_color(Color::BLACK));
    mat_list.white = materials.add(StandardMaterial::from_color(Color::WHITE));

    mesh_list.cube = meshes.add(Cuboid::default());

    // `--boids` works in normal launches too; the default is the full
    // historical 99x99 grid. A test scene spawns its own props instead.
    let boid_budget = if scene_active { 0 } else { launch.boids };
    let mut boids_spawned = 0;
    'grid: for i in 1..100 {
        for j in 1..100 {
            if boids_spawned >= boid_budget {
                break 'grid;
            }
            boids_spawned += 1;
            commands.spawn(BoidBundle::with_id(
                ids.next(),
                Target {
                    pos: Vec3::from_array([(i - 50) as f32, 0.0, (j - 50) as f32]),
                    ..default()
                },
                &variations,
            ));
        }
    }

    // Isolated flat/crowd scene: no obstacle scatter — nothing but the
    // units should cost render time. Any `--scene` also implies this.
    if !launch.flat && !scene_active {
        for _ in 1..100 {
            let mut rng = rand::rng();
            let x = rng.random_range(-100.0..100.0);
            let z = rng.random_range(-100.0..100.0);
            // Obstacles sit on the terrain surface (cube is 1 m, so +0.5 to
            // its centre).
            let y = field.height(x, z) + 0.5;

            commands.spawn(ObstacleBundle::new(
                mesh_list.cube.clone(),
                mat_list.black.clone(),
                Vec3::from_array([1.0, 0.0, 0.0]),
                Vec3::from_array([x, y, z]),
            ));
        }
    }

    // A test scene brings its own plain camera; skipping the RTS one
    // keeps the scene single-camera (several input systems `single()` it)
    // and the scene pose immune to bench zoom pinning.
    // A test scene brings its own camera; skipping the RTS one
    // keeps the scene single-camera (several input systems `single()` it)
    // and the scene pose immune to bench zoom pinning. (The crowd scene
    // spawns the shared RTS camera itself, see scene.rs.)
    let camera = if scene_active {
        None
    } else {
        Some(spawn_rts_camera(&mut commands))
    };

    // Atmosphere is opt-in for now: Bevy 0.19 refilters its environment-map
    // cubemap every frame, and the caching/on-demand fix is still open
    // upstream. `--atmosphere` and `--env-map` work in normal launches too;
    // the bench just inherits the same configuration.
    let environment_map = launch.atmosphere && launch.environment_map;
    let Some(camera) = camera else {
        return;
    };
    if launch.atmosphere {
        // Enables atmosphere rendering for this view; requires HDR.
        commands
            .entity(camera)
            .insert(AtmosphereSettings::default());
    }
    if environment_map {
        // Sky-driven ambient and reflections. Bevy re-renders and refilters
        // this cubemap every frame; 128 px matches upstream's new default
        // (bevyengine/bevy#24738).
        commands
            .entity(camera)
            .insert(AtmosphereEnvironmentMapLight {
                size: ENVIRONMENT_MAP_SIZE_PX,
                ..default()
            });
        // The cubemap is the ambient source; the flat ambient would wash it out.
        *ambient_light = GlobalAmbientLight::NONE;
    } else {
        // With the atmosphere probe off, the flat ambient keeps shadowed
        // terrain readable instead of falling to black (GlobalAmbientLight's
        // 80 cd/m² default).
        *ambient_light = GlobalAmbientLight::default();
    }

    // Makes bright pixels glow; independent of the atmosphere stack.
    if launch.bloom {
        commands.entity(camera).insert(Bloom::default());
    }
}
