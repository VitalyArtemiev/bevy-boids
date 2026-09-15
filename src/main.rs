mod billboard;
mod boid;
mod crowd;
mod formations;
mod freecam;
mod horse;
mod kinematics;
mod launch;
mod pbd;
mod player;
mod preprocess;
mod resources;
mod scene;
mod sky;
mod target;
mod terrain;
mod ui;
mod util;

use crate::billboard::{BillboardPlugin, force_render, swap_boid_lod, update_billboard_yaw};
use crate::boid::{BoidIds, BoidTuning, bob};
use crate::crowd::CrowdPlugin;
use crate::formations::{
    FormationTuning, LODGuard, assign_slots, dispatch_formation_goals, init_formation_speed,
    plan_formation_goals, propagate_formation_targets, transition_formation_orders,
};
use crate::freecam::{CameraMode, Freecam, apply_camera_mode, freecam_move};
use crate::kinematics::*;
use crate::launch::{BenchPlugin, LaunchConfig, parse_launch_args};
use crate::player::{
    FormationSelectionGizmo, Player, SelectionGizmo, draw_cursor, frontage_position_system,
    height_scaled_zoom, mouse_click_system, quick_group_system, selection_indicator_face,
};
use crate::resources::{Materials, Meshes};
use crate::scene::{LoadWorld, ScenePlugin, load_world};
use crate::sky::{ENVIRONMENT_MAP_SIZE_PX, SkyPlugin, SkyTuning};
use crate::target::follow_target;
use crate::terrain::{
    CameraClearance, ErosionDemoPlugin, TerrainMesh, camera_terrain_clearance,
    focus_camera_on_ground, ground_boids, project_obstacles_onto_field, reset_ground_caches,
};
use crate::ui::debug::DebugUiPlugin;
use crate::ui::input::InputPlugin;
use crate::ui::radial::{RadialPlugin, radial_closed};
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
    let (bench, shadows, no_vsync) = (launch.bench, launch.shadows, launch.no_vsync);
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

    let mut default_plugins =
        DefaultPlugins
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
        // Boid identity counter: spawn order today, persisted formation
        // state once boids stream with LOD. (The world assembler resets
        // it on every load; scene ids are spawn-order-derived.)
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
        // Runtime-tunable values exposed by the debug panel (F1).
        .init_resource::<KinematicsTuning>()
        .init_resource::<BoidTuning>()
        .init_resource::<FormationTuning>()
        .init_resource::<pbd::PbdTuning>()
        .init_resource::<pbd::PbdScratch>()
        .insert_resource(ClearColor(Color::srgb(0.38, 0.62, 0.86)))
        .init_resource::<SelectionGizmo>()
        .init_resource::<FormationSelectionGizmo>()
        .add_plugins(
            AutomaticUpdate::<TrackedByTree>::new()
                .with_frequency(Duration::from_secs_f32(1.0))
                .with_transform(TransformMode::Transform),
        )
        .add_systems(
            Startup,
            // Shared handles first (the world assembler scatters
            // obstacles through them), then the initial world — seeded
            // by ScenePlugin from `--scene`/the default launch and built
            // through the same loader as every runtime switch.
            (setup, load_world.run_if(resource_exists::<LoadWorld>)).chain(),
        );

    // The erosion terrain machinery always exists, whatever the launch
    // mode, so the normal world can be reloaded from any scene at
    // runtime; which ground actually spawns is the world assembler's
    // call (`scene::assemble_world`). In scene launches the rebuild
    // stands down (scenes install their own height fields) and the F3
    // panel's `Res<DemoTerrain>` stays valid.
    app.init_resource::<TerrainMesh>()
        .add_plugins(ErosionDemoPlugin);

    app.add_systems(
        Update,
        (
            move_step,
            // Positional collision response: after integration, before
            // grounding re-seats y (contact, anticipation, obstacles).
            pbd::pbd_contact.after(move_step).before(ground_boids),
            ground_boids.after(move_step).before(bob),
            bob,
            draw_cursor,
            // Input-consuming systems stand down while egui has the
            // pointer/keyboard (an open menu or text field) and while
            // the radial menu owns the pointer.
            mouse_click_system.run_if(radial_closed.and_then(not(egui_wants_any_pointer_input))),
            quick_group_system.run_if(not(egui_wants_any_keyboard_input)),
            frontage_position_system.run_if(
                not(egui_wants_any_pointer_input).and_then(not(egui_wants_any_keyboard_input)),
            ),
            height_scaled_zoom.run_if(radial_closed.and_then(not(egui_wants_any_pointer_input))),
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
                not(egui_wants_any_pointer_input).and_then(not(egui_wants_any_keyboard_input)),
            ),
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

/// The app's one and only camera, spawned once at startup. bevy_egui
/// attaches its primary context — every menu — to the first camera it
/// sees and NEVER attaches another for the app's lifetime, so this entity
/// must never despawn or the UI dies with it (world switches reconfigure
/// it instead: [`reset_app_camera_rts`] / [`configure_app_camera_plain`]).
/// The billboard scene runs it as a plain camera; every other world runs
/// it as the game's shared RTS camera.
#[derive(Component)]
pub struct AppCamera;

/// The single [`AppCamera`], spawning the bare entity if it doesn't exist
/// yet (tests assemble worlds without main's `setup`).
pub(crate) fn app_camera(world: &mut World) -> Entity {
    if let Some(camera) = world
        .query_filtered::<Entity, With<AppCamera>>()
        .iter(world)
        .next()
    {
        return camera;
    }
    world.spawn((Camera3d::default(), AppCamera)).id()
}

/// Fresh game RTS camera state, as a normal launch starts with (see the
/// field comments for the load-bearing tuning).
fn fresh_rts_camera() -> RtsCamera {
    RtsCamera {
        // 30 km ceiling: regional view over the LOD tiles (which cover a
        // continent). height_scaled_zoom keeps low-altitude zooming at
        // the legacy feel and accelerates with altitude.
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
    }
}

fn fresh_rts_controls() -> RtsCameraControls {
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
        // frontage_position_system disables it while a selection exists.
        button_drag: Option::from(MouseButton::Right),
        lock_on_drag: false,
        edge_pan_width: 0.00,
        edge_pan_restrict_to_viewport: false,
        pan_speed: 15.0,
        // Neutralized: zoom input is ours (height_scaled_zoom), whose
        // step is anchored in height metres and grows with altitude.
        zoom_sensitivity: 0.0,
        enabled: true,
    }
}

/// Reinstalls the full RTS camera rig on the app camera with fresh state
/// — the reset every world load performs. Strips scene leftovers (the
/// billboard's plain pose, a freecam takeover) so each world starts from
/// the same clean slate a fresh camera used to give.
pub(crate) fn reset_app_camera_rts(world: &mut World) -> Entity {
    let camera = app_camera(world);
    world.entity_mut(camera).remove::<Freecam>().insert((
        // Smoothed terrain-clearance lift state (see camera_terrain_clearance).
        CameraClearance::default(),
        Projection::Perspective(PerspectiveProjection {
            far: CAMERA_FAR_PLANE_M,
            ..default()
        }),
        fresh_rts_camera(),
        fresh_rts_controls(),
    ));
    camera
}

/// The billboard scene's rig: a plain camera — no RTS components (the
/// input systems that expect an RTS camera find none and stand down, and
/// the bench's zoom pinning doesn't touch the pose), default projection
/// and the given fixed transform.
pub(crate) fn configure_app_camera_plain(world: &mut World, pose: Transform) -> Entity {
    let camera = app_camera(world);
    world
        .entity_mut(camera)
        .remove::<(Freecam, RtsCamera, RtsCameraControls, CameraClearance)>()
        .insert((
            Projection::Perspective(PerspectiveProjection::default()),
            pose,
        ));
    camera
}

/// The normal world's launch-flag dressing on the app camera:
/// atmosphere, environment map, bloom. Atmosphere is opt-in for now:
/// Bevy 0.19 refilters its environment-map cubemap every frame, and the
/// caching/on-demand fix is still open upstream. `--atmosphere` and
/// `--env-map` work in normal launches too; the bench just inherits the
/// same configuration. Re-applied on every load — scenes strip it.
pub(crate) fn apply_camera_dressing(world: &mut World, launch: &LaunchConfig) {
    let camera = app_camera(world);
    let environment_map = launch.atmosphere && launch.environment_map;
    if launch.atmosphere {
        // Enables atmosphere rendering for this view; requires HDR.
        world
            .entity_mut(camera)
            .insert(AtmosphereSettings::default());
    }
    if environment_map {
        // Sky-driven ambient and reflections. Bevy re-renders and refilters
        // this cubemap every frame; 128 px matches upstream's new default
        // (bevyengine/bevy#24738).
        world
            .entity_mut(camera)
            .insert(AtmosphereEnvironmentMapLight {
                size: ENVIRONMENT_MAP_SIZE_PX,
                ..default()
            });
        // The cubemap is the ambient source; the flat ambient would wash it out.
        *world.resource_mut::<GlobalAmbientLight>() = GlobalAmbientLight::NONE;
    } else {
        // With the atmosphere probe off, the flat ambient keeps shadowed
        // terrain readable instead of falling to black (GlobalAmbientLight's
        // 80 cd/m² default).
        *world.resource_mut::<GlobalAmbientLight>() = GlobalAmbientLight::default();
    }
    // Makes bright pixels glow; independent of the atmosphere stack.
    if launch.bloom {
        world.entity_mut(camera).insert(Bloom::default());
    }
}

/// Strips the launch dressing back off: scenes run bare cameras (fresh
/// state, no atmosphere/bloom), and the ambient returns to its default.
pub(crate) fn strip_camera_dressing(world: &mut World) {
    let camera = app_camera(world);
    world
        .entity_mut(camera)
        .remove::<(AtmosphereSettings, AtmosphereEnvironmentMapLight, Bloom)>();
    *world.resource_mut::<GlobalAmbientLight>() = GlobalAmbientLight::default();
}

/// Shared handles, gizmo config and the persistent app camera. World
/// content itself (boids, obstacles, grounds) is `scene::assemble_world`'s
/// job — this runs first, at startup, before the initial world loads (the
/// load then configures the camera for its world).
pub(crate) fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut mesh_list: ResMut<Meshes>,
    mut mat_list: ResMut<Materials>,
    mut gizmo_store: ResMut<GizmoConfigStore>,
) {
    // Debug gizmos (cursor, selection box, frontage line) must read through
    // the terrain: pull them to the near plane so hills can't bury them.
    let (gizmo_config, _) = gizmo_store.config_mut::<DefaultGizmoConfigGroup>();
    gizmo_config.depth_bias = -1.0;

    mat_list.black = materials.add(StandardMaterial::from_color(Color::BLACK));
    mat_list.white = materials.add(StandardMaterial::from_color(Color::WHITE));

    mesh_list.cube = meshes.add(Cuboid::default());

    // The app camera: bare here — whichever world loads first (this same
    // frame) configures its rig. bevy_egui finds it on the first PreUpdate
    // and its context rides this entity for the whole app lifetime.
    commands.spawn((Camera3d::default(), AppCamera));
}
