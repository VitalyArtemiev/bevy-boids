mod boid;
mod formations;
mod horse;
mod kinematics;
mod player;
mod resources;
mod target;
mod terrain;
mod util;

use crate::boid::*;
use crate::formations::{
    LODGuard, assign_slots, dispatch_formation_goals, init_formation_speed,
    plan_formation_goals, propagate_formation_targets, transition_formation_orders,
};
use crate::kinematics::*;
use crate::player::{
    FormationSelectionGizmo, Player, SelectionGizmo, draw_cursor, frontage_position_system,
    height_scaled_zoom, mouse_click_system, quick_group_system, selection_indicator_face,
};
use crate::resources::{Materials, Meshes};
use crate::target::{Target, follow_target};
use crate::terrain::{
    CameraClearance, MainWorld, ObstacleBundle, TerrainBrush, camera_terrain_clearance,
    ground_boids, terrain_brush_system, terrain_height,
};
use crate::terrain::walkability::debug_walkability;
use bevy::asset::RenderAssetUsages;
use bevy::gizmos::config::{DefaultGizmoConfigGroup, GizmoConfigStore};
use bevy::math::bounding::Aabb2d;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings};
use bevy_rts_camera::{RtsCamera, RtsCameraControls, RtsCameraPlugin, RtsCameraSystemSet};
use bevy_spatial::{AutomaticUpdate, TransformMode};
use bevy_voxel_world::prelude::{VoxelWorldCamera, VoxelWorldPlugin};
use rand::Rng;
use std::time::Duration;

fn main() {
    let mut wgpu_settings = WgpuSettings::default();
    // Browsers have no Vulkan; let wgpu pick (WebGL2/WebGPU) on wasm
    #[cfg(not(target_arch = "wasm32"))]
    {
        wgpu_settings.backends = Some(Backends::VULKAN);
    }

    App::new()
        .init_resource::<Materials>()
        .init_resource::<Meshes>()
        .init_resource::<Player>()
        .init_resource::<LODGuard>()
        .init_resource::<TerrainBrush>()
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(wgpu_settings)),
                    ..default()
                }),
        )
        .add_plugins(RtsCameraPlugin)
        .add_plugins(VoxelWorldPlugin::with_config(MainWorld))
        .init_resource::<SelectionGizmo>()
        .init_resource::<FormationSelectionGizmo>()
        .add_plugins(
            AutomaticUpdate::<TrackedByTree>::new()
                .with_frequency(Duration::from_secs_f32(1.0))
                .with_transform(TransformMode::Transform),
        )
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                soft_collisions,
                move_step,
                ground_boids.after(move_step).before(bob),
                bob,
                draw_cursor,
                mouse_click_system,
                quick_group_system,
                frontage_position_system,
                height_scaled_zoom,
                selection_indicator_face,
                terrain_brush_system,
                debug_walkability,
                camera_terrain_clearance::<MainWorld>.after(RtsCameraSystemSet),
                hard_collisions.after(soft_collisions),
            ),
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
                .chain(),
        )
        .run();
}

#[derive(Component)]
struct Container {
    pos: Vec3,
    vel: Vec3,
    target: Vec3,
}

const X_EXTENT: f32 = 14.5;

fn uv_debug_texture() -> Image {
    const TEXTURE_SIZE: usize = 8;

    let mut palette: [u8; 32] = [
        255, 102, 159, 255, 255, 159, 102, 255, 236, 255, 102, 255, 121, 255, 102, 255, 102, 255,
        198, 255, 102, 198, 255, 255, 121, 102, 255, 255, 236, 102, 255, 255,
    ];

    let mut texture_data = [0; TEXTURE_SIZE * TEXTURE_SIZE * 4];
    for y in 0..TEXTURE_SIZE {
        let offset = TEXTURE_SIZE * y * 4;
        texture_data[offset..(offset + TEXTURE_SIZE * 4)].copy_from_slice(&palette);
        palette.rotate_right(4);
    }

    Image::new_fill(
        Extent3d {
            width: TEXTURE_SIZE as u32,
            height: TEXTURE_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &texture_data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut mesh_list: ResMut<Meshes>,
    mut mat_list: ResMut<Materials>,
    mut gizmo_store: ResMut<GizmoConfigStore>,
) {
    // Debug gizmos (cursor, selection box, brushes, walkability) must read
    // through the terrain: pull them to the near plane so slopes can't
    // bury them.
    let (gizmo_config, _) = gizmo_store.config_mut::<DefaultGizmoConfigGroup>();
    gizmo_config.depth_bias = -1.0;

    mat_list.black = materials.add(StandardMaterial::from_color(Color::BLACK));
    mat_list.white = materials.add(StandardMaterial::from_color(Color::WHITE));
    mat_list.debug_material = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(uv_debug_texture())),
        ..default()
    });

    mesh_list.cube = meshes.add(Cuboid::default());
    mesh_list.capsule = meshes.add(Capsule3d::default());

    for i in 1..100 {
        for j in 1..100 {
            let mut ent = commands
                .spawn(BoidBundle::with_target(
                    Target {
                        pos: Vec3::from_array([(i - 50) as f32, 0.0, (j - 50) as f32]),
                        dir: Default::default(),
                    },
                    mesh_list.capsule.clone(),
                    mat_list.debug_material.clone(),
                ))
                .id();

            // commands.entity(ent).insert(NoAutomaticBatching{});
        }
    }

    for i in 1..100 {
        let mut rng = rand::rng();
        let x = rng.random_range(-100.0..100.0);
        let z = rng.random_range(-100.0..100.0);
        // Obstacles sit on the terrain surface (cube is 1 m, so +0.5 to
        // its centre), not on the y=0 plane the boids spawn at.
        let y = terrain_height(x, z) + 0.5;

        commands.spawn(ObstacleBundle::new(
            mesh_list.cube.clone(),
            mat_list.black.clone(),
            Vec3::from_array([1.0, 0.0, 0.0]),
            Vec3::from_array([x, y, z]),
        ));
    }

    commands.spawn((
        PointLight {
            color: Default::default(),
            intensity: 9000.0,
            range: 100.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 10.0, 0.0),
    ));

    // Terrain is lit by a sun; its shadows stay off for now — chunk meshes
    // stream in constantly and shadow-map redraws would churn.
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(50.0, 80.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        // Marks the camera the voxel world streams chunks around.
        VoxelWorldCamera::<MainWorld>::default(),
        // Smoothed terrain-clearance lift state (see camera_terrain_clearance).
        CameraClearance::default(),
        RtsCamera {
            bounds: Aabb2d::new(Vec2::ZERO, Vec2::new(10000.0, 10000.0)),
            height_min: 2.0,
            height_max: 300.0,
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
            // frontage_position_system disables it while a selection exists.
            button_drag: Option::from(MouseButton::Right),
            lock_on_drag: false,
            edge_pan_width: 0.00,
            edge_pan_restrict_to_viewport: false,
            pan_speed: 15.0,
            // Neutralized: zoom input is ours (height_scaled_zoom), whose
            // speed scales with camera height instead of staying constant.
            zoom_sensitivity: 0.0,
            enabled: true,
        },
    ));
}
