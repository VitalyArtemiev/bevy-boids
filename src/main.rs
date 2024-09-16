mod boid;
mod kinematics;
mod formations;
mod terrain;
mod util;
mod horse;
mod player;
mod target;

use crate::boid::*;
use crate::kinematics::*;
use crate::player::{draw_cursor, mouse_click_system, Player};
use crate::target::{follow_target, Target};
use crate::terrain::{Obstacle, ObstacleBundle, TerrainBundle};
use crate::util::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings};
use bevy::render::{mesh, RenderPlugin};
use bevy::prelude::*;
use bevy_rts_camera::{RtsCamera, RtsCameraControls, RtsCameraPlugin};
use bevy_spatial::{AutomaticUpdate, TransformMode};
use std::time::Duration;
use bevy::math::bounding::Aabb2d;
use bevy::render::render_resource::TextureViewDimension::Cube;
use rand::Rng;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()).set(RenderPlugin {
            render_creation: RenderCreation::Automatic(WgpuSettings {
                backends: Some(Backends::VULKAN),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(RtsCameraPlugin)
        .add_plugins(AutomaticUpdate::<(TrackedByTree)>::new()
            .with_frequency(Duration::from_secs_f32(1.0))
            .with_transform(TransformMode::Transform))
        .add_systems(Startup, setup)
        .add_systems(Update, (soft_collisions, move_step, bob, draw_cursor, mouse_click_system, hard_collisions.after(soft_collisions)))
        .add_systems(FixedUpdate, (follow_target))
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
) {
    let debug_material = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(uv_debug_texture())),
        ..default()
    });
    let black_mat = materials.add(StandardMaterial::from_color(Color::BLACK));

    let capsule = meshes.add(Capsule3d::default());
    let cube = meshes.add(Cuboid::default());


    for i in 1..100 {
        for j in 1..100 {
            let mut ent = commands.spawn(BoidBundle::with_target(
                Target {
                    pos: Vec3::from_array([(i - 50) as f32, 0.0, (j-50) as f32]),
                    dir: Default::default(),
                },
                capsule.clone(),
                debug_material.clone(),
            )).id();

            // commands.entity(ent).insert(NoAutomaticBatching{});
        }
    }


    for i in 1..100 {
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(-100.0..100.0);
        let z = rng.gen_range(-100.0..100.0);

        let mut ent = commands.spawn(ObstacleBundle::new(
            cube.clone(),
            black_mat.clone(),
            Vec3::from_array([1.0, 0.0, 0.0]),
            Vec3::from_array([x as f32, 1.0, z as f32]),
        )).id();
    }

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 9000.0,
            range: 100.,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(10.0, 10.0, 0.0),
        ..default()
    });

    // ground plane
    commands.spawn(TerrainBundle::default(&mut meshes, &mut images, &mut materials));

    commands.spawn((
        Player::default(),
        Camera3dBundle::default(),
        RtsCamera {
            bounds: Aabb2d::new(Vec2::ZERO, Vec2::new(10000.0, 10000.0)),
            height_min: 2.0,
            height_max: 150.0,
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
            button_drag: Option::from(MouseButton::Right),
            lock_on_drag: false,
            edge_pan_width: 0.00,
            pan_speed: 15.0,
            zoom_sensitivity: 0.5,
            enabled: true,
        },
    ));
}