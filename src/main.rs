mod boid;
mod kinematics;
mod formations;
mod terrain;
mod util;
mod horse;
mod player;

use std::time::Duration;
use bevy::{
    prelude::*,
};

use bevy_rts_camera::{RtsCamera, RtsCameraControls, RtsCameraPlugin};
use bevy_spatial::{AutomaticUpdate, TransformMode};
use crate::boid::*;
use crate::kinematics::*;
use crate::player::{draw_cursor, mouse_click_system, Player};
use crate::util::*;
use crate::terrain::{TerrainBundle};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .add_plugins(RtsCameraPlugin)
        .add_plugins(AutomaticUpdate::<(TrackedByTree)>::new()
            .with_frequency(Duration::from_secs_f32(1.0))
            .with_transform(TransformMode::Transform))
        .add_systems(Startup, setup)
        .add_systems(Update, (avoid_collisions, move_step, bob, draw_cursor, mouse_click_system))
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

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for i in 1..20 {
        for j in 1..20 {
            commands.spawn(BoidBundle::with_boid(
                Boid{ 
                    target: Vec3::from_array([(i - 10) as f32, 0.0, (j-10) as f32]) 
                },
                &mut meshes,
                &mut images,
                &mut materials
            ));
        }
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
        RtsCamera::default(),
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