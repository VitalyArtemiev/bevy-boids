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
use bevy::render::RenderPlugin;
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings};
use bevy_rts_camera::{Ground, RtsCamera, RtsCameraControls, RtsCameraPlugin};
use bevy_spatial::{AutomaticUpdate, SpatialAccess, TransformMode};
use crate::boid::*;
use crate::kinematics::*;
use crate::util::*;
use crate::terrain::TerrainBundle;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .add_plugins(RtsCameraPlugin)
        .add_plugins(AutomaticUpdate::<(SoftCollision)>::new()
            .with_frequency(Duration::from_secs_f32(1.0))
            .with_transform(TransformMode::Transform))
        .add_systems(Startup, setup)
        .add_systems(Update, (move_step, avoid_collisions, bob))
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
        Camera3dBundle::default(),
        RtsCamera::default(),
        RtsCameraControls {
            key_up: KeyCode::KeyW,
            key_down: KeyCode::KeyS,
            key_left: KeyCode::KeyA,
            key_right: KeyCode::KeyD,
            button_rotate: MouseButton::Middle,
            edge_pan_width: 0.00,
            pan_speed: 15.0,
            enabled: true,
        },
    ));
}