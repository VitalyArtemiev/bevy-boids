mod boid;
mod kinematics;
mod formations;
mod terrain;
mod util;

use std::time::Duration;
use bevy::{
    prelude::*,
};
use bevy_spatial::{AutomaticUpdate, SpatialAccess, TransformMode};
use crate::boid::*;
use crate::kinematics::*;
use crate::util::*;
use crate::terrain::TerrainBundle;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
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
    for i in 1..50 {
        for j in 1..50 {
            commands.spawn(BoidBundle::with_boid(Boid{ target: Vec3::from_array([(i - 10) as f32, 0.0, (j-10) as f32]) }, &mut meshes, &mut images, &mut materials));
        }
    }

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 9000.0,
            range: 100.,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(8.0, 16.0, 8.0),
        ..default()
    });

    // ground plane
    commands.spawn(TerrainBundle::default(&mut meshes, &mut images, &mut materials));

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 6., 12.0).looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
        ..default()
    });
}