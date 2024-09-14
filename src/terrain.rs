use bevy::prelude::*;
use bevy_rts_camera::Ground;
use crate::kinematics::{HardCollision, TrackedByTree};
use crate::util::BundleDefault;

#[derive(Component, Default)]
pub struct Terrain {}

#[derive(Bundle)]
pub struct TerrainBundle {
    pbr: PbrBundle,
    terrain: Terrain,
    ground: Ground
}

impl BundleDefault for TerrainBundle {
    fn default(meshes: &mut ResMut<Assets<Mesh>>, images: &mut ResMut<Assets<Image>>, materials: &mut ResMut<Assets<StandardMaterial>>) -> Self {
        TerrainBundle {
            pbr: PbrBundle
            {
                mesh: meshes.add(
                    Plane3d{ normal: Dir3::Y, half_size: Vec2::new(250., 250.)}
                ),
                material: materials.add(Color::WHITE),
                ..default()
            },
            terrain: Default::default(),
            ground: Ground,
        }
    }
}

#[derive(Component, Default)]
pub struct Obstacle {
    pub(crate) normal: Vec3,
    hard_collision: HardCollision,
    tracked: TrackedByTree,
    pbr: PbrBundle
}

impl Obstacle {
    fn new(mesh: Handle<Mesh>, material: Handle<StandardMaterial>, normal: Vec3, pos: Vec3) -> Self {

        Obstacle{
            normal,
            hard_collision: Default::default(),
            tracked: Default::default(),
            pbr: PbrBundle {
            mesh,
            material,
            transform: Transform::from_translation(pos),
            ..default()
        },
        }
    }
}