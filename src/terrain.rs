use bevy::prelude::*;
use bevy_rts_camera::Ground;
use crate::boid::{Bob, Boid};
use crate::kinematics::{HardCollision, SoftCollision, TrackedByTree, Velocity};
use crate::target::Target;
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
                    Plane3d{ normal: Dir3::Y, half_size: Vec2::new(2500., 2500.)}
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
}

#[derive(Bundle, Default)]
pub struct ObstacleBundle {
    obstacle: Obstacle,
    pbr: PbrBundle,
    hard_collision: HardCollision,
    tracked: TrackedByTree,
}

impl ObstacleBundle {
    pub(crate) fn new(mesh: Handle<Mesh>, material: Handle<StandardMaterial>, normal: Vec3, pos: Vec3) -> Self {
        ObstacleBundle{
            obstacle: Obstacle{ normal },
            hard_collision: Default::default(),
            tracked: Default::default(),
            pbr: PbrBundle {
                mesh,
                material,
                transform: Transform::from_xyz(pos.x, pos.y, pos.z),
                ..default()
            },
        }
    }
}

