use crate::boid::{Bob, Boid};
use crate::kinematics::{HardCollision, SoftCollision, TrackedByTree, Velocity};
use crate::target::Target;
use crate::util::BundleDefault;
use bevy::prelude::*;
use bevy_rts_camera::Ground;

#[derive(Component, Default)]
pub struct Terrain {}

#[derive(Bundle)]
pub struct TerrainBundle {
    transform: Transform,
    mesh: Mesh3d,
    material: MeshMaterial3d<StandardMaterial>,
    terrain: Terrain,
    ground: Ground,
}

impl BundleDefault for TerrainBundle {
    fn default(
        meshes: &mut ResMut<Assets<Mesh>>,
        images: &mut ResMut<Assets<Image>>,
        materials: &mut ResMut<Assets<StandardMaterial>>,
    ) -> Self {
        TerrainBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            mesh: Mesh3d(meshes.add(Plane3d {
                normal: Dir3::Y,
                half_size: Vec2::new(2500., 2500.),
            })),
            material: MeshMaterial3d(materials.add(Color::WHITE)),
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
    mesh: Mesh3d,
    material: MeshMaterial3d<StandardMaterial>,
    transform: Transform,
    hard_collision: HardCollision,
    tracked: TrackedByTree,
}

impl ObstacleBundle {
    pub(crate) fn new(
        mesh: Handle<Mesh>,
        material: Handle<StandardMaterial>,
        normal: Vec3,
        pos: Vec3,
    ) -> Self {

        ObstacleBundle {
            obstacle: Obstacle { normal },
            hard_collision: Default::default(),
            tracked: Default::default(),
            mesh: Mesh3d(mesh),
            material: MeshMaterial3d(material),
            transform: Transform::from_xyz(pos.x, pos.y, pos.z),
            // pbr: PbrBundle {
            //     mesh,
            //     material,
            //     transform: Transform::from_xyz(pos.x, pos.y, pos.z),
            //     ..default()
            // },
        }
    }
}
