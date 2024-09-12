use bevy::prelude::*;
use bevy_rts_camera::Ground;
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
                mesh: meshes.add(Plane3d{ normal: Dir3::Y, half_size: Vec2::new(25., 25.)}),
                material: materials.add(Color::WHITE),
                ..default()
            },
            terrain: Default::default(),
            ground: Ground,
        }
    }
}