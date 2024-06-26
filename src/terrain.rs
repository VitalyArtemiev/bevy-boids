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
                mesh: meshes.add(shape::Plane::from_size(50.0)),
                material: materials.add(Color::SILVER),
                ..default()
            },
            terrain: Default::default(),
            ground: Ground,
        }
    }
}