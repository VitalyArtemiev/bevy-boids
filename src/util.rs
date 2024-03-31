use bevy::prelude::*;

pub trait BundleDefault{
    fn default(meshes: &mut ResMut<Assets<Mesh>>,
               images: &mut ResMut<Assets<Image>>,
               materials: &mut ResMut<Assets<StandardMaterial>>) -> Self;
}
