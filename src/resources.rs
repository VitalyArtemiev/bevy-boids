use bevy::math::Vec3;
use bevy::prelude::{Handle, Mesh, Resource, StandardMaterial};

#[derive(Resource, Default)]
pub struct Meshes {
    pub cube: Handle<Mesh>,
    pub capsule: Handle<Mesh>,
    pub plane: Handle<Mesh>,
}
#[derive(Resource, Default)]
pub struct Materials {
    pub debug_material: Handle<StandardMaterial>,
    pub black: Handle<StandardMaterial>,
    pub white: Handle<StandardMaterial>,
}