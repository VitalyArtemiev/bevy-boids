use bevy::prelude::{Handle, Mesh, Resource, StandardMaterial};

#[derive(Resource, Default)]
pub struct Meshes {
    pub cube: Handle<Mesh>,
}
#[derive(Resource, Default)]
pub struct Materials {
    pub black: Handle<StandardMaterial>,
    pub white: Handle<StandardMaterial>,
}
