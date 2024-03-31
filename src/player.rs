use bevy::prelude::{Camera3d, Component, Query, Res, Time, Transform, Vec2, With};
use crate::boid::{Boid, NNTree};
use crate::kinematics::Velocity;
use crate::terrain::Terrain;

#[derive(Component,Default)]
pub struct Player {
    selecting: bool,
    corner1: Vec2,
    corner2: Vec2
}


pub fn start_select(mut q_player: Query<(&Camera3d, &mut Player)>, q_terrain: Query<(&Terrain)>) {

}

pub fn end_select(mut q_boids: Query<(&mut Transform, &crate::boid::Boid)>, tree: Res<NNTree>, q_terrain: Query<(&Terrain)>, time: Res<Time>) {
   
}