use bevy::prelude::*;
use crate::boid::Boid;
use crate::kinematics::Velocity;

pub type FormationFunction = fn(usize, usize) -> Vec2;

#[derive(Component)]

pub struct Formation {

}

#[derive(Component)]
pub struct FormationMember {
    formation: Entity,
    index_number: u32,
}

pub fn form_up(
    mut q_boids: Query<(&mut Boid, &FormationMember)>,
    mut q_formations: Query<(&Formation)>,
) {

    for (mut boid, member) in &mut q_boids {
        q_formations.get(member.formation);
    }
}