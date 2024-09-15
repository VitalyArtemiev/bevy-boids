use crate::kinematics::*;
use crate::target::Target;
use crate::terrain::Obstacle;
use bevy::pbr::PbrBundle;
use bevy::prelude::Bundle;
use bevy::prelude::*;
use bevy_spatial::SpatialAccess;
use rand::Rng;

#[derive(Component, Default)]
pub struct Boid {}

#[derive(Bundle, Default)]
pub struct BoidBundle {
    boid: Boid,
    target: Target,
    vel: Velocity,
    pbr: PbrBundle,
    bob: Bob,
    collision: SoftCollision,
    tracked: TrackedByTree,
}

impl BoidBundle {
    pub fn with_target(target: Target, mesh: Handle<Mesh>,
                       material: Handle<StandardMaterial>) -> Self {
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(-10.0..10.0);
        let z = rng.gen_range(-10.0..10.0);
        let bob_offset = rng.gen_range(-20.0..20.0);

        BoidBundle {
            target,
            pbr: PbrBundle {
                mesh,
                material,
                transform: Transform::from_xyz(
                    x,
                    0.5,
                    z,
                ),
                ..default()
            },
            bob: Bob { offset: bob_offset },
            ..default()
        }
    }
    pub fn random(mesh: Handle<Mesh>,
                  material: Handle<StandardMaterial>) -> Self {
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(-10.0..10.0);
        let z = rng.gen_range(-10.0..10.0);
        let bob_offset = rng.gen_range(-10.0..10.0);

        BoidBundle {
            target: Target { pos: Vec3::from_array([-x, 0.0, -z]), dir: Default::default() },
            pbr: PbrBundle {
                mesh,
                material,
                transform: Transform::from_xyz(
                    x,
                    0.5,
                    z,
                ),
                ..default()
            },
            bob: Bob { offset: bob_offset },
            ..default()
        }
    }
}


const INTERACTION_RADIUS: f32 = 1.0;
const REPEL_COEF: f32 = 0.05;
const MAX_REPEL_ACCELERATION: f32 = MAX_ACCELERATION * 0.5;

pub fn soft_collisions(mut query: Query<(&Transform, &mut Velocity), With<Boid>>, tree: Res<NNTree>) {
    //replace with iter_combinations_mut?
    query.par_iter_mut().for_each(|(transform, mut vel)| {
        let this = transform.translation;
        let mut dir = Vec3::default();

        if let Some((other, _)) = tree.nearest_neighbour(this) {
            let vec = -other + this;
            let len = vec.length().max(0.01);
            //Don't need a branch - if len is large, effect is small
            dir += vec.normalize() / len;
        }
        //Maybe don't need more than one? Should bench but this is slower at 10k
        // for (other, _) in tree.within_distance(this, INTERACTION_RADIUS) {
        //     let vec = - other + this;
        //     let len = vec.length() + 0.01;
        //     dir += vec.normalize() / len;
        // }

        let min_a = (vel.a.length() * REPEL_COEF).min(MAX_REPEL_ACCELERATION);

        // vel.push = (dir).clamp_length_max(min_a);
        vel.a += (dir).clamp_length_max(min_a);
    })
}

const OBSTACLE_INTERACTION_RADIUS: f32 = 0.5;

pub fn hard_collisions(mut q_boids: Query<(&Transform, &mut Velocity), With<Boid>>,
                       q_walls: Query<(&Obstacle, &Transform), With<HardCollision>>, tree: Res<NNTree>) {
    /// Find wall. Find all ents near wall. Remove vel along normal.
    q_walls.iter().for_each(|(obstacle, transform)| {
        for (_other, entity) in tree.within_distance(transform.translation, OBSTACLE_INTERACTION_RADIUS) {
            if let Ok((_transform, mut velocity)) = q_boids.get_mut(entity.unwrap()) {
                let p_v = velocity.v.project_onto(obstacle.normal);
                velocity.v -= p_v;
                let m_a = velocity.a.length();
                let p_a = velocity.a.project_onto(obstacle.normal);
                velocity.a -= p_a;
                velocity.a = velocity.a.normalize() * m_a;
            }
        }
    });
}

#[derive(Component, Default)]
pub struct Bob {
    pub offset: f32,
}

const BOB_AMPLITUDE: f32 = 0.1;
const BOB_FREQ_COEF: f32 = 0.18;
const BOB_FREQ_MIN: f32 = 0.05;

pub fn bob(mut q_boids: Query<(&mut Transform, &Velocity, &Bob), With<Boid>>, time: Res<Time>) {
    for (mut transform, vel, bob) in &mut q_boids {
        let freq = (vel.v.length() * BOB_FREQ_COEF).clamp(BOB_FREQ_MIN, BOB_FREQ_MIN * 4.);
        let time_elapsed = time.elapsed_seconds();
        transform.translation.y = BOB_AMPLITUDE * f32::sin(freq * (bob.offset + time_elapsed))
    }
}
