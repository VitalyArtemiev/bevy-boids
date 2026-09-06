use crate::kinematics::*;
use crate::target::Target;
use crate::terrain::{GroundY, Obstacle};
use bevy::prelude::Bundle;
use bevy::prelude::*;
use bevy_spatial::SpatialAccess;
use rand::Rng;

#[derive(Component, Default)]
pub struct Boid {}

#[derive(Bundle, Default)]
pub struct BoidBundle {
    boid: Boid,
    transform: Transform,
    target: Target,
    vel: Velocity,
    mesh: Mesh3d,
    material: MeshMaterial3d<StandardMaterial>,
    bob: Bob,
    ground: GroundY,
    collision: SoftCollision,
    tracked: TrackedByTree,
}

impl BoidBundle {
    pub fn with_target(
        target: Target,
        mesh: Handle<Mesh>,
        material: Handle<StandardMaterial>,
    ) -> Self {
        let mut rng = rand::rng();
        let x = rng.random_range(-10.0..10.0);
        let z = rng.random_range(-10.0..10.0);
        let bob_offset = rng.random_range(-20.0..20.0);

        BoidBundle {
            transform: Transform::from_xyz(x, 0.5, z),
            target,
            mesh: Mesh3d(mesh),
            material: MeshMaterial3d(material),
            bob: Bob { offset: bob_offset },
            ..default()
        }
    }
    pub fn random(mesh: Handle<Mesh>, material: Handle<StandardMaterial>) -> Self {
        let mut rng = rand::rng();
        let x = rng.random_range(-10.0..10.0);
        let z = rng.random_range(-10.0..10.0);
        let bob_offset = rng.random_range(-10.0..10.0);

        BoidBundle {
            transform: Transform::from_xyz(x, 0.5, z),
            target: Target {
                pos: Vec3::from_array([-x, 1.0, -z]),
                dir: Default::default(),
            },
            mesh: Mesh3d(mesh),
            material: MeshMaterial3d(material),
            bob: Bob { offset: bob_offset },
            ..default()
        }
    }
}

const REPEL_COEF: f32 = 0.05;

/// Runtime-tunable separation/bob parameters; defaults mirror the consts
/// above, which stay authoritative for comments and docs.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct BoidTuning {
    /// Fraction of current acceleration reused as the repulsion cap.
    pub repel_coef: f32,
    /// Query radius for obstacle (hard) collisions, metres.
    pub obstacle_interaction_radius_m: f32,
    /// Idle bob height, metres.
    pub bob_amplitude_m: f32,
    /// Bob frequency per m/s of speed.
    pub bob_freq_coef: f32,
    /// Floor for the bob frequency, Hz.
    pub bob_freq_min_hz: f32,
}

impl Default for BoidTuning {
    fn default() -> Self {
        Self {
            repel_coef: REPEL_COEF,
            obstacle_interaction_radius_m: OBSTACLE_INTERACTION_RADIUS,
            bob_amplitude_m: BOB_AMPLITUDE,
            bob_freq_coef: BOB_FREQ_COEF,
            bob_freq_min_hz: BOB_FREQ_MIN,
        }
    }
}

pub fn soft_collisions(
    mut query: Query<(Entity, &Transform, &mut Velocity), With<Boid>>,
    tree: Res<NNTree>,
    kin: Res<KinematicsTuning>,
    boid: Res<BoidTuning>,
) {
    //replace with iter_combinations_mut?
    query
        .par_iter_mut()
        .for_each(|(entity, transform, mut vel)| {
            let this = transform.translation;
            let mut dir = Vec3::default();

            // Skip the self-match: `this` is in the tree, so the nearest neighbour is the boid itself
            for (other, other_entity) in tree.k_nearest_neighbour(this, 2) {
                if other_entity == Some(entity) {
                    continue;
                }
                let vec = -other + this;
                let len = vec.length().max(0.01);
                //Don't need a branch - if len is large, effect is small
                dir += vec.normalize_or_zero() / len;
            }
            //Maybe don't need more than one? Should bench but this is slower at 10k
            // for (other, _) in tree.within_distance(this, INTERACTION_RADIUS) {
            //     let vec = - other + this;
            //     let len = vec.length() + 0.01;
            //     dir += vec.normalize() / len;
            // }

            // Repulsion can add at most half of MAX_ACCELERATION.
            let min_a = (vel.a.length() * boid.repel_coef).min(kin.max_acceleration_mpss * 0.5);

            // vel.push = (dir).clamp_length_max(min_a);
            vel.a += (dir).clamp_length_max(min_a);
        })
}

const OBSTACLE_INTERACTION_RADIUS: f32 = 1.5;

pub fn hard_collisions(
    mut q_boids: Query<(&Transform, &mut Velocity), With<Boid>>,
    q_walls: Query<(&Obstacle, &Transform), With<HardCollision>>,
    tree: Res<NNTree>,
    tuning: Res<BoidTuning>,
) {
    // Find wall. Find all ents near wall. Remove vel along normal.
    q_walls.iter().for_each(|(obstacle, transform)| {
        for (_other, entity) in
            tree.within_distance(transform.translation, tuning.obstacle_interaction_radius_m)
        {
            if let Ok((_transform, mut velocity)) = q_boids.get_mut(entity.unwrap()) {
                let p_v = velocity.v.project_onto(obstacle.normal);
                velocity.v -= p_v;
                let m_a = velocity.a.length();
                let p_a = velocity.a.project_onto(obstacle.normal);
                velocity.a -= p_a;
                velocity.a = velocity.a.normalize_or_zero() * m_a;
            }
        }
    });
}

#[derive(Component, Default)]
pub struct Bob {
    pub offset: f32,
}

const BOB_AMPLITUDE: f32 = 0.1;
const BOB_FREQ_COEF: f32 = 0.15;
const BOB_FREQ_MIN: f32 = 0.05;
/// Capsule3d::default() is 1 m tall; its centre rides half a metre above
/// the terrain surface tracked by GroundY.
const BOID_HALF_HEIGHT: f32 = 0.5;

pub fn bob(
    mut q_boids: Query<(&mut Transform, &Velocity, &Bob, &GroundY), With<Boid>>,
    time: Res<Time>,
    tuning: Res<BoidTuning>,
) {
    for (mut transform, vel, bob, ground) in &mut q_boids {
        let freq = (vel.v.length() * tuning.bob_freq_coef)
            .clamp(tuning.bob_freq_min_hz, tuning.bob_freq_min_hz * 4.);
        let time_elapsed = time.elapsed_secs();
        transform.translation.y = ground.surface
            + BOID_HALF_HEIGHT
            + tuning.bob_amplitude_m * f32::sin(freq * (bob.offset + time_elapsed))
    }
}
