use bevy::pbr::PbrBundle;
use bevy::prelude::Bundle;
use bevy::{
    prelude::*,
};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy_spatial::kdtree::KDTree3;
use bevy_spatial::SpatialAccess;
use rand::Rng;
use crate::kinematics::*;
use crate::terrain::Terrain;
use crate::util::BundleDefault;

#[derive(Component,Default)]
pub struct Boid {
    pub(crate) target: Vec3,
}

#[derive(Bundle, Default)]
pub struct BoidBundle {
    boid: Boid,
    vel: Velocity,
    pbr: PbrBundle,
    bob: Bob,
    collision: SoftCollision,
    tracked: TrackedByTree,
}

fn uv_debug_texture() -> Image {
    const TEXTURE_SIZE: usize = 8;

    let mut palette: [u8; 32] = [
        255, 102, 159, 255, 255, 159, 102, 255, 236, 255, 102, 255, 121, 255, 102, 255, 102, 255,
        198, 255, 102, 198, 255, 255, 121, 102, 255, 255, 236, 102, 255, 255,
    ];

    let mut texture_data = [0; TEXTURE_SIZE * TEXTURE_SIZE * 4];
    for y in 0..TEXTURE_SIZE {
        let offset = TEXTURE_SIZE * y * 4;
        texture_data[offset..(offset + TEXTURE_SIZE * 4)].copy_from_slice(&palette);
        palette.rotate_right(4);
    }

    Image::new_fill(
        Extent3d {
            width: TEXTURE_SIZE as u32,
            height: TEXTURE_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &texture_data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

impl BundleDefault for BoidBundle {
    fn default(meshes: &mut ResMut<Assets<Mesh>>,
               images: &mut ResMut<Assets<Image>>,
               materials: &mut ResMut<Assets<StandardMaterial>>) -> Self {

        let debug_material = materials.add(StandardMaterial {
            base_color_texture: Some(images.add(uv_debug_texture())),
            ..default()
        });
        let capsule = meshes.add(Capsule3d::default());

        BoidBundle {
            boid: Default::default(),
            vel: Default::default(),
            pbr: PbrBundle {
                mesh: capsule,
                material: debug_material,
                transform: Transform::from_xyz(
                    -10.0,
                    0.0,
                    0.0,
                ),
                ..default()
            },
            ..default()
        }
    }
}

impl BoidBundle {
    pub fn with_boid(boid: Boid, meshes: &mut ResMut<Assets<Mesh>>,
                     images: &mut ResMut<Assets<Image>>,
                     materials: &mut ResMut<Assets<StandardMaterial>>) -> Self {
        let debug_material = materials.add(StandardMaterial {
            base_color_texture: Some(images.add(uv_debug_texture())),
            ..default()
        });
        let capsule = meshes.add(Capsule3d::default());

        let mut rng = rand::thread_rng();
        let x = rng.gen_range(-10.0..10.0);
        let z = rng.gen_range(-10.0..10.0);
        let bob_offset = rng.gen_range(-20.0..20.0);

        BoidBundle {
            boid,
            pbr: PbrBundle {
                mesh: capsule,
                material: debug_material,
                transform: Transform::from_xyz(
                    x,
                    0.0,
                    z,
                ),
                ..default()
            },
            bob: Bob { offset: bob_offset },
            ..default()
        }
    }
    pub fn random(meshes: &mut ResMut<Assets<Mesh>>,
                         images: &mut ResMut<Assets<Image>>,
                         materials: &mut ResMut<Assets<StandardMaterial>>) -> Self {

        let debug_material = materials.add(StandardMaterial {
            base_color_texture: Some(images.add(uv_debug_texture())),
            ..default()
        });
        let capsule = meshes.add(Capsule3d::default());

        let mut rng = rand::thread_rng();
        let x = rng.gen_range(-10.0..10.0);
        let z = rng.gen_range(-10.0..10.0);
        let bob_offset = rng.gen_range(-10.0..10.0);

        BoidBundle {
            boid: Boid { target: Vec3::from_array([-x, 0.0, -z])},
            pbr: PbrBundle {
                mesh: capsule,
                material: debug_material,
                transform: Transform::from_xyz(
                    x,
                    0.0,
                    z,
                ),
                ..default()
            },
            bob: Bob { offset: bob_offset },
            ..default()
        }
    }
}

///Add force in target direction
pub fn follow_target(mut query: Query<(&Transform, &Boid, &mut Velocity)>) {
    for (transform, boid, mut vel) in &mut query {
        let dir: Vec3 = boid.target - transform.translation;
        let v_sign = dir.dot(vel.v).signum();
        let l = dir.length();
        let v = vel.v.length() * v_sign;
        //we always wanna be there in DECELERATION_TIME_SEC
        //a = (l-vt)/t2
        let a = ((l - v * DECELERATION_TIME_SEC)/DECELERATION_TIME_SEC_SQUARED);

        vel.a = (dir.normalize() * a).clamp_length_max(MAX_ACCELERATION);
    }
}

const INTERACTION_RADIUS: f32 = 1.0;
const REPEL_COEF: f32 = 0.05;
const MAX_REPEL_ACCELERATION: f32 = MAX_ACCELERATION * 0.5;

pub fn avoid_collisions(mut query: Query<(&Transform, &mut Velocity), With<Boid>>, tree: Res<NNTree>){
    for (transform, mut vel) in &mut query {
        let this = transform.translation;
        let mut dir = Vec3::default();
        
        if let Some((other, _)) = tree.nearest_neighbour(this) {
            let vec = -other + this;
            let len = vec.length() + 0.01;
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
    }
}

#[derive(Component,Default)]
pub struct Bob {
    pub offset: f32,
}

const BOB_AMPLITUDE: f32 = 0.1;
const BOB_FREQ_COEF: f32 = 0.3;

pub fn bob(mut q_boids: Query<(&mut Transform,  &Velocity, &Bob), With<Boid>>, time: Res<Time>) {
    for (mut transform, vel, bob) in &mut q_boids {
        let freq = vel.v.length() * BOB_FREQ_COEF;
        let time_elapsed = time.elapsed_seconds();
        transform.translation.y = BOB_AMPLITUDE * f32::sin(freq * (bob.offset + time_elapsed))
    }
}
