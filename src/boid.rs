use crate::kinematics::*;
use crate::pbd::{Body, Faction};
use crate::target::Target;
use crate::terrain::GroundY;
use bevy::asset::RenderAssetUsages;
use bevy::prelude::Bundle;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use rand::Rng;

/// The UV-palette checker the debug boid material has always worn — the
/// catalog's placeholder variations tint it per variation, and the
/// preprocessor bakes it, so mesh, billboard and bake all read one texture.
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

/// One boid. `id` is a stable identity assigned at spawn (spawn order
/// today, persisted formation state once boids stream with LOD): it alone
/// decides the boid's visual variation via [`variation_for`], so a soldier
/// can never "switch places" with another across zoom cycles or respawn.
/// Equipment is a separate future concern — the id only picks the variant
/// within one equipment type.
#[derive(Component, Default)]
pub struct Boid {
    pub id: u32,
}

/// Spawn-order id counter; insert before any boid spawn.
#[derive(Resource, Default)]
pub struct BoidIds(pub u32);

impl BoidIds {
    pub fn next(&mut self) -> u32 {
        let id = self.0;
        self.0 += 1;
        id
    }
}

/// How many visual variations the catalog ships. Must match
/// [`boid_variations`]'s list length (a test pins it).
pub const VARIATION_COUNT: usize = 3;

/// Deterministic visual variation for a boid id — a pure round-robin over
/// the catalog. Never random: the same id must always show the same
/// soldier, even after the mesh detaches for the billboard LOD or the boid
/// despawns/respawns with formation streaming.
pub const fn variation_for(id: u32) -> usize {
    (id as usize) % VARIATION_COUNT
}

/// One flavor of the boid model — the catalog entry shared by the spawner
/// (mesh up close) and the impostor preprocessor (one atlas per entry).
#[derive(Clone)]
pub struct BoidVariation {
    /// Doubles as the atlas filename: `assets/impostors/<name>.png`.
    pub name: &'static str,
    pub mesh: Handle<Mesh>,
    /// The lit material the mesh renders with up close. The preprocessor
    /// bakes an unlit clone of it for the atlas albedo.
    pub material: Handle<StandardMaterial>,
    /// Bounding-sphere centre in model-local space: what the bake camera
    /// orbits and the billboard quad offsets by (yaw-rotated).
    pub center: Vec3,
    /// Bounding-sphere radius, metres: the bake fits the ortho frustum to
    /// it, and the billboard sizes its quad `radius × FIT_MARGIN` so the
    /// swap never pops in apparent size.
    pub radius: f32,
}

/// The shared catalog resource. Built once at startup from the same
/// constructor the preprocessor uses — one list, so the game's meshes and
/// the baked atlases can never drift apart.
#[derive(Resource, Clone)]
pub struct BoidVariations(pub Vec<BoidVariation>);

impl FromWorld for BoidVariations {
    fn from_world(world: &mut World) -> Self {
        // Sequential blocks: each `Mut` guard holds its `&mut World` borrow
        // to the end of its block, so the three stores are fetched one at a
        // time and the shared builder still does the real work.
        let (mesh, texture, center, radius) = {
            let mut meshes = world.resource_mut::<Assets<Mesh>>();
            let mesh = meshes.add(Capsule3d::default());
            let (center, radius) =
                crate::preprocess::mesh_bounds(meshes.get(&mesh).expect("just added"));
            drop(meshes);
            let texture = world
                .resource_mut::<Assets<Image>>()
                .add(uv_debug_texture());
            (mesh, texture, center, radius)
        };
        let mut materials = world.resource_mut::<Assets<StandardMaterial>>();
        Self(tinted_capsules(
            mesh,
            texture,
            center,
            radius,
            &mut materials,
        ))
    }
}

/// The catalog. Placeholder entries — the same capsule tinted three ways
/// over the UV-debug texture — exist to prove the multi-variation machinery
/// end to end; real soldier meshes replace this list and nothing else.
pub fn boid_variations(
    meshes: &mut Assets<Mesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<StandardMaterial>,
) -> Vec<BoidVariation> {
    let mesh = meshes.add(Capsule3d::default());
    let texture = images.add(uv_debug_texture());
    let (center, radius) = crate::preprocess::mesh_bounds(meshes.get(&mesh).expect("just added"));
    tinted_capsules(mesh, texture, center, radius, materials)
}

/// Builds the three placeholder variations around one shared mesh.
fn tinted_capsules(
    mesh: Handle<Mesh>,
    texture: Handle<Image>,
    center: Vec3,
    radius: f32,
    materials: &mut Assets<StandardMaterial>,
) -> Vec<BoidVariation> {
    ["capsule-red", "capsule-green", "capsule-blue"]
        .map(|name| {
            let tint = match name {
                "capsule-red" => Color::srgb(1.0, 0.35, 0.35),
                "capsule-green" => Color::srgb(0.40, 1.0, 0.45),
                _ => Color::srgb(0.45, 0.55, 1.0),
            };
            BoidVariation {
                name,
                mesh: mesh.clone(),
                material: materials.add(StandardMaterial {
                    base_color: tint,
                    base_color_texture: Some(texture.clone()),
                    ..default()
                }),
                center,
                radius,
            }
        })
        .into_iter()
        .collect()
}

#[derive(Bundle, Default)]
pub struct BoidBundle {
    boid: Boid,
    transform: Transform,
    target: Target,
    vel: Velocity,
    faction: Faction,
    body: Body,
    mesh: Mesh3d,
    material: MeshMaterial3d<StandardMaterial>,
    ground: GroundY,
    tracked: TrackedByTree,
}

impl BoidBundle {
    /// Spawns a boid whose visual variation is `variation_for(id)` — the
    /// canonical spawn path.
    pub fn with_id(id: u32, target: Target, variations: &BoidVariations) -> Self {
        let variation = &variations.0[variation_for(id)];
        let mut rng = rand::rng();
        let x = rng.random_range(-10.0..10.0);
        let z = rng.random_range(-10.0..10.0);

        BoidBundle {
            boid: Boid { id },
            transform: Transform::from_xyz(x, 0.5, z),
            target,
            mesh: Mesh3d(variation.mesh.clone()),
            material: MeshMaterial3d(variation.material.clone()),
            ..default()
        }
    }

    /// Legacy spawn with explicit handles; variation is id 0's.
    pub fn with_target(
        target: Target,
        mesh: Handle<Mesh>,
        material: Handle<StandardMaterial>,
    ) -> Self {
        let mut rng = rand::rng();
        let x = rng.random_range(-10.0..10.0);
        let z = rng.random_range(-10.0..10.0);

        BoidBundle {
            transform: Transform::from_xyz(x, 0.5, z),
            target,
            mesh: Mesh3d(mesh),
            material: MeshMaterial3d(material),
            ..default()
        }
    }
}

/// Runtime-tunable bob parameters; defaults mirror the consts above, which
/// stay authoritative for comments and docs. (Separation/obstacle response
/// lives in [`crate::pbd::PbdTuning`].)
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct BoidTuning {
    /// Bob height, metres.
    pub bob_amplitude_m: f32,
    /// Idle sway cadence, Hz (stationary units visibly breathe).
    pub bob_freq_idle_hz: f32,
    /// Marching cadence, Hz.
    pub bob_freq_walk_hz: f32,
    /// Sprint cadence, Hz.
    pub bob_freq_run_hz: f32,
}

impl Default for BoidTuning {
    fn default() -> Self {
        Self {
            bob_amplitude_m: BOB_AMPLITUDE,
            bob_freq_idle_hz: BOB_FREQ_IDLE,
            bob_freq_walk_hz: BOB_FREQ_WALK,
            bob_freq_run_hz: BOB_FREQ_RUN,
        }
    }
}

const BOB_AMPLITUDE: f32 = 0.1;
const BOB_FREQ_IDLE: f32 = 0.2;
const BOB_FREQ_WALK: f32 = 2.5;
const BOB_FREQ_RUN: f32 = 5.0;
/// Speed ranges the cadence weights sweep across, m/s: idle fades out by
/// `IDLE_FADE_END`, run takes over from `RUN_FADE_START`.
const IDLE_FADE_END: f32 = 3.0;
const RUN_FADE_START: f32 = 6.0;
const RUN_FADE_END: f32 = 14.0;
/// Capsule3d::default() is 1 m tall; its centre rides half a metre above
/// the terrain surface tracked by GroundY.
const BOID_HALF_HEIGHT: f32 = 0.5;

/// Per-boid bob phase de-sync, derived from the stable spawn id — a
/// golden-ratio hash spreads consecutive ids evenly around the circle, so
/// a crowd never bobs in lockstep and no per-boid state is stored.
fn bob_offset(id: u32) -> f32 {
    use std::f32::consts::TAU;
    ((id as f32) * 0.618_034).fract() * TAU
}

fn smoothstep(x: f32, start: f32, end: f32) -> f32 {
    let t = ((x - start) / (end - start)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Cosmetic vertical bob. Three fixed cadences (idle / walk / run) blended
/// by speed — a speed change morphs the weights, it cannot jump the phase,
/// because every cadence is a pure function of global time. This keeps the
/// effect stateless (no accumulated phase per boid; 200k-unit fields pay
/// three `sin`s and nothing else) at the cost of the cadence being
/// approximate rather than exactly `coef × speed`.
pub fn bob(
    mut q_boids: Query<(&mut Transform, &Velocity, &Boid, &GroundY)>,
    time: Res<Time>,
    tuning: Res<BoidTuning>,
) {
    use std::f32::consts::TAU;
    let t = time.elapsed_secs();
    for (mut transform, vel, boid, ground) in &mut q_boids {
        let v = vel.v.length();
        let w_idle = 1.0 - smoothstep(v, IDLE_FADE_END * 0.25, IDLE_FADE_END);
        let w_run = smoothstep(v, RUN_FADE_START, RUN_FADE_END);
        let w_walk = (1.0 - w_idle - w_run).clamp(0.0, 1.0);
        let arg = t + bob_offset(boid.id);
        let bob_height = tuning.bob_amplitude_m
            * (w_idle * f32::sin(TAU * tuning.bob_freq_idle_hz * arg)
                + w_walk * f32::sin(TAU * tuning.bob_freq_walk_hz * arg)
                + w_run * f32::sin(TAU * tuning.bob_freq_run_hz * arg));
        transform.translation.y = ground.surface + BOID_HALF_HEIGHT + bob_height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variation_for_is_a_deterministic_round_robin() {
        assert_eq!(variation_for(0), 0);
        assert_eq!(variation_for(2), 2);
        assert_eq!(variation_for(3), 0);
        assert_eq!(variation_for(7), 1);
        for id in 0..100u32 {
            assert!(variation_for(id) < VARIATION_COUNT);
        }
    }

    #[test]
    fn catalog_matches_variation_count_with_unique_names() {
        let mut meshes = Assets::<Mesh>::default();
        let mut images = Assets::<Image>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let catalog = boid_variations(&mut meshes, &mut images, &mut materials);
        assert_eq!(catalog.len(), VARIATION_COUNT);
        let mut names: Vec<_> = catalog.iter().map(|v| v.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), VARIATION_COUNT, "names must be unique");
        // The bake fit is a plausible bounding sphere of the shared mesh,
        // and every entry shares one mesh handle.
        assert!(catalog.iter().all(|v| (0.2..2.0).contains(&v.radius)));
        for variation in &catalog[1..] {
            assert_eq!(variation.mesh, catalog[0].mesh);
        }
    }

    /// Bob is stateless: the animation is a pure function of global time
    /// and current speed, so (a) a speed change can never jump the height
    /// — the failure that made bobbing read as acceleration-driven when
    /// phase was `freq × elapsed` — and (b) a stationary boid still sways
    /// at the idle cadence, while a sprinting one bobs visibly faster.
    #[test]
    fn bob_is_stateless_continuous_and_speed_responsive() {
        use bevy::app::App;
        use bevy::time::TimeUpdateStrategy;
        use std::time::Duration;

        let tuning = BoidTuning::default();
        let dt = 1.0 / 60.0;
        let mut app = App::new();
        app.add_plugins(bevy::time::TimePlugin)
            .init_resource::<BoidTuning>()
            .add_systems(bevy::app::Update, bob);
        let e = app
            .world_mut()
            .spawn((
                Boid { id: 7 },
                Transform::IDENTITY,
                Velocity::default(),
                GroundY::default(),
            ))
            .id();

        let tick = |app: &mut App, secs: f32| {
            app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
                secs,
            )));
            app.update();
            app.world().get::<Transform>(e).unwrap().translation.y
        };
        let set_speed = |app: &mut App, v: f32| {
            app.world_mut().get_mut::<Velocity>(e).unwrap().v = Vec3::X * v;
        };

        // (a) Smooth under speed wobble after a long session: with the old
        // freq×elapsed phase, ±0.1 m/s of per-frame jitter moved the phase
        // by TAU·coef·Δv·elapsed (hundreds of radians by now) — chatter.
        // Here the sine arguments do not depend on speed at all, so the
        // per-frame move is bounded by the fastest cadence times dt.
        tick(&mut app, 600.0);
        set_speed(&mut app, 20.0);
        let mut prev = tick(&mut app, dt);
        for i in 0..120 {
            set_speed(&mut app, 20.0 + if i % 2 == 0 { 0.1 } else { -0.1 });
            let y = tick(&mut app, dt);
            assert!(
                (y - prev).abs()
                    <= tuning.bob_amplitude_m * std::f32::consts::TAU * tuning.bob_freq_run_hz * dt
                        + 1e-4,
                "bob must stay smooth under speed wobble, moved {}",
                (y - prev).abs()
            );
            prev = y;
        }

        // (b) Idle sway: stationary, the height sweeps the full amplitude
        // range within one idle period.
        set_speed(&mut app, 0.0);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for _ in 0..(1.0 / tuning.bob_freq_idle_hz / dt) as usize {
            let y = tick(&mut app, dt);
            lo = lo.min(y);
            hi = hi.max(y);
        }
        assert!(
            hi - lo >= tuning.bob_amplitude_m,
            "idle bob must be visible, sweep {}",
            hi - lo
        );

        // (b) Cadence follows speed: sign changes of the bob around its
        // seat over one second — idle slower than sprint.
        let crossings = |app: &mut App, v: f32| {
            set_speed(app, v);
            let mut sign = 0i32;
            let mut count = 0;
            for _ in 0..(1.0 / dt) as usize {
                let y = tick(app, dt) - (BOID_HALF_HEIGHT);
                let s = (y > 0.0) as i32;
                if sign != 0 && s != sign {
                    count += 1;
                }
                sign = s;
            }
            count
        };
        let idle = crossings(&mut app, 0.0);
        let run = crossings(&mut app, 20.0);
        assert!(
            run > idle,
            "sprint must bob faster than idle ({run} vs {idle} crossings)"
        );
    }
}
