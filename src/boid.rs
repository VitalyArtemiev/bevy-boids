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
            let texture = world.resource_mut::<Assets<Image>>().add(uv_debug_texture());
            (mesh, texture, center, radius)
        };
        let mut materials = world.resource_mut::<Assets<StandardMaterial>>();
        Self(tinted_capsules(mesh, texture, center, radius, &mut materials))
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
    let (center, radius) =
        crate::preprocess::mesh_bounds(meshes.get(&mesh).expect("just added"));
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
    bob: Bob,
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
        let bob_offset = rng.random_range(-20.0..20.0);

        BoidBundle {
            boid: Boid { id },
            transform: Transform::from_xyz(x, 0.5, z),
            target,
            mesh: Mesh3d(variation.mesh.clone()),
            material: MeshMaterial3d(variation.material.clone()),
            bob: Bob { offset: bob_offset, ..default() },
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
        let bob_offset = rng.random_range(-20.0..20.0);

        BoidBundle {
            transform: Transform::from_xyz(x, 0.5, z),
            target,
            mesh: Mesh3d(mesh),
            material: MeshMaterial3d(material),
            bob: Bob { offset: bob_offset, ..default() },
            ..default()
        }
    }
}

/// Runtime-tunable bob parameters; defaults mirror the consts above, which
/// stay authoritative for comments and docs. (Separation/obstacle response
/// lives in [`crate::pbd::PbdTuning`].)
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct BoidTuning {
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
            bob_amplitude_m: BOB_AMPLITUDE,
            bob_freq_coef: BOB_FREQ_COEF,
            bob_freq_min_hz: BOB_FREQ_MIN,
        }
    }
}

#[derive(Component, Default)]
pub struct Bob {
    /// Per-boid phase de-sync so a crowd never bobs in lockstep.
    pub offset: f32,
    /// Accumulated bob phase (`phase += freq·dt` each frame). Accumulating
    /// is what keeps the visible frequency equal to `freq` while speed —
    /// and therefore frequency — changes: computing `sin(freq · elapsed)`
    /// instead turns every frequency change into a phase jump of
    /// `Δfreq · elapsed`, i.e. acceleration-driven chatter that worsens
    /// the longer the app runs.
    pub phase: f32,
}

const BOB_AMPLITUDE: f32 = 0.1;
/// Bob frequency per m/s of speed: a 5 m/s march bobs at ~0.75 Hz, a
/// 20 m/s run at ~3 Hz.
const BOB_FREQ_COEF: f32 = 0.15;
/// Idle sway floor, Hz — visible breathing at rest, not the sub-perceptual
/// 0.05 Hz this used to be.
const BOB_FREQ_MIN: f32 = 0.5;
/// Capsule3d::default() is 1 m tall; its centre rides half a metre above
/// the terrain surface tracked by GroundY.
const BOID_HALF_HEIGHT: f32 = 0.5;

pub fn bob(
    mut q_boids: Query<(&mut Transform, &Velocity, &mut Bob, &GroundY), With<Boid>>,
    time: Res<Time>,
    tuning: Res<BoidTuning>,
) {
    use std::f32::consts::TAU;
    let dt = time.delta_secs();
    for (mut transform, vel, mut bob, ground) in &mut q_boids {
        // `freq` is Hz, phase is radians — the TAU is what the original
        // `sin(freq · elapsed)` was missing, another ~6x shrink on top of
        // the clamps.
        let freq = (vel.v.length() * tuning.bob_freq_coef).max(tuning.bob_freq_min_hz);
        bob.phase += TAU * freq * dt;
        transform.translation.y = ground.surface
            + BOID_HALF_HEIGHT
            + tuning.bob_amplitude_m * f32::sin(bob.phase + bob.offset)
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

    /// Bob advances its accumulated phase by exactly `freq·dt` — the
    /// property whose absence made bobbing read as acceleration-driven:
    /// `sin(freq · elapsed)` jumps phase by `Δfreq · elapsed` whenever
    /// speed changes, chatter that worsens the longer the app runs.
    /// A stationary boid must still advance at the floor frequency.
    #[test]
    fn bob_advances_phase_by_frequency_times_dt() {
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
                Boid::default(),
                Transform::IDENTITY,
                Velocity::default(),
                Bob::default(),
                GroundY::default(),
            ))
            .id();
        app.insert_resource(TimeUpdateStrategy::ManualDuration(
            Duration::from_secs_f32(dt),
        ));

        let phase = |app: &App, e| app.world().get::<Bob>(e).unwrap().phase;

        use std::f32::consts::TAU;
        // Stationary: the floor frequency, visible idle sway.
        app.update();
        let p0 = phase(&app, e);
        app.update();
        let p1 = phase(&app, e);
        assert!(
            (p1 - p0 - TAU * tuning.bob_freq_min_hz * dt).abs() < 1e-5,
            "stationary phase advance must be the floor frequency"
        );

        // Quarter period at the floor: the boid visibly rises off the seat.
        for _ in 0..29 {
            app.update();
        }
        let y = app.world().get::<Transform>(e).unwrap().translation.y;
        assert!(
            (y - (BOID_HALF_HEIGHT + tuning.bob_amplitude_m)).abs() < 1e-3,
            "idle bob must reach full amplitude, y {y}"
        );

        // Speeding up mid-run: advance is exactly coef·speed·dt — no
        // elapsed-time phase jump from the frequency change itself.
        app.world_mut()
            .get_mut::<Velocity>(e)
            .unwrap()
            .v = Vec3::X * 10.0;
        app.update();
        let p2 = phase(&app, e);
        app.update();
        let p3 = phase(&app, e);
        assert!(
            (p3 - p2 - TAU * tuning.bob_freq_coef * 10.0 * dt).abs() < 1e-5,
            "moving phase advance must be coef × speed × dt"
        );
    }
}
