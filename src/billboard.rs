//! Per-boid impostor LOD: past a camera-distance threshold the PBR mesh is
//! detached and the boid renders as one billboard quad sampling the baked
//! impostor atlas (see `docs/plans/impostor-billboard-lod.md` and
//! `preprocess::atlas` for the layout).
//!
//! Which atlas a boid uses is a pure function of its stable id
//! (`boid::variation_for`) — never spawn-time randomness — so zooming out
//! and back in restores exactly the mesh the soldier had (no "switching
//! places"), and future formation streaming keeps identity for free.
//! Presence of [`Billboard`] is the far-LOD state, mirroring the
//! component-presence LOD convention on formations. The simulation never
//! notices the swap: `bob`, grounding, the kd-tree and formation steering
//! key on `Transform`/`GroundY`/`TrackedByTree`, never on render
//! components.
//!
//! The quad itself is built in the shader: per-instance data rides the
//! engine's [`MeshTag`] channel (automatic instancing — every billboard of
//! a variation is one draw call), the vertex stage picks the atlas cell
//! from the view direction and the folded yaw, and the sprite is always
//! displayed upright — world up projects up-image in every baked cell and
//! up-screen at runtime, so yaw lives entirely in which cell gets picked.
//! The fragment stage lights the unlit albedo with the mirrored view-space
//! normal (`nuv = 1 - uv`), reading ambient and the directional sun
//! straight from the engine's `lights` view binding — the same uniform
//! PBR meshes use — so F1 sun edits, illuminance and any future day/night
//! state apply to both render paths from one source (no sync system, and
//! billboards go dark with the capsules at night instead of glowing at a
//! baked ambient floor).
//!
//! Flat-scene bench, 5k boids, camera ~302 m at 25° from nadir, shadows
//! on, dev profile (`--flat --force-* --no-vsync`, 12 s runs, measured
//! back to back): full capsules 60.0 fps, cutout billboards 70.0 fps —
//! and 35.8 fps when the billboards blend instead, because the
//! transparent phase's per-instance sort defeats the instancing (why
//! [`BillboardMaterial::alpha_mode`] is a cutout). Soft alpha WITHOUT
//! losing instancing exists: `AlphaMode::AlphaToCoverage` + `Msaa::Sample4`
//! on the camera also stays in the opaque phase and measured 70.2 fps —
//! but it drags the whole scene into 4× MSAA, so it's a deliberate
//! switch to make, not a default.

use bevy::asset::Asset;
use bevy::camera::Camera;
use bevy::material::specialize::SpecializedMeshPipelineError;
use bevy::mesh::MeshTag;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{AsBindGroup, RenderPipelineDescriptor};
use bevy::shader::ShaderRef;

use crate::boid::{Boid, BoidBundle, BoidVariations, variation_for};
use crate::debug_ui::DebugConfig;
use crate::kinematics::Velocity;
use crate::launch::LaunchConfig;
use crate::preprocess::atlas;
use crate::target::Target;
use std::f32::consts::TAU;

/// Swap-out distance, metres: beyond it the mesh detaches. Default leaves
/// the bench camera (~300 m) deep in billboard territory while keeping the
/// walk-around view fully meshed.
const SWAP_DISTANCE_M: f32 = 50.0;

/// Hysteresis band, metres: a boid closer than
/// `swap_distance - hysteresis` returns to its mesh. Without the band a
/// soldier parked exactly at the threshold would thrash between draw paths
/// every frame as the camera breathes.
const HYSTERESIS_M: f32 = 5.0;

/// Final multiplier on the billboard's light sum. 1.0 is PBR parity by
/// construction (same `lights` binding, same `view.exposure`); the F1
/// slider exists to eyeball-match the lab twins — the baked-normal
/// approximation shades a touch differently from real normals, and a
/// scalar beats re-deriving the difference analytically.
const BRIGHTNESS: f32 = 1.0;

/// Billboard pose occupying the tag's pose bits (see [`pack_tag`]). The
/// idle frame; a future animation system writes walk/attack poses.
pub const IDLE_POSE: u32 = 0;

/// Far-LOD marker: this boid currently renders as an atlas billboard.
#[derive(Component)]
pub struct Billboard;

/// Runtime-tunable swap distances; exposed as sliders by the debug UI.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct BillboardTuning {
    /// Camera distance beyond which the mesh detaches, metres.
    pub swap_distance_m: f32,
    /// Width of the hysteresis band inside the swap distance, metres.
    pub hysteresis_m: f32,
    /// Multiplier on the billboard's final light sum (1.0 = PBR parity).
    /// Pushed into the shared materials live by [`sync_billboard_brightness`].
    pub brightness: f32,
}

impl Default for BillboardTuning {
    fn default() -> Self {
        Self {
            swap_distance_m: SWAP_DISTANCE_M,
            hysteresis_m: HYSTERESIS_M,
            brightness: BRIGHTNESS,
        }
    }
}

/// The shared billboard handles: one quad mesh, one material per catalog
/// variation (each bound to that variation's baked atlas). Built once at
/// startup from the same catalog the spawner and the baker use.
#[derive(Resource)]
pub struct BillboardAssets {
    pub quad: Handle<Mesh>,
    pub materials: Vec<Handle<BillboardMaterial>>,
}

impl FromWorld for BillboardAssets {
    fn from_world(world: &mut World) -> Self {
        // Collect the per-variation specs under immutable borrows first —
        // the asset stores are then borrowed mutably one at a time.
        let specs: Vec<_> = world
            .resource::<BoidVariations>()
            .0
            .iter()
            .map(|variation| {
                (
                    // Quad side equals the bake frustum's width so the
                    // billboard is exactly as large on screen as the mesh
                    // it replaces — no pop.
                    2.0 * variation.radius * atlas::FIT_MARGIN,
                    variation.center,
                )
            })
            .collect();

        let server = world.resource::<AssetServer>().clone();
        let catalog = world.resource::<BoidVariations>().0.clone();
        // The tuning is initialised before this resource in the plugin, so
        // freshly created materials already carry the slider value; the
        // sync system only has to cover later edits.
        let brightness = world.resource::<BillboardTuning>().brightness;
        let quad = world
            .resource_mut::<Assets<Mesh>>()
            .add(Plane3d::default().mesh().size(1.0, 1.0));
        let mut materials = world.resource_mut::<Assets<BillboardMaterial>>();
        let handles = catalog
            .iter()
            .zip(specs)
            .map(|(variation, (span, center))| {
                materials.add(BillboardMaterial {
                    atlas: server.load(format!("impostors/{}.png", variation.name)),
                    // xyz: bake centre; w: quad side in metres.
                    center_span: center.extend(span),
                    brightness,
                })
            })
            .collect();
        Self {
            // ±0.5 corners in xz; the shader builds the camera-facing quad
            // from them directly.
            quad,
            materials: handles,
        }
    }
}

/// One camera-facing quad sampling a variation's impostor atlas. Uniform
/// layout mirrors `assets/shaders/impostor_billboard.wgsl`. Lighting comes
/// from the engine's `lights` view binding (ambient + directional sun,
/// premultiplied by illuminance), so billboards and PBR meshes share one
/// lighting authority — F1 sun edits and any future day/night state apply
/// to both from one source. The one material-side lighting input is the
/// scalar [`BillboardTuning::brightness`] match knob (see
/// [`sync_billboard_brightness`]).
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct BillboardMaterial {
    /// xyz: the variation's bake centre in model-local space (usually
    /// zero; matters once models lean off their origin); w: quad side in
    /// world metres (the bake frustum width).
    #[uniform(0)]
    pub center_span: Vec4,
    /// Multiplier on the light sum; 1.0 = PBR parity.
    #[uniform(1)]
    pub brightness: f32,
    /// The baked atlas: per-pose albedo bands in the top half, mirrored
    /// view-space normals in the bottom half.
    #[texture(2, dimension = "2d")]
    #[sampler(3)]
    pub atlas: Handle<Image>,
}

impl Material for BillboardMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/impostor_billboard.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/impostor_billboard.wgsl".into()
    }
    // Cutout transparency: keeps billboards in the opaque pass where
    // automatic instancing holds (one draw per variation). Blended quads
    // land in the transparent phase, whose per-instance sort defeats the
    // batching — the flat-scene bench measured that at 5k boids as 35.8
    // fps blended vs 70.0 cutout vs 60.0 full meshes, dev profile. The
    // fragment shader performs the discard itself (see
    // impostor_billboard.wgsl — `Mask` only routes the pipeline; a custom
    // shader that never discards writes the atlas's black background).
    // Soft edges without losing instancing: switch to
    // `AlphaMode::AlphaToCoverage` + `Msaa::Sample4` on the camera
    // (measured 70.2 fps — same speed, per-pixel alpha via MSAA coverage,
    // at the cost of 4× MSAA on the whole scene).
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Mask(0.5)
    }
    // The normal half of the atlas already encodes what a prepass would
    // sample, and a billboard shadow would be a rectangle.
    fn enable_prepass() -> bool {
        false
    }
    fn enable_shadows() -> bool {
        false
    }
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // The quad spins around the view axis with the folded yaw; skipping
        // face culling makes its winding irrelevant.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Packs yaw (radians, any value) and pose index into a [`MeshTag`].
/// Low 16 bits: yaw as a fraction of τ (1/65536 of a turn ≈ 0.0055°);
/// bits 16..24: pose index. [`update_billboard_yaw`] rewrites the yaw
/// bits only, a future animation system rewrites the pose bits only.
pub fn pack_tag(yaw: f32, pose: u32) -> u32 {
    let frac = (yaw.rem_euclid(TAU) / TAU * 65536.0) as u32;
    (pose << 16) | (frac & 0xffff)
}

/// The yaw a tag carries, radians. The runtime lookup lives in the shader
/// (the wgsl mirrors `atlas::view_index`); this inverse exists so tests and
/// debugging can read a tag back.
#[cfg_attr(not(test), allow(dead_code))]
pub fn tag_yaw(tag: u32) -> f32 {
    f32::from((tag & 0xffff) as u16) / 65536.0 * TAU
}

/// The boid's facing azimuth (0 = +X, growing toward +Z — the same frame
/// `preprocess::atlas` measures view azimuths in): the formation executor's
/// facing when set (`Target.dir`), else the direction of travel. None when
/// the boid is stationary and facingless — the last packed yaw stands.
fn facing_yaw(target: &Target, vel: &Velocity) -> Option<f32> {
    let dir = if target.dir.length_squared() > 1e-6 {
        target.dir
    } else {
        vel.v
    };
    let horizontal = dir * Vec3::new(1.0, 0.0, 1.0);
    (horizontal.length_squared() > 1e-6).then(|| horizontal.z.atan2(horizontal.x))
}

pub struct BillboardPlugin;

impl Plugin for BillboardPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BillboardTuning>()
            // The catalog (BoidVariations) is built by the same FromWorld
            // constructor the preprocessor uses; BillboardAssets reads it.
            .init_resource::<BoidVariations>()
            .add_plugins(MaterialPlugin::<BillboardMaterial>::default())
            .init_resource::<BillboardAssets>()
            .add_systems(
                Update,
                (
                    swap_boid_lod,
                    update_billboard_yaw,
                    // --billboard-lab's one-shot conversion trails the swap.
                    billboard_lab_convert,
                )
                    .chain(),
            )
            // The brightness slider's push into the shared materials. Runs
            // on any BillboardTuning edit (the swap sliders share the
            // resource — re-pushing a scalar is free).
            .add_systems(
                Update,
                sync_billboard_brightness.run_if(resource_changed::<BillboardTuning>),
            )
            .add_systems(Startup, spawn_billboard_lab.run_if(launch_billboard_lab));
    }
}

/// Run condition for the lab scene: `--billboard-lab` (implies the `--flat`
/// ground and suppresses the normal camera/boid spawn — see main.rs).
pub fn launch_billboard_lab(launch: Res<LaunchConfig>) -> bool {
    launch.billboard_lab
}

/// The test pair: a billboard boid beside its mesh twin, up close. Both
/// stand still (target = own position) at the same yaw (0), so the two
/// render paths show the same soldier under the same sun — move the sun
/// with the F1 sliders (or `--sun-azimuth`/`--sun-elevation`) and compare.
/// The camera parks at 25° from nadir (inside the bake cone), ~7 m out:
/// the sprite fills a healthy slice of the screen.
fn spawn_billboard_lab(mut commands: Commands, variations: Res<BoidVariations>) {
    // Both twins stand still (target = own position) at the same yaw (0),
    // so the two render paths show the same soldier under the same sun —
    // move the sun (F1 sliders, or `--sun-azimuth`/`--sun-elevation`) and
    // compare. The pair is split along Z, not X: this camera looks down
    // the X axis, so Z offsets separate the twins cleanly left/right on
    // screen instead of overlapping them along the view axis — pixel
    // analysis (and eyeballs) get two disjoint blobs. Placement overrides
    // ride `insert` because the bundle's fields are boid.rs-private.
    commands
        .spawn((
            BoidBundle::with_id(0, Target::default(), &variations),
            BillboardLabMesh,
        ))
        .insert(Transform::from_xyz(0.0, 0.5, -1.5))
        .insert(Target {
            pos: Vec3::new(0.0, 0.5, -1.5),
            ..default()
        });
    commands
        .spawn((
            BoidBundle::with_id(0, Target::default(), &variations),
            BillboardLabSprite,
        ))
        .insert(Transform::from_xyz(0.0, 0.5, 1.5))
        .insert(Target {
            pos: Vec3::new(0.0, 0.5, 1.5),
            ..default()
        });

    let distance = 7.0;
    let polar = 25.0_f32.to_radians();
    commands.spawn((
        Camera3d::default(),
        // Plain camera, not an RtsCamera: the input systems that expect
        // exactly one RTS camera find none and stand down, and the bench's
        // zoom pinning doesn't touch this pose. 25° from nadir (inside the
        // bake cone), on the +X side, looking at the pair.
        Transform::from_xyz(distance * polar.sin(), distance * polar.cos(), 0.0)
            .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));
}

/// Mesh twin (z = −1.5, screen right): stays on its full mesh — the
/// lighting reference.
#[derive(Component)]
pub struct BillboardLabMesh;

/// Billboard twin (z = +1.5, screen left): converted to its billboard
/// once, first frame it can.
#[derive(Component)]
pub struct BillboardLabSprite;

/// One-shot lab conversion (the `Local` guard): swaps the marked twin to
/// its billboard through the same `attach_billboard` path as the LOD swap.
fn billboard_lab_convert(
    sprites: Query<(Entity, &Boid, &Target, &Velocity), With<BillboardLabSprite>>,
    assets: Res<BillboardAssets>,
    mut done: Local<bool>,
    mut commands: Commands,
) {
    if *done {
        return;
    }
    for (entity, boid, target, vel) in &sprites {
        *done = true;
        let yaw = facing_yaw(target, vel).unwrap_or(0.0);
        attach_billboard(&mut commands, entity, boid.id, yaw, &assets);
    }
}

/// Pushes [`BillboardTuning::brightness`] edits into the shared billboard
/// materials — the materials start at the tuning's value (see
/// [`BillboardAssets::from_world`]), so this only has to cover live
/// slider edits. All far boids and the lab twin share one material per
/// variation, so one pass reaches everything on screen.
fn sync_billboard_brightness(
    tuning: Res<BillboardTuning>,
    mut materials: ResMut<Assets<BillboardMaterial>>,
    live: Query<&MeshMaterial3d<BillboardMaterial>>,
) {
    for handle in &live {
        if let Some(mut material) = materials.get_mut(&handle.0) {
            material.brightness = tuning.brightness;
        }
    }
}

/// Detaches/reattaches meshes by camera distance. Runs after the RTS
/// camera settles so the distance is measured against the smoothed view
/// position, not the pre-smooth one.
pub fn swap_boid_lod(
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    boids: Query<(Entity, &Boid, &Transform, Option<&Billboard>)>,
    facing: Query<(&Target, &Velocity)>,
    variations: Res<BoidVariations>,
    assets: Res<BillboardAssets>,
    tuning: Res<BillboardTuning>,
    debug: Res<DebugConfig>,
    mut commands: Commands,
) {
    if !debug.impostor_lod {
        return;
    }
    let Ok((_, camera)) = cameras.single() else {
        return;
    };
    let camera_pos = camera.translation();
    let swap_at = tuning.swap_distance_m * tuning.swap_distance_m;
    let restore_at = (tuning.swap_distance_m - tuning.hysteresis_m).max(0.0).powi(2);

    for (entity, boid, transform, billboard) in &boids {
        let distance_sq = camera_pos.distance_squared(transform.translation);
        match billboard {
            // Meshed and far: detach the mesh, attach the billboard quad
            // with this variation's atlas.
            None if distance_sq > swap_at => {
                let yaw = facing
                    .get(entity)
                    .ok()
                    .and_then(|(target, vel)| facing_yaw(target, vel))
                    .unwrap_or(0.0);
                attach_billboard(&mut commands, entity, boid.id, yaw, &assets);
            }
            // Billmarked and near: restore the exact mesh/material handles
            // the spawn attached (identity is the id — see variation_for).
            Some(_) if distance_sq < restore_at => {
                let variation = &variations.0[variation_for(boid.id)];
                commands
                    .entity(entity)
                    .remove::<(Billboard, MeshTag, Mesh3d, MeshMaterial3d<BillboardMaterial>)>()
                    .insert((
                        Mesh3d(variation.mesh.clone()),
                        MeshMaterial3d(variation.material.clone()),
                    ));
            }
            // Inside the hysteresis band: hold the current state.
            _ => {}
        }
    }
}

/// Detaches the mesh and attaches the billboard render components for
/// `boid_id`'s variation — the one conversion both the distance swap and
/// the manual `--force-billboards` bench override go through.
fn attach_billboard(
    commands: &mut Commands,
    entity: Entity,
    boid_id: u32,
    yaw: f32,
    assets: &BillboardAssets,
) {
    commands
        .entity(entity)
        .remove::<(Mesh3d, MeshMaterial3d<StandardMaterial>)>()
        .insert((
            Mesh3d(assets.quad.clone()),
            MeshMaterial3d(assets.materials[variation_for(boid_id)].clone()),
            MeshTag(pack_tag(yaw, IDLE_POSE)),
            Billboard,
        ));
}

/// One-shot `--force-billboards` bench override: converts every boid once
/// (the `Local` guard) and leaves them there — `main.rs` stands the
/// distance swap down while either force flag is set, so the render path
/// stays under manual control for clean A/B benches.
pub fn force_render(
    launch: Res<LaunchConfig>,
    boids: Query<(Entity, &Boid, &Target, &Velocity), Without<Billboard>>,
    assets: Res<BillboardAssets>,
    mut done: Local<bool>,
    mut commands: Commands,
) {
    if *done || !launch.force_billboards {
        return;
    }
    *done = true;
    for (entity, boid, target, vel) in &boids {
        let yaw = facing_yaw(target, vel).unwrap_or(0.0);
        attach_billboard(&mut commands, entity, boid.id, yaw, &assets);
    }
    info!(
        "billboard: --force-billboards converted {} boid(s) for benching",
        boids.iter().len()
    );
}

/// Keeps billboard yaw current: the formation executor writes facing into
/// `Target.dir`, free boids reveal theirs through velocity. Only the yaw
/// bits are rewritten (pose bits belong to the future animation system),
/// and an unchanged quantized yaw doesn't touch the component — no
/// per-frame re-extraction for boids walking straight.
pub fn update_billboard_yaw(
    mut billboards: Query<(&Target, &Velocity, &mut MeshTag), With<Billboard>>,
) {
    for (target, vel, mut tag) in &mut billboards {
        let Some(yaw) = facing_yaw(target, vel) else {
            continue;
        };
        let merged = (tag.0 & !0xffff) | (pack_tag(yaw, 0) & 0xffff);
        if merged != tag.0 {
            tag.0 = merged;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boid::{BoidBundle, boid_variations};

    #[test]
    fn pack_tag_round_trips_yaw_and_pose() {
        for yaw in [0.0_f32, 0.3, 2.9, -0.7, TAU - 0.001, 100.0] {
            // The tag quantizes to 1/65536 of a turn (~0.0055°).
            let quantized = (yaw.rem_euclid(TAU) / TAU * 65536.0).floor() / 65536.0 * TAU;
            assert!((tag_yaw(pack_tag(yaw, IDLE_POSE)) - quantized).abs() < 1e-6);
        }
        for pose in [0u32, 1, 7, 255] {
            assert_eq!((pack_tag(1.0, pose) >> 16) & 0xff, pose);
        }
    }

    #[test]
    fn yaw_updates_preserve_pose_bits() {
        let tag = pack_tag(0.0, 3);
        let updated = (tag & !0xffff) | (pack_tag(2.0, 0) & 0xffff);
        assert_eq!((updated >> 16) & 0xff, 3, "pose bits must survive");
        assert!((tag_yaw(updated) - 2.0).abs() < TAU / 65536.0 + 1e-6);
    }

    #[test]
    fn facing_yaw_prefers_formation_facing_then_velocity() {
        let mut target = Target::default();
        let mut vel = Velocity::default();
        assert_eq!(facing_yaw(&target, &vel), None);

        vel.v = Vec3::new(0.0, 0.0, 1.0);
        assert_eq!(facing_yaw(&target, &vel), Some(90f32.to_radians()));

        target.dir = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(facing_yaw(&target, &vel), Some(0.0));

        // Vertical-only motion carries no facing.
        target.dir = Vec3::ZERO;
        vel.v = Vec3::Y;
        assert_eq!(facing_yaw(&target, &vel), None);
    }

    /// A minimal app with only the swap system, a camera at the origin,
    /// and hand-built boids — the `zoom_app` pattern from player.rs. No
    /// update is burned here; each test drives its own frames.
    fn lod_app() -> (App, Entity, Entity) {
        let mut app = App::new();
        app.init_resource::<DebugConfig>()
            .init_resource::<BillboardTuning>()
            .insert_resource(BoidVariations(boid_variations(
                &mut Assets::<Mesh>::default(),
                &mut Assets::<Image>::default(),
                &mut Assets::<StandardMaterial>::default(),
            )))
            .insert_resource(BillboardAssets {
                quad: Handle::default(),
                materials: vec![Handle::default(); crate::boid::VARIATION_COUNT],
            })
            .add_systems(Update, swap_boid_lod);

        app.world_mut()
            .spawn((Camera3d::default(), GlobalTransform::IDENTITY));
        let near = app
            .world_mut()
            .spawn(BoidBundle::with_target(
                Target::default(),
                Handle::default(),
                Handle::default(),
            ))
            .insert(Transform::from_xyz(5.0, 0.0, 0.0))
            .id();
        let far = app
            .world_mut()
            .spawn(BoidBundle::with_target(
                Target::default(),
                Handle::default(),
                Handle::default(),
            ))
            .insert((Boid { id: 1 }, Transform::from_xyz(0.0, 0.0, 500.0)))
            .id();
        (app, near, far)
    }

    #[test]
    fn far_boids_swap_and_near_boids_keep_their_mesh() {
        let (mut app, near, far) = lod_app();
        app.update();

        let world = app.world();
        assert!(world.get::<Billboard>(far).is_some(), "far boid billboards");
        assert!(world.get::<MeshTag>(far).is_some());
        assert!(world.get::<Mesh3d>(far).is_some(), "quad mesh attached");
        assert!(
            world.get::<MeshMaterial3d<BillboardMaterial>>(far).is_some(),
            "atlas material attached"
        );
        assert!(world.get::<Billboard>(near).is_none(), "near boid stays meshed");
        assert!(world.get::<Mesh3d>(near).is_some());
    }

    #[test]
    fn hysteresis_holds_state_inside_the_band() {
        let (mut app, _near, far) = lod_app();
        // Default: swap at 50 m, restore below 45 m. Park the far boid at
        // 47 m — inside the band — and confirm neither direction flips.
        app.world_mut()
            .entity_mut(far)
            .insert(Transform::from_xyz(0.0, 0.0, 47.0));
        app.update();
        assert!(
            app.world().get::<Billboard>(far).is_none(),
            "47 m from mesh state: still meshed inside the band"
        );

        // Billmarked first, then parked inside the band: stays billmarked.
        let (mut app, _near, far) = lod_app();
        app.update();
        assert!(app.world().get::<Billboard>(far).is_some());
        app.world_mut()
            .entity_mut(far)
            .insert(Transform::from_xyz(0.0, 0.0, 47.0));
        app.update();
        assert!(
            app.world().get::<Billboard>(far).is_some(),
            "47 m from billboard state: still billmarked inside the band"
        );
    }

    #[test]
    fn swap_back_restores_each_boids_own_variation() {
        let (mut app, near, far) = lod_app();
        let catalog = app.world().resource::<BoidVariations>().0.clone();
        let near_before = app
            .world()
            .get::<MeshMaterial3d<StandardMaterial>>(near)
            .expect("near boid meshed")
            .0
            .clone();
        // near has id 0, far id 1 — different variations by construction.
        assert_ne!(
            catalog[variation_for(0)].material,
            catalog[variation_for(1)].material
        );
        app.update();
        assert!(app.world().get::<Billboard>(far).is_some());

        // Walk the far boid back under the restore threshold.
        app.world_mut()
            .entity_mut(far)
            .insert(Transform::from_xyz(0.0, 0.0, 20.0));
        app.update();

        let world = app.world();
        assert!(world.get::<Billboard>(far).is_none());
        let restored = world
            .get::<MeshMaterial3d<StandardMaterial>>(far)
            .expect("mesh material restored");
        assert_eq!(
            restored.0, catalog[variation_for(1)].material,
            "the soldier returns to ITS OWN mesh"
        );
        let kept = world
            .get::<MeshMaterial3d<StandardMaterial>>(near)
            .expect("near boid untouched");
        assert_eq!(kept.0, near_before, "near boid never swapped");
    }

    #[test]
    fn impostor_toggle_disables_the_swap() {
        let (mut app, _near, far) = lod_app();
        app.world_mut()
            .resource_mut::<DebugConfig>()
            .bypass_change_detection()
            .impostor_lod = false;
        app.update();
        assert!(
            app.world().get::<Billboard>(far).is_none(),
            "kill-switch holds meshes"
        );
    }

    /// Brightness is a live material input: a tuning edit must reach the
    /// shared materials the same frame (the slider's whole point — the
    /// materials are seeded from the tuning at creation, this covers edits).
    #[test]
    fn brightness_edits_reach_live_billboard_materials() {
        let mut app = App::new();
        app.init_resource::<BillboardTuning>().add_systems(
            Update,
            sync_billboard_brightness.run_if(resource_changed::<BillboardTuning>),
        );
        let mut materials = Assets::<BillboardMaterial>::default();
        let handle = materials.add(BillboardMaterial {
            center_span: Vec4::default(),
            brightness: 1.0,
            atlas: Handle::default(),
        });
        app.world_mut().insert_resource(materials);
        app.world_mut()
            .spawn(MeshMaterial3d::<BillboardMaterial>(handle.clone()));
        app.update();

        app.world_mut().resource_mut::<BillboardTuning>().brightness = 0.5;
        app.update();
        let material = app
            .world()
            .resource::<Assets<BillboardMaterial>>()
            .get(&handle)
            .unwrap();
        assert_eq!(material.brightness, 0.5, "slider edit pushed live");
    }
}
