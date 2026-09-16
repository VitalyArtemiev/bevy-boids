//! Standalone test scenes and the world assembler that swaps them at
//! runtime.
//!
//! A "world" is the full playable set: boids and formations, obstacles,
//! the camera and the ground. Exactly one is loaded at a time — the
//! normal RTS sandbox ([`ActiveScene`](None)) or one test scene
//! ([`ActiveScene`](Some(..)), the `--scene` launch values). Every load,
//! at startup and at runtime alike, goes through [`assemble_world`]: the
//! old world is despawned first, then the requested one spawns. The F4
//! scene menu and the main-menu Scenes picker send [`LoadWorld`] requests
//! while playing; `--scene <name>` seeds the initial request.
//!
//! While a scene is active it owns the camera rig, props and render
//! paths: `DebugConfig.impostor_lod` stands the billboard auto-swap down
//! so each boid stays on exactly the render path the scene gave it, and
//! the scene gates ([`in_scene`]) follow [`ActiveScene`], so a runtime
//! switch re-targets every scene-gated system exactly like a scene
//! launch. (The camera ENTITY persists across worlds — see
//! [`crate::AppCamera`] — only its rig changes.)
//!
//! Adding a scene (keep this list honest):
//! 1. Add a [`TestScene`] variant plus its kebab-case [`TestScene::name`].
//! 2. Write a `spawn_<name>_scene(world: &mut World, ..)` builder.
//! 3. Call it from [`assemble_scene`]'s match.
//! 4. If it needs ground beyond the universal flat plane (the crowd owns
//!    its rolling hills), branch on it in [`assemble_scene`].

use bevy::prelude::*;
use bevy_rts_camera::{Ground, RtsCamera};
use std::collections::VecDeque;

use crate::billboard::{Billboard, BillboardAssets, BillboardOwner, attach_billboard, facing_yaw};
use crate::boid::{Boid, BoidBundle, BoidIds, BoidVariations};
use crate::crowd::{CrowdArmy, CrowdDust, CrowdGround};
use crate::formations::{Formation, FormationKind, FormationOrder, MemberOf};
use crate::freecam::CameraMode;
use crate::kinematics::Velocity;
use crate::launch::LaunchConfig;
use crate::pbd::{Body, Faction};
use crate::player::Player;
use crate::resources::{Materials, Meshes};
use crate::target::Target;
use crate::terrain::{DemoTerrain, HeightField, Obstacle, ObstacleBundle, WaterPlane};
use crate::ui::GameState;
use crate::ui::debug::DebugConfig;
use crate::ui::radial::RadialMenu;
use rand::Rng;

/// The available `--scene` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestScene {
    /// The billboard comparison pair: a mesh capsule beside its forced
    /// billboard twin, up close on the flat plane under one sun — move
    /// the sun (`--sun-azimuth`/`--sun-elevation`, or the F1 sliders)
    /// and compare the two render paths.
    Billboard,
    /// The formation crowd-shell experiment (src/crowd.rs): two opposing
    /// armies rendered by the procedural crowd shader on rolling test
    /// hills, viewed through the game's RTS camera. Pair with
    /// `--cam-angle`/`--shot` for reproducible captures inside the
    /// experiment's 30° design cone.
    Crowd,
    /// One unit from standstill to a distant point — plain arrival
    /// dynamics (acceleration, momentum, deceleration onto the target).
    Arrival,
    /// One unit already at speed, target off to the side — the orbit
    /// case `follow_target`'s misalignment damping exists to break: the
    /// unit must brake and turn, not circle.
    Perpendicular,
    /// Two friendly units on opposing offset lanes — the long-range
    /// anticipation constraint must sidestep them past each other
    /// without contact and without either slowing to a crawl.
    HeadOn,
    /// The same meeting, hostile: no anticipation, bodies slam into
    /// contact and grind (the "march into the enemy" case).
    Clash,
    /// Hostile head-on with mass asymmetry: the 8x heavier unit bowls
    /// the light one back (inverse-mass contact weighting).
    Shove,
    /// Three factions converging on one point — the free-for-all melee
    /// pile; watch penetration stay bounded and the pile grind.
    Melee,
    /// Two friendly formations on crossing courses — member-level
    /// anticipation must braid the blocks through each other.
    FormationCross,
    /// Two friendly formations HEAD-ON on the same lane — the hard braid
    /// case: unlike [`TestScene::FormationCross`]'s transient 90°
    /// crossing, the conflict zone persists, formation slots pull members
    /// back into the lane, and the block geometry is mirror-symmetric
    /// (no inherent sidestep side). Braiding works for the lone pair of
    /// `head-on`; this scene is the formation-scale stress test.
    FormationBraid,
    /// Two hostile formations marching straight through each other —
    /// contact-only, the clash at formation scale.
    FormationClash,
    /// One formation cycling through every [`FormationKind`] while
    /// marching a triangular course: at each corner-to-corner leg it
    /// reforms into the next kind, marches onto the corner and reorients
    /// toward the following one. The order queue is built from
    /// [`FormationKind::ALL`], so a new formation kind extends the parade
    /// with no scene-code change.
    FormationParade,
}

impl TestScene {
    /// Every scene, for listing and tests.
    pub const ALL: [TestScene; 12] = [
        TestScene::Billboard,
        TestScene::Crowd,
        TestScene::Arrival,
        TestScene::Perpendicular,
        TestScene::HeadOn,
        TestScene::Clash,
        TestScene::Shove,
        TestScene::Melee,
        TestScene::FormationCross,
        TestScene::FormationBraid,
        TestScene::FormationClash,
        TestScene::FormationParade,
    ];

    /// The `--scene` value that selects this scene.
    pub fn name(&self) -> &'static str {
        match *self {
            TestScene::Billboard => "billboard",
            TestScene::Crowd => "crowd",
            TestScene::Arrival => "arrival",
            TestScene::Perpendicular => "perpendicular",
            TestScene::HeadOn => "head-on",
            TestScene::Clash => "clash",
            TestScene::Shove => "shove",
            TestScene::Melee => "melee",
            TestScene::FormationCross => "formation-cross",
            TestScene::FormationBraid => "formation-braid",
            TestScene::FormationClash => "formation-clash",
            TestScene::FormationParade => "formation-parade",
        }
    }

    /// Parses a `--scene` value; `None` = unknown name.
    pub fn from_name(name: &str) -> Option<Self> {
        TestScene::ALL
            .into_iter()
            .find(|scene| scene.name() == name)
    }

    /// One-line hover tooltip for the scene picker — the condensed version
    /// of this variant's docs.
    pub fn tooltip(&self) -> &'static str {
        match *self {
            TestScene::Billboard => {
                "A mesh capsule beside its billboard twin under one sun — compare the render paths."
            }
            TestScene::Crowd => {
                "The crowd-shell experiment: two armies drawn by the procedural crowd shader on test hills."
            }
            TestScene::Arrival => {
                "One unit from standstill onto a distant point — plain arrival dynamics."
            }
            TestScene::Perpendicular => {
                "One unit at speed, target off to the side — must brake and turn, not orbit."
            }
            TestScene::HeadOn => {
                "Two friendly units on offset lanes — anticipation sidesteps them past without contact."
            }
            TestScene::Clash => {
                "The same meeting, hostile — no anticipation; bodies slam and grind."
            }
            TestScene::Shove => {
                "Hostile head-on with mass asymmetry — the 8× heavier unit bowls the light one."
            }
            TestScene::Melee => {
                "Three factions converging on one point — the free-for-all pile; penetration stays bounded."
            }
            TestScene::FormationCross => {
                "Two friendly formations crossing at 90° — members braid the blocks through each other."
            }
            TestScene::FormationBraid => {
                "Two friendly formations head-on on one lane — the persistent-conflict braid stress test."
            }
            TestScene::FormationClash => {
                "Two hostile formations marching through each other — clash at formation scale."
            }
            TestScene::FormationParade => {
                "One formation cycling through every formation kind while marching a triangular course."
            }
        }
    }

    /// Whether the scene wants the flat plane ([`spawn_flat_ground`])
    /// instead of any terrain — every movement scene does; only the crowd
    /// experiment wants its rolling hills.
    pub fn wants_flat_ground(&self) -> bool {
        !matches!(self, TestScene::Crowd)
    }
}

/// The world currently loaded: `None` = the normal RTS sandbox,
/// `Some(scene)` = that test scene owns camera, props and ground. Seeded
/// from `--scene` at startup; [`LoadWorld`] requests switch it at runtime.
/// Every [`in_scene`] gate reads this, so runtime switches and scene
/// launches drive the same systems.
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveScene(pub Option<TestScene>);

/// Run condition factory: true while `scene` is the loaded world.
pub fn in_scene(scene: TestScene) -> impl FnMut(Res<ActiveScene>) -> bool + Clone {
    move |active: Res<ActiveScene>| active.0 == Some(scene)
}

/// Run condition: the normal RTS world is loaded. Gates what only makes
/// sense there — the erosion terrain rebuild (a scene installs its own
/// height field, and a stale rebuild would stomp it).
pub fn normal_world(active: Res<ActiveScene>) -> bool {
    active.0.is_none()
}

/// A world (re)load request from UI: the F4 scene menu, the main-menu
/// Scenes picker, or Start (back to the normal world after a scene).
/// [`load_world`] consumes it.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadWorld(pub Option<TestScene>);

/// The historical 4×3 marching block — the default [`SceneUnits`] and the
/// anchor every formation scene's stage geometry is expressed against.
const DEFAULT_UNITS_PER_FORMATION: usize = 12;

/// The formation-parade scene's unit count — and the anchor its course
/// geometry is expressed against.
const DEFAULT_PARADE_UNITS: usize = 20;

/// Unit counts the scene picker sizes its scenes with, set from the F4
/// menu's sliders. `per_formation` sizes the formation scenes' marching
/// blocks (cross, braid, clash): the default lands on the historical 4×3
/// blocks and stages, bigger counts spawn near-square blocks and scale
/// each stage by [`stage_scale`] so the scene's relative layout (gaps and
/// course lengths against the block's size) holds at any count.
/// `parade_units` is the formation-parade scene's single formation.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneUnits {
    pub per_formation: usize,
    pub parade_units: usize,
}

impl Default for SceneUnits {
    fn default() -> Self {
        SceneUnits {
            per_formation: DEFAULT_UNITS_PER_FORMATION,
            parade_units: DEFAULT_PARADE_UNITS,
        }
    }
}

pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        let scene = app.world().resource::<LaunchConfig>().scene;
        // Runtime truth of which world is loaded, plus the initial load
        // request: startup assembles through the same loader as every
        // runtime switch. main.rs runs the Startup copy of `load_world`
        // after `setup` (shared handles first); the Update copy here
        // serves the menus.
        app.insert_resource(ActiveScene(scene))
            .insert_resource(LoadWorld(scene))
            // The world assembler's resource surface (idempotent; main
            // and the tests may provide their own).
            .init_resource::<HeightField>()
            .init_resource::<DebugConfig>()
            .init_resource::<GlobalAmbientLight>()
            .init_resource::<SceneUnits>()
            .add_systems(
                Update,
                (
                    load_world.run_if(resource_exists::<LoadWorld>),
                    pin_scene_camera.run_if(resource_exists::<SceneCameraPose>),
                    billboard_scene_convert.run_if(in_scene(TestScene::Billboard)),
                ),
            );
        if scene.is_some() {
            // A scene runs straight into Playing — no main menu (the same
            // state bypass BenchPlugin uses, minus the bench's pinning,
            // timed exit and FPS logging). Inserted before UiPlugin's
            // `init_state`, which is idempotent and keeps this value, so
            // the menu's OnEnter(MainMenu) pause never happens either.
            app.insert_state(GameState::Playing);
        }
    }
}

/// Consumes a [`LoadWorld`] request through [`assemble_world`], with
/// exclusive world access: the switch is atomic and nothing runs
/// alongside it. NOTE: exclusivity alone does NOT make it safe — the
/// multithreaded executor applies other systems' command buffers only at
/// sync points, never before an exclusive system, so a completed-but-
/// unapplied buffer can still hold entity commands that outlive the
/// teardown (the F4-menu load raced `swap_boid_lod`'s queued LOD
/// conversions into "Entity despawned" panics this way). Three guards
/// cover it: main.rs orders its Update systems `.before(load_world)`
/// (an ordering edge from a deferred system inserts an auto sync point,
/// applying its buffer first), the billboard conversions check
/// entity validity at application time, and so does `Selected`'s
/// on_remove hook — the teardown's own recursive despawns fire it
/// mid-despawn, and its queued indicator cleanup would otherwise apply
/// against the just-despawned entity after this system returns.
pub fn load_world(world: &mut World) {
    let Some(LoadWorld(target)) = world.remove_resource::<LoadWorld>() else {
        return;
    };
    assemble_world(world, target);
}

/// Tears the loaded world down and builds `target` (`None` = the normal
/// RTS sandbox). The single spawn path for startup and runtime loads
/// alike; directly callable from tests.
fn assemble_world(world: &mut World, target: Option<TestScene>) {
    let launch = world.resource::<LaunchConfig>().clone();
    teardown_world(world);

    // Which world is loaded — every in_scene gate follows this.
    world.insert_resource(ActiveScene(target));
    // A scene pins its boids' render paths (the billboard twins must stay
    // put); the distance swap only runs in the normal world, unless a
    // bench flag forces a render path by hand.
    let forced_render = launch.force_meshes || launch.force_billboards;
    world.resource_mut::<DebugConfig>().impostor_lod = !forced_render && target.is_none();

    match target {
        Some(scene) => assemble_scene(world, scene),
        None => assemble_normal_world(world, &launch),
    }
}

/// Despawns every world-owned entity — boids (with their selection
/// indicator children), formations, obstacles, the three ground kinds,
/// the crowd props — and resets the per-world UI state. Shared resources
/// (settings, asset handles, tuning) survive; height fields are installed
/// by whichever world assembles next. The camera is NOT world-owned: the
/// app's single camera (see [`crate::AppCamera`]) persists across worlds —
/// bevy_egui's primary context rides it for the app's lifetime, and
/// moving or despawning it breaks egui (menus, gates, the works).
fn teardown_world(world: &mut World) {
    let mut doomed = world.query_filtered::<Entity, Or<(
        With<Boid>,
        With<Formation>,
        With<Obstacle>,
        With<Ground>,
        With<WaterPlane>,
        With<FlatGround>,
        With<CrowdGround>,
        With<CrowdArmy>,
        With<CrowdDust>,
    )>>();
    let doomed: Vec<Entity> = doomed.iter(world).collect();
    for entity in doomed {
        // Recursive: takes relationship children (selection indicators)
        // along, firing their cleanup hooks.
        let _ = world.despawn(entity);
    }

    // One-shots and menus that referenced the old world.
    world.remove_resource::<SceneCameraPose>();
    world.remove_resource::<RadialMenu>();
    // Fresh camera contract: each world reconfigures the app camera
    // (fresh RTS state, no freecam takeover). Re-inserting also re-arms
    // `apply_camera_mode` harmlessly.
    world.insert_resource(CameraMode::Rts);
    // Drag/selection state referenced despawned entities.
    world.insert_resource(Player::default());
}

/// Builds a test scene: its ground first (every scene but the crowd runs
/// on the flat plane; the crowd builds its rolling hills and installs the
/// matching height field), then its props and camera. Scenes always run
/// the app camera BARE — no launch dressing (bloom/atmosphere) — matching
/// what a direct `--scene` launch looks like.
fn assemble_scene(world: &mut World, scene: TestScene) {
    let variations = world.resource::<BoidVariations>().clone();
    // Fresh id counter: scene ids pick faction colours by spawn order
    // (`id = faction + 3·seq`), so a reload is bit-identical to the first
    // load.
    let mut ids = BoidIds::default();
    let units = world.resource::<SceneUnits>().clone();

    crate::strip_camera_dressing(world);

    if scene.wants_flat_ground() {
        world.insert_resource(HeightField::default());
        spawn_flat_ground(world);
    } else {
        crate::crowd::spawn_crowd_ground(world);
    }

    match scene {
        TestScene::Billboard => spawn_billboard_scene(world, &variations),
        TestScene::Crowd => spawn_crowd_scene(world),
        TestScene::Arrival => spawn_arrival_scene(world, &variations, &mut ids),
        TestScene::Perpendicular => spawn_perpendicular_scene(world, &variations, &mut ids),
        TestScene::HeadOn => spawn_head_on_scene(world, &variations, &mut ids),
        TestScene::Clash => spawn_clash_scene(world, &variations, &mut ids),
        TestScene::Shove => spawn_shove_scene(world, &variations, &mut ids),
        TestScene::Melee => spawn_melee_scene(world, &variations, &mut ids),
        TestScene::FormationCross => {
            spawn_formation_cross_scene(world, &variations, &mut ids, units.per_formation)
        }
        TestScene::FormationBraid => {
            spawn_formation_braid_scene(world, &variations, &mut ids, units.per_formation)
        }
        TestScene::FormationClash => {
            spawn_formation_clash_scene(world, &variations, &mut ids, units.per_formation)
        }
        TestScene::FormationParade => {
            spawn_formation_parade_scene(world, &variations, &mut ids, units.parade_units)
        }
    }
    world.insert_resource(ids);
}

/// The normal RTS sandbox: launch-mode ground, the obstacle scatter, the
/// `--boids` grid and the dressed RTS camera (moved here from main's
/// `setup`, so Start after a scene rebuilds the same world).
fn assemble_normal_world(world: &mut World, launch: &LaunchConfig) {
    let variations = world.resource::<BoidVariations>().clone();
    let mut ids = BoidIds::default();
    let field = world.resource::<HeightField>().clone();

    // Ground: the flat plane for `--flat` render-path benches, the
    // erosion square otherwise (`--no-terrain` keeps the world
    // groundless, as at startup). The erosion mesh and height field
    // rebuild from `DemoTerrain` on the next update; obstacles re-seat
    // onto the rebuilt surface per frame.
    if launch.flat {
        world.insert_resource(HeightField::default());
        spawn_flat_ground(world);
    } else if launch.terrain {
        crate::terrain::spawn_erosion_ground(world);
        crate::terrain::demo::spawn_water(world);
        world.resource_mut::<DemoTerrain>().dirty = true;
    }

    // Obstacle scatter: nothing but the units should cost render time on
    // the isolation grounds (`--flat` and every scene).
    if !launch.flat {
        let cube = world.resource::<Meshes>().cube.clone();
        let black = world.resource::<Materials>().black.clone();
        for _ in 1..100 {
            let mut rng = rand::rng();
            let x = rng.random_range(-100.0..100.0);
            let z = rng.random_range(-100.0..100.0);
            // Obstacles sit on the terrain surface (cube is 1 m, so +0.5
            // to its centre).
            let y = field.height(x, z) + 0.5;
            world.spawn(ObstacleBundle::new(
                cube.clone(),
                black.clone(),
                Vec3::from_array([1.0, 0.0, 0.0]),
                Vec3::from_array([x, y, z]),
            ));
        }
    }

    // `--boids` works in normal launches too; the default is the full
    // historical 99x99 grid.
    let mut boids_spawned = 0;
    'grid: for i in 1..100 {
        for j in 1..100 {
            if boids_spawned >= launch.boids {
                break 'grid;
            }
            boids_spawned += 1;
            world.spawn(BoidBundle::with_id(
                ids.next(),
                Target {
                    pos: Vec3::from_array([(i - 50) as f32, 0.0, (j - 50) as f32]),
                    ..default()
                },
                &variations,
            ));
        }
    }

    crate::reset_app_camera_rts(world);
    crate::apply_camera_dressing(world, launch);
    world.insert_resource(ids);
}

/// The crowd scene's camera: the game's own RTS camera (the experiment's
/// design cone is defined for it, and bench `--cam-angle` pinning targets
/// `RtsCamera`), reconfigured fresh on the persistent app camera.
fn spawn_crowd_scene(world: &mut World) {
    crate::reset_app_camera_rts(world);
}

// ---- Movement scenes -------------------------------------------------------
//
// Shared plumbing: every movement scene runs on the flat plane through the
// game's full pipeline (planner -> move_step -> pbd_contact, formations on
// FixedUpdate), with the RTS camera pinned once onto its stage. Boid ids
// double as faction colours: `id % 3` picks the variation, so a faction
// whose ids are `faction + 3n` wears one colour (0 red, 1 green, 2 blue).

/// Where a movement scene wants its camera (focus point + zoom), consumed
/// once by [`pin_scene_camera`].
#[derive(Resource, Default)]
struct SceneCameraPose {
    focus: Vec3,
    zoom: f32,
}

/// One-shot camera pin per world load: focuses the shared RTS camera on
/// the scene's stage, then drops the request. Runs while the pose resource
/// exists; the camera may arrive a frame after the load.
fn pin_scene_camera(
    mut cameras: Query<&mut RtsCamera>,
    pose: Res<SceneCameraPose>,
    mut commands: Commands,
) {
    if cameras.is_empty() {
        return;
    }
    for mut camera in &mut cameras {
        camera.focus.translation = pose.focus;
        camera.target_focus.translation = pose.focus;
        camera.zoom = pose.zoom;
        camera.target_zoom = pose.zoom;
    }
    commands.remove_resource::<SceneCameraPose>();
}

/// One unit spec for [`spawn_scene_boid`].
struct BoidSpec {
    pos: Vec3,
    vel: Vec3,
    target: Vec3,
    faction: u8,
    body: Body,
}

impl BoidSpec {
    fn at(pos: Vec3, target: Vec3, faction: u8) -> Self {
        BoidSpec {
            pos,
            vel: Vec3::ZERO,
            target,
            faction,
            body: Body::default(),
        }
    }
}

fn scene_velocity(v: Vec3) -> Velocity {
    let mut vel = Velocity::default();
    vel.v = v;
    vel
}

/// Spawns one boid whose visual variation matches its faction
/// (`id = faction + 3·seq` keeps one colour per side).
fn spawn_scene_boid(
    world: &mut World,
    ids: &mut BoidIds,
    variations: &BoidVariations,
    spec: BoidSpec,
) -> Entity {
    let id = spec.faction as u32 + 3 * ids.next();
    world
        .spawn(BoidBundle::with_id(
            id,
            Target {
                pos: spec.target,
                ..default()
            },
            variations,
        ))
        .insert(Transform::from_translation(spec.pos))
        .insert(scene_velocity(spec.vel))
        .insert(Faction(spec.faction))
        .insert(spec.body)
        .id()
}

/// Spawns a marching block of `count` units under one formation with a
/// `Move` order. The block is near-square — the default 12 lands on
/// exactly the historical 4×3 (4 columns, 3 rows) — and spawns exactly
/// `count` units; a tail count that doesn't tile the grid closes up ragged
/// at the rear rank. `colour` decouples looks from allegiance (friendly
/// blocks in a crossing still want distinguishing colours). Members spawn
/// in their facing-frame grid around `origin`; `assign_slots` refines the
/// mapping.
fn spawn_marching_block(
    world: &mut World,
    ids: &mut BoidIds,
    variations: &BoidVariations,
    faction: u8,
    colour: u8,
    origin: Vec3,
    move_to: Vec3,
    count: usize,
) {
    if count == 0 {
        return;
    }
    let facing = (move_to - origin).normalize_or_zero();
    let columns = ((count as f32).sqrt().ceil() as usize).max(1);
    let formation = world
        .spawn((
            Formation {
                kind: FormationKind::Grid,
                columns: Some(columns),
                dir: facing,
                tasks: VecDeque::from([FormationOrder::Move {
                    pos: move_to,
                    facing_dir: facing,
                }]),
                ..default()
            },
            Transform::from_translation(origin),
        ))
        .id();
    spawn_block_members(
        world, ids, variations, formation, faction, colour, origin, facing, count,
    );
}

/// Spawns `count` boid members for `formation` in the near-square
/// facing-frame grid around `origin` that [`spawn_marching_block`] lays
/// out (see its docs for the tail-rank and `colour` conventions) — the
/// shared placement half, for formations that come with their own task
/// queue.
fn spawn_block_members(
    world: &mut World,
    ids: &mut BoidIds,
    variations: &BoidVariations,
    formation: Entity,
    faction: u8,
    colour: u8,
    origin: Vec3,
    facing: Vec3,
    count: usize,
) {
    if count == 0 {
        return;
    }
    let columns = ((count as f32).sqrt().ceil() as usize).max(1);
    let rows = count.div_ceil(columns);
    let side = Vec3::new(-facing.z, 0.0, facing.x);
    let mut spawned = 0;
    'block: for row in 0..rows {
        for col in 0..columns {
            if spawned >= count {
                break 'block;
            }
            spawned += 1;
            let offset = side * (col as f32 - (columns - 1) as f32 / 2.0) * FormationKind::SPACING
                - facing * row as f32 * FormationKind::SPACING;
            let id = colour as u32 + 3 * ids.next();
            world
                .spawn(BoidBundle::with_id(id, Target::default(), variations))
                .insert(Transform::from_translation(origin + offset + Vec3::Y * 0.5))
                .insert(MemberOf(formation))
                .insert(Faction(faction))
                .insert(Body::default());
        }
    }
}

/// Stage-side scale for a formation scene loaded with `count` units per
/// block: the block's side grows as √count, so course lengths and camera
/// heights grow the same way — every count keeps the scene's relative
/// layout, and the default count is exactly the historical stage (1.0).
fn stage_scale(count: usize) -> f32 {
    (count as f32 / DEFAULT_UNITS_PER_FORMATION as f32).sqrt()
}

/// Scales a formation scene's stage point on x/z (y is the 0.5 ground
/// offset, not a stage distance).
fn stage_point(p: Vec3, k: f32) -> Vec3 {
    Vec3::new(p.x * k, p.y, p.z * k)
}

/// Points the app camera at the stage as a fresh RTS camera, pinned
/// through the usual [`SceneCameraPose`] one-shot. `height_m` is the
/// wanted camera height in metres (converted to `RtsCamera` zoom units,
/// where 0.0 = 30 km and 1.0 = `height_min` = 2 m — `--zoom`-style linear
/// interpolation, so 0.99 ≈ 302 m).
fn scene_camera(world: &mut World, focus: Vec3, height_m: f32) {
    crate::reset_app_camera_rts(world);
    let zoom = ((30_000.0 - height_m) / (30_000.0 - 2.0)).clamp(0.0, 1.0);
    world.insert_resource(SceneCameraPose { focus, zoom });
}

fn spawn_arrival_scene(world: &mut World, variations: &BoidVariations, ids: &mut BoidIds) {
    spawn_scene_boid(
        world,
        ids,
        variations,
        BoidSpec::at(Vec3::new(-20.0, 0.5, 0.0), Vec3::new(20.0, 0.5, 0.0), 0),
    );
    scene_camera(world, Vec3::new(0.0, 0.0, 0.0), 35.0);
}

fn spawn_perpendicular_scene(world: &mut World, variations: &BoidVariations, ids: &mut BoidIds) {
    let mut spec = BoidSpec::at(Vec3::new(-30.0, 0.5, 0.0), Vec3::new(-10.0, 0.5, 25.0), 0);
    spec.vel = Vec3::new(15.0, 0.0, 0.0);
    spawn_scene_boid(world, ids, variations, spec);
    scene_camera(world, Vec3::new(-15.0, 0.0, 10.0), 45.0);
}

fn spawn_head_on_scene(world: &mut World, variations: &BoidVariations, ids: &mut BoidIds) {
    spawn_scene_boid(
        world,
        ids,
        variations,
        BoidSpec::at(Vec3::new(-30.0, 0.5, 0.0), Vec3::new(30.0, 0.5, 0.0), 0),
    );
    // Same army (green), offset lane: anticipation must braid the pass.
    spawn_scene_boid(
        world,
        ids,
        variations,
        BoidSpec::at(Vec3::new(30.0, 0.5, 1.5), Vec3::new(-30.0, 0.5, 1.5), 0),
    );
    scene_camera(world, Vec3::ZERO, 40.0);
}

fn spawn_clash_scene(world: &mut World, variations: &BoidVariations, ids: &mut BoidIds) {
    spawn_scene_boid(
        world,
        ids,
        variations,
        BoidSpec::at(Vec3::new(-25.0, 0.5, 0.0), Vec3::new(25.0, 0.5, 0.0), 0),
    );
    spawn_scene_boid(
        world,
        ids,
        variations,
        BoidSpec::at(Vec3::new(25.0, 0.5, 0.0), Vec3::new(-25.0, 0.5, 0.0), 1),
    );
    scene_camera(world, Vec3::ZERO, 35.0);
}

fn spawn_shove_scene(world: &mut World, variations: &BoidVariations, ids: &mut BoidIds) {
    let mut heavy = BoidSpec::at(Vec3::new(-20.0, 0.5, 0.0), Vec3::new(20.0, 0.5, 0.0), 0);
    heavy.body = Body {
        radius_m: 0.5,
        mass_kg: 8.0,
    };
    spawn_scene_boid(world, ids, variations, heavy);
    spawn_scene_boid(
        world,
        ids,
        variations,
        BoidSpec::at(Vec3::new(20.0, 0.5, 0.0), Vec3::new(-20.0, 0.5, 0.0), 1),
    );
    scene_camera(world, Vec3::ZERO, 35.0);
}

fn spawn_melee_scene(world: &mut World, variations: &BoidVariations, ids: &mut BoidIds) {
    // Three 3-wide ranks per faction, converging through the centre from
    // evenly spaced bearings — every pair hostile, so the pile is pure
    // contact dynamics.
    for (faction, bearing) in (0u8..3).zip([90.0f32, 210.0, 330.0]) {
        let bearing = bearing.to_radians();
        let dir = Vec3::new(bearing.cos(), 0.0, bearing.sin());
        let side = Vec3::new(-dir.z, 0.0, dir.x);
        for row in 0..3 {
            for col in 0..3 {
                let pos = dir * (22.0 - row as f32 * FormationKind::SPACING)
                    + side * (col as f32 - 1.0) * FormationKind::SPACING
                    + Vec3::Y * 0.5;
                // Through the centre and out the far side: a guaranteed
                // three-way pile, not a polite ring.
                spawn_scene_boid(world, ids, variations, BoidSpec::at(pos, -pos, faction));
            }
        }
    }
    scene_camera(world, Vec3::ZERO, 60.0);
}

fn spawn_formation_cross_scene(
    world: &mut World,
    variations: &BoidVariations,
    ids: &mut BoidIds,
    count: usize,
) {
    // One army, two blocks with distinguishing colours, courses crossing
    // at the centre — friendly throughout, so members of different blocks
    // anticipate while slot-keeping holds each block together.
    let k = stage_scale(count);
    spawn_marching_block(
        world,
        ids,
        variations,
        0,
        0,
        stage_point(Vec3::new(-45.0, 0.5, 0.0), k),
        stage_point(Vec3::new(45.0, 0.5, 0.0), k),
        count,
    );
    spawn_marching_block(
        world,
        ids,
        variations,
        0,
        1,
        stage_point(Vec3::new(0.0, 0.5, 45.0), k),
        stage_point(Vec3::new(0.0, 0.5, -45.0), k),
        count,
    );
    scene_camera(world, Vec3::ZERO, 75.0 * k);
}

fn spawn_formation_braid_scene(
    world: &mut World,
    variations: &BoidVariations,
    ids: &mut BoidIds,
    count: usize,
) {
    // One army, head-on, starting CLOSE (20 m between origins at the
    // default count): the blocks meet within a couple of seconds, before
    // they have spooled up — the low-speed braid regime (short
    // anticipation lookahead in metres, contact and slot-keeping
    // relatively stronger). The gap scales with the blocks so bigger
    // counts keep that regime instead of spawning pre-interpenetrated.
    let k = stage_scale(count);
    spawn_marching_block(
        world,
        ids,
        variations,
        0,
        0,
        stage_point(Vec3::new(-10.0, 0.5, 0.0), k),
        stage_point(Vec3::new(10.0, 0.5, 0.0), k),
        count,
    );
    spawn_marching_block(
        world,
        ids,
        variations,
        0,
        1,
        stage_point(Vec3::new(10.0, 0.5, 0.0), k),
        stage_point(Vec3::new(-10.0, 0.5, 0.0), k),
        count,
    );
    scene_camera(world, Vec3::ZERO, 40.0 * k);
}

fn spawn_formation_clash_scene(
    world: &mut World,
    variations: &BoidVariations,
    ids: &mut BoidIds,
    count: usize,
) {
    // Hostile blocks on the same lane, marching straight through each
    // other: no anticipation between armies, contact only — the melee at
    // formation scale.
    let k = stage_scale(count);
    spawn_marching_block(
        world,
        ids,
        variations,
        0,
        0,
        stage_point(Vec3::new(-40.0, 0.5, 0.0), k),
        stage_point(Vec3::new(40.0, 0.5, 0.0), k),
        count,
    );
    spawn_marching_block(
        world,
        ids,
        variations,
        2,
        2,
        stage_point(Vec3::new(40.0, 0.5, 0.0), k),
        stage_point(Vec3::new(-40.0, 0.5, 0.0), k),
        count,
    );
    scene_camera(world, Vec3::ZERO, 75.0 * k);
}

// ---- Formation parade scene -------------------------------------------------
//
// One friendly formation exercising the whole order surface in sequence:
// for every corner-to-corner leg of a triangular course it reforms into
// the next formation kind, marches onto the corner, and reorients toward
// the following one. The queue is derived from FormationKind::ALL and the
// corner list — a new formation kind (one more ALL entry) lengthens the
// parade with no change here.

/// Circumradius of the parade's triangular course at the default unit
/// count: corner-to-corner legs of R·√3 ≈ 60 m.
const PARADE_COURSE_RADIUS_M: f32 = 35.0;

/// Stage-side scale for the parade loaded with `count` units, anchored at
/// [`DEFAULT_PARADE_UNITS`] (see [`stage_scale`]'s rationale): the block's
/// side grows as √count, so the course and camera grow the same way.
fn parade_scale(count: usize) -> f32 {
    (count as f32 / DEFAULT_PARADE_UNITS as f32).sqrt()
}

/// The parade course: the corners of an equilateral triangle centred on
/// the stage origin, corner 0 at +Z ("north"), visited 0 → 1 → 2 → 0.
fn parade_corners(radius_m: f32) -> [Vec3; 3] {
    [90.0f32, 210.0, 330.0].map(|deg| {
        let a = deg.to_radians();
        Vec3::new(a.cos() * radius_m, 0.5, a.sin() * radius_m)
    })
}

/// The parade's task queue: one leg per (formation kind × course corner),
/// so every kind in [`FormationKind::ALL`] gets a full circuit's worth of
/// legs and adding a kind extends the parade. Each leg is the requested
/// triple — change formation to the next kind, march onto this leg's
/// corner (arriving facing along the march), then reorient toward the
/// next corner. The first march starts from `muster` (the spawn origin);
/// "next kind" is relative to `initial_kind`, the kind the block spawns
/// in.
fn parade_orders(
    muster: Vec3,
    corners: &[Vec3],
    initial_kind: FormationKind,
) -> VecDeque<FormationOrder> {
    let kinds = FormationKind::ALL.len();
    let start = FormationKind::ALL
        .iter()
        .position(|&kind| kind == initial_kind)
        .expect("the spawn kind is an ALL member")
        + 1;
    let dir = |from: Vec3, to: Vec3| (to - from).normalize_or_zero();

    let mut tasks = VecDeque::new();
    for leg in 0..kinds * corners.len() {
        let corner = corners[leg % corners.len()];
        let from = if leg == 0 {
            muster
        } else {
            corners[(leg - 1) % corners.len()]
        };
        tasks.push_back(FormationOrder::Reform {
            kind: FormationKind::ALL[(start + leg) % kinds],
            columns: None,
        });
        tasks.push_back(FormationOrder::Move {
            pos: corner,
            facing_dir: dir(from, corner),
        });
        tasks.push_back(FormationOrder::Rotate {
            to: dir(corner, corners[(leg + 1) % corners.len()]),
        });
    }
    tasks
}

/// Spawns the parade: one friendly formation of `count` units mustered at
/// the course's centre facing corner 0, carrying [`parade_orders`]'s
/// whole queue.
fn spawn_formation_parade_scene(
    world: &mut World,
    variations: &BoidVariations,
    ids: &mut BoidIds,
    count: usize,
) {
    let k = parade_scale(count);
    let corners = parade_corners(PARADE_COURSE_RADIUS_M * k);
    let muster = Vec3::new(0.0, 0.5, 0.0);
    let facing = (corners[0] - muster).normalize_or_zero();
    let formation = world
        .spawn((
            Formation {
                kind: FormationKind::Grid,
                dir: facing,
                tasks: parade_orders(muster, &corners, FormationKind::Grid),
                ..default()
            },
            Transform::from_translation(muster),
        ))
        .id();
    spawn_block_members(
        world, ids, variations, formation, 0, 0, muster, facing, count,
    );
    scene_camera(world, Vec3::ZERO, 75.0 * k);
}

/// The flat-plane stand-in ground for `--flat` render-path benches and
/// every scene but the crowd: nothing but the units should cost GPU time.
fn spawn_flat_ground(world: &mut World) {
    const GROUND_M: f32 = 800.0;
    let mesh = world
        .resource_mut::<Assets<Mesh>>()
        .add(Plane3d::default().mesh().size(GROUND_M, GROUND_M));
    let material = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::srgb(0.40, 0.42, 0.31),
            ..default()
        });
    world.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::IDENTITY,
        FlatGround,
    ));
}

/// Marks the flat stand-in ground so the world teardown can despawn it
/// (the erosion ground carries the crate's `Ground` tag instead).
#[derive(Component)]
pub struct FlatGround;

/// Builds the billboard scene's test pair: a full mesh capsule (screen
/// LEFT, z = +1.5) beside its billboard twin (screen RIGHT, z = −1.5),
/// up close under the same sun. Both stand still (target = own position)
/// at the same yaw (0), so the two render paths show the same soldier.
/// The pair splits along Z because this camera looks down the X axis: Z
/// offsets separate the twins cleanly left/right on screen instead of
/// overlapping them along the view axis. Placement overrides ride
/// `insert` because the bundle's fields are boid.rs-private.
fn spawn_billboard_scene(world: &mut World, variations: &BoidVariations) {
    world
        .spawn((
            BoidBundle::with_id(0, Target::default(), variations),
            BillboardSceneMesh,
        ))
        .insert(Transform::from_xyz(0.0, 0.5, 1.5))
        .insert(Target {
            pos: Vec3::new(0.0, 0.5, 1.5),
            ..default()
        });
    world
        .spawn((
            BoidBundle::with_id(0, Target::default(), variations),
            BillboardSceneSprite,
        ))
        .insert(Transform::from_xyz(0.0, 0.5, -1.5))
        .insert(Target {
            pos: Vec3::new(0.0, 0.5, -1.5),
            ..default()
        });

    let distance = 7.0;
    let polar = 25.0_f32.to_radians();
    // Plain camera pose on the app camera, not an RTS rig: the input
    // systems that expect an RTS camera find none and stand down, and the
    // bench's zoom pinning doesn't touch this pose. 25° from nadir
    // (inside the bake cone), on the +X side, looking at the pair.
    crate::configure_app_camera_plain(
        world,
        Transform::from_xyz(distance * polar.sin(), distance * polar.cos(), 0.0)
            .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    );
}

/// Mesh twin (screen LEFT): stays on its full mesh — the lighting
/// reference. The auto-swap is stood down for the whole scene (see the
/// module docs), so it stays a mesh however the camera moves.
#[derive(Component)]
pub struct BillboardSceneMesh;

/// Billboard twin (screen RIGHT): converted to its billboard on the first
/// frame it exists unconverted — the `Without<Billboard>` filter is the
/// guard, so a world reload that respawns the twin reconverts it.
#[derive(Component)]
pub struct BillboardSceneSprite;

/// Converts every unconverted sprite twin through the same
/// `attach_billboard` path as the LOD swap — but as a MANUAL placement:
/// the scene stands the auto-swap down, and even with the F1 toggle
/// flipped back on the swap must leave the twin billboarded (the pair's
/// whole point is showing both render paths side by side).
fn billboard_scene_convert(
    sprites: Query<
        (Entity, &Boid, &Target, &Velocity),
        (With<BillboardSceneSprite>, Without<Billboard>),
    >,
    assets: Res<BillboardAssets>,
    mut commands: Commands,
) {
    for (entity, boid, target, vel) in &sprites {
        let yaw = facing_yaw(target, vel).unwrap_or(0.0);
        attach_billboard(
            &mut commands,
            entity,
            boid.id,
            yaw,
            &assets,
            BillboardOwner::Manual,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::state::app::StatesPlugin;

    /// The scene registry stays parseable: unique kebab names, every one
    /// round-trips through `from_name`, and the flat-ground request is
    /// exactly "everything but the crowd hills".
    #[test]
    fn scene_names_round_trip_and_flat_ground_is_everything_but_crowd() {
        let mut names: Vec<_> = TestScene::ALL.iter().map(|s| s.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), TestScene::ALL.len(), "names must be unique");
        for scene in TestScene::ALL {
            assert_eq!(TestScene::from_name(scene.name()), Some(scene));
        }
        assert!(!TestScene::Crowd.wants_flat_ground());
        for scene in TestScene::ALL {
            assert_eq!(
                scene.wants_flat_ground(),
                scene != TestScene::Crowd,
                "{:?} flat-ground request drifted",
                scene.name()
            );
        }
    }

    /// A scene launch lands in `Playing` straight away (no main menu), and
    /// UiPlugin's later `init_state` — idempotent — must not clobber it.
    #[test]
    fn scene_launch_skips_the_main_menu() {
        let mut scene_app = App::new();
        scene_app
            .add_plugins(StatesPlugin)
            .insert_resource(LaunchConfig {
                scene: Some(TestScene::Billboard),
                ..Default::default()
            })
            .add_plugins(ScenePlugin)
            .init_state::<GameState>();
        assert_eq!(
            *scene_app.world().resource::<State<GameState>>(),
            GameState::Playing
        );

        // Without a scene the default menu state stands.
        let mut plain_app = App::new();
        plain_app
            .add_plugins(StatesPlugin)
            .insert_resource(LaunchConfig::default())
            .add_plugins(ScenePlugin)
            .init_state::<GameState>();
        assert_eq!(
            *plain_app.world().resource::<State<GameState>>(),
            GameState::default()
        );
    }

    /// Headless app with the assembler's resource surface (no rendering):
    /// bare asset stores for the variation catalog and the grounds, the
    /// shared handles, the terrain settings. `ScenePlugin` seeds the
    /// initial `LoadWorld`, so one `update()` builds the launch world
    /// exactly like the game.
    fn world_app(launch: LaunchConfig) -> App {
        let mut app = App::new();
        app.add_plugins(StatesPlugin)
            .insert_resource(launch)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<Image>>()
            .init_resource::<BoidVariations>()
            .init_resource::<Meshes>()
            .init_resource::<Materials>()
            .init_resource::<BoidIds>()
            .init_resource::<Player>()
            .init_resource::<CameraMode>()
            .init_resource::<DemoTerrain>()
            .init_resource::<crate::terrain::TerrainMesh>()
            .add_plugins(ScenePlugin);
        app
    }

    fn count<T: Component>(world: &mut World) -> usize {
        world.query::<&T>().iter(world).count()
    }

    /// Loading a world discards the previous one, whatever it was: a
    /// scene replaces the normal sandbox and vice versa, with the right
    /// camera, ground and bookkeeping following each switch.
    #[test]
    fn world_switches_discard_the_previous_world() {
        let launch = LaunchConfig {
            boids: 4,
            ..Default::default()
        };
        let mut app = world_app(launch);
        app.update(); // consumes the seeded LoadWorld(None)
        let world = app.world_mut();
        assert_eq!(count::<Boid>(world), 4, "the startup grid did not spawn");
        assert_eq!(count::<Obstacle>(world), 99);
        assert_eq!(
            world.resource::<ActiveScene>().0,
            None,
            "the startup world must be the normal sandbox"
        );
        assert_eq!(count::<RtsCamera>(world), 1);

        // Into a scene: the sandbox's boids, obstacles, camera and
        // erosion ground are all gone.
        let world = app.world_mut();
        assemble_world(world, Some(TestScene::FormationClash));
        assert_eq!(count::<Boid>(world), 24, "two 4x3 blocks");
        assert_eq!(count::<Formation>(world), 2);
        assert_eq!(count::<Obstacle>(world), 0, "sandbox obstacles survived");
        assert_eq!(count::<FlatGround>(world), 1);
        assert_eq!(count::<RtsCamera>(world), 1);
        assert!(!world.resource::<DebugConfig>().impostor_lod);

        // Back to the sandbox: the scene's props are gone, the grid,
        // scatter and RTS camera are back.
        let world = app.world_mut();
        assemble_world(world, None);
        assert_eq!(count::<Boid>(world), 4);
        assert_eq!(count::<Formation>(world), 0, "scene formations survived");
        assert_eq!(count::<Obstacle>(world), 99);
        assert_eq!(count::<FlatGround>(world), 0, "scene ground survived");
        assert!(world.resource::<DebugConfig>().impostor_lod);
    }

    /// The formation scenes take their per-block unit count from
    /// [`SceneUnits`]: the default keeps the historical 4×3 blocks (24
    /// units across two), bigger counts spawn exactly that many units per
    /// block (near-square, ragged tail ranks close up short), and the
    /// stage scales with the block so bigger counts don't spawn
    /// pre-interpenetrated.
    #[test]
    fn formation_scenes_take_their_unit_count_from_scene_units() {
        let mut app = world_app(LaunchConfig::default());

        assemble_world(app.world_mut(), Some(TestScene::FormationClash));
        assert_eq!(
            count::<Boid>(app.world_mut()),
            24,
            "the default must keep the historical two 4×3 blocks"
        );

        // 100 per block: two 10×10 blocks on a stage scaled by √(100/12).
        app.world_mut().insert_resource(SceneUnits {
            per_formation: 100,
            ..Default::default()
        });
        assemble_world(app.world_mut(), Some(TestScene::FormationClash));
        let world = app.world_mut();
        assert_eq!(count::<Boid>(world), 200);
        assert_eq!(count::<Formation>(world), 2);
        let k = (100.0_f32 / 12.0).sqrt();
        let mut formations = world.query_filtered::<&Transform, With<Formation>>();
        let mut xs: Vec<f32> = formations.iter(world).map(|t| t.translation.x).collect();
        xs.sort_by(|a, b| a.total_cmp(b));
        assert!(
            (xs[0] + 40.0 * k).abs() < 1e-4,
            "left origin not stage-scaled"
        );
        assert!(
            (xs[1] - 40.0 * k).abs() < 1e-4,
            "right origin not stage-scaled"
        );

        // A count that doesn't tile the near-square grid still spawns
        // exactly (13 = a 4-wide grid with a ragged tail rank).
        app.world_mut().insert_resource(SceneUnits {
            per_formation: 13,
            ..Default::default()
        });
        assemble_world(app.world_mut(), Some(TestScene::FormationCross));
        assert_eq!(count::<Boid>(app.world_mut()), 26);
    }

    /// The parade course is an equilateral triangle centred on the stage
    /// origin, corner 0 at +Z ("north").
    #[test]
    fn parade_course_is_a_centred_equilateral_triangle() {
        let corners = parade_corners(35.0);
        let centre = corners.iter().sum::<Vec3>() / 3.0;
        assert!(
            centre.x.abs() < 1e-4 && centre.z.abs() < 1e-4,
            "centroid drifted: {centre:?}"
        );
        let side = corners[0].distance(corners[1]);
        assert!((side - corners[1].distance(corners[2])).abs() < 1e-4);
        assert!((side - corners[2].distance(corners[0])).abs() < 1e-4);
        assert!((side - 35.0 * 3.0_f32.sqrt()).abs() < 1e-3, "side = R·√3");
        assert!(
            corners[0].z > 0.0 && corners[0].x.abs() < 1e-4,
            "corner 0 must sit on +Z: {:?}",
            corners[0]
        );
    }

    /// The parade queue walks the requested triple — reform into the next
    /// kind, march onto the corner, reorient toward the next one — with
    /// kinds and corners cycling and every kind in `ALL` getting a full
    /// circuit's worth of legs.
    #[test]
    fn parade_orders_cycle_kinds_and_corners() {
        let muster = Vec3::new(0.0, 0.5, 0.0);
        let corners = parade_corners(35.0);
        let tasks = parade_orders(muster, &corners, FormationKind::Grid);
        assert_eq!(tasks.len(), 3 * FormationKind::ALL.len() * corners.len());

        // "Next" is relative to the spawn kind: the cycle starts just
        // past it in ALL.
        let start = FormationKind::ALL
            .iter()
            .position(|&kind| kind == FormationKind::Grid)
            .unwrap()
            + 1;
        let orders: Vec<FormationOrder> = tasks.into_iter().collect();
        let mut reformed_kinds = Vec::new();
        for (leg, chunk) in orders.chunks(3).enumerate() {
            let corner = corners[leg % corners.len()];
            let from = if leg == 0 {
                muster
            } else {
                corners[(leg - 1) % corners.len()]
            };
            let next = corners[(leg + 1) % corners.len()];
            let [
                FormationOrder::Reform { kind, columns },
                FormationOrder::Move { pos, facing_dir },
                FormationOrder::Rotate { to },
            ] = chunk
            else {
                panic!("leg {leg} is not the reform-march-reorient triple");
            };
            assert_eq!(
                *kind,
                FormationKind::ALL[(start + leg) % FormationKind::ALL.len()]
            );
            assert_eq!(*columns, None);
            assert_eq!(*pos, corner);
            assert!(facing_dir.distance((corner - from).normalize()) < 1e-5);
            assert!(to.distance((next - corner).normalize()) < 1e-5);
            reformed_kinds.push(*kind);
        }
        for kind in FormationKind::ALL {
            assert!(
                reformed_kinds.contains(&kind),
                "every formation kind must parade; {kind:?} never reformed into"
            );
        }
    }

    /// The parade scene spawns one formation of `SceneUnits::parade_units`
    /// units carrying the full parade queue — the picker's slider sizes it.
    #[test]
    fn parade_scene_spawns_parade_units_with_orders() {
        let mut app = world_app(LaunchConfig::default());
        assemble_world(app.world_mut(), Some(TestScene::FormationParade));
        {
            let world = app.world_mut();
            assert_eq!(count::<Boid>(world), 20, "default parade unit count");
            assert_eq!(count::<Formation>(world), 1);
            let mut formations = world.query_filtered::<Entity, With<Formation>>();
            let formation = formations.iter(world).next().expect("the parade formation");
            let formation = world.get::<Formation>(formation).unwrap();
            assert_eq!(formation.tasks.len(), 3 * FormationKind::ALL.len() * 3);
            assert!(
                matches!(formation.tasks.front(), Some(FormationOrder::Reform { .. })),
                "the parade must open by changing formation"
            );
        }

        app.world_mut().resource_mut::<SceneUnits>().parade_units = 7;
        assemble_world(app.world_mut(), Some(TestScene::FormationParade));
        assert_eq!(
            count::<Boid>(app.world_mut()),
            7,
            "the slider count applies on load"
        );
    }

    /// The parade executes through the real formation pipeline: the block
    /// reforms, marches onto corner 0 of the course and the queue advances
    /// past the opening leg (the scene assembly test above only pins the
    /// spawn state). Mirrors formations.rs's harness: manual clocks, the
    /// FixedUpdate executor chain, `move_step` in Update.
    #[test]
    fn parade_queue_executes_through_the_formation_pipeline() {
        use crate::formations::{
            FormationTuning, LODGuard, assign_slots, dispatch_formation_goals,
            init_formation_speed, plan_formation_goals, propagate_formation_targets,
            transition_formation_orders,
        };
        use crate::kinematics::{KinematicsTuning, move_step};
        use crate::target::follow_target;
        use bevy::gizmos::AppGizmoBuilder;
        use bevy::gizmos::config::{DefaultGizmoConfigGroup, GizmoConfigStore};
        use bevy::time::{Fixed, Time, TimePlugin, TimeUpdateStrategy};
        use std::time::Duration;

        let mut app = world_app(LaunchConfig {
            scene: Some(TestScene::FormationParade),
            ..Default::default()
        });
        app.add_plugins(TimePlugin)
            .insert_resource(Time::<Fixed>::from_duration(Duration::from_secs_f32(
                1.0 / 60.0,
            )))
            .init_resource::<LODGuard>()
            .init_resource::<FormationTuning>()
            .init_resource::<KinematicsTuning>()
            .init_resource::<GizmoConfigStore>()
            .init_gizmo_group::<DefaultGizmoConfigGroup>()
            .init_resource::<Assets<GizmoAsset>>()
            .add_systems(
                FixedUpdate,
                (
                    init_formation_speed,
                    propagate_formation_targets,
                    transition_formation_orders,
                    assign_slots,
                    plan_formation_goals,
                    dispatch_formation_goals,
                    follow_target,
                )
                    .chain(),
            )
            .add_systems(Update, move_step);
        app.update(); // builds the parade world; clocks not yet advanced

        let corners = parade_corners(PARADE_COURSE_RADIUS_M);
        let com = |world: &mut World| -> Vec3 {
            let mut boids = world.query_filtered::<&Transform, With<Boid>>();
            let (sum, n) = boids.iter(world).fold((Vec3::ZERO, 0usize), |(sum, n), t| {
                (sum + t.translation, n + 1)
            });
            sum / n as f32
        };
        let formation = {
            let world = app.world_mut();
            let mut formations = world.query_filtered::<Entity, With<Formation>>();
            formations.iter(world).next().expect("the parade formation")
        };
        let full_queue = app.world().get::<Formation>(formation).unwrap().tasks.len();

        // One manual step per update (the timestep matches dt).
        let mut step = |app: &mut App| {
            app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
                1.0 / 60.0,
            )));
            app.update();
        };
        let mut reached_corner = false;
        for _ in 0..2400 {
            step(&mut app);
            if !reached_corner && com(app.world_mut()).distance(corners[0]) < 6.0 {
                reached_corner = true;
            }
        }
        assert!(
            reached_corner,
            "the parade never reached corner 0: center of mass {:?}",
            com(app.world_mut())
        );

        let world = app.world_mut();
        let f = world.get::<Formation>(formation).unwrap();
        assert!(
            f.tasks.len() < full_queue,
            "the queue must have advanced past the opening leg"
        );
        assert_ne!(
            f.kind,
            FormationKind::Grid,
            "the opening Reform must have changed the spawn kind"
        );
    }

    /// A live selection must survive a world switch: `Selected`'s
    /// on_remove hook queues a deferred `despawn_related::<Children>`
    /// while `teardown_world` is mid-despawn of that very entity, so the
    /// command applies against an already-despawned entity on the next
    /// update — the billboard→crowd switch panicked exactly there ("Entity
    /// despawned", attributed to `load_world`). The hook skips a gone
    /// entity instead.
    #[test]
    fn world_switches_survive_a_live_selection() {
        use crate::player::{Selected, SelectionGizmo, SelectionIndicator};
        let mut app = world_app(LaunchConfig {
            boids: 4,
            ..Default::default()
        });
        // Indicator children spawn through the shared gizmo asset.
        app.init_resource::<Assets<GizmoAsset>>()
            .init_resource::<SelectionGizmo>();
        app.update(); // the startup world

        // Select two boids; one update lets the insert hook's indicator
        // children spawn (they are deferred, like the removal cleanup).
        let selected: Vec<Entity> = {
            let world = app.world_mut();
            let mut boids = world.query_filtered::<Entity, With<Boid>>();
            let selected: Vec<Entity> = boids.iter(world).take(2).collect();
            for boid in &selected {
                world.entity_mut(*boid).insert(Selected);
            }
            selected
        };
        app.update();
        assert_eq!(count::<SelectionIndicator>(app.world_mut()), 2);

        // Switch worlds: teardown despawns the selected boids, firing the
        // hook mid-despawn; its deferred command applies on this update.
        assemble_world(app.world_mut(), Some(TestScene::Crowd));
        app.update();
        assert_eq!(count::<Selected>(app.world_mut()), 0);
        assert_eq!(
            count::<SelectionIndicator>(app.world_mut()),
            0,
            "indicator children must not outlive their entities"
        );
        assert_eq!(
            count::<CrowdGround>(app.world_mut()),
            1,
            "the switch must still complete into the crowd scene"
        );
    }

    /// Reloading the same scene resets its state: the id counter starts
    /// over (faction colours are spawn-order-derived, so a reload is
    /// bit-identical) and per-world resources snap back.
    #[test]
    fn reloading_a_scene_resets_spawn_state() {
        let mut app = world_app(LaunchConfig::default());
        let ids_of = |world: &mut World| -> Vec<u32> {
            world
                .query::<&Boid>()
                .iter(world)
                .map(|boid| boid.id)
                .collect()
        };
        let world = app.world_mut();
        assemble_world(world, Some(TestScene::Melee));
        let first = ids_of(world);
        assert_eq!(first.len(), 27);

        let world = app.world_mut();
        assemble_world(world, Some(TestScene::Melee));
        assert_eq!(ids_of(world), first, "reload must reproduce the same ids");
        assert_eq!(
            world.resource::<BoidIds>().0,
            27,
            "counter must restart with the reload (one seq per unit)"
        );
    }

    /// A runtime `LoadWorld` request flows through the registered Update
    /// system: one update() consumes it and the resource is gone (no
    /// double-load on the next frame).
    #[test]
    fn load_world_requests_are_consumed_once() {
        let launch = LaunchConfig {
            scene: Some(TestScene::Arrival),
            ..Default::default()
        };
        let mut app = world_app(launch);
        app.update();
        assert!(app.world().get_resource::<LoadWorld>().is_none());
        assert_eq!(count::<Boid>(app.world_mut()), 1);
        assert_eq!(
            app.world().resource::<ActiveScene>().0,
            Some(TestScene::Arrival)
        );

        app.world_mut()
            .insert_resource(LoadWorld(Some(TestScene::Clash)));
        app.update();
        assert_eq!(count::<Boid>(app.world_mut()), 2, "clash has two units");
        assert!(app.world().get_resource::<LoadWorld>().is_none());
    }

    /// The egui primary context must survive any number of world switches.
    /// bevy_egui attaches it to the first camera it sees and never attaches
    /// another (its `egui_context_exists` latch is app-lifetime), and it
    /// must ride a camera for the render pass to extract it — so the app
    /// keeps ONE persistent camera that world switches reconfigure but
    /// never despawn or replace. (The earlier despawn-and-reseat approach
    /// churned egui's internal state — a phantom button click and an
    /// emath "time shouldn't move backwards" panic by the fifth load.)
    #[test]
    fn world_switches_keep_one_persistent_camera_for_egui() {
        use bevy_egui::PrimaryEguiContext;
        let mut app = world_app(LaunchConfig::default());
        app.update(); // the startup world

        // The context as bevy_egui attaches it (PreUpdate, first camera).
        let camera = {
            let world = app.world_mut();
            let camera = world
                .query_filtered::<Entity, With<crate::AppCamera>>()
                .iter(world)
                .next()
                .expect("the app camera");
            world
                .entity_mut(camera)
                .insert(bevy_egui::PrimaryEguiContext)
                .id()
        };

        // Switch worlds repeatedly: the camera entity — and the egui
        // context on it — must never be replaced, while the rig follows
        // each world (plain for the billboard scene, RTS otherwise).
        for target in [
            Some(TestScene::Billboard),
            Some(TestScene::Clash),
            None,
            Some(TestScene::Arrival),
            None,
            Some(TestScene::Billboard),
        ] {
            let world = app.world_mut();
            assemble_world(world, target);
            assert_eq!(
                count::<crate::AppCamera>(world),
                1,
                "the app camera must be singular"
            );
            let mut carriers = world.query_filtered::<Entity, With<PrimaryEguiContext>>();
            assert_eq!(
                carriers.iter(world).count(),
                1,
                "the egui context must stay exactly once"
            );
            assert_eq!(
                carriers.iter(world).next().unwrap(),
                camera,
                "the egui context must stay on the persistent app camera"
            );
            assert_eq!(
                world.get::<RtsCamera>(camera).is_some(),
                target != Some(TestScene::Billboard),
                "the rig must follow the world"
            );
        }
    }

    /// Scenes run the app camera bare: loading any scene from the sandbox
    /// (which dresses the camera per the launch flags — bloom is on by
    /// default) must strip that dressing, exactly like a direct `--scene`
    /// launch whose camera never dressed; returning to the sandbox
    /// re-applies it. The billboard scene is the regression case — its
    /// plain-camera config doesn't touch dressing itself.
    #[test]
    fn scenes_strip_the_launch_camera_dressing() {
        use bevy::post_process::bloom::Bloom;
        let mut app = world_app(LaunchConfig::default());
        app.update(); // normal world: dressed (bloom on by default)
        assert_eq!(
            count::<Bloom>(app.world_mut()),
            1,
            "the sandbox camera must carry the launch dressing"
        );

        assemble_world(app.world_mut(), Some(TestScene::Billboard));
        assert_eq!(
            count::<Bloom>(app.world_mut()),
            0,
            "the billboard scene must run the camera bare"
        );

        assemble_world(app.world_mut(), Some(TestScene::Clash));
        assert_eq!(
            count::<Bloom>(app.world_mut()),
            0,
            "movement scenes must run the camera bare"
        );

        assemble_world(app.world_mut(), None);
        assert_eq!(
            count::<Bloom>(app.world_mut()),
            1,
            "returning to the sandbox must re-apply the launch dressing"
        );
    }
}
