//! Standalone test scenes: `--scene <name>` replaces the normal launch's
//! RTS camera + boid grid with one hand-built micro-scene, running
//! straight into `GameState::Playing` (no main menu — the `--bench`
//! bypass without the bench behaviours). Every scene owns its camera and
//! props; while any scene is active, `setup` in main.rs skips the RTS
//! camera, the grid boids and the obstacle scatter, and
//! `DebugConfig.impostor_lod` stands the billboard auto-swap down so
//! each boid stays on exactly the render path the scene gave it.
//!
//! Adding a scene (keep this list honest):
//! 1. Add a [`TestScene`] variant plus its kebab-case [`TestScene::name`].
//! 2. Write a `Startup` system (`spawn_<name>_scene`) that builds it.
//! 3. Register it in [`ScenePlugin::build`] with
//!    `.run_if(in_scene(TestScene::NewVariant))`.
//! 4. If it needs main-setup accommodations beyond the universal ones
//!    (the billboard scene also asks for the `--flat` ground), gate those
//!    in `setup` on `launch.in_scene(TestScene::NewVariant)`.

use bevy::prelude::*;
use bevy_rts_camera::RtsCamera;
use std::collections::VecDeque;

use crate::billboard::{BillboardAssets, attach_billboard, facing_yaw};
use crate::boid::{Boid, BoidBundle, BoidIds, BoidVariations};
use crate::formations::{Formation, FormationKind, FormationOrder, MemberOf};
use crate::kinematics::Velocity;
use crate::launch::LaunchConfig;
use crate::pbd::Body;
use crate::pbd::Faction;
use crate::target::Target;
use crate::ui::GameState;

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
    /// Two hostile formations marching straight through each other —
    /// contact-only, the clash at formation scale.
    FormationClash,
}

impl TestScene {
    /// Every scene, for listing and tests.
    pub const ALL: [TestScene; 10] = [
        TestScene::Billboard,
        TestScene::Crowd,
        TestScene::Arrival,
        TestScene::Perpendicular,
        TestScene::HeadOn,
        TestScene::Clash,
        TestScene::Shove,
        TestScene::Melee,
        TestScene::FormationCross,
        TestScene::FormationClash,
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
            TestScene::FormationClash => "formation-clash",
        }
    }

    /// Parses a `--scene` value; `None` = unknown name.
    pub fn from_name(name: &str) -> Option<Self> {
        TestScene::ALL.into_iter().find(|scene| scene.name() == name)
    }

    /// Whether the scene wants the flat plane (`scene::spawn_flat_ground`)
    /// instead of any terrain — every movement scene does; only the crowd
    /// experiment wants its rolling hills.
    pub fn wants_flat_ground(&self) -> bool {
        !matches!(self, TestScene::Crowd)
    }
}

/// Run condition factory: true while `--scene <this scene>` is active.
pub fn in_scene(scene: TestScene) -> impl FnMut(Res<LaunchConfig>) -> bool + Clone {
    move |launch: Res<LaunchConfig>| launch.scene == Some(scene)
}

pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        if app.world().resource::<LaunchConfig>().scene.is_some() {
            // A scene runs straight into Playing — no main menu (the same
            // state bypass BenchPlugin uses, minus the bench's pinning,
            // timed exit and FPS logging). Inserted before UiPlugin's
            // `init_state`, which is idempotent and keeps this value, so
            // the menu's OnEnter(MainMenu) pause never happens either.
            app.insert_state(GameState::Playing);
        }
        app.add_systems(
            Startup,
            spawn_billboard_scene.run_if(in_scene(TestScene::Billboard)),
        )
        .add_systems(
            Update,
            // Trails the spawn: the twin entities only exist after the
            // Startup commands apply.
            billboard_scene_convert.run_if(in_scene(TestScene::Billboard)),
        )
        .add_systems(
            Startup,
            spawn_crowd_scene.run_if(in_scene(TestScene::Crowd)),
        )
        // Movement scenes: each pins the shared RTS camera onto its stage
        // once, then the one-shot guard keeps it stable (the camera stays
        // freely controllable afterwards).
        .add_systems(
            Update,
            pin_scene_camera.run_if(resource_exists::<SceneCameraPose>),
        )
        .add_systems(
            Startup,
            (
                spawn_arrival_scene.run_if(in_scene(TestScene::Arrival)),
                spawn_perpendicular_scene.run_if(in_scene(TestScene::Perpendicular)),
                spawn_head_on_scene.run_if(in_scene(TestScene::HeadOn)),
                spawn_clash_scene.run_if(in_scene(TestScene::Clash)),
                spawn_shove_scene.run_if(in_scene(TestScene::Shove)),
                spawn_melee_scene.run_if(in_scene(TestScene::Melee)),
                spawn_formation_cross_scene.run_if(in_scene(TestScene::FormationCross)),
                spawn_formation_clash_scene.run_if(in_scene(TestScene::FormationClash)),
            ),
        );
    }
}

/// The crowd scene's camera: the game's own RTS camera (the experiment's
/// design cone is defined for it, and bench `--cam-angle` pinning targets
/// `RtsCamera`). The armies and their rolling-hill ground are spawned by
/// `CrowdPlugin`'s systems, gated on the same scene — the materials and
/// the shader-module load gate live there.
fn spawn_crowd_scene(mut commands: Commands) {
    crate::spawn_rts_camera(&mut commands);
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

/// One-shot camera pin: focuses the shared RTS camera on the scene's
/// stage. Runs only while a scene inserted the pose resource, and the
/// `Local` guard keeps it to the first frame the camera exists.
fn pin_scene_camera(
    mut cameras: Query<&mut RtsCamera>,
    pose: Res<SceneCameraPose>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    for mut camera in &mut cameras {
        camera.focus.translation = pose.focus;
        camera.target_focus.translation = pose.focus;
        camera.zoom = pose.zoom;
        camera.target_zoom = pose.zoom;
        *done = true;
    }
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
    commands: &mut Commands,
    ids: &mut BoidIds,
    variations: &BoidVariations,
    spec: BoidSpec,
) -> Entity {
    let id = spec.faction as u32 + 3 * ids.next();
    commands
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

/// Spawns a rectangular marching block under one formation with a `Move`
/// order. `colour` decouples looks from allegiance (friendly blocks in a
/// crossing still want distinguishing colours). Members spawn in their
/// facing-frame grid around `origin`; `assign_slots` refines the mapping.
fn spawn_marching_block(
    commands: &mut Commands,
    ids: &mut BoidIds,
    variations: &BoidVariations,
    faction: u8,
    colour: u8,
    origin: Vec3,
    move_to: Vec3,
    columns: usize,
    rows: usize,
) {
    let facing = (move_to - origin).normalize_or_zero();
    let formation = commands
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
    let side = Vec3::new(-facing.z, 0.0, facing.x);
    for row in 0..rows {
        for col in 0..columns {
            let offset = side * (col as f32 - (columns - 1) as f32 / 2.0) * FormationKind::SPACING
                - facing * row as f32 * FormationKind::SPACING;
            let id = colour as u32 + 3 * ids.next();
            commands
                .spawn(BoidBundle::with_id(id, Target::default(), variations))
                .insert(Transform::from_translation(origin + offset + Vec3::Y * 0.5))
                .insert(MemberOf(formation))
                .insert(Faction(faction))
                .insert(Body::default());
        }
    }
}

/// Spawns the shared RTS camera pinned on the stage. `height_m` is the
/// wanted camera height in metres (converted to `RtsCamera` zoom units,
/// where 0.0 = 30 km and 1.0 = `height_min` = 2 m — `--zoom`-style linear
/// interpolation, so 0.99 ≈ 302 m).
fn scene_camera(commands: &mut Commands, focus: Vec3, height_m: f32) {
    crate::spawn_rts_camera(commands);
    let zoom = ((30_000.0 - height_m) / (30_000.0 - 2.0)).clamp(0.0, 1.0);
    commands.insert_resource(SceneCameraPose { focus, zoom });
}

fn spawn_arrival_scene(mut commands: Commands, variations: Res<BoidVariations>, mut ids: ResMut<BoidIds>) {
    spawn_scene_boid(
        &mut commands,
        &mut ids,
        &variations,
        BoidSpec::at(Vec3::new(-20.0, 0.5, 0.0), Vec3::new(20.0, 0.5, 0.0), 0),
    );
    scene_camera(&mut commands, Vec3::new(0.0, 0.0, 0.0), 35.0);
}

fn spawn_perpendicular_scene(mut commands: Commands, variations: Res<BoidVariations>, mut ids: ResMut<BoidIds>) {
    let mut spec = BoidSpec::at(
        Vec3::new(-30.0, 0.5, 0.0),
        Vec3::new(-10.0, 0.5, 25.0),
        0,
    );
    spec.vel = Vec3::new(15.0, 0.0, 0.0);
    spawn_scene_boid(&mut commands, &mut ids, &variations, spec);
    scene_camera(&mut commands, Vec3::new(-15.0, 0.0, 10.0), 45.0);
}

fn spawn_head_on_scene(mut commands: Commands, variations: Res<BoidVariations>, mut ids: ResMut<BoidIds>) {
    spawn_scene_boid(
        &mut commands,
        &mut ids,
        &variations,
        BoidSpec::at(Vec3::new(-30.0, 0.5, 0.0), Vec3::new(30.0, 0.5, 0.0), 0),
    );
    // Same army (green), offset lane: anticipation must braid the pass.
    spawn_scene_boid(
        &mut commands,
        &mut ids,
        &variations,
        BoidSpec::at(Vec3::new(30.0, 0.5, 1.5), Vec3::new(-30.0, 0.5, 1.5), 0),
    );
    scene_camera(&mut commands, Vec3::ZERO, 40.0);
}

fn spawn_clash_scene(mut commands: Commands, variations: Res<BoidVariations>, mut ids: ResMut<BoidIds>) {
    spawn_scene_boid(
        &mut commands,
        &mut ids,
        &variations,
        BoidSpec::at(Vec3::new(-25.0, 0.5, 0.0), Vec3::new(25.0, 0.5, 0.0), 0),
    );
    spawn_scene_boid(
        &mut commands,
        &mut ids,
        &variations,
        BoidSpec::at(Vec3::new(25.0, 0.5, 0.0), Vec3::new(-25.0, 0.5, 0.0), 1),
    );
    scene_camera(&mut commands, Vec3::ZERO, 35.0);
}

fn spawn_shove_scene(mut commands: Commands, variations: Res<BoidVariations>, mut ids: ResMut<BoidIds>) {
    let mut heavy = BoidSpec::at(
        Vec3::new(-20.0, 0.5, 0.0),
        Vec3::new(20.0, 0.5, 0.0),
        0,
    );
    heavy.body = Body {
        radius_m: 0.5,
        mass_kg: 8.0,
    };
    spawn_scene_boid(&mut commands, &mut ids, &variations, heavy);
    spawn_scene_boid(
        &mut commands,
        &mut ids,
        &variations,
        BoidSpec::at(Vec3::new(20.0, 0.5, 0.0), Vec3::new(-20.0, 0.5, 0.0), 1),
    );
    scene_camera(&mut commands, Vec3::ZERO, 35.0);
}

fn spawn_melee_scene(mut commands: Commands, variations: Res<BoidVariations>, mut ids: ResMut<BoidIds>) {
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
                spawn_scene_boid(
                    &mut commands,
                    &mut ids,
                    &variations,
                    BoidSpec::at(pos, -pos, faction),
                );
            }
        }
    }
    scene_camera(&mut commands, Vec3::ZERO, 60.0);
}

fn spawn_formation_cross_scene(mut commands: Commands, variations: Res<BoidVariations>, mut ids: ResMut<BoidIds>) {
    // One army, two blocks with distinguishing colours, courses crossing
    // at the centre — friendly throughout, so members of different blocks
    // anticipate while slot-keeping holds each block together.
    spawn_marching_block(
        &mut commands,
        &mut ids,
        &variations,
        0,
        0,
        Vec3::new(-45.0, 0.5, 0.0),
        Vec3::new(45.0, 0.5, 0.0),
        4,
        3,
    );
    spawn_marching_block(
        &mut commands,
        &mut ids,
        &variations,
        0,
        1,
        Vec3::new(0.0, 0.5, 45.0),
        Vec3::new(0.0, 0.5, -45.0),
        4,
        3,
    );
    scene_camera(&mut commands, Vec3::ZERO, 75.0);
}

fn spawn_formation_clash_scene(mut commands: Commands, variations: Res<BoidVariations>, mut ids: ResMut<BoidIds>) {
    // Hostile blocks on the same lane, marching straight through each
    // other: no anticipation between armies, contact only — the melee at
    // formation scale.
    spawn_marching_block(
        &mut commands,
        &mut ids,
        &variations,
        0,
        0,
        Vec3::new(-40.0, 0.5, 0.0),
        Vec3::new(40.0, 0.5, 0.0),
        4,
        3,
    );
    spawn_marching_block(
        &mut commands,
        &mut ids,
        &variations,
        2,
        2,
        Vec3::new(40.0, 0.5, 0.0),
        Vec3::new(-40.0, 0.5, 0.0),
        4,
        3,
    );
    scene_camera(&mut commands, Vec3::ZERO, 75.0);
}

/// The flat-plane stand-in ground for `--flat` render-path benches and
/// the billboard scene: nothing but the units should cost GPU time.
/// (Registered from main.rs alongside `DemoTerrain`, not scene-gated,
/// because `--flat` is not a scene.)
pub fn spawn_flat_ground(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    const GROUND_M: f32 = 800.0;
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(GROUND_M, GROUND_M))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.40, 0.42, 0.31),
            ..default()
        })),
        Transform::IDENTITY,
    ));
}

/// Builds the billboard scene's test pair: a full mesh capsule (screen
/// LEFT, z = +1.5) beside its billboard twin (screen RIGHT, z = −1.5),
/// up close under the same sun. Both stand still (target = own position)
/// at the same yaw (0), so the two render paths show the same soldier.
/// The pair splits along Z because this camera looks down the X axis: Z
/// offsets separate the twins cleanly left/right on screen instead of
/// overlapping them along the view axis. Placement overrides ride
/// `insert` because the bundle's fields are boid.rs-private.
fn spawn_billboard_scene(mut commands: Commands, variations: Res<BoidVariations>) {
    commands
        .spawn((
            BoidBundle::with_id(0, Target::default(), &variations),
            BillboardSceneMesh,
        ))
        .insert(Transform::from_xyz(0.0, 0.5, 1.5))
        .insert(Target {
            pos: Vec3::new(0.0, 0.5, 1.5),
            ..default()
        });
    commands
        .spawn((
            BoidBundle::with_id(0, Target::default(), &variations),
            BillboardSceneSprite,
        ))
        .insert(Transform::from_xyz(0.0, 0.5, -1.5))
        .insert(Target {
            pos: Vec3::new(0.0, 0.5, -1.5),
            ..default()
        });

    let distance = 7.0;
    let polar = 25.0_f32.to_radians();
    commands.spawn((
        Camera3d::default(),
        // Plain camera, not an RtsCamera: the input systems that expect
        // exactly one RTS camera find none and stand down, and the bench's
        // zoom pinning doesn't touch this pose. 25° from nadir (inside
        // the bake cone), on the +X side, looking at the pair.
        Transform::from_xyz(distance * polar.sin(), distance * polar.cos(), 0.0)
            .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));
}

/// Mesh twin (screen LEFT): stays on its full mesh — the lighting
/// reference. The auto-swap is stood down for the whole scene (see the
/// module docs), so it stays a mesh however the camera moves.
#[derive(Component)]
pub struct BillboardSceneMesh;

/// Billboard twin (screen RIGHT): converted to its billboard once, first
/// frame it can.
#[derive(Component)]
pub struct BillboardSceneSprite;

/// One-shot conversion (the `Local` guard): swaps the marked twin to its
/// billboard through the same `attach_billboard` path as the LOD swap.
fn billboard_scene_convert(
    sprites: Query<(Entity, &Boid, &Target, &Velocity), With<BillboardSceneSprite>>,
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
}
