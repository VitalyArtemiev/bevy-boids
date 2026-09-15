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

use crate::billboard::{BillboardAssets, attach_billboard, facing_yaw};
use crate::boid::{Boid, BoidBundle, BoidVariations};
use crate::kinematics::Velocity;
use crate::launch::LaunchConfig;
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
}

impl TestScene {
    /// Every scene, for listing and tests.
    pub const ALL: [TestScene; 2] = [TestScene::Billboard, TestScene::Crowd];

    /// The `--scene` value that selects this scene.
    pub fn name(&self) -> &'static str {
        match *self {
            TestScene::Billboard => "billboard",
            TestScene::Crowd => "crowd",
        }
    }

    /// Parses a `--scene` value; `None` = unknown name.
    pub fn from_name(name: &str) -> Option<Self> {
        TestScene::ALL.into_iter().find(|scene| scene.name() == name)
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
