//! Standalone test scenes: `--scene <name>` replaces the normal launch's
//! RTS camera + boid grid with one hand-built micro-scene. Every scene
//! owns its camera and props; while any scene is active, `setup` in
//! main.rs skips the RTS camera, the grid boids and the obstacle scatter,
//! and `DebugConfig.impostor_lod` stands the billboard auto-swap down so
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

/// The available `--scene` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestScene {
    /// The billboard comparison pair: a mesh capsule beside its forced
    /// billboard twin, up close on the flat plane under one sun — move
    /// the sun (`--sun-azimuth`/`--sun-elevation`, or the F1 sliders)
    /// and compare the two render paths.
    Billboard,
}

impl TestScene {
    /// Every scene, for listing and tests.
    pub const ALL: [TestScene; 1] = [TestScene::Billboard];

    /// The `--scene` value that selects this scene.
    pub fn name(&self) -> &'static str {
        match *self {
            TestScene::Billboard => "billboard",
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
        app.add_systems(
            Startup,
            spawn_billboard_scene.run_if(in_scene(TestScene::Billboard)),
        )
        .add_systems(
            Update,
            // Trails the spawn: the twin entities only exist after the
            // Startup commands apply.
            billboard_scene_convert.run_if(in_scene(TestScene::Billboard)),
        );
    }
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
