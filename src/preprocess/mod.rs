//! `--preprocess`: bake boid model variations into impostor atlas PNGs.
//!
//! At large camera distances the plan is to drop the mesh and draw one
//! textured billboard per boid. The billboard needs an unlit albedo and a
//! normal for every view direction in a steep-down cone (plus the boid's
//! facing, which [`atlas`] folds into the view azimuth); one disk of
//! pre-rendered views per animation pose per model variation covers all of
//! it.
//!
//! This is a self-contained app mode, never merged into the game's plugin
//! graph: `cargo run -- --preprocess` renders every catalog variation (see
//! `boid::boid_variations` — the game spawns from the same list) in every
//! [`bake_poses`] pose from every [`atlas::view_samples`] direction, with
//! two co-located orthographic cameras (albedo and normals on separate
//! render layers, so both captures happen in the same frame), captures each
//! through the renderer's own screenshot readback, and composites the cells
//! into one zero-waste RGBA PNG per variation under `assets/impostors/` —
//! each pose's albedo cells fill a band of the top half, normal cells sit
//! mirrored through the texture centre, so a runtime shader pairs them with
//! `nuv = 1.0 - uv`.
//!
//! Nothing is lit at bake time: baked lighting would be wrong at every
//! yaw except the one the model was baked at (the folding note in
//! [`atlas`]), so the albedo renders an unlit clone of the game material
//! and the normals render as raw view-space directions for the runtime
//! shader to shade with. Normal cells are stored sRGB-encoded like the
//! albedo — decode with `n = 2 * sampled - 1` after sampling the atlas as
//! an sRGB texture.

pub mod atlas;

use std::path::Path;

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{ClearColorConfig, RenderTarget, ScalingMode};
use bevy::pbr::{Material, MaterialPlugin};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::RenderPlugin;
use bevy::render::mesh::VertexAttributeValues;
use bevy::render::render_resource::{AsBindGroup, Extent3d, TextureDimension, TextureFormat};
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings};
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::shader::ShaderRef;
use bevy::window::{WindowPlugin, WindowResolution};

use crate::boid::boid_variations;
use atlas::ViewSample;

/// Where baked atlases land, under `assets/` so the billboard LOD can
/// `AssetServer::load` them.
const OUTPUT_DIR: &str = "assets/impostors";

/// Render layers of the bake: the albedo and normal models sit at the same
/// place, so each model/camera pair lives on its own layer and the two
/// renders never see each other.
const ALBEDO_LAYER: usize = 1;
const NORMAL_LAYER: usize = 2;

/// Camera distance from the model centre, in fitted-span units. Far enough
/// that near/far planes never clip the model for any view in the cone.
const DISTANCE_SPANS: f32 = 4.0;

/// Far plane, in fitted-span units (the model spans about ±1 span).
const FAR_SPANS: f32 = 7.0;

/// Frames burned between stages — only the mesh swap or pose transform
/// needs to settle.
const RETUNE_WARMUP_FRAMES: u32 = 2;

/// Marker for the two offscreen baking cameras.
#[derive(Component)]
struct PreprocessCamera;

/// Marker for both model entities — the bake swaps their mesh between
/// variations.
#[derive(Component)]
struct PreprocessModel;

/// Marker for the unlit-albedo model, the one whose material varies.
#[derive(Component)]
struct AlbedoModel;

/// Outputs view-space normals (`assets/shaders/impostor_normal.wgsl`); one
/// shared instance for every variation — only the mesh changes between
/// bakes.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Default)]
struct NormalMaterial {}

impl Material for NormalMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/impostor_normal.wgsl".into()
    }
}

/// One bake job derived from a catalog entry: the same mesh the game
/// spawns, with an unlit clone of its material (the bake stores raw
/// albedo; runtime lighting comes from the normal half).
#[derive(Clone)]
struct BakeEntry {
    name: &'static str,
    mesh: Handle<Mesh>,
    albedo: Handle<StandardMaterial>,
    center: Vec3,
    /// Bounding-sphere radius, metres. The frustum is fitted to the sphere,
    /// not the box, so every view direction — and every pose, however the
    /// model is transformed — frames the whole model.
    radius: f32,
}

/// One baked animation state. Only the idle pose ships — the axis (and the
/// stage machine that walks it) is plumbed for any pose count, but no
/// animation exists to feed it yet. Real soldier meshes plug skeletal
/// animation in here (each animation frame becomes a pose) at the same
/// stage boundary: set the model's pose, `POSE_COUNT` in `atlas.rs` and
/// its wgsl mirror, and re-run the bake. Cameras and the frustum stay
/// fixed across poses so all poses of a variation share framing.
struct Pose {
    name: &'static str,
    transform: Transform,
}

fn bake_poses() -> Vec<Pose> {
    vec![Pose {
        name: "idle",
        transform: Transform::IDENTITY,
    }]
}

/// Drives the bake: which view is in flight, what has landed.
#[derive(Resource)]
struct PreprocessState {
    albedo_target: Handle<Image>,
    normal_target: Handle<Image>,
    entries: Vec<BakeEntry>,
    poses: Vec<Pose>,
    /// Current variation / pose stage.
    current: usize,
    pose: usize,
    samples: Vec<ViewSample>,
    /// Flat `pose · views + view` capture storage for the current stage
    /// block (all poses of the current variation).
    captured_albedo: Vec<Option<Image>>,
    captured_normals: Vec<Option<Image>>,
    /// Next sample to dispatch within the current pose; strictly
    /// serialised so both of a view's captures are confirmed before the
    /// cameras move on.
    next: usize,
    /// Frames left to burn before the next dispatch.
    warmup: u32,
    /// Set once the respective probe capture returns non-blank pixels —
    /// that pipeline, mesh and texture are uploaded and real captures can
    /// start. How long this takes varies per machine and cache state, which
    /// is why a fixed frame count cannot gate it. Both renders must be
    /// warm: they compile separate pipelines.
    probe_albedo_seen: bool,
    probe_normal_seen: bool,
    started: std::time::Instant,
}

impl PreprocessState {
    fn entry(&self) -> &BakeEntry {
        &self.entries[self.current]
    }

    fn primed(&self) -> bool {
        self.probe_albedo_seen && self.probe_normal_seen
    }

    fn previous_view_complete(&self) -> bool {
        self.next == 0
            || (self.captured_albedo[self.stage_slot(self.next - 1)].is_some()
                && self.captured_normals[self.stage_slot(self.next - 1)].is_some())
    }

    /// Flat capture index of view `view` of the current pose.
    fn stage_slot(&self, view: usize) -> usize {
        self.pose * self.samples.len() + view
    }

    /// Whether the current pose's every view has landed on the CPU.
    fn pose_complete(&self) -> bool {
        let views = self.samples.len();
        self.captured_albedo[self.pose * views..][..views].iter().all(|c| c.is_some())
            && self.captured_normals[self.pose * views..][..views]
                .iter()
                .all(|c| c.is_some())
    }
}

/// Runs the standalone preprocessor app (see module docs) and returns when
/// it exits.
pub fn run() {
    let mut wgpu_settings = WgpuSettings::default();
    // Match the game's native backend pin (see `main.rs`).
    #[cfg(not(target_arch = "wasm32"))]
    {
        wgpu_settings.backends = Some(Backends::VULKAN);
    }

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-boids impostor preprocess".into(),
                        resolution: WindowResolution::new(128, 128),
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(wgpu_settings)),
                    ..default()
                }),
        )
        .add_plugins(MaterialPlugin::<NormalMaterial>::default())
        .add_plugins(PreprocessPlugin)
        .run();
}

struct PreprocessPlugin;

impl Plugin for PreprocessPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, preprocess_setup).add_systems(
            Update,
            // Dispatch before finish so the last capture's completion is
            // acted on the same frame it arrives.
            (dispatch_next_view, advance_stage).chain(),
        );
    }
}

/// Centre and radius of a mesh's bounding sphere, from its position
/// attribute. Called at startup only — after the first render frame the
/// mesh data is extracted to the render world and no longer readable.
/// Shared with `boid::boid_variations`, which fits the catalog's bake data
/// from the same meshes the game spawns.
pub(crate) fn mesh_bounds(mesh: &Mesh) -> (Vec3, f32) {
    let positions = mesh
        .attributes()
        .find(|(attribute, _)| attribute.id == Mesh::ATTRIBUTE_POSITION.id)
        .and_then(|(_, values)| match values {
            VertexAttributeValues::Float32x3(positions) => Some(positions),
            _ => None,
        })
        .expect("variation meshes need float3 positions");
    let mut min = Vec3::from_slice(&positions[0]);
    let mut max = min;
    for position in positions {
        let position = Vec3::from_slice(position);
        min = min.min(position);
        max = max.max(position);
    }
    let center = (min + max) / 2.0;
    let radius = positions
        .iter()
        .map(|p| (Vec3::from_slice(p) - center).length())
        .fold(0.0_f32, f32::max);
    (center, radius)
}

/// Derives the bake list from the shared catalog: same meshes, unlit
/// material clones for the albedo pass.
fn bake_entries(
    catalog: &[crate::boid::BoidVariation],
    materials: &mut Assets<StandardMaterial>,
) -> Vec<BakeEntry> {
    catalog
        .iter()
        .map(|variation| {
            let mut albedo = materials
                .get(&variation.material)
                .expect("catalog material just built")
                .clone();
            albedo.unlit = true;
            BakeEntry {
                name: variation.name,
                mesh: variation.mesh.clone(),
                albedo: materials.add(albedo),
                center: variation.center,
                radius: variation.radius,
            }
        })
        .collect()
}

/// Ortho projection fitted to a bounding sphere: a square frustum of
/// `atlas::FIT_MARGIN · radius`, the only projection that shows the whole model
/// from every direction in the cone at a constant scale. `area` mirrors
/// the scaling mode (the camera system recomputes it anyway).
fn fitted_projection(radius: f32) -> Projection {
    let span = radius * atlas::FIT_MARGIN;
    Projection::Orthographic(OrthographicProjection {
        scaling_mode: ScalingMode::Fixed {
            width: 2.0 * span,
            height: 2.0 * span,
        },
        near: span,
        far: span * FAR_SPANS,
        viewport_origin: Vec2::new(0.5, 0.5),
        scale: 1.0,
        area: Rect::new(-span, -span, span, span),
    })
}

/// One offscreen capture target: a single cell in size. Every capture
/// redirects through the screenshot machinery into its own texture, so the
/// image only pins the capture size and colour format (sRGB bytes,
/// transparent clear — the alpha channel is the billboard coverage mask).
fn capture_target(images: &mut Assets<Image>) -> Handle<Image> {
    images.add(Image::new_target_texture(
        atlas::CELL_SIZE_PX,
        atlas::CELL_SIZE_PX,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    ))
}

fn preprocess_setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut normal_materials: ResMut<Assets<NormalMaterial>>,
) {
    // The game spawns from the same constructor — one catalog, no drift.
    let catalog = boid_variations(&mut meshes, &mut images, &mut materials);
    let entries = bake_entries(&catalog, &mut materials);
    let poses = bake_poses();
    assert_eq!(
        poses.len() as u32,
        atlas::POSE_COUNT,
        "atlas::POSE_COUNT must match bake_poses()"
    );

    let first = entries[0].clone();
    let samples = atlas::view_samples();

    let albedo_target = capture_target(&mut images);
    let normal_target = capture_target(&mut images);

    // The same mesh rendered twice at the same place: once unlit for
    // albedo, once with the normal material — on separate layers so each
    // camera sees exactly one of them.
    commands.spawn((
        Mesh3d(first.mesh.clone()),
        MeshMaterial3d(first.albedo.clone()),
        poses[0].transform,
        RenderLayers::layer(ALBEDO_LAYER),
        PreprocessModel,
        AlbedoModel,
    ));
    commands.spawn((
        Mesh3d(first.mesh),
        MeshMaterial3d(normal_materials.add(NormalMaterial {})),
        poses[0].transform,
        RenderLayers::layer(NORMAL_LAYER),
        PreprocessModel,
    ));

    // Parked on the first sample's pose so the warmup probes (which capture
    // before `dispatch_next_view` ever sets a pose) actually frame the
    // model.
    let initial_pose = atlas::camera_transform(
        &samples[0],
        first.center,
        first.radius * atlas::FIT_MARGIN * DISTANCE_SPANS,
    );
    for (target, layer) in [
        (albedo_target.clone(), ALBEDO_LAYER),
        (normal_target.clone(), NORMAL_LAYER),
    ] {
        commands.spawn((
            Camera3d::default(),
            Camera {
                clear_color: ClearColorConfig::Custom(Color::NONE),
                ..default()
            },
            RenderTarget::Image(target.into()),
            fitted_projection(first.radius),
            initial_pose,
            RenderLayers::layer(layer),
            PreprocessCamera,
        ));
    }

    let cells_per_variation = samples.len() * poses.len();
    commands.insert_resource(PreprocessState {
        albedo_target,
        normal_target,
        entries,
        poses,
        current: 0,
        pose: 0,
        captured_albedo: vec![None; cells_per_variation],
        captured_normals: vec![None; cells_per_variation],
        samples,
        next: 0,
        warmup: 0,
        probe_albedo_seen: false,
        probe_normal_seen: false,
        started: std::time::Instant::now(),
    });
}

/// Points the cameras at the next unbaked sample and captures one frame of
/// each target.
///
/// Until both probes have seen non-blank pixels, one throwaway capture pair
/// per frame measures render warmth instead (the probes' observers only
/// flip the flags; their pixels are discarded with the entity). Real
/// captures are strictly serialised — the next pose is only dispatched once
/// both of the previous view's captures have landed on the CPU — because
/// the screenshot readback arrives a frame or two after the render;
/// pipelining would risk pairing a late capture with the wrong cell.
fn dispatch_next_view(
    mut state: ResMut<PreprocessState>,
    mut cameras: Query<&mut Transform, With<PreprocessCamera>>,
    mut commands: Commands,
) {
    if state.warmup > 0 {
        state.warmup -= 1;
        return;
    }
    if !state.primed() {
        // One probe pair per frame; late/duplicate probes are harmless, and
        // the first to see rendered geometry unlocks the bake.
        spawn_capture(
            &mut commands,
            &state.albedo_target,
            move |state: &mut PreprocessState, image: &Image| {
                if opaque_px(image) > 0 {
                    state.probe_albedo_seen = true;
                }
            },
        );
        spawn_capture(
            &mut commands,
            &state.normal_target,
            move |state: &mut PreprocessState, image: &Image| {
                if opaque_px(image) > 0 {
                    state.probe_normal_seen = true;
                }
            },
        );
        return;
    }
    if !state.previous_view_complete() {
        return;
    }
    let Some(sample) = state.samples.get(state.next).copied() else {
        return;
    };
    let span = state.entry().radius * atlas::FIT_MARGIN;
    for mut transform in &mut cameras {
        *transform =
            atlas::camera_transform(&sample, state.entry().center, span * DISTANCE_SPANS);
    }

    let slot = state.stage_slot(state.next);
    let name = state.entry().name;
    let pose = state.poses[state.pose].name;
    let index = state.next;
    let total = state.samples.len();
    info!(
        "preprocess: {name}/{pose} view {}/{} at polar {:.0}° azimuth {:.0}°",
        index + 1,
        total,
        sample.polar.to_degrees(),
        sample.azimuth.to_degrees()
    );
    spawn_capture(
        &mut commands,
        &state.albedo_target,
        move |state: &mut PreprocessState, image: &Image| {
            debug!(
                "preprocess: {}/{} view {} albedo arrived ({} opaque px)",
                state.entries[state.current].name,
                state.poses[state.pose].name,
                index + 1,
                opaque_px(image)
            );
            state.captured_albedo[slot] = Some(image.clone());
        },
    );
    spawn_capture(
        &mut commands,
        &state.normal_target,
        move |state: &mut PreprocessState, image: &Image| {
            debug!(
                "preprocess: {}/{} view {} normals arrived ({} opaque px)",
                state.entries[state.current].name,
                state.poses[state.pose].name,
                index + 1,
                opaque_px(image)
            );
            state.captured_normals[slot] = Some(image.clone());
        },
    );
    state.next += 1;
}

/// Spawns one screenshot entity whose observer hands the captured image to
/// `store`. The entity (and the observer with it) is despawned by the
/// screenshot plugin after the capture completes.
fn spawn_capture(
    commands: &mut Commands,
    target: &Handle<Image>,
    store: impl Fn(&mut PreprocessState, &Image) + Send + Sync + 'static,
) {
    let target = target.clone();
    commands
        .spawn(Screenshot::image(target))
        .observe(move |event: On<ScreenshotCaptured>, mut state: ResMut<PreprocessState>| {
            store(&mut state, &event.image);
        });
}

/// Diagnostics helper: count of pixels with nonzero alpha.
fn opaque_px(image: &Image) -> usize {
    image
        .data
        .as_deref()
        .map(|data| data.chunks_exact(4).filter(|px| px[3] != 0).count())
        .unwrap_or(0)
}

/// Completes the current pose: saves the variation's atlas after the last
/// pose, or rolls the stage machine to the next pose / variation.
fn advance_stage(
    mut state: ResMut<PreprocessState>,
    mut cameras: Query<&mut Projection, With<PreprocessCamera>>,
    // Both queries touch `Mesh3d`/`Transform`, so they must be provably
    // disjoint: the normal model versus the albedo model (which also swaps
    // its material).
    mut normal_models: Query<
        (&mut Transform, &mut Mesh3d),
        (With<PreprocessModel>, Without<AlbedoModel>),
    >,
    mut albedo_models: Query<
        (&mut Transform, &mut Mesh3d, &mut MeshMaterial3d<StandardMaterial>),
        With<AlbedoModel>,
    >,
    mut app_exit: MessageWriter<AppExit>,
) {
    if !state.pose_complete() {
        return;
    }

    // More poses of this variation? Roll to the next one; earlier poses'
    // captures stay for the compose at the end.
    if state.pose + 1 < state.poses.len() {
        state.pose += 1;
        let pose_transform = state.poses[state.pose].transform;
        for (mut transform, _) in &mut normal_models {
            *transform = pose_transform;
        }
        for (mut transform, _, _) in &mut albedo_models {
            *transform = pose_transform;
        }
        state.next = 0;
        state.warmup = RETUNE_WARMUP_FRAMES;
        return;
    }

    // Last pose of the variation: compose and save.
    let name = state.entry().name;
    let path = Path::new(OUTPUT_DIR).join(format!("{name}.png"));
    let baked =
        compose_atlas(&state.samples, &state.captured_albedo, &state.captured_normals);
    for (kind, captured) in [
        ("albedo", &state.captured_albedo),
        ("normals", &state.captured_normals),
    ] {
        if let Some(empty) = empty_cells(captured) {
            warn!(
                "preprocess: {name}: {empty} {kind} cell(s) captured nothing — blank model, missing asset or too little warmup?"
            );
        }
    }
    match save_atlas(&baked, &path) {
        Ok(()) => info!(
            "preprocess: wrote {} ({} poses × {} views)",
            path.display(),
            state.poses.len(),
            state.samples.len()
        ),
        Err(error) => {
            error!("preprocess: failed to write {}: {error}", path.display());
            app_exit.write(AppExit::error());
            return;
        }
    }

    // More variations? Retarget the models at the next one.
    if state.current + 1 < state.entries.len() {
        state.current += 1;
        state.pose = 0;
        let next = state.entry().clone();
        let pose_transform = state.poses[0].transform;
        for mut projection in &mut cameras {
            *projection = fitted_projection(next.radius);
        }
        for (mut transform, mut mesh) in &mut normal_models {
            *transform = pose_transform;
            mesh.0 = next.mesh.clone();
        }
        for (mut transform, mut mesh, mut material) in &mut albedo_models {
            *transform = pose_transform;
            mesh.0 = next.mesh.clone();
            material.0 = next.albedo.clone();
        }
        state.reset_stage();
        return;
    }

    info!(
        "preprocess: all {} variation(s) baked in {:.1}s",
        state.entries.len(),
        state.started.elapsed().as_secs_f32()
    );
    app_exit.write(AppExit::Success);
}

impl PreprocessState {
    /// Clears the current variation's captures and parks the dispatcher.
    fn reset_stage(&mut self) {
        let cells = self.samples.len() * self.poses.len();
        self.captured_albedo = vec![None; cells];
        self.captured_normals = vec![None; cells];
        self.next = 0;
        self.warmup = RETUNE_WARMUP_FRAMES;
    }
}

/// Stitches the per-view captures into one atlas-sized RGBA8 image: each
/// pose's albedo cells fill a band of the top half, normal cells mirror
/// them through the texture centre (see the module docs).
fn compose_atlas(
    samples: &[ViewSample],
    albedo: &[Option<Image>],
    normals: &[Option<Image>],
) -> Image {
    let mut atlas = Image::new_fill(
        Extent3d {
            width: atlas::ATLAS_SIZE_PX.x,
            height: atlas::ATLAS_SIZE_PX.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD,
    );
    let data = atlas.data.as_mut().expect("new_fill keeps CPU data");
    for pose in 0..atlas::POSE_COUNT as usize {
        for (view, (albedo, normal)) in albedo
            .iter()
            .skip(pose * samples.len())
            .zip(normals.iter().skip(pose * samples.len()))
            .enumerate()
        {
            if let Some(albedo) = albedo {
                debug_assert_eq!(
                    (albedo.width(), albedo.height()),
                    (atlas::CELL_SIZE_PX, atlas::CELL_SIZE_PX)
                );
                let src = albedo.data.as_deref().expect("captures carry CPU data");
                atlas::blit_cell(
                    data,
                    atlas::ATLAS_SIZE_PX.x,
                    atlas::albedo_cell(view, pose),
                    atlas::CELL_SIZE_PX,
                    src,
                );
            }
            if let Some(normal) = normal {
                debug_assert_eq!(
                    (normal.width(), normal.height()),
                    (atlas::CELL_SIZE_PX, atlas::CELL_SIZE_PX)
                );
                let src = normal.data.as_deref().expect("captures carry CPU data");
                atlas::blit_cell(
                    data,
                    atlas::ATLAS_SIZE_PX.x,
                    atlas::normal_cell(view, pose),
                    atlas::CELL_SIZE_PX,
                    src,
                );
            }
        }
    }
    atlas
}

/// Count of captures that are fully transparent — a blank bake diagnostic.
fn empty_cells(captured: &[Option<Image>]) -> Option<usize> {
    let empty = captured
        .iter()
        .filter(|image| {
            image
                .as_ref()
                .map(|img| {
                    img.data
                        .as_deref()
                        .is_some_and(|d| d.chunks_exact(4).all(|px| px[3] == 0))
                })
                .unwrap_or(true)
        })
        .count();
    (empty > 0).then_some(empty)
}

/// Writes the atlas as an RGBA PNG (extension picks the format; alpha is the
/// coverage mask the billboard material will blend on).
fn save_atlas(atlas: &Image, path: &Path) -> Result<(), String> {
    let dynamic = atlas
        .clone()
        .try_into_dynamic()
        .map_err(|error| error.to_string())?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        dynamic
            .to_rgba8()
            .save(path)
            .map_err(|error| error.to_string())?;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = dynamic;
        return Err("impostor atlas saving is native-only".to_string());
    }
    Ok(())
}
