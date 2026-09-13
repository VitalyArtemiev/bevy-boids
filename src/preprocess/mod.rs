//! `--preprocess`: bake boid model variations into impostor atlas PNGs.
//!
//! At large camera distances the plan is to drop the mesh and draw one
//! textured billboard per boid. The billboard needs an unlit albedo and a
//! normal for every view direction in a steep-down cone (plus the boid's
//! facing, which [`atlas`] folds into the view azimuth); one disk of
//! pre-rendered views per model variation covers both.
//!
//! This is a self-contained app mode, never merged into the game's plugin
//! graph: `cargo run -- --preprocess` renders every registered variation
//! from every [`atlas::view_samples`] direction with two co-located
//! orthographic cameras (albedo and normals on separate render layers,
//! so both captures happen in the same frame), captures each through the
//! renderer's own screenshot readback, and composites the cells into one
//! zero-waste RGBA PNG under `assets/impostors/` — albedo cells fill the
//! top half, normal cells sit mirrored through the texture centre, so a
//! runtime shader pairs them with `nuv = 1.0 - uv`.
//!
//! Nothing is lit at bake time: baked lighting would be wrong at every
//! yaw except the one the model was baked at (the folding note in
//! [`atlas`]), so the albedo renders unlit and the normals render as raw
//! view-space directions for the runtime shader to shade with. Normal
//! cells are stored sRGB-encoded like the albedo — decode with
//! `n = 2 * sampled - 1` after sampling the atlas as an sRGB texture.

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

use crate::uv_debug_texture;
use atlas::ViewSample;

/// Where baked atlases land, under `assets/` so the future billboard LOD can
/// `AssetServer::load` them.
const OUTPUT_DIR: &str = "assets/impostors";

/// Render layers of the bake: the albedo and normal models sit at the same
/// place, so each model/camera pair lives on its own layer and the two
/// renders never see each other.
const ALBEDO_LAYER: usize = 1;
const NORMAL_LAYER: usize = 2;

/// Headroom around the fitted bounding sphere, fraction of its radius: the
/// silhouette must not touch the cell edge, or bilinear filtering at
/// billboard time would bleed the neighbouring cell in.
const FIT_MARGIN: f32 = 1.1;

/// Camera distance from the model centre, in fitted-span units. Far enough
/// that near/far planes never clip the model for any view in the cone.
const DISTANCE_SPANS: f32 = 4.0;

/// Far plane, in fitted-span units (the model spans about ±1 span).
const FAR_SPANS: f32 = 7.0;

/// Frames burned between variations — only the mesh swap needs to settle.
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

/// One bake job: a model variation and the parameters its bake needs.
#[derive(Clone)]
struct VariationBake {
    name: &'static str,
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    /// Mesh AABB centre in world space — what the cameras orbit.
    center: Vec3,
    /// Bounding-sphere radius, metres: the frustum is fitted to the sphere,
    /// not the box, so every view direction in the cone sees the whole model.
    radius: f32,
}

/// Drives the bake: which view is in flight, what has landed.
#[derive(Resource)]
struct PreprocessState {
    albedo_target: Handle<Image>,
    normal_target: Handle<Image>,
    variations: Vec<VariationBake>,
    current: usize,
    samples: Vec<ViewSample>,
    captured_albedo: Vec<Option<Image>>,
    captured_normals: Vec<Option<Image>>,
    /// Next sample to dispatch; strictly serialised so both of a view's
    /// captures are confirmed before the cameras move on.
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
    fn variation(&self) -> &VariationBake {
        &self.variations[self.current]
    }

    fn primed(&self) -> bool {
        self.probe_albedo_seen && self.probe_normal_seen
    }

    fn previous_view_complete(&self) -> bool {
        self.next == 0
            || (self.captured_albedo[self.next - 1].is_some()
                && self.captured_normals[self.next - 1].is_some())
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
            (dispatch_next_view, finish_variation).chain(),
        );
    }
}

/// The bake list. Today: the classic capsule boid with its UV-debug
/// material, baked unlit (runtime lighting comes from the normal half).
/// Extend with armed/armoured/cavalry meshes here — each entry bakes to
/// `<OUTPUT_DIR>/<name>.png` with the same view sampling.
fn variations(
    meshes: &mut Assets<Mesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<StandardMaterial>,
) -> Vec<VariationBake> {
    let mesh = meshes.add(Capsule3d::default());
    let material = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(uv_debug_texture())),
        unlit: true,
        ..default()
    });
    let (center, radius) = mesh_bounds(&meshes.get(&mesh).expect("just added"));
    vec![VariationBake {
        name: "boid-capsule",
        mesh,
        material,
        center,
        radius,
    }]
}

/// Centre and radius of a mesh's bounding sphere, from its position
/// attribute. Called at startup only — after the first render frame the
/// mesh data is extracted to the render world and no longer readable.
fn mesh_bounds(mesh: &Mesh) -> (Vec3, f32) {
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

/// Ortho projection fitted to a bounding sphere: a square frustum of
/// `FIT_MARGIN · radius`, the only projection that shows the whole model
/// from every direction in the cone at a constant scale. `area` mirrors
/// the scaling mode (the camera system recomputes it anyway).
fn fitted_projection(radius: f32) -> Projection {
    let span = radius * FIT_MARGIN;
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
    let variations = variations(&mut meshes, &mut images, &mut materials);
    let first = variations[0].clone();
    let samples = atlas::view_samples();

    let albedo_target = capture_target(&mut images);
    let normal_target = capture_target(&mut images);

    // The same mesh rendered twice at the same spot: once unlit for albedo,
    // once with the normal material — on separate layers so each camera
    // sees exactly one of them.
    commands.spawn((
        Mesh3d(first.mesh.clone()),
        MeshMaterial3d(first.material.clone()),
        Transform::IDENTITY,
        RenderLayers::layer(ALBEDO_LAYER),
        PreprocessModel,
        AlbedoModel,
    ));
    commands.spawn((
        Mesh3d(first.mesh),
        MeshMaterial3d(normal_materials.add(NormalMaterial {})),
        Transform::IDENTITY,
        RenderLayers::layer(NORMAL_LAYER),
        PreprocessModel,
    ));

    // Parked on the first sample's pose so the warmup probes (which capture
    // before `dispatch_next_view` ever sets a pose) actually frame the
    // model.
    let initial_pose = atlas::camera_transform(
        &samples[0],
        first.center,
        first.radius * FIT_MARGIN * DISTANCE_SPANS,
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

    commands.insert_resource(PreprocessState {
        albedo_target,
        normal_target,
        variations,
        current: 0,
        captured_albedo: vec![None; samples.len()],
        captured_normals: vec![None; samples.len()],
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
    let span = state.variation().radius * FIT_MARGIN;
    for mut transform in &mut cameras {
        *transform = atlas::camera_transform(&sample, state.variation().center, span * DISTANCE_SPANS);
    }

    let index = state.next;
    let total = state.samples.len();
    info!(
        "preprocess: dispatching view {}/{} at polar {:.0}° azimuth {:.0}°",
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
                "preprocess: view {} albedo arrived ({} opaque px)",
                index + 1,
                opaque_px(image)
            );
            state.captured_albedo[index] = Some(image.clone());
        },
    );
    spawn_capture(
        &mut commands,
        &state.normal_target,
        move |state: &mut PreprocessState, image: &Image| {
            debug!(
                "preprocess: view {} normals arrived ({} opaque px)",
                index + 1,
                opaque_px(image)
            );
            state.captured_normals[index] = Some(image.clone());
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

/// Saves the finished variation's atlas, then either retargets the model at
/// the next variation or exits the app.
fn finish_variation(
    mut state: ResMut<PreprocessState>,
    mut cameras: Query<&mut Projection, With<PreprocessCamera>>,
    // Both queries touch `Mesh3d`, so they must be provably disjoint: the
    // normal model versus the albedo model (which also swaps its material).
    mut normal_models: Query<&mut Mesh3d, (With<PreprocessModel>, Without<AlbedoModel>)>,
    mut albedo_models: Query<
        (&mut Mesh3d, &mut MeshMaterial3d<StandardMaterial>),
        With<AlbedoModel>,
    >,
    mut app_exit: MessageWriter<AppExit>,
) {
    if state.captured_albedo.iter().any(|c| c.is_none())
        || state.captured_normals.iter().any(|c| c.is_none())
    {
        return;
    }

    let name = state.variation().name;
    let path = Path::new(OUTPUT_DIR).join(format!("{name}.png"));
    let baked = compose_atlas(&state.captured_albedo, &state.captured_normals);
    for (kind, captured) in [("albedo", &state.captured_albedo), ("normals", &state.captured_normals)] {
        if let Some(empty) = empty_cells(captured) {
            warn!(
                "preprocess: {name}: {empty} {kind} cell(s) captured nothing — blank model, missing asset or too little warmup?"
            );
        }
    }
    match save_atlas(&baked, &path) {
        Ok(()) => info!("preprocess: wrote {} ({} views)", path.display(), state.samples.len()),
        Err(error) => {
            error!("preprocess: failed to write {}: {error}", path.display());
            app_exit.write(AppExit::error());
            return;
        }
    }

    state.current += 1;
    if state.current >= state.variations.len() {
        info!(
            "preprocess: all {} variation(s) baked in {:.1}s",
            state.variations.len(),
            state.started.elapsed().as_secs_f32()
        );
        app_exit.write(AppExit::Success);
        return;
    }

    let next = state.variation().clone();
    for mut projection in &mut cameras {
        *projection = fitted_projection(next.radius);
    }
    for mut mesh in &mut normal_models {
        mesh.0 = next.mesh.clone();
    }
    for (mut mesh, mut material) in &mut albedo_models {
        mesh.0 = next.mesh.clone();
        material.0 = next.material.clone();
    }
    state.captured_albedo = vec![None; state.samples.len()];
    state.captured_normals = vec![None; state.samples.len()];
    state.next = 0;
    state.warmup = RETUNE_WARMUP_FRAMES;
}

/// Stitches the per-view captures into one atlas-sized RGBA8 image:
/// albedo cells row-major in the top half, normal cells mirrored through
/// the texture centre (see the module docs).
fn compose_atlas(albedo: &[Option<Image>], normals: &[Option<Image>]) -> Image {
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
    for (index, (albedo, normal)) in albedo.iter().zip(normals).enumerate() {
        if let Some(albedo) = albedo {
            debug_assert_eq!(
                (albedo.width(), albedo.height()),
                (atlas::CELL_SIZE_PX, atlas::CELL_SIZE_PX)
            );
            let src = albedo.data.as_deref().expect("captures carry CPU data");
            atlas::blit_cell(
                data,
                atlas::ATLAS_SIZE_PX.x,
                atlas::albedo_cell(index),
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
                atlas::normal_cell(index),
                atlas::CELL_SIZE_PX,
                src,
            );
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
