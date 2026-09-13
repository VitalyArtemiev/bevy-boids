//! `--preprocess`: bake boid model variations into impostor atlas PNGs.
//!
//! At large camera distances the plan is to drop the mesh and draw one
//! textured billboard per boid. The billboard texture depends on the view
//! direction within a steep-down cone and on the boid's facing; [`atlas`]
//! explains how facing folds into the view azimuth, so one disk of
//! pre-rendered views per model variation suffices.
//!
//! This is a self-contained app mode, never merged into the game's plugin
//! graph: `cargo run -- --preprocess` renders every registered variation
//! from every [`atlas::view_samples`] direction with an orthographic camera
//! into an offscreen target, captures each frame through the renderer's own
//! screenshot readback, composites the cells into one RGBA PNG under
//! `assets/impostors/` and exits.
//!
//! Lighting matches the game's sky (same sun position, same flat ambient)
//! minus shadows: a baked shadow would be baked at the wrong relative yaw
//! at runtime (see the orientation-folding note in [`atlas`]), and soft
//! unshadowed shading reads better at billboard size anyway.

pub mod atlas;

use std::path::Path;

use bevy::asset::RenderAssetUsages;
use bevy::camera::{ClearColorConfig, RenderTarget, ScalingMode};
use bevy::light::light_consts::lux;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::mesh::VertexAttributeValues;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings};
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::{WindowPlugin, WindowResolution};

use crate::sky::{SkyTuning, flat_ambient, sun_transform};
use crate::uv_debug_texture;
use atlas::ViewSample;

/// Where baked atlases land, under `assets/` so the future billboard LOD can
/// `AssetServer::load` them.
const OUTPUT_DIR: &str = "assets/impostors";

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

/// Marker for the offscreen baking camera.
#[derive(Component)]
struct PreprocessCamera;

/// Marker for the model under the camera.
#[derive(Component)]
struct PreprocessModel;

/// One bake job: a model variation and the parameters its bake needs.
#[derive(Clone)]
struct VariationBake {
    name: &'static str,
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    /// Mesh AABB centre in world space — what the camera orbits.
    center: Vec3,
    /// Bounding-sphere radius, metres: the frustum is fitted to the sphere,
    /// not the box, so every view direction in the cone sees the whole model.
    radius: f32,
}

/// Drives the bake: which sample is in flight, what has landed.
#[derive(Resource)]
struct PreprocessState {
    target: Handle<Image>,
    variations: Vec<VariationBake>,
    current: usize,
    samples: Vec<ViewSample>,
    captured: Vec<Option<Image>>,
    /// Next sample to dispatch; strictly serialised so each capture is
    /// confirmed before the camera moves on.
    next: usize,
    /// Frames left to burn before the next dispatch.
    warmup: u32,
    /// Flips once a probe capture returns non-blank pixels — mesh, texture
    /// and shader pipelines are uploaded, so real captures can start. How
    /// long that takes varies per machine and cache state, which is why a
    /// fixed frame count cannot gate it.
    primed: bool,
    started: std::time::Instant,
}

impl PreprocessState {
    fn variation(&self) -> &VariationBake {
        &self.variations[self.current]
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
/// material. Extend with armed/armoured/cavalry meshes here — each entry
/// bakes to `<OUTPUT_DIR>/<name>.png` with the same view sampling.
fn variations(
    meshes: &mut Assets<Mesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<StandardMaterial>,
) -> Vec<VariationBake> {
    let mesh = meshes.add(Capsule3d::default());
    let material = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(uv_debug_texture())),
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

fn preprocess_setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut images: ResMut<Assets<Image>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let variations = variations(&mut meshes, &mut images, &mut materials);
    let first = variations[0].clone();

    // The capture target: one cell in size. Every capture redirects through
    // the screenshot machinery into its own texture, so this image only
    // pins the capture size and colour format (sRGB bytes, transparent
    // clear — the alpha channel is the billboard coverage mask).
    let target = images.add(Image::new_target_texture(
        atlas::CELL_SIZE_PX,
        atlas::CELL_SIZE_PX,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    ));

    commands.spawn((
        Mesh3d(first.mesh),
        MeshMaterial3d(first.material),
        Transform::IDENTITY,
        PreprocessModel,
    ));

    // Same sun as the game's sky (elevation/azimuth from `SkyTuning`
    // defaults), shadows off — see the module docs for why baking shadows
    // would lie at runtime.
    let sky = SkyTuning::default();
    commands.spawn((
        DirectionalLight {
            illuminance: lux::FULL_DAYLIGHT,
            shadow_maps_enabled: false,
            ..default()
        },
        sun_transform(sky.sun_elevation_deg, sky.sun_azimuth_deg),
    ));
    commands.insert_resource(flat_ambient());

    let samples = atlas::view_samples();
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
        RenderTarget::Image(target.clone().into()),
        fitted_projection(first.radius),
        // Parked on the first sample's pose so the warmup probes (which
        // capture before `dispatch_next_view` ever sets a pose) actually
        // frame the model.
        atlas::camera_transform(
            &samples[0],
            first.center,
            first.radius * FIT_MARGIN * DISTANCE_SPANS,
        ),
        PreprocessCamera,
    ));

    commands.insert_resource(PreprocessState {
        target,
        variations,
        current: 0,
        captured: vec![None; samples.len()],
        samples,
        next: 0,
        warmup: 0,
        primed: false,
        started: std::time::Instant::now(),
    });
}

/// Points the camera at the next unbaked sample and captures one frame.
///
/// Until the first non-blank probe arrives, one throwaway capture per frame
/// measures render warmth instead (the probes' observers only flip
/// `primed`; their pixels are discarded with the entity). Real captures
/// are strictly serialised — the next pose is only dispatched once the
/// previous capture has landed on the CPU — because the screenshot
/// readback arrives a frame or two after the render; pipelining would risk
/// pairing a late capture with the wrong cell.
fn dispatch_next_view(
    mut state: ResMut<PreprocessState>,
    mut cameras: Query<&mut Transform, With<PreprocessCamera>>,
    mut commands: Commands,
) {
    if state.warmup > 0 {
        state.warmup -= 1;
        return;
    }
    if !state.primed {
        // One probe per frame; late/duplicate probes are harmless, and the
        // first one to see rendered geometry unlocks the bake.
        commands
            .spawn(Screenshot::image(state.target.clone()))
            .observe(
                |event: On<ScreenshotCaptured>, mut state: ResMut<PreprocessState>| {
                    if opaque_px(&event.image) > 0 {
                        state.primed = true;
                    }
                },
            );
        return;
    }
    if state.next > 0 && state.captured[state.next - 1].is_none() {
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
    commands
        .spawn(Screenshot::image(state.target.clone()))
        .observe(
            move |event: On<ScreenshotCaptured>, mut state: ResMut<PreprocessState>| {
                debug!(
                    "preprocess: view {} capture arrived ({} opaque px)",
                    index + 1,
                    opaque_px(&event.image)
                );
                state.captured[index] = Some(event.image.clone());
            },
        );
    state.next += 1;
}

/// Diagnostics helper: count of pixels with nonzero alpha.
fn opaque_px(image: &Image) -> usize {
    image
        .data
        .as_deref()
        .map(|data| {
            data.chunks_exact(4)
                .filter(|px| px[3] != 0)
                .count()
        })
        .unwrap_or(0)
}

/// Saves the finished variation's atlas, then either retargets the model at
/// the next variation or exits the app.
fn finish_variation(
    mut state: ResMut<PreprocessState>,
    mut cameras: Query<&mut Projection, With<PreprocessCamera>>,
    mut models: Query<(&mut Mesh3d, &mut MeshMaterial3d<StandardMaterial>), With<PreprocessModel>>,
    mut app_exit: MessageWriter<AppExit>,
) {
    if state.captured.iter().any(|c| c.is_none()) {
        return;
    }

    let name = state.variation().name;
    let path = Path::new(OUTPUT_DIR).join(format!("{name}.png"));
    let baked = compose_atlas(&state.samples, &state.captured);
    if let Some(empty) = empty_cells(&state.samples, &state.captured) {
        warn!(
            "preprocess: {name}: {empty} cell(s) captured nothing — blank model, missing asset or too little warmup?"
        );
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
    for (mut mesh, mut material) in &mut models {
        mesh.0 = next.mesh.clone();
        material.0 = next.material.clone();
    }
    state.captured = vec![None; state.samples.len()];
    state.next = 0;
    state.warmup = RETUNE_WARMUP_FRAMES;
}

/// Stitches the per-sample captures into one atlas-sized RGBA8 image.
fn compose_atlas(samples: &[ViewSample], captured: &[Option<Image>]) -> Image {
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
    for (sample, image) in samples.iter().zip(captured) {
        let Some(image) = image else { continue };
        debug_assert_eq!(
            (image.width(), image.height()),
            (atlas::CELL_SIZE_PX, atlas::CELL_SIZE_PX)
        );
        debug!(
            "preprocess: composing view at cell ({}, {}) polar {:.0}° azimuth {:.0}° ({} opaque px)",
            sample.cell.x,
            sample.cell.y,
            sample.polar.to_degrees(),
            sample.azimuth.to_degrees(),
            opaque_px(image)
        );
        let src = image.data.as_deref().expect("captures carry CPU data");
        atlas::blit_cell(data, atlas::ATLAS_SIZE_PX.x, sample.cell, atlas::CELL_SIZE_PX, src);
    }
    atlas
}

/// Count of captures that are fully transparent — a blank bake diagnostic.
fn empty_cells(samples: &[ViewSample], captured: &[Option<Image>]) -> Option<usize> {
    let empty = samples
        .iter()
        .zip(captured)
        .filter(|(_, image)| {
            image
                .as_ref()
                .map(|img| img.data.as_deref().is_some_and(|d| d.chunks_exact(4).all(|px| px[3] == 0)))
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
