//! The formation "crowd shell" experiment: rendering a massed formation
//! as one deformed box plus a procedural crowd shader, instead of one draw
//! per unit.
//!
//! Each army is a subdivided box the height of a unit (plus headroom for
//! helmets and spear tips) — the mesh is the formation's bounding volume,
//! dense enough (2 m grid) for the CPU to deform along a fluid frontline
//! later. The experiment deforms it in the vertex stage (`swell_y` in
//! `assets/shaders/crowd_common.wgsl`) to prove the crowd rendering
//! survives the geometry moving.
//!
//! The shader does the rest: a jittered grid of soldiers (cylinder body +
//! helmet sphere per cell) lives inside the box, sampled by walking the
//! view ray through the cells (2-D DDA — exact, because jitter keeps every
//! body disk inside its cell). Occupancy is fully static — no soldier ever
//! pops in or out; the fluid frontline instead moves by WARPING the walk's
//! z coordinate with the surge (`front_z(x, t) − front_z(x, 0)`), so the
//! whole column breathes back and forth while keeping its identity. A
//! sparse skirmish screen leads and solid ranks follow. Within the
//! top-down 30° cone the trace yields shoulder disks and helmets from
//! nadir and body-wall silhouettes at the rim, with periodic armour and
//! spear-tip glints. The box itself is invisible: the material is alpha
//! masked, and rays that neither hit a soldier nor pass beneath occupied
//! cells are discarded — only the crowd and its dark shadowed interior
//! draw. A translucent dust volume rides each box (`crowd_dust`).
//!
//! Two armies face off across a gap near the origin as the `--scene=crowd`
//! test scene (see src/scene.rs), so the bench camera (`--cam-angle`,
//! `--shot`) captures the experiment head-on.

use bevy::asset::Asset;
use bevy::light::NotShadowCaster;
use bevy::pbr::{Material, MaterialPlugin, MeshMaterial3d};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::mesh::{Indices, Mesh};
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat,
};
use bevy::shader::ShaderRef;

use crate::scene::{TestScene, in_scene};
use crate::sky::{SkyTuning, sun_transform};
use crate::terrain::HeightField;

/// Footprint shared by the crowd and dust boxes (must match `LEN_M` /
/// `DEPTH_M` in `assets/shaders/crowd_common.wgsl`).
pub const CROWD_LENGTH_M: f32 = 140.0;
pub const CROWD_DEPTH_M: f32 = 36.0;
/// Box height: a unit is up to 1.95 m tall (helmet included); the rest is
/// headroom for the swell (±0.19 m), which rides the lid. Keep this tight
/// to the soldiers: the lid must stay above every helmet at every phase,
/// or heads poke through it and vanish from above.
pub const CROWD_HEIGHT_M: f32 = 2.3;
/// Z offset of each army's box origin from the world centre; the second
/// army is rotated π so local -z (toward its own front) faces the first.
/// Rendered front lines sit `front_z` ≈ 5 m deeper inside each box.
const CROWD_GAP_M: f32 = 7.0;
/// Dust volume height above the local ground. Tall enough for the plume
/// to read as haze hanging over the fight, not a thin skirt on the roof.
const DUST_HEIGHT_M: f32 = 3.0;
/// Dust volume's floor above the local ground: the plume hangs over the
/// fight instead of wrapping the soldiers' legs.
const DUST_BASE_M: f32 = 1.0;
/// Soldier grid cell — the DDA's march step (matches the default the
/// shader receives through `CrowdMaterial::sun_spacing.w`).
const SOLDIER_SPACING_M: f32 = 0.85;
/// Vertex grid step of the deformable box.
const VERTEX_STEP_M: f32 = 2.0;
/// Vertical subdivision of the box's side faces.
const SIDE_Y_STEP_M: f32 = 1.2;
/// Side of the isolated scene's ground plane.
const TEST_GROUND_M: f32 = 800.0;
/// How far below the highest soldier the trace box's floor reaches
/// (mirrors the headroom the WGSL used to hard-code).
const TRACE_Y_PAD_M: f32 = 0.7;
/// Terrain heightmap side, in metres and texels (≈0.47 m/texel — coarse,
/// but the heightfield is low-frequency by design).
const HEIGHT_MAP_SPAN_M: f32 = 240.0;
const HEIGHT_MAP_RES: u32 = 512;

/// Rolling test hills for the `--scene=crowd` test scene: gentle enough to
/// march armies across (worst grade ~3%), steep enough to exercise
/// terrain following. The ground mesh, the HeightField and the crowd
/// heightmap all come from this one function, so grounding, camera and
/// crowd agree by construction.
pub fn crowd_test_height(x: f32, z: f32) -> f32 {
    4.0 * (x * 0.008).sin() * (z * 0.0075).cos()
        + 2.0 * (x * 0.019 + 1.3).sin() * (z * 0.016 + 0.7).sin()
}

/// Runtime-tunable crowd look; exposed as sliders by the debug UI and
/// pushed into the material uniforms by [`sync_crowd_tuning`].
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct CrowdTuning {
    /// Occupancy scale over the field's own row-density curve.
    pub density: f32,
    /// Periodic armour-glint frequency (cycles per soldier per second).
    pub glint_rate_hz: f32,
    /// Glint brightness multiplier.
    pub glint_strength: f32,
    /// Dust volume density scale.
    pub dust_density: f32,
}

impl Default for CrowdTuning {
    fn default() -> Self {
        Self {
            density: 1.0,
            glint_rate_hz: 0.5,
            glint_strength: 1.0,
            dust_density: 1.6,
        }
    }
}

/// Terrain mapping shared by both crowd materials (mirrors `TerrainUniforms`
/// in `crowd_common.wgsl`): box-local xz onto the battlefield heightmap.
#[derive(ShaderType, Debug, Clone, Copy)]
pub struct TerrainUniforms {
    /// (origin.x, origin.z, sin(yaw), cos(yaw)) — box-local xz → world xz.
    pub a: Vec4,
    /// (map_min.x, map_min.z, 1/map_span, seat) — world xz → heightmap,
    /// and the seat height h_rel is measured from.
    pub b: Vec4,
    /// (h_min, h_max, trace_y_min, trace_y_max) — height decode and the
    /// local y range the trace clips to.
    pub c: Vec4,
}

/// The opaque formation crowd. All scalars ride ONE uniform buffer
/// (`CrowdUniforms`, binding 0) beside the heightmap texture: browser
/// backends — WebGL2 and in-browser WebGPU alike — cap uniform buffers per
/// shader stage at 12, and the engine's view bindings already spend 8, so
/// the six separate bindings this replaced overflowed the pipeline layout
/// and killed pipeline creation on wasm (native Vulkan allows far more,
/// so it never showed up locally).
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CrowdMaterial {
    #[uniform(0)]
    pub uniforms: CrowdUniforms,
    /// Battlefield heightmap (R8Unorm over `HEIGHT_MAP_SPAN_M`), decoded
    /// with `terrain.c.xy`.
    #[texture(1)]
    pub height_map: Handle<Image>,
}

#[derive(ShaderType, Debug, Clone, Copy)]
pub struct CrowdUniforms {
    /// rgb: team tunic colour; w: per-team hash seed.
    pub team: Vec4,
    /// x: density, y: glint rate (rad/s), z: glint strength, w: unused.
    pub params: Vec4,
    /// xyz: direction TO the sun in the box's local space, w: spacing (m).
    pub sun_spacing: Vec4,
    /// Heightmap mapping — see [`TerrainUniforms`].
    pub terrain: TerrainUniforms,
}

impl Material for CrowdMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/crowd.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/crowd.wgsl".into()
    }
    // Cutout transparency: the shader discards rays that miss every
    // soldier and never pass beneath occupied cells, so the box shows
    // only the crowd and its shadowed interior — bare terrain elsewhere.
    // Masked (not blended) keeps the crowd in the opaque pass: no sorting
    // against the dust volume, no depth-write surprises.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Mask(0.5)
    }
}

/// The translucent dust volume above a crowd box. Same single-buffer layout
/// as [`CrowdMaterial`] (one uniforms struct + the shared heightmap) — the
/// packing exists for the wasm uniform-buffer limit, see its docs.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CrowdDustMaterial {
    #[uniform(0)]
    pub uniforms: CrowdDustUniforms,
    /// See `CrowdMaterial` — same battlefield heightmap.
    #[texture(1)]
    pub height_map: Handle<Image>,
}

#[derive(ShaderType, Debug, Clone, Copy)]
pub struct CrowdDustUniforms {
    /// x: density, y: box height (m), z/w: unused.
    pub params: Vec4,
    /// xyz: direction TO the sun in local space.
    pub sun: Vec4,
    /// Heightmap mapping — see [`TerrainUniforms`].
    pub terrain: TerrainUniforms,
}

impl Material for CrowdDustMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/crowd_dust.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/crowd_dust.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

/// Marks a crowd shell entity (the opaque soldier box).
#[derive(Component)]
pub struct CrowdArmy;

/// Marks a dust volume entity.
#[derive(Component)]
pub struct CrowdDust;

/// Keeps the shared crowd module loaded. `crowd.wgsl` and `crowd_dust.wgsl`
/// import it by namespace (`#import bevy_boids::crowd_common`), and naga-oil
/// resolves namespace imports only against shader assets that are actually
/// loaded — no path references the file, so without this load the crowd
/// pipelines silently never specialize and nothing draws (the deleted
/// terrain render module carried the same resource for `terrain_common`).
#[derive(Resource)]
pub struct CrowdCommonShader(pub Handle<Shader>);

pub struct CrowdPlugin;

impl Plugin for CrowdPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CrowdTuning>()
            .add_plugins(MaterialPlugin::<CrowdMaterial>::default())
            .add_plugins(MaterialPlugin::<CrowdDustMaterial>::default())
            .add_systems(Startup, load_crowd_module)
            .add_systems(
                Update,
                // Armies spawn once the shared shader module is loaded —
                // creating the material earlier races the async load.
                // (The scene's ground and camera are the world
                // assembler's job, see `scene::assemble_scene`.)
                spawn_crowd
                    .run_if(in_scene(TestScene::Crowd))
                    .run_if(crowd_module_loaded),
            )
            .add_systems(
                Update,
                sync_crowd_tuning.run_if(resource_changed::<CrowdTuning>),
            );
    }
}

/// Loads `crowd_common.wgsl` so its `#define_import_path` namespace exists
/// before any crowd pipeline specializes.
fn load_crowd_module(mut commands: Commands, server: Res<AssetServer>) {
    commands.insert_resource(CrowdCommonShader(server.load("shaders/crowd_common.wgsl")));
}

/// Run condition for `spawn_crowd`: the shared module must be fully loaded
/// first. Creating the material earlier races the async load, and a
/// pipeline that specializes against a missing import silently never draws.
fn crowd_module_loaded(server: Res<AssetServer>, module: Res<CrowdCommonShader>) -> bool {
    server.is_loaded_with_dependencies(&module.0)
}

/// The crowd scene's stand-in terrain: a gently rolling grid mesh
/// and a HeightField from the same [`crowd_test_height`], so the crowd's
/// boxes have real slopes to ride (the scene replaces the erosion demo
/// entirely). Called by the world assembler whenever the crowd scene
/// loads — at startup and on runtime switches alike.
pub(crate) fn spawn_crowd_ground(world: &mut World) {
    let quads = 128;
    let step = TEST_GROUND_M / quads as f32;
    let half = TEST_GROUND_M / 2.0;
    let eps = step;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for iz in 0..=quads {
        for ix in 0..=quads {
            let x = -half + ix as f32 * step;
            let z = -half + iz as f32 * step;
            positions.push([x, crowd_test_height(x, z), z]);
            // Central-difference normal of the height function.
            let dx = (crowd_test_height(x + eps, z) - crowd_test_height(x - eps, z)) / (2.0 * eps);
            let dz = (crowd_test_height(x, z + eps) - crowd_test_height(x, z - eps)) / (2.0 * eps);
            normals.push(Vec3::new(-dx, 1.0, -dz).normalize().to_array());
            let v = iz * (quads + 1) + ix;
            if ix < quads && iz < quads {
                // Winding so faces point up.
                indices.extend_from_slice(&[v, v + quads + 1, v + 1]);
                indices.extend_from_slice(&[v + 1, v + quads + 1, v + quads + 2]);
            }
        }
    }
    let mut mesh = Mesh::new(
        bevy::render::render_resource::PrimitiveTopology::TriangleList,
        Default::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    // Grounding, camera focus and the crowd heightmap sample this exact
    // function, so everything agrees with the rendered mesh.
    *world.resource_mut::<HeightField>() = HeightField::from_fn(crowd_test_height);

    let mesh = world.resource_mut::<Assets<Mesh>>().add(mesh);
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
        CrowdGround,
    ));
}

/// Marks the crowd scene's rolling-hill ground so the world teardown can
/// despawn it with the rest of the scene.
#[derive(Component)]
pub struct CrowdGround;

/// Bakes the battlefield heights into an R8Unorm heightmap texture for
/// the crowd shaders (they cannot call the CPU `HeightField`). The decode
/// range `(min, max)` rides to the GPU in `terrain.c.xy`.
fn bake_height_map(images: &mut Assets<Image>, field: &HeightField) -> (Handle<Image>, f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut heights = Vec::with_capacity((HEIGHT_MAP_RES * HEIGHT_MAP_RES) as usize);
    for iz in 0..HEIGHT_MAP_RES {
        for ix in 0..HEIGHT_MAP_RES {
            let x = -HEIGHT_MAP_SPAN_M / 2.0
                + (ix as f32 + 0.5) / HEIGHT_MAP_RES as f32 * HEIGHT_MAP_SPAN_M;
            let z = -HEIGHT_MAP_SPAN_M / 2.0
                + (iz as f32 + 0.5) / HEIGHT_MAP_RES as f32 * HEIGHT_MAP_SPAN_M;
            let h = field.height(x, z);
            min = min.min(h);
            max = max.max(h);
            heights.push(h);
        }
    }
    let range = (max - min).max(1e-3);
    let data = heights
        .iter()
        .map(|h| (((h - min) / range).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect::<Vec<u8>>();
    let image = Image::new_fill(
        Extent3d {
            width: HEIGHT_MAP_RES,
            height: HEIGHT_MAP_RES,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &data,
        TextureFormat::R8Unorm,
        Default::default(),
    );
    (images.add(image), min, max)
}

/// Spawns the two opposing armies and their dust volumes, seated on the
/// terrain. Runs on every frame the scene allows while no army exists —
/// presence is the one-shot guard, so a world reload (which despawns the
/// armies) respawns them fresh.
fn spawn_crowd(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut crowd_materials: ResMut<Assets<CrowdMaterial>>,
    mut dust_materials: ResMut<Assets<CrowdDustMaterial>>,
    field: Res<HeightField>,
    sky: Res<SkyTuning>,
    tuning: Res<CrowdTuning>,
    armies: Query<(), With<CrowdArmy>>,
) {
    if !armies.is_empty() {
        return;
    }

    let crowd_mesh = meshes.add(crowd_box(
        CROWD_LENGTH_M,
        CROWD_DEPTH_M,
        0.0,
        CROWD_HEIGHT_M,
    ));
    // Dust floats one metre above the local ground so the plume hangs
    // over the fight rather than wrapping the soldiers' legs.
    let dust_mesh = meshes.add(crowd_box(
        CROWD_LENGTH_M,
        CROWD_DEPTH_M,
        DUST_BASE_M,
        DUST_BASE_M + DUST_HEIGHT_M,
    ));
    let (height_map, h_min, h_max) = bake_height_map(&mut images, &field);
    let sun_world = sun_transform(sky.sun_elevation_deg, sky.sun_azimuth_deg)
        .translation
        .normalize();

    // (tunic rgb + team seed, yaw, z offset). The second army is the same
    // box rotated to face the first: local -z is "toward the enemy".
    let armies = [
        (Vec4::new(0.62, 0.12, 0.08, 1.0), 0.0, CROWD_GAP_M / 2.0),
        (
            Vec4::new(0.10, 0.22, 0.68, 7.3),
            std::f32::consts::PI,
            -CROWD_GAP_M / 2.0,
        ),
    ];
    for (team, yaw, z) in armies {
        let rotation = Quat::from_rotation_y(yaw);
        let (min_h, max_h) = footprint_y_range(&field, rotation, z);
        // Sit slightly buried: hiding beats floating.
        let seat = min_h + 0.05;
        // Lighting runs in local space, so the sun rotates with the box.
        let sun_local = rotation.inverse() * sun_world;
        let terrain = terrain_uniforms(yaw, z, seat, h_min, h_max, min_h, max_h);

        commands.spawn((
            Mesh3d(crowd_mesh.clone()),
            MeshMaterial3d(crowd_materials.add(CrowdMaterial {
                uniforms: CrowdUniforms {
                    team,
                    params: Vec4::new(
                        tuning.density,
                        tuning.glint_rate_hz * std::f32::consts::TAU,
                        tuning.glint_strength,
                        0.0,
                    ),
                    sun_spacing: Vec4::new(
                        sun_local.x,
                        sun_local.y,
                        sun_local.z,
                        SOLDIER_SPACING_M,
                    ),
                    terrain,
                },
                height_map: height_map.clone(),
            })),
            Transform::from_xyz(0.0, seat, z).with_rotation(rotation),
            // The crowd's own shading handles self-occlusion; a lid-less
            // box shadow from the stock prepass would read wrong.
            NotShadowCaster,
            CrowdArmy,
        ));
        commands.spawn((
            Mesh3d(dust_mesh.clone()),
            MeshMaterial3d(dust_materials.add(CrowdDustMaterial {
                uniforms: CrowdDustUniforms {
                    params: Vec4::new(tuning.dust_density, DUST_HEIGHT_M, 0.0, 0.0),
                    sun: Vec4::new(sun_local.x, sun_local.y, sun_local.z, 0.0),
                    terrain,
                },
                height_map: height_map.clone(),
            })),
            Transform::from_xyz(0.0, seat, z).with_rotation(rotation),
            NotShadowCaster,
            CrowdDust,
        ));
    }
}

/// The terrain uniforms shared by both crowd materials: (origin.x,
/// origin.z, sin(yaw), cos(yaw)), the heightmap window plus this army's
/// seat, and `c` = the heightmap's BAKE decode range (the map spans
/// `h_min..h_max` over the whole battlefield) followed by the trace's
/// local y range (this box's footprint rise plus box height).
fn terrain_uniforms(
    yaw: f32,
    z_offset: f32,
    seat: f32,
    h_min: f32,
    h_max: f32,
    footprint_min_h: f32,
    footprint_max_h: f32,
) -> TerrainUniforms {
    TerrainUniforms {
        a: Vec4::new(0.0, z_offset, yaw.sin(), yaw.cos()),
        b: Vec4::new(
            -HEIGHT_MAP_SPAN_M / 2.0,
            -HEIGHT_MAP_SPAN_M / 2.0,
            1.0 / HEIGHT_MAP_SPAN_M,
            seat,
        ),
        c: Vec4::new(
            h_min,
            h_max,
            (footprint_min_h - seat) - TRACE_Y_PAD_M,
            (footprint_max_h - footprint_min_h) + CROWD_HEIGHT_M + TRACE_Y_PAD_M,
        ),
    }
}

/// Local y range a box's trace must cover on this ground: heights run
/// from the footprint's lowest sample (the seat, minus a hair of burial)
/// to its highest plus the box height. Returned as `(min_h, max_h)` in
/// world y; [`terrain_uniforms`] converts to box-local.
fn footprint_y_range(field: &HeightField, rotation: Quat, z_offset: f32) -> (f32, f32) {
    let half = CROWD_LENGTH_M / 2.0;
    let mut lowest = f32::MAX;
    let mut highest = f32::MIN;
    for ix in 0..=8usize {
        for iz in 0..=4usize {
            let local = Vec3::new(
                -half + CROWD_LENGTH_M * ix as f32 / 8.0,
                0.0,
                CROWD_DEPTH_M * iz as f32 / 4.0,
            );
            let world = rotation * local + Vec3::new(0.0, 0.0, z_offset);
            let h = field.height(world.x, world.z);
            lowest = lowest.min(h);
            highest = highest.max(h);
        }
    }
    (lowest, highest)
}

/// Pushes [`CrowdTuning`] edits into the live material instances.
fn sync_crowd_tuning(
    tuning: Res<CrowdTuning>,
    mut crowd_materials: ResMut<Assets<CrowdMaterial>>,
    mut dust_materials: ResMut<Assets<CrowdDustMaterial>>,
    armies: Query<&MeshMaterial3d<CrowdMaterial>>,
    clouds: Query<&MeshMaterial3d<CrowdDustMaterial>>,
) {
    for material in &armies {
        if let Some(mut m) = crowd_materials.get_mut(&material.0) {
            m.uniforms.params.x = tuning.density;
            m.uniforms.params.y = tuning.glint_rate_hz * std::f32::consts::TAU;
            m.uniforms.params.z = tuning.glint_strength;
        }
    }
    for material in &clouds {
        if let Some(mut m) = dust_materials.get_mut(&material.0) {
            m.uniforms.params.x = tuning.dust_density;
        }
    }
}

/// Appends one quad grid face. Winding is chosen so the geometric normal
/// `dv × du` points along the face's outward normal — see the call site
/// comments in [`crowd_box`].
fn face(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
    origin: Vec3,
    du: Vec3,
    dv: Vec3,
    nu: usize,
    nv: usize,
) {
    let normal = dv.cross(du).normalize();
    let base = positions.len() as u32;
    for iv in 0..=nv {
        for iu in 0..=nu {
            let p = origin + du * iu as f32 + dv * iv as f32;
            positions.push(p.to_array());
            normals.push(normal.to_array());
        }
    }
    let stride = (nu + 1) as u32;
    for iv in 0..nv {
        for iu in 0..nu {
            let v0 = base + iv as u32 * stride + iu as u32;
            indices.extend_from_slice(&[
                v0,
                v0 + stride,
                v0 + 1,
                v0 + 1,
                v0 + stride,
                v0 + stride + 1,
            ]);
        }
    }
}

/// The deformable formation box: a closed, subdivided cuboid spanning
/// x ∈ [-length/2, length/2] (along the front), z ∈ [0, depth] (enemy at
/// -z), y ∈ [y_min, y_max]. Local coordinates only — entities carry the
/// world placement.
pub fn crowd_box(length_m: f32, depth_m: f32, y_min: f32, y_max: f32) -> Mesh {
    let half = length_m / 2.0;
    let nx = (length_m / VERTEX_STEP_M).round() as usize;
    let nz = (depth_m / VERTEX_STEP_M).round() as usize;
    let ny = (((y_max - y_min) / SIDE_Y_STEP_M).round() as usize).max(1);

    let x = Vec3::X * VERTEX_STEP_M;
    let z = Vec3::Z * VERTEX_STEP_M;
    let y = Vec3::Y * SIDE_Y_STEP_M;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // (origin, du, dv, nu, nv) with dv × du = outward normal:
    // top z×x=+y, bottom x×z=-y, front(+x,+y) y×x=-z, back(+y,+x) x×y=+z,
    // left(+y,+z) z×y=-x, right(+z,+y) y×z=+x.
    face(
        &mut positions,
        &mut normals,
        &mut indices,
        Vec3::new(-half, y_max, 0.0),
        x,
        z,
        nx,
        nz,
    );
    face(
        &mut positions,
        &mut normals,
        &mut indices,
        Vec3::new(-half, y_min, 0.0),
        z,
        x,
        nz,
        nx,
    );
    face(
        &mut positions,
        &mut normals,
        &mut indices,
        Vec3::new(-half, y_min, 0.0),
        x,
        y,
        nx,
        ny,
    );
    face(
        &mut positions,
        &mut normals,
        &mut indices,
        Vec3::new(-half, y_min, depth_m),
        y,
        x,
        ny,
        nx,
    );
    face(
        &mut positions,
        &mut normals,
        &mut indices,
        Vec3::new(-half, y_min, 0.0),
        y,
        z,
        ny,
        nz,
    );
    face(
        &mut positions,
        &mut normals,
        &mut indices,
        Vec3::new(half, y_min, 0.0),
        z,
        y,
        nz,
        ny,
    );

    let mut mesh = Mesh::new(
        bevy::render::render_resource::PrimitiveTopology::TriangleList,
        Default::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::mesh::VertexAttributeValues;

    fn triangles(mesh: &Mesh) -> Vec<[Vec3; 3]> {
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(p) => p.clone(),
            _ => panic!("unexpected position format"),
        };
        let indices = match mesh.indices().unwrap() {
            Indices::U32(i) => i.clone(),
            _ => panic!("unexpected index format"),
        };
        indices
            .chunks_exact(3)
            .map(|t| {
                let idx: [u32; 3] = t.try_into().expect("chunks_exact(3)");
                idx.map(|i| Vec3::from(positions[i as usize]))
            })
            .collect()
    }

    #[test]
    fn crowd_box_is_closed_with_outward_faces() {
        let mesh = crowd_box(10.0, 4.0, 0.0, 2.5);
        let centre = Vec3::new(0.0, 1.25, 2.0);
        for tri in triangles(&mesh) {
            let normal = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
            let centroid = (tri[0] + tri[1] + tri[2]) / 3.0;
            assert!(
                normal.dot(centroid - centre) > 0.0,
                "triangle {tri:?} faces inward ({normal:?})"
            );
        }
    }

    #[test]
    fn crowd_box_counts_match_the_face_grids() {
        // 10/2 = 5 x-steps, 4/2 = 2 z-steps, max(2.5/1.2, 1) = 2 y-steps.
        // Verts: top+bottom 2·(6·3), front+back 2·(6·3), sides 2·(3·3) = 90.
        // Quads: 2·(5·2) + 2·(5·2) + 2·(2·2) = 48 → 96 triangles.
        let mesh = crowd_box(10.0, 4.0, 0.0, 2.5);
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(p) => p.clone(),
            _ => panic!("unexpected position format"),
        };
        assert_eq!(positions.len(), 90);
        assert_eq!(triangles(&mesh).len(), 96);
        for p in positions {
            assert!(p[0] >= -5.0 && p[0] <= 5.0, "x out of bounds: {p:?}");
            assert!(p[1] >= 0.0 && p[1] <= 2.5, "y out of bounds: {p:?}");
            assert!(p[2] >= 0.0 && p[2] <= 4.0, "z out of bounds: {p:?}");
        }
    }

    #[test]
    fn footprint_y_range_spans_the_ground_under_the_box() {
        let flat = HeightField::from_fn(|_, _| 0.0);
        let (min, max) = footprint_y_range(&flat, Quat::IDENTITY, 10.0);
        assert!((min - 0.0).abs() < 1e-4 && (max - 0.0).abs() < 1e-4);

        // A slope rising with z: the range must cover the whole rotated
        // footprint, not just under the entity centre.
        let slope = HeightField::from_fn(|_, z| z);
        let (min, max) =
            footprint_y_range(&slope, Quat::from_rotation_y(std::f32::consts::PI), 5.0);
        // Rotated π, the footprint spans world z ∈ [5 − 36, 5].
        assert!((min - (5.0 - CROWD_DEPTH_M)).abs() < 1e-4);
        assert!((max - 5.0).abs() < 1e-4);
    }
}
