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
//! body disk inside its cell). A time-varying occupancy field carves the
//! fluid leading edge, sparse like a skirmish screen up front and solid
//! behind. Within the top-down 30° cone the trace yields shoulder disks
//! and helmets from nadir and body-wall silhouettes at the rim, with
//! periodic armour/spear glints; rays that thread the gaps shade a dark
//! trampled floor. A translucent dust volume rides each box (`crowd_dust`).
//!
//! Two armies face off across a gap near the origin so the bench camera
//! (`--crowd --shot ...`, optionally `--cam-angle <deg from nadir>`)
//! captures the experiment head-on.

use bevy::asset::Asset;
use bevy::light::NotShadowCaster;
use bevy::pbr::{Material, MaterialPlugin, MeshMaterial3d};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::mesh::{Indices, Mesh};
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

use crate::launch::launch_crowd_enabled;
use crate::sky::{SkyTuning, sun_transform};
use crate::terrain::HeightField;

/// Footprint shared by the crowd and dust boxes (must match `LEN_M` /
/// `DEPTH_M` in `assets/shaders/crowd_common.wgsl`).
pub const CROWD_LENGTH_M: f32 = 140.0;
pub const CROWD_DEPTH_M: f32 = 36.0;
/// Box height: a unit is up to 1.95 m; the rest is helmet/spear headroom
/// (must exceed the shader's max unit height, or heads clip at the lid).
pub const CROWD_HEIGHT_M: f32 = 2.5;
/// Empty ground between the two armies' front edges.
const CROWD_GAP_M: f32 = 7.0;
/// Dust volume height above the crowd box lid.
const DUST_HEIGHT_M: f32 = 1.8;
/// Soldier grid cell — the DDA's march step (matches the default the
/// shader receives through `CrowdMaterial::sun_spacing.w`).
const SOLDIER_SPACING_M: f32 = 0.85;
/// Vertex grid step of the deformable box.
const VERTEX_STEP_M: f32 = 2.0;
/// Vertical subdivision of the box's side faces.
const SIDE_Y_STEP_M: f32 = 1.2;
/// Side of the isolated scene's ground plane.
const TEST_GROUND_M: f32 = 800.0;

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
            glint_rate_hz: 0.2,
            glint_strength: 1.0,
            dust_density: 1.0,
        }
    }
}

/// The opaque formation crowd. Uniform layout mirrors the declarations
/// at the top of `assets/shaders/crowd.wgsl`.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CrowdMaterial {
    /// rgb: team tunic colour; w: per-team hash seed.
    #[uniform(0)]
    pub team: Vec4,
    /// x: density, y: glint rate (rad/s), z: glint strength, w: box height.
    #[uniform(1)]
    pub params: Vec4,
    /// xyz: direction TO the sun in the box's local space, w: spacing (m).
    #[uniform(2)]
    pub sun_spacing: Vec4,
}

impl Material for CrowdMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/crowd.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/crowd.wgsl".into()
    }
}

/// The translucent dust volume above a crowd box.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CrowdDustMaterial {
    /// x: density, y: box height (m), z/w: unused.
    #[uniform(0)]
    pub params: Vec4,
    /// xyz: direction TO the sun in local space.
    #[uniform(1)]
    pub sun: Vec4,
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
                // `--crowd` runs as an isolated scene (see main.rs): the
                // default flat HeightField, no erosion demo — so there is
                // no `rebuild_terrain` to order against.
                spawn_crowd.run_if(launch_crowd_enabled),
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

/// The isolated test scene's stand-in terrain: one matte plane at y = 0
/// over the flat default HeightField, so the experiment renders against a
/// clean ground instead of the erosion demo's gullies (main.rs skips the
/// erosion plugin entirely while `--crowd` is set).
pub fn spawn_crowd_ground(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(
            Plane3d::default().mesh().size(TEST_GROUND_M, TEST_GROUND_M),
        )),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.40, 0.42, 0.31),
            ..default()
        })),
        Transform::IDENTITY,
    ));
}

/// Spawns the two opposing armies and their dust volumes, seated on the
/// terrain. Runs once (the `Local` guard) on the first frame the launch
/// flag allows; entities stay put afterwards.
fn spawn_crowd(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut crowd_materials: ResMut<Assets<CrowdMaterial>>,
    mut dust_materials: ResMut<Assets<CrowdDustMaterial>>,
    field: Res<HeightField>,
    sky: Res<SkyTuning>,
    tuning: Res<CrowdTuning>,
    mut spawned: Local<bool>,
) {
    if *spawned {
        return;
    }
    *spawned = true;

    let crowd_mesh = meshes.add(crowd_box(
        CROWD_LENGTH_M,
        CROWD_DEPTH_M,
        0.0,
        CROWD_HEIGHT_M,
    ));
    let dust_mesh = meshes.add(crowd_box(CROWD_LENGTH_M, CROWD_DEPTH_M, 0.0, DUST_HEIGHT_M));
    let sun_world = sun_transform(sky.sun_elevation_deg, sky.sun_azimuth_deg)
        .translation
        .normalize();

    // (tunic rgb + team seed, yaw, z offset). The second army is the same
    // box rotated to face the first: local -z is "toward the enemy".
    let armies = [
        (Vec4::new(0.55, 0.14, 0.10, 1.0), 0.0, CROWD_GAP_M / 2.0),
        (
            Vec4::new(0.16, 0.26, 0.48, 7.3),
            std::f32::consts::PI,
            -CROWD_GAP_M / 2.0,
        ),
    ];
    for (team, yaw, z) in armies {
        let rotation = Quat::from_rotation_y(yaw);
        let seat = seat_height(&field, rotation, z);
        info!("crowd: spawning army team={team:?} seat={seat} z={z}");
        // Lighting runs in local space, so the sun rotates with the box.
        let sun_local = rotation.inverse() * sun_world;

        commands.spawn((
            Mesh3d(crowd_mesh.clone()),
            MeshMaterial3d(crowd_materials.add(CrowdMaterial {
                team,
                params: Vec4::new(
                    tuning.density,
                    tuning.glint_rate_hz * std::f32::consts::TAU,
                    tuning.glint_strength,
                    CROWD_HEIGHT_M,
                ),
                sun_spacing: Vec4::new(sun_local.x, sun_local.y, sun_local.z, SOLDIER_SPACING_M),
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
                params: Vec4::new(tuning.dust_density, DUST_HEIGHT_M, 0.0, 0.0),
                sun: Vec4::new(sun_local.x, sun_local.y, sun_local.z, 0.0),
            })),
            Transform::from_xyz(0.0, seat + CROWD_HEIGHT_M, z).with_rotation(rotation),
            NotShadowCaster,
            CrowdDust,
        ));
    }
}

/// Ground height to seat a box at: the minimum over a coarse sample of
/// its rotated footprint plus a hair — burying beats floating.
fn seat_height(field: &HeightField, rotation: Quat, z_offset: f32) -> f32 {
    let half = CROWD_LENGTH_M / 2.0;
    let mut lowest = f32::MAX;
    for ix in 0..=8usize {
        for iz in 0..=4usize {
            let local = Vec3::new(
                -half + CROWD_LENGTH_M * ix as f32 / 8.0,
                0.0,
                CROWD_DEPTH_M * iz as f32 / 4.0,
            );
            let world = rotation * local + Vec3::new(0.0, 0.0, z_offset);
            lowest = lowest.min(field.height(world.x, world.z));
        }
    }
    lowest + 0.05
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
            m.params.x = tuning.density;
            m.params.y = tuning.glint_rate_hz * std::f32::consts::TAU;
            m.params.z = tuning.glint_strength;
        }
    }
    for material in &clouds {
        if let Some(mut m) = dust_materials.get_mut(&material.0) {
            m.params.x = tuning.dust_density;
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
    fn seating_takes_the_lowest_sampled_ground() {
        let flat = HeightField::from_fn(|_, _| 0.0);
        assert!((seat_height(&flat, Quat::IDENTITY, 10.0) - 0.05).abs() < 1e-4);

        // A slope rising with z: the seat must sit at the low (-z) edge of
        // the rotated footprint, not under the entity centre.
        let slope = HeightField::from_fn(|_, z| z);
        let seat = seat_height(&slope, Quat::from_rotation_y(std::f32::consts::PI), 5.0);
        // Rotated π, the footprint spans world z ∈ [5 - 36, 5].
        assert!((seat - (5.0 - CROWD_DEPTH_M) - 0.05).abs() < 1e-4);
    }
}
