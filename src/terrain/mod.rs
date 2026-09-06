//! The terrain: a single flat 1 km² square (no LOD, no streaming) whose
//! surface `HeightField` describes. Boid grounding, camera focus-follow
//! and terrain clearance all sample the field; the mesh exists for
//! rendering and for the drag-pan grab raycast (`Ground`).

use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;
use std::sync::Arc;
use bevy::render::mesh::{Indices, Mesh};
use bevy_rts_camera::Ground;

mod camera;
pub mod demo;
mod grounding;

pub use camera::{CameraClearance, camera_terrain_clearance, focus_camera_on_ground};
pub use demo::{DemoTerrain, ErosionDemoPlugin, ViewMode};
pub use grounding::{GroundY, ground_boids, reset_ground_caches};

/// The water plane riding `DemoTerrain::water_level`.
#[derive(Component)]
pub struct WaterPlane;

/// The playable square's side, in metres.
pub const TERRAIN_EXTENT_M: f32 = 1000.0;

/// The terrain surface. Closure-backed so the erosion-demo generators
/// (and the tests) can install any height function; the default is flat.
#[derive(Resource, Clone)]
pub struct HeightField {
    sampler: Arc<dyn Fn(f32, f32) -> f32 + Send + Sync>,
}

impl Default for HeightField {
    fn default() -> Self {
        HeightField::from_fn(|_, _| 0.0)
    }
}

impl HeightField {
    pub fn from_fn(sampler: impl Fn(f32, f32) -> f32 + Send + Sync + 'static) -> Self {
        HeightField {
            sampler: Arc::new(sampler),
        }
    }

    /// Surface height at a world XZ point.
    pub fn height(&self, x: f32, z: f32) -> f32 {
        (self.sampler)(x, z)
    }
}

/// The rendered square: one grid mesh (vertex Y holds the surface, so the
/// mesh IS the field's render — kept in a resource so later systems can
/// rewrite heights in place through `Assets<Mesh>`).
#[derive(Resource)]
pub struct TerrainMesh {
    pub handle: Handle<Mesh>,
}

/// Grid resolution per side (quads). Matches the erosion-filter demo's
/// 256×256 lattice.
pub const TERRAIN_RESOLUTION: usize = 256;

impl FromWorld for TerrainMesh {
    fn from_world(world: &mut World) -> Self {
        let handle = world.resource_mut::<Assets<Mesh>>().add(grid_mesh(0.0));
        TerrainMesh { handle }
    }
}

/// A `TERRAIN_RESOLUTION`² grid over the square with every vertex at
/// `y` — the shared substrate of the barebones plane and the erosion
/// demo's height updates.
pub(crate) fn grid_mesh(y: f32) -> Mesh {
    let verts = TERRAIN_RESOLUTION + 1;
    let half = TERRAIN_EXTENT_M / 2.0;
    let step = TERRAIN_EXTENT_M / TERRAIN_RESOLUTION as f32;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(verts * verts);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(verts * verts);
    for iz in 0..verts {
        for ix in 0..verts {
            positions.push([-half + ix as f32 * step, y, -half + iz as f32 * step]);
            uvs.push([ix as f32, iz as f32]);
        }
    }
    let mut indices: Vec<u32> = Vec::with_capacity(TERRAIN_RESOLUTION * TERRAIN_RESOLUTION * 6);
    for iz in 0..TERRAIN_RESOLUTION {
        for ix in 0..TERRAIN_RESOLUTION {
            let v0 = (iz * verts + ix) as u32;
            // Winding so faces point up (+y).
            indices.extend_from_slice(&[v0, v0 + verts as u32, v0 + 1]);
            indices.extend_from_slice(&[v0 + 1, v0 + verts as u32, v0 + verts as u32 + 1]);
        }
    }
    let mut mesh = Mesh::new(
        bevy::render::render_resource::PrimitiveTopology::TriangleList,
        Default::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        vec![[0.0f32, 1.0, 0.0]; verts * verts],
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_COLOR,
        vec![[1.0f32, 1.0, 1.0, 1.0]; verts * verts],
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Spawn the rendered square. `Ground` marks it for the drag-pan grab
/// raycast (lock_on_drag is off, so it is only a tag).
pub fn spawn_ground(
    mut commands: Commands,
    terrain: Res<TerrainMesh>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(terrain.handle.clone()),
        // The grid's COLOR_0 attribute arms vertex coloring; the demo's
        // albedo rides it.
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            ..default()
        })),
        Transform::IDENTITY,
        Ground,
    ));
}

use crate::kinematics::{HardCollision, TrackedByTree};

/// Cuboid obstacles are 1 m; their centre rides half a metre above the
/// surface.
pub(crate) const OBSTACLE_HALF_HEIGHT: f32 = 0.5;

/// A cuboid obstacle on the terrain surface, repelling boids from its
/// `normal` side.
#[derive(Component, Default)]
pub struct Obstacle {
    pub(crate) normal: Vec3,
}

/// Spawn bundle: obstacle + mesh + surface transform + spatial-tree
/// registration.
#[derive(Bundle, Default)]
pub struct ObstacleBundle {
    obstacle: Obstacle,
    mesh: Mesh3d,
    material: MeshMaterial3d<StandardMaterial>,
    transform: Transform,
    hard_collision: HardCollision,
    tracked: TrackedByTree,
}

impl ObstacleBundle {
    pub(crate) fn new(
        mesh: Handle<Mesh>,
        material: Handle<StandardMaterial>,
        normal: Vec3,
        pos: Vec3,
    ) -> Self {
        ObstacleBundle {
            obstacle: Obstacle { normal },
            hard_collision: Default::default(),
            tracked: Default::default(),
            mesh: Mesh3d(mesh),
            material: MeshMaterial3d(material),
            transform: Transform::from_xyz(pos.x, pos.y, pos.z),
        }
    }
}

/// Re-seat obstacles on the terrain surface (cheap at ~100 obstacles, so
/// it simply runs every frame).
pub fn project_obstacles_onto_field(
    mut query: Query<(&mut Transform, &Obstacle)>,
    field: Res<HeightField>,
) {
    for (mut transform, _obstacle) in &mut query {
        let t = transform.translation;
        transform.translation.y = field.height(t.x, t.z) + OBSTACLE_HALF_HEIGHT;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::mesh::VertexAttributeValues;

    #[test]
    fn grid_mesh_faces_up_with_full_coverage() {
        // Backface culling uses winding: every triangle faces +y, and the
        // grid is exactly (resolution+1)² verts / resolution²×2 tris.
        let mesh = grid_mesh(0.0);
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(p) => p.clone(),
            _ => panic!("unexpected position format"),
        };
        let indices = match mesh.indices().unwrap() {
            Indices::U32(i) => i.clone(),
            _ => panic!("unexpected index format"),
        };
        let verts = TERRAIN_RESOLUTION + 1;
        assert_eq!(positions.len(), verts * verts);
        assert_eq!(indices.len(), TERRAIN_RESOLUTION * TERRAIN_RESOLUTION * 6);
        for tri in 0..(indices.len() / 3) {
            let (a, b, c) = (
                indices[tri * 3] as usize,
                indices[tri * 3 + 1] as usize,
                indices[tri * 3 + 2] as usize,
            );
            let normal = (Vec3::from(positions[b]) - Vec3::from(positions[a]))
                .cross(Vec3::from(positions[c]) - Vec3::from(positions[a]));
            assert!(normal.y > 0.0, "triangle {tri} faces {normal:?}, not up");
        }
    }

}
