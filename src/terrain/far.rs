//! Far-field terrain: a quadtree of heightfield tiles for continent-scale
//! views, covering everything beyond the voxel near-field.
//!
//! bevy_voxel_world's chunk grid is fixed at 32 m per chunk no matter the
//! LOD level (LOD only downsamples voxels *within* a chunk), so voxel
//! coverage grows as chunks² with radius — continent views are impossible
//! that way. Here each LOD level is a ring of tiles around the camera
//! focus, with cell size doubling per level: bounded tile count, coverage
//! growing exponentially. Tiles sample the same pure `terrain_height` the
//! voxel lookup uses, so far and near terrain agree by construction
//! (modulo the voxel grid's integer height quantization).
//!
//! Far tiles are visual only: they carry no `Ground` marker, so camera and
//! cursor raycasts keep hitting the full-resolution voxel chunks.

use super::noise::terrain_height;
use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::prelude::*;
use bevy_rts_camera::RtsCamera;
use std::collections::HashMap;

/// Tile resolution: 32x32 quads per tile, any level.
const TILE_QUADS: usize = 32;
const TILE_VERTS: usize = TILE_QUADS + 1;
/// Cell size (metres per quad edge) per LOD level; each level doubles the
/// previous. Top level: 65536 m cells, ~2.1 Mm tiles — the 5x5 ring of the
/// two top levels spans the 10,000 km continent requirement.
const LEVEL_CELL_M: [f32; 8] = [
    4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0,
];
/// Tiles per level: the 5x5 ring around the focus tile.
const LEVEL_RING: i32 = 2;
/// Tiles whose centre is closer to the focus than this multiple of their
/// own tile size are skipped: the level below already renders them.
const INNER_CUT_TILE_MULT: f32 = 1.5;

/// Tile identity: LOD level + grid position at that level's tile size.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct TileKey {
    level: u8,
    x: i32,
    z: i32,
}

impl TileKey {
    fn tile_size(self) -> f32 {
        LEVEL_CELL_M[self.level as usize] * TILE_QUADS as f32
    }

    fn centre(self) -> Vec2 {
        let half = self.tile_size() / 2.0;
        Vec2::new(
            self.x as f32 * self.tile_size() + half,
            self.z as f32 * self.tile_size() + half,
        )
    }
}

/// Spawned far-terrain tiles, keyed for streaming (spawn/evict on focus
/// movement).
#[derive(Resource, Default)]
pub struct FarTerrain {
    tiles: HashMap<TileKey, Entity>,
}

impl FarTerrain {
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Distance (metres) from `origin` to the farthest rendered tile edge:
    /// the guaranteed covered radius.
    pub fn covered_radius_m(&self, origin: Vec2) -> f32 {
        self.tiles
            .keys()
            .map(|key| {
                let half = key.tile_size() / 2.0;
                (key.centre() - origin).length() + half
            })
            .fold(0.0, f32::max)
    }
}

/// Build one tile mesh: a heightfield grid with a downward skirt ring that
/// hides cracks against neighbouring (possibly coarser) tiles.
fn tile_mesh(key: TileKey) -> Mesh {
    let cell = LEVEL_CELL_M[key.level as usize];
    let tile_size = key.tile_size();
    let origin_x = key.x as f32 * tile_size;
    let origin_z = key.z as f32 * tile_size;
    let skirt = (cell * 1.5).max(8.0);

    // Interior grid heights, sampled once and reused for normals.
    let mut heights = [[0.0f32; TILE_VERTS]; TILE_VERTS];
    for iz in 0..TILE_VERTS {
        for ix in 0..TILE_VERTS {
            heights[iz][ix] = terrain_height(origin_x + ix as f32 * cell, origin_z + iz as f32 * cell);
        }
    }

    // Grid vertices + a duplicated border ring pushed down by `skirt`.
    let vert_count = TILE_VERTS * TILE_VERTS + 4 * TILE_VERTS;
    let mut positions = Vec::with_capacity(vert_count);
    let mut normals = Vec::with_capacity(vert_count);
    let height_at = |ix: usize, iz: usize| heights[iz][ix];

    for iz in 0..TILE_VERTS {
        for ix in 0..TILE_VERTS {
            let x = origin_x + ix as f32 * cell;
            let z = origin_z + iz as f32 * cell;
            positions.push([x, height_at(ix, iz), z]);
            // Central-difference normal from neighbouring samples (clamped
            // at borders — one cell of curl at the edge, hidden by skirts).
            let dx = height_at((ix + 1).min(TILE_QUADS), iz)
                - height_at(ix.saturating_sub(1), iz);
            let dz = height_at(ix, (iz + 1).min(TILE_QUADS))
                - height_at(ix, iz.saturating_sub(1));
            normals.push(Vec3::new(-dx, 2.0 * cell, -dz).normalize().to_array());
        }
    }

    // Skirt: duplicate the border ring, sunk by `skirt`. Winding runs
    // around the tile edge so the walls face outward.
    let border: Vec<usize> = (0..TILE_VERTS)
        .chain((1..TILE_VERTS).map(|i| i * TILE_VERTS + TILE_QUADS))
        .chain((0..TILE_QUADS).rev().map(|i| TILE_VERTS * TILE_QUADS + i))
        .chain((1..TILE_VERTS).rev().map(|i| i * TILE_VERTS))
        .collect();
    let mut skirt_start = positions.len();
    for &v in &border {
        let [x, y, z] = positions[v];
        positions.push([x, y - skirt, z]);
        normals.push([0.0, 1.0, 0.0]);
    }

    let mut indices: Vec<u32> = Vec::with_capacity(TILE_QUADS * TILE_QUADS * 6 + border.len() * 6);
    for iz in 0..TILE_QUADS {
        for ix in 0..TILE_QUADS {
            let v0 = (iz * TILE_VERTS + ix) as u32;
            indices.extend_from_slice(&[v0, v0 + 1, v0 + TILE_VERTS as u32]);
            indices.extend_from_slice(&[v0 + 1, v0 + TILE_VERTS as u32 + 1, v0 + TILE_VERTS as u32]);
        }
    }
    let n = border.len();
    for i in 0..n {
        let top_a = border[i] as u32;
        let top_b = border[(i + 1) % n] as u32;
        let bot_a = (skirt_start + i) as u32;
        let bot_b = (skirt_start + (i + 1) % n) as u32;
        indices.extend_from_slice(&[top_a, bot_a, top_b]);
        indices.extend_from_slice(&[top_b, bot_a, bot_b]);
    }

    let mut mesh = Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        Default::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(positions),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        VertexAttributeValues::Float32x3(normals),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Stream far-terrain tiles around the camera focus: compute the desired
/// tile set, spawn missing (mesh built synchronously — ~1k noise samples
/// per tile), evict the rest.
pub fn far_terrain_stream(
    mut commands: Commands,
    mut far: ResMut<FarTerrain>,
    cameras: Query<&RtsCamera>,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Res<crate::resources::Materials>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let focus = camera.focus.translation.xz();

    let mut desired: Vec<TileKey> = Vec::new();
    for (level, &cell) in LEVEL_CELL_M.iter().enumerate() {
        let tile_size = cell * TILE_QUADS as f32;
        let focus_tile = (focus / tile_size).floor().as_ivec2();
        for dx in -LEVEL_RING..=LEVEL_RING {
            for dz in -LEVEL_RING..=LEVEL_RING {
                let key = TileKey {
                    level: level as u8,
                    x: focus_tile.x + dx,
                    z: focus_tile.y + dz,
                };
                // Inner cut: the finer level below renders this area.
                if level > 0 && (key.centre() - focus).length() < INNER_CUT_TILE_MULT * tile_size
                {
                    continue;
                }
                desired.push(key);
            }
        }
    }

    // Evict tiles that left the desired set.
    let desired_set: Vec<TileKey> = desired.clone();
    far.tiles.retain(|&key, &mut entity| {
        if desired_set.contains(&key) {
            true
        } else {
            commands.entity(entity).despawn();
            false
        }
    });

    // Spawn new tiles.
    for key in desired {
        if far.tiles.contains_key(&key) {
            continue;
        }
        let mesh = meshes.add(tile_mesh(key));
        let entity = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(materials.white.clone()),
                Transform::IDENTITY,
            ))
            .id();
        far.tiles.insert(key, entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::MinimalPlugins;
    use bevy::app::App;
    use bevy::transform::TransformPlugin;

    fn far_app(focus: Vec3) -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin))
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<FarTerrain>()
            .init_resource::<crate::resources::Materials>()
            .add_systems(Startup, move |mut commands: Commands| {
                let focus = focus;
                commands.spawn((
                    Camera3d::default(),
                    RtsCamera {
                        focus: Transform::from_translation(focus),
                        target_focus: Transform::from_translation(focus),
                        ..Default::default()
                    },
                ));
            })
            .add_systems(Update, far_terrain_stream);
        // The Materials resource holds handles that setup normally fills;
        // give the white material a real asset before streaming runs.
        let white = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::from_color(Color::WHITE));
        app.world_mut()
            .resource_mut::<crate::resources::Materials>()
            .white = white;
        // Startup + a couple of streaming passes.
        app.update();
        app.update();
        app
    }

    #[test]
    fn far_tiles_cover_continent_radius_with_bounded_tile_count() {
        let app = far_app(Vec3::ZERO);
        let far = app.world().resource::<FarTerrain>();
        // 8 levels x 25 tiles max, minus inner cuts: comfortably under 200.
        assert!(
            far.tile_count() > 8,
            "expected tiles at multiple levels, got {}",
            far.tile_count()
        );
        assert!(
            far.tile_count() < 200,
            "tile budget exploded: {}",
            far.tile_count()
        );
        // 10,000 km across = 5,000 km radius from the focus.
        let covered = far.covered_radius_m(Vec2::ZERO);
        assert!(
            covered >= 5_000_000.0,
            "covered only {covered} m of the required 5,000,000 m radius"
        );
    }

    #[test]
    fn far_tiles_stream_with_a_moving_focus() {
        let mut app = far_app(Vec3::ZERO);
        // Teleport the focus ~3,000 km away: tiles must respawn around the
        // new position with the same bounded budget.
        let entity = app
            .world_mut()
            .query_filtered::<Entity, With<Camera3d>>()
            .single(app.world())
            .unwrap();
        let mut entity = app.world_mut().get_entity_mut(entity).unwrap();
        let mut rts = entity.get_mut::<RtsCamera>().unwrap();
        let new_focus = Transform::from_xyz(3_000_000.0, 0.0, 0.0);
        rts.focus = new_focus;
        rts.target_focus = new_focus;
        drop(rts);
        app.update();
        app.update();

        let far = app.world().resource::<FarTerrain>();
        assert!(far.tile_count() < 200, "tile budget exploded after move");
        let covered = far.covered_radius_m(Vec2::new(3_000_000.0, 0.0));
        assert!(
            covered >= 5_000_000.0,
            "covered only {covered} m around the new focus"
        );
    }
}
