//! The terrain: streamed heightfield tiles in LOD rings.
//!
//! Each LOD level is a 5x5 ring of tiles around the camera focus, with
//! cell size growing ×4 per level (1 m near the camera to 65 km at
//! continental distance). Coverage is continuous by construction: a
//! coarser tile is skipped only when the finer level's square fully
//! contains it, so there are no annular gaps between levels; where rings
//! overlap, the coarser level sits slightly lower (per-level height bias)
//! so the finer surface wins the depth test. Level transitions hide their
//! step behind skirts. Heights come from [`HeightField`], so meshes,
//! grounding, and camera clearance agree by construction.

use super::HeightField;
use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::prelude::*;
use bevy_rts_camera::{Ground, RtsCamera};
use std::collections::HashMap;

/// Tile resolution: 32x32 quads per tile, any level.
const TILE_QUADS: usize = 32;
const TILE_VERTS: usize = TILE_QUADS + 1;
/// Cell size (metres per quad edge) per LOD level, ×4 per level. Level 0
/// gives 1 m detail in a 32 m tile; the top level's 65 km cells put the
/// 5x5 ring at ~10.5 Mm span — a small continent.
const LEVEL_CELL_M: [f32; 9] = [
    1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0,
];
/// Half-extent of a level's rendered square, in multiples of that level's
/// own tile size: the 5x5 ring around the focus tile.
const LEVEL_RING: i32 = 2;
/// Coarser levels sit this fraction of their cell size below the field, so
/// wherever rings overlap the finer (more accurate) surface wins depth
/// testing instead of z-fighting.
const LEVEL_DROP_CELL_FRACTION: f32 = 0.25;

/// Tile identity: LOD level + grid position at that level's tile size.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TileKey {
    level: u8,
    x: i32,
    z: i32,
}

impl TileKey {
    pub fn cell(self) -> f32 {
        LEVEL_CELL_M[self.level as usize]
    }

    pub fn tile_size(self) -> f32 {
        self.cell() * TILE_QUADS as f32
    }

    pub fn centre(self) -> Vec2 {
        let half = self.tile_size() / 2.0;
        Vec2::new(
            self.x as f32 * self.tile_size() + half,
            self.z as f32 * self.tile_size() + half,
        )
    }

    /// Does this tile's XZ footprint contain the point?
    pub fn contains(self, point: Vec2) -> bool {
        let half = self.tile_size() / 2.0;
        let c = self.centre();
        (point.x - c.x).abs() <= half && (point.y - c.y).abs() <= half
    }
}

/// Spawned terrain tiles, keyed for streaming (spawn/evict on focus
/// movement).
#[derive(Resource, Default)]
pub struct TerrainTiles {
    tiles: HashMap<TileKey, Entity>,
}

impl TerrainTiles {
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    pub fn keys(&self) -> impl Iterator<Item = TileKey> + '_ {
        self.tiles.keys().copied()
    }

    /// Is this world point inside any streamed tile's footprint? The
    /// streaming guarantee: every point between the finest ring and the
    /// continental rim is covered.
    pub fn covers(&self, point: Vec2) -> bool {
        self.tiles.keys().any(|&key| key.contains(point))
    }
}

/// A level's tile is redundant when the finer level's square fully
/// contains it: the finer 5x5 ring covers +/- (LEVEL_RING + 0.5) tiles of
/// the level below.
fn covered_by_finer_level(level: u8, key: TileKey, focus: Vec2) -> bool {
    if level == 0 {
        return false;
    }
    let finer_coverage = (LEVEL_RING as f32 + 0.5) * LEVEL_CELL_M[(level - 1) as usize]
        * TILE_QUADS as f32;
    let half = key.tile_size() / 2.0;
    let centre = key.centre();
    (centre.x - focus.x).abs() + half <= finer_coverage
        && (centre.y - focus.y).abs() + half <= finer_coverage
}

/// Build one tile mesh: a heightfield grid with a downward skirt ring that
/// hides cracks against neighbouring (possibly coarser) tiles.
pub(crate) fn tile_mesh(field: &HeightField, key: TileKey) -> Mesh {
    let cell = key.cell();
    let tile_size = key.tile_size();
    let origin_x = key.x as f32 * tile_size;
    let origin_z = key.z as f32 * tile_size;
    let drop = cell * LEVEL_DROP_CELL_FRACTION;
    let skirt = (cell * 1.5).max(8.0);

    // Interior grid heights, sampled once and reused for normals.
    let mut heights = [[0.0f32; TILE_VERTS]; TILE_VERTS];
    for iz in 0..TILE_VERTS {
        for ix in 0..TILE_VERTS {
            heights[iz][ix] =
                field.height(origin_x + ix as f32 * cell, origin_z + iz as f32 * cell) - drop;
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
    // around the tile edge so the walls face outward. Each corner appears
    // exactly once: edge D starts at iz = TILE_QUADS - 1 because edge C
    // already ends at the (iz = TILE_QUADS, ix = 0) corner.
    let border: Vec<usize> = (0..TILE_VERTS)
        .chain((1..TILE_VERTS).map(|i| i * TILE_VERTS + TILE_QUADS))
        .chain((0..TILE_QUADS).rev().map(|i| TILE_VERTS * TILE_QUADS + i))
        .chain((1..TILE_QUADS).rev().map(|i| i * TILE_VERTS))
        .collect();
    let skirt_start = positions.len();
    for &v in &border {
        let [x, y, z] = positions[v];
        positions.push([x, y - skirt, z]);
        normals.push([0.0, 1.0, 0.0]);
    }

    let mut indices: Vec<u32> = Vec::with_capacity(TILE_QUADS * TILE_QUADS * 6 + border.len() * 6);
    for iz in 0..TILE_QUADS {
        for ix in 0..TILE_QUADS {
            let v0 = (iz * TILE_VERTS + ix) as u32;
            // Winding so faces point UP (+y): (B - A) x (C - A) with
            // +x then +z edges gives -y, so the +z corner comes second.
            indices.extend_from_slice(&[v0, v0 + TILE_VERTS as u32, v0 + 1]);
            indices.extend_from_slice(&[
                v0 + 1,
                v0 + TILE_VERTS as u32,
                v0 + TILE_VERTS as u32 + 1,
            ]);
        }
    }
    let n = border.len();
    for i in 0..n {
        let top_a = border[i] as u32;
        let top_b = border[(i + 1) % n] as u32;
        let bot_a = (skirt_start + i) as u32;
        let bot_b = (skirt_start + (i + 1) % n) as u32;
        // Skirt walls face outward from the tile.
        indices.extend_from_slice(&[top_a, top_b, bot_a]);
        indices.extend_from_slice(&[top_b, bot_b, bot_a]);
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

/// Stream terrain tiles around the camera focus: compute the desired tile
/// set, spawn missing (mesh built synchronously — ~1k field samples per
/// tile), evict the rest.
pub fn stream_terrain_tiles(
    mut commands: Commands,
    mut tiles: ResMut<TerrainTiles>,
    cameras: Query<&RtsCamera>,
    field: Res<HeightField>,
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
                if covered_by_finer_level(level as u8, key, focus) {
                    continue;
                }
                desired.push(key);
            }
        }
    }

    // Evict tiles that left the desired set.
    let desired: Vec<TileKey> = desired;
    tiles.tiles.retain(|&key, &mut entity| {
        if desired.contains(&key) {
            true
        } else {
            commands.entity(entity).despawn();
            false
        }
    });

    // Spawn new tiles. Ground-marked so bevy_rts_camera raycasts (pan
    // focus, drag) follow the actual terrain surface.
    for key in desired {
        if tiles.tiles.contains_key(&key) {
            continue;
        }
        let mesh = meshes.add(tile_mesh(&field, key));
        let entity = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(materials.ground.clone()),
                Transform::IDENTITY,
                Ground,
            ))
            .id();
        tiles.tiles.insert(key, entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_field() -> HeightField {
        HeightField::from_fn(|_, _| 3.0)
    }

    #[test]
    fn tile_faces_point_up_and_skirts_outward() {
        // Backface culling uses winding: interior faces must have +y
        // geometric normals, and skirt walls must face away from the tile.
        let key = TileKey { level: 2, x: -3, z: 7 };
        let mesh = tile_mesh(&flat_field(), key);
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("positions") {
            VertexAttributeValues::Float32x3(p) => p.clone(),
            _ => panic!("unexpected position format"),
        };
        let indices: Vec<u32> = match mesh.indices().expect("indices") {
            Indices::U32(i) => i.clone(),
            _ => panic!("unexpected index format"),
        };

        let face_normal = |a: usize, b: usize, c: usize| {
            let pa = Vec3::from(positions[a]);
            let pb = Vec3::from(positions[b]);
            let pc = Vec3::from(positions[c]);
            (pb - pa).cross(pc - pa)
        };

        let interior_count = TILE_QUADS * TILE_QUADS * 2;
        for tri in 0..interior_count {
            let (a, b, c) = (
                indices[tri * 3] as usize,
                indices[tri * 3 + 1] as usize,
                indices[tri * 3 + 2] as usize,
            );
            let normal = face_normal(a, b, c);
            assert!(
                normal.y > 0.0,
                "interior triangle {tri} faces {normal:?}, not up"
            );
        }

        let c = key.centre();
        let centre = Vec3::new(c.x, 0.0, c.y);
        for tri in interior_count..(indices.len() / 3) {
            let (a, b, cc) = (
                indices[tri * 3] as usize,
                indices[tri * 3 + 1] as usize,
                indices[tri * 3 + 2] as usize,
            );
            let normal = face_normal(a, b, cc);
            let face_mid = (Vec3::from(positions[a]) + Vec3::from(positions[b])
                + Vec3::from(positions[cc]))
                / 3.0;
            assert!(
                (face_mid - centre).dot(normal) > 0.0,
                "skirt triangle {tri} faces inward"
            );
        }
    }

    #[test]
    fn coverage_is_continuous_from_the_camera_to_continental_distance() {
        // Ring-sampling: every point from beside the focus out to the
        // continental rim must fall inside some streamed tile — the
        // no-annular-gaps guarantee of the skip criterion.
        let mut covered = 0usize;
        let mut total = 0usize;
        for radius in [
            10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0, 4_000_000.0,
        ] {
            for angle in 0..16 {
                let a = angle as f32 * std::f32::consts::TAU / 16.0;
                let point = Vec2::new(a.cos(), a.sin()) * radius;
                total += 1;
                covered += tiles_covering_point(
                    desired_set_for_focus(Vec2::ZERO),
                    point,
                ) as usize;
            }
        }
        assert_eq!(covered, total, "some ring points are not covered");
    }

    /// The desired-set computation extracted for tests: which tiles would
    /// stream around this focus.
    fn desired_set_for_focus(focus: Vec2) -> Vec<TileKey> {
        let mut desired = Vec::new();
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
                    if covered_by_finer_level(level as u8, key, focus) {
                        continue;
                    }
                    desired.push(key);
                }
            }
        }
        desired
    }

    fn tiles_covering_point(tiles: Vec<TileKey>, point: Vec2) -> bool {
        tiles.iter().any(|&key| key.contains(point))
    }

    #[test]
    fn finest_level_streams_at_the_focus() {
        let tiles = desired_set_for_focus(Vec2::new(123.4, -45.6));
        assert!(
            tiles
                .iter()
                .any(|key| key.level == 0 && key.contains(Vec2::new(123.4, -45.6))),
            "no 1 m-resolution tile covers the focus"
        );
    }

    #[test]
    fn tile_budget_stays_bounded() {
        let tiles = desired_set_for_focus(Vec2::ZERO);
        // Hard bound: 25 tiles per level. With the focus on a tile corner
        // (the origin is one) even a coarse centre tile pokes out of the
        // finer square, so nothing is skipped; mid-tile focuses skip more.
        assert!(
            tiles.len() <= LEVEL_CELL_M.len() * 25,
            "tile budget exploded: {}",
            tiles.len()
        );
        assert!(tiles.len() > 9, "expected tiles at every level");
    }
}
