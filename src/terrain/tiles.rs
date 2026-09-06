//! The terrain: streamed heightfield tiles in LOD rings.
//!
//! Each LOD level is a 5x5 ring of tiles around the RTS camera's focus
//! (the camera's ground position in freecam), with cell size growing ×4
//! per level (1 m near the camera to 65 km at continental distance); the
//! finest rendered level is gated by the camera's distance (see
//! [`LodMode`]). Coverage is continuous by construction: a
//! coarser tile is skipped only when the finer level's square fully
//! contains it, so there are no annular gaps between levels; where rings
//! overlap, the coarser level sits slightly lower (per-level height bias)
//! so the finer surface wins the depth test. Level transitions hide their
//! step behind skirts. Heights come from [`HeightField`], so meshes,
//! grounding, and camera clearance agree by construction; the same field
//! samples tint the tiles (rock/grass/dirt by slope and altitude,
//! drainage streaks from the erosion ridge map).

use super::{HeightField, TerrainSample};
use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::prelude::*;
use bevy_rts_camera::{Ground, RtsCamera};
use std::collections::HashMap;
use bevy::platform::time::Instant; // web-time on wasm: std's Instant::now() panics ("time not implemented on this platform")
use std::time::Duration;

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
/// Vertical separation between adjacent levels: each level above the
/// second sits a further half-cell below the one finer, so overlap fringes
/// resolve by depth test even on steep terrain (chord-over-valley error is
/// bounded by slope * fringe width, and fringes are under half a cell).
const LEVEL_DROP_STEP_FRACTION: f32 = 0.5;
/// The stitched fine rim sits this fraction of the parent cell BELOW the
/// parent plane: the rim is otherwise exactly coplanar with the coarse
/// fringe quads, which z-fights where they overlap.
const STITCH_BIAS_CELL_FRACTION: f32 = 0.02;

/// Coarser levels sit below the field so wherever rings overlap the finer
/// (more accurate) surface wins depth testing instead of z-fighting. The
/// two finest levels share a 1 m drop, making their ring boundary seamless
/// by construction; above that each level sinks a further half-cell.
fn level_drop(level: u8) -> f32 {
    let mut drop = 1.0;
    for l in 2..=level as usize {
        drop += LEVEL_CELL_M[l] * LEVEL_DROP_STEP_FRACTION;
    }
    drop
}

/// Parent (= next coarser) level's geometry, as seen from `level`.
const PARENT_CELL_MULT: f32 = 4.0;

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
/// movement). The value pairs the entity with the [`TileStamp`] the mesh
/// was baked with and its [`Handle`](bevy::asset::Handle): rims and holes
/// are computed from the ring position at spawn time, so a stamp mismatch
/// (focus crossed a tile boundary) queues a replacement — the old mesh
/// keeps rendering until the replacement spawns, so streaming never opens
/// a hole (see `stream_terrain_tiles`).
#[derive(Resource, Default)]
pub struct TerrainTiles {
    tiles: HashMap<TileKey, (Entity, TileStamp, Handle<Mesh>)>,
}

/// Ring position a tile's mesh was baked with: the focus tile at the
/// tile's own level (which edges are stitching rims), at the finer level
/// (where the cut-out hole lies), and whether this tile's mesh actually
/// cuts that hole (the finer level renders AND this tile's footprint
/// overlaps its square — the distance gate, [`LodMode`], lifts whole fine
/// levels in and out). Only a flip of what the mesh depends on forces a
/// respawn; tiles whose geometry is unchanged by a ring move keep their
/// stamp and are not rebuilt.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TileStamp {
    focus_tile: IVec2,
    fine_focus_tile: IVec2,
    cuts_hole: bool,
}

/// Retired tile meshes, kept for instant re-spawn. Zooming and panning
/// back over ground you just left re-uses the baked mesh instead of
/// re-running the erosion-filtered field ~1k times — a cache hit costs no
/// mesh time and no [`StreamBudget`]. Bounded: the least-recently-used
/// meshes beyond [`TILE_CACHE_TILES`] are dropped from `Assets<Mesh>`
/// (live tiles are never cached, so nothing referenced is freed).
#[derive(Resource, Default)]
pub struct TileMeshCache {
    entries: HashMap<(TileKey, TileStamp), (Handle<Mesh>, u64)>,
    clock: u64,
}

/// Cache capacity in tiles. ~75 KB of vertex/index data per tile →
/// ~20 MB; a full desired set is ~120-150 tiles, so a zoom gate cycle or
/// a pan detour fits whole.
const TILE_CACHE_TILES: usize = 256;

impl TileMeshCache {
    /// Take a mesh out of the cache for re-spawn, if present.
    fn take(&mut self, key: TileKey, stamp: TileStamp) -> Option<Handle<Mesh>> {
        self.clock += 1;
        self.entries.remove(&(key, stamp)).map(|(handle, _)| handle)
    }

    /// Park a retired tile's mesh in the cache; evict the LRU overflow
    /// (freeing its asset).
    fn put(
        &mut self,
        key: TileKey,
        stamp: TileStamp,
        handle: Handle<Mesh>,
        meshes: &mut Assets<Mesh>,
    ) {
        self.clock += 1;
        let clock = self.clock;
        self.entries.insert((key, stamp), (handle, clock));
        while self.entries.len() > TILE_CACHE_TILES {
            let Some((&oldest, _)) = self.entries.iter().min_by_key(|(_, (_, used))| *used)
            else {
                break;
            };
            if let Some((handle, _)) = self.entries.remove(&oldest) {
                meshes.remove(&handle);
            }
        }
    }

    /// Drop every cached mesh (the height field changed; the meshes are
    /// baked from the old field).
    fn drain(&mut self, meshes: &mut Assets<Mesh>) {
        for (_, (handle, _)) in self.entries.drain() {
            meshes.remove(&handle);
        }
    }
}

/// Per-frame meshing budget for streaming. A tuning edit respawns every
/// tile at once, and with the erosion filter each tile mesh costs several
/// times more field work — instead of freezing one frame, the rebuild
/// spreads across frames and the new world floods outward from the
/// camera. Absent resource means unlimited (tests, loading screens).
#[derive(Resource, Clone, Copy)]
pub struct StreamBudget {
    pub mesh_millis: u64,
}

impl Default for StreamBudget {
    fn default() -> Self {
        StreamBudget { mesh_millis: 2 }
    }
}

impl StreamBudget {
    // No UNLIMITED constant: an absent resource already means unlimited
    // (tests, loading screens) — see `stream_terrain_tiles`.
}

/// How the finest rendered LOD level is chosen. Toggled from the F3 panel.
///
/// `CameraDistance` (the default) is correct LOD: levels whose cells are
/// too fine for the camera's distance to the ring centre are not
/// rendered, so detail scales with how far the camera actually is — a
/// camera 30 km up never streams 1 m cells, one 2 m off the ground does.
///
/// The rings themselves always centre on the RTS focus, which zoom does
/// not move. Centring them on the camera body — the literal reading of
/// "distance from the camera" — is unworkable with this RTS camera: its
/// XZ position slides ~11 km over one zoom gesture (the body trails the
/// focus by `height · tan(pitch)` and the pitch widens as you zoom in),
/// which re-invalidates every fine tile's stamp at frame rate, starves
/// the [`StreamBudget`] (the spawn queue is finest-first) and leaves
/// voids until seconds after the gesture ends.
///
/// `FinestAtFocus` is the debug mode this shipped with: the finest level
/// always renders at the ring centre regardless of camera distance —
/// full detail wherever you look, at any altitude.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum LodMode {
    #[default]
    CameraDistance,
    FinestAtFocus,
}

/// Quads across the near view at the finest level — the screen-error
/// budget of the distance gate. 256: walking height keeps the 1 m cells,
/// ~300 m altitude keeps 4 m cells, the 30 km ceiling keeps 256 m cells.
const LOD_CELLS_PER_VIEW: f32 = 256.0;

/// The finest level worth rendering at this camera-to-centre distance:
/// the first level whose cell size subtends at most
/// [`LOD_CELLS_PER_VIEW`] quads across the near view. Distance 0 (or the
/// debug mode) keeps level 0.
fn finest_level(camera_distance_m: f32) -> u8 {
    let min_cell_m = camera_distance_m / LOD_CELLS_PER_VIEW;
    LEVEL_CELL_M
        .iter()
        .position(|&cell| cell >= min_cell_m)
        .unwrap_or(0) as u8
}

/// The XZ point the LOD rings centre on: the RTS focus — the projection
/// of the camera's view ray onto the terrain — or, without an RTS camera
/// (freecam), the camera's own ground position.
fn ring_center(rts_camera: Option<&RtsCamera>, camera_xz: Vec2) -> Vec2 {
    rts_camera
        .map(|camera| camera.focus.translation.xz())
        .unwrap_or(camera_xz)
}

impl TerrainTiles {
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    pub fn keys(&self) -> impl Iterator<Item = TileKey> + '_ {
        self.tiles.keys().copied()
    }

    pub(crate) fn entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.tiles.values().map(|&(entity, _, _)| entity)
    }

    /// Is this world point inside any streamed tile's footprint? The
    /// streaming guarantee: every point between the finest ring and the
    /// continental rim is covered.
    pub fn covers(&self, point: Vec2) -> bool {
        self.tiles.keys().any(|&key| key.contains(point))
    }
}

/// World-space XZ square a coarse level's mesh cuts out: where the finer
/// level's 5x5 ring renders, ±[`LEVEL_RING`] finer tiles around the finer
/// focus tile. Tile-aligned (computed from the finer centre tile), because
/// that is exactly what is on screen — a centre-centred approximation can
/// both over-skip (opening slit holes) and under-skip (leaving whole-tile
/// overlap sheets) depending on where inside its tile the centre sits.
fn hole_square(level: u8, fine_focus_tile: IVec2) -> (Vec2, Vec2) {
    let finer_size = LEVEL_CELL_M[(level - 1) as usize] * TILE_QUADS as f32;
    (
        (fine_focus_tile - LEVEL_RING).as_vec2() * finer_size,
        (fine_focus_tile + LEVEL_RING + 1).as_vec2() * finer_size,
    )
}

/// The hole square for a ring centre: [`hole_square`] at the centre's
/// finer-level focus tile.
fn finer_render_square(level: u8, center: Vec2) -> (Vec2, Vec2) {
    let finer_size = LEVEL_CELL_M[(level - 1) as usize] * TILE_QUADS as f32;
    hole_square(level, (center / finer_size).floor().as_ivec2())
}

/// Does this tile's baked mesh render the ground at `point`? False where
/// the tile is absent or its cut-out hole swallows the point — the mesh
/// cuts quads whose centre is strictly inside the hole square.
fn renders_at(level: u8, stamp: &TileStamp, point: Vec2) -> bool {
    if !stamp.cuts_hole {
        return true;
    }
    let (hmin, hmax) = hole_square(level, stamp.fine_focus_tile);
    !(hmin.x < point.x && point.x < hmax.x && hmin.y < point.y && point.y < hmax.y)
}

/// A level's tile is redundant when the finer level's rendered square
/// fully contains it (the clipmap invariant: coarse levels render only
/// where finer ones don't). At or below `min_level` (the distance-gated
/// finest level) nothing finer renders, so those levels are never
/// covered-and-skipped. Partially-overlapping tiles are NOT skipped —
/// tile_mesh cuts their overlap out per-quad instead.
fn covered_by_finer_level(level: u8, key: TileKey, center: Vec2, min_level: u8) -> bool {
    if level <= min_level {
        return false;
    }
    let (min, max) = finer_render_square(level, center);
    let half = key.tile_size() / 2.0;
    let c = key.centre();
    c.x - half >= min.x && c.x + half <= max.x && c.y - half >= min.y && c.y + half <= max.y
}

/// Parent (= next coarser) level's surface height at a boundary point:
/// linear interpolation between the two enclosing parent grid samples
/// along the boundary direction, minus the parent's drop and a small bias.
/// Tile edges are multiples of the parent cell, so edge endpoints coincide
/// with parent samples and the interpolation is exact there. On corners
/// both axes are aligned and both branches reduce to the same parent
/// sample. The bias drops the stitched rim a hair below the parent plane
/// so overlapping coarse fringe quads (same plane, by construction) win
/// cleanly instead of z-fighting.
fn parent_lerp_height(field: &HeightField, level: u8, x: f32, z: f32) -> f32 {
    let parent_cell = LEVEL_CELL_M[level as usize] * PARENT_CELL_MULT;
    let parent_drop = level_drop(level + 1) + parent_cell * STITCH_BIAS_CELL_FRACTION;
    let gx = x / parent_cell;
    let gz = z / parent_cell;
    // Whichever axis is off the parent grid is the varying one.
    if (gz - gz.round()).abs() > 1e-4 {
        let i = gz.floor();
        let z0 = i * parent_cell;
        let t = gz - i;
        let h0 = field.height(x, z0);
        let h1 = field.height(x, z0 + parent_cell);
        h0 + (h1 - h0) * t - parent_drop
    } else {
        let i = gx.floor();
        let x0 = i * parent_cell;
        let t = gx - i;
        let h0 = field.height(x0, z);
        let h1 = field.height(x0 + parent_cell, z);
        h0 + (h1 - h0) * t - parent_drop
    }
}

/// Build one tile mesh: a heightfield grid with a downward skirt ring that
/// hides cracks against neighbouring (possibly coarser) tiles.
///
/// `focus_tile` is the focus tile's grid position at this key's level: the
/// four outer rim edges of the 5x5 ring border a COARSER level, and their
/// border vertices are stitched onto the parent level's linear
/// interpolation (minus the parent's drop) — the CPU equivalent of
/// terrain_renderer's vertex morphing, so the fine surface meets the
/// coarse one along the exact shared boundary with no crack or step.
///
/// `hole` is the finer level's rendered square: interior quads whose
/// centre falls inside it are CUT, so this coarse tile does not sheet a
/// second surface underneath the finer one (the clipmap invariant —
/// coarse renders only where fine doesn't). Quads kept by the centre rule
/// may overlap the square by under half a cell; the level drop and stitch
/// bias make the fine surface win that fringe.
pub(crate) fn tile_mesh(
    field: &HeightField,
    key: TileKey,
    focus_tile: IVec2,
    hole: Option<(Vec2, Vec2)>,
) -> Mesh {
    let cell = key.cell();
    let tile_size = key.tile_size();
    let origin_x = key.x as f32 * tile_size;
    let origin_z = key.z as f32 * tile_size;
    let drop = level_drop(key.level);
    let skirt = (cell * 1.5).max(8.0);
    // Rim edges: the neighbour in that direction lies outside the 5x5
    // ring, so the adjacent level there is coarser. The TOP level's outer
    // rim borders the void, not a coarser level — nothing to stitch to.
    let can_stitch = (key.level as usize + 1) < LEVEL_CELL_M.len();
    let rim_x_plus = can_stitch && key.x == focus_tile.x + LEVEL_RING;
    let rim_x_minus = can_stitch && key.x == focus_tile.x - LEVEL_RING;
    let rim_z_plus = can_stitch && key.z == focus_tile.y + LEVEL_RING;
    let rim_z_minus = can_stitch && key.z == focus_tile.y - LEVEL_RING;

    // Interior grid: one field sample per vertex feeds positions, normals
    // and colors. `heights` is what the mesh shows (drop applied, rims
    // re-stitched); `samples` keeps the unstitched world-space fields for
    // shading, since altitude gates must not see LOD drop offsets.
    let mut heights = [[0.0f32; TILE_VERTS]; TILE_VERTS];
    let mut samples = [[TerrainSample::default(); TILE_VERTS]; TILE_VERTS];
    for iz in 0..TILE_VERTS {
        for ix in 0..TILE_VERTS {
            let sample =
                field.sample(origin_x + ix as f32 * cell, origin_z + iz as f32 * cell);
            heights[iz][ix] = sample.height - drop;
            samples[iz][ix] = sample;
        }
    }

    // Stitch rim edges onto the parent level's surface.
    let mut stitch = |ix: usize, iz: usize| {
        let x = origin_x + ix as f32 * cell;
        let z = origin_z + iz as f32 * cell;
        heights[iz][ix] = parent_lerp_height(field, key.level, x, z);
    };
    if rim_x_plus {
        for iz in 0..TILE_VERTS {
            stitch(TILE_QUADS, iz);
        }
    }
    if rim_x_minus {
        for iz in 0..TILE_VERTS {
            stitch(0, iz);
        }
    }
    if rim_z_plus {
        for ix in 0..TILE_VERTS {
            stitch(ix, TILE_QUADS);
        }
    }
    if rim_z_minus {
        for ix in 0..TILE_VERTS {
            stitch(ix, 0);
        }
    }

    // Grid vertices + a duplicated border ring pushed down by `skirt`.
    let vert_count = TILE_VERTS * TILE_VERTS + 4 * TILE_VERTS;
    let mut positions = Vec::with_capacity(vert_count);
    let mut normals = Vec::with_capacity(vert_count);
    let mut colors = Vec::with_capacity(vert_count);
    let palette = ground_palette();

    for iz in 0..TILE_VERTS {
        for ix in 0..TILE_VERTS {
            let x = origin_x + ix as f32 * cell;
            let z = origin_z + iz as f32 * cell;
            positions.push([x, heights[iz][ix], z]);
            // Normal from the field's analytic slope: field sampling is
            // tile-independent, so neighbouring tiles agree exactly on
            // shared vertices (sampling this tile's clamped grid instead
            // drew a lighting grid over the whole world), and the gullies
            // carve the lighting with no extra field samples.
            let slope = samples[iz][ix].slope;
            normals.push(Vec3::new(-slope.x, 1.0, -slope.y).normalize().to_array());
            colors.push(ground_color(&palette, samples[iz][ix]).to_array());
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
        colors.push(colors[v]);
    }

    let mut indices: Vec<u32> = Vec::with_capacity(TILE_QUADS * TILE_QUADS * 6 + border.len() * 6);
    for iz in 0..TILE_QUADS {
        for ix in 0..TILE_QUADS {
            // Cut the hole: no coarse quads under the finer level's square.
            if let Some((hmin, hmax)) = hole {
                let cx = origin_x + (ix as f32 + 0.5) * cell;
                let cz = origin_z + (iz as f32 + 0.5) * cell;
                if cx > hmin.x && cx < hmax.x && cz > hmin.y && cz < hmax.y {
                    continue;
                }
            }
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
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, VertexAttributeValues::Float32x4(colors));
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// The terrain palette as linear RGBA — sRGB values from the erosion
/// reference demo's texturing: [grass low, grass high, dirt, cliff,
/// drainage chalk, snow]. Converted once per mesh, not per vertex.
fn ground_palette() -> [Vec4; 6] {
    [
        Color::srgb(0.15, 0.30, 0.10),
        Color::srgb(0.40, 0.50, 0.20),
        Color::srgb(0.60, 0.50, 0.40),
        Color::srgb(0.22, 0.20, 0.20),
        Color::srgb(0.82, 0.76, 0.62),
        Color::srgb(0.92, 0.94, 0.96),
    ]
    .map(|c| {
        let l = c.to_linear();
        Vec4::new(l.red, l.green, l.blue, l.alpha)
    })
}

/// Per-vertex tint from one field sample — the CPU adaptation of the
/// reference demo's per-fragment material: grass two-tone by altitude,
/// dirt then cliff as steeps rise, chalky drainage streaks where the
/// ridge map marks gully creases, snow on the highest ridges.
fn ground_color(palette: &[Vec4; 6], sample: TerrainSample) -> Vec4 {
    // Slope is m/m: grass holds to ~31°, sheer cliff past ~45°.
    let steepness = sample.slope.length();
    let grass = Vec4::lerp(palette[0], palette[1], smoothstep(4.0, 28.0, sample.height));
    let mut c = Vec4::lerp(grass, palette[2], smoothstep(0.3, 0.55, steepness));
    c = Vec4::lerp(c, palette[3], smoothstep(0.55, 1.0, steepness));
    // ridge_map ≈ -1 at crease centres; only the narrow core band paints
    // the stream beds, or the chalk speckles over every slope (tuned on
    // screen: 0.3 fired as isolated dots).
    let ridgemap = (sample.ridge_map * 0.5 + 0.5).clamp(0.0, 1.0);
    let drainage = (1.0 - ridgemap / 0.15).clamp(0.0, 1.0);
    c = Vec4::lerp(c, palette[4], drainage);
    Vec4::lerp(c, palette[5], smoothstep(55.0, 70.0, sample.height))
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The desired tile set around a ring centre with `min_level` as the
/// finest level that renders (the camera-distance gate; 0 = the full
/// stack): one 5x5 ring per level from there up, minus tiles the finer
/// level fully covers, each stamped with the ring position it must be
/// baked with. Shared by the streaming system and the tests.
pub(crate) fn desired_tiles(center: Vec2, min_level: u8) -> Vec<(TileKey, TileStamp)> {
    let mut desired = Vec::new();
    for (level, &cell) in LEVEL_CELL_M.iter().enumerate() {
        let level = level as u8;
        if level < min_level {
            continue;
        }
        let tile_size = cell * TILE_QUADS as f32;
        let center_tile = (center / tile_size).floor().as_ivec2();
        let fine_size = if level == 0 {
            tile_size
        } else {
            LEVEL_CELL_M[(level - 1) as usize] * TILE_QUADS as f32
        };
        let fine_center_tile = (center / fine_size).floor().as_ivec2();
        // The hole this level's tiles cut: the finer square, but only
        // where it exists (level > min_level) and only on tiles whose
        // footprint actually overlaps it — a ring move or gate flip then
        // restamps just the tiles whose mesh really changed.
        let hole = if level > min_level {
            Some(finer_render_square(level, center))
        } else {
            None
        };
        for dx in -LEVEL_RING..=LEVEL_RING {
            for dz in -LEVEL_RING..=LEVEL_RING {
                let key = TileKey {
                    level,
                    x: center_tile.x + dx,
                    z: center_tile.y + dz,
                };
                if covered_by_finer_level(level, key, center, min_level) {
                    continue;
                }
                let cuts_hole = hole.is_some_and(|(hmin, hmax)| {
                    let half = tile_size / 2.0;
                    let c = key.centre();
                    c.x + half > hmin.x
                        && c.x - half < hmax.x
                        && c.y + half > hmin.y
                        && c.y - half < hmax.y
                });
                desired.push((
                    key,
                    TileStamp {
                        focus_tile: center_tile,
                        fine_focus_tile: fine_center_tile,
                        cuts_hole,
                    },
                ));
            }
        }
    }
    desired
}

/// Stream terrain tiles: compute the desired tile set, spawn missing
/// (mesh built synchronously — ~1k field samples per tile, at most a
/// [`StreamBudget`] worth of *builds* per frame; [`TileMeshCache`] hits
/// are free), retire the rest. The rings centre on the RTS focus (the
/// camera's ground position in freecam) and the finest level is gated by
/// the camera's distance to that centre — see [`LodMode`].
///
/// Streaming never opens a hole in the ground. Stale tiles (their ring
/// position moved, or the distance gate flipped their hole) keep
/// rendering until their replacement spawns in the same frame — an atomic
/// swap, not a despawn-then-build. Tiles dropped from the desired set
/// (gate lifted their whole level, focus moved on) keep rendering until a
/// current tile actually renders over their ground: the coarse ring's
/// baked hole would otherwise expose the void beneath. A stale rim or a
/// few frames of coarse-over-fine overlap beat a hole every time.
pub fn stream_terrain_tiles(
    mut commands: Commands,
    mut tiles: ResMut<TerrainTiles>,
    mut cache: ResMut<TileMeshCache>,
    cameras: Query<(&Transform, Option<&RtsCamera>), With<Camera3d>>,
    lod_mode: Option<Res<LodMode>>,
    field: Res<HeightField>,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Res<crate::resources::Materials>,
    budget: Option<Res<StreamBudget>>,
) {
    let Ok((camera_transform, rts_camera)) = cameras.single() else {
        return;
    };
    let center = ring_center(rts_camera, camera_transform.translation.xz());
    // The camera-to-centre distance gates the finest level. The RTS focus
    // already sits on the field (focus_camera_on_ground); a freecam has no
    // focus, so sample the field under the camera.
    let center_ground_y = rts_camera
        .map(|camera| camera.focus.translation.y)
        .unwrap_or_else(|| field.height(center.x, center.y));
    let camera_distance = match lod_mode.as_deref().copied().unwrap_or_default() {
        LodMode::CameraDistance => camera_transform
            .translation
            .distance(Vec3::new(center.x, center_ground_y, center.y)),
        LodMode::FinestAtFocus => 0.0,
    };
    let min_level = finest_level(camera_distance);

    // The height field changed (tuning edit): the entire rendered world is
    // stale, cached meshes included. Drop it all — the spawn pass below
    // then rebuilds the desired set from the new field, spreading the
    // meshes across frames under the budget so a slider drag never
    // freezes the frame.
    if field.is_changed() {
        for (_, (entity, ..)) in tiles.tiles.drain() {
            commands.entity(entity).despawn();
        }
        cache.drain(&mut meshes);
    }

    let desired = desired_tiles(center, min_level);

    // Half-extent of everything the desired set can ever cover (the top
    // ring): beyond it, a dropped tile will never be covered — retire it
    // immediately instead of leaking it.
    let top_tile = LEVEL_CELL_M[LEVEL_CELL_M.len() - 1] * TILE_QUADS as f32;
    let continental_reach = (LEVEL_RING as f32 + 0.5) * top_tile;

    // Retire pass. Current and stale-stamped tiles stay (the spawn pass
    // swaps the stale ones atomically); dropped keys — the gate lifted
    // their whole level, or the focus moved on — stay until current tiles
    // render over their centre, then retire into the cache.
    let mut retired: Vec<TileKey> = Vec::new();
    for (&key, _) in tiles.tiles.iter() {
        if desired.iter().any(|(k, _)| *k == key) {
            continue; // current or stale: the spawn pass owns it
        }
        // Dropped: nothing will ever cover it out at the continental rim,
        // and the desired set always covers everything closer, so the
        // wait is bounded.
        let centre = key.centre();
        if (centre - center).abs().max_element() > continental_reach {
            retired.push(key);
            continue;
        }
        let covered = desired.iter().any(|(cover_key, cover_stamp)| {
            cover_key.contains(centre)
                && matches!(
                    tiles.tiles.get(cover_key),
                    Some((_, live, _)) if *live == *cover_stamp
                )
                && renders_at(cover_key.level, cover_stamp, centre)
        });
        if covered {
            retired.push(key);
        }
    }
    for key in retired {
        if let Some((entity, stamp, handle)) = tiles.tiles.remove(&key) {
            cache.put(key, stamp, handle, &mut meshes);
            commands.entity(entity).despawn();
        }
    }

    // Spawn pass. Cache hits apply free; misses are meshed in parallel
    // waves — one tile costs ~2 ms of field sampling serially (see the
    // `tile_mesh` perf test), so a 2 ms budget buys several tiles only
    // across cores. Camera focus-following samples the HeightField
    // directly (terrain::camera); the Ground marker remains for
    // drag-pan's grab-point raycast. Absent budget means unlimited
    // (tests, loading screens).
    let deadline = budget
        .as_deref()
        .map(|b| Instant::now() + Duration::from_millis(b.mesh_millis));
    let mut built = 0usize;
    let mut index = 0usize;
    while index < desired.len() {
        if built > 0 && deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        // Assemble the next wave: misses to mesh (a full parallel wave,
        // or a single guaranteed-progress tile when the budget is already
        // spent), cache hits passing straight through.
        let spent = deadline.is_some_and(|d| Instant::now() >= d);
        let wave_cap = if spent { 1 } else { parallelism().max(1) };
        let mut wave: Vec<(TileKey, TileStamp, Option<Handle<Mesh>>)> = Vec::new();
        while index < desired.len() && wave.len() < wave_cap {
            let (key, stamp) = desired[index];
            index += 1;
            if let Some((_, live, _)) = tiles.tiles.get(&key) {
                if *live == stamp {
                    continue; // current
                }
            }
            wave.push((key, stamp, cache.take(key, stamp)));
        }
        if wave.is_empty() {
            continue;
        }
        let fresh = mesh_wave(&field, &wave);
        built += fresh.len();
        for (i, mesh) in fresh {
            let handle = meshes.add(mesh);
            wave[i].2 = Some(handle);
        }
        for (key, stamp, mesh) in wave {
            // Every wave entry has a handle by now: cache hit or fresh.
            let mesh = mesh.expect("wave entry left unmeshed");
            // Atomic swap: the replacement enters the map this frame; the
            // retired mesh parks in the cache.
            let entity = commands
                .spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(materials.ground.clone()),
                    Transform::IDENTITY,
                    Ground,
                ))
                .id();
            if let Some((old_entity, old_stamp, old_handle)) =
                tiles.tiles.insert(key, (entity, stamp, mesh))
            {
                cache.put(key, old_stamp, old_handle, &mut meshes);
                commands.entity(old_entity).despawn();
            }
        }
    }
}

/// Usable cores for the meshing waves (wasm reports 1; the serial
/// fallback there needs no threads).
fn parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Mesh this wave's cache misses across cores, returning `(wave index,
/// mesh)` pairs. Every tile costs about the same (~equal vertex counts),
/// so even slices need no work stealing. Single-core falls back to
/// serial — `std::thread::scope` needs real threads, which wasm does not
/// have.
fn mesh_wave(
    field: &HeightField,
    wave: &[(TileKey, TileStamp, Option<Handle<Mesh>>)],
) -> Vec<(usize, Mesh)> {
    let to_build: Vec<usize> = wave
        .iter()
        .enumerate()
        .filter(|(_, (_, _, cached))| cached.is_none())
        .map(|(i, _)| i)
        .collect();
    let build = |indexes: &[usize]| -> Vec<(usize, Mesh)> {
        indexes
            .iter()
            .map(|&i| {
                let (key, stamp, _) = wave[i];
                let hole = if stamp.cuts_hole {
                    Some(hole_square(key.level, stamp.fine_focus_tile))
                } else {
                    None
                };
                (i, tile_mesh(field, key, stamp.focus_tile, hole))
            })
            .collect()
    };
    let workers = parallelism().min(to_build.len()).max(1);
    if workers <= 1 || to_build.len() <= 1 {
        return build(&to_build);
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = to_build
            .chunks(to_build.len().div_ceil(workers))
            .map(|chunk| scope.spawn(move || build(chunk)))
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("tile meshing panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::super::{TerrainTuning, rebuild_height_field};
    use super::*;

    fn flat_field() -> HeightField {
        HeightField::from_fn(|_, _| 3.0)
    }

    #[test]
    fn tile_faces_point_up_and_skirts_outward() {
        // Backface culling uses winding: interior faces must have +y
        // geometric normals, and skirt walls must face away from the tile.
        let key = TileKey {
            level: 2,
            x: -3,
            z: 7,
        };
        // Focus on the tile itself: no rim edges, pure interior test.
        let mesh = tile_mesh(&flat_field(), key, IVec2::new(-3, 7), None);
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
            let face_mid =
                (Vec3::from(positions[a]) + Vec3::from(positions[b]) + Vec3::from(positions[cc]))
                    / 3.0;
            assert!(
                (face_mid - centre).dot(normal) > 0.0,
                "skirt triangle {tri} faces inward"
            );
        }
    }

    #[test]
    fn tuning_change_rebuilds_the_whole_tile_set() {
        // The live-regeneration contract: mutating the tuning resource
        // replaces every streamed tile with fresh meshes.
        let mut app = App::new();
        app.insert_resource(TerrainTuning::default())
            .insert_resource(HeightField::default())
            .init_resource::<TerrainTiles>()
            .init_resource::<TileMeshCache>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<crate::resources::Materials>()
            .add_systems(
                Update,
                (
                    rebuild_height_field,
                    stream_terrain_tiles.after(rebuild_height_field),
                ),
            );
        // One camera entity: streaming keys the ring centre off the
        // Camera3d entity's transform (plus its RtsCamera focus in Focus
        // mode) via `cameras.single()`.
        app.world_mut()
            .spawn((Camera3d::default(), RtsCamera::default()));
        let ground = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::from_color(Color::WHITE));
        app.world_mut()
            .resource_mut::<crate::resources::Materials>()
            .ground = ground;
        app.update();
        app.update();

        let before = tile_entities(&app);
        assert!(!before.is_empty(), "no tiles streamed initially");

        app.world_mut().resource_mut::<TerrainTuning>().ridge_max_m += 10.0;
        app.update();
        app.update();

        let after = tile_entities(&app);
        assert!(!after.is_empty());
        assert!(
            before.iter().all(|e| !after.contains(e)),
            "tiles were not rebuilt: {} entities survived the tuning change",
            before.iter().filter(|e| after.contains(e)).count()
        );
    }

    fn tile_entities(app: &App) -> Vec<Entity> {
        app.world().resource::<TerrainTiles>().entities().collect()
    }

    #[test]
    fn coverage_is_continuous_from_the_camera_to_continental_distance() {
        // Ring-sampling: every point from beside the focus out to the
        // continental rim must fall inside some streamed tile — the
        // no-annular-gaps guarantee of the skip criterion. Dense radii
        // near the finest ring (16 m steps) so thin slit holes between
        // levels cannot hide between samples. The guarantee must hold at
        // any distance-gated finest level, not just the full stack.
        for min_level in [0u8, 2, 4] {
            let mut covered = 0usize;
            let mut total = 0usize;
            let mut radii: Vec<f32> = (1..24).map(|i| i as f32 * 16.0).collect();
            radii.extend_from_slice(&[1_000.0, 10_000.0, 100_000.0, 1_000_000.0, 4_000_000.0]);
            for radius in radii {
                for angle in 0..16 {
                    let a = angle as f32 * std::f32::consts::TAU / 16.0;
                    let point = Vec2::new(a.cos(), a.sin()) * radius;
                    total += 1;
                    covered +=
                        tiles_covering_point(desired_set_around(Vec2::ZERO, min_level), point)
                            as usize;
                }
            }
            assert_eq!(covered, total, "min_level {min_level}: uncovered points");
        }
    }

    /// The desired-set computation extracted for tests: which tiles would
    /// stream around this ring centre at this finest level.
    fn desired_set_around(center: Vec2, min_level: u8) -> Vec<TileKey> {
        desired_tiles(center, min_level)
            .into_iter()
            .map(|(key, _)| key)
            .collect()
    }

    #[test]
    fn ring_center_prefers_the_rts_focus() {
        // Rings follow the focus (which zoom never moves); only a freecam
        // — no RTS camera — centres on the camera's own ground position.
        let camera_xz = Vec2::new(100.0, -200.0);
        let rts = RtsCamera {
            focus: Transform::from_translation(Vec3::new(5_000.0, 3.0, 5_000.0)),
            ..Default::default()
        };
        assert_eq!(
            ring_center(Some(&rts), camera_xz),
            Vec2::new(5_000.0, 5_000.0)
        );
        assert_eq!(ring_center(None, camera_xz), camera_xz);
    }

    #[test]
    fn finest_level_grows_with_camera_distance() {
        assert_eq!(finest_level(0.0), 0);
        // Walking height keeps 1 m cells; a few hundred metres needs 4 m
        // cells; the 30 km ceiling's slant distance keeps 256 m cells.
        assert_eq!(finest_level(10.0), 0);
        assert_eq!(finest_level(400.0), 1);
        assert_eq!(finest_level(30_000.0 / 20f32.to_radians().cos()), 4);
        // Monotone across the range.
        let mut last = 0;
        for distance in [0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0] {
            let level = finest_level(distance);
            assert!(level >= last, "not monotone at {distance}");
            last = level;
        }
    }

    #[test]
    fn camera_distance_gates_the_finest_rings() {
        // With the gate at level 2: no finer tiles stream, the gate level
        // covers the centre un-cut, and exactly the tiles overlapping the
        // hole carry it in their stamp — ring corners don't, so a gate
        // flip restamps only the affected tiles.
        let tiles = desired_set_around(Vec2::ZERO, 2);
        assert!(
            tiles.iter().all(|key| key.level >= 2),
            "levels finer than the gate streamed"
        );
        assert!(
            tiles
                .iter()
                .any(|key| key.level == 2 && key.contains(Vec2::ZERO)),
            "gate level must cover the centre"
        );

        let gated = desired_tiles(Vec2::ZERO, 2);
        assert!(
            gated
                .iter()
                .filter(|(key, _)| key.level == 2)
                .all(|(_, stamp)| !stamp.cuts_hole),
            "gate level must be un-cut"
        );
        assert!(
            gated
                .iter()
                .any(|(key, stamp)| key.level == 3 && stamp.cuts_hole),
            "the level above must cut where it overlaps the gate ring"
        );
        assert!(
            gated
                .iter()
                .any(|(key, stamp)| key.level == 3 && !stamp.cuts_hole),
            "ring corners away from the hole must not cut"
        );

        // Lowering the gate one level flips only the overlapping tiles'
        // stamps; a corner tile above the gate stamps byte-identical, or
        // the flip would respawn the whole ring for nothing.
        let lower = desired_tiles(Vec2::ZERO, 1);
        let stamp_of = |set: &[(TileKey, TileStamp)], key: TileKey| {
            set.iter().find(|(k, _)| *k == key).map(|(_, s)| *s)
        };
        let corner = TileKey {
            level: 3,
            x: -2,
            z: -2,
        };
        assert_eq!(stamp_of(&lower, corner), stamp_of(&gated, corner));
        let centre = TileKey {
            level: 2,
            x: 0,
            z: 0,
        };
        assert_eq!(
            stamp_of(&gated, centre).map(|s| s.cuts_hole),
            Some(false)
        );
        assert_eq!(
            stamp_of(&lower, centre).map(|s| s.cuts_hole),
            Some(true),
            "the newly-holed tile must restamp, or its stale un-cut mesh would sheet the fine level"
        );
    }

    fn tiles_covering_point(tiles: Vec<TileKey>, point: Vec2) -> bool {
        tiles.iter().any(|&key| key.contains(point))
    }

    #[test]
    fn no_spawned_tile_is_fully_covered_by_the_finer_level() {
        // The clipmap invariant at tile granularity: anything the finer
        // square fully contains is skipped, so no coarse sheet can render
        // unseen underneath the fine surface.
        for focus in [Vec2::ZERO, Vec2::new(17.3, -45.6), Vec2::new(990.0, 12.5)] {
            for key in desired_set_around(focus, 0) {
                assert!(
                    !covered_by_finer_level(key.level, key, focus, 0),
                    "level {} tile at {:?} fully covered but desired",
                    key.level,
                    key.centre()
                );
            }
        }
    }

    #[test]
    fn coarse_tiles_cut_holes_where_the_finer_level_renders() {
        // The clipmap invariant at quad granularity: a coarse tile
        // straddling the finer square emits fewer quads — the overlap is
        // cut out, not sheeted underneath (which showed as coarse mesh
        // clipping through the fine surface over valleys).
        let field = flat_field();
        let focus = Vec2::new(64.0, 64.0); // level-0 square: [0, 160]^2
        let (hmin, hmax) = finer_render_square(1, focus);
        // A level-1 tile (128 m) straddling the square's +x edge.
        let key = TileKey {
            level: 1,
            x: 1,
            z: 1,
        };
        let with_hole = tile_mesh(&field, key, IVec2::new(1, 1), Some((hmin, hmax)));
        let without_hole = tile_mesh(&field, key, IVec2::new(1, 1), None);
        let quad_count = |mesh: &Mesh| mesh.indices().expect("indices").len() as usize;
        assert!(
            quad_count(&with_hole) < quad_count(&without_hole),
            "hole not cut: {} vs {} indices",
            quad_count(&with_hole),
            quad_count(&without_hole)
        );
        // And the cut must not be everything: the part outside the square
        // still renders.
        assert!(quad_count(&with_hole) > 0);
    }

    #[test]
    fn focus_movement_respawns_stale_stamped_tiles() {
        // Rims and holes are baked from the ring position at spawn time;
        // crossing a tile boundary must rebuild affected tiles instead of
        // leaving stale geometry mid-ring.
        let mut app = App::new();
        app.insert_resource(HeightField::default())
            .init_resource::<TerrainTiles>()
            .init_resource::<TileMeshCache>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<crate::resources::Materials>()
            .add_systems(Update, stream_terrain_tiles);
        let ground = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::from_color(Color::WHITE));
        app.world_mut()
            .resource_mut::<crate::resources::Materials>()
            .ground = ground;
        let camera = app
            .world_mut()
            .spawn((Camera3d::default(), RtsCamera::default()))
            .id();
        app.update();
        app.update();
        let before = tile_entities(&app);
        assert!(!before.is_empty());

        // Pan across a level-0 tile boundary (32 m): move both the camera
        // body and its focus — the centre follows the focus, the distance
        // gate reads the body.
        let mut entity = app.world_mut().get_entity_mut(camera).unwrap();
        let mut transform = entity.get_mut::<Transform>().unwrap();
        transform.translation.x = 40.0;
        drop(transform);
        let mut entity = app.world_mut().get_entity_mut(camera).unwrap();
        let mut rts = entity.get_mut::<RtsCamera>().unwrap();
        rts.focus.translation.x = 40.0;
        rts.target_focus = rts.focus;
        drop(rts);
        app.update();
        app.update();

        let after = tile_entities(&app);
        assert!(!after.is_empty());
        assert!(
            before.iter().any(|e| !after.contains(e)),
            "no tiles were respawned after the focus crossed a tile boundary"
        );
    }

    #[test]
    fn finest_level_streams_at_the_focus() {
        let tiles = desired_set_around(Vec2::new(123.4, -45.6), 0);
        assert!(
            tiles
                .iter()
                .any(|key| key.level == 0 && key.contains(Vec2::new(123.4, -45.6))),
            "no 1 m-resolution tile covers the focus"
        );
    }

    #[test]
    fn tile_budget_stays_bounded() {
        let tiles = desired_set_around(Vec2::ZERO, 0);
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

    #[test]
    fn rim_edges_are_stitched_to_the_parent_surface() {
        // Nonlinear in z, so the parent chord differs from the true field:
        // a stitched rim vertex must sit on the chord, an interior one on
        // the field itself (minus their respective drops).
        let field = HeightField::from_fn(|_, z| 0.001 * z * z);
        let key = TileKey {
            level: 0,
            x: 2,
            z: 0,
        }; // +x rim of the focus ring
        let mesh = tile_mesh(&field, key, IVec2::ZERO, None);
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("positions") {
            VertexAttributeValues::Float32x3(p) => p.clone(),
            _ => panic!("unexpected position format"),
        };

        let vertex_y = |ix: usize, iz: usize| match &positions[iz * TILE_VERTS + ix] {
            [_, y, _] => *y,
            _ => unreachable!(),
        };

        // Level 0 and level 1 share the same drop (1 m), so a +x rim edge
        // vertex at (x=96, z) must equal chord_lerp(h(96,z0), h(96,z0+4)) - 1.
        assert_eq!(level_drop(0), level_drop(1), "finest two drops must match");
        let parent_cell = LEVEL_CELL_M[0] * PARENT_CELL_MULT;
        for iz in 0..TILE_VERTS {
            let z = iz as f32;
            let expected = {
                let i = (z / parent_cell).floor();
                let z0 = i * parent_cell;
                let t = (z - z0) / parent_cell;
                let h0 = 0.001 * z0 * z0;
                let h1 = 0.001 * (z0 + parent_cell) * (z0 + parent_cell);
                h0 + (h1 - h0) * t - level_drop(1) - parent_cell * STITCH_BIAS_CELL_FRACTION
            };
            assert!(
                (vertex_y(TILE_QUADS, iz) - expected).abs() < 1e-4,
                "rim vertex iz={iz}: {} != {expected}",
                vertex_y(TILE_QUADS, iz)
            );
        }
        // Interior vertices stay on the true field (minus own drop), e.g.
        // the centre: h(80, 16) - 1.
        let centre_expected = 0.001 * 16.0 * 16.0 - level_drop(0);
        assert!(
            (vertex_y(16, 16) - centre_expected).abs() < 1e-4,
            "interior vertex must be un-stitched"
        );
    }

    #[test]
    fn top_level_rim_does_not_stitch_into_the_void() {
        // Regression: the top level's outer rim borders no coarser level;
        // stitching used to index LEVEL_CELL_M[9] and panic at runtime.
        // Periodic sub-scale heights, because real-amplitude fields at
        // 2 Mm tile distances exceed f32 vertex precision in the test's
        // exact-value comparison.
        let field = HeightField::from_fn(|_, z| 0.001 * (z % 1000.0));
        let key = TileKey {
            level: (LEVEL_CELL_M.len() - 1) as u8,
            x: 2,
            z: 0,
        };
        let mesh = tile_mesh(&field, key, IVec2::ZERO, None); // must not panic
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("positions") {
            VertexAttributeValues::Float32x3(p) => p.clone(),
            _ => panic!("unexpected position format"),
        };
        // Rim vertices stay on the plain field (minus the top level's drop).
        let drop = level_drop(key.level);
        let z = 5.0 * LEVEL_CELL_M[8];
        let expected = 0.001 * (z % 1000.0) - drop;
        let [_, y, _] = positions[5 * TILE_VERTS + TILE_QUADS] else {
            unreachable!()
        };
        assert!((y - expected).abs() < 1e-2, "{y} != {expected}");
    }

    #[test]
    fn normals_agree_across_tile_boundaries() {
        // Normals come from the field at world positions, so the shared
        // edge of two adjacent tiles must carry identical normals — this
        // is what kills the per-tile lighting grid.
        let field = HeightField::default();
        let a = tile_mesh(
            &field,
            TileKey {
                level: 0,
                x: 0,
                z: 0,
            },
            IVec2::ZERO,
            None,
        );
        let b = tile_mesh(
            &field,
            TileKey {
                level: 0,
                x: 1,
                z: 0,
            },
            IVec2::ZERO,
            None,
        );
        let normals = |mesh: &Mesh| match mesh.attribute(Mesh::ATTRIBUTE_NORMAL).expect("normals") {
            VertexAttributeValues::Float32x3(n) => n.clone(),
            _ => panic!("unexpected normal format"),
        };
        let (na, nb) = (normals(&a), normals(&b));
        // Shared edge x = 32: A's ix = TILE_QUADS column, B's ix = 0.
        for iz in 0..TILE_VERTS {
            assert_eq!(
                na[iz * TILE_VERTS + TILE_QUADS],
                nb[iz * TILE_VERTS],
                "normal mismatch on shared edge at iz={iz}"
            );
        }
    }

    #[test]
    fn colors_match_terrain_shape() {
        // Shading must agree with the fields it comes from: steepness
        // loses the grass green for cliff, creases brighten into chalky
        // drainage streaks.
        let palette = ground_palette();
        let sample = |slope: Vec2, ridge: f32| {
            ground_color(
                &palette,
                TerrainSample {
                    height: 10.0,
                    slope,
                    ridge_map: ridge,
                },
            )
        };
        let grass = sample(Vec2::ZERO, 1.0);
        let cliff = sample(Vec2::new(1.5, 0.0), 1.0);
        assert!(
            cliff.y < grass.y,
            "cliff must lose the grass green: {cliff} vs {grass}"
        );
        let crease = sample(Vec2::ZERO, -1.0);
        assert!(
            crease.x > grass.x && crease.y > grass.y,
            "drainage brightens creases: {crease} vs {grass}"
        );
    }

    #[test]
    fn streaming_respects_the_per_frame_budget() {
        // Zero budget: exactly one tile per frame (the progress
        // guarantee), and the full desired set eventually lands.
        let mut app = App::new();
        app.insert_resource(HeightField::default())
            .insert_resource(StreamBudget { mesh_millis: 0 })
            .init_resource::<TerrainTiles>()
            .init_resource::<TileMeshCache>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<crate::resources::Materials>()
            .add_systems(Update, stream_terrain_tiles);
        let ground = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::from_color(Color::WHITE));
        app.world_mut()
            .resource_mut::<crate::resources::Materials>()
            .ground = ground;
        app.world_mut()
            .spawn((Camera3d::default(), RtsCamera::default()));
        app.update();

        let first = tile_entities(&app).len();
        assert_eq!(first, 1, "zero budget must build exactly one tile per frame");

        let total = desired_tiles(Vec2::ZERO, 0).len();
        for _ in 0..(4 * total) {
            app.update();
            if tile_entities(&app).len() == total {
                break;
            }
        }
        assert_eq!(
            tile_entities(&app).len(),
            total,
            "budgeted streaming never finished the desired set"
        );
    }

    /// A streaming test app: the system plus the resources it needs, one
    /// RTS camera at the origin. `budget_millis: None` = no budget
    /// (unlimited builds per frame).
    fn streaming_app(budget_millis: Option<u64>) -> (App, Entity) {
        let mut app = App::new();
        app.insert_resource(HeightField::default())
            .init_resource::<TerrainTiles>()
            .init_resource::<TileMeshCache>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<crate::resources::Materials>()
            .add_systems(Update, stream_terrain_tiles);
        if let Some(millis) = budget_millis {
            app.insert_resource(StreamBudget { mesh_millis: millis });
        }
        let ground = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::from_color(Color::WHITE));
        app.world_mut()
            .resource_mut::<crate::resources::Materials>()
            .ground = ground;
        let camera = app
            .world_mut()
            .spawn((Camera3d::default(), RtsCamera::default()))
            .id();
        (app, camera)
    }

    /// Move the camera body and its focus together (a pan); both the ring
    /// centre and the distance gate read them.
    fn pan_camera(app: &mut App, camera: Entity, x: f32) {
        let mut entity = app.world_mut().get_entity_mut(camera).unwrap();
        entity.get_mut::<Transform>().unwrap().translation.x = x;
        let mut rts = entity.get_mut::<RtsCamera>().unwrap();
        rts.focus.translation.x = x;
        rts.target_focus = rts.focus;
    }

    fn converge(app: &mut App, total: usize) {
        for _ in 0..(4 * total + 8) {
            app.update();
        }
    }

    #[test]
    fn stale_tiles_render_until_their_replacement_spawns() {
        // The no-void invariant: crossing a tile boundary restamps the
        // rings, but the old meshes stay in the world until each
        // replacement lands (zero budget = one build per frame, so the
        // overlap window is long) — and the streamed set keeps covering
        // the focus throughout.
        let (mut app, camera) = streaming_app(Some(0));
        let total = desired_tiles(Vec2::ZERO, 0).len();
        converge(&mut app, total);
        assert_eq!(tile_entities(&app).len(), total);
        let before = tile_entities(&app);

        pan_camera(&mut app, camera, 40.0);
        app.update(); // exactly one replacement built this frame

        let alive = |app: &App, entity: Entity| app.world().get_entity(entity).is_ok();
        let survivors = before.iter().filter(|e| alive(&app, **e)).count();
        assert!(
            survivors > before.len() / 2,
            "old meshes vanished before their replacements: {survivors}/{}",
            before.len()
        );
        assert!(
            app.world()
                .resource::<TerrainTiles>()
                .covers(Vec2::new(40.0, 0.0)),
            "the new focus is not covered mid-transition"
        );

        converge(&mut app, total);
        assert_eq!(tile_entities(&app).len(), total);
    }

    #[test]
    fn dropped_level_tiles_wait_until_covered() {
        // Zooming out lifts the finest levels (the distance gate): their
        // tiles must not vanish while the coarser ring still cuts the
        // hole that exposed them — the cover has to land first.
        let (mut app, _camera) = streaming_app(Some(0));
        let total = desired_tiles(Vec2::ZERO, 0).len();
        converge(&mut app, total);

        // Lift the camera 40 km (focus stays): distance gates the finest
        // level up to L4, dropping the L0..L3 rings.
        app.world_mut()
            .get_entity_mut(_camera)
            .unwrap()
            .get_mut::<Transform>()
            .unwrap()
            .translation = Vec3::new(0.0, 40_000.0, 0.0);
        app.update(); // one build: the un-holed covers have not landed

        let l0 = app
            .world()
            .resource::<TerrainTiles>()
            .tiles
            .keys()
            .filter(|key| key.level == 0)
            .count();
        assert!(l0 > 0, "dropped L0 tiles vanished before their cover landed");

        converge(&mut app, total);
        let l0 = app
            .world()
            .resource::<TerrainTiles>()
            .tiles
            .keys()
            .filter(|key| key.level == 0)
            .count();
        assert_eq!(l0, 0, "dropped L0 tiles never retired");
    }

    #[test]
    fn recycled_tiles_reuse_cached_meshes() {
        // Panning away and back re-spawns from the mesh cache: the return
        // trip creates no new Assets<Mesh> entries.
        let (mut app, camera) = streaming_app(None);
        let total = desired_tiles(Vec2::ZERO, 0).len();
        converge(&mut app, total);

        let asset_count =
            |app: &App| app.world().resource::<Assets<Mesh>>().iter().count();
        pan_camera(&mut app, camera, 200.0);
        converge(&mut app, total);
        let away = asset_count(&app);

        pan_camera(&mut app, camera, 0.0);
        converge(&mut app, total);
        assert_eq!(
            asset_count(&app),
            away,
            "returning over visited ground built new meshes instead of cache hits"
        );
        assert_eq!(tile_entities(&app).len(), total);
    }

    #[test]
    fn tile_mesh_builds_in_about_a_millisecond() {
        // Pins the SERIAL cost of one tile build — the field sampling is
        // ~2 ms (release ~2 ms, dev/cranelift a touch more) and the spawn
        // pass meshes whole waves of them in parallel (see `mesh_wave`).
        // A noise/erosion regression that multiplies this starves
        // streaming even across cores. Wall-clock like
        // `nearest_solver_scales_to_10k_members`; run in release for the
        // real number.
        let field = HeightField::default();
        let set = desired_tiles(Vec2::new(123.4, -45.6), 0);
        let start = Instant::now();
        for (key, stamp) in &set {
            let _ = tile_mesh(&field, *key, stamp.focus_tile, None);
        }
        let per_tile = start.elapsed() / set.len() as u32;
        eprintln!("tile_mesh: {per_tile:?}/tile over {} tiles", set.len());
        assert!(
            per_tile < Duration::from_millis(4),
            "tile mesh avg {per_tile:?} — parallel waves assume ~2 ms serial builds"
        );
    }
}
