//! The terrain: streamed LOD rings baked into small render textures.
//!
//! Each LOD level is a 5x5 ring of tiles around the RTS camera's focus
//! (the camera's ground position in freecam), with cell size growing ×4
//! per level (1 m near the camera to 65 km at continental distance); the
//! finest rendered level is gated by the camera's distance (see
//! [`LodMode`]). Coverage is continuous by construction: a coarser tile
//! is skipped only when the finer level's square fully contains it, so
//! there are no annular gaps between levels; where rings overlap, the
//! coarser level sits slightly lower (per-level height bias) so the
//! finer surface wins the depth test. Heights come from
//! [`HeightField`], so the baked render, grounding, and camera clearance
//! agree by construction; the same field samples tint the tiles
//! (rock/grass/dirt by slope and altitude, drainage streaks from the
//! erosion ridge map).
//!
//! Rendering is `render`'s job: every tile draws the ONE shared displaced
//! mesh, placed by its baked textures — the UDLOD approach.

use super::render::{
    SharedTileMesh, TerrainMaterial, TerrainExtension, TileAtlas, TileUniform,
    bake_texture_f32, bake_texture_rg32f, bake_texture_rgba8,
};
use super::{HeightField, TerrainSample, VERTICAL_BIAS};
use bevy::prelude::*;
use bevy_rts_camera::{Ground, RtsCamera};
use bevy::platform::time::Instant; // web-time on wasm: std's Instant::now() panics ("time not implemented on this platform")
use std::collections::HashMap;
use std::time::Duration;

/// Tile resolution: 32x32 quads per tile, any level.
pub(crate) const TILE_QUADS: usize = 32;
pub(crate) const TILE_VERTS: usize = TILE_QUADS + 1;
/// Cell size (metres per quad edge) per LOD level, ×4 per level. Level 0
/// gives 1 m detail in a 32 m tile; the top level's 65 km cells put the
/// 5x5 ring at ~10.5 Mm span — a small continent.
const LEVEL_CELL_M: [f32; 9] = [
    1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0,
];
/// Half-extent of a level's rendered square, in multiples of that level's
/// own tile size: the 5x5 ring around the focus tile.
const LEVEL_RING: i32 = 2;
/// Vertical separation between adjacent levels: a fixed sub-metre nudge
/// so transiently-overlapping rings (the swap window while a coarse
/// cover sheets over the fine ring it retires) resolve by depth test.
/// Historically this scaled with cell size (0.3 cells per level,
/// CUMULATIVE — 26 km at the top level), which rim stitching compressed
/// into one-cell ramps: concentric square cliffs around every ring,
/// receding to the horizon. Rings no longer overlap at steady state
/// (holes are cut exactly and rims stitch to the parent chord), so the
/// scaled drop bought nothing visible and cost the terraces.
const LEVEL_DROP: f32 = 0.5;
/// The stitched fine rim sits this far BELOW the parent plane: the rim
/// is otherwise exactly coplanar with the coarse fringe quads, which
/// z-fights where they overlap. Flat for the same reason as
/// [`LEVEL_DROP` — cell-scaled bias was kilometres wide at the top.
const STITCH_BIAS: f32 = 0.1;

/// Coarser levels sit a hair below the field so transient ring overlaps
/// resolve by depth test instead of z-fighting. Flat across levels.
fn level_drop(_level: u8) -> f32 {
    LEVEL_DROP
}

/// Parent (= next coarser) level's geometry, as seen from `level`.
const PARENT_CELL_MULT: f32 = 4.0;

/// Tile identity: LOD level + grid position at that level's tile size.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TileKey {
    pub(crate) level: u8,
    pub(crate) x: i32,
    pub(crate) z: i32,
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
/// movement). The value pairs the entity with the [`TileStamp`] the
/// bake was made with and its [`TileRender`] (the per-tile material and
/// its textures): rims and holes are computed from the ring position at
/// spawn time, so a stamp mismatch (focus crossed a tile boundary)
/// queues a replacement — the old surface keeps rendering until the
/// replacement spawns, so streaming never opens a hole (see
/// `stream_terrain_tiles`).
#[derive(Resource, Default)]
pub struct TerrainTiles {
    tiles: HashMap<TileKey, (Entity, TileStamp, TileRender)>,
}

/// Ring position a tile's bake was made with: the focus tile at the
/// tile's own level (which edges are stitching rims), at the finer level
/// (where the cut-out hole lies), and whether this tile's bake actually
/// cuts that hole (the finer level renders AND this tile's footprint
/// overlaps its square — the distance gate, [`LodMode`], lifts whole fine
/// levels in and out). Only a flip of what the bake depends on forces a
/// respawn; tiles whose geometry is unchanged by a ring move keep their
/// stamp and are not rebuilt.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TileStamp {
    focus_tile: IVec2,
    fine_focus_tile: IVec2,
    cuts_hole: bool,
}

/// Live or cached render state for one tile: the per-tile
/// [`TerrainMaterial`] plus the tile's layer in the shared [`TileAtlas`]
/// and the baked height range. The material's uniform carries the layer
/// index, so render state and atlas stay in lockstep.
pub(crate) struct TileRender {
    material: Handle<TerrainMaterial>,
    slot: u32,
    min_y: f32,
    max_y: f32,
}

/// Free what a retired tile render holds (called on cache eviction and
/// field rebuilds, when nothing references it anymore): the material
/// goes, the atlas layer returns to the free list.
impl TileRender {
    fn free(self, materials: &mut Assets<TerrainMaterial>, atlas: &mut TileAtlas) {
        materials.remove(&self.material);
        atlas.release(self.slot);
    }
}

/// Retired tile renders, kept for instant re-spawn. Zooming and panning
/// back over ground you just left re-uses the baked material instead of
/// re-running the erosion-filtered field ~1k times — a cache hit costs
/// no bake time and no [`StreamBudget`]. Bounded: the
/// least-recently-used renders beyond [`TILE_CACHE_TILES`] are freed
/// (live tiles are never cached, so nothing referenced is freed).
#[derive(Resource, Default)]
pub struct TileRenderCache {
    entries: HashMap<(TileKey, TileStamp), (TileRender, u64)>,
    clock: u64,
}

/// Cache capacity in tiles. ~21 KB of bake textures per tile → ~5 MB; a
/// full desired set is ~120-150 tiles, so a zoom gate cycle or a pan
/// detour fits whole.
const TILE_CACHE_TILES: usize = 256;

impl TileRenderCache {
    /// Take a render out of the cache for re-spawn, if present.
    fn take(&mut self, key: TileKey, stamp: TileStamp) -> Option<TileRender> {
        self.clock += 1;
        self.entries.remove(&(key, stamp)).map(|(render, _)| render)
    }

    /// Free the least-recently-used cached render, if any — the atlas's
    /// pressure valve when a wave of bakes would claim more layers than
    /// are free. Cache misses cost a re-bake, never a hole.
    fn evict_oldest(
        &mut self,
        materials: &mut Assets<TerrainMaterial>,
        atlas: &mut TileAtlas,
    ) -> bool {
        let Some((&oldest, _)) = self.entries.iter().min_by_key(|(_, (_, used))| *used) else {
            return false;
        };
        if let Some((render, _)) = self.entries.remove(&oldest) {
            render.free(materials, atlas);
        }
        true
    }

    /// Park a retired tile's render in the cache; free the LRU overflow.
    fn put(
        &mut self,
        key: TileKey,
        stamp: TileStamp,
        render: TileRender,
        materials: &mut Assets<TerrainMaterial>,
        atlas: &mut TileAtlas,
    ) {
        self.clock += 1;
        let clock = self.clock;
        self.entries.insert((key, stamp), (render, clock));
        while self.entries.len() > TILE_CACHE_TILES {
            let Some((&oldest, _)) = self.entries.iter().min_by_key(|(_, (_, used))| *used)
            else {
                break;
            };
            if let Some((render, _)) = self.entries.remove(&oldest) {
                render.free(materials, atlas);
            }
        }
    }

    /// Drop every cached render (the height field changed; the bakes are
    /// from the old field).
    fn drain(&mut self, materials: &mut Assets<TerrainMaterial>, atlas: &mut TileAtlas) {
        for (_, (render, _)) in self.entries.drain() {
            render.free(materials, atlas);
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

/// Number of LOD levels (entries in `LEVEL_CELL_M`).
pub fn level_count() -> usize {
    LEVEL_CELL_M.len()
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

    /// Audit atlas-layer usage across live and cached renders: counts,
    /// layers claimed by more than one render (aliasing — renders would
    /// read each other's bakes), and out-of-range layer indices.
    pub fn slot_audit(&self, cache: &TileRenderCache) -> (usize, usize, usize, usize) {
        let mut use_count: HashMap<u32, usize> = HashMap::new();
        let mut oor = 0;
        let mut live = 0;
        for (_, _, render) in self.tiles.values() {
            live += 1;
            if render.slot >= super::render::ATLAS_LAYERS {
                oor += 1;
            }
            *use_count.entry(render.slot).or_default() += 1;
        }
        let mut cached = 0;
        for (render, _) in cache.entries.values() {
            cached += 1;
            if render.slot >= super::render::ATLAS_LAYERS {
                oor += 1;
            }
            *use_count.entry(render.slot).or_default() += 1;
        }
        let aliased = use_count.values().filter(|&&n| n > 1).count();
        (live, cached, aliased, oor)
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

/// How far the morph band reaches in from a rim edge, in quads: the fine
/// surface blends onto the parent surface across this band, making the
/// LOD boundary a continuous ramp instead of a cliff — without it the
/// parent's boundary strip (its own sampled heights, which overshoot
/// massif peaks at 4x cells) can tower over the fine ring's valleys and
/// occlude them. CDLOD's transition zone, baked statically.
const MORPH_BAND_QUADS: f32 = 8.0;

/// One tile's baked render data: the small textures the shared displaced
/// mesh renders from (see `crate::terrain::render`). `height` is the
/// FINAL vertex surface — level drop and rim stitching included, i.e.
/// exactly what `tile_mesh` used to push as vertex Y — `slope` and
/// `color` are the 3×3-blurred shading fields.
pub(crate) struct BakedTile {
    pub height: Image,
    pub slope: Image,
    pub color: Image,
    pub min_y: f32,
    pub max_y: f32,
}

/// The hole rectangle for a tile, in tile-local TEXEL units for the
/// vertex shader's clamp projection (see `terrain_common.wgsl`): world
/// square → texels, or huge bounds when this tile cuts no hole (clamp
/// becomes the identity).
fn hole_texels(key: TileKey, hole: Option<(Vec2, Vec2)>) -> Vec4 {
    const EMPTY: f32 = 1.0e9;
    match hole {
        None => Vec4::new(-EMPTY, EMPTY, -EMPTY, EMPTY),
        Some((hmin, hmax)) => {
            let cell = key.cell();
            let origin = Vec2::new(
                key.x as f32 * key.tile_size(),
                key.z as f32 * key.tile_size(),
            );
            Vec4::new(
                (hmin.x - origin.x) / cell,
                (hmax.x - origin.x) / cell,
                (hmin.y - origin.y) / cell,
                (hmax.y - origin.y) / cell,
            )
        }
    }
}

/// Bake one tile: sample the field on the 33×33 grid (level-truncated),
/// stitch rim vertices onto the parent surface, blur the shading fields,
/// and emit the render textures.
///
/// `focus_tile` is the focus tile's grid position at this key's level: the
/// four outer rim edges of the 5x5 ring border a COARSER level, and their
/// border vertices are stitched onto the parent level's linear
/// interpolation (minus the parent's drop) — the CPU equivalent of
/// terrain_renderer's vertex morphing, so the fine surface meets the
/// coarse one along the exact shared boundary with no crack or step.
///
/// Hole-cutting is shader-side: the coarse tile must not sheet a second
/// surface underneath the finer one (the clipmap invariant), which the
/// vertex shader enforces by projecting the hole rectangle away — the
/// rectangle rides the tile uniform (`hole_texels`), not the bake.
pub(crate) fn bake_tile(
    field: &HeightField,
    key: TileKey,
    focus_tile: IVec2,
) -> BakedTile {
    let cell = key.cell();
    let tile_size = key.tile_size();
    let origin_x = key.x as f32 * tile_size;
    let origin_z = key.z as f32 * tile_size;
    let drop = level_drop(key.level);
    // Rim edges: the neighbour in that direction lies outside the 5x5
    // ring, so the adjacent level there is coarser. The TOP level's outer
    // rim borders the void, not a coarser level — nothing to stitch to.
    let can_stitch = (key.level as usize + 1) < LEVEL_CELL_M.len();
    let rim_x_plus = can_stitch && key.x == focus_tile.x + LEVEL_RING;
    let rim_x_minus = can_stitch && key.x == focus_tile.x - LEVEL_RING;
    let rim_z_plus = can_stitch && key.z == focus_tile.y + LEVEL_RING;
    let rim_z_minus = can_stitch && key.z == focus_tile.y - LEVEL_RING;

    // Interior grid: one field sample per vertex feeds the height, and
    // (blurred) the shading. `heights` is what renders (drop applied,
    // rims re-stitched); `samples` keeps the unstitched world-space
    // fields for shading, since altitude gates must not see LOD drop
    // offsets.
    let mut heights = [[0.0f32; TILE_VERTS]; TILE_VERTS];
    let mut samples = [[TerrainSample::default(); TILE_VERTS]; TILE_VERTS];
    for iz in 0..TILE_VERTS {
        for ix in 0..TILE_VERTS {
            let sample = field.sample_lod(
                origin_x + ix as f32 * cell,
                origin_z + iz as f32 * cell,
                key.level,
            );
            heights[iz][ix] = sample.height - drop;
            samples[iz][ix] = sample;
        }
    }

    // Morph band: rim-facing vertices blend onto the parent surface
    // across MORPH_BAND_QUADS, the rim edge itself landing exactly on it
    // (the old edge-only stitch). The blend is smoothstep, so the
    // surface leaves the parent plane tangentially — no step, no
    // occluding cliff, and the band's screen error decays like the LOD
    // error itself. The parent surface is a 9x9 sample lattice (the
    // tile spans exactly 8 parent cells) bilinearly upsampled — the
    // same surface `parent_lerp_height` interpolates, at 81 field
    // samples for the whole band instead of four per vertex.
    if rim_x_plus || rim_x_minus || rim_z_plus || rim_z_minus {
        let parent_cell = cell * PARENT_CELL_MULT;
        let parent_drop = level_drop(key.level + 1) + STITCH_BIAS;
        let mut lattice = [[0.0f32; 9]; 9];
        for lz in 0..9usize {
            for lx in 0..9usize {
                lattice[lz][lx] = field.height(
                    origin_x + lx as f32 * parent_cell,
                    origin_z + lz as f32 * parent_cell,
                ) - parent_drop;
            }
        }
        for iz in 0..TILE_VERTS {
            for ix in 0..TILE_VERTS {
                let u = ix as f32 / TILE_QUADS as f32;
                let v = iz as f32 / TILE_QUADS as f32;
                let mut d = f32::INFINITY;
                if rim_x_plus {
                    d = d.min(1.0 - u);
                }
                if rim_x_minus {
                    d = d.min(u);
                }
                if rim_z_plus {
                    d = d.min(1.0 - v);
                }
                if rim_z_minus {
                    d = d.min(v);
                }
                if d.is_infinite() {
                    continue; // interior tile of the ring: no band
                }
                let fx = ix as f32 / PARENT_CELL_MULT;
                let fz = iz as f32 / PARENT_CELL_MULT;
                let (px, pz) = ((fx.floor() as usize).min(7), (fz.floor() as usize).min(7));
                let (tx, tz) = (fx - px as f32, fz - pz as f32);
                let a = lattice[pz][px] + (lattice[pz][px + 1] - lattice[pz][px]) * tx;
                let b =
                    lattice[pz + 1][px] + (lattice[pz + 1][px + 1] - lattice[pz + 1][px]) * tx;
                let parent = a + (b - a) * tz;
                let fine = heights[iz][ix];
                let band = d * TILE_QUADS as f32 / MORPH_BAND_QUADS;
                let morph = smoothstep(0.0, 1.0, band.min(1.0));
                heights[iz][ix] = parent + (fine - parent) * morph;
            }
        }
    }

    // Shading: SLOPE and COLOR from a 3x3-blurred neighborhood: on 50°
    // carved terrain the exact per-vertex slope flips shading and color
    // thresholds between adjacent facets every cell — a shattered-glass
    // speckle (the reference shades per-fragment from interpolated
    // values). The blur is a free low-pass — all nine samples are
    // already in the grid — and geometry keeps the exact height, so only
    // shading softens. Field sampling is tile-independent, so
    // neighbouring tiles agree exactly on shared vertices.
    let palette = ground_palette();
    let relief_m = field.relief_m();
    let mut slopes = [[Vec2::ZERO; TILE_VERTS]; TILE_VERTS];
    let mut colors = [[Vec4::ONE; TILE_VERTS]; TILE_VERTS];
    for iz in 0..TILE_VERTS {
        for ix in 0..TILE_VERTS {
            let (mut slope, mut h, mut rg) = (Vec2::ZERO, 0.0, 0.0);
            for dz in -1..=1i32 {
                for dx in -1..=1i32 {
                    let (jx, jz) = (ix as i32 + dx, iz as i32 + dz);
                    let s = if jx >= 0
                        && jx <= TILE_QUADS as i32
                        && jz >= 0
                        && jz <= TILE_QUADS as i32
                    {
                        samples[jz as usize][jx as usize]
                    } else {
                        // Border vertices blur into the neighbour tile's
                        // grid positions: sampling the field at the same
                        // world points keeps the blur — like the exact
                        // slope — tile-independent, so shared vertices
                        // shade identically from both sides.
                        field.sample_lod(
                            origin_x + jx as f32 * cell,
                            origin_z + jz as f32 * cell,
                            key.level,
                        )
                    };
                    slope += s.slope;
                    h += s.height;
                    rg += s.ridge_map;
                }
            }
            let slope = slope / 9.0;
            slopes[iz][ix] = slope;
            colors[iz][ix] = ground_color(
                &palette,
                slope.length(),
                h / 9.0 / relief_m.max(1.0) + VERTICAL_BIAS,
                rg / 9.0,
                1.0 + 0.7 * key.level as f32,
            );
        }
    }

    // Ring-edge color morph: each LOD ring jumps 4x in cell size, and
    // even with level-widened color bands the two rings' vertices read
    // differently — the ring boundary showed up as a hard color step.
    // Rim-facing verts fade toward the tile average over a SEAM-scale
    // band (a few texels): enough to turn the step into a gradient,
    // small enough not to read as a band of wrong-resolution terrain
    // (the original 35%-of-tile band did exactly that at coarse
    // levels). Strength grows mildly with level.
    let can_morph = can_stitch;
    if can_morph && (rim_x_plus || rim_x_minus || rim_z_plus || rim_z_minus) {
        let mut avg = Vec4::ZERO;
        for row in &colors {
            for c in row {
                avg += *c;
            }
        }
        avg /= (TILE_VERTS * TILE_VERTS) as f32;
        let strength = (0.1 * key.level as f32).min(0.5);
        for iz in 0..TILE_VERTS {
            for ix in 0..TILE_VERTS {
                let u = ix as f32 / TILE_QUADS as f32;
                let v = iz as f32 / TILE_QUADS as f32;
                let mut d = 1.0f32;
                if rim_x_plus {
                    d = d.min(u);
                }
                if rim_x_minus {
                    d = d.min(1.0 - u);
                }
                if rim_z_plus {
                    d = d.min(v);
                }
                if rim_z_minus {
                    d = d.min(1.0 - v);
                }
                if d < 1.0 {
                    let t = smoothstep(0.0, 0.12, d);
                    colors[iz][ix] = colors[iz][ix].lerp(avg, (1.0 - t) * strength);
                }
            }
        }
    }

    // Pack into textures.
    let flat = |g: &[[f32; TILE_VERTS]; TILE_VERTS]| -> Vec<f32> {
        let mut v = Vec::with_capacity(TILE_VERTS * TILE_VERTS);
        for row in g {
            v.extend_from_slice(row);
        }
        v
    };
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for row in &heights {
        for &h in row {
            min_y = min_y.min(h);
            max_y = max_y.max(h);
        }
    }
    BakedTile {
        height: bake_texture_f32(&flat(&heights)),
        slope: bake_texture_rg32f(&{
            let mut v = Vec::with_capacity(TILE_VERTS * TILE_VERTS);
            for row in &slopes {
                for s in row {
                    v.push([s.x, s.y]);
                }
            }
            v
        }),
        color: bake_texture_rgba8(&{
            let mut v = Vec::with_capacity(TILE_VERTS * TILE_VERTS);
            for row in &colors {
                for c in row {
                    v.push(c.to_array());
                }
            }
            v
        }),
        min_y,
        max_y,
    }
}

/// The terrain palette as linear RGBA — sRGB values from the erosion
/// reference demo's texturing: [grass low, grass high, dirt, cliff,
/// drainage chalk, snow]. Converted once per mesh, not per vertex.
fn ground_palette() -> [Vec4; 6] {
    [
        Color::srgb(0.15, 0.30, 0.10),
        Color::srgb(0.40, 0.50, 0.20),
        Color::srgb(0.60, 0.50, 0.40),
        Color::srgb(0.30, 0.28, 0.27),
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
/// ridge map marks gully creases, snow on the highest ridges. Altitude
/// gates run on unit height (height / relief + 0.43) like the demo's, so
/// the bands follow the mountains when relief changes scale.
fn ground_color(
    palette: &[Vec4; 6],
    steepness: f32,
    unit_height: f32,
    ridgemap_raw: f32,
    widen: f32,
) -> Vec4 {
    // Every band's transition widens with tile level: coarse vertices
    // sample the field 16-64 m apart, and narrow thresholds then
    // decorrelate into confetti. Widened bands shade coarse rings as
    // coherent averages of what the fine ring resolves.
    let grass = Vec4::lerp(
        palette[0],
        palette[1],
        smoothstep_wide(0.40, 0.55, unit_height, widen),
    );
    // Slope gates stay sharp: coarse vertices UNDERSAMPLE slopes (they
    // miss narrow gully flanks), so widening would fabricate dirt on
    // terrain the fine ring renders green. Height gates below widen —
    // heights survive coarsening.
    let mut c = Vec4::lerp(grass, palette[2], smoothstep(0.5, 1.2, steepness));
    c = Vec4::lerp(c, palette[3], smoothstep(1.4, 2.4, steepness));
    // ridge_map ≈ -1 at crease centres; only the narrow core band paints
    // the stream beds, or the chalk speckles over every slope (tuned on
    // screen: 0.3 fired as isolated dots). Chalk needs steep gully
    // flanks to read as drainage: without the steepness gate it fires
    // on the shallow creases of flat plains as pale dandruff.
    let ridgemap = (ridgemap_raw * 0.5 + 0.5).clamp(0.0, 1.0);
    // Chalk is as fine as the gullies it traces: coarse rings sample
    // `ridge_map` decorrelated, so the band would fire as pale dither.
    // Fade it out with level, like the gully geometry itself.
    let drainage = (1.0 - ridgemap / 0.15).clamp(0.0, 1.0)
        * smoothstep(0.2, 0.5, steepness)
        / widen.sqrt();
    c = Vec4::lerp(c, palette[4], drainage);
    Vec4::lerp(
        c,
        palette[5],
        smoothstep_wide(0.80, 0.95, unit_height, widen),
    )
}

/// `smoothstep` with its transition band widened by `widen` around the
/// same centre.
fn smoothstep_wide(a: f32, b: f32, x: f32, widen: f32) -> f32 {
    let centre = (a + b) * 0.5;
    let half = (b - a) * 0.5 * widen;
    smoothstep(centre - half, centre + half, x)
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
/// (baked synchronously — ~1k field samples per tile, at most a
/// [`StreamBudget`] worth of *bakes* per frame; [`TileRenderCache`] hits
/// are free), retire the rest. The rings centre on the RTS focus (the
/// camera's ground position in freecam) and the finest level is gated by
/// the camera's distance to that centre — see [`LodMode`].
///
/// Streaming never opens a hole in the ground. Stale tiles (their ring
/// position moved, or the distance gate flipped their hole) keep
/// rendering until their replacement spawns in the same frame — an atomic
/// swap, not a despawn-then-build. Tiles dropped from the desired set
/// (gate lifted their whole level, focus moved on) keep rendering until
/// a current tile actually renders over their ground: the coarse ring's
/// cut hole would otherwise expose the void beneath. A stale rim or a
/// few frames of coarse-over-fine overlap beat a hole every time.
#[allow(clippy::too_many_arguments)]
pub fn stream_terrain_tiles(
    mut commands: Commands,
    mut tiles: ResMut<TerrainTiles>,
    mut cache: ResMut<TileRenderCache>,
    cameras: Query<(&Transform, Option<&RtsCamera>), With<Camera3d>>,
    lod_mode: Option<Res<LodMode>>,
    field: Res<HeightField>,
    shared_mesh: Res<SharedTileMesh>,
    mut atlas: ResMut<TileAtlas>,
    mut images: ResMut<Assets<Image>>,
    mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
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
    // stale, cached renders included. Drop it all — the spawn pass below
    // then rebuilds the desired set from the new field, spreading the
    // bakes across frames under the budget so a slider drag never
    // freezes the frame.
    if field.is_changed() {
        for (_, (entity, _, render)) in tiles.tiles.drain() {
            commands.entity(entity).despawn();
            render.free(&mut terrain_materials, &mut atlas);
        }
        cache.drain(&mut terrain_materials, &mut atlas);
    }

    let desired = desired_tiles(center, min_level);

    // Half-extent of everything the desired set can ever cover (the top
    // ring): beyond it, a dropped tile will never be covered — retire it
    // immediately instead of leaking it.
    let top_tile = LEVEL_CELL_M[LEVEL_CELL_M.len() - 1] * TILE_QUADS as f32;
    let continental_reach = (LEVEL_RING as f32 + 0.5) * top_tile;

    // Retire pass. Current and stale-stamped tiles stay (the spawn pass
    // swaps the stale ones atomically); dropped keys — the gate lifted
    // their whole level, or the focus moved on — start fading out and
    // stay until they have faded AND current tiles render over their
    // centre, then retire into the cache.
    let mut retired: Vec<TileKey> = Vec::new();
    for (&key, &(_, _, _)) in tiles.tiles.iter() {
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
        if let Some((entity, stamp, render)) = tiles.tiles.remove(&key) {
            cache.put(key, stamp, render, &mut terrain_materials, &mut atlas);
            commands.entity(entity).despawn();
        }
    }

    // Immediate pass: apply every cached render swap NOW, unbudgeted.
    // A zoom gate flip restamps a whole coarse level at once (holes in
    // or out), and those renders are almost always cached from the last
    // pass through this zoom — processing them in the same frame as the
    // fine rings keeps every live tile's hole in step with what renders
    // beneath it. Without this, the budgeted loop reaches the coarse
    // restamp frames after the fine ring spawned, and the stale HOLELESS
    // sheet covers the fresh detail for everyone to see. A hole-adding
    // swap still waits until a finer live tile renders over the hole
    // (the void guard), so nothing opens a gap.
    for &(key, stamp) in &desired {
        if let Some((_, live, _)) = tiles.tiles.get(&key) {
            if *live == stamp {
                continue; // current
            }
        }
        if stamp.cuts_hole && !hole_covered(&tiles.tiles, key, &stamp) {
            continue; // deferred: fine coverage not live yet
        }
        let Some(render) = cache.take(key, stamp) else {
            continue; // a bake; the budgeted pass below owns it
        };
        let entity = commands
            .spawn((
                Mesh3d(shared_mesh.0.clone()),
                MeshMaterial3d(render.material.clone()),
                bevy::camera::visibility::NoFrustumCulling,
                Transform::IDENTITY,
                // The drag-pan grab raycast is off (lock_on_drag =
                // false); the marker just tags terrain meshes.
                Ground,
            ))
            .id();
        if let Some((old_entity, old_stamp, old_render)) =
            tiles.tiles.insert(key, (entity, stamp, render))
        {
            cache.put(key, old_stamp, old_render, &mut terrain_materials, &mut atlas);
            commands.entity(old_entity).despawn();
        }
    }

    // Spawn pass. Misses are baked in parallel waves — one tile costs
    // ~2 ms of field sampling serially (see the `bake_tile` perf test),
    // so a 2 ms budget buys several tiles only across cores. Absent
    // budget means unlimited (tests, loading screens).
    let deadline = budget
        .as_deref()
        .map(|b| Instant::now() + Duration::from_millis(b.mesh_millis));
    let mut built = 0usize;
    let mut index = 0usize;
    while index < desired.len() {
        if built > 0 && deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        // Assemble the next wave of bakes (a full parallel wave, or a
        // single guaranteed-progress tile when the budget is spent).
        let spent = deadline.is_some_and(|d| Instant::now() >= d);
        let wave_cap = if spent { 1 } else { parallelism().max(1) };
        let mut wave: Vec<(TileKey, TileStamp)> = Vec::new();
        while index < desired.len() && wave.len() < wave_cap {
            let (key, stamp) = desired[index];
            index += 1;
            if let Some((_, live, _)) = tiles.tiles.get(&key) {
                if *live == stamp {
                    continue; // current (or an immediate-pass deferred hit)
                }
            }
            wave.push((key, stamp));
        }
        if wave.is_empty() {
            continue;
        }
        // Atlas pressure valve: the wave will claim fresh layers; evict
        // cached renders until they fit (a miss re-bakes, a shortfall
        // would assert).
        while atlas.available() < wave.len() as u32
            && cache.evict_oldest(&mut terrain_materials, &mut atlas)
        {}
        let fresh = bake_wave(&field, &wave);
        built += fresh.len();
        for (i, bake) in fresh {
            let (key, stamp) = wave[i];
            let render = upload_tile(
                key,
                stamp,
                bake,
                &mut atlas,
                &mut images,
                &mut terrain_materials,
            );
            let entity = commands
                .spawn((
                    Mesh3d(shared_mesh.0.clone()),
                    MeshMaterial3d(render.material.clone()),
                    bevy::camera::visibility::NoFrustumCulling,
                    Transform::IDENTITY,
                    // The drag-pan grab raycast is off (lock_on_drag =
                    // false); the marker just tags terrain meshes.
                    Ground,
                ))
                .id();
            if let Some((old_entity, old_stamp, old_render)) =
                tiles.tiles.insert(key, (entity, stamp, render))
            {
                cache.put(key, old_stamp, old_render, &mut terrain_materials, &mut atlas);
                commands.entity(old_entity).despawn();
            }
        }
    }
}

/// Does a finer live tile render over every part of `key`'s would-be
/// hole? The immediate pass refuses to open a hole the fine level has
/// not filled in yet (the void guard, same predicate the retire pass
/// and `hole_cutters_never_expose_their_hole` use).
fn hole_covered(
    tiles: &HashMap<TileKey, (Entity, TileStamp, TileRender)>,
    key: TileKey,
    stamp: &TileStamp,
) -> bool {
    let (hmin, hmax) = hole_square(key.level, stamp.fine_focus_tile);
    [hmin + Vec2::splat(1.0),
        Vec2::new(hmax.x - 1.0, hmin.y + 1.0),
        Vec2::new(hmin.x + 1.0, hmax.y - 1.0),
        hmax - Vec2::splat(1.0),
        (hmin + hmax) / 2.0]
    .iter()
    .all(|&p| {
        tiles.iter().any(|(&k, &(_, st, _))| {
            k.level < key.level && k.contains(p) && renders_at(k.level, &st, p)
        })
    })
}

/// Turn a fresh [`BakedTile`] into live render state: claim an atlas
/// layer, patch the bakes into it and build the per-tile material.
fn upload_tile(
    key: TileKey,
    stamp: TileStamp,
    bake: BakedTile,
    atlas: &mut TileAtlas,
    images: &mut Assets<Image>,
    materials: &mut Assets<TerrainMaterial>,
) -> TileRender {
    let slot = atlas.alloc();
    atlas.write(images, slot, &bake);
    let hole = if stamp.cuts_hole {
        Some(hole_square(key.level, stamp.fine_focus_tile))
    } else {
        None
    };
    let material = materials.add(TerrainMaterial {
        base: StandardMaterial {
            base_color: Color::WHITE,
            ..Default::default()
        },
        extension: TerrainExtension {
            height: atlas.height.clone(),
            slope: atlas.slope.clone(),
            color: atlas.color.clone(),
            tile: TileUniform {
                origin_size: Vec4::new(
                    key.x as f32 * key.tile_size(),
                    key.z as f32 * key.tile_size(),
                    key.tile_size(),
                    key.cell(),
                ),
                hole: hole_texels(key, hole),
                // (level, unused, unused, atlas layer).
                shading: Vec4::new(key.level as f32, 0.0, 0.0, slot as f32),
            },
        },
    });
    TileRender {
        material,
        slot,
        min_y: bake.min_y,
        max_y: bake.max_y,
    }
}

/// Usable cores for the baking waves (wasm reports 1; the serial
/// fallback there needs no threads).
fn parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Bake this wave's cache misses across cores, returning `(wave index,
/// bake)` pairs. Every tile costs about the same (~equal sample counts),
/// so even slices need no work stealing. Single-core falls back to
/// serial — `std::thread::scope` needs real threads, which wasm does not
/// have.
fn bake_wave(field: &HeightField, wave: &[(TileKey, TileStamp)]) -> Vec<(usize, BakedTile)> {
    let build = |indexes: &[usize]| -> Vec<(usize, BakedTile)> {
        indexes
            .iter()
            .map(|&i| {
                let (key, stamp) = wave[i];
                (i, bake_tile(field, key, stamp.focus_tile))
            })
            .collect()
    };
    let to_build: Vec<usize> = (0..wave.len()).collect();
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
            .flat_map(|handle| handle.join().expect("tile baking panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::super::render::SharedTileMesh;
    use bevy::mesh::{Indices, VertexAttributeValues};
    use super::super::{TerrainTuning, rebuild_height_field};
    use super::*;

    fn flat_field() -> HeightField {
        HeightField::from_fn(|_, _| 3.0)
    }

    /// One baked f32 texel (R32Float layout: 4 bytes per texel).
    fn texel(image: &Image, ix: usize, iz: usize) -> f32 {
        let data = image.data.as_ref().expect("cpu-side data");
        let i = (iz * TILE_VERTS + ix) * 4;
        f32::from_le_bytes(data[i..i + 4].try_into().unwrap())
    }

    /// One baked RG32Float texel (slope components).
    fn texel2(image: &Image, ix: usize, iz: usize) -> [f32; 2] {
        let data = image.data.as_ref().expect("cpu-side data");
        let i = (iz * TILE_VERTS + ix) * 8;
        [
            f32::from_le_bytes(data[i..i + 4].try_into().unwrap()),
            f32::from_le_bytes(data[i + 4..i + 8].try_into().unwrap()),
        ]
    }

    #[test]
    fn shared_mesh_faces_up_with_no_excess_geometry() {
        // Backface culling uses winding: the one shared grid every tile
        // renders must have +y faces. It is EXACTLY the 33x33 grid — no
        // skirt ring: rims stitch instead (a hanging wall showed up as
        // coincident z-fighting polygons at ring seams), so any extra
        // vertices or triangles are a regression.
        let mut world = World::new();
        world.init_resource::<Assets<Mesh>>();
        let shared = SharedTileMesh::from_world(&mut world);
        let mesh = world
            .resource::<Assets<Mesh>>()
            .get(&shared.0)
            .expect("shared mesh");
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).expect("positions") {
            VertexAttributeValues::Float32x3(p) => p.clone(),
            _ => panic!("unexpected position format"),
        };
        let indices: Vec<u32> = match mesh.indices().expect("indices") {
            Indices::U32(i) => i.clone(),
            _ => panic!("unexpected index format"),
        };
        assert_eq!(positions.len(), TILE_VERTS * TILE_VERTS, "skirt verts crept back");
        assert_eq!(
            indices.len(),
            TILE_QUADS * TILE_QUADS * 6,
            "extra triangles crept back"
        );
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

    #[test]
    fn tuning_change_rebuilds_the_whole_tile_set() {
        // The live-regeneration contract: mutating the tuning resource
        // replaces every streamed tile with fresh bakes.
        let mut app = terrain_app(None);
        app.insert_resource(TerrainTuning::default());
        // One camera entity: streaming keys the ring centre off the
        // Camera3d entity's transform (plus its RtsCamera focus in Focus
        // mode) via `cameras.single()`.
        app.world_mut()
            .spawn((Camera3d::default(), RtsCamera::default()));
        app.add_systems(
            Update,
            (
                rebuild_height_field,
                stream_terrain_tiles.after(rebuild_height_field),
            ),
        );
        app.update();
        app.update();

        let before = tile_entities(&app);
        assert!(!before.is_empty(), "no tiles streamed initially");

        app.world_mut()
            .resource_mut::<TerrainTuning>()
            .tectonics
            .mountain_relief_m += 10.0;
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

    /// A streaming test app: the systems and resources the terrain needs,
    /// one RTS camera at the origin (spawned by the caller when ready).
    /// `budget_millis: None` = no budget (unlimited bakes per frame).
    fn terrain_app(budget_millis: Option<u64>) -> App {
        let mut app = App::new();
        app.insert_resource(HeightField::default())
            .init_resource::<TerrainTiles>()
            .init_resource::<TileRenderCache>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<Image>>()
            .init_resource::<Assets<TerrainMaterial>>()
            .init_resource::<SharedTileMesh>()
            .init_resource::<TileAtlas>()
            .insert_resource(Time::<()>::default())
            .add_systems(Update, stream_terrain_tiles);
        if let Some(millis) = budget_millis {
            app.insert_resource(StreamBudget { mesh_millis: millis });
        }
        app
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
                    covered += tiles_covering_point(
                        desired_set_around(Vec2::ZERO, min_level),
                        point,
                    ) as usize;
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
    fn hole_texels_place_the_cut_rectangle_and_empty_is_identity() {
        // The shader cuts a coarse tile's overlap with the finer square by
        // clamping verts into the hole rectangle (texel units); an empty
        // rect must encode to bounds clamp() never reaches.
        let key = TileKey {
            level: 1,
            x: 1,
            z: 1,
        }; // 128 m tile at (128, 128), 4 m cells
        let empty = hole_texels(key, None);
        assert!(empty.x < -1.0e8 && empty.y > 1.0e8, "empty rect: {empty:?}");

        // Finer square [0, 160]^2 world → texels relative to the tile
        // origin (128, 128): [-32, 8] in x and z.
        let rect = hole_texels(
            key,
            Some((Vec2::ZERO, Vec2::new(160.0, 160.0))),
        );
        assert!((rect.x - (-32.0)).abs() < 1e-4);
        assert!((rect.y - 8.0).abs() < 1e-4);
        assert!((rect.z - (-32.0)).abs() < 1e-4);
        assert!((rect.w - 8.0).abs() < 1e-4);
    }

    #[test]
    fn focus_movement_respawns_stale_stamped_tiles() {
        // Rims and holes are baked from the ring position at spawn time;
        // crossing a tile boundary must rebuild affected tiles instead of
        // leaving stale geometry mid-ring.
        let mut app = terrain_app(None);
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
        // a stitched rim texel must sit on the chord, an interior one on
        // the field itself (minus their respective drops).
        let field = HeightField::from_fn(|_, z| 0.001 * z * z);
        let key = TileKey {
            level: 0,
            x: 2,
            z: 0,
        }; // +x rim of the focus ring
        let bake = bake_tile(&field, key, IVec2::ZERO);

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
                h0 + (h1 - h0) * t - level_drop(1) - STITCH_BIAS
            };
            assert!(
                (texel(&bake.height, TILE_QUADS, iz) - expected).abs() < 1e-4,
                "rim texel iz={iz}: {} != {expected}",
                texel(&bake.height, TILE_QUADS, iz)
            );
        }
        // Interior vertices stay on the true field (minus own drop), e.g.
        // the centre: h(80, 16) - 1.
        let centre_expected = 0.001 * 16.0 * 16.0 - level_drop(0);
        assert!(
            (texel(&bake.height, 16, 16) - centre_expected).abs() < 1e-4,
            "interior texel must be un-stitched"
        );
    }

    #[test]
    fn top_level_rim_does_not_stitch_into_the_void() {
        // Regression: the top level's outer rim borders no coarser level;
        // stitching used to index LEVEL_CELL_M[9] and panic at runtime.
        // Periodic sub-scale heights, because real-amplitude fields at
        // 2 Mm tile distances exceed f32 precision in the test's
        // exact-value comparison.
        let field = HeightField::from_fn(|_, z| 0.001 * (z % 1000.0));
        let key = TileKey {
            level: (LEVEL_CELL_M.len() - 1) as u8,
            x: 2,
            z: 0,
        };
        let bake = bake_tile(&field, key, IVec2::ZERO); // must not panic
        // Rim texels stay on the plain field (minus the top level's drop).
        let drop = level_drop(key.level);
        let z = 5.0 * LEVEL_CELL_M[8];
        let expected = 0.001 * (z % 1000.0) - drop;
        assert!(
            (texel(&bake.height, TILE_QUADS, 5) - expected).abs() < 1e-2,
            "{} != {expected}",
            texel(&bake.height, TILE_QUADS, 5)
        );
    }

    #[test]
    fn slopes_agree_across_tile_boundaries() {
        // Slopes come from the field at world positions, so the shared
        // edge of two adjacent tiles must carry identical baked slopes —
        // this is what kills the per-tile lighting grid.
        let field = HeightField::default();
        let a = bake_tile(
            &field,
            TileKey {
                level: 0,
                x: 0,
                z: 0,
            },
            IVec2::ZERO,
                    );
        let b = bake_tile(
            &field,
            TileKey {
                level: 0,
                x: 1,
                z: 0,
            },
            IVec2::ZERO,
                    );
        // Shared edge x = 32: A's ix = TILE_QUADS column, B's ix = 0.
        for iz in 0..TILE_VERTS {
            assert_eq!(
                texel2(&a.slope, TILE_QUADS, iz),
                texel2(&b.slope, 0, iz),
                "slope mismatch on shared edge at iz={iz}"
            );
        }
    }

    #[test]
    fn colors_match_terrain_shape() {
        // Shading must agree with the fields it comes from: steepness
        // loses the grass green for cliff, creases brighten into chalky
        // drainage streaks — but only on STEEP ground: on flat plains
        // the same crease stays grass (chalk there read as pale
        // dandruff).
        let palette = ground_palette();
        let sample = |slope: Vec2, ridge: f32| {
            ground_color(
                &palette,
                slope.length(),
                10.0 / TerrainTuning::default().tectonics.mountain_relief_m
                    + super::VERTICAL_BIAS,
                ridge,
                1.0,
            )
        };
        let grass = sample(Vec2::ZERO, 1.0);
        let cliff = sample(Vec2::new(2.5, 0.0), 1.0);
        assert!(
            cliff.y < grass.y,
            "cliff must lose the grass green: {cliff} vs {grass}"
        );
        let crease = sample(Vec2::new(0.9, 0.0), -1.0);
        let flat_crease = sample(Vec2::ZERO, -1.0);
        assert!(
            crease.x > grass.x && crease.y > grass.y,
            "drainage brightens steep creases: {crease} vs {grass}"
        );
        assert_eq!(
            flat_crease, grass,
            "chalk must not fire on flat creases: {flat_crease} vs {grass}"
        );
    }

    #[test]
    fn streaming_respects_the_per_frame_budget() {
        // Zero budget: exactly one tile per frame (the progress
        // guarantee), and the full desired set eventually lands.
        let mut app = terrain_app(Some(0));
        app.world_mut()
            .spawn((Camera3d::default(), RtsCamera::default()));
        app.update();

        let first = tile_entities(&app).len();
        assert_eq!(first, 1, "zero budget must build exactly one tile per frame");

        let total = desired_tiles(Vec2::ZERO, 0).len();
        for _ in 0..(4 * total) {
            app.world_mut()
                .resource_mut::<Time>()
                .advance_by(Duration::from_secs_f32(1.0 / 60.0));
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
            app.world_mut()
                .resource_mut::<Time>()
                .advance_by(Duration::from_secs_f32(1.0 / 60.0));
            app.update();
        }
    }

    #[test]
    fn stale_tiles_render_until_their_replacement_spawns() {
        // The no-void invariant: crossing a tile boundary restamps the
        // rings, but the old surfaces stay in the world until each
        // replacement lands (zero budget = one bake per frame, so the
        // overlap window is long) — and the streamed set keeps covering
        // the focus throughout.
        let mut app = terrain_app(Some(0));
        let camera = app
            .world_mut()
            .spawn((Camera3d::default(), RtsCamera::default()))
            .id();
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
            "old surfaces vanished before their replacements: {survivors}/{}",
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
        // tiles fade out through the coarse representation and must not
        // retire while the coarser ring still cuts the hole that exposed
        // them — the cover has to land first.
        let mut app = terrain_app(Some(0));
        let camera = app
            .world_mut()
            .spawn((Camera3d::default(), RtsCamera::default()))
            .id();
        let total = desired_tiles(Vec2::ZERO, 0).len();
        converge(&mut app, total);

        // Lift the camera 40 km (focus stays): distance gates the finest
        // level up to L4, dropping the L0..L3 rings.
        app.world_mut()
            .get_entity_mut(camera)
            .unwrap()
            .get_mut::<Transform>()
            .unwrap()
            .translation = Vec3::new(0.0, 40_000.0, 0.0);
        app.update(); // one bake: the un-holed covers have not landed

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
    fn tile_atlas_layers_are_recycled_across_rebuilds() {
        // A tuning edit drops every live render and the whole cache; the
        // atlas must hand those layers back out instead of running its
        // high-water mark up — unbounded slider-dragging against a
        // bounded atlas would otherwise assert mid-frame.
        let mut app = terrain_app(None);
        app.world_mut()
            .spawn((Camera3d::default(), RtsCamera::default()));
        let total = desired_tiles(Vec2::ZERO, 0).len();
        converge(&mut app, total);
        let first_pass = app.world().resource::<TileAtlas>().high_water();
        assert!(first_pass > 0, "no atlas layers were claimed");

        for _ in 0..2 {
            app.world_mut()
                .resource_mut::<HeightField>()
                .set_changed();
            converge(&mut app, total);
        }
        assert_eq!(
            app.world().resource::<TileAtlas>().high_water(),
            first_pass,
            "atlas layers leaked across world rebuilds"
        );
    }

    #[test]
    fn recycled_tiles_reuse_cached_renders() {        // Panning away and back re-spawns from the render cache: the
        // return trip creates no new terrain materials.
        let mut app = terrain_app(None);
        let camera = app
            .world_mut()
            .spawn((Camera3d::default(), RtsCamera::default()))
            .id();
        let total = desired_tiles(Vec2::ZERO, 0).len();
        converge(&mut app, total);

        let asset_count = |app: &App| {
            app.world()
                .resource::<Assets<TerrainMaterial>>()
                .iter()
                .count()
        };
        pan_camera(&mut app, camera, 200.0);
        converge(&mut app, total);
        let away = asset_count(&app);

        pan_camera(&mut app, camera, 0.0);
        converge(&mut app, total);
        assert_eq!(
            asset_count(&app),
            away,
            "returning over visited ground built new renders instead of cache hits"
        );
        assert_eq!(tile_entities(&app).len(), total);
    }

    #[test]
    fn morph_band_blends_rim_tiles_onto_the_parent() {
        // A rim tile's outer band ramps onto the parent surface: the rim
        // edge lands exactly on it, the band's inner edge keeps the fine
        // height, and between them the blend is strictly monotone. This
        // band is what keeps the LOD boundary continuous — without it
        // the parent's boundary strip can tower over the fine ring's
        // valleys and occlude them.
        let field = HeightField::from_fn(|x, z| 0.001 * x * z);
        let key = TileKey { level: 0, x: 2, z: 0 }; // +x rim of the ring
        let bake = bake_tile(&field, key, IVec2::ZERO);
        let h = |ix: usize, iz: usize| texel(&bake.height, ix, iz);

        // Parent surface at the rim edge (x = 96): the edge sits exactly
        // on parent lattice lines along x, so only the z chord varies.
        let parent_at_rim = |z: f32| {
            let pz = (z / 4.0).floor();
            let z0 = pz * 4.0;
            let t = (z - z0) / 4.0;
            let a = 0.001 * 96.0 * z0;
            let b = 0.001 * 96.0 * (z0 + 4.0);
            a + (b - a) * t - level_drop(1) - STITCH_BIAS
        };

        // The tile's world origin is (64, 0): texel ix is world x = 64 + ix.
        let fine_at = |ix: f32, z: f32| 0.001 * (64.0 + ix) * z - level_drop(0);


        for iz in [0usize, 7, 16, 27, 32] {
            let z = iz as f32;
            let rim = h(TILE_QUADS, iz);
            let want_parent = parent_at_rim(z);
            assert!(
                (rim - want_parent).abs() < 1e-3,
                "rim iz={iz}: {rim} != parent {want_parent}"
            );
            // Inner edge of the band: the fine height, untouched.
            let band_inner = TILE_QUADS - MORPH_BAND_QUADS as usize;
            let inner = h(band_inner, iz);
            let want_fine = fine_at(band_inner as f32, z);
            assert!(
                (inner - want_fine).abs() < 1e-3,
                "band-inner iz={iz}: {inner} != fine {want_fine}"
            );
            // Halfway through the band: strictly between the two.
            let mid = h(TILE_QUADS - 4, iz);
            let (lo, hi) = (
                want_parent.min(want_fine),
                want_parent.max(want_fine),
            );
            assert!(
                mid >= lo - 1e-3 && mid <= hi + 1e-3,
                "band-mid iz={iz}: {mid} escaped [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn gate_flip_restamps_coarse_tiles_the_same_frame() {
        // Regression (the "mesh sitting on detailed terrain"): a zoom
        // gate flip makes a whole coarse level's stamps flip their hole
        // in or out. Those renders are cached from the last pass through
        // the zoom, so they must apply the SAME frame — unbudgeted — or
        // the stale holeless sheet covers the fresh fine ring (the old
        // giant LOD drops used to hide this; flat drops expose it).
        // Budget 0 (one bake per frame) maximizes the starvation window.
        let mut app = terrain_app(Some(0));
        let camera = app
            .world_mut()
            .spawn((Camera3d::default(), RtsCamera::default()))
            .id();
        // Zoomed in: fine levels render, coarse tiles cut holes.
        converge(&mut app, desired_tiles(Vec2::ZERO, 0).len());
        let zoomed_in = app.world().resource::<TerrainTiles>().tile_count();

        // Zoom out (camera far above): the gate lifts the fine levels.
        app.world_mut()
            .get_entity_mut(camera)
            .unwrap()
            .get_mut::<Transform>()
            .unwrap()
            .translation = Vec3::new(0.0, 40_000.0, 0.0);
        converge(&mut app, desired_tiles(Vec2::ZERO, 2).len());
        let zoomed_out = app.world().resource::<TerrainTiles>().tile_count();
        assert!(zoomed_out < zoomed_in);

        // Zoom back in. After ONE update every live tile must carry its
        // desired stamp — all the restamps were cache hits.
        app.world_mut()
            .get_entity_mut(camera)
            .unwrap()
            .get_mut::<Transform>()
            .unwrap()
            .translation = Vec3::new(0.0, 30.0, 40.0);
        app.update();
        let desired: HashMap<TileKey, TileStamp> =
            desired_tiles(Vec2::ZERO, 0).into_iter().collect();
        let stale: Vec<TileKey> = app
            .world()
            .resource::<TerrainTiles>()
            .tiles
            .iter()
            .filter(|(key, (_, live, _))| desired.get(key) != Some(live))
            .map(|(key, _)| *key)
            .collect();
        assert!(
            stale.is_empty(),
            "after one frame back at fine zoom, {stale:?} still render stale stamps (the holeless-sheet window)"
        );
    }

    #[test]
    fn hole_cutters_never_expose_their_hole() {
        // The void guard that replaced LOD fades: a coarse tile spawns
        // with its hole already cut, so anything under the hole must be
        // rendering BEFORE the cutter goes live. The spawn order is
        // finest-first and the budget here admits one tile per frame —
        // the widest exposure window — and still no hole may open.
        let mut app = terrain_app(Some(0));
        app.world_mut()
            .spawn((Camera3d::default(), RtsCamera::default()));
        let total = desired_tiles(Vec2::ZERO, 0).len();
        for _ in 0..(total * 2) {
            app.update();
            let tiles = &app.world().resource::<TerrainTiles>().tiles;
            for (&key, &(_, stamp, _)) in tiles {
                if !stamp.cuts_hole {
                    continue;
                }
                let (hmin, hmax) = hole_square(key.level, stamp.fine_focus_tile);
                // Corners-in plus centre sample the hole; the cover may
                // be any live tile that renders over the point.
                for p in [
                    Vec2::new(hmin.x + 1.0, hmin.y + 1.0),
                    Vec2::new(hmax.x - 1.0, hmin.y + 1.0),
                    Vec2::new(hmin.x + 1.0, hmax.y - 1.0),
                    Vec2::new(hmax.x - 1.0, hmax.y - 1.0),
                    (hmin + hmax) / 2.0,
                ] {
                    let covered = tiles.iter().any(|(&k, &(_, st, _))| {
                        k.contains(p) && renders_at(k.level, &st, p)
                    });
                    assert!(
                        covered,
                        "live hole-cutter {key:?} exposes {p:?}: spawn order broke the void guard"
                    );
                }
            }
        }
    }

    #[test]
    fn bake_tile_costs_about_two_milliseconds() {
        // Pins the SERIAL cost of one tile bake. Two-layer baseline
        // (tectonic Worley + erosion filter, 2026-09): ~1.9 ms release,
        // ~3.9 ms dev/cranelift isolated (~2.5 µs/sample: ~0.7 µs
        // tectonic + ~1.5 µs filter), up to ~5 ms when the whole test
        // binary competes for cores, and ~8 ms with heavy EXTERNAL
        // system load on top (load average ~11). Wall-clock timing, so
        // the best of three passes is the number that's pinned —
        // contention inflates individual passes, while a real
        // noise/erosion regression slows every one. The spawn pass bakes
        // whole waves in parallel (see `bake_wave`); a regression that
        // MULTIPLIES this starves streaming even across cores. Run in
        // release for the real number.
        let field = HeightField::default();
        let set = desired_tiles(Vec2::new(123.4, -45.6), 0);
        let mut best = Duration::from_secs(1);
        for _ in 0..3 {
            let start = Instant::now();
            for (key, stamp) in &set {
                let _ = bake_tile(&field, *key, stamp.focus_tile);
            }
            best = best.min(start.elapsed() / set.len() as u32);
        }
        eprintln!("bake_tile: {best:?}/tile over {} tiles", set.len());
        assert!(
            best < Duration::from_millis(8),
            "tile bake avg {best:?} — parallel waves assume ~2 ms serial release bakes"
        );
    }
}
