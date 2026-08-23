use crate::boid::Boid;
use crate::kinematics::{HardCollision, TrackedByTree, Velocity};
use bevy::prelude::*;
use bevy_rts_camera::{Ground, RtsCamera};
use bevy_voxel_world::prelude::{
    TextureIndexMapperFn, VoxelLookupDelegate, VoxelWorld, VoxelWorldConfig, WorldVoxel,
};
use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Procedural terrain: fBm cascade
//
// The authoritative terrain is a pure height function `TerrainNoise::height`
// (world metres, y up, one voxel = 1 m). Everything else — voxel lookup,
// obstacle placement, later walkability — samples the same function so the
// world stays self-consistent no matter which subsystem asks.
// ---------------------------------------------------------------------------

/// Deterministic world seed; later becomes part of the saved world state.
const WORLD_SEED: i32 = 1337;

// Amplitudes in metres, wavelengths in metres (noise frequency = 1/wavelength).
const WARP_AMPLITUDE_M: f32 = 140.0;
const WARP_WAVELENGTH_M: f32 = 900.0;
const BASE_AMPLITUDE_M: f32 = 12.0;
const BASE_WAVELENGTH_M: f32 = 420.0;
const RIDGE_MAX_M: f32 = 34.0;
const RIDGE_WAVELENGTH_M: f32 = 780.0;
/// Mountains only where the low-frequency mask is high, so they cluster into
/// ranges instead of covering the map.
const MASK_WAVELENGTH_M: f32 = 1300.0;
const DETAIL_AMPLITUDE_M: f32 = 1.4;
const DETAIL_WAVELENGTH_M: f32 = 56.0;

// Voxel materials; indices into the default 4x4 texture palette.
const MATERIAL_BEDROCK: u8 = 0;
const MATERIAL_GRASS: u8 = 1;
const MATERIAL_ROCK: u8 = 2;
const MATERIAL_SHORE: u8 = 3;

/// Everything below this height is bedrock, so no hole can open into the void.
const BEDROCK_TOP_Y: i32 = 0;

fn surface_material(surface_y: i32) -> u8 {
    if surface_y <= 1 {
        MATERIAL_SHORE
    } else if surface_y < 16 {
        MATERIAL_GRASS
    } else {
        MATERIAL_ROCK
    }
}

/// The configured noise cascade. Construct once (per lookup delegate, per
/// obstacle scatter) — `FastNoiseLite` is plain data, cheap to clone around
/// but pointless to rebuild per sample.
pub struct TerrainNoise {
    warp: FastNoiseLite,
    base: FastNoiseLite,
    ridge: FastNoiseLite,
    mask: FastNoiseLite,
    detail: FastNoiseLite,
}

impl Default for TerrainNoise {
    fn default() -> Self {
        let mut warp = FastNoiseLite::new();
        warp.set_seed(Some(WORLD_SEED));
        warp.set_noise_type(Some(NoiseType::OpenSimplex2));
        warp.set_frequency(Some(1.0 / WARP_WAVELENGTH_M));
        warp.set_domain_warp_amp(Some(WARP_AMPLITUDE_M));

        let fbm = |wavelength: f32, octaves: i32| {
            let mut n = FastNoiseLite::new();
            n.set_seed(Some(WORLD_SEED));
            n.set_noise_type(Some(NoiseType::OpenSimplex2));
            n.set_fractal_type(Some(FractalType::FBm));
            n.set_fractal_octaves(Some(octaves));
            n.set_frequency(Some(1.0 / wavelength));
            n
        };

        let mut ridge = fbm(RIDGE_WAVELENGTH_M, 4);
        ridge.set_fractal_type(Some(FractalType::Ridged));

        TerrainNoise {
            warp,
            base: fbm(BASE_WAVELENGTH_M, 4),
            ridge,
            mask: fbm(MASK_WAVELENGTH_M, 3),
            detail: fbm(DETAIL_WAVELENGTH_M, 3),
        }
    }
}

impl TerrainNoise {
    /// Terrain height in world metres at (x, z). Deterministic for a seed.
    pub fn height(&self, x: f32, z: f32) -> f32 {
        let (wx, wz) = self.warp.domain_warp_2d(x, z);
        let base = self.base.get_noise_2d(wx, wz) * BASE_AMPLITUDE_M;
        // Ridged noise peaks where the noise crosses zero: |n| gives sharp
        // ridge lines, the mask confines them to mountain ranges.
        let ridge = 1.0 - self.ridge.get_noise_2d(wx, wz).abs();
        let mask = (self.mask.get_noise_2d(x, z) * 0.5 + 0.5).clamp(0.0, 1.0);
        let detail = self.detail.get_noise_2d(x, z) * DETAIL_AMPLITUDE_M;
        base + ridge * mask * RIDGE_MAX_M + detail
    }
}

/// Shared height function: terrain height in metres at world (x, z).
/// Used by systems that just need a height probe without a voxel query.
pub fn terrain_height(x: f32, z: f32) -> f32 {
    TerrainNoise::default().height(x, z)
}

// ---------------------------------------------------------------------------
// bevy_voxel_world configuration
// ---------------------------------------------------------------------------

/// Marker config for our main (and currently only) voxel world.
#[derive(Resource, Default, Clone)]
pub struct MainWorld;

impl VoxelWorldConfig for MainWorld {
    type MaterialIndex = u8;
    /// Mark every chunk as camera ground so bevy_rts_camera raycasts (pan
    /// focus, drag) hit the actual terrain instead of a flat plane.
    type ChunkUserBundle = (Ground,);

    /// Chunks are 32 voxels; 16 chunks = 512 m radius around the camera.
    fn spawning_distance(&self) -> u32 {
        16
    }

    /// Keep a core radius spawned even when looking away (viewport culling
    /// otherwise despawns everything behind the camera between frames).
    fn min_despawn_distance(&self) -> u32 {
        4
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        Box::new(|_chunk_pos, _lod, _previous| {
            let noise = TerrainNoise::default();
            // Column heights are pure in (x, z): cache per chunk so the
            // noise cascade runs once per column, not once per voxel.
            let mut height_cache: HashMap<(i32, i32), i32> = HashMap::new();
            Box::new(move |pos: IVec3, _previous| {
                let h = *height_cache
                    .entry((pos.x, pos.z))
                    .or_insert_with(|| noise.height(pos.x as f32, pos.z as f32).floor() as i32);
                if pos.y <= BEDROCK_TOP_Y {
                    WorldVoxel::Solid(MATERIAL_BEDROCK)
                } else if pos.y < h {
                    WorldVoxel::Solid(surface_material(h))
                } else {
                    WorldVoxel::Air
                }
            })
        })
    }

    fn texture_index_mapper(&self) -> TextureIndexMapperFn<Self::MaterialIndex> {
        // The bundled fallback texture is a 4x4 palette; map materials onto
        // the first row until real terrain texturing lands.
        Arc::new(|material| [(material % 4) as u32; 3])
    }
}

// ---------------------------------------------------------------------------
// Boid grounding
// ---------------------------------------------------------------------------

/// Terrain grounding state per boid: the surface height under the boid and
/// the voxel column it was computed for. The surface is rescanned only when
/// the boid enters a new voxel column; `bob` renders from it every frame.
#[derive(Component, Debug)]
pub struct GroundY {
    /// World-space height of the terrain surface (top face of the top solid
    /// voxel) under the boid's current column.
    pub surface: f32,
    /// Column the surface was computed for; sentinel `i32::MIN` forces a
    /// wide first scan because the spawn height can be far off terrain.
    col: (i32, i32),
}

impl Default for GroundY {
    fn default() -> Self {
        GroundY {
            surface: 0.0,
            col: (i32::MIN, i32::MIN),
        }
    }
}

/// Scan a voxel column downward for the first solid voxel with air above it.
/// Returns the world-space surface height. `Unset` counts as air: chunks
/// stream in over several frames, so the world under a boid may legitimately
/// not exist yet.
pub(crate) fn find_surface<I>(
    get_voxel: &dyn Fn(IVec3) -> WorldVoxel<I>,
    x: i32,
    z: i32,
    top_y: i32,
    bottom_y: i32,
) -> Option<f32> {
    let mut y = top_y;
    while y >= bottom_y {
        let below = get_voxel(IVec3::new(x, y, z));
        if !matches!(below, WorldVoxel::Air | WorldVoxel::Unset) {
            let above = get_voxel(IVec3::new(x, y + 1, z));
            if matches!(above, WorldVoxel::Air | WorldVoxel::Unset) {
                return Some((y + 1) as f32);
            }
        }
        y -= 1;
    }
    None
}

/// Keep every boid on the terrain surface. Runs after `move_step` (fresh
/// positions) and before `bob` (which renders the y). Vertical velocity is
/// cancelled: steering targets the y boids spawned at, and `bob` overwrites
/// the rendered y anyway — without the reset both would accumulate against
/// the terrain-following height.
pub fn ground_boids(
    mut query: Query<(&Transform, &mut GroundY, &mut Velocity), With<Boid>>,
    voxel_world: VoxelWorld<MainWorld>,
) {
    let get_voxel = voxel_world.get_voxel_fn();
    query.par_iter_mut().for_each(|(transform, mut ground, mut vel)| {
        let t = transform.translation;
        let col = (t.x.floor() as i32, t.z.floor() as i32);
        if col != ground.col {
            let (top, bottom) = if ground.col.0 == i32::MIN {
                ((t.y + 64.0) as i32, (t.y - 64.0) as i32)
            } else {
                ((t.y + 8.0) as i32, (t.y - 16.0) as i32)
            };
            if let Some(surface) = find_surface(&*get_voxel, col.0, col.1, top, bottom) {
                ground.surface = surface;
            }
            // Keep the last surface when nothing was found: the chunk is
            // probably still generating, better to walk on stale ground
            // than to fall.
            ground.col = col;
        }
        vel.v.y = 0.0;
        vel.a.y = 0.0;
    });
}

// ---------------------------------------------------------------------------
// Debug terrain brushes
//
// 7/8/9/0 select raise/lower/flatten/smooth, [ and ] resize the brush, hold
// G to apply at the cursor. Moats and earthen walls are composites of these
// primitives; this is the spike harness for the deformation feel.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BrushMode {
    Raise,
    Lower,
    Flatten,
    Smooth,
}

impl BrushMode {
    fn from_key(code: KeyCode) -> Option<Self> {
        Some(match code {
            KeyCode::Digit7 => BrushMode::Raise,
            KeyCode::Digit8 => BrushMode::Lower,
            KeyCode::Digit9 => BrushMode::Flatten,
            KeyCode::Digit0 => BrushMode::Smooth,
            _ => return None,
        })
    }
}

/// Debug brush state, shared with the brush gizmo preview.
#[derive(Resource)]
pub struct TerrainBrush {
    pub mode: BrushMode,
    pub radius_m: f32,
}

impl Default for TerrainBrush {
    fn default() -> Self {
        TerrainBrush {
            mode: BrushMode::Raise,
            radius_m: 4.0,
        }
    }
}

const BRUSH_STRENGTH_M_PER_SEC: f32 = 6.0;
const BRUSH_MIN_RADIUS_M: f32 = 1.0;
const BRUSH_MAX_RADIUS_M: f32 = 16.0;
/// Column scan window around the cursor hit when looking for the current
/// surface: tall enough to find the top of a pile built by repeated raising.
const BRUSH_SCAN_UP_M: i32 = 32;
const BRUSH_SCAN_DOWN_M: i32 = 48;
/// Flatten/smooth approach rate, as a fraction of the remaining difference
/// per second.
const BRUSH_LEVEL_RATE_PER_SEC: f32 = 10.0;
const BRUSH_MIN_SURFACE_Y: i32 = BEDROCK_TOP_Y + 1;

/// Smooth (quadratic) falloff weight of a column at distance `d` from the
/// brush centre.
fn brush_falloff(d2: f32, radius: f32) -> f32 {
    (1.0 - d2 / (radius * radius)).clamp(0.0, 1.0)
}

/// New surface height for one column under the brush, before falloff
/// weighting. Pure: the caller supplies the current surfaces (centre, this
/// column, and the column's 4-neighbourhood average for smoothing) and the
/// delta time to integrate at.
fn brush_target_height(
    mode: BrushMode,
    centre_h: f32,
    current: f32,
    neighbour_avg: f32,
    dt: f32,
) -> f32 {
    match mode {
        BrushMode::Raise => current + BRUSH_STRENGTH_M_PER_SEC * dt,
        BrushMode::Lower => current - BRUSH_STRENGTH_M_PER_SEC * dt,
        BrushMode::Flatten => current + (centre_h - current) * BRUSH_LEVEL_RATE_PER_SEC * dt,
        BrushMode::Smooth => current + (neighbour_avg - current) * BRUSH_LEVEL_RATE_PER_SEC * dt,
    }
}

pub fn terrain_brush_system(
    mut brush: ResMut<TerrainBrush>,
    keys: Res<ButtonInput<KeyCode>>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    windows: Query<&Window>,
    time: Res<Time>,
    mut voxel_world: VoxelWorld<MainWorld>,
    mut gizmos: Gizmos,
) {
    for code in [
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
        KeyCode::Digit0,
    ] {
        if keys.just_pressed(code) {
            if let Some(mode) = BrushMode::from_key(code) {
                brush.mode = mode;
            }
        }
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        brush.radius_m = (brush.radius_m - 1.0).max(BRUSH_MIN_RADIUS_M);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        brush.radius_m = (brush.radius_m + 1.0).min(BRUSH_MAX_RADIUS_M);
    }

    let Ok((camera, camera_transform)) = camera_query.single() else {
        return;
    };
    let Some(cursor) = windows.single().ok().and_then(|w| w.cursor_position()) else {
        return;
    };
    let Some(point) = crate::player::get_intersection(&voxel_world, &cursor, camera, camera_transform)
    else {
        return;
    };

    // Brush outline preview at the cursor: colour hints at the mode.
    let colour = match brush.mode {
        BrushMode::Raise => Color::srgb(0.2, 0.9, 0.2),
        BrushMode::Lower => Color::srgb(0.9, 0.3, 0.2),
        BrushMode::Flatten => Color::srgb(0.9, 0.8, 0.2),
        BrushMode::Smooth => Color::srgb(0.3, 0.5, 0.9),
    };
    gizmos.circle(
        Isometry3d::new(point + Vec3::Y * 0.05, Quat::from_rotation_x(-FRAC_PI_2)),
        brush.radius_m,
        colour,
    );

    if !keys.pressed(KeyCode::KeyG) {
        return;
    }

    let dt = time.delta_secs().max(0.001);
    let get_voxel = voxel_world.get_voxel_fn();
    let centre_voxel = point.floor().as_ivec3();
    let top = centre_voxel.y + 1 + BRUSH_SCAN_UP_M;
    let bottom = (centre_voxel.y + 1 - BRUSH_SCAN_DOWN_M).max(BRUSH_MIN_SURFACE_Y);
    let radius = brush.radius_m;

    // Read pass: current surfaces of every column in the disc.
    let r = radius.ceil() as i32;
    let mut surfaces: Vec<(i32, i32, f32)> = Vec::with_capacity((2 * r + 1).pow(2) as usize);
    for dx in -r..=r {
        for dz in -r..=r {
            let d2 = (dx * dx + dz * dz) as f32;
            if d2 > radius * radius {
                continue;
            }
            let x = centre_voxel.x + dx;
            let z = centre_voxel.z + dz;
            if let Some(surface) = find_surface(&*get_voxel, x, z, top, bottom) {
                surfaces.push((x, z, surface));
            }
        }
    }
    let centre_h = surfaces
        .iter()
        .find(|(x, z, _)| *x == centre_voxel.x && *z == centre_voxel.z)
        .map(|&(_, _, h)| h)
        .unwrap_or(centre_voxel.y as f32 + 1.0);

    // Compute + write pass. Smooth uses the average of the 4-neighbourhood
    // from the pre-edit snapshot so the result does not smear asymmetrically.
    let neighbour_avg = |x: i32, z: i32| -> f32 {
        let mut sum = 0.0;
        let mut n = 0.0;
        for (nx, nz, nh) in &surfaces {
            if (nx - x).abs() + (nz - z).abs() == 1 {
                sum += nh;
                n += 1.0;
            }
        }
        if n > 0.0 {
            sum / n
        } else {
            centre_h
        }
    };

    for &(x, z, current) in &surfaces {
        let d2 = ((x - centre_voxel.x).pow(2) + (z - centre_voxel.z).pow(2)) as f32;
        let weight = brush_falloff(d2, radius);
        if weight <= 0.0 {
            continue;
        }
        let avg = neighbour_avg(x, z);
        let target = brush_target_height(brush.mode, centre_h, current, avg, dt);
        let new_h = (current + (target - current) * weight).max(BRUSH_MIN_SURFACE_Y as f32);
        let old_i = current.floor() as i32;
        let new_i = new_h.floor() as i32;
        let material = surface_material(new_i);
        if new_i > old_i {
            for y in (old_i + 1)..=new_i {
                voxel_world.set_voxel(IVec3::new(x, y, z), WorldVoxel::Solid(material));
            }
        } else if new_i < old_i {
            for y in (new_i + 1)..=old_i {
                voxel_world.set_voxel(IVec3::new(x, y, z), WorldVoxel::Air);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Camera terrain clearance
// ---------------------------------------------------------------------------

/// Minimum altitude kept between the camera and the highest terrain column
/// within `CAMERA_CLEARANCE_RADIUS_M` of its XZ position.
const CAMERA_TERRAIN_MARGIN_M: f32 = 1.0;
/// Horizontal clearance radius. Clearing only the column directly below is
/// wrong on slopes: the camera slides along a mountainside with the columns
/// *beside* it higher, clipping into their sides. The camera must clear the
/// highest surface in this disc (the 8 neighbours plus a partial second
/// ring at 1 m voxels), which is what makes a zoomed, rotated view from a
/// mountaintop possible.
const CAMERA_CLEARANCE_RADIUS_M: f32 = 2.0;
/// How fast the clearance lift relaxes when terrain no longer demands it.
/// Rising is instant (never clip); falling is eased so the camera doesn't
/// staircase down rough ground in 1 m voxel steps.
const CAMERA_CLEARANCE_RELAX_M_PER_SEC: f32 = 8.0;
/// Column scan floor below the camera: valley surfaces far underneath
/// cannot clip the camera, so scans stop here.
const CAMERA_SCAN_BELOW_M: i32 = 64;

/// Smoothed clearance lift per camera. The plugin re-derives the camera
/// transform every frame, so the clamp is recomputed from scratch each
/// frame against the plugin's placement; this component carries only the
/// relax smoothing between frames.
#[derive(Component, Default)]
pub struct CameraClearance {
    offset: f32,
}

/// Keep the RTS camera out of the terrain. bevy_rts_camera terrain-follows
/// only the *focus* point (`follow_ground`), so zooming in on a mountain
/// lowers the camera into the peak and its surrounding columns. A
/// focus->camera ray does not work (the ~20 degree RTS angle makes it graze
/// terrain immediately); instead, sample the highest surface in a small
/// disc around the camera's XZ position and keep the camera above it.
/// Runs after `RtsCameraSystemSet`; only the height is touched — the
/// plugin owns XZ and rotation. Generic over the world so tests can run
/// it against deterministic synthetic terrain.
pub fn camera_terrain_clearance<C: VoxelWorldConfig>(
    mut cameras: Query<(&mut Transform, &RtsCamera, &mut CameraClearance), With<Camera3d>>,
    voxel_world: VoxelWorld<C>,
    time: Res<Time>,
) {
    let get_voxel = voxel_world.get_voxel_fn();
    for (mut transform, rts, mut clearance) in &mut cameras {
        let pos = transform.translation;
        let cx = pos.x.floor() as i32;
        let cz = pos.z.floor() as i32;

        // Scan from above anything the plugin can place under us (so a
        // buried camera, or a cliff wall beside it, still finds the top
        // surface) down to a floor well below: deep surfaces can't clip.
        let top = pos.y as i32 + rts.height_max as i32;
        let bottom = pos.y as i32 - CAMERA_SCAN_BELOW_M;
        let r = CAMERA_CLEARANCE_RADIUS_M.ceil() as i32;
        let mut max_surface = f32::NEG_INFINITY;
            for dx in -r..=r {
                for dz in -r..=r {
                    if (dx * dx + dz * dz) as f32 > CAMERA_CLEARANCE_RADIUS_M.powi(2) {
                        continue;
                    }
                if let Some(h) = find_surface(&*get_voxel, cx + dx, cz + dz, top, bottom) {
                    max_surface = max_surface.max(h);
                }
            }
        }
        if max_surface == f32::NEG_INFINITY {
            continue; // no terrain loaded around the camera yet
        }

        let min_y = max_surface + CAMERA_TERRAIN_MARGIN_M;
        let needed = (min_y - pos.y).max(0.0);
        if needed > clearance.offset {
            clearance.offset = needed;
        } else {
            clearance.offset =
                (clearance.offset - CAMERA_CLEARANCE_RELAX_M_PER_SEC * time.delta_secs())
                    .max(needed);
        }
        if clearance.offset > 0.0 {
            transform.translation.y = pos.y + clearance.offset;
        }
    }
}

// ---------------------------------------------------------------------------
// Obstacles
// ---------------------------------------------------------------------------

#[derive(Component, Default)]
pub struct Obstacle {
    pub(crate) normal: Vec3,
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::MinimalPlugins;
    use bevy::app::App;
    use bevy::ecs::system::RunSystemOnce;
    use bevy::input::mouse::{MouseMotion, MouseWheel};
    use bevy::time::{TimePlugin, TimeUpdateStrategy};
    use bevy::transform::TransformPlugin;
    use bevy_rts_camera::{RtsCameraPlugin, RtsCameraSystemSet};
    use bevy_voxel_world::prelude::{
        ChunkDespawnStrategy, ChunkSpawnStrategy, VoxelWorldCamera, VoxelWorldPlugin,
    };
    use std::time::Duration;

    /// Synthetic worst-case terrain: a `TEST_CLIFF_HEIGHT` wall for x >= 0
    /// against a low plain for x < 0 — a vertical cliff, the steepest a
    /// voxel world can be. Everything downstream (clearance, later
    /// walkability) runs against this instead of the fBm world so the
    /// geometry under test is deterministic.
    const TEST_CLIFF_HEIGHT: i32 = 30;

    #[derive(Resource, Default, Clone)]
    struct TestWorld;

    impl VoxelWorldConfig for TestWorld {
        type MaterialIndex = u8;
        type ChunkUserBundle = (Ground,);

        fn spawning_distance(&self) -> u32 {
            // Generous enough that terrain generates under the camera at
            // every zoom level (height_max = 300 m is ~10 chunks up).
            12
        }
        fn min_despawn_distance(&self) -> u32 {
            4
        }
        /// Viewport-based culling needs a real window; FarAway despawn plus
        /// Close spawn keeps chunk lifecycle purely distance-based, which a
        /// headless test can drive deterministically.
        fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
            ChunkDespawnStrategy::FarAway
        }

        fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
            ChunkSpawnStrategy::Close
        }

        fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
            Box::new(|_, _, _| {
                Box::new(|pos: IVec3, _| {
                    let surface = if pos.x >= 0 {
                        TEST_CLIFF_HEIGHT
                    } else {
                        1
                    };
                    if pos.y <= 0 {
                        WorldVoxel::Solid(0)
                    } else if pos.y < surface {
                        WorldVoxel::Solid(1)
                    } else {
                        WorldVoxel::Air
                    }
                })
            })
        }
    }

    /// Per-tick observation of the camera against the voxel world, written
    /// by `probe_camera` so the assertions can run outside the schedule.
    #[derive(Resource, Default)]
    struct CamProbe {
        camera: Vec3,
        /// Any solid voxel within one voxel of the camera position in every
        /// axis (a 3x3x3 box, the conservative reading of "at least 1 m
        /// from terrain geometry").
        solid_within_1m: bool,
        /// Surface height in the camera's own column, once terrain there is
        /// generated.
        surface_below: Option<f32>,
    }

    fn probe_camera(
        mut probe: ResMut<CamProbe>,
        cameras: Query<&Transform, With<Camera3d>>,
        voxel_world: VoxelWorld<TestWorld>,
    ) {
        let Ok(transform) = cameras.single() else {
            return;
        };
        let pos = transform.translation;
        let get_voxel = voxel_world.get_voxel_fn();
        let base = pos.floor().as_ivec3();
        let mut solid = false;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let v = get_voxel(base + IVec3::new(dx, dy, dz));
                    if !matches!(v, WorldVoxel::Air | WorldVoxel::Unset) {
                        solid = true;
                    }
                }
            }
        }
        probe.camera = pos;
        probe.solid_within_1m = solid;
        // Deep scan: at far zoom the camera is up to height_max above the
        // ground, and the interesting assertion is still "clear of the
        // surface far below".
        probe.surface_below = find_surface(
            &*get_voxel,
            base.x,
            base.z,
            (pos.y + 2.0) as i32,
            (pos.y - 400.0) as i32,
        );
    }

    /// Headless camera rig: task pool (voxel world generates chunks on it),
    /// manual time, transform propagation for the plugin's ground raycast,
    /// the RTS camera plugin, and the voxel test world with the clearance
    /// clamp and probe chained after the plugin's camera update.
    fn camera_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin))
            // RtsCameraPlugin bundles the input controller, whose systems
            // require input resources; stub them empty so those systems
            // no-op and the camera chain runs undisturbed.
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<ButtonInput<KeyCode>>()
            // MeshRayCast (follow_ground, grab_pan) needs the mesh store;
            // the voxel plugin runs meshless here, so provide it empty.
            .init_resource::<Assets<Mesh>>()
            .add_message::<MouseMotion>()
            .add_message::<MouseWheel>()
            .add_plugins(RtsCameraPlugin)
            // Custom-material mode: the chunk-task polling system is gated
            // on `LoadingTexture::is_loaded`, which only the custom-material
            // path sets synchronously (`minimal()` never loads a texture,
            // so generated voxel data would never land in the chunk map).
            // A plain StandardMaterial keeps everything headless: no render
            // plugins, but real chunk generation, meshing, and our Ground
            // bundles. The asset stores below are pure storage the plugin's
            // mesh/material bookkeeping expects to exist.
            .init_resource::<Assets<Shader>>()
            .init_resource::<Assets<StandardMaterial>>()
            .add_plugins(
                VoxelWorldPlugin::with_config(TestWorld)
                    .with_material(StandardMaterial::default()),
            )
            .init_resource::<CamProbe>()
            .add_systems(
                Update,
                (
                    camera_terrain_clearance::<TestWorld>.after(RtsCameraSystemSet),
                    probe_camera.after(camera_terrain_clearance::<TestWorld>),
                ),
            );
        // Burn the first update: it only records the initial clock instant.
        app.update();
        app
    }

    fn tick(app: &mut App, dt: f32) {
        app.insert_resource(TimeUpdateStrategy::ManualDuration(
            Duration::from_secs_f32(dt),
        ));
        app.update();
    }

    fn spawn_camera(app: &mut App, focus: Vec3) -> Entity {
        app.world_mut()
            .spawn((
                Camera3d::default(),
                VoxelWorldCamera::<TestWorld>::default(),
                CameraClearance::default(),
                RtsCamera {
                    // height 2..300, fixed 20 degree angle (dynamic angle
                    // depends on zoom smoothing on real time; keep the test
                    // deterministic).
                    height_min: 2.0,
                    height_max: 300.0,
                    angle: 20.0f32.to_radians(),
                    target_angle: 20.0f32.to_radians(),
                    min_angle: 20.0f32.to_radians(),
                    dynamic_angle: false,
                    smoothness: 0.3,
                    focus: Transform::from_translation(focus),
                    target_focus: Transform::from_translation(focus),
                    zoom: 0.0,
                    target_zoom: 0.0,
                    snap: false,
                    ..Default::default()
                },
            ))
            .id()
    }

    /// Teleport the camera (focus snapped with a facing, zoom applied
    /// immediately — the plugin's own smoothing runs on real time, which a
    /// headless test cannot advance) and reset the clearance relax state so
    /// it cannot mask a failure with a leftover lift from the previous
    /// position. `face` is the focus forward direction: the plugin pushes
    /// the camera backwards along it, which is what puts the camera beside
    /// (rather than on top of) terrain features.
    fn place_camera(app: &mut App, camera: Entity, focus: Vec3, face: Vec3, zoom: f32) {
        let orientation =
            Transform::from_translation(focus).looking_at(focus + face, Vec3::Y);
        let mut entity = app.world_mut().get_entity_mut(camera).unwrap();
        let mut rts = entity.get_mut::<RtsCamera>().unwrap();
        rts.focus = orientation;
        rts.target_focus = orientation;
        // zoom 1.0 is height_min (closest), 0.0 is height_max (far).
        rts.zoom = zoom;
        rts.target_zoom = zoom;
        drop(rts);
        entity.get_mut::<CameraClearance>().unwrap().offset = 0.0;
    }

    /// Chunk generation is async on the task pool: tick until the terrain
    /// under the camera exists, bounded so a generation failure fails the
    /// test instead of hanging it.
    fn wait_for_terrain(app: &mut App) {
        for _ in 0..600 {
            tick(app, 1.0 / 60.0);
            if app.world().resource::<CamProbe>().surface_below.is_some() {
                return;
            }
        }
        let camera = app.world().resource::<CamProbe>().camera;
        let any_solid = app
            .world_mut()
            .run_system_once(scan_for_solid_near_camera)
            .ok()
            .flatten();
        let (zoom, target_zoom, focus) = app
            .world_mut()
            .query::<&RtsCamera>()
            .single(app.world())
            .map(|rts| (rts.zoom, rts.target_zoom, rts.focus.translation))
            .unwrap_or((f32::NAN, f32::NAN, Vec3::ZERO));
        panic!(
            "terrain under the camera never generated: camera at {camera:?}, \
             zoom {zoom} target {target_zoom} focus {focus:?}, \
             any solid within 16 columns: {any_solid:?}"
        );
    }

    /// Diagnostic for wait_for_terrain failures: is there any solid voxel
    /// in a coarse box around the camera?
    fn scan_for_solid_near_camera(
        cameras: Query<&Transform, With<Camera3d>>,
        voxel_world: VoxelWorld<TestWorld>,
    ) -> Option<bool> {
        let pos = cameras.single().ok()?.translation;
        let get_voxel = voxel_world.get_voxel_fn();
        let base = pos.floor().as_ivec3();
        for dx in -16..=16 {
            for dz in -16..=16 {
                for dy in -400..=40 {
                    if !matches!(
                        get_voxel(base + IVec3::new(dx, dy, dz)),
                        WorldVoxel::Air | WorldVoxel::Unset
                    ) {
                        return Some(true);
                    }
                }
            }
        }
        Some(false)
    }

    #[test]
    fn camera_keeps_clear_of_steep_terrain_across_zoom_levels() {
        let mut app = camera_test_app();
        let camera = spawn_camera(&mut app, Vec3::new(-20.0, 1.0, 0.0));

        // Lowest and highest ground, the cliff crest, and — the case that
        // needs neighbour sampling — parked just off the wall on the plain
        // side, facing along +X so the plugin's backwards camera offset
        // pushes it over the plain beside the 30 m cliff face.
        let forward = Vec3::Z;
        let scenarios: [(Vec3, Vec3, &str); 4] = [
            (Vec3::new(-20.0, 1.0, 0.0), forward, "valley floor"),
            (
                Vec3::new(1.0, TEST_CLIFF_HEIGHT as f32, 4.0),
                forward,
                "cliff crest",
            ),
            (
                Vec3::new(10.0, TEST_CLIFF_HEIGHT as f32, 0.0),
                forward,
                "ridge top",
            ),
            (
                Vec3::new(-1.5, 1.0, 4.0),
                Vec3::NEG_X,
                "beside the cliff wall",
            ),
        ];

        for (focus, face, name) in scenarios {
            // Closest zoom first (worst case: camera_height = height_min),
            // then mid and far.
            for zoom in [1.0, 0.5, 0.0] {
                place_camera(&mut app, camera, focus, face, zoom);
                wait_for_terrain(&mut app);
                // A few more ticks so the clamp reacts to freshly loaded
                // terrain, not just the camera placement.
                for _ in 0..5 {
                    tick(&mut app, 1.0 / 60.0);
                }
                let probe = app.world().resource::<CamProbe>();
                assert!(
                    !probe.solid_within_1m,
                    "{name} @ zoom {zoom}: camera {:?} clips terrain",
                    probe.camera
                );
                if let Some(surface) = probe.surface_below {
                    assert!(
                        probe.camera.y >= surface + CAMERA_TERRAIN_MARGIN_M - 1e-3,
                        "{name} @ zoom {zoom}: camera {:?} below surface {surface} + margin",
                        probe.camera
                    );
                }
            }
        }
    }
}
