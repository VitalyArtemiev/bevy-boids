use crate::boid::Boid;
use crate::kinematics::{HardCollision, TrackedByTree, Velocity};
use bevy::prelude::*;
use bevy_rts_camera::Ground;
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
pub(crate) fn find_surface(
    get_voxel: &dyn Fn(IVec3) -> WorldVoxel,
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
