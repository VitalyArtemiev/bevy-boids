//! Pure procedural terrain generation: the fBm cascade behind every voxel.
//!
//! The authoritative terrain is the height function `TerrainNoise::height`
//! (world metres, y up, one voxel = 1 m). Everything else — the voxel
//! lookup, obstacle placement, later walkability — samples the same
//! function so the world stays self-consistent no matter which subsystem
//! asks.

use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use std::sync::LazyLock;

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
pub(crate) const MATERIAL_BEDROCK: u8 = 0;
pub(crate) const MATERIAL_GRASS: u8 = 1;
pub(crate) const MATERIAL_ROCK: u8 = 2;
pub(crate) const MATERIAL_SHORE: u8 = 3;

/// Everything below this height is bedrock, so no hole can open into the void.
pub(crate) const BEDROCK_TOP_Y: i32 = 0;

pub(crate) fn surface_material(surface_y: i32) -> u8 {
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

/// Shared cascade behind `terrain_height`: built once, sampled forever.
static WORLD_TERRAIN: LazyLock<TerrainNoise> = LazyLock::new(TerrainNoise::default);

/// Terrain height in metres at world (x, z). For systems that need a height
/// probe without a voxel query (obstacle placement, tests). Cheap: samples
/// the shared cascade, no per-call setup.
pub fn terrain_height(x: f32, z: f32) -> f32 {
    WORLD_TERRAIN.height(x, z)
}
