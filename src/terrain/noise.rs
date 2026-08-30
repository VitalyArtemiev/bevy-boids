//! Pure procedural terrain generation: the fBm cascade behind the world.
//!
//! The authoritative terrain is the height function `TerrainNoise::height`
//! (world metres, y up). Everything else — tile meshes, boid grounding,
//! camera clearance, cursor rays, later edits and pathfinding — samples the
//! same function (through [`crate::terrain::HeightField`]) so the world
//! stays self-consistent no matter which subsystem asks.

use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

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

/// The configured noise cascade. Construct once and share — `FastNoiseLite`
/// is plain data, cheap to clone around but pointless to rebuild per sample.
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
