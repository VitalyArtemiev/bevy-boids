//! Pure procedural terrain generation: the fBm cascade behind the world.
//!
//! The authoritative terrain is the height function `TerrainNoise::height`
//! (world metres, y up). Everything else — tile meshes, boid grounding,
//! camera clearance, cursor rays, later edits and pathfinding — samples the
//! same function (through [`crate::terrain::HeightField`]) so the world
//! stays self-consistent no matter which subsystem asks.
//!
//! All parameters live in [`TerrainTuning`], a runtime resource: the debug
//! panel mutates it, and the change-detection cascade (rebuild
//! `HeightField` -> rebuild tiles -> reset caches) regenerates the world
//! live. See also `tiles::LEVEL_CELL_M` — wavelengths below ~2x a level's
//! cell size shimmer at that LOD.

use bevy::prelude::Resource;
use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

/// Runtime-tunable world generation parameters. Defaults reproduce the
/// original hand-tuned constants.
#[derive(Resource, Clone, Debug)]
pub struct TerrainTuning {
    /// Changes every layer at once: the "new world" knob.
    pub seed: i32,
    /// Domain warp: input coordinates are pushed up to this far by a slow
    /// swirl, which makes valleys and ridgelines meander.
    pub warp_amplitude_m: f32,
    pub warp_wavelength_m: f32,
    /// The rolling hills the landscape is built on.
    pub base_amplitude_m: f32,
    pub base_wavelength_m: f32,
    /// Sharp crest lines (`1 - |n|` noise) on top of the base.
    pub ridge_max_m: f32,
    pub ridge_wavelength_m: f32,
    /// Slow noise in [0, 1] gating where mountains rise — clusters peaks
    /// into ranges with real gaps instead of uniform spikes.
    pub mask_wavelength_m: f32,
    /// Small-scale surface texture.
    pub detail_amplitude_m: f32,
    pub detail_wavelength_m: f32,
}

impl Default for TerrainTuning {
    fn default() -> Self {
        TerrainTuning {
            seed: 1337,
            warp_amplitude_m: 140.0,
            warp_wavelength_m: 900.0,
            base_amplitude_m: 12.0,
            base_wavelength_m: 420.0,
            ridge_max_m: 34.0,
            ridge_wavelength_m: 780.0,
            mask_wavelength_m: 1300.0,
            detail_amplitude_m: 1.4,
            detail_wavelength_m: 56.0,
        }
    }
}

/// The configured noise cascade. Construct once and share — `FastNoiseLite`
/// is plain data, cheap to clone around but pointless to rebuild per sample.
/// Octave counts are fixed (smaller layers get fewer octaves); amplitudes
/// and wavelengths are the tuning surface.
pub struct TerrainNoise {
    warp: FastNoiseLite,
    base: FastNoiseLite,
    ridge: FastNoiseLite,
    mask: FastNoiseLite,
    detail: FastNoiseLite,
    ridge_max_m: f32,
    base_amplitude_m: f32,
    detail_amplitude_m: f32,
}

impl TerrainNoise {
    pub fn from_tuning(tuning: &TerrainTuning) -> Self {
        let mut warp = FastNoiseLite::new();
        warp.set_seed(Some(tuning.seed));
        warp.set_noise_type(Some(NoiseType::OpenSimplex2));
        warp.set_frequency(Some(1.0 / tuning.warp_wavelength_m));
        warp.set_domain_warp_amp(Some(tuning.warp_amplitude_m));

        let fbm = |seed: i32, wavelength: f32, octaves: i32| {
            let mut n = FastNoiseLite::new();
            n.set_seed(Some(seed));
            n.set_noise_type(Some(NoiseType::OpenSimplex2));
            n.set_fractal_type(Some(FractalType::FBm));
            n.set_fractal_octaves(Some(octaves));
            n.set_frequency(Some(1.0 / wavelength));
            n
        };

        let mut ridge = fbm(tuning.seed, tuning.ridge_wavelength_m, 4);
        ridge.set_fractal_type(Some(FractalType::Ridged));

        TerrainNoise {
            warp,
            base: fbm(tuning.seed, tuning.base_wavelength_m, 4),
            ridge,
            mask: fbm(tuning.seed, tuning.mask_wavelength_m, 3),
            detail: fbm(tuning.seed, tuning.detail_wavelength_m, 3),
            ridge_max_m: tuning.ridge_max_m,
            base_amplitude_m: tuning.base_amplitude_m,
            detail_amplitude_m: tuning.detail_amplitude_m,
        }
    }

    /// Terrain height in world metres at (x, z). Deterministic for a seed.
    pub fn height(&self, x: f32, z: f32) -> f32 {
        let (wx, wz) = self.warp.domain_warp_2d(x, z);
        let base = self.base.get_noise_2d(wx, wz) * self.base_amplitude_m;
        // Ridged noise peaks where the noise crosses zero: |n| gives sharp
        // ridge lines, the mask confines them to mountain ranges.
        let ridge = 1.0 - self.ridge.get_noise_2d(wx, wz).abs();
        let mask = (self.mask.get_noise_2d(x, z) * 0.5 + 0.5).clamp(0.0, 1.0);
        let detail = self.detail.get_noise_2d(x, z) * self.detail_amplitude_m;
        base + ridge * mask * self.ridge_max_m + detail
    }
}

impl Default for TerrainNoise {
    fn default() -> Self {
        TerrainNoise::from_tuning(&TerrainTuning::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_tuning_is_deterministic() {
        let a = TerrainNoise::from_tuning(&TerrainTuning::default());
        let b = TerrainNoise::from_tuning(&TerrainTuning::default());
        for x in [-1000.0, 0.0, 17.3, 90_000.0] {
            for z in [-400.0, 3.1, 60_000.0] {
                assert_eq!(a.height(x, z), b.height(x, z));
            }
        }
    }

    #[test]
    fn different_seed_changes_the_world() {
        let mut tuning = TerrainTuning::default();
        let a = TerrainNoise::from_tuning(&tuning);
        tuning.seed += 1;
        let b = TerrainNoise::from_tuning(&tuning);
        assert_ne!(
            a.height(123.0, 456.0),
            b.height(123.0, 456.0),
            "a seed bump must move terrain somewhere"
        );
    }

    #[test]
    fn zeroed_amplitudes_give_a_flat_zero_world() {
        let mut tuning = TerrainTuning::default();
        tuning.base_amplitude_m = 0.0;
        tuning.ridge_max_m = 0.0;
        tuning.detail_amplitude_m = 0.0;
        let noise = TerrainNoise::from_tuning(&tuning);
        for x in [0.0, 55.5, -1234.0] {
            for z in [0.0, -77.7, 4000.0] {
                assert_eq!(noise.height(x, z), 0.0, "flat at ({x}, {z})");
            }
        }
    }
}
