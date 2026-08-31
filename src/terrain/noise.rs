//! Pure procedural terrain generation: the erosion-filtered height
//! function behind the world.
//!
//! The authoritative terrain is the height function `TerrainNoise::sample`
//! (world metres, y up). Everything else — tile meshes, boid grounding,
//! camera clearance, cursor rays, later edits and pathfinding — samples the
//! same function (through [`crate::terrain::HeightField`]) so the world
//! stays self-consistent no matter which subsystem asks.
//!
//! Heights are Rune Skovbo Johansen's "Advanced Terrain Erosion Filter"
//! (blog.runevision.com, March 2026) evaluated per sample: an IQ
//! gradient-noise fBm base plus our fastnoise ridged mountains feed the
//! filter, which carves branching gullies along the downhill direction and
//! reports a ridge map used for tile coloring. The filter is analytic — no
//! neighbour reads, no simulation — so any point can be sampled alone, the
//! same contract the old pure-fBm cascade had. We use the pure-Rust port in
//! `bevy_erosion_filter::cpu` (MPL-2.0, mirrors the author's WGSL).
//!
//! Units: the filter works in "relief units" over a coordinate space of
//! `world metres / feature_size_m`, so carve depth is independent of the
//! amplitude sliders and every threshold (`onset`, `assumed_slope`) keeps
//! its reference-demo meaning. Slopes chain-rule between the two spaces.
//!
//! All parameters live in [`TerrainTuning`], a runtime resource: the F3
//! panel mutates it, and the change-detection cascade (rebuild
//! `HeightField` -> rebuild tiles -> reset caches) regenerates the world
//! live. See also `tiles::LEVEL_CELL_M` — wavelengths below ~2x a level's
//! cell size shimmer at that LOD.

use bevy::math::{Vec2, Vec3, Vec4};
use bevy::prelude::Resource;
use bevy_erosion_filter::cpu::{ErosionFilterParams, erosion_filter, fbm};
use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

/// Runtime-tunable world generation parameters.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct TerrainTuning {
    /// Changes every layer at once: the "new world" knob.
    pub seed: i32,
    /// The rolling hills the landscape is built on: IQ gradient-noise fBm
    /// with analytic derivatives, the substrate the gullies carve into.
    pub base_amplitude_m: f32,
    pub base_wavelength_m: f32,
    pub base_octaves: i32,
    /// Sharp crest lines (`1 - |n|` noise) on top of the base.
    pub ridge_max_m: f32,
    pub ridge_wavelength_m: f32,
    /// Slow noise in [0, 1] gating where mountains rise — clusters peaks
    /// into ranges with real gaps instead of uniform spikes.
    pub mask_wavelength_m: f32,
    /// Domain warp: mountain coordinates are pushed up to this far by a
    /// slow swirl, which makes ridgelines meander. (The fBm base is
    /// unwarped so its analytic gradient stays exact.)
    pub warp_amplitude_m: f32,
    pub warp_wavelength_m: f32,
    /// Gully erosion filter parameters.
    pub erosion: ErosionTuning,
}

/// Erosion filter knobs, 1:1 with `ErosionFilterParams` in
/// `bevy_erosion_filter::cpu` (defaults are its reference look, which is
/// the blog post's shadertoy). Fixed-size shader vectors are arrays here
/// so every component binds to a slider directly.
#[derive(Clone, Debug, PartialEq)]
pub struct ErosionTuning {
    /// Horizontal size of the gully pattern in filter units: bigger =
    /// wider, smoother gullies.
    pub scale: f32,
    /// How deep the filter carves per octave; 0 disables erosion entirely.
    pub strength: f32,
    /// Visible gully magnitude inside the steepness band.
    pub gully_weight: f32,
    /// How far high octaves spread beyond steep slopes.
    pub detail: f32,
    /// Stripe-cell density: smaller = grainier, larger = curvier gullies.
    pub cell_scale: f32,
    /// Partial normalization of the stripe wave (tames cancellation
    /// spikes where waves cancel).
    pub normalization: f32,
    pub octaves: i32,
    pub lacunarity: f32,
    pub gain: f32,
    /// Edge rounding of the carved profile: [ridge, ridge/k, crease,
    /// crease/k].
    pub rounding: [f32; 4],
    /// Steepness thresholds: [strength onset, strength onset 2, ridge-map
    /// mask, ridge-map fade].
    pub onset: [f32; 4],
    /// Pretend input slope for gully directions: [magnitude, blend].
    pub assumed_slope: [f32; 2],
    /// Feature size of the filter's coordinate space: world points are
    /// fed to the filter as (x, z) divided by this many metres.
    pub feature_size_m: f32,
    /// Fade-target range: heights are divided by this fraction of total
    /// relief before clamping to [-1, 1] (peaks bias to +1, valleys to
    /// -1).
    pub fade_fraction: f32,
}

impl Default for TerrainTuning {
    fn default() -> Self {
        TerrainTuning {
            seed: 1337,
            warp_amplitude_m: 140.0,
            warp_wavelength_m: 900.0,
            base_amplitude_m: 12.0,
            base_wavelength_m: 420.0,
            base_octaves: 3,
            ridge_max_m: 34.0,
            ridge_wavelength_m: 780.0,
            mask_wavelength_m: 1300.0,
            erosion: ErosionTuning::default(),
        }
    }
}

impl Default for ErosionTuning {
    fn default() -> Self {
        ErosionTuning {
            scale: 0.15,
            strength: 0.22,
            gully_weight: 0.5,
            detail: 1.5,
            cell_scale: 0.7,
            normalization: 0.5,
            octaves: 5,
            lacunarity: 2.0,
            gain: 0.5,
            rounding: [0.1, 0.0, 0.1, 2.0],
            onset: [1.25, 1.25, 2.8, 1.5],
            assumed_slope: [0.7, 1.0],
            // Gully wavelength = feature_size * scale * cell_scale ≈ 16 m
            // at the port's defaults — fine enough to read as dendritic
            // drainage, coarse enough to survive our 1 m near cells.
            feature_size_m: 150.0,
            fade_fraction: 0.6,
        }
    }
}

/// fBm character consts — deliberately not tuning sliders: the shapes the
/// amplitude/wavelength/octave sliders control are the tuning surface.
const BASE_LACUNARITY: f32 = 2.0;
const BASE_GAIN: f32 = 0.5;
/// Forward-difference step for the mountains' gradient. The layer is
/// smooth (wavelengths >= 56 m), so 2 m is far from its detail limit, and
/// gully directions don't need better.
const MOUNTAIN_GRADIENT_EPS_M: f32 = 2.0;

/// One terrain evaluation: the height plus the shading fields the filter
/// produces on the way. Slope is d(height)/d(x, z) in metres per metre,
/// so `Vec3::new(-slope.x, 1.0, -slope.y)` is the surface normal.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TerrainSample {
    pub height: f32,
    pub slope: Vec2,
    /// ~ +1 on ridges, ~ -1 in gully creases, ~ 0 on flats. Drives
    /// drainage-streak coloring in the tile meshes.
    pub ridge_map: f32,
}

/// The configured terrain function. Construct once and share — the
/// fastnoise layers are plain data, and the erosion parameters are copied
/// out of the tuning at build time.
pub struct TerrainNoise {
    warp: FastNoiseLite,
    ridge: FastNoiseLite,
    mask: FastNoiseLite,
    base_amplitude_m: f32,
    /// Base-fBm frequency in filter units (p-space), derived from
    /// `base_wavelength_m` so the slider keeps its metre meaning.
    base_frequency: f32,
    base_octaves: i32,
    ridge_max_m: f32,
    erosion: ErosionTuning,
    /// Domain offset for the filter: the IQ hash behind it is unseeded, so
    /// seeds sample unrelated regions by jumping far in p-space.
    offset: Vec2,
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
            ridge,
            mask: fbm(tuning.seed, tuning.mask_wavelength_m, 3),
            base_amplitude_m: tuning.base_amplitude_m,
            base_frequency: tuning.erosion.feature_size_m / tuning.base_wavelength_m,
            base_octaves: tuning.base_octaves,
            ridge_max_m: tuning.ridge_max_m,
            erosion: tuning.erosion.clone(),
            offset: seed_offset(tuning.seed),
        }
    }

    /// Terrain height plus the slope and ridge map at world (x, z), in
    /// one evaluation. Deterministic for a seed.
    pub fn sample(&self, x: f32, z: f32) -> TerrainSample {
        let e = &self.erosion;
        let p = Vec2::new(x, z) / e.feature_size_m + self.offset;
        // Total relief scales the filter's height units, so the carve
        // depth is amplitude-independent (the demo normalizes the same way).
        let relief_m = (self.base_amplitude_m + self.ridge_max_m).max(1.0);

        let raw = fbm(
            p,
            self.base_frequency,
            self.base_octaves,
            BASE_LACUNARITY,
            BASE_GAIN,
        );
        let h_mount = self.mountains(x, z);
        // Forward difference: mountains are smooth and low-frequency, so
        // this is plenty for gully direction and keeps the sample cost at
        // two extra mountain evals instead of four.
        let d_mount = Vec2::new(
            (self.mountains(x + MOUNTAIN_GRADIENT_EPS_M, z) - h_mount) / MOUNTAIN_GRADIENT_EPS_M,
            (self.mountains(x, z + MOUNTAIN_GRADIENT_EPS_M) - h_mount) / MOUNTAIN_GRADIENT_EPS_M,
        );

        let h_m = raw.x * self.base_amplitude_m + h_mount;
        // Into filter units: height in relief fractions, slope in h-units
        // per p-unit (chain rule across `p = metres / feature_size_m`).
        let slope_u =
            (Vec2::new(raw.y, raw.z) * self.base_amplitude_m + d_mount * e.feature_size_m)
                / relief_m;
        let input = Vec3::new(h_m / relief_m, slope_u.x, slope_u.y);

        // Peaks bias the fade toward ridge (+1), valleys toward crease
        // (-1), matching the demo's `h / (amp * 0.6)` normalization.
        let fade_target = (input.x / e.fade_fraction).clamp(-1.0, 1.0);
        let params = ErosionFilterParams {
            scale: e.scale,
            strength: e.strength,
            gully_weight: e.gully_weight,
            detail: e.detail,
            rounding: Vec4::from_array(e.rounding),
            onset: Vec4::from_array(e.onset),
            assumed_slope: Vec2::from_array(e.assumed_slope),
            cell_scale: e.cell_scale,
            normalization: e.normalization,
            octaves: e.octaves,
            lacunarity: e.lacunarity,
            gain: e.gain,
        };
        let out = erosion_filter(p, input, fade_target, &params);

        TerrainSample {
            height: (input.x + out.delta.x) * relief_m,
            slope: (Vec2::new(input.y, input.z) + Vec2::new(out.delta.y, out.delta.z))
                * relief_m
                / e.feature_size_m,
            ridge_map: out.ridge_map,
        }
    }

    /// The fastnoise mountain layer in metres: ridged crests gated by the
    /// range mask, on warped coordinates.
    fn mountains(&self, x: f32, z: f32) -> f32 {
        let (wx, wz) = self.warp.domain_warp_2d(x, z);
        // Ridged noise peaks where the noise crosses zero: |n| gives sharp
        // ridge lines, the mask confines them to mountain ranges.
        let ridge = 1.0 - self.ridge.get_noise_2d(wx, wz).abs();
        let mask = (self.mask.get_noise_2d(x, z) * 0.5 + 0.5).clamp(0.0, 1.0);
        ridge * mask * self.ridge_max_m
    }
}

impl Default for TerrainNoise {
    fn default() -> Self {
        TerrainNoise::from_tuning(&TerrainTuning::default())
    }
}

/// The IQ hash behind the filter has no seed input; unrelated seeds must
/// sample unrelated regions, so map the seed onto a large domain offset
/// (±4 k p-units ≈ ±600 km at the default feature size) with a cheap
/// integer hash — bit-exact on every platform.
fn seed_offset(seed: i32) -> Vec2 {
    let mut x = seed as u32 ^ 0x9e37_79b9;
    let mut h = || {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^= x >> 16;
        (x >> 8) as f32 / 16_777_216.0 - 0.5
    };
    let ox = h();
    let oz = h();
    Vec2::new(ox * 8192.0, oz * 8192.0)
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
                assert_eq!(a.sample(x, z).height, b.sample(x, z).height);
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
            a.sample(123.0, 456.0).height,
            b.sample(123.0, 456.0).height,
            "a seed bump must move terrain somewhere"
        );
    }

    #[test]
    fn zeroed_amplitudes_and_no_erosion_give_a_flat_zero_world() {
        let mut tuning = TerrainTuning::default();
        tuning.base_amplitude_m = 0.0;
        tuning.ridge_max_m = 0.0;
        tuning.erosion.strength = 0.0;
        let noise = TerrainNoise::from_tuning(&tuning);
        for x in [0.0, 55.5, -1234.0] {
            for z in [0.0, -77.7, 4000.0] {
                assert_eq!(noise.sample(x, z).height, 0.0, "flat at ({x}, {z})");
            }
        }
    }

    #[test]
    fn zero_strength_reproduces_the_unfiltered_input() {
        // The filter only ever adds `faded * strength` to the height, so
        // strength = 0 must reduce to the raw fBm + mountains composition
        // (up to the one relief-normalize round-trip) — the kill switch
        // for comparing eroded and raw looks.
        let mut tuning = TerrainTuning::default();
        tuning.erosion.strength = 0.0;
        let noise = TerrainNoise::from_tuning(&tuning);
        let relief_m = (tuning.base_amplitude_m + tuning.ridge_max_m).max(1.0);
        for x in [0.0, 137.2, -9_600.0] {
            for z in [0.0, -515.0, 32_000.0] {
                let p = Vec2::new(x, z) / tuning.erosion.feature_size_m + noise.offset;
                let raw = fbm(
                    p,
                    noise.base_frequency,
                    noise.base_octaves,
                    BASE_LACUNARITY,
                    BASE_GAIN,
                );
                let expected = raw.x * tuning.base_amplitude_m + noise.mountains(x, z);
                assert!(
                    (noise.sample(x, z).height - expected).abs() < 1e-3 * relief_m,
                    "strength = 0 drifted from the raw composition at ({x}, {z})"
                );
            }
        }
    }

    #[test]
    fn sample_fields_stay_in_range_and_finite() {
        let noise = TerrainNoise::default();
        for x in [-5000.0, 0.0, 12.5, 3400.0] {
            for z in [-5000.0, 1.25, 6800.0] {
                let s = noise.sample(x, z);
                assert!(s.height.is_finite());
                assert!(s.slope.x.is_finite() && s.slope.y.is_finite());
                assert!(
                    (-1.5..=1.5).contains(&s.ridge_map),
                    "ridge map out of range at ({x}, {z}): {}",
                    s.ridge_map
                );
            }
        }
    }

    #[test]
    fn erosion_actually_carves() {
        // Gullies redistribute height: with erosion on, at least some
        // sampled points must differ from the unfiltered input.
        let eroded = TerrainNoise::default();
        let mut tuning = TerrainTuning::default();
        tuning.erosion.strength = 0.0;
        let raw = TerrainNoise::from_tuning(&tuning);
        let mut differs = 0;
        for x in 0..40 {
            for z in 0..40 {
                let (x, z) = (x as f32 * 37.0, z as f32 * 41.0);
                if (eroded.sample(x, z).height - raw.sample(x, z).height).abs() > 1e-4 {
                    differs += 1;
                }
            }
        }
        assert!(differs > 400, "erosion moved only {differs}/1600 points");
    }

    #[test]
    fn analytic_slope_tracks_the_height_field() {
        // The reported slope (used for tile normals) must track the height
        // field's own finite difference: same direction, similar grade.
        // Exact equality is not the filter's contract — its slope state is
        // an analytic estimate of the carved surface, accurate to tens of
        // percent on gully flanks, which is fine for shading.
        let noise = TerrainNoise::default();
        for (x, z) in [(13.0, 71.0), (-420.0, 90.0), (1234.5, -777.25)] {
            let s = noise.sample(x, z);
            const H: f32 = 0.5;
            let fd = Vec2::new(
                (noise.sample(x + H, z).height - noise.sample(x - H, z).height) / (2.0 * H),
                (noise.sample(x, z + H).height - noise.sample(x, z - H).height) / (2.0 * H),
            );
            assert!(
                s.slope.dot(fd) > 0.0,
                "slope disagrees in direction with the height field at ({x}, {z}): {:?} vs {fd:?}",
                s.slope
            );
            let ratio = s.slope.length() / fd.length().max(1e-3);
            assert!(
                (0.25..=4.0).contains(&ratio),
                "slope magnitude off at ({x}, {z}): {:?} vs {fd:?} (ratio {ratio})",
                s.slope
            );
        }
    }
}
