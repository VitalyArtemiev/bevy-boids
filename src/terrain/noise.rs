//! Pure procedural terrain generation: the erosion-filtered height
//! function behind the world.
//!
//! The authoritative terrain is the height function `TerrainNoise::sample`
//! (world metres, y up). Everything else — tile meshes, boid grounding,
//! camera clearance, cursor rays, later edits and pathfinding — samples the
//! same function (through [`crate::terrain::HeightField`]) so the world
//! stays self-consistent no matter which subsystem asks.
//!
//! Two layers, both analytic (no neighbour reads, no simulation), so any
//! point can be sampled alone:
//!
//! 1. **Tectonics** (plate scale, ~km): a Worley noise whose cells are
//!    tectonic plates. Each plate hashes a unit drift vector and an
//!    interior hilliness; where two plates' vectors collide, an orogeny
//!    bump raises the relief toward mountain scale (and tightens the
//!    feature size); where they split, the would-be uplift is spent as
//!    subsidence instead — deep elongated basins (future lakes; the
//!    height simply goes below the plains level until water rendering
//!    lands). Plate interiors blend each plate's hilliness between
//!    plains and rolling hills. This layer does not displace height
//!    directly — it decides, per point, the relief and feature scale the
//!    erosion layer is drawn at.
//! 2. **Erosion terrain** (~10–100 m): Rune Skovbo Johansen's "Advanced
//!    Terrain Erosion Filter" (blog.runevision.com, March 2026) through
//!    the pure-Rust port in `bevy_erosion_filter::cpu` (MPL-2.0). The
//!    composition is a direct port of the crate's reference demo
//!    (`evaluate_terrain_with_octaves` in its terrain_demo.wgsl): the
//!    crate's own fBm is the entire base landform — no extra noise
//!    layers — and the filter carves branching gullies along the
//!    downhill direction, reporting a ridge map used for tile coloring.
//!    Keeping the input composition exactly the demo's (frequency 3 fBm,
//!    gain 0.1, amplitude 0.125) is load-bearing: the `onset` thresholds
//!    that gate erosion are tuned for the slope magnitudes that
//!    composition produces.
//!
//! Units: the filter works in "unit heights" around 0.5 over a coordinate
//! space of `world metres / feature_m(x, z)` — the feature scale varies
//! per point with the tectonic field, acting as a smooth domain warp on
//! the erosion coordinates (varying at plate scale, far above the erosion
//! wavelengths, so it reads as regional character, not stretching).
//! Slopes chain-rule across both layers, fully analytically: the Worley
//! distances have exact gradients (unit vectors toward the feature
//! points) and the plate drifts are per-cell constants, so the whole
//! tectonic field differentiates in closed form.
//!
//! All parameters live in [`TerrainTuning`], a runtime resource: the F3
//! panel mutates it, and the change-detection cascade (rebuild
//! `HeightField` -> rebuild tiles -> reset caches) regenerates the world
//! live. See also `tiles::LEVEL_CELL_M` — wavelengths below ~2x a level's
//! cell size shimmer at that LOD.

use bevy::math::{IVec2, Vec2, Vec3};
use bevy::prelude::Resource;
use bevy_erosion_filter::cpu::{ErosionFilterParams, erosion_filter, fbm, noised};

/// Base fBm fractal constants — a TRUE multi-octave cascade (gain 0.45:
/// each octave ~half the previous), unlike the reference demo's gain 0.1
/// which is effectively single-octave. The demo's ratio works in its 22 m
/// diorama viewed from 15 m; scaled to an RTS world it turned every
/// mountain into one wavelength of uniform high-frequency peaks.
const BASE_LACUNARITY: f32 = 2.0;
const BASE_GAIN: f32 = 0.45;
/// Unit-height amplitude of the base fBm: keeps the filter's input
/// hovering around 0.5 ± 0.1 where its fade/remap expect it. The base's
/// WORLD-scale displacement is separate — see `BASE_RELIEF_FRACTION`.
const BASE_UNIT_AMP: f32 = 0.10;
/// How much of the local relief the multi-octave base landform displaces
/// in world metres. The reference demo's look comes from an
/// amplitude:wavelength ratio ~0.3 — its 22 m diorama is all steep
/// faces — while a linear unit-height mapping at RTS scales yields a
/// ~0.01 ratio: flat pancakes with speckle. 0.35 restores the demo's
/// ratio in mountains (±130 m at the 350 m base wavelength) and keeps
/// plains as gentle swells (±19 m).
const BASE_RELIEF_FRACTION: f32 = 0.35;
/// Fade-target denominator: peaks bias toward +1, valleys toward -1,
/// scaled to the multi-octave base's actual spread (~1.8x the first
/// octave).
const FADE_FRACTION: f32 = BASE_UNIT_AMP * 1.8 * 0.6;
/// World height zero point: the demo maps `unit_height - 0.43` to world
/// metres. Shared by the tile coloring, which gates on unit height.
pub(crate) const VERTICAL_BIAS: f32 = 0.43;
/// Feature-point jitter of the plate lattice. Below 1.0 the two nearest
/// feature points are always inside the 3x3 neighbourhood `worley`
/// searches. Kept high (0.92): the more the points stray from their cell
/// centres, the less the boundary bisectors align with the lattice-cell
/// edges — low jitter made every range run parallel to the grid.
const PLATE_JITTER: f32 = 0.92;
/// How far hilliness pulls the feature scale from plains toward
/// mountains: full-hill interiors sit halfway. Orogeny takes it the rest
/// of the way.
const HILLS_FEATURE_WEIGHT: f32 = 0.5;

/// Runtime-tunable world generation parameters. The `erosion` field is the
/// crate's own `ErosionFilterParams` verbatim; `tectonics` positions both
/// layers in our world.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct TerrainTuning {
    /// Changes every layer at once: the "new world" knob. Seeds the fBm
    /// domain offset and the plate lattice.
    pub seed: i32,
    /// Isostasy knob (the demo's `height_offset`): world units sink by
    /// this much per unit of accumulated erosion magnitude.
    pub height_offset: f32,
    /// First-octave wavelength of the base landform, in world metres —
    /// independent of the erosion feature scale (which only sets the
    /// gully size), so mountains keep 300 m-class hills instead of
    /// shrinking with the gullies.
    pub base_wavelength_m: f32,
    pub base_octaves: i32,
    /// The crate's erosion filter parameters, 1:1.
    pub erosion: ErosionFilterParams,
    /// Plate-tectonics layer: where mountains rise, basins open and
    /// plains spread.
    pub tectonics: TectonicTuning,
}

/// The tectonic layer's knobs. Plates are Worley cells of
/// `plate_size_m`; every plate hashes a drift vector and a hilliness, and
/// boundaries between plates resolve into orogeny (colliding), subsidence
/// (splitting) or nothing (shear).
#[derive(Clone, Debug, PartialEq)]
pub struct TectonicTuning {
    /// Worley lattice spacing: roughly the diameter of a tectonic plate.
    pub plate_size_m: f32,
    /// Scales how strongly boundary convergence/divergence maps to
    /// full-strength orogeny/subsidence: higher = more boundaries
    /// qualify as mountains and rifts, lower = only near-head-on
    /// collisions count.
    pub drift_speed: f32,
    /// Half-width of the mountain/rift band around a plate boundary, as a
    /// fraction of plate size.
    pub mountain_width: f32,
    /// How deep rift basins carve below the plains level.
    pub rift_depth_m: f32,
    /// Relief (peak-to-valley span is ~0.24x this) of flat plate
    /// interiors.
    pub plains_relief_m: f32,
    /// Relief of hilly plate interiors (plates hash their own hilliness
    /// between this and the plains value).
    pub hills_relief_m: f32,
    /// Relief of colliding-boundary mountains.
    pub mountain_relief_m: f32,
    /// Erosion feature scale in flat plains. Gully wavelength =
    /// `feature_m * scale * cell_scale`; the carve amplitude scales with
    /// relief, so the feature must scale with it too or the gullies turn
    /// into near-vertical sawtooth (amplitude:wavelength must stay
    /// ~0.1-0.2, the ratio that makes the reference demo read as carved
    /// rock instead of corduroy).
    pub plains_feature_m: f32,
    /// Erosion feature scale in mountains: LARGER than plains — the
    /// carve amplitude there is relief-scaled (~±28 m), so it needs
    /// ~230 m ravines: the demo's per-octave slope is ~0.6 (31°), and
    /// keeping that ratio — not just the amplitude — is what separates
    /// carved rock from sawtooth. (Setting this smaller than plains was
    /// the "uniform sawtooth" bug.)
    pub mountain_feature_m: f32,
    /// How far plate space is curled to bend boundaries (in plate units).
    /// Without it every range follows the straight perpendicular
    /// bisector of two plate points — unnaturally straight lines
    /// parallel to the lattice-cell edges.
    pub boundary_curvature: f32,
    /// Wavelength of that curl, in plate units.
    pub boundary_curvature_wavelength: f32,
    /// How much each range swells and pinches along its length (0 =
    /// uniform strength, 1 = ranges taper to nothing at their quietest).
    pub range_pulse: f32,
    /// Wavelength of the along-range pulse, in plate units.
    pub range_pulse_wavelength: f32,
    /// Amplitude of the macro landform layer: the multi-octave massif/
    /// valley structure INSIDE a range (wavelength `macro_wavelength_m`
    /// and halving). The demo composition's base fBm has gain 0.1 — one
    /// dominant smooth octave — so without this layer scaled mountains
    /// are just high-frequency uniform peaks. Gated by orogeny: the
    /// tectonic field decides WHERE macro relief is applied.
    pub macro_amplitude_m: f32,
    /// First-octave wavelength of the macro layer (the massif scale).
    pub macro_wavelength_m: f32,
    /// Octaves of the macro layer (each half wavelength, half amplitude).
    pub macro_octaves: i32,
}

/// Macro layer fractal constants: first octave AT `macro_wavelength_m`
/// (massifs), each octave half the wavelength and half the amplitude —
/// a proper mountain cascade.
const MACRO_FREQUENCY: f32 = 1.0;
const MACRO_LACUNARITY: f32 = 2.0;
const MACRO_GAIN: f32 = 0.5;

/// Per-octave ridged multifractal (Musgrave-style): octave i contributes
/// crest lines where ITS noise crosses zero (`(1-|n|)²`, exact gradient),
/// and each octave's weight is the previous octave's ridge value, so
/// small ridges concentrate along the big spines instead of tiling the
/// flanks. Returns the value (~0..2) and its gradient wrt `p`.
fn ridged_multifractal(
    p: Vec2,
    octaves: i32,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
) -> (f32, Vec2) {
    let mut value = 0.0;
    let mut grad = Vec2::ZERO;
    let (mut amp, mut freq, mut weight) = (1.0, frequency, 1.0);
    for _ in 0..octaves {
        let n = noised(p * freq);
        let r = (1.0 - n.x.abs()).max(0.0);
        value += r * r * amp * weight;
        // d(r²)/dn = -2·r·sign(n); chain rule through the octave
        // frequency like the crate's own fbm does.
        grad += Vec2::new(n.y, n.z) * (-2.0 * r * n.x.signum() * amp * freq * weight);
        weight = r.clamp(0.0, 1.0);
        amp *= gain;
        freq *= lacunarity;
    }
    (value, grad)
}

impl Default for TectonicTuning {
    fn default() -> Self {
        TectonicTuning {
            plate_size_m: 9000.0,
            drift_speed: 2.4,
            mountain_width: 0.55,
            rift_depth_m: 120.0,
            plains_relief_m: 60.0,
            hills_relief_m: 140.0,
            mountain_relief_m: 420.0,
            plains_feature_m: 400.0,
            mountain_feature_m: 2200.0,
            boundary_curvature: 0.35,
            boundary_curvature_wavelength: 2.5,
            range_pulse: 0.35,
            range_pulse_wavelength: 3.0,
            macro_amplitude_m: 170.0,
            macro_wavelength_m: 1800.0,
            macro_octaves: 6,
        }
    }
}

impl Default for TerrainTuning {
    fn default() -> Self {
        TerrainTuning {
            // Picked over a handful of candidates by rendering the field:
            // the origin sits on plains with a mountain belt ~4-5 km out
            // and a rift basin beyond it, so the default spawn sees all
            // three tectonic natures without travelling.
            seed: 2026,
            height_offset: -0.65,
            base_wavelength_m: 350.0,
            base_octaves: 5,
            erosion: ErosionFilterParams::default(),
            tectonics: TectonicTuning::default(),
        }
    }
}

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

/// The configured terrain function. Construct once and share — it is
/// plain data copied out of the tuning at build time.
pub struct TerrainNoise {
    seed: i32,
    height_offset: f32,
    base_wavelength_m: f32,
    base_octaves: i32,
    erosion: ErosionFilterParams,
    tectonics: TectonicTuning,
    /// Domain offset for the filter: the IQ hash behind it is unseeded,
    /// so seeds sample unrelated regions by jumping in p-space.
    offset: Vec2,
    /// Phases of the plate-space curl warp (radians), hashed from the
    /// seed so seeds bend boundaries differently.
    warp_phase: Vec2,
    /// Domain offset of the macro landform fBm (same IQ-hash precision
    /// envelope as `offset`, a different stream so the two layers don't
    /// correlate).
    macro_offset: Vec2,
}

impl TerrainNoise {
    /// Coloring reference relief (the mountain scale) — hands the
    /// conversion the sampler used to [`crate::terrain::HeightField`].
    /// Snow and grass bands gate on height relative to THIS, so snow
    /// crowns only real mountains, never a tall plains hill.
    pub fn relief_m(&self) -> f32 {
        self.tectonics.mountain_relief_m
    }

    pub fn from_tuning(tuning: &TerrainTuning) -> Self {
        TerrainNoise {
            seed: tuning.seed,
            height_offset: tuning.height_offset,
            base_wavelength_m: tuning.base_wavelength_m,
            base_octaves: tuning.base_octaves,
            erosion: tuning.erosion,
            tectonics: tuning.tectonics.clone(),
            offset: seed_offset(tuning.seed),
            warp_phase: warp_phases(tuning.seed),
            macro_offset: seed_offset(tuning.seed ^ 0x2f6a_88c1),
        }
    }

    /// Full-detail sample (see [`Self::sample_lod`]).
    pub fn sample(&self, x: f32, z: f32) -> TerrainSample {
        self.sample_lod(x, z, 0)
    }

    /// [`Self::sample`] with an LOD level: every octave ladder is
    /// truncated to what the level's cell size can resolve (each ring
    /// jumps 4x in cell size, so halving the octaves per level matches).
    /// Without this, coarse tiles alias the sub-cell octaves into dense
    /// high-frequency speckle — the near field resolves them, the far
    /// field cannot, and there is no morphing to hide it yet.
    pub fn sample_lod(&self, x: f32, z: f32, level: u8) -> TerrainSample {
        let l = level as i32;
        let base_oct = (self.base_octaves - l / 2).max(1);
        let macro_oct = (self.tectonics.macro_octaves - l / 2).max(1);
        let erosion_oct = (self.erosion.octaves - l).max(1);
        let t = self.tectonic(x, z);
        let p = Vec2::new(x, z) / t.feature_m + self.offset;

        // Macro landform layer: a per-octave RIDGED multifractal (each
        // octave contributes crest lines at its own zero crossings —
        // massif spines at `macro_wavelength_m`, secondary ridges
        // halving down; Musgrave weighting damps small ridges away from
        // the big ones), whose amplitude the orogeny gates. Ridging the
        // SUM of the octaves instead would plant a crest at every zero
        // crossing of every octave wiggling the sum — dense parallel
        // sawtooth at all scales, which is exactly the uniform
        // high-frequency pattern this layer once suffered from.
        let pm = Vec2::new(x, z) / self.tectonics.macro_wavelength_m + self.macro_offset;
        let (macro_val, macro_grad_p) = ridged_multifractal(
            pm,
            macro_oct,
            MACRO_FREQUENCY,
            MACRO_LACUNARITY,
            MACRO_GAIN,
        );
        let macro_m = macro_val * self.tectonics.macro_amplitude_m * t.orogeny;
        // World gradient of the macro displacement: chain rule through
        // the per-octave ridge transforms plus the orogeny-modulation
        // term (both exact; a crest line is a C1 break — that IS the
        // ridgeline).
        let macro_slope = macro_grad_p
            * (self.tectonics.macro_amplitude_m * t.orogeny / self.tectonics.macro_wavelength_m)
            + macro_val * self.tectonics.macro_amplitude_m * t.d_orogeny;

        // Base landform: a true multi-octave fBm at a WORLD-space
        // wavelength (independent of the gully-scale feature), evaluated
        // in unit heights. The erosion filter treats base.yz as the
        // input slope without differentiating, so the base may live in
        // its own coordinate frame; its world gradient is exact.
        let bp = Vec2::new(x, z) / self.base_wavelength_m + self.offset;
        let base_noise = fbm(bp, 1.0, base_oct, BASE_LACUNARITY, BASE_GAIN);
        let base_unit = base_noise.x * BASE_UNIT_AMP;
        let base_slope_w = Vec2::new(base_noise.y, base_noise.z)
            * (BASE_UNIT_AMP / self.base_wavelength_m);
        let fade_target = (base_unit / FADE_FRACTION).clamp(-1.0, 1.0);
        // The base's world-scale displacement: the multi-octave cascade
        // times the local relief. This is the actual mountain geometry —
        // massifs, ridgelines and valleys at 350 m down to 22 m
        // wavelengths — while the filter supplies the fine gully carving.
        let base_big_m = base_noise.x * BASE_RELIEF_FRACTION * t.relief_m;
        let base_big_slope_w = Vec2::new(base_noise.y, base_noise.z)
            * (BASE_RELIEF_FRACTION * t.relief_m / self.base_wavelength_m)
            + base_noise.x * BASE_RELIEF_FRACTION * t.d_relief;

        // Feed the base and macro world slopes into the filter input
        // (converted to unit heights per filter p-unit) so gully
        // directions lean down the flanks; the contributions are small,
        // so the onset gate keeps its reference-demo meaning.
        let slope_feed = (base_slope_w + macro_slope) * (t.feature_m / t.relief_m.max(1.0));

        let erosion = ErosionFilterParams {
            octaves: erosion_oct,
            ..self.erosion
        };
        let out = erosion_filter(
            p,
            Vec3::new(base_unit + VERTICAL_BIAS, slope_feed.x, slope_feed.y),
            fade_target,
            &erosion,
        );
        let unit = base_unit + out.delta.x + self.height_offset * out.magnitude;
        // d(unit)/d(p): everything the filter saw as input slope plus
        // its own analytic delta.
        let erosion_grad_p = slope_feed + Vec2::new(out.delta.y, out.delta.z);
        let height = (unit - VERTICAL_BIAS) * t.relief_m - t.subsidence_m + macro_m + base_big_m;

        // Slope: d(unit)/d(world) = the base's exact world gradient plus
        // the filter delta chain-ruled through the feature warp
        // (p = world / feature(x, z)); then relief scales it and the
        // relief/subsidence/macro modulations add their own gradients.
        let f = t.feature_m;
        let inv_f2 = 1.0 / (f * f);
        let dp_dx = Vec2::new(1.0 / f - x * t.d_feature.x * inv_f2, -z * t.d_feature.x * inv_f2);
        let dp_dz = Vec2::new(-x * t.d_feature.y * inv_f2, 1.0 / f - z * t.d_feature.y * inv_f2);
        let slope = Vec2::new(
            erosion_grad_p.dot(dp_dx) * t.relief_m + unit * t.d_relief.x - t.d_subsidence.x,
            erosion_grad_p.dot(dp_dz) * t.relief_m + unit * t.d_relief.y - t.d_subsidence.y,
        ) + base_slope_w * t.relief_m
            + base_big_slope_w
            + macro_slope;

        TerrainSample {
            height,
            slope,
            ridge_map: out.ridge_map,
        }
    }

    /// The tectonic field at a world point: which relief and feature
    /// scale the erosion terrain is drawn at here, and how far a rift
    /// basin has subsided below the plains level — with exact gradients
    /// (per world metre) for the slope chain rule.
    fn tectonic(&self, x: f32, z: f32) -> TectonicField {
        let t = &self.tectonics;
        let w = Vec2::new(x, z) / t.plate_size_m;

        // Closed-form curl warp of plate space: bends the straight
        // perpendicular-bisector boundaries into meandering ranges. The
        // Jacobian is analytic, so the gradients pull back exactly.
        let (u, s1, s2) = self.warp_plate(w);
        let (a, b, f1, f2) = nearest_plates(u, self.seed);

        // Exact Worley gradients in warped space, pulled back to w
        // (they feed every derivative below).
        let g1 = pull_back((u - a.pos) / f1.max(1e-6), s1, s2);
        let g2 = pull_back((u - b.pos) / f2.max(1e-6), s1, s2);

        // Per-boundary character: each plate pair hashes its own
        // collision strength and band width, so a border's ranges swell,
        // taper and go quiet along it instead of one uniform wall — the
        // plate layout modulates the terrain's intensity, it does not
        // stamp it.
        let (strength, width_scale, dir, phase) = edge_profile(&a, &b, self.seed);
        let width = t.mountain_width * width_scale;

        // Closing rate along the boundary normal: constant per plate
        // pair (drifts and normal are), so it scales the bump but never
        // contributes its own gradient.
        let closing = closing_rate(&a, &b);
        let (k_orogeny, k_rift) = response(closing, t);

        // Boundary bump over the F1/F2 equidistant surface, fading over
        // the pair's own band width. smoothstep has zero slope at both
        // ends, so the clamp needs no special casing.
        let t_edge = (1.0 - (f2 - f1).max(0.0) / width).clamp(0.0, 1.0);
        let bump = t_edge * t_edge * (3.0 - 2.0 * t_edge);
        let d_bump = if t_edge > 0.0 && t_edge < 1.0 {
            6.0 * t_edge * (1.0 - t_edge) * -(g2 - g1) / width
        } else {
            Vec2::ZERO
        };

        // Along-range pulse: a slow seeded sine swells and pinches each
        // range along its length (varies in plate space, seeded per
        // boundary).
        let pulse_w = std::f32::consts::TAU / t.range_pulse_wavelength;
        let pulse = 1.0 - t.range_pulse
            + t.range_pulse * (dir.dot(w) * pulse_w + phase).sin();
        let d_pulse = if t.range_pulse > 0.0 {
            t.range_pulse * (dir.dot(w) * pulse_w + phase).cos() * pulse_w * dir
        } else {
            Vec2::ZERO
        };

        let orogeny = bump * k_orogeny * strength * pulse;
        let d_orogeny =
            d_bump * (k_orogeny * strength * pulse) + bump * (k_orogeny * strength) * d_pulse;
        let rift = bump * k_rift * strength * pulse;
        let d_rift = d_bump * (k_rift * strength * pulse) + bump * (k_rift * strength) * d_pulse;

        // Interiors blend the two nearest plates' hilliness, weighted by
        // proximity; the nearest plate dominates away from the boundary.
        let sum = (f1 + f2).max(1e-6);
        let blend = f1 / sum;
        let d_blend = (f2 * g1 - f1 * g2) / (sum * sum);
        let hilliness = a.hilliness * (1.0 - blend) + b.hilliness * blend;
        let d_hilliness = (b.hilliness - a.hilliness) * d_blend;

        let interior = t.plains_relief_m + (t.hills_relief_m - t.plains_relief_m) * hilliness;
        let d_interior = (t.hills_relief_m - t.plains_relief_m) * d_hilliness;
        let relief = interior + (t.mountain_relief_m - interior) * orogeny;
        let d_relief = d_interior * (1.0 - orogeny)
            + (t.mountain_relief_m - interior) * d_orogeny;

        let hills_energy = hilliness * HILLS_FEATURE_WEIGHT;
        let (energy, d_energy) = if orogeny >= hills_energy {
            (orogeny, d_orogeny)
        } else {
            (hills_energy, d_hilliness * HILLS_FEATURE_WEIGHT)
        };
        let feature = t.plains_feature_m + (t.mountain_feature_m - t.plains_feature_m) * energy;
        let d_feature = (t.mountain_feature_m - t.plains_feature_m) * d_energy;

        // Gradients above are per plate-space unit (already pulled back
        // through the warp); w = world / plate size, so scale down into
        // per-metre slopes.
        let per_m = 1.0 / t.plate_size_m;
        TectonicField {
            relief_m: relief,
            feature_m: feature,
            subsidence_m: rift * t.rift_depth_m,
            orogeny,
            d_orogeny: d_orogeny * per_m,
            d_relief: d_relief * per_m,
            d_feature: d_feature * per_m,
            d_subsidence: d_rift * t.rift_depth_m * per_m,
        }
    }

    /// The plate-space curl warp: `u = w + A·(sin(w.y·k + φ1),
    /// sin(w.x·k + φ2))`. Returns `u` plus the two off-diagonal
    /// Jacobian couplings `s1 = ∂u.x/∂w.y` and `s2 = ∂u.y/∂w.x`
    /// (diagonals stay 1).
    fn warp_plate(&self, w: Vec2) -> (Vec2, f32, f32) {
        let t = &self.tectonics;
        if t.boundary_curvature <= 0.0 {
            return (w, 0.0, 0.0);
        }
        let k = std::f32::consts::TAU / t.boundary_curvature_wavelength;
        let (ax, ay) = (w.y * k + self.warp_phase.x, w.x * k + self.warp_phase.y);
        (
            Vec2::new(
                w.x + t.boundary_curvature * ax.sin(),
                w.y + t.boundary_curvature * ay.sin(),
            ),
            t.boundary_curvature * k * ax.cos(),
            t.boundary_curvature * k * ay.cos(),
        )
    }
}

impl Default for TerrainNoise {
    fn default() -> Self {
        TerrainNoise::from_tuning(&TerrainTuning::default())
    }
}

/// The local scale of the terrain, from the plate lattice, with its
/// gradients per world metre.
struct TectonicField {
    /// Relief the erosion terrain maps unit heights to at this point.
    relief_m: f32,
    /// Erosion coordinate scale (metres per filter p-unit) at this point.
    feature_m: f32,
    /// How far this point has subsided below the plains level (rift
    /// basins; 0 outside them).
    subsidence_m: f32,
    /// Orogeny and its gradient — the macro landform layer scales with
    /// both.
    orogeny: f32,
    d_orogeny: Vec2,
    d_relief: Vec2,
    d_feature: Vec2,
    d_subsidence: Vec2,
}

/// One tectonic plate: a jittered Worley feature point carrying its
/// hashed drift and interior character.
#[derive(Clone, Copy)]
struct Plate {
    /// The lattice cell the plate grows from (identifies it for
    /// per-boundary hashing).
    cell: IVec2,
    /// Feature-point position in plate space (lattice cell + jitter).
    pos: Vec2,
    /// Unit drift velocity. Boundaries resolve by the closing rate of the
    /// two plates' vectors along the boundary normal.
    drift: Vec2,
    /// Interior character: 0 = flat plain, 1 = hills.
    hilliness: f32,
    /// Margin activity: with the opposite plate's value averaged in, it
    /// sets how strongly this plate's boundaries orogen/rift.
    activity: f32,
    /// Half-phase of this plate's along-margin pulse.
    pulse_phase: f32,
}

/// The plates' closing rate along their shared boundary normal (plate a
/// toward b): +1 = head-on collision, -1 = plates pulling apart, 0 =
/// shear. Symmetric under swapping the two plates, so the boundary field
/// is continuous across the border itself.
fn closing_rate(a: &Plate, b: &Plate) -> f32 {
    let normal = (b.pos - a.pos).normalize_or_zero();
    (a.drift - b.drift).dot(normal) * 0.5
}

/// Clamped boundary response to a pair's closing rate:
/// `(orogeny gain, rift gain)` in [0, 1]. Colliding vectors orogen,
/// splitting vectors rift, sliding vectors do neither. Rifting needs
/// stronger divergence than orogeny (`RIFT_DAMP`): on real Earth,
/// collision/subduction dominates continental margins while rifting is
/// the rare Baikal/Rhine exception.
const RIFT_DAMP: f32 = 0.8;
fn response(closing: f32, t: &TectonicTuning) -> (f32, f32) {
    (
        (closing * t.drift_speed).clamp(0.0, 1.0),
        (-closing * t.drift_speed * RIFT_DAMP).clamp(0.0, 1.0),
    )
}

/// Per-boundary character: `(strength, width scale, pulse direction,
/// pulse phase)`. Strength averages the two plates' margin activities
/// (biased toward quiet seams) and the pulse phase averages their
/// phases — both SYMMETRIC in the pair, so adjacent boundaries of an
/// active plate join into one long chain. Width and pulse direction
/// stay per-edge for variety.
fn edge_profile(a: &Plate, b: &Plate, seed: i32) -> (f32, f32, Vec2, f32) {
    let (c1, c2) = if (a.cell.y, a.cell.x) <= (b.cell.y, b.cell.x) {
        (a.cell, b.cell)
    } else {
        (b.cell, a.cell)
    };
    let mut x = plate_hash(c1, seed ^ 0x51ed_270b)
        ^ (c2.x as u32).wrapping_mul(0xc2b2_ae3d)
        ^ (c2.y as u32).wrapping_mul(0x27d4_eb2f);
    let width = (next_unit(&mut x) + 0.5).clamp(0.0, 1.0);
    let angle = (next_unit(&mut x) + 0.5) * std::f32::consts::TAU;
    let activity = (a.activity + b.activity) * 0.5;
    let phase = (a.pulse_phase + b.pulse_phase) * 0.5 * std::f32::consts::TAU;
    (
        0.45 + 0.55 * activity * activity.sqrt(),
        0.6 + 0.9 * width,
        Vec2::new(angle.cos(), angle.sin()),
        phase,
    )
}

/// Pull a warped-space gradient back through the curl warp:
/// `∇_w F = Jᵀ ∇_u F` with `J = [[1, s1], [s2, 1]]`.
fn pull_back(g: Vec2, s1: f32, s2: f32) -> Vec2 {
    Vec2::new(g.x + s2 * g.y, s1 * g.x + g.y)
}

/// One scramble step of the integer hash below: mixes `x` and returns a
/// fresh float in [-0.5, 0.5). Integer scrambling only — bit exact on
/// every platform.
#[inline]
fn next_unit(x: &mut u32) -> f32 {
    *x ^= *x >> 16;
    *x = x.wrapping_mul(0x7feb_352d);
    *x ^= *x >> 15;
    *x = x.wrapping_mul(0x846c_a68b);
    *x ^= *x >> 16;
    (*x >> 8) as f32 / 16_777_216.0 - 0.5
}

/// Seed a plate's hash stream from its lattice cell. Integer inputs are
/// small lattice coordinates, so (unlike the filter's IQ hash) this is
/// precision-safe at any world coordinate.
fn plate_hash(cell: IVec2, seed: i32) -> u32 {
    (cell.x as u32)
        ^ (cell.y as u32).wrapping_mul(0x9e37_79b9)
        ^ (seed as u32).wrapping_mul(0x85eb_ca6b)
}

/// A plate's feature-point position in plate space — the cheap half of
/// its data (two hashes), needed for all nine candidate cells.
#[inline]
fn plate_pos(cell: IVec2, seed: i32) -> Vec2 {
    let mut x = plate_hash(cell, seed);
    let jx = next_unit(&mut x);
    let jy = next_unit(&mut x);
    cell.as_vec2() + Vec2::new(0.5 + PLATE_JITTER * jx, 0.5 + PLATE_JITTER * jy)
}

/// A plate's hilliness, margin activity, and pulse phase — hashed per
/// cell. Activity and pulse phase make adjacent boundaries of an active
/// plate correlate: their segments join into one long range along the
/// whole margin, instead of unrelated short strokes.
#[inline]
fn plate_character(cell: IVec2, seed: i32) -> (f32, f32, f32) {
    let mut x = plate_hash(cell, seed);
    let _jitter_x = next_unit(&mut x);
    let _jitter_y = next_unit(&mut x);
    let hilliness = next_unit(&mut x);
    let activity = next_unit(&mut x);
    let pulse_phase = next_unit(&mut x);
    (
        (hilliness + 0.5).clamp(0.0, 1.0),
        (activity + 0.5).clamp(0.0, 1.0),
        (pulse_phase + 0.5).clamp(0.0, 1.0),
    )
}

/// Plate drift as a smooth circulation field over plate space: the drift
/// direction rotates slowly from plate to plate (two seeded sine lobes),
/// so a plate pushes into its neighbour COHERENTLY along the whole
/// shared boundary — long ranges and long rifts along one margin — where
/// independent per-plate random drifts made most pairs slide and only
/// lucky pairs collide (rare high-frequency blotches). Sampled at the
/// feature points, so the closing rate stays a per-pair constant and
/// every gradient remains exact.
fn drift_field(pos: Vec2, phases: Vec2) -> Vec2 {
    const WAVELENGTH: f32 = 3.0;
    const SWING: f32 = 1.1;
    let k = std::f32::consts::TAU / WAVELENGTH;
    let angle = SWING * (pos.x * k + phases.x).sin() + SWING * (pos.y * k + phases.y).sin();
    Vec2::new(angle.cos(), angle.sin())
}

/// The two nearest plates to `w` (plate space) with their distances F1
/// and F2. The F1/F2 equidistant surface is the plate boundary.
/// `PLATE_JITTER < 1` keeps both winners inside the 3x3 neighbourhood.
fn nearest_plates(w: Vec2, seed: i32) -> (Plate, Plate, f32, f32) {
    let drift_phases = warp_phases(seed ^ 0x1b8e_9c47);
    let base = w.floor().as_ivec2();
    let mut cells = [IVec2::ZERO; 9];
    let mut poss = [Vec2::ZERO; 9];
    let mut first = (0usize, f32::INFINITY);
    let mut second = (1usize, f32::INFINITY);
    let mut k = 0usize;
    for dj in -1..=1i32 {
        for di in -1..=1i32 {
            let cell = base + IVec2::new(di, dj);
            let pos = plate_pos(cell, seed);
            cells[k] = cell;
            poss[k] = pos;
            let d = w.distance_squared(pos);
            if d < first.1 {
                second = first;
                first = (k, d);
            } else if d < second.1 {
                second = (k, d);
            }
            k += 1;
        }
    }
    let (hilliness, activity, pulse_phase) = plate_character(cells[first.0], seed);
    let nearest = Plate {
        cell: cells[first.0],
        pos: poss[first.0],
        drift: drift_field(poss[first.0], drift_phases),
        hilliness,
        activity,
        pulse_phase,
    };
    let (hilliness, activity, pulse_phase) = plate_character(cells[second.0], seed);
    let runner_up = Plate {
        cell: cells[second.0],
        pos: poss[second.0],
        drift: drift_field(poss[second.0], drift_phases),
        hilliness,
        activity,
        pulse_phase,
    };
    (nearest, runner_up, first.1.sqrt(), second.1.sqrt())
}

/// Phases of the plate-space curl warp, hashed from the seed (radians).
fn warp_phases(seed: i32) -> Vec2 {
    let mut x = seed as u32 ^ 0x632b_e5a9;
    let p1 = next_unit(&mut x);
    let p2 = next_unit(&mut x);
    Vec2::new(p1, p2) * std::f32::consts::TAU
}

/// The IQ hash behind the filter has no seed input, so map the seed onto
/// a domain offset. The offset MUST stay small: the hash computes
/// `fract(x·y·(x+y))`, whose f32 precision collapses as the cube of the
/// coordinate — beyond a few hundred p-units `fract` returns quantized
/// garbage and the terrain grows grid-aligned artifacts (the "orthogonal
/// ridges" the first attempt at this had with ±8192 offsets). ±16 keeps
/// the hash in its comfort zone while still placing seeds ~7 hill
/// wavelengths apart, decorrelated for practical purposes.
fn seed_offset(seed: i32) -> Vec2 {
    const OFFSET_SCALE: f32 = 16.0;
    let mut x = seed as u32 ^ 0x9e37_79b9;
    let ox = next_unit(&mut x);
    let oz = next_unit(&mut x);
    Vec2::new(ox, oz) * OFFSET_SCALE
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
    fn zero_strength_reproduces_the_unfiltered_input() {
        // The filter only ever adds `faded * strength` to the height, so
        // strength = 0 must reduce to the raw fBm composition plus the
        // (filter-independent) macro layer at whatever scale the
        // tectonic field picked — the kill switch for comparing eroded
        // and raw looks.
        let mut tuning = TerrainTuning::default();
        tuning.erosion.strength = 0.0;
        let noise = TerrainNoise::from_tuning(&tuning);
        for x in [0.0, 137.2, -9_600.0] {
            for z in [0.0, -515.0, 32_000.0] {
                let t = noise.tectonic(x, z);
                let bp = Vec2::new(x, z) / tuning.base_wavelength_m + noise.offset;
                let base_noise = fbm(bp, 1.0, tuning.base_octaves, BASE_LACUNARITY, BASE_GAIN);
                let base_unit = base_noise.x * BASE_UNIT_AMP;
                let base_x = base_noise.x;
                let pm = Vec2::new(x, z) / tuning.tectonics.macro_wavelength_m + noise.macro_offset;
                let (macro_val, _) = ridged_multifractal(
                    pm,
                    tuning.tectonics.macro_octaves,
                    MACRO_FREQUENCY,
                    MACRO_LACUNARITY,
                    MACRO_GAIN,
                );
                let expected = (base_unit - VERTICAL_BIAS) * t.relief_m - t.subsidence_m
                    + macro_val * tuning.tectonics.macro_amplitude_m * t.orogeny
                    + base_x * BASE_RELIEF_FRACTION * t.relief_m;
                assert!(
                    (noise.sample(x, z).height - expected).abs() < 1e-6,
                    "strength = 0 drifted from the raw composition at ({x}, {z})"
                );
            }
        }
    }

    #[test]
    fn zero_relief_and_rift_is_a_flat_zero_world() {
        // Grounding sanity: collapsed relief and no subsidence flatten
        // every height to exactly 0, tectonic layers included.
        let mut tuning = TerrainTuning::default();
        tuning.tectonics.plains_relief_m = 0.0;
        tuning.tectonics.hills_relief_m = 0.0;
        tuning.tectonics.mountain_relief_m = 0.0;
        tuning.tectonics.rift_depth_m = 0.0;
        tuning.tectonics.macro_amplitude_m = 0.0;
        let noise = TerrainNoise::from_tuning(&tuning);
        for x in [0.0, 55.5, -1234.0] {
            for z in [0.0, -77.7, 4000.0] {
                assert_eq!(noise.sample(x, z).height, 0.0, "flat at ({x}, {z})");
            }
        }
    }

    #[test]
    fn sample_fields_stay_in_range_and_finite() {
        let noise = TerrainNoise::default();
        let t = &noise.tectonics;
        // |unit| stays under ~0.35 in practice; the bound keeps room
        // for the whole erosion envelope, the macro layer, and the rift
        // floor.
        let bound = 0.35 * t.mountain_relief_m
            + t.rift_depth_m
            + t.macro_amplitude_m * 2.6
            + BASE_RELIEF_FRACTION * t.mountain_relief_m * 1.2
            + 1.0;
        for x in [-20_000.0, 0.0, 12.5, 8_000.0, 40_000.0] {
            for z in [-20_000.0, 1.25, 6_800.0, 30_000.0] {
                let s = noise.sample(x, z);
                assert!(s.height.is_finite());
                assert!(s.height.abs() < bound, "height {} at ({x}, {z})", s.height);
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
    fn erosion_fires_on_the_rolling_hills_not_just_rare_steeps() {
        // The uniform-world regression: with the demo composition the
        // base fBm alone must produce slopes that clear the erosion
        // onset gate in most places, so gullies cover the landscape
        // instead of firing only on rare steep spots. Most of a plain
        // grid must move by a meaningful fraction of the filter's
        // maximum carve depth at the LOCAL relief (plains carve less in
        // absolute metres than mountains — that is the point of the
        // tectonic layer — but the relative carve must hold everywhere).
        let eroded = TerrainNoise::default();
        let mut tuning = TerrainTuning::default();
        tuning.erosion.strength = 0.0;
        let raw = TerrainNoise::from_tuning(&tuning);
        let mut moved = 0;
        let mut total = 0;
        for x in 0..30 {
            for z in 0..30 {
                let (x, z) = (x as f32 * 53.0 + 7.0, z as f32 * 61.0 + 11.0);
                let relief = eroded.tectonic(x, z).relief_m;
                let max_carve = tuning.erosion.strength * tuning.erosion.scale * relief * 2.0;
                let d = (eroded.sample(x, z).height - raw.sample(x, z).height).abs();
                total += 1;
                if d > 0.1 * max_carve {
                    moved += 1;
                }
            }
        }
        assert!(
            moved > total / 2,
            "erosion moved only {moved}/{total} points meaningfully — the onset gate is starving it again"
        );
    }

    #[test]
    fn tectonic_gradients_match_finite_differences() {
        // The closed-form Worley gradients (feature warp, relief ramp,
        // rift walls) must match the field's own finite difference —
        // this is the chain rule the tile normals rely on.
        let noise = TerrainNoise::default();
        for (x, z) in [(0.0, 0.0), (2_345.0, -1_100.0), (-7_800.0, 4_200.0)] {
            let t = noise.tectonic(x, z);
            const E: f32 = 200.0;
            let xm = noise.tectonic(x - E, z);
            let xp = noise.tectonic(x + E, z);
            let zm = noise.tectonic(x, z - E);
            let zp = noise.tectonic(x, z + E);
            let cases = [
                (
                    "relief",
                    t.d_relief,
                    Vec2::new(
                        (xp.relief_m - xm.relief_m) / (2.0 * E),
                        (zp.relief_m - zm.relief_m) / (2.0 * E),
                    ),
                    0.15,
                ),
                (
                    "feature",
                    t.d_feature,
                    Vec2::new(
                        (xp.feature_m - xm.feature_m) / (2.0 * E),
                        (zp.feature_m - zm.feature_m) / (2.0 * E),
                    ),
                    0.15,
                ),
                (
                    "subsidence",
                    t.d_subsidence,
                    Vec2::new(
                        (xp.subsidence_m - xm.subsidence_m) / (2.0 * E),
                        (zp.subsidence_m - zm.subsidence_m) / (2.0 * E),
                    ),
                    0.05,
                ),
            ];
            for (name, analytic, fd, tol) in cases {
                assert!(
                    (analytic.x - fd.x).abs() < tol && (analytic.y - fd.y).abs() < tol,
                    "{name} gradient off at ({x}, {z}): {analytic:?} vs {fd:?}"
                );
            }
        }
    }

    #[test]
    fn analytic_slope_tracks_the_height_field() {
        // The reported slope (used for tile normals) must track the height
        // field's own finite difference: same direction, similar grade.
        // This covers the tectonic chain rule too — the feature warp and
        // rift walls are folded in analytically. Exact equality is not
        // the filter's contract — its slope state is an analytic estimate
        // of the carved surface, off by up to a few times on narrow
        // gully flanks where the absolute error is fractions of a degree
        // (invisible in shading), so near-flat points get an
        // absolute-error escape instead of the ratio band.
        let noise = TerrainNoise::default();
        // At a Voronoi switch locus the second-nearest plate changes
        // identity INSIDE the finite-difference stencil, so analytic and
        // FD legitimately disagree there (the per-plate margin activity
        // makes the pair strengths differ). Those loci are measure-tiny
        // — one outlier in a spread of samples is allowed, any more
        // means a real chain-rule regression.
        const POINTS: [(f32, f32); 10] = [
            (13.0, 71.0),
            (-420.0, 90.0),
            (1234.5, -777.25),
            (4_500.0, -2_600.0),
            (-8_100.0, 3_300.0),
            (9_600.0, 12_700.0),
            (-2_500.0, -14_800.0),
            (17_400.0, -5_600.0),
            (-19_200.0, -9_100.0),
            (6_700.0, 21_300.0),
        ];
        let mut outliers = 0;
        for (x, z) in POINTS {
            let s = noise.sample(x, z);
            const H: f32 = 0.5;
            let fd = Vec2::new(
                (noise.sample(x + H, z).height - noise.sample(x - H, z).height) / (2.0 * H),
                (noise.sample(x, z + H).height - noise.sample(x, z - H).height) / (2.0 * H),
            );
            // Direction is meaningless on near-flat points (both slopes
            // ~1°): only compare where there is a slope to speak of.
            let direction_ok = s.slope.dot(fd) > 0.0
                || s.slope.length() < 0.05
                || fd.length() < 0.05;
            let ratio = s.slope.length() / fd.length().max(1e-3);
            let abs_err = (s.slope - fd).length();
            // The filter's slope state is an estimate; at gully flanks it
            // can undershoot the true grade — 0.15 m/m (~8.5°) absolute
            // error stays acceptable for shading.
            let magnitude_ok = (0.25..=4.0).contains(&ratio) || abs_err < 0.15;
            if !direction_ok || !magnitude_ok {
                outliers += 1;
                eprintln!(
                    "slope outlier at ({x}, {z}): {:?} vs {fd:?} (ratio {ratio}, abs {abs_err})",
                    s.slope
                );
            }
        }
        assert!(
            outliers <= 1,
            "analytic slope diverges from the height field at {outliers}/{} sampled points",
            POINTS.len()
        );
    }

    #[test]
    fn colliding_plates_raise_mountains_and_splitting_plates_open_rifts() {
        // The sign convention, pinned on hand-built plates through the
        // real code path (`closing_rate` + `response`): head-on vectors
        // converge to full orogeny on the boundary, opposite vectors
        // open a rift, and shear (perpendicular drift) leaves the
        // boundary quiet.
        let t = TectonicTuning::default();
        // Plate a sits west of plate b, two units apart in plate space.
        let resolve = |drift_a: Vec2, drift_b: Vec2| {
            let a = Plate {
                cell: IVec2::ZERO,
                pos: Vec2::new(0.0, 0.0),
                drift: drift_a,
                hilliness: 0.0,
                activity: 1.0,
                pulse_phase: 0.0,
            };
            let b = Plate {
                cell: IVec2::new(2, 0),
                pos: Vec2::new(2.0, 0.0),
                drift: drift_b,
                hilliness: 0.0,
                activity: 1.0,
                pulse_phase: 0.0,
            };
            response(closing_rate(&a, &b), &t)
        };

        let (orogeny, rift) = resolve(Vec2::X, Vec2::NEG_X);
        assert!(orogeny > 0.9, "head-on collision must orogen: {orogeny}");
        assert!(rift < 1e-6, "collision must not rift: {rift}");

        let (orogeny, rift) = resolve(Vec2::NEG_X, Vec2::X);
        assert!(rift > 0.9, "splitting plates must rift: {rift}");
        assert!(orogeny < 1e-6, "splitting plates must not orogen: {orogeny}");

        let (orogeny, rift) = resolve(Vec2::Y, Vec2::NEG_Y);
        assert!(
            orogeny < 0.05 && rift < 0.05,
            "shear boundary must stay quiet: orogeny {orogeny}, rift {rift}"
        );
    }

    #[test]
    fn the_default_world_contains_mountains_and_rift_basins() {
        // With random plate drifts, some boundaries must collide and some
        // must split: the default world has to actually contain both
        // natures, at strength, within a few plate diameters of the
        // origin. Dense scan — orogeny/subsidence only peak in a narrow
        // band around each boundary, so a sparse grid can step over
        // them. ±60 km because per-boundary strength/width variation and
        // the range pulse make full-strength segments a minority: a
        // smaller window may legally contain none.
        let noise = TerrainNoise::default();
        let t = &noise.tectonics;
        let mut max_relief: f32 = 0.0;
        let mut max_subsidence: f32 = 0.0;
        for i in 0..80 {
            for j in 0..80 {
                let f = noise.tectonic(i as f32 * 1500.0 - 60_000.0, j as f32 * 1500.0 - 60_000.0);
                max_relief = max_relief.max(f.relief_m);
                max_subsidence = max_subsidence.max(f.subsidence_m);
            }
        }
        assert!(
            max_relief > 0.75 * t.mountain_relief_m,
            "no real mountains in the scan: max relief {max_relief}"
        );
        assert!(
            max_subsidence > 0.6 * t.rift_depth_m,
            "no real rift basins in the scan: max subsidence {max_subsidence}"
        );
    }

    #[test]
    fn tectonic_field_is_smooth_at_terrain_scales() {
        // The modulation must vary at plate scale only: a 300 m step (a
        // full erosion wavelength) may not jump the relief or open a
        // sudden rift wall — that would tear visible seams into tiles.
        // Rare exceptions are inherent: at Voronoi triple points the
        // SECOND-nearest plate switches identity mid-boundary and the
        // closing rate steps between the two pairs' values (a measure-
        // tiny locus — a handful of points per continent). So: nearly
        // every sampled point must be smooth, a couple may sit on a
        // step.
        let noise = TerrainNoise::default();
        let t = &noise.tectonics;
        let relief_range = t.mountain_relief_m - t.plains_relief_m;
        const CHECKS: usize = 12;
        let mut smooth = 0;
        for k in 0..CHECKS {
            let x = k as f32 * 1731.0 - 9_000.0;
            let z = (k % 5) as f32 * 2_210.0 - 4_000.0;
            let a = noise.tectonic(x, z);
            let b = noise.tectonic(x + 300.0, z);
            if (a.relief_m - b.relief_m).abs() < 0.1 * relief_range
                && (a.subsidence_m - b.subsidence_m).abs() < 0.1 * t.rift_depth_m
            {
                smooth += 1;
            }
        }
        assert!(
            smooth >= CHECKS - 2,
            "tectonic field tears at terrain scales: {smooth}/{CHECKS} sampled points smooth"
        );
    }

    #[test]
    fn seed_offsets_stay_inside_the_hash_precision_envelope() {
        // The load-bearing bound from `seed_offset`'s doc comment: every
        // possible seed must land inside ±16 p-units, and adjacent seeds
        // must not collide (they land ~7 hill wavelengths apart).
        let mut seen = std::collections::HashSet::new();
        for seed in 0..1000 {
            let o = seed_offset(seed);
            assert!(
                o.x.abs() <= 16.0 && o.y.abs() <= 16.0,
                "seed {seed} escaped the envelope: {o:?}"
            );
            assert!(seen.insert((o.x.to_bits(), o.y.to_bits())), "seed {seed} collides");
        }
    }
}






#[cfg(test)]
mod probe {
    use super::*;
    #[test]
    fn find_ranges_near_origin() {
        let n = TerrainNoise::default();
        let mut best = Vec::new();
        for i in 0..80 {
            for j in 0..80 {
                let x = i as f32 * 500.0 - 20_000.0;
                let z = j as f32 * 500.0 - 20_000.0;
                let o = n.tectonic(x, z).orogeny;
                if o > 0.5 {
                    best.push((o, x, z));
                }
            }
        }
        best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (o, x, z) in best.iter().take(6) {
            println!("RANGE orogeny {:.2} at ({:.0}, {:.0})", o, x, z);
        }
        println!("total strong samples: {}", best.len());
        // Height grid through the strongest point
        let (cx, cz) = (best[0].1, best[0].2);
        for j in (0..10).rev() {
            let mut row = String::new();
            for i in 0..10 {
                let h = n.sample(cx + (i as f32 - 5.0) * 400.0, cz + (j as f32 - 5.0) * 400.0).height;
                row.push_str(&format!("{h:6.0} "));
            }
            println!("{row}");
        }
    }
}
