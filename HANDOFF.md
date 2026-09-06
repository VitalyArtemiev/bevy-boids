# Session handoff — erosion-filter terrain (2026-09-05, updated 2026-09-06)

Point-in-time summary of the terrain-erosion session, written so a fresh
session on any machine can pick up the work. Commits `c60cd2a`…`76b5569`
on top of `5919964`. Read together with `.agents/skills/bevy-boids/SKILL.md`
(conventions, commands, hard constraints) — that file is the living source
of truth; this file is narrative.

## What landed

### Tectonic plate layer (2026-09-06, uncommitted)
Regional terrain character on top of the (rewritten) erosion terrain: a
Worley noise whose cells are tectonic plates (`plate_size_m` 9 km
default). Each plate hashes a 16-direction drift vector and an interior
hilliness. Per point, the two nearest plates resolve their shared
boundary by the closing rate of their drifts along the boundary normal:
colliding → orogeny bump (relief ramps to `mountain_relief_m`, feature
tightens to `mountain_feature_m`), splitting → the would-be uplift is
spent as subsidence (`rift_depth_m` below the plains — Baikal-style
elongated basins; water rendering is still H3), shear → quiet.
Interiors blend the two nearest plates' hilliness between
`plains_relief_m` and `hills_relief_m`. The erosion terrain itself is
UNCHANGED — the tectonic layer only sets the per-point relief and
feature scale it is drawn at (`feature_m(x,z)` acts as a smooth domain
warp on the filter coordinates).

- `TerrainTuning` gains `tectonics: TectonicTuning` (9 knobs, F3
  "Tectonics" section); the global `feature_size_m`/`relief_m` are gone.
- Slopes chain-rule analytically through the modulation: Worley F1/F2
  have exact gradients (unit vectors to the feature points), drifts are
  per-pair constants, so the whole tectonic field differentiates in
  closed form — `tectonic_gradients_match_finite_differences` pins the
  math. Perf note: an earlier FD version needed 5 Worley evals/sample
  (7.7 ms/tile in dev); the analytic version needs 1 and lands at
  ~3.9 ms/tile dev isolated (~1.9 ms release) — ~2.5 µs/sample: ~0.7 µs
  tectonic + ~1.5 µs filter, cranelift opt-1 (the filter crate runs
  opt-3 LLVM as a dep). `plate_pos` is computed for all 9 cells but
  drift/hilliness (`plate_character`) only for the two winners. The
  tile-mesh dev pin moved 4 → 6 ms with the new baseline (the old pin
  was calibrated to the single-layer world and flaked at a 3% margin
  under parallel test load).
- `HeightField.relief_m` is now the COLORING REFERENCE
  (`tectonics.mountain_relief_m`): snow/grass unit-height gates convert
  with it, so snow crowns only real mountains. The live relief varies
  per point.
- Known artifact: where the second-nearest plate switches identity at
  Voronoi triple points on a boundary, orogeny jumps between two
  collision strengths — an inherent step of the 2-plate approximation,
  isolated to triple points. If it ever reads badly, blend the top-3
  candidates there.
- **Organic boundaries (second iteration, same day).** First render
  passed the tests but the ranges ran as straight lines parallel to the
  lattice-cell edges: the F1/F2 equidistant locus IS the perpendicular
  bisector — a straight line near cell edges at jitter 0.8, with
  constant per-pair strength along it. Fixes, all analytic so gradients
  stay exact:
  * Plate space runs through a seeded **curl warp**
    (`u = w + A·(sin(w.y k+φ1), sin(w.x k+φ2))`, `boundary_curvature`
    0.35, wavelength 6 plates); Worley gradients pull back through Jᵀ —
    `tectonic_gradients_match_finite_differences` pins the math.
  * **Margin activity** is hashed per PLATE and averaged per boundary
    (`0.3 + 0.7·m^1.5`), and the **pulse phase** averages per-plate
    phases — adjacent boundaries of an active plate join into one long
    range instead of unrelated short strokes. Width and pulse direction
    stay per-edge. Rifting is damped ×0.8 (`RIFT_DAMP`): collisions
    dominate, rifts stay the rare Baikal exception.
  * `PLATE_JITTER` 0.8 → 0.92 (bisectors less grid-parallel).
  * Tuned by rendering the relief field to PPM/PNG and by in-game
    `--shot` at zooms 0.1/0.3/0.45/0.85: curved snow-capped chains with
    dirt foothill fringes, mottled plains, chalk drainage streaks, no
    grid artifacts.
- **Color-side fixes from the same pass**: drainage chalk now gates on
  steepness (`smoothstep(0.08, 0.25, slope)`) — on flat plains the
  creases fired pale dandruff; dirt gate starts earlier (0.18) so
  foothills transition instead of green→cliff snapping; grass two-tone
  widened (0.40–0.55 units) for plains mottling; `plains_relief_m`
  18 → 35 m so plains shade as landform instead of a sheet. The
  analytic-vs-FD slope test now tolerates one switch-locus outlier in
  ten samples, skips direction on near-flat points, and allows 0.15 m/m
  absolute error (the filter's slope state locally undershoots gully
  flanks; fine for shading).
- Still visible: the tile LOD ring drop on hilly ground reads as a
  faint linear seam from high zoom (geometry sits one `level_drop`
  lower per level) — pre-existing clipmap tradeoff, H5 vertex morphing
  is the fix.
- Default seed is **2026**, picked by rendering the field over a few
  candidates (a PPM-writer probe, since deleted): the origin sits on
  plains with a mountain belt ~4-5 km out and a rift basin beyond — the
  default spawn sees all three natures. Seed 1337 spawned mid-plains
  (flat green to the horizon — correct, boring).
- Visual verification: new bench-only `--shot <path>` flag
  (`launch.rs`) captures the framebuffer via Bevy's screenshot API near
  the end of the run. This machine's Wayland compositor throttles the
  occluded game window to exactly ~20 fps WITH OR WITHOUT terrain (CPU
  idle) — the bench FPS number is meaningless here; use the tile-mesh
  test and `--shot` output instead. Screenshot at zoom 0.5: plains at
  spawn, snow-capped range 4-5 km out, rift basin beyond, no grid
  artifacts.
- Sign convention pinned by
  `colliding_plates_raise_mountains_and_splitting_plates_open_rifts`
  (head-on → orogeny > 0.9, split → rift > 0.9, shear → both ~0;
  `boundary_interaction` is symmetric under plate swap, so the field is
  continuous across borders).

### Multi-octave rebuild + coherent margins (2026-09-06, third iteration)
Zoomed-in verification (new bench `--focus x,z` flag parks the camera on a
chosen world spot) exposed the real story behind "uniform peaks, no
multi-octave variation": the demo composition's base fBm has gain 0.1 —
EFFECTIVELY SINGLE-OCTAVE. In the 22 m diorama viewed from 15 m that
reads as steep carved terrain; scaled to a 420 m-relief RTS world it is
one wavelength of bumps — no massifs, no valleys, nothing for the
tectonics to amplify. Fixes, in sampling order:

1. **Base landform is now a true cascade**: world-space wavelength
   `base_wavelength_m` (350 m, decoupled from the gully-scale feature),
   5 octaves, gain 0.45 (was 3 octaves/gain 0.1 keyed to feature).
2. **Relief-scaled big forms** (`BASE_RELIEF_FRACTION` 0.35): the base
   displaces world metres by 0.35 × local relief — restoring the demo's
   amplitude:wavelength ratio (~0.3) in mountains; the unit-height
   filter input stays gentle so the crate's tuned thresholds keep their
   meaning. The filter's job is gully detail on top of the big forms.
3. **Macro landform layer**: 6-octave fBm at `macro_wavelength_m`
   (1.8 km), RIDGED ((1−|n|)² — crest lines where the noise crosses
   zero), amplitude `macro_amplitude_m` (170 m) gated by orogeny — the
   tectonic field decides WHERE the macro relief goes, per the original
   design intent. Its analytic gradient (including the ridge transform
   and the orogeny-modulation term) feeds both the final normals and
   the filter's input slope so gullies lean down-massif.
4. **Coherent plate drift**: per-plate random drift vectors made most
   boundaries slide (|closing| < 0.2 for 70% of spine samples) — ranges
   shrank to rare speckles. Plate drift now comes from a smooth
   circulation field (two seeded sine lobes, wavelength 3 plates)
   sampled at the feature points: a plate pushes into its neighbour
   coherently along the WHOLE boundary — long ranges, long rifts.
   drift_speed 2.4, strength floor 0.45, RIFT_DAMP 0.8.
5. **LOD-aware colors** (tiles.rs): dirt/cliff/chalk steepness gates
   stay SHARP (coarse vertices undersample slopes — widening fabricated
   dirt), height gates (grass/snow) widen with level, chalk amplitude
   fades with level (decorrelated ridge_map at 256 m cells fired pale
   dither), and ring-facing tile edges morph vertex colors toward the
   tile average (`LEVEL_DROP_STEP_FRACTION` 0.5 → 0.3). The streaming
   rectangle at mid zoom went from tan-vs-green clash to a faint
   boundary; full geometry+color morphing remains H5.

**Gully-ratio inversion (same-day fix):** the user's screenshots showed
high-frequency uni-directional sawtooth on dark steep faces. Root cause:
the filter carve's world amplitude scales with relief (±28 m at 420) but
its wavelength scaled DOWN with orogeny (feature 150 m → 16 m gullies) —
amplitude:wavelength ~1.7, near-vertical teeth; the demo's ratio is
~0.1. `mountain_feature_m` now LARGER than plains (1200 vs 400 → ~126 m
ravines, ratio ~0.22); cliff gate raised (slope 1.4+, palette lightened
to 0.30 grey) so only true rock walls read black.

**Sawtooth round two (same-day):** default-settings zoom-ins still
showed dense parallel teeth. Two causes, both fixed:
1. The macro's ridge transform was applied to the SUM of the 6-octave
   fBm — the sum crosses zero at every octave wiggle, planting a sharp
   crest at each crossing: dense parallel sawtooth at all scales.
   Replaced with a per-octave **ridged multifractal**
   (`ridged_multifractal`, Musgrave weight = previous ridge): crests
   only at each octave's own zero crossings, small ridges concentrated
   along the big spines. Exact gradients preserved
   (`tectonic_gradients…` + `zero_strength` pin it).
2. The remaining crest speckle was RENDERING, not noise: all octaves
   were evaluated at every LOD, so 8-32 m ravine detail on 16-64 m
   cells alias-shaded into shattered-glass facets. `TerrainNoise::
   sample_lod(x, z, level)` truncates each octave ladder per level
   (base/macro −level/2, erosion −level); `HeightField::sample_lod`
   carries it, `tile_mesh` renders with `key.level`, and gameplay
   callers keep full detail via `height()`/`sample()` (level 0).
3. Shading-side: vertex NORMALS and COLORS now come from a 3x3 blur of
   the sample neighborhood (blur samples cross tile borders via
   `sample_lod` at the same world points, so shared vertices still
   agree — `normals_agree_across_tile_boundaries` pins it); on 50°
   carved terrain the exact per-vertex slope flipped color/shading
   thresholds per facet (shattered glass). Chalk fades with level;
   snow gate raised to unit 0.80-0.95 (caps only — the macro's mean
   uplift had turned whole crests white); dirt/cliff gates at slope
   0.5/1.4-2.4 with a lighter cliff grey. `sky.rs` adds a
   `GlobalAmbientLight` at 7% of full daylight (sky-tinted): the
   reference keeps steep faces readable with a ~30% bounce term; with
   Bevy's near-zero ambient, shadowed 45° faces collapsed to black.

Verified by in-game `--shot` at zooms 0.6/0.9/0.95 over a range
(`--focus 6000,-7500` on seed 2026): curved snow-dusted massifs with
crest lines, gully-carved green flanks, dirt foothill fringes, natural
range taper, no grid artifacts. All 76 tests green incl. the exact
macro/ridge chain rule (`zero_strength` + `tectonic_gradients…`).

### Noise rewrite: crate-only composition (2026-09-06, uncommitted)
The 2026-09-05 composition (below) was scrapped — the result was uniform
terrain with unnatural orthogonal ridges. Both symptoms had one root
cause each, and the fix was to port the crate's reference demo
*exactly* instead of adapting it:

1. **Orthogonal ridges — IQ hash degeneracy.** The old `seed_offset`
   mapped seeds to ±8192 p-units, but the crate's `hash2` computes
   `fract(x·y·(x+y))`, which at coordinates that large returns a
   *constant* gradient over thousands of lattice cells (verified
   numerically: 1 distinct value across a 64×64 lattice vs ~3300 at the
   origin). Constant gradients left only lattice-boundary discontinuities
   as structure — grid-aligned ridges over a featureless world. The
   rewrite keeps offsets within ±16 p-units
   (`seed_offsets_stay_inside_the_hash_precision_envelope` pins it).
2. **Uniformity — starving the onset gate.** The old custom base (fBm
   frequency 0.357 in p-space + fastnoise mountains) delivered ~10×
   smaller slopes than the demo's frequency-3 fBm, so the erosion
   `onset` threshold kept the carve mask near zero except on mountain
   flanks. The rewrite is a line-for-line port of the demo's
   `evaluate_terrain_with_octaves`: `fbm(p, freq, oct, 2.0, 0.1) *
   0.125`, fade `/0.6`, `base*0.5+0.5`, filter, `height = (unit − 0.43)
   × relief_m` (the tree bump is skipped — demo-only coloring).

`TerrainTuning` is now `{ seed, feature_size_m, relief_m, height_offset,
base_frequency, base_octaves, erosion: cpu::ErosionFilterParams }` — the
crate's parameter struct embedded verbatim (the `ErosionTuning` mirror
and every fastnoise knob: gone, along with the `fastnoise-lite` dep).
Defaults: feature 150 m (16 m gullies), relief 120 m, height offset
−0.65, hill frequency 3 (50 m hills). `HeightField` carries `relief_m`
so `ground_color` gates on unit height like the demo (snow at 0.53–0.60
units — the bands follow the relief slider now). New regression test
`erosion_fires_on_the_rolling_hills_not_just_rare_steeps` pins >50% of
sampled points moving meaningfully. F3 panel: "Landform" + "Erosion"
sections over exactly these fields.

### Terrain: runevision erosion filter (`a872eac`) — superseded above
The height function (`src/terrain/noise.rs`) is now Rune Skovbo Johansen's
"Advanced Terrain Erosion Filter" (blog.runevision.com, 2026-03; shadertoy
`wXcfWn`), evaluated per sample via the pure-Rust port
`bevy_erosion_filter` 0.2 (`default-features = false` → glam-only cpu
module, wasm-safe, MPL-2.0). Composition per `TerrainNoise::sample`:

1. **Base land**: crate `fbm` (IQ gradient noise, analytic derivatives)
   — replaced the old fastnoise base + detail layers.
2. **Mountains**: unchanged fastnoise ridged×mask layer (+domain warp);
   gradient by forward difference at `MOUNTAIN_GRADIENT_EPS_M = 2 m`
   (smooth/low-frequency, so this is plenty for gully direction).
3. **Filter**: `erosion_filter(p, h_and_slope, fade_target, params)` —
   carves branching gullies over N octaves, returns height/slope delta +
   `ridge_map` (+1 ridge / −1 crease).

Unit conventions (load-bearing):
- Filter works in **relief units**: heights divided by
  `base_amplitude_m + ridge_max_m`, slopes chain-ruled across
  `p = metres / feature_size_m`. This keeps carve depth
  amplitude-independent and the `onset`/`assumed_slope` thresholds at
  their reference-demo meaning.
- `feature_size_m` default **150** → gully wavelength ≈
  `feature_size · scale · cell_scale` ≈ 16 m. The port's raw defaults
  imply ~3 m gullies — sub-cell at our 1 m near LOD. Don't "fix" the
  default back.
- The IQ hash is unseeded: `seed_offset` maps the tuning seed onto a
  large integer-hash domain offset. `strength = 0` is an exact filter
  bypass (test-pinned).

`HeightField` now stores a sampler returning `TerrainSample
{ height, slope, ridge_map }`; `.height()` delegates for the
height-only consumers (grounding, camera, cursor march).

### Tiles: vertex colors, analytic normals, streaming budget (`a872eac`)
- `tile_mesh` takes ONE field sample per vertex for position + normal +
  color. Ground material is `base_color: WHITE` — Bevy 0.19 multiplies
  the mesh `COLOR` attribute into base color automatically (there is no
  `vertex_colors` flag anymore).
- `ground_color` (tiles.rs) is the CPU adaptation of the demo's
  per-fragment material: grass two-tone by altitude, dirt → cliff by
  slope (0.3/0.55 m-m gates), drainage chalk only in the narrow crease
  band (`ridgemap < 0.15` — 0.3 speckled), snow above 55 m (since the
  2026-09-06 rewrite: altitude gates run on unit height, snow at
  0.53–0.60 units). Values were
  tuned by screenshot; retune on screen, not on paper.
- Normals come from the analytic slope (tile-seam-exact, halves per-tile
  field samples). Known limit: the filter's slope state is an
  approximation — up to ~2.7× local error vs the geometry's finite
  difference on gully flanks. `analytic_slope_tracks_the_height_field`
  documents the tolerance; fine for shading.
- `StreamBudget` (2 ms/frame, absent resource = unlimited) throttles
  tile meshing so F3 edits flood the world back in instead of hitching —
  the filter makes each tile mesh several times costlier.

### F3 panel (`a872eac`)
`terrain_tuning_ui` (src/ui.rs) restructured into collapsing sections
(Seed / Base / Mountains / Erosion) with every filter parameter as a
slider + Reset to defaults. Same `bypass_change_detection` /
`set_changed` discipline as before — do not hold `&mut` through a ResMut
inside egui bodies.

### Preceding commits in range
- `c60cd2a`: debug_ui generic `tuned()` collapsing sections with reset
  buttons (why the tuning structs derive `PartialEq`) + SKILL.md
  conventions (options persistence, change-detection pattern).
- `5919964` (earlier session, context): `launch.rs` bench harness
  (`--bench`, `--secs`, `--zoom`, scenario flags), atmosphere opt-in
  with 128 px env map, `dynamic`/`tracing` cargo features opt-in,
  vendored `bevy_rts_camera` (its per-frame `follow_ground` mesh raycast
  cost ~5 ms/frame with 10k boids; we sample the HeightField
  analytically instead).

## Performance ledger (dev profile, ~1 MP window, AMD 780M)
- Default bench (9801 boids + shadows, atmosphere off): **~53 fps**
  (was ~51 pre-erosion; the removed detail layer offsets filter cost).
- Atmosphere isolated: ~77 fps off / ~53 sky-only / ~47 with 128 px
  cubemap. Bevy 0.19 refilters the env map every frame (upstream
  #24522 tracks on-demand regeneration) — atmosphere stays off by
  default.
- Unfiltered chrome trace distorts to ~8 fps; use the RUST_LOG target
  filter recorded in SKILL.md.
- Distant tiles undersample the ~16 m gullies (aliasing/shimmer at
  coarse LODs) — accepted for now; per-octave LOD fading is H5 work.

## Open threads (plan-file milestones: .zcode/plans/, gitignored)
- **H2**: height-delta edit layer + brushes + edit log (HeightField docs
  already promise the wrap-don't-replace pattern).
- **H3 remainder**: hydrology/water — erosion gullies exist; no water
  plane or flow accumulation yet.
- **H4**: sparse voxel detail volumes + footprint protocol.
- **H5**: UDLOD upgrades — vertex morphing at seams, per-octave gully
  fading by LOD (the aliasing above), minmax culling, GPU tile
  refinement. The cpu module mirrors the WGSL numerically, so a GPU
  port can match the CPU field exactly.
- **H6**: horizon structures/trees (the demo's tree canopy logic was
  deliberately skipped).
- Small: upstream bevy_rts_camera follow_ground opt-out would let us
  drop the vendored fork.

## Environment notes
- Windows MSVC, nightly pinned via rust-toolchain.toml, cranelift for
  profile.dev, 150% display scaling. For visual verification on this
  machine: full-screen capture returns black on a locked session;
  `--bench` mode + Win32 `PrintWindow` (flag 2, DPI-aware) into a temp
  PNG is the reliable path — the game window exposes no useful
  accessibility tree.
- Not pushed: `c60cd2a…76b5569` are local on master.
