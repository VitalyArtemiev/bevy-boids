# Session handoff — erosion-filter terrain (2026-09-05)

Point-in-time summary of the terrain-erosion session, written so a fresh
session on any machine can pick up the work. Commits `c60cd2a`…`76b5569`
on top of `5919964`. Read together with `.agents/skills/bevy-boids/SKILL.md`
(conventions, commands, hard constraints) — that file is the living source
of truth; this file is narrative.

## What landed

### Terrain: runevision erosion filter (`a872eac`)
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
  band (`ridgemap < 0.15` — 0.3 speckled), snow above 55 m. Values were
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
