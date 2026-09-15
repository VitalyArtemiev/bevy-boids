# Impostor billboard LOD — design

Status: implemented (`src/billboard.rs`, `src/boid.rs` catalog, pose axis in
`src/preprocess/`). Deviations from the original plan found during
implementation: the yaw source is `Target.dir` (the formation executor
writes facing there) with `Velocity.v` as the free-boid fallback, and the
plan's "rotate cell-local uv by the folded yaw" was wrong — world up
projects up-image in every baked cell and up-screen at runtime, so sprites
are displayed upright and yaw lives entirely in which cell the fold
selects (an early spin implementation made every capsule render sideways;
it was removed).

## Goal

Past a camera-distance threshold, each boid drops its PBR mesh
(`Mesh3d` + `MeshMaterial3d<StandardMaterial>`) and renders as one
GPU-billboard quad driven by the baked impostor atlas (normal-lit albedo,
`nuv = 1 - uv` pairing). Two requirements folded into the design:

- **Variation identity is a pure function of a stable boid id.** Soldiers
  must never "switch places" across zoom cycles or future LOD streaming:
  the visual variant (3-4 per formation within the *same* equipment type)
  is `variation_for(boid.id)`, never spawn-time randomness. Equipment
  changes are a separate future system and out of scope here.
- **The bake pipeline has an animation-pose axis from day one.** Walk and
  attack animation frames must bake into the same atlas later without
  layout changes.

## Two roles, kept distinct

- `BoidVariation` — the CATALOG: a `Resource` (`BoidVariations(Vec<...>)`)
  built once at startup (same lifetime as `Meshes`/`Materials`). Fields:
  `name`, `mesh`/`material` handles, bake-fit `center`/`radius` (the mesh's
  bounding sphere from `mesh_bounds()` — the baker orbits `center` and fits
  the orthographic frustum to the sphere; the runtime sizes the quad
  `radius × FIT_MARGIN` and offsets it by yaw-rotated `center` so billboard
  and mesh agree at the swap boundary).
- `Mesh3d` + `MeshMaterial3d<StandardMaterial>` — the real per-boid render
  components, attached at spawn exactly as today; temporarily removed when
  far, restored from the catalog on return.

No `ModelLod` component (considered and dropped): `variation_for(id)` is
pure, so the swap system computes the variation on the fly and clones
handles from the catalog. Handles compare by asset id, so restoration
returns the identical handles by construction.

```
spawn:  Mesh3d + Material from catalog[variation_for(id)]
far:    Mesh3d + Material REMOVED; Billboard + quad + atlas material INSERTED
near:   Billboard etc. REMOVED; Mesh3d + Material restored from
        catalog[variation_for(id)]
```

## 1. Shared variation registry — `boid.rs`

- `boid_variations(meshes, images, materials) -> Vec<BoidVariation>` — the
  preprocess's `VariationBake` renamed and moved so spawner and baker share
  ONE constructor. Ships three tinted capsule placeholders (same capsule,
  `base_color` tints over the UV-debug texture) to prove the multi-variation
  machinery before real soldier meshes exist; replacing them later is
  editing this one list.
- `Boid` gains `pub id: u32`, from a spawn-order counter (`BoidIds`) in
  `setup` — later from persisted formation state when boids stream.
  `pub const fn variation_for(id: u32) -> usize = (id as usize) %
  VARIATION_COUNT` (const pinned to the catalog length by a test). The id
  controls only the visual variation within same-equipment units.
- `BoidBundle::with_id(id, target, &BoidVariations)` attaches mesh/material
  via `variation_for(id)` — spawn convenience only; identity lives in the
  id. Existing constructors delegate with id 0.

## 2. `src/billboard.rs` — `BillboardPlugin` (self-contained, like `SkyPlugin`)

### Components / resources

- `Billboard` marker — presence is the far-LOD state ("component presence
  is state", mirroring `Velocity`-LOD on formations).
- `BillboardAssets` (plugin Startup): shared quad mesh + one
  `Handle<BillboardMaterial>` per catalog entry
  (`AssetServer::load("impostors/<name>.png")`), indexed by variation.
- `BillboardTuning { swap_distance_m: 50.0, hysteresis_m: 5.0 }` (`*Tuning`
  convention); `DebugConfig` gains `impostor_lod: bool` kill-switch.
- `pack_tag(yaw, pose) -> u32` — `MeshTag` packing: low 16 bits yaw
  fraction of τ, next 8 bits pose index. `update_billboard_yaw` rewrites
  yaw bits only; a future animation system rewrites pose bits only.

### Systems (into the existing `Update` / `GameState::Playing` block)

1. `swap_boid_lod` `.after(RtsCameraSystemSet)` (camera settled, like
   `camera_terrain_clearance`): camera via
   `Query<(&Camera, &GlobalTransform), With<Camera3d>>` + `single()`
   (draw_cursor pattern; works for RTS and freecam). One O(n) pass over
   `Query<(Entity, &Boid, &Transform, Option<&Billboard>)>` + the catalog
   resource, squared distance:
   - beyond `swap_distance_m` → remove `Mesh3d` /
     `MeshMaterial3d<StandardMaterial>`, insert
     `(Mesh3d(quad), MeshMaterial3d(assets[variation_for(boid.id)]),
     MeshTag(pack_tag(yaw_from_target, idle)), Billboard)`;
   - inside `swap_distance_m - hysteresis_m` → restore mesh/material from
     `catalog[variation_for(boid.id)]`, remove `Billboard`/`MeshTag`;
   - in the hysteresis band → no change (anti-thrash).
   Simulation untouched: `bob`, `ground_boids`, kd-tree selection,
   formation steering key on `Transform`/`GroundY`/`TrackedByTree`, never
   render components.
2. `update_billboard_yaw` — `Query<(&Target, &mut MeshTag), With<Billboard>>`:
   yaw from `Target.dir` (`atan2(z, x)`), folding into view azimuth per the
   atlas design. Billboards only, trivial cost.

### Material + shader (`assets/shaders/impostor_billboard.wgsl`)

- `BillboardMaterial { atlas, world_span, sun_dir, sun_color, ambient }` —
  custom `Material` (pattern: preprocess `NormalMaterial`):
  `enable_prepass() -> false`, `enable_shadows() -> false`,
  `AlphaMode::Mask(0.5)` — cutout keeps billboards in the opaque pass
  where automatic instancing holds; the flat-scene bench measured 35.8 fps
  blended vs 70.0 cutout vs 60.0 full meshes (5k, dev profile), because
  the transparent phase's per-instance sort defeats the batching. The
  custom fragment must `discard` below the cutoff itself — `Mask` only
  routes the pipeline; without the discard the atlas's black background
  writes behind every sprite. Soft alpha with instancing is possible via
  `AlphaMode::AlphaToCoverage` + `Msaa::Sample4` (measured 70.2 fps), at
  the cost of 4× MSAA on the whole scene.
  `world_span` =
  `radius × FIT_MARGIN` from the catalog; sun from `SkyTuning::default()`
  angles, same anchoring as the baker.
- Vertex: `MeshTag` + automatic instancing = one draw call per variation
  (`automatic_instancing` pattern, `mesh_functions::get_tag`). Quad
  centered at instance world position + yaw-rotated `center`, expanded
  along view right/up × `world_span`; view direction from
  `view.world_position`; ring/slot consts mirrored from `atlas.rs` (30°
  cone, rings [8,16,24], 7-wide); fold yaw from the tag; display the
  sprite upright — no uv rotation (world up projects up-image in every
  baked cell and up-screen at runtime, so yaw lives entirely in cell
  choice); the image u axis maps along screen-right (a mirrored display
  is invisible on symmetric albedo but flips the baked normal field
  against the runtime view basis — the sun then lights the sprite from
  the wrong side horizontally; this and the normal half's 180° content
  rotation are a paired invariant, each alone re-opens the opposite
  axis's bug); apply the pose band offset
  (uv y += pose · pose_stride) before the normal mirror.
- Fragment: `albedo = sample(uv)`, `n = 2·sample(1 - uv) - 1` (sRGB decode
  free via sRGB sampling); lighting reads the engine's `lights` view
  binding directly — `albedo·(ambient_color + Σ light_color·NdotL/π)`
  with the light colour premultiplied by illuminance and the direction
  rotated into view space. This replaced an earlier material-uniform
  design (sun/ambient constants + a `sync_billboard_sun` bridge), which
  was both a startup snapshot the F1 sliders couldn't reach and a baked
  ambient floor that would have glowed at night; one lighting authority
  now serves meshes and billboards. Two magnitude steps are mandatory
  alongside the binding read, both mirroring `pbr_functions.wgsl`'s
  fragment tail: (1) scale the summed light by `view.exposure` — the
  uniform is photometric (sun ≈ 10⁴ lux) and every PBR fragment applies
  the camera's exposure (`Exposure::BLENDER`, EV100 9.7 ≈ ×1.4e-3)
  before writing; skipping it rides the tonemap shoulder ~700× too
  high, which reads as "colourful at grazing sun angles, a white blob
  wherever NdotL nears 1" (the 302 m bench view — the second shipped
  bug), and (2) run the standard chain's in-shader `tone_mapping(color,
  view.color_grading)` under `TONEMAP_IN_SHADER` — or the raw value
  saturates the sRGB target and the whole sprite clips to white (first
  attempt shipped exactly that bug). Zero directional lights leaves
  ambient only — the night case. Alpha = coverage; after the exposure
  fix the 302 m bench shows billboard tint parity with the mesh-mode
  baseline (0.85% vs 0.76% tinted pixels, 0% near-white in both; the
  residue vs close-up meshes: PBR's specular rim and Burley vs plain
  Lambert). One material-side lighting input exists on top: the scalar
  `BillboardTuning.brightness` (F1 slider, 1.0 = parity) — a manual
  match knob for that residue, seeded into the shared materials at
  creation and pushed live by `sync_billboard_brightness`.

## 3. Pose axis in the atlas + bake (`src/preprocess/atlas.rs`, `mod.rs`)

- `atlas.rs`: `POSE_COUNT` and per-pose cells —
  `albedo_cell(view, pose) = (x, pose·HALF_ROWS + y)`, `normal_cell` stays
  its whole-texture 180° rotation, so `nuv = 1 - uv` keeps working
  globally for any pose count — the rotation covers the cell placement
  AND the image content (`blit_cell_rot180` composes the normal capture
  rotated; an upright blit pairs every albedo pixel with the opposite
  sprite point's normal and lights the billboard from the anti-sun
  side). Grid: `7 × (2·HALF_ROWS·POSE_COUNT)` —
  the shipped idle pose (49 views) packs exactly (448×896, zero waste);
  tests pin exact fit, the mirror per pose, and the band-stacking
  arithmetic for future pose counts.
- `mod.rs`: bake state machine iterates **variations × poses × views**
  (strictly serialized as now). `bake_poses()` ships only the idle pose
  (an early tilted "walk" placeholder proved the axis, then was dropped —
  unused poses only bloat the texture); the stage-boundary transform swap
  is the extension point future skeletal animation plugs into (each
  animation frame becomes a pose: bump `POSE_COUNT` in `atlas.rs` and its
  wgsl mirror, then re-bake); `center`/frustum stay fixed across poses so
  all poses of a variation share framing.

## Wiring

`mod billboard;` + plugin + systems in `main.rs`; `setup` assigns ids and
spawns via the catalog; `debug_ui.rs` gains a `tuned("Billboards")` section
(two sliders: swap 5..=500 m, hysteresis 0..=50 m) and an `impostor_lod`
toggle, exact `tuned()`/`slider()`/bypass recipe.

## Tests (inline, mini-app pattern like `zoom_app`/`app_with`)

1. far boid swaps to billboard components; near boid keeps mesh;
2. hysteresis band preserves state both directions;
3. swap-back restores exactly `catalog[variation_for(id)]` handles — and
   two boids with different ids keep different variations across a
   zoom-out/zoom-in cycle (the "switch places" regression test);
4. `variation_for` round-robin covers all variations, deterministic;
5. `pack_tag` yaw/pose round-trip; yaw updates preserve pose bits;
6. atlas: pose bands mirror correctly, exact-fit invariant for the shipped
   pose count and band stacking for future ones.

## Verification

- `cargo test`, wasm check; re-run `--preprocess` (3 atlases) and
  pixel-inspect (98 cells, per-variation tint visible).
- Run the game: zoom out — billboards swap in per-variation-tinted,
  normal-lit; zoom in — each soldier returns to *its own* mesh; F1 slider
  live-tunes. Drag-select unaffected.
- `--bench --shot` (bench pins ~302 m → exercises the billboard path).
- Carried caveat: the current RTS camera sits ~70° from vertical, so views
  clamp to the outer 30° ring until the far camera steepens.

## Out of scope (documented hooks)

Equipment system (separate), real soldier meshes replacing tints,
skeletal-animation pose enumeration, per-formation gating,
neighbor-cell blending.
