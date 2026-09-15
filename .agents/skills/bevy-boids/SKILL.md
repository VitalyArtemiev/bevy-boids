---
name: bevy-boids
description: Conventions, hard constraints, and build commands for this Bevy 0.19 boids/RTS simulation project. Use for ANY task in this repo that touches Rust source, Cargo.toml or .cargo config, builds, tests, wasm deploys, or performance — writing or editing systems, components, bundles, or plugins; adding or changing dependencies; debugging; profiling. Trigger even when "Bevy" is not mentioned explicitly, since almost all code here is Bevy code.
---

# bevy-boids — project conventions

End goal is a massive-scale RTS game with ~200k units on screen (some are
rendered as simple billboards or abstracted as formations where the whole
formation might be rendered as two triangles with fancy shaders) at the same
time, not counting background simulation. The scale will vary from viewing the
whole continent to being present on the battlefield among the units. To this
end, everything should be subject to strict LOD culling. For example, it should
be possible to quickly load/unload individual boids from the formation if the
distance changes. Almost every simulated system should be able to be
approximatex on every LOD level. For example, when individual boids are loaded,
we track health, energy and morale for them individually. On unload, we save
average data/approx. distribution of health/tiredness/morale into formation. On
load we again put the data into boids. The same should be true about civillian
population in cities. You should be able to follow a laborer as they go about
their day in a city at small scale, but zoomed all the way out it should fall
back to a rough economic output simulation. Units behave boid-like in most
circumstances to impart the feeling of weight and fluid crowd dynamics. They
should generally behve like they are avoiding collissions, unless it is a
frontal assault charge in battle, and even then it might depend on morale.
Performance is a significant concern, use existing kd-tree from bevy_spatial and
propose other acceleration structures when applicable. Units are subdivided into
a hierarchy of formations, there can be formations of formations to define
complex maneuvers. Each formation has a queue of orders formation hierarchy with
task queues, drag-selection and frontage designation, RTS camera, spatial
kd-tree queries. Ships native (Windows/Linux dev) and to the
web via GitHub Pages (wasm). Single crate, one module per domain in `src/`,
tests inline under `#[cfg(test)]`. This file distills the rules the codebase
already follows — when it and the code disagree, the code wins; fix this file.

## Ground rule: Maintainability

Important: code needs to be extremely readable, extendable and maintainable.
This doesn't mean 'more comments' - instead, bevy systems should be small,
understandable, easily testable. If a system needs to be long and complex, split
inner logic into independably-testable functions. Bevy infrastructure is
well-designed - use it whenever reasonable instead of trying to cram everything
into one complex system. We need compartmentalization, message passing via
inserting components etc. No sphagetti code. When in trouble, look at bevy
examples and bevy cheatbook - principles are still the same even api is
different.

## Ground rule: the codebase is the Bevy 0.19 source of truth

This repo compiles against Bevy 0.19. Most tutorials, forum answers, and
training-data-era code describe Bevy ≤ 0.15, whose APIs differ pervasively
(input, parenting, asset components, naming). Before using any Bevy API from
memory, grep `src/` for it and copy the in-repo spelling. If the codebase is
silent, check `references/bevy-019-idioms.md` (read it before writing new
Bevy code), and verify against the 0.19 docs rather than guessing.

Custom material shaders (the `crowd.rs` / wgsl stack) have two silent
failure modes that cost a full debugging session each:

- A custom wgsl module (`#define_import_path bevy_boids::foo`) is only
  visible to `#import bevy_boids::foo` if the module's `.wgsl` is actually
  a *loaded* asset — nothing references it by path, so it must be
  preloaded (`CrowdCommonShader` + the `crowd_module_loaded` spawn gate in
  `crowd.rs`; the deleted terrain render module needed the same resource).
  Otherwise the pipeline silently never specializes: nothing draws, no
  error is logged.
- naga-oil shader errors DO get logged by `pipeline_cache` as
  `failed to process shader error:` — but the message spans two lines, so
  grepping logs for `error|naga` with `head` can hide it behind cargo
  warnings. Grep for `failed to process` specifically, or run with
  `RUST_LOG` trace. The classic trigger: a type imported from a module
  must be fully qualified in *every* signature (`fn fragment(in:
  crowd_common::CrowdOut)`, not bare `CrowdOut` — WGSL has no type
  aliases and no implicit module-name resolution).

## Hard constraints — breaking these breaks the build or CI

1. **Bevy 0.19 + Rust nightly, both pinned.** Nightly is required:
   `.cargo/config.toml` uses `-Zthreads` (Linux target) and
   `[unstable] codegen-backend = true` (cranelift for `profile.dev`).
   `rust-toolchain.toml` pins nightly and CI sets `RUSTUP_TOOLCHAIN`.
   Don't move the project to stable or reorder profiles.
2. **Dynamic linking and tracing are opt-in, native-only conveniences.**
   No Cargo feature is enabled by default. `dynamic` adds Bevy's dylib for
   fast iteration; `tracing` adds the base span layer; `chrome`, `tracy`, and
   `tracy_mem` add tracing sinks. Never enable `dynamic`, `tracy`, or
   `tracy_mem` for wasm (tracy-client-sys cannot target wasm). `dynamic` is
   also incompatible with the LTO release profile — release builds stay
   static by default.
3. **wasm needs the getrandom backend cfg.** `.cargo/config.toml` sets
   `--cfg getrandom_backend="wasm_js"` for `wasm32-unknown-unknown`, matching
   the `getrandom = { features = ["wasm_js"] }` dep. Removing either breaks
   the wasm build.
4. **CI checks native AND wasm on every push/PR** (`cargo check` and
   `cargo check --target wasm32-unknown-unknown`). Both must stay green; a
   change is not done until both pass locally. CI needs
   `rustup component add rustc-codegen-cranelift-preview` even for `check`
   (profile.dev selects cranelift).
5. **`bevy_spatial` is a fork, not the crates.io crate** — git dependency on
   the `within` branch of `VitalyArtemiev/bevy-spatial`, `kdtree` feature
   only. Selection (`within_rect`) and obstacle queries use its AABB `within`
   API, which upstream lacks. Don't "fix" the manifest back to crates.io.
6. **`bevy_rts_camera` is vendored at `vendor/bevy_rts_camera`** (upstream
   0.14.0 plus one change): the per-frame `follow_ground` mesh raycast is
   removed because it AABB-culled the whole mesh world (~5 ms/frame with 10k
   boid meshes). The game sets focus height analytically in
   `terrain::camera::focus_camera_on_ground`; `Ground` only marks drag-pan
   grab anchors. Don't swap the manifest back to crates.io.
7. **wasm-bindgen-cli is version-pinned in CI (0.2.128) to match
   Cargo.lock.** Cargo.lock is committed and CI cargo invocations use
   `--locked`, so local and CI resolve identical versions. wasm-bindgen moves
   as a lockstep family (wasm-bindgen, wasm-bindgen-futures, js-sys, web-sys
   exact-pin each other): update all four together
   (`cargo update -p wasm-bindgen -p wasm-bindgen-futures -p js-sys
   -p web-sys`), and update the CI CLI pin in `.github/workflows/ci.yaml` in
   the same change. `scripts/check-wasm-bindgen-pin.sh` enforces the invariant:
   it runs as `.githooks/pre-commit` (activate per clone with
   `git config core.hooksPath .githooks`) and as the first step of CI's check
   job. The bindgen output name `bevy_boids`
   (`--out-name`) is referenced by `assets/index.html`.
8. **Performance is a feature.** The sim aims to run ~200k boids. Never add a
   per-frame O(n²) loop over boids — use the kd-tree (`Res<NNTree>`:
   `k_nearest_neighbour`, `within_distance`, `within`) and
   `query.par_iter_mut()` like the existing systems do. There is a test with
   an explicit wall-clock budget (`nearest_solver_scales_to_10k_members`,
   ~7ms release for 10k) — don't weaken its budgets to make a change pass.

## Commands

- **Iterate with `cargo run --features dynamic`** — Bevy as a dylib, builds
  in a small fraction of the static time. `cargo build` (static) and
  `--release` are for profiling and shipping only; don't sit through
  release builds while experimenting.
- Static dev run: `cargo run` (the default; also what release uses).
- Fast iteration dev run: `cargo run --features dynamic` (native only).
- Base tracing spans: `cargo run --features tracing` (native only).
- Chrome trace: `cargo run --features chrome` (native only; implies tracing).
- Tracy: `cargo run --features tracy` (native only; implies tracing).
- Tracy + memory: `cargo run --features tracy_mem` (native only).
- Fast iteration with a Tracy sink: `cargo run --features dynamic,tracy`.
- Profiling: `cargo run --release --features tracy`, connect Tracy.
- Release run (static LTO, no dev features): `cargo run --release`.
- Tests: `cargo test`; run perf-budget tests in release for realistic
  timings (`cargo test --release nearest_solver`).
- Bench: `cargo run -- --bench` (plays 30 s with no main menu, camera
  pinned at zoom 0.99 ≈ 302 m, logs FPS on exit). Bench-only values:
  `--secs N`, `--zoom 0..1`, `--cam-angle <deg>` (pin the view angle from
  nadir, 0 = straight down — default dynamic angle swings to 72° at high
  zoom, outside the crowd experiment's 30° design cone),
  `--shot <path>` (captures one screenshot
  near the end of the run via the renderer's own readback — visual
  verification that needs no desktop capture tooling and works on
  occluded/locked Wayland sessions, where compositor throttling makes
  the FPS number meaningless). Scenario flags work with or without `--bench`:
  `--boids N`, `--zoom-sweep low:high` (oscillate zoom, exercising LOD churn), `--shadows|--no-shadows`, `--terrain|--no-terrain`,
  `--atmosphere|--no-atmosphere`, `--env-map|--no-env-map` (implies
  atmosphere), `--bloom|--no-bloom`, `--flat` (bare plane ground, no
  obstacles — for render-path benches), `--force-meshes`/`--force-billboards` (hold every
  boid on one render path, auto-swap off; last flag wins), `--no-vsync`
  (`PresentMode::Immediate` — light scenes ride the refresh ceiling at 60
  fps and compare nothing), `--sun-azimuth <deg>`/`--sun-elevation <deg>`
  (seed `SkyTuning` for reproducible lighting; F1 sliders still work
  live), `--scene <name>` (standalone test scene instead of the normal
  RTS launch — currently `billboard` (implies `--flat`, suppresses the
  RTS camera and grid boids, and stands the billboard auto-swap down;
  a mesh capsule on the left beside its forced-billboard twin on the
  right up close under the same sun — pair with the sun flags and
  `--shot` to compare the render paths) and `crowd` (the crowd-shell
  experiment: two armies rendered by the procedural crowd shader on
  rolling test hills, through the game's own RTS camera — pair with
  `--cam-angle`/`--shot` for captures inside the 30° design cone; the
  ground and armies are CrowdPlugin's own, gated on the scene; see
  `scene.rs` to add more). Render-path A/B recipe:
  `cargo run --features dynamic -- --bench --secs 12 --boids 5000 --flat --cam-angle 25 --zoom 0.99 --no-vsync --force-meshes`
  vs the same with `--force-billboards` (see the measured numbers in
  `billboard.rs`'s module docs).
  Atmosphere is off by default (see `sky.rs`).
- Chrome trace of the bench (low-overhead: the full unfiltered trace drops
  the run to ~8 FPS):
  `RUST_LOG='warn,bevy_boids=trace,bevy_ecs::system::function_system=trace,bevy_ecs::schedule=trace,bevy_app=trace,bevy_render=trace,bevy_time=trace' TRACE_CHROME=trace.json cargo run --features chrome -- --bench --secs 5`
- Wasm check: `cargo check --target wasm32-unknown-unknown`.
- Impostor atlas bake (native tool mode): `cargo run -- --preprocess` —
  bakes every catalog variation (unlit albedo + view-space normals, one
  animation pose axis, packed 180°-mirrored into one zero-waste texture per
  variation) into RGBA PNGs under `assets/impostors/` and exits; never
  enters the game's plugin graph.
- Web build (what CI deploys to Pages):
  `cargo build --profile wasm-release --target wasm32-unknown-unknown`, then
  `wasm-bindgen --target web --out-dir dist --out-name bevy_boids
  target/wasm32-unknown-unknown/wasm-release/bevy-boids.wasm`, then assemble
  the site: `dist/` + `assets/index.html` + copy `assets/` into
  `dist/assets/`.

## Architecture map

| Module | Contents |
| --- | --- |
| `main.rs` | App assembly, all schedules, `setup` (boids/obstacles/camera/ground; the sun/sky lives in `sky.rs`) |
| `ui/input.rs` | `InputPlugin`: the single `PlayerContext` enhanced-input context with all gameplay actions (`Select`, `Frontage`, quick groups 1-6, `Pause`, panel toggles), `ActionId` identity + defaults, `BindingsSettings` persistence (bevy-settings), rebind logic (`rebind`/`replace_binding` — `Binding` is immutable, so rebinds despawn/respawn binding entities) and the Key Bindings window (capture next press, conflict swap, reset). Poll helpers: `started`/`fired`/`completed` over `Query<(&ActionTag, &TriggerState, &ActionEvents)>` |
| `ui/radial.rs` | `RadialPlugin`: radial context menu. RMB *click* (press+release within `CLICK_TOLERANCE_PX`, with a selection) opens `RadialMenu` (resource presence = state; `radial_closed` gates gameplay input systems while open). egui painter wedges, hover navigation, sub-rings exclude the parent-direction wedge (that gap + the inner hole = back). Commands: Walk (keep formation / reform into 4-wide Grid column then restore via `Reform{kind, columns}` orders) and Run (`Target::speed_scale` walk 0.5 / run 1.0) |
| `launch.rs` | `LaunchConfig` + short CLI flags (`--bench`, `--secs`, `--zoom`, `--cam-angle`, `--boids`, `--shadows`, `--terrain`, `--atmosphere`, `--env-map`, `--bloom`, `--flat`, `--force-meshes`/`--force-billboards`, `--no-vsync`, `--sun-azimuth`/`--sun-elevation`, `--scene`, `--preprocess`); `BenchPlugin` skips the main menu, pins camera zoom, disables camera input, exits after a duration and logs FPS |
| `scene.rs` | Standalone test scenes (`--scene <name>`, `ScenePlugin`): a scene replaces the normal RTS launch with one hand-built micro-scene, running straight into `GameState::Playing` (no main menu — the `--bench` bypass without the bench behaviours) — it owns its camera and props, `setup` skips the RTS camera/grid boids/obstacles while any scene is active, and `DebugConfig.impostor_lod` stands the billboard auto-swap down so each boid stays on the render path the scene gave it. `TestScene` is the name registry (`ALL`/`name`/`from_name`); `in_scene(variant)` is the run-condition factory for the spawn systems; `wants_flat_ground()` gates the flat plane in `main.rs`. Movement scenes (`arrival`, `perpendicular`, `head-on`, `clash`, `shove`, `melee`, `formation-cross`, `formation-braid` (the head-on jam stress test), `formation-clash` — growing PBD-collision complexity, one unit from standstill up to hostile formations marching through each other) share helpers: `spawn_scene_boid` (ids pick faction-matching colours: `id = faction + 3·seq`), `spawn_marching_block` (Grid block + `Move` order), `scene_camera` (RTS camera pinned via the one-shot `SceneCameraPose`; `height_m` converts to zoom units where 1.0 = 2 m and 0.0 = 30 km). The `billboard` scene: a mesh capsule (screen LEFT) beside its billboard twin (screen RIGHT) on the flat plane, both still at yaw 0, plain camera 25° from nadir ~7 m out — the module docs list the four steps to add a scene (variant → spawn system → `run_if` registration → any main-setup accommodations). The `crowd` scene: the crowd-shell experiment viewed through the game RTS camera (`scene::spawn_crowd_scene` calls `spawn_rts_camera`); its hills ground and armies are CrowdPlugin systems gated on `in_scene(TestScene::Crowd)`. Visual captures: `--bench --scene=<name> --zoom ~0.998 --cam-angle 25 --shot <path>` (the shot fires at `--secs` − 5 s, so pass `--secs 12` to catch t≈7 s) |
| `crowd.rs` | `CrowdPlugin` — the formation crowd-shell experiment (`--scene=crowd`, isolated scene: gently rolling test hills from `crowd_test_height` replace the erosion terrain; ground mesh, `HeightField` and a baked heightmap texture all share that one function). A whole massed formation renders as one deformable box plus a procedural crowd shader instead of one draw per unit. `crowd_box` builds the subdivided bounding-volume box (unit height + swell headroom, 2 m vertex grid for CPU deformation later). The box rides the terrain: vertex + fragment stages displace by `terrain_h_rel` (the baked R8Unorm heightmap, bindings 3-6 in `crowd_common.wgsl`, shared by both materials with per-army origin/yaw/seat and the footprint's trace y-range), and soldier feet sample the same heights. The fragment shader walks the view ray through a jittered soldier grid (cylinder body + helmet sphere per cell — exact per-cell tests because jitter keeps every disk inside its cell), warping z by the frontline surge so the whole column breathes without any soldier popping (occupancy rolls are static). It shades helmet/shoulder disks from nadir, body-wall silhouettes at the cone rim, and periodic armour/spear glints (a per-soldier emissive flash carries the shimmer — pure specular can't fire from nadir); rays beneath occupied cells shade dark crowd shadow, every other ray is discarded — the material is alpha masked, so the box is invisible against the terrain. `CrowdDustMaterial` rides a taller translucent box above each crowd, gated to the churn band around the front line. Two opposing armies face off near the origin. Shaders: `assets/shaders/crowd.wgsl` + `crowd_dust.wgsl` over shared `crowd_common.wgsl`. `CrowdTuning` (F1) pushes density/glint/dust into the material uniforms |
| `preprocess/` | The `--preprocess` far-LOD impostor baker (standalone app, never added to the game). `atlas.rs` — pure layout: views sampled as concentric rings inside `MAX_ANGLE_FROM_VERTICAL` (30°) from nadir (nadir + rings at 10°/20°/30° with 8/16/24 azimuth slots; `view_index` is the (polar, azimuth) → index lookup and the wgsl twin's test-pinned reference; boid yaw folds into view azimuth because models are upright — no separate yaw axis; `FIT_MARGIN` sizes both the bake frustum and the runtime quad). Packing: each pose's albedo cells band the top half flat row-major (`POSE_COUNT` per variation), normal cells sit at the whole-texture 180° rotation with their content rotated too (`blit_cell_rot180`; `nuv = 1 - uv` pairs them for any pose — an upright normal blit lights billboards from the anti-sun side) — 7×14 cells for the shipped idle pose × 49 views = zero waste; tests pin the exact fit and the band-stacking arithmetic for future pose counts. `mod.rs` — the bake walks the shared `boid::boid_variations` catalog (unlit clones of the game materials) × `bake_poses()` (the idle pose today; the stage-boundary transform swap is where skeletal animation frames land — bump `POSE_COUNT` in `atlas.rs` AND its wgsl mirror and re-bake) × views: two co-located orthographic cameras on separate render layers, `Screenshot::image` captures both targets per view (probes repeat until non-blank on BOTH pipelines — compile time outlasts any fixed warmup), strictly serialised dispatch pairs captures with cells, one `assets/impostors/<name>.png` per variation |
| `boid.rs` | `Boid { id }` (stable spawn-order identity from `BoidIds`; `variation_for(id)` deterministically picks the visual variation — soldiers never "switch places" across LOD swaps or future streaming), the `BoidVariations` catalog (`FromWorld`; three tinted-capsule placeholders prove the machinery; the baker and spawner share this ONE list), `BoidBundle::with_id` (includes `Faction` + `Body` for the PBD solver), `bob`, `BoidTuning` |
| `billboard.rs` | `BillboardPlugin` — per-boid impostor LOD (design: `docs/plans/impostor-billboard-lod.md`). `swap_boid_lod` (Update, `.after(RtsCameraSystemSet)`, gated by `DebugConfig.impostor_lod`) detaches `Mesh3d`/material beyond `BillboardTuning.swap_distance_m` and attaches a quad + per-variation atlas `BillboardMaterial` + `MeshTag` (shared `attach_billboard` also serves the one-shot `--force-billboards` bench override and the `--scene billboard` twin; force flags and scenes stand the swap down via `DebugConfig`); restore inside `swap - hysteresis` pulls the exact catalog handles for `variation_for(id)` — identity is the id, no `ModelLod` component. `update_billboard_yaw` refreshes the tag's yaw bits from `Target.dir` (formation facing) falling back to velocity. The shader (`assets/shaders/impostor_billboard.wgsl`, custom `Material`, `AlphaMode::Mask(0.5)` — cutout keeps the opaque pass and automatic instancing: 5k `--flat` bench 35.8 blended / 70.0 cutout / 60.0 meshes; the fragment `discard`s the cutoff itself (a custom shader without it writes the atlas's black background), and `AlphaToCoverage` + `Msaa::Sample4` is the soft-alpha variant at equal speed if 4× MSAA scene-wide is acceptable; no prepass/shadows, cull off) builds the camera-facing quad in the vertex stage from `MeshTag` (yaw + pose bits), folds yaw into view azimuth against the mirrored `atlas.rs` consts, displays the sprite upright with the image u axis along screen-right (world up projects up-image in every baked cell and up-screen at runtime — yaw lives entirely in cell choice; the un-mirrored u axis is load-bearing: mirroring the display is invisible on symmetric albedo but flips the baked normals against the runtime view basis, lighting the sprite from the wrong side horizontally — it pairs with the normal half's 180° content rotation, each alone re-opens the opposite axis's bug), offsets by the yaw-rotated bake centre, and lights the unlit albedo with the mirrored view-space normal reading ambient + directional sun straight from the engine's `lights` view binding (PBR's own source: colour premultiplied by illuminance, Lambert `dot/π` ≈ Burley for the matte capsules) — one lighting authority for meshes and billboards (lighting itself needs no sync system; the one material-side input is the scalar brightness knob), and billboards darken with the scene at night instead of glowing at a baked ambient floor; the uniform is photometric HDR (sun ≈ 10⁴ lux), so the fragment tail mirrors `pbr_functions.wgsl` twice — scale the summed light by `view.exposure` (the default `Exposure::BLENDER`, EV100 9.7 ≈ ×1.4e-3, is what makes PBR output display-scale; without it sprites ride the tonemap shoulder ~700× too high — colourful at grazing sun angles, white blobs wherever NdotL nears 1, i.e. exactly the 302 m bench), then run the in-shader `tone_mapping(color, view.color_grading)` under `TONEMAP_IN_SHADER` (without it the raw value saturates the sRGB target and clips to white); 302 m bench after both steps: 0.85% tinted pixels / 0% near-white vs the mesh-mode baseline's 0.76% / 0%; `BillboardTuning.brightness` (F1 slider, 0..=3×, 1.0 = parity) is the one material-side lighting input — an eyeball-match knob for the baked-normal residue, seeded into the materials at creation (`BillboardAssets::from_world`) and pushed live by `sync_billboard_brightness` (`resource_changed`-gated) |
| `kinematics.rs` | `Velocity { v, a, push, target_v, slope, slope_col }` (slope = cached uphill gradient, resampled per metre column), tuning consts + `KinematicsTuning`, `arrival_plan` (arrival-damped, misalignment-slowed preferred velocity — the anti-orbit fix), slope-aware `move_step` (downhill gravity, uphill thrust loss, downhill cap relaxation; sheds cap excess with bounded decel, never snaps), `NNTree`/`TrackedByTree` |
| `target.rs` | `Target { pos, dir, speed_scale }` (scale multiplies the tuning velocity cap — Walk/Run), `follow_target` desired-velocity steering (brakes toward the preferred velocity; see `kinematics::arrival_plan`) |
| `pbd.rs` | Position-based collision response (Weiss et al. MiG 2017, arXiv:1802.02673 — design record in `docs/pbd-anticipation.md`): `pbd_contact` (Update, `.after(move_step).before(ground_boids)`) projects boids out of overlap instead of adding forces — frictional contact with inverse-mass weighting (`Body { radius_m, mass_kg }` — heavy units get shoved less), Macklin-style positional friction (cone-clamped, no velocity reads), long-range time-to-collision anticipation for friendly pairs (`time_to_collision` quadratic; the §4.5 sidestep sheds the braking component so units slide past instead of halting), and infinite-mass obstacle-cuboid seating. Velocity picks corrections up as `Δv = Δx/Δt` at write-back. Clash mode: `Faction` (different ids hostile, symmetric) keeps contact but drops anticipation — bodies slam and grind; `Charging` drops anticipation against everyone. Solver: kd-tree candidate gather once per step (`k_nearest` + margin), Jacobi iterations over a reused flat SoA scratch (`PbdScratch`, double-buffered positions, `ComputeTaskPool::scope` chunk-parallel) — 1.3 ms at 10k boids in release (budget test `pbd_contact_scales_to_10k_boids`). Pure fns unit-tested; integration tests pin non-penetration, clash-hold, shove-by-mass, and the three-faction melee bound |
| `formations.rs` | `Formation`, `FormationKind` (Line/Column/Grid/Wedge/Ring), `FormationSlot`, relationship components, `FormationOrder` queue, `SlotsStale`/`FormationGoal` message components, the chained executor pipeline (`transition_formation_orders`, `plan_formation_goals`, `dispatch_formation_goals`), Morton-order slot assignment, LOD, `FormationTuning`, most tests |
| `player.rs` | Selection state, drag-select, frontage designation, quick groups, selection gizmos, component hooks, `height_scaled_zoom` (constant-ratio wheel steps: `max(h × 0.125, 0.5 m)` per notch × the Options zoom-speed multiplier; easing into max zoom; tuned by feel, half the original 0.25 ratio). The crate's own zoom MUST stay neutralized (`zoom_sensitivity: 0`, see `apply_options`) — its constant-units step (7.5 km/notch) stacks on top of ours and dominates near the ground |
| `sky.rs` | `SkyPlugin`: directional sun + opt-in atmosphere (`Atmosphere`, `ScatteringMedium`, `SunDisk`, `Bloom`), `SkyTuning` + live `update_sun`, 128 px env-map default (atmosphere stays off because Bevy 0.19 refilters it every frame; upstream #24522/#24738 track the fix/lower default); `sun_transform` + `flat_ambient` helpers with unit tests, shared with the impostor preprocessor |
| `ui/mod.rs` | `UiPlugin`: egui (`bevy_egui`), `GameState { MainMenu, Playing, Paused }`, main/pause menus, Options menu (`OptionsSettings` persisted via bevy-settings, `apply_options` pushes to live values), terrain-tuning panel on F3 (world-gen sliders + the `LodMode` and `CameraMode` debug toggles), pause plumbing (`Time<Virtual>` + camera kill-switch) |
| `ui/debug.rs` | `DebugUiPlugin`: F1 debug panel — sliders over the `*Tuning` resources, gizmo/LOD/sky toggles, FPS; `DebugConfig` consumed by gizmo systems |
| `freecam.rs` | `CameraMode { Rts, Free }` resource (F3 toggle) + `Freecam`: free-fly camera. `apply_camera_mode` (gated `resource_changed`) swaps by component presence — freecam removes `RtsCamera`/`RtsCameraControls` (standing the crate's systems down) parked on `Freecam::saved` for restoration; `freecam_move` is WASD/QE/RMB-drag/wheel input, inert without a `Freecam` camera |
| `terrain/` | THE terrain: the `bevy_erosion_filter` terrain demo recreated on the CPU over a flat-substrate 1 km² square (`demo.rs`, no LOD): a 3-octave gain-0.1 fBm base + the crate's `cpu::erosion_filter` carve gullies and the ridge map, the demo's albedo cascade (cliff/dirt/snow/sand/grass/trees/drainage) rides vertex colors on the 256×256 `TerrainMesh` grid, a translucent `WaterPlane` rides `water_level`, and the demo's 22 m world stretches uniformly by `WORLD_SCALE` (1000/22) so slopes and gully sizes match the reference at game scale. `DemoTerrain` (F3 panel: filter sliders, terrain scales, four view modes) marks dirty; `rebuild_terrain` re-evaluates all 66k vertices (~160 ms — deliberately on slider RELEASE, drags coalesce) and rebuilds the closure-backed `HeightField` from the same evaluation, so grounding (`grounding.rs`, `GroundY`), camera focus/clearance (`camera.rs`), obstacle seating and the cursor raycast agree with the render by construction. Barebones pieces in `mod.rs` (`grid_mesh`, `spawn_ground`, `ObstacleBundle`); `rebuild_mesh`'s buffers-then-attributes shape is a borrow-checker requirement. The streamed-tile LOD system is deleted (`tiles.rs`/`noise.rs`/`render.rs`, see the a84a4f4 reset) |
| `resources.rs`, `util.rs` | Shared-handle Resources (`Meshes`, `Materials`); geometry helpers (`within_rect`) |
| `horse.rs` | Stub for future cavalry behavior |

## Scheduling conventions

- Input, selection, camera, and per-frame animation run in `Update`, along
  with the variable-step integrator (`move_step`) and collisions.
- Camera focus terrain-following is ours, not the crate's:
  `focus_camera_on_ground` samples `HeightField` before
  `RtsCameraSystemSet`; `camera_terrain_clearance` lifts the camera body
  after it.
- Everything gameplay-facing is gated `run_if(in_state(GameState::Playing))`
  (see `main.rs`) — sim, input, and the FixedUpdate pipeline all stop in
  menus. Input systems additionally use
  `not(egui_wants_any_pointer_input)`/`not(egui_wants_any_keyboard_input)`
  from `bevy_egui::input` so egui keeps clicks/keys when a panel is open.
  Menus pause `Time<Virtual>` (via `ui/mod.rs`) so the fixed clock doesn't
  accumulate a catch-up storm, and disable the camera via
  `RtsCameraControls.enabled`.
- All egui systems live in the `EguiPrimaryContextPass` schedule (required
  by `bevy_egui` 0.42), take `EguiContexts`, and may return `Result`. Panels
  need a hand-built root `egui::Ui` (see `root_ui` in `ui/mod.rs`); `Window::show`
  takes the `&Context` directly.
- **egui panels over `ResMut` must bypass change detection**: any `&mut`
  through a `ResMut`/`Mut` (even an unused reborrow like `&mut *res` to
  satisfy widget signatures) marks the resource/component changed every
  frame the panel is open. With change-detection-driven consumers
  (`resource_changed` conditions, `is_changed()` early-returns) that means
  per-frame rebuilds — the F3 terrain panel was regenerating the entire
  world at frame rate this way. Pattern (see `tuned()` in `ui/debug.rs`,
  `terrain_tuning_ui`/`options_ui` in `ui/mod.rs`):
  `res.bypass_change_detection()` for the widgets, accumulate
  `Response::changed()` from every widget, call `res.set_changed()` only if
  something actually changed.
- **Gameplay input is actions, not raw keys**: systems poll
  `Query<(&ActionTag, &TriggerState, &ActionEvents)>` through the
  `ui/input.rs` helpers (`started` ≈ just-pressed, `fired` ≈ held,
  `completed` ≈ just-released). New actions = new `InputAction` struct +
  `ActionId` variant + default binding + a row in `spawn_input_context`.
  Raw `ButtonInput` reads are confined to the rebind capture, the camera
  plugin's own fields, and `freecam.rs` (not yet migrated). `Binding`
  components are immutable — rebinds go through `input::replace_binding`
  (despawn/respawn).
- The radial menu owns RMB *clicks*: `frontage_position_system` skips
  releases within `radial::CLICK_TOLERANCE_PX` of the press, and gameplay
  input systems carry the `radial_closed` run condition. New radial
  commands = `RadialCommand` variant + a wedge in `radial::menu_items` +
  handling in `radial::execute`.
- The whole formation pipeline runs in `FixedUpdate`, explicitly chained
  (see `main.rs`): `init_formation_speed` → `propagate_formation_targets`
  (LOD `Velocity` state) → `transition_formation_orders` (task state
  machine, inserts `SlotsStale`) → `assign_slots` (Morton re-mapping, pops
  finished `Rotate`/`Reform`) → `plan_formation_goals` (writes
  `FormationGoal`) → `dispatch_formation_goals` (origin snap, arrival pops,
  `Target` writes / sub-order injection, gizmos) → `follow_target`.
- Pipeline order is load-bearing. Per frame: the FixedUpdate chain above
  (… → `follow_target`, steering) runs first, then Update integrates
  `move_step` → `pbd_contact` (positional collision projection) →
  `ground_boids` → `bob`. Express ordering explicitly where it matters
  (`pbd_contact.after(move_step).before(ground_boids)` in `main.rs`); tests
  pin full order with `.chain()`. Read `main.rs` before adding a system and
  place it in the right schedule.

## ECS patterns in use — match them

- **Shared asset handles live in Resources** (`Meshes`, `Materials`) built
  once in `setup`; clone `Handle`s into spawned entities. Shared gizmo
  shapes are `GizmoAsset`s in Resources built via `FromWorld`
  (`SelectionGizmo` in `player.rs`).
- **Hierarchies are relationships**: one `MemberOf`/`Members` relationship
  covers all formation occupants — boids and sub-formations alike — via
  `#[relationship]`/`#[relationship_target]`; "is this occupant a
  sub-formation" is a `Has<Formation>` check, not a separate relationship.
  Parenting uses `ChildOf`/`Children`.
- **Component presence is state**: a formation carries `Velocity` if it is
  the lowest loaded LOD level; `propagate_formation_targets` inserts/removes
  it. Don't add parallel bool flags.
- **Required components guarantee creation-path invariants**:
  `#[require(NeedsSpeedInit, Target)]` on `Formation` means every spawn
  path gets the marker; `init_formation_speed` later derives `max_speed`
  from the member list and removes it. Data derived from relationship
  targets must be computed by a system on a later tick (targets only
  populate when spawn commands apply) — never set it by hand at spawn
  sites, never in a spawn hook.
- **Self-contained features are plugins**: `SkyPlugin` bundles its own
  startup system; `main.rs` just adds it. Prefer a plugin over loose
  `add_systems` calls when a module owns a cohesive feature.
- **Pipeline stages communicate through components, not snapshot vectors**:
  when a chain of read-after-write steps would force one giant `ParamSet`
  system (the former `process_formation_orders`), split it into chained
  systems that pass small marker/data components instead — `SlotsStale`
  (re-map needed) and `FormationGoal` (this tick's steering plan) in
  `formations.rs`. Presence/absence is the state; each system's queries are
  then non-overlapping and no `ParamSet` is needed. Extract the pure math
  (`plan_goal`) as world-free functions and unit-test those directly.
  `ParamSet` remains the tool when a single system genuinely must touch
  overlapping queries in one pass.
- **Deferred structural changes use `commands.queue(move |world: &mut
  World| ...)`** when the change depends on values computed mid-system (Reform
  pop, sub-formation task injection in `formations.rs`).
- **Spawn/despawn side effects use component lifecycle hooks**: manual
  `impl Component` returning `on_insert()`/`on_remove()` hooks (`Selected`
  in `player.rs` spawns indicator children; its `on_remove` cleans them up
  via `despawn_related::<Children>()`). Observers exist in 0.19 but this
  codebase doesn't use them — reach for hooks/systems first for
  consistency.
- **Spatial membership is a component**: entities queried via the kd-tree
  need the `TrackedByTree` marker (see `SoftCollision`/`HardCollision`
  embedding it); the `AutomaticUpdate::<TrackedByTree>` plugin refreshes the
  tree periodically (tests configure a faster refresh for determinism).
- **Parallelism comes free — don't add async.** Bevy systems are plain
  synchronous functions: the multithreaded scheduler already runs
  non-conflicting systems in parallel based on their data access, and
  `query.par_iter_mut()` parallelizes independent per-entity work within a
  system (the `pbd_contact` write-back in `pbd.rs`; its gather/iteration
  phases chunk plain slices through `ComputeTaskPool::scope` because they
  need per-index writes into SoA arrays, which query iterators can't give).
  This codebase has zero async
  code — don't reach for async/await, tokio, or an executor crate to "get
  concurrency"; parallel ≠ async, and systems can't be `async fn`. The
  escape hatch for genuinely background work (asset warmup, network
  interop) is `bevy::tasks` (`AsyncComputeTaskPool`) with results fed back
  into a system each frame; propose that before adding it.
- **Movement model**: planners write `vel.a` / `vel.target_v`
  (`follow_target`); `move_step` integrates semi-implicit-Euler with slope
  physics (gravity along the terrain, slope-scaled thrust and speed cap)
  and clamps — the steering cap sheds excess speed with bounded
  deceleration, never a snap; `pbd_contact` owns ALL collision response
  positionally (`Δv = Δx/Δt` pick-up at write-back) — steering never adds
  separation forces, the solver never steers. Never teleport entities from
  steering code; the intentional exceptions (formation origin snapping to
  center of mass, `bob` animating y, the PBD solver's own projections) are
  marked by their comments. Design record: `docs/pbd-anticipation.md`.

## Testing conventions

- Headless `App` harness: `test_app()` in `formations.rs` adds `TimePlugin`
  (the plugin owns the fixed-loop runner), sets `Time::<Fixed>` to the tick
  dt, inits the tuning resources (`FormationTuning`,
  `KinematicsTuning`, `DebugConfig` — systems take them as `Res`), and chains
  the FixedUpdate pipeline; `tick(app, dt)` drives time via
  `TimeUpdateStrategy::ManualDuration` so every tick runs exactly one
  FixedUpdate. Bevy's first `update()` never advances the clocks, so
  `test_app()` burns it before any entities exist — and a self-test
  (`harness_advances_one_fixed_step_per_tick`) pins that guarantee.
- If a system starts becoming complex, split it into testable fuctions with
  simple inputs and outputs, test it without instantiating the world if possible.
- Headless gizmos need `init_gizmo_group::<DefaultGizmoConfigGroup>()`,
  `GizmoConfigStore`, and `Assets<GizmoAsset>` resources — copy that block
  from `test_app()` when a tested system draws gizmos.
- Integration tests spawn real entities and tick to convergence (hundreds of
  ticks for a march), then assert on world state. Synthetic entity ids in
  pure unit tests: `Entity::from_raw_u32(i).unwrap()`.
- quickcheck is available in dev-deps for property tests.

## Style conventions

- Doc comments on public types/systems explain *why* and the invariants
  (see `Formation`, `dispatch_formation_goals`) — keep that density for new ones.
- Runtime-tunable values live in `*Tuning` resources (`KinematicsTuning`,
  `BoidTuning`, `FormationTuning`, `SkyTuning`, `TerrainTuning`) next to
  their systems, with units in field names and `Default`s derived from the
  module-level `const`s (which stay authoritative for spawn paths and
  tests). Exposing a value in the F1 debug panel = add the field, read it
  from the system, add one `Slider::new` line in `ui/debug.rs`. Pure tuning
  stays as plain consts.
- **User-facing options** (`OptionsSettings` in `ui/mod.rs`) are separate from
  dev tuning and persisted by Bevy's first-party `bevy-settings` crate
  (`bevy = { features = ["bevy_settings"] }`): TOML at
  `%LOCALAPPDATA%/<app name>/settings.toml` (browser localStorage on wasm).
  Gotchas: settings structs need `#[derive(Resource, Reflect, SettingsGroup)]`
  with **`#[reflect(Default, Resource)]`** (the `Resource` reflect attribute
  attaches type data bevy-settings unwraps on; missing it panics at startup),
  and the type must be registered *before* `SettingsPlugin` builds. Flow is
  one-way: options → live resources (`apply_options`, gated on
  `resource_changed`); debug-panel edits to live values don't persist. Save
  is explicit: `SaveSettingsDeferred` after UI changes.
- Comments recording measurements ("this is slower at 10k", "~7ms in
  release for 10k") are load-bearing — preserve them and add your own when
  you bench.
