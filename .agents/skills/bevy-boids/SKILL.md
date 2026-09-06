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
  `--secs N`, `--zoom 0..1`, `--shot <path>` (captures one screenshot
  near the end of the run via the renderer's own readback — visual
  verification that needs no desktop capture tooling and works on
  occluded/locked Wayland sessions, where compositor throttling makes
  the FPS number meaningless). Scenario flags work with or without `--bench`:
  `--boids N`, `--zoom-sweep low:high` (oscillate zoom, exercising LOD churn), `--shadows|--no-shadows`, `--terrain|--no-terrain`,
  `--atmosphere|--no-atmosphere`, `--env-map|--no-env-map` (implies
  atmosphere), `--bloom|--no-bloom`. Atmosphere is off by default (see
  `sky.rs`).
- Chrome trace of the bench (low-overhead: the full unfiltered trace drops
  the run to ~8 FPS):
  `RUST_LOG='warn,bevy_boids=trace,bevy_ecs::system::function_system=trace,bevy_ecs::schedule=trace,bevy_app=trace,bevy_render=trace,bevy_time=trace' TRACE_CHROME=trace.json cargo run --features chrome -- --bench --secs 5`
- Wasm check: `cargo check --target wasm32-unknown-unknown`.
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
| `launch.rs` | `LaunchConfig` + short CLI flags (`--bench`, `--secs`, `--zoom`, `--boids`, `--shadows`, `--terrain`, `--atmosphere`, `--env-map`, `--bloom`); `BenchPlugin` skips the main menu, pins camera zoom, disables camera input, exits after a duration and logs FPS |
| `boid.rs` | `Boid`, `BoidBundle`, separation (`soft_collisions`), walls (`hard_collisions`), `bob`, `BoidTuning` |
| `kinematics.rs` | `Velocity { v, a, push, target_v }`, tuning consts + `KinematicsTuning`, `move_step` integrator, `NNTree`/`TrackedByTree` |
| `target.rs` | `Target` component, `follow_target` steering |
| `formations.rs` | `Formation`, `FormationKind` (Line/Column/Grid/Wedge/Ring), `FormationSlot`, relationship components, `FormationOrder` queue, `SlotsStale`/`FormationGoal` message components, the chained executor pipeline (`transition_formation_orders`, `plan_formation_goals`, `dispatch_formation_goals`), Morton-order slot assignment, LOD, `FormationTuning`, most tests |
| `player.rs` | Selection state, drag-select, frontage designation, quick groups, selection gizmos, component hooks, `height_scaled_zoom` (constant-ratio wheel steps: `max(h × 0.125, 0.5 m)` per notch × the Options zoom-speed multiplier; easing into max zoom; tuned by feel, half the original 0.25 ratio). The crate's own zoom MUST stay neutralized (`zoom_sensitivity: 0`, see `apply_options`) — its constant-units step (7.5 km/notch) stacks on top of ours and dominates near the ground |
| `sky.rs` | `SkyPlugin`: directional sun + opt-in atmosphere (`Atmosphere`, `ScatteringMedium`, `SunDisk`, `Bloom`), `SkyTuning` + live `update_sun`, 128 px env-map default (atmosphere stays off because Bevy 0.19 refilters it every frame; upstream #24522/#24738 track the fix/lower default); pure `sun_transform` helper with unit tests |
| `ui.rs` | `UiPlugin`: egui (`bevy_egui`), `GameState { MainMenu, Playing, Paused }`, main/pause menus, Options menu (`OptionsSettings` persisted via bevy-settings, `apply_options` pushes to live values), terrain-tuning panel on F3 (world-gen sliders + the `LodMode` and `CameraMode` debug toggles), pause plumbing (`Time<Virtual>` + camera kill-switch) |
| `debug_ui.rs` | `DebugUiPlugin`: F1 debug panel — sliders over the `*Tuning` resources, gizmo/LOD/sky toggles, FPS; `DebugConfig` consumed by gizmo systems |
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
  Menus pause `Time<Virtual>` (via `ui.rs`) so the fixed clock doesn't
  accumulate a catch-up storm, and disable the camera via
  `RtsCameraControls.enabled`.
- All egui systems live in the `EguiPrimaryContextPass` schedule (required
  by `bevy_egui` 0.42), take `EguiContexts`, and may return `Result`. Panels
  need a hand-built root `egui::Ui` (see `root_ui` in `ui.rs`); `Window::show`
  takes the `&Context` directly.
- **egui panels over `ResMut` must bypass change detection**: any `&mut`
  through a `ResMut`/`Mut` (even an unused reborrow like `&mut *res` to
  satisfy widget signatures) marks the resource/component changed every
  frame the panel is open. With change-detection-driven consumers
  (`resource_changed` conditions, `is_changed()` early-returns) that means
  per-frame rebuilds — the F3 terrain panel was regenerating the entire
  world at frame rate this way. Pattern (see `tuned()` in `debug_ui.rs`,
  `terrain_tuning_ui`/`options_ui` in `ui.rs`):
  `res.bypass_change_detection()` for the widgets, accumulate
  `Response::changed()` from every widget, call `res.set_changed()` only if
  something actually changed.
- The whole formation pipeline runs in `FixedUpdate`, explicitly chained
  (see `main.rs`): `init_formation_speed` → `propagate_formation_targets`
  (LOD `Velocity` state) → `transition_formation_orders` (task state
  machine, inserts `SlotsStale`) → `assign_slots` (Morton re-mapping, pops
  finished `Rotate`/`Reform`) → `plan_formation_goals` (writes
  `FormationGoal`) → `dispatch_formation_goals` (origin snap, arrival pops,
  `Target` writes / sub-order injection, gizmos) → `follow_target`.
- Pipeline order is load-bearing: collisions → task propagation →
  `follow_target` → `move_step`. Express ordering explicitly where it
  matters (`.after(soft_collisions)` in `main.rs`); tests pin full order
  with `.chain()`. Read `main.rs` before adding a system and place it in the
  right schedule.

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
  system (`soft_collisions` in `boid.rs`). This codebase has zero async
  code — don't reach for async/await, tokio, or an executor crate to "get
  concurrency"; parallel ≠ async, and systems can't be `async fn`. The
  escape hatch for genuinely background work (asset warmup, network
  interop) is `bevy::tasks` (`AsyncComputeTaskPool`) with results fed back
  into a system each frame; propose that before adding it.
- **Movement model**: steering writes `vel.a` / `vel.target_v`
  (`follow_target`, `soft_collisions`); `move_step` integrates
  semi-implicit-Euler and clamps. Never teleport entities from steering
  code; the intentional exceptions (formation origin snapping to center of
  mass, `bob` animating y) are marked by their comments.

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
  from the system, add one `Slider::new` line in `debug_ui.rs`. Pure tuning
  stays as plain consts.
- **User-facing options** (`OptionsSettings` in `ui.rs`) are separate from
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
