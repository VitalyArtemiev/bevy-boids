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
2. **Tracy and dynamic linking are native-only.** The `tracy` feature (and
   `dynamic_linking`, `trace`, `trace_tracy`) live under
   `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]` for a reason:
   tracy-client-sys cannot build for wasm. Keep the native/wasm dependency
   split intact; never enable those features for wasm targets.
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
6. **wasm-bindgen-cli is version-pinned in CI (0.2.126) to match
   Cargo.lock.** If the lockfile moves wasm-bindgen, update the CI pin in
   `.github/workflows/ci.yaml`. The bindgen output name `bevy_boids`
   (`--out-name`) is referenced by `assets/index.html`.
7. **Performance is a feature.** The sim aims to run ~200k boids. Never add a
   per-frame O(n²) loop over boids — use the kd-tree (`Res<NNTree>`:
   `k_nearest_neighbour`, `within_distance`, `within`) and
   `query.par_iter_mut()` like the existing systems do. There is a test with
   an explicit wall-clock budget (`nearest_solver_scales_to_10k_members`,
   ~7ms release for 10k) — don't weaken its budgets to make a change pass.

## Commands

- Dev run: `cargo run` (native builds already carry Tracy instrumentation
  via target deps).
    # "trace_tracy",
- Dev run with tracing: `cargo run --features trace_tracy`
- Profiling: `cargo run --release --features trace_tracy`, connect Tracy.
- Tests: `cargo test`; run perf-budget tests in release for realistic
  timings (`cargo test --release nearest_solver`).
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
| `main.rs` | App assembly, all schedules, `setup` (boids/obstacles/light/camera/ground) |
| `boid.rs` | `Boid`, `BoidBundle`, separation (`soft_collisions`), walls (`hard_collisions`), `bob` |
| `kinematics.rs` | `Velocity { v, a, push, target_v }`, tuning consts, `move_step` integrator, `NNTree`/`TrackedByTree` |
| `target.rs` | `Target` component, `follow_target` steering |
| `formations.rs` | `Formation`, `FormationKind` (Line/Column/Grid/Wedge/Ring), `FormationSlot`, relationship components, `FormationTask` queue, Morton-order slot assignment, LOD, most tests |
| `player.rs` | Selection state, drag-select, frontage designation, quick groups, selection gizmos, component hooks |
| `terrain.rs`, `resources.rs`, `util.rs` | Ground/obstacles; shared-handle Resources (`Meshes`, `Materials`); geometry helpers (`within_rect`) |
| `horse.rs` | Stub for future cavalry behavior |

## Scheduling conventions

- Input, selection, camera, per-frame animation, and the task executor
  (`process_formation_orders`, which also draws gizmos) run in `Update`.
  Formation bookkeeping (`assign_slots`, `propagate_formation_targets`,
  `follow_target`) runs in `FixedUpdate`.
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
- **Hierarchies are relationships**: `MemberOf`/`Members` and
  `FormationOf`/`Formations` via `#[relationship]`/`#[relationship_target]`;
  parenting uses `ChildOf`/`Children`.
- **Component presence is state**: a formation carries `Velocity` if it is
  the lowest loaded LOD level; `propagate_formation_targets` inserts/removes
  it. Don't add parallel bool flags.
- **Multi-view systems use `ParamSet` with named passes** — see
  `process_formation_orders` (pass A mutates state, then snapshots, then applies).
  This is the project's answer to borrow conflicts in big systems; prefer it
  over splitting data through `Local` temporaries.
- **Deferred structural changes use `commands.queue(move |world: &mut
  World| ...)`** when the change depends on values computed mid-system (Reform
  pop, sub-formation task injection in `formations.rs`).
- **Spawn/despawn side effects use component lifecycle hooks**: manual
  `impl Component` returning `on_insert()`/`on_remove()` hooks
  (`Selected` in `player.rs`). Observers exist in 0.19 but this codebase
  doesn't use them — reach for hooks/systems first for consistency.
- **Spatial membership is a component**: entities queried via the kd-tree
  need the `TrackedByTree` marker (see `SoftCollision`/`HardCollision`
  embedding it); the `AutomaticUpdate::<TrackedByTree>` plugin refreshes the
  tree periodically (tests configure a faster refresh for determinism).
- **Movement model**: steering writes `vel.a` / `vel.target_v`
  (`follow_target`, `soft_collisions`); `move_step` integrates
  semi-implicit-Euler and clamps. Never teleport entities from steering
  code; the intentional exceptions (formation origin snapping to center of
  mass, `bob` animating y) are marked by their comments.

## Testing conventions

- Headless `App` harness: `test_app()` in `formations.rs` builds a minimal
  app with the real `AutomaticUpdate` plugin and the pipeline `.chain()`ed;
  `tick(app, dt)` advances `Time` manually via `advance_by`.
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
  (see `Formation`, `process_formation_orders`) — keep that density for new ones.
- Tuning constants live as module-level `const`s near their system, with
  units in the name (`MAX_VELOCITY`, `DECELERATION_TIME_SEC`, `LEAD_TIME`).
- Comments recording measurements ("this is slower at 10k", "~7ms in
  release for 10k") are load-bearing — preserve them and add your own when
  you bench.
