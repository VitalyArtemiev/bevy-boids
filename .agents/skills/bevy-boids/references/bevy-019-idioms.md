# Bevy 0.19 idioms (vs. the pre-0.16 habits most online material teaches)

Every "write this" column entry is spelled exactly as used in this repo, so
copying it is safe. When something you need isn't listed here or in `src/`,
check the Bevy 0.19 docs — do not fall back to older-API memory.

| If you're about to write (≤0.15 era) | Write instead (0.19, as used here) | Where here |
| --- | --- | --- |
| `Res<Input<KeyCode>>`, `Input<MouseButton>` | `ButtonInput<KeyCode>`, `ButtonInput<MouseButton>` | `player.rs` |
| `Parent` component for parenting | `ChildOf` relationship component (+ `Children` target) | `player.rs` |
| Custom hierarchies with manual `Entity` bookkeeping | `#[relationship(relationship_target = Members)]` / `#[relationship_target(relationship = MemberOf)]` | `formations.rs` |
| `Handle<Mesh>` / `Handle<StandardMaterial>` as bundle/component fields | `Mesh3d(handle)` / `MeshMaterial3d::<StandardMaterial>(handle)` | `boid.rs` |
| `PointLight { shadows_enabled: .. }` | `shadow_maps_enabled: true` | `main.rs` |
| `time.delta_seconds()` | `time.delta_secs()` | `kinematics.rs` |
| `Color::rgb(..)` / `Color::rgba(..)` | `Color::srgb(..)` (colorspace-explicit constructors); named constants via palettes, e.g. `bevy::color::palettes::basic::YELLOW` | `player.rs`, `formations.rs` |
| `v.try_normalize().unwrap_or(Vec3::ZERO)` | `v.normalize_or_zero()` | everywhere |
| Spawn-side effects via observer-less `Added<T>` polling or ad-hoc systems | component lifecycle hooks: manual `impl Component` (`const STORAGE_TYPE`, `type Mutability = Mutable`) with `on_insert()` / `on_remove()` returning a `ComponentHook` taking `(&mut DeferredWorld, HookContext)` | `player.rs` (`Selected`) |
| Synthetic entity ids in tests: `Entity::from_raw(i)` | `Entity::from_raw_u32(i).unwrap()` (returns `Option` in 0.19) | `formations.rs` tests |
| Cramming a read-after-write pipeline into one system with `ParamSet` passes | chained systems passing small marker/data components (`SlotsStale`, `FormationGoal`) — the chain's auto sync points flush `Commands` between stages; keep `ParamSet` only when one system genuinely needs overlapping access in one pass | `formations.rs` executor pipeline |
| Mutating another entity's state mid-iteration via `World` access hacks | `commands.queue(move \|world: &mut World\| ...)` deferred closure | `formations.rs` |
| `async fn` systems, tokio/futures to "get concurrency" | Plain sync systems — the scheduler runs non-conflicting systems in parallel; `par_iter_mut()` for per-entity loops | `soft_collisions` in `boid.rs` |

Notes:

- "Parallel by default" ≠ async: systems are synchronous functions run
  concurrently by the multithreaded scheduler (there is no first-class
  async system support in Bevy). Async exists only at the engine level via
  `bevy::tasks` task pools for background work — see the parallelism bullet
  in SKILL.md before reaching for it.
- Both tuple-spawns (`commands.spawn((A, B, C))`) and derived `Bundle`
  structs with constructors (`BoidBundle::with_target`) are in use; either is
  fine — bundle constructors are preferred when spawn sites repeat.
- Required components (0.16+) are in use: `#[require(NeedsSpeedInit,
  Target)]` on `Formation` guarantees every spawn path gets them. Prefer
  `require` over repeating components at each spawn site.
- Observers (`Trigger`, `.observe(..)`) exist in 0.19 but are unused here;
  prefer the existing hook/system/queued-task patterns unless asked.
- wasm: don't force wgpu backends. `main.rs` pins `Backends::VULKAN` only
  under `#[cfg(not(target_arch = "wasm32"))]`; browsers must let wgpu pick
  (WebGL2/WebGPU). Keep any new render settings wasm-conditional if they
  touch backends. The atmosphere (`Atmosphere`/`ScatteringMedium`) builds
  its lookup textures with compute shaders, so the sky needs WebGPU on
  wasm — WebGL2 keeps sun lighting but falls back to the flat `ClearColor`
  sky (see `sky.rs`).
- Extra fields on a relationship component are a footgun upstream:
  re-inserting the relationship silently resets them to `Default`
  (bevy#19589). Keep companion data as its own component — `FormationSlot`
  deliberately stays separate from `MemberOf`; see its doc comment.
