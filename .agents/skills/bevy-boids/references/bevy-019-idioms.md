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
| Spawn-side effects via observer-less `Added<T>` polling or ad-hoc systems | component lifecycle hooks: manual `impl Component` with `on_insert()` / `on_remove()` returning a `ComponentHook` taking `(&mut DeferredWorld, HookContext)` | `player.rs` (`Selected`) |
| Synthetic entity ids in tests: `Entity::from_raw(i)` | `Entity::from_raw_u32(i).unwrap()` (returns `Option` in 0.19) | `formations.rs` tests |
| Per-frame `Vec` gathering across disjoint queries via `Local` + multiple systems | one system, `ParamSet<(P0, P1, P2)>` with sequential passes | `process_formation_orders` |
| Mutating another entity's state mid-iteration via `World` access hacks | `commands.queue(move \|world: &mut World\| ...)` deferred closure | `formations.rs` |

Notes:

- Both tuple-spawns (`commands.spawn((A, B, C))`) and derived `Bundle`
  structs with constructors (`BoidBundle::with_target`) are in use; either is
  fine — bundle constructors are preferred when spawn sites repeat.
- Required components (0.16+) are available; this codebase mostly spells
  components out explicitly for clarity. Follow the local file's style.
- Observers (`Trigger`, `.observe(..)`) exist in 0.19 but are unused here;
  prefer the existing hook/system/queued-task patterns unless asked.
- wasm: don't force wgpu backends. `main.rs` pins `Backends::VULKAN` only
  under `#[cfg(not(target_arch = "wasm32"))]`; browsers must let wgpu pick
  (WebGL2/WebGPU). Keep any new render settings wasm-conditional if they
  touch backends.
- `#[allow(clippy::type_complexity)]` on systems with long `ParamSet`/`Query`
  types is accepted practice here (see `process_formation_orders`).
