# UI session — egui stack M1–M5, radial menu, visual polish, src/ui/ move

Multiple sessions on `master`, from the stack decision through the module
reorganization. UI work lives in `src/ui/` (`mod.rs`, `debug.rs`, `input.rs`,
`radial.rs`); commits `068a3c8` (M4+M5), `fe4a6af` (move + scene wiring).

## Stack decision (approved plan)

- **bevy_egui 0.42** for all UI (immediate mode, one stack, wasm-proven) —
  Bevy 0.19 has no first-party immediate-mode UI, and Feathers is
  editor-oriented, so "immediate mode" and "first-party" were mutually
  exclusive; immediate mode won.
- **bevy_enhanced_input 0.26** for remappable bindings.
- Persistence via first-party **bevy-settings** (`bevy_settings` feature):
  `%LOCALAPPDATA%/com.github.VitalyArtemiev.bevy-boids/settings.toml`.

## M1 — egui scaffold, states, pause

`UiPlugin` (`ui/mod.rs`): `GameState { MainMenu, Playing, Paused }`; sim +
input gated `in_state(Playing)`; menus pause `Time<Virtual>` (prevents
fixed-timestep catch-up storm) and kill the camera via
`RtsCameraControls.enabled`. egui systems run in `EguiPrimaryContextPass`
(0.42 requirement); panels need a hand-built root `egui::Ui` (egui 0.36
panels take `&mut Ui`, not `&Context` — `root_ui` helper); `Window::show`
takes `&Context` directly. Gameplay input systems carry
`not(egui_wants_any_pointer_input)` / `..._keyboard_input` run conditions.

## M2/M3 — debug panels, options, persistence

- `*Tuning` resources (`Kinematics/Boid/Formation/Sky/Terrain`) with const
  defaults; F1 debug panel sliders; F3 terrain panel (user-built).
- `OptionsSettings` (shadows, bloom, camera speeds) persisted by
  bevy-settings; `apply_options` pushes to live values, gated
  `resource_changed` (fires once at startup + on edits).
- **Gotchas recorded in the skill**: settings structs need
  `#[derive(Resource, Reflect, SettingsGroup)]` with
  `#[reflect(Default, Resource)]` — without `Resource` the loader panics at
  startup (unwraps ReflectComponent); types must be registered *before*
  `SettingsPlugin` builds. Storage is `preferences_dir()` = **LOCALAppData**
  on Windows, not Roaming. Saving is explicit: `SaveSettingsDeferred`
  (debounced 1 s) after UI edits.
- **egui-over-ResMut rule** (two bugs found): any `&mut` through a
  `ResMut`/`Mut` — even an unused `&mut *res` reborrow — marks the resource
  changed every frame; change-driven consumers then rebuild per frame (the
  F3 panel regenerated the whole terrain at frame rate). Pattern:
  `bypass_change_detection()` for widgets, accumulate `Response::changed()`,
  `set_changed()` only on real edits (`tuned()` helper in `ui/debug.rs`).

## M4 — enhanced-input migration + rebind UI (`ui/input.rs`)

- One `PlayerContext`, 19 actions (Select, SelectAdditive, Frontage,
  AdjustWidth, Pause, ToggleDebug, ToggleTerrain, Recall/AssignGroup 1–6);
  recall = digit, assign = Ctrl+digit (`ModKeys::CONTROL`; more-modifier
  bindings evaluate first + consume input). Systems poll
  `Query<(&ActionTag, &TriggerState, &ActionEvents)>` via
  `started`/`fired`/`completed`; action entities are tagged `ActionTag`
  (stable `ActionId`) so no per-type queries.
- Rebind UI ("Key bindings…" in Options): capture next press
  (raw `ButtonInput`), conflict = swap the two actions' bindings,
  reset-all. Persistence: `BindingsSettings.overrides` (sparse; entries
  dropped when back at default).
- **`Binding` is immutable** (`#[component(immutable)]` + on_insert hook) —
  rebinds despawn/respawn binding entities (`replace_binding`), never
  mutate. Enhanced-input types aren't re-exported at crate root — `ui/input.rs`
  re-exports `ActionEvents`/`TriggerState`.
- Camera keys stay on the vendored bevy_rts_camera fields; `freecam.rs`
  still reads raw keys (unmigrated).

## M5 — radial context menu (`ui/radial.rs`)

- **Gesture**: RMB press→release within `CLICK_TOLERANCE_PX` (6 px) with a
  selection opens the menu; drags keep frontage designation
  (`frontage_position_system` skips undragged releases via
  `Player.front_press`). Gameplay input stands down via the `radial_closed`
  condition (RadialMenu resource presence = open state).
- **Tree layout** (user-directed, iterated three times): parent ring stays
  visible; each child ring's center sits exactly on the parent's outer
  radius in the spawning wedge's direction (no gap), rotated so the
  wedge facing the parent is **omitted** (60° corridor) — the corridor is
  what keeps the child's flanks off the parent's sibling wedges
  (test-pinned: `child_ring_never_covers_parent_siblings`). Children draw
  **behind** their parents (deepest ring first).
- **Interaction**: hover deepest-ring-first (corridor falls through to the
  parent), hovering a sibling command folds deeper rings, LMB executes,
  hole pops (root hole cancels), Esc / click outside closes. **Crash
  fixed**: rings laid out from the pre-navigation path indexed a truncated
  path — re-layout in the same frame when navigation changes it
  (+ `layout_always_matches_the_path_length` test).
- **Commands** (placeholder set): Walk (sub-menu: Keep formation / Column
  of 4) and Run. Column-of-4 queues `Reform{Grid, cols:4}` → `Move` →
  `Reform{original}`; speeds via new `Target::speed_scale` (walk 0.5, run
  1.0) set on formations + loaded members + free boids. `FormationOrder::Reform`
  now carries `{kind, columns}`.
- **Visuals**: wedges are opaque checker-textured quad strips
  (`egui::Mesh`, world-space UVs, vertex-color tints) — the earlier
  `convex_polygon` hull fan drew a stray triangle over the hole; rims at 48
  arc samples (n-gon silhouette gone).
- **Labels**: tangential via `TextShape.angle` (galley + rotation; flip a
  half-turn on the lower half so nothing reads upside-down). Horizontal
  labels *cannot* fit sideways wedges (98 px label vs 70 px radial band) —
  the clipping test proved it.
- **Label clipping test** (layer 1 of the proposed testing method):
  `labels_fit_inside_their_wedges` measures every label × every menu state
  on a headless `egui::Context` (needs one burn-in `begin_pass`/`end_pass`
  with `textures_delta.clear()`); asserts rotated-quad corners stay in the
  wedge sector — strict radial+angular at 1× font, angular-only at 1.5×
  (font-only bumps are config-coupled; pokes past the rim = re-tune radii
  with the font). Layers 2 (debug bounds overlay) and 3 (`--shot-menu`
  screenshot set for vision review) proposed, not built.

## Repo reorganization

All UI modules moved to `src/ui/` (`fe4a6af`): `ui.rs`→`ui/mod.rs` kept
every `crate::ui::` path working; submodules `debug`/`input`/`radial`
declared in `mod.rs`; importers rewritten (`crate::ui::input::` etc.).
Commit also carries the user's interleaved `--scene=crowd` flag refactor —
kept together so every snapshot compiles.

## Verification state

89/89 tests, wasm check clean, boots without panics. Radial interaction
was exercised by the user in-game (one crash found+fixed); label clipping
is enforced headlessly. Untracked leftovers: two `*_backup.wgsl` scratch
files. Nothing pushed.

## Natural follow-ups

More radial commands (stances, facing, formation kinds); lateral spread for
multi-formation radial orders (all currently converge on one point);
freecam migration onto actions; clipping-test layers 2–3; per-wedge icons
instead of text labels eventually.
