# Session: Sun & atmospheric effects (2026-08-29)

Added first-party sun + atmosphere to the game. Commit `6fdb18a` —
*"Sky: Earth atmosphere, sun disk, sky-driven ambient and shadows"* (master).

## What was added

- **`src/sky.rs`** (new module): `SkyPlugin` spawns
  - `Atmosphere::earth` entity with a `ScatteringMedium::earth(256, 256)` asset.
    Spawned without a Transform, the component's `on_add` hook centers the
    planet one inner_radius below the origin, so the planet surface sits at
    y = 0 — flush with the ground plane.
  - The sun: `DirectionalLight` at `lux::FULL_DAYLIGHT` with
    `shadow_maps_enabled: true` + `SunDisk::EARTH`. Direction set by pure
    helper `sun_transform(SUN_ELEVATION_DEG, SUN_AZIMUTH_DEG)` (35°/125°,
    tuning consts), covered by 3 inline unit tests.
- **`src/main.rs`**: camera gained `Projection::Perspective { far: 30 km }`
  (default 1 km clipped most of the 5 km plane), `AtmosphereSettings`
  (auto-requires HDR), `AtmosphereEnvironmentMapLight`, `Bloom` (sun glow).
  `GlobalAmbientLight::NONE` + sky-tinted `ClearColor`; spawn `PointLight`
  removed.

## API notes (Bevy 0.19, verified against vendored sources)

- `Atmosphere` is a **standalone entity** in 0.19 (moved off the camera;
  migration guide 0.18→0.19, PR #23651). `AtmosphereSettings` stays on the
  camera and enables rendering per view; nearest atmosphere wins.
- `SunDisk` attaches to the `DirectionalLight` entity (`#[require(DirectionalLight)]`);
  the sky's sun direction follows that light's transform.
- All atmosphere LUTs are **compute shaders** → sky renders on WebGPU only.
  On WebGL2 the scene keeps direct sun lighting but no sky, and with
  `GlobalAmbientLight::NONE` no sky-driven ambient either (accepted
  trade-off; remedy if it ever matters: keep a small `GlobalAmbientLight`).
- `bevy_rts_camera` never touches `Projection`, so the explicit far plane is safe.

## Decisions (user-selected)

WebGPU-only (no WebGL2 fallback path) · full relight (sky-driven ambient,
PointLight removed) · sun shadows enabled + bench.

## Benchmarks (dev profile, 1080p, ~60 m camera, idle machine)

| Configuration | Frame time |
| --- | --- |
| Boids hidden from render (10k sim still runs) | ~49.5 ms |
| Boids visible, shadows off | ~51.5 ms |
| Boids visible, shadows on | ~61 ms |

- Shadow cascades cost **~9.5 ms (~16%)**; recorded next to
  `shadow_maps_enabled` in `sky.rs`. First measurement was taken while other
  workloads ran; user asked for a re-run on an idle machine — numbers held.
- **Mesh instancing is working** (user challenged an earlier "10k draw calls"
  claim, which was wrong): all boids share one `Handle<Mesh>` +
  `Handle<StandardMaterial>`, no `NoAutomaticBatching`, and bevy_pbr sets
  `AUTOMATIC_BATCHING` unless that component exists. Rendering the whole
  boid field costs ~2 ms; the frame is dominated by the **dev-profile CPU
  sim** (cranelift opt-1 for our crate; the same kd-tree work has a ~7 ms
  release budget). Shadow-cost levers if needed: lower-poly capsule or
  `NotShadowCaster` on far LODs — not batching.

## Pre-existing issues discovered (reproduce at clean HEAD, not from this change)

1. **`cargo build --release` cannot link**: always-on native
   `bevy/dynamic_linking` is incompatible with `[profile.release] lto = true`
   (MSVC LNK2019 on reflect/inventory statics). CI only runs `cargo check`,
   so it went unnoticed. Workaround that links fine (9 min):
   `cargo build --release --config 'profile.release.lto="off"'`.
2. **The release exe won't launch on this machine**: dies resolving
   `api-ms-win-crt-heap-l1-1-0.dll` — same error as the repo's stale
   `run.log` from Aug 20. Dev builds run fine; CRT/runtime-install specific,
   unresolved. This is why bench numbers are dev-profile.

## Process notes

- An early exploration pass described a voxel-world/far-terrain repo state
  that does not exist (real HEAD `42bb710`, flat plane + PointLight); direct
  file reads were the ground truth and the plan targeted the real code.
  Confirms the skill rule: the codebase is the source of truth.
- All API spellings were verified in vendored registry sources before use
  (e.g. `shadow_maps_enabled`, `illuminance`, `GlobalAmbientLight.brightness`).

## Verification

`cargo check` + `cargo check --target wasm32-unknown-unknown` green,
19 tests pass, live-run visual check (sky renders — the region beyond the
plane edge shows haze, not the clear color; shadows visible on obstacles).

## Follow-up ideas

- Day/night cycle: rotate the sun entity — sky, SunDisk, and aerial haze follow.
- Volumetric light shafts (`VolumetricFog`): WebGPU-only, known glitchy
  upstream (bevy issue #22574).
- Investigate the release CRT launch failure; adopt the LTO-off override or
  a `fast`-profile fix for local release profiling.
