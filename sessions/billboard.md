# Billboard session — PBR-parity lighting, the orientation invariant, `--scene`

Multiple sessions on `master`, picking up after the impostor-LOD commit
`bb745e5` (see `docs/plans/impostor-billboard-lod.md` for the design).
Commits: `f922df6` (lighting + orientation), `52466ec` (`--scene`
framework), `a0f0610` (scene menu bypass).

## Lighting: one authority, two mandatory magnitude steps

The fragment now reads ambient + directional sun straight from the
engine's `lights` view binding (`bevy_pbr::mesh_view_bindings`) — the
same uniform PBR meshes read — instead of material-uniform snapshots:
F1 sun sliders, illuminance and future day/night state apply to both
render paths with no sync system, and billboards go dark with the
capsules at night. The uniform is **photometric HDR** (sun colour
premultiplied by `lux::FULL_DAYLIGHT` = 20 000), so the fragment tail
must mirror `pbr_functions.wgsl` twice — shipping without either step
produced a shipped-and-fixed bug apiece:

1. **`view.exposure`** — every PBR fragment scales its summed light by
   it; the default (`Exposure::BLENDER`, EV100 9.7 ≈ ×1.4e-3) is what
   brings daylight to display scale. Without it the sprite rides the
   tonemap shoulder ~700× too high: *colourful at grazing sun angles,
   a white blob wherever NdotL nears 1*. That split is why the lab
   (el 45°) passed numeric checks while the 302 m bench (default sun,
   NdotL ≈ 1) blew out — and why the bug looked distance-dependent but
   wasn't.
2. **In-shader tonemapping** — `tone_mapping(color, view.color_grading)`
   under `#ifdef TONEMAP_IN_SHADER` (defined for non-HDR cameras;
   this game's cameras are LDR). Without it the raw value saturates
   the sRGB target and the whole sprite clips to white at any range.

Measured after both: 302 m bench billboards 0.85% tinted / 0% near-white
vs mesh-mode baseline 0.76% / 0% — parity within noise.

`BillboardTuning.brightness` (F1 slider, 0..=3×, 1.0 = parity) is the
one material-side lighting input — an eyeball-match knob for the
baked-normal residue. It rides `#[uniform(1)] f32` (atlas bindings
renumbered 0/1/2/3 on both sides), is seeded into the shared materials
at creation (`BillboardAssets::from_world`) and pushed live by
`sync_billboard_brightness` (`resource_changed`-gated). The user has
not yet reported a preferred value; the default stays 1.0.

## The orientation invariant (two mirrored errors hiding each other)

User report: sun at zenith lit the sprite's *bottom*. Root cause chain:

- The atlas bakes normals into cells at the whole-texture 180° rotation
  but **blitted the image upright** — the shader's `nuv = 1 - uv` read
  then paired every albedo pixel with the *opposite sprite point's*
  normal: lighting rotated 180° on the sprite.
- Fixing that (`blit_cell_rot180` in the compositor, re-baked atlases)
  made **left/right** wrong instead: the runtime display had been
  mirroring the cell horizontally (`spun = -corner` flipped both axes),
  invisible on the symmetric checker albedo, and the two horizontal
  errors had been cancelling. Display fix: `spun = vec2(corner.x,
  -corner.y)` — image u along screen-right, v stays top-to-bottom.

**Paired invariant** (recorded in SKILL.md + the shader comments): the
normal half's 180° content rotation and the un-mirrored display u axis
each alone re-open the opposite axis's bug. Verified numerically at
el 85° (both twins top-lit) and az 90°/270° el 45° (lit sides match the
mesh twin's signs); atlases re-baked via `--preprocess`.

## `--scene` framework (replaces `--billboard-lab`)

`src/scene.rs`: `TestScene` is the name registry (`ALL`/`name`/
`from_name`), `in_scene(variant)` the run-condition factory,
`ScenePlugin` the registration point — module docs list the four steps
to add a scene. A scene owns its camera and props; while any is active,
`setup` skips the RTS camera/grid boids/obstacles,
`DebugConfig.impostor_lod` stands the billboard auto-swap down, and the
launch runs straight into `GameState::Playing` (the `--bench` state
bypass minus bench behaviours; inserted before UiPlugin's idempotent
`init_state`, pinned by a test). Unknown names error with the list.

The `billboard` scene: mesh capsule screen **LEFT** (z = +1.5), forced
billboard twin **RIGHT** (z = −1.5), both still at yaw 0, plain camera
25° from nadir ~7 m out. The pair splits along Z because the camera
looks down X — Z offsets separate the twins on screen instead of
overlapping them along the view axis (the old ±X layout stacked them in
depth, which frustrated every pixel measurement). The spawn/convert
code moved out of `billboard.rs`; `attach_billboard` and `facing_yaw`
are its `pub(crate)` seams.

## Verification techniques that worked

- **Numeric pixels over vision models** (a vision MCP call once rated
  the blown-out lab "90–95% parity"): tinted fraction ((max−min) > 60
  && max > 60), near-white (min > 220), hue buckets. PowerShell
  System.Drawing scripts under `target/` (gitignored, recreate as
  needed; note `powershell -File` needs `-ExecutionPolicy Bypass`).
- **Red-dominant mask** (`R − G > 90`) isolates the checker capsules
  from warm sun-glow terrain/sky that defeats generic spread masks.
- **Before/after image diff** isolates the billboard when twins
  overlapped: the mesh is identical across runs, so changed pixels are
  billboard pixels.
- **ASCII pixel rendering** (luma ramp + `R` for red-mask) when the
  vision MCP 400s — it is flaky, and URLs containing the Cyrillic repo
  path (`Документы`) make it worse.
- Bench recipe for lighting checks:
  `--bench --secs 8 --scene billboard --sun-azimuth <a> --sun-elevation <e> --shot target/x.png`.

## Gotchas

- Killing a `cargo run` mid-link leaves corrupt artifacts: the next
  dynamic build fails with LNK2019 on `anon.*.llvm.*` symbols that
  look like a code error but aren't — `cargo clean -p bevy-boids`
  fixes it. (Same code links fine under `cargo test`.)
- The `lights` uniform's directional `.color` is `color × illuminance`
  and `ambient_color` is `color × brightness` (lux-scale) — anything
  reading them raw must apply `view.exposure` itself.
- `App::insert_state` panics without `StatesPlugin` — tests need
  `app.add_plugins(StatesPlugin)` first (real app gets it from
  DefaultPlugins).

Suite: 89 tests green, wasm check green at every commit.
