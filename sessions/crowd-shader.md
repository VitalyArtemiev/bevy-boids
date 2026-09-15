# Crowd shader session — formation crowd-shell experiment (box + procedural crowd)

Goal from the user: render massed units in tight formations cheaply — a
unit-height box with extra vertices that can deform (a fluid frontline),
plus a shader that "roughly resembles a crowd of soldiers" from the top
and the upper 30° cone, with unit silhouettes, periodic weapon/armor
glint, and some dust. Everything lives in `src/crowd.rs` +
`assets/shaders/crowd{,_common,_dust}.wgsl`; verified via
`--scene=crowd --cam-angle <deg> --shot <png>` bench runs.

Commits: user's WIP snapshot `6dd7606`, mine `93711a0` (transparent box,
breathing columns) and `c4c1a4e` (terrain following). The final
`--scene=crowd` migration is **uncommitted** — it landed while the user's
`src/ui/` refactor was half-staged in the same files (`main.rs` mixes
both), so no clean commit exists through that state.

## What happened

### The design that shipped
One alpha-masked box per army (~1,300 verts, 2 m grid, ready for CPU
frontline deformation later) + per-pixel raytraced crowd: the fragment
walks the view ray through a jittered 0.85 m soldier grid (2-D DDA;
exact because jitter clamps every body disk inside its cell), solving
each cell's vertical cylinder + helmet sphere analytically (rays
parametrized by y — the cone guarantees steep rays). Soldiers = shoulder
disks/helmets from nadir, body-wall silhouettes at the cone rim.
Occupancy is fully static; the frontline surge is a z-warp of the DDA
walk, so the whole column breathes without any soldier popping. Rays
that hit nothing and never passed under occupied cells are discarded
(alpha mask) — the box is invisible against real terrain; rays under the
ranks shade dark team-tinted crowd shadow. Glints: per-soldier periodic
emissive flash (pure specular mathematically can't fire from nadir) +
spear-tip mirror sparks. Dust: a translucent volume above, gated to the
churn band around the front. Lighting all in box-local space (sun
rotated per army at spawn) — units move with the box rigidly.

### Two invisible-shader bugs (the long debugging saga)
The shader silently drew nothing for most of a session, with zero
logged errors. Sequence of real causes, all in SKILL.md now:
1. `position_world_to_clip` takes **vec3**, not vec4.
2. A custom wgsl module (`#define_import_path bevy_boids::crowd_common`)
   only resolves `#import` if the file is a **loaded asset** — nothing
   references it by path, so `CrowdPlugin` preloads it
   (`CrowdCommonShader`) and gates spawning on the load; otherwise
   pipelines silently never specialize (the deleted terrain render
   module had the same resource).
3. The actual killer: the fragment signature used bare `CrowdOut` while
   the type lives in the imported module — WGSL has no aliases; every
   signature must say `crowd_common::CrowdOut`. The error WAS logged
   (`pipeline_cache: failed to process shader error:`) but spans two
   lines and head-limited `grep error` never reached it past cargo
   warnings. Lesson recorded: grep `failed to process` specifically.
Also learned: `Mut<T>` holds its `&mut World` until last use — multiple
`resource_mut` calls need nested `resource_scope` (fixed the user's
`BoidVariations::from_world` with this when it blocked the build).

### Vision verification was a minefield
The external image analyzer repeatedly **hallucinated armies that
weren't there** ("distinct crimson and steel-blue masses with rank
structure" over a uniform green frame). Built objective fallbacks:
PowerShell pixel sampling (regional averages, row/band profiles,
variance stddev, 3× crops) — these caught every hallucination and drove
all tuning decisions. In-shader debug views (normal output, DDA
visit/hit/occupancy encoding) cracked the two structural bugs below.
Direct vision (once available) + pixel stats together are the recipe.

### The two structural bugs the audit found
When real eyes finally looked: flat slabs, no soldiers. In-shader DDA
diagnostics exposed:
1. **The frontline sat at 0.68 × box depth** — each army occupied the
   rear third; the visible "mass" was mostly empty floor. Front base is
   now 0.15 × depth.
2. **An inverted nearest-hit comparison** (`t >= t_min` with a 1e9
   sentinel — rejected every hit): introduced during the struct
   restructure for bug 3 above; meant *no soldier had ever rendered* —
   everything seen until then was tinted floor + spear sparks. All prior
   "tuning" had been papering over this.
After the fixes: dark body-shadowed interior floor (no direct sun under
ranks), per-soldier albedo variation (the variation that survives at
pixel scale), wrap diffuse (packed ranks never shade to black),
helmets smaller than shoulders (tunic colour must ring the steel).

### User feedback round (all five points)
1. **Box transparent**: alpha-masked discard for empty rays; the debug
   floor rectangle is gone, real terrain shows to the troop edges.
2. **Dust visible**: density + alpha cap raised, plume concentrated on
   the frontline band, box 3 m tall (later gated to the churn band — no
   haze over quiet rear ranks).
3. **Glints back**: per-soldier emissive flash added (see above).
4. **No popping**: static existence + surge-warp (see design). Verified
   with captures 8 s apart: front silhouettes shift, zero pop-in.
5. **Rear ranks full**: straight formed flanks/rear (militarily
   correct; the "hard rectangle" artifact was the fake floor, gone with
   transparency).

### Terrain following (commit `c4c1a4e`)
Boxes, ray anchors and soldier feet all sample one baked heightmap:
bindings 3–6 in `crowd_common.wgsl` (R8Unorm 512² over 240 m, per-army
origin/yaw/seat, height decode range, footprint trace y-range), shared
by both materials. The isolated scene got rolling hills
(`crowd_test_height` drives ground mesh + `HeightField` + heightmap
bake, so grounding/camera/crowd agree by construction). Intersection
tests needed no changes — local-space and verticality-agnostic; soldiers
stay upright on slopes. During the `--scene` migration the compiler
surfaced a latent bug from this round: the decode range was wired to
the per-army footprint min/max instead of the map's bake range —
visually identical on symmetric hills, wrong on offset terrain.

### `--crowd` → `--scene=crowd` migration (uncommitted)
The user built `src/scene.rs` (TestScene registry, `--scene <name>`,
per-scene `in_scene()` run conditions, scenes own camera/props).
Migration: `TestScene::Crowd`; `spawn_crowd_scene` spawns the camera —
the crowd scene uses the game's own RTS camera (design cone + bench
`--cam-angle` pinning are defined for it), so the RTS spawn was
extracted from `setup` into `pub(crate) fn spawn_rts_camera` in
main.rs. CrowdPlugin's army/hills systems re-gated on the scene.
`--flat`/billboard got their plain plane back (they'd silently
inherited the crowd's hills). `launch.rs` dropped `crowd`/`--crowd`/
`launch_crowd_enabled`; tests assert both `--scene` flag forms and that
the old flag selects nothing. Crowd scene also needs
`init_resource::<DemoTerrain>()` (F3 panel `Res` panics without it —
hit this twice). Verified: 89/89 tests, wasm green, `--scene=crowd`
and `--scene=billboard` bench captures.

## Known gaps / next steps
- Sun is baked per-army at spawn (`sun_local`) — live sun changes don't
  re-light existing armies.
- No shadow casting/receiving, no aerial perspective.
- The swell is vertex-shader demo deformation; real CPU frontline
  deformation through the mesh vertices is the intended production path.
- DDA cap (48 cells) is generous for the cone; a tighter cap is a small
  far-LOD win.
- Heightmap is nearest-texel (0.47 m/texel) — fine for low-frequency
  terrain, worth revisiting for gullies.
