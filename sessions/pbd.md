# PBD session — braid polish, tuning, review, stateless bob (merged to master)

Session on the `pbd` branch, fast-forwarded to `master` at `3e5cb6a`.
Continues the PBD collision-solver work (Weiss et al., MiG 2017 —
`docs/pbd-anticipation.md`, `src/pbd.rs`).

## What happened

### Low-speed braid verified (`f89fe6f`)
`formation-braid` close-start: origins ±10 m instead of ±40 (20 m gap,
meet within ~2 s before spooling up), camera 40 m. Vision analysis of
t=3/5.5/10 s captures: colors interpenetrate into a packed braided core
(more compressed than high speed — the 1.5 s anticipation horizon covers
only ~3–4 m), some lane knock-offs mid-pass, complete pass with both
blocks reformed by t=10 s, no re-jam. Soft slot arrivals hold at low
speed.

### Experiment 1 — contact radius 0.5 → 0.45 (`cbed2ae`)
User report: stationary-through-moving braids fine; both-moving snags
rear ranks. 10% radius cut under the visual capsule: less core stacking
at the encounter, snag reduced to isolated pairs (one locked red-green
pair mid-pass, resolves by t=10). Accepted cost: slight visual overlap
in a push. Residual failure mode is a mutually-stopped overlapping pair
(friction resists sliding but nothing shears them apart) — radius can't
fix that alone.

### Friction semantics (Q&A, no code)
Friction was already per-pair by hostility: friendly μ = `friction`
(0.4), hostile = `friction × hostile_friction_scale` (0.1) — resolved in
`pair_policy`, carried per-candidate (`nbr.friction`), so the Jacobi loop
never sees a global μ. Extending by other criteria is a one-function
change; the invariant is *symmetry* — both sides of a pair must compute
the same answer.

### Defaults retuned + slider widened (`8f0fef5`)
Manual scene testing by the user: **12 neighbours × 1 Jacobi pass** is
the behavior they wanted. `NEIGHBOR_COUNT` 6→12, `SOLVER_ITERATIONS`
4→1. Perf at 10k boids release: **0.8 ms** (was 1.3 at 6×4). Hostile
friction slider now 0..=5 (above 1× = deliberate sticky grind, e.g.
shield walls).

**Discarding behind-neighbors: rejected** (answered, not implemented).
The expensive part is the kd-tree query/heap, which runs before any
direction is known — a post-filter saves only cheap constraint
arithmetic. Querying fewer to actually save time = smaller effective
neighbour count, which the 12-neighbour finding contradicts. Also
"behind" is the wrong predicate: contact is omni-directional (rear
pressure in a jam), a heading filter breaks the symmetric-pair invariant
(half-rate separation), and fast followers from behind are real future
collisions — the τ formulation already gives receding pairs near-zero
stiffness for free. Scaling levers if gather ever bottlenecks: spatial
hash + radius buckets, candidate reuse across frames.

### Code review pass (`f99ff8c`)
- **Dead code out**: `Velocity::push` (write-only since PBD replaced
  force pushes) and `SoftCollision` (empty wrapper; `BoidBundle` already
  carries `TrackedByTree`). `HardCollision` is now a plain obstacle
  marker.
- **Obstacle seating fix**: query reach uses the *widest* boid radius
  (was first-boid's — under-reached with mixed radii); cuboid half-extent
  named (`OBSTACLE_HALF_EXTENT`, was three magic 0.5s); loop skipped when
  no obstacles.
- **Comments/docs matched to as-built**: candidate-margin comment (the
  keep-filter runs on live scratch positions — it cannot compensate tree
  staleness; it keeps near-contact pairs for later iterations);
  `NEIGHBOR_SLOTS` sizing (candidate list, not contact count); doc's
  bogus "query radius += speed × refresh" gather description; the
  never-applied 0.25 s tree-refresh "decision" is now an honest known
  gap (refresh still 1 s; mitigated by live-position filter and
  symmetric gather); phantom "obstacle stop test" replaced by a gap note
  (obstacle seating is scene-verified only).

### Architecture Q&As (no code)
- **CPU vs GPU**: everything simulates on CPU (`ComputeTaskPool` chunk
  parallelism); GPU only renders. GPU compute + spatial hash is the
  ~200k-unit migration path, not needed at 0.8 ms/10k.
- **Velocity component**: still the backbone (`follow_target` writes
  `a`/`target_v`, `move_step` integrates with slopes, `pbd_contact`
  snapshots `v` and folds corrections back as `Δv = Δx/Δt`). Only the
  `push` field died.
- **Formation-as-agent LOD**: mostly holds. `move_step`/`follow_target`
  are filter-free, and formations.rs:369's Velocity-LOD split already
  makes a lowest-LOD formation integrate like one boid — marching works
  today. PBD is structurally ready (scale-free disks, inverse-mass,
  τ-anticipation) but **not wired**: the LOD flip doesn't stamp
  `Body { radius = extent, mass = Σ members }` + `Faction`, so distant
  formations currently collide with nothing. Approximation gaps
  acceptable exactly at LOD distance: disk-vs-block shape, friction cone
  scaling with metre-overlaps (needs retune at scale), point kinematics
  ignoring wheeling (outer members travel faster on turns), and re-LOD
  member reconciliation is unbuilt.

### The bob saga (`dd4d82d` → `66d17a9` → `3e5cb6a`)
User report: units bobbed only when starting/stopping (as if frequency
depended on acceleration), nothing when stationary despite a floor, too
fast during transitions. **Three stacked bugs**:

1. Phase computed as `freq × elapsed` (the 2023 hand-written form) —
   effective frequency is `freq + t·Δfreq/Δt`: an acceleration term
   multiplied by session elapsed-time, worsening as the app runs.
2. Tuning-era `clamp(floor, floor×4)` = 0.05–0.2 Hz pinned every speed
   above a 1.33 m/s walk at an invisible 0.2 Hz; the 0.05 Hz "floor" is
   a 20 s period.
3. Hz fed to `sin()` as radians — every frequency ~6× under intent (the
   original had this too; caught by the new test's quarter-period check).

Fix history: first accumulated phase per boid (`phase += TAU·freq·dt`)
with floor raised to 0.5 Hz — correct, but the user challenged the 4
bytes/boid at 200k scale ("stay vigilant; many such systems may follow";
phase of a time-varying oscillator is path-dependent, but this is
cosmetic). **Final form: fully stateless** — the `Bob` component is
deleted. Three fixed cadences (`sin(TAU·f·(t + hash(id)))` for
idle/walk/run) blended by speed through smoothsteps; sine arguments
never contain speed, so no phase can jump. Per-boid de-sync is a
golden-ratio hash of `Boid::id`. Tuning: three cadence sliders (F1).
User then retuned to 0.2 / 2.5 / 5.0 Hz (`3e5cb6a`).

House rule established: cosmetic effects prefer pure functions of global
time + already-stored state (id, velocity, transform) over per-boid
accumulators.

### Merge
Master had no divergent commits → `git checkout master && git merge
--ff-only pbd` (no rebase needed, history preserved). Gate before merge:
108/108 tests, wasm clean, clippy at baseline (zero new warnings).
Local master is 13 commits ahead of origin (not pushed). `pbd` branch
now redundant.

## State at session end
- All 13 commits on master (`175bc49`..`3e5cb6a`), clean tree, 108
  tests green, release perf 0.8 ms @ 10k, wasm clean.
- Defaults of record: contact radius 0.45, 12 neighbours × 1 iteration,
  friendly μ 0.4, hostile ×0.25 (slider to 5), anticipation 1.5 s /
  stiffness 0.35 / braking keep 0.3, slot soft radius 6 m (authority
  floor 0.25), bob 0.2/2.5/5.0 Hz.

## Open items / next levers
- Rear-pair snag in both-moving braids: friendly friction 0.4→0.25–0.3,
  stronger anticipation stiffness, or a consistent-passing-side bias
  (candidates listed in docs/pbd-anticipation.md).
- Formation-as-agent LOD wiring (Body/Faction on the Velocity flip) +
  per-scale friction retune + re-LOD member reconciliation.
- Obstacle seating integration test (documented gap).
- Tree refresh 1 s → 0.25 s if fast units visibly clip after a refresh.
- GPU compute + spatial hash at ~200k units.

## Operational notes
- Vision analysis of screenshots: the CDN URL embeds the file path — a
  Cyrillic path segment (`Документы`) breaks the analyzer's fetcher;
  re-encode (downscale) to an ASCII path and re-upload. Content-dedupe
  means identical bytes return the same (broken) URL — re-encoding also
  sidesteps that.
- `--shot` fires at `--secs − 5 s`; zoom 0.9987 ≈ 41 m height,
  `--cam-angle` is degrees from nadir.
- Git Bash mangles `$` in inline `powershell -Command` strings — write a
  .ps1 (ASCII-only paths) and run with `-File`.
