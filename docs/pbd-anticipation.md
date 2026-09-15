# PBD collision solver — implemented (Stages 2+3), design record

Status: **implemented** (all three stages). Stage 1 — the anti-orbit
planner (`kinematics::arrival_plan`, desired-velocity `follow_target`) and
slope-aware `move_step`. Stage 2 — the frictional-contact solver in
`src/pbd.rs` (`pbd_contact`), replacing `soft_collisions` and
`hard_collisions`. Stage 3 — the long-range anticipation constraint and
the §4.5 sidestep, same solver. The melee/scene validation described below
lands as `--scene` test scenes (`melee`, `formation-cross`,
`formation-clash`, …) and the `pbd` test module.

Deviations from the paper as written (deliberate, see the review trail):

- **Hybrid, not pure PBD**: the steering contract from Stage 1 is intact —
  planners write `vel.a`/`vel.target_v`, `move_step` integrates; the
  solver runs *after* integration and its corrections reach velocity as
  `Δv = Δx/Δt`. The paper's velocity blend `vb=(1−α)v+α·vp` is the same
  first-order relaxation as our accel-clamped desired-velocity steering.
- **One-sided per-agent Jacobi** over a flat SoA scratch (each boid
  resolves against its neighbours' last-iteration positions,
  double-buffered), not the paper's GPU hash grids. Candidates come from
  the shared kd-tree `k_nearest` with a staleness margin — no per-step
  spatial hash so far; revisit if anticipation candidate sets starve.
- **Anticipation applied to current positions** (direction from the
  predicted future pair), with the §4.5 braking-shed implemented as "remove
  the anti-parallel-to-own-velocity component, keep `braking_keep` of it"
  (tunable; 0 = pure sidestep). Symmetric dead-head-on pairs get no
  sidestep — contact handles them, exactly the paper's degeneracy.
- **Obstacle seating is post-solve** (single projection pass), not an
  in-iteration constraint.

Technique source: *Position-Based Multi-Agent Dynamics for Real-Time Crowd
Simulation*, Weiss, Litteneker, Jiang, Terzopoulos — MiG 2017,
[arXiv:1802.02673](https://arxiv.org/abs/1802.02673). Agents are PBD
particles: a planner produces a preferred velocity, positions are predicted,
then projected onto constraint manifolds (frictional contact, anticipatory
long-range collision) instead of accumulating repulsion forces.

## How the stages relate

- `follow_target` is the paper's "velocity planner" (its §4.1): it writes
  `vel.a` toward an arrival-damped, misalignment-slowed preferred velocity
  (`kinematics::arrival_plan`), which fixes the orbit failure of bearing-only
  steering (a perpendicular target used to ask for pure centripetal
  acceleration and the boid circled forever at the turn radius v²/a = 80 m).
- `move_step` remains the integrator, now with slope physics (downhill
  gravity, uphill thrust loss, downhill cap relaxation, bounded-deceleration
  cap shedding).
- Stage 2 replaced force-based collision response (`soft_collisions`,
  `hard_collisions`) with positional constraints. Stage 3 added anticipatory
  constraints on top. The steering/integration contract from Stage 1 is
  untouched by both.

## Honest scope note

The paper has **no faction or per-pair avoidance-toggle concept** — it is a
single-crown simulation (its two-species demo is about mass, not hostility).
Skipping constraints for hostile pairs is *our extension*. It is
mechanically trivial and cannot destabilise PBD (a smaller constraint set is
still a valid projection), but it must be validated by our own three-faction
melee stress test, not assumed from the paper. Hostility must be a
**symmetric** relation (both sides skip), otherwise one army politely steps
aside while the other plows through it.

## Stage 2 — frictional contact solver (paper §4.2, §4.7) — as built

Replaces `soft_collisions` + `hard_collisions`; new module `src/pbd.rs`,
system `pbd_contact` in Update, ordered after `move_step` and before
`ground_boids`.

Components:
- `Faction { id: u8 }` on `BoidBundle` (default 0).
- `Charging` marker — frontal assault: drops anticipatory avoidance (Stage 3)
  against everyone; contact always stays.
- `mass: f32` — inverse-mass weighting (the paper's bears-vs-rabbits
  mechanics: heavy units barely move, light ones get shoved — cavalry vs
  infantry).
- Pair policy as a pure, unit-tested, symmetric fn:
  friend → full constraint set; enemy → contact only (reduced friction, no
  anticipation, no cohesion — the physical clash); `Charging` → anticipation
  off for all pairs.

Solver shape:
- Scratch SoA resource (positions, deltas, entity→index map) — reused
  buffers, zero steady-state allocation.
- Candidate pairs gathered **once per step** from `Res<NNTree>` with a
  staleness margin (query radius += max relative speed × tree refresh
  period); the tree gives candidates, live positions come from the scratch
  copy.
- **Jacobi iterations designed in from day one** (`iterations` tuning,
  default ~4; the paper uses ~6 with delta-averaging coefficient 1.2).
  `iterations = 1` degenerates to single-pass. Do not land the buffers
  single-pass-only: retrofitting the loop is the expensive part.
- Frictional contact: inequality distance constraint
  `C(xi,xj) = |xi−xj| − (ri+rj) ≥ 0`, split by inverse mass, with kinematic
  friction on tangential slip (Macklin et al. unified-particles style).
- Obstacles: infinite-mass projection against the actual `Obstacle` cuboid
  AABB (replaces the center-distance/fixed-normal hack in
  `hard_collisions`).
- Position corrections translate to velocity inside the solver:
  `Δv = Δx/Δt` (the PBD velocity update), then §4.6-style speed/accel
  clamps.

Validation gates:
- Three-faction melee stress test: three groups converging on one point,
  iterations at melee default — assert bounded penetration (< slop) at every
  tick slice. This is the test that validates the per-pair-skip extension.
- Friendly head-on pass without interpenetration; heavier unit shoves the
  lighter; obstacle stop test.
- Wall-clock perf test at 10k agents in the `nearest_solver_scales_to_10k`
  style (contact degree is geometrically bounded ~6–8 for disks, so a melee
  is not asymptotically worse than a march; enemy pairs also skip the
  expensive Stage 3 constraints exactly where density peaks).
- Bench decision: kd-tree `AutomaticUpdate` refresh 1 s → 0.25 s (position
  staleness at 20 m/s is up to 20 m today).

## Stage 3 — long-range anticipation (paper §4.4–4.5) — as built

Only worth adding once playtesting shows last-instant jams in cross-formation
traffic; formation slotting already deconflicts destinations within a
formation.

Math (paper §4.4), per pair, using predicted positions
`x̂i = xi + Δt·vi`, `x̂j = xj + Δt·vj`:

```
a = (vi − vj)·(vi − vj)
b = 2(vi − vj)·(x̂i − x̂j)
c = (x̂i − x̂j)·(x̂i − x̂j) − (ri + rj)²
τ = (−b + √(b² − 4ac)) / 2a          // valid when 0 < τ < τ0
τ̂ = Δt·⌊τ/Δt⌋                       // whole steps just before contact
x̃i = x̂i + (τ̂ + Δt)·vi,  x̃j = x̂j + (τ̂ + Δt)·vj
C(x̃i, x̃j) = |x̃i − x̃j| − (ri + rj) ≥ 0
```

with adaptive stiffness `k·exp(−τ̂²/τ0)`: imminent collisions are stiff,
distant futures fade out. Jacobi-solved like contact, same pair loop.

Avoidance model (§4.5) — how corrections are applied so agents *sidestep*
instead of braking: decompose the positional correction

```
d = (x̃i′ − x̃i) − (x̃j′ − x̃j)      // relative correction
dn = (d·n̂)·n̂,  n̂ = (x̃i − x̃j)/|x̃i − x̃j|
d ← d − dn                          // keep only the tangential part
```

which preserves the pairwise closing speed while sliding the pair past each
other — the paper credits this with non-jittering flows.

Parameter transfer (the paper's numbers are 1.4 m/s pedestrians; ours are up
to 20 m/s units):
- τ0 = 20 s must shrink to ~1–2 s (lookahead 20–40 m, comparable to the
  paper's ~28 m in absolute metres).
- Long-range stiffness k needs retuning upward at our Δt; exact value found
  by bench, not copied.
- Candidate gathering for the large lookahead radius is what forces the
  per-step spatial-hash decision: kd-tree `within_distance` at 40 m radius
  returns half the battlefield. Practical CPU/wasm route: `k_nearest(K≈10)`
  + τ-validity filter (pairs that actually collide soon are almost always
  nearest neighbours). Revisit a real hash grid only if that starves the
  constraint set.
- Same-faction only; `Charging` skips it against everyone (that is the
  frontal-assault exception the project brief calls for).

XSPH cohesion (§4.3) is **rejected** for now: formations already provide
cohesion, and velocity-matching toward neighbours fights slot targets.
Revisit only as a per-LOD flavour term, off by default.

## The head-on formation braid — fixed by soft slot arrivals

`--scene formation-braid` (two friendly blocks, same lane, marching at
each other) pinned the original failure: the lone-pair `head-on` scene
braids cleanly, but the formation-scale meeting stalled into a deadlocked
mixed phalanx wall at the centre. Three mechanisms stacked:

1. **Mirror symmetry.** A block-block meeting has no inherent sidestep
   side: each interior member's anticipated pairs sit symmetrically left
   and right, so their lateral corrections cancel. The lone pair only
   braids because of its deliberate lane offset (and numerical noise).
2. **Slots fight the sidestep.** `dispatch_formation_goals` rewrites each
   member's `Target` to its slot every tick, and while a `Move` is active
   `plan_goal`'s `MIN_LEAD` keeps every slot ~4 m ahead of the centre of
   mass — so the jam's forward pressure is full-strength slot steering
   into the wall.
3. **Friendly contact boxes the wall in.** Contact applies between
   compatriots too; once the blocks compress, same-army neighbours at
   1 m spacing seal the ends of the wall.

**The fix (built): soft slot arrivals.** Member slot targets now carry
`Target::soft_arrival_m` (set from `FormationTuning::slot_soft_radius_m`,
default 6 m, F1-tunable): inside the radius, `follow_target`'s
acceleration clamp ramps down quadratically (`kinematics::arrival_authority`,
floored at 0.25). Near their slots, members steer at ≤ ~1.2 m/s² — well
below the solver's avoidance authority — so anticipation opens lateral
lanes instead of being crushed flat. The floor matters: authority fading
to *zero* let post-arrival momentum drift formations past their
destination (the march test caught this); the quarter floor keeps enough
settle/brake authority for crisp arrivals. Radius 6 m rather than the
naive ~2 m because the march lead already keeps members ~4 m from their
slots — a smaller fade never engages where the pressure builds.
Verified in the scene: the blocks now interweave like a braided cable and
pass (previously a stable deadlocked wall); the 90° `formation-cross` and
all 102 tests unchanged.

Remaining candidates if multi-block traffic still clogs (not built):

- **Consistent passing side** — bias the §4.5 sidestep laterally by a
  fixed handedness relative to the closing direction ("keep right"),
  keyed per formation so both blocks pick the same global convention.
  Would make the braid deterministic instead of noise-broken, and scale
  to more than two blocks.
- **Formation-level deconfliction** — plan-level: two formations whose
  planned paths intersect resolve at the `FormationGoal` layer (one
  yields, detours or sidesteps as a whole). This is what RTS games
  actually do for large blocks; the paper has nothing equivalent (its
  agents have no group structure).
