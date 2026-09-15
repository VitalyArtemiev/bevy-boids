# Session handoff — formation relationships & executor pipeline (2026-09-16)

Point-in-time summary of the formations/relationships session, written so a
fresh session on any machine can pick up the work. Commits `b7f4d89`
(mid-session checkpoint, "Start refactoring into a more readable form") and
`42bb710` ("Merge MemberOf/FormationOf into one relationship") on top of
`f966c00`. Read together with `.agents/skills/bevy-boids/SKILL.md`
(conventions, commands, hard constraints) — that file is the living source
of truth; this file is narrative.

The session started from a review of ten `todo:` comments (one written
`todo;` and missed by naive grep — grep for the bare word). All are resolved
or consciously deferred; the deferred ones are listed at the bottom.

## What landed

### Compile fixes + executor cleanup (in `b7f4d89`)

- Finished the interrupted `FormationTask` → `FormationOrder` rename in
  `player.rs`; gave `FormationOrder::Hold` a no-op transition arm (it
  formalizes the idle state — planning/dispatch treat it exactly like an
  empty queue; its `pos`/`facing_dir` fields are unused until something
  starts pushing it).
- `Rotate` no longer rewrites itself into a fake `Reform`; it sets `dir`
  and flags for re-mapping like a `Move` facing change. The `AssignmentJob`
  struct, the jobs-collection loop, `reform_front` plumbing, the
  `complete`/`usize::MAX` dressing, and both "member despawned mid-frame"
  guards are gone — assignment is one loop, and "pop when done" is a local
  `matches!` on `tasks.front()`.
- Stale-`FormationSlot` bug: verified real with a test (detach members,
  mirror their positions, regroup → slots stayed crossed `[2,1,0]` instead
  of re-deriving `[0,1,2]`), then fixed by removing `FormationSlot`
  together with membership on every detach path.
- `Formation::max_speed` is no longer spawn-site data: `#[require(
  NeedsSpeedInit)]` + `init_formation_speed` derive it from the member
  list a tick after creation (relationship targets aren't populated at
  spawn-hook time — that's why a hook can't work). Sub-formations
  initialize bottom-up; a parent seeing a still-marked child waits a tick.
  `from_member_speeds` is deleted.

### LOD: universal command path for the lowest loaded level (in `b7f4d89`)

Resolves the formations.rs:124 todo ("universal way to command the lowest
loaded level"). Design, refined from the user's wording ("receive direct
*tasks*"):

- **`Velocity` presence on a formation IS the marker** for "lowest loaded
  level of its branch". `propagate_formation_targets` maintains it with a
  local, bottom-up-consistent rule: lowest loaded iff no member is
  simulated (has `Velocity`) or is a container (has occupants). Unloading
  a formation's boids flips `Velocity` onto the formation; reloading flips
  it back.
- **The task queue stays the single command channel** — no last-writer
  conflicts. A lowest loaded formation executes its front order *on
  itself*: `Formation` `#[require]`s `Target` (the steering actuator), it
  keeps its integrated transform (no COM snapping — `move_step` owns it),
  and dispatch writes the intermediate goal into its own `Target`.
  Parents command it exactly like a loaded sub: by injecting a `Move`
  order into its queue.
- Two bugs this exposed: a container's COM only scanned boids (sub-only
  parents never propagated — nothing exercised `FormationOf` before), and
  a holding parent re-injected a `Move`-to-current-position into subs
  every frame (endless pop/re-inject churn). COM now covers everything
  simulated below; injection happens only while an order is active.
- `Rotate`/`Reform` on a self-simulated formation pop immediately in the
  transition system (no slots to re-map).

### Executor split into chained FixedUpdate systems (in `b7f4d89`)

`process_formation_orders` (325 lines) is deleted. The user asked why it
was long and approved the split; everything now lives in `FixedUpdate`,
explicitly chained (main.rs):

```
init_formation_speed → propagate_formation_targets →
transition_formation_orders → assign_slots → plan_formation_goals →
dispatch_formation_goals → follow_target
```

- Pipeline stages communicate through **components, not snapshot
  vectors**: `SlotsStale` (re-map needed; inserted by transitions,
  consumed+removed by `assign_slots`, which also pops the finished
  `Rotate`/`Reform`) and `FormationGoal` (the per-tick steering plan:
  COM/goal/facing/task_pos; overwritten in place via `&mut` while
  marching, inserted/removed when "something to steer" flips). Presence
  is the state.
- `plan_goal(&FormationFrame, &[Vec3]) -> Option<FormationGoal>` is
  **pure** (no world) and unit-tested with synthetic data — the project's
  "test without instantiating the world" convention, applied.
- The executor's old assignment pass merged into `assign_slots`
  (trigger: invalid OR stale), deleting ~60 lines of duplicated Morton
  code. **No `ParamSet` remains anywhere**: each system's accesses don't
  conflict; the chain's auto sync points make each stage's commands
  visible to the next.
- Consistency gains: slot re-maps are visible to planning the same tick;
  dispatch reads post-pop task state; orders pushed from `Update` systems
  are picked up on the next fixed step, deterministically.

### Test harness on fixed time (in `b7f4d89`)

Headless `test_app` learned real fixed-timestep semantics. Three traps,
all documented in the code:

1. A bare `App` never runs `FixedUpdate` — the fixed-loop runner
   (`run_fixed_main_schedule`) is added by **`TimePlugin`**, not by
   `App::new()`.
2. Drive time with `TimeUpdateStrategy::ManualDuration(dt)` (bevy's own
   test pattern), timestep set to 1/60 matching `tick(dt)`.
3. Bevy's **first `app.update()` never advances the clocks**
   (`update_with_instant` only records `first_update` on first call), so
   `test_app()` burns one warmup update before returning — every
   `tick()` afterwards is exactly one fixed step, pinned by
   `harness_advances_one_fixed_step_per_tick`.

### Slots for sub-formations, then the relationship merge (in `42bb710`)

Sequenced deliberately (slots first — merging with boid-only slot
invariants would break the validity check):

- `assign_slots` assigns over **all occupants** — sub-formations matched
  by their origins through the same Morton pairing. `member_total` is
  gone; there is just `total = occupants.len()`. Dispatch reads each
  sub's assigned slot (positional expression survives only as the
  pre-assignment fallback). The recurring boids-then-subs iteration is
  one `occupants(members)` helper.
- `FormationOf`/`Formations` are **deleted**: one `MemberOf`/`Members`
  relationship covers boids and sub-formations alike. Dispatch branches
  per occupant on `Has<Formation>`: injected order vs `Target` write.
  Quick-group detach is a single path (`remove::<MemberOf>()` +
  `remove::<FormationSlot>()` for every occupant). Tests attach subs
  with plain `MemberOf(parent)`; all LOD/slot/march tests passed
  unchanged after the merge.

## Deliberate decisions (user)

- **Uniform slot spacing stays.** Heterogeneous/extent-based offsets for
  sub-formation occupants are deferred; noted on the `Members` doc.
- **`FormationSlot` is NOT folded into `MemberOf`** (`MemberOf { formation,
  slot }` is legal in Bevy 0.19 — extra relationship fields). The
  alternative stays documented as a comment on `FormationSlot`; revisit
  if detach paths multiply.
- "Doesn't need to be frame-perfect, but consistency would be nice" →
  the whole executor pipeline on `FixedUpdate`, explicitly chained.

## Tests (16, all green; wasm check clean)

March/reorient (2), stale-slot regression, `init_formation_speed` pending
propagation, lowest-loaded self-execution, parent→lowest-loaded-sub
through orders, sub-holds-slot-in-parent-layout, harness fixed-step
guarantee, `plan_goal` unit tests (5), Morton solver tests (3, incl. the
10k perf budget).

## Known issues / leftover

- **`cargo test --release` fails to link on this Windows machine** (LNK2019
  on bevy_reflect `type_info` statics). Verified pre-existing at `f966c00`
  in a throwaway worktree — not caused by this work; the `.cargo/config.toml`
  LLVM override (also pre-existing) doesn't fix the release-test target.
  The perf test passes its debug budget; re-verify in release once the
  link issue is solved.
- Open todos by design: `Rotate` **wheeling** (smooth rotation retaining
  slots, `Reform` only for excessive rotation — the doc todo on
  `FormationOrder::Rotate`), the `horse.rs` cavalry stubs, extent-based
  slot spacing, slot-into-relationship fold.
