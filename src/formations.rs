use crate::kinematics::Velocity;
use crate::target::Target;
use bevy::prelude::*;
use std::collections::VecDeque;

/// A maneuver a formation executes, one at a time, front of the queue first.
/// Player/ai code only *enqueues* tasks; the [`process_formation_orders`] system
/// executes them and pops each as it finishes.
#[derive(Debug, Clone, Copy)]
pub enum FormationOrder {
    /// March the formation to a world position, presenting `facing_dir`.
    /// On start, if `facing_dir` differs from the current facing, members are
    /// re-mapped to slots in the new frame (symmetric formations re-orient
    /// without moving: different slot, same position). Finished when the
    /// center of mass arrives within [`ARRIVE_TOLERANCE`] of `pos`.
    Move { pos: Vec3, facing_dir: Vec3 },
    /// Re-fill slots from current member positions (after a kind/column
    /// change, or when a boid died or left). Finished once every member has
    /// a valid slot.
    Reform,
    /// Rotate the slot frame to `to`; executes as a facing change followed by
    /// a [`FormationOrder::Reform`].
    /// todo: this is wrong. Rotate should be smooth rotation about center with
    /// units retaining their slots. Reform is for when rotation is excessive and
    /// the individual units can just change direction or after a formation changes.
    Rotate { to: Vec3 },
    /// Hold position. This is the default order that doesn't get removed.
    Hold { pos: Vec3, facing_dir: Vec3 }
}

/// A formation groups boids (and possibly sub-formations) and assigns each
/// member a target position relative to the formation origin.
///
/// The origin is stored as the entity's [`Transform`]; the desired origin (used
/// when this formation is itself a member of a parent formation) in [`Target`].
#[derive(Component)]
#[require(NeedsSpeedInit, Target)]
pub struct Formation {
    /// Maps member index -> desired position relative to the formation origin.
    /// Intended to become player-defined, with maneuvers transitioning
    /// between kinds (e.g. blending offsets over time).
    pub kind: FormationKind,
    /// Column override for [`FormationKind::Grid`]: when set, the grid lays
    /// out `columns` wide regardless of member count (rows grow instead).
    /// Set by Ctrl+RMB frontage designation to fit the formation to the
    /// dragged frontage width.
    pub columns: Option<usize>,
    /// Side length of a square (centered on the origin) that encloses all
    /// member slots of the current kind/member count. Maintained by
    /// [`propagate_formation_targets`].
    pub extent: f32,
    /// Where members face (per frontage designation).
    pub dir: Vec3,
    /// The formation's maximum movement speed: the slowest member's max
    /// speed (MAX_VELOCITY for plain boids), derived from the member list by
    /// [`init_formation_speed`] shortly after creation - creation sites
    /// never set it by hand. Drives the intermediate-goal lead distance.
    pub max_speed: f32,
    /// Pending maneuvers, executed front-to-back; an empty queue means
    /// plain marching.
    pub tasks: VecDeque<FormationOrder>,
}

/// Marker: [`Formation::max_speed`] has not been derived from the member
/// list yet. Inserted automatically with `Formation` (via `require`), so
/// every creation path gets it; [`init_formation_speed`] removes it once
/// the speed is computed. Presence is the state - no parallel bool flags.
#[derive(Component, Default)]
pub struct NeedsSpeedInit;

impl Default for Formation {
    fn default() -> Self {
        Self {
            kind: FormationKind::default(),
            columns: None,
            extent: FormationKind::SPACING,
            dir: Vec3::ZERO,
            max_speed: crate::kinematics::MAX_VELOCITY,
            tasks: VecDeque::new(),
        }
    }
}

impl Formation {
    /// Slot offset honoring the column override (Grid only).
    pub fn slot_offset(&self, index: usize, total: usize) -> Vec3 {
        self.kind.offset_with_cols(index, total, self.columns)
    }

    /// Enclosing-square side honoring the column override (Grid only).
    pub fn slot_extent(&self, total: usize) -> f32 {
        self.kind.extent_with_cols(total, self.columns)
    }
}

/// Slot identity of a boid within its formation: member `FormationSlot(i)`
/// occupies slot `i` of [`FormationKind::offset`]. Slots are persistent; if a
/// boid dies or leaves, [`assign_slots`] backfills vacancies with the
/// remaining members, minimizing total movement.
///
/// Persistence is deliberate: re-deriving slots every frame would reshuffle
/// members (jitter), so slots only change when the current assignment is
/// invalidated (membership, kind/column, or facing change). Because a slot is
/// meaningless without membership, every detach path must remove this
/// component together with `MemberOf` - a stale slot passes the validity
/// check in [`assign_slots`] and pins the member to an arbitrary slot in its
/// next formation.
///
/// Alternative considered: fold the slot into the relationship component as
/// `MemberOf { formation: Entity, slot: usize }` (Bevy 0.19 relationship
/// components may carry extra fields). Membership and slot would then be
/// created/dropped atomically and stale slots would be impossible; the cost
/// is losing `With<FormationSlot>` query filters and having to mutate the
/// relationship component on every re-map. Revisit if detach paths multiply.
#[derive(Component, Copy, Clone, Debug, PartialEq, Eq)]
pub struct FormationSlot(pub usize);

/// Relationship: this entity (a boid) is a member of a formation.
#[derive(Component)]
#[relationship(relationship_target = Members)]
pub struct MemberOf(pub Entity);

/// Reverse relationship: all boids that are members of this formation.
#[derive(Component)]
#[relationship_target(relationship = MemberOf)]
pub struct Members(Vec<Entity>);

/// Relationship: this formation is a sub-formation of a larger formation.
#[derive(Component)]
#[relationship(relationship_target = Formations)]
pub struct FormationOf(pub Entity);

/// Reverse relationship: all sub-formations of this formation.
///
/// Order dispatch does NOT depend on this split anymore: a lowest loaded
/// member (see [`propagate_formation_targets`]) executes orders on itself
/// either way, and containers propagate down whatever they are. The split
/// remains because boids and sub-formations consume different mechanics:
/// boids hold [`FormationSlot`]s and receive `Target`s, sub-formations have
/// their own kind/extent and receive injected orders.
/// todo: give sub-formations slots too (replacing the positional
/// `members.len() + i` indexing in `process_formation_orders`), then merge
/// `MemberOf`/`FormationOf` into one relationship - with dispatch
/// marker-based, the second relationship has no remaining job.
#[derive(Component)]
#[relationship_target(relationship = FormationOf)]
pub struct Formations(Vec<Entity>);

/// Quick-command-group slot (RTS hotkey groups 1-6).
#[derive(Component)]
pub struct QuickCommandGroup(pub u8);

/// Simple built-in formation functions. X = right, Z = forward, on the ground
/// plane; the formation origin is at the centroid of its slots.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum FormationKind {
    Line,
    Column,
    #[default]
    Grid,
    Wedge,
    Ring,
}

impl FormationKind {
    pub const SPACING: f32 = 2.0;

    /// Number of wedge rows needed for `total` members (rows of 1, 2, 3, ...).
    fn wedge_rows(total: usize) -> usize {
        let mut rows = 1;
        while rows * (rows + 1) / 2 < total {
            rows += 1;
        }
        rows
    }

    /// Grid column count for `total` members: the override when set (Ctrl+RMB
    /// width fitting), else the near-square default.
    fn grid_cols(&self, total: usize, cols: Option<usize>) -> usize {
        cols.unwrap_or_else(|| (total as f32).sqrt().ceil().max(1.0) as usize)
            .max(1)
    }

    /// Side of the square (centered on the formation origin) that encloses
    /// all member slots for this kind and member count.
    pub fn extent(&self, total: usize) -> f32 {
        self.extent_with_cols(total, None)
    }

    /// [`extent`](Self::extent) honoring a Grid column override.
    pub fn extent_with_cols(&self, total: usize, cols: Option<usize>) -> f32 {
        const S: f32 = FormationKind::SPACING;
        if total == 0 {
            return S;
        }
        let side = match self {
            FormationKind::Line | FormationKind::Column => (total.saturating_sub(1)) as f32 * S,
            FormationKind::Grid => {
                let cols = self.grid_cols(total, cols);
                let rows = total.div_ceil(cols);
                ((cols - 1) as f32 * S).max((rows - 1) as f32 * S)
            }
            FormationKind::Wedge => {
                // Last (widest) row has `rows` members, rows extend forward.
                let rows = Self::wedge_rows(total);
                (rows - 1) as f32 * S
            }
            FormationKind::Ring => 2.0 * (total as f32 * S / std::f32::consts::TAU).max(S),
        };
        side.max(S)
    }

    /// Desired position of the member with `index` (out of `total` members,
    /// counting both boids and sub-formations) relative to the formation origin.
    pub fn offset(&self, index: usize, total: usize) -> Vec3 {
        self.offset_with_cols(index, total, None)
    }

    /// [`offset`](Self::offset) honoring a Grid column override.
    pub fn offset_with_cols(&self, index: usize, total: usize, cols: Option<usize>) -> Vec3 {
        const S: f32 = FormationKind::SPACING;
        match self {
            FormationKind::Line => {
                let c = (total.saturating_sub(1)) as f32 * S / 2.0;
                Vec3::new(index as f32 * S - c, 0.0, 0.0)
            }
            FormationKind::Column => {
                let c = (total.saturating_sub(1)) as f32 * S / 2.0;
                Vec3::new(0.0, 0.0, index as f32 * S - c)
            }
            FormationKind::Grid => {
                let cols = self.grid_cols(total, cols);
                let rows = total.div_ceil(cols);
                let col = index % cols;
                let row = index / cols;
                Vec3::new(
                    col as f32 * S - (cols - 1) as f32 * S / 2.0,
                    0.0,
                    row as f32 * S - (rows - 1) as f32 * S / 2.0,
                )
            }
            FormationKind::Wedge => {
                // Rows of 1, 2, 3, ... members, apex pointing +Z.
                let mut row = 0;
                let mut before = 0; // members in rows before `row`
                while before + (row + 1) <= index {
                    before += row + 1;
                    row += 1;
                }
                let in_row = index - before;
                let row_len = row + 1;
                let rows = Self::wedge_rows(total);
                Vec3::new(
                    in_row as f32 * S - (row_len - 1) as f32 * S / 2.0,
                    0.0,
                    row as f32 * S - (rows - 1) as f32 * S / 2.0,
                )
            }
            FormationKind::Ring => {
                let radius = (total as f32 * S / std::f32::consts::TAU).max(S);
                let angle = index as f32 / total as f32 * std::f32::consts::TAU;
                Vec3::new(angle.cos() * radius, 0.0, angle.sin() * radius)
            }
        }
    }
}

/// Level-of-detail control for formation simulation. The idea: exactly one
/// level of the formation hierarchy is "the lowest loaded one" per branch -
/// below it, member boids/sub-formations are not simulated individually.
/// That level carries a [`Velocity`] (maintained by
/// [`propagate_formation_targets`]): it integrates like a boid, receives
/// its commands through the ordinary task queue, and executes them on
/// itself; levels above propagate orders downward instead.
#[derive(Resource, Debug)]
pub struct LODGuard {
    /// Propagate parent formation targets to *direct* members only.
    /// Nested levels converge on later frames (each level re-emits to its own
    /// members), so disabling this cheaply freezes distant detail.
    pub propagate_targets: bool,
}

impl Default for LODGuard {
    fn default() -> Self {
        Self {
            propagate_targets: true,
        }
    }
}

/// Derive [`Formation::max_speed`] = min over member max speeds, once per
/// formation. `NeedsSpeedInit` rides along with every `Formation` spawn
/// (`require`), so all creation paths are covered without spawn-site code.
/// Relationship targets are only populated after the spawn commands apply,
/// which is why this is a system in a later frame rather than a spawn hook.
/// Sub-formations initialize bottom-up: while a child still carries the
/// marker its `max_speed` is the default, so the parent waits a tick instead
/// of reading a bogus speed. Plain boids have no per-entity speed (yet) and
/// contribute `MAX_VELOCITY`.
pub fn init_formation_speed(
    q_marked: Query<
        (Entity, Option<&Members>, Option<&Formations>),
        (With<Formation>, With<NeedsSpeedInit>),
    >,
    q_details: Query<(&Formation, Option<&NeedsSpeedInit>)>,
    mut commands: Commands,
) {
    for (entity, members, subs) in &q_marked {
        let mut max_speed = f32::INFINITY;
        let mut pending = false;
        let mut any = false;
        // `Members` only exists once a boid has attached; sub-only parents
        // carry `Formations` alone.
        let members = members
            .into_iter()
            .flat_map(|m| m.iter())
            .chain(subs.into_iter().flat_map(|s| s.iter()));
        for member in members {
            any = true;
            match q_details.get(member) {
                Ok((child, child_pending)) => {
                    pending |= child_pending.is_some();
                    max_speed = max_speed.min(child.max_speed);
                }
                Err(_) => max_speed = max_speed.min(crate::kinematics::MAX_VELOCITY),
            }
        }
        if !any || pending {
            continue; // still assembling, or a sub-formation is not initialized yet
        }
        let speed = max_speed.min(crate::kinematics::MAX_VELOCITY);
        commands.queue(move |world: &mut World| {
            if let Some(mut formation) = world.get_mut::<Formation>(entity) {
                formation.max_speed = speed;
            }
            world.entity_mut(entity).remove::<NeedsSpeedInit>();
        });
    }
}

/// Maintain per-formation bookkeeping (extent) and the LOD Velocity split:
/// a formation WITH `Velocity` is the lowest loaded level of its branch -
/// nothing below it needs simulating, so it integrates like a single boid
/// (`move_step` + `follow_target`) and [`process_formation_orders`] executes
/// its orders on itself. WITHOUT `Velocity` it is a container that
/// propagates orders down to its members instead.
///
/// The rule is local and bottom-up consistent: a formation is lowest loaded
/// if none of its members is simulated (carries `Velocity`) or is itself a
/// container (has relationship targets). Unloading a formation's boids
/// (removing their detail) therefore flips `Velocity` onto the formation;
/// reloading flips it back off. `LODGuard` freezes all of this when
/// propagation is off.
pub fn propagate_formation_targets(
    lod: Res<LODGuard>,
    mut q_formations: Query<
        (
            Entity,
            &mut Formation,
            Option<&Members>,
            Option<&Formations>,
            Option<&Velocity>,
        ),
        With<Formation>,
    >,
    q_member_state: Query<(Has<Velocity>, Has<Members>, Has<Formations>)>,
    mut commands: Commands,
) {
    if !lod.propagate_targets {
        return;
    }
    for (entity, mut formation, members, subs, velocity) in &mut q_formations {
        let total = members.map_or(0, |m| m.len()) + subs.map_or(0, |s| s.len());
        formation.extent = formation.slot_extent(total);

        // Lowest loaded iff nothing below is simulated or propagates
        // further. Unresolvable members are gone; they simulate nothing.
        let should_have_velocity = members
            .into_iter()
            .flat_map(|m| m.iter())
            .chain(subs.into_iter().flat_map(|s| s.iter()))
            .filter_map(|m| q_member_state.get(m).ok())
            .all(|(simulated, has_members, has_formations)| {
                !simulated && !has_members && !has_formations
            });
        match (velocity.is_some(), should_have_velocity) {
            (true, false) => {
                commands.entity(entity).remove::<Velocity>();
            }
            (false, true) => {
                commands.entity(entity).insert(Velocity::default());
            }
            _ => {}
        }
    }
}

/// Automatic slot maintenance: (re)assigns members whenever the current
/// assignment is invalid - group creation (no slots yet), a member dying or
/// leaving (gap), or a kind/column change. Explicit player-driven reforms go
/// through [`FormationOrder::Reform`] in [`process_formation_orders`]; both paths
/// share [`assign_slots_nearest`].
pub fn assign_slots(
    q_formations: Query<
        (Entity, &Transform, &Formation, Option<&Members>, Option<&Formations>),
        With<Formation>,
    >,
    q_members: Query<(&Transform, Option<&FormationSlot>)>,
    mut commands: Commands,
) {
    for (entity, transform, formation, members, subs) in &q_formations {
        let member_total = members.map_or(0, |m| m.len());
        let total = member_total + subs.map_or(0, |s| s.len());
        if total == 0 {
            continue;
        }

        // Validity check (cheap): every member has an in-range, unique slot.
        let mut valid = true;
        let mut seen = vec![false; total];
        for member in members.into_iter().flat_map(|m| m.iter()) {
            match q_members.get(member) {
                Ok((_, Some(slot)))
                    if (slot.0 as usize) < member_total && !seen[slot.0 as usize] =>
                {
                    seen[slot.0 as usize] = true;
                }
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if valid {
            continue;
        }

        let rotation = yaw_quat(formation.dir).unwrap_or(Quat::IDENTITY);
        let origin = transform.translation;
        let slot_positions: Vec<Vec3> = (0..member_total)
            .map(|i| origin + rotation * formation.slot_offset(i, total))
            .collect();
        let member_positions: Vec<(Entity, Vec3)> = members
            .into_iter()
            .flat_map(|m| m.iter())
            .filter_map(|m| q_members.get(m).ok().map(|(t, _)| (m, t.translation)))
            .collect();
        // Members cannot despawn mid-system (commands are deferred), so every
        // member resolves; a short list would mis-pair the Morton matching.
        debug_assert_eq!(member_positions.len(), member_total);
        let assignment = assign_slots_nearest(origin, &member_positions, &slot_positions);
        for (&(member, _), &slot) in member_positions.iter().zip(&assignment) {
            if slot != usize::MAX {
                commands.entity(member).insert(FormationSlot(slot));
            }
        }
    }
}

/// Solve members -> slots by Morton-order matching: both sides are sorted
/// by a 2D Z-order curve code of their positions and paired in order.
/// Locality-preserving (nearby members go to nearby slots), deterministic,
/// and O((members + slots) log) with no contention handling - the pairing is
/// a pure sort, independent of how slot positions were generated (custom
/// formation functions included). A re-orientation to a symmetric layout
/// maps each member onto the slot now at its own position ("different slot,
/// same position") because the slot point set is unchanged.
fn assign_slots_nearest(
    _origin: Vec3,
    member_positions: &[(Entity, Vec3)],
    slot_positions: &[Vec3],
) -> Vec<usize> {
    let n = member_positions.len();
    debug_assert_eq!(n, slot_positions.len());

    let mut members: Vec<usize> = (0..n).collect();
    let mut slots: Vec<usize> = (0..n).collect();
    members.sort_by_key(|&i| morton2(member_positions[i].1));
    slots.sort_by_key(|&s| morton2(slot_positions[s]));

    let mut result = vec![usize::MAX; n];
    for (&m, &s) in members.iter().zip(&slots) {
        result[m] = s;
    }
    result
}

/// 2D Morton (Z-order) code of the ground-plane projection, quantized to
/// 0.25 world units. Purely for locality ordering - collisions are harmless.
fn morton2(p: Vec3) -> u64 {
    const BITS: u32 = 21;
    const SCALE: f32 = 4.0; // units per bit step (0.25 per step)
    let qx = ((p.x * SCALE).round() as i64 + (1 << BITS) / 2).clamp(0, (1 << BITS) - 1) as u64;
    let qz = ((p.z * SCALE).round() as i64 + (1 << BITS) / 2).clamp(0, (1 << BITS) - 1) as u64;
    let mut code = 0u64;
    for i in 0..BITS {
        code |= ((qx >> i) & 1) << (2 * i);
        code |= ((qz >> i) & 1) << (2 * i + 1);
    }
    code
}

/// Yaw-only facing quaternion for a ground-plane direction.
fn yaw_quat(dir: Vec3) -> Option<Quat> {
    let d = dir.normalize_or_zero();
    if d.length_squared() < 1e-6 {
        None
    } else {
        Some(Quat::from_rotation_y(d.x.atan2(d.z)))
    }
}

/// Task executor: runs the front of each formation's task queue until
/// finished, then pops it and starts the next. With an empty queue the
/// formation holds position: the origin tracks the members' center of mass
/// and members hold their slots in the current facing.
///
/// Every frame, for every formation:
/// 1. Task transitions: `Rotate` and a `Move` whose facing differs turn the
///    slot frame and flag the formation for member re-mapping (symmetric
///    formations re-orient without moving: different slot, same position);
///    `Reform` flags it unconditionally. The assignment pass then re-maps
///    members to slots via Morton-order locality matching and pops the
///    finished `Rotate`/`Reform`.
/// 2. The origin snaps to the center of mass of the members; a finished
///    `Move` (center of mass within [`ARRIVE_TOLERANCE`] of `pos`) pops.
/// 3. Targets propagate from the *intermediate goal*: for `Move`, offset
///    from the center of mass toward `pos` by `slowest_member_speed *
///    LEAD_TIME`, clamped to the remaining distance - members keep
///    formation along the path and are never asked to cover more than the
///    lead distance. Otherwise the goal is the center of mass itself (hold).
///
/// A lowest loaded formation (carrying `Velocity`, see
/// [`propagate_formation_targets`]) is instead simulated as one unit: it
/// keeps its integrated transform (no center-of-mass snapping) and step 3
/// writes the intermediate goal into its own `Target`, so orders execute on
/// the formation itself rather than propagating to unloaded members.
pub const LEAD_TIME: f32 = 10.0;

/// Minimum lead distance so a formation ordered to march from a standstill
/// bootstraps: without it, lead = slowest x LEAD_TIME is zero at rest, and
/// the goal lands on the center of mass (no member ever gains speed).
pub const MIN_LEAD: f32 = 2.0 * FormationKind::SPACING;

/// Center-of-mass arrival tolerance for [`FormationOrder::Move`].
pub const ARRIVE_TOLERANCE: f32 = 2.0;

#[allow(clippy::type_complexity)]
pub fn process_formation_orders(
    mut params: ParamSet<(
        Query<
            (
                Entity,
                &mut Transform,
                &mut Formation,
                Option<&Members>,
                Option<&Formations>,
                Option<&Velocity>,
            ),
            With<Formation>,
        >,
        Query<(&Transform, &Velocity, Option<&FormationSlot>)>,
        Query<&mut Target>,
    )>,
    mut commands: Commands,
    mut gizmos: Gizmos,
) {
    // Pass A - task transitions. Mutates Formation state only; slot
    // assignment follows right after (it needs read-only access to both
    // formations and member transforms).
    let mut needs_assign: Vec<Entity> = Vec::new();
    for (formation_entity, _, mut formation, _, _, _) in params.p0().iter_mut() {
        let Some(task) = formation.tasks.front().copied() else {
            continue;
        };
        match task {
            FormationOrder::Rotate { to } => {
                formation.dir = to;
                needs_assign.push(formation_entity);
                // Popped in the assignment pass below once members re-map.
            }
            FormationOrder::Reform => {
                needs_assign.push(formation_entity);
                // Popped in the assignment pass below when members re-map.
            }
            FormationOrder::Move { facing_dir, .. } => {
                // Facing change: re-map slots into the new frame before marching.
                if formation.dir.distance_squared(facing_dir) > 1e-4 {
                    formation.dir = facing_dir;
                    needs_assign.push(formation_entity);
                }
            }
            // Hold formalizes the idle state; the passes below treat it
            // exactly like an empty queue (hold at the center of mass).
            FormationOrder::Hold { .. } => {}
        }
    }

    // Assignment pass: re-map the members of the formations flagged above
    // into their (possibly re-oriented) slot frames. Each formation's frame
    // data is copied out so the p0 borrow ends before member transforms are
    // read through p1; assignment is immediate, no intermediate jobs.
    for formation_entity in needs_assign {
        let q0 = params.p0();
        let Ok((_, transform, formation, members, subs, velocity)) = q0.get(formation_entity)
        else {
            continue;
        };
        // A lowest loaded formation's members are abstracted away (they do
        // not even match p1, which requires Velocity); slot re-mapping
        // waits until they reload - `assign_slots` revalidates then.
        if velocity.is_some() {
            continue;
        }
        let member_total = members.map_or(0, |m| m.len());
        if member_total == 0 {
            continue;
        }
        let total = member_total + subs.map_or(0, |s| s.len());
        let origin = transform.translation;
        let rotation = yaw_quat(formation.dir).unwrap_or(Quat::IDENTITY);
        let slot_positions: Vec<Vec3> = (0..member_total)
            .map(|i| origin + rotation * formation.slot_offset(i, total))
            .collect();
        let member_ids: Vec<Entity> = members
            .into_iter()
            .flat_map(|m| m.iter())
            .collect();
        // Only Reform and Rotate finish by re-mapping; a Move facing change
        // re-maps but keeps marching.
        let pop_when_assigned = matches!(
            formation.tasks.front(),
            Some(FormationOrder::Reform | FormationOrder::Rotate { .. })
        );
        // Everything above is owned; the p0 borrow ends here.

        let member_positions: Vec<(Entity, Vec3)> = member_ids
            .iter()
            .filter_map(|&m| params.p1().get(m).ok().map(|(t, _, _)| (m, t.translation)))
            .collect();
        // Members cannot despawn mid-system (commands are deferred), so every
        // member resolves; a short list would mis-pair the Morton matching.
        debug_assert_eq!(member_positions.len(), member_total);
        let assignment = assign_slots_nearest(origin, &member_positions, &slot_positions);
        for (&(member, _), &slot) in member_positions.iter().zip(&assignment) {
            if slot != usize::MAX {
                commands.entity(member).insert(FormationSlot(slot));
            }
        }
        if pop_when_assigned {
            commands.queue(move |world: &mut World| {
                let poppable = matches!(
                    world
                        .get::<Formation>(formation_entity)
                        .and_then(|f| f.tasks.front().copied()),
                    Some(FormationOrder::Reform | FormationOrder::Rotate { .. })
                );
                if poppable {
                    world
                        .get_mut::<Formation>(formation_entity)
                        .expect("formation existed a moment ago")
                        .tasks
                        .pop_front();
                }
            });
        }
    }

    // Pass B - snapshot members (slots), the active task, max speed, and
    // whether the formation itself is the lowest loaded level (carries
    // `Velocity`). `max_speed` (derived by `init_formation_speed` after
    // creation) drives the lead distance; the slowest-member scan is gone.
    struct Snapshot {
        entity: Entity,
        own_pos: Vec3,
        self_simulated: bool,
        member_slots: Vec<(Entity, Option<usize>)>,
        subs: Vec<Entity>,
        task: Option<FormationOrder>,
        max_speed: f32,
    }
    let mut snapshots: Vec<Snapshot> = params
        .p0()
        .iter()
        .map(|(entity, transform, formation, members, subs, velocity)| Snapshot {
            entity,
            own_pos: transform.translation,
            self_simulated: velocity.is_some(),
            member_slots: members
                .into_iter()
                .flat_map(|m| m.iter())
                .map(|m| (m, None))
                .collect(),
            subs: subs.into_iter().flat_map(|s| s.iter()).collect(),
            task: formation.tasks.front().copied(),
            max_speed: formation.max_speed,
        })
        .collect();
    for snapshot in &mut snapshots {
        for (member, slot) in &mut snapshot.member_slots {
            if let Ok((_, _, s)) = params.p1().get(*member) {
                *slot = s.map(|s| s.0);
            }
        }
    }

    struct Plan {
        center_of_mass: Vec3,
        goal: Vec3,
        facing: Vec3,
        task_pos: Option<Vec3>,
    }
    let mut plans: Vec<Option<Plan>> = Vec::with_capacity(snapshots.len());
    for snapshot in &snapshots {
        // A formation simulated as a unit IS its own center of mass; a
        // container's is the center of mass of everything simulated below
        // it: loaded boids and lowest loaded sub-formations alike.
        let com = if snapshot.self_simulated {
            snapshot.own_pos
        } else {
            let mut com = Vec3::ZERO;
            let mut count = 0usize;
            let simulated = snapshot
                .member_slots
                .iter()
                .map(|(m, _)| *m)
                .chain(snapshot.subs.iter().copied());
            for member in simulated {
                if let Ok((transform, _, _)) = params.p1().get(member) {
                    com += transform.translation;
                    count += 1;
                }
            }
            if count == 0 {
                plans.push(None);
                continue;
            }
            com / count as f32
        };

        // Active Move: goal is the intermediate point toward the task
        // position. Anything else (idle, Reform, pre-Move): hold at COM.
        let (goal, facing, task_pos) = match snapshot.task {
            Some(FormationOrder::Move { pos, facing_dir }) => {
                let to_target = pos - com;
                let distance = to_target.length();
                let lead = (snapshot.max_speed * LEAD_TIME)
                    .max(MIN_LEAD)
                    .min(distance);
                let goal = if distance > 1e-4 {
                    com + to_target * (lead / distance)
                } else {
                    pos
                };
                (goal, facing_dir, Some(pos))
            }
            _ => (com, Vec3::ZERO, None),
        };

        plans.push(Some(Plan {
            center_of_mass: com,
            goal,
            facing,
            task_pos,
        }));
    }

    // Pass C - snap the origin to the center of mass (containers only; a
    // self-simulated formation's transform belongs to `move_step`); marker
    // rotation follows the effective facing; pop finished Move tasks.
    for ((_, mut transform, mut formation, _, _, velocity), plan) in
        params.p0().iter_mut().zip(&plans)
    {
        let Some(plan) = plan else {
            continue;
        };
        if velocity.is_none() {
            transform.translation = plan.center_of_mass;
        }
        let facing = if plan.facing == Vec3::ZERO {
            formation.dir
        } else {
            plan.facing
        };
        if let Some(desired) = yaw_quat(facing) {
            transform.rotation = desired;
        }
        if let Some(pos) = plan.task_pos {
            if plan.center_of_mass.distance(pos) < ARRIVE_TOLERANCE {
                formation.tasks.pop_front();
            }
        }
    }

    // Pass D - propagate the goal downward. A lowest loaded formation
    // (carries `Velocity`) executes its order on itself: Target is the
    // steering actuator and the queue stays the single command channel.
    // Otherwise members are placed by slot identity (list order fallback
    // before the first assignment): boids get Target directly;
    // sub-formations receive a Move task (the parent fully dictates the
    // child's placement) and propagate next frame.
    let mut assignments: Vec<(Entity, Vec3, Vec3)> = Vec::new();
    let mut sub_tasks: Vec<(Entity, FormationOrder)> = Vec::new();
    for (((entity, _, formation, _, subs, velocity), snapshot), plan) in
        params.p0().iter().zip(&snapshots).zip(&plans)
    {
        let Some(plan) = plan else {
            continue;
        };
        let facing = if plan.facing == Vec3::ZERO {
            formation.dir
        } else {
            plan.facing
        };
        let rotation = yaw_quat(facing).unwrap_or(Quat::IDENTITY);
        if velocity.is_some() {
            assignments.push((entity, plan.goal, facing));
        } else {
            let member_slots = &snapshot.member_slots;
            let total = member_slots.len() + subs.map_or(0, |s| s.len());
            for ((member, slot), fallback) in member_slots.iter().zip(0..) {
                let slot = slot.unwrap_or(fallback);
                assignments.push((
                    *member,
                    plan.goal + rotation * formation.slot_offset(slot, total),
                    facing,
                ));
            }
            if let Some(subs) = subs {
                // Inject only while an order is active: a holding parent
                // leaves the sub's queue alone, so the sub finishes its
                // last injected order and goes idle (empty queue) instead
                // of receiving a Move-to-current-position every frame.
                if plan.task_pos.is_some() {
                    for (i, sub) in subs.iter().enumerate() {
                        sub_tasks.push((
                            sub,
                            FormationOrder::Move {
                                pos: plan.goal
                                    + rotation
                                        * formation.slot_offset(member_slots.len() + i, total),
                                facing_dir: facing,
                            },
                        ));
                    }
                }
            }
        }
        // Debug: center of mass -> goal, and goal -> final task target.
        gizmos.line(plan.center_of_mass, plan.goal, Color::srgb(0.2, 0.6, 1.0));
        if let Some(pos) = plan.task_pos {
            let lifted = pos + Vec3::new(0.0, 0.2, 0.0);
            if plan.goal.distance(lifted) > 1e-3 {
                gizmos.line(plan.goal, lifted, Color::srgb(0.5, 0.5, 0.5));
            }
        }
    }
    for (entity, pos, dir) in assignments {
        if let Ok(mut target) = params.p2().get_mut(entity) {
            target.pos = pos;
            target.dir = dir;
        }
    }
    for (sub, task) in sub_tasks {
        commands.queue(move |world: &mut World| {
            if let Some(mut formation) = world.get_mut::<Formation>(sub) {
                // The parent dictates the sub-formation's orders wholesale.
                formation.tasks.clear();
                formation.tasks.push_back(task);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::{TrackedByTree, move_step};
    use crate::target::follow_target;
    use bevy::gizmos::AppGizmoBuilder;
    use bevy::gizmos::config::{DefaultGizmoConfigGroup, GizmoConfigStore};
    use bevy::time::Time;
    use std::time::Duration;

    /// Headless app: manual time, kd tree via the real AutomaticUpdate
    /// plugin (fast refresh), and the formation + kinematics pipeline in
    /// execution order.
    fn test_app() -> App {
        use bevy_spatial::{AutomaticUpdate, TransformMode};
        let mut app = App::new();
        app.init_resource::<Time>()
            .init_resource::<LODGuard>()
            .init_resource::<GizmoConfigStore>()
            .init_gizmo_group::<DefaultGizmoConfigGroup>()
            .init_resource::<Assets<GizmoAsset>>()
            .add_plugins(
                AutomaticUpdate::<TrackedByTree>::new()
                    .with_frequency(Duration::from_secs_f32(1.0 / 20.0))
                    .with_transform(TransformMode::Transform),
            )
            .add_systems(
                Update,
                (
                    init_formation_speed,
                    propagate_formation_targets,
                    assign_slots,
                    process_formation_orders,
                    follow_target,
                    move_step,
                )
                    .chain(),
            );
        app
    }

    fn tick(app: &mut App, dt: f32) {
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_secs_f32(dt));
        app.update();
    }

    fn spawn_formation(app: &mut App, positions: &[Vec3]) -> Entity {
        let formation = app
            .world_mut()
            .spawn((Formation::default(), Transform::default()))
            .id();
        for pos in positions {
            app.world_mut().spawn((
                Transform::from_translation(*pos),
                Velocity::default(),
                TrackedByTree,
                Target::default(),
                MemberOf(formation),
            ));
        }
        formation
    }

    fn center_of_mass(world: &mut World, formation: Entity) -> Vec3 {
        let mut query = world.query_filtered::<(&Transform, &MemberOf), With<Velocity>>();
        let mut com = Vec3::ZERO;
        let mut n = 0;
        for (transform, member_of) in query.iter(world) {
            if member_of.0 == formation {
                com += transform.translation;
                n += 1;
            }
        }
        com / n as f32
    }

    #[test]
    fn move_task_transports_formation_to_destination() {
        let mut app = test_app();
        let positions: Vec<Vec3> = (0..3)
            .flat_map(|r| {
                (0..3).map(move |c| Vec3::new(c as f32 * 2.0 - 2.0, 0.0, r as f32 * 2.0 - 2.0))
            })
            .collect();
        let formation = spawn_formation(&mut app, &positions);
        // Let the kd tree populate and slots get initially assigned.
        for _ in 0..10 {
            tick(&mut app, 1.0 / 60.0);
        }

        let dest = Vec3::new(60.0, 0.0, 40.0);
        app.world_mut()
            .get_mut::<Formation>(formation)
            .unwrap()
            .tasks
            .push_back(FormationOrder::Move {
                pos: dest,
                facing_dir: Vec3::new(0.0, 0.0, 1.0),
            });

        for _ in 0..900 {
            tick(&mut app, 1.0 / 60.0);
        }

        let world = app.world_mut();
        let com = center_of_mass(world, formation);
        assert!(
            com.distance(dest) < 5.0,
            "center of mass {com:?} did not reach {dest:?}"
        );
        let tasks = &world.get::<Formation>(formation).unwrap().tasks;
        assert!(tasks.is_empty(), "Move task should have finished");
    }

    #[test]
    fn move_task_reorients_without_moving_symmetric_formation() {
        let mut app = test_app();
        let positions: Vec<Vec3> = (0..3)
            .flat_map(|r| {
                (0..3).map(move |c| Vec3::new(c as f32 * 2.0 - 2.0, 0.0, r as f32 * 2.0 - 2.0))
            })
            .collect();
        let formation = spawn_formation(&mut app, &positions);
        for _ in 0..10 {
            tick(&mut app, 1.0 / 60.0);
        }

        let before: Vec<(Entity, Vec3, usize)> = {
            let world = app.world_mut();
            let mut query =
                world.query_filtered::<(Entity, &Transform, &FormationSlot), With<MemberOf>>();
            query
                .iter(world)
                .map(|(e, t, s)| (e, t.translation, s.0))
                .collect()
        };
        assert_eq!(before.len(), 9, "slots should be assigned");

        let com = center_of_mass(app.world_mut(), formation);
        // 180-degree reorientation to the same spot: different slot, same position.
        app.world_mut()
            .get_mut::<Formation>(formation)
            .unwrap()
            .tasks
            .push_back(FormationOrder::Move {
                pos: com,
                facing_dir: Vec3::new(0.0, 0.0, -1.0),
            });

        for _ in 0..240 {
            tick(&mut app, 1.0 / 60.0);
        }

        let world = app.world_mut();
        let mut query =
            world.query_filtered::<(Entity, &Transform, &FormationSlot), With<MemberOf>>();
        let after: Vec<(Entity, Vec3, usize)> = query
            .iter(world)
            .map(|(e, t, s)| (e, t.translation, s.0))
            .collect();
        let mut slots_changed = 0;
        for (entity, pos_after, slot_after) in &after {
            let (_, pos_before, slot_before) = before.iter().find(|(e, _, _)| e == entity).unwrap();
            assert!(
                pos_after.distance(*pos_before) < 1.5,
                "boid {entity:?} moved {} on reorientation",
                pos_after.distance(*pos_before)
            );
            if slot_before != slot_after {
                slots_changed += 1;
            }
        }
        assert!(
            slots_changed > 0,
            "a 180-degree reorientation must re-map slots"
        );
    }

    #[test]
    fn detached_members_do_not_carry_stale_slots() {
        // Regrouping detaches members and forms a new formation over them.
        // A stale FormationSlot passes the validity check in assign_slots
        // (in-range, unique), so the new formation would keep the old
        // mapping and members cross instead of re-deriving slots from their
        // current positions.
        let mut app = test_app();
        let positions = vec![
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let formation = spawn_formation(&mut app, &positions);
        for _ in 0..10 {
            tick(&mut app, 1.0 / 60.0);
        }
        // Line offsets ascend with +X and Morton matching is local, so the
        // leftmost boid holds slot 0 and the rightmost slot 2.
        let members: Vec<(f32, Entity, usize)> = {
            let world = app.world_mut();
            let mut query =
                world.query_filtered::<(Entity, &Transform, &FormationSlot), With<MemberOf>>();
            query
                .iter(world)
                .filter(|(e, _, _)| world.get::<MemberOf>(*e).unwrap().0 == formation)
                .map(|(e, t, s)| (t.translation.x, e, s.0))
                .collect()
        };
        assert_eq!(members.len(), 3, "slots should be assigned");
        let mut sorted = members.clone();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(
            sorted.iter().map(|(_, _, s)| *s).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );

        // Detach everyone (as quick-group overwrite does) and mirror their
        // positions, so the old mapping is exactly crossed.
        let formation_b = {
            let world = app.world_mut();
            for (_, e, _) in &members {
                world
                    .entity_mut(*e)
                    .remove::<MemberOf>()
                    .remove::<FormationSlot>();
            }
            for (x, e, _) in &members {
                world.get_mut::<Transform>(*e).unwrap().translation.x = -*x;
            }
            let formation_b = world
                .spawn((Formation::default(), Transform::default()))
                .id();
            for (_, e, _) in &members {
                world.entity_mut(*e).insert(MemberOf(formation_b));
            }
            formation_b
        };
        tick(&mut app, 1.0 / 60.0);

        // Slots must be re-derived from the mirrored positions: leftmost
        // boid -> 0, rightmost -> 2. With stale slots the mapping stays
        // crossed (the boid now on the left keeps slot 2).
        let world = app.world_mut();
        let mut query =
            world.query_filtered::<(Entity, &Transform, &FormationSlot), With<MemberOf>>();
        let mut remapped: Vec<(f32, usize)> = query
            .iter(world)
            .filter(|(e, _, _)| world.get::<MemberOf>(*e).unwrap().0 == formation_b)
            .map(|(e, t, s)| (t.translation.x, s.0))
            .collect();
        remapped.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(
            remapped.iter().map(|(_, s)| *s).collect::<Vec<_>>(),
            vec![0, 1, 2],
            "slots must be re-derived from member positions, not carried over"
        );
    }

    #[test]
    fn init_formation_speed_derives_from_slowest_subformation() {
        let mut app = test_app();
        // Sub-formation with boid members initializes to MAX_VELOCITY on the
        // first tick; the parent must wait for that before deriving its own
        // speed (a pending child still reports the default).
        let sub = spawn_formation(
            &mut app,
            &[Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
        );
        let parent = app
            .world_mut()
            .spawn((Formation::default(), Transform::default()))
            .id();
        app.world_mut().entity_mut(sub).insert(FormationOf(parent));
        tick(&mut app, 1.0 / 60.0);
        assert!(
            app.world().get::<NeedsSpeedInit>(parent).is_some(),
            "parent must wait while the sub-formation is pending"
        );

        // Slow the sub below the default and let the parent derive.
        app.world_mut().get_mut::<Formation>(sub).unwrap().max_speed = 3.0;
        tick(&mut app, 1.0 / 60.0);

        let world = app.world_mut();
        assert_eq!(world.get::<Formation>(parent).unwrap().max_speed, 3.0);
        assert!(world.get::<NeedsSpeedInit>(parent).is_none());
    }

    /// Members of a formation, queried from the world (helper).
    fn members_of(world: &mut World, formation: Entity) -> Vec<Entity> {
        let mut query = world.query_filtered::<Entity, With<MemberOf>>();
        query
            .iter(world)
            .filter(|e| world.get::<MemberOf>(*e).unwrap().0 == formation)
            .collect()
    }

    #[test]
    fn lowest_loaded_formation_executes_own_orders_as_a_unit() {
        let mut app = test_app();
        let formation = spawn_formation(
            &mut app,
            &[Vec3::new(-2.0, 0.0, 0.0), Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)],
        );
        for _ in 0..10 {
            tick(&mut app, 1.0 / 60.0);
        }
        let dest = Vec3::new(30.0, 0.0, 20.0);
        app.world_mut()
            .get_mut::<Formation>(formation)
            .unwrap()
            .tasks
            .push_back(FormationOrder::Move {
                pos: dest,
                facing_dir: Vec3::new(0.0, 0.0, 1.0),
            });
        // Unload the members: they keep existing but stop simulating (no
        // Velocity), so nothing below the formation is simulated.
        let members = members_of(app.world_mut(), formation);
        for &member in &members {
            app.world_mut().entity_mut(member).remove::<Velocity>();
        }
        tick(&mut app, 1.0 / 60.0);
        assert!(
            app.world().get::<Velocity>(formation).is_some(),
            "propagate must flip Velocity onto the lowest loaded formation"
        );

        // The members are unloaded: where they stand must not change.
        let frozen: Vec<(Entity, Vec3)> = members
            .iter()
            .map(|&m| (m, app.world().get::<Transform>(m).unwrap().translation))
            .collect();

        for _ in 0..900 {
            tick(&mut app, 1.0 / 60.0);
        }

        let world = app.world_mut();
        let pos = world.get::<Transform>(formation).unwrap().translation;
        assert!(
            pos.distance(dest) < 5.0,
            "formation {pos:?} did not reach {dest:?}"
        );
        assert!(
            world.get::<Formation>(formation).unwrap().tasks.is_empty(),
            "Move should have popped on arrival"
        );
        for (member, before) in frozen {
            let after = world.get::<Transform>(member).unwrap().translation;
            assert_eq!(after, before, "unloaded member {member:?} must not move");
        }
    }

    #[test]
    fn parent_commands_lowest_loaded_subformation_through_orders() {
        let mut app = test_app();
        let sub = spawn_formation(
            &mut app,
            &[Vec3::new(-2.0, 0.0, 0.0), Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)],
        );
        for _ in 0..10 {
            tick(&mut app, 1.0 / 60.0);
        }
        // Unload the sub's boids; the sub becomes the lowest loaded level
        // and the parent above it stays a container that propagates orders.
        let members = members_of(app.world_mut(), sub);
        for &member in &members {
            app.world_mut().entity_mut(member).remove::<Velocity>();
        }
        tick(&mut app, 1.0 / 60.0);
        assert!(app.world().get::<Velocity>(sub).is_some());

        let dest = Vec3::new(40.0, 0.0, -15.0);
        let parent = app
            .world_mut()
            .spawn((Formation::default(), Transform::default()))
            .id();
        app.world_mut().entity_mut(sub).insert(FormationOf(parent));
        app.world_mut()
            .get_mut::<Formation>(parent)
            .unwrap()
            .tasks
            .push_back(FormationOrder::Move {
                pos: dest,
                facing_dir: Vec3::new(0.0, 0.0, 1.0),
            });

        // The parent commands the sub through its task queue: a one-wide
        // formation's slot 0 is the origin, so the sub's goal is the
        // parent's goal. Mid-march the injected Move must be visible.
        for _ in 0..60 {
            tick(&mut app, 1.0 / 60.0);
        }
        assert!(
            app.world()
                .get::<Formation>(sub)
                .unwrap()
                .tasks
                .front()
                .is_some_and(|t| matches!(t, FormationOrder::Move { .. })),
            "parent must inject a Move order into the lowest loaded sub"
        );

        for _ in 0..900 {
            tick(&mut app, 1.0 / 60.0);
        }

        let world = app.world_mut();
        let sub_pos = world.get::<Transform>(sub).unwrap().translation;
        assert!(
            sub_pos.distance(dest) < 5.0,
            "sub-formation {sub_pos:?} did not reach {dest:?}"
        );
        for formation in [parent, sub] {
            assert!(
                world.get::<Formation>(formation).unwrap().tasks.is_empty(),
                "Move should have popped on arrival"
            );
        }
    }

    #[test]
    fn nearest_solver_maps_mirrored_layout_onto_itself() {
        // Slots and members identical: zero movement, identity mapping.
        let slots: Vec<Vec3> = (0..4).map(|i| Vec3::new(i as f32 * 2.0, 0.0, 0.0)).collect();
        let members: Vec<(Entity, Vec3)> = slots
            .iter()
            .enumerate()
            .map(|(i, &p)| (Entity::from_raw_u32(i as u32).unwrap(), p))
            .collect();
        let assignment = assign_slots_nearest(Vec3::new(3.0, 0.0, 0.0), &members, &slots);
        assert_eq!(assignment, vec![0, 1, 2, 3]);
    }

    #[test]
    fn nearest_solver_assigns_distinct_slots_on_collision() {
        // Two members piled on one slot: both must get distinct slots.
        let slots = vec![Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0)];
        let members: Vec<(Entity, Vec3)> = [
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(-0.1, 0.0, 0.0),
            Vec3::new(3.9, 0.0, 0.0),
        ]
        .iter()
        .enumerate()
        .map(|(i, &p)| (Entity::from_raw_u32(i as u32).unwrap(), p))
        .collect();
        let assignment = assign_slots_nearest(Vec3::new(2.0, 0.0, 0.0), &members, &slots);
        let mut seen = std::collections::HashSet::new();
        for &s in &assignment {
            assert!(seen.insert(s), "duplicate slot {s}");
        }
    }

    #[test]
    fn nearest_solver_scales_to_10k_members() {
        let n = 10_000usize;
        let cols = (n as f32).sqrt().ceil() as usize;
        let s = FormationKind::SPACING;
        let slots: Vec<Vec3> = (0..n)
            .map(|i| {
                Vec3::new(
                    (i % cols) as f32 * s,
                    0.0,
                    (i / cols) as f32 * s,
                )
            })
            .collect();
        // Members jittered around the slots (post-selection blob).
        let members: Vec<(Entity, Vec3)> = slots
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let j = |x: f32| x + ((i * 2654435761 % 97) as f32 / 97.0 - 0.5) * 2.0;
                (Entity::from_raw_u32(i as u32).unwrap(), Vec3::new(j(p.x), 0.0, j(p.z)))
            })
            .collect();
        let t0 = std::time::Instant::now();
        let assignment = assign_slots_nearest(Vec3::new(cols as f32 * s / 2.0, 0.0, cols as f32 * s / 2.0), &members, &slots);
        let t1 = std::time::Instant::now();
        let elapsed = t1 - t0;
        let mut seen = std::collections::HashSet::new();
        for &sl in &assignment {
            assert!(seen.insert(sl), "duplicate slot");
        }
        eprintln!("10k assignment took {elapsed:?}");
        // ~7ms in release for 10k; debug builds are ~15x slower.
        let budget_ms = if cfg!(debug_assertions) { 500 } else { 50 };
        assert!(
            elapsed.as_millis() < budget_ms,
            "too slow: {elapsed:?} (budget {budget_ms}ms)"
        );
    }
}
