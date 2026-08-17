use crate::kinematics::{NNTree, Velocity};
use crate::target::Target;
use bevy::prelude::*;
use bevy_spatial::SpatialAccess;
use std::collections::{HashSet, VecDeque};

/// A maneuver a formation executes before (or while) marching.
#[derive(Debug, Clone, Copy)]
pub enum FormationTask {
    /// Rotate the slot frame about the formation center until it aligns with
    /// `to`. Members hold their (rotating) slots; marching is suspended
    /// meanwhile. Used by asymmetric kinds that would otherwise make members
    /// cross paths on large turns.
    Rotate { to: Vec3 },
}

/// A formation groups boids (and possibly sub-formations) and assigns each
/// member a target position relative to the formation origin.
///
/// The origin is stored as the entity's [`Transform`]; the desired origin (used
/// when this formation is itself a member of a parent formation) in [`Target`].
#[derive(Component, Default)]
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
    /// Pending maneuvers, executed front-to-back; an empty queue means
    /// plain marching.
    pub tasks: VecDeque<FormationTask>,
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
#[derive(Component, Copy, Clone, Debug, PartialEq, Eq)]
pub struct FormationSlot(pub usize);

/// Marker: the formation's member-to-slot assignment must be recomputed
/// (orientation/kind change). Inserted by frontage designation and group
/// creation; removed by [`assign_slots`].
#[derive(Component)]
pub struct NeedsSlotAssignment;

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
#[derive(Component)]
#[relationship_target(relationship = FormationOf)]
pub struct Formations(Vec<Entity>);

/// Quick-command-group slot (RTS hotkey groups 1-6).
#[derive(Component)]
pub struct CommandSlot(pub u8);

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

/// Master simulation-detail switches for formations. Intended as the LOD hook:
/// when the camera is far away, boids or whole sub-formations may not be
/// fully simulated; flags here gate how far target propagation reaches.
#[derive(Resource, Debug)]
pub struct FormationSim {
    /// Propagate parent formation targets to *direct* members only.
    /// Nested levels converge on later frames (each level re-emits to its own
    /// members), so disabling this cheaply freezes distant detail.
    pub propagate_targets: bool,
}

impl Default for FormationSim {
    fn default() -> Self {
        Self {
            propagate_targets: true,
        }
    }
}

/// Maintain per-formation bookkeeping (extent). Member target propagation
/// lives in [`move_formations`], which drives it from the intermediate goal.
pub fn propagate_formation_targets(
    sim: Res<FormationSim>,
    mut q_formations: Query<(&mut Formation, &Members, Option<&Formations>), With<Formation>>,
) {
    if !sim.propagate_targets {
        return;
    }
    for (mut formation, members, subs) in &mut q_formations {
        let total = members.len() + subs.map_or(0, |s| s.len());
        formation.extent = formation.slot_extent(total);
    }
}

/// (Re)assign members to formation slots so that total movement to the new
/// slot positions is minimized. Runs when a formation is flagged
/// [`NeedsSlotAssignment`] (orientation/kind change, group creation) or when
/// the current assignment is invalid (member died or left, leaving a gap).
///
/// Assignment: greedy nearest-edge matching, then repeated pairwise swaps
/// while they reduce total distance. Cheap, and near-optimal for the
/// formations involved.
pub fn assign_slots(
    q_formations: Query<
        (
            Entity,
            &Transform,
            &Formation,
            &Members,
            Option<&Formations>,
            Option<&NeedsSlotAssignment>,
        ),
        With<Formation>,
    >,
    q_members: Query<Option<&FormationSlot>>,
    tree: Res<NNTree>,
    mut commands: Commands,
) {
    for (entity, transform, formation, members, subs, flagged) in &q_formations {
        let member_total = members.len();
        let total = member_total + subs.map_or(0, |s| s.len());
        if total == 0 {
            if flagged.is_some() {
                commands.entity(entity).remove::<NeedsSlotAssignment>();
            }
            continue;
        }

        // Validity check (cheap): every member has an in-range, unique slot.
        let mut valid = flagged.is_none();
        if valid {
            let mut seen = vec![false; total];
            for member in members.iter() {
                match q_members.get(member) {
                    Ok(Some(slot)) if (slot.0 as usize) < member_total
                        && !seen[slot.0 as usize] =>
                    {
                        seen[slot.0 as usize] = true;
                    }
                    _ => {
                        valid = false;
                        break;
                    }
                }
            }
        }
        if valid {
            continue;
        }

        // Slot offsets in the NEW frame (current origin, rotated to the newly
        // designated direction), iterated inside-out: inner slots claim the
        // members nearest the center first, the rest flow outward. Boid slots
        // are 0..members.len(); indices at/above that are reserved for
        // sub-formations (see move_formations stage 3).
        let member_total = members.len();
        let rotation = yaw_quat(formation.dir).unwrap_or(Quat::IDENTITY);
        let origin = transform.translation;
        let offsets: Vec<Vec3> = (0..member_total)
            .map(|i| formation.slot_offset(i, total))
            .collect();
        let mut slot_order: Vec<usize> = (0..member_total).collect();
        slot_order.sort_by(|&a, &b| {
            offsets[a]
                .length_squared()
                .total_cmp(&offsets[b].length_squared())
        });

        // Slot-driven greedy via the kd tree: for each slot (inside-out),
        // take the nearest *unassigned member of this formation*. The tree
        // tracks all boids, so the k-NN window grows until a member matches.
        let member_set: HashSet<Entity> = members.iter().collect();
        let mut assigned: HashSet<Entity> = HashSet::new();
        let mut total_shuffle = 0.0f32;
        for &slot in &slot_order {
            let slot_pos = origin + rotation * offsets[slot];
            let mut k = 4usize;
            loop {
                let found = tree
                    .k_nearest_neighbour(slot_pos, k)
                    .into_iter()
                    .filter_map(|(pos, e)| e.map(|e| (pos, e)))
                    .find(|(_, e)| member_set.contains(e) && !assigned.contains(e));
                match found {
                    Some((member_pos, member)) => {
                        assigned.insert(member);
                        total_shuffle += member_pos.distance(slot_pos);
                        commands.entity(member).insert(FormationSlot(slot));
                        break;
                    }
                    None => {
                        if k >= member_total.max(tree.tree.len()) {
                            break; // give up on this slot; retried next frame
                        }
                        k *= 2;
                    }
                }
            }
        }

        if assigned.len() == member_total {
            info!(
                "[slots] formation {entity:?}: {member_total} members, mean shuffle {:#.2}",
                total_shuffle / member_total as f32
            );
            commands.entity(entity).remove::<NeedsSlotAssignment>();
        } else {
            // Incomplete (stale tree entries, dead members): clear leftover
            // stale slots and let the validity check re-run next frame.
            for member in members.iter() {
                if !assigned.contains(&member) {
                    commands.entity(member).remove::<FormationSlot>();
                }
            }
        }
    }
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

/// Formation origin follows an *intermediate goal* so members keep formation
/// along the path instead of the center outrunning them:
/// 1. The origin snaps to the center of mass of the (boid) members every
///    frame - the center is defined by where its members actually are.
/// 2. The intermediate goal is offset from that center of mass toward the
///    real target by `slowest_member_speed * LEAD_TIME`, clamped to the
///    remaining distance.
/// 3. Member targets are propagated from the intermediate goal (and its
///    facing), replacing the raw-target propagation.
///
/// A maneuver system can later drive goals along paths instead.
pub const LEAD_TIME: f32 = 10.0;

pub fn move_formations(
    mut params: ParamSet<(
        Query<
            (
                &mut Transform,
                &Formation,
                &Target,
                &Members,
                Option<&Formations>,
            ),
            With<Formation>,
        >,
        Query<(&Transform, &Velocity, Option<&FormationSlot>)>,
        Query<&mut Target>,
    )>,
    mut gizmos: Gizmos,
) {
    struct Plan {
        center_of_mass: Vec3,
        goal: Vec3,
        target_pos: Vec3,
    }
    // Stage 1: snapshot what we need from p0 (small data; avoids overlapping
    // ParamSet borrows when reading member states via p1). Member entries
    // carry their slot id (filled in the next pass) so placement is by slot
    // identity, not list order.
    let mut snapshots: Vec<(Vec<(Entity, Option<usize>)>, Vec3, Vec3, f32)> = params
        .p0()
        .iter()
        .map(|(_, formation, target, members, _)| {
            (
                members.iter().map(|m| (m, None)).collect(),
                target.pos,
                formation.dir,
                formation.extent,
            )
        })
        .collect();
    for formation_snapshot in &mut snapshots {
        for (member, slot) in &mut formation_snapshot.0 {
            if let Ok((_, _, s)) = params.p1().get(*member) {
                *slot = s.map(|s| s.0);
            }
        }
    }
    // Aligned with p0 iteration order; None for formations with no members
    // (keeps stage 2/3 zips in sync).
    let mut plans: Vec<Option<Plan>> = Vec::with_capacity(snapshots.len());
    for (member_list, target_pos, _facing, _extent) in &snapshots {
        let mut com = Vec3::ZERO;
        let mut slowest = f32::INFINITY;
        let mut count = 0usize;
        for (member, _) in member_list {
            if let Ok((transform, velocity, _)) = params.p1().get(*member) {
                com += transform.translation;
                slowest = slowest.min(velocity.target_v);
                count += 1;
            }
        }
        if count == 0 {
            plans.push(None);
            continue;
        }
        let com = com / count as f32;

        // Intermediate goal: offset from the center of mass toward the target
        // by the distance the slowest member covers in LEAD_TIME seconds,
        // clamped to the remaining distance (so it lands exactly on the target).
        let to_target = *target_pos - com;
        let distance = to_target.length();
        let lead = (slowest * LEAD_TIME).min(distance);
        let goal = if distance > 1e-4 {
            com + to_target * (lead / distance)
        } else {
            *target_pos
        };

        plans.push(Some(Plan {
            center_of_mass: com,
            goal,
            target_pos: *target_pos,
        }));
    }

    // Stage 2: the formation origin *is* the center of mass of its members;
    // facing follows the plan.
    for ((mut transform, _, _, _, _), plan) in params.p0().iter_mut().zip(&plans) {
        let Some(plan) = plan else {
            continue;
        };
        transform.translation = plan.center_of_mass;
    }
    // Marker frame: rotate to the designated facing (read-only pass; stage 2
    // and 3 are separate because of the mixed &mut/& borrows).
    for (((mut transform, _, _, _, _), (_, _, facing, _)), plan) in
        params.p0().iter_mut().zip(&snapshots).zip(&plans)
    {
        let Some(plan) = plan else {
            continue;
        };
        let _ = plan;
        if let Some(desired) = yaw_quat(*facing) {
            transform.rotation = desired;
        }
    }

    // Stage 3: propagate member targets from the *intermediate goal*,
    // placing each member by slot identity (falls back to list order before
    // the first assignment). The slot frame is ALWAYS the designated facing
    // (never the travel direction): orientation changes are absorbed by
    // re-mapping slots (assign_slots), so a symmetric formation re-orienting
    // 90/180 degrees keeps every boid where it stands - different slot, same
    // position. Boids get Target directly; sub-formations get their origin
    // Target set and propagate to their own members next frame.
    let mut assignments: Vec<(Entity, Vec3, Vec3)> = Vec::new();
    for (((_, formation, _, _, subs), (member_slots, _, facing, _)), plan) in
        params.p0().iter().zip(&snapshots).zip(&plans)
    {
        let Some(plan) = plan else {
            continue;
        };
        let rotation = yaw_quat(*facing).unwrap_or(Quat::IDENTITY);
        let total = member_slots.len() + subs.map_or(0, |s| s.len());
        for ((member, slot), fallback) in member_slots.iter().zip(0..) {
            let slot = slot.unwrap_or(fallback);
            assignments.push((
                *member,
                plan.goal + rotation * formation.slot_offset(slot, total),
                *facing,
            ));
        }
        if let Some(subs) = subs {
            for (i, sub) in subs.iter().enumerate() {
                assignments.push((
                    sub,
                    plan.goal + rotation * formation.slot_offset(member_slots.len() + i, total),
                    *facing,
                ));
            }
        }
        // Debug: line from center of mass to the intermediate goal, and the
        // remaining leg from the goal to the final target.
        gizmos.line(plan.center_of_mass, plan.goal, Color::srgb(0.2, 0.6, 1.0));
        let lifted_target = plan.target_pos + Vec3::new(0.0, 0.2, 0.0);
        if plan.goal.distance(lifted_target) > 1e-3 {
            gizmos.line(plan.goal, lifted_target, Color::srgb(0.5, 0.5, 0.5));
        }
    }
    for (entity, pos, dir) in assignments {
        if let Ok(mut target) = params.p2().get_mut(entity) {
            target.pos = pos;
            target.dir = dir;
        }
    }
}

