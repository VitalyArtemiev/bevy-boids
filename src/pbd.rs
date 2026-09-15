//! Position-based collision response for boids — Weiss, Litteneker, Jiang,
//! Terzopoulos, *Position-Based Multi-Agent Dynamics for Real-Time Crowd
//! Simulation* (MiG 2017, arXiv:1802.02673). Implements stages 2 and 3 of
//! `docs/pbd-anticipation.md`.
//!
//! `pbd_contact` runs after `move_step` and projects boids out of overlap
//! (frictional contact, §4.2), nudges friendly pairs whose predicted paths
//! collide within the anticipation horizon (long-range constraint, §4.4)
//! with the avoidance flavour of §4.5 (sidestep rather than brake), and
//! seats boids against obstacle cuboids (§4.7). Everything is positional:
//! velocity picks the corrections up as `Δv = Δx/Δt` at write-back, so the
//! steering contract stays intact — planners write `vel.a`/`vel.target_v`,
//! the solver owns collision response, and neither fights the other.
//!
//! Hostile factions are the project's clash mode: enemy pairs keep the
//! contact constraint — bodies slam and shove by mass, the paper's
//! bears-vs-rabbits mechanics — but skip anticipation, and a [`Charging`]
//! unit skips anticipation against everyone (frontal assault). Per-pair
//! constraint skipping is *our* extension of the paper (it is a
//! single-crowd sim with no factions); a smaller constraint set cannot
//! destabilise PBD, and the melee tests below validate it.

use crate::boid::Boid;
use crate::kinematics::{HardCollision, KinematicsTuning, NNTree, Velocity};
use crate::terrain::Obstacle;
use bevy::prelude::*;
use bevy::tasks::ComputeTaskPool;
use bevy_spatial::SpatialAccess;
use std::collections::HashMap;

/// Which side a boid fights on. Boids on different factions are hostile to
/// each other (symmetric — both sides of a pair skip avoidance, or one army
/// would politely step aside while the other plows through). A relation
/// matrix replaces the id comparison when alliances arrive.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Faction(pub u8);

/// A unit in frontal assault charge: anticipation is dropped against
/// *everyone* (it plows through friends too — the "unless it is a frontal
/// assault charge" clause of the project brief). Contact always stays;
/// bodies still collide.
#[derive(Component, Default)]
pub struct Charging;

/// The physical body of a unit for the solver: a disk in the ground plane
/// of `radius_m`, weighing `mass_kg`. Mass only ever enters as inverse-mass
/// weighting, so heavier units are shoved less — cavalry vs infantry.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Body {
    pub radius_m: f32,
    pub mass_kg: f32,
}

impl Default for Body {
    fn default() -> Self {
        // Contact radius sits 10% under the visual capsule (0.5): moving
        // blocks sliding through each other shear past instead of snagging
        // on the rear ranks, at the cost of slight visual overlap in a push.
        Body {
            radius_m: 0.45,
            mass_kg: 1.0,
        }
    }
}

pub const SOLVER_ITERATIONS: usize = 4;
/// Candidate neighbours gathered per boid (`k_nearest`). Must stay within
/// [`NEIGHBOR_SLOTS`].
pub const NEIGHBOR_COUNT: usize = 6;
/// Share of the tangential relative slip a contact removes per step,
/// clamped by the friction cone `μ × overlap`.
pub const FRICTION: f32 = 0.4;
/// Hostile contacts are slicker — clashing bodies slide off each other
/// rather than grinding to a standstill.
pub const HOSTILE_FRICTION_SCALE: f32 = 0.25;
/// How far ahead (seconds) the long-range constraint looks for friendly
/// pairs. The paper's 20 s horizon is tuned for 1.4 m/s pedestrians; our
/// 20 m/s units meet their futures ~10× sooner.
pub const ANTICIPATION_HORIZON_SEC: f32 = 1.5;
/// Per-step strength of the long-range correction. The paper retunes its
/// pedestrian value for Δt and its solver, and so must we — this default
/// is calibrated against the `--scene` crossings.
pub const ANTICIPATION_STIFFNESS: f32 = 0.35;
/// How much of the long-range correction's braking component (the part
/// anti-parallel to the boid's own velocity) survives: 0 = pure §4.5
/// sidestep (agents never slow for each other), 1 = plain anticipation.
pub const BRAKING_KEEP: f32 = 0.3;
/// Extra metres on the contact-distance test when filtering neighbour
/// candidates, buying back some of the kd-tree's staleness (tree positions
/// can be a refresh period old).
pub const CANDIDATE_MARGIN_M: f32 = 2.0;
/// Extra metres on the obstacle proximity query.
pub const OBSTACLE_MARGIN_M: f32 = 1.0;
/// Fixed per-boid neighbour slots — the scratch arrays are flat, not CSR:
/// one gather pass, no two-phase offset build, and 16 slots comfortably
/// cover the ~6–8 simultaneous contacts a disk can even have.
pub const NEIGHBOR_SLOTS: usize = 16;
/// Solver work is chunk-parallel over these many boids per task.
const PAR_CHUNK: usize = 256;

/// Runtime-tunable solver parameters, exposed by the debug UI.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct PbdTuning {
    pub iterations: usize,
    pub neighbors: usize,
    pub friction: f32,
    pub hostile_friction_scale: f32,
    pub anticipation_horizon_sec: f32,
    pub anticipation_stiffness: f32,
    pub braking_keep: f32,
    pub candidate_margin_m: f32,
    pub obstacle_margin_m: f32,
}

impl Default for PbdTuning {
    fn default() -> Self {
        Self {
            iterations: SOLVER_ITERATIONS,
            neighbors: NEIGHBOR_COUNT,
            friction: FRICTION,
            hostile_friction_scale: HOSTILE_FRICTION_SCALE,
            anticipation_horizon_sec: ANTICIPATION_HORIZON_SEC,
            anticipation_stiffness: ANTICIPATION_STIFFNESS,
            braking_keep: BRAKING_KEEP,
            candidate_margin_m: CANDIDATE_MARGIN_M,
            obstacle_margin_m: OBSTACLE_MARGIN_M,
        }
    }
}

/// Which constraints apply to a boid pair, by hostility and charge state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairPolicy {
    /// Long-range anticipation (never hostile; dropped while either side
    /// charges).
    pub anticipate: bool,
    /// Contact friction for the pair. Contact itself is always on: bodies
    /// collide, friends and foes alike — hostility only changes how much
    /// they grind.
    pub friction: f32,
}

/// The pair rule set. Symmetric by construction (both sides compute the
/// same hostility answer), which the solver depends on: a one-sided skip
/// would make one boid of a pair do all the avoiding.
pub fn pair_policy(
    my: &Faction,
    my_charging: bool,
    their: &Faction,
    their_charging: bool,
    tuning: &PbdTuning,
) -> PairPolicy {
    let hostile = my.0 != their.0;
    PairPolicy {
        anticipate: !hostile && !my_charging && !their_charging,
        friction: if hostile {
            tuning.friction * tuning.hostile_friction_scale
        } else {
            tuning.friction
        },
    }
}

/// One-sided positional contact correction for `i` against `j` (paper
/// §4.2, frictionless part): push `i` out of overlap along the contact
/// normal, taking the inverse-mass share `inv_i / (inv_i + inv_j)` so the
/// pair splits the correction and heavier bodies barely move. Zero when
/// the pair does not overlap.
pub fn contact_correction(
    pos_i: Vec3,
    pos_j: Vec3,
    radius_i: f32,
    radius_j: f32,
    inv_i: f32,
    inv_j: f32,
) -> Vec3 {
    let d = pos_i - pos_j;
    let dist = d.length();
    let min_dist = radius_i + radius_j;
    if dist >= min_dist || dist < 1e-6 || inv_i + inv_j <= 0.0 {
        return Vec3::ZERO;
    }
    d * ((min_dist - dist) / dist * (inv_i / (inv_i + inv_j)))
}

/// One-sided friction correction for a contact (Macklin-style kinematic
/// friction): oppose the tangential part of this step's *relative*
/// displacement (`step_rel`), clamped by the friction cone `μ × overlap`.
/// PBD friction works on positions, so it needs no velocity reads and
/// races nothing.
pub fn friction_correction(
    step_rel: Vec3,
    normal: Vec3,
    overlap: f32,
    mu: f32,
    share: f32,
) -> Vec3 {
    if overlap <= 0.0 || mu <= 0.0 {
        return Vec3::ZERO;
    }
    let tangential = step_rel - normal * step_rel.dot(normal);
    // Oppose the relative slip, capped by this side's share of the cone.
    (-tangential * mu).clamp_length_max(mu * overlap * share)
}

/// Time to first collision for a pair on current course (paper §4.4):
/// solves `|p_rel + v_rel·t| = min_dist` and returns the smaller root when
/// it lies inside `(0, horizon)`. Separating pairs, non-colliding courses,
/// parallel courses and far futures return `None` — the constraint simply
/// does not apply.
pub fn time_to_collision(
    pos_i: Vec3,
    vel_i: Vec3,
    pos_j: Vec3,
    vel_j: Vec3,
    min_dist: f32,
    horizon: f32,
) -> Option<f32> {
    let v = vel_i - vel_j;
    let a = v.dot(v);
    if a < 1e-8 {
        return None;
    }
    let p = pos_i - pos_j;
    let b = 2.0 * v.dot(p);
    let c = p.dot(p) - min_dist * min_dist;
    if c <= 0.0 {
        return None; // already overlapping: contact's job
    }
    let disc = b * b - 4.0 * a * c;
    if disc <= 0.0 {
        return None;
    }
    let tau = (-b - disc.sqrt()) / (2.0 * a);
    (tau > 0.0 && tau < horizon).then_some(tau)
}

/// Precomputed long-range correction for one side of a pair (paper §4.4
/// with the §4.5 avoidance flavour). Evaluates the pair at `τ̂ + Δt` —
/// one step past the whole steps before contact, i.e. where they *would*
/// collide — and scales the push by the adaptive stiffness
/// `k·exp(−τ̂²/τ0)`, so imminent collisions are firm and distant ones
/// fade. The §4.5 part: the component anti-parallel to the boid's own
/// velocity (the braking part) is shed, leaving a sidestep — agents keep
/// their speed and slide past each other instead of halting.
pub fn anticipation_correction(
    pos_i: Vec3,
    vel_i: Vec3,
    pos_j: Vec3,
    vel_j: Vec3,
    min_dist: f32,
    dt: f32,
    tuning: &PbdTuning,
) -> Vec3 {
    let Some(tau) = time_to_collision(
        pos_i,
        vel_i,
        pos_j,
        vel_j,
        min_dist,
        tuning.anticipation_horizon_sec,
    ) else {
        return Vec3::ZERO;
    };
    let tau_hat = (dt * (tau / dt).floor()).max(0.0);
    let lead = tau_hat + dt;
    let future_i = pos_i + vel_i * lead;
    let future_j = pos_j + vel_j * lead;
    let d = future_i - future_j;
    let dist = d.length();
    if dist >= min_dist || dist < 1e-6 {
        return Vec3::ZERO;
    }
    let horizon = tuning.anticipation_horizon_sec;
    let stiffness =
        tuning.anticipation_stiffness * (-(tau_hat * tau_hat) / (horizon * horizon)).exp();
    let mut correction = d * ((min_dist - dist) / dist * stiffness);
    let heading = vel_i.normalize_or_zero();
    let brake = correction.dot(heading).min(0.0);
    correction -= heading * (brake * (1.0 - tuning.braking_keep));
    correction
}

/// One neighbour slot, filled once per step during candidate gathering.
#[derive(Clone, Copy, Default)]
struct Neighbor {
    idx: u32,
    /// Precomputed long-range correction for this side (zero when the pair
    /// does not anticipate).
    antic: Vec3,
    /// Contact friction coefficient for the pair.
    friction: f32,
}

/// Reused solver scratch: a flat SoA over the boid set, rebuilt (not
/// reallocated) every step.
#[derive(Resource, Default)]
pub struct PbdScratch {
    entity_to_idx: HashMap<Entity, usize>,
    /// Positions at step start (post-`move_step`): the friction slip
    /// reference and the write-back `Δv = Δx/Δt` base.
    pos_start: Vec<Vec3>,
    vel: Vec<Vec3>,
    radius: Vec<f32>,
    inv_mass: Vec<f32>,
    charging: Vec<bool>,
    factions: Vec<Faction>,
    /// Double-buffered working positions (Jacobi: every boid reads one
    /// buffer, writes the other; after the final swap the newest positions
    /// are back in `pos_a`).
    pos_a: Vec<Vec3>,
    pos_b: Vec<Vec3>,
    nbr_counts: Vec<u32>,
    /// `n × NEIGHBOR_SLOTS` flat neighbour slots.
    nbrs: Vec<Neighbor>,
}

/// The positional collision stage: snapshot boids → gather pair candidates
/// once (the kd-tree query is the expensive part; iterations only reread
/// the scratch) → Jacobi iterations over contact + friction +
/// anticipation → seat against obstacle cuboids → write positions back
/// with `Δv = Δx/Δt`, clamped to the hard speed ceiling.
pub fn pbd_contact(
    mut q_boids: Query<
        (
            Entity,
            &mut Transform,
            &mut Velocity,
            &Faction,
            Option<&Charging>,
            &Body,
        ),
        With<Boid>,
    >,
    q_obstacles: Query<(&Obstacle, &Transform), (With<HardCollision>, Without<Boid>)>,
    tree: Res<NNTree>,
    time: Res<Time>,
    kin: Res<KinematicsTuning>,
    tuning: Res<PbdTuning>,
    mut scratch: ResMut<PbdScratch>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }

    // -- Snapshot (serial: builds the entity->index map) ------------------
    let s = &mut *scratch;
    s.entity_to_idx.clear();
    s.pos_start.clear();
    s.vel.clear();
    s.radius.clear();
    s.inv_mass.clear();
    s.charging.clear();
    s.factions.clear();
    // Capacity persists across frames, so the pushes don't allocate in
    // steady state.
    for (entity, transform, vel, faction, charging, body) in &mut q_boids {
        s.entity_to_idx.insert(entity, s.pos_start.len());
        s.pos_start.push(transform.translation);
        s.vel.push(vel.v);
        s.radius.push(body.radius_m);
        s.inv_mass.push(1.0 / body.mass_kg.max(1e-6));
        s.charging.push(charging.is_some());
        s.factions.push(*faction);
    }
    let n = s.pos_start.len();
    s.pos_a.clear();
    s.pos_a.extend_from_slice(&s.pos_start);
    s.pos_b.clear();
    s.pos_b.resize(n, Vec3::ZERO);
    s.nbr_counts.clear();
    s.nbr_counts.resize(n, 0);
    s.nbrs.clear();
    s.nbrs.resize(n * NEIGHBOR_SLOTS, Neighbor::default());

    // -- Gather pair candidates (parallel, once per step) -----------------
    {
        let PbdScratch {
            pos_start,
            vel,
            radius,
            inv_mass: _,
            charging,
            factions,
            pos_a: _,
            pos_b: _,
            nbr_counts,
            nbrs,
            entity_to_idx,
        } = &mut *s;
        let k = tuning.neighbors.min(NEIGHBOR_SLOTS);
        let margin = tuning.candidate_margin_m;
        let tree = &*tree;
        let tuning = &*tuning;
        // Shared reborrowed views; only the per-chunk count/slot slices
        // stay mutable.
        let starts: &[Vec3] = pos_start;
        let vels: &[Vec3] = vel;
        let radii: &[f32] = radius;
        let charges: &[bool] = charging;
        let sides: &[Faction] = factions;
        let index_of: &HashMap<Entity, usize> = entity_to_idx;
        ComputeTaskPool::get().scope(|scope| {
            for (chunk_i, (counts, slots)) in nbr_counts
                .as_mut_slice()
                .chunks_mut(PAR_CHUNK)
                .zip(nbrs.as_mut_slice().chunks_mut(PAR_CHUNK * NEIGHBOR_SLOTS))
                .enumerate()
            {
                scope.spawn(async move {
                    for local in 0..counts.len() {
                        let i = chunk_i * PAR_CHUNK + local;
                        let mut count = 0usize;
                        let pos_i = starts[i];
                        let vel_i = vels[i];
                        for (_other_pos, other_entity) in tree.k_nearest_neighbour(pos_i, k + 1) {
                            if count >= k {
                                break;
                            }
                            let Some(&j) =
                                other_entity.and_then(|e| index_of.get(&e))
                            else {
                                continue;
                            };
                            if j == i {
                                continue;
                            }
                            let pos_j = starts[j];
                            let contact_range = radii[i] + radii[j] + margin;
                            let policy = pair_policy(
                                &sides[i],
                                charges[i],
                                &sides[j],
                                charges[j],
                                tuning,
                            );
                            let antic = if policy.anticipate {
                                anticipation_correction(
                                    pos_i,
                                    vel_i,
                                    pos_j,
                                    vels[j],
                                    radii[i] + radii[j],
                                    dt,
                                    tuning,
                                )
                            } else {
                                Vec3::ZERO
                            };
                            // A slot is worth keeping for contact proximity
                            // or a live anticipation push.
                            if antic == Vec3::ZERO && pos_i.distance(pos_j) >= contact_range {
                                continue;
                            }
                            slots[local * NEIGHBOR_SLOTS + count] = Neighbor {
                                idx: j as u32,
                                antic,
                                friction: policy.friction,
                            };
                            count += 1;
                        }
                        counts[local] = count as u32;
                    }
                });
            }
        });
    }

    // -- Jacobi iterations (parallel over chunks of the write buffer) -----
    let iterations = tuning.iterations.max(1);
    for _ in 0..iterations {
        {
            let PbdScratch {
                pos_a,
                pos_b,
                pos_start,
                inv_mass,
                radius,
                nbr_counts,
                nbrs,
                ..
            } = &mut *s;
            // Anticipation corrections are velocity-derived (constant
            // across iterations), so each iteration applies a
            // 1/iterations share of the precomputed vector; contact and
            // friction re-evaluate on fresh positions and take their full
            // share.
            let antic_share = 1.0 / iterations as f32;
            // Shared reborrowed views (implicit &mut -> & reborrow, so the
            // destructured &mut fields are not moved).
            let read: &[Vec3] = pos_a;
            let starts: &[Vec3] = pos_start;
            let masses: &[f32] = inv_mass;
            let radii: &[f32] = radius;
            let counts: &[u32] = nbr_counts;
            let nbr_list: &[Neighbor] = nbrs;
            ComputeTaskPool::get().scope(|scope| {
                for (chunk_i, out) in pos_b.as_mut_slice().chunks_mut(PAR_CHUNK).enumerate() {
                    scope.spawn(async move {
                        for local in 0..out.len() {
                            let i = chunk_i * PAR_CHUNK + local;
                            let inv_i = masses[i];
                            let mut corr = Vec3::ZERO;
                            for slot in 0..counts[i] as usize {
                                let nbr = nbr_list[i * NEIGHBOR_SLOTS + slot];
                                let j = nbr.idx as usize;
                                let inv_j = masses[j];
                                corr += contact_correction(
                                    read[i], read[j], radii[i], radii[j], inv_i, inv_j,
                                );
                                // Friction rides the contact: this step's
                                // relative slip, cone-clamped.
                                let d = read[i] - read[j];
                                let dist = d.length();
                                let min_dist = radii[i] + radii[j];
                                if dist < min_dist && dist > 1e-6 {
                                    let share = inv_i / (inv_i + inv_j);
                                    let step_rel =
                                        (read[i] - starts[i]) - (read[j] - starts[j]);
                                    corr += friction_correction(
                                        step_rel,
                                        d / dist,
                                        min_dist - dist,
                                        nbr.friction,
                                        share,
                                    );
                                }
                                corr += nbr.antic * (inv_i / (inv_i + inv_j)) * antic_share;
                            }
                            out[local] = read[i] + corr;
                        }
                    });
                }
            });
        }
        std::mem::swap(&mut s.pos_a, &mut s.pos_b);
    }

    // -- Obstacle seating (§4.7: static obstacles have infinite mass) -----
    // Few obstacles with few contacts each: a serial obstacle-centric
    // loop, projecting disks out of the cuboid's XZ square.
    for (_obstacle, transform) in &q_obstacles {
        let center = transform.translation;
        let reach = 0.5 + s.radius.first().copied().unwrap_or(0.5) + tuning.obstacle_margin_m;
        for (_pos, hit) in tree.within_distance(center, reach) {
            let Some(&i) = hit.and_then(|e| s.entity_to_idx.get(&e)) else {
                continue;
            };
            let p = s.pos_a[i];
            let closest = Vec3::new(
                p.x.clamp(center.x - 0.5, center.x + 0.5),
                p.y,
                p.z.clamp(center.z - 0.5, center.z + 0.5),
            );
            let d = p - closest;
            let dist = d.xz().length();
            let r = s.radius[i];
            if dist < r && dist > 1e-6 {
                let push = d.xz() / dist * (r - dist);
                s.pos_a[i] += Vec3::new(push.x, 0.0, push.y);
            } else if dist <= 1e-6 {
                // Dead centre on the square: any push works; the next step
                // has gradient again.
                s.pos_a[i].x += r;
            }
        }
    }

    // -- Write-back: positions + Δv = Δx/Δt -------------------------------
    let max_v = kin.max_velocity_mps;
    q_boids.par_iter_mut().for_each(|(entity, mut transform, mut vel, _, _, _)| {
        let Some(&i) = s.entity_to_idx.get(&entity) else {
            return;
        };
        let delta = s.pos_a[i] - s.pos_start[i];
        if delta == Vec3::ZERO {
            return;
        }
        transform.translation += delta;
        vel.v = (vel.v + delta / dt).clamp_length_max(max_v);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boid::{BoidBundle, BoidVariations};
    use crate::target::{Target, follow_target};
    use crate::kinematics::move_step;
    use bevy::app::TaskPoolPlugin;
    use bevy::time::{TimePlugin, TimeUpdateStrategy};
    use bevy_spatial::{AutomaticUpdate, TransformMode};
    use std::time::{Duration, Instant};

    #[test]
    fn contact_correction_splits_by_inverse_mass() {
        // Equal masses: each side takes half the overlap.
        let c = contact_correction(
            Vec3::new(0.8, 0.0, 0.0),
            Vec3::ZERO,
            0.5,
            0.5,
            1.0,
            1.0,
        );
        assert!((c.x - 0.1).abs() < 1e-5, "{c:?}");

        // A 10x heavier `i` barely moves: share 0.1/1.1.
        let c = contact_correction(
            Vec3::new(0.8, 0.0, 0.0),
            Vec3::ZERO,
            0.5,
            0.5,
            0.1,
            1.0,
        );
        assert!((c.x - 0.2 * (0.1 / 1.1)).abs() < 1e-5, "{c:?}");

        // No overlap, no correction; identical centres are ambiguous and
        // left to other neighbours.
        assert_eq!(
            contact_correction(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, 0.5, 0.5, 1.0, 1.0),
            Vec3::ZERO
        );
        assert_eq!(
            contact_correction(Vec3::ZERO, Vec3::ZERO, 0.5, 0.5, 1.0, 1.0),
            Vec3::ZERO
        );
    }

    #[test]
    fn friction_correction_opposes_slip_within_the_cone() {
        // Pure tangential slip along +x on a contact normal along z.
        let step_rel = Vec3::new(0.2, 0.0, 0.0);
        let normal = Vec3::Z;
        let c = friction_correction(step_rel, normal, 0.1, 0.4, 0.5);
        assert!(c.x < 0.0, "must oppose the slip, got {c:?}");
        // |slip|*mu = 0.08 exceeds the cone share 0.4*0.1*0.5 = 0.02.
        assert!((c.length() - 0.02).abs() < 1e-5, "{c:?}");
        // Below the cone: full opposing share.
        let c = friction_correction(Vec3::new(0.01, 0.0, 0.0), normal, 0.5, 0.4, 0.5);
        assert!((c.length() - 0.004).abs() < 1e-5, "{c:?}");
        // No overlap or no friction: nothing.
        assert_eq!(
            friction_correction(step_rel, normal, 0.0, 0.4, 0.5),
            Vec3::ZERO
        );
        assert_eq!(
            friction_correction(step_rel, normal, 0.1, 0.0, 0.5),
            Vec3::ZERO
        );
    }

    #[test]
    fn time_to_collision_finds_first_contact() {
        // Head-on, 10 m apart, closing at 4 m/s, meeting disks at 1 m:
        // first contact at (10-1)/4.
        let tau = time_to_collision(
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(-2.0, 0.0, 0.0),
            1.0,
            5.0,
        );
        assert!((tau.unwrap() - 2.25).abs() < 1e-4);

        // Beyond the horizon: no constraint.
        assert!(time_to_collision(
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(-2.0, 0.0, 0.0),
            1.0,
            1.0
        )
        .is_none());
        // Separating, parallel, and non-colliding courses never trigger.
        assert!(time_to_collision(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            1.0,
            5.0
        )
        .is_none());
        assert!(time_to_collision(
            Vec3::ZERO,
            Vec3::X,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::X,
            1.0,
            5.0
        )
        .is_none());
        assert!(time_to_collision(
            Vec3::ZERO,
            Vec3::Y,
            Vec3::new(10.0, 0.0, 0.0),
            -Vec3::Y,
            1.0,
            5.0
        )
        .is_none());
    }

    #[test]
    fn pair_policy_hostility_and_charge() {
        let tuning = PbdTuning::default();
        let red = Faction(0);
        let blue = Faction(1);
        // Friends anticipate with full friction.
        let p = pair_policy(&red, false, &red, false, &tuning);
        assert!(p.anticipate);
        assert_eq!(p.friction, tuning.friction);
        // Enemies clash: no anticipation, slicker contact.
        let p = pair_policy(&red, false, &blue, false, &tuning);
        assert!(!p.anticipate);
        assert_eq!(p.friction, tuning.friction * tuning.hostile_friction_scale);
        // A charge drops anticipation even between friends.
        assert!(!pair_policy(&red, true, &red, false, &tuning).anticipate);
        assert!(!pair_policy(&red, false, &red, true, &tuning).anticipate);
    }

    #[test]
    fn anticipation_correction_sidesteps_instead_of_braking() {
        // braking_keep 0 isolates the §4.5 shed: nothing of the anti-parallel
        // component may survive.
        let tuning = PbdTuning {
            braking_keep: 0.0,
            ..default()
        };
        // Crossing courses at right angles: the push is essentially
        // lateral for the +x traveller, so it survives the braking shed.
        let c = anticipation_correction(
            Vec3::new(-10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -10.0),
            Vec3::new(0.0, 0.0, 10.0),
            1.0,
            1.0 / 60.0,
            &tuning,
        );
        assert!(c.length() > 1e-4, "crossing pair must sidestep, got {c:?}");
        assert!(
            c.dot(Vec3::X) > -1e-4,
            "no braking for the +x traveller: {c:?}"
        );

        // Dead head-on: the entire push is braking and §4.5 sheds it —
        // contact handles the rest (symmetric charges cannot sidestep
        // anyway).
        let c = anticipation_correction(
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(-5.0, 0.0, 0.0),
            1.0,
            1.0 / 60.0,
            &tuning,
        );
        assert!(c.length() < 1e-4, "head-on braking must be shed, got {c:?}");
    }

    // ---- Integration harness -------------------------------------------

    /// Headless solver app: manual time, real kd-tree with fast refresh,
    /// and the movement chain (plan -> integrate -> resolve) in Update.
    fn solver_app() -> App {
        let mut app = App::new();
        app.add_plugins(TaskPoolPlugin::default())
            .add_plugins(TimePlugin)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<Image>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<BoidVariations>()
            .init_resource::<crate::kinematics::KinematicsTuning>()
            .init_resource::<crate::terrain::HeightField>()
            .init_resource::<PbdTuning>()
            .init_resource::<PbdScratch>()
            .add_plugins(
                AutomaticUpdate::<crate::kinematics::TrackedByTree>::new()
                    .with_frequency(Duration::from_secs_f32(1.0 / 20.0))
                    .with_transform(TransformMode::Transform),
            )
            .add_systems(Update, (follow_target, move_step, pbd_contact).chain());
        app.update(); // burn the first update (no clock advance)
        app
    }

    fn tick(app: &mut App, dt: f32) {
        app.insert_resource(TimeUpdateStrategy::ManualDuration(
            Duration::from_secs_f32(dt),
        ));
        app.update();
    }

    fn spawn_boid(
        app: &mut App,
        id: u32,
        pos: Vec3,
        vel: Vec3,
        target: Vec3,
        faction: u8,
        body: Body,
    ) {
        let variations = app.world_mut().resource::<BoidVariations>().clone();
        app.world_mut()
            .spawn(BoidBundle::with_id(
                id,
                Target {
                    pos: target,
                    ..default()
                },
                &variations,
            ))
            .insert(Transform::from_translation(pos))
            .insert({
                // Velocity's slope-cache field is module-private, so build
                // through Default instead of struct-update syntax.
                let mut velocity = Velocity::default();
                velocity.v = vel;
                velocity
            })
            .insert(Faction(faction))
            .insert(body);
    }

    fn positions(app: &mut App) -> Vec<(Entity, Vec3)> {
        let mut query = app
            .world_mut()
            .query_filtered::<(Entity, &Transform), With<Boid>>();
        query
            .iter(app.world())
            .map(|(e, t)| (e, t.translation))
            .collect()
    }

    fn min_pair_distance(app: &mut App) -> f32 {
        let positions = positions(app);
        let mut min = f32::INFINITY;
        for a in 0..positions.len() {
            for b in (a + 1)..positions.len() {
                min = min.min(positions[a].1.distance(positions[b].1));
            }
        }
        min
    }

    /// Friendly oblique approach: anticipation must part them without any
    /// interpenetration beyond solver slop, and both must complete the
    /// pass (the whole point over braking to a halt).
    #[test]
    fn friendly_oblique_pair_passes_cleanly() {
        let mut app = solver_app();
        spawn_boid(
            &mut app,
            0,
            Vec3::new(-30.0, 0.5, 0.0),
            Vec3::ZERO,
            Vec3::new(30.0, 0.5, 0.0),
            0,
            Body::default(),
        );
        spawn_boid(
            &mut app,
            1,
            Vec3::new(30.0, 0.5, 1.2),
            Vec3::ZERO,
            Vec3::new(-30.0, 0.5, 1.2),
            0,
            Body::default(),
        );

        let mut min_dist = f32::INFINITY;
        for _ in 0..900 {
            tick(&mut app, 1.0 / 60.0);
            min_dist = min_dist.min(min_pair_distance(&mut app));
        }
        let positions = positions(&mut app);
        assert!(
            min_dist > 0.8,
            "friends must not interpenetrate, min {min_dist}"
        );
        // Both travellers completed the pass.
        assert!(positions[0].1.x > 20.0, "{:?}",
            positions.iter().map(|p| p.1).collect::<Vec<_>>()
        );
        assert!(positions[1].1.x < -20.0);
    }

    /// The clash: enemies do NOT avoid — they run into contact and stay
    /// engaged (nobody runs through anybody).
    #[test]
    fn hostile_pair_slams_into_contact_and_holds() {
        let mut app = solver_app();
        // True head-on (same lane): at the crossing instant the contact
        // normal is purely along the closing axis, so the slam kills the
        // closing velocity instead of sheering the pair past each other —
        // the frontal-collision case, as opposed to the oblique glance the
        // friendly test exercises.
        spawn_boid(
            &mut app,
            0,
            Vec3::new(-25.0, 0.5, 0.0),
            Vec3::ZERO,
            Vec3::new(25.0, 0.5, 0.0),
            0,
            Body::default(),
        );
        spawn_boid(
            &mut app,
            1,
            Vec3::new(25.0, 0.5, 0.0),
            Vec3::ZERO,
            Vec3::new(-25.0, 0.5, 0.0),
            1,
            Body::default(),
        );

        let (mut touched, mut final_gap) = (false, f32::INFINITY);
        for _ in 0..900 {
            tick(&mut app, 1.0 / 60.0);
            let dist = min_pair_distance(&mut app);
            touched |= dist <= 1.0 + 0.25;
            final_gap = dist;
        }
        assert!(touched, "enemies must run into contact");
        // Still engaged at the end: the contact constraint holds them at
        // disk distance, grinding, not passing.
        assert!(
            final_gap < 2.5,
            "clash must hold contact, gap {final_gap}"
        );
    }

    /// Mass decides the clash (the paper's bears vs rabbits): a 10x heavier
    /// unit pushes the contact point toward the lighter one's origin.
    #[test]
    fn heavier_unit_shoves_the_lighter() {
        let mut app = solver_app();
        spawn_boid(
            &mut app,
            0,
            Vec3::new(-20.0, 0.5, 0.0),
            Vec3::ZERO,
            Vec3::new(20.0, 0.5, 0.0),
            0,
            Body {
                radius_m: 0.5,
                mass_kg: 10.0,
            },
        );
        spawn_boid(
            &mut app,
            1,
            Vec3::new(20.0, 0.5, 0.0),
            Vec3::ZERO,
            Vec3::new(-20.0, 0.5, 0.0),
            1,
            Body::default(),
        );

        for _ in 0..900 {
            tick(&mut app, 1.0 / 60.0);
        }
        let positions = positions(&mut app);
        let mid = (positions[0].1 + positions[1].1) * 0.5;
        assert!(
            mid.x > 1.0,
            "the pair's centre must drift toward the light unit's side, {mid:?}"
        );
    }

    /// The melee gate: three hostile groups converging — the extension the
    /// paper never tested. Penetration must stay bounded throughout.
    #[test]
    fn three_faction_melee_penetration_stays_bounded() {
        let mut app = solver_app();
        let mut id = 0;
        for (faction, angle) in (0u8..3).zip([0.0f32, 2.1, 4.2]) {
            for row in 0..2 {
                for col in 0..3 {
                    let dir = Vec3::new(angle.cos(), 0.0, angle.sin());
                    let base = dir * 18.0;
                    let side = Vec3::new(-dir.z, 0.0, dir.x);
                    let pos = base + side * (col as f32 - 1.0) * 2.0 + dir * row as f32 * 2.0;
                    spawn_boid(&mut app, id, pos + Vec3::Y * 0.5, Vec3::ZERO, Vec3::Y * 0.5, faction, Body::default());
                    id += 1;
                }
            }
        }

        let mut worst = f32::INFINITY;
        for _ in 0..1200 {
            tick(&mut app, 1.0 / 60.0);
            worst = worst.min(min_pair_distance(&mut app));
        }
        assert!(
            worst > 0.5,
            "melee penetration must stay bounded, worst {worst}"
        );
    }

    /// Wall-clock budget for the whole solver step at 10k boids (gather +
    /// iterations + write-back), in the style of
    /// `nearest_solver_scales_to_10k_members`. Measured 1.3 ms in release
    /// on the dev box; the budget keeps ~8x headroom for CI noise and
    /// slower machines.
    #[test]
    fn pbd_contact_scales_to_10k_boids() {
        let mut app = App::new();
        app.add_plugins(TaskPoolPlugin::default())
            .add_plugins(TimePlugin)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<Image>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<BoidVariations>()
            .init_resource::<crate::kinematics::KinematicsTuning>()
            .init_resource::<crate::terrain::HeightField>()
            .init_resource::<PbdTuning>()
            .init_resource::<PbdScratch>()
            .add_plugins(
                AutomaticUpdate::<crate::kinematics::TrackedByTree>::new()
                    // Long period: the timed update must not pay for a
                    // tree rebuild, only for the solver step.
                    .with_frequency(Duration::from_secs_f32(60.0))
                    .with_transform(TransformMode::Transform),
            )
            .add_systems(Update, pbd_contact);
        app.update(); // builds the tree; also burns the zero-delta update

        let variations = app.world_mut().resource::<BoidVariations>().clone();
        let side = 100.0_f32;
        for i in 0..10_000u32 {
            let x = (i % 100) as f32 / 100.0 * side - side / 2.0;
            let z = (i / 100) as f32 / 100.0 * side - side / 2.0;
            app.world_mut()
                .spawn(BoidBundle::with_id(i, Target::default(), &variations))
                .insert(Transform::from_xyz(x, 0.5, z))
                .insert(Faction(0))
                .insert(Body::default());
        }
        app.update(); // tree refresh picks the boids up

        let start = Instant::now();
        app.insert_resource(TimeUpdateStrategy::ManualDuration(
            Duration::from_secs_f32(1.0 / 60.0),
        ));
        app.update();
        let elapsed = start.elapsed();
        println!("pbd_contact at 10k boids: {elapsed:?}");
        let budget_ms = if cfg!(debug_assertions) { 200 } else { 10 };
        assert!(
            elapsed.as_millis() as usize <= budget_ms,
            "pbd_contact at 10k took {:?} (budget {budget_ms} ms)",
            elapsed
        );
    }
}
