//! Pure terrain walkability derivation, plus a debug overlay.
//!
//! Walkability is reduced from the voxel world to a 2D grid of per-column
//! surface heights (`Option<f32>`: `None` = no surface in the scanned
//! window / chunk not generated). The current model is the top surface
//! only — multi-span columns (tunnels, ramparts, halls) arrive with fort
//! structures; the API takes surfaces as plain data so the span model can
//! extend it without touching consumers.
//!
//! Two primitives back the later pathfinding tiers (flow fields for units,
//! clearance-aware sector search for formations):
//! - [`traversable`]: can a unit step between two adjacent surfaces?
//! - [`WalkField::clearance`]: how many cells from the nearest impassable
//!   cell — a formation of footprint W plans routes where clearance >= W.

use super::MainWorld;
use super::grounding::find_surface;
use bevy::prelude::*;
use bevy_rts_camera::RtsCamera;
use bevy_voxel_world::prelude::VoxelWorld;
use std::f32::consts::FRAC_PI_2;

/// Maximum height difference a unit can step up or down between adjacent
/// columns (metres; one voxel = 1 m).
pub const MAX_STEP_M: f32 = 1.0;

/// A cell whose neighbourhood drops faster than this is a cliff: standable
/// in principle but unusable for planning, so it is marked impassable.
pub const CLIFF_DROP_M: f32 = 3.0;

/// Can a unit step between two adjacent column surfaces? Void-to-void and
/// step-too-high transitions are not traversable.
pub fn traversable(from: Option<f32>, to: Option<f32>) -> bool {
    match (from, to) {
        (Some(a), Some(b)) => (a - b).abs() <= MAX_STEP_M,
        _ => false,
    }
}

/// Grid of passability and clearance over a rectangular window of column
/// surfaces, row-major, `width * height` cells.
pub struct WalkField {
    pub width: usize,
    pub height: usize,
    /// A cell is passable when it has a surface and does not drop more than
    /// [`CLIFF_DROP_M`] toward any existing neighbour.
    passable: Vec<bool>,
    /// Manhattan distance (in cells, saturating) to the nearest impassable
    /// cell; 0 on impassable cells themselves.
    clearance: Vec<u8>,
}

impl WalkField {
    pub fn from_surfaces(surfaces: &[Option<f32>], width: usize, height: usize) -> Self {
        assert_eq!(
            surfaces.len(),
            width * height,
            "surface window must be width * height"
        );
        let idx = |x: usize, z: usize| z * width + x;

        let mut passable = vec![false; surfaces.len()];
        for z in 0..height {
            for x in 0..width {
                let Some(h) = surfaces[idx(x, z)] else {
                    continue;
                };
                // Cliff check, one-sided: a cell is impassable when it drops
                // more than CLIFF_DROP_M toward some existing neighbour (a
                // mesa edge). The low side of a drop stays walkable, and
                // cells with no existing neighbours (window borders) carry
                // no evidence of a cliff and stay passable.
                let mut cliff = false;
                for (nx, nz) in [
                    (x + 1, z),
                    (x.wrapping_sub(1), z),
                    (x, z + 1),
                    (x, z.wrapping_sub(1)),
                ] {
                    if nx < width && nz < height {
                        if let Some(nh) = surfaces[idx(nx, nz)] {
                            if h - nh > CLIFF_DROP_M {
                                cliff = true;
                                break;
                            }
                        }
                    }
                }
                passable[idx(x, z)] = !cliff;
            }
        }

        let clearance = clearance_transform(&passable, width, height);
        WalkField {
            width,
            height,
            passable,
            clearance,
        }
    }

    pub fn passable(&self, x: usize, z: usize) -> bool {
        self.passable[z * self.width + x]
    }

    pub fn clearance(&self, x: usize, z: usize) -> u8 {
        self.clearance[z * self.width + x]
    }
}

/// Two-pass Manhattan distance transform to the nearest impassable cell,
/// saturating at u8::MAX. Classic chamfer sweep: forward (top-left to
/// bottom-right) propagating from up/left, backward from down/right.
fn clearance_transform(passable: &[bool], width: usize, height: usize) -> Vec<u8> {
    let mut dist: Vec<u8> = passable
        .iter()
        .map(|&p| if p { u8::MAX } else { 0 })
        .collect();
    let idx = |x: usize, z: usize| z * width + x;

    for z in 0..height {
        for x in 0..width {
            let i = idx(x, z);
            if !passable[i] {
                continue;
            }
            if x > 0 {
                dist[i] = dist[i].min(dist[idx(x - 1, z)].saturating_add(1));
            }
            if z > 0 {
                dist[i] = dist[i].min(dist[idx(x, z - 1)].saturating_add(1));
            }
        }
    }
    for z in (0..height).rev() {
        for x in (0..width).rev() {
            let i = idx(x, z);
            if !passable[i] {
                continue;
            }
            if x + 1 < width {
                dist[i] = dist[i].min(dist[idx(x + 1, z)].saturating_add(1));
            }
            if z + 1 < height {
                dist[i] = dist[i].min(dist[idx(x, z + 1)].saturating_add(1));
            }
        }
    }
    dist
}

// ---------------------------------------------------------------------------
// Debug visualization: hold H to overlay passability around the camera
// focus. Samples the voxel world's column surfaces through the same scan
// the boid grounding uses, so what you see is what they walk on.
// ---------------------------------------------------------------------------

/// Half-extent of the sampled window (cells; one cell = one voxel column).
const DEBUG_WINDOW_HALF: usize = 24;
/// Column scan window above/below the camera focus (metres).
const DEBUG_SCAN_UP_M: i32 = 24;
const DEBUG_SCAN_DOWN_M: i32 = 24;

pub fn debug_walkability(
    keys: Res<ButtonInput<KeyCode>>,
    camera_query: Query<&RtsCamera>,
    voxel_world: VoxelWorld<MainWorld>,
    mut gizmos: Gizmos,
) {
    if !keys.pressed(KeyCode::KeyH) {
        return;
    }
    let Ok(camera) = camera_query.single() else {
        return;
    };
    let focus = camera.focus.translation;
    let get_voxel = voxel_world.get_voxel_fn();

    let w = DEBUG_WINDOW_HALF * 2 + 1;
    let x0 = focus.x.floor() as i32 - DEBUG_WINDOW_HALF as i32;
    let z0 = focus.z.floor() as i32 - DEBUG_WINDOW_HALF as i32;
    let top = (focus.y + DEBUG_SCAN_UP_M as f32) as i32;
    let bottom = (focus.y - DEBUG_SCAN_DOWN_M as f32) as i32;

    let mut surfaces = Vec::with_capacity(w * w);
    for dz in 0..w as i32 {
        for dx in 0..w as i32 {
            surfaces.push(find_surface(&*get_voxel, x0 + dx, z0 + dz, top, bottom));
        }
    }
    let field = WalkField::from_surfaces(&surfaces, w, w);

    for dz in 0..w {
        for dx in 0..w {
            if !field.passable(dx, dz) {
                let h = surfaces[dz * w + dx].unwrap_or(focus.y);
                gizmos.rect(
                    Isometry3d::new(
                        Vec3::new(
                            x0 as f32 + dx as f32 + 0.5,
                            h + 0.05,
                            z0 as f32 + dz as f32 + 0.5,
                        ),
                        Quat::from_rotation_x(-FRAC_PI_2),
                    ),
                    Vec2::new(0.9, 0.9),
                    Color::srgb(0.9, 0.2, 0.2),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::quickcheck;

    #[test]
    fn traversable_requires_two_surfaces_within_step() {
        assert!(traversable(Some(5.0), Some(5.0)));
        assert!(traversable(Some(5.0), Some(6.0)));
        assert!(!traversable(Some(5.0), Some(6.01)));
        assert!(!traversable(Some(5.0), None));
        assert!(!traversable(None, Some(5.0)));
        assert!(!traversable(None, None));
    }

    #[test]
    fn flat_ground_is_fully_passable_with_saturated_clearance() {
        let w = 8;
        let surfaces = vec![Some(3.0); w * w];
        let field = WalkField::from_surfaces(&surfaces, w, w);
        for z in 0..w {
            for x in 0..w {
                assert!(field.passable(x, z));
                assert_eq!(field.clearance(x, z), u8::MAX);
            }
        }
    }

    #[test]
    fn single_void_gets_manhattan_clearance() {
        let w = 5;
        let mut surfaces = vec![Some(2.0); w * w];
        surfaces[2 * w + 2] = None; // centre is void
        let field = WalkField::from_surfaces(&surfaces, w, w);
        assert!(!field.passable(2, 2));
        assert_eq!(field.clearance(2, 2), 0);
        // Manhattan ring around the void.
        assert_eq!(field.clearance(1, 2), 1);
        assert_eq!(field.clearance(2, 1), 1);
        assert_eq!(field.clearance(0, 2), 2);
        assert_eq!(field.clearance(2, 0), 2);
        assert_eq!(field.clearance(0, 0), 4);
    }

    #[test]
    fn cliff_edges_are_impassable() {
        // A 1-wide mesa 5 m tall: every top cell borders a 5 m drop.
        let w = 3;
        let mut surfaces = vec![Some(0.0); w * w];
        surfaces[1 * w + 1] = Some(5.0);
        let field = WalkField::from_surfaces(&surfaces, w, w);
        assert!(!field.passable(1, 1), "isolated mesa is not passable");
        // The ground ring keeps passability: it has flat neighbours.
        assert!(field.passable(0, 0));
    }

    quickcheck! {
        fn clearance_matches_bruteforce_manhattan(
            passable: Vec<bool>,
            width: u8
        ) -> bool {
            let width = (width as usize).max(1);
            let height = passable.len().div_ceil(width).max(1);
            let mut padded = passable;
            padded.resize(width * height, true);
            let field = WalkField::from_surfaces(
                &padded.iter().map(|&p| if p { Some(1.0) } else { None }).collect::<Vec<_>>(),
                width,
                height,
            );
            // Brute force: exact Manhattan distance to nearest impassable.
            for z in 0..height {
                for x in 0..width {
                    let expect = (0..height)
                        .flat_map(|zz| (0..width).map(move |xx| (xx, zz)))
                        .filter(|&(xx, zz)| !padded[zz * width + xx])
                        .map(|(xx, zz)| xx.abs_diff(x) + zz.abs_diff(z))
                        .min()
                        .unwrap_or(u8::MAX as usize)
                        .min(u8::MAX as usize) as u8;
                    if field.clearance(x, z) != expect {
                        return false;
                    }
                }
            }
            true
        }
    }
}
