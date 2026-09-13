//! Impostor atlas layout: which view directions a baked boid atlas covers
//! and where each lands in the texture.
//!
//! Far-away boids render as billboards whose texture is picked by the view
//! direction. Because the billboard camera looks mostly down, only a cone
//! around straight down (nadir) is baked: [`MAX_ANGLE_FROM_VERTICAL`] wide,
//! sampled as concentric rings — the straight-down render sits at the centre
//! of the conceptual disk (atlas row 0) and every further row is one ring at
//! a larger polar angle, its cells spread evenly around the azimuth.
//!
//! # Why rings-in-rows instead of a literal disk
//!
//! Rasterising a disk of views into a square cell grid leaves collisions
//! (two samples rounding into the same cell) and holes (grid cells no ring
//! passes through). Unrolling each ring into its own atlas row keeps the
//! spiral-around-the-centre structure while making every sample addressable
//! with plain `(row, column)` arithmetic: row = polar angle, column =
//! azimuth, with the column count per row growing with ring circumference
//! so adjacent samples stay roughly equal arc length apart.
//!
//! # Orientation folding
//!
//! Boid models are upright, so rotating the model by yaw φ and
//! counter-rotating the view azimuth by φ produce the same projection. The
//! atlas therefore needs no separate yaw axis: a boid at world yaw φ seen at
//! view azimuth α uses the cell for azimuth α − φ (see [`cell_for`]). This
//! folds the (view direction × boid yaw) parameter space into the 2-D disk.
//! The cost is lighting: the sun stays fixed while the geometry effectively
//! rotates, so baked shading is exact only near the yaw the model was baked
//! at (identity) and drifts with the folded angle — acceptable at billboard
//! size, worth revisiting if distant units ever look flat.
//!
//! # Addressing
//!
//! Cell `(x, y)`: `y == 0` is the nadir render; `y == r` (for
//! `r` in `1..=RING_CELL_COUNTS.len()`) is the ring at polar angle
//! `r · MAX_ANGLE_FROM_VERTICAL / RING_CELL_COUNTS.len()` with
//! `RING_CELL_COUNTS[r - 1]` azimuth slots, slot `x` covering
//! `[x/n · τ, (x+1)/n · τ)` sampled at its centre.

use std::f32::consts::TAU;

use bevy::math::Vec3;
use bevy::prelude::{Transform, UVec2};

/// Maximum angle between the camera's view direction and straight down that
/// the impostor atlas covers. Beyond it a camera would see a boid from the
/// side — a projection the atlas deliberately does not store.
///
/// NOTE: the current `RtsCamera` pitch floor is 20° above the horizon (70°
/// from vertical). The 30° cap is the forward-looking assumption for the
/// steep far camera of the large-battle LOD work; the runtime billboard
/// lookup must clamp to this cone (or fall back to the mesh) beyond it.
pub const MAX_ANGLE_FROM_VERTICAL: f32 = 30.0_f32.to_radians();

/// Azimuth slots per ring, innermost first. Proportional to ring
/// circumference (`sin` of the polar angle), keeping the angular distance
/// between neighbouring samples roughly constant across the disk.
pub const RING_CELL_COUNTS: [usize; 3] = [8, 16, 24];

/// Edge length of one atlas cell, pixels.
pub const CELL_SIZE_PX: u32 = 64;

/// Atlas size in cells: width = widest ring, height = rings + nadir row.
/// (`RING_CELL_COUNTS` is ascending; a test pins that so the indexing here
/// can rely on it.)
pub const ATLAS_GRID: UVec2 = UVec2::new(
    RING_CELL_COUNTS[RING_CELL_COUNTS.len() - 1] as u32,
    RING_CELL_COUNTS.len() as u32 + 1,
);

/// Atlas size in pixels.
pub const ATLAS_SIZE_PX: UVec2 = UVec2::new(
    ATLAS_GRID.x * CELL_SIZE_PX,
    ATLAS_GRID.y * CELL_SIZE_PX,
);

/// One baked view: where the pre-render camera sits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewSample {
    /// Angle of the view direction away from straight down, radians.
    pub polar: f32,
    /// Azimuth of the view direction around the vertical, radians
    /// (0 = world +X, growing toward +Z, matching `sky::sun_transform`).
    pub azimuth: f32,
    /// Atlas grid cell the render lands in.
    pub cell: UVec2,
}

/// Every view the preprocessor bakes, nadir first then rings outward.
pub fn view_samples() -> Vec<ViewSample> {
    let mut samples = vec![ViewSample {
        polar: 0.0,
        azimuth: 0.0,
        cell: UVec2::ZERO,
    }];
    for (ring, count) in RING_CELL_COUNTS.iter().enumerate() {
        let polar = (ring + 1) as f32 / RING_CELL_COUNTS.len() as f32 * MAX_ANGLE_FROM_VERTICAL;
        for slot in 0..*count {
            let azimuth = (slot as f32 + 0.5) * TAU / *count as f32;
            samples.push(ViewSample {
                polar,
                azimuth,
                cell: UVec2::new(slot as u32, ring as u32 + 1),
            });
        }
    }
    samples
}

/// Direction a camera looks from, `polar` away from straight down at
/// `azimuth` around the vertical.
pub fn view_direction(polar: f32, azimuth: f32) -> Vec3 {
    let (polar_sin, polar_cos) = polar.sin_cos();
    let (az_sin, az_cos) = azimuth.sin_cos();
    Vec3::new(polar_sin * az_cos, -polar_cos, polar_sin * az_sin)
}

/// Camera pose rendering the model at `center` from the given sample. The
/// up hint is the horizontal azimuth vector, which is perpendicular to the
/// view direction at nadir and never parallel to it inside the cone, so the
/// pose varies continuously across the whole disk (at nadir the image up
/// axis is exactly world +X rotated to the sample's azimuth).
pub fn camera_transform(sample: &ViewSample, center: Vec3, distance: f32) -> Transform {
    let view = view_direction(sample.polar, sample.azimuth);
    let (az_sin, az_cos) = sample.azimuth.sin_cos();
    let up = Vec3::new(az_cos, 0.0, az_sin);
    Transform::from_translation(center - view * distance).looking_to(view, up)
}

/// Nearest atlas cell for a runtime view at `polar` from nadir and
/// `azimuth` (any value; wrapped). Polar angles beyond
/// [`MAX_ANGLE_FROM_VERTICAL`] clamp to the outermost ring — the documented
/// approximation for views outside the baked cone.
pub fn cell_for(polar: f32, azimuth: f32) -> UVec2 {
    let rings = RING_CELL_COUNTS.len() as f32;
    let ring = (polar / MAX_ANGLE_FROM_VERTICAL * rings).round().clamp(0.0, rings) as u32;
    if ring == 0 {
        // Nadir: an upright model's projection does not depend on azimuth.
        return UVec2::ZERO;
    }
    let count = RING_CELL_COUNTS[(ring - 1) as usize] as u32;
    let slot = (azimuth.rem_euclid(TAU) / TAU * count as f32).floor() as u32 % count;
    UVec2::new(slot, ring)
}

/// Copies one `cell_size`-square RGBA8 cell into a row-major atlas buffer.
pub fn blit_cell(dst: &mut [u8], dst_width_px: u32, cell: UVec2, cell_size: u32, src: &[u8]) {
    let cell_size = cell_size as usize;
    let row_bytes = cell_size * 4;
    let dst_x = cell.x as usize * cell_size;
    let dst_y = cell.y as usize * cell_size;
    for y in 0..cell_size {
        let src_row = y * row_bytes;
        let dst_row = ((dst_y + y) * dst_width_px as usize + dst_x) * 4;
        dst[dst_row..dst_row + row_bytes]
            .copy_from_slice(&src[src_row..src_row + row_bytes]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_fill_unique_cells_within_the_grid() {
        // `ATLAS_GRID` indexes the widest ring last, so the counts must
        // ascend for the const to hold.
        assert!(
            RING_CELL_COUNTS.is_sorted(),
            "ring cell counts must ascend with circumference"
        );
        let samples = view_samples();
        assert_eq!(
            samples.len(),
            1 + RING_CELL_COUNTS.iter().sum::<usize>(),
            "one nadir cell plus every ring slot"
        );
        let mut seen = std::collections::HashSet::new();
        for sample in &samples {
            assert!(seen.insert(sample.cell), "duplicate cell {:?}", sample.cell);
            assert!(sample.cell.x < ATLAS_GRID.x && sample.cell.y < ATLAS_GRID.y);
            assert!((0.0..=MAX_ANGLE_FROM_VERTICAL).contains(&sample.polar));
        }
        for (ring, count) in RING_CELL_COUNTS.iter().enumerate() {
            let in_row = samples
                .iter()
                .filter(|s| s.cell.y == ring as u32 + 1)
                .count();
            assert_eq!(in_row, *count, "ring {ring} row occupancy");
        }
    }

    #[test]
    fn nadir_view_points_straight_down() {
        assert_eq!(view_direction(0.0, 0.0), Vec3::NEG_Y);
        assert_eq!(view_direction(0.0, 2.1), Vec3::NEG_Y);
    }

    #[test]
    fn polar_argument_sets_the_angle_from_nadir() {
        for polar in [0.0, 0.05, MAX_ANGLE_FROM_VERTICAL / 2.0, MAX_ANGLE_FROM_VERTICAL] {
            let angle = view_direction(polar, 1.7)
                .angle_between(Vec3::NEG_Y)
                .abs();
            assert!((angle - polar).abs() < 1e-5, "polar {polar} gave {angle}");
        }
    }

    #[test]
    fn camera_poses_look_at_the_model_along_the_view_direction() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        for sample in view_samples() {
            let pose = camera_transform(&sample, center, 10.0);
            let view = view_direction(sample.polar, sample.azimuth);
            assert!(pose.forward().angle_between(view).abs() < 1e-5);
            assert!((pose.translation - (center - view * 10.0)).length() < 1e-5);
        }
    }

    #[test]
    fn camera_up_follows_the_azimuth_and_never_flips() {
        for sample in view_samples() {
            let up = camera_transform(&sample, Vec3::ZERO, 10.0).up();
            let (az_sin, az_cos) = sample.azimuth.sin_cos();
            let hint = Vec3::new(az_cos, 0.0, az_sin);
            // Orthogonalizing the hint against the view direction tilts it
            // by exactly the polar angle — up keeps the azimuth and sheds
            // only the view-parallel component, never flipping sides.
            assert!(
                (up.dot(hint) - sample.polar.cos()).abs() < 1e-5,
                "up {up} lost the azimuth hint {hint}"
            );
        }
        // At nadir the hint is already perpendicular to the view, so the
        // camera up must land on it exactly — the property the runtime
        // billboard rotation relies on.
        let nadir = camera_transform(&view_samples()[0], Vec3::ZERO, 10.0);
        assert!((*nadir.up() - Vec3::X).length() < 1e-5);
    }

    #[test]
    fn cell_for_round_trips_every_sample() {
        for sample in view_samples() {
            assert_eq!(cell_for(sample.polar, sample.azimuth), sample.cell);
        }
    }

    #[test]
    fn cell_for_wraps_azimuth_and_clamps_polar() {
        assert_eq!(cell_for(0.0, -0.1), UVec2::ZERO);
        assert_eq!(cell_for(MAX_ANGLE_FROM_VERTICAL, TAU - 0.01), {
            let count = RING_CELL_COUNTS[RING_CELL_COUNTS.len() - 1] as u32;
            UVec2::new(count - 1, RING_CELL_COUNTS.len() as u32)
        });
        // Far beyond the cone clamps onto the outer ring instead of
        // indexing out of the atlas.
        let outer = cell_for(2.0 * MAX_ANGLE_FROM_VERTICAL, 0.0);
        assert_eq!(outer.y, RING_CELL_COUNTS.len() as u32);
        assert!(outer.x < RING_CELL_COUNTS[RING_CELL_COUNTS.len() - 1] as u32);
    }

    quickcheck::quickcheck! {
        fn cell_for_stays_in_grid(polar: f32, azimuth: f32) -> bool {
            let cell = cell_for(polar, azimuth);
            cell.x < ATLAS_GRID.x && cell.y < ATLAS_GRID.y
        }
    }

    #[test]
    fn blit_places_cells_side_by_side() {
        // Two 2×2-px cells in a 4-px-wide atlas: cell (1,0) must land right
        // of cell (0,0) without overwriting it.
        let mut dst = [0u8; 4 * 2 * 4];
        let a = [1u8; 2 * 2 * 4];
        let b = [2u8; 2 * 2 * 4];
        blit_cell(&mut dst, 4, UVec2::new(0, 0), 2, &a);
        blit_cell(&mut dst, 4, UVec2::new(1, 0), 2, &b);
        for y in 0..2 {
            let row = y * 4 * 4;
            assert_eq!(&dst[row..row + 2 * 4], &[1u8; 8]);
            assert_eq!(&dst[row + 2 * 4..row + 4 * 4], &[2u8; 8]);
        }
    }
}
