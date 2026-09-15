//! Impostor atlas layout: which view directions a baked boid atlas covers
//! and where each lands in the texture.
//!
//! Far-away boids render as billboards whose texture is picked by the view
//! direction. Because the billboard camera looks mostly down, only a cone
//! around straight down (nadir) is baked: [`MAX_ANGLE_FROM_VERTICAL`] wide,
//! sampled as concentric rings — the straight-down render first, then every
//! ring at a larger polar angle, its samples spread evenly around the
//! azimuth.
//!
//! # Packing: albedo and mirrored normals in one texture
//!
//! Every view is baked twice — an unlit albedo render and a view-space
//! normal render — for every animation pose, and all of it lives in one
//! atlas with zero wasted cells: each pose's albedo cells fill a horizontal
//! band of the top half in flat view-index order, and normal cells sit at
//! the whole-texture 180° rotation of their albedo cell (bottom half,
//! reversed). A runtime shader therefore samples the normal for any albedo
//! uv with the single mirror `nuv = 1.0 - uv` — for any pose — and the
//! texture spends every cell on data.
//!
//! Both halves are stored sRGB-encoded (the capture target writes through
//! an sRGB view): load the atlas as an sRGB texture — Bevy's default for
//! color images — and decode normals with `n = 2 * sampled - 1`.
//!
//! # Orientation folding
//!
//! Boid models are upright, so rotating the model by yaw φ and
//! counter-rotating the view azimuth by φ produce the same projection. The
//! atlas therefore needs no separate yaw axis: a boid at world yaw φ seen at
//! view azimuth α uses the view sampled at azimuth α − φ (see
//! [`view_index`]). This folds the (view direction × boid yaw) parameter
//! space into the 2-D cone. The cost is lighting, which is exactly why the
//! albedo bakes unlit and normals bake separately: shading is applied at
//! runtime from the normal render instead of being baked at one yaw.
//!
//! # Addressing
//!
//! View `i` (in [`view_samples`] order: nadir, then ring slots innermost to
//! outermost) in pose `p` occupies [`albedo_cell`]`(i, p)` and
//! [`normal_cell`]`(i, p)`. A runtime view at (polar, azimuth) maps to its
//! nearest view index via [`view_index`].

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

/// Azimuth samples per ring, innermost first. Proportional to ring
/// circumference (`sin` of the polar angle), keeping the angular distance
/// between neighbouring samples roughly constant across the cone.
pub const RING_CELL_COUNTS: [usize; 3] = [8, 16, 24];

/// Edge length of one atlas cell, pixels.
pub const CELL_SIZE_PX: u32 = 64;

/// Headroom around the fitted bounding sphere, fraction of its radius: the
/// silhouette must not touch the cell edge, or bilinear filtering at
/// billboard time would bleed the neighbouring cell in. The bake frustum
/// and the runtime billboard quad both size themselves from this, so the
/// swap never pops in apparent size.
pub const FIT_MARGIN: f32 = 1.1;

/// Animation poses baked per variation (see `bake_poses` in the bake
/// module). Every pose reuses the same view cone, stacked as horizontal
/// bands — future walk/attack frames become additional poses without any
/// layout change. Currently just the idle pose: the axis is plumbed and
/// tested, but no animation exists to feed it yet.
pub const POSE_COUNT: u32 = 1;

/// Total baked views: the nadir plus every ring slot.
const VIEW_COUNT: usize = 1 + RING_CELL_COUNTS[0] + RING_CELL_COUNTS[1] + RING_CELL_COUNTS[2];

/// Atlas width in cells. Chosen so the albedo half packs exactly for the
/// current [`RING_CELL_COUNTS`] (49 views = a clean 7×7 half per pose,
/// zero waste for any pose count); a test pins the exact fit so ring
/// changes must revisit it.
pub const ATLAS_WIDTH_CELLS: u32 = 7;

/// Rows used by one render kind (albedo or normals) of one pose.
pub const HALF_ROWS: u32 = (VIEW_COUNT as u32 + ATLAS_WIDTH_CELLS - 1) / ATLAS_WIDTH_CELLS;

/// Atlas size in cells: every pose's albedo band stacks in the top half,
/// the mirrored normal bands in the bottom half.
pub const ATLAS_GRID: UVec2 = UVec2::new(ATLAS_WIDTH_CELLS, 2 * HALF_ROWS * POSE_COUNT);

/// Atlas size in pixels.
pub const ATLAS_SIZE_PX: UVec2 =
    UVec2::new(ATLAS_GRID.x * CELL_SIZE_PX, ATLAS_GRID.y * CELL_SIZE_PX);

/// One baked view: where the pre-render camera sits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewSample {
    /// Angle of the view direction away from straight down, radians.
    pub polar: f32,
    /// Azimuth of the view direction around the vertical, radians
    /// (0 = world +X, growing toward +Z, matching `sky::sun_transform`).
    pub azimuth: f32,
}

/// Every view the preprocessor bakes, nadir first then rings outward. The
/// position in this list is the view's index into the atlas.
pub fn view_samples() -> Vec<ViewSample> {
    let mut samples = vec![ViewSample {
        polar: 0.0,
        azimuth: 0.0,
    }];
    for (ring, count) in RING_CELL_COUNTS.iter().enumerate() {
        let polar = (ring + 1) as f32 / RING_CELL_COUNTS.len() as f32 * MAX_ANGLE_FROM_VERTICAL;
        for slot in 0..*count {
            let azimuth = (slot as f32 + 0.5) * TAU / *count as f32;
            samples.push(ViewSample { polar, azimuth });
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
/// pose varies continuously across the whole cone (at nadir the image up
/// axis is exactly world +X rotated to the sample's azimuth).
pub fn camera_transform(sample: &ViewSample, center: Vec3, distance: f32) -> Transform {
    let view = view_direction(sample.polar, sample.azimuth);
    let (az_sin, az_cos) = sample.azimuth.sin_cos();
    let up = Vec3::new(az_cos, 0.0, az_sin);
    Transform::from_translation(center - view * distance).looking_to(view, up)
}

/// Flat index of the nearest baked view for a runtime view at `polar` from
/// nadir and `azimuth` (any value; wrapped) — the index
/// [`albedo_cell`]/[`normal_cell`] address. Polar angles beyond
/// [`MAX_ANGLE_FROM_VERTICAL`] clamp to the outermost ring — the documented
/// approximation for views outside the baked cone.
///
/// The billboard shader mirrors this lookup in wgsl
/// (`assets/shaders/impostor_billboard.wgsl`); this Rust twin is the
/// test-pinned reference the shader must match.
#[cfg_attr(not(test), allow(dead_code))]
pub fn view_index(polar: f32, azimuth: f32) -> usize {
    let rings = RING_CELL_COUNTS.len() as f32;
    let ring = (polar / MAX_ANGLE_FROM_VERTICAL * rings)
        .round()
        .clamp(0.0, rings) as usize;
    if ring == 0 {
        // Nadir: an upright model's projection does not depend on azimuth.
        return 0;
    }
    let count = RING_CELL_COUNTS[ring - 1];
    let slot = (azimuth.rem_euclid(TAU) / TAU * count as f32).floor() as usize % count;
    1 + RING_CELL_COUNTS[..ring - 1].iter().sum::<usize>() + slot
}

/// Albedo cell of view `index` in pose `pose`: flat row-major order within
/// the pose's band of the top half.
pub fn albedo_cell(index: usize, pose: usize) -> UVec2 {
    UVec2::new(
        index as u32 % ATLAS_GRID.x,
        pose as u32 * HALF_ROWS + index as u32 / ATLAS_GRID.x,
    )
}

/// Normal cell of view `index` in pose `pose`: the albedo cell mirrored
/// through the texture centre — the whole-texture 180° rotation that lets a
/// shader fetch it with `nuv = 1.0 - uv` regardless of pose count.
pub fn normal_cell(index: usize, pose: usize) -> UVec2 {
    let albedo = albedo_cell(index, pose);
    UVec2::new(ATLAS_GRID.x - 1 - albedo.x, ATLAS_GRID.y - 1 - albedo.y)
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
        dst[dst_row..dst_row + row_bytes].copy_from_slice(&src[src_row..src_row + row_bytes]);
    }
}

/// [`blit_cell`], but rotating the source image 180° on the way in. The
/// normal half's mirror convention is the *whole-texture* 180° rotation:
/// the cell placement ([`normal_cell`]) AND the image content rotate, so a
/// runtime shader recovers the normal for any albedo uv with the single
/// mirror `nuv = 1.0 - uv`. Blitting upright pairs every albedo pixel with
/// the opposite sprite point's normal — the billboard then lights from the
/// anti-sun side (sun at zenith glows the sprite's bottom half).
pub fn blit_cell_rot180(
    dst: &mut [u8],
    dst_width_px: u32,
    cell: UVec2,
    cell_size: u32,
    src: &[u8],
) {
    let cell_size = cell_size as usize;
    let row_bytes = cell_size * 4;
    let dst_x = cell.x as usize * cell_size;
    let dst_y = cell.y as usize * cell_size;
    for y in 0..cell_size {
        let src_row = (cell_size - 1 - y) * row_bytes;
        let dst_row = ((dst_y + y) * dst_width_px as usize + dst_x) * 4;
        for x in 0..cell_size {
            let src_px = src_row + (cell_size - 1 - x) * 4;
            let dst_px = dst_row + x * 4;
            dst[dst_px..dst_px + 4].copy_from_slice(&src[src_px..src_px + 4]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_cover_the_cone_in_ring_order() {
        // `ATLAS_WIDTH_CELLS` packs the albedo half exactly for the current
        // counts — keep the two in sync when retuning the rings.
        assert!(
            RING_CELL_COUNTS.is_sorted(),
            "ring cell counts must ascend with circumference"
        );
        assert_eq!(
            view_samples().len(),
            VIEW_COUNT,
            "one nadir sample plus every ring slot"
        );
        assert_eq!(
            ATLAS_GRID.x * ATLAS_GRID.y,
            2 * VIEW_COUNT as u32 * POSE_COUNT,
            "the atlas must spend every cell on data"
        );
        for (ring, count) in RING_CELL_COUNTS.iter().enumerate() {
            let in_ring = view_samples()
                .iter()
                .filter(|s| (s.polar.to_degrees() - 10.0 * (ring + 1) as f32).abs() < 0.5)
                .count();
            assert_eq!(in_ring, *count, "ring {ring} occupancy");
        }
        assert!(
            view_samples()
                .iter()
                .all(|s| s.polar <= MAX_ANGLE_FROM_VERTICAL)
        );
    }

    #[test]
    fn nadir_view_points_straight_down() {
        assert_eq!(view_direction(0.0, 0.0), Vec3::NEG_Y);
        assert_eq!(view_direction(0.0, 2.1), Vec3::NEG_Y);
    }

    #[test]
    fn polar_argument_sets_the_angle_from_nadir() {
        for polar in [
            0.0,
            0.05,
            MAX_ANGLE_FROM_VERTICAL / 2.0,
            MAX_ANGLE_FROM_VERTICAL,
        ] {
            let angle = view_direction(polar, 1.7).angle_between(Vec3::NEG_Y).abs();
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
    fn view_index_round_trips_every_sample() {
        for (index, sample) in view_samples().iter().enumerate() {
            assert_eq!(view_index(sample.polar, sample.azimuth), index);
        }
    }

    #[test]
    fn view_index_wraps_azimuth_and_clamps_polar() {
        assert_eq!(view_index(0.0, -0.1), 0);
        assert_eq!(
            view_index(MAX_ANGLE_FROM_VERTICAL, TAU - 0.01),
            VIEW_COUNT - 1
        );
        // Far beyond the cone clamps onto the outer ring instead of
        // indexing out of the atlas: azimuth 0 maps to its first slot.
        let outer_ring_start = 1 + RING_CELL_COUNTS[0] + RING_CELL_COUNTS[1];
        assert!(view_index(2.0 * MAX_ANGLE_FROM_VERTICAL, 0.0) >= outer_ring_start);
    }

    quickcheck::quickcheck! {
        fn view_index_stays_in_range(polar: f32, azimuth: f32) -> bool {
            view_index(polar, azimuth) < view_samples().len()
        }
    }

    #[test]
    fn normals_mirror_albedo_through_the_texture_centre() {
        let mut seen = std::collections::HashSet::new();
        for pose in 0..POSE_COUNT as usize {
            for index in 0..VIEW_COUNT {
                let albedo = albedo_cell(index, pose);
                let normal = normal_cell(index, pose);
                assert!(albedo.x < ATLAS_GRID.x && albedo.y < HALF_ROWS * (pose as u32 + 1));
                assert!(albedo.y >= HALF_ROWS * pose as u32, "band containment");
                assert!(normal.y >= ATLAS_GRID.y - HALF_ROWS * (pose as u32 + 1));
                // The exact whole-texture 180° rotation — the `nuv = 1 - uv`
                // pairing the runtime shader relies on.
                assert_eq!(
                    normal,
                    UVec2::new(ATLAS_GRID.x - 1 - albedo.x, ATLAS_GRID.y - 1 - albedo.y)
                );
                assert!(
                    seen.insert(albedo),
                    "duplicate albedo cell for view {index}"
                );
                assert!(
                    seen.insert(normal),
                    "duplicate normal cell for view {index}"
                );
            }
        }
        assert_eq!(
            seen.len(),
            2 * VIEW_COUNT * POSE_COUNT as usize,
            "no cell is shared or wasted"
        );
    }

    #[test]
    fn pose_bands_stack_for_future_pose_counts() {
        // The pose axis ships a single idle pose, so pin the band
        // arithmetic for indices the atlas doesn't currently use: albedo
        // bands stack at HALF_ROWS stride. (`normal_cell`'s mirror is
        // defined against the shipped grid height, so a larger pose count
        // regenerates the atlas and the mirror together — the 180°
        // relation itself is pinned by the test above for shipped poses.)
        for pose in 1..=3usize {
            assert_eq!(
                albedo_cell(0, pose).y,
                pose as u32 * HALF_ROWS,
                "band starts at its stride"
            );
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

    #[test]
    fn blit_rot180_reverses_both_axes() {
        // A 2×2 cell with four distinct pixels; the rot180 blit must place
        // each at the diagonally opposite corner — the content rotation
        // `nuv = 1 - uv` reads back (upright blits light the billboard from
        // the anti-sun side).
        let mut dst = [0u8; 2 * 2 * 4];
        let px = |r: u8, g: u8, b: u8| [r, g, b, 255];
        let src = [px(10, 0, 0), px(20, 0, 0), px(30, 0, 0), px(40, 0, 0)].concat();
        blit_cell_rot180(&mut dst, 2, UVec2::new(0, 0), 2, &src);
        // src row-major [10 20 / 30 40] → rotated [40 30 / 20 10].
        assert_eq!(&dst[0..4], px(40, 0, 0));
        assert_eq!(&dst[4..8], px(30, 0, 0));
        assert_eq!(&dst[8..12], px(20, 0, 0));
        assert_eq!(&dst[12..16], px(10, 0, 0));
    }
}
