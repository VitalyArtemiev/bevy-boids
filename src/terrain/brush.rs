//! Debug terrain brushes: the spike harness for the deformation feel.
//!
//! 7/8/9/0 select raise/lower/flatten/smooth, [ and ] resize the brush,
//! hold G to apply at the cursor. Moats and earthen walls are composites
//! of these primitives.

use super::grounding::find_surface;
use super::noise::{BEDROCK_TOP_Y, surface_material};
use super::MainWorld;
use bevy::prelude::*;
use bevy_voxel_world::prelude::{VoxelWorld, WorldVoxel};
use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BrushMode {
    Raise,
    Lower,
    Flatten,
    Smooth,
}

/// Debug brush state, shared with the brush gizmo preview.
#[derive(Resource)]
pub struct TerrainBrush {
    pub mode: BrushMode,
    pub radius_m: f32,
}

impl Default for TerrainBrush {
    fn default() -> Self {
        TerrainBrush {
            mode: BrushMode::Raise,
            radius_m: 4.0,
        }
    }
}

const BRUSH_STRENGTH_M_PER_SEC: f32 = 6.0;
const BRUSH_MIN_RADIUS_M: f32 = 1.0;
const BRUSH_MAX_RADIUS_M: f32 = 16.0;
/// Column scan window around the cursor hit when looking for the current
/// surface: tall enough to find the top of a pile built by repeated raising.
const BRUSH_SCAN_UP_M: i32 = 32;
const BRUSH_SCAN_DOWN_M: i32 = 48;
/// Flatten/smooth approach rate, as a fraction of the remaining difference
/// per second.
const BRUSH_LEVEL_RATE_PER_SEC: f32 = 10.0;
const BRUSH_MIN_SURFACE_Y: i32 = BEDROCK_TOP_Y + 1;

/// Key bindings for the brush modes, in BrushMode declaration order.
const BRUSH_MODE_KEYS: [(KeyCode, BrushMode); 4] = [
    (KeyCode::Digit7, BrushMode::Raise),
    (KeyCode::Digit8, BrushMode::Lower),
    (KeyCode::Digit9, BrushMode::Flatten),
    (KeyCode::Digit0, BrushMode::Smooth),
];

/// Smooth (quadratic) falloff weight of a column at squared distance `d2`
/// from the brush centre.
fn brush_falloff(d2: f32, radius: f32) -> f32 {
    (1.0 - d2 / (radius * radius)).clamp(0.0, 1.0)
}

/// New surface height for one column under the brush, before falloff
/// weighting. Pure: the caller supplies the current surfaces (centre, this
/// column, and the column's 4-neighbourhood average for smoothing) and the
/// delta time to integrate at.
fn brush_target_height(
    mode: BrushMode,
    centre_h: f32,
    current: f32,
    neighbour_avg: f32,
    dt: f32,
) -> f32 {
    match mode {
        BrushMode::Raise => current + BRUSH_STRENGTH_M_PER_SEC * dt,
        BrushMode::Lower => current - BRUSH_STRENGTH_M_PER_SEC * dt,
        BrushMode::Flatten => current + (centre_h - current) * BRUSH_LEVEL_RATE_PER_SEC * dt,
        BrushMode::Smooth => current + (neighbour_avg - current) * BRUSH_LEVEL_RATE_PER_SEC * dt,
    }
}

pub fn terrain_brush_system(
    mut brush: ResMut<TerrainBrush>,
    keys: Res<ButtonInput<KeyCode>>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    windows: Query<&Window>,
    time: Res<Time>,
    mut voxel_world: VoxelWorld<MainWorld>,
    mut gizmos: Gizmos,
) {
    for (key, mode) in BRUSH_MODE_KEYS {
        if keys.just_pressed(key) {
            brush.mode = mode;
        }
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        brush.radius_m = (brush.radius_m - 1.0).max(BRUSH_MIN_RADIUS_M);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        brush.radius_m = (brush.radius_m + 1.0).min(BRUSH_MAX_RADIUS_M);
    }

    let Ok((camera, camera_transform)) = camera_query.single() else {
        return;
    };
    let Some(cursor) = windows.single().ok().and_then(|w| w.cursor_position()) else {
        return;
    };
    let Some(point) =
        crate::player::get_intersection(&voxel_world, &cursor, camera, camera_transform)
    else {
        return;
    };

    // Brush outline preview at the cursor: colour hints at the mode.
    let colour = match brush.mode {
        BrushMode::Raise => Color::srgb(0.2, 0.9, 0.2),
        BrushMode::Lower => Color::srgb(0.9, 0.3, 0.2),
        BrushMode::Flatten => Color::srgb(0.9, 0.8, 0.2),
        BrushMode::Smooth => Color::srgb(0.3, 0.5, 0.9),
    };
    gizmos.circle(
        Isometry3d::new(point + Vec3::Y * 0.05, Quat::from_rotation_x(-FRAC_PI_2)),
        brush.radius_m,
        colour,
    );

    if !keys.pressed(KeyCode::KeyG) {
        return;
    }

    let dt = time.delta_secs().max(0.001);
    let get_voxel = voxel_world.get_voxel_fn();
    let centre_voxel = point.floor().as_ivec3();
    let top = centre_voxel.y + 1 + BRUSH_SCAN_UP_M;
    let bottom = (centre_voxel.y + 1 - BRUSH_SCAN_DOWN_M).max(BRUSH_MIN_SURFACE_Y);
    let radius = brush.radius_m;

    // Read pass: current surfaces of every column in the disc, indexed for
    // O(1) neighbour lookups in the smooth pass (the disc is up to ~800
    // columns; scanning it per column would be quadratic).
    let r = radius.ceil() as i32;
    let mut surfaces: HashMap<(i32, i32), f32> = HashMap::with_capacity((2 * r + 1).pow(2) as usize);
    for dx in -r..=r {
        for dz in -r..=r {
            if (dx * dx + dz * dz) as f32 > radius * radius {
                continue;
            }
            let x = centre_voxel.x + dx;
            let z = centre_voxel.z + dz;
            if let Some(surface) = find_surface(&*get_voxel, x, z, top, bottom) {
                surfaces.insert((x, z), surface);
            }
        }
    }
    let centre_h = surfaces
        .get(&(centre_voxel.x, centre_voxel.z))
        .copied()
        .unwrap_or(centre_voxel.y as f32 + 1.0);

    // Compute + write pass. Smooth uses the average of the 4-neighbourhood
    // from the pre-edit snapshot so the result does not smear asymmetrically.
    let neighbour_avg = |x: i32, z: i32| -> f32 {
        let mut sum = 0.0;
        let mut n = 0.0;
        for (nx, nz) in [(x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)] {
            if let Some(&nh) = surfaces.get(&(nx, nz)) {
                sum += nh;
                n += 1.0;
            }
        }
        if n > 0.0 {
            sum / n
        } else {
            centre_h
        }
    };

    for (&(x, z), &current) in &surfaces {
        let d2 = ((x - centre_voxel.x).pow(2) + (z - centre_voxel.z).pow(2)) as f32;
        let weight = brush_falloff(d2, radius);
        if weight <= 0.0 {
            continue;
        }
        let avg = neighbour_avg(x, z);
        let target = brush_target_height(brush.mode, centre_h, current, avg, dt);
        let new_h = (current + (target - current) * weight).max(BRUSH_MIN_SURFACE_Y as f32);
        let old_i = current.floor() as i32;
        let new_i = new_h.floor() as i32;
        let material = surface_material(new_i);
        if new_i > old_i {
            for y in (old_i + 1)..=new_i {
                voxel_world.set_voxel(IVec3::new(x, y, z), WorldVoxel::Solid(material));
            }
        } else if new_i < old_i {
            for y in (new_i + 1)..=old_i {
                voxel_world.set_voxel(IVec3::new(x, y, z), WorldVoxel::Air);
            }
        }
    }
}
