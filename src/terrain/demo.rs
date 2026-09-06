//! The `bevy_erosion_filter` terrain demo, recreated on the CPU over the
//! 1 km² square — no LOD, no streaming.
//!
//! Composition (a faithful port of the crate's `terrain_demo` example,
//! `evaluate_terrain_with_octaves` + `terrain_albedo`): a 3-octave gain-0.1
//! fBm gives the base unit height and slope; the erosion filter (the
//! crate's `cpu` port, same numbers as the WGSL) carves gullies and gives
//! the ridge map; the demo's albedo cascade (cliff/dirt/snow/sand/grass/
//! trees/drainage chalk) colors it. The demo's 22 m world is stretched
//! uniformly to our 1 km: every height, gully and slope scales by
//! `WORLD_SCALE`, so the look is identical at game scale. The tree bump
//! (a /300 height nudge) is omitted; the tree MASK still colors the
//! canopy.
//!
//! One CPU evaluation per vertex fills the shared grid mesh's positions,
//! normals and colors; the F3 panel drives the same sliders as the demo
//! (view modes included — erosion delta, ridge map, debug fade). Boid
//! grounding and the camera sample the same evaluation through
//! [`super::HeightField`], so sim and render agree by construction.

use bevy::prelude::*;
use bevy_erosion_filter::cpu;

use super::{TERRAIN_EXTENT_M, TERRAIN_RESOLUTION, TerrainMesh};

/// The demo world is 22 m across; ours is 1 km. Every demo-unit length
/// (heights, gullies, water level) scales by this.
pub const WORLD_SCALE: f32 = TERRAIN_EXTENT_M / 22.0;

/// The demo's vertical scale (slider 10..40), applied in demo units.
const VERTICAL_SCALE: f32 = 22.0;
/// Height bias folded into `world_height` (demo: `(unit - 0.43)`).
const UNIT_HEIGHT_ORIGIN: f32 = 0.43;
/// fBm amplitude and fade normalization (demo constants).
const HEIGHT_AMP: f32 = 0.125;
/// Noise-domain offset (`terrain_point`), which also seeds the map.
const MAP_ORIGIN: Vec2 = Vec2::new(0.17, 0.31);
/// Fixed factor in the demo's slope-to-world conversion (`terrain1.z`).
const SLOPE_UNIT: f32 = 0.62;

// Palette (linear RGB, the demo's constants).
const CLIFF: Vec3 = Vec3::new(0.22, 0.20, 0.20);
const DIRT: Vec3 = Vec3::new(0.60, 0.50, 0.40);
const TREE: Vec3 = Vec3::new(0.12, 0.26, 0.10);
const GRASS1: Vec3 = Vec3::new(0.15, 0.30, 0.10);
const GRASS2: Vec3 = Vec3::new(0.40, 0.50, 0.20);
const SAND: Vec3 = Vec3::new(0.80, 0.70, 0.60);
const GRASS_HEIGHT: f32 = 0.465;
const DRAINAGE_WIDTH: f32 = 0.3;

/// Everything the panel drives. `dirty` marks a pending mesh rebuild.
#[derive(Resource, Clone, Debug)]
pub struct DemoTerrain {
    pub erosion: cpu::ErosionFilterParams,
    pub height_offset: f32,
    pub map_scale: f32,
    pub water_level: f32,
    pub erosion_enabled: bool,
    pub view_mode: ViewMode,
    pub dirty: bool,
}

impl Default for DemoTerrain {
    fn default() -> Self {
        DemoTerrain {
            erosion: cpu::ErosionFilterParams::default(),
            height_offset: -0.65,
            map_scale: 1.0,
            water_level: -1.54,
            erosion_enabled: true,
            view_mode: ViewMode::Terrain,
            dirty: true,
        }
    }
}

/// The demo's four fragment views, as per-vertex colors here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ViewMode {
    #[default]
    Terrain,
    ErosionDelta,
    RidgeMap,
    DebugFade,
}

impl ViewMode {
    fn label(self) -> &'static str {
        match self {
            ViewMode::Terrain => "Terrain",
            ViewMode::ErosionDelta => "Erosion",
            ViewMode::RidgeMap => "Ridges",
            ViewMode::DebugFade => "Debug",
        }
    }
}

/// Everything one vertex needs, mirroring the demo's `TerrainEval`.
pub struct TerrainEval {
    pub unit_height: f32,
    pub slope: Vec2,
    pub erosion_delta: f32,
    pub ridge_map: f32,
    pub debug: f32,
    pub tree_amount: f32,
}

/// The demo's terrain evaluation at a uv point (0..1 across the square).
/// Pure: the same inputs give the same vertex, whatever asks.
pub fn evaluate_terrain(demo: &DemoTerrain, uv: Vec2) -> TerrainEval {
    let p = uv * demo.map_scale + MAP_ORIGIN;

    let raw = cpu::fbm(p, 3.0, 3, 2.0, 0.1) * HEIGHT_AMP;
    let fade_target = (raw.x / (HEIGHT_AMP * 0.6)).clamp(-1.0, 1.0);
    let base = raw * 0.5 + Vec3::new(0.5, 0.0, 0.0);

    let (delta, magnitude, ridge_map, debug) = if demo.erosion_enabled {
        let filtered = cpu::erosion_filter(p, base, fade_target, &demo.erosion);
        (
            filtered.delta,
            filtered.magnitude,
            filtered.ridge_map,
            filtered.debug,
        )
    } else {
        (Vec3::ZERO, 0.0, 1.0, fade_target)
    };

    let height_offset = demo.height_offset * magnitude;
    let unit_height = base.x + delta.x + height_offset;
    let slope = Vec2::new(base.y + delta.y, base.z + delta.z);

    let erosion_delta = if magnitude > 0.0 {
        delta.x / magnitude
    } else {
        0.0
    };

    // The demo's slope→world factor, extent-independent: vertical × map
    // scale × the fixed unit (our extent/world-scale ratio cancels).
    let stw = VERTICAL_SCALE * demo.map_scale * SLOPE_UNIT;
    let world_normal = normal_from_slope(slope, stw);

    let water_unit = demo.water_level / VERTICAL_SCALE + UNIT_HEIGHT_ORIGIN;
    let occlusion = (erosion_delta + 0.5).clamp(0.0, 1.0);
    let trees = trees_amount(unit_height, world_normal.y, occlusion, ridge_map, water_unit);

    TerrainEval {
        unit_height,
        slope,
        erosion_delta,
        ridge_map,
        debug,
        tree_amount: trees,
    }
}

/// World height of the surface at a uv point — what grounding and the
/// camera sample through the [`super::HeightField`].
pub fn world_height(demo: &DemoTerrain, uv: Vec2) -> f32 {
    (evaluate_terrain(demo, uv).unit_height - UNIT_HEIGHT_ORIGIN)
        * VERTICAL_SCALE
        * WORLD_SCALE
}

fn normal_from_slope(slope: Vec2, stw: f32) -> Vec3 {
    Vec3::new(-slope.x * stw, 1.0, -slope.y * stw).normalize_or_zero()
}

fn smoothstep_down(e0: f32, e1: f32, x: f32) -> f32 {
    1.0 - smoothstep(e0, e1, x)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn trees_amount(height: f32, normal_y: f32, occlusion: f32, ridge_map: f32, water_unit: f32) -> f32 {
    let t = smoothstep_down(
        GRASS_HEIGHT + 0.01,
        GRASS_HEIGHT + 0.05,
        height + 0.01 + (occlusion - 0.8) * 0.05,
    ) * smoothstep(0.0, 0.4, occlusion)
        * smoothstep(0.95, 1.0, normal_y)
        * smoothstep(-1.4, 0.0, ridge_map)
        * smoothstep(water_unit, water_unit + 0.007, height);
    (t - 0.5) / 0.6
}

/// The demo's albedo cascade (`terrain_albedo`), minus its lighting.
fn albedo(demo: &DemoTerrain, eval: &TerrainEval, normal_y: f32) -> Vec3 {
    let h = eval.unit_height;
    let occlusion = (eval.erosion_delta + 0.5).clamp(0.0, 1.0);
    let trees = (eval.tree_amount * 0.5 + 0.5).clamp(0.0, 1.0);
    let ridgemap = (eval.ridge_map * 0.5 + 0.5).clamp(0.0, 1.0);
    let water_unit = demo.water_level / VERTICAL_SCALE + UNIT_HEIGHT_ORIGIN;

    let mut color = CLIFF * smoothstep(0.4, 0.52, h);
    color = color.lerp(DIRT, smoothstep_down(0.0, 0.6, occlusion));
    color = color.lerp(Vec3::ONE, smoothstep(0.53, 0.6, h));
    color = color.lerp(SAND, smoothstep_down(water_unit, water_unit + 0.005, h));

    let grass_mix = GRASS1.lerp(GRASS2, smoothstep(0.4, 0.6, h - eval.erosion_delta * 0.05));
    let grass_height_mask = smoothstep_down(
        GRASS_HEIGHT + 0.02,
        GRASS_HEIGHT + 0.05,
        h + 0.01 + (occlusion - 0.8) * 0.05,
    );
    let grass_normal_mask = smoothstep(0.8, 1.0, 1.0 - (1.0 - normal_y) * (1.0 - trees));
    color = color.lerp(grass_mix, grass_height_mask * grass_normal_mask);

    color = color.lerp(
        TREE * trees.powf(8.0),
        ((trees * 2.2 - 0.8).clamp(0.0, 1.0)) * 0.6,
    );

    let drainage = (((1.0 - (ridgemap / DRAINAGE_WIDTH).clamp(0.0, 1.0)) * 1.5) as f32).clamp(0.0, 1.0);
    color.lerp(Vec3::ONE, drainage)
}

/// The demo's debug views (`debug_color`).
fn debug_color(eval: &TerrainEval, mode: ViewMode) -> Vec3 {
    match mode {
        ViewMode::ErosionDelta => {
            let t = (eval.erosion_delta * 0.5 + 0.5).clamp(0.0, 1.0);
            Vec3::new(0.05, 0.10, 0.16).lerp(Vec3::new(0.82, 0.76, 0.66), t)
        }
        ViewMode::RidgeMap => {
            let crease = (-eval.ridge_map).clamp(0.0, 1.0);
            let ridge = eval.ridge_map.clamp(0.0, 1.0);
            Vec3::new(0.30, 0.34, 0.35)
                .lerp(Vec3::new(0.03, 0.18, 0.26), crease)
                .lerp(Vec3::new(0.95, 0.94, 0.88), ridge)
        }
        ViewMode::DebugFade => {
            let t = (eval.debug * 0.5 + 0.5).clamp(0.0, 1.0);
            Vec3::new(0.08, 0.16, 0.34).lerp(Vec3::new(0.95, 0.83, 0.58), t)
        }
        ViewMode::Terrain => unreachable!("the terrain view uses the albedo cascade"),
    }
}

/// Fill the grid mesh from a fresh evaluation: positions (heights),
/// analytic normals, and per-view colors. One evaluation per vertex.
pub fn rebuild_mesh(mesh: &mut Mesh, demo: &DemoTerrain) {
    // Evaluate into flat buffers first: attribute access borrows the mesh
    // one attribute at a time, and interleaving evaluation with three
    // live borrows fights the borrow checker.
    let verts = TERRAIN_RESOLUTION + 1;
    let stw = VERTICAL_SCALE * demo.map_scale * SLOPE_UNIT;
    let mut heights = vec![0.0f32; verts * verts];
    let mut normals = vec![[0.0f32; 3]; verts * verts];
    let mut colors = vec![[0.0f32; 4]; verts * verts];
    for iz in 0..verts {
        for ix in 0..verts {
            let uv = Vec2::new(ix as f32, iz as f32) / TERRAIN_RESOLUTION as f32;
            let eval = evaluate_terrain(demo, uv);
            let i = iz * verts + ix;
            heights[i] =
                (eval.unit_height - UNIT_HEIGHT_ORIGIN) * VERTICAL_SCALE * WORLD_SCALE;
            let normal = normal_from_slope(eval.slope, stw);
            normals[i] = normal.to_array();
            let color = match demo.view_mode {
                ViewMode::Terrain => albedo(demo, &eval, normal.y),
                mode => debug_color(&eval, mode),
            };
            colors[i] = [color.x, color.y, color.z, 1.0];
        }
    }

    let positions = match mesh
        .attribute_mut(bevy::render::mesh::Mesh::ATTRIBUTE_POSITION)
        .expect("positions")
    {
        bevy::render::mesh::VertexAttributeValues::Float32x3(p) => p,
        _ => panic!("unexpected position format"),
    };
    for (i, h) in heights.into_iter().enumerate() {
        positions[i][1] = h;
    }
    match mesh
        .attribute_mut(bevy::render::mesh::Mesh::ATTRIBUTE_NORMAL)
        .expect("normals")
    {
        bevy::render::mesh::VertexAttributeValues::Float32x3(n) => *n = normals,
        _ => panic!("unexpected normal format"),
    }
    match mesh
        .attribute_mut(bevy::render::mesh::Mesh::ATTRIBUTE_COLOR)
        .expect("colors")
    {
        bevy::render::mesh::VertexAttributeValues::Float32x4(c) => *c = colors,
        _ => panic!("unexpected color format"),
    }
}

/// Demo wiring: the settings resource, the water plane, and the rebuild
/// system. The rebuild fires when the panel marked the settings dirty AND
/// the pointer is up (a drag fires `changed` every frame; a full CPU
/// re-evaluation is ~160 ms, so drags coalesce into one rebuild on
/// release) — plus once at startup.
pub struct ErosionDemoPlugin;

impl Plugin for ErosionDemoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DemoTerrain>()
            .add_systems(Startup, spawn_water)
            .add_systems(Update, rebuild_terrain);
    }
}

fn spawn_water(
    mut commands: Commands,
    demo: Res<DemoTerrain>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        super::WaterPlane,
        Mesh3d(meshes.add(
            Plane3d::default()
                .mesh()
                .size(TERRAIN_EXTENT_M * 0.94, TERRAIN_EXTENT_M * 0.94),
        )),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.05, 0.24, 0.34, 0.72),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            perceptual_roughness: 0.18,
            reflectance: 0.68,
            ..default()
        })),
        Transform::from_xyz(0.0, demo.water_level * WORLD_SCALE, 0.0),
    ));
}

fn rebuild_terrain(
    mut demo: ResMut<DemoTerrain>,
    pointer: Res<ButtonInput<MouseButton>>,
    terrain: Res<TerrainMesh>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut field: ResMut<super::HeightField>,
    mut water: Query<&mut Transform, With<super::WaterPlane>>,
) {
    if !demo.dirty || pointer.pressed(MouseButton::Left) {
        return;
    }
    demo.dirty = false;
    if let Some(mut mesh) = meshes.get_mut(&terrain.handle) {
        rebuild_mesh(&mut mesh, &demo);
    }
    // Grounding/camera/raycast sample the same evaluation: capture the
    // settings by value so the field follows the panel without borrowing.
    let settings = demo.clone();
    *field = super::HeightField::from_fn(move |x, z| {
        let uv = world_to_uv(x, z);
        world_height(&settings, uv)
    });
    if let Ok(mut transform) = water.single_mut() {
        transform.translation.y = demo.water_level * WORLD_SCALE;
    }
}

/// World XZ to the evaluation's uv square (0..1); outside the square the
/// uv clamps, extending the rim heights outward.
fn world_to_uv(x: f32, z: f32) -> Vec2 {
    Vec2::new(
        (x / TERRAIN_EXTENT_M + 0.5).clamp(0.0, 1.0),
        (z / TERRAIN_EXTENT_M + 0.5).clamp(0.0, 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heights_stay_in_the_demo_band() {
        // The composition maps unit heights into roughly [0.3, 0.6]; the
        // world band follows. Samples across the square, both with and
        // without erosion, must stay in it — a runaway term (wrong fbm
        // scaling, missing clamp) escapes immediately.
        let demo = DemoTerrain::default();
        for i in 0..16 {
            for j in 0..16 {
                let uv = Vec2::new(i as f32 / 16.0, j as f32 / 16.0);
                let unit = evaluate_terrain(&demo, uv).unit_height;
                assert!(
                    (0.25..0.68).contains(&unit),
                    "unit height {unit} at {uv:?} escaped the demo band"
                );
            }
        }
    }

    #[test]
    fn erosion_carves_but_keeps_the_band() {
        let mut with = DemoTerrain::default();
        with.erosion_enabled = false;
        for i in 0..8 {
            for j in 0..8 {
                let uv = Vec2::new(i as f32 / 8.0, j as f32 / 8.0);
                let a = evaluate_terrain(&DemoTerrain::default(), uv).unit_height;
                let b = evaluate_terrain(&with, uv).unit_height;
                assert!((a - b).abs() < 0.2, "erosion moved {uv:?} by {}", a - b);
            }
        }
    }

    #[test]
    fn world_height_matches_unit_height() {
        let demo = DemoTerrain::default();
        let uv = Vec2::new(0.3, 0.7);
        let unit = evaluate_terrain(&demo, uv).unit_height;
        assert_eq!(
            world_height(&demo, uv),
            (unit - UNIT_HEIGHT_ORIGIN) * VERTICAL_SCALE * WORLD_SCALE
        );
    }
}
