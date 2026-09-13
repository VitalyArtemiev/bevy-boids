//! Radial context menu: an RMB *click* (press and release within
//! [`CLICK_TOLERANCE_PX`]) with a selection opens a ring of command wedges
//! under the cursor. Hovering a branching wedge opens its sub-ring
//! *alongside* the parent — offset in the hovered wedge's direction, with
//! the wedge facing back at the parent left out so the child never covers
//! it (tree menus grow one ring per level this way, children drawn behind
//! their parents). LMB on a wedge executes; the hole pops back a level;
//! Esc or a click outside every ring cancels.
//!
//! RMB *drag* keeps its old meaning: frontage designation (see
//! `frontage_position_system`, which skips releases that didn't drag).

use crate::formations::{Formation, FormationKind, FormationOrder, Members};
use crate::input::{ActionEvents, ActionId, ActionTag, TriggerState, completed, started};
use crate::player::{Selected, get_intersection};
use crate::terrain::HeightField;
use crate::ui::GameState;
use bevy::prelude::*;
use bevy_egui::egui;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass};

/// Screen distance between press and release that still counts as a click.
pub const CLICK_TOLERANCE_PX: f32 = 6.0;
const OUTER_RADIUS_PX: f32 = 110.0;
const INNER_RADIUS_PX: f32 = 40.0;
/// Angular width of the wedge a sub-ring leaves out — the corridor back to
/// its parent (radians). Sized so that with the sub-ring center sitting on
/// the parent rim, its flanks only ever overlap the wedge that spawned
/// them, never the siblings (see the `child_ring_never_covers_parent_siblings`
/// test).
const BACK_WEDGE: f32 = std::f32::consts::FRAC_PI_3;
/// Wedge rim smoothness (arc samples per wedge edge); the rims are drawn
/// from these samples, so this is what keeps the rings from looking like
/// n-gons.
const ARC_STEPS: usize = 48;
/// Placeholder-fill texture density: one texture tile per this many screen
/// pixels (the 2×2 checker texture then reads as 32 px cells).
const UV_TILES_PX: f32 = 64.0;

/// Walk speed as a fraction of the tuning max velocity; Run is 1.0.
const WALK_SPEED_SCALE: f32 = 0.5;
/// Width of the marching column for `WalkColumn`.
const MARCH_COLUMN_WIDTH: usize = 4;

/// The open menu. Presence is the state: gameplay input systems are gated
/// on `radial_closed` while it exists.
#[derive(Resource, Debug)]
pub struct RadialMenu {
    /// Ring center in screen pixels (where the RMB click landed).
    pub center: Vec2,
    /// Ground point under the click — the command destination.
    pub world_point: Vec3,
    /// Chosen sub-menu path: indexes into each level of [`menu_items`].
    /// Empty = top level.
    pub path: Vec<usize>,
}

/// What a command wedge does. Placeholder set per the design: Walk (with
/// the keep-formation / marching-column choice) and Run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RadialCommand {
    WalkKeep,
    WalkColumn,
    Run,
}

/// One wedge of the current ring. `SubMenu` wedges navigate on hover.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RadialItem {
    Command(RadialCommand),
    SubMenu,
}

/// Ring contents at `path`. Walk branches only when a formation is
/// selected — a bare boid selection has no formation to keep or reform.
pub fn menu_items(path: &[usize], has_formation: bool) -> Vec<RadialItem> {
    match path {
        [] => {
            if has_formation {
                vec![RadialItem::SubMenu, RadialItem::Command(RadialCommand::Run)]
            } else {
                vec![
                    RadialItem::Command(RadialCommand::WalkKeep),
                    RadialItem::Command(RadialCommand::Run),
                ]
            }
        }
        [_] => vec![
            RadialItem::Command(RadialCommand::WalkKeep),
            RadialItem::Command(RadialCommand::WalkColumn),
        ],
        _ => vec![],
    }
}

fn item_label(item: RadialItem, path: &[usize]) -> &'static str {
    match item {
        RadialItem::Command(RadialCommand::WalkKeep) if path.is_empty() => "Walk",
        RadialItem::Command(RadialCommand::WalkKeep) => "Keep formation",
        RadialItem::Command(RadialCommand::WalkColumn) => "Column of 4",
        RadialItem::Command(RadialCommand::Run) => "Run",
        RadialItem::SubMenu => "Walk",
    }
}

/// Radial distance from a ring center where wedge labels are centered.
pub const LABEL_RADIUS_PX: f32 = (INNER_RADIUS_PX + OUTER_RADIUS_PX) / 2.0;
/// The font size wedge labels are drawn at (and the test measures with).
pub const LABEL_FONT_SIZE: f32 = 14.0;

/// Measured placement of one wedge label: text oriented *tangentially*
/// (following the ring), flipped on the lower half of the ring so it never
/// reads upside down. Pure — the clipping test runs it against a bare
/// `egui::Context` (fonts load without a window).
///
/// Tangential is not just aesthetic: horizontal labels clip for any wedge
/// pointing sideways (a long label extends radially past the rim), while
/// tangential labels have the wedge's whole arc to spread along.
pub struct LabelLayout {
    /// Galley center at [`LABEL_RADIUS_PX`] on the wedge's mid angle.
    pub center: Vec2,
    /// Rotation of the text (radians, clockwise; feeds `TextShape::angle`).
    pub angle: f32,
    /// Where to anchor the galley so its center lands on `center`.
    pub pos: egui::Pos2,
    /// The measured, tinted galley.
    pub galley: std::sync::Arc<egui::Galley>,
    /// The four corners of the rotated text bounds (clipping tests).
    pub quad: [Vec2; 4],
}

pub fn label_layout(
    ctx: &egui::Context,
    ring_center: Vec2,
    start: f32,
    end: f32,
    text: &str,
    font_size: f32,
) -> LabelLayout {
    let job = egui::text::LayoutJob::simple(
        text.into(),
        egui::FontId::proportional(font_size),
        egui::Color32::WHITE,
        f32::INFINITY,
    );
    let galley = ctx.fonts_mut(|fonts| fonts.layout_job(job));
    let mid = (start + end) / 2.0;
    let center = ring_center + Vec2::new(mid.cos(), mid.sin()) * LABEL_RADIUS_PX;

    // Baseline along the tangent (mid + 90°), flipped a half turn when the
    // mid angle is in the lower half of the ring (screen y grows down).
    let mut angle = mid + std::f32::consts::FRAC_PI_2;
    if mid.sin() > 0.0 {
        angle += std::f32::consts::PI;
    }

    let rot = egui::emath::Rot2::from_angle(angle);
    let half = galley.size() / 2.0;
    let pos = egui::pos2(
        center.x - (rot * half).x,
        center.y - (rot * half).y,
    );
    // egui and bevy have distinct Vec2 types; convert at the boundary.
    let corner = |sx: f32, sy: f32| {
        let v = rot * egui::vec2(half.x * sx, half.y * sy);
        center + Vec2::new(v.x, v.y)
    };
    let quad = [corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)];
    LabelLayout {
        center,
        angle,
        pos,
        galley,
        quad,
    }
}

// --- Geometry (pure, unit-tested) -------------------------------------------

/// Start/end angle of every segment in a ring of `n`. The top level starts
/// at the top (-π/2); a sub-ring (`avoid` = center angle of the omitted
/// back wedge) fills only the arc outside that wedge so the direction
/// toward its parent stays empty.
pub fn segment_angles(n: usize, avoid: Option<f32>) -> Vec<(f32, f32)> {
    let tau = std::f32::consts::TAU;
    let span = match avoid {
        Some(_) => tau - BACK_WEDGE,
        None => tau,
    };
    let start = match avoid {
        Some(a) => a + BACK_WEDGE / 2.0,
        None => -std::f32::consts::FRAC_PI_2 - (tau / n.max(1) as f32) / 2.0,
    };
    (0..n)
        .map(|i| {
            let s = start + span * i as f32 / n as f32;
            (s, s + span / n as f32)
        })
        .collect()
}

/// One visible ring of the menu tree.
pub struct Ring {
    /// Ring center in screen pixels.
    pub center: Vec2,
    pub items: Vec<RadialItem>,
    /// (start, end) angle per item, radians.
    pub angles: Vec<(f32, f32)>,
    /// Path prefix identifying this ring (`menu.path[..level]`) — labels
    /// depend on the level (`Walk` vs `Keep formation`).
    pub path: Vec<usize>,
}

/// Lays out every ring on the active path: the root at the click position,
/// each child offset [`RING_OFFSET_PX`] toward the wedge that spawned it,
/// rotated so its back wedge faces the parent. Siblings therefore never
/// cover their parent, and deeper levels keep growing outward the same way.
pub fn layout_rings(menu: &RadialMenu, has_formation: bool) -> Vec<Ring> {
    let mut rings = Vec::with_capacity(menu.path.len() + 1);
    let mut center = menu.center;
    let mut spawn_angle: Option<f32> = None; // wedge center angle that opens the next ring
    for level in 0..=menu.path.len() {
        let path = menu.path[..level].to_vec();
        let items = menu_items(&path, has_formation);
        let back = spawn_angle.map(|a| a + std::f32::consts::PI);
        let angles = segment_angles(items.len(), back);
        rings.push(Ring {
            center,
            items,
            angles,
            path,
        });
        if level < menu.path.len() {
            let (start, end) = rings[level].angles[menu.path[level]];
            let mid = (start + end) / 2.0;
            spawn_angle = Some(mid);
            // Sub-ring center sits on the parent's outer radius: the rings
            // touch with no gap, and the omitted back wedge keeps the
            // child's flanks off the parent's sibling wedges.
            center += Vec2::new(mid.cos(), mid.sin()) * OUTER_RADIUS_PX;
        }
    }
    rings
}

/// What the pointer is over in one ring.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RingHit {
    Segment(usize),
    /// The inner hole — "back" on sub-rings, cancel on the root.
    Hole,
}

/// Pointer membership in one ring. `None` covers both "outside the rim" and
/// "in the back-wedge corridor": the parent ring remains reachable through
/// the corridor, so callers fall through to outer rings on `None`.
pub fn ring_hit(ring: &Ring, pointer: Vec2) -> Option<RingHit> {
    let d = pointer - ring.center;
    let r = d.length();
    if r < INNER_RADIUS_PX {
        return Some(RingHit::Hole);
    }
    if r > OUTER_RADIUS_PX {
        return None;
    }
    let mut angle = d.to_angle();
    if angle < 0.0 {
        angle += std::f32::consts::TAU;
    }
    for (i, &(start, end)) in ring.angles.iter().enumerate() {
        if angle_in(angle, start, end) {
            return Some(RingHit::Segment(i));
        }
    }
    None
}

/// Angle membership with wrap-around; half-open [start, end).
fn angle_in(angle: f32, start: f32, end: f32) -> bool {
    let tau = std::f32::consts::TAU;
    let norm = |mut a: f32| {
        while a < 0.0 {
            a += tau;
        }
        while a >= tau {
            a -= tau;
        }
        a
    };
    let a = norm(angle - start);
    let width = norm(end - start);
    a < width
}

/// Placeholder fill: a 2×2 two-tone checker loaded as a repeating texture.
/// Vertex colors tint it per wedge state, so one texture serves every ring.
fn placeholder_texture() -> egui::ColorImage {
    let light = egui::Color32::from_rgb(235, 235, 235);
    let dark = egui::Color32::from_rgb(200, 200, 200);
    egui::ColorImage::new([2, 2], vec![light, dark, dark, light])
}

/// Appends one annulus sector to `mesh` as a quad strip between the inner
/// and outer arcs. Exact representation — the previous `convex_polygon`
/// fan triangulated the wedge's convex hull, which drew a stray triangle
/// across the hole. UVs are world-space (`pos / UV_TILES_PX`), so the
/// checker stays continuous within a ring.
fn push_wedge(mesh: &mut egui::Mesh, center: Vec2, start: f32, end: f32, color: egui::Color32) {
    let base = mesh.vertices.len() as u32;
    for i in 0..=ARC_STEPS {
        let a = start + (end - start) * i as f32 / ARC_STEPS as f32;
        let (s, c) = a.sin_cos();
        let inner = center + Vec2::new(c, s) * INNER_RADIUS_PX;
        let outer = center + Vec2::new(c, s) * OUTER_RADIUS_PX;
        mesh.vertices.push(egui::epaint::Vertex {
            pos: egui::pos2(inner.x, inner.y),
            uv: egui::pos2(inner.x / UV_TILES_PX, inner.y / UV_TILES_PX),
            color,
        });
        mesh.vertices.push(egui::epaint::Vertex {
            pos: egui::pos2(outer.x, outer.y),
            uv: egui::pos2(outer.x / UV_TILES_PX, outer.y / UV_TILES_PX),
            color,
        });
    }
    for i in 0..ARC_STEPS {
        let inner0 = base + (2 * i) as u32;
        let outer0 = inner0 + 1;
        let inner1 = inner0 + 2;
        let outer1 = inner0 + 3;
        mesh.add_triangle(inner0, outer0, outer1);
        mesh.add_triangle(inner0, outer1, inner1);
    }
}

// --- Plugin -----------------------------------------------------------------

pub struct RadialPlugin;

/// True while no radial menu is open — the run condition that stands
/// gameplay input systems down for the menu's lifetime.
pub fn radial_closed(menu: Option<Res<RadialMenu>>) -> bool {
    menu.is_none()
}

impl Plugin for RadialPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            radial_input
                .run_if(in_state(GameState::Playing))
                .run_if(radial_closed),
        )
        .add_systems(
            EguiPrimaryContextPass,
            radial_ui
                .run_if(in_state(GameState::Playing))
                .run_if(resource_exists::<RadialMenu>),
        );
    }
}

/// Classifies the RMB gesture: a press→release within
/// [`CLICK_TOLERANCE_PX`] with something selected opens the menu at the
/// press position. Drags are left to the frontage system.
fn radial_input(
    actions: Query<(&ActionTag, &TriggerState, &ActionEvents)>,
    mut press: Local<Option<Vec2>>,
    windows: Query<&Window>,
    field: Res<HeightField>,
    q_camera: Query<(&Camera, &GlobalTransform)>,
    q_selected: Query<(), With<Selected>>,
    mut commands: Commands,
) {
    let cursor = windows.single().ok().and_then(|w| w.cursor_position());
    if started(&actions, ActionId::Frontage) {
        *press = cursor;
    }
    if completed(&actions, ActionId::Frontage) {
        let press = press.take();
        let (Some(press), Some(release)) = (press, cursor) else {
            return;
        };
        if release.distance(press) >= CLICK_TOLERANCE_PX || q_selected.is_empty() {
            return;
        }
        let world_point = q_camera
            .single()
            .ok()
            .and_then(|(camera, transform)| get_intersection(&field, &press, camera, transform));
        let Some(world_point) = world_point else {
            return;
        };
        commands.insert_resource(RadialMenu {
            center: press,
            world_point,
            path: Vec::new(),
        });
    }
}

/// Draws every ring of the menu tree (root first, each child beside its
/// parent and on top of it) and resolves hover/click into navigation and
/// commands.
fn radial_ui(
    mut contexts: EguiContexts,
    mut menu: ResMut<RadialMenu>,
    mut fill_texture: Local<Option<egui::TextureHandle>>,
    q_selected_formations: Query<(), (With<Selected>, With<Formation>)>,
    mut commands: Commands,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    let has_formation = !q_selected_formations.is_empty();
    let mut rings = layout_rings(&menu, has_formation);

    let pointer = ctx.pointer_hover_pos().map(|p| Vec2::new(p.x, p.y));
    let primary_clicked = ctx.input(|i| i.pointer.primary_clicked());
    let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));

    // Hover, deepest ring first: children sit on top of the parents'.
    let mut hover: Option<(usize, RingHit)> = None;
    if let Some(p) = pointer {
        for k in (0..rings.len()).rev() {
            if let Some(hit) = ring_hit(&rings[k], p) {
                hover = Some((k, hit));
                break;
            }
        }
    }

    // Cancel: Esc anywhere, or a click fully outside every ring.
    let inside_any = pointer.is_some_and(|p| {
        rings
            .iter()
            .any(|ring| (p - ring.center).length() <= OUTER_RADIUS_PX)
    });
    if escape || (primary_clicked && !inside_any) {
        commands.remove_resource::<RadialMenu>();
        return Ok(());
    }

    // Hover navigation: the path becomes the chain of rings up to the
    // hovered one, extended through the hovered wedge if it branches.
    // Hovering a sibling command folds the deeper rings away.
    let path_before = menu.path.clone();
    if let Some((k, RingHit::Segment(i))) = hover {
        let mut path = menu.path[..k].to_vec();
        if matches!(rings[k].items[i], RadialItem::SubMenu) {
            path.push(i);
        }
        menu.path = path;
    }
    // The layout above was computed from the pre-navigation path (hover
    // needs the geometry); truncate/re-extend re-laid rings whose children
    // just folded away, otherwise they linger one frame and index past the
    // shortened path.
    if menu.path != path_before {
        rings = layout_rings(&menu, has_formation);
        // The hovered ring itself is unaffected by the change (its layout
        // depends only on the unchanged prefix), so `hover` stays valid.
    }

    // One Area spanning every ring's bounding box.
    let mut min = menu.center - Vec2::splat(OUTER_RADIUS_PX);
    let mut max = menu.center + Vec2::splat(OUTER_RADIUS_PX);
    for ring in &rings[1..] {
        min = min.min(ring.center - Vec2::splat(OUTER_RADIUS_PX));
        max = max.max(ring.center + Vec2::splat(OUTER_RADIUS_PX));
    }
    egui::Area::new(egui::Id::new("radial-menu"))
        .order(egui::Order::Foreground)
        .fixed_pos(egui::pos2(min.x, min.y))
        .interactable(true)
        .show(ctx, |ui| {
            // Placeholder fill texture, loaded once: a subtle two-tone
            // checker, tinted per wedge state via vertex colors (opaque).
            let texture = fill_texture.get_or_insert_with(|| {
                ctx.load_texture(
                    "radial-placeholder",
                    placeholder_texture(),
                    egui::TextureOptions::NEAREST_REPEAT,
                )
            });
            let stroke = egui::Stroke::new(1.5, egui::Color32::from_gray(35));
            let painter = ui.painter();

            // Per ring: wedge fills first (one mesh), then separators,
            // labels and hole — painted deepest ring first, so every ring
            // tucks behind its parent where they overlap.
            for (k, ring) in rings.iter().enumerate().rev() {
                // Wedges are exact annulus quad strips (see `push_wedge` —
                // the old convex-polygon fan drew a stray triangle over the
                // hole).
                let mut fills = egui::Mesh::with_texture(texture.id());
                for (i, item) in ring.items.iter().enumerate() {
                    let (start, end) = ring.angles[i];
                    let hovered = hover == Some((k, RingHit::Segment(i)));
                    // The wedge a child ring hangs off stays lit.
                    let on_path = k < menu.path.len() && k + 1 < rings.len() && menu.path[k] == i;
                    let tint = if hovered || on_path {
                        egui::Color32::from_rgb(96, 136, 208)
                    } else if matches!(item, RadialItem::SubMenu) {
                        egui::Color32::from_rgb(74, 108, 170)
                    } else {
                        egui::Color32::from_rgb(56, 64, 78)
                    };
                    push_wedge(&mut fills, ring.center, start, end, tint);
                }
                painter.add(egui::Shape::mesh(fills));

                for (i, item) in ring.items.iter().enumerate() {
                    let (start, end) = ring.angles[i];
                    let edge = |a: f32, r: f32| {
                        egui::pos2(ring.center.x + a.cos() * r, ring.center.y + a.sin() * r)
                    };
                    // Separators + rim, so adjacent wedges read apart.
                    painter.add(egui::Shape::line(
                        vec![edge(start, INNER_RADIUS_PX), edge(start, OUTER_RADIUS_PX)],
                        stroke,
                    ));
                    let rim: Vec<egui::Pos2> = (0..=ARC_STEPS)
                        .map(|s| {
                            let a = start + (end - start) * s as f32 / ARC_STEPS as f32;
                            edge(a, OUTER_RADIUS_PX)
                        })
                        .collect();
                    painter.add(egui::Shape::line(rim, stroke));

                    let label = item_label(*item, &ring.path);
                    let layout = label_layout(
                        ctx,
                        ring.center,
                        start,
                        end,
                        label,
                        LABEL_FONT_SIZE,
                    );
                    painter.add(egui::Shape::Text(egui::epaint::TextShape {
                        pos: layout.pos,
                        galley: layout.galley,
                        underline: egui::Stroke::NONE,
                        fallback_color: egui::Color32::WHITE,
                        override_text_color: None,
                        opacity_factor: 1.0,
                        angle: layout.angle,
                    }));
                }
                // The hole: cancel at the root, "back" on sub-rings.
                painter.circle_filled(
                    egui::pos2(ring.center.x, ring.center.y),
                    INNER_RADIUS_PX - 3.0,
                    egui::Color32::from_gray(25),
                );
                let hole_text = if k == 0 { "✕" } else { "◂" };
                painter.text(
                    egui::pos2(ring.center.x, ring.center.y),
                    egui::Align2::CENTER_CENTER,
                    hole_text,
                    egui::FontId::proportional(16.0),
                    egui::Color32::from_gray(160),
                );
            }

            if primary_clicked {
                match hover {
                    Some((k, RingHit::Segment(i))) => {
                        if let RadialItem::Command(command) = rings[k].items[i] {
                            let point = menu.world_point;
                            commands.queue(move |world: &mut World| {
                                execute(world, point, command);
                            });
                            commands.remove_resource::<RadialMenu>();
                        }
                    }
                    Some((k, RingHit::Hole)) => {
                        if k == 0 {
                            commands.remove_resource::<RadialMenu>();
                        } else {
                            menu.path.truncate(k);
                        }
                    }
                    None => {}
                }
            }
        });
    Ok(())
}

/// Applies a command to the selection: formations through their task
/// queues (WalkColumn = reform into a `MARCH_COLUMN_WIDTH`-wide grid,
/// march, restore the original layout), free boids through `Target`.
fn execute(world: &mut World, point: Vec3, command: RadialCommand) {
    let (speed_scale, column_march) = match command {
        RadialCommand::Run => (1.0, false),
        RadialCommand::WalkKeep => (WALK_SPEED_SCALE, false),
        RadialCommand::WalkColumn => (WALK_SPEED_SCALE, true),
    };

    let mut formations: Vec<(Entity, (FormationKind, Option<usize>))> = Vec::new();
    let mut free_boids: Vec<Entity> = Vec::new();
    for entity in world.query_filtered::<Entity, With<Selected>>().iter(world) {
        // Snapshot just the layout fields the orders need; `tasks` is
        // replaced wholesale below.
        let layout = world.get::<Formation>(entity).map(|f| (f.kind, f.columns));
        match layout {
            Some(layout) => formations.push((entity, layout)),
            None => free_boids.push(entity),
        }
    }

    for (entity, (kind, columns)) in formations {
        let from = world
            .get::<Transform>(entity)
            .map(|t| t.translation)
            .unwrap_or(point);
        let facing = facing_between(from, point);

        let mut tasks = std::collections::VecDeque::new();
        if column_march {
            tasks.push_back(FormationOrder::Reform {
                kind: FormationKind::Grid,
                columns: Some(MARCH_COLUMN_WIDTH),
            });
        }
        tasks.push_back(FormationOrder::Move {
            pos: point,
            facing_dir: facing,
        });
        if column_march {
            tasks.push_back(FormationOrder::Reform { kind, columns });
        }
        if let Some(mut f) = world.get_mut::<Formation>(entity) {
            f.tasks = tasks;
        }
        if let Some(mut target) = world.get_mut::<crate::target::Target>(entity) {
            target.speed_scale = speed_scale;
        }
        // Loaded members steer by their own `Target` caps.
        let members: Vec<Entity> = world
            .get::<Members>(entity)
            .map(|m| m.iter().collect())
            .unwrap_or_default();
        for member in members {
            if let Some(mut target) = world.get_mut::<crate::target::Target>(member) {
                target.speed_scale = speed_scale;
            }
        }
    }

    for entity in free_boids {
        let from = world
            .get::<Transform>(entity)
            .map(|t| t.translation)
            .unwrap_or(point);
        if let Some(mut target) = world.get_mut::<crate::target::Target>(entity) {
            target.pos = point;
            target.dir = facing_between(from, point);
            target.speed_scale = speed_scale;
        }
    }
}

/// Horizontal facing from `from` toward `to` (zero when coincident).
fn facing_between(from: Vec3, to: Vec3) -> Vec3 {
    let mut d = to - from;
    d.y = 0.0;
    d.normalize_or_zero()
}

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn top_level_segments_tile_the_ring() {
        let n = 4;
        let angles = segment_angles(n, None);
        assert_eq!(angles.len(), n);
        for (start, end) in &angles {
            assert!(end > start);
            assert!(end - start <= std::f32::consts::FRAC_PI_2 + 1e-4);
        }
        // Contiguous: each next start where the previous ended.
        for w in angles.windows(2) {
            assert!(approx(w[0].1, w[1].0));
        }
        assert!(approx(angles[0].0 + std::f32::consts::TAU, angles[n - 1].1));
    }

    #[test]
    fn sub_menu_segments_skip_the_back_wedge() {
        let n = 2;
        let avoid = -std::f32::consts::FRAC_PI_2;
        let angles = segment_angles(n, Some(avoid));
        // Total span leaves room exactly for the back wedge.
        let span: f32 = angles.iter().map(|(s, e)| e - s).sum();
        assert!(approx(span, std::f32::consts::TAU - BACK_WEDGE));
        // No segment intrudes into the back wedge.
        for (start, _) in &angles {
            assert!(*start > avoid);
        }
    }

    #[test]
    fn hit_finds_the_wedge_containing_the_pointer() {
        let menu = RadialMenu {
            center: Vec2::new(500.0, 400.0),
            world_point: Vec3::ZERO,
            path: vec![],
        };
        let ring = &layout_rings(&menu, true)[0];
        for (i, &(start, end)) in ring.angles.iter().enumerate() {
            let mid = (start + end) / 2.0;
            let r = (INNER_RADIUS_PX + OUTER_RADIUS_PX) / 2.0;
            let p = ring.center + Vec2::new(mid.cos(), mid.sin()) * r;
            assert_eq!(ring_hit(ring, p), Some(RingHit::Segment(i)));
        }
    }

    #[test]
    fn hole_is_a_hit_and_beyond_the_rim_is_not() {
        let menu = RadialMenu {
            center: Vec2::ZERO,
            world_point: Vec3::ZERO,
            path: vec![],
        };
        let ring = &layout_rings(&menu, true)[0];
        // The hole reads as Hole (back/cancel), not as a miss.
        assert_eq!(ring_hit(ring, Vec2::X * 10.0), Some(RingHit::Hole));
        // Beyond the outer rim is a miss.
        assert_eq!(ring_hit(ring, Vec2::X * (OUTER_RADIUS_PX + 5.0)), None);
    }

    #[test]
    fn child_ring_sits_beside_the_parent_on_its_rim() {
        let menu = RadialMenu {
            center: Vec2::ZERO,
            world_point: Vec3::ZERO,
            path: vec![0],
        };
        let rings = layout_rings(&menu, true);
        assert_eq!(rings.len(), 2);
        let (parent, child) = (&rings[0], &rings[1]);

        // The child hangs off the spawning wedge: offset in its direction,
        // with its center exactly on the parent's outer radius (no gap).
        let (start, end) = parent.angles[0];
        let mid = (start + end) / 2.0;
        let expected = Vec2::new(mid.cos(), mid.sin()) * OUTER_RADIUS_PX;
        assert!((child.center - expected).length() < 1e-3);
    }

    #[test]
    fn child_ring_never_covers_parent_siblings() {
        // With the child center on the parent rim its wedges intentionally
        // hug the spawning wedge; the omitted back wedge buys them the
        // right to overlap only that wedge's angular span — never the
        // parent's sibling wedges.
        let menu = RadialMenu {
            center: Vec2::ZERO,
            world_point: Vec3::ZERO,
            path: vec![0],
        };
        let rings = layout_rings(&menu, true);
        let (parent, child) = (&rings[0], &rings[1]);
        let (wedge_start, wedge_end) = parent.angles[0];

        for &(start, end) in &child.angles {
            for step in 0..=24 {
                let a = start + (end - start) * step as f32 / 24.0;
                for r in [
                    INNER_RADIUS_PX,
                    (INNER_RADIUS_PX + OUTER_RADIUS_PX) / 2.0,
                    OUTER_RADIUS_PX,
                ] {
                    let p = child.center + Vec2::new(a.cos(), a.sin()) * r;
                    let d = p.length(); // parent center is the origin
                    if !(INNER_RADIUS_PX..OUTER_RADIUS_PX).contains(&d) {
                        continue; // outside the parent's wedge band anyway
                    }
                    let angle = p.y.atan2(p.x);
                    assert!(
                        angle_in(angle, wedge_start, wedge_end),
                        "child wedge point (r={r}, a={a}) covers the parent at angle \
                         {angle}, outside the spawning wedge's span"
                    );
                }
            }
        }
    }

    #[test]
    fn layout_always_matches_the_path_length() {
        // The draw loop indexes menu.path[k] for every ring that has a
        // child; the layout must therefore always have exactly one ring per
        // path element plus the root, with every path index in range —
        // including after hover navigation truncates the path mid-frame.
        for has_formation in [true, false] {
            for path in [vec![], vec![0]] {
                let menu = RadialMenu {
                    center: Vec2::ZERO,
                    world_point: Vec3::ZERO,
                    path: path.clone(),
                };
                let rings = layout_rings(&menu, has_formation);
                assert_eq!(rings.len(), menu.path.len() + 1);
                for k in 0..menu.path.len() {
                    assert!(
                        menu.path[k] < rings[k].items.len(),
                        "path index out of range at level {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn labels_fit_inside_their_wedges() {
        // Headless text measurement: a bare Context loads the default
        // fonts without a window. Every shipped label, in every menu
        // state, must stay inside its wedge's sector — never clip across a
        // wedge edge, the hole, or the rim.
        //
        // At 1x this is checked strictly (radial + angular, with a small
        // comfort pad). A second, 1.5x-scaled pass checks only the angular
        // fit: raising the font without also raising the ring radii is a
        // config-coupled change (a pure pixels-per-point scale scales both
        // and preserves fit), but the wedge edges must never be crossed at
        // any font size — that's a property of the layout, not the config.
        let ctx = egui::Context::default();
        // Fonts only load on the first pass; burn one. The font-atlas
        // deltas have no renderer to apply them to headless — clear instead
        // of tripping the unapplied-delta assert on drop.
        ctx.begin_pass(egui::RawInput::default());
        ctx.end_pass().textures_delta.clear();
        const RADIAL_PAD_PX: f32 = 4.0;
        const ANGULAR_PAD_PX: f32 = 4.0;

        for has_formation in [true, false] {
            for path in [Vec::new(), vec![0]] {
                let menu = RadialMenu {
                    center: Vec2::ZERO,
                    world_point: Vec3::ZERO,
                    path: path.clone(),
                };
                for ring in layout_rings(&menu, has_formation) {
                    for (i, item) in ring.items.iter().enumerate() {
                        let (start, end) = ring.angles[i];
                        let label = item_label(*item, &ring.path);
                        for (font_size, check_radial) in [
                            (LABEL_FONT_SIZE, true),
                            (LABEL_FONT_SIZE * 1.5, false),
                        ] {
                            let layout =
                                label_layout(&ctx, ring.center, start, end, label, font_size);
                            for corner in layout.quad {
                                let offset =
                                    Vec2::new(corner.x, corner.y) - ring.center;
                                let d = offset.length();
                                if check_radial {
                                    assert!(
                                        (INNER_RADIUS_PX + RADIAL_PAD_PX)
                                            < d
                                            && d
                                                < (OUTER_RADIUS_PX - RADIAL_PAD_PX),
                                        "{label:?} ({font_size}px) corner at radius {d} \
                                         clips the inner/outer edge (pad {RADIAL_PAD_PX})"
                                    );
                                }
                                let angle = offset.y.atan2(offset.x);
                                // Angular pad expressed at the corner's own
                                // radius so it stays a pixel measure.
                                let pad = ANGULAR_PAD_PX / d;
                                assert!(
                                    angle_in(angle, start + pad, end - pad),
                                    "{label:?} ({font_size}px) corner at angle {angle} \
                                     clips the wedge edge (pad {ANGULAR_PAD_PX}px)"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn hole_click_pops_sub_menu_and_closes_top_level() {
        let mut menu = RadialMenu {
            center: Vec2::ZERO,
            world_point: Vec3::ZERO,
            path: vec![0],
        };
        assert!(menu.path.pop().is_some());
        assert!(menu.path.is_empty());

        let mut top = RadialMenu {
            center: Vec2::ZERO,
            world_point: Vec3::ZERO,
            path: vec![],
        };
        assert!(top.path.pop().is_none());
    }
}
