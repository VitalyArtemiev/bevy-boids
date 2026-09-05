use crate::boid::Boid;
use crate::formations::{
    Formation, FormationKind, FormationOrder, FormationSlot, FormationTuning, MemberOf, Members,
    QuickCommandGroup,
};
use crate::kinematics::{NNTree, Velocity};
use crate::target::Target;
use crate::terrain::HeightField;
use crate::util::within_rect;
use bevy::color::palettes::basic::YELLOW;
use bevy::ecs::component::{Mutable, StorageType};
use bevy::ecs::lifecycle::{ComponentHook, HookContext};
use bevy::ecs::relationship::RelationshipTarget as _;
use bevy::ecs::world::DeferredWorld;
use bevy::gizmos::GizmoAsset;
use bevy::gizmos::config::GizmoLineConfig;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::math::{Isometry3d, Quat, Ray3d, Vec3};
use bevy::prelude::{
    Assets, ButtonInput, Camera, ChildOf, Children, Color, Commands, Component, Dir3, Entity,
    FromWorld, Gizmo, Gizmos, GlobalTransform, Handle, InfinitePlane3d, KeyCode, MessageReader,
    MouseButton, Query, Res, ResMut, Resource, Transform, Vec2, Window, With, Without, World,
    default, info, warn,
};
use bevy_rts_camera::{RtsCamera, RtsCameraControls};
use std::f32::consts::FRAC_PI_2;

#[derive(Resource, Default)]
pub struct Player {
    selecting: bool,
    corner1: Vec3,
    corner3: Vec3,
    /// Left front corner of a frontage being designated by RMB drag.
    front_left: Option<Vec3>,
}

pub struct Selected;

/// Shared gizmo asset for selection rings.
#[derive(Resource)]
pub struct SelectionGizmo(pub Handle<GizmoAsset>);

impl FromWorld for SelectionGizmo {
    fn from_world(world: &mut World) -> Self {
        // Elongated triangle outline on the ground plane, apex pointing +Z
        // (local forward). Oriented per-boid by the indicator child's
        // Transform rotation (see selection_indicator_face system).
        let mut gizmo = GizmoAsset::default();
        // Triangle in the XZ ground plane (x = right, z = forward), apex +Z.
        let tip = Vec3::new(0.0, 0.0, 0.6);
        let left = Vec3::new(-0.25, 0.0, -0.4);
        let right = Vec3::new(0.25, 0.0, -0.4);
        gizmo.linestrip([tip, right, left, tip], Color::srgb(1.0, 0.9, 0.0));
        let handle = world.resource_mut::<Assets<GizmoAsset>>().add(gizmo);
        Self(handle)
    }
}

/// Shared gizmo asset for formation selection: a flat unit square (side 1,
/// centered); the child entity's Transform scale sizes it to the formation extent.
#[derive(Resource)]
pub struct FormationSelectionGizmo(pub Handle<GizmoAsset>);

impl FromWorld for FormationSelectionGizmo {
    fn from_world(world: &mut World) -> Self {
        let mut gizmo = GizmoAsset::default();
        gizmo.rect(
            Isometry3d::from_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
            Vec2::ONE,
            Color::srgb(1.0, 0.9, 0.0),
        );
        let handle = world.resource_mut::<Assets<GizmoAsset>>().add(gizmo);
        Self(handle)
    }
}

/// Marker on the child entity that visualizes a selection.
#[derive(Component, Default)]
pub struct SelectionIndicator;

fn on_selected_insert(mut world: DeferredWorld, ctx: HookContext) {
    // Formations get an extent-sized square; boids get the small ring.
    let formation_extent = world
        .get_entity_mut(ctx.entity)
        .ok()
        .and_then(|entity| entity.get::<Formation>().map(|formation| formation.extent));
    let handle = match formation_extent {
        Some(extent) => {
            let Some(gizmo) = world.get_resource::<FormationSelectionGizmo>() else {
                warn!("Selected inserted before FormationSelectionGizmo resource exists");
                return;
            };
            let handle = gizmo.0.clone();
            world.commands().entity(ctx.entity).with_children(|parent| {
                parent.spawn((
                    SelectionIndicator,
                    Gizmo {
                        handle,
                        line_config: GizmoLineConfig {
                            width: 3.0,
                            ..default()
                        },
                        depth_bias: -1.0,
                    },
                    Transform::from_xyz(0.0, 0.05, 0.0).with_scale(Vec3::splat(extent)),
                ));
            });
            return;
        }
        None => {
            let Some(gizmo) = world.get_resource::<SelectionGizmo>() else {
                warn!("Selected inserted before SelectionGizmo resource exists");
                return;
            };
            gizmo.0.clone()
        }
    };
    world.commands().entity(ctx.entity).with_children(|parent| {
        parent.spawn((
            SelectionIndicator,
            Gizmo {
                handle,
                line_config: GizmoLineConfig {
                    width: 3.0,
                    ..default()
                },
                depth_bias: -1.0,
            },
            Transform::from_xyz(0.0, 0.05, 0.0),
        ));
    });
}

fn on_selected_remove(mut world: DeferredWorld, ctx: HookContext) {
    world
        .commands()
        .entity(ctx.entity)
        .despawn_related::<Children>();
}

impl Component for Selected {
    const STORAGE_TYPE: StorageType = StorageType::Table;
    type Mutability = Mutable;

    fn on_insert() -> Option<ComponentHook> {
        Some(on_selected_insert)
    }

    fn on_remove() -> Option<ComponentHook> {
        Some(on_selected_remove)
    }
}

/// March a ray against the height field: exponential steps outward,
/// bisection refinement at the first surface crossing. ~50 field samples
/// typical, accurate to well under a metre at any practical range.
fn march_ray_against_height(field: &HeightField, ray: Ray3d) -> Option<Vec3> {
    let mut prev_t = 0.0f32;
    let mut t = 0.5f32;
    let mut step = 1.0f32;
    for _ in 0..256 {
        let p = ray.get_point(t);
        if p.y <= field.height(p.x, p.z) {
            // Crossed below the surface: bisect the bracket.
            let (mut lo, mut hi) = (prev_t, t);
            for _ in 0..16 {
                let mid = (lo + hi) / 2.0;
                let m = ray.get_point(mid);
                if m.y <= field.height(m.x, m.z) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return Some(ray.get_point(hi));
        }
        prev_t = t;
        t += step;
        step *= 1.06;
        if t > 60_000.0 {
            return None;
        }
    }
    None
}

fn get_intersection(
    field: &HeightField,
    cursor_position: &Vec2,
    camera: &Camera,
    camera_transform: &GlobalTransform,
) -> Option<Vec3> {
    // Calculate a ray pointing from the camera into the world based on the cursor's position.
    let ray = camera
        .viewport_to_world(camera_transform, *cursor_position)
        .ok()?;

    march_ray_against_height(field, ray)
}

pub fn draw_cursor(
    camera_query: Query<(&Camera, &GlobalTransform) /*With<Player>*/>,
    field: Res<HeightField>,
    windows: Query<&Window>,
    mut gizmos: Gizmos,
    debug: Res<crate::debug_ui::DebugConfig>,
) {
    if !debug.show_cursor_circle {
        return;
    }
    match camera_query.single() {
        Ok((camera, camera_transform)) => {
            let Some(cursor_position) = windows.single().unwrap().cursor_position() else {
                return;
            };

            let Some(point) = get_intersection(&field, &cursor_position, camera, camera_transform)
            else {
                return;
            };

            // Draw a circle just above the terrain at that position,
            // rotated to lie flat (circle default normal is +Z, ground is +Y).
            gizmos.circle(
                Isometry3d::new(point + Vec3::Y * 0.01, Quat::from_rotation_x(-FRAC_PI_2)),
                0.2,
                Color::WHITE,
            );
        }
        _ => {}
    }
}

pub fn mouse_click_system(
    mut player: ResMut<Player>,
    field: Res<HeightField>,
    mut q_camera: Query<(&Camera, &GlobalTransform)>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    q_selected: Query<(Entity, &Children), With<Selected>>,
    tree: Res<NNTree>,
    mut gizmos: Gizmos,
    mut commands: Commands,
) {
    let (camera, camera_transform) = q_camera.single_mut().unwrap();
    let Some(cursor_position) = windows.single().unwrap().cursor_position() else {
        return;
    };
    let Some(point) = get_intersection(&field, &cursor_position, camera, camera_transform) else {
        return;
    };

    if mouse_button_input.just_pressed(MouseButton::Left) {
        player.selecting = true;
        player.corner1 = point;
    }

    if mouse_button_input.just_released(MouseButton::Left) && player.selecting {
        player.selecting = false;

        if !keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
            for (entity, _) in &q_selected {
                commands.entity(entity).remove::<Selected>();
            }
        }

        // Click without drag: deselect (above) but no box select. This also
        // guards against a stale corner1 producing a phantom selection.
        if player.corner1 == point {
            return;
        }

        player.corner3 = point;

        let right = camera_transform.right();
        let dif = player.corner3 - player.corner1;

        let dif_hor = dif.project_onto(right.as_vec3());
        let dif_vert = dif - dif_hor;

        let corner1 = player.corner1;
        let corner2 = corner1 + dif_vert;
        let corner3 = player.corner3;
        let corner4 = corner1 + dif_hor;

        for (_, entity) in within_rect(corner1, corner2, corner3, corner4, tree) {
            commands.entity(entity.unwrap()).insert(Selected);
        }
    }

    if mouse_button_input.pressed(MouseButton::Left) {
        player.corner3 = point;

        let right = camera_transform.right();
        let dif = player.corner3 - player.corner1;

        let dif_hor = dif.project_onto(right.as_vec3());
        let dif_vert = dif - dif_hor;

        let corner1 = player.corner1;
        let corner2 = corner1 + dif_vert;
        let corner3 = player.corner3;
        let corner4 = corner1 + dif_hor;

        gizmos.line(corner1, corner2, Color::WHITE);
        gizmos.line(corner2, corner3, Color::WHITE);
        gizmos.line(corner3, corner4, Color::WHITE);
        gizmos.line(corner4, corner1, Color::WHITE);
    }
}

/// Hotkeys 1-6, RTS quick command groups:
/// - Ctrl+N -> assign current selection to slot N (boids: new formation;
///   a selected formation: re-slot it)
/// - N alone -> select the formation stored in slot N (replacing selection)
pub fn quick_group_system(
    keys: Res<ButtonInput<KeyCode>>,
    q_selected: Query<(Entity, &Transform), (With<Selected>, Without<Formation>)>,
    q_selected_formations: Query<Entity, (With<Selected>, With<Formation>)>,
    q_formations: Query<(Entity, &QuickCommandGroup, &Members), With<Formation>>,
    mut commands: Commands,
) {
    const SLOT_KEYS: [KeyCode; 6] = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
    ];
    let Some(slot) = SLOT_KEYS
        .iter()
        .position(|key| keys.just_pressed(*key))
        .map(|i| i as u8)
    else {
        return;
    };

    let ctrl = keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]);

    if ctrl {
        if let Ok(selected_formation) = q_selected_formations.single() {
            // Re-slot the selected formation to this number.
            commands
                .entity(selected_formation)
                .insert(QuickCommandGroup(slot));
        } else if !q_selected.is_empty() {
            // Assign: build a new formation at the selection's centroid.
            let mut centroid = Vec3::ZERO;
            let count = q_selected.iter().count();
            for (_, transform) in &q_selected {
                centroid += transform.translation;
            }
            centroid /= count as f32;

            // Free the slot: detach occupants of the formation currently in
            // it. Detached occupants must also drop their FormationSlot: a
            // stale slot survives the validity check in assign_slots
            // (in-range, unique) and would pin the member to an arbitrary
            // slot in whatever formation it joins next.
            if let Some((old, _, members)) = q_formations.iter().find(|(_, s, _)| s.0 == slot) {
                for member in members.iter() {
                    commands
                        .entity(member)
                        .remove::<MemberOf>()
                        .remove::<FormationSlot>();
                }
                commands.entity(old).despawn();
            }

            // max_speed is derived from the member list by
            // `init_formation_speed` on the ticks after the members attach.
            let formation = commands
                .spawn((
                    Formation::default(),
                    Transform::from_translation(centroid),
                    QuickCommandGroup(slot),
                ))
                .id();
            for (entity, _) in &q_selected {
                commands.entity(entity).insert(MemberOf(formation));
            }
        }
    } else if let Some((formation_entity, _, _)) = q_formations.iter().find(|(_, s, _)| s.0 == slot)
    {
        // Plain number: select this group, replacing the current selection.
        for (entity, _) in &q_selected {
            commands.entity(entity).remove::<Selected>();
        }
        for entity in &q_selected_formations {
            commands.entity(entity).remove::<Selected>();
        }
        commands.entity(formation_entity).insert(Selected);
    }
}

/// Right-click drag designates a frontage for the selected entities:
/// press = left front corner, release = right front corner. Selected units
/// (free boids, formations, and formations of member boids) are arranged in
/// a grid along the frontage, facing perpendicular to it; depth is automatic
/// from the minimum spacing: 0.5 per boid, or the maximum extent among the
/// formations being positioned.
pub fn frontage_position_system(
    mut player: ResMut<Player>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    field: Res<HeightField>,
    q_camera: Query<(&Camera, &GlobalTransform)>,
    windows: Query<&Window>,
    q_selected_boids: Query<Entity, (With<Selected>, With<Boid>, Without<Formation>)>,
    q_selected_formations: Query<Entity, (With<Selected>, With<Formation>)>,
    q_member_of: Query<&MemberOf>,
    mut q_formation_mut: Query<&mut Formation>,
    mut commands: Commands,
    mut q_targets: Query<&mut Target>,
    mut q_camera_controls: Query<&mut RtsCameraControls>,
    mut gizmos: Gizmos,
    tuning: Res<FormationTuning>,
) {
    // Nothing selected: RMB stays the camera drag-pan control. With a
    // selection, RMB becomes frontage designation, so disable camera drag.
    let has_selection = !q_selected_boids.is_empty() || !q_selected_formations.is_empty();
    for mut controls in &mut q_camera_controls {
        controls.button_drag = (!has_selection).then_some(MouseButton::Right);
    }
    if !has_selection {
        player.front_left = None;
        return;
    }

    let Ok((camera, camera_transform)) = q_camera.single() else {
        return;
    };
    let Some(cursor) = windows.single().ok().and_then(|w| w.cursor_position()) else {
        return;
    };
    let Some(point) = get_intersection(&field, &cursor, camera, camera_transform) else {
        return;
    };

    if mouse.just_pressed(MouseButton::Right) {
        player.front_left = Some(point);
    }

    if let Some(left) = player.front_left {
        if mouse.pressed(MouseButton::Right) {
            gizmos.line(left, point, Color::srgb(0.3, 1.0, 0.3));
        }
        if mouse.just_released(MouseButton::Right) {
            player.front_left = None;
            let adjust_width = keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]);
            designate_frontage(
                left,
                point,
                adjust_width,
                tuning.spacing_m,
                &q_selected_boids,
                &q_selected_formations,
                &q_member_of,
                &mut q_formation_mut,
                &mut q_targets,
            );
        }
    }
}

/// Arrange `units` in a grid spanning the frontage from `left` to `right_pt`.
/// `spacing_m` is the slot spacing (see [`FormationTuning`]).
fn designate_frontage(
    left: Vec3,
    right_pt: Vec3,
    adjust_width: bool,
    spacing_m: f32,
    q_selected_boids: &Query<Entity, (With<Selected>, With<Boid>, Without<Formation>)>,
    q_selected_formations: &Query<Entity, (With<Selected>, With<Formation>)>,
    q_member_of: &Query<&MemberOf>,
    q_formation_mut: &mut Query<&mut Formation>,
    q_targets: &mut Query<&mut Target>,
) {
    let right_vec = right_pt - left;
    let width = right_vec.length();
    if width < 0.1 {
        return;
    }
    let right_dir = right_vec / width;
    // The frontage line has a normal: dragging left->right faces the formation
    // "up" (+Z on screen), right->left faces it "down". Rows extend BEHIND the
    // line (opposite the facing), so the line is the formation's front edge.
    let forward = right_dir.cross(Vec3::Y).normalize();

    // Collect positionable units: formations directly, member boids via their
    // formation (its origin takes the slot; propagation moves the members).
    let mut units: Vec<Entity> = Vec::new();
    for formation in q_selected_formations.iter() {
        if !units.contains(&formation) {
            units.push(formation);
        }
    }
    for boid in q_selected_boids.iter() {
        match q_member_of.get(boid) {
            Ok(MemberOf(formation)) => {
                if !units.contains(formation) {
                    units.push(*formation);
                }
            }
            Err(_) => {
                if !units.contains(&boid) {
                    units.push(boid);
                }
            }
        }
    }
    if units.is_empty() {
        return;
    }

    // Minimum spacing: boid slot spacing, widened to the largest formation
    // extent when any formation is among the units.
    // Ctrl held: fit each formation's internal grid width to the frontage
    // (columns = width / spacing); the slot system re-maps members.
    if adjust_width {
        let new_cols = (width / spacing_m).round().max(1.0) as usize;
        for &unit in &units {
            if let Ok(mut formation) = q_formation_mut.get_mut(unit) {
                formation.columns = Some(new_cols);
                info!("[width] formation {unit:?} columns={new_cols}");
            }
        }
    }

    let mut spacing: f32 = 1.0;
    for &unit in &units {
        if let Ok(formation) = q_formation_mut.get_mut(unit) {
            spacing = spacing.max(formation.extent);
        }
    }

    let n = units.len();
    let cols = ((width / spacing) as usize + 1).clamp(1, n);
    let midpoint = left + right_vec / 2.0;

    for (k, &unit) in units.iter().enumerate() {
        let col = k % cols;
        let row = k / cols;
        let col_x = if cols > 1 {
            width * col as f32 / (cols - 1) as f32 - width / 2.0
        } else {
            0.0
        };
        // The line is the FRONT edge: each unit's center sits half its depth
        // behind it, so the body (boid radius / formation extent) is flush
        // against the line rather than straddling it.
        let pos = midpoint + right_dir * col_x - forward * (row as f32 * spacing + spacing * 0.5);
        if let Ok(mut formation) = q_formation_mut.get_mut(unit) {
            // Formation control goes through the task queue: a new order
            // replaces pending tasks. Move handles the facing-change slot
            // re-map; a width change reforms first (new columns).
            formation.tasks.clear();
            if adjust_width {
                formation.tasks.push_back(FormationOrder::Reform);
            }
            formation.tasks.push_back(FormationOrder::Move {
                pos,
                facing_dir: forward,
            });
        } else if let Ok(mut target) = q_targets.get_mut(unit) {
            // Free boids (not in any formation): direct target.
            target.pos = pos;
            target.dir = forward;
        }
    }
}

/// Point each selected boid's triangle indicator along its current movement
/// direction (velocity if moving, else its target direction). Formation
/// indicators (squares) are skipped - their facing is the formation's.
pub fn selection_indicator_face(
    q_boids: Query<(&Velocity, &Target), With<Boid>>,
    mut q_indicators: Query<(&mut Transform, &ChildOf), With<SelectionIndicator>>,
) {
    for (mut transform, parent) in &mut q_indicators {
        let Ok((velocity, target)) = q_boids.get(parent.parent()) else {
            continue; // formation square: orientation handled by the formation
        };
        let dir = if velocity.v.length_squared() > 0.01 {
            velocity.v
        } else {
            target.dir
        };
        if dir.length_squared() > 1e-6 {
            transform.rotation = Quat::from_rotation_y(dir.x.atan2(dir.z));
        }
    }
}

/// Height-scaled zoom input: replaces the plugin's stock zoom (neutralized
/// by `zoom_sensitivity: 0` on the camera controls).
///
/// Stock adds a constant zoom delta per wheel unit, which only felt right
/// when the whole height range was 300 m; with a 30 km ceiling it would
/// move the camera 7.5 km per notch near the ground. Instead the step is
/// a constant FRACTION of the current height (constant-ratio "map zoom"):
///
/// `step(h) = max(h * ZOOM_STEP_FRACTION, ZOOM_STEP_MIN_M)`
///
/// Every notch covers the same fraction of the remaining altitude, so the
/// camera decelerates smoothly into the ground (the legacy flat 75 m step
/// below 300 m slammed 97% of the altitude in one notch near max zoom).
/// Tuned by feel on the near-ground zoom: half the original 0.25 ratio.
/// Ground -> 30 km is ~72 notches.
const ZOOM_STEP_FRACTION: f32 = 0.125;
/// Step floor so a notch stays meaningful at walking height (max zoom).
const ZOOM_STEP_MIN_M: f32 = 0.5;
/// Pixel-unit wheel deltas are scaled down like the plugin does.
const ZOOM_PIXEL_UNIT_SCALE: f32 = 0.001;

pub fn height_scaled_zoom(
    mut mouse_wheel: MessageReader<MouseWheel>,
    options: Option<Res<crate::ui::OptionsSettings>>,
    mut cam_q: Query<(&mut RtsCamera, &mut RtsCameraControls)>,
) {
    // The Options "zoom speed" knob, 1.0 without the resource (tests,
    // headless). It scales our step ONLY — the crate's own zoom stays
    // neutralized (`zoom_sensitivity: 0`), see `apply_options`.
    let zoom_speed = options
        .map(|settings| settings.camera_zoom_sensitivity)
        .unwrap_or(1.0);
    for (mut cam, controls) in cam_q.iter_mut().filter(|(_, c)| c.enabled) {
        let wheel = mouse_wheel
            .read()
            .map(|message| match message.unit {
                MouseScrollUnit::Line => message.y,
                MouseScrollUnit::Pixel => message.y * ZOOM_PIXEL_UNIT_SCALE,
            })
            .fold(0.0, |acc, val| acc + val);
        if wheel == 0.0 {
            continue;
        }
        let span = cam.height_max - cam.height_min;
        // Positive wheel = zoom in = descend.
        let height = cam.height_max - cam.target_zoom * span;
        let step = (height * ZOOM_STEP_FRACTION * zoom_speed).max(ZOOM_STEP_MIN_M);
        let target_height = (height - wheel * step).clamp(cam.height_min, cam.height_max);
        cam.target_zoom = (cam.height_max - target_height) / span;
    }
}

#[cfg(test)]
mod zoom_tests {
    use super::*;
    use bevy::app::App;
    use bevy::input::mouse::MouseScrollUnit;
    use bevy::input::touch::TouchPhase;
    use bevy::prelude::Update;

    fn zoom_app(target_zoom: f32) -> (App, Entity) {
        let mut app = App::new();
        app.add_message::<MouseWheel>()
            .add_systems(Update, height_scaled_zoom);
        let camera = app
            .world_mut()
            .spawn((
                RtsCamera {
                    height_min: 2.0,
                    height_max: 30_000.0,
                    zoom: target_zoom,
                    target_zoom,
                    ..Default::default()
                },
                RtsCameraControls::default(),
            ))
            .id();
        (app, camera)
    }

    fn scroll(app: &mut App, y: f32) {
        app.world_mut().write_message(MouseWheel {
            unit: MouseScrollUnit::Line,
            x: 0.0,
            y,
            window: Entity::PLACEHOLDER,
            phase: TouchPhase::Moved,
        });
        app.update();
    }

    fn target_height(app: &App, camera: Entity) -> f32 {
        let rts = app.world().get::<RtsCamera>(camera).unwrap();
        rts.height_max - rts.target_zoom * (rts.height_max - rts.height_min)
    }

    #[test]
    fn notch_at_max_zoom_eases_in_with_the_height_floor() {
        // h = 2 m: step = max(0.5, 0.125 × 2) = 0.5 m — the camera
        // decelerates into the ground instead of the legacy flat 75 m
        // slam.
        let (mut app, camera) = zoom_app(1.0);
        scroll(&mut app, -1.0);
        assert!((target_height(&app, camera) - 2.5).abs() < 1e-3);
    }

    #[test]
    fn options_zoom_speed_scales_the_step() {
        // The Options knob is a multiplier over the height fraction:
        // 2× doubles the 12.5 m step at h = 100 m to 25 m. (The knob
        // must scale OUR step only — see `apply_options`, which keeps the
        // crate's constant-units zoom disarmed.)
        let (mut app, camera) = zoom_app(29_900.0 / 29_998.0);
        app.insert_resource(crate::ui::OptionsSettings {
            camera_zoom_sensitivity: 2.0,
            ..Default::default()
        });
        scroll(&mut app, -1.0);
        assert!((target_height(&app, camera) - 125.0).abs() < 1e-3);
    }

    #[test]
    fn notch_is_a_constant_fraction_of_the_height() {
        // h = 100 m: step = 12.5 m (half the original 25 m feel — one
        // notch covers an eighth of the altitude). The step compounds on
        // the new height, so a second notch adds 12.5% of 112.5.
        let (mut app, camera) = zoom_app(29_900.0 / 29_998.0);
        scroll(&mut app, -1.0);
        assert!((target_height(&app, camera) - 112.5).abs() < 1e-3);
        scroll(&mut app, -1.0);
        assert!((target_height(&app, camera) - 126.5625).abs() < 1e-2);
    }

    #[test]
    fn step_scales_up_with_altitude_at_mid_range() {
        // h = 300 m (the old ceiling): step = 37.5 m — the ratio holds
        // everywhere above the floor.
        let (mut app, camera) = zoom_app(29_700.0 / 29_998.0);
        scroll(&mut app, -1.0);
        assert!((target_height(&app, camera) - 337.5).abs() < 1.0);
    }

    #[test]
    fn step_grows_with_altitude_at_the_top() {
        // h = 20 km: step = 20 000 × 0.125 = 2500 m.
        let (mut app, camera) = zoom_app(10_000.0 / 29_998.0);
        scroll(&mut app, -1.0);
        assert!((target_height(&app, camera) - 22_500.0).abs() < 1.0);
    }

    #[test]
    fn zooming_in_at_the_ceiling_clamps_without_overshoot() {
        let (mut app, camera) = zoom_app(0.0); // h = 30 km
        scroll(&mut app, -5.0); // hard zoom out at max altitude
        assert!((target_height(&app, camera) - 30_000.0).abs() < 1e-3);
        scroll(&mut app, 20.0); // hard zoom in toward the ground
        assert!(
            (target_height(&app, camera) - 2.0).abs() < 1.0,
            "must clamp at height_min"
        );
    }
}
