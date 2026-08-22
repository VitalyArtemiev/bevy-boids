use crate::boid::Boid;
use crate::formations::{
    Formation, FormationKind, FormationOf, FormationOrder, FormationSlot, Formations, MemberOf,
    Members, QuickCommandGroup,
};
use crate::kinematics::{NNTree, Velocity};
use crate::target::Target;
use crate::util::within_rect;
use bevy::color::palettes::basic::YELLOW;
use bevy::ecs::component::{Mutable, StorageType};
use bevy::ecs::lifecycle::{ComponentHook, HookContext};
use bevy::ecs::relationship::RelationshipTarget as _;
use bevy::ecs::world::DeferredWorld;
use bevy::gizmos::GizmoAsset;
use bevy::gizmos::config::GizmoLineConfig;
use bevy::math::{Isometry3d, Quat, Vec3};
use bevy::prelude::{
    Assets, ButtonInput, Camera, ChildOf, Children, Color, Commands, Component, Dir3, Entity,
    FromWorld, Gizmo, Gizmos, GlobalTransform, Handle, InfinitePlane3d, KeyCode, MouseButton,
    Query, Res, ResMut, Resource, Transform, Vec2, Window, With, Without, World, default, info,
    warn,
};
use bevy_rts_camera::{Ground, RtsCameraControls};
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

fn get_intersection(
    cursor_position: &Vec2,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    ground_transform: &GlobalTransform,
) -> Option<Vec3> {
    // Calculate a ray pointing from the camera into the world based on the cursor's position.
    let ray = camera
        .viewport_to_world(camera_transform, *cursor_position)
        .unwrap();

    // Calculate if and where the ray is hitting the ground plane.
    let distance = ray.intersect_plane(
        ground_transform.translation(),
        InfinitePlane3d { normal: Dir3::Y },
    )?;

    Some(ray.get_point(distance))
}

pub fn draw_cursor(
    camera_query: Query<(&Camera, &GlobalTransform) /*With<Player>*/>,
    ground_query: Query<&GlobalTransform, With<Ground>>,
    windows: Query<&Window>,
    mut gizmos: Gizmos,
) {
    match camera_query.single() {
        Ok((camera, camera_transform)) => {
            let ground = ground_query.single().unwrap();

            let Some(cursor_position) = windows.single().unwrap().cursor_position() else {
                return;
            };

            let Some(point) = get_intersection(&cursor_position, camera, camera_transform, ground)
            else {
                return;
            };

            // Draw a circle just above the ground plane at that position,
            // rotated to lie flat (circle default normal is +Z, ground is +Y).
            gizmos.circle(
                Isometry3d::new(
                    point + ground.up() * 0.01, // Up vector is already normalized.
                    Quat::from_rotation_x(-FRAC_PI_2),
                ),
                0.2,
                Color::WHITE,
            );
        }
        _ => {}
    }
}

pub fn mouse_click_system(
    mut player: ResMut<Player>,
    mut q_camera: Query<(&Camera, &GlobalTransform)>,
    q_ground: Query<&GlobalTransform, With<Ground>>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    q_selected: Query<(Entity, &Children), With<Selected>>,
    tree: Res<NNTree>,
    mut gizmos: Gizmos,
    mut commands: Commands,
) {
    let (camera, camera_transform) = q_camera.single_mut().unwrap();
    let ground = q_ground.single().unwrap();
    let Some(cursor_position) = windows.single().unwrap().cursor_position() else {
        return;
    };
    let Some(point) = get_intersection(&cursor_position, camera, camera_transform, ground) else {
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
    q_formations: Query<(Entity, &QuickCommandGroup, &Members, Option<&Formations>), With<Formation>>,
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

            // Free the slot: detach members of the formation currently in it.
            // Detached members must also drop their FormationSlot: a stale
            // slot survives the validity check in assign_slots (in-range,
            // unique) and would pin the member to an arbitrary slot in
            // whatever formation it joins next.
            if let Some((old, _, members, subs)) =
                q_formations.iter().find(|(_, s, _, _)| s.0 == slot)
            {
                for member in members.iter() {
                    commands
                        .entity(member)
                        .remove::<MemberOf>()
                        .remove::<FormationSlot>();
                }
                if let Some(subs) = subs {
                    for sub in subs.iter() {
                        commands.entity(sub).remove::<FormationOf>();
                    }
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
    } else if let Some((formation_entity, _, _, _)) =
        q_formations.iter().find(|(_, s, _, _)| s.0 == slot)
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
    q_camera: Query<(&Camera, &GlobalTransform)>,
    q_ground: Query<&GlobalTransform, With<Ground>>,
    windows: Query<&Window>,
    q_selected_boids: Query<Entity, (With<Selected>, With<Boid>, Without<Formation>)>,
    q_selected_formations: Query<Entity, (With<Selected>, With<Formation>)>,
    q_member_of: Query<&MemberOf>,
    mut q_formation_mut: Query<&mut Formation>,
    mut commands: Commands,
    mut q_targets: Query<&mut Target>,
    mut q_camera_controls: Query<&mut RtsCameraControls>,
    mut gizmos: Gizmos,
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
    let Ok(ground) = q_ground.single() else {
        return;
    };
    let Some(cursor) = windows.single().ok().and_then(|w| w.cursor_position()) else {
        return;
    };
    let Some(point) = get_intersection(&cursor, camera, camera_transform, ground) else {
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
fn designate_frontage(
    left: Vec3,
    right_pt: Vec3,
    adjust_width: bool,
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
    // (columns = width / SPACING); the slot system re-maps members.
    if adjust_width {
        let new_cols = (width / FormationKind::SPACING).round().max(1.0) as usize;
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
