use crate::kinematics::NNTree;
use crate::util::within_rect;
use std::f32::consts::FRAC_PI_2;
use bevy::color::palettes::basic::YELLOW;
use bevy::ecs::component::{Mutable, StorageType};
use bevy::ecs::lifecycle::{ComponentHook, HookContext};
use bevy::ecs::world::DeferredWorld;
use bevy::gizmos::GizmoAsset;
use bevy::gizmos::config::GizmoLineConfig;
use bevy::math::{Isometry3d, Quat, Vec3};
use bevy::prelude::{
    Assets, ButtonInput, Camera, Children, Color, Commands, Component, Dir3, Entity, FromWorld,
    Gizmo, Gizmos, GlobalTransform, Handle, InfinitePlane3d, KeyCode, MouseButton, Query, Res,
    ResMut, Resource,     Transform, Vec2, Window, With, World, default, info, warn,
};
use bevy_rts_camera::Ground;

#[derive(Resource, Default)]
pub struct Player {
    selecting: bool,
    corner1: Vec3,
    corner3: Vec3,
}

pub struct Selected;

/// Shared gizmo asset for selection rings.
#[derive(Resource)]
pub struct SelectionGizmo(pub Handle<GizmoAsset>);

impl FromWorld for SelectionGizmo {
    fn from_world(world: &mut World) -> Self {
        let mut gizmo = GizmoAsset::default();
        gizmo
            .circle(
                Isometry3d::from_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
                0.5,
                Color::srgb(1.0, 0.9, 0.0),
            )
            .resolution(32);
        let handle = world.resource_mut::<Assets<GizmoAsset>>().add(gizmo);
        Self(handle)
    }
}

/// Marker on the child entity that visualizes a selection.
#[derive(Component, Default)]
pub struct SelectionIndicator;

fn on_selected_insert(mut world: DeferredWorld, ctx: HookContext) {
    let Some(gizmo) = world.get_resource::<SelectionGizmo>() else {
        warn!("Selected inserted before SelectionGizmo resource exists");
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
    let ray = camera.viewport_to_world(camera_transform, *cursor_position).unwrap();

    // Calculate if and where the ray is hitting the ground plane.
    let distance = ray.intersect_plane(
        ground_transform.translation(),
        InfinitePlane3d { normal: Dir3::Y },
    )?;

    Some(ray.get_point(distance))
}

pub fn draw_cursor(
    camera_query: Query<(&Camera, &GlobalTransform), /*With<Player>*/>,
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

            let Some(point) = get_intersection(&cursor_position, camera, camera_transform, ground) else {
                return;
            };

            // Draw a circle just above the ground plane at that position.
            gizmos.circle(
                point + ground.up() * 0.01, // Up vector is already normalized.
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
        info!("[sel] no cursor position");
        return;
    };
    let Some(point) = get_intersection(&cursor_position, camera, camera_transform, ground) else {
        info!("[sel] ray missed ground, cursor={:?}", cursor_position);
        return;
    };

    if mouse_button_input.just_pressed(MouseButton::Left) {
        info!("[sel] PRESS cursor={:?} point={:?}", cursor_position, point);
        player.selecting = true;
        player.corner1 = point;
    }

    if mouse_button_input.just_released(MouseButton::Left) {
        info!(
            "[sel] RELEASE cursor={:?} point={:?} corner1={:?} selecting={}",
            cursor_position, point, player.corner1, player.selecting
        );
    }

    if mouse_button_input.just_released(MouseButton::Left) && player.selecting {
        player.selecting = false;
        if player.corner1 == point {
            return;
        }

        player.corner3 = point;

        if !keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
            for (entity, _) in &q_selected {
                commands.entity(entity).remove::<Selected>();
            }
        }


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

        // let dif_x = dif.project_onto(Dir3::X.as_vec3());
        // let dif_z = dif - dif_x;
        // let corner1 = player.corner1 + up;
        // let corner2 = corner1 + dif_x;
        // let corner3 = player.corner3 + up;
        // let corner4 = corner1 + dif_z;
        //
        // gizmos.line(corner1, corner2, Color::WHITE);
        // gizmos.line(corner2, corner3, Color::WHITE);
        // gizmos.line(corner3, corner4, Color::WHITE);
        // gizmos.line(corner4, corner1, Color::WHITE);
    }
}
