use crate::kinematics::{NNTree, Velocity};
use crate::util::within_rect;
use bevy::math::Vec3;
use bevy::pbr::{MeshMaterial3d, PointLight, StandardMaterial};
use bevy::prelude::{default, BuildChildren, ButtonInput, Camera, Children, Color, Commands, Component, Dir3, Entity, Gizmos, GlobalTransform, InfinitePlane3d, KeyCode, Mesh3d, MouseButton, Query, Res, ResMut, Resource, Transform, Vec2, Window, With};
use bevy_rts_camera::Ground;
use crate::target::Target;

#[derive(Resource, Default)]
pub struct Player {
    selecting: bool,
    corner1: Vec3,
    corner3: Vec3,
}

#[derive(Component)]
pub struct Selected{
    transform: Transform,
    mesh: Mesh3d,
    material: MeshMaterial3d<StandardMaterial>,
}

impl Default for Selected {
    fn default() -> Self {
        Selected {
            transform: Transform::from_xyz(0.0, 1.0, 0.0),
            mesh: Default::default(),
            material: Default::default(),
        }
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
    let (camera, camera_transform) = camera_query.single();
    let ground = ground_query.single();

    let Some(cursor_position) = windows.single().cursor_position() else {
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
    let (camera, camera_transform) = q_camera.single_mut();
    let ground = q_ground.single();
    let Some(cursor_position) = windows.single().cursor_position() else {
        return;
    };
    let Some(point) = get_intersection(&cursor_position, camera, camera_transform, ground) else {
        return;
    };

    if mouse_button_input.just_pressed(MouseButton::Left) {
        player.selecting = true;
        player.corner1 = point;
    }

    if mouse_button_input.just_released(MouseButton::Left) {
        player.corner3 = point;
        player.selecting = false;

        if !keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
            for (entity, children) in &q_selected {
                commands.entity(entity).remove::<Selected>();
                commands.entity(entity).remove_children(children);
                for child in children {
                    commands.entity(*child).despawn()
                }

                // commands.entity(entity).clear_children();
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
            commands.entity(entity.unwrap()).insert(Selected::default());

            commands
                // .spawn(PointLightBundle {
                //     point_light: PointLight {
                //         intensity: 1000.0,
                //         range: 0.2,
                //         shadows_enabled: false,
                //         ..default()
                //     },
                //     transform: Transform::from_xyz(0., 1.1, 0.),
                //     ..default()
                // })
                .spawn(
                    PointLight {
                        color: Default::default(),
                        intensity: 1000.0,
                        range: 5.0,
                        radius: 5.0,
                        shadows_enabled: false,
                        shadow_depth_bias: 0.0,
                        shadow_normal_bias: 0.0,
                        shadow_map_near_z: 0.0,
                    },
                )
                .set_parent(entity.unwrap());

            // if let Ok(mut boid) = query.get_mut(entity.unwrap()) {
            //     entity.
            //     boid.
            // }
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
