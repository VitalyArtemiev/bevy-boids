use bevy::math::Vec3;
use bevy::prelude::{Bundle, ButtonInput, Camera, Camera3d, Color, Component, Direction3d, Entity, Gizmos, GlobalTransform, KeyCode, MouseButton, Plane3d, Query, Res, Time, Transform, Vec2, Window, With};
use bevy_rts_camera::{Ground, RtsCamera};
use crate::boid::{Boid, NNTree};
use crate::kinematics::Velocity;
use crate::terrain::Terrain;

#[derive(Component,Default)]
pub struct Player {
    selecting: bool,
    corner1: Vec3,
    corner4: Vec3,
    selected: Vec<Entity>
}


fn get_intersection(cursor_position: &Vec2, camera: &Camera, camera_transform: &GlobalTransform, ground_transform: &GlobalTransform) -> Option<Vec3> {
    // Calculate a ray pointing from the camera into the world based on the cursor's position.
    let ray = camera.viewport_to_world(camera_transform, *cursor_position)?;
    
    // Calculate if and where the ray is hitting the ground plane.
    let distance = ray.intersect_plane(ground_transform.translation(), Plane3d::new(ground_transform.up()))?;

    Some(ray.get_point(distance))
}


pub fn draw_cursor(
    camera_query: Query<(&Camera, &GlobalTransform), With<Player>>,
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
        point + ground.up() * 0.01,
        Direction3d::new_unchecked(ground.up()), // Up vector is already normalized.
        0.2,
        Color::WHITE,
    );
}


pub fn mouse_click_system(mut q_player: Query<(&Camera, &GlobalTransform, &mut Player)>, 
                          q_ground: Query<&GlobalTransform, With<Ground>>, 
                          mouse_button_input: Res<ButtonInput<MouseButton>>,
                          windows: Query<&Window>,
                          mut gizmos: Gizmos) {
    let (camera, camera_transform, mut player) = q_player.single_mut();
    let ground = q_ground.single();
    let Some(cursor_position) = windows.single().cursor_position() else {
        return;
    };
    let Some(point) = get_intersection(&cursor_position, camera, camera_transform, ground) else {
        return
    };

    if mouse_button_input.just_pressed(MouseButton::Left) {
        player.selecting = true;
        player.corner1 = point;
    }

    if mouse_button_input.just_released(MouseButton::Left) {
        player.corner4 = point;
        player.selecting = false;
    }
    
    if mouse_button_input.pressed(MouseButton::Left) {
        player.corner4 = point;

        let right = camera_transform.right();
        let up = ground.up() * 0.01;

        let dif = player.corner4 - player.corner1;
        
        let dif_hor = dif.project_onto(right);
        let dif_vert = dif - dif_hor;
        let corner1 = player.corner1 + up;
        let corner2 = corner1 + dif_hor;
        let corner3 = player.corner4 + up;
        let corner4 = corner1 + dif_vert;
        
        gizmos.line(corner1, corner2, Color::BLUE);
        gizmos.line(corner2, corner3, Color::BLUE);
        gizmos.line(corner3, corner4, Color::BLUE);
        gizmos.line(corner4, corner1, Color::BLUE);
    }
}