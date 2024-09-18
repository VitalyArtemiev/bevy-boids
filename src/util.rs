use crate::kinematics::NNTree;
use bevy::prelude::*;
use bevy_spatial::kdtree::KDTree3;
use bevy_spatial::SpatialAABBAccess;

pub trait BundleDefault {
    fn default(
        meshes: &mut ResMut<Assets<Mesh>>,
        images: &mut ResMut<Assets<Image>>,
        materials: &mut ResMut<Assets<StandardMaterial>>,
    ) -> Self;
}

pub fn side(start: Vec3, end: Vec3, query: &Vec3) -> f32 {
    (end.z - start.z) * (query.x - start.x) + (-end.x + start.x) * (query.z - start.z)
}

pub fn point_in_triangle(tri: Triangle3d, query: &Vec3) -> bool {
    let p1 = tri.vertices[0];
    let p2 = tri.vertices[1];
    let p3 = tri.vertices[2];
    let side1: bool = side(p1, p2, query) >= 0.;
    let side2: bool = side(p2, p3, query) >= 0.;
    let side3: bool = side(p3, p1, query) >= 0.;
    side1 && side2 && side3
}

pub fn within_rect(
    corner1: Vec3,
    corner2: Vec3,
    corner3: Vec3,
    corner4: Vec3,
    tree: Res<NNTree>,
) -> Vec<(Vec3, Option<Entity>)> {
    let xs = [corner1.x, corner2.x, corner3.x, corner4.x];
    let ys = [corner1.y, corner2.y, corner3.y, corner4.y];
    let zs = [corner1.z, corner2.z, corner3.z, corner4.z];

    let tri1 = Triangle3d {
        vertices: [
            Vec3 {
                x: corner1.x,
                y: 0.,
                z: corner1.z,
            },
            Vec3 {
                x: corner2.x,
                y: 0.,
                z: corner2.z,
            },
            Vec3 {
                x: corner3.x,
                y: 0.,
                z: corner3.z,
            },
        ],
    };

    let tri2 = Triangle3d {
        vertices: [
            Vec3 {
                x: corner1.x,
                y: 0.,
                z: corner1.z,
            },
            Vec3 {
                x: corner3.x,
                y: 0.,
                z: corner3.z,
            },
            Vec3 {
                x: corner4.x,
                y: 0.,
                z: corner4.z,
            },
        ],
    };

    let loc1 = Vec3 {
        x: xs.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        y: ys.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        z: zs.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
    } - Dir3::Y.as_vec3();

    let loc2 = Vec3 {
        x: xs.iter().fold(f32::INFINITY, |a, &b| a.max(b)),
        y: ys.iter().fold(f32::INFINITY, |a, &b| a.max(b)),
        z: zs.iter().fold(f32::INFINITY, |a, &b| a.max(b)),
    } + Dir3::Y.as_vec3();

    tree.within(loc1, loc2)
        .into_iter()
        .filter(|(pos, _)| point_in_triangle(tri1, pos) || point_in_triangle(tri2, pos))
        .collect()
}
