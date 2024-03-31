use bevy::prelude::*;
use rand::Rng;
use crate::boid::Boid;
use crate::kinematics::Velocity;

pub enum FormationType {
    Random(u32), // Total
    Square(u32, u32), // Total, Index 
    Rect(u32, u32), // Total, Index, Depth
    Circle(u32, u32), // Total, Index
    //Custom(u32, FormationFunction) //I dunno
}
pub type FormationFunction = fn(usize, FormationType) -> Vec2;

// pub fn random_formation(_: usize, ftype: FormationType) -> Vec2 {
//     if let FormationType::Random(total) = ftype {
//         let mut rng = rand::thread_rng();
//         let range = (total as f32).sqrt();
// 
//         return Vec2 {
//             x: rng.gen_range(-range..range),
//             y: rng.gen_range(-range..range)
//         }
//     }
//     Vec2::Default()
// }

pub fn square(_: usize, total: usize) -> Vec2 {
    let mut rng = rand::thread_rng();
    let range = (total as f32).sqrt();

    Vec2 {
        x: rng.gen_range(-range..range),
        y: rng.gen_range(-range..range)
    }
}


#[derive(Component)]

pub struct Formation {
    width: f32,
    depth: f32,
}
// 
// impl Default for Formation {
//     fn default() -> Self {
//         Formation { width: 20, depth: 5 }
//         
//     }
// }

#[derive(Component)]
pub struct FormationMember {
    formation: Entity,
    index_number: u32,
}

pub fn form_up(
    mut q_boids: Query<(&mut Boid, &FormationMember)>,
    mut q_formations: Query<(&Formation)>,
) {

    for (mut boid, member) in &mut q_boids {
        let _ = q_formations.get(member.formation);
    }
}