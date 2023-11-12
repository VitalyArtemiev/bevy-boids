use bevy::prelude::*;

#[derive(Component,Default)]
pub struct Horse {
    pub(crate) target: Vec3,
}

pub fn avoid_tight_formations(mut query: Query<(&Transform)>){
    //todo: insert waypoints to avoid tight formations, if not blinded
    //see waterloo film
}

pub fn throw_rider() {
    //todo: throw rider if decelerated quickly
}