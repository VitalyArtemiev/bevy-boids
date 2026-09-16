//! The scene picker: one window, two doors — F4 while playing and the
//! "Scenes" button on the main menu. Lists every [`TestScene`] plus a
//! reset of the currently loaded world; each pick sends a
//! [`LoadWorld`] request, and `scene::assemble_world` tears the old world
//! down before the new one spawns. Sliders over [`SceneUnits`] size the
//! scenes at load: units per formation (the formation scenes' marching
//! blocks) and the formation-parade scene's unit count.

use bevy::prelude::*;
use bevy_egui::egui;
use bevy_egui::input::egui_wants_any_keyboard_input;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass};

use crate::scene::{ActiveScene, LoadWorld, SceneUnits, TestScene};
use crate::ui::GameState;
use crate::ui::input::{ActionEvents, ActionId, ActionTag, TriggerState};

/// Whether the scene picker window is open. A resource (not egui `Local`
/// state): the F4 hotkey and the main-menu button both toggle it.
#[derive(Resource, Default)]
pub struct ScenesOpen(pub bool);

pub struct ScenesUiPlugin;

impl Plugin for ScenesUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScenesOpen>()
            .add_systems(
                Update,
                toggle_scenes_menu
                    .run_if(in_state(GameState::Playing))
                    // Don't let F4 leak into egui while a text field has
                    // the keyboard (and vice versa).
                    .run_if(not(egui_wants_any_keyboard_input)),
            )
            .add_systems(
                EguiPrimaryContextPass,
                scenes_menu_ui
                    .run_if(in_state(GameState::MainMenu).or_else(in_state(GameState::Playing))),
            );
    }
}

/// F4 shows/hides the picker while playing.
fn toggle_scenes_menu(
    actions: Query<(&ActionTag, &TriggerState, &ActionEvents)>,
    mut open: ResMut<ScenesOpen>,
) {
    if crate::ui::input::started(&actions, ActionId::ToggleScenes) {
        open.0 = !open.0;
    }
}

/// The picker itself. Clicking a scene loads it (discarding the current
/// world) and, from the main menu, enters play; "Reset current scene"
/// reloads whatever is loaded — the sandbox included. After any click the
/// picker surrenders egui keyboard focus: a clicked button keeps focus
/// otherwise, and while egui wants the keyboard the F1-F4 toggle systems
/// stand down — this menu exists for click-scene-then-keep-iterating, so
/// the binds must come back the instant the click lands.
fn scenes_menu_ui(
    mut contexts: EguiContexts,
    mut open: ResMut<ScenesOpen>,
    active: Res<ActiveScene>,
    state: Res<State<GameState>>,
    mut next_state: ResMut<NextState<GameState>>,
    mut units: ResMut<SceneUnits>,
    mut commands: Commands,
) -> Result {
    if !open.0 {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    let mut clicked = false;
    let mut units_changed = false;
    egui::Window::new("Scenes")
        .open(&mut open.0)
        .min_width(220.0)
        .show(ctx, |ui| {
            ui.label("Loading a scene discards the current world.");
            for scene in TestScene::ALL {
                let current = active.0 == Some(scene);
                let label = if current {
                    format!("{} (loaded)", scene.name())
                } else {
                    scene.name().to_owned()
                };
                if ui.button(label).on_hover_text(scene.tooltip()).clicked() {
                    clicked = true;
                    commands.insert_resource(LoadWorld(Some(scene)));
                    if *state.get() == GameState::MainMenu {
                        next_state.set(GameState::Playing);
                    }
                }
            }
            ui.separator();
            // Sizes the marching blocks in the formation scenes; read once
            // at the next scene load. Bypass-then-re-arm like every panel:
            // the raw `&mut` through `ResMut` would flag the resource
            // changed every frame the window is open.
            let units: &mut SceneUnits = units.bypass_change_detection();
            units_changed |= ui
                .add(
                    egui::Slider::new(&mut units.per_formation, 1..=10_000)
                        .text("units per formation"),
                )
                .on_hover_text("Units per marching block in the formation scenes (cross, braid, clash; default 12 = the historical 4×3). The stage scales with the block. Applies when a scene loads.")
                .changed();
            units_changed |= ui
                .add(
                    egui::Slider::new(&mut units.parade_units, 1..=10_000)
                        .text("parade units"),
                )
                .on_hover_text("Units in the formation-parade scene's single formation (default 20). The course scales with the block. Applies when the scene loads.")
                .changed();
            ui.separator();
            if ui
                .button("Reset current scene")
                .on_hover_text("Reload the current world from scratch — the sandbox included.")
                .clicked()
            {
                clicked = true;
                commands.insert_resource(LoadWorld(active.0));
            }
        });
    if units_changed {
        units.set_changed();
    }
    if clicked {
        // Surrender whichever widget took focus with the click (there is
        // at most one — the clicked button).
        if let Some(id) = ctx.memory(|m| m.focused()) {
            ctx.memory_mut(|m| m.surrender_focus(id));
        }
    }
    Ok(())
}
