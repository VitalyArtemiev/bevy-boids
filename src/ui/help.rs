//! The Help panel: a gesture-oriented controls reference, opened by the
//! Help buttons on the main and pause menus. The Key Bindings window lists
//! *what* every action is bound to; this panel explains the parts a plain
//! binding list can't show — which inputs are drags versus clicks, which
//! modifiers combine with a drag (additive select, width-fitting
//! frontages), and the F1/F3/F4 panels.

use bevy::prelude::*;
use bevy_egui::egui;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass};

use crate::ui::GameState;
use crate::ui::input::{ActionId, ActionTag, Binding, Bindings, current_binding};

/// Whether the Help window is open. A resource (not egui `Local` state):
/// the toggle buttons live in the main-menu and pause-menu systems.
#[derive(Resource, Default)]
pub struct HelpOpen(pub bool);

pub struct HelpUiPlugin;

impl Plugin for HelpUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HelpOpen>().add_systems(
            EguiPrimaryContextPass,
            help_ui.run_if(in_state(GameState::MainMenu).or_else(in_state(GameState::Paused))),
        );
    }
}

fn help_ui(
    mut contexts: EguiContexts,
    mut open: ResMut<HelpOpen>,
    q_actions: Query<(&ActionTag, &Bindings)>,
    q_bindings: Query<&Binding>,
) -> Result {
    if !open.0 {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    // Current binding labels, resolved up front: the rows below compose
    // them into gestures ("Mouse Left drag", "ControlLeft + Mouse Right
    // drag"), so remaps keep the explanations accurate.
    let bind = |id: ActionId| current_binding(&q_actions, &q_bindings, id).to_string();
    let select = bind(ActionId::Select);
    let additive = bind(ActionId::SelectAdditive);
    let frontage = bind(ActionId::Frontage);
    let adjust_width = bind(ActionId::AdjustWidth);
    let (recall_lo, recall_hi) = (
        bind(ActionId::RecallGroup(1)),
        bind(ActionId::RecallGroup(6)),
    );
    let (assign_lo, assign_hi) = (
        bind(ActionId::AssignGroup(1)),
        bind(ActionId::AssignGroup(6)),
    );
    let pause = bind(ActionId::Pause);
    let debug = bind(ActionId::ToggleDebug);
    let terrain = bind(ActionId::ToggleTerrain);
    let scenes = bind(ActionId::ToggleScenes);

    egui::Window::new("Help")
        .open(&mut open.0)
        .min_width(480.0)
        .show(ctx, |ui| {
            // Camera keys are the RTS rig's fixed controls (fresh_rts_controls),
            // not rebindable actions, so this section is static text.
            ui.heading("Camera");
            controls(ui, "help_camera", |ui| {
                control(ui, "Mouse wheel", "Zoom — the step grows with camera height");
                control(ui, "W A S D", "Pan");
                control(ui, "Q / E  or  Middle drag", "Rotate around the focus");
                control(
                    ui,
                    "Mouse Right drag",
                    "Pan — while nothing is selected",
                );
            });

            ui.add_space(6.0);
            ui.heading("Selection");
            controls(ui, "help_selection", |ui| {
                control(ui, &format!("{select} drag"), "Box select");
                control(
                    ui,
                    &format!("{additive} + {select} drag"),
                    "Add to the current selection",
                );
                control(ui, &format!("{select} click"), "Clear the selection");
            });

            ui.add_space(6.0);
            ui.heading("Orders");
            controls(ui, "help_orders", |ui| {
                control(
                    ui,
                    &format!("{frontage} drag"),
                    "Designate a frontage — press at one end of the line, release at the other; the drag direction sets the facing",
                );
                control(
                    ui,
                    &format!("{adjust_width} + {frontage} drag"),
                    "Fit the formation's columns to the frontage width",
                );
                control(
                    ui,
                    &format!("{frontage} click"),
                    "Radial command menu (walk / run)",
                );
            });

            ui.add_space(6.0);
            ui.heading("Quick groups");
            controls(ui, "help_groups", |ui| {
                control(
                    ui,
                    &format!("{recall_lo} … {recall_hi}"),
                    "Select the group stored in that slot",
                );
                control(
                    ui,
                    &format!("{assign_lo} … {assign_hi}"),
                    "Store the current selection in that slot",
                );
            });

            ui.add_space(6.0);
            ui.heading("Menus & panels");
            controls(ui, "help_menus", |ui| {
                control(
                    ui,
                    &pause,
                    "Pause / resume — with a menu window open, close it instead",
                );
                control(
                    ui,
                    &debug,
                    "Debug panel — tuning sliders, gizmos, freecam toggle",
                );
                control(
                    ui,
                    &terrain,
                    "Terrain panel — world-generation sliders and view modes",
                );
                control(ui, &scenes, "Scene picker — test scenes and world reset");
            });

            ui.add_space(8.0);
            ui.label("Inputs above follow your remaps — rebind them in Options → Key bindings.");
        });
    Ok(())
}

/// One section's two-column "input — what it does" grid.
fn controls(ui: &mut egui::Ui, id: &str, body: impl FnOnce(&mut egui::Ui)) {
    egui::Grid::new(id)
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, body);
}

/// One grid row: the input in monospace, the explanation beside it.
fn control(ui: &mut egui::Ui, key: &str, tip: &str) {
    ui.monospace(key);
    ui.label(tip);
    ui.end_row();
}
