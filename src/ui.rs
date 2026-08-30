//! UI shell: egui plumbing, the top-level [`GameState`] flow (main menu /
//! playing / paused) and the state menus themselves.
//!
//! All egui systems run in the `EguiPrimaryContextPass` schedule required by
//! `bevy_egui` (not `Update`). Gameplay input is kept out of egui's way with
//! the `egui_wants_*` run conditions applied where the input systems are
//! registered in `main.rs`.

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy_egui::egui;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use bevy_rts_camera::RtsCameraControls;

use crate::terrain::TerrainTuning;

/// App-wide flow state. The simulation and gameplay input only run in
/// [`GameState::Playing`]; menus pause both the fixed timestep (via
/// `Time<Virtual>`) and the camera controls.
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GameState {
    #[default]
    MainMenu,
    Playing,
    Paused,
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
            .init_state::<GameState>()
            .add_systems(
                EguiPrimaryContextPass,
                (
                    main_menu_ui.run_if(in_state(GameState::MainMenu)),
                    pause_menu_ui.run_if(in_state(GameState::Paused)),
                    terrain_tuning_ui.run_if(in_state(GameState::Playing)),
                ),
            )
            .add_systems(Update, toggle_pause)
            // Pausing `Time<Virtual>` is what actually stops the fixed
            // timestep from accumulating (gating systems alone would cause a
            // catch-up storm on resume). The camera plugin's systems can't
            // get run conditions from us, but its controls have a kill
            // switch. Exit events run before enter events on a transition,
            // so Paused -> MainMenu still ends up disabled.
            .add_systems(
                OnEnter(GameState::MainMenu),
                (pause_virtual_time, disable_camera_controls),
            )
            .add_systems(
                OnExit(GameState::MainMenu),
                (unpause_virtual_time, enable_camera_controls),
            )
            .add_systems(OnEnter(GameState::Paused), disable_camera_controls)
            .add_systems(OnExit(GameState::Paused), enable_camera_controls);
    }
}

fn main_menu_ui(
    mut contexts: EguiContexts,
    mut next_state: ResMut<NextState<GameState>>,
    mut app_exit: MessageWriter<AppExit>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    let mut root = root_ui(ctx);
    egui::CentralPanel::default().show(&mut root, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() / 3.0);
            ui.heading("bevy-boids");
            ui.add_space(40.0);
            if ui.button("Start").clicked() {
                next_state.set(GameState::Playing);
            }
            // Browsers own the page; there is nothing to exit to.
            #[cfg(not(target_arch = "wasm32"))]
            if ui.button("Quit").clicked() {
                app_exit.write(AppExit::Success);
            }
        });
    });
    Ok(())
}

fn pause_menu_ui(
    mut contexts: EguiContexts,
    mut next_state: ResMut<NextState<GameState>>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    let mut root = root_ui(ctx);
    let frame = egui::Frame {
        fill: egui::Color32::from_black_alpha(160),
        ..Default::default()
    };
    egui::CentralPanel::default()
        .frame(frame)
        .show(&mut root, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() / 3.0);
                ui.heading("Paused");
                ui.add_space(40.0);
                if ui.button("Resume").clicked() {
                    next_state.set(GameState::Playing);
                }
                if ui.button("Main Menu").clicked() {
                    next_state.set(GameState::MainMenu);
                }
            });
        });
    Ok(())
}

/// egui 0.36 panels take a root `&mut Ui` instead of a `&Context`; this is
/// the full-viewport root the bevy_egui examples build by hand.
fn root_ui(ctx: &egui::Context) -> egui::Ui {
    egui::Ui::new(
        ctx.clone(),
        "viewport".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    )
}

/// The terrain tuning panel: every world-generation parameter as a live
/// slider. Mutations propagate through change detection the same frame.
///
/// F3 toggles it here (not in a separate system: `Local` state is
/// per-system, so a toggle elsewhere flips its own flag, never this one) —
/// and the window's X button closes it through the same `Local`.
fn terrain_tuning_ui(
    mut contexts: EguiContexts,
    keys: Res<ButtonInput<KeyCode>>,
    mut shown: Local<bool>,
    mut tuning: ResMut<TerrainTuning>,
) -> Result {
    if keys.just_pressed(KeyCode::F3) {
        *shown = !*shown;
    }
    if !*shown {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    egui::Window::new("Terrain tuning")
        .open(&mut *shown)
        .show(ctx, |ui| {
            let tuning: &mut TerrainTuning = &mut *tuning;
            ui.label("World regenerates live while you drag.");
            ui.add_space(6.0);

            ui.heading("Seed");
            ui.add(egui::DragValue::new(&mut tuning.seed).speed(1));

            ui.heading("Base (rolling hills)");
            amplitude_wavelength(
                ui,
                &mut tuning.base_amplitude_m,
                &mut tuning.base_wavelength_m,
                60.0,
                2000.0,
            );

            ui.heading("Mountains");
            ui.add(
                egui::Slider::new(&mut tuning.ridge_max_m, 0.0..=150.0).text("ridge height (m)"),
            );
            ui.add(
                egui::Slider::new(&mut tuning.ridge_wavelength_m, 100.0..=3000.0)
                    .text("ridge wavelength (m)"),
            );
            ui.add(
                egui::Slider::new(&mut tuning.mask_wavelength_m, 200.0..=5000.0)
                    .text("range spacing (m)"),
            );

            ui.heading("Domain warp (meander)");
            amplitude_wavelength(
                ui,
                &mut tuning.warp_amplitude_m,
                &mut tuning.warp_wavelength_m,
                500.0,
                3000.0,
            );

            ui.heading("Detail (surface texture)");
            amplitude_wavelength(
                ui,
                &mut tuning.detail_amplitude_m,
                &mut tuning.detail_wavelength_m,
                5.0,
                200.0,
            );
        });
    Ok(())
}

/// Paired amplitude + wavelength sliders with metre suffixes.
fn amplitude_wavelength(
    ui: &mut egui::Ui,
    amplitude: &mut f32,
    wavelength: &mut f32,
    amplitude_max: f32,
    wavelength_max: f32,
) {
    ui.add(egui::Slider::new(amplitude, 0.0..=amplitude_max).text("amplitude (m)"));
    ui.add(egui::Slider::new(wavelength, 10.0..=wavelength_max).text("wavelength (m)"));
}

/// Escape toggles between playing and paused.
fn toggle_pause(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<GameState>>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if !keys.just_pressed(KeyCode::Escape) {
        return;
    }
    match state.get() {
        GameState::Playing => next_state.set(GameState::Paused),
        GameState::Paused => next_state.set(GameState::Playing),
        GameState::MainMenu => {}
    }
}

fn pause_virtual_time(mut time: ResMut<Time<Virtual>>) {
    time.pause();
}

fn unpause_virtual_time(mut time: ResMut<Time<Virtual>>) {
    time.unpause();
}

fn disable_camera_controls(mut controls: Query<&mut RtsCameraControls>) {
    for mut c in &mut controls {
        c.enabled = false;
    }
}

fn enable_camera_controls(mut controls: Query<&mut RtsCameraControls>) {
    for mut c in &mut controls {
        c.enabled = true;
    }
}
