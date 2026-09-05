//! UI shell: egui plumbing, the top-level [`GameState`] flow (main menu /
//! playing / paused) and the state menus themselves.
//!
//! All egui systems run in the `EguiPrimaryContextPass` schedule required by
//! `bevy_egui` (not `Update`). Gameplay input is kept out of egui's way with
//! the `egui_wants_*` run conditions applied where the input systems are
//! registered in `main.rs`.

use bevy::app::AppExit;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::settings::{ReflectSettingsGroup, SaveSettingsDeferred, SettingsGroup, SettingsPlugin};
use bevy_egui::egui;
use bevy_egui::input::egui_wants_any_keyboard_input;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use bevy_rts_camera::RtsCameraControls;

use crate::freecam::CameraMode;
use crate::sky::SkyTuning;
use crate::terrain::{LodMode, TerrainTuning};
use std::ops::RangeInclusive;

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

/// User-facing options, persisted by `bevy-settings` (TOML on native,
/// browser localStorage on wasm). This is the single source of truth;
/// [`apply_options`] pushes values into the live resources/components.
/// One-way: the debug panel and code edits to the live values do NOT flow
/// back here, so touching an options control re-asserts the stored value.
#[derive(Resource, Debug, Clone, Copy, Reflect, SettingsGroup)]
// `Resource` pulls in ReflectResource, whose registration also attaches the
// ReflectComponent data bevy-settings' load/save paths unwrap on (resources
// live as components on their own entities in 0.19).
#[reflect(Default, Resource)]
pub struct OptionsSettings {
    /// Directional-light shadow maps (see the cost note on `setup_sky`).
    pub shadows: bool,
    /// Camera bloom post-process.
    pub bloom: bool,
    /// Camera pan speed, m/s at the reference height.
    pub camera_pan_speed: f32,
    /// Zoom speed multiplier over the height-scaled step
    /// (`height_scaled_zoom`): 0.5 halves it, 2.0 doubles it. Must NEVER
    /// feed the crate's `zoom_sensitivity` — the crate's zoom is
    /// deliberately neutralized (`0` in `setup`) because its
    /// constant-height step (7.5 km per notch at any altitude) stacks on
    /// top of ours and dominates near the ground.
    pub camera_zoom_sensitivity: f32,
}

impl Default for OptionsSettings {
    fn default() -> Self {
        Self {
            shadows: true,
            bloom: true,
            camera_pan_speed: 15.0,
            camera_zoom_sensitivity: 1.0,
        }
    }
}

/// Whether the Options window is open. A resource (not egui `Local` state):
/// the toggle buttons live in the main-menu and pause-menu systems.
#[derive(Resource, Default)]
pub struct OptionsOpen(pub bool);

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
            // Register the settings group before SettingsPlugin builds: it
            // scans the type registry once, at build time.
            .register_type::<OptionsSettings>()
            .register_type_data::<OptionsSettings, ReflectSettingsGroup>()
            .add_plugins(SettingsPlugin::new(
                "com.github.VitalyArtemiev.bevy-boids",
            ))
            .init_resource::<OptionsOpen>()
            .init_state::<GameState>()
            .add_systems(
                EguiPrimaryContextPass,
                (
                    main_menu_ui.run_if(in_state(GameState::MainMenu)),
                    pause_menu_ui.run_if(in_state(GameState::Paused)),
                    options_ui.run_if(
                        in_state(GameState::MainMenu).or_else(in_state(GameState::Paused)),
                    ),
                    terrain_tuning_ui
                        .run_if(in_state(GameState::Playing))
                        // Don't let F3 leak into egui while a drag value is
                        // being keyboard-edited (and vice versa).
                        .run_if(not(egui_wants_any_keyboard_input)),
                ),
            )
        .add_systems(
            Update,
            (
                toggle_pause,
                // Runs on startup (the loaded resource's insert counts
                // as a change) and whenever Options edits something.
                apply_options.run_if(resource_changed::<OptionsSettings>),
            )
                .chain(),
        )
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
    mut options_open: ResMut<OptionsOpen>,
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
            if ui.button("Options").clicked() {
                options_open.0 = !options_open.0;
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
    mut options_open: ResMut<OptionsOpen>,
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
                if ui.button("Options").clicked() {
                    options_open.0 = !options_open.0;
                }
                if ui.button("Main Menu").clicked() {
                    next_state.set(GameState::MainMenu);
                }
            });
        });
    Ok(())
}

/// The Options window (reachable from the main menu and the pause menu).
/// Edits [`OptionsSettings`] in place; [`apply_options`] pushes changes to
/// the live values and the save is debounced through `bevy-settings`.
fn options_ui(
    mut contexts: EguiContexts,
    mut options_open: ResMut<OptionsOpen>,
    mut settings: ResMut<OptionsSettings>,
    mut commands: Commands,
) -> Result {
    if !options_open.0 {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    let mut changed = false;
    egui::Window::new("Options")
        .open(&mut options_open.0)
        .show(ctx, |ui| {
            // Same deal as the F3 panel: don't flag the resource changed
            // just because widgets hold `&mut` for a frame.
            let settings: &mut OptionsSettings = settings.bypass_change_detection();
            ui.heading("Graphics");
            changed |= ui.checkbox(&mut settings.shadows, "Shadow maps").changed();
            changed |= ui.checkbox(&mut settings.bloom, "Bloom").changed();
            ui.add_space(6.0);
            ui.heading("Camera");
            changed |= ui
                .add(
                    egui::Slider::new(&mut settings.camera_pan_speed, 1.0..=100.0)
                        .text("pan speed"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut settings.camera_zoom_sensitivity, 0.25..=4.0)
                        .text("zoom speed"),
                )
                .changed();
            ui.add_space(6.0);
            if ui.button("Reset to defaults").clicked() {
                *settings = OptionsSettings::default();
                changed = true;
            }
        });
    if changed {
        settings.set_changed();
        // Debounced write (1 s after the last change) — safe to send every
        // frame of a slider drag.
        commands.queue(SaveSettingsDeferred::default());
    }
    Ok(())
}

/// Pushes [`OptionsSettings`] into the live values. Runs on startup (the
/// loaded resource's insert counts as a change) and whenever the Options
/// window edits something.
fn apply_options(
    settings: Res<OptionsSettings>,
    mut sky: ResMut<SkyTuning>,
    q_camera: Query<(Entity, Has<Bloom>), With<Camera3d>>,
    mut q_controls: Query<&mut RtsCameraControls>,
    mut commands: Commands,
) {
    sky.shadows = settings.shadows;
    for (entity, has_bloom) in &q_camera {
        match (settings.bloom, has_bloom) {
            (true, false) => {
                commands.entity(entity).insert(Bloom::default());
            }
            (false, true) => {
                commands.entity(entity).remove::<Bloom>();
            }
            _ => {}
        }
    }
    for mut controls in &mut q_controls {
        controls.pan_speed = settings.camera_pan_speed;
        // Zoom sensitivity is deliberately NOT pushed: `setup` neutralizes
        // the crate's zoom (`zoom_sensitivity: 0`) because its
        // constant-height step stacks on top of `height_scaled_zoom` and
        // dominates near the ground. The options knob scales our step
        // instead, inside `height_scaled_zoom`.
    }
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
    mut lod_mode: ResMut<LodMode>,
    mut camera_mode: ResMut<CameraMode>,
) -> Result {
    if keys.just_pressed(KeyCode::F3) {
        *shown = !*shown;
    }
    if !*shown {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;
    // Holding `&mut` through the ResMut would flag the resource as changed
    // every frame the window is open, rebuilding the whole terrain at frame
    // rate; bypass the flag and re-arm it only for real edits.
    let mut changed = false;
    // Same bypass for the two debug toggles: `apply_camera_mode` is gated
    // on `resource_changed::<CameraMode>`.
    let mut lod_changed = false;
    let mut mode_changed = false;
    egui::Window::new("Terrain tuning")
        .open(&mut *shown)
        .show(ctx, |ui| {
            let tuning: &mut TerrainTuning = tuning.bypass_change_detection();
            let lod_mode = lod_mode.bypass_change_detection();
            let camera_mode = camera_mode.bypass_change_detection();
            ui.label("World regenerates live while you drag.");
            ui.add_space(6.0);

            ui.heading("Debug");
            ui.horizontal(|ui| {
                ui.label("LOD:");
                let r_distance = ui.selectable_value(
                    lod_mode,
                    LodMode::CameraDistance,
                    "by camera distance (correct)",
                );
                let r_focus = ui.selectable_value(
                    lod_mode,
                    LodMode::FinestAtFocus,
                    "finest at focus (debug)",
                );
                lod_changed |= r_distance.changed() | r_focus.changed();
            });
            let mut free = *camera_mode == CameraMode::Free;
            if ui
                .checkbox(
                    &mut free,
                    "Freecam (WASD fly, Q/E down/up, RMB-drag look, wheel = speed)",
                )
                .changed()
            {
                *camera_mode = if free { CameraMode::Free } else { CameraMode::Rts };
                mode_changed = true;
            }
            ui.add_space(6.0);

            changed |= ui
                .horizontal(|ui| {
                    ui.label("seed:");
                    ui.add(egui::DragValue::new(&mut tuning.seed).speed(1)).changed()
                })
                .inner;

            changed |= section(ui, "Base (rolling hills)", |ui| {
                let mut c = amplitude_wavelength(
                    ui,
                    &mut tuning.base_amplitude_m,
                    &mut tuning.base_wavelength_m,
                    60.0,
                    2000.0,
                );
                c |= slider(ui, &mut tuning.base_octaves, 1..=8, "octaves").changed();
                c
            });

            changed |= section(ui, "Mountains", |ui| {
                let mut c = slider(
                    ui,
                    &mut tuning.ridge_max_m,
                    0.0..=150.0,
                    "ridge height (m)",
                )
                .changed();
                c |= slider(
                    ui,
                    &mut tuning.ridge_wavelength_m,
                    100.0..=3000.0,
                    "ridge wavelength (m)",
                )
                .changed();
                c |= slider(
                    ui,
                    &mut tuning.mask_wavelength_m,
                    200.0..=5000.0,
                    "range spacing (m)",
                )
                .changed();
                c |= amplitude_wavelength(
                    ui,
                    &mut tuning.warp_amplitude_m,
                    &mut tuning.warp_wavelength_m,
                    500.0,
                    3000.0,
                );
                c
            });

            changed |= section(ui, "Erosion (gullies)", |ui| {
                let e = &mut tuning.erosion;
                let mut c =
                    slider(ui, &mut e.strength, 0.0..=0.5, "strength (0 = off)").changed();
                c |= slider(ui, &mut e.gully_weight, 0.0..=1.0, "gully depth").changed();
                c |= slider(ui, &mut e.scale, 0.04..=0.32, "gully size").changed();
                c |= slider(ui, &mut e.detail, 0.1..=4.0, "high-slope spread").changed();
                c |= slider(ui, &mut e.cell_scale, 0.25..=4.0, "cell density").changed();
                c |= slider(
                    ui,
                    &mut e.normalization,
                    0.0..=1.0,
                    "wave normalization",
                )
                .changed();
                c |= slider(ui, &mut e.octaves, 1..=8, "octaves").changed();
                c |= slider(ui, &mut e.lacunarity, 1.5..=3.0, "lacunarity").changed();
                c |= slider(ui, &mut e.gain, 0.1..=0.9, "gain").changed();
                c |= component_sliders(
                    ui,
                    &mut e.rounding,
                    0.0..=4.0,
                    &[
                        "round ridges",
                        "ridge falloff",
                        "round creases",
                        "crease falloff",
                    ],
                );
                c |= component_sliders(
                    ui,
                    &mut e.onset,
                    0.0..=4.0,
                    &["onset slope 1", "onset slope 2", "ridge mask", "ridge fade"],
                );
                c |= component_sliders(
                    ui,
                    &mut e.assumed_slope,
                    0.0..=2.0,
                    &["pretend slope", "pretend blend"],
                );
                c |= slider(
                    ui,
                    &mut e.feature_size_m,
                    5.0..=500.0,
                    "feature size (m)",
                )
                .changed();
                c |= slider(ui, &mut e.fade_fraction, 0.2..=2.0, "fade range").changed();
                c
            });

            ui.add_space(6.0);
            if ui.button("Reset to defaults").clicked() {
                *tuning = TerrainTuning::default();
                changed = true;
            }
        });
    if changed {
        tuning.set_changed();
    }
    if lod_changed {
        lod_mode.set_changed();
    }
    if mode_changed {
        camera_mode.set_changed();
    }
    Ok(())
}

/// A collapsing section whose body reports whether it changed anything.
fn section(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui) -> bool) -> bool {
    egui::CollapsingHeader::new(title)
        .default_open(true)
        .show(ui, body)
        .body_returned
        .unwrap_or(false)
}

fn slider<T: egui::emath::Numeric>(
    ui: &mut egui::Ui,
    value: &mut T,
    range: RangeInclusive<T>,
    label: &str,
) -> egui::Response {
    ui.add(egui::Slider::new(value, range).text(label))
}

/// One slider per array component — the erosion filter's shader vectors
/// (rounding/onset/assumed slope) bind component-wise.
fn component_sliders(
    ui: &mut egui::Ui,
    values: &mut [f32],
    range: RangeInclusive<f32>,
    labels: &[&str],
) -> bool {
    let mut changed = false;
    for (value, label) in values.iter_mut().zip(labels) {
        changed |= ui
            .add(egui::Slider::new(value, range.clone()).text(*label))
            .changed();
    }
    changed
}

/// Paired amplitude + wavelength sliders with metre suffixes. Returns
/// whether either value actually changed.
fn amplitude_wavelength(
    ui: &mut egui::Ui,
    amplitude: &mut f32,
    wavelength: &mut f32,
    amplitude_max: f32,
    wavelength_max: f32,
) -> bool {
    let a = ui.add(egui::Slider::new(amplitude, 0.0..=amplitude_max).text("amplitude (m)"));
    let w = ui.add(egui::Slider::new(wavelength, 10.0..=wavelength_max).text("wavelength (m)"));
    a.changed() | w.changed()
}

/// Escape pauses/resumes; with the Options window open it closes that
/// instead, and in the main menu it just closes the Options window.
fn toggle_pause(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<GameState>>,
    mut next_state: ResMut<NextState<GameState>>,
    mut options_open: ResMut<OptionsOpen>,
) {
    if !keys.just_pressed(KeyCode::Escape) {
        return;
    }
    match state.get() {
        GameState::Playing => {
            if options_open.0 {
                options_open.0 = false;
            } else {
                next_state.set(GameState::Paused);
            }
        }
        GameState::Paused => {
            if options_open.0 {
                options_open.0 = false;
            } else {
                next_state.set(GameState::Playing);
            }
        }
        GameState::MainMenu => options_open.0 = false,
    }
}

fn pause_virtual_time(mut time: ResMut<Time<Virtual>>) {
    time.pause();
}

fn unpause_virtual_time(mut time: ResMut<Time<Virtual>>) {
    time.unpause();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The crate's zoom must stay disarmed: pushing the options knob into
    /// `RtsCameraControls::zoom_sensitivity` re-armed the crate's
    /// constant-units zoom (7.5 km per notch at any altitude) on top of
    /// `height_scaled_zoom` — one scroll out near the ground rocketed the
    /// camera to the sky.
    #[test]
    fn apply_options_keeps_the_crate_zoom_disarmed() {
        let mut app = App::new();
        app.insert_resource(OptionsSettings::default())
            .init_resource::<SkyTuning>()
            .add_systems(Update, apply_options);
        let camera = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                RtsCameraControls {
                    zoom_sensitivity: 0.0,
                    pan_speed: 0.0,
                    ..Default::default()
                },
            ))
            .id();
        app.update(); // the resource insert counts as a change

        let controls = app.world().get::<RtsCameraControls>(camera).unwrap();
        assert_eq!(
            controls.pan_speed, 15.0,
            "the pan speed option no longer applies"
        );
        assert_eq!(
            controls.zoom_sensitivity, 0.0,
            "the crate's constant-units zoom got re-armed on top of height_scaled_zoom"
        );
    }
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
