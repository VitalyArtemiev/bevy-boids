//! Debug panel: egui sliders over the per-domain tuning resources plus
//! gizmo/LOD/render toggles, toggled with F1 while playing.
//!
//! Adding a slider for a new value is three steps: add the field to the
//! owning module's `*Tuning` resource, read it from the system, and add one
//! `Slider::new` line in the matching section here.

use crate::boid::BoidTuning;
use crate::formations::{FormationTuning, LODGuard};
use crate::kinematics::KinematicsTuning;
use crate::sky::SkyTuning;
use crate::ui::GameState;
use bevy::prelude::*;
use bevy_egui::egui;
use bevy_egui::input::egui_wants_any_keyboard_input;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass};
use bevy_rts_camera::RtsCameraControls;

/// Debug-only view state. Consumers (`draw_cursor`,
/// `dispatch_formation_goals`) early-return on their toggle.
#[derive(Resource, Debug)]
pub struct DebugConfig {
    /// Debug panel visible (F1).
    pub open: bool,
    /// Ground circle under the cursor (`draw_cursor`).
    pub show_cursor_circle: bool,
    /// Formation steering lines (`dispatch_formation_goals`).
    pub show_formation_goals: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            open: false,
            show_cursor_circle: true,
            show_formation_goals: true,
        }
    }
}

pub struct DebugUiPlugin;

impl Plugin for DebugUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DebugConfig>()
            .add_systems(
                EguiPrimaryContextPass,
                debug_panel.run_if(in_state(GameState::Playing)),
            )
            .add_systems(
                Update,
                toggle_debug_panel
                    .run_if(in_state(GameState::Playing))
                    .run_if(not(egui_wants_any_keyboard_input)),
            );
    }
}

/// F1 shows/hides the debug panel.
fn toggle_debug_panel(keys: Res<ButtonInput<KeyCode>>, mut config: ResMut<DebugConfig>) {
    if keys.just_pressed(KeyCode::F1) {
        config.open = !config.open;
    }
}

#[allow(clippy::too_many_arguments)]
fn debug_panel(
    mut contexts: EguiContexts,
    mut config: ResMut<DebugConfig>,
    mut kin: ResMut<KinematicsTuning>,
    mut boid: ResMut<BoidTuning>,
    mut form: ResMut<FormationTuning>,
    mut sky: ResMut<SkyTuning>,
    mut lod: ResMut<LODGuard>,
    mut q_controls: Query<&mut RtsCameraControls>,
    time: Res<Time>,
    mut fps: Local<f32>,
) -> Result {
    // Exponentially smoothed FPS; the window title doubles as the readout.
    if time.delta_secs() > 0.0 {
        *fps = fps.mul_add(0.9, 0.1 / time.delta_secs());
    }
    if !config.open {
        return Ok(());
    }

    let ctx = contexts.ctx_mut()?;
    // Local copy: egui's close button and the closure body both want &mut
    // config, so sync back only after the window is drawn.
    let mut open = config.open;
    // Fixed title and pinned min width: egui windows auto-size to content,
    // so a width that changes per frame (FPS in the title) makes the
    // undragged window reposition itself.
    egui::Window::new("Debug")
        .default_width(320.0)
        .min_width(320.0)
        .open(&mut open)
        .show(ctx, |ui| {
            ui.label(format!(
                "{:.0} FPS ({:.1} ms)",
                *fps,
                time.delta_secs() * 1000.0
            ));
            ui.separator();
            egui::CollapsingHeader::new("Kinematics")
                .default_open(true)
                .show(ui, |ui| {
                    slider(
                        ui,
                        &mut kin.max_velocity_mps,
                        0.0..=50.0,
                        "max velocity (m/s)",
                    );
                    slider(
                        ui,
                        &mut kin.brownian_velocity_mps,
                        0.0..=0.5,
                        "brownian jitter (m/s)",
                    );
                    slider(
                        ui,
                        &mut kin.max_acceleration_mpss,
                        0.0..=50.0,
                        "max acceleration (m/s²)",
                    );
                    slider(
                        ui,
                        &mut kin.deceleration_time_sec,
                        0.1..=5.0,
                        "deceleration time (s)",
                    );
                    reset(ui, &mut *kin);
                });

            egui::CollapsingHeader::new("Boid")
                .default_open(false)
                .show(ui, |ui| {
                    slider(ui, &mut boid.repel_coef, 0.0..=0.5, "repel coefficient");
                    slider(
                        ui,
                        &mut boid.obstacle_interaction_radius_m,
                        0.5..=5.0,
                        "obstacle radius (m)",
                    );
                    slider(
                        ui,
                        &mut boid.bob_amplitude_m,
                        0.0..=0.5,
                        "bob amplitude (m)",
                    );
                    slider(
                        ui,
                        &mut boid.bob_freq_coef,
                        0.0..=1.0,
                        "bob frequency coefficient",
                    );
                    slider(
                        ui,
                        &mut boid.bob_freq_min_hz,
                        0.0..=0.5,
                        "bob frequency minimum (Hz)",
                    );
                    reset(ui, &mut *boid);
                });

            egui::CollapsingHeader::new("Formations")
                .default_open(false)
                .show(ui, |ui| {
                    slider(ui, &mut form.spacing_m, 0.5..=10.0, "slot spacing (m)");
                    slider(ui, &mut form.lead_time_sec, 0.0..=60.0, "lead time (s)");
                    slider(
                        ui,
                        &mut form.arrive_tolerance_m,
                        0.0..=10.0,
                        "arrival tolerance (m)",
                    );
                    reset(ui, &mut *form);
                });

            egui::CollapsingHeader::new("Sky")
                .default_open(false)
                .show(ui, |ui| {
                    slider(
                        ui,
                        &mut sky.sun_elevation_deg,
                        0.0..=89.0,
                        "sun elevation (°)",
                    );
                    slider(ui, &mut sky.sun_azimuth_deg, 0.0..=360.0, "sun azimuth (°)");
                    ui.checkbox(&mut sky.shadows, "shadow maps");
                    reset(ui, &mut *sky);
                });

            egui::CollapsingHeader::new("Camera")
                .default_open(false)
                .show(ui, |ui| {
                    if let Ok(mut controls) = q_controls.single_mut() {
                        slider(ui, &mut controls.pan_speed, 1.0..=100.0, "pan speed");
                        slider(
                            ui,
                            &mut controls.zoom_sensitivity,
                            0.05..=2.0,
                            "zoom sensitivity",
                        );
                    }
                });

            egui::CollapsingHeader::new("Toggles")
                .default_open(true)
                .show(ui, |ui| {
                    ui.checkbox(&mut config.show_cursor_circle, "cursor ground circle");
                    ui.checkbox(&mut config.show_formation_goals, "formation goal lines");
                    ui.checkbox(&mut lod.propagate_targets, "LOD target propagation");
                });
        });
    config.open = open;
    Ok(())
}

fn slider(ui: &mut egui::Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, label: &str) {
    ui.add(egui::Slider::new(value, range).text(label));
}

/// Restores a tuning resource to its `Default` (the const-derived values).
fn reset<T: Default>(ui: &mut egui::Ui, tuning: &mut T) {
    if ui.button("reset").clicked() {
        *tuning = T::default();
    }
}
