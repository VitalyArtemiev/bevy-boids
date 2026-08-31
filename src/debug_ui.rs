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
use bevy::ecs::component::Mutable;
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
    // Exponentially smoothed FPS readout.
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

            tuned(&mut kin, ui, "Kinematics", true, |ui, kin: &mut KinematicsTuning| {
                let mut changed = false;
                changed |= slider(
                    ui,
                    &mut kin.max_velocity_mps,
                    0.0..=50.0,
                    "max velocity (m/s)",
                )
                .changed();
                changed |= slider(
                    ui,
                    &mut kin.brownian_velocity_mps,
                    0.0..=0.5,
                    "brownian jitter (m/s)",
                )
                .changed();
                changed |= slider(
                    ui,
                    &mut kin.max_acceleration_mpss,
                    0.0..=50.0,
                    "max acceleration (m/s²)",
                )
                .changed();
                changed |= slider(
                    ui,
                    &mut kin.deceleration_time_sec,
                    0.1..=5.0,
                    "deceleration time (s)",
                )
                .changed();
                changed |= reset(ui, kin);
                changed
            });

            tuned(&mut boid, ui, "Boid", false, |ui, boid: &mut BoidTuning| {
                let mut changed = false;
                changed |=
                    slider(ui, &mut boid.repel_coef, 0.0..=0.5, "repel coefficient").changed();
                changed |= slider(
                    ui,
                    &mut boid.obstacle_interaction_radius_m,
                    0.5..=5.0,
                    "obstacle radius (m)",
                )
                .changed();
                changed |=
                    slider(ui, &mut boid.bob_amplitude_m, 0.0..=0.5, "bob amplitude (m)")
                        .changed();
                changed |= slider(
                    ui,
                    &mut boid.bob_freq_coef,
                    0.0..=1.0,
                    "bob frequency coefficient",
                )
                .changed();
                changed |= slider(
                    ui,
                    &mut boid.bob_freq_min_hz,
                    0.0..=0.5,
                    "bob frequency minimum (Hz)",
                )
                .changed();
                changed |= reset(ui, boid);
                changed
            });

            tuned(&mut form, ui, "Formations", false, |ui, form: &mut FormationTuning| {
                let mut changed = false;
                changed |= slider(ui, &mut form.spacing_m, 0.5..=10.0, "slot spacing (m)")
                    .changed();
                changed |=
                    slider(ui, &mut form.lead_time_sec, 0.0..=60.0, "lead time (s)").changed();
                changed |= slider(
                    ui,
                    &mut form.arrive_tolerance_m,
                    0.0..=10.0,
                    "arrival tolerance (m)",
                )
                .changed();
                changed |= reset(ui, form);
                changed
            });

            tuned(&mut sky, ui, "Sky", false, |ui, sky: &mut SkyTuning| {
                let mut changed = false;
                changed |= slider(
                    ui,
                    &mut sky.sun_elevation_deg,
                    0.0..=89.0,
                    "sun elevation (°)",
                )
                .changed();
                changed |= slider(
                    ui,
                    &mut sky.sun_azimuth_deg,
                    0.0..=360.0,
                    "sun azimuth (°)",
                )
                .changed();
                changed |= ui.checkbox(&mut sky.shadows, "shadow maps").changed();
                changed |= reset(ui, sky);
                changed
            });

            // The camera controls are a component on the camera entity, not
            // a tuning resource, but the don't-flag-changes-without-edits
            // rule is the same.
            let mut controls_changed = false;
            egui::CollapsingHeader::new("Camera")
                .default_open(false)
                .show(ui, |ui| {
                    if let Ok(mut controls) = q_controls.single_mut() {
                        let controls: &mut RtsCameraControls =
                            controls.bypass_change_detection();
                        let mut changed = false;
                        changed |= slider(
                            ui,
                            &mut controls.pan_speed,
                            1.0..=100.0,
                            "pan speed",
                        )
                        .changed();
                        changed |= slider(
                            ui,
                            &mut controls.zoom_sensitivity,
                            0.05..=2.0,
                            "zoom sensitivity",
                        )
                        .changed();
                        controls_changed |= changed;
                    }
                });
            if controls_changed {
                if let Ok(mut controls) = q_controls.single_mut() {
                    controls.set_changed();
                }
            }

            let mut toggles_changed = false;
            egui::CollapsingHeader::new("Toggles")
                .default_open(true)
                .show(ui, |ui| {
                    let config: &mut DebugConfig = config.bypass_change_detection();
                    toggles_changed |= ui
                        .checkbox(&mut config.show_cursor_circle, "cursor ground circle")
                        .changed();
                    toggles_changed |= ui
                        .checkbox(&mut config.show_formation_goals, "formation goal lines")
                        .changed();
                    let lod: &mut LODGuard = lod.bypass_change_detection();
                    toggles_changed |= ui
                        .checkbox(&mut lod.propagate_targets, "LOD target propagation")
                        .changed();
                });
            if toggles_changed {
                config.set_changed();
                lod.set_changed();
            }
        });
    config.open = open;
    Ok(())
}

/// A collapsing section over one tuning resource. Change detection on the
/// resource is bypassed while egui holds `&mut` (which would otherwise flag
/// the resource changed every frame the panel is open — the F3 terrain
/// panel rebuilt the whole world that way) and re-armed only when `body`
/// reports a real edit.
fn tuned<T: Resource<Mutability = Mutable>>(
    res: &mut ResMut<T>,
    ui: &mut egui::Ui,
    header: &str,
    default_open: bool,
    body: impl FnOnce(&mut egui::Ui, &mut T) -> bool,
) {
    let mut changed = false;
    egui::CollapsingHeader::new(header)
        .default_open(default_open)
        .show(ui, |ui| {
            let value: &mut T = res.bypass_change_detection();
            changed = body(ui, value);
        });
    if changed {
        res.set_changed();
    }
}

fn slider(
    ui: &mut egui::Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    label: &str,
) -> egui::Response {
    ui.add(egui::Slider::new(value, range).text(label))
}

/// Restores a tuning resource to its `Default` (the const-derived values).
/// Returns whether it actually changed.
fn reset<T: Default + PartialEq>(ui: &mut egui::Ui, tuning: &mut T) -> bool {
    if ui.button("reset").clicked() && *tuning != T::default() {
        *tuning = T::default();
        return true;
    }
    false
}
