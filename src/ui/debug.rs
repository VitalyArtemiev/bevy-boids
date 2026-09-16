//! Debug panel: egui sliders over the per-domain tuning resources plus
//! gizmo/LOD/render toggles, toggled with F1 while playing.
//!
//! Adding a slider for a new value is three steps: add the field to the
//! owning module's `*Tuning` resource, read it from the system, and add one
//! `Slider::new` line in the matching section here. Every control carries a
//! short hover tooltip distilled from the field docs of its tuning resource.

use crate::billboard::BillboardTuning;
use crate::boid::BoidTuning;
use crate::crowd::CrowdTuning;
use crate::formations::{FormationTuning, LODGuard};
use crate::freecam::{CameraMode, Freecam, FREECAM_MAX_SPEED_MPS, FREECAM_MIN_SPEED_MPS};
use crate::kinematics::KinematicsTuning;
use crate::pbd::PbdTuning;
use crate::sky::SkyTuning;
use crate::ui::GameState;
use crate::ui::input::{ActionEvents, ActionId, ActionTag, TriggerState};
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
    /// Kill-switch for the per-boid impostor LOD (`swap_boid_lod`): off
    /// holds every boid on its mesh regardless of distance.
    pub impostor_lod: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            open: false,
            show_cursor_circle: true,
            show_formation_goals: true,
            impostor_lod: true,
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
fn toggle_debug_panel(
    actions: Query<(&ActionTag, &TriggerState, &ActionEvents)>,
    mut config: ResMut<DebugConfig>,
) {
    if crate::ui::input::started(&actions, ActionId::ToggleDebug) {
        config.open = !config.open;
    }
}

#[allow(clippy::too_many_arguments)]
fn debug_panel(
    mut contexts: EguiContexts,
    mut config: ResMut<DebugConfig>,
    mut kin: ResMut<KinematicsTuning>,
    mut boid: ResMut<BoidTuning>,
    mut pbd: ResMut<PbdTuning>,
    mut form: ResMut<FormationTuning>,
    mut sky: ResMut<SkyTuning>,
    mut crowd: ResMut<CrowdTuning>,
    mut billboard: ResMut<BillboardTuning>,
    mut lod: ResMut<LODGuard>,
    mut camera_mode: ResMut<CameraMode>,
    mut q_controls: Query<&mut RtsCameraControls>,
    mut q_freecam: Query<&mut Freecam>,
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

            tuned(
                &mut kin,
                ui,
                "Kinematics",
                true,
                |ui, kin: &mut KinematicsTuning| {
                    let mut changed = false;
                    changed |= slider(
                        ui,
                        &mut kin.max_velocity_mps,
                        0.0..=50.0,
                        "max velocity (m/s)",
                        "Hard ceiling on any boid's speed.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut kin.brownian_velocity_mps,
                        0.0..=0.5,
                        "brownian jitter (m/s)",
                        "Speed slack above the steering cap — keeps idle units fidgeting.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut kin.max_acceleration_mpss,
                        0.0..=50.0,
                        "max acceleration (m/s²)",
                        "Ceiling on steering thrust; also bounds how fast excess speed is shed.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut kin.deceleration_time_sec,
                        0.1..=5.0,
                        "deceleration time (s)",
                        "Arrival braking budget: preferred speed = distance left ÷ this.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut kin.steer_response_sec,
                        0.05..=2.0,
                        "steer response (s)",
                        "Time constant for velocity to settle onto the planned velocity.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut kin.misalign_slowdown,
                        0.0..=1.0,
                        "misalignment slowdown",
                        "Slows preferred speed for off-heading targets (0 = off, 1 = full) — breaks flyby orbits.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut kin.gravity_mpss,
                        0.0..=30.0,
                        "gravity (m/s²)",
                        "Downhill pull along the terrain surface.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut kin.slope_accel_coef,
                        0.0..=3.0,
                        "uphill thrust loss",
                        "Steering thrust lost per unit of uphill grade (floored at 0.2×).",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut kin.slope_cap_coef,
                        0.0..=3.0,
                        "downhill cap gain",
                        "Speed-cap relaxation per unit of downhill grade (ceiling 1.5×).",
                    )
                    .changed();
                    changed |= reset(ui, kin);
                    changed
                },
            );

            tuned(&mut boid, ui, "Boid", false, |ui, boid: &mut BoidTuning| {
                let mut changed = false;
                changed |= slider(
                    ui,
                    &mut boid.bob_amplitude_m,
                    0.0..=0.5,
                    "bob amplitude (m)",
                    "Height of the cosmetic vertical bob.",
                )
                .changed();
                changed |= slider(
                    ui,
                    &mut boid.bob_freq_idle_hz,
                    0.0..=2.0,
                    "bob idle cadence (Hz)",
                    "Sway rate of stationary units; cadences crossfade by speed.",
                )
                .changed();
                changed |= slider(
                    ui,
                    &mut boid.bob_freq_walk_hz,
                    0.0..=4.0,
                    "bob walk cadence (Hz)",
                    "Bob rate at marching speed.",
                )
                .changed();
                changed |= slider(
                    ui,
                    &mut boid.bob_freq_run_hz,
                    0.0..=8.0,
                    "bob run cadence (Hz)",
                    "Bob rate at sprint speed.",
                )
                .changed();
                changed |= reset(ui, boid);
                changed
            });

            tuned(
                &mut pbd,
                ui,
                "PBD collisions",
                false,
                |ui, pbd: &mut PbdTuning| {
                    let mut changed = false;
                    changed |= slider_usize(
                        ui,
                        &mut pbd.iterations,
                        1..=12,
                        "solver iterations",
                        "Jacobi passes per step; 1 plus next frame's fresh positions suffices.",
                    )
                    .changed();
                    changed |= slider_usize(
                        ui,
                        &mut pbd.neighbors,
                        1..=16,
                        "neighbours per boid",
                        "kd-tree candidates per boid — look-ahead beyond the ~6–8 real contacts is what makes blocks braid.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut pbd.friction,
                        0.0..=1.0,
                        "contact friction",
                        "Share of tangential slip each contact removes (cone-clamped by overlap).",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut pbd.hostile_friction_scale,
                        // Above 1.0 it is a deliberate grind multiplier: sticky
                        // enemy contact (shield walls), not just slicker slide.
                        0.0..=5.0,
                        "hostile friction scale",
                        "Friction multiplier for enemy pairs: 0.25 slides off, above 1 grinds like a shield wall.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut pbd.anticipation_horizon_sec,
                        0.0..=5.0,
                        "anticipation horizon (s)",
                        "Look-ahead for friendly pairs; hostile pairs and charging units never anticipate.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut pbd.anticipation_stiffness,
                        0.0..=1.0,
                        "anticipation stiffness",
                        "Per-step strength of the long-range avoidance nudge.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut pbd.braking_keep,
                        0.0..=1.0,
                        "anticipation braking keep",
                        "Kept fraction of the correction's braking part: 0 = pure sidestep, 1 = full brake.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut pbd.candidate_margin_m,
                        0.0..=10.0,
                        "candidate margin (m)",
                        "Extra metres keeping near-touching pairs visible to later solver passes.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut pbd.obstacle_margin_m,
                        0.0..=10.0,
                        "obstacle margin (m)",
                        "Extra metres on the obstacle-cuboid proximity query.",
                    )
                    .changed();
                    changed |= reset(ui, pbd);
                    changed
                },
            );

            tuned(
                &mut form,
                ui,
                "Formations",
                false,
                |ui, form: &mut FormationTuning| {
                    let mut changed = false;
                    changed |= slider(
                        ui,
                        &mut form.spacing_m,
                        0.5..=10.0,
                        "slot spacing (m)",
                        "Metres between neighbouring slots in every formation kind.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut form.lead_time_sec,
                        0.0..=60.0,
                        "lead time (s)",
                        "Marching goal lead distance, as seconds of the slowest member's speed.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut form.arrive_tolerance_m,
                        0.0..=10.0,
                        "arrival tolerance (m)",
                        "Centre-of-mass distance at which a Move order counts as arrived.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut form.slot_soft_radius_m,
                        0.0..=15.0,
                        "slot soft radius (m)",
                        "Slot steering fades inside this radius so collisions win the last stretch (the head-on jam fix). 0 = crisp slots.",
                    )
                    .changed();
                    changed |= reset(ui, form);
                    changed
                },
            );

            tuned(&mut sky, ui, "Sky", false, |ui, sky: &mut SkyTuning| {
                let mut changed = false;
                changed |= slider(
                    ui,
                    &mut sky.sun_elevation_deg,
                    0.0..=89.0,
                    "sun elevation (°)",
                    "Sun height above the horizon.",
                )
                .changed();
                changed |= slider(
                    ui,
                    &mut sky.sun_azimuth_deg,
                    0.0..=360.0,
                    "sun azimuth (°)",
                    "Sun direction around the horizon, measured from +X.",
                )
                .changed();
                changed |= ui
                    .checkbox(&mut sky.shadows, "shadow maps")
                    .on_hover_text("Directional-light shadow maps (~16% of frame time at 10k casters).")
                    .changed();
                changed |= reset(ui, sky);
                changed
            });

            tuned(
                &mut crowd,
                ui,
                "Crowd",
                false,
                |ui, crowd: &mut CrowdTuning| {
                    let mut changed = false;
                    changed |= slider(
                        ui,
                        &mut crowd.density,
                        0.2..=1.2,
                        "crowd density",
                        "Occupancy scale over the army's own row-density curve.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut crowd.glint_rate_hz,
                        0.0..=1.0,
                        "glint rate (Hz)",
                        "Armour-glint frequency per soldier.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut crowd.glint_strength,
                        0.0..=3.0,
                        "glint strength",
                        "Glint brightness multiplier.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut crowd.dust_density,
                        0.0..=3.0,
                        "dust density",
                        "Dust volume density in the churn band at the front line.",
                    )
                    .changed();
                    changed |= reset(ui, crowd);
                    changed
                },
            );

            tuned(
                &mut billboard,
                ui,
                "Billboards",
                false,
                |ui, billboard: &mut BillboardTuning| {
                    let mut changed = false;
                    changed |= slider(
                        ui,
                        &mut billboard.swap_distance_m,
                        5.0..=500.0,
                        "swap distance (m)",
                        "Camera distance beyond which meshes detach for baked impostor quads.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut billboard.hysteresis_m,
                        0.0..=50.0,
                        "hysteresis band (m)",
                        "A boid must come this far back inside the swap distance before its mesh returns — stops threshold thrash.",
                    )
                    .changed();
                    changed |= slider(
                        ui,
                        &mut billboard.brightness,
                        0.0..=3.0,
                        "brightness (×, 1 = mesh parity)",
                        "Light-sum multiplier; 1.0 matches meshes (eyeball knob for the baked normals).",
                    )
                    .changed();
                    changed |= reset(ui, billboard);
                    changed
                },
            );

            // The camera controls are a component on the camera entity, not
            // a tuning resource, but the don't-flag-changes-without-edits
            // rule is the same. No zoom slider here: the crate's
            // `zoom_sensitivity` is deliberately neutralized (`0` in
            // `setup`) — its constant-units step stacks on top of
            // `height_scaled_zoom`; the Options "zoom speed" knob scales
            // our step instead.
            let mut mode_changed = false;
            let mut freecam_changed = false;
            let mut controls_changed = false;
            egui::CollapsingHeader::new("Camera")
                .default_open(false)
                .show(ui, |ui| {
                    let camera_mode = camera_mode.bypass_change_detection();
                    let mut free = *camera_mode == CameraMode::Free;
                    if ui
                        .checkbox(
                            &mut free,
                            "Freecam (WASD fly, Q/E down/up, RMB-drag look, wheel = speed)",
                        )
                        .on_hover_text("Swap the RTS rig for a free-fly camera; focus and zoom survive the round trip.")
                        .changed()
                    {
                        *camera_mode = if free {
                            CameraMode::Free
                        } else {
                            CameraMode::Rts
                        };
                        mode_changed = true;
                    }
                    // The fly-speed slider only exists while freecam is
                    // active (the component is the state; the wheel retunes
                    // the same field). Logarithmic: the clamp range spans
                    // walking pace to continental sweeps.
                    if let Ok(mut freecam) = q_freecam.single_mut() {
                        let freecam: &mut Freecam = freecam.bypass_change_detection();
                        freecam_changed |= ui
                            .add(
                                egui::Slider::new(
                                    &mut freecam.speed_mps,
                                    FREECAM_MIN_SPEED_MPS..=FREECAM_MAX_SPEED_MPS,
                                )
                                .text("fly speed (m/s)")
                                .logarithmic(true),
                            )
                            .on_hover_text("Base freecam speed; Shift sprints ×5.")
                            .changed();
                    }
                    if let Ok(mut controls) = q_controls.single_mut() {
                        let controls: &mut RtsCameraControls = controls.bypass_change_detection();
                        controls_changed |= slider(
                            ui,
                            &mut controls.pan_speed,
                            1.0..=100.0,
                            "pan speed",
                            "RTS camera pan speed, m/s at the reference height.",
                        )
                        .changed();
                    }
                });
            if mode_changed {
                camera_mode.set_changed();
            }
            if freecam_changed {
                if let Ok(mut freecam) = q_freecam.single_mut() {
                    freecam.set_changed();
                }
            }
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
                        .on_hover_text("Gizmo circle on the terrain under the cursor.")
                        .changed();
                    toggles_changed |= ui
                        .checkbox(&mut config.show_formation_goals, "formation goal lines")
                        .on_hover_text("Gizmo lines from each member to its steering goal.")
                        .changed();
                    toggles_changed |= ui
                        .checkbox(&mut config.impostor_lod, "impostor billboards by distance")
                        .on_hover_text("Kill-switch for the distance swap — off holds every boid on its mesh.")
                        .changed();
                    let lod: &mut LODGuard = lod.bypass_change_detection();
                    toggles_changed |= ui
                        .checkbox(&mut lod.propagate_targets, "LOD target propagation")
                        .on_hover_text("Formations push targets to direct members only; off cheaply freezes distant detail.")
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
    tip: &str,
) -> egui::Response {
    ui.add(egui::Slider::new(value, range).text(label))
        .on_hover_text(tip)
}

/// Integer slider (whole counts: solver iterations, neighbour counts).
fn slider_usize(
    ui: &mut egui::Ui,
    value: &mut usize,
    range: std::ops::RangeInclusive<usize>,
    label: &str,
    tip: &str,
) -> egui::Response {
    ui.add(egui::Slider::new(value, range).text(label))
        .on_hover_text(tip)
}

/// Restores a tuning resource to its `Default` (the const-derived values).
/// Returns whether it actually changed.
fn reset<T: Default + PartialEq>(ui: &mut egui::Ui, tuning: &mut T) -> bool {
    if ui
        .button("reset")
        .on_hover_text("Restore this section's default tuning.")
        .clicked()
        && *tuning != T::default()
    {
        *tuning = T::default();
        return true;
    }
    false
}
