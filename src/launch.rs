//! Launch options plus the `--bench` scenario: every gameplay-facing flag
//! works with or without benching, because they configure the world at
//! startup — only auto-play, camera pinning, and timed exit are bench-only.
//!
//! Baseline (dev profile, ~1 MP window, AMD 780M, default zoom, 9801
//! boids + shadows): ~51 fps with the atmosphere disabled (the default).
//! Opt-in atmosphere, isolated with no boids/shadows: ~77 fps off, ~53 with
//! sky/LUTs only, ~47 with the 128 px environment cubemap too. An unfiltered
//! `--features chrome` trace distorts runs to ~8 fps; profile with the
//! `RUST_LOG` target filter recorded in SKILL.md.

use std::time::Duration;

use bevy::prelude::*;
use bevy_rts_camera::{RtsCamera, RtsCameraControls, RtsCameraSystemSet};

use crate::scene::TestScene;
use crate::ui::GameState;

/// Bench zoom expressed in `RtsCamera` units, where 0.0 = `height_max`
/// (30 km) and 1.0 = `height_min` (2 m). 0.99 lerps to ~302 m, the legacy
/// pre-continental camera height the bench baseline was measured at.
const BENCH_ZOOM: f32 = 0.99;
const BENCH_DURATION_SECS: f32 = 30.0;
pub const DEFAULT_BOIDS: usize = 99 * 99;

/// Parsed command-line configuration, inserted before the plugins so every
/// startup system (world spawn, sky, camera, terrain streaming) can read it.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct LaunchConfig {
    /// Skip the main menu, pin the camera, and exit after `duration`.
    pub bench: bool,
    /// Bench-only: how long to stay in the playing state before `AppExit`.
    pub duration: Duration,
    /// Bench-only: camera zoom pinned for the whole run (0.0 high, 1.0 low).
    pub zoom: f32,
    /// Bench-only: when set, oscillate zoom between these bounds instead
    /// of pinning (exercises LOD churn; `--zoom-sweep low:high`).
    pub zoom_sweep: Option<(f32, f32)>,
    /// Bench-only: write one screenshot here near the end of the run —
    /// self-contained visual verification with no desktop capture tooling
    /// (works over ssh and on locked/occluded sessions).
    pub shot: Option<std::path::PathBuf>,
    /// Bench-only: park the camera focus at this world XZ instead of the
    /// origin, so terrain verification can look at a specific spot (e.g.
    /// a mountain range).
    pub focus: Option<bevy::math::Vec2>,
    /// How many boids `setup` spawns (0 isolates terrain/sky cost).
    pub boids: usize,
    /// Whether the directional light casts shadows (`SkyTuning.shadows`).
    pub shadows: bool,
    /// Whether terrain tiles stream and render.
    pub terrain: bool,
    /// Whether cameras render the atmosphere scattering stack.
    pub atmosphere: bool,
    /// Whether the atmosphere spawns its per-frame environment-map probe.
    /// Enabling this also enables the atmosphere: a cubemap makes no sense
    /// without the sky it samples.
    pub environment_map: bool,
    /// Whether cameras get the bloom post-process.
    pub bloom: bool,
    /// Flat-plane scene: skip the erosion demo and obstacles, spawn the
    /// plain ground mesh (the render-path bench stand-in terrain) — used
    /// by render-path benches so nothing but the boids costs GPU time.
    pub flat: bool,
    /// Bench scenario: hold every boid on its full mesh, auto-swap off.
    /// Mutually exclusive with [`Self::force_billboards`] (last flag wins).
    pub force_meshes: bool,
    /// Bench scenario: convert every boid to its impostor billboard once,
    /// auto-swap off — the manual counterpart of [`Self::force_meshes`],
    /// so the two paths can be compared at the same camera and scale.
    pub force_billboards: bool,
    /// Present without waiting for vblank (`PresentMode::Immediate`), for
    /// benches on scenes light enough to ride the refresh-rate ceiling —
    /// a vsync-capped 60 fps comparison says nothing.
    pub no_vsync: bool,
    /// Initial sun azimuth, degrees (0 = +X, growing toward +Z — the
    /// `sun_transform` frame). For reproducible lighting in test shots;
    /// the F1 sliders still move it live afterwards.
    pub sun_azimuth_deg: Option<f32>,
    /// Initial sun elevation above the horizon, degrees.
    pub sun_elevation_deg: Option<f32>,
    /// Standalone test scene to launch instead of the normal RTS setup —
    /// `--scene <name>` (see src/scene.rs for the available scenes and
    /// how to add more). Scenes bring their own camera and props; main
    /// setup skips the RTS camera, the grid boids and the obstacles while
    /// one is active.
    pub scene: Option<TestScene>,
    /// Bench-only: pin the camera at this angle from nadir, degrees
    /// (0 = looking straight down). The crowd experiment's design cone is
    /// the upper 30°; this flag makes `--shot` captures reproducible
    /// inside it instead of riding the zoom-dependent dynamic angle.
    pub cam_angle_deg: Option<f32>,
    /// Bake impostor atlases (`preprocess::run`) and exit instead of
    /// launching the game.
    pub preprocess: bool,
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self {
            bench: false,
            duration: Duration::from_secs_f32(BENCH_DURATION_SECS),
            zoom: BENCH_ZOOM,
            zoom_sweep: None,
            shot: None,
            focus: None,
            boids: DEFAULT_BOIDS,
            shadows: true,
            terrain: true,
            // Bevy 0.19 refilters atmosphere LUTs and environment cubemaps
            // every frame; keep the expensive stack opt-in until upstream
            // lands on-demand regeneration.
            atmosphere: false,
            environment_map: false,
            bloom: true,
            flat: false,
            force_meshes: false,
            force_billboards: false,
            no_vsync: false,
            sun_azimuth_deg: None,
            sun_elevation_deg: None,
            scene: None,
            cam_angle_deg: None,
            preprocess: false,
        }
    }
}

impl LaunchConfig {
    /// Whether the given test scene is active (`--scene`).
    pub fn in_scene(&self, scene: TestScene) -> bool {
        self.scene == Some(scene)
    }
}

/// Consumes a valued option: either the `--flag=value` form or the next argv
/// entry. Advances `i` past what it consumed.
fn value(
    args: &[String],
    i: &mut usize,
    flag: &str,
    inline: Option<&str>,
) -> Result<String, String> {
    if let Some(value) = inline {
        *i += 1;
        return Ok(value.to_string());
    }
    let Some(value) = args.get(*i + 1) else {
        return Err(format!("{flag} needs a value"));
    };
    *i += 2;
    Ok(value.clone())
}

/// Parses `--bench`, `--secs <s>`, `--zoom <z>`, `--shot <path>`,
/// `--focus <x,z>`, `--cam-angle <deg>`,
/// `--boids <n>`, `--shadows|--no-shadows`, `--terrain|--no-terrain`,
/// `--atmosphere|--no-atmosphere`, `--env-map|--no-env-map`,
/// `--bloom|--no-bloom`, `--flat`, `--force-meshes`,
/// `--force-billboards`, `--no-vsync`, `--scene <name>` (see
/// [`crate::scene::TestScene`]), and `--preprocess`. Later flags win.
/// Unrelated arguments are ignored so other tooling can pass through.
pub fn parse_launch_args(args: &[String]) -> Result<LaunchConfig, String> {
    let mut config = LaunchConfig::default();

    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let (flag, inline_value) = match arg.split_once('=') {
            Some((flag, value)) => (flag, Some(value)),
            None => (arg, None),
        };
        match flag {
            "--bench" => {
                config.bench = true;
                i += 1;
            }
            "--secs" => {
                let parsed: f32 = value(args, &mut i, flag, inline_value)?
                    .parse()
                    .map_err(|_| format!("{flag}: value is not a number"))?;
                if !parsed.is_finite() || parsed <= 0.0 {
                    return Err(format!("{flag} must be positive, got {parsed}"));
                }
                config.duration = Duration::from_secs_f32(parsed);
            }
            "--zoom-sweep" => {
                let (lo, hi) = args[i + 1]
                    .split_once(':')
                    .ok_or_else(|| "expected low:high".to_string())?;
                let lo: f32 = lo.parse().map_err(|e| format!("bad zoom: {e}"))?;
                let hi: f32 = hi.parse().map_err(|e| format!("bad zoom: {e}"))?;
                config.zoom_sweep = Some((lo, hi));
                i += 1;
            }
            "--zoom" => {
                let parsed: f32 = value(args, &mut i, flag, inline_value)?
                    .parse()
                    .map_err(|_| format!("{flag}: value is not a number"))?;
                if !(0.0..=1.0).contains(&parsed) {
                    return Err(format!("{flag} must be within 0.0..=1.0, got {parsed}"));
                }
                config.zoom = parsed;
            }
            "--shot" => {
                config.shot = Some(value(args, &mut i, flag, inline_value)?.into());
            }
            "--focus" => {
                let raw = value(args, &mut i, flag, inline_value)?;
                let (x, z) = raw.split_once(',').ok_or("--focus wants <x>,<z>")?;
                let parse = |v: &str| {
                    v.trim()
                        .parse::<f32>()
                        .map_err(|_| format!("--focus: {v:?} is not a number"))
                };
                config.focus = Some(bevy::math::Vec2::new(parse(x)?, parse(z)?));
            }
            "--boids" => {
                let raw = value(args, &mut i, flag, inline_value)?;
                config.boids = raw
                    .parse()
                    .map_err(|_| format!("{flag}: {raw:?} is not a whole number"))?;
                if config.boids > DEFAULT_BOIDS {
                    return Err(format!(
                        "{flag} is capped at the spawn grid's {DEFAULT_BOIDS}, got {}",
                        config.boids
                    ));
                }
            }
            "--shadows" | "--no-shadows" => {
                config.shadows = !flag.starts_with("--no-");
                i += 1;
            }
            "--terrain" | "--no-terrain" => {
                config.terrain = !flag.starts_with("--no-");
                i += 1;
            }
            "--atmosphere" | "--no-atmosphere" => {
                config.atmosphere = !flag.starts_with("--no-");
                if !config.atmosphere {
                    config.environment_map = false;
                }
                i += 1;
            }
            "--env-map" | "--no-env-map" => {
                config.environment_map = !flag.starts_with("--no-");
                if config.environment_map {
                    config.atmosphere = true;
                }
                i += 1;
            }
            "--bloom" | "--no-bloom" => {
                config.bloom = !flag.starts_with("--no-");
                i += 1;
            }
            "--flat" => {
                config.flat = true;
                i += 1;
            }
            "--force-meshes" => {
                config.force_meshes = true;
                config.force_billboards = false;
                i += 1;
            }
            "--force-billboards" => {
                config.force_billboards = true;
                config.force_meshes = false;
                i += 1;
            }
            "--no-vsync" => {
                config.no_vsync = true;
                i += 1;
            }
            "--sun-azimuth" => {
                let parsed: f32 = value(args, &mut i, flag, inline_value)?
                    .parse()
                    .map_err(|_| format!("{flag}: value is not a number"))?;
                if !parsed.is_finite() || !(0.0..=360.0).contains(&parsed) {
                    return Err(format!("{flag} must be within 0.0..=360.0, got {parsed}"));
                }
                config.sun_azimuth_deg = Some(parsed);
            }
            "--sun-elevation" => {
                let parsed: f32 = value(args, &mut i, flag, inline_value)?
                    .parse()
                    .map_err(|_| format!("{flag}: value is not a number"))?;
                if !parsed.is_finite() || !(0.0..=89.0).contains(&parsed) {
                    return Err(format!("{flag} must be within 0.0..=89.0, got {parsed}"));
                }
                config.sun_elevation_deg = Some(parsed);
            }
            "--scene" => {
                let name = value(args, &mut i, flag, inline_value)?;
                let Some(scene) = TestScene::from_name(&name) else {
                    let available = TestScene::ALL
                        .iter()
                        .map(TestScene::name)
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(format!(
                        "{flag}: unknown scene {name:?} (available: {available})"
                    ));
                };
                config.scene = Some(scene);
            }
            "--cam-angle" => {
                let parsed: f32 = value(args, &mut i, flag, inline_value)?
                    .parse()
                    .map_err(|_| format!("{flag}: value is not a number"))?;
                // 70° is past the RTS camera's shallowest angle (72° at
                // max zoom); anything more would look at the underside.
                if !parsed.is_finite() || !(0.0..=70.0).contains(&parsed) {
                    return Err(format!("{flag} must be within 0.0..=70.0, got {parsed}"));
                }
                config.cam_angle_deg = Some(parsed);
            }
            "--preprocess" => {
                config.preprocess = true;
                i += 1;
            }
            _ => i += 1,
        }
    }

    Ok(config)
}

/// Run condition for `stream_terrain_tiles`: `--no-terrain` skips tile
/// spawning even in a normal (non-bench) launch.
pub fn launch_terrain_enabled(config: Res<LaunchConfig>) -> bool {
    config.terrain
}

/// Added unconditionally by `main`; it is a no-op without `--bench`.
pub struct BenchPlugin(pub bool);

impl Plugin for BenchPlugin {
    fn build(&self, app: &mut App) {
        if !self.0 {
            return;
        }
        app.insert_state(GameState::Playing).add_systems(
            Update,
            (
                pin_bench_camera.before(RtsCameraSystemSet),
                sweep_bench_zoom.before(RtsCameraSystemSet),
                bench_screenshot,
                exit_bench_when_elapsed,
            )
                .chain()
                .run_if(in_state(GameState::Playing)),
        );
    }
}

/// Freeze the camera at the bench zoom on the first playing frame and cut
/// input out of the loop: pans or scrolls would change the tile set under
/// test and make runs incomparable.
fn pin_bench_camera(
    config: Res<LaunchConfig>,
    mut cameras: Query<(&mut RtsCamera, &mut RtsCameraControls)>,
    mut pinned: Local<bool>,
) {
    if *pinned {
        return;
    }
    for (mut camera, mut controls) in &mut cameras {
        camera.zoom = config.zoom;
        camera.target_zoom = config.zoom;
        if let Some(focus) = config.focus {
            camera.focus.translation.x = focus.x;
            camera.focus.translation.z = focus.y;
            camera.target_focus.translation.x = focus.x;
            camera.target_focus.translation.z = focus.y;
        }
        if let Some(angle) = config.cam_angle_deg {
            // Freeze the view at a fixed angle from nadir: the default
            // dynamic angle swings with zoom (up to 72°), which would put
            // captures outside the crowd experiment's 30° design cone.
            camera.angle = angle.to_radians();
            camera.target_angle = angle.to_radians();
            camera.min_angle = angle.to_radians();
            camera.dynamic_angle = false;
        }
        controls.enabled = false;
    }
    *pinned = true;
    info!(
        "bench: zoom {:.2}, duration {:.0}s, {} boids, shadows {}, terrain {}, atmosphere {}, env map {}, bloom {}, scene {}",
        config.zoom,
        config.duration.as_secs_f32(),
        config.boids,
        config.shadows,
        config.terrain,
        config.atmosphere,
        config.environment_map,
        config.bloom,
        config.scene.map(|scene| scene.name()).unwrap_or("-")
    );
}

/// Bench-only (`--massif <m>`): override the tectonic macro amplitude
/// before the first rebuild, so benches can pin extreme worlds (the F3

/// Bench-only (`--retune <secs>`): flip the massif amplitude between
/// 170 m and 600 m every frame for the given window, exactly like
/// dragging the F3 slider, then hold 600 m and let the world settle —

/// Bench-only (`--zoom-sweep <low>:<high>`): oscillate the zoom
/// sinusoidally across the run, exercising LOD level gating and ring
/// churn the way an interactive zoom does. The period spans a few level
/// flips so screenshots can catch mid-transition states.
fn sweep_bench_zoom(
    config: Res<LaunchConfig>,
    time: Res<Time>,
    mut cameras: Query<&mut RtsCamera>,
) {
    let Some((low, high)) = config.zoom_sweep else {
        return;
    };
    let t = time.elapsed_secs() * std::f32::consts::TAU / 8.0;
    let z = low + (high - low) * (0.5 - 0.5 * t.cos());
    for mut camera in &mut cameras {
        camera.zoom = z;
        camera.target_zoom = z;
    }
}

/// Capture one screenshot near the end of a bench run — tiles streamed,
/// camera settled — and write it to the `--shot` path. The capture is the
/// renderer's own framebuffer readback, so it needs no desktop capture
/// tooling and works regardless of window visibility (a Wayland compositor
/// throttles occluded windows to a crawl, but frames still happen).
fn bench_screenshot(
    config: Res<LaunchConfig>,
    time: Res<Time>,
    mut elapsed: Local<Duration>,
    mut taken: Local<bool>,
    mut commands: Commands,
    cameras: Query<(&Transform, Option<&RtsCamera>), With<Camera3d>>,
) {
    let Some(path) = &config.shot else {
        return;
    };
    *elapsed += time.delta();
    // Late in the run, but with a few seconds of margin so the async
    // readback completes and saves before `AppExit`.
    let at = config.duration.saturating_sub(Duration::from_secs(5));
    if !*taken && *elapsed >= at {
        *taken = true;
        for (transform, rts) in &cameras {
            info!(
                "bench: camera at {:?} focus {:?} zoom {:?}",
                transform.translation,
                rts.map(|r| r.focus.translation),
                rts.map(|r| r.zoom)
            );
        }
        info!("bench: capturing screenshot to {}", path.display());
        commands
            .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
            .observe(bevy::render::view::screenshot::save_to_disk(path.clone()));
    }
}

/// Play for the configured duration, then exit gracefully — a clean
/// `AppExit`, not a killed process, is what lets the chrome trace writer
/// flush its file.
fn exit_bench_when_elapsed(
    config: Res<LaunchConfig>,
    time: Res<Time>,
    mut elapsed: Local<Duration>,
    mut frames: Local<u32>,
    mut recent_window: Local<Duration>,
    mut recent_frames: Local<u32>,
    mut app_exit: MessageWriter<AppExit>,
) {
    *elapsed += time.delta();
    *frames += 1;
    *recent_window += time.delta();
    *recent_frames += 1;
    // Steady-state rate over the trailing window — startup streaming and
    // its atlas re-uploads dominate the whole-run average, which makes
    // short benches lie.
    if *recent_window >= Duration::from_secs(5) {
        info!(
            "bench: last {:.1}s: {:.1} fps",
            recent_window.as_secs_f32(),
            *recent_frames as f32 / recent_window.as_secs_f32()
        );
        *recent_window = Duration::ZERO;
        *recent_frames = 0;
    }
    if *elapsed >= config.duration {
        info!(
            "bench: duration reached, {} frames, {:.1} fps",
            *frames,
            *frames as f32 / elapsed.as_secs_f32()
        );
        app_exit.write(AppExit::Success);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::TestScene;
    use bevy::state::app::StatesPlugin;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn defaults_are_a_normal_atmosphere_off_launch() {
        let config = parse_launch_args(&[]).unwrap();
        assert_eq!(config, LaunchConfig::default());
        assert!(!config.bench);
        assert_eq!(config.boids, DEFAULT_BOIDS);
        assert!(config.shadows);
        assert!(config.terrain);
        assert!(!config.atmosphere);
        assert!(!config.environment_map);
        assert!(config.bloom);
    }

    #[test]
    fn short_flags_apply_outside_bench() {
        let config =
            parse_launch_args(&args(&["--atmosphere", "--no-bloom", "--no-shadows"])).unwrap();
        assert!(!config.bench);
        assert!(config.atmosphere);
        assert!(!config.bloom);
        assert!(!config.shadows);
    }

    #[test]
    fn valued_flags_accept_inline_and_separate_values() {
        let config = parse_launch_args(&args(&[
            "--bench",
            "--secs=10",
            "--zoom",
            "0.5",
            "--boids",
            "100",
        ]))
        .unwrap();
        assert!(config.bench);
        assert_eq!(config.duration, Duration::from_secs_f32(10.0));
        assert_eq!(config.zoom, 0.5);
        assert_eq!(config.boids, 100);
    }

    #[test]
    fn environment_map_implies_atmosphere() {
        let config = parse_launch_args(&args(&["--env-map"])).unwrap();
        assert!(config.atmosphere);
        assert!(config.environment_map);

        let config = parse_launch_args(&args(&["--atmosphere", "--no-atmosphere"])).unwrap();
        assert!(!config.atmosphere);
        assert!(!config.environment_map);
    }

    #[test]
    fn focus_flag_parses_xz() {
        let config = parse_launch_args(&args(&["--focus=100,-50"])).unwrap();
        assert_eq!(config.focus, Some(bevy::math::Vec2::new(100.0, -50.0)));
        assert!(parse_launch_args(&args(&["--focus", "100"])).is_err());
        assert!(parse_launch_args(&args(&["--focus=abc,0"])).is_err());
    }

    #[test]
    fn shot_flag_parses_a_path() {
        let config = parse_launch_args(&args(&["--bench", "--shot=/tmp/shot.png"])).unwrap();
        assert_eq!(
            config.shot.as_deref(),
            Some(std::path::Path::new("/tmp/shot.png"))
        );
        assert_eq!(
            parse_launch_args(&args(&["--shot"])).unwrap_err(),
            "--shot needs a value"
        );
    }

    #[test]
    fn preprocess_flag_switches_to_the_baker() {
        assert!(!parse_launch_args(&[]).unwrap().preprocess);
        assert!(parse_launch_args(&args(&["--preprocess"])).unwrap().preprocess);
    }

    #[test]
    fn crowd_scene_parses_and_cam_angle_validates() {
        // The crowd experiment is a `--scene` value now; both flag forms.
        let config = parse_launch_args(&args(&["--scene=crowd"])).unwrap();
        assert_eq!(config.scene, Some(TestScene::Crowd));
        let config = parse_launch_args(&args(&["--scene", "crowd"])).unwrap();
        assert_eq!(config.scene, Some(TestScene::Crowd));

        // The old standalone flag is gone.
        assert!(parse_launch_args(&args(&["--crowd"])).unwrap().scene.is_none());

        let config = parse_launch_args(&args(&["--bench", "--scene=crowd", "--cam-angle=25"]))
            .unwrap();
        assert_eq!(config.scene, Some(TestScene::Crowd));
        assert_eq!(config.cam_angle_deg, Some(25.0));

        assert!(parse_launch_args(&args(&["--cam-angle", "90"])).is_err());
        assert!(parse_launch_args(&args(&["--cam-angle", "-5"])).is_err());
        assert!(parse_launch_args(&args(&["--cam-angle", "abc"])).is_err());
        assert!(parse_launch_args(&args(&["--cam-angle"])).is_err());
    }

    #[test]
    fn render_force_flags_are_mutually_exclusive_last_wins() {
        assert!(!parse_launch_args(&[]).unwrap().force_meshes);
        assert!(!parse_launch_args(&[]).unwrap().force_billboards);

        let config = parse_launch_args(&args(&["--force-billboards"])).unwrap();
        assert!(config.force_billboards && !config.force_meshes);

        // Giving both keeps only the last one.
        let config =
            parse_launch_args(&args(&["--force-billboards", "--force-meshes"])).unwrap();
        assert!(config.force_meshes && !config.force_billboards);

        assert!(parse_launch_args(&args(&["--flat"])).unwrap().flat);
    }

    #[test]
    fn sun_and_scene_flags_parse_with_validation() {
        let config = parse_launch_args(&args(&["--sun-azimuth", "125", "--sun-elevation", "35"]))
            .unwrap();
        assert_eq!(config.sun_azimuth_deg, Some(125.0));
        assert_eq!(config.sun_elevation_deg, Some(35.0));

        assert!(parse_launch_args(&args(&["--sun-azimuth", "400"])).is_err());
        assert!(parse_launch_args(&args(&["--sun-elevation", "90"])).is_err());
        assert!(parse_launch_args(&args(&["--sun-azimuth", "abc"])).is_err());

        let scene = parse_launch_args(&args(&["--scene", "billboard"])).unwrap();
        assert_eq!(scene.scene, Some(TestScene::Billboard));
        assert!(scene.in_scene(TestScene::Billboard));
        // Inline form too, and unknown names list what's available.
        assert_eq!(
            parse_launch_args(&args(&["--scene=billboard"]))
                .unwrap()
                .scene,
            Some(TestScene::Billboard)
        );
        let err = parse_launch_args(&args(&["--scene", "arena"])).unwrap_err();
        assert!(err.contains("unknown scene") && err.contains("billboard"), "{err}");
        assert!(parse_launch_args(&args(&["--scene"])).is_err());
    }

    #[test]
    fn bad_launch_values_are_rejected() {
        assert!(parse_launch_args(&args(&["--bench", "--secs", "0"])).is_err());
        assert!(parse_launch_args(&args(&["--zoom", "1.5"])).is_err());
        assert!(parse_launch_args(&args(&["--boids", "many"])).is_err());
        assert!(parse_launch_args(&args(&["--boids", "20000"])).is_err());
        assert!(parse_launch_args(&args(&["--secs"])).is_err());
    }

    #[test]
    fn bench_plugin_skips_the_main_menu_only_in_bench_mode() {
        let mut app = App::new();
        app.add_plugins(StatesPlugin);
        app.init_state::<GameState>();
        app.add_plugins(BenchPlugin(true));
        assert_eq!(
            *app.world().resource::<State<GameState>>().get(),
            GameState::Playing
        );

        let mut app = App::new();
        app.add_plugins(StatesPlugin);
        app.init_state::<GameState>();
        app.add_plugins(BenchPlugin(false));
        assert_eq!(
            *app.world().resource::<State<GameState>>().get(),
            GameState::MainMenu
        );
    }
}
