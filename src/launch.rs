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

use crate::ui::GameState;

/// Bench zoom expressed in `RtsCamera` units, where 0.0 = `height_max`
/// (30 km) and 1.0 = `height_min` (2 m). 0.99 lerps to ~302 m: the
/// `ZOOM_ANCHOR_M` altitude the zoom-speed curve treats as the legacy feel.
const BENCH_ZOOM: f32 = 0.99;
const BENCH_DURATION_SECS: f32 = 30.0;
pub const DEFAULT_BOIDS: usize = 99 * 99;

/// Parsed command-line configuration, inserted before the plugins so every
/// startup system (world spawn, sky, camera, terrain streaming) can read it.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct LaunchConfig {
    /// Skip the main menu, pin the camera, and exit after `duration`.
    pub bench: bool,
    /// Bench-only: how long to stay in the playing state before `AppExit`.
    pub duration: Duration,
    /// Bench-only: camera zoom pinned for the whole run (0.0 high, 1.0 low).
    pub zoom: f32,
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
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self {
            bench: false,
            duration: Duration::from_secs_f32(BENCH_DURATION_SECS),
            zoom: BENCH_ZOOM,
            boids: DEFAULT_BOIDS,
            shadows: true,
            terrain: true,
            // Bevy 0.19 refilters atmosphere LUTs and environment cubemaps
            // every frame; keep the expensive stack opt-in until upstream
            // lands on-demand regeneration.
            atmosphere: false,
            environment_map: false,
            bloom: true,
        }
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

/// Parses `--bench`, `--secs <s>`, `--zoom <z>`, `--boids <n>`,
/// `--shadows|--no-shadows`, `--terrain|--no-terrain`,
/// `--atmosphere|--no-atmosphere`, `--env-map|--no-env-map`, and
/// `--bloom|--no-bloom`. Later flags win. Unrelated arguments are ignored so
/// other tooling can pass through.
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
            "--zoom" => {
                let parsed: f32 = value(args, &mut i, flag, inline_value)?
                    .parse()
                    .map_err(|_| format!("{flag}: value is not a number"))?;
                if !(0.0..=1.0).contains(&parsed) {
                    return Err(format!("{flag} must be within 0.0..=1.0, got {parsed}"));
                }
                config.zoom = parsed;
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
        controls.enabled = false;
    }
    *pinned = true;
    info!(
        "bench: zoom {:.2}, duration {:.0}s, {} boids, shadows {}, terrain {}, atmosphere {}, env map {}, bloom {}",
        config.zoom,
        config.duration.as_secs_f32(),
        config.boids,
        config.shadows,
        config.terrain,
        config.atmosphere,
        config.environment_map,
        config.bloom
    );
}

/// Play for the configured duration, then exit gracefully — a clean
/// `AppExit`, not a killed process, is what lets the chrome trace writer
/// flush its file.
fn exit_bench_when_elapsed(
    config: Res<LaunchConfig>,
    time: Res<Time>,
    mut elapsed: Local<Duration>,
    mut frames: Local<u32>,
    mut app_exit: MessageWriter<AppExit>,
) {
    *elapsed += time.delta();
    *frames += 1;
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
