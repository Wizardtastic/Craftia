//! `voxel-app` — executable entry point for the voxel sandbox.
//!
//! Boots the engine with configuration from `config.toml` (if present) merged
//! with CLI flags. For automated verification, pass `--capture <frames>` to
//! render that many frames, save a screenshot, and exit.

use std::path::{Path, PathBuf};

use clap::Parser;
use voxel_engine::run;
use voxel_engine::settings::GameSettings;

/// Voxel sandbox engine — a custom Vulkan-based voxel game.
#[derive(Parser, Debug)]
#[command(name = "voxel", version, about)]
struct Cli {
    /// Capture N frames and save a screenshot, then exit.
    #[arg(long)]
    capture: Option<usize>,

    /// World generation seed.
    #[arg(long)]
    seed: Option<i32>,

    /// Enable Vulkan validation layers.
    #[arg(long)]
    validation: bool,

    /// Disable VSync.
    #[arg(long)]
    no_vsync: bool,

    /// Path to config file (default: config.toml).
    #[arg(long, default_value = "config.toml")]
    config: String,

    /// Window width in pixels.
    #[arg(long)]
    width: Option<u32>,

    /// Window height in pixels.
    #[arg(long)]
    height: Option<u32>,

    /// Start in fullscreen mode.
    #[arg(long)]
    fullscreen: bool,

    /// Enable debug overlay on startup.
    #[arg(long)]
    debug: bool,
    /// MSAA sample count (1, 2, 4, or 8).
    #[arg(long, value_name = "N")]
    msaa: Option<u32>,
    /// Disable hardware occlusion culling.
    #[arg(long)]
    no_occlusion_culling: bool,
    /// Capture-run camera position override "x,y,z" (teleports + flying).
    #[arg(long, value_name = "x,y,z")]
    campos: Option<String>,
    /// Capture-run camera yaw/pitch override "yaw,pitch" in radians.
    #[arg(long, value_name = "yaw,pitch")]
    camrot: Option<String>,
}

/// Parse a comma-separated list of f32 values, e.g. "10.5,64,-3.25".
fn parse_float_list(s: &str, expected: usize) -> Option<Vec<f32>> {
    let v: Vec<f32> = s
        .split(',')
        .filter_map(|p| p.trim().parse::<f32>().ok())
        .collect();
    if v.len() == expected {
        Some(v)
    } else {
        None
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    let cli = Cli::parse();

    // Load config file (from --config path or default).
    let settings = GameSettings::load(Path::new(&cli.config));

    // Build EngineConfig via the canonical `GameSettings::to_engine_config()`
    // mapping (the same helper that `EngineConfig::default()` delegates to),
    // then layer CLI overrides on top.
    let mut config = settings.to_engine_config();
    if let Some(seed) = cli.seed {
        config.seed = seed;
    }
    if let Some(w) = cli.width {
        config.window_size.0 = w;
    }
    if let Some(h) = cli.height {
        config.window_size.1 = h;
    }
    config.capture_after_frames = cli.capture;
    config.exit_after_capture = cli.capture.is_some();
    if let Some(pos) = cli.campos.as_deref().and_then(|s| parse_float_list(s, 3)) {
        config.capture_cam_pos = Some([pos[0], pos[1], pos[2]]);
    }
    if let Some(rot) = cli.camrot.as_deref().and_then(|s| parse_float_list(s, 2)) {
        config.capture_cam_rot = Some((rot[0], rot[1]));
    }
    config.fullscreen = cli.fullscreen;
    // Track the config file so the file-watcher can react to edits.
    config.config_path = PathBuf::from(&cli.config);

    // CLI-side toggles for renderer / debug overlay.
    if cli.validation {
        config.render.validation = true;
    }
    if cli.no_vsync {
        config.render.vsync = false;
    }
    if let Some(msaa) = cli.msaa {
        config.render.msaa_samples = msaa;
    }
    if cli.no_occlusion_culling {
        config.render.occlusion_culling = false;
    }
    if cli.debug {
        // Set debug overlay on via config — the engine reads this
        // from the config. For now, we'll just pass it through.
        // The engine can check config.debug or a separate field.
    }
    if cli.fullscreen {
        log::info!("starting in borderless fullscreen");
    }

    log::info!("starting voxel engine (seed={})", config.seed);
    run(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_verify() {
        Cli::command().debug_assert();
    }

    #[test]
    fn cli_defaults() {
        let cli = Cli::try_parse_from(["voxel"]).unwrap();
        assert!(cli.capture.is_none());
        assert!(cli.seed.is_none());
        assert!(!cli.validation);
        assert!(!cli.no_vsync);
        assert_eq!(cli.config, "config.toml");
        assert!(cli.width.is_none());
        assert!(cli.height.is_none());
        assert!(!cli.fullscreen);
        assert!(!cli.debug);
    }

    #[test]
    fn cli_with_args() {
        let cli = Cli::try_parse_from([
            "voxel",
            "--capture",
            "180",
            "--seed",
            "42",
            "--validation",
            "--no-vsync",
            "--width",
            "1920",
            "--height",
            "1080",
            "--fullscreen",
            "--debug",
        ])
        .unwrap();
        assert_eq!(cli.capture, Some(180));
        assert_eq!(cli.seed, Some(42));
        assert!(cli.validation);
        assert!(cli.no_vsync);
        assert_eq!(cli.width, Some(1920));
        assert_eq!(cli.height, Some(1080));
        assert!(cli.fullscreen);
        assert!(cli.debug);
    }

    #[test]
    fn cli_custom_config() {
        let cli = Cli::try_parse_from(["voxel", "--config", "my_config.toml"]).unwrap();
        assert_eq!(cli.config, "my_config.toml");
    }
}
