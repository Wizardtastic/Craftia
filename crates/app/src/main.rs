//! `voxel-app` — executable entry point for the voxel sandbox.
//!
//! Boots the engine with configuration from `config.toml` (if present) merged
//! with CLI flags. For automated verification, pass `--capture <frames>` to
//! render that many frames, save a screenshot, and exit.

use std::fs::OpenOptions;
use std::io::{self, Write};
use std::panic;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use clap::Parser;
use voxel_engine::run;
use voxel_engine::settings::GameSettings;

// ---- Build-time metadata (set by `crates/app/build.rs`) ----
//
// Read via `env!` so a build-script failure (or a missing var) is a
// hard compile error rather than silent "unknown" strings at runtime
// — keeps the banner honest.
const GIT_SHA: &str = env!("VOXEL_GIT_SHA");
const GIT_DIRTY: &str = env!("VOXEL_GIT_DIRTY");
const BUILD_UNIX: &str = env!("VOXEL_BUILD_UNIX");
const TARGET_TRIPLE: &str = env!("VOXEL_TARGET_TRIPLE");
const RUSTC_VERSION: &str = env!("VOXEL_RUSTC_VERSION");

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
    /// Enable the GPU-driven chunk rendering pipeline (indirect multi-draw).
    #[arg(long)]
    gpu_driven: bool,
    /// Enable the GPU compute chunk mesher (implies --gpu-driven).
    #[arg(long)]
    gpu_meshing: bool,
    /// Distance (in chunks) beyond which chunks are GPU-meshed (implies --gpu-meshing).
    #[arg(long, value_name = "N")]
    gpu_mesh_distance: Option<i32>,
    /// Load radius (in chunks) — how far around the player the world streams.
    #[arg(long, value_name = "N")]
    load_radius: Option<i32>,
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

/// Writer that tees every log record to stderr AND a file. Used as the
/// `env_logger` pipe target so format / filter / RUST_LOG handling is
/// unchanged — we only add a second sink.
///
/// The inner file is guarded by a `Mutex<File>` so concurrent log calls
/// from ECS worker threads don't interleave within a single line.
struct TeeWriter {
    file: Mutex<std::fs::File>,
}

impl Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Best-effort stderr: a broken pipe shouldn't block logging.
        let _ = io::stderr().write_all(buf);
        self.file
            .lock()
            .expect("log file mutex poisoned")
            .write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stderr().flush()?;
        self.file.lock().expect("log file mutex poisoned").flush()
    }
}

/// `Box<dyn Write + Send + Sync>` adapter so `TeeWriter` can be shared
/// between the env_logger pipe target and the panic hook closures.
struct SharedTeeWriter(Arc<Mutex<TeeWriter>>);

impl Write for SharedTeeWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("log file mutex poisoned").write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.lock().expect("log file mutex poisoned").flush()
    }
}

/// Promote the file at `latest` to `previous`, replacing any existing
/// `previous`. Used during `init_logging` so every launch keeps exactly
/// one generation of history (the immediately prior run) for comparison.
///
/// We deliberately do NOT use `std::fs::rename` directly to replace
/// `previous` — on some platforms that is atomic, on others it is an
/// error if the destination exists. Going through `remove_file` first
/// is portable and the racing window (no one else should be touching
/// these files) is irrelevant for a single-user game binary.
///
/// Returns `Ok(true)` if a rotation actually happened (latest existed
/// and was successfully promoted), `Ok(false)` if there was nothing
/// to rotate.
fn rotate_previous_log(latest: &Path, previous: &Path) -> std::io::Result<bool> {
    if !latest.exists() {
        return Ok(false);
    }
    let _ = std::fs::remove_file(previous);
    std::fs::rename(latest, previous)?;
    Ok(true)
}

/// Emit a startup banner identifying the run. The banner is written as
/// the first lines of `logs/latest.log` (and to stderr) so the user can
/// answer "what build is this?" at a glance from any captured log.
fn emit_startup_banner() {
    let dirty = if GIT_DIRTY == "true" { " (dirty)" } else { "" };
    log::info!("============ voxel sandbox startup ============");
    log::info!(
        "  binary:    {} {}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    );
    log::info!("  git:       {}{}", GIT_SHA, dirty);
    log::info!("  rustc:     {}", RUSTC_VERSION);
    log::info!("  target:    {}", TARGET_TRIPLE);
    log::info!("  build unx: {}", BUILD_UNIX);
    log::info!("  pid:       {}", std::process::id());
    log::info!("===============================================");
}

/// Initialize logging so every launch keeps the immediately prior run
/// as `logs/previous.log` and writes the current run's output to
/// `logs/latest.log` (truncated to zero on open).
///
/// Rotation runs BEFORE the new log file is opened, so its outcome
/// can't be surfaced via `log::*!()` yet (no logger has been installed).
/// We capture the result in `rotation_note` and emit it into the file
/// once env_logger is live, just above the startup banner.
fn init_logging() {
    // Best-effort directory creation; failure is OK because OpenOptions
    // will report it and we fall back to stderr.
    let _ = std::fs::create_dir_all("logs");

    let latest_path = PathBuf::from("logs/latest.log");
    let previous_path = PathBuf::from("logs/previous.log");
    let rotation_note: Option<String> = match rotate_previous_log(&latest_path, &previous_path) {
        Ok(true) => Some(format!("rotated prior run -> {}", previous_path.display())),
        Ok(false) => None, // no prior run — first launch on this machine
        Err(e) => {
            // Surface to stderr in case env_logger never starts falling
            // back (e.g., a later file-open failure will already print
            // its own warning), AND keep the message so we can re-log
            // it into the file when the logger does come up.
            eprintln!(
                "warning: could not rotate {} -> {}: {e}",
                latest_path.display(),
                previous_path.display()
            );
            Some(format!("rotation FAILED: {e}"))
        }
    };

    let log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&latest_path);

    match log_file {
        Ok(file) => {
            let tee = Arc::new(Mutex::new(TeeWriter {
                file: Mutex::new(file),
            }));

            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
                .format_timestamp_secs()
                .target(env_logger::Target::Pipe(Box::new(SharedTeeWriter(
                    tee.clone(),
                ))))
                .init();

            // Panics bypass destructors and can otherwise lose the tail
            // of the log. Capture to the file before the default hook
            // prints the panic message, then flush so the trace lands
            // on disk even if the process exits immediately after.
            let tee_for_hook = tee.clone();
            let prev_hook = panic::take_hook();
            panic::set_hook(Box::new(move |info| {
                if let Some(loc) = info.location() {
                    log::error!("PANIC at {}:{}: {}", loc.file(), loc.line(), info);
                } else {
                    log::error!("PANIC: {}", info);
                }
                if let Ok(mut w) = tee_for_hook.lock() {
                    let _ = w.flush();
                }
                prev_hook(info);
            }));

            // Now that env_logger is live, surface the rotation outcome
            // into the file as the very first line so users opening
            // `logs/latest.log` see whether the prior run was archived.
            if let Some(note) = rotation_note {
                log::info!("[rotation] {}", note);
            }

            // First thing into the freshly-opened latest.log so cat'ing
            // the file always answers "what run produced this?".
            emit_startup_banner();
        }
        Err(e) => {
            eprintln!(
                "warning: could not open logs/latest.log: {e}; falling back to stderr-only logging"
            );
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
                .format_timestamp_secs()
                .init();
        }
    }
}

fn main() -> anyhow::Result<()> {
    init_logging();

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
    if cli.gpu_driven {
        config.render.gpu_driven = true;
    }
    if cli.gpu_meshing {
        config.render.gpu_meshing = true;
        config.render.gpu_driven = true;
    }
    if let Some(d) = cli.gpu_mesh_distance {
        config.stream.gpu_mesh_distance = d;
        config.render.gpu_meshing = true;
        config.render.gpu_driven = true;
    }
    if let Some(r) = cli.load_radius {
        config.stream.load_radius = r.max(1);
        // Keep unload radius strictly larger so the wider band doesn't thrash.
        config.stream.unload_radius = (r + 2).max(config.stream.unload_radius);
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

    /// Exercise the file-sink end of `TeeWriter` so we know that each
    /// "launch" really does open with `truncate(true)` (i.e., the file
    /// reflects only the most-recent run, never accumulates). stderr
    /// output is intentionally not inspected here — it's a side effect.
    #[test]
    fn tee_writer_truncates_on_each_open() {
        let dir = std::env::temp_dir().join("voxel-app-log-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("truncate.log");
        let _ = std::fs::remove_file(&path);

        // First "launch".
        {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .unwrap();
            let mut w = TeeWriter {
                file: Mutex::new(f),
            };
            writeln!(w, "first run payload, much longer than the second").unwrap();
            w.flush().unwrap();
        }
        let first_size = std::fs::metadata(&path).unwrap().len();
        assert!(first_size > 0);

        // Second "launch" — the new content must fully replace the old.
        {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .unwrap();
            let mut w = TeeWriter {
                file: Mutex::new(f),
            };
            writeln!(w, "second run").unwrap();
            w.flush().unwrap();
        }
        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            contents, "second run\n",
            "each launch must truncate, not append"
        );
        // Sanity: the file size shrank, proving the old payload is gone.
        assert!(std::fs::metadata(&path).unwrap().len() < first_size);
    }

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

    /// Verify the rotation policy used by `init_logging`:
    /// * launch #1 → writes `latest.log`, no `previous.log` exists yet.
    /// * launch #2 → `latest.log` is promoted to `previous.log`, fresh
    ///   `latest.log` is opened.
    /// * launch #3 → the prior launch's content now lives in
    ///   `previous.log`, and a new `latest.log` is started.
    ///
    /// In every step, exactly one of `latest.log` / `previous.log`
    /// carries the OLD content and the other is the current run.
    #[test]
    fn rotation_keeps_previous_run() {
        use std::io::Write as _;
        let dir = std::env::temp_dir().join("voxel-app-rotate-test");
        let _ = std::fs::create_dir_all(&dir);
        let latest = dir.join("latest.log");
        let previous = dir.join("previous.log");
        let _ = std::fs::remove_file(&latest);
        let _ = std::fs::remove_file(&previous);

        // Launch #1: write some content to latest.log, leave previous alone.
        let write = |path: &Path, line: &str| {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)
                .unwrap();
            let mut w = TeeWriter {
                file: Mutex::new(f),
            };
            writeln!(w, "{}", line).unwrap();
            w.flush().unwrap();
        };
        write(&latest, "first run payload");

        // Launch #2: rotate, write a new latest.log.
        assert!(rotate_previous_log(&latest, &previous).unwrap());
        assert!(
            !latest.exists(),
            "after rotate, latest.log should not exist"
        );
        assert!(previous.exists(), "after rotate, previous.log must exist");
        assert_eq!(
            std::fs::read_to_string(&previous).unwrap(),
            "first run payload\n"
        );
        write(&latest, "second run payload");

        // Launch #3: rotate again. previous.log must now carry the
        // *second* run's content (not the first anymore, because we
        // keep exactly one generation of history).
        assert!(rotate_previous_log(&latest, &previous).unwrap());
        assert_eq!(
            std::fs::read_to_string(&previous).unwrap(),
            "second run payload\n"
        );
        write(&latest, "third run payload");

        // Final state: latest = third-only, previous = second-only.
        assert_eq!(
            std::fs::read_to_string(&latest).unwrap(),
            "third run payload\n"
        );
        assert_eq!(
            std::fs::read_to_string(&previous).unwrap(),
            "second run payload\n"
        );

        // And on a brand-new machine (no latest.log), rotation is a
        // no-op — the function must NOT fail or create a stray file.
        let _ = std::fs::remove_file(&latest);
        assert!(!rotate_previous_log(&latest, &previous).unwrap());
    }
}
