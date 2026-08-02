//! Asset hot-reload: watches shader / texture / config directories on disk and
//! dispatches events via a `mpsc::Receiver` for the renderer (and engine) to
//! consume at the start of each frame.
//!
//! Lifecycle: `FileWatcher::new` spawns a background thread that owns a
//! `notify::RecommendedWatcher`. The thread waits on a `shutdown` channel
//! until `FileWatcher` is dropped, at which point it sends `()` and joins.
//!
//! Use [`compile_shader`] to compile a `.vert`/`.frag` source file to SPIR-V
//! at runtime — the renderer uses this to perform live shader recompilation
//! triggered by `HotReloadEvent::ShaderChanged`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};

/// Events emitted by [`FileWatcher`] for the renderer to react to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HotReloadEvent {
    /// A shader source file (`*.vert` / `*.frag`) under `shader_dir` changed.
    /// `name` is the short filename (e.g. `"chunk.vert"`).
    ShaderChanged(String),
    /// A texture atlas asset (any `*.png` or `textures.toml` under the
    /// textures directory) changed.
    TextureAtlasChanged,
    /// A texture pack `.zip` file was added, removed, or modified.
    TexturePackChanged,
    /// The config file changed.
    ConfigChanged,
}

/// Cap on dedupe-map entries. When full, the oldest entry by timestamp is
/// evicted before inserting a new key. Bounds memory for long dev sessions
/// to ~32 paths (one per watched shader file plus the atlas/config slots).
const MAX_DEDUPE_ENTRIES: usize = 32;

/// Background file watcher. Drop to tear down the thread.
pub struct FileWatcher {
    _thread: std::thread::JoinHandle<()>,
    shutdown: mpsc::Sender<()>,
    pub receiver: mpsc::Receiver<HotReloadEvent>,
}

impl FileWatcher {
    /// Watch `shader_dir` recursively for `*.vert`/`*.frag`, `textures_dir`
    /// recursively for `*.png`/`textures.toml`, and `config_path` (a file).
    /// Events are coalesced: identical (kind, name) pairs are dropped if they
    /// fire more than once within 100ms to avoid stampede reloads from
    /// editors that write in bursts.
    pub fn new(
        shader_dir: Option<PathBuf>,
        textures_dir: Option<PathBuf>,
        texture_packs_dir: Option<PathBuf>,
        config_path: PathBuf,
    ) -> Self {
        let (event_tx, event_rx) = mpsc::channel::<HotReloadEvent>();
        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("voxel-file-watcher".into())
            .spawn(move || {
                run_watcher_thread(shader_dir, textures_dir, texture_packs_dir, config_path, event_tx, shutdown_rx);
            })
            .expect("failed to spawn file-watcher thread");
        Self {
            _thread: thread,
            shutdown: shutdown_tx,
            receiver: event_rx,
        }
    }
}

impl Drop for FileWatcher {
    fn drop(&mut self) {
        // Tell the thread to exit. The thread holds the watcher until it
        // sees this signal, then drops it (which closes notify handles).
        let _ = self.shutdown.send(());
    }
}

fn run_watcher_thread(
    shader_dir: Option<PathBuf>,
    textures_dir: Option<PathBuf>,
    texture_packs_dir: Option<PathBuf>,
    config_path: PathBuf,
    event_tx: mpsc::Sender<HotReloadEvent>,
    shutdown_rx: mpsc::Receiver<()>,
) {
    use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};

    // Clone the path arguments before they're moved into the closure so we
    // can still pass them to `watcher.watch(...)` after construction.
    let shader_watch = shader_dir.clone();
    let textures_watch = textures_dir.clone();
    let packs_watch = texture_packs_dir.clone();
    let config_watch = config_path.clone();

    let result: Result<RecommendedWatcher> = (|| {
        let mut watcher = notify::recommended_watcher({
            // Per-shader dedupe: keep one entry per shader filename so saving
            // different shaders back-to-back both fire. Editors that save in
            // bursts (write-temp -> rename) emit multiple notify events for
            // one logical save; we drop the duplicates per name within 100ms.
            //
            // Atlas + config use single slots (a path-level change is the only
            // signal — repeated saves of the same atlas entry or same config
            // file are coalesced within the window).
            let last_emit: std::sync::Mutex<
                std::collections::HashMap<String, Instant>,
            > = std::sync::Mutex::new(std::collections::HashMap::new());
            move |res: notify::Result<notify::Event>| {
                let event = match res {
                    Ok(e) => e,
                    Err(e) => {
                        log::warn!("notify: {e}");
                        return;
                    }
                };
                if !matches!(
                    event.kind,
                    EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                ) {
                    return;
                }
                for path in &event.paths {
                    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                        continue;
                    };
                    let (key, ev) = if shader_dir
                        .as_ref()
                        .map(|d| path.starts_with(d))
                        .unwrap_or(false)
                        && (name.ends_with(".vert") || name.ends_with(".frag"))
                    {
                        // Per-shader key. Editors that save in bursts emit
                        // multiple notify events for one logical save; the
                        // 100ms window drops duplicates per-name. We use a
                        // String key (one allocation per save) so that the
                        // dedupe map never leaks heap memory.
                        let k = format!("shader:{}", name);
                        (k, Some(HotReloadEvent::ShaderChanged(name.to_string())))
                    } else if textures_dir
                        .as_ref()
                        .map(|d| path.starts_with(d))
                        .unwrap_or(false)
                        && (name.ends_with(".png") || name == "textures.toml")
                    {
                        ("atlas".to_string(), Some(HotReloadEvent::TextureAtlasChanged))
                    } else if texture_packs_dir
                        .as_ref()
                        .map(|d| path.starts_with(d))
                        .unwrap_or(false)
                        && name.ends_with(".zip")
                    {
                        ("texture_pack".to_string(), Some(HotReloadEvent::TexturePackChanged))
                    } else if path == &config_path
                        || path.file_name() == config_path.file_name()
                    {
                        ("config".to_string(), Some(HotReloadEvent::ConfigChanged))
                    } else {
                        // Fallback: skip events for files we don't classify.
                        continue;
                    };
                    let Some(ev) = ev else { continue };
                    let mut map = match last_emit.lock() {
                        Ok(m) => m,
                        Err(_) => continue,
                    };
                    let now = Instant::now();
                    if let Some(prev) = map.get(key.as_str()) {
                        if now.duration_since(*prev) < Duration::from_millis(100) {
                            // Timestamp refresh so a third save outside the
                            // window will fire even though we dropped this one.
                            map.insert(key, now);
                            continue;
                        }
                    }
                    // Cap the dedupe map so long dev sessions don't accumulate
                    // keys forever. The oldest entry (by timestamp) is evicted
                    // before the new key is inserted.
                    if map.len() >= MAX_DEDUPE_ENTRIES {
                        if let Some(oldest_key) = map
                            .iter()
                            .min_by_key(|(_, ts)| **ts)
                            .map(|(k, _)| k.clone())
                        {
                            map.remove(&oldest_key);
                        }
                    }
                    map.insert(key, now);
                    if event_tx.send(ev).is_err() {
                        return;
                    }
                }
            }
        })?;
        // Recursive watch for shaders + textures (so users can keep shaders
        // organized in sub-folders); non-recursive for the config parent dir
        // (and we then add the file itself).
        if let Some(sdir) = shader_watch.as_ref() {
            if sdir.is_dir() {
                if let Err(e) = watcher.watch(sdir, RecursiveMode::Recursive) {
                    log::warn!("file-watcher: failed to watch {}: {e}", sdir.display());
                }
            } else {
                log::warn!(
                    "file-watcher: shader_dir {} is not a directory; skipping watch",
                    sdir.display()
                );
            }
        } else {
            log::info!("file-watcher: shader_dir not configured; skipping shader watch");
        }
        if let Some(tdir) = textures_watch.as_ref() {
            if tdir.is_dir() {
                if let Err(e) = watcher.watch(tdir, RecursiveMode::Recursive) {
                    log::warn!("file-watcher: failed to watch {}: {e}", tdir.display());
                }
            } else {
                log::warn!(
                    "file-watcher: textures_dir {} is not a directory; skipping watch",
                    tdir.display()
                );
            }
        }
        if let Some(pdir) = packs_watch.as_ref() {
            if pdir.is_dir() {
                if let Err(e) = watcher.watch(pdir, RecursiveMode::Recursive) {
                    log::warn!("file-watcher: failed to watch texture_packs_dir {}: {e}", pdir.display());
                }
            } else {
                log::warn!(
                    "file-watcher: texture_packs_dir {} is not a directory; skipping watch",
                    pdir.display()
                );
            }
        }
        // Watch the parent dir so we get create + delete events too.
        let parent = config_watch.parent().unwrap_or_else(|| Path::new("."));
        if parent.is_dir() {
            if let Err(e) = watcher.watch(parent, RecursiveMode::NonRecursive) {
                log::warn!("file-watcher: failed to watch {}: {e}", parent.display());
            }
        }
        // Also watch the file itself if available.
        if config_watch.is_file() {
            if let Err(e) = watcher.watch(&config_watch, RecursiveMode::NonRecursive) {
                log::warn!("file-watcher: failed to watch {}: {e}", config_watch.display());
            }
        }
        Ok(watcher)
    })();

    let watcher = match result {
        Ok(w) => w,
        Err(e) => {
            log::error!("file-watcher: failed to construct notify watcher: {e:?}");
            // Still wait for shutdown so Drop can join cleanly.
            let _ = shutdown_rx.recv();
            return;
        }
    };

    // Block on shutdown signal. Holding the watcher alive here keeps notify
    // running until the consumer drops the FileWatcher.
    let _ = shutdown_rx.recv();
    drop(watcher);
    log::info!("file-watcher: shut down");
}

/// Compile a GLSL source file to SPIR-V bytes using `glslangValidator` from
/// the Vulkan SDK (or `PATH`). Writes to a temp file (next to the source) and
/// reads it back. Returns the raw SPIR-V bytes suitable for `vk::ShaderModule`.
pub fn compile_shader(src: &Path) -> Result<Vec<u8>> {
    let glslang = find_glslang_validator().context("locating glslangValidator")?;
    let src = src.canonicalize().unwrap_or_else(|_| src.to_path_buf());
    if !src.is_file() {
        return Err(anyhow!("shader source not found: {}", src.display()));
    }
    // Use a deterministic temp filename so multiple compiles of the same
    // source in a session don't multiply temp files.
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("shader");
    let tmp_dir = std::env::temp_dir();
    let tmp_out = tmp_dir.join(format!("voxel-shader-{}-{}.spv", stem, std::process::id()));        let output = Command::new(&glslang)
            .arg("-V")
            .arg("-o")
            .arg(&tmp_out)
            .arg(&src)
            .output()
            .with_context(|| format!("failed to spawn glslangValidator ({:?})", glslang))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::warn!("glslangValidator failed for {}: {}", src.display(), stderr);
        return Err(anyhow!(
            "glslangValidator compile error for {}: {}",
            src.display(),
            stderr.trim()
        ));
    }
    let bytes = std::fs::read(&tmp_out).with_context(|| {
        format!(
            "failed to read compiled SPIR-V at {}",
            tmp_out.display()
        )
    })?;
    Ok(bytes)
}

/// Locate `glslangValidator`. Mirrors `build.rs`'s path resolution: prefer
/// `$VULKAN_SDK/Bin/`, then fall back to `PATH`. Returns `Err` (not `panic!`)
/// so the renderer can fail gracefully at runtime.
pub fn find_glslang_validator() -> Result<PathBuf> {
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let candidate = Path::new(&sdk).join("Bin").join("glslangValidator.exe");
        if candidate.is_file() {
            return Ok(candidate);
        }
        let candidate2 = Path::new(&sdk).join("Bin").join("glslangValidator");
        if candidate2.is_file() {
            return Ok(candidate2);
        }
    }
    // PATH fallback (Windows `where`, then POSIX `which`).
    if cfg!(target_os = "windows") {
        if let Ok(out) = Command::new("where").arg("glslangValidator").output() {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout);
                if let Some(line) = s.lines().next() {
                    let p = PathBuf::from(line.trim());
                    if p.is_file() {
                        return Ok(p);
                    }
                }
            }
        }
    } else if let Ok(out) = Command::new("which").arg("glslangValidator").output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            if let Some(line) = s.lines().next() {
                let p = PathBuf::from(line.trim());
                if p.is_file() {
                    return Ok(p);
                }
            }
        }
    }
    Err(anyhow!(
        "glslangValidator not found; install the Vulkan SDK or set VULKAN_SDK"
    ))
}
