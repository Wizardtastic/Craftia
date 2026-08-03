//! Configuration file loading and management.

//!

//! Settings are loaded from `config.toml` at the workspace root. CLI arguments

//! override file values. If the file is missing, all defaults are used.

use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use voxel_game::input::Action;

use voxel_game::PlayerConfig;

use voxel_render::RendererConfig;

use voxel_world::StreamConfig;

/// Top-level configuration file structure (maps to `config.toml`).

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct GameSettings {
    pub graphics: GraphicsSettings,

    pub world: WorldSettings,

    pub player: PlayerSettings,

    pub keys: KeybindSettings,

    pub debug: DebugSettings,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct GraphicsSettings {
    pub width: u32,

    pub height: u32,

    pub vsync: bool,

    pub fog_distance: f32,

    pub shadow_enabled: bool,

    pub shadow_resolution: u32,

    pub exposure: f32,

    pub vignette_strength: f32,
    /// Directory containing PNG texture overrides (filenames `<tile_index>.png`).
    /// If `None` or the directory doesn't exist, the procedural atlas is used.
    pub textures_dir: Option<String>,
    /// Directory containing texture pack `.zip` files.
    /// Texture packs override base textures on a per-tile basis.
    /// Later packs in alphabetical order win over earlier ones.
    pub texture_packs_dir: Option<String>,

    /// MSAA sample count (1, 2, 4, or 8). Set to 1 to disable MSAA.
    pub msaa_samples: u32,

    /// Enable hardware occlusion culling for chunks.
    pub occlusion_culling: bool,

    /// Enable screen-space ambient occlusion (SSAO).
    pub ssao_enabled: bool,

    /// Enable the Phase-1 GPU-driven chunk rendering pipeline (indirect
    /// multi-draw + compute-shader frustum culling + bindless origins).
    pub gpu_driven: bool,

    /// SSAO sampling radius (world-space distance).
    pub ssao_radius: f32,

    /// SSAO depth bias to prevent self-occlusion.
    pub ssao_bias: f32,

    /// SSAO darkening strength multiplier.
    pub ssao_strength: f32,

    /// Global water surface Y (world units). Feeds the chunk shader's
    /// submerged-terrain caustics path and the cheap wet-edge band
    /// approximation. Defaults to sea level at chunk Y=0.
    pub water_y: f32,
    /// Strength of the cool-tint wet-edge added to non-water blocks just
    /// below the water surface. 0.0 disables.
    pub wet_edge_strength: f32,
    /// Strength of sun caustics on submerged (non-water) terrain. 0.0
    /// disables.
    pub caustics_strength: f32,
    /// Strength of the cherry-yellow chlorophyll backlight on leaves when
    /// the sun hits their back face. 0.0 disables.
    pub leaves_sss_strength: f32,
    /// Master reflection strength in [0, 1] for the water/glass/glossy-tile
    /// reflection paths (chunk shader binding 8). 0.0 disables reflections.
    pub reflection_strength: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct WorldSettings {
    pub seed: i32,

    pub load_radius: i32,

    pub unload_radius: i32,

    pub day_length: f64,

    pub assets_path: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct PlayerSettings {
    pub walk_speed: f32,

    pub sprint_speed: f32,

    pub sneak_speed: f32,

    pub jump_speed: f32,

    pub gravity: f32,

    pub terminal_velocity: f32,

    pub mouse_sensitivity: f32,

    pub fly_speed: f32,
}

/// Keybind settings. Each value is a key name string (e.g. "F3", "T", "Escape").
/// These are matched against `winit::keyboard::KeyCode` names at runtime.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct KeybindSettings {
    pub debug_overlay: String,

    pub wireframe: String,

    pub fly: String,

    pub chat: String,

    pub render_distance_up: String,

    pub render_distance_down: String,

    pub screenshot: String,

    pub pause: String,

    pub block_picker: String,

    pub profiler: String,

    pub chunk_debug: String,

    pub ecs_inspector: String,

    pub pin_unpin: String,

    pub pin_cycle_next: String,

    pub pin_cycle_prev: String,

    pub pin_clear: String,

    pub developer_console: String,

    pub telemetry_dashboard: String,

    pub edit_mode: String,

    pub fullscreen_map: String,
}

/// Resolved mapping from `KeyCode` → `Action`. Built once from `KeybindSettings`.
pub type KeybindMap = HashMap<winit::keyboard::KeyCode, Action>;

impl KeybindSettings {
    /// Parse the string key names into a `KeyCode → Action` map.
    pub fn resolve(&self) -> KeybindMap {
        let mut map = KeybindMap::new();

        if let Some(k) = parse_key(&self.debug_overlay) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::DebugOverlay);
        }

        if let Some(k) = parse_key(&self.wireframe) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::Wireframe);
        }

        if let Some(k) = parse_key(&self.fly) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::Fly);
        }

        if let Some(k) = parse_key(&self.chat) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::Chat);
        }

        if let Some(k) = parse_key(&self.render_distance_up) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::RenderDistanceUp);
        }

        if let Some(k) = parse_key(&self.render_distance_down) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::RenderDistanceDown);
        }

        if let Some(k) = parse_key(&self.screenshot) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::Screenshot);
        }

        if let Some(k) = parse_key(&self.pause) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::Pause);
        }

        if let Some(k) = parse_key(&self.block_picker) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::BlockPicker);
        }

        if let Some(k) = parse_key(&self.profiler) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::Profiler);
        }

        if let Some(k) = parse_key(&self.chunk_debug) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::ChunkDebug);
        }

        if let Some(k) = parse_key(&self.ecs_inspector) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::EcsInspector);
        }

        if let Some(k) = parse_key(&self.pin_unpin) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::PinUnpin);
        }

        if let Some(k) = parse_key(&self.pin_cycle_next) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::PinCycleNext);
        }

        if let Some(k) = parse_key(&self.pin_cycle_prev) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::PinCyclePrev);
        }

        if let Some(k) = parse_key(&self.pin_clear) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::PinClear);
        }

        if let Some(k) = parse_key(&self.developer_console) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::DeveloperConsole);
        }

        if let Some(k) = parse_key(&self.telemetry_dashboard) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::TelemetryDashboard);
        }

        if let Some(k) = parse_key(&self.edit_mode) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::EditMode);
        }

        if let Some(k) = parse_key(&self.fullscreen_map) {
            if map.contains_key(&k) {
                log::warn!(
                    "duplicate keybind: key {:?} bound to multiple actions, last one wins",
                    k
                );
            }

            map.insert(k, Action::FullscreenMap);
        }

        map
    }
}

/// Resolve the workspace-root `shaders/` directory relative to this crate's
/// manifest dir. Used by `GameSettings::to_renderer_config` to populate
/// `RendererConfig::shader_dir` so the renderer can locate GLSL source files
/// for runtime hot-reload.
pub fn shader_source_dir() -> Option<std::path::PathBuf> {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let candidate = manifest.join("..").join("..").join("shaders");

    if candidate.is_dir() {
        Some(candidate)
    } else {
        log::debug!(
            "shader source dir not found at {}, hot-reload disabled",
            candidate.display()
        );

        None
    }
}

/// Parse a key name string (e.g. "F3", "T", "Escape", "Space") into a `KeyCode`.
fn parse_key(name: &str) -> Option<winit::keyboard::KeyCode> {
    use winit::keyboard::KeyCode;

    match name {
        "F1" => Some(KeyCode::F1),

        "F2" => Some(KeyCode::F2),

        "F3" => Some(KeyCode::F3),

        "F4" => Some(KeyCode::F4),

        "F5" => Some(KeyCode::F5),

        "F6" => Some(KeyCode::F6),

        "F7" => Some(KeyCode::F7),

        "F8" => Some(KeyCode::F8),

        "F9" => Some(KeyCode::F9),

        "F10" => Some(KeyCode::F10),

        "F11" => Some(KeyCode::F11),

        "F12" => Some(KeyCode::F12),

        "Escape" => Some(KeyCode::Escape),

        "Space" => Some(KeyCode::Space),

        "Enter" => Some(KeyCode::Enter),

        "Backspace" => Some(KeyCode::Backspace),

        "Tab" => Some(KeyCode::Tab),

        "T" => Some(KeyCode::KeyT),

        "Y" => Some(KeyCode::KeyY),

        "B" => Some(KeyCode::KeyB),

        "E" => Some(KeyCode::KeyE),

        "G" => Some(KeyCode::KeyG),

        "H" => Some(KeyCode::KeyH),

        "K" => Some(KeyCode::KeyK),

        "N" => Some(KeyCode::KeyN),

        "O" => Some(KeyCode::KeyO),

        "P" => Some(KeyCode::KeyP),

        "Q" => Some(KeyCode::KeyQ),

        "R" => Some(KeyCode::KeyR),

        "U" => Some(KeyCode::KeyU),

        "V" => Some(KeyCode::KeyV),

        "X" => Some(KeyCode::KeyX),

        "Z" => Some(KeyCode::KeyZ),

        "Backquote" => Some(KeyCode::Backquote),

        "ScrollLock" => Some(KeyCode::ScrollLock),

        "Minus" => Some(KeyCode::Minus),

        "Equal" => Some(KeyCode::Equal),

        "BracketLeft" => Some(KeyCode::BracketLeft),

        "BracketRight" => Some(KeyCode::BracketRight),

        "Backslash" => Some(KeyCode::Backslash),

        "Semicolon" => Some(KeyCode::Semicolon),

        "Quote" => Some(KeyCode::Quote),

        "Comma" => Some(KeyCode::Comma),

        "Period" => Some(KeyCode::Period),

        "Slash" => Some(KeyCode::Slash),

        "0" => Some(KeyCode::Digit0),

        "1" => Some(KeyCode::Digit1),

        "2" => Some(KeyCode::Digit2),

        "3" => Some(KeyCode::Digit3),

        "4" => Some(KeyCode::Digit4),

        "5" => Some(KeyCode::Digit5),

        "6" => Some(KeyCode::Digit6),

        "7" => Some(KeyCode::Digit7),

        "8" => Some(KeyCode::Digit8),

        "9" => Some(KeyCode::Digit9),

        "Up" => Some(KeyCode::ArrowUp),

        "Down" => Some(KeyCode::ArrowDown),

        "Left" => Some(KeyCode::ArrowLeft),

        "Right" => Some(KeyCode::ArrowRight),

        _ => None,
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct DebugSettings {
    pub show_overlay: bool,
}
impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            vsync: true,
            fog_distance: 320.0,
            shadow_enabled: true,
            shadow_resolution: 2048,
            exposure: 0.6,
            vignette_strength: 0.15,
            textures_dir: None,
            texture_packs_dir: None,
            msaa_samples: 4,
            occlusion_culling: true,
            ssao_enabled: true,
            gpu_driven: false,
            ssao_radius: 1.5,
            ssao_bias: 0.025,
            ssao_strength: 1.0,
            // Water + translucency upgrade defaults. Master scalars shipped
            // at modest values so the visual effects show up gently on first
            // boot; users can dial each to taste from the settings UI.
            // water_y matches the standard MC overworld sea level so chunk
            // blocks placed at Y=62 are seen as wet/shaded, and blocks
            // below 62 pick up the submerged caustics tint, all by default.
            water_y: 62.0,
            wet_edge_strength: 0.5,
            caustics_strength: 0.6,
            leaves_sss_strength: 0.65,
            // Reflections default ON at a strong-but-not-mirror level; dial
            // down for weaker water/glass sheen or to 0 to disable.
            reflection_strength: 0.85,
        }
    }
}

impl Default for WorldSettings {
    fn default() -> Self {
        Self {
            seed: 1337,

            load_radius: 6,

            unload_radius: 8,

            day_length: 1200.0,

            assets_path: None,
        }
    }
}

impl Default for PlayerSettings {
    fn default() -> Self {
        let pc = PlayerConfig::default();

        Self {
            walk_speed: pc.walk_speed,

            sprint_speed: pc.sprint_speed,

            sneak_speed: pc.sneak_speed,

            jump_speed: pc.jump_speed,

            gravity: pc.gravity,

            terminal_velocity: pc.terminal_velocity,

            mouse_sensitivity: pc.mouse_sensitivity,

            fly_speed: pc.fly_speed,
        }
    }
}

impl Default for KeybindSettings {
    fn default() -> Self {
        Self {
            debug_overlay: "F3".into(),

            wireframe: "F4".into(),

            fly: "F5".into(),

            chat: "T".into(),

            render_distance_up: "F8".into(),

            render_distance_down: "F9".into(),

            screenshot: "F2".into(),

            pause: "Escape".into(),

            block_picker: "E".into(),

            profiler: "F6".into(),

            chunk_debug: "F7".into(),

            ecs_inspector: "F10".into(),

            pin_unpin: "F11".into(),

            pin_cycle_next: "BracketRight".into(),

            pin_cycle_prev: "BracketLeft".into(),

            pin_clear: "Backslash".into(),

            developer_console: "Backquote".into(),

            telemetry_dashboard: "ScrollLock".into(),

            edit_mode: "X".into(),

            fullscreen_map: "M".into(),
        }
    }
}

/// Macro to copy graphics fields from `GameSettings` into `EngineConfig`,
/// reducing the field-by-field boilerplate. Usage:
/// ```ignore
/// let mut cfg = graphics_fields!(self);
/// cfg.seed = self.world.seed;
/// // ... fill non-graphics fields ...
/// ```
macro_rules! graphics_fields {
    ($settings:expr) => {{
        let g = &$settings.graphics;
        crate::EngineConfig {
            // Graphics fields (copied from GraphicsSettings).
            shadow_enabled: g.shadow_enabled,
            shadow_resolution: g.shadow_resolution,
            exposure: g.exposure,
            vignette_strength: g.vignette_strength,
            ssao_enabled: g.ssao_enabled,
            ssao_radius: g.ssao_radius,
            ssao_bias: g.ssao_bias,
            ssao_strength: g.ssao_strength,
            water_y: g.water_y,
            wet_edge_strength: g.wet_edge_strength,
            caustics_strength: g.caustics_strength,
            leaves_sss_strength: g.leaves_sss_strength,
            reflection_strength: g.reflection_strength,
            window_size: (g.width, g.height),
            // Defaults for non-graphics fields (overwritten by caller).
            seed: 0,
            title: String::new(),
            render: RendererConfig::default(),
            stream: StreamConfig::default(),
            world: WorldSettings::default(),
            player: PlayerConfig::default(),
            capture_after_frames: None,
            capture_path: std::path::PathBuf::new(),
            exit_after_capture: false,
            capture_cam_pos: None,
            capture_cam_rot: None,
            spawn: None,
            day_length: 0.0,
            keybinds: KeybindSettings::default(),
            assets_path: None,
            fullscreen: false,
            profile_cpu_systems: false,
            config_path: std::path::PathBuf::new(),
        }
    }};
}

impl GameSettings {
    /// Load from a TOML file. Returns defaults if the file doesn't exist.
    pub fn load(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(content) => match toml::from_str::<GameSettings>(&content) {
                Ok(settings) => {
                    log::info!("Loaded config from {}", path.display());

                    settings
                }

                Err(e) => {
                    log::warn!("Failed to parse {}: {e}. Using defaults.", path.display());

                    Self::default()
                }
            },

            Err(_) => {
                log::info!("No config file at {}. Using defaults.", path.display());

                Self::default()
            }
        }
    }

    /// Save settings to a TOML file on disk.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(std::io::Error::other)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)?;
        log::info!("Saved config to {}", path.display());
        Ok(())
    }

    /// Convert to the engine's `RendererConfig`.
    pub fn to_renderer_config(&self) -> RendererConfig {
        RendererConfig {
            validation: false,

            clear_color: [0.52, 0.72, 0.95, 1.0],

            vsync: self.graphics.vsync,

            fog_color: [0.62, 0.80, 0.96],

            fog_distance: self.graphics.fog_distance,

            textures_dir: self
                .graphics
                .textures_dir
                .as_ref()
                .map(std::path::PathBuf::from),
            texture_packs_dir: self
                .graphics
                .texture_packs_dir
                .as_ref()
                .map(std::path::PathBuf::from),

            shader_dir: shader_source_dir(),

            particle_softness: 0.3,

            msaa_samples: self.graphics.msaa_samples,

            occlusion_culling: self.graphics.occlusion_culling,

            ssao_enabled: self.graphics.ssao_enabled,

            gpu_driven: self.graphics.gpu_driven,

            gpu_meshing: false,
        }
    }

    /// Convert to the engine's `StreamConfig`.
    pub fn to_stream_config(&self) -> StreamConfig {
        StreamConfig {
            load_radius: self.world.load_radius,

            unload_radius: self.world.unload_radius,

            vertical_half: 3,

            gpu_mesh_distance: 0,

            gen_batch: 24,

            mesh_batch: 24,

            structure_batch: 8,
        }
    }

    /// Convert to the game's `PlayerConfig`.
    pub fn to_player_config(&self) -> PlayerConfig {
        PlayerConfig {
            walk_speed: self.player.walk_speed,

            sprint_speed: self.player.sprint_speed,

            sneak_speed: self.player.sneak_speed,

            jump_speed: self.player.jump_speed,

            gravity: self.player.gravity,

            terminal_velocity: self.player.terminal_velocity,

            mouse_sensitivity: self.player.mouse_sensitivity,

            fly_speed: self.player.fly_speed,
        }
    }

    /// Build an `EngineConfig` from this `GameSettings`. CLI/engine-only
    /// fields (`title`, `capture_*`, `spawn`, `fullscreen`,
    /// `profile_cpu_systems`, `config_path`) take their hardcoded
    /// defaults; the caller should layer per-field CLI overrides on top
    /// of the returned value. This is the canonical mapping shared by
    /// `EngineConfig::default()` (via
    /// `GameSettings::default().to_engine_config()`) and
    /// `crates/app/src/main.rs` (after applying CLI flags), so the two
    /// paths can never drift apart.
    pub fn to_engine_config(&self) -> crate::EngineConfig {
        let render = self.to_renderer_config();
        let stream = self.to_stream_config();
        let player = self.to_player_config();
        let keybinds = self.keys.clone();
        let assets_path = self
            .world
            .assets_path
            .as_ref()
            .map(std::path::PathBuf::from);

        let mut cfg = graphics_fields!(self);
        // Override defaults with computed values.
        cfg.seed = self.world.seed;
        cfg.title = "voxel — custom Vulkan engine".into();
        cfg.render = render;
        cfg.stream = stream;
        cfg.world = self.world.clone();
        cfg.player = player;
        cfg.capture_after_frames = None;
        cfg.capture_path = std::path::PathBuf::from("capture.png");
        cfg.exit_after_capture = false;
        cfg.capture_cam_pos = None;
        cfg.capture_cam_rot = None;
        cfg.spawn = None;
        cfg.day_length = self.world.day_length;
        cfg.keybinds = keybinds;
        cfg.assets_path = assets_path;
        cfg.fullscreen = false;
        // Default to enabled so the F6 profiler overlay has data.
        cfg.profile_cpu_systems = true;
        cfg.config_path = std::path::PathBuf::from("config.toml");
        cfg
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]

    fn default_settings() {
        let s = GameSettings::default();

        assert_eq!(s.graphics.width, 1280);

        assert_eq!(s.graphics.height, 720);

        assert!(s.graphics.vsync);

        assert_eq!(s.world.seed, 1337);

        assert_eq!(s.world.load_radius, 6);

        assert_eq!(s.world.day_length, 1200.0);

        assert_eq!(s.keys.debug_overlay, "F3");

        assert_eq!(s.keys.wireframe, "F4");

        assert_eq!(s.keys.fly, "F5");

        assert_eq!(s.keys.chat, "T");

        assert!(!s.debug.show_overlay);
    }

    #[test]

    fn partial_toml_fills_defaults() {
        let toml = r#"

            [world]

            seed = 42

        "#;

        let s: GameSettings = toml::from_str(toml).unwrap();

        assert_eq!(s.world.seed, 42);

        assert_eq!(s.world.load_radius, 6); // default

        assert_eq!(s.graphics.width, 1280); // default
    }

    #[test]

    fn empty_toml_fills_all_defaults() {
        let s: GameSettings = toml::from_str("").unwrap();

        assert_eq!(s.graphics.width, 1280);

        assert_eq!(s.world.seed, 1337);

        assert_eq!(s.player.walk_speed, 4.317);
    }

    #[test]

    fn player_settings_override() {
        let toml = r#"

            [player]

            walk_speed = 10.0

            fly_speed = 50.0

        "#;

        let s: GameSettings = toml::from_str(toml).unwrap();

        assert!((s.player.walk_speed - 10.0).abs() < f32::EPSILON);

        assert!((s.player.fly_speed - 50.0).abs() < f32::EPSILON);

        assert!((s.player.sprint_speed - 5.6).abs() < f32::EPSILON); // default
    }

    #[test]

    fn keybind_override() {
        let toml = r#"

            [keys]

            chat = "Y"

            debug_overlay = "F12"

        "#;

        let s: GameSettings = toml::from_str(toml).unwrap();

        assert_eq!(s.keys.chat, "Y");

        assert_eq!(s.keys.debug_overlay, "F12");

        assert_eq!(s.keys.wireframe, "F4"); // default
    }

    #[test]

    fn graphics_override() {
        let toml = r#"

            [graphics]

            width = 1920

            height = 1080

            vsync = false

            fog_distance = 500.0

        "#;

        let s: GameSettings = toml::from_str(toml).unwrap();

        assert_eq!(s.graphics.width, 1920);

        assert_eq!(s.graphics.height, 1080);

        assert!(!s.graphics.vsync);

        assert!((s.graphics.fog_distance - 500.0).abs() < f32::EPSILON);
    }

    #[test]

    fn to_renderer_config() {
        let s = GameSettings::default();

        let rc = s.to_renderer_config();

        assert!(!rc.validation);

        assert!(rc.vsync);

        assert!((rc.fog_distance - 320.0).abs() < f32::EPSILON);
    }

    #[test]

    fn to_stream_config() {
        let s = GameSettings::default();

        let sc = s.to_stream_config();

        assert_eq!(sc.load_radius, 6);

        assert_eq!(sc.unload_radius, 8);

        assert_eq!(sc.vertical_half, 3);
    }

    #[test]

    fn to_player_config() {
        let s = GameSettings::default();

        let pc = s.to_player_config();

        assert!((pc.walk_speed - 4.317).abs() < f32::EPSILON);

        assert!((pc.fly_speed - 20.0).abs() < f32::EPSILON);
    }

    #[test]

    fn debug_settings_override() {
        let toml = r#"

            [debug]

            show_overlay = true

        "#;

        let s: GameSettings = toml::from_str(toml).unwrap();

        assert!(s.debug.show_overlay);
    }

    #[test]

    fn keybind_resolve_defaults() {
        use winit::keyboard::KeyCode;

        let s = KeybindSettings::default();

        let map = s.resolve();

        assert_eq!(map.get(&KeyCode::F3), Some(&Action::DebugOverlay));

        assert_eq!(map.get(&KeyCode::F4), Some(&Action::Wireframe));

        assert_eq!(map.get(&KeyCode::F5), Some(&Action::Fly));

        assert_eq!(map.get(&KeyCode::KeyT), Some(&Action::Chat));

        assert_eq!(map.get(&KeyCode::F8), Some(&Action::RenderDistanceUp));

        assert_eq!(map.get(&KeyCode::F9), Some(&Action::RenderDistanceDown));

        assert_eq!(map.get(&KeyCode::F2), Some(&Action::Screenshot));

        assert_eq!(map.get(&KeyCode::Escape), Some(&Action::Pause));

        assert_eq!(map.get(&KeyCode::F6), Some(&Action::Profiler));

        assert_eq!(map.get(&KeyCode::F7), Some(&Action::ChunkDebug));

        assert_eq!(map.get(&KeyCode::F10), Some(&Action::EcsInspector));

        assert_eq!(map.get(&KeyCode::F11), Some(&Action::PinUnpin));

        assert_eq!(map.get(&KeyCode::BracketRight), Some(&Action::PinCycleNext));

        assert_eq!(map.get(&KeyCode::BracketLeft), Some(&Action::PinCyclePrev));

        assert_eq!(map.get(&KeyCode::Backslash), Some(&Action::PinClear));

        assert_eq!(
            map.get(&KeyCode::Backquote),
            Some(&Action::DeveloperConsole)
        );

        assert_eq!(
            map.get(&KeyCode::ScrollLock),
            Some(&Action::TelemetryDashboard)
        );

        assert_eq!(map.get(&KeyCode::KeyX), Some(&Action::EditMode));

        assert_eq!(map.len(), 19);
    }

    #[test]

    fn keybind_resolve_custom() {
        use winit::keyboard::KeyCode;

        let s = KeybindSettings {
            debug_overlay: "F12".into(),

            chat: "Y".into(),

            ..Default::default()
        };

        let map = s.resolve();

        assert_eq!(map.get(&KeyCode::F12), Some(&Action::DebugOverlay));

        assert_eq!(map.get(&KeyCode::KeyY), Some(&Action::Chat));

        assert_eq!(map.get(&KeyCode::F3), None); // no longer bound
    }

    #[test]

    fn parse_key_known() {
        assert!(parse_key("F1").is_some());

        assert!(parse_key("Escape").is_some());

        assert!(parse_key("Space").is_some());

        assert!(parse_key("T").is_some());

        assert!(parse_key("Backquote").is_some());
    }

    #[test]

    fn parse_key_unknown() {
        assert!(parse_key("NotAKey").is_none());

        assert!(parse_key("").is_none());
    }
}
