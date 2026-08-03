//! `voxel-engine` — application shell.
//!
//! Owns the winit window, the Vulkan [`Renderer`], the shared [`World`], the
//! background [`ChunkStreamer`], and the gameplay state ([`Player`], [`Hotbar`],
//! [`InputState`]). Drives a fixed-timestep simulation with interpolated
//! rendering, translates raw window events into resolved input, and streams
//! chunk meshes to the GPU as the world loads.
//!
//! This is also the future host for the plugin/scripting runtime and the
//! dedicated-server entry point (same shell, headless renderer).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Ime, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::KeyCode;
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowAttributes, WindowId};

use crate::keybinds::physical_key_to_char;

use voxel_game::input::Action;
use voxel_game::{ChatState, CommandResult, DeveloperConsole, Hotbar, InputState, PlayerConfig};
use voxel_render::{FileWatcher, FontAtlas, GpuTimings, Renderer, RendererConfig};
use voxel_world::{ChunkStreamer, StreamConfig, World};

use crate::sim::Simulation;

mod commands;
mod console_commands;
mod edit;
mod frame;
mod keybinds;
mod map;
mod save;
pub mod settings;
mod sim;
mod telemetry;
pub mod test_sim;
mod ui;

/// High-level application state. Drives which UI is shown and how input is
/// routed. The flow is:
///
/// ```text
/// TitleScreen ──Singleplayer──ΓåÆ WorldSelect ──Pick World──ΓåÆ Playing
///      Γöé                            Γöé
///      Γö£──Options──────ΓåÆ SettingsMenu ΓåÉ──ΓöÉ
///      Γöö──Quit─────────ΓåÆ exit            Γöé
///                                        Γöé
/// Playing ──Escape──ΓåÆ PauseMenu ──Back to Game──ΓåÆ Playing
///                             Γö£──Options──────────────ΓåÆ SettingsMenu
///                             Γö£──Save & Quit to Title──ΓåÆ TitleScreen
///                             Γöö──Quit Game─────────────ΓåÆ exit
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GameState {
    TitleScreen,
    WorldSelect,
    Playing,
    PauseMenu,
    SettingsMenu,
}

/// Computed day/night parameters for the current game time.
struct DayParams {
    horizon: [f32; 3],
    zenith: [f32; 3],
    fog: [f32; 3],
    daylight: f32,
    sun_altitude: f32,
    sun_angle: f32,
}

/// Rolling profiler state for the debug overlay.
struct ProfilerState {
    frame_times_ms: VecDeque<f64>,
    gpu_timings: VecDeque<GpuTimings>,
    enabled: bool,
    /// Last chunk-debug-minimap refresh timestamp. Used to gate the
    /// `chunk_debug_info_batch` query so we re-poll the world at most
    /// once per ~100 ms (Phase 4 #11).
    last_minimap_update: Option<Instant>,
    /// Player chunk position at the time of the last minimap refresh;
    /// used to invalidate the cache when the player crosses a chunk
    /// boundary.
    last_minimap_chunk: Option<(i32, i32)>,
    /// Cached `(chunk_pos, loaded, dirty, palette_mode, has_mesh)` batch
    /// for the minimap so successive frames within the rate-limit window
    /// don't have to walk the world's chunk map.
    cached_minimap_batch: Vec<(voxel_core::math::ChunkPos, bool, bool, bool, bool)>,
    /// Per-system elapsed microseconds from the most recent ECS step.
    /// Refreshed by `frame()` after each `sched.run()`. Empty when
    /// `EngineConfig::profile_cpu_systems` is false.
    system_timings: Vec<(String, u64)>,
}

impl Default for ProfilerState {
    fn default() -> Self {
        Self {
            frame_times_ms: VecDeque::with_capacity(120),
            gpu_timings: VecDeque::with_capacity(120),
            enabled: false,
            last_minimap_update: None,
            last_minimap_chunk: None,
            cached_minimap_batch: Vec::new(),
            system_timings: Vec::new(),
        }
    }
}

impl ProfilerState {
    /// Record frame completion and push the frame time (in ms) into the
    /// rolling 120-frame history.
    fn end_frame(&mut self, frame_time_ms: f64) {
        self.frame_times_ms.push_back(frame_time_ms);
        if self.frame_times_ms.len() > 120 {
            self.frame_times_ms.pop_front();
        }
    }

    /// Average frame time over the rolling window (in ms).
    fn avg_ms(&self) -> f64 {
        if self.frame_times_ms.is_empty() {
            0.0
        } else {
            self.frame_times_ms.iter().sum::<f64>() / self.frame_times_ms.len() as f64
        }
    }

    /// Average FPS over the rolling window.
    fn avg_fps(&self) -> f64 {
        let avg = self.avg_ms();
        if avg > 0.0 {
            1000.0 / avg
        } else {
            0.0
        }
    }
}

/// Render-related state: the winit window, the Vulkan renderer, window size,
/// and the bitmap font used for UI text.
struct RenderState {
    pub renderer: Option<Renderer>,
    pub window: Option<Window>,
    pub window_size: (u32, u32),
    /// Monitor DPI scale factor of the window (e.g. 1.0 = 100%, 2.0 = 200%
    /// HiDPI). UI is laid out in *logical* pixels (`window_size / ui_scale`)
    /// and vertex positions are multiplied back by `ui_scale` before upload,
    /// so HUD/menus stay the same physical size on high-DPI displays.
    pub ui_scale: f32,
    pub font: FontAtlas,
}

impl RenderState {
    fn new(_config: &EngineConfig) -> Self {
        Self {
            renderer: None,
            window: None,
            window_size: (1280, 720),
            ui_scale: 1.0,
            font: FontAtlas::new(),
        }
    }

    /// Logical (DPI-independent) UI size in pixels: the physical window size
    /// divided by the monitor scale factor. All UI layout happens in this
    /// space so the HUD/menus render at a consistent size on HiDPI displays.
    fn logical_size(&self) -> (f32, f32) {
        let s = self.ui_scale.max(0.25);
        (self.window_size.0 as f32 / s, self.window_size.1 as f32 / s)
    }

    fn resize(&mut self, size: (u32, u32)) {
        if let Some(r) = self.renderer.as_mut() {
            r.resize(size);
        }
    }
}

/// Engine-level input & timing state. Embeds the gameplay [`InputState`] and
/// holds cursor lock, keybind map, per-frame timing, and scratch buffers.
///
/// Note: the fixed-timestep simulation accumulator lives on
/// [`crate::sim::Simulation`] now (private field, exposed via
/// `reset_accumulator`); the engine calls `tick_fixed(frame_dt)` and the
/// accumulator is updated implicitly.
struct EngineInputState {
    /// Game-level input state (held actions, clicks, mouse delta).
    pub input: InputState,
    /// True when the OS cursor is locked for FPS look.
    pub cursor_locked: bool,
    /// Last frame timestamp used for delta computation.
    pub last_time: Instant,
    /// Last frame delta in seconds.
    pub frame_time: f64,
    /// Total frames rendered (used for auto-capture).
    pub frame_count: usize,
    /// True once the auto-capture screenshot has been written.
    pub captured: bool,
    /// True when a screenshot has been requested and should be taken in frame().
    pub screenshot_requested: bool,
    /// True after the window has been created and the render loop started.
    pub running: bool,
    /// True once the player's spawn chunk has loaded; physics is paused until then.
    pub spawned: bool,
    /// Resolved keybind map: KeyCode ΓåÆ Action.
    pub keybinds: settings::KeybindMap,
    /// Last sun_dir sent to streamer (avoid redundant sends).
    pub last_sun_dir: Vec3,
    /// Last player position sent to streamer focus (avoid redundant sends).
    pub last_focus_pos: Vec3,
}

impl EngineInputState {
    fn new(keybinds: settings::KeybindMap) -> Self {
        Self {
            input: InputState::default(),
            cursor_locked: false,
            last_time: Instant::now(),
            frame_time: 0.016,
            frame_count: 0,
            captured: false,
            screenshot_requested: false,
            running: false,
            spawned: false,
            keybinds,
            last_sun_dir: Vec3::ZERO,
            last_focus_pos: Vec3::new(f32::MAX, f32::MAX, f32::MAX),
        }
    }
}

/// World state: the shared [`World`] handle and the background [`ChunkStreamer`].
struct WorldState {
    pub world: Arc<World>,
    pub streamer: Option<ChunkStreamer>,
}

impl WorldState {
    fn new(world: Arc<World>) -> Self {
        Self {
            world,
            streamer: None,
        }
    }
}

/// Creative inventory item with display information.
#[derive(Clone, Debug)]
pub(crate) struct CreativeItem {
    pub id: voxel_core::BlockId,
    pub name: String,
    pub category: String,
    pub tile: u16,
}

/// Crosshair display mode based on what player is looking at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CrosshairMode {
    /// Default crosshair (small +).
    Default,
    /// Looking at a block (slightly brighter, corner dots).
    BlockTarget,
    /// Looking at an interactable (center square).
    Interact,
}

/// Forward block-break/place particles to the renderer's particle manager.
/// The renderer owns simulation + GPU upload (subpass 1 in the main render
/// pipeline). The engine used to draw a CPU-projected 2D dot per particle
/// in the UI overlay; with Phase 1 ships, particles are real 3D
/// camera-aligned billboard quads in the world.
pub fn spawn_particles_break(
    renderer: &mut Option<Renderer>,
    pos: Vec3,
    color: [u8; 4],
    normal: Vec3,
) {
    if let Some(r) = renderer.as_mut() {
        r.spawn_particles_break(pos, color, normal);
    }
}

pub fn spawn_particles_place(renderer: &mut Option<Renderer>, pos: Vec3, color: [u8; 4]) {
    if let Some(r) = renderer.as_mut() {
        r.spawn_particles_place(pos, color);
    }
}

/// Step particle physics + push the per-instance VBO. Called once per frame.
pub fn update_particles(renderer: &mut Option<Renderer>, dt: f32) {
    if let Some(r) = renderer.as_mut() {
        r.update_particles(dt);
    }
}

/// Gameplay state: hotbar, chat, undo/redo, pause/menu flags, etc.
pub(crate) struct GamePlayState {
    /// Spawn position (used as fallback before ECS player is ready).
    pub spawn_pos: Vec3,
    /// 9-slot hotbar.
    pub hotbar: Hotbar,
    /// Current game state (playing vs pause menu).
    pub game_state: GameState,
    /// Game time in seconds (wraps at day_length).
    pub game_time: f64,
    /// Length of a full day/night cycle in seconds.
    pub day_length: f64,
    /// Mouse position in logical UI pixels (physical / ui_scale), for
    /// pause-menu and block-picker hit-testing against UI rects.
    pub mouse_pos: (f32, f32),
    /// Pre-computed pause-menu button rects (4 buttons: back, options, save&quit, quit).
    pub pause_buttons: Option<[(f32, f32, f32, f32); 4]>,
    /// Chat and command system.
    pub chat: ChatState,
    /// Undo/redo stack for block edits.
    pub undo_redo: voxel_game::UndoRedoState,
    /// Block picker (inventory) open.
    pub block_picker_open: bool,
    /// Schematic clipboard: ((x1,y1,z1), (x2,y2,z2), blocks)
    pub clipboard: Option<Clipboard>,
    /// Debug overlay enabled (F3 toggle).
    pub debug_overlay: bool,
    /// Chunk debug visualization enabled (F7 toggle).
    pub chunk_debug_enabled: bool,
    /// Set by the Exit Game button; checked in about_to_wait to exit the loop.
    pub want_exit: bool,
    /// Cached `(BlockId, name)` list for the block picker overlay.
    pub block_picker_cache: Vec<(voxel_core::BlockId, String)>,
    /// Runtime ECS inspector overlay toggle (F10 by default).
    pub ecs_inspector: bool,
    /// Developer console (toggled with ` or F2).
    pub console: DeveloperConsole,
    /// Queued script lines from /exec (drained one per frame).
    pub console_script: Option<Vec<String>>,
    /// Entity that the engine camera should follow, if any.
    pub pinned_entity: Option<voxel_ecs::Entity>,
    /// World editor state.
    pub edit: edit::EditState,
    /// Minimap / fullscreen map state.
    pub map: map::MapState,
    /// Title screen / world select / settings fields.
    pub title_buttons: Option<[(f32, f32, f32, f32); 4]>,
    pub world_list: Vec<save::WorldInfo>,
    pub selected_world_index: Option<usize>,
    pub world_select_buttons: Option<WorldSelectButtons>,
    pub create_world_state: Option<CreateWorldState>,
    pub settings_previous: GameState,
    pub listening_rebind: Option<usize>,
    pub current_world_path: Option<std::path::PathBuf>,
    pub panorama_rotation: f32,
    pub settings_back_btn: Option<(f32, f32, f32, f32)>,
    pub last_click_time: Option<std::time::Instant>,
    pub last_click_row: Option<usize>,
    pub pending_delete: Option<usize>,
    pub play_time_accumulator: f64,
    pub settings_widgets: Option<SettingsWidgets>,
    /// Index of the slider currently being dragged (if any).
    pub settings_slider_dragging: Option<usize>,
    /// Whether the left mouse button is currently held in the settings menu.
    pub settings_left_mouse_held: bool,
    pub cheats_enabled: bool,
    // Visual polish state
    pub looked_at_block: Option<[i32; 3]>,
    pub mining_crack: Option<([i32; 3], u8)>,
    pub crosshair_mode: CrosshairMode,
    // Creative inventory
    pub creative_items: Vec<CreativeItem>,
    pub creative_tab: usize,
    pub creative_search: String,
    pub creative_scroll: usize,
}

/// Pre-computed interactive widget rects from the settings menu draw pass.
#[derive(Clone, Debug)]
pub(crate) struct SettingsWidgets {
    /// (x, y, w, h) for each slider: render_distance, fog_distance, exposure,
    /// mouse_sensitivity, walk_speed, fly_speed
    pub sliders: Vec<SliderWidget>,
    /// (x, y, w, h) for each toggle: vsync, shadows, vignette
    pub toggles: Vec<(f32, f32, f32, f32, String)>, // x, y, w, h, label
    /// Apply / Defaults button rects
    pub apply_btn: (f32, f32, f32, f32),
    pub defaults_btn: (f32, f32, f32, f32),
}

impl GamePlayState {
    fn new(spawn_pos: Vec3, hotbar: Hotbar, day_length: f64) -> Self {
        Self {
            spawn_pos,
            hotbar,
            game_state: GameState::TitleScreen,
            game_time: 300.0, // start at dawn
            day_length,
            mouse_pos: (0.0, 0.0),
            pause_buttons: None,
            chat: ChatState::default(),
            undo_redo: voxel_game::UndoRedoState::default(),
            block_picker_open: false,
            clipboard: None,
            debug_overlay: false,
            chunk_debug_enabled: false,
            want_exit: false,
            block_picker_cache: Vec::new(),
            ecs_inspector: false,
            console: DeveloperConsole::new(),
            console_script: None,
            pinned_entity: None,
            edit: edit::EditState::default(),
            map: map::MapState::default(),
            title_buttons: None,
            world_list: Vec::new(),
            selected_world_index: None,
            world_select_buttons: None,
            create_world_state: None,
            settings_previous: GameState::TitleScreen,
            listening_rebind: None,
            current_world_path: None,
            panorama_rotation: 0.0,
            settings_back_btn: None,
            last_click_time: None,
            last_click_row: None,
            pending_delete: None,
            play_time_accumulator: 0.0,
            settings_widgets: None,
            settings_slider_dragging: None,
            settings_left_mouse_held: false,
            cheats_enabled: false,
            creative_items: Vec::new(),
            creative_tab: 0,
            creative_search: String::new(),
            creative_scroll: 0,
            looked_at_block: None,
            mining_crack: None,
            crosshair_mode: CrosshairMode::Default,
        }
    }

    /// Populate the block-picker cache from the world's block registry. The
    /// picker overlays the same `(id, name)` pair list regardless of which
    /// blocks are currently loaded, so caching once at startup (and on
    /// registry change) is correct. Called from `EngineApp::new`.
    pub fn populate_block_picker_cache(&mut self, reg: &voxel_world::BlockRegistry) {
        self.block_picker_cache.clear();
        self.creative_items.clear();
        for id_u16 in 0..reg.count() as u16 {
            let id = voxel_core::BlockId(id_u16);
            let def = reg.get(id);
            let name = def.name.as_ref().to_string();
            // Get the top face tile index for the icon.
            let tile = def.textures.tile(voxel_world::registry::Face::PosY);
            // Categorize based on block properties.
            let category = categorize_block(&name, def);
            self.block_picker_cache.push((id, name.clone()));
            self.creative_items.push(CreativeItem {
                id,
                name,
                category,
                tile,
            });
        }
    }

    // Player accessors (`player_pos`, `player_flying`, etc.) live on
    // [`crate::sim::Simulation`] now; the engine goes through
    // `self.simulation` for any ECS-backed player query. This struct
    // holds engine-level concerns only: hotbar, chat, undo/redo,
    // pause/menu state, ECS inspector toggles, the camera pin target.
}

/// Categorize a block for the creative inventory tabs.
fn categorize_block(name: &str, def: &voxel_world::registry::BlockDef) -> String {
    let n = name.to_lowercase();
    // Nature: plants, organic
    if n.contains("grass") && !n.contains("block")
        || n.contains("leaves")
        || n.contains("log")
        || n.contains("sapling")
        || n.contains("flower")
        || n.contains("mushroom")
        || n.contains("cactus")
        || n.contains("vine")
        || n.contains("tall")
        || n.contains("fern")
        || n.contains("poppy")
        || n.contains("dandelion")
        || n.contains("kelp")
        || n.contains("coral")
    {
        return "Nature".to_string();
    }
    // Building: construction blocks
    if n.contains("stone")
        || n.contains("cobble")
        || n.contains("brick")
        || n.contains("plank")
        || n.contains("wood")
        || n.contains("sand")
        || n.contains("gravel")
        || n.contains("glass")
        || n.contains("wool")
        || n.contains("concrete")
        || n.contains("terracotta")
    {
        return "Building".to_string();
    }
    // Ores / minerals
    if n.contains("ore")
        || n.contains("diamond")
        || n.contains("gold")
        || n.contains("iron")
        || n.contains("coal")
        || n.contains("emerald")
        || n.contains("lapis")
        || n.contains("redstone")
    {
        return "Ores".to_string();
    }
    // Decoration
    if n.contains("torch")
        || n.contains("lantern")
        || n.contains("candle")
        || n.contains("carpet")
        || n.contains("banner")
        || n.contains("painting")
        || n.contains("bookshelf")
    {
        return "Decoration".to_string();
    }
    // Liquids
    if def.kind == voxel_world::registry::BlockKind::Liquid {
        return "Liquids".to_string();
    }
    // Default
    "Blocks".to_string()
}

/// Pre-computed button rects for the world selection screen.
#[derive(Clone, Debug)]
pub(crate) struct WorldSelectButtons {
    /// Per-world row rects: (x, y, w, h) for each world entry.
    pub rows: Vec<(f32, f32, f32, f32)>,
    /// Delete button rects parallel to `rows`.
    pub delete_buttons: Vec<(f32, f32, f32, f32)>,
    /// "Create New World" button.
    pub create_btn: (f32, f32, f32, f32),
    /// "Play Selected World" button.
    pub play_btn: (f32, f32, f32, f32),
    /// "Close" button.
    pub close_btn: (f32, f32, f32, f32),
}

/// State for the create-world mini-dialog.
#[derive(Clone, Debug)]
pub(crate) struct CreateWorldState {
    pub name: String,
    pub seed: String,
    pub game_mode: String,
    pub allow_cheats: bool,
    pub error: Option<String>,
    /// 0 = name field, 1 = seed field (for keyboard input routing).
    pub active_field: usize,
    /// Pre-computed rects for click handling: (name_input, seed_input, cancel_btn, create_btn, mode_survival, mode_creative).
    pub rects: Option<CreateWorldRects>,
}

#[derive(Clone, Debug)]
pub(crate) struct CreateWorldRects {
    pub name_input: (f32, f32, f32, f32),
    pub seed_input: (f32, f32, f32, f32),
    pub cancel_btn: (f32, f32, f32, f32),
    pub create_btn: (f32, f32, f32, f32),
    pub mode_survival: (f32, f32, f32, f32),
    pub mode_creative: (f32, f32, f32, f32),
    pub cheats_toggle: (f32, f32, f32, f32),
}

impl Default for CreateWorldState {
    fn default() -> Self {
        Self {
            name: "New World".into(),
            seed: String::new(),
            game_mode: "survival".into(),
            allow_cheats: false,
            error: None,
            active_field: 0,
            rects: None,
        }
    }
}

/// Engine configuration.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// World generation seed.
    pub seed: i32,
    /// Window title.
    pub title: String,
    /// Initial window size in physical pixels.
    pub window_size: (u32, u32),
    /// Render configuration.
    pub render: RendererConfig,
    /// Chunk streaming configuration.
    pub stream: StreamConfig,
    /// Mirror of `GameSettings::world`. The ConfigChanged handler gates
    /// world-related updates on whole-struct inequality via
    /// `WorldSettings`'s `PartialEq` derive; the older per-field mirrors
    /// (`self.config.stream.{load_radius, unload_radius}` and
    /// `self.config.day_length`) are kept in sync for downstream systems
    /// that read them, while `self.config.seed` is intentionally NOT
    /// mirrored (a worker holds the original startup seed for terrain gen).
    pub world: crate::settings::WorldSettings,
    /// Player configuration.
    pub player: PlayerConfig,
    /// If set, automatically capture a screenshot after this many frames and
    /// save it to `capture_path` (used for headless verification).
    pub capture_after_frames: Option<usize>,
    /// Where to write an auto-capture screenshot.
    pub capture_path: PathBuf,
    /// Exit the process shortly after the auto-capture completes.
    pub exit_after_capture: bool,
    /// Optional camera position override for auto-capture runs (teleports the
    /// player, enables flying). Set via `--campos`.
    pub capture_cam_pos: Option<[f32; 3]>,
    /// Optional camera yaw/pitch override (radians) for auto-capture runs.
    /// Set via `--camrot`.
    pub capture_cam_rot: Option<(f32, f32)>,
    /// Spawn position (world space). If `None`, a surface height is found.
    pub spawn: Option<[f32; 3]>,
    /// Day length in seconds.
    pub day_length: f64,
    /// Keybind settings from config.
    pub keybinds: settings::KeybindSettings,
    /// Path to the assets directory (for data-driven block definitions).
    pub assets_path: Option<PathBuf>,
    /// Enable cascaded shadow mapping.
    pub shadow_enabled: bool,
    /// Shadow map resolution per cascade (texels per side).
    pub shadow_resolution: u32,
    /// HDR tonemapping exposure.
    pub exposure: f32,
    /// Vignette darkening strength at screen edges.
    pub vignette_strength: f32,
    /// Enable screen-space ambient occlusion.
    pub ssao_enabled: bool,
    /// SSAO sampling radius (world-space distance).
    pub ssao_radius: f32,
    /// SSAO depth bias to prevent self-occlusion.
    pub ssao_bias: f32,
    /// SSAO darkening strength multiplier.
    pub ssao_strength: f32,
    /// Global water surface Y (world units). Drives the chunk shader's
    /// caustics path + wet-edge band. Defaults to sea level at chunk Y=0.
    pub water_y: f32,
    /// Cool-tint wet-edge strength applied to blocks near the water surface.
    pub wet_edge_strength: f32,
    /// Sun-caustics strength on submerged terrain.
    pub caustics_strength: f32,
    /// Cherry-yellow chlorophyll backlight on leaves (sun-behind-leaf case).
    pub leaves_sss_strength: f32,
    /// Master reflection strength in [0, 1] for the chunk shader's slice-3
    /// reflection paths (water SSR + sky, glass sheen, opaque glossy tiles).
    /// 0 disables every reflection.
    pub reflection_strength: f32,
    /// Open the window in borderless-fullscreen mode (instead of windowed).
    pub fullscreen: bool,
    /// Opt in to per-system CPU timing on the ECS schedule. When true,
    /// `EngineApp::new` builds the schedule with `.with_timing()` and
    /// `frame()` refreshes `ProfilerState::system_timings` after each
    /// `sched.run()`. Default true so the profiler overlay has data.
    pub profile_cpu_systems: bool,
    /// Path to the live config file. Used by the file-watcher to detect
    /// live edits and by `reload_config_from_disk` to re-read settings.
    pub config_path: PathBuf,
}

impl Default for EngineConfig {
    fn default() -> Self {
        // Single source of truth: delegate to
        // `GameSettings::default().to_engine_config()`. Both
        // `app::main` (after applying CLI overrides) and this
        // `Default` impl share the same canonical mapping, so
        // they cannot drift apart.
        crate::settings::GameSettings::default().to_engine_config()
    }
}

/// Entry point: create the event loop and run the engine until the window
/// closes (or until an auto-capture + exit completes).
pub fn run(config: EngineConfig) -> Result<()> {
    let event_loop = EventLoop::new().map_err(|e| anyhow!("EventLoop::new: {e}"))?;
    let mut app = EngineApp::new(config)?;
    event_loop
        .run_app(&mut app)
        .map_err(|e| anyhow!("event loop ended with error: {e}"))?;
    Ok(())
}

/// Schematic clipboard: origin corner, opposite corner, flattened block list.
pub(crate) type Clipboard = ((i32, i32, i32), (i32, i32, i32), Vec<voxel_core::BlockId>);

/// Pre-computed settings slider widget rect: (x, y, w, h, label, min, max).
pub(crate) type SliderWidget = (f32, f32, f32, f32, String, f32, f32);

/// Manages loaded texture packs and their UI state.
#[derive(Default)]
pub struct TexturePackManager {
    /// Currently loaded texture packs (in load order).
    pub loaded_packs: Vec<TexturePackInfo>,
    /// Whether the pack manager panel is open.
    pub panel_open: bool,
}

/// Info about a loaded texture pack, displayed in the UI.
#[derive(Clone, Debug)]
pub struct TexturePackInfo {
    /// Display name.
    pub name: String,
    /// Description from pack.toml.
    pub description: String,
    /// Version from pack.toml.
    pub version: String,
    /// Author from pack.toml.
    pub author: String,
    /// Number of overridden tiles.
    pub tile_count: usize,
    /// Number of animation definitions.
    pub animation_count: usize,
    /// Whether this pack is currently enabled.
    pub enabled: bool,
}


pub(crate) struct EngineApp {
    config: EngineConfig,
    /// Render state: window, renderer, font, window size.
    render: RenderState,
    /// Engine input & timing: cursor lock, keybinds, frame timing, scratch buffers.
    input: EngineInputState,
    /// World state: shared world handle and chunk streamer.
    world_state: WorldState,
    /// Headless gameplay step. Owns the ECS world, the system schedule,
    /// and a clone of the `Arc<World>` (shared with `world_state`).
    /// `frame()` calls `simulation.set_player_input` + `tick_fixed`
    /// each tick; pause / spawn-wait logic short-circuits the call.
    simulation: Simulation,
    /// Gameplay state: hotbar, chat, undo/redo, pause/menu state, etc.
    /// Player accessors live on `simulation` now.
    gameplay: GamePlayState,
    /// Profiler state (rolling frame time, GPU timings).
    profiler: ProfilerState,
    /// Telemetry collector for the metrics dashboard.
    telemetry: telemetry::TelemetryCollector,
    /// Asset hot-reload: file-watcher background thread + receiver. Populated
    /// in `EngineApp::new` once shader / texture / config paths are known.
    hot_reload: Option<FileWatcher>,
    /// Audio system.
    audio: voxel_audio::AudioManager,
    /// System info for RSS memory reporting.
    sysinfo: sysinfo::System,
    /// Counter to throttle sysinfo refreshes (every N frames).
    sysinfo_refresh_counter: u32,
    /// Texture pack manager: tracks loaded packs and their metadata.
    texture_pack_manager: TexturePackManager,
}

impl EngineApp {
    fn new(config: EngineConfig) -> Result<Self> {
        let world = World::new_with_path(config.seed, config.assets_path.as_deref());
        // Find a land spawn (above sea level) using the height function directly
        // — no chunk loading needed. Falls back to a high spawn at origin.
        let (sx, sy, sz) = world.terrain().find_spawn();
        let spawn = config
            .spawn
            .unwrap_or([sx as f32 + 0.5, sy as f32 + 2.0, sz as f32 + 0.5]);
        log::info!(
            "spawn search: land column at ({}, {}, {}) -> spawn pos {:?}",
            sx,
            sy,
            sz,
            spawn
        );
        // Compute spawn_pos early so GamePlayState::new can read it (Phase 4 #10
        // also wants the block-picker cache populated before the Self literal).
        let spawn_pos = Vec3::from(spawn);
        let day_length = config.day_length;
        let mut hotbar = Hotbar::new();
        hotbar.populate_defaults(&world.registry());
        let keybinds = config.keybinds.resolve();
        let render = RenderState::new(&config);

        // Build gameplay state and populate the block-picker cache up-front so
        // `draw_block_picker` / `handle_block_picker_click` can read from the
        // precomputed `(BlockId, name)` list instead of walking the registry
        // every frame (Phase 4 #10).
        let mut gameplay = GamePlayState::new(spawn_pos, hotbar, day_length);
        gameplay.populate_block_picker_cache(&world.registry());
        // Populate edit palette categories.
        {
            let mut categories: std::collections::HashMap<
                edit::BlockCategory,
                Vec<(voxel_core::BlockId, String)>,
            > = std::collections::HashMap::new();
            for (id, name) in &gameplay.block_picker_cache {
                let cat = edit::categorize(&name.to_lowercase());
                categories.entry(cat).or_default().push((*id, name.clone()));
            }
            gameplay.edit.categories = categories.into_iter().collect();
            gameplay.edit.categories.sort_by_key(|(cat, _)| *cat);
        }
        gameplay.console.register_commands(&[
            "/tp",
            "/time set",
            "/time speed",
            "/give",
            "/setblock",
            "/fill",
            "/hollow",
            "/sphere",
            "/cylinder",
            "/pyramid",
            "/replace",
            "/line",
            "/schematic save",
            "/schematic list",
            "/schematic load",
            "/schematic paste",
            "/gamemode",
            "/pos",
            "/chunk",
            "/fps",
            "/reload",
            "/clear",
            "/save",
            "/load",
            "/copy",
            "/paste",
            "/help",
            "/ecs list",
            "/ecs inspect",
            "/ecs resources",
            "/ecs resource",
            "/get",
            "/set",
            "/exec",
            "/waypoint add",
            "/waypoint list",
            "/waypoint remove",
            "/waypoint save",
            "/waypoint load",
        ]);

        // Load saved waypoints.
        let wp_path = std::path::Path::new("assets").join("waypoints.json");
        if let Err(e) = gameplay.map.load_waypoints(&wp_path) {
            log::warn!("failed to load waypoints: {e}");
        }

        // --- Simulation ---
        // Build the headless sim with the SAME world Arc the streamer
        // operates on. Both sides share chunk storage via the Arc, so a
        // chunk loaded by the streamer is immediately visible to
        // movement_system's collision queries. The day/night clock
        // lives on `GamePlayState`, so `Simulation::new` doesn't need
        // a `day_length` argument.
        let simulation = Simulation::new(world.clone(), spawn_pos, config.profile_cpu_systems);

        // Note: the file-watcher is spawned AFTER `Renderer::new` returns in
        // `resumed()`, because we need the (shader_dir, textures_dir,
        // config_path) triple that the user-facing config actually resolves.
        Ok(Self {
            config,
            render,
            input: EngineInputState::new(keybinds),
            world_state: WorldState::new(world),
            simulation,
            gameplay,
            profiler: ProfilerState::default(),
            telemetry: telemetry::TelemetryCollector::default(),
            hot_reload: None,
            audio: voxel_audio::AudioManager::null(),
            sysinfo: sysinfo::System::new(),
            sysinfo_refresh_counter: 0,
            texture_pack_manager: TexturePackManager::default(),
        })
    }

    fn lock_cursor(&mut self) {
        if let Some(w) = &self.render.window {
            let _ = w
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| w.set_cursor_grab(CursorGrabMode::Confined));
            w.set_cursor_visible(false);
            self.input.cursor_locked = true;
        }
    }

    fn unlock_cursor(&mut self) {
        if let Some(w) = &self.render.window {
            let _ = w.set_cursor_grab(CursorGrabMode::None);
            w.set_cursor_visible(true);
            self.input.cursor_locked = false;
        }
    }

    /// Compute day/night parameters from game_time.
    /// Step the pin forward (+1) or backward (ΓÇô1) through every live
    /// entity in the ECS world, sorted by `Entity.index`. Reads
    /// `simulation.ecs_world().archetypes()` for the live set; treats
    /// the currently-pinned entity (or the end of the list, if no pin
    /// is active) as the rotation pivot. Always announces the new
    /// pin via `chat.push_message`.
    fn cycle_pin(&mut self, dir: i32) {
        let mut live: Vec<voxel_ecs::Entity> = Vec::new();
        for arch in self.simulation.ecs_world().archetypes() {
            for e in arch.entities() {
                live.push(*e);
            }
        }
        live.sort_by_key(|e| e.index);
        if live.is_empty() {
            self.gameplay
                .chat
                .push_message("Pin: no live entities to cycle".into());
            return;
        }
        // Step relative to the currently-pinned entity (if any). When
        // no pin is active, always land on the FIRST entity so the
        // first press after a clear is predictable.
        let cur_pos_opt = self
            .gameplay
            .pinned_entity
            .and_then(|cur| live.iter().position(|e| *e == cur));
        let next = match (cur_pos_opt, dir > 0) {
            (Some(i), true) => (i + 1) % live.len(),
            (Some(0), false) => live.len() - 1,
            (Some(i), false) => i - 1,
            (None, _) => 0,
        };
        let chosen = live[next];
        self.gameplay.pinned_entity = Some(chosen);
        self.gameplay
            .chat
            .push_message(format!("Pin: e[{}:{}]", chosen.index, chosen.generation));
    }

    fn day_params(&self) -> DayParams {
        let t = (self.gameplay.game_time % self.gameplay.day_length) / self.gameplay.day_length; // 0..1
        let sun_angle = (t as f32) * std::f32::consts::TAU;
        let sun_altitude = (sun_angle - std::f32::consts::FRAC_PI_2).sin(); // -1..1
        let daylight = (sun_altitude * 1.2 + 0.6).clamp(0.0, 1.0);

        // Day sky colours.
        let day_horizon = [0.62, 0.78, 0.95];
        let day_zenith = [0.35, 0.55, 0.90];
        let day_fog = [0.62, 0.80, 0.96];
        // Night sky colours.
        let night_horizon = [0.05, 0.06, 0.12];
        let night_zenith = [0.01, 0.02, 0.05];
        let night_fog = [0.04, 0.05, 0.10];

        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        };

        DayParams {
            horizon: lerp(night_horizon, day_horizon, daylight),
            zenith: lerp(night_zenith, day_zenith, daylight),
            fog: lerp(night_fog, day_fog, daylight),
            daylight,
            sun_altitude,
            sun_angle,
        }
    }

    /// Transition to TitleScreen state (unlock cursor, stop sim).
    fn enter_title_screen(&mut self) {
        self.gameplay.game_state = GameState::TitleScreen;
        self.unlock_cursor();
        self.input.input.held.clear();
        self.gameplay.block_picker_open = false;
        self.gameplay.chat.open = false;
        self.gameplay.console.open = false;
        self.gameplay.edit.mode = edit::EditModeState::Inactive;
        self.gameplay.map.fullscreen_open = false;
        self.gameplay.settings_slider_dragging = None;
        self.gameplay.settings_left_mouse_held = false;
        self.simulation.reset_accumulator();
        self.input.spawned = false;
        // Play title screen music.
        self.audio.push_event(voxel_audio::AudioEvent::PlayMusic {
            track: "music.menu".into(),
            volume: 1.0,
            loop_: true,
        });
    }

    /// Transition to WorldSelect state (scan saves directory).
    fn enter_world_select(&mut self) {
        let saves_dir = std::path::PathBuf::from("saves");
        self.gameplay.world_list = save::list_world_info(&saves_dir);
        self.gameplay.selected_world_index = None;
        self.gameplay.create_world_state = None;
        self.gameplay.game_state = GameState::WorldSelect;
        self.unlock_cursor();
        self.input.input.held.clear();
    }

    /// Transition to SettingsMenu, remembering where we came from.
    fn enter_settings(&mut self, previous: GameState) {
        self.gameplay.settings_previous = previous;
        self.gameplay.game_state = GameState::SettingsMenu;
        self.unlock_cursor();
        self.input.input.held.clear();
        self.gameplay.listening_rebind = None;
        self.gameplay.settings_slider_dragging = None;
        self.gameplay.settings_left_mouse_held = false;
    }

    /// Transition to Playing state (lock cursor, resume sim).
    fn enter_playing(&mut self) {
        self.gameplay.game_state = GameState::Playing;
        self.lock_cursor();
        self.input.running = true;
        self.input.last_time = Instant::now();
        self.simulation.reset_accumulator();
        // Stop title music.
        self.audio.push_event(voxel_audio::AudioEvent::StopMusic);
    }

    /// Transition to PauseMenu state (unlock cursor, pause sim).
    fn enter_pause(&mut self) {
        self.gameplay.game_state = GameState::PauseMenu;
        self.unlock_cursor();
        self.input.input.held.clear();
        self.gameplay.block_picker_open = false;
        self.gameplay.chat.open = false;
        self.gameplay.edit.mode = edit::EditModeState::Inactive;
        self.gameplay.map.fullscreen_open = false;
        self.gameplay.settings_slider_dragging = None;
        self.gameplay.settings_left_mouse_held = false;
        // Drop any in-flight fixed-step accumulator so a long pause
        // doesn't unleash a burst of catch-up steps on resume.
        self.simulation.reset_accumulator();
    }
}

impl ApplicationHandler for EngineApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.render.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title(self.config.title.clone())
            .with_inner_size(PhysicalSize::new(
                self.config.window_size.0,
                self.config.window_size.1,
            ))
            .with_fullscreen(if self.config.fullscreen {
                Some(Fullscreen::Borderless(None))
            } else {
                None
            });
        let window = match event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(e) => {
                log::error!("create_window: {e}");
                self.gameplay.want_exit = true;
                return;
            }
        };
        // Capture the initial DPI scale factor (HiDPI laptops report > 1.0)
        // so the HUD/menus lay out in logical pixels from the first frame.
        self.render.ui_scale = window.scale_factor().max(0.25) as f32;
        self.render.window = Some(window);

        // Build the renderer from the window's raw handles.
        if let Some(window) = &self.render.window {
            use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
            let wh = match window.window_handle() {
                Ok(h) => h.as_raw(),
                Err(e) => {
                    log::error!("window_handle: {e}");
                    self.gameplay.want_exit = true;
                    return;
                }
            };
            let dh = match window.display_handle() {
                Ok(h) => h.as_raw(),
                Err(e) => {
                    log::error!("display_handle: {e}");
                    self.gameplay.want_exit = true;
                    return;
                }
            };
            // Query the actual physical window size: on Wayland the
            // compositor may assign a size different from the requested one,
            // and the swapchain must match whatever the window really is.
            let window_size = window.inner_size();
            match Renderer::new(
                wh,
                dh,
                self.config.render.clone(),
                (window_size.width, window_size.height),
            ) {
                Ok(r) => {
                    self.render.window_size = (r.extent().width, r.extent().height);
                    // Populate texture pack manager from initial renderer pack info.
                    self.texture_pack_manager.loaded_packs = r
                        .pack_infos()
                        .iter()
                        .map(|p| crate::TexturePackInfo {
                            name: p.name.clone(),
                            description: p.description.clone(),
                            version: p.version.clone(),
                            author: p.author.clone(),
                            tile_count: p.tile_count,
                            animation_count: p.animation_count,
                            enabled: p.enabled,
                        })
                        .collect();
                    self.render.renderer = Some(r);
                }
                Err(e) => {
                    log::error!("Renderer::new: {e}");
                    self.gameplay.want_exit = true;
                    return;
                }
            }
        }

        // Spawn the chunk streamer and populate the hotbar.
        let focus_pos = self
            .simulation
            .player_pos()
            .unwrap_or(self.gameplay.spawn_pos);
        match ChunkStreamer::spawn(self.world_state.world.clone(), self.config.stream) {
            Ok(s) => {
                s.set_focus(focus_pos);
                self.world_state.streamer = Some(s);
            }
            Err(e) => {
                log::error!("ChunkStreamer::spawn: {e}");
                self.gameplay.want_exit = true;
                return;
            }
        }

        // Start on the title screen — don't lock cursor or start sim yet.
        // The chunk streamer runs in the background so terrain is ready
        // when the player clicks "Play".
        self.input.running = true;
        self.input.last_time = Instant::now();

        // Spawn the asset file-watcher. It hot-reloads shader / texture /
        // config edits at runtime by sending HotReloadEvent variants over
        // an mpsc channel that `frame()` drains each tick. `shader_dir`
        // is now Optional; the watcher thread simply skips shader events
        // when it is None, so a textures-only workflow works as well.
        let shader_dir = self.config.render.shader_dir.clone();
        let textures_dir = self.config.render.textures_dir.clone();
        let texture_packs_dir = self.config.render.texture_packs_dir.clone();
        let config_path = self.config.config_path.clone();
        self.hot_reload = Some(FileWatcher::new(
            shader_dir,
            textures_dir,
            texture_packs_dir,
            config_path,
        ));
        log::info!(
            "file-watcher: active (config = {})",
            self.config.config_path.display()
        );

        // Initialize audio system.
        let audio_dir = std::path::PathBuf::from("assets/audio");
        match voxel_audio::AudioManager::new(voxel_audio::AudioConfig::default(), &audio_dir) {
            Ok(a) => {
                self.audio = a;
                log::info!("Audio system initialized");
            }
            Err(e) => {
                log::warn!("Audio init failed: {e}. Running without sound.");
                self.audio = voxel_audio::AudioManager::null();
            }
        }

        // Request the first redraw to kick-start the render loop.
        if let Some(w) = &self.render.window {
            w.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                // DPI changed (e.g. the window moved between monitors).
                // Update the UI scale; the next redraw re-lays-out the HUD
                // at the new scale. The physical window size is unchanged
                // (the existing `Resized` path handles any swapchain work).
                self.render.ui_scale = scale_factor.max(0.25) as f32;
                if let Some(w) = &self.render.window {
                    w.request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                self.render.window_size = (size.width, size.height);
                self.render.resize((size.width, size.height));
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::PhysicalKey;
                let pressed = event.state == ElementState::Pressed;

                // Developer console input takes priority when open.
                if self.gameplay.console.open {
                    if pressed {
                        if let PhysicalKey::Code(code) = event.physical_key {
                            match code {
                                KeyCode::Escape | KeyCode::Backquote => {
                                    self.gameplay.console.close();
                                    self.lock_cursor();
                                }
                                KeyCode::Enter => {
                                    let text = self.gameplay.console.submit();
                                    if text.starts_with('/') {
                                        let result = voxel_game::ChatState::parse_command(
                                            &text,
                                            self.simulation
                                                .player_pos()
                                                .unwrap_or_default(),
                                        );
                                        match &result {
                                            CommandResult::EcsList
                                            | CommandResult::EcsInspect { .. }
                                            | CommandResult::EcsResources
                                            | CommandResult::EcsResource { .. }
                                            | CommandResult::Get { .. }
                                            | CommandResult::Set { .. }
                                            | CommandResult::Exec(_) => {
                                                let output =
                                                    self.execute_console_command(result);
                                                for line in output {
                                                    self.gameplay.console.println(line);
                                                }
                                            }
                                            _ => {
                                                self.execute_command(result);
                                            }
                                        }
                                    } else {
                                        self.gameplay
                                            .console
                                            .println("commands start with /".into());
                                    }
                                }
                                KeyCode::Backspace => self.gameplay.console.backspace(),
                                KeyCode::Delete => self.gameplay.console.delete(),
                                KeyCode::ArrowLeft => self.gameplay.console.cursor_left(),
                                KeyCode::ArrowRight => self.gameplay.console.cursor_right(),
                                KeyCode::ArrowUp => self.gameplay.console.cursor_up(),
                                KeyCode::ArrowDown => self.gameplay.console.cursor_down(),
                                KeyCode::Home => self.gameplay.console.cursor_home(),
                                KeyCode::End => self.gameplay.console.cursor_end(),
                                KeyCode::Tab => self.gameplay.console.tab_complete(),
                                _ => {
                                    if let Some(ch) = physical_key_to_char(code) {
                                        self.gameplay.console.insert_char(ch);
                                    }
                                }
                            }
                        }
                    }
                    return;
                }

                // Chat input takes priority when open.
                if self.gameplay.chat.open {
                    if pressed {
                        if let PhysicalKey::Code(code) = event.physical_key {
                            match code {
                                KeyCode::Escape => {
                                    self.gameplay.chat.close();
                                    self.lock_cursor();
                                    if let Some(w) = &self.render.window {
                                        w.set_ime_allowed(false);
                                    }
                                }
                                KeyCode::Enter => {
                                    let pos = self.simulation.player_pos()
                                        .unwrap_or(self.gameplay.spawn_pos);
                                    let result = self.gameplay.chat.submit_with_pos(pos);
                                    self.execute_command(result);
                                    self.lock_cursor();
                                    if let Some(w) = &self.render.window {
                                        w.set_ime_allowed(false);
                                    }
                                }
                                KeyCode::Backspace => {
                                    self.gameplay.chat.backspace();
                                }
                                KeyCode::Space => {
                                    self.gameplay.chat.push_char(' ');
                                }
                                KeyCode::Tab => {
                                    self.gameplay.chat.tab_complete();
                                }
                                KeyCode::ArrowUp => {
                                    self.gameplay.chat.history_up();
                                }
                                KeyCode::ArrowDown => {
                                    self.gameplay.chat.history_down();
                                }
                                _ => {
                                    // Map physical key codes to characters directly.
                                    if let Some(ch) = physical_key_to_char(code) {
                                        self.gameplay.chat.push_char(ch);
                                    }
                                }
                            }
                        }
                    }
                    return;
                }

                // Palette search input takes priority when focused.
                if self.gameplay.edit.mode.is_active() && self.gameplay.edit.search_focused {
                    if pressed {
                        if let PhysicalKey::Code(code) = event.physical_key {
                            match code {
                                KeyCode::Escape => {
                                    self.gameplay.edit.search_focused = false;
                                }
                                KeyCode::Backspace => {
                                    self.gameplay.edit.palette_search.pop();
                                }
                                _ => {
                                    if let Some(ch) = physical_key_to_char(code) {
                                        self.gameplay.edit.palette_search.push(ch);
                                    }
                                }
                            }
                        }
                    }
                    return;
                }

                // Creative inventory search input takes priority when search tab is active.
                if self.gameplay.block_picker_open {
                    let tab_count = 8; // Must match draw_block_picker
                    let is_search = self.gameplay.creative_tab == tab_count - 1;
                    if is_search && pressed {
                        if let PhysicalKey::Code(code) = event.physical_key {
                            match code {
                                KeyCode::Escape => {
                                    self.gameplay.block_picker_open = false;
                                    self.lock_cursor();
                                }
                                KeyCode::Backspace => {
                                    self.gameplay.creative_search.pop();
                                }
                                _ => {
                                    if let Some(ch) = physical_key_to_char(code) {
                                        self.gameplay.creative_search.push(ch);
                                    }
                                }
                            }
                        }
                        return;
                    }
                    // If not in search tab, E or Escape closes the inventory.
                    if pressed {
                        if let PhysicalKey::Code(code) = event.physical_key {
                            if code == KeyCode::Escape || code == KeyCode::KeyE {
                                self.gameplay.block_picker_open = false;
                                self.lock_cursor();
                                return;
                            }
                        }
                    }
                    // Block other input while inventory is open.
                    return;
                }

                if let PhysicalKey::Code(code) = event.physical_key {
                    // Ctrl+Z / Ctrl+Y for undo/redo (check before other handlers).
                    if pressed && self.input.input.held(Action::Sprint) {
                        match code {
                            KeyCode::KeyZ => {
                                if let Some(action) = self.gameplay.undo_redo.pop_undo() {
                                    let mut chunks = std::collections::HashSet::new();
                                    for edit in action.edits.iter().rev() {
                                        let id = voxel_core::BlockId(edit.old_block);
                                        self.world_state.world.set_block(edit.x, edit.y, edit.z, id);
                                        let cp = voxel_core::math::block_to_chunk(
                                            glam::IVec3::new(edit.x, edit.y, edit.z),
                                        );
                                        chunks.insert(cp);
                                    }
                                    if let Some(s) = &self.world_state.streamer {
                                        for cp in chunks {
                                            s.request_remesh(cp);
                                        }
                                    }
                                    self.gameplay.chat.push_message(format!(
                                        "Undid {} block changes",
                                        action.edits.len()
                                    ));
                                }
                                return;
                            }
                            KeyCode::KeyY => {
                                if let Some(action) = self.gameplay.undo_redo.pop_redo() {
                                    let mut chunks = std::collections::HashSet::new();
                                    for edit in &action.edits {
                                        let id = voxel_core::BlockId(edit.new_block);
                                        self.world_state.world.set_block(edit.x, edit.y, edit.z, id);
                                        let cp = voxel_core::math::block_to_chunk(
                                            glam::IVec3::new(edit.x, edit.y, edit.z),
                                        );
                                        chunks.insert(cp);
                                    }
                                    if let Some(s) = &self.world_state.streamer {
                                        for cp in chunks {
                                            s.request_remesh(cp);
                                        }
                                    }
                                    self.gameplay.chat.push_message(format!(
                                        "Redid {} block changes",
                                        action.edits.len()
                                    ));
                                }
                                return;
                            }
                            _ => {}
                        }
                    }

                    // Movement actions (held): only in Playing state.
                    let action = if self.gameplay.game_state == GameState::Playing {
                        match code {
                            KeyCode::KeyW => Some(Action::Forward),
                            KeyCode::KeyS => Some(Action::Back),
                            KeyCode::KeyA => Some(Action::Left),
                            KeyCode::KeyD => Some(Action::Right),
                            KeyCode::Space => Some(Action::Jump),
                            KeyCode::ShiftLeft => Some(Action::Sneak),
                            KeyCode::ControlLeft => Some(Action::Sprint),
                            _ => None,
                        }
                    } else {
                        None
                    };

                    // Non-movement keybinds: looked up from config (only in Playing state).
                    if self.gameplay.game_state == GameState::Playing {
                    if let Some(&bound_action) = self.input.keybinds.get(&code) {
                        if pressed {
                            match bound_action {
                                Action::DebugOverlay => {
                                    self.gameplay.debug_overlay = !self.gameplay.debug_overlay;
                                }
                                Action::Wireframe => {
                                    if let Some(r) = self.render.renderer.as_mut() {
                                        r.toggle_wireframe();
                                    }
                                }
                                Action::Fly => {
                                    let new_flying = !self.simulation.player_flying();
                                    // Keep the ECS source of truth in sync.
                                    self.simulation.set_player_flying(new_flying);
                                    log::info!(
                                        "fly mode: {}",
                                        if new_flying { "ON" } else { "OFF" }
                                    );
                                }
                                Action::Chat => {
                                    if self.gameplay.game_state == GameState::Playing {
                                        self.gameplay.chat.open();
                                        self.unlock_cursor();
                                        self.input.input.held.clear();
                                        if let Some(w) = &self.render.window {
                                            w.set_ime_allowed(true);
                                        }
                                    }
                                }
                                Action::Screenshot => {
                                    self.input.screenshot_requested = true;
                                }
                                Action::Pause => {
                                    // Pause key only toggles Playing Γåö PauseMenu.
                                    if self.gameplay.game_state == GameState::Playing {
                                        self.gameplay.block_picker_open = false;
                                        self.gameplay.chat.open = false;
                                        self.enter_pause();
                                    } else if self.gameplay.game_state == GameState::PauseMenu {
                                        self.enter_playing();
                                    }
                                }
                                Action::RenderDistanceUp => {
                                    let r = (self.config.stream.load_radius + 1).min(16);
                                    self.config.stream.load_radius = r;
                                    if let Some(s) = &self.world_state.streamer {
                                        s.set_load_radius(r as u32);
                                    }
                                    self.gameplay.chat.push_message(format!("Render distance: {r}"));
                                }
                                Action::RenderDistanceDown => {
                                    let r = (self.config.stream.load_radius - 1).max(2);
                                    self.config.stream.load_radius = r;
                                    if let Some(s) = &self.world_state.streamer {
                                        s.set_load_radius(r as u32);
                                    }
                                    self.gameplay.chat.push_message(format!("Render distance: {r}"));
                                }
                                Action::BlockPicker => {
                                    if self.gameplay.game_state == GameState::Playing {
                                        self.gameplay.block_picker_open = !self.gameplay.block_picker_open;
                                        if self.gameplay.block_picker_open {
                                            self.unlock_cursor();
                                            self.input.input.held.clear();
                                        } else {
                                            self.lock_cursor();
                                        }
                                    }
                                }
                                Action::Profiler => {
                                    self.profiler.enabled = !self.profiler.enabled;
                                }
                                Action::ChunkDebug => {
                                    self.gameplay.chunk_debug_enabled = !self.gameplay.chunk_debug_enabled;
                                    self.gameplay.chat.push_message(format!(
                                        "Chunk debug: {}",
                                        if self.gameplay.chunk_debug_enabled {
                                            "ON"
                                        } else {
                                            "OFF"
                                        }
                                    ));
                                }
                                Action::EcsInspector => {
                                    self.gameplay.ecs_inspector = !self.gameplay.ecs_inspector;
                                    self.gameplay.chat.push_message(format!(
                                        "ECS inspector: {}",
                                        if self.gameplay.ecs_inspector {
                                            "ON"
                                        } else {
                                            "OFF"
                                        }
                                    ));
                                }
                                Action::PinUnpin => {
                                    if self.gameplay.pinned_entity.is_some() {
                                        self.gameplay.pinned_entity = None;
                                    } else {
                                        // Default the pin to the player so
                                        // a fresh F11 press immediately
                                        // snaps the camera to the
                                        // player (a no-op for the
                                        // existing camera math, but the
                                        // cycle keybinds then step to
                                        // the debug entities).
                                        self.gameplay.pinned_entity =
                                            self.simulation.player_entity();
                                    }
                                    let msg = match self.gameplay.pinned_entity {
                                        Some(e) => {
                                            format!("Pin: ON (e[{}:{}])", e.index, e.generation)
                                        }
                                        None => "Pin: OFF".to_string(),
                                    };
                                    self.gameplay.chat.push_message(msg);
                                }
                                Action::PinCycleNext => self.cycle_pin(1),
                                Action::PinCyclePrev => self.cycle_pin(-1),
                                Action::PinClear => {
                                    self.gameplay.pinned_entity = None;
                                    self.gameplay.chat.push_message("Pin: OFF".into());
                                }
                                Action::DeveloperConsole => {
                                    if self.gameplay.game_state == GameState::Playing {
                                        if self.gameplay.console.open {
                                            self.gameplay.console.close();
                                            self.lock_cursor();
                                        } else {
                                            self.gameplay.console.open();
                                            self.unlock_cursor();
                                            self.input.input.held.clear();
                                        }
                                    }
                                }
                                Action::TelemetryDashboard => {
                                    self.telemetry.toggle();
                                }
                                Action::EditMode => {
                                    if self.gameplay.game_state == GameState::Playing {
                                        self.gameplay.edit.toggle();
                                        if self.gameplay.edit.mode.is_active() {
                                            self.unlock_cursor();
                                        } else {
                                            self.lock_cursor();
                                        }
                                    }
                                }
                                Action::FullscreenMap
                                    if self.gameplay.game_state == GameState::Playing => {
                                        self.gameplay.map.fullscreen_open = !self.gameplay.map.fullscreen_open;
                                        if self.gameplay.map.fullscreen_open {
                                            self.unlock_cursor();
                                        } else {
                                            self.lock_cursor();
                                        }
                                    }
                                _ => {}
                            }
                        }
                    }
                    // Hotbar slot selection (Digit1-9) — always hardcoded.
                    if pressed {
                        let slot = match code {
                            KeyCode::Digit1 => Some(0),
                            KeyCode::Digit2 => Some(1),
                            KeyCode::Digit3 => Some(2),
                            KeyCode::Digit4 => Some(3),
                            KeyCode::Digit5 => Some(4),
                            KeyCode::Digit6 => Some(5),
                            KeyCode::Digit7 => Some(6),
                            KeyCode::Digit8 => Some(7),
                            KeyCode::Digit9 => Some(8),
                            _ => None,
                        };
                        if let Some(idx) = slot {
                            self.gameplay.hotbar.select(idx);
                            }
                    }
                    } // end Playing-only keybind guard
                    else if pressed {
                        // Escape is handled separately (not in keybind map) to
                        // avoid accidental re-binding.
                        if code == KeyCode::Escape {
                            match self.gameplay.game_state {
                                GameState::TitleScreen => {
                                    self.gameplay.want_exit = true;
                                }
                                GameState::WorldSelect => {
                                    if self.gameplay.create_world_state.is_some() {
                                        self.gameplay.create_world_state = None;
                                    } else {
                                        self.enter_title_screen();
                                    }
                                }
                                GameState::Playing => {
                                    if self.gameplay.block_picker_open {
                                        self.gameplay.block_picker_open = false;
                                        self.lock_cursor();
                                    } else {
                                        self.enter_pause();
                                    }
                                }
                                GameState::PauseMenu => {
                                    self.enter_playing();
                                }
                                GameState::SettingsMenu => {
                                    match self.gameplay.settings_previous {
                                        GameState::TitleScreen | GameState::WorldSelect => {
                                            self.enter_title_screen();
                                        }
                                        GameState::PauseMenu => {
                                            self.enter_pause();
                                        }
                                        _ => {
                                            self.enter_title_screen();
                                        }
                                    }
                                }
                            }
                        }
                        // Create world dialog keyboard input.
                        if self.gameplay.game_state == GameState::WorldSelect
                            && self.gameplay.create_world_state.is_some()
                        {
                            if let Some(ref mut state) = self.gameplay.create_world_state {
                                match code {
                                    KeyCode::Escape => {
                                        self.gameplay.create_world_state = None;
                                    }
                                    KeyCode::Tab => {
                                        state.active_field = 1 - state.active_field;
                                    }
                                    KeyCode::Enter => {
                                        let name = state.name.trim().to_string();
                                        if !name.is_empty() {
                                            let seed = state.seed.trim().to_string();
                                            let mode = state.game_mode.clone();
                                            let cheats = state.allow_cheats;
                                            self.gameplay.create_world_state = None;
                                            self.create_world_from_dialog(name, seed, mode, cheats);
                                        } else {
                                            state.error = Some("World name cannot be empty".into());
                                        }
                                    }
                                    KeyCode::Backspace => {
                                        let field = if state.active_field == 0 {
                                            &mut state.name
                                        } else {
                                            &mut state.seed
                                        };
                                        field.pop();
                                    }
                                    _ => {
                                        if let Some(ch) = physical_key_to_char(code) {
                                            let field = if state.active_field == 0 {
                                                &mut state.name
                                            } else {
                                                &mut state.seed
                                            };
                                            // Limit field length.
                                            if field.len() < 32 {
                                                field.push(ch);
                                            }
                                        }
                                    }
                                }
                            }
                            return;
                        }
                    }

                    if let Some(a) = action {
                        if pressed {
                            self.input.input.held.insert(a);
                        } else {
                            self.input.input.held.remove(&a);
                        }
                    }
                }
            }
            WindowEvent::Ime(Ime::Commit(text)) => {
                if self.gameplay.chat.open {
                    for ch in text.chars() {
                        self.gameplay.chat.push_char(ch);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match self.gameplay.game_state {
                    GameState::Playing => {
                        // When edit mode is active and cursor is unlocked, register
                        // clicks for UI interaction instead of re-locking the cursor.
                        if !self.input.cursor_locked && pressed {
                            if self.gameplay.edit.mode.is_active() {
                                if button == MouseButton::Left { self.gameplay.edit.ui_click = true }
                                return;
                            }
                            // If the block picker (creative inventory) is open,
                            // register the click for UI hit-testing instead of
                            // re-locking the cursor.
                            if self.gameplay.block_picker_open {
                                match button {
                                    MouseButton::Left => self.input.input.clicks.left = true,
                                    MouseButton::Right => self.input.input.clicks.right = true,
                                    _ => {}
                                }
                                return;
                            }
                            // If the player is dead, register the click for the
                            // death screen buttons instead of re-locking the cursor.
                            let is_dead = self.simulation.ecs_world()
                                .resource::<voxel_game::PlayerEntity>()
                                .and_then(|p| p.0)
                                .and_then(|e| self.simulation.ecs_world().get::<voxel_game::Health>(e))
                                .map(|h| h.dead)
                                .unwrap_or(false);
                            if is_dead {
                                match button {
                                    MouseButton::Left => self.input.input.clicks.left = true,
                                    MouseButton::Right => self.input.input.clicks.right = true,
                                    _ => {}
                                }
                                return;
                            }
                            // Clicking the window re-locks the cursor for mouse look.
                            self.lock_cursor();
                            return;
                        }
                        if self.input.cursor_locked {
                            match button {
                                MouseButton::Left => self.input.input.clicks.left = pressed,
                                MouseButton::Right => self.input.input.clicks.right = pressed,
                                _ => {}
                            }
                        }
                    }
                    GameState::PauseMenu | GameState::TitleScreen
                    | GameState::WorldSelect | GameState::SettingsMenu => {
                        // In menu states, register clicks for UI hit-testing.
                        if pressed {
                            match button {
                                MouseButton::Left => self.input.input.clicks.left = true,
                                MouseButton::Right => self.input.input.clicks.right = true,
                                _ => {}
                            }
                        }
                        // Track left-mouse held state for slider dragging.
                        if self.gameplay.game_state == GameState::SettingsMenu
                            && button == MouseButton::Left {
                                self.gameplay.settings_left_mouse_held = pressed;
                                if !pressed {
                                    self.gameplay.settings_slider_dragging = None;
                                }
                            }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                };
                // Accumulate scroll for palette when edit mode is active.
                if self.gameplay.edit.mode.is_active() {
                    self.gameplay.edit.scroll_delta += dy;
                }
                // Scroll creative inventory when open.
                if self.gameplay.block_picker_open {
                    if dy < 0.0 {
                        // Scroll down.
                        self.gameplay.creative_scroll = self.gameplay.creative_scroll.saturating_add(1);
                    } else if dy > 0.0 {
                        // Scroll up.
                        self.gameplay.creative_scroll = self.gameplay.creative_scroll.saturating_sub(1);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Track mouse position for UI hit-testing, in logical pixels
                // (divided by the DPI scale) to match UI layout coordinates.
                let s = self.render.ui_scale.max(0.25);
                self.gameplay.mouse_pos = (position.x as f32 / s, position.y as f32 / s);
            }
            WindowEvent::RedrawRequested => {
                if self.input.running {
                    self.frame();
                }
            }
            WindowEvent::Focused(false)
                // Auto-pause when the window loses focus — but not in capture
                // mode (headless verification), where the window may not have focus.
                if self.gameplay.game_state == GameState::Playing
                    && self.config.capture_after_frames.is_none() =>
            {
                self.enter_pause();
            }
            WindowEvent::Focused(true) => {
                // Re-set last_time so the first frame after refocus doesn't
                // compute a huge delta.
                self.input.last_time = Instant::now();
            }
            _ => {}
        }

        // Auto-exit after capture, if configured.
        if self.config.exit_after_capture && self.input.captured {
            event_loop.exit();
        }
        // Exit game button.
        if self.gameplay.want_exit {
            event_loop.exit();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if !self.input.cursor_locked {
            return;
        }
        if let DeviceEvent::MouseMotion { delta } = event {
            self.input.input.mouse_delta.0 += delta.0 as f32;
            self.input.input.mouse_delta.1 += delta.1 as f32;
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.render.window {
            if self.input.running && !self.gameplay.want_exit {
                w.request_redraw();
            }
        }
    }
}

#[cfg(test)]
mod eol_invariant {
    //! Walk every `crates/**/*.rs` file (excluding `target/`) and assert:
    //!   * valid UTF-8 (catches Windows-cp1252 leaks like `0x97` em-dash
    //!     that rustc's UTF-8 reader rejects at parse time)
    //!   * no bare `\r` byte (catches CRLF re-introduction — Rust raw
    //!     strings `r"\r"` reject bare CRs at parse time)
    //!
    //! Policy is enforced at three layers:
    //!   1. `.gitattributes` (`eol=lf` on git checkout)
    //!   2. `.editorconfig` (`end_of_line = lf` at editor save time)
    //!   3. **this test** — runs as part of `cargo test --workspace`
    use std::fs;
    use std::path::{Path, PathBuf};

    /// `crates/engine/Cargo.toml` ΓåÆ two directory levels up is the
    /// workspace root.
    fn workspace_root() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // crates/engine -> crates -> root
        p.pop();
        p.pop();
        p
    }

    /// Recursively walk `*.rs` files under `root`, skipping any `target/`.
    fn walk_rs(root: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        visit(root, &mut out);
        out
    }

    fn visit(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(read) = fs::read_dir(dir) else {
            return;
        };
        for e in read.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.file_name().and_then(|n| n.to_str()) == Some("target") {
                    continue;
                }
                visit(&p, out);
            } else if p.extension().and_then(|x| x.to_str()) == Some("rs") {
                out.push(p);
            }
        }
    }

    // NOTE: panic messages below use the curly `\u{2014}` form for the
    // em-dash. rustc 1.96 rejected the legacy 4-digit `\u2014` form in
    // this specific panic-string context with
    // `error: incorrect unicode escape sequence`. The curly form
    // compiles cleanly here and keeps the source ASCII. If you change
    // the form, verify with `cargo build -p voxel-engine` first;
    // do NOT switch back to legacy `\u2014` without re-validating
    // against rustc 1.96. The `mod eol_invariant` `//!` doc-comment
    // above intentionally uses raw UTF-8 em-dash bytes instead of
    // `\u{2014}` — doc comments don't trigger the panic-string parse
    // state, so either form works there; lazy uniformity didn't seem
    // worth a forced refactor. `docs/notes/u2014_repro.md` records the
    // 8-case empirical matrix that drove this form choice.

    #[test]
    fn all_crates_rs_files_are_pure_lf_utf8() {
        let files = walk_rs(&workspace_root());
        assert!(
            !files.is_empty(),
            "expected at least one .rs file under crates/"
        );

        for path in &files {
            let bytes =
                fs::read(path).unwrap_or_else(|e| panic!("read {} failed: {e}", path.display()));

            // UTF-8 validity.
            if let Err(e) = std::str::from_utf8(&bytes) {
                panic!(
                    "`{}` is not valid UTF-8: {e}\n                     (typical cause: a Windows-cp1252 byte leaked through \u{2014}                       re-save the file as UTF-8)",
                    path.display()
                );
            }

            // No bare CR bytes (CRLF is forbidden because Rust raw strings
            // cannot contain a bare CR).
            for (i, &b) in bytes.iter().enumerate() {
                if b == b'\r' {
                    let line = bytes[..i].iter().filter(|&&c| c == b'\n').count() + 1;
                    panic!(
                        "`{}` contains a bare CR at byte offset {} (line {}, 1-based) \u{2014}                          normalize to LF (see .gitattributes / .editorconfig)",
                        path.display(),
                        i,
                        line
                    );
                }
            }
        }
    }
}
