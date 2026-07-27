//! Minimap / fullscreen map state and rendering.
//!
//! The map system samples the world's top-block columns into a CPU-side
//! RGBA framebuffer, then uploads it as a GPU texture displayed via
//! `quad_uv` with `tex_id = 2.0`.

pub mod renderer;

use std::time::{Duration, Instant};

use voxel_world::map::ColumnSample;

/// Waypoint marker visible on the minimap / fullscreen map.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Waypoint {
    pub name: String,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub color: [u8; 4],
}

/// State for the minimap and fullscreen map system.
pub struct MapState {
    /// Whether the minimap HUD overlay is visible.
    pub visible: bool,
    /// Whether the fullscreen map is open.
    pub fullscreen_open: bool,

    // --- Rendering ---
    /// CPU-side RGBA framebuffer for the minimap texture.
    /// Built by `rebuild_framebuffer()` and uploaded to GPU when dirty.
    pub framebuffer: Vec<u8>,
    /// Dimensions of the framebuffer in pixels (square).
    pub fb_size: u32,

    // --- View parameters ---
    /// World-space center of the map (player XZ).
    pub center_x: i32,
    pub center_z: i32,
    /// Blocks per pixel (zoom level). Higher = more zoomed out.
    pub blocks_per_pixel: u32,
    /// Half-radius in blocks.
    pub radius_blocks: u32,

    // --- Dirtiness ---
    /// True when the framebuffer needs to be rebuilt.
    pub dirty: bool,
    /// True when the GPU texture needs to be re-uploaded.
    pub texture_dirty: bool,
    /// Last update time (for rate limiting).
    pub last_update: Instant,
    /// Minimum interval between rebuilds.
    pub update_interval: Duration,

    // --- Cached data ---
    /// Cached column samples from the world.
    pub samples: Vec<ColumnSample>,
    /// Last player chunk position for invalidation.
    pub last_player_x: i32,
    pub last_player_z: i32,

    // --- Waypoints ---
    pub waypoints: Vec<Waypoint>,
}

impl Default for MapState {
    fn default() -> Self {
        Self {
            visible: true,
            fullscreen_open: false,
            framebuffer: Vec::new(),
            fb_size: 256,
            center_x: 0,
            center_z: 0,
            blocks_per_pixel: 2,
            radius_blocks: 256,
            dirty: true,
            texture_dirty: false,
            last_update: Instant::now() - Duration::from_secs(10),
            update_interval: Duration::from_millis(500),
            samples: Vec::new(),
            last_player_x: i32::MAX,
            last_player_z: i32::MAX,
            waypoints: Vec::new(),
        }
    }
}

impl MapState {
    /// Check if the player has moved far enough to need a rebuild.
    pub fn check_dirty(&mut self, player_x: i32, player_z: i32) {
        let dx = (player_x - self.last_player_x).abs();
        let dz = (player_z - self.last_player_z).abs();
        let threshold = self.blocks_per_pixel.max(4) as i32;
        if dx > threshold || dz > threshold {
            self.center_x = player_x;
            self.center_z = player_z;
            self.last_player_x = player_x;
            self.last_player_z = player_z;
            self.dirty = true;
        }
    }

    /// Add a waypoint at the given position.
    pub fn add_waypoint(&mut self, name: String, x: i32, y: i32, z: i32, color: [u8; 4]) {
        self.waypoints.push(Waypoint {
            name,
            x,
            y,
            z,
            color,
        });
    }

    /// Remove a waypoint by name. Returns true if removed.
    pub fn remove_waypoint(&mut self, name: &str) -> bool {
        let len = self.waypoints.len();
        self.waypoints.retain(|w| w.name != name);
        self.waypoints.len() < len
    }

    /// Save waypoints to a JSON file.
    pub fn save_waypoints(&self, path: &std::path::Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.waypoints).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    /// Load waypoints from a JSON file.
    pub fn load_waypoints(&mut self, path: &std::path::Path) -> Result<(), String> {
        if !path.exists() {
            return Ok(());
        }
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        self.waypoints = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        Ok(())
    }
}
