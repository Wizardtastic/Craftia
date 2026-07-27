//! Entity (player) persistence to / from `<save_dir>/entities.json`.
//!
//! World chunk persistence lives in `voxel_world::save`. This module handles
//! only the ECS-side player state (transform, velocity, AABB, input, state)
//! and world metadata (world_info.json) for the world selection screen.

use std::path::Path;

use voxel_game::Aabb;
use voxel_game::Experience;
use voxel_game::GameMode;
use voxel_game::Health;
use voxel_game::Hunger;
use voxel_game::PlayerEntity;
use voxel_game::PlayerInput;
use voxel_game::PlayerState;
use voxel_game::Transform;
use voxel_game::Velocity;

/// Metadata for a saved world, displayed on the world selection screen.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WorldInfo {
    pub name: String,
    pub seed: i32,
    pub created: String,
    pub last_played: String,
    pub game_mode: String,
    pub play_time_seconds: u64,
    pub version: u32,
    /// Flags for special world settings (e.g., "cheats" to enable commands).
    #[serde(default)]
    pub flags: Vec<String>,
    /// World difficulty: peaceful, easy, normal, hard.
    #[serde(default = "default_difficulty")]
    pub difficulty: String,
    /// Spawn position (x, y, z). None = auto-detect from terrain.
    #[serde(default)]
    pub spawn_position: Option<[i32; 3]>,
    /// Path to the save directory (not serialized, filled at scan time).
    #[serde(skip)]
    pub path: std::path::PathBuf,
}

fn default_difficulty() -> String {
    "normal".to_string()
}

impl WorldInfo {
    /// Create a default WorldInfo for a new world.
    pub fn new_default(name: &str, seed: i32) -> Self {
        Self {
            name: name.to_string(),
            seed,
            created: chrono_now(),
            last_played: "never".to_string(),
            game_mode: "survival".to_string(),
            play_time_seconds: 0,
            version: 1,
            flags: Vec::new(),
            difficulty: "normal".to_string(),
            spawn_position: None,
            path: std::path::PathBuf::new(),
        }
    }

    /// Whether cheats are enabled for this world.
    pub fn cheats_enabled(&self) -> bool {
        self.flags.contains(&"cheats".to_string())
    }
}

pub fn chrono_now() -> String {
    // Simple timestamp without pulling in chrono crate.
    // Returns seconds since epoch as a rough placeholder.
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_else(|_| "0".to_string())
}

/// Read world_info.json from a save directory. Returns None if missing.
pub fn read_world_info(dir: &Path) -> Option<WorldInfo> {
    let path = dir.join("world_info.json");
    let json = std::fs::read_to_string(&path).ok()?;
    let mut info: WorldInfo = serde_json::from_str(&json).ok()?;
    info.path = dir.to_path_buf();
    Some(info)
}

/// Write world_info.json to a save directory.
pub fn write_world_info(dir: &Path, info: &WorldInfo) -> anyhow::Result<()> {
    let path = dir.join("world_info.json");
    let json = serde_json::to_string_pretty(info)?;
    std::fs::write(&path, json)?;
    Ok(())
}

/// Scan the saves directory and return all world infos.
/// For directories with meta.bin but no world_info.json, creates a default one.
pub fn list_world_info(saves_dir: &Path) -> Vec<WorldInfo> {
    let mut worlds = Vec::new();
    if !saves_dir.exists() {
        return worlds;
    }
    let Ok(entries) = std::fs::read_dir(saves_dir) else {
        return worlds;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        // A valid save directory must have meta.bin.
        if !path.join("meta.bin").exists() {
            continue;
        }
        let info = if let Some(info) = read_world_info(&path) {
            info
        } else {
            // Backward compat: create world_info.json from directory name.
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Unknown")
                .to_string();
            let default_info = WorldInfo::new_default(&name, 0);
            let _ = write_world_info(&path, &default_info);
            let mut info = default_info;
            info.path = path.clone();
            info
        };
        worlds.push(info);
    }
    // Sort by last_played descending (most recent first).
    worlds.sort_by(|a, b| b.last_played.cmp(&a.last_played));
    worlds
}

/// Single source of truth for the JSON shape of a saved entity ΓÇö used by
/// both `save_entities` and `load_entities` since the field set is identical.
#[derive(serde::Serialize, serde::Deserialize)]
struct EntitySave {
    name: String,
    transform: Option<Transform>,
    velocity: Option<Velocity>,
    aabb: Option<Aabb>,
    player_input: Option<PlayerInput>,
    player_state: Option<PlayerState>,
    #[serde(default)]
    health: Option<Health>,
    #[serde(default)]
    hunger: Option<Hunger>,
    #[serde(default)]
    experience: Option<Experience>,
    #[serde(default)]
    game_mode: Option<GameMode>,
}

impl crate::EngineApp {
    /// Save all entity state (currently just the player's transform + state)
    /// to a JSON file in the save directory. The file is named
    /// `entities.json` and uses serde_json for human-readable format.
    pub(crate) fn save_entities(&self, save_dir: &std::path::Path) -> anyhow::Result<()> {
        let mut entries: Vec<EntitySave> = Vec::new();

        // Find all entities with CameraOwner (currently just the player).
        let ecs = self.simulation.ecs_world();
        for (_entity, camera_owner) in ecs.query::<&voxel_game::CameraOwner>() {
            let _ = camera_owner;
            // We found the player. Read their components.
            let player = match ecs.resource::<PlayerEntity>().and_then(|p| p.0) {
                Some(e) => e,
                None => continue,
            };
            entries.push(EntitySave {
                name: "player".to_string(),
                transform: ecs.get::<Transform>(player).copied(),
                velocity: ecs.get::<Velocity>(player).copied(),
                aabb: ecs.get::<Aabb>(player).copied(),
                player_input: ecs.get::<PlayerInput>(player).copied(),
                player_state: ecs.get::<PlayerState>(player).copied(),
                health: ecs.get::<Health>(player).copied(),
                hunger: ecs.get::<Hunger>(player).copied(),
                experience: ecs.get::<Experience>(player).copied(),
                game_mode: ecs.get::<GameMode>(player).copied(),
            });
            break;
        }

        let json = serde_json::to_string_pretty(&entries)?;
        std::fs::write(save_dir.join("entities.json"), json)?;
        Ok(())
    }

    /// Load entity state from a JSON file in the save directory and restore
    /// components on the existing player entity. A missing file is treated
    /// as "no entities to load" (not an error).
    pub(crate) fn load_entities(&mut self, save_dir: &std::path::Path) -> anyhow::Result<()> {
        let path = save_dir.join("entities.json");
        let json = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => return Ok(()), // no entities file = no entities to load
        };
        let entries: Vec<EntitySave> = serde_json::from_str(&json)?;

        let player = match self
            .simulation
            .ecs_world()
            .resource::<PlayerEntity>()
            .and_then(|p| p.0)
        {
            Some(e) => e,
            None => return Ok(()),
        };

        let ecs = self.simulation.ecs_world_mut();
        for entry in entries {
            if entry.name != "player" {
                continue;
            }
            if let Some(t) = entry.transform {
                ecs.set(player, t);
            }
            if let Some(v) = entry.velocity {
                ecs.set(player, v);
            }
            if let Some(a) = entry.aabb {
                ecs.set(player, a);
            }
            if let Some(pi) = entry.player_input {
                ecs.set(player, pi);
            }
            if let Some(ps) = entry.player_state {
                ecs.set(player, ps);
            }
            if let Some(h) = entry.health {
                ecs.set(player, h);
            }
            if let Some(hu) = entry.hunger {
                ecs.set(player, hu);
            }
            if let Some(xp) = entry.experience {
                ecs.set(player, xp);
            }
            if let Some(gm) = entry.game_mode {
                ecs.set(player, gm);
            }
            break;
        }
        Ok(())
    }
}
