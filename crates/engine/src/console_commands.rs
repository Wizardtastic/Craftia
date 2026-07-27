//! Console command dispatcher.
//!
//! The `execute_console_command` method on [`crate::EngineApp`] handles
//! introspection commands that return text to the console scrollback
//! rather than mutating game state (unlike the world-mutation commands
//! in `commands.rs`).

use voxel_game::CommandResult;

impl crate::EngineApp {
    /// Handle a console-specific command (non-block commands that return
    /// text to the scrollback rather than mutating game state).
    pub(crate) fn execute_console_command(&mut self, cmd: CommandResult) -> Vec<String> {
        match cmd {
            CommandResult::EcsList => {
                let mut lines = Vec::new();
                for arch in self.simulation.ecs_world().archetypes() {
                    let names: Vec<&str> = arch
                        .component_names
                        .iter()
                        .map(|n| n.rsplit("::").next().unwrap_or(n))
                        .collect();
                    lines.push(format!(
                        "Archetype[{}] {} (N={})",
                        arch.id.0,
                        names.join(", "),
                        arch.len()
                    ));
                    for &entity in arch.entities() {
                        lines.push(format!("  Entity[{}:{}]", entity.index, entity.generation));
                    }
                }
                if lines.is_empty() {
                    lines.push("No archetypes".into());
                }
                lines
            }
            CommandResult::EcsInspect { entity_id } => {
                let mut lines = Vec::new();
                for arch in self.simulation.ecs_world().archetypes() {
                    for (row, &entity) in arch.entities().iter().enumerate() {
                        if entity.index == entity_id {
                            lines.push(format!("Entity[{}:{}]", entity.index, entity.generation));
                            for (col_idx, &type_id) in arch.component_types.iter().enumerate() {
                                let name =
                                    arch.component_names.get(col_idx).copied().unwrap_or("?");
                                let short = name.rsplit("::").next().unwrap_or(name);
                                if let Some(any_ref) =
                                    arch.columns()[col_idx].value_as_any(row as u32)
                                {
                                    let formatted = self
                                        .simulation
                                        .ecs_world()
                                        .format_component(type_id, any_ref)
                                        .unwrap_or_else(|| "<opaque>".into());
                                    for line in formatted.lines() {
                                        lines.push(format!("  {}: {}", short, line));
                                    }
                                }
                            }
                            return lines;
                        }
                    }
                }
                vec![format!("Entity[{entity_id}] not found")]
            }
            CommandResult::EcsResources => {
                let mut lines = vec!["Resources:".into()];
                for tid in self.simulation.ecs_world().resource_type_ids() {
                    let name = self.simulation.ecs_world().name_for(tid);
                    if let Some(text) = self.simulation.ecs_world().format_resource(&tid) {
                        lines.push(format!("  {}:", name));
                        for sub in text.lines() {
                            lines.push(format!("    {}", sub));
                        }
                    } else {
                        lines.push(format!("  {}: <no formatter>", name));
                    }
                }
                if lines.len() == 1 {
                    lines.push("  (none)".into());
                }
                lines
            }
            CommandResult::EcsResource { type_name } => {
                let mut lines = Vec::new();
                for tid in self.simulation.ecs_world().resource_type_ids() {
                    let name = self.simulation.ecs_world().name_for(tid);
                    if name == type_name {
                        if let Some(text) = self.simulation.ecs_world().format_resource(&tid) {
                            lines.push(format!("{}:", name));
                            for sub in text.lines() {
                                lines.push(format!("  {}", sub));
                            }
                        } else {
                            lines.push(format!("{}: <no formatter>", name));
                        }
                        return lines;
                    }
                }
                vec![format!("Resource '{}' not found", type_name)]
            }
            CommandResult::Get { path } => {
                // Simple path-based config accessor:
                //   /get player.gravity
                //   /get render.fog_distance
                let strs: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
                match strs.as_slice() {
                    ["player", "gravity"] => {
                        vec![format!("player.gravity = {}", self.config.player.gravity)]
                    }
                    ["player", "walk_speed"] => {
                        vec![format!(
                            "player.walk_speed = {}",
                            self.config.player.walk_speed
                        )]
                    }
                    ["player", "sprint_speed"] => {
                        vec![format!(
                            "player.sprint_speed = {}",
                            self.config.player.sprint_speed
                        )]
                    }
                    ["player", "fly_speed"] => {
                        vec![format!(
                            "player.fly_speed = {}",
                            self.config.player.fly_speed
                        )]
                    }
                    ["player", "jump_speed"] => {
                        vec![format!(
                            "player.jump_speed = {}",
                            self.config.player.jump_speed
                        )]
                    }
                    ["render", "fog_distance"] => {
                        vec![format!(
                            "render.fog_distance = {}",
                            self.config.render.fog_distance
                        )]
                    }
                    ["world", "seed"] => {
                        vec![format!("world.seed = {}", self.config.seed)]
                    }
                    ["world", "load_radius"] => {
                        vec![format!(
                            "world.load_radius = {}",
                            self.config.stream.load_radius
                        )]
                    }
                    _ => vec![format!("unknown path: {:?}", path)],
                }
            }
            CommandResult::Set { path, value } => {
                // Hot-patch a value at runtime.
                let strs: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
                match strs.as_slice() {
                    ["player", "gravity"] => {
                        let v: f32 = match value.parse() {
                            Ok(v) => v,
                            Err(_) => return vec![format!("invalid float: {value}")],
                        };
                        self.config.player.gravity = v;
                        vec![format!("player.gravity = {v}")]
                    }
                    ["player", "walk_speed"] => {
                        let v: f32 = match value.parse() {
                            Ok(v) => v,
                            Err(_) => return vec![format!("invalid float: {value}")],
                        };
                        self.config.player.walk_speed = v;
                        vec![format!("player.walk_speed = {v}")]
                    }
                    ["player", "sprint_speed"] => {
                        let v: f32 = match value.parse() {
                            Ok(v) => v,
                            Err(_) => return vec![format!("invalid float: {value}")],
                        };
                        self.config.player.sprint_speed = v;
                        vec![format!("player.sprint_speed = {v}")]
                    }
                    ["player", "fly_speed"] => {
                        let v: f32 = match value.parse() {
                            Ok(v) => v,
                            Err(_) => return vec![format!("invalid float: {value}")],
                        };
                        self.config.player.fly_speed = v;
                        vec![format!("player.fly_speed = {v}")]
                    }
                    ["player", "jump_speed"] => {
                        let v: f32 = match value.parse() {
                            Ok(v) => v,
                            Err(_) => return vec![format!("invalid float: {value}")],
                        };
                        self.config.player.jump_speed = v;
                        vec![format!("player.jump_speed = {v}")]
                    }
                    ["render", "fog_distance"] => {
                        let v: f32 = match value.parse() {
                            Ok(v) => v,
                            Err(_) => return vec![format!("invalid float: {value}")],
                        };
                        self.config.render.fog_distance = v;
                        vec![format!("render.fog_distance = {v}")]
                    }
                    _ => vec![format!("unknown path: {:?}", path)],
                }
            }
            CommandResult::Exec(path) => match std::fs::read_to_string(&path) {
                Ok(content) => {
                    let lines: Vec<String> = content.lines().map(String::from).collect();
                    let count = lines.len();
                    self.gameplay.console_script = Some(lines);
                    vec![format!("Executing {count} commands from {path}")]
                }
                Err(e) => vec![format!("exec {path}: {e}")],
            },
            CommandResult::WaypointAdd { name, pos } => {
                let p = pos.unwrap_or_else(|| {
                    let pp = self
                        .simulation
                        .player_pos()
                        .unwrap_or(self.gameplay.spawn_pos);
                    glam::IVec3::new(
                        pp.x.floor() as i32,
                        pp.y.floor() as i32,
                        pp.z.floor() as i32,
                    )
                });
                self.gameplay
                    .map
                    .add_waypoint(name.clone(), p.x, p.y, p.z, [255, 200, 60, 255]);
                vec![format!(
                    "Waypoint '{}' added at ({}, {}, {})",
                    name, p.x, p.y, p.z
                )]
            }
            CommandResult::WaypointList => {
                if self.gameplay.map.waypoints.is_empty() {
                    vec!["No waypoints".into()]
                } else {
                    self.gameplay
                        .map
                        .waypoints
                        .iter()
                        .map(|wp| format!("  {} ({}, {}, {})", wp.name, wp.x, wp.y, wp.z))
                        .collect()
                }
            }
            CommandResult::WaypointRemove(name) => {
                if self.gameplay.map.remove_waypoint(&name) {
                    vec![format!("Waypoint '{}' removed", name)]
                } else {
                    vec![format!("Waypoint '{}' not found", name)]
                }
            }
            CommandResult::WaypointSave => {
                let path = std::path::Path::new("assets").join("waypoints.json");
                match self.gameplay.map.save_waypoints(&path) {
                    Ok(()) => vec!["Waypoints saved".into()],
                    Err(e) => vec![format!("Save failed: {}", e)],
                }
            }
            CommandResult::WaypointLoad => {
                let path = std::path::Path::new("assets").join("waypoints.json");
                match self.gameplay.map.load_waypoints(&path) {
                    Ok(()) => vec![format!(
                        "Loaded {} waypoints",
                        self.gameplay.map.waypoints.len()
                    )],
                    Err(e) => vec![format!("Load failed: {}", e)],
                }
            }
            _ => vec![],
        }
    }
}
