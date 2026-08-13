use glam::Vec3;
use std::collections::VecDeque;

const MAX_MESSAGES: usize = 64;
const MAX_HISTORY: usize = 50;

/// All recognized commands and their sub-commands for autocomplete.
const COMMANDS: &[&str] = &[
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
    "/schematic",
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
];

/// Result of parsing a single `/`-prefixed chat input. Each variant carries
/// exactly the arguments the dispatcher needs to act; parsing errors are
/// surfaced as [`CommandResult::Unknown`].
///
/// Feature 4 added six volume-shape commands (`Hollow`, `Sphere`,
/// `Cylinder`, `Pyramid`, `Replace`, `Line`) and four schematic commands
/// (`SchematicSave`, `SchematicList`, `SchematicLoad`, `SchematicPaste`).
/// `SchematicPaste` uses a named struct so its optional `[ox oy oz]`
/// triple and named rotation/mirror flags remain readable inside the
/// dispatcher's `match` arms; the volume commands stay positional to match
/// the existing [`CommandResult::Fill`] shape.
pub enum CommandResult {
    Teleport(Vec3),
    SetTime(f64),
    TimeSpeed(f64),
    Give(String, i32),
    SetBlock(i32, i32, i32, String),
    Fill(i32, i32, i32, i32, i32, i32, String),
    /// `/hollow x1 y1 z1 x2 y2 z2 <shell> <block>` — fill the AABB shell of
    /// `shell` thick with `block`.
    Hollow(i32, i32, i32, i32, i32, i32, i32, String),
    /// `/sphere cx cy cz <radius> <block>` — Euclidean sphere fill.
    Sphere(i32, i32, i32, f32, String),
    /// `/cylinder bx by bz <radius> <height> <block>` — Y-up cylinder.
    Cylinder(i32, i32, i32, f32, f32, String),
    /// `/pyramid x1 y1 z1 x2 y2 z2 <block>` — square-base pyramid.
    Pyramid(i32, i32, i32, i32, i32, i32, String),
    /// `/replace x1 y1 z1 x2 y2 z2 <target> <replacement>` AABB swap.
    Replace(i32, i32, i32, i32, i32, i32, String, String),
    /// `/line ax ay az bx by bz <thickness> <block>` — Bresenham line.
    Line(i32, i32, i32, i32, i32, i32, i32, String),
    /// `/schematic save <name>` — snapshot the world through the chat's
    /// selection clipboard and write to `./schematics/<name>.schem`.
    SchematicSave(String),
    /// `/schematic list` — list pasted schematics in this world.
    SchematicList,
    /// `/schematic load <name>` — read `./schematics/<name>.schem`
    /// back into a Schematic-ready buffer.
    SchematicLoad(String),
    /// `/schematic paste <name> [ox oy oz] [rot0|rot90|rot180|rot270]
    /// [mx|my|mz|mxy|mxz|myz|mxyz|mnone]` — paste with optional transform.
    SchematicPaste {
        name: String,
        origin: Option<glam::IVec3>,
        rotation: voxel_world::schematic::Rotation90,
        mirror: voxel_world::schematic::MirrorAxes,
    },
    Gamemode(String),
    /// `/difficulty peaceful|easy|normal|hard` — set world difficulty.
    Difficulty(String),
    /// `/kill` — instant kill the player.
    Kill,
    Position,
    ChunkInfo,
    Fps,
    Reload,
    Clear,
    Save(String),
    Load(String),
    Copy(i32, i32, i32, i32, i32, i32),
    Paste,
    Help,
    Unknown(String),
    Empty,
    // -- REPL / introspection variants --
    /// `/ecs list` — return list of all entities with archetype info
    EcsList,
    /// `/ecs inspect <entity_id>` — return component dump for one entity
    EcsInspect {
        entity_id: u32,
    },
    /// `/ecs resources` — list all resources
    EcsResources,
    /// `/ecs resource <name>` — dump a specific resource
    EcsResource {
        type_name: String,
    },
    /// `/get <path>` — read a config/runtime value
    Get {
        path: Vec<String>,
    },
    /// `/set <path> <value>` — hot-patch a config/runtime value
    Set {
        path: Vec<String>,
        value: String,
    },
    /// `/exec <filename>` — run commands from a file
    Exec(String),
    // -- Waypoint commands --
    /// `/waypoint add <name> [x y z]` — add a waypoint at position.
    WaypointAdd {
        name: String,
        pos: Option<glam::IVec3>,
    },
    /// `/waypoint list` — list all waypoints.
    WaypointList,
    /// `/waypoint remove <name>` — remove a waypoint by name.
    WaypointRemove(String),
    /// `/waypoint save` — save waypoints to file.
    WaypointSave,
    /// `/waypoint load` — load waypoints from file.
    WaypointLoad,
}

#[derive(Default)]
pub struct ChatState {
    pub open: bool,
    pub input_buf: String,
    pub messages: VecDeque<String>,
    history: VecDeque<String>,
    history_index: Option<usize>,
}

impl ChatState {
    pub fn open(&mut self) {
        self.open = true;
        self.input_buf.clear();
        self.history_index = None;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.input_buf.clear();
        self.history_index = None;
    }

    pub fn submit(&mut self) -> CommandResult {
        let input = self.input_buf.trim().to_string();
        self.input_buf.clear();
        self.open = false;
        self.history_index = None;

        if input.is_empty() {
            return CommandResult::Empty;
        }

        // Record in history (no duplicates of the last entry).
        if self.history.front().map(|s| s.as_str()) != Some(&input) {
            self.history.push_front(input.clone());
            while self.history.len() > MAX_HISTORY {
                self.history.pop_back();
            }
        }

        self.push_message(format!("> {input}"));
        let result = Self::parse_command(&input, Vec3::ZERO);
        if let CommandResult::Unknown(msg) = &result {
            self.push_message(msg.clone());
        }
        result
    }

    pub fn submit_with_pos(&mut self, player_pos: Vec3) -> CommandResult {
        let input = self.input_buf.trim().to_string();
        self.input_buf.clear();
        self.open = false;
        self.history_index = None;

        if input.is_empty() {
            return CommandResult::Empty;
        }

        if self.history.front().map(|s| s.as_str()) != Some(&input) {
            self.history.push_front(input.clone());
            while self.history.len() > MAX_HISTORY {
                self.history.pop_back();
            }
        }

        self.push_message(format!("> {input}"));
        Self::parse_command(&input, player_pos)
    }

    /// Cycle history up (older).
    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.history_index {
            None => 0,
            Some(i) if i + 1 < self.history.len() => i + 1,
            _ => return,
        };
        self.history_index = Some(next);
        self.input_buf = self.history[next].clone();
    }

    /// Cycle history down (newer).
    pub fn history_down(&mut self) {
        match self.history_index {
            None => {}
            Some(0) => {
                self.history_index = None;
                self.input_buf.clear();
            }
            Some(i) => {
                let next = i - 1;
                self.history_index = Some(next);
                self.input_buf = self.history[next].clone();
            }
        }
    }

    /// Tab-complete the current input buffer against known commands.
    pub fn tab_complete(&mut self) {
        let input = self.input_buf.trim_start();
        if input.is_empty() {
            return;
        }
        let matches: Vec<&str> = COMMANDS
            .iter()
            .filter(|cmd| cmd.starts_with(input))
            .copied()
            .collect();
        if matches.len() == 1 {
            self.input_buf = format!("{} ", matches[0]);
        } else if matches.len() > 1 {
            let prefix = matches.iter().fold(matches[0], |acc, &s| {
                let common_len = acc
                    .chars()
                    .zip(s.chars())
                    .take_while(|(a, b)| a == b)
                    .count();
                &acc[..common_len]
            });
            if prefix.len() > input.len() {
                self.input_buf = prefix.to_string();
            } else {
                self.push_message(format!("Commands: {}", matches.join(", ")));
            }
        }
    }

    pub fn push_message(&mut self, msg: String) {
        self.messages.push_front(msg);
        while self.messages.len() > MAX_MESSAGES {
            self.messages.pop_back();
        }
    }

    pub fn push_char(&mut self, ch: char) {
        if self.open && !ch.is_control() {
            self.input_buf.push(ch);
            self.history_index = None;
        }
    }

    pub fn backspace(&mut self) {
        if self.open {
            self.input_buf.pop();
            self.history_index = None;
        }
    }

    pub fn parse_tp_args(args: &str, player_pos: Vec3) -> CommandResult {
        let parts: Vec<&str> = args.split_whitespace().collect();
        if parts.len() < 3 {
            return CommandResult::Unknown("/tp requires x y z".into());
        }
        let parse_coord = |s: &str, current: f32| -> Result<f32, String> {
            if let Some(rest) = s.strip_prefix('~') {
                let offset: f32 = if rest.is_empty() {
                    0.0
                } else {
                    rest.parse().map_err(|_| format!("invalid offset: {s}"))?
                };
                Ok(current + offset)
            } else {
                s.parse().map_err(|_| format!("invalid coordinate: {s}"))
            }
        };
        match (
            parse_coord(parts[0], player_pos.x),
            parse_coord(parts[1], player_pos.y),
            parse_coord(parts[2], player_pos.z),
        ) {
            (Ok(x), Ok(y), Ok(z)) => CommandResult::Teleport(Vec3::new(x, y, z)),
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => CommandResult::Unknown(e),
        }
    }

    pub fn parse_command(input: &str, player_pos: Vec3) -> CommandResult {
        let parts = parse_args(input);
        if parts.is_empty() {
            return CommandResult::Empty;
        }

        match parts[0].as_str() {
            "/tp" => Self::parse_tp_args(&parts[1..].join(" "), player_pos),
            "/time" => {
                if parts.len() < 2 {
                    return CommandResult::Unknown("/time requires set/speed".into());
                }
                match parts[1].as_str() {
                    "set" => {
                        if parts.len() < 3 {
                            return CommandResult::Unknown("/time set requires a value".into());
                        }
                        match parts[2].as_str() {
                            "day" => CommandResult::SetTime(0.0),
                            "night" => CommandResult::SetTime(0.5),
                            "dawn" => CommandResult::SetTime(0.15),
                            "dusk" => CommandResult::SetTime(0.65),
                            val => match val.parse::<f64>() {
                                Ok(v) => CommandResult::SetTime(v),
                                Err(_) => CommandResult::Unknown(format!("invalid time: {val}")),
                            },
                        }
                    }
                    "speed" => {
                        if parts.len() < 3 {
                            return CommandResult::Unknown(
                                "/time speed requires multiplier".into(),
                            );
                        }
                        match parts[2].parse::<f64>() {
                            Ok(v) if v > 0.0 => CommandResult::TimeSpeed(v),
                            _ => CommandResult::Unknown(format!("invalid speed: {}", parts[2])),
                        }
                    }
                    sub => CommandResult::Unknown(format!("unknown /time subcommand: {sub}")),
                }
            }
            "/give" => {
                if parts.len() < 2 {
                    return CommandResult::Unknown("/give requires <block> [count]".into());
                }
                let block = parts[1].clone();
                let count = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
                CommandResult::Give(block, count)
            }
            "/setblock" => {
                if parts.len() < 5 {
                    return CommandResult::Unknown("/setblock requires x y z <block>".into());
                }
                let x = parts[1].parse::<i32>();
                let y = parts[2].parse::<i32>();
                let z = parts[3].parse::<i32>();
                let block = parts[4].clone();
                match (x, y, z) {
                    (Ok(x), Ok(y), Ok(z)) => CommandResult::SetBlock(x, y, z, block),
                    _ => CommandResult::Unknown("invalid coordinates".into()),
                }
            }
            "/fill" => {
                if parts.len() < 8 {
                    return CommandResult::Unknown(
                        "/fill requires x1 y1 z1 x2 y2 z2 <block>".into(),
                    );
                }
                let x1 = parts[1].parse::<i32>();
                let y1 = parts[2].parse::<i32>();
                let z1 = parts[3].parse::<i32>();
                let x2 = parts[4].parse::<i32>();
                let y2 = parts[5].parse::<i32>();
                let z2 = parts[6].parse::<i32>();
                let block = parts[7].clone();
                match (x1, y1, z1, x2, y2, z2) {
                    (Ok(x1), Ok(y1), Ok(z1), Ok(x2), Ok(y2), Ok(z2)) => {
                        CommandResult::Fill(x1, y1, z1, x2, y2, z2, block)
                    }
                    _ => CommandResult::Unknown("invalid coordinates".into()),
                }
            }
            "/hollow" => {
                if parts.len() < 9 {
                    return CommandResult::Unknown(
                        "/hollow requires x1 y1 z1 x2 y2 z2 <shell> <block>".into(),
                    );
                }
                let x1 = parts[1].parse::<i32>();
                let y1 = parts[2].parse::<i32>();
                let z1 = parts[3].parse::<i32>();
                let x2 = parts[4].parse::<i32>();
                let y2 = parts[5].parse::<i32>();
                let z2 = parts[6].parse::<i32>();
                let shell = parts[7].parse::<i32>();
                let block = parts[8].clone();
                match (x1, y1, z1, x2, y2, z2, shell) {
                    (Ok(x1), Ok(y1), Ok(z1), Ok(x2), Ok(y2), Ok(z2), Ok(shell)) => {
                        CommandResult::Hollow(x1, y1, z1, x2, y2, z2, shell, block)
                    }
                    _ => CommandResult::Unknown("invalid /hollow arguments".into()),
                }
            }
            "/sphere" => {
                if parts.len() < 6 {
                    return CommandResult::Unknown(
                        "/sphere requires cx cy cz <radius> <block>".into(),
                    );
                }
                let cx = parts[1].parse::<i32>();
                let cy = parts[2].parse::<i32>();
                let cz = parts[3].parse::<i32>();
                let radius = parts[4].parse::<f32>();
                let block = parts[5].clone();
                match (cx, cy, cz, radius) {
                    (Ok(cx), Ok(cy), Ok(cz), Ok(radius)) => {
                        CommandResult::Sphere(cx, cy, cz, radius, block)
                    }
                    _ => CommandResult::Unknown("invalid /sphere arguments".into()),
                }
            }
            "/cylinder" => {
                if parts.len() < 7 {
                    return CommandResult::Unknown(
                        "/cylinder requires bx by bz <radius> <height> <block>".into(),
                    );
                }
                let bx = parts[1].parse::<i32>();
                let by = parts[2].parse::<i32>();
                let bz = parts[3].parse::<i32>();
                let radius = parts[4].parse::<f32>();
                let height = parts[5].parse::<f32>();
                let block = parts[6].clone();
                match (bx, by, bz, radius, height) {
                    (Ok(bx), Ok(by), Ok(bz), Ok(radius), Ok(height)) => {
                        CommandResult::Cylinder(bx, by, bz, radius, height, block)
                    }
                    _ => CommandResult::Unknown("invalid /cylinder arguments".into()),
                }
            }
            "/pyramid" => {
                if parts.len() < 8 {
                    return CommandResult::Unknown(
                        "/pyramid requires x1 y1 z1 x2 y2 z2 <block>".into(),
                    );
                }
                let x1 = parts[1].parse::<i32>();
                let y1 = parts[2].parse::<i32>();
                let z1 = parts[3].parse::<i32>();
                let x2 = parts[4].parse::<i32>();
                let y2 = parts[5].parse::<i32>();
                let z2 = parts[6].parse::<i32>();
                let block = parts[7].clone();
                match (x1, y1, z1, x2, y2, z2) {
                    (Ok(x1), Ok(y1), Ok(z1), Ok(x2), Ok(y2), Ok(z2)) => {
                        CommandResult::Pyramid(x1, y1, z1, x2, y2, z2, block)
                    }
                    _ => CommandResult::Unknown("invalid /pyramid arguments".into()),
                }
            }
            "/replace" => {
                if parts.len() < 9 {
                    return CommandResult::Unknown(
                        "/replace requires x1 y1 z1 x2 y2 z2 <target> <replacement>".into(),
                    );
                }
                let x1 = parts[1].parse::<i32>();
                let y1 = parts[2].parse::<i32>();
                let z1 = parts[3].parse::<i32>();
                let x2 = parts[4].parse::<i32>();
                let y2 = parts[5].parse::<i32>();
                let z2 = parts[6].parse::<i32>();
                let target = parts[7].clone();
                let replacement = parts[8].clone();
                match (x1, y1, z1, x2, y2, z2) {
                    (Ok(x1), Ok(y1), Ok(z1), Ok(x2), Ok(y2), Ok(z2)) => {
                        CommandResult::Replace(x1, y1, z1, x2, y2, z2, target, replacement)
                    }
                    _ => CommandResult::Unknown("invalid /replace arguments".into()),
                }
            }
            "/line" => {
                if parts.len() < 9 {
                    return CommandResult::Unknown(
                        "/line requires ax ay az bx by bz <thickness> <block>".into(),
                    );
                }
                let ax = parts[1].parse::<i32>();
                let ay = parts[2].parse::<i32>();
                let az = parts[3].parse::<i32>();
                let bx = parts[4].parse::<i32>();
                let by = parts[5].parse::<i32>();
                let bz = parts[6].parse::<i32>();
                let thickness = parts[7].parse::<i32>();
                let block = parts[8].clone();
                match (ax, ay, az, bx, by, bz, thickness) {
                    (Ok(ax), Ok(ay), Ok(az), Ok(bx), Ok(by), Ok(bz), Ok(thickness)) => {
                        CommandResult::Line(ax, ay, az, bx, by, bz, thickness, block)
                    }
                    _ => CommandResult::Unknown("invalid /line arguments".into()),
                }
            }
            "/schematic" => Self::parse_schematic(&parts[1..]),
            "/gamemode" => {
                let mode = parts
                    .get(1)
                    .map(|s| s.as_str())
                    .unwrap_or("creative")
                    .to_string();
                CommandResult::Gamemode(mode)
            }
            "/difficulty" => {
                let diff = parts
                    .get(1)
                    .map(|s| s.as_str())
                    .unwrap_or("normal")
                    .to_string();
                CommandResult::Difficulty(diff)
            }
            "/kill" => CommandResult::Kill,
            "/pos" => CommandResult::Position,
            "/chunk" => CommandResult::ChunkInfo,
            "/fps" => CommandResult::Fps,
            "/reload" => CommandResult::Reload,
            "/clear" => CommandResult::Clear,
            "/save" => {
                let path = parts
                    .get(1)
                    .map(|s| s.as_str())
                    .unwrap_or("world_save")
                    .to_string();
                CommandResult::Save(path)
            }
            "/load" => {
                let path = parts
                    .get(1)
                    .map(|s| s.as_str())
                    .unwrap_or("world_save")
                    .to_string();
                CommandResult::Load(path)
            }
            "/copy" => {
                if parts.len() < 7 {
                    return CommandResult::Unknown("/copy requires x1 y1 z1 x2 y2 z2".into());
                }
                let x1 = parts[1].parse::<i32>();
                let y1 = parts[2].parse::<i32>();
                let z1 = parts[3].parse::<i32>();
                let x2 = parts[4].parse::<i32>();
                let y2 = parts[5].parse::<i32>();
                let z2 = parts[6].parse::<i32>();
                match (x1, y1, z1, x2, y2, z2) {
                    (Ok(x1), Ok(y1), Ok(z1), Ok(x2), Ok(y2), Ok(z2)) => {
                        // Cap selection at 32x32x32 *cells* (inclusive). Each
                        // span from corner `a` to corner `b` covers
                        // (`a-b).abs() + 1` cells, so we compare the cell
                        // count rather than the bare extent — otherwise a
                        // user asking for "32 blocks wide" would silently get
                        // 33.
                        let cell_x = (x1 - x2).unsigned_abs() + 1;
                        let cell_y = (y1 - y2).unsigned_abs() + 1;
                        let cell_z = (z1 - z2).unsigned_abs() + 1;
                        if cell_x > 32 || cell_y > 32 || cell_z > 32 {
                            return CommandResult::Unknown(format!(
                                "/copy selection too large: \
                                 {cell_x}x{cell_y}x{cell_z} cells (max 32x32x32)"
                            ));
                        }
                        CommandResult::Copy(x1, y1, z1, x2, y2, z2)
                    }
                    _ => CommandResult::Unknown("invalid coordinates".into()),
                }
            }
            "/paste" => CommandResult::Paste,
            "/help" => CommandResult::Help,
            "/ecs" => Self::parse_ecs(&parts[1..]),
            "/get" => {
                if parts.len() < 2 {
                    return CommandResult::Unknown("/get requires a path".into());
                }
                let path: Vec<String> = parts[1].split('.').map(String::from).collect();
                CommandResult::Get { path }
            }
            "/set" => {
                if parts.len() < 3 {
                    return CommandResult::Unknown("/set requires <path> <value>".into());
                }
                let path: Vec<String> = parts[1].split('.').map(String::from).collect();
                let value = parts[2..].join(" ");
                CommandResult::Set { path, value }
            }
            "/exec" => {
                if parts.len() < 2 {
                    return CommandResult::Unknown("/exec requires a filename".into());
                }
                CommandResult::Exec(parts[1].clone())
            }
            "/waypoint" => {
                if parts.len() < 2 {
                    return CommandResult::Unknown(
                        "/waypoint requires add|list|remove|save|load".into(),
                    );
                }
                match parts[1].as_str() {
                    "add" => {
                        if parts.len() < 3 {
                            return CommandResult::Unknown("/waypoint add <name> [x y z]".into());
                        }
                        let name = parts[2].clone();
                        let pos = if parts.len() >= 6 {
                            let x = parts[3].parse::<i32>();
                            let y = parts[4].parse::<i32>();
                            let z = parts[5].parse::<i32>();
                            match (x, y, z) {
                                (Ok(x), Ok(y), Ok(z)) => Some(glam::IVec3::new(x, y, z)),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        CommandResult::WaypointAdd { name, pos }
                    }
                    "list" => CommandResult::WaypointList,
                    "remove" => {
                        if parts.len() < 3 {
                            return CommandResult::Unknown("/waypoint remove <name>".into());
                        }
                        CommandResult::WaypointRemove(parts[2].clone())
                    }
                    "save" => CommandResult::WaypointSave,
                    "load" => CommandResult::WaypointLoad,
                    _ => CommandResult::Unknown("/waypoint: unknown subcommand".into()),
                }
            }
            cmd if cmd.starts_with('/') => CommandResult::Unknown(cmd.to_string()),
            _ => CommandResult::Unknown("commands start with /".into()),
        }
    }

    /// Parse the post-`/ecs` tail. Subcommands:
    /// * `list`                   — list all entities with archetype info.
    /// * `inspect <entity_id>`    — component dump for one entity.
    /// * `resources`              — list all resources.
    /// * `resource <name>`        — dump a specific resource.
    fn parse_ecs(parts: &[String]) -> CommandResult {
        if parts.is_empty() {
            return CommandResult::Unknown("/ecs requires a subcommand".into());
        }
        match parts[0].as_str() {
            "list" => CommandResult::EcsList,
            "inspect" => {
                if parts.len() < 2 {
                    return CommandResult::Unknown("/ecs inspect requires an entity id".into());
                }
                match parts[1].parse::<u32>() {
                    Ok(id) => CommandResult::EcsInspect { entity_id: id },
                    Err(_) => CommandResult::Unknown(format!("invalid entity id: {}", parts[1])),
                }
            }
            "resources" => CommandResult::EcsResources,
            "resource" => {
                if parts.len() < 2 {
                    return CommandResult::Unknown("/ecs resource requires a type name".into());
                }
                CommandResult::EcsResource {
                    type_name: parts[1].clone(),
                }
            }
            sub => CommandResult::Unknown(format!("unknown /ecs subcommand: {sub}")),
        }
    }

    /// Parse the post-`/schematic` tail. Subcommands:
    /// * `save <name>`         — capture current selection.
    /// * `list`                — list pasted schematics in this world.
    /// * `load <name>`         — load a `.schem` from `./schematics/`.
    /// * `paste <name> [ox oy oz] [rot 0|90|180|270] [mirror x|y|z|xy|xz|yz|xyz]` —
    ///   paste with optional transform.
    fn parse_schematic(parts: &[String]) -> CommandResult {
        if parts.is_empty() {
            return CommandResult::Unknown("/schematic requires a subcommand".into());
        }
        match parts[0].as_str() {
            "save" => {
                if parts.len() < 2 {
                    return CommandResult::Unknown("/schematic save requires <name>".into());
                }
                CommandResult::SchematicSave(parts[1].clone())
            }
            "list" => CommandResult::SchematicList,
            "load" => {
                if parts.len() < 2 {
                    return CommandResult::Unknown("/schematic load requires <name>".into());
                }
                CommandResult::SchematicLoad(parts[1].clone())
            }
            "paste" => {
                if parts.len() < 2 {
                    return CommandResult::Unknown("/schematic paste requires <name>".into());
                }
                let name = parts[1].clone();
                let mut idx = 2;
                let mut origin = None;
                // Optional `[ox oy oz]` triple — only consumed if all three are
                // parseable as i32. Any non-i32 token falls through to the
                // flag parser below.
                if parts.len() >= idx + 3 {
                    if let (Ok(x), Ok(y), Ok(z)) = (
                        parts[idx].parse::<i32>(),
                        parts[idx + 1].parse::<i32>(),
                        parts[idx + 2].parse::<i32>(),
                    ) {
                        origin = Some(glam::IVec3::new(x, y, z));
                        idx += 3;
                    }
                }
                let mut rotation = voxel_world::schematic::Rotation90::Deg0;
                let mut mirror = voxel_world::schematic::MirrorAxes::NONE;
                // Named flags in any order.
                while idx < parts.len() {
                    match parts[idx].as_str() {
                        "rot0" => rotation = voxel_world::schematic::Rotation90::Deg0,
                        "rot90" => rotation = voxel_world::schematic::Rotation90::Deg90,
                        "rot180" => rotation = voxel_world::schematic::Rotation90::Deg180,
                        "rot270" => rotation = voxel_world::schematic::Rotation90::Deg270,
                        "mx" => mirror = voxel_world::schematic::MirrorAxes::X,
                        "my" => mirror = voxel_world::schematic::MirrorAxes::Y,
                        "mz" => mirror = voxel_world::schematic::MirrorAxes::Z,
                        "mxy" => mirror = voxel_world::schematic::MirrorAxes::XY,
                        "mxz" => mirror = voxel_world::schematic::MirrorAxes::XZ,
                        "myz" => mirror = voxel_world::schematic::MirrorAxes::YZ,
                        "mxyz" => mirror = voxel_world::schematic::MirrorAxes::ALL,
                        "mnone" => mirror = voxel_world::schematic::MirrorAxes::NONE,
                        other => {
                            return CommandResult::Unknown(format!(
                                "unknown /schematic paste flag: {other}"
                            ));
                        }
                    }
                    idx += 1;
                }
                CommandResult::SchematicPaste {
                    name,
                    origin,
                    rotation,
                    mirror,
                }
            }
            sub => CommandResult::Unknown(format!("unknown /schematic subcommand: {sub}")),
        }
    }
}

fn parse_args(input: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    for ch in input.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ' ' if !in_quotes => {
                if !current.is_empty() {
                    args.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty() {
        assert!(matches!(
            ChatState::parse_command("", Vec3::ZERO),
            CommandResult::Empty
        ));
    }

    #[test]
    fn parse_help() {
        assert!(matches!(
            ChatState::parse_command("/help", Vec3::ZERO),
            CommandResult::Help
        ));
    }

    #[test]
    fn parse_unknown_command() {
        assert!(matches!(
            ChatState::parse_command("/foo", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_no_slash() {
        assert!(matches!(
            ChatState::parse_command("hello", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_time_set_day() {
        let r = ChatState::parse_command("/time set day", Vec3::ZERO);
        assert!(matches!(r, CommandResult::SetTime(0.0)));
    }

    #[test]
    fn parse_time_set_night() {
        let r = ChatState::parse_command("/time set night", Vec3::ZERO);
        assert!(matches!(r, CommandResult::SetTime(0.5)));
    }

    #[test]
    fn parse_time_set_numeric() {
        let r = ChatState::parse_command("/time set 42.5", Vec3::ZERO);
        assert!(matches!(r, CommandResult::SetTime(v) if (v - 42.5).abs() < f64::EPSILON));
    }

    #[test]
    fn parse_time_set_invalid() {
        assert!(matches!(
            ChatState::parse_command("/time set abc", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_time_set_missing() {
        assert!(matches!(
            ChatState::parse_command("/time set", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_time_speed() {
        let r = ChatState::parse_command("/time speed 2.0", Vec3::ZERO);
        assert!(matches!(r, CommandResult::TimeSpeed(v) if (v - 2.0).abs() < f64::EPSILON));
    }

    #[test]
    fn parse_time_speed_zero() {
        assert!(matches!(
            ChatState::parse_command("/time speed 0", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_time_speed_negative() {
        assert!(matches!(
            ChatState::parse_command("/time speed -1", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_time_unknown_sub() {
        assert!(matches!(
            ChatState::parse_command("/time foo", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_time_missing_args() {
        assert!(matches!(
            ChatState::parse_command("/time", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_tp_missing_args() {
        assert!(matches!(
            ChatState::parse_command("/tp", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_tp_too_few() {
        assert!(matches!(
            ChatState::parse_command("/tp 1 2", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_tp_absolute() {
        let r = ChatState::parse_tp_args("10 20 30", Vec3::ZERO);
        if let CommandResult::Teleport(pos) = r {
            assert!((pos.x - 10.0).abs() < f32::EPSILON);
            assert!((pos.y - 20.0).abs() < f32::EPSILON);
            assert!((pos.z - 30.0).abs() < f32::EPSILON);
        } else {
            panic!("expected Teleport");
        }
    }

    #[test]
    fn parse_tp_relative() {
        let r = ChatState::parse_tp_args("~ ~10 ~-5", Vec3::new(100.0, 50.0, 200.0));
        if let CommandResult::Teleport(pos) = r {
            assert!((pos.x - 100.0).abs() < f32::EPSILON);
            assert!((pos.y - 60.0).abs() < f32::EPSILON);
            assert!((pos.z - 195.0).abs() < f32::EPSILON);
        } else {
            panic!("expected Teleport");
        }
    }

    #[test]
    fn parse_tpbare_tilde() {
        let r = ChatState::parse_tp_args("~ ~ ~", Vec3::new(5.0, 10.0, 15.0));
        if let CommandResult::Teleport(pos) = r {
            assert!((pos.x - 5.0).abs() < f32::EPSILON);
            assert!((pos.y - 10.0).abs() < f32::EPSILON);
            assert!((pos.z - 15.0).abs() < f32::EPSILON);
        } else {
            panic!("expected Teleport");
        }
    }

    #[test]
    fn chat_submit_empty() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf.clear();
        assert!(matches!(chat.submit(), CommandResult::Empty));
        assert!(!chat.open);
    }

    #[test]
    fn chat_submit_adds_message() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "/help".into();
        chat.submit();
        assert!(!chat.messages.is_empty());
        assert!(chat.messages[0].contains("/help"));
    }

    #[test]
    fn chat_push_char_ignores_when_closed() {
        let mut chat = ChatState::default();
        chat.push_char('a');
        assert!(chat.input_buf.is_empty());
    }

    #[test]
    fn chat_push_char_adds_when_open() {
        let mut chat = ChatState::default();
        chat.open();
        chat.push_char('h');
        chat.push_char('i');
        assert_eq!(chat.input_buf, "hi");
    }

    #[test]
    fn chat_push_char_ignores_control() {
        let mut chat = ChatState::default();
        chat.open();
        chat.push_char('\n');
        chat.push_char('\x1b');
        assert!(chat.input_buf.is_empty());
    }

    #[test]
    fn chat_backspace() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "abc".into();
        chat.backspace();
        assert_eq!(chat.input_buf, "ab");
        chat.backspace();
        assert_eq!(chat.input_buf, "a");
        chat.backspace();
        assert_eq!(chat.input_buf, "");
        chat.backspace();
        assert_eq!(chat.input_buf, "");
    }

    #[test]
    fn chat_message_limit() {
        let mut chat = ChatState::default();
        for i in 0..100 {
            chat.push_message(format!("msg {i}"));
        }
        assert!(chat.messages.len() <= 64);
    }

    // --- History tests ---

    #[test]
    fn history_up_populates_input() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "/help".into();
        chat.submit();
        chat.open();
        chat.history_up();
        assert_eq!(chat.input_buf, "/help");
    }

    #[test]
    fn history_down_clears() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "/help".into();
        chat.submit();
        chat.open();
        chat.history_up();
        chat.history_down();
        assert!(chat.input_buf.is_empty());
    }

    #[test]
    fn history_no_duplicates() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "/help".into();
        chat.submit();
        chat.open();
        chat.input_buf = "/help".into();
        chat.submit();
        assert_eq!(chat.history.len(), 1);
    }

    // --- Tab completion tests ---

    #[test]
    fn tab_complete_unique() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "/he".into();
        chat.tab_complete();
        assert_eq!(chat.input_buf, "/help ");
    }

    #[test]
    fn tab_complete_common_prefix() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "/ti".into();
        chat.tab_complete();
        assert!(chat.input_buf.len() > 3);
    }

    // --- Pre-existing tests continue below ---

    #[test]
    fn parse_give() {
        let r = ChatState::parse_command("/give stone 64", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Give(name, 64) if name == "stone"));
    }

    #[test]
    fn parse_give_default_count() {
        let r = ChatState::parse_command("/give dirt", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Give(name, 1) if name == "dirt"));
    }

    #[test]
    fn parse_give_missing() {
        assert!(matches!(
            ChatState::parse_command("/give", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_setblock() {
        let r = ChatState::parse_command("/setblock 10 20 30 stone", Vec3::ZERO);
        assert!(matches!(r, CommandResult::SetBlock(10, 20, 30, ref b) if b == "stone"));
    }

    #[test]
    fn parse_setblock_missing() {
        assert!(matches!(
            ChatState::parse_command("/setblock 1 2", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_fill() {
        let r = ChatState::parse_command("/fill 0 0 0 5 5 5 stone", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Fill(0, 0, 0, 5, 5, 5, ref b) if b == "stone"));
    }

    #[test]
    fn parse_fill_missing() {
        assert!(matches!(
            ChatState::parse_command("/fill 0 0 0 1 1", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_gamemode() {
        let r = ChatState::parse_command("/gamemode creative", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Gamemode(ref m) if m == "creative"));
    }

    #[test]
    fn parse_gamemode_default() {
        let r = ChatState::parse_command("/gamemode", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Gamemode(ref m) if m == "creative"));
    }

    #[test]
    fn parse_pos() {
        assert!(matches!(
            ChatState::parse_command("/pos", Vec3::ZERO),
            CommandResult::Position
        ));
    }

    #[test]
    fn parse_chunk() {
        assert!(matches!(
            ChatState::parse_command("/chunk", Vec3::ZERO),
            CommandResult::ChunkInfo
        ));
    }

    #[test]
    fn parse_fps() {
        assert!(matches!(
            ChatState::parse_command("/fps", Vec3::ZERO),
            CommandResult::Fps
        ));
    }

    #[test]
    fn parse_reload() {
        assert!(matches!(
            ChatState::parse_command("/reload", Vec3::ZERO),
            CommandResult::Reload
        ));
    }

    #[test]
    fn parse_clear() {
        assert!(matches!(
            ChatState::parse_command("/clear", Vec3::ZERO),
            CommandResult::Clear
        ));
    }

    #[test]
    fn parse_tp_relative_via_parse_command() {
        let r = ChatState::parse_command("/tp ~ ~10 ~", Vec3::new(100.0, 50.0, 200.0));
        if let CommandResult::Teleport(pos) = r {
            assert!((pos.x - 100.0).abs() < f32::EPSILON);
            assert!((pos.y - 60.0).abs() < f32::EPSILON);
            assert!((pos.z - 200.0).abs() < f32::EPSILON);
        } else {
            panic!("expected Teleport");
        }
    }

    // --- Feature 4 tests: volume commands ---

    #[test]
    fn parse_hollow_full() {
        let r = ChatState::parse_command("/hollow 0 0 0 5 5 5 2 stone", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Hollow(0,0,0,5,5,5,2,ref b) if b == "stone"));
    }

    #[test]
    fn parse_hollow_missing_args() {
        assert!(matches!(
            ChatState::parse_command("/hollow 0 0 0 5 5 5 stone", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_sphere_full() {
        let r = ChatState::parse_command("/sphere 10 20 30 4.5 glass", Vec3::ZERO);
        if let CommandResult::Sphere(cx, cy, cz, r, name) = r {
            assert_eq!(cx, 10);
            assert_eq!(cy, 20);
            assert_eq!(cz, 30);
            assert!((r - 4.5).abs() < f32::EPSILON);
            assert_eq!(name, "glass");
        } else {
            panic!("expected Sphere");
        }
    }

    #[test]
    fn parse_sphere_invalid_radius() {
        assert!(matches!(
            ChatState::parse_command("/sphere 0 0 0 abc stone", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_cylinder_full() {
        let r = ChatState::parse_command("/cylinder 0 0 0 3.0 5.0 sand", Vec3::ZERO);
        if let CommandResult::Cylinder(bx, by, bz, rad, h, name) = r {
            assert_eq!(bx, 0);
            assert_eq!(by, 0);
            assert_eq!(bz, 0);
            assert!((rad - 3.0).abs() < f32::EPSILON);
            assert!((h - 5.0).abs() < f32::EPSILON);
            assert_eq!(name, "sand");
        } else {
            panic!("expected Cylinder");
        }
    }

    #[test]
    fn parse_pyramid_full() {
        let r = ChatState::parse_command("/pyramid 0 0 0 5 10 5 stone", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Pyramid(0,0,0,5,10,5,ref b) if b == "stone"));
    }

    #[test]
    fn parse_replace_full() {
        let r = ChatState::parse_command("/replace 0 0 0 5 5 5 dirt grass", Vec3::ZERO);
        assert!(matches!(
            r,
            CommandResult::Replace(0,0,0,5,5,5,ref t, ref rp) if t == "dirt" && rp == "grass"
        ));
    }

    #[test]
    fn parse_replace_missing_replacement() {
        assert!(matches!(
            ChatState::parse_command("/replace 0 0 0 5 5 5 dirt", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_line_full() {
        let r = ChatState::parse_command("/line 0 0 0 10 10 10 2 stone", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Line(0,0,0,10,10,10,2,ref b) if b == "stone"));
    }

    // --- Feature 4 tests: schematic commands ---

    #[test]
    fn parse_schematic_missing_subcommand() {
        assert!(matches!(
            ChatState::parse_command("/schematic", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_schematic_save_full() {
        let r = ChatState::parse_command("/schematic save castle", Vec3::ZERO);
        assert!(matches!(r, CommandResult::SchematicSave(name) if name == "castle"));
    }

    #[test]
    fn parse_schematic_save_missing_name() {
        assert!(matches!(
            ChatState::parse_command("/schematic save", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_schematic_list() {
        assert!(matches!(
            ChatState::parse_command("/schematic list", Vec3::ZERO),
            CommandResult::SchematicList
        ));
    }

    #[test]
    fn parse_schematic_load_full() {
        let r = ChatState::parse_command("/schematic load castle", Vec3::ZERO);
        assert!(matches!(r, CommandResult::SchematicLoad(name) if name == "castle"));
    }

    #[test]
    fn parse_schematic_paste_basic() {
        let r = ChatState::parse_command("/schematic paste castle", Vec3::ZERO);
        if let CommandResult::SchematicPaste {
            name,
            origin,
            rotation,
            mirror,
        } = r
        {
            assert_eq!(name, "castle");
            assert!(origin.is_none());
            assert_eq!(rotation, voxel_world::schematic::Rotation90::Deg0);
            assert_eq!(mirror, voxel_world::schematic::MirrorAxes::NONE);
        } else {
            panic!("expected SchematicPaste");
        }
    }

    #[test]
    fn parse_schematic_paste_with_origin_and_transform() {
        let r = ChatState::parse_command("/schematic paste castle 5 10 15 rot90 mxz", Vec3::ZERO);
        if let CommandResult::SchematicPaste {
            name,
            origin,
            rotation,
            mirror,
        } = r
        {
            assert_eq!(name, "castle");
            assert_eq!(origin, Some(glam::IVec3::new(5, 10, 15)));
            assert_eq!(rotation, voxel_world::schematic::Rotation90::Deg90);
            assert_eq!(mirror, voxel_world::schematic::MirrorAxes::XZ);
        } else {
            panic!("expected SchematicPaste");
        }
    }

    #[test]
    fn parse_schematic_paste_flag_order_invariant() {
        // Mirror flag before rotation flag — both should still parse.
        let r = ChatState::parse_command("/schematic paste castle mx rot180", Vec3::ZERO);
        if let CommandResult::SchematicPaste {
            name,
            origin,
            rotation,
            mirror,
        } = r
        {
            assert_eq!(name, "castle");
            assert!(origin.is_none());
            assert_eq!(mirror, voxel_world::schematic::MirrorAxes::X);
            assert_eq!(rotation, voxel_world::schematic::Rotation90::Deg180);
        } else {
            panic!("expected SchematicPaste");
        }
    }

    #[test]
    fn parse_schematic_paste_unknown_flag_rejected() {
        assert!(matches!(
            ChatState::parse_command("/schematic paste castle roty", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_schematic_paste_missing_name() {
        assert!(matches!(
            ChatState::parse_command("/schematic paste", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_schematic_unknown_subcommand() {
        assert!(matches!(
            ChatState::parse_command("/schematic foo", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    // --- /copy size-limit regression tests (Bug #22) --------------

    /// 32 cells along each axis is the documented upper bound; the extent
    /// (corner-to-corner distance) is 31 so the total cell count is 32×32×32.
    #[test]
    fn parse_copy_just_at_limit_accepted() {
        // extent 31 -> 32 cells per axis -> 32 768 cells, accepted.
        let r = ChatState::parse_command("/copy 0 0 0 31 31 31 air", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Copy(0, 0, 0, 31, 31, 31)));
    }

    /// 33 cells along any axis exceeds the 32-cell bound.
    #[test]
    fn parse_copy_too_large_x_rejected() {
        let r = ChatState::parse_command("/copy 0 0 0 32 0 0 air", Vec3::ZERO);
        if let CommandResult::Unknown(msg) = r {
            assert!(msg.contains("too large"), "got: {msg}");
        } else {
            panic!("expected Unknown for oversized selection");
        }
    }

    /// Volume=32x32x31 cells is large enough to trip the limit (1<<5^3 = 32768 < 32^3 = 32768, so volumes in cells equal the limit when all axes match).
    #[test]
    fn parse_copy_too_large_volume_rejected() {
        // 32 cells per axis -> extent 31 -> inside limit; 33 cells -> extent 32 -> rejected.
        let r = ChatState::parse_command("/copy 0 0 0 32 32 32 air", Vec3::ZERO);
        if let CommandResult::Unknown(msg) = r {
            assert!(msg.contains("too large"), "got: {msg}");
            assert!(msg.contains("32"), "error should mention cell count: {msg}");
        } else {
            panic!("expected Unknown for oversized volume");
        }
    }

    /// Swap corners — the limit check must not assume x1<x2, y1<y2, z1<z2.
    #[test]
    fn parse_copy_swapped_corners_respected() {
        // Same as parse_copy_too_large_x_rejected but with corners reversed.
        let r = ChatState::parse_command("/copy 32 0 0 0 0 0 air", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Unknown(_)));
    }

    /// 1-cell selection is always allowed.
    #[test]
    fn parse_copy_single_cell_accepted() {
        let r = ChatState::parse_command("/copy 5 10 15 5 10 15 air", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Copy(5, 10, 15, 5, 10, 15)));
    }
}
