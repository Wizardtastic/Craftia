//! Chat command dispatcher.
//!
//! The `execute_command` method on [`crate::EngineApp`] is the single
//! destination for `/`-prefixed chat commands. Each volume-producing
//! command routes through the [`Self::run_volume`] helper, which opens
//! a Feature 1 `UndoRedoState` batch, streams edits through the matching
//! `World` volume method, and commits ΓÇö so thousands of `set_block`
//! calls collapse to one undo step.

use voxel_core::BlockId;

use voxel_game::CommandResult;
use voxel_world::schematic::{Schematic, SchematicId};
use voxel_world::volume::BlockChange;

impl crate::EngineApp {
    pub(crate) fn execute_command(&mut self, result: CommandResult) {
        match result {
            CommandResult::Teleport(target) => {
                // Player position is ECS-backed; the ECS path is the source of
                // truth. The legacy `GamePlayState::player` field was removed
                // when the player migrated to an entity + components.
                self.simulation.set_player_pos(target);
                if let Some(cam) = self
                    .simulation
                    .ecs_world_mut()
                    .resource_mut::<voxel_game::CameraResource>()
                {
                    cam.0.pos = target + glam::Vec3::new(0.0, voxel_game::EYE_HEIGHT, 0.0);
                }
                self.gameplay.chat.push_message(format!(
                    "Teleported to ({:.1}, {:.1}, {:.1})",
                    target.x, target.y, target.z
                ));
            }
            CommandResult::SetTime(t) => {
                self.gameplay.game_time = t % self.gameplay.day_length;
                self.gameplay
                    .chat
                    .push_message(format!("Time set to {:.0}s", self.gameplay.game_time));
            }
            CommandResult::TimeSpeed(mult) => {
                if mult <= 0.0 {
                    self.gameplay
                        .chat
                        .push_message("time speed must be positive".into());
                    return;
                }
                self.gameplay.day_length = self.config.day_length / mult;
                self.gameplay.chat.push_message(format!(
                    "Day length: {:.0}s ({:.1}x speed)",
                    self.gameplay.day_length, mult
                ));
            }
            CommandResult::Give(block, count) => {
                self.gameplay
                    .chat
                    .push_message(format!("Gave {count} {block} (not yet implemented)"));
            }
            CommandResult::SetBlock(x, y, z, block) => {
                let reg = self.world_state.world.registry_ref();
                let Some(id) = reg.id_of(&block) else {
                    self.gameplay
                        .chat
                        .push_message(format!("Unknown block: {block}"));
                    return;
                };
                self.gameplay
                    .undo_redo
                    .begin_batch(format!("setblock {block}"));
                let mut edits = Vec::new();
                let old = self.world_state.world.get_block(x, y, z);
                let written = self.world_state.world.set_block(x, y, z, id);
                if written && old != id {
                    edits.push(voxel_game::BlockEdit {
                        x,
                        y,
                        z,
                        old_block: old.0,
                        new_block: id.0,
                    });
                }
                // Push the per-block edits through the active batch.
                for e in &edits {
                    self.gameplay.undo_redo.push_edit_batched(e.clone());
                }
                self.gameplay.undo_redo.commit_batch();
                if edits.is_empty() {
                    self.gameplay
                        .chat
                        .push_message(format!("Block at ({x}, {y}, {z}) unchanged"));
                } else {
                    self.gameplay
                        .chat
                        .push_message(format!("Block set at ({x}, {y}, {z})"));
                }
            }
            CommandResult::Fill(x1, y1, z1, x2, y2, z2, block) => {
                let reg = self.world_state.world.registry_ref();
                let Some(id) = reg.id_of(&block) else {
                    self.gameplay
                        .chat
                        .push_message(format!("Unknown block: {block}"));
                    return;
                };
                let bounds = bounds_from_corners(x1, y1, z1, x2, y2, z2);
                let undo = &mut self.gameplay.undo_redo;
                let world = &self.world_state.world;
                undo.begin_batch(format!("fill {block}"));
                let count = world.fill_aabb(bounds, id, push_to_undo(undo));
                undo.commit_batch();
                self.gameplay
                    .chat
                    .push_message(format!("Filled {count} blocks"));
            }

            // --- Feature 4: volumetric shape commands ---------------------
            CommandResult::Hollow(x1, y1, z1, x2, y2, z2, shell, block) => {
                self.run_volume("hollow", &block, |world, id, undo| {
                    let bounds = bounds_from_corners(x1, y1, z1, x2, y2, z2);
                    let shell = shell.max(1);
                    world.hollow_aabb(bounds, id, shell, push_to_undo(undo))
                });
            }
            CommandResult::Sphere(cx, cy, cz, radius, block) => {
                self.run_volume("sphere", &block, |world, id, undo| {
                    world.fill_sphere((cx, cy, cz), radius, id, push_to_undo(undo))
                });
            }
            CommandResult::Cylinder(bx, by, bz, radius, height, block) => {
                self.run_volume("cylinder", &block, |world, id, undo| {
                    world.fill_cylinder((bx, by, bz), radius, height, id, push_to_undo(undo))
                });
            }
            CommandResult::Pyramid(x1, y1, z1, x2, y2, z2, block) => {
                self.run_volume("pyramid", &block, |world, id, undo| {
                    let bounds = bounds_from_corners(x1, y1, z1, x2, y2, z2);
                    world.fill_pyramid(bounds, id, push_to_undo(undo))
                });
            }
            CommandResult::Replace(x1, y1, z1, x2, y2, z2, target, replacement) => {
                self.run_volume_replace(&target, &replacement, |world, undo| {
                    let bounds = bounds_from_corners(x1, y1, z1, x2, y2, z2);
                    world.replace_in_aabb(
                        bounds,
                        world.registry_ref().id_of(&target).unwrap_or(BlockId::AIR),
                        world
                            .registry_ref()
                            .id_of(&replacement)
                            .unwrap_or(BlockId::AIR),
                        push_to_undo(undo),
                    )
                });
            }
            CommandResult::Line(ax, ay, az, bx, by, bz, thickness, block) => {
                self.run_volume("line", &block, |world, id, undo| {
                    world.fill_line(
                        (ax, ay, az),
                        (bx, by, bz),
                        thickness,
                        id,
                        push_to_undo(undo),
                    )
                });
            }

            // --- Feature 4: schematic commands -----------------------------
            CommandResult::SchematicSave(name) => {
                let Some(((min_x, min_y, min_z), (max_x, max_y, max_z), _)) =
                    &self.gameplay.clipboard
                else {
                    self.gameplay
                        .chat
                        .push_message("/schematic save requires a /copy'd selection".into());
                    return;
                };
                let bounds = voxel_core::Aabb {
                    min: glam::Vec3::new(*min_x as f32, *min_y as f32, *min_z as f32),
                    max: glam::Vec3::new(
                        (*max_x as f32) + 1.0,
                        (*max_y as f32) + 1.0,
                        (*max_z as f32) + 1.0,
                    ),
                };
                let id = SchematicId::new(name.clone());
                let reg = self.world_state.world.registry_ref();
                let schem = Schematic::capture(id, bounds, &self.world_state.world, reg);

                let dir = std::env::current_dir()
                    .unwrap_or_default()
                    .join("schematics");
                let _ = std::fs::create_dir_all(&dir);
                let path = dir.join(format!("{name}.schem"));
                match schem.save(&path) {
                    Ok(()) => self
                        .gameplay
                        .chat
                        .push_message(format!("Saved schematic to {}", path.display())),
                    Err(e) => self.gameplay.chat.push_message(format!("Save failed: {e}")),
                }
            }
            CommandResult::SchematicList => {
                let list = self.world_state.world.pasted_schematics();
                if list.is_empty() {
                    self.gameplay
                        .chat
                        .push_message("No schematics pasted yet".into());
                    return;
                }
                let mut msg = format!("Pasted schematics ({}):\n", list.len());
                for entity in &list {
                    msg.push_str(&format!(
                        "  {} - origin {:?} rot {:?} mirror {} blocks - {} blocks written\n",
                        entity.id,
                        entity.origin,
                        entity.rotation,
                        entity.mirror.0,
                        entity.pasted_blocks,
                    ));
                }
                self.gameplay.chat.push_message(msg);
            }
            CommandResult::SchematicLoad(name) => {
                // v1: just verify the file is readable from ./schematics/.
                // The actual paste is via /schematic paste which loads
                // fresh on demand.
                let dir = std::env::current_dir()
                    .unwrap_or_default()
                    .join("schematics");
                let path = dir.join(format!("{name}.schem"));
                let id = SchematicId::new(name.clone());
                match Schematic::load(&path, id) {
                    Ok(s) => self.gameplay.chat.push_message(format!(
                        "Schematic '{}' is valid ({} voxels). Use /schematic paste {} [ox oy oz] [rot] [mirror] to apply.",
                        name,
                        s.voxel_count(),
                        name
                    )),
                    Err(e) => self
                        .gameplay
                        .chat
                        .push_message(format!("Load failed: {e}")),
                }
            }
            CommandResult::SchematicPaste {
                name,
                origin,
                rotation,
                mirror,
            } => {
                let dir = std::env::current_dir()
                    .unwrap_or_default()
                    .join("schematics");
                let path = dir.join(format!("{name}.schem"));
                let id = SchematicId::new(name.clone());
                let schem = match Schematic::load(&path, id) {
                    Ok(s) => s,
                    Err(e) => {
                        self.gameplay.chat.push_message(format!("Load failed: {e}"));
                        return;
                    }
                };
                let paste_origin = origin.unwrap_or_else(|| glam::IVec3::ZERO);
                let undo = &mut self.gameplay.undo_redo;
                undo.begin_batch(format!("schematic_paste {name}"));
                let entity_opt = self.world_state.world.paste_schematic(
                    &schem,
                    paste_origin,
                    rotation,
                    mirror,
                    push_to_undo(undo),
                );
                undo.commit_batch();
                match entity_opt {
                    Some(entity) => self.gameplay.chat.push_message(format!(
                        "Pasted '{}' ({} blocks at {:?})",
                        entity.id, entity.pasted_blocks, entity.origin
                    )),
                    None => self.gameplay.chat.push_message(format!(
                        "Paste of '{name}' wrote zero blocks (chunks unloaded?)"
                    )),
                }
            }

            // --- Pre-existing commands -------------------------------------
            CommandResult::Gamemode(mode) => {
                if !self.gameplay.cheats_enabled {
                    self.gameplay
                        .chat
                        .push_message("Cheats must be enabled to use /gamemode".into());
                    return;
                }
                let new_mode = match voxel_game::GameMode::from_str_loose(&mode) {
                    Some(m) => m,
                    None => {
                        self.gameplay.chat.push_message(
                            "Usage: /gamemode survival|creative|adventure|spectator".into(),
                        );
                        return;
                    }
                };
                // Update the player's GameMode component in the ECS.
                if let Some(player) = self.simulation.player_entity() {
                    self.simulation.ecs_world_mut().set(player, new_mode);
                }
                // Update world_info.json to persist the change.
                if let Some(ref save_path) = self.gameplay.current_world_path {
                    if let Some(mut info) = crate::save::read_world_info(save_path) {
                        info.game_mode = new_mode.display_name().to_lowercase();
                        let _ = crate::save::write_world_info(save_path, &info);
                    }
                }
                self.gameplay
                    .chat
                    .push_message(format!("Game mode set to {}", new_mode.display_name()));
            }
            CommandResult::Difficulty(diff) => {
                if !self.gameplay.cheats_enabled {
                    self.gameplay
                        .chat
                        .push_message("Cheats must be enabled to use /difficulty".into());
                    return;
                }
                let new_diff = match voxel_hunger::Difficulty::from_str(&diff) {
                    Some(d) => d,
                    None => {
                        self.gameplay
                            .chat
                            .push_message("Usage: /difficulty peaceful|easy|normal|hard".into());
                        return;
                    }
                };
                // Update the difficulty resource.
                if let Some(diff_res) = self
                    .simulation
                    .ecs_world_mut()
                    .resource_mut::<voxel_game::DifficultyResource>()
                {
                    diff_res.0 = new_diff;
                }
                // Update world_info.json to persist the change.
                if let Some(ref save_path) = self.gameplay.current_world_path {
                    if let Some(mut info) = crate::save::read_world_info(save_path) {
                        info.difficulty = new_diff.display_name().to_lowercase();
                        let _ = crate::save::write_world_info(save_path, &info);
                    }
                }
                self.gameplay
                    .chat
                    .push_message(format!("Difficulty set to {}", new_diff.display_name()));
            }
            CommandResult::Kill => {
                if !self.gameplay.cheats_enabled {
                    self.gameplay
                        .chat
                        .push_message("Cheats must be enabled to use /kill".into());
                    return;
                }
                // Deal massive damage to kill the player.
                if let Some(player) = self.simulation.player_entity() {
                    if let Some(dq) = self
                        .simulation
                        .ecs_world_mut()
                        .resource_mut::<voxel_game::DamageQueue>()
                    {
                        dq.push(
                            player,
                            voxel_game::DamageEvent {
                                source: voxel_game::DamageSource::Void,
                                amount: 1000.0,
                            },
                        );
                    }
                }
                self.gameplay.chat.push_message("Killed player".into());
            }
            CommandResult::Position => {
                let p = self
                    .simulation
                    .player_pos()
                    .unwrap_or(self.gameplay.spawn_pos);
                self.gameplay
                    .chat
                    .push_message(format!("Pos: ({:.1}, {:.1}, {:.1})", p.x, p.y, p.z));
            }
            CommandResult::ChunkInfo => {
                let p = self
                    .simulation
                    .player_pos()
                    .unwrap_or(self.gameplay.spawn_pos);
                let block = voxel_core::math::world_to_block(p);
                let cp = voxel_core::math::block_to_chunk(block);
                let loaded = self.world_state.world.loaded_chunk_count();
                let meshed = self.world_state.world.meshed_chunk_count();
                self.gameplay.chat.push_message(format!(
                    "Chunk ({}, {}), loaded: {loaded}, meshed: {meshed}",
                    cp.x(),
                    cp.z()
                ));
            }
            CommandResult::Fps => {
                self.gameplay.chat.push_message(format!(
                    "Frame time: {:.1}ms ({:.0} fps)",
                    self.input.frame_time * 1000.0,
                    1.0 / self.input.frame_time.max(0.001)
                ));
            }
            CommandResult::Reload => {
                self.gameplay
                    .chat
                    .push_message("Config reloaded (not yet implemented)".into());
            }
            CommandResult::Clear => {
                self.gameplay.chat.messages.clear();
                self.gameplay.chat.push_message("Chat cleared".into());
            }
            CommandResult::Save(path) => {
                let save_dir = std::path::PathBuf::from(&path);
                match voxel_world::save::save_world(&self.world_state.world, &save_dir) {
                    Ok(()) => {
                        if let Err(e) = self.save_entities(&save_dir) {
                            self.gameplay
                                .chat
                                .push_message(format!("World saved but entity save failed: {e}"));
                        } else {
                            self.gameplay
                                .chat
                                .push_message(format!("World saved to {path}"));
                        }
                    }
                    Err(e) => self.gameplay.chat.push_message(format!("Save failed: {e}")),
                }
            }
            CommandResult::Load(path) => {
                let save_dir = std::path::PathBuf::from(&path);
                match voxel_world::save::load_world(&save_dir) {
                    Ok((seed, chunks)) => {
                        self.config.seed = seed;
                        let count = chunks.len();
                        self.world_state.world.insert_chunks(chunks);
                        if let Err(e) = self.load_entities(&save_dir) {
                            self.gameplay.chat.push_message(format!(
                                "Loaded {count} chunks but entity load failed: {e}"
                            ));
                        } else {
                            self.gameplay
                                .chat
                                .push_message(format!("Loaded {count} chunks from {path}"));
                        }
                        let pos = self
                            .simulation
                            .player_pos()
                            .unwrap_or(self.gameplay.spawn_pos);
                        if let Some(s) = &self.world_state.streamer {
                            s.set_focus(pos);
                        }
                    }
                    Err(e) => self.gameplay.chat.push_message(format!("Load failed: {e}")),
                }
            }
            CommandResult::Copy(x1, y1, z1, x2, y2, z2) => {
                let min_x = x1.min(x2);
                let max_x = x1.max(x2);
                let min_y = y1.min(y2);
                let max_y = y1.max(y2);
                let min_z = z1.min(z2);
                let max_z = z1.max(z2);
                let mut blocks = Vec::new();
                for x in min_x..=max_x {
                    for y in min_y..=max_y {
                        for z in min_z..=max_z {
                            blocks.push(self.world_state.world.get_block(x, y, z));
                        }
                    }
                }
                let count = blocks.len();
                self.gameplay.clipboard =
                    Some(((min_x, min_y, min_z), (max_x, max_y, max_z), blocks));
                let size = (max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1);
                self.gameplay
                    .chat
                    .push_message(format!("Copied {size} blocks ({count} stored)"));
            }
            CommandResult::Paste => {
                let Some(((min_x, min_y, min_z), (max_x, max_y, max_z), blocks)) =
                    &self.gameplay.clipboard
                else {
                    self.gameplay.chat.push_message("Clipboard empty".into());
                    return;
                };
                let sx = max_x - min_x + 1;
                let sy = max_y - min_y + 1;
                let mut count = 0u32;
                let mut idx = 0;
                self.gameplay
                    .undo_redo
                    .begin_batch("clipboard_paste".to_string());
                for x in 0..sx {
                    for y in 0..sy {
                        for z in 0..(max_z - min_z + 1) {
                            let id = blocks[idx];
                            let old =
                                self.world_state
                                    .world
                                    .get_block(min_x + x, min_y + y, min_z + z);
                            let _ = self.world_state.world.set_block(
                                min_x + x,
                                min_y + y,
                                min_z + z,
                                id,
                            );
                            self.gameplay
                                .undo_redo
                                .push_edit_batched(voxel_game::BlockEdit {
                                    x: min_x + x,
                                    y: min_y + y,
                                    z: min_z + z,
                                    old_block: old.0,
                                    new_block: id.0,
                                });
                            count += 1;
                            idx += 1;
                        }
                    }
                }
                self.gameplay.undo_redo.commit_batch();
                self.gameplay
                    .chat
                    .push_message(format!("Pasted {count} blocks"));
            }
            CommandResult::Help => {
                self.gameplay.chat.push_message("Commands:".into());
                self.gameplay
                    .chat
                    .push_message("  /tp x y z        - teleport (~ for relative)".into());
                self.gameplay.chat.push_message(
                    "  /time set <val>  - set time (day/night/dawn/dusk/seconds)".into(),
                );
                self.gameplay
                    .chat
                    .push_message("  /time speed <x>  - set time speed multiplier".into());
                self.gameplay
                    .chat
                    .push_message("  /give <block> [n]- give block items (WIP)".into());
                self.gameplay
                    .chat
                    .push_message("  /setblock x y z <block>".into());
                self.gameplay
                    .chat
                    .push_message("  /fill x1 y1 z1 x2 y2 z2 <block>".into());
                self.gameplay
                    .chat
                    .push_message("  /hollow x1 y1 z1 x2 y2 z2 <shell> <block>".into());
                self.gameplay
                    .chat
                    .push_message("  /sphere cx cy cz <radius> <block>".into());
                self.gameplay
                    .chat
                    .push_message("  /cylinder bx by bz <radius> <height> <block>".into());
                self.gameplay
                    .chat
                    .push_message("  /pyramid x1 y1 z1 x2 y2 z2 <block>".into());
                self.gameplay
                    .chat
                    .push_message("  /replace x1 y1 z1 x2 y2 z2 <target> <replacement>".into());
                self.gameplay
                    .chat
                    .push_message("  /line ax ay az bx by bz <thickness> <block>".into());
                self.gameplay.chat.push_message(
                    "  /schematic save <name>          - snapshot /copy'd selection".into(),
                );
                self.gameplay.chat.push_message(
                    "  /schematic list                  - list pasted schematics".into(),
                );
                self.gameplay.chat.push_message(
                    "  /schematic load <name>          - validate a .schem file".into(),
                );
                self.gameplay.chat.push_message(
                    "  /schematic paste <name> [ox oy oz] [rot0|90|180|270] [mx|my|mz|...]".into(),
                );
                self.gameplay
                    .chat
                    .push_message("  /gamemode <mode> - set gamemode (WIP)".into());
                self.gameplay
                    .chat
                    .push_message("  /difficulty <d>  - set difficulty (WIP)".into());
                self.gameplay
                    .chat
                    .push_message("  /kill            - kill the player".into());
                self.gameplay
                    .chat
                    .push_message("  /pos             - show current position".into());
                self.gameplay
                    .chat
                    .push_message("  /chunk           - show chunk info".into());
                self.gameplay
                    .chat
                    .push_message("  /fps             - show frame rate".into());
                self.gameplay
                    .chat
                    .push_message("  /clear           - clear chat".into());
                self.gameplay
                    .chat
                    .push_message("  /save [path]     - save world to disk".into());
                self.gameplay
                    .chat
                    .push_message("  /load [path]     - load world from disk".into());
                self.gameplay
                    .chat
                    .push_message("  /copy x1 y1 z1 x2 y2 z2 - copy region".into());
                self.gameplay
                    .chat
                    .push_message("  /paste           - paste clipboard".into());
                self.gameplay
                    .chat
                    .push_message("  /help            - show this help".into());
                self.gameplay
                    .chat
                    .push_message("  Tab = autocomplete, Up/Down = history".into());
            }
            CommandResult::Empty => {}
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
                self.gameplay.chat.push_message(format!(
                    "Waypoint '{}' added at ({}, {}, {})",
                    name, p.x, p.y, p.z
                ));
            }
            CommandResult::WaypointList => {
                if self.gameplay.map.waypoints.is_empty() {
                    self.gameplay.chat.push_message("No waypoints".into());
                } else {
                    for wp in &self.gameplay.map.waypoints {
                        self.gameplay
                            .chat
                            .push_message(format!("  {} ({}, {}, {})", wp.name, wp.x, wp.y, wp.z));
                    }
                }
            }
            CommandResult::WaypointRemove(name) => {
                if self.gameplay.map.remove_waypoint(&name) {
                    self.gameplay
                        .chat
                        .push_message(format!("Waypoint '{}' removed", name));
                } else {
                    self.gameplay
                        .chat
                        .push_message(format!("Waypoint '{}' not found", name));
                }
            }
            CommandResult::WaypointSave => {
                let path = std::path::Path::new("assets").join("waypoints.json");
                match self.gameplay.map.save_waypoints(&path) {
                    Ok(()) => self.gameplay.chat.push_message("Waypoints saved".into()),
                    Err(e) => self
                        .gameplay
                        .chat
                        .push_message(format!("Save failed: {}", e)),
                }
            }
            CommandResult::WaypointLoad => {
                let path = std::path::Path::new("assets").join("waypoints.json");
                match self.gameplay.map.load_waypoints(&path) {
                    Ok(()) => self.gameplay.chat.push_message(format!(
                        "Loaded {} waypoints",
                        self.gameplay.map.waypoints.len()
                    )),
                    Err(e) => self
                        .gameplay
                        .chat
                        .push_message(format!("Load failed: {}", e)),
                }
            }
            CommandResult::Unknown(msg) => {
                self.gameplay.chat.push_message(msg);
            }
            _ => unreachable!("Feature 4 added variants ΓÇö dispatcher must match them all"),
        }
    }

    /// Resolves `block_name` via `BlockRegistry::id_of`, then opens an
    /// `UndoRedoState` batch, runs the supplied volume method, and
    /// commits. The `op` closure is given the World (`&World`) and a
    /// mutable `UndoRedoState` reference so it can construct the
    /// `BlockChange ΓåÆ BlockEdit` pipeline via [`push_to_undo`].
    ///
    /// Returns the number of distinct block changes recorded (via the
    /// chat feedback the dispatcher writes itself).
    fn run_volume<F>(&mut self, label: &str, block_name: &str, op: F)
    where
        F: FnOnce(&voxel_world::World, BlockId, &mut voxel_game::UndoRedoState) -> usize,
    {
        let reg = self.world_state.world.registry_ref();
        let Some(id) = reg.id_of(block_name) else {
            self.gameplay
                .chat
                .push_message(format!("Unknown block: {block_name}"));
            return;
        };
        self.gameplay
            .undo_redo
            .begin_batch(format!("{label} {block_name}"));
        let count = {
            let undo = &mut self.gameplay.undo_redo;
            let world = &self.world_state.world;
            op(world, id, undo)
        };
        self.gameplay.undo_redo.commit_batch();
        self.gameplay
            .chat
            .push_message(format!("{label} {block_name}: {count} blocks"));
    }

    /// Replace variant of [`Self::run_volume`] ΓÇö needs lookup of two
    /// block names (target + replacement) before opening the batch, so a
    /// single-name helper doesn't fit cleanly.
    fn run_volume_replace<F>(&mut self, target: &str, replacement: &str, op: F)
    where
        F: FnOnce(&voxel_world::World, &mut voxel_game::UndoRedoState) -> usize,
    {
        let reg = self.world_state.world.registry_ref();
        let Some(target_id) = reg.id_of(target) else {
            self.gameplay
                .chat
                .push_message(format!("Unknown block: {target}"));
            return;
        };
        let Some(replace_id) = reg.id_of(replacement) else {
            self.gameplay
                .chat
                .push_message(format!("Unknown block: {replacement}"));
            return;
        };
        if target_id == replace_id {
            self.gameplay.chat.push_message(format!(
                "/replace: target '{target}' == replacement '{replacement}' ΓÇö no-op"
            ));
            return;
        }
        self.gameplay
            .undo_redo
            .begin_batch(format!("replace {target}->{replacement}"));
        let count = {
            let undo = &mut self.gameplay.undo_redo;
            let world = &self.world_state.world;
            // IDs validated above; the closure re-resolves via the
            // registry so any caller-side aliasing is consistent.
            let _ = (target_id, replace_id);
            op(world, undo)
        };
        self.gameplay.undo_redo.commit_batch();
        self.gameplay
            .chat
            .push_message(format!("replace: {count} blocks swapped"));
    }
}

/// Convert two opposing integer AABB corners into the half-open
/// `voxel_core::Aabb` the World volume methods expect (`max` exclusive).
fn bounds_from_corners(x1: i32, y1: i32, z1: i32, x2: i32, y2: i32, z2: i32) -> voxel_core::Aabb {
    voxel_core::Aabb {
        min: glam::Vec3::new(x1.min(x2) as f32, y1.min(y2) as f32, z1.min(z2) as f32),
        max: glam::Vec3::new(
            (x1.max(x2) as f32) + 1.0,
            (y1.max(y2) as f32) + 1.0,
            (z1.max(z2) as f32) + 1.0,
        ),
    }
}

/// Forward each `BlockChange` from a volume method into a batched undo
/// entry. Returns a `FnMut(BlockChange)` that auto-reborrows `&mut
/// UndoRedoState` on each call (so consecutive invocations during a
/// single volume run are sequenced correctly).
fn push_to_undo<'a>(undo: &'a mut voxel_game::UndoRedoState) -> impl FnMut(BlockChange) + 'a {
    move |change| {
        let _ = undo.push_edit_batched(voxel_game::BlockEdit {
            x: change.x,
            y: change.y,
            z: change.z,
            old_block: change.old.0,
            new_block: change.new.0,
        });
    }
}
