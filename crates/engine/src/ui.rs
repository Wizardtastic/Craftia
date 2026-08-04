//! UI overlay rendering and click-handling.
//!
//! All `draw_*` methods on `EngineApp` plus their paired click handlers
//! (`handle_*_click`) live here. The split keeps the frame loop in
//! `lib.rs` focused on per-frame orchestration rather than HUD layout.

use voxel_core::{Point, Rect};
use voxel_render::{GraphStyle, UiDrawData};

use crate::edit;
use crate::edit::terrain::TerrainOp;
use crate::GameState;

/// Inclusive min/max range for a slider. Shared by the settings sliders and
/// the edit-panel slider rows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SliderRange {
    pub(crate) min: f32,
    pub(crate) max: f32,
}

/// Row position: x/y origin plus available width. The row height is implied
/// by the row's fixed dimensions. Shared by the settings sliders and the
/// edit-panel row helpers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Row {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) w: f32,
}

/// Sanitize a string for use as a directory/file name.
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | ' ' => c,
            _ => '_',
        })
        .collect::<String>()
        .trim()
        .to_string()
        .replace(' ', "_")
}

/// Format a timestamp string (seconds since epoch) as a human-readable relative time.
fn format_last_played(timestamp_secs: &str) -> String {
    let Ok(secs) = timestamp_secs.parse::<u64>() else {
        return "never".to_string();
    };
    if secs == 0 {
        return "never".to_string();
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if now <= secs {
        return "just now".to_string();
    }
    let diff = now - secs;
    if diff < 60 {
        "just now".to_string()
    } else if diff < 3600 {
        let mins = diff / 60;
        format!("{} min ago", mins)
    } else if diff < 86400 {
        let hours = diff / 3600;
        format!("{} hour{} ago", hours, if hours == 1 { "" } else { "s" })
    } else if diff < 604800 {
        let days = diff / 86400;
        format!("{} day{} ago", days, if days == 1 { "" } else { "s" })
    } else {
        "a long time ago".to_string()
    }
}

impl crate::EngineApp {
    /// Build the UI overlay for this frame: crosshair + hotbar when playing,
    /// or the pause/exit menu when paused.
    pub(crate) fn build_ui(&mut self) -> UiDrawData {
        let mut ui = UiDrawData::default();
        // Lay out in logical (DPI-independent) pixels; the vertices are
        // scaled back to physical pixels below, and the UI shader maps
        // against the physical swapchain size.
        let (w, h) = self.render.logical_size();

        match self.gameplay.game_state {
            GameState::TitleScreen => {
                self.draw_title_screen(&mut ui, w, h);
            }
            GameState::WorldSelect => {
                self.draw_world_select(&mut ui, w, h);
                // Draw create world dialog overlay if active.
                if self.gameplay.create_world_state.is_some() {
                    self.draw_create_world_dialog(&mut ui, w, h);
                }
                // Draw delete confirmation overlay if pending.
                if self.gameplay.pending_delete.is_some() {
                    self.draw_delete_confirm_dialog(&mut ui, w, h);
                }
            }
            GameState::SettingsMenu => {
                // Draw the previous state underneath (dimmed).
                match self.gameplay.settings_previous {
                    GameState::TitleScreen => {
                        self.draw_title_screen(&mut ui, w, h);
                    }
                    GameState::PauseMenu => {
                        self.draw_pause_menu(&mut ui, w, h);
                    }
                    _ => {}
                }
                self.draw_settings_menu(&mut ui, w, h);
            }
            GameState::Playing => {
                // Check if player is dead.
                let is_dead = self
                    .simulation
                    .ecs_world()
                    .resource::<voxel_game::PlayerEntity>()
                    .and_then(|p| p.0)
                    .and_then(|e| self.simulation.ecs_world().get::<voxel_game::Health>(e))
                    .map(|h| h.dead)
                    .unwrap_or(false);

                if is_dead {
                    // Show death screen instead of normal HUD.
                    self.draw_death_screen(&mut ui, w, h);
                } else {
                    self.draw_crosshair(&mut ui, w, h);
                    self.draw_hotbar(&mut ui, w, h);
                    self.draw_held_item_name(&mut ui, w, h);
                    self.draw_health_bar(&mut ui, w, h);
                    self.draw_hunger_bar(&mut ui, w, h);
                    self.draw_xp_bar(&mut ui, w, h);
                    self.draw_bubble_bar(&mut ui, w, h);
                    self.draw_damage_vignette(&mut ui, w, h);
                    self.draw_player_arm(&mut ui, w, h);
                }
                if self.gameplay.debug_overlay {
                    self.draw_debug_overlay(&mut ui, w, h);
                }
                if self.profiler.enabled {
                    self.draw_profiler_overlay(&mut ui, w, h);
                }
                if self.gameplay.ecs_inspector {
                    self.draw_ecs_inspector(&mut ui, w, h);
                }
                if self.gameplay.block_picker_open {
                    self.draw_block_picker(&mut ui, w, h);
                }
                self.draw_chat(&mut ui, w, h);
                if self.gameplay.console.open {
                    self.draw_console(&mut ui, w, h);
                }
                if self.telemetry.enabled() {
                    self.draw_telemetry_dashboard(&mut ui, w, h);
                }
                self.draw_fullscreen_map(&mut ui, w, h);
                self.draw_minimap(&mut ui, w, h);
                if self.gameplay.edit.mode.is_active() {
                    self.draw_editor_ui(&mut ui, w, h);
                }
                self.gameplay.edit.consume_frame();
            }
            GameState::PauseMenu => {
                self.draw_pause_menu(&mut ui, w, h);
            }
        }

        // Scale the logical-pixel layout up to physical pixels so the UI
        // pipeline (which divides by the physical swapchain size) renders
        // it at the correct size on HiDPI displays. Skip the no-op when
        // the scale is 1.0 (typical Windows desktop).
        let s = self.render.ui_scale;
        if (s - 1.0).abs() > f32::EPSILON {
            for v in &mut ui.vertices {
                v.pos[0] *= s;
                v.pos[1] *= s;
            }
        }

        ui
    }

    /// Draw the full VoxEdit-style editor UI:
    /// menu bar, category bar, left panel, right panel, status bar.
    fn draw_editor_ui(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        let mouse = self.gameplay.mouse_pos;

        // Compute FPS for menu bar.
        let fps = self.profiler.avg_fps();

        // Menu bar (top).
        let menu_action;
        {
            let font = &self.render.font;
            menu_action =
                edit::menu_bar::draw_menu_bar(ui, &self.gameplay.edit, w, mouse, font, fps);
        }
        if menu_action == edit::menu_bar::MenuAction::ExitEditor {
            self.gameplay.edit.toggle();
            self.lock_cursor();
        }

        // Category bar (left edge).
        let cat_action;
        {
            let font = &self.render.font;
            cat_action = edit::toolbar::draw_category_bar(ui, &self.gameplay.edit, h, mouse, font);
        }
        if let edit::toolbar::CategoryAction::Select(cat) = cat_action {
            self.gameplay.edit.active_category = cat;
            self.gameplay.edit.active_tool_id = edit::default_tool(cat).to_string();
        }

        // Left panel (tool grid + active block + palette + history).
        let left_action;
        {
            let reg = self.world_state.world.registry();
            let font = &self.render.font;
            left_action = edit::left_panel::draw_left_panel(
                ui,
                &mut self.gameplay.edit,
                h,
                mouse,
                font,
                &reg,
            );
        }
        match left_action {
            edit::left_panel::LeftPanelAction::SelectTool(id) => {
                self.gameplay.edit.active_tool_id = id;
            }
            edit::left_panel::LeftPanelAction::SelectBlock(block_id) => {
                if let Some(brush) = self.gameplay.edit.brush_mut() {
                    brush.block = block_id;
                }
                self.gameplay.edit.add_recent(block_id);
            }
            edit::left_panel::LeftPanelAction::None => {}
        }

        // Right panel (tool options + target info + world properties).
        let right_action;
        {
            let cursor_pos = self.gameplay.edit.brush_center.map(|c| (c.x, c.y, c.z));
            let font = &self.render.font;
            right_action = edit::right_panel::draw_right_panel(
                ui,
                &mut self.gameplay.edit,
                (w, h),
                mouse,
                font,
                cursor_pos,
                &self.texture_pack_manager.loaded_packs,
            );
        }
        self.handle_right_panel_action(right_action);

        // Status bar (bottom).
        {
            let font = &self.render.font;
            edit::status_bar::draw_status_bar(ui, &self.gameplay.edit, w, h, font);
        }
    }

    /// Process right panel actions.
    fn handle_right_panel_action(&mut self, action: edit::right_panel::RightPanelAction) {
        use edit::right_panel::RightPanelAction;
        match action {
            RightPanelAction::None => {}
            RightPanelAction::SetShape(shape) => {
                if let Some(brush) = self.gameplay.edit.brush_mut() {
                    brush.shape = shape;
                }
                // Also update terrain/paint tool shape if active.
                if let Some(t) = self.gameplay.edit.terrain_mut() {
                    t.shape = shape;
                }
                if let Some(p) = self.gameplay.edit.paint_mut() {
                    p.shape = shape;
                }
            }
            RightPanelAction::SetPaintMode(mode) => {
                if let Some(brush) = self.gameplay.edit.brush_mut() {
                    brush.paint_mode = mode;
                }
            }
            RightPanelAction::ToggleHollow => {
                if let Some(brush) = self.gameplay.edit.brush_mut() {
                    brush.hollow = !brush.hollow;
                }
            }
            RightPanelAction::ToggleSurfaceOnly => {
                if let Some(brush) = self.gameplay.edit.brush_mut() {
                    brush.surface_only = !brush.surface_only;
                }
            }
            RightPanelAction::ToggleReplace => {
                if let Some(brush) = self.gameplay.edit.brush_mut() {
                    brush.replace = !brush.replace;
                }
            }
            RightPanelAction::ToggleShowGrid => {
                self.gameplay.edit.show_grid = !self.gameplay.edit.show_grid;
            }
            RightPanelAction::ToggleShowChunks => {
                self.gameplay.edit.show_chunks = !self.gameplay.edit.show_chunks;
            }
            RightPanelAction::RadiusDelta(delta) => {
                if let Some(brush) = self.gameplay.edit.brush_mut() {
                    brush.radius = (brush.radius + delta).clamp(1.0, 25.0);
                }
                if let Some(t) = self.gameplay.edit.terrain_mut() {
                    t.radius = (t.radius + delta).clamp(1.0, 25.0);
                }
                if let Some(p) = self.gameplay.edit.paint_mut() {
                    p.radius = (p.radius + delta).clamp(1.0, 25.0);
                }
            }
            RightPanelAction::StrengthDelta(delta) => {
                if let Some(brush) = self.gameplay.edit.brush_mut() {
                    brush.strength = (brush.strength + delta).clamp(0.0, 1.0);
                }
            }
            // Phase 2: Replace target
            RightPanelAction::PickReplaceTarget => {
                // Target is picked via shift+right-click in frame.rs.
                // This action is a UI hint; actual pick happens on next click.
            }
            RightPanelAction::ClearReplaceTarget => {
                if let Some(brush) = self.gameplay.edit.brush_mut() {
                    brush.target = None;
                }
            }
            // Phase 3: Multi-block palette
            RightPanelAction::ToggleMultiBlock => {
                if let Some(brush) = self.gameplay.edit.brush_mut() {
                    brush.palette.enabled = !brush.palette.enabled;
                    if brush.palette.enabled && brush.palette.entries.is_empty() {
                        // Seed with current brush block.
                        brush.palette.add(brush.block, 1.0);
                    }
                }
            }
            RightPanelAction::AddPaletteBlock => {
                if let Some(brush) = self.gameplay.edit.brush_mut() {
                    // Add current brush block to palette.
                    brush.palette.add(brush.block, 1.0);
                }
            }
            RightPanelAction::ClearPalette => {
                if let Some(brush) = self.gameplay.edit.brush_mut() {
                    brush.palette.clear();
                    brush.palette.enabled = false;
                }
            }
            // Phase 4: Terrain
            RightPanelAction::SetTerrainOp(op) => {
                if let Some(t) = self.gameplay.edit.terrain_mut() {
                    t.op = op;
                }
            }
            RightPanelAction::TerrainAmountDelta(delta) => {
                if let Some(t) = self.gameplay.edit.terrain_mut() {
                    match &mut t.op {
                        TerrainOp::Raise { amount } | TerrainOp::Lower { amount } => {
                            *amount = (*amount + delta).clamp(1.0, 20.0);
                        }
                        _ => {}
                    }
                }
            }
            // Phase 5: Gradient
            RightPanelAction::SetGradientShape(gs) => {
                if let Some(p) = self.gameplay.edit.paint_mut() {
                    p.gradient = gs;
                }
            }
            RightPanelAction::SetInterpolation(im) => {
                if let Some(p) = self.gameplay.edit.paint_mut() {
                    p.interpolation = im;
                }
            }
            RightPanelAction::SwapGradientBlocks => {
                if let Some(p) = self.gameplay.edit.paint_mut() {
                    std::mem::swap(&mut p.block_a, &mut p.block_b);
                }
            }
            // Phase 6: Filters
            RightPanelAction::AddFilter(op) => {
                if let Some(f) = self.gameplay.edit.filter_mut() {
                    f.add(op);
                }
            }
            RightPanelAction::RemoveFilter(idx) => {
                if let Some(f) = self.gameplay.edit.filter_mut() {
                    f.remove(idx);
                }
            }
            RightPanelAction::ClearFilters => {
                if let Some(f) = self.gameplay.edit.filter_mut() {
                    f.clear();
                }
            }
            RightPanelAction::ApplyFilters => {
                // Apply filters to selection or brush area.
                if let Some(f) = self.gameplay.edit.filter_ref().cloned() {
                    let (min, max) = if f.apply_to_selection {
                        if let Some(sel) = self.gameplay.edit.select_ref() {
                            sel.bounds().unwrap_or((
                                self.gameplay.edit.brush_center.unwrap_or(glam::IVec3::ZERO),
                                self.gameplay.edit.brush_center.unwrap_or(glam::IVec3::ZERO),
                            ))
                        } else {
                            let c = self.gameplay.edit.brush_center.unwrap_or(glam::IVec3::ZERO);
                            (c, c)
                        }
                    } else {
                        let c = self.gameplay.edit.brush_center.unwrap_or(glam::IVec3::ZERO);
                        let r = 5;
                        (c - glam::IVec3::splat(r), c + glam::IVec3::splat(r))
                    };
                    let affected = edit::filter::apply_filters(
                        &f,
                        &self.world_state.world,
                        min,
                        max,
                        &mut self.gameplay.undo_redo,
                    );
                    if let Some(streamer) = &self.world_state.streamer {
                        for cp in affected {
                            streamer.request_remesh(cp);
                        }
                    }
                }
            }
            // Phase 7: Transform
            RightPanelAction::TransformMove(delta) => {
                if let Some((min, max)) = self.gameplay.edit.select_ref().and_then(|s| s.bounds()) {
                    let affected = apply_transform_move(
                        &self.world_state.world,
                        min,
                        max,
                        delta,
                        &mut self.gameplay.undo_redo,
                    );
                    if let Some(streamer) = &self.world_state.streamer {
                        for cp in affected {
                            streamer.request_remesh(cp);
                        }
                    }
                    // Update selection bounds.
                    if let Some(sel) = self.gameplay.edit.select_mut() {
                        sel.active_selection = Some((min + delta, max + delta));
                    }
                }
            }
            RightPanelAction::TransformRotate(degrees) => {
                if let Some((min, max)) = self.gameplay.edit.select_ref().and_then(|s| s.bounds()) {
                    let affected = apply_transform_rotate(
                        &self.world_state.world,
                        min,
                        max,
                        degrees,
                        &mut self.gameplay.undo_redo,
                    );
                    if let Some(streamer) = &self.world_state.streamer {
                        for cp in affected {
                            streamer.request_remesh(cp);
                        }
                    }
                }
            }
            RightPanelAction::TransformScale(factor) => {
                if let Some((min, max)) = self.gameplay.edit.select_ref().and_then(|s| s.bounds()) {
                    let affected = apply_transform_scale(
                        &self.world_state.world,
                        min,
                        max,
                        factor,
                        &mut self.gameplay.undo_redo,
                    );
                    if let Some(streamer) = &self.world_state.streamer {
                        for cp in affected {
                            streamer.request_remesh(cp);
                        }
                    }
                }
            }
            // Phase 1: Selection ops
            RightPanelAction::SelectionCopy => {
                if let Some((min, max)) = self.gameplay.edit.select_ref().and_then(|s| s.bounds()) {
                    let clipboard = edit::select::copy_selection(&self.world_state.world, min, max);
                    self.gameplay.clipboard = Some(clipboard);
                    self.gameplay
                        .chat
                        .push_message("Copied selection to clipboard".into());
                }
            }
            RightPanelAction::SelectionCut => {
                if let Some((min, max)) = self.gameplay.edit.select_ref().and_then(|s| s.bounds()) {
                    let clipboard = edit::select::copy_selection(&self.world_state.world, min, max);
                    self.gameplay.clipboard = Some(clipboard);
                    let affected = edit::select::delete_selection(
                        &self.world_state.world,
                        min,
                        max,
                        &mut self.gameplay.undo_redo,
                    );
                    if let Some(streamer) = &self.world_state.streamer {
                        for cp in affected {
                            streamer.request_remesh(cp);
                        }
                    }
                    self.gameplay
                        .chat
                        .push_message("Cut selection to clipboard".into());
                }
            }
            RightPanelAction::SelectionDelete => {
                if let Some((min, max)) = self.gameplay.edit.select_ref().and_then(|s| s.bounds()) {
                    let affected = edit::select::delete_selection(
                        &self.world_state.world,
                        min,
                        max,
                        &mut self.gameplay.undo_redo,
                    );
                    if let Some(streamer) = &self.world_state.streamer {
                        for cp in affected {
                            streamer.request_remesh(cp);
                        }
                    }
                    self.gameplay.chat.push_message("Deleted selection".into());
                }
            }
            RightPanelAction::SelectionClear => {
                if let Some(sel) = self.gameplay.edit.select_mut() {
                    sel.clear();
                }
            }
            // Phase 8: Undo/Redo
            RightPanelAction::UndoAction => {
                if let Some(action) = self.gameplay.undo_redo.pop_undo() {
                    let mut chunks = std::collections::HashSet::new();
                    for edit in action.edits.iter().rev() {
                        let id = voxel_core::BlockId(edit.old_block);
                        self.world_state.world.set_block(edit.x, edit.y, edit.z, id);
                        let cp = voxel_core::math::block_to_chunk(glam::IVec3::new(
                            edit.x, edit.y, edit.z,
                        ));
                        chunks.insert(cp);
                    }
                    if let Some(s) = &self.world_state.streamer {
                        for cp in chunks {
                            s.request_remesh(cp);
                        }
                    }
                    self.gameplay.chat.push_message("Undo".into());
                }
            }
            RightPanelAction::RedoAction => {
                if let Some(action) = self.gameplay.undo_redo.pop_redo() {
                    let mut chunks = std::collections::HashSet::new();
                    for edit in &action.edits {
                        let id = voxel_core::BlockId(edit.new_block);
                        self.world_state.world.set_block(edit.x, edit.y, edit.z, id);
                        let cp = voxel_core::math::block_to_chunk(glam::IVec3::new(
                            edit.x, edit.y, edit.z,
                        ));
                        chunks.insert(cp);
                    }
                    if let Some(s) = &self.world_state.streamer {
                        for cp in chunks {
                            s.request_remesh(cp);
                        }
                    }
                    self.gameplay.chat.push_message("Redo".into());
                }
            }
        }
    }

    /// Draw a centred crosshair (two thin white bars forming a +).
    /// Mode changes the appearance based on what the player is looking at.
    fn draw_crosshair(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let cx = w * 0.5;
        let cy = h * 0.5;
        let len = 10.0;
        let thick = 2.0;

        match self.gameplay.crosshair_mode {
            crate::CrosshairMode::Default => {
                // Default crosshair: white + shape.
                let color = [255, 255, 255, 200];
                ui.quad(cx - len, cy - thick * 0.5, len * 2.0, thick, color);
                ui.quad(cx - thick * 0.5, cy - len, thick, len * 2.0, color);
            }
            crate::CrosshairMode::BlockTarget => {
                // Block targeted: brighter + with corner dots.
                let color = [255, 255, 255, 255];
                ui.quad(cx - len, cy - thick * 0.5, len * 2.0, thick, color);
                ui.quad(cx - thick * 0.5, cy - len, thick, len * 2.0, color);

                // Corner dots for precision.
                let dot_size = 2.0;
                let dot_offset = 6.0;
                let dot_color = [255, 255, 255, 180];
                ui.quad(
                    cx - dot_offset - dot_size,
                    cy - dot_offset - dot_size,
                    dot_size,
                    dot_size,
                    dot_color,
                );
                ui.quad(
                    cx + dot_offset,
                    cy - dot_offset - dot_size,
                    dot_size,
                    dot_size,
                    dot_color,
                );
                ui.quad(
                    cx - dot_offset - dot_size,
                    cy + dot_offset,
                    dot_size,
                    dot_size,
                    dot_color,
                );
                ui.quad(
                    cx + dot_offset,
                    cy + dot_offset,
                    dot_size,
                    dot_size,
                    dot_color,
                );
            }
            crate::CrosshairMode::Interact => {
                // Interactable: + with center square.
                let color = [255, 255, 255, 255];
                ui.quad(cx - len, cy - thick * 0.5, len * 2.0, thick, color);
                ui.quad(cx - thick * 0.5, cy - len, thick, len * 2.0, color);

                // Center square.
                let square_size = 4.0;
                let square_color = [255, 255, 255, 200];
                ui.quad(
                    cx - square_size * 0.5,
                    cy - square_size * 0.5,
                    square_size,
                    square_size,
                    square_color,
                );
            }
        }
    }

    /// Draw the 9-slot hotbar at the bottom-centre of the screen.
    fn draw_hotbar(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let slot = 48.0;
        let gap = 4.0;
        let total = slot * 9.0 + gap * 8.0;
        let x0 = (w - total) * 0.5;
        let y0 = h - slot - 12.0;
        let reg = self.world_state.world.registry();

        for i in 0..9 {
            let x = x0 + i as f32 * (slot + gap);
            ui.quad(x, y0, slot, slot, [40, 40, 40, 180]);
            ui.rect_border(x, y0, slot, slot, 2.0, [80, 80, 80, 220]);

            let block_id = self
                .gameplay
                .hotbar
                .slot(i)
                .unwrap_or(voxel_core::BlockId::AIR);
            if !block_id.is_air() {
                let def = reg.get(block_id);
                let tile = def.textures.tile(voxel_world::registry::Face::PosX);
                let icon_size = slot - 8.0;
                ui.block_icon(
                    x + 4.0,
                    y0 + 4.0,
                    icon_size,
                    icon_size,
                    tile,
                    [255, 255, 255, 255],
                );
            }

            if i == self.gameplay.hotbar.selected {
                ui.rect_border(
                    x - 2.0,
                    y0 - 2.0,
                    slot + 4.0,
                    slot + 4.0,
                    3.0,
                    [255, 255, 255, 255],
                );
            }
        }
    }

    /// Draw the name of the currently held item above the hotbar.
    fn draw_held_item_name(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let slot = 48.0;
        let gap = 4.0;
        let total = slot * 9.0 + gap * 8.0;
        let _x0 = (w - total) * 0.5;
        let y0 = h - slot - 12.0;

        // Get the selected block.
        let block_id = self
            .gameplay
            .hotbar
            .selected_block()
            .unwrap_or(voxel_core::BlockId::AIR);
        if block_id.is_air() {
            return;
        }

        // Get the block name from the registry.
        let reg = self.world_state.world.registry();
        let def = reg.get(block_id);
        let name = def.name.as_ref();

        // Draw the name centered above the hotbar.
        let text_scale = 1.0;
        let text_width = self.render.font.text_width(name, text_scale);
        let text_x = (w - text_width) * 0.5;
        let text_y = y0 - 16.0; // 16px above hotbar

        // Draw with a slight shadow for readability.
        ui.text(
            name,
            text_x + 1.0,
            text_y + 1.0,
            text_scale,
            [0, 0, 0, 180],
            &self.render.font,
        );
        ui.text(
            name,
            text_x,
            text_y,
            text_scale,
            [255, 255, 255, 255],
            &self.render.font,
        );
    }

    /// Draw the health bar (hearts) for survival mode.
    /// Only shown in Survival/Adventure modes.
    fn draw_health_bar(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        // Check if we're in a survival-like mode.
        let game_mode = self
            .simulation
            .ecs_world()
            .resource::<voxel_game::PlayerEntity>()
            .and_then(|p| p.0)
            .and_then(|e| self.simulation.ecs_world().get::<voxel_game::GameMode>(e))
            .copied()
            .unwrap_or(voxel_game::GameMode::Survival);

        // Only show health bar in survival and adventure modes.
        if !game_mode.has_hunger() && game_mode != voxel_game::GameMode::Adventure {
            return;
        }

        // Get the player's health.
        let health = self
            .simulation
            .ecs_world()
            .resource::<voxel_game::PlayerEntity>()
            .and_then(|p| p.0)
            .and_then(|e| self.simulation.ecs_world().get::<voxel_game::Health>(e))
            .copied()
            .unwrap_or_default();

        // Position: above the hotbar, left-aligned.
        let slot = 48.0;
        let gap = 4.0;
        let total = slot * 9.0 + gap * 8.0;
        let x0 = (w - total) * 0.5;
        let y0 = h - slot - 12.0;
        let heart_size = 9.0;
        let heart_gap = 1.0;
        let bar_y = y0 - 20.0; // 20px above hotbar

        // Draw 10 hearts.
        for i in 0..10 {
            let heart_x = x0 + i as f32 * (heart_size + heart_gap);
            let health_value = health.current - i as f32 * 2.0;

            // Determine heart state: 0 = empty, 1 = half, 2 = full
            let stage = if health_value >= 2.0 {
                2
            } else if health_value >= 1.0 {
                1
            } else {
                0
            };

            // Draw heart background (empty).
            ui.quad(heart_x, bar_y, heart_size, heart_size, [40, 10, 10, 200]);

            // Draw heart fill based on stage.
            match stage {
                2 => {
                    // Full heart - red
                    ui.quad(heart_x, bar_y, heart_size, heart_size, [200, 30, 30, 255]);
                }
                1 => {
                    // Half heart - red on left half, empty on right
                    ui.quad(
                        heart_x,
                        bar_y,
                        heart_size * 0.5,
                        heart_size,
                        [200, 30, 30, 255],
                    );
                }
                _ => {
                    // Empty heart - dark red outline
                    ui.rect_border(
                        heart_x,
                        bar_y,
                        heart_size,
                        heart_size,
                        1.0,
                        [100, 20, 20, 200],
                    );
                }
            }
        }

        // Draw health text (optional, for debugging).
        let health_text = format!("{}/{}", health.current as i32, health.max as i32);
        let text_x = x0 + 10.0 * (heart_size + heart_gap) + 4.0;
        ui.text(
            &health_text,
            text_x,
            bar_y,
            0.7,
            [200, 200, 200, 255],
            &self.render.font,
        );
    }

    /// Draw the hunger bar (drumsticks) for survival mode.
    fn draw_hunger_bar(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let game_mode = self
            .simulation
            .ecs_world()
            .resource::<voxel_game::PlayerEntity>()
            .and_then(|p| p.0)
            .and_then(|e| self.simulation.ecs_world().get::<voxel_game::GameMode>(e))
            .copied()
            .unwrap_or(voxel_game::GameMode::Survival);

        // Only show hunger bar in survival mode.
        if !game_mode.has_hunger() {
            return;
        }

        let hunger = self
            .simulation
            .ecs_world()
            .resource::<voxel_game::PlayerEntity>()
            .and_then(|p| p.0)
            .and_then(|e| self.simulation.ecs_world().get::<voxel_game::Hunger>(e))
            .copied()
            .unwrap_or_default();

        // Position: right side, same row as health bar.
        let slot = 48.0;
        let gap = 4.0;
        let total = slot * 9.0 + gap * 8.0;
        let x0 = (w - total) * 0.5;
        let y0 = h - slot - 12.0;
        let drumstick_size = 9.0;
        let drumstick_gap = 1.0;
        let bar_y = y0 - 20.0;

        // Draw 10 drumsticks from the right.
        for i in 0..10 {
            let drumstick_x = x0 + total - (i as f32 + 1.0) * (drumstick_size + drumstick_gap);
            let food_value = hunger.food - i as f32 * 2.0;

            let stage = if food_value >= 2.0 {
                2
            } else if food_value >= 1.0 {
                1
            } else {
                0
            };

            // Draw drumstick background (empty).
            ui.quad(
                drumstick_x,
                bar_y,
                drumstick_size,
                drumstick_size,
                [40, 40, 10, 200],
            );

            // Draw drumstick fill.
            match stage {
                2 => {
                    // Full - brown
                    ui.quad(
                        drumstick_x,
                        bar_y,
                        drumstick_size,
                        drumstick_size,
                        [180, 120, 40, 255],
                    );
                }
                1 => {
                    // Half - brown on right half
                    ui.quad(
                        drumstick_x + drumstick_size * 0.5,
                        bar_y,
                        drumstick_size * 0.5,
                        drumstick_size,
                        [180, 120, 40, 255],
                    );
                }
                _ => {
                    // Empty - outline
                    ui.rect_border(
                        drumstick_x,
                        bar_y,
                        drumstick_size,
                        drumstick_size,
                        1.0,
                        [100, 80, 20, 200],
                    );
                }
            }
        }
    }

    /// Draw the XP bar above the hotbar.
    fn draw_xp_bar(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let game_mode = self
            .simulation
            .ecs_world()
            .resource::<voxel_game::PlayerEntity>()
            .and_then(|p| p.0)
            .and_then(|e| self.simulation.ecs_world().get::<voxel_game::GameMode>(e))
            .copied()
            .unwrap_or(voxel_game::GameMode::Survival);

        // Only show XP bar in survival/adventure modes.
        if game_mode == voxel_game::GameMode::Creative
            || game_mode == voxel_game::GameMode::Spectator
        {
            return;
        }

        let experience = self
            .simulation
            .ecs_world()
            .resource::<voxel_game::PlayerEntity>()
            .and_then(|p| p.0)
            .and_then(|e| self.simulation.ecs_world().get::<voxel_game::Experience>(e))
            .copied()
            .unwrap_or_default();

        // Position: between hotbar and health/hunger bars.
        let slot = 48.0;
        let gap = 4.0;
        let total = slot * 9.0 + gap * 8.0;
        let x0 = (w - total) * 0.5;
        let y0 = h - slot - 12.0;
        let bar_width = total;
        let bar_height = 5.0;
        let bar_y = y0 - 32.0; // Above health/hunger bars

        // Draw XP bar background.
        ui.quad(x0, bar_y, bar_width, bar_height, [40, 40, 40, 200]);

        // Draw XP bar fill (green).
        let fill_width = bar_width * experience.progress;
        ui.quad(x0, bar_y, fill_width, bar_height, [50, 200, 50, 255]);

        // Draw level text.
        let level_text = format!("{}", experience.level);
        let text_width = self.render.font.text_width(&level_text, 0.8);
        ui.text(
            &level_text,
            x0 + (bar_width - text_width) * 0.5,
            bar_y - 12.0,
            0.8,
            [200, 200, 200, 255],
            &self.render.font,
        );
    }

    /// Draw bubble bar when underwater.
    fn draw_bubble_bar(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let air = self
            .simulation
            .ecs_world()
            .resource::<voxel_game::PlayerEntity>()
            .and_then(|p| p.0)
            .and_then(|e| self.simulation.ecs_world().get::<voxel_game::AirSupply>(e))
            .copied()
            .unwrap_or_default();

        // Only show bubbles when not at full air.
        if air.current >= air.max {
            return;
        }

        // Position: same row as health bar, right of hunger.
        let slot = 48.0;
        let gap = 4.0;
        let total = slot * 9.0 + gap * 8.0;
        let x0 = (w - total) * 0.5;
        let y0 = h - slot - 12.0;
        let bubble_size = 9.0;
        let bubble_gap = 1.0;
        let bar_y = y0 - 32.0; // Above XP bar

        // Draw 10 bubbles.
        for i in 0..10 {
            let bubble_x = x0 + total - (i as f32 + 1.0) * (bubble_size + bubble_gap);
            let air_value = air.current - i as f32 * (air.max / 10.0);

            let stage = if air_value >= air.max / 10.0 {
                2
            } else if air_value > 0.0 {
                1
            } else {
                0
            };

            // Draw bubble background (empty).
            ui.quad(bubble_x, bar_y, bubble_size, bubble_size, [10, 20, 40, 200]);

            // Draw bubble fill.
            match stage {
                2 => {
                    // Full - blue
                    ui.quad(
                        bubble_x,
                        bar_y,
                        bubble_size,
                        bubble_size,
                        [40, 100, 200, 255],
                    );
                }
                1 => {
                    // Half - blue on left half
                    ui.quad(
                        bubble_x,
                        bar_y,
                        bubble_size * 0.5,
                        bubble_size,
                        [40, 100, 200, 255],
                    );
                }
                _ => {
                    // Empty - outline
                    ui.rect_border(
                        bubble_x,
                        bar_y,
                        bubble_size,
                        bubble_size,
                        1.0,
                        [20, 50, 100, 200],
                    );
                }
            }
        }
    }

    /// Draw damage vignette (red flash on hit).
    fn draw_damage_vignette(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let health = self
            .simulation
            .ecs_world()
            .resource::<voxel_game::PlayerEntity>()
            .and_then(|p| p.0)
            .and_then(|e| self.simulation.ecs_world().get::<voxel_game::Health>(e))
            .copied()
            .unwrap_or_default();

        // Show red vignette when recently damaged.
        if health.invulnerability_ticks > 0 {
            // Fade out over the invulnerability period.
            let alpha = (health.invulnerability_ticks as f32 / 20.0 * 80.0) as u8;
            // Draw red overlay at screen edges.
            ui.quad(0.0, 0.0, w, h, [200, 0, 0, alpha]);
        }
    }

    /// Draw the player's arm below the held item.
    fn draw_player_arm(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        // Don't draw arm in creative/spectator.
        let game_mode = self
            .simulation
            .ecs_world()
            .resource::<voxel_game::PlayerEntity>()
            .and_then(|p| p.0)
            .and_then(|e| self.simulation.ecs_world().get::<voxel_game::GameMode>(e))
            .copied()
            .unwrap_or(voxel_game::GameMode::Survival);
        if game_mode == voxel_game::GameMode::Creative
            || game_mode == voxel_game::GameMode::Spectator
        {
            return;
        }

        // Get player state for mining swing.
        let mining_swing = self
            .simulation
            .ecs_world()
            .resource::<voxel_game::PlayerEntity>()
            .and_then(|p| p.0)
            .and_then(|e| {
                self.simulation
                    .ecs_world()
                    .get::<voxel_game::PlayerState>(e)
            })
            .map(|s| s.mining_swing)
            .unwrap_or(0.0);

        // Arm position: bottom-right of screen.
        let arm_w = 24.0;
        let arm_h = 64.0;
        let arm_x = w * 0.5 + 40.0; // Right of center
        let arm_y = h - 120.0; // Above hotbar

        // Apply mining swing rotation (visual offset).
        let swing_offset = if mining_swing > 0.0 {
            let progress = 1.0 - mining_swing;
            let eased = 1.0 - (1.0 - progress).powi(2);
            eased * 20.0
        } else {
            0.0
        };

        // Draw arm (skin-colored rectangle).
        let skin_color = [180, 140, 100, 255];
        ui.quad(arm_x + swing_offset, arm_y, arm_w, arm_h, skin_color);

        // Draw hand (slightly darker).
        let hand_color = [160, 120, 80, 255];
        ui.quad(
            arm_x + swing_offset,
            arm_y + arm_h - 16.0,
            arm_w,
            16.0,
            hand_color,
        );
    }

    /// Draw the death screen overlay.
    fn draw_death_screen(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        // Dark red background.
        ui.quad(0.0, 0.0, w, h, [100, 0, 0, 180]);

        // "You died!" title.
        let title = "You died!";
        let title_size = 3.0;
        let title_width = self.render.font.text_width(title, title_size);
        ui.text(
            title,
            (w - title_width) * 0.5,
            h * 0.3,
            title_size,
            [255, 255, 255, 255],
            &self.render.font,
        );

        // Death message (if available).
        let message = "You died!";
        let msg_size = 1.5;
        let msg_width = self.render.font.text_width(message, msg_size);
        ui.text(
            message,
            (w - msg_width) * 0.5,
            h * 0.4,
            msg_size,
            [200, 200, 200, 255],
            &self.render.font,
        );

        // Respawn button.
        let btn_w = 200.0;
        let btn_h = 40.0;
        let btn_x = (w - btn_w) * 0.5;
        let btn_y = h * 0.5;
        self.draw_button(
            ui,
            Rect {
                x: btn_x,
                y: btn_y,
                w: btn_w,
                h: btn_h,
            },
            "RESPAWN",
            [40, 80, 40, 220],
            [60, 120, 60, 255],
            [120, 200, 120, 255],
        );

        // Title Screen button.
        let btn2_y = btn_y + 60.0;
        self.draw_button(
            ui,
            Rect {
                x: btn_x,
                y: btn2_y,
                w: btn_w,
                h: btn_h,
            },
            "TITLE SCREEN",
            [80, 40, 40, 220],
            [120, 60, 60, 255],
            [200, 100, 100, 255],
        );
    }

    /// Draw the pause/exit menu (4 buttons: back, options, save & quit, quit).
    fn draw_pause_menu(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        ui.quad(0.0, 0.0, w, h, [0, 0, 0, 160]);

        let panel_w = 320.0;
        let panel_h = 340.0;
        let px = (w - panel_w) * 0.5;
        let py = (h - panel_h) * 0.5;

        ui.quad(px, py, panel_w, panel_h, [30, 30, 40, 240]);
        ui.rect_border(px, py, panel_w, panel_h, 2.0, [100, 100, 120, 255]);

        let title = "PAUSED";
        let tw = self.render.font.text_width(title, 2.0);
        ui.text(
            title,
            px + (panel_w - tw) * 0.5,
            py + 20.0,
            2.0,
            [255, 255, 255, 255],
            &self.render.font,
        );

        let btn_w = 240.0;
        let btn_h = 40.0;
        let btn_x = px + (panel_w - btn_w) * 0.5;
        let spacing = 52.0;
        let btn_y0 = py + 60.0;

        // Back to Game
        self.draw_button(
            ui,
            Rect {
                x: btn_x,
                y: btn_y0,
                w: btn_w,
                h: btn_h,
            },
            "BACK TO GAME",
            [40, 80, 40, 220],
            [60, 120, 60, 255],
            [120, 200, 120, 255],
        );
        // Options
        self.draw_button(
            ui,
            Rect {
                x: btn_x,
                y: btn_y0 + spacing,
                w: btn_w,
                h: btn_h,
            },
            "OPTIONS",
            [50, 50, 80, 220],
            [70, 70, 110, 255],
            [120, 120, 200, 255],
        );
        // Save & Quit to Title
        self.draw_button(
            ui,
            Rect {
                x: btn_x,
                y: btn_y0 + spacing * 2.0,
                w: btn_w,
                h: btn_h,
            },
            "SAVE & QUIT",
            [80, 60, 30, 220],
            [120, 90, 40, 255],
            [200, 160, 80, 255],
        );
        // Quit Game
        self.draw_button(
            ui,
            Rect {
                x: btn_x,
                y: btn_y0 + spacing * 3.0,
                w: btn_w,
                h: btn_h,
            },
            "QUIT GAME",
            [90, 30, 30, 220],
            [140, 50, 50, 255],
            [220, 100, 100, 255],
        );

        self.gameplay.pause_buttons = Some([
            Rect {
                x: btn_x,
                y: btn_y0,
                w: btn_w,
                h: btn_h,
            },
            Rect {
                x: btn_x,
                y: btn_y0 + spacing,
                w: btn_w,
                h: btn_h,
            },
            Rect {
                x: btn_x,
                y: btn_y0 + spacing * 2.0,
                w: btn_w,
                h: btn_h,
            },
            Rect {
                x: btn_x,
                y: btn_y0 + spacing * 3.0,
                w: btn_w,
                h: btn_h,
            },
        ]);
    }

    fn draw_button(
        &self,
        ui: &mut UiDrawData,
        rect: Rect,
        label: &str,
        fill_normal: [u8; 4],
        fill_hover: [u8; 4],
        border: [u8; 4],
    ) {
        let Rect { x, y, w, h } = rect;
        let hovered = rect.contains(self.gameplay.mouse_pos);
        let fill = if hovered { fill_hover } else { fill_normal };
        ui.quad(x, y, w, h, fill);
        ui.rect_border(x, y, w, h, 2.0, border);
        let scale = 1.5;
        let lw = self.render.font.text_width(label, scale);
        ui.text(
            label,
            x + (w - lw) * 0.5,
            y + 14.0,
            scale,
            [255, 255, 255, 255],
            &self.render.font,
        );
    }

    /// Handle a click in the creative inventory overlay.
    pub(crate) fn handle_block_picker_click(&mut self) {
        let (w, h) = self.render.logical_size();

        // ── Recompute layout (must match draw_block_picker) ──
        let slot_size = 40.0f32;
        let slot_gap = 3.0f32;
        let cols = 9usize;
        let panel_pad = 12.0f32;
        let tabs = [
            "Blocks",
            "Nature",
            "Building",
            "Ores",
            "Decoration",
            "Liquids",
            "All",
            "Search",
        ];
        let tab_count = tabs.len();
        let active_tab = self.gameplay.creative_tab;
        let is_search = active_tab == tab_count - 1;

        let filtered_items: Vec<&crate::CreativeItem> = if is_search {
            let q = self.gameplay.creative_search.to_ascii_lowercase();
            if q.is_empty() {
                self.gameplay.creative_items.iter().collect()
            } else {
                self.gameplay
                    .creative_items
                    .iter()
                    .filter(|it| it.name.to_ascii_lowercase().contains(&q))
                    .collect()
            }
        } else if active_tab == tab_count - 2 {
            self.gameplay.creative_items.iter().collect()
        } else {
            let cat = tabs[active_tab];
            self.gameplay
                .creative_items
                .iter()
                .filter(|it| it.category == cat)
                .collect()
        };

        let item_count = filtered_items.len();
        let rows = item_count.div_ceil(cols);
        let visible_rows = rows.min(5);

        let grid_w = cols as f32 * (slot_size + slot_gap) - slot_gap;
        let panel_w = grid_w + panel_pad * 2.0;
        let tab_height = 32.0f32;
        let titlebar_h = 28.0f32;
        let search_h = 28.0f32;
        let grid_h = visible_rows as f32 * (slot_size + slot_gap) - slot_gap;
        let hotbar_h = slot_size;
        let divider_h = 2.0f32;

        let panel_h = panel_pad
            + titlebar_h
            + 6.0
            + if is_search { search_h + 6.0 } else { 0.0 }
            + grid_h
            + panel_pad
            + divider_h
            + 6.0
            + hotbar_h
            + panel_pad;

        let panel_x = (w - panel_w) * 0.5;
        let panel_y = (h - panel_h) * 0.5 - 10.0;

        let mx = self.gameplay.mouse_pos.x;
        let my = self.gameplay.mouse_pos.y;

        // ── Check tab clicks ──
        let tab_w = 40.0f32;
        let tab_gap = 2.0f32;
        let tab_row_w = tab_count as f32 * tab_w + (tab_count - 1) as f32 * tab_gap;
        let tab_x0 = panel_x + (panel_w - tab_row_w) * 0.5;
        let tab_y = panel_y - tab_height;

        for i in 0..tab_count {
            let tx = tab_x0 + i as f32 * (tab_w + tab_gap);
            let is_active = i == active_tab;
            let t_h = if is_active {
                tab_height + 6.0
            } else {
                tab_height
            };
            let ty = if is_active { tab_y - 6.0 } else { tab_y };

            if mx >= tx && mx < tx + tab_w && my >= ty && my < ty + t_h {
                self.gameplay.creative_tab = i;
                self.gameplay.creative_search.clear();
                self.gameplay.creative_scroll = 0;
                return;
            }
        }

        // ── Check close button ──
        let close_x = panel_x + panel_w - panel_pad - 20.0;
        let close_y = panel_y + panel_pad;
        if mx >= close_x && mx < close_x + 20.0 && my >= close_y && my < close_y + 20.0 {
            self.gameplay.block_picker_open = false;
            self.lock_cursor();
            return;
        }

        // ── Check item grid clicks ──
        let mut cy = panel_y + panel_pad + titlebar_h + 6.0;
        if is_search {
            cy += search_h + 6.0;
        }
        let grid_x0 = panel_x + panel_pad;
        let grid_y0 = cy;

        if mx >= grid_x0 && mx < grid_x0 + grid_w && my >= grid_y0 && my < grid_y0 + grid_h {
            let col = ((mx - grid_x0) / (slot_size + slot_gap)) as usize;
            let row = ((my - grid_y0) / (slot_size + slot_gap)) as usize;
            let idx = row * cols + col;
            if col < cols && row < visible_rows && idx < filtered_items.len() {
                let item = filtered_items[idx];
                self.gameplay
                    .hotbar
                    .set_slot(self.gameplay.hotbar.selected, item.id);
                self.gameplay
                    .chat
                    .push_message(format!("Selected: {}", item.name));
                self.gameplay.block_picker_open = false;
                self.lock_cursor();
            }
            return;
        }

        // ── Check hotbar clicks ──
        cy += grid_h + panel_pad + divider_h + 6.0;
        let hotbar_x0 = panel_x + panel_pad;
        if mx >= hotbar_x0 && mx < hotbar_x0 + grid_w && my >= cy && my < cy + hotbar_h {
            let slot_idx = ((mx - hotbar_x0) / (slot_size + slot_gap)) as usize;
            if slot_idx < 9 {
                self.gameplay.hotbar.selected = slot_idx;
            }
        }
    }

    /// Resolve the entity rendered at the mouse cursor position in the ECS inspector panel.
    pub(crate) fn entity_at_slot(&self, w: f32, h: f32, mouse: Point) -> Option<voxel_ecs::Entity> {
        let panel_w = 380.0;
        let panel_x = w - panel_w - 8.0;
        let panel_y = 8.0;
        let line_h = 14.0;
        let pad = 4.0;
        let cutoff = h - 12.0;
        let Point { x: mx, y: my } = mouse;

        if !(mx >= panel_x && mx <= panel_x + panel_w) {
            return None;
        }

        let mut y = panel_y + pad;
        if y + line_h >= cutoff {
            return None;
        }
        y += line_h;

        for arch in self.simulation.ecs_world().archetypes() {
            if y + line_h >= cutoff {
                break;
            }
            y += line_h;
            for (row, entity) in arch.entities().iter().enumerate() {
                if y + line_h >= cutoff {
                    break;
                }
                if my >= y && my < y + line_h {
                    return Some(*entity);
                }
                y += line_h;
                for col_idx in 0..arch.component_types.len() {
                    if y + line_h >= cutoff {
                        break;
                    }
                    y += line_h;
                    let value_lines = arch.columns()[col_idx]
                        .value_as_any(row as u32)
                        .and_then(|raw| {
                            self.simulation
                                .ecs_world()
                                .format_component(arch.component_types[col_idx], raw)
                        })
                        .map(|t| t.lines().count())
                        .unwrap_or(1);
                    y += value_lines as f32 * line_h;
                }
            }
        }
        None
    }

    /// Click handler for the ECS inspector.
    pub(crate) fn handle_ecs_inspector_click(&mut self) {
        let (w, h) = self.render.logical_size();
        if let Some(e) = self.entity_at_slot(w, h, self.gameplay.mouse_pos) {
            self.gameplay.pinned_entity = Some(e);
            self.gameplay
                .chat
                .push_message(format!("Pin: e[{}:{}]", e.index, e.generation));
        }
    }

    /// Handle a click in the pause menu (4 buttons).
    pub(crate) fn handle_pause_click(&mut self) {
        if let Some(buttons) = self.gameplay.pause_buttons {
            // Play UI click sound.
            self.audio.push_event(voxel_audio::AudioEvent::PlaySfx {
                sound: "ui.click".into(),
                position: None,
                volume: 1.0,
                pitch: None,
                group: voxel_audio::AudioGroup::Sfx,
            });
            // Back to Game
            if buttons[0].contains(self.gameplay.mouse_pos) {
                self.enter_playing();
            }
            // Options
            if buttons[1].contains(self.gameplay.mouse_pos) {
                self.enter_settings(GameState::PauseMenu);
            }
            // Save & Quit to Title
            if buttons[2].contains(self.gameplay.mouse_pos) {
                // Save the world and update metadata.
                if let Some(ref save_path) = self.gameplay.current_world_path {
                    let _ = self.save_entities(save_path);
                    let _ = voxel_world::save::save_world(&self.world_state.world, save_path);
                    // Update world_info.json with play time and last_played.
                    if let Some(mut info) = crate::save::read_world_info(save_path) {
                        info.play_time_seconds += self.gameplay.play_time_accumulator as u64;
                        info.last_played = crate::save::chrono_now();
                        let _ = crate::save::write_world_info(save_path, &info);
                    }
                }
                self.gameplay.play_time_accumulator = 0.0;
                self.enter_title_screen();
            }
            // Quit Game
            if buttons[3].contains(self.gameplay.mouse_pos) {
                log::info!("exit game requested");
                self.gameplay.want_exit = true;
            }
        }
    }

    /// Handle a click on the death screen (Respawn / Title Screen).
    pub(crate) fn handle_death_screen_click(&mut self) {
        let (w, h) = self.render.logical_size();

        let btn_w = 200.0;
        let btn_h = 40.0;
        let btn_x = (w - btn_w) * 0.5;
        let btn_y = h * 0.5;
        let respawn_btn = Rect {
            x: btn_x,
            y: btn_y,
            w: btn_w,
            h: btn_h,
        };
        let title_btn = Rect {
            x: btn_x,
            y: btn_y + 60.0,
            w: btn_w,
            h: btn_h,
        };

        // Respawn
        if respawn_btn.contains(self.gameplay.mouse_pos) {
            self.respawn_player();
        }
        // Title Screen
        if title_btn.contains(self.gameplay.mouse_pos) {
            // Save and return to title.
            if let Some(ref save_path) = self.gameplay.current_world_path {
                let _ = self.save_entities(save_path);
                let _ = voxel_world::save::save_world(&self.world_state.world, save_path);
                if let Some(mut info) = crate::save::read_world_info(save_path) {
                    info.play_time_seconds += self.gameplay.play_time_accumulator as u64;
                    info.last_played = crate::save::chrono_now();
                    let _ = crate::save::write_world_info(save_path, &info);
                }
            }
            self.gameplay.play_time_accumulator = 0.0;
            self.enter_title_screen();
        }
    }

    /// Respawn the player: reset health, hunger, position; lock cursor.
    fn respawn_player(&mut self) {
        let ecs = self.simulation.ecs_world_mut();
        if let Some(player) = ecs.resource::<voxel_game::PlayerEntity>().and_then(|p| p.0) {
            // Reset health.
            if let Some(health) = ecs.get_mut::<voxel_game::Health>(player) {
                health.reset();
            }
            // Reset hunger.
            if let Some(hunger) = ecs.get_mut::<voxel_game::Hunger>(player) {
                hunger.reset();
            }
            // Reset air supply.
            if let Some(air) = ecs.get_mut::<voxel_game::AirSupply>(player) {
                *air = voxel_game::AirSupply::default();
            }
            // Teleport to spawn.
            let spawn = self.gameplay.spawn_pos;
            if let Some(t) = ecs.get_mut::<voxel_game::Transform>(player) {
                t.pos = spawn;
            }
            if let Some(v) = ecs.get_mut::<voxel_game::Velocity>(player) {
                v.lin = glam::Vec3::ZERO;
            }
        }
        self.lock_cursor();
    }

    // ── Title Screen ────────────────────────────────────────────────────

    fn draw_title_screen(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        // Background: dark gradient (panorama renders behind, this is just a fallback).
        ui.quad(0.0, 0.0, w, h, [10, 10, 15, 255]);

        let panel_w = 320.0;
        let btn_h = 50.0;
        let spacing = 70.0;
        let _total_h = btn_h * 4.0 + spacing * 3.0;
        let btn_start_y = h * 0.38;
        let btn_x = (w - panel_w) * 0.5;

        // Title — positioned above the buttons with comfortable spacing.
        let title = "VOXEL ENGINE";
        let tw = self.render.font.text_width(title, 4.0);
        ui.text(
            title,
            (w - tw) * 0.5,
            btn_start_y - 70.0,
            4.0,
            [255, 255, 255, 255],
            &self.render.font,
        );

        // Subtitle
        let sub = "voxel engine v0.1";
        let sw = self.render.font.text_width(sub, 1.0);
        ui.text(
            sub,
            (w - sw) * 0.5,
            btn_start_y - 36.0,
            1.0,
            [150, 150, 150, 255],
            &self.render.font,
        );

        // Buttons
        let btn_labels = ["SINGLEPLAYER", "MULTIPLAYER", "OPTIONS", "QUIT GAME"];
        let btn_colors: [([u8; 4], [u8; 4], [u8; 4]); 4] = [
            ([40, 80, 40, 220], [60, 120, 60, 255], [120, 200, 120, 255]),
            ([50, 50, 50, 200], [60, 60, 60, 255], [100, 100, 100, 200]),
            ([50, 50, 80, 220], [70, 70, 110, 255], [120, 120, 200, 255]),
            ([90, 30, 30, 220], [140, 50, 50, 255], [220, 100, 100, 255]),
        ];

        let mut buttons = [Rect {
            x: 0.0,
            y: 0.0,
            w: 0.0,
            h: 0.0,
        }; 4];
        for (i, label) in btn_labels.iter().enumerate() {
            let by = btn_start_y + i as f32 * spacing;
            let (fill, hover, border) = btn_colors[i];
            self.draw_button(
                ui,
                Rect {
                    x: btn_x,
                    y: by,
                    w: panel_w,
                    h: btn_h,
                },
                label,
                fill,
                hover,
                border,
            );
            buttons[i] = Rect {
                x: btn_x,
                y: by,
                w: panel_w,
                h: btn_h,
            };
        }

        // "Multiplayer" is grayed out — draw "Coming Soon" on hover.
        let mp_btn = buttons[1];
        if mp_btn.contains(self.gameplay.mouse_pos) {
            let cs = "Coming Soon";
            let csw = self.render.font.text_width(cs, 0.8);
            ui.text(
                cs,
                mp_btn.x + (mp_btn.w - csw) * 0.5,
                mp_btn.y + mp_btn.h + 4.0,
                0.8,
                [150, 150, 150, 200],
                &self.render.font,
            );
        }

        // Copyright
        let copyright = "Copyright 2026";
        let cw = self.render.font.text_width(copyright, 0.7);
        ui.text(
            copyright,
            (w - cw) * 0.5,
            h - 24.0,
            0.7,
            [80, 80, 80, 200],
            &self.render.font,
        );

        self.gameplay.title_buttons = Some(buttons);
    }

    pub(crate) fn handle_title_click(&mut self) {
        if let Some(buttons) = self.gameplay.title_buttons {
            // Play UI click sound.
            self.audio.push_event(voxel_audio::AudioEvent::PlaySfx {
                sound: "ui.click".into(),
                position: None,
                volume: 1.0,
                pitch: None,
                group: voxel_audio::AudioGroup::Sfx,
            });
            // Singleplayer — show world selection screen.
            if buttons[0].contains(self.gameplay.mouse_pos) {
                self.enter_world_select();
            }
            // Multiplayer (not implemented)
            if buttons[1].contains(self.gameplay.mouse_pos) {
                self.gameplay
                    .chat
                    .push_message("Multiplayer not yet implemented".into());
            }
            // Options
            if buttons[2].contains(self.gameplay.mouse_pos) {
                self.enter_settings(GameState::TitleScreen);
            }
            // Quit
            if buttons[3].contains(self.gameplay.mouse_pos) {
                self.gameplay.want_exit = true;
            }
        }
    }

    // ── World Select ────────────────────────────────────────────────────

    fn draw_world_select(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        // Full-screen dimmed overlay.
        ui.quad(0.0, 0.0, w, h, [0, 0, 0, 200]);

        let panel_w = 600.0f32;
        let panel_h = 500.0f32;
        let px = (w - panel_w) * 0.5;
        let py = (h - panel_h) * 0.5;

        ui.quad(px, py, panel_w, panel_h, [25, 25, 30, 240]);
        ui.rect_border(px, py, panel_w, panel_h, 2.0, [80, 80, 100, 255]);

        // Title
        ui.text(
            "Select World",
            px + 16.0,
            py + 12.0,
            1.5,
            [255, 255, 255, 255],
            &self.render.font,
        );

        // Close button
        let close_x = px + panel_w - 30.0;
        let close_y = py + 8.0;
        let close_hovered = Rect {
            x: close_x,
            y: close_y,
            w: 22.0,
            h: 22.0,
        }
        .contains(self.gameplay.mouse_pos);
        ui.quad(
            close_x,
            close_y,
            22.0,
            22.0,
            if close_hovered {
                [140, 50, 50, 255]
            } else {
                [80, 30, 30, 200]
            },
        );
        ui.text(
            "X",
            close_x + 5.0,
            close_y + 3.0,
            1.0,
            [255, 200, 200, 255],
            &self.render.font,
        );

        // World list
        let list_y = py + 40.0;
        let list_h = panel_h - 100.0;
        let row_h = 60.0;
        let mut buttons = crate::WorldSelectButtons {
            rows: Vec::new(),
            delete_buttons: Vec::new(),
            create_btn: Rect {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            },
            play_btn: Rect {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            },
            close_btn: Rect {
                x: close_x,
                y: close_y,
                w: 22.0,
                h: 22.0,
            },
        };

        let mut ry = list_y;
        for (i, world) in self.gameplay.world_list.iter().enumerate() {
            if ry + row_h > list_y + list_h {
                break;
            }
            let selected = self.gameplay.selected_world_index == Some(i);
            let row_hovered = Rect {
                x: px + 8.0,
                y: ry,
                w: panel_w - 16.0,
                h: row_h - 4.0,
            }
            .contains(self.gameplay.mouse_pos);

            let bg = if selected {
                [50, 70, 100, 255]
            } else if row_hovered {
                [40, 40, 50, 255]
            } else {
                [30, 30, 35, 255]
            };
            ui.quad(px + 8.0, ry, panel_w - 16.0, row_h - 4.0, bg);
            ui.rect_border(
                px + 8.0,
                ry,
                panel_w - 16.0,
                row_h - 4.0,
                1.0,
                if selected {
                    [74, 127, 212, 255]
                } else {
                    [50, 50, 60, 255]
                },
            );

            // World name
            ui.text(
                &world.name,
                px + 16.0,
                ry + 6.0,
                1.2,
                [220, 220, 220, 255],
                &self.render.font,
            );
            // Seed + last played (human-readable relative time)
            let played_str = format_last_played(&world.last_played);
            let info = format!("Seed: {}  |  Played: {}", world.seed, played_str);
            ui.text(
                &info,
                px + 16.0,
                ry + 24.0,
                0.8,
                [140, 140, 140, 255],
                &self.render.font,
            );
            // Game mode
            ui.text(
                &world.game_mode,
                px + 16.0,
                ry + 38.0,
                0.7,
                [100, 100, 100, 200],
                &self.render.font,
            );

            // Delete button
            let del_x = px + panel_w - 60.0;
            let del_hovered = Rect {
                x: del_x,
                y: ry + 10.0,
                w: 40.0,
                h: 20.0,
            }
            .contains(self.gameplay.mouse_pos);
            ui.quad(
                del_x,
                ry + 10.0,
                40.0,
                20.0,
                if del_hovered {
                    [140, 50, 50, 255]
                } else {
                    [80, 30, 30, 200]
                },
            );
            ui.text(
                "Del",
                del_x + 6.0,
                ry + 12.0,
                0.7,
                [255, 200, 200, 255],
                &self.render.font,
            );

            buttons.rows.push(Rect {
                x: px + 8.0,
                y: ry,
                w: panel_w - 16.0,
                h: row_h - 4.0,
            });
            buttons.delete_buttons.push(Rect {
                x: del_x,
                y: ry + 10.0,
                w: 40.0,
                h: 20.0,
            });

            ry += row_h;
        }

        // Create New World button
        let create_y = list_y + list_h + 4.0;
        let create_w = 160.0;
        let create_x = px + 8.0;
        let create_hovered = Rect {
            x: create_x,
            y: create_y,
            w: create_w,
            h: 28.0,
        }
        .contains(self.gameplay.mouse_pos);
        ui.quad(
            create_x,
            create_y,
            create_w,
            28.0,
            if create_hovered {
                [50, 70, 50, 255]
            } else {
                [35, 50, 35, 200]
            },
        );
        ui.rect_border(create_x, create_y, create_w, 28.0, 1.0, [80, 120, 80, 255]);
        ui.text(
            "+ Create New World",
            create_x + 8.0,
            create_y + 6.0,
            0.8,
            [180, 220, 180, 255],
            &self.render.font,
        );
        buttons.create_btn = Rect {
            x: create_x,
            y: create_y,
            w: create_w,
            h: 28.0,
        };

        // Play Selected World button
        let play_w = 180.0;
        let play_x = px + panel_w - play_w - 8.0;
        let play_enabled = self.gameplay.selected_world_index.is_some();
        let play_hovered = Rect {
            x: play_x,
            y: create_y,
            w: play_w,
            h: 28.0,
        }
        .contains(self.gameplay.mouse_pos);
        let play_bg = if !play_enabled {
            [40, 40, 40, 150]
        } else if play_hovered {
            [50, 80, 50, 255]
        } else {
            [35, 60, 35, 220]
        };
        ui.quad(play_x, create_y, play_w, 28.0, play_bg);
        ui.rect_border(
            play_x,
            create_y,
            play_w,
            28.0,
            1.0,
            if play_enabled {
                [100, 180, 100, 255]
            } else {
                [60, 60, 60, 150]
            },
        );
        ui.text(
            "Play Selected World",
            play_x + 12.0,
            create_y + 6.0,
            0.8,
            if play_enabled {
                [200, 255, 200, 255]
            } else {
                [100, 100, 100, 150]
            },
            &self.render.font,
        );
        buttons.play_btn = Rect {
            x: play_x,
            y: create_y,
            w: play_w,
            h: 28.0,
        };

        self.gameplay.world_select_buttons = Some(buttons);
    }

    pub(crate) fn handle_world_select_click(&mut self) {
        // If delete confirmation is showing, route click there.
        if self.gameplay.pending_delete.is_some() {
            self.handle_delete_confirm_click();
            return;
        }
        // If create dialog is showing, route click there.
        if self.gameplay.create_world_state.is_some() {
            self.handle_create_world_click();
            return;
        }

        let Some(buttons) = self.gameplay.world_select_buttons.clone() else {
            return;
        };

        // Close button
        if buttons.close_btn.contains(self.gameplay.mouse_pos) {
            self.enter_title_screen();
            return;
        }

        // Delete buttons — check BEFORE row selection (delete sits inside the row rect).
        for (i, &rect) in buttons.delete_buttons.iter().enumerate() {
            if rect.contains(self.gameplay.mouse_pos) {
                if i < self.gameplay.world_list.len() {
                    self.gameplay.pending_delete = Some(i);
                }
                return;
            }
        }

        // World row selection (with double-click detection)
        for (i, &rect) in buttons.rows.iter().enumerate() {
            if rect.contains(self.gameplay.mouse_pos) {
                self.gameplay.selected_world_index = Some(i);

                // Double-click detection: if same row clicked within 400ms, play it.
                let now = std::time::Instant::now();
                let is_double = self.gameplay.last_click_row == Some(i)
                    && self
                        .gameplay
                        .last_click_time
                        .map(|t| now.duration_since(t).as_millis() < 400)
                        .unwrap_or(false);
                self.gameplay.last_click_time = Some(now);
                self.gameplay.last_click_row = Some(i);

                if is_double && i < self.gameplay.world_list.len() {
                    let save_path = self.gameplay.world_list[i].path.clone();
                    self.load_and_play_world(save_path);
                }
                return;
            }
        }

        // Create New World
        if buttons.create_btn.contains(self.gameplay.mouse_pos) {
            self.gameplay.create_world_state = Some(crate::CreateWorldState::default());
            return;
        }

        // Play Selected World
        if buttons.play_btn.contains(self.gameplay.mouse_pos) {
            if let Some(idx) = self.gameplay.selected_world_index {
                if idx < self.gameplay.world_list.len() {
                    let save_path = self.gameplay.world_list[idx].path.clone();
                    self.load_and_play_world(save_path);
                }
            }
        }
    }

    /// Load a world from a save directory and start playing.
    fn load_and_play_world(&mut self, save_path: std::path::PathBuf) {
        // Load world chunks.
        match voxel_world::save::load_world(&save_path) {
            Ok((seed, chunks)) => {
                log::info!(
                    "loaded world from {} (seed={}, {} chunks)",
                    save_path.display(),
                    seed,
                    chunks.len()
                );
                // Clear existing world and insert loaded chunks.
                for (pos, chunk) in chunks {
                    self.world_state.world.insert_chunk(pos, chunk);
                }
                // Load entity state.
                let _ = self.load_entities(&save_path);
                // Load world_info and restore cheats flag.
                if let Some(mut info) = crate::save::read_world_info(&save_path) {
                    info.last_played = crate::save::chrono_now();
                    let _ = crate::save::write_world_info(&save_path, &info);
                    self.gameplay.cheats_enabled = info.cheats_enabled();
                }
                self.gameplay.current_world_path = Some(save_path);
                self.gameplay.play_time_accumulator = 0.0;
                self.input.spawned = false;
                self.enter_playing();
            }
            Err(e) => {
                log::error!("failed to load world: {e}");
                self.gameplay
                    .chat
                    .push_message(format!("Failed to load world: {e}"));
            }
        }
    }

    /// Create a world from the create dialog's user input.
    pub(crate) fn create_world_from_dialog(
        &mut self,
        name: String,
        seed_str: String,
        game_mode: String,
        allow_cheats: bool,
    ) {
        let saves_dir = std::path::PathBuf::from("saves");

        // Parse seed: empty = random, otherwise parse as i32.
        let seed = if seed_str.is_empty() {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i32)
                .unwrap_or(1337)
        } else {
            seed_str.parse::<i32>().unwrap_or_else(|_| {
                // Hash the string to a seed.
                seed_str
                    .bytes()
                    .fold(0i32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as i32))
            })
        };

        // Sanitize the world name for use as a directory name.
        let dir_name = sanitize_filename(&name);

        // Handle name collision by appending _1, _2, etc.
        let mut save_path = saves_dir.join(&dir_name);
        if save_path.exists() {
            for i in 1..1000 {
                let candidate = format!("{}_{}", dir_name, i);
                save_path = saves_dir.join(&candidate);
                if !save_path.exists() {
                    break;
                }
            }
        }

        // Create the save directory.
        if let Err(e) = std::fs::create_dir_all(&save_path) {
            log::error!("failed to create save dir: {e}");
            self.gameplay
                .chat
                .push_message(format!("Failed to create world: {e}"));
            return;
        }

        // Write world_info.json.
        let mut info = crate::save::WorldInfo::new_default(&name, seed);
        info.game_mode = game_mode;
        if allow_cheats {
            info.flags.push("cheats".to_string());
        }
        let _ = crate::save::write_world_info(&save_path, &info);

        // Save the initial world state.
        if let Err(e) = voxel_world::save::save_world(&self.world_state.world, &save_path) {
            log::error!("failed to save initial world: {e}");
        }

        log::info!(
            "created new world '{}' (seed={}, mode={}, cheats={}) at {}",
            name,
            seed,
            info.game_mode,
            allow_cheats,
            save_path.display()
        );
        self.gameplay.current_world_path = Some(save_path);
        self.gameplay.play_time_accumulator = 0.0;
        self.gameplay.cheats_enabled = allow_cheats;
        self.input.spawned = false;
        self.enter_playing();
    }

    // ── Create World Dialog ─────────────────────────────────────────────

    fn draw_create_world_dialog(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        // Dimmed overlay.
        ui.quad(0.0, 0.0, w, h, [0, 0, 0, 160]);

        let panel_w = 400.0f32;
        let panel_h = 320.0f32;
        let px = (w - panel_w) * 0.5;
        let py = (h - panel_h) * 0.5;

        ui.quad(px, py, panel_w, panel_h, [25, 25, 30, 245]);
        ui.rect_border(px, py, panel_w, panel_h, 2.0, [80, 80, 100, 255]);

        // Title + close button.
        ui.text(
            "Create New World",
            px + 16.0,
            py + 12.0,
            1.3,
            [255, 255, 255, 255],
            &self.render.font,
        );
        let close_x = px + panel_w - 28.0;
        let close_y = py + 8.0;
        let close_hovered = Rect {
            x: close_x,
            y: close_y,
            w: 20.0,
            h: 20.0,
        }
        .contains(self.gameplay.mouse_pos);
        ui.quad(
            close_x,
            close_y,
            20.0,
            20.0,
            if close_hovered {
                [140, 50, 50, 255]
            } else {
                [80, 30, 30, 200]
            },
        );
        ui.text(
            "x",
            close_x + 5.0,
            close_y + 2.0,
            0.9,
            [255, 200, 200, 255],
            &self.render.font,
        );

        let Some(ref state) = self.gameplay.create_world_state else {
            return;
        };
        let lx = px + 16.0;
        let input_w = panel_w - 32.0;
        let input_h = 24.0;

        // World Name label + input.
        let name_label_y = py + 42.0;
        ui.text(
            "World Name:",
            lx,
            name_label_y,
            0.8,
            [160, 160, 160, 255],
            &self.render.font,
        );
        let name_y = name_label_y + 16.0;
        let name_active = state.active_field == 0;
        ui.quad(lx, name_y, input_w, input_h, [15, 15, 20, 255]);
        ui.rect_border(
            lx,
            name_y,
            input_w,
            input_h,
            1.0,
            if name_active {
                [74, 127, 212, 255]
            } else {
                [60, 60, 70, 255]
            },
        );
        // Draw text with blinking cursor.
        let name_display = if name_active {
            let blink = ((std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
                / 500)
                % 2)
                == 0;
            if blink {
                format!("{}|", state.name)
            } else {
                format!("{} ", state.name)
            }
        } else {
            state.name.clone()
        };
        ui.text(
            &name_display,
            lx + 4.0,
            name_y + 5.0,
            0.8,
            [220, 220, 220, 255],
            &self.render.font,
        );

        // Seed label + input.
        let seed_label_y = name_y + input_h + 8.0;
        ui.text(
            "Seed (blank = random):",
            lx,
            seed_label_y,
            0.8,
            [160, 160, 160, 255],
            &self.render.font,
        );
        let seed_y = seed_label_y + 16.0;
        let seed_active = state.active_field == 1;
        ui.quad(lx, seed_y, input_w, input_h, [15, 15, 20, 255]);
        ui.rect_border(
            lx,
            seed_y,
            input_w,
            input_h,
            1.0,
            if seed_active {
                [74, 127, 212, 255]
            } else {
                [60, 60, 70, 255]
            },
        );
        let seed_display = if seed_active {
            let blink = ((std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
                / 500)
                % 2)
                == 0;
            if blink {
                format!("{}|", state.seed)
            } else {
                format!("{} ", state.seed)
            }
        } else {
            state.seed.clone()
        };
        ui.text(
            &seed_display,
            lx + 4.0,
            seed_y + 5.0,
            0.8,
            [220, 220, 220, 255],
            &self.render.font,
        );

        // Game Mode radio buttons.
        let mode_y = seed_y + input_h + 10.0;
        ui.text(
            "Game Mode:",
            lx,
            mode_y,
            0.8,
            [160, 160, 160, 255],
            &self.render.font,
        );
        let radio_y = mode_y + 16.0;
        let surv_active = state.game_mode == "survival";
        let crea_active = state.game_mode == "creative";

        // Survival radio.
        let surv_x = lx + 10.0;
        ui.quad(surv_x, radio_y, 12.0, 12.0, [15, 15, 20, 255]);
        ui.rect_border(surv_x, radio_y, 12.0, 12.0, 1.0, [60, 60, 70, 255]);
        if surv_active {
            ui.quad(surv_x + 3.0, radio_y + 3.0, 6.0, 6.0, [74, 127, 212, 255]);
        }
        ui.text(
            "Survival",
            surv_x + 18.0,
            radio_y,
            0.8,
            [200, 200, 200, 255],
            &self.render.font,
        );

        // Creative radio.
        let crea_x = surv_x + 100.0;
        ui.quad(crea_x, radio_y, 12.0, 12.0, [15, 15, 20, 255]);
        ui.rect_border(crea_x, radio_y, 12.0, 12.0, 1.0, [60, 60, 70, 255]);
        if crea_active {
            ui.quad(crea_x + 3.0, radio_y + 3.0, 6.0, 6.0, [74, 127, 212, 255]);
        }
        ui.text(
            "Creative",
            crea_x + 18.0,
            radio_y,
            0.8,
            [200, 200, 200, 255],
            &self.render.font,
        );

        // Allow Cheats checkbox.
        let cheats_y = radio_y + 22.0;
        let cheats_x = lx + 10.0;
        ui.quad(cheats_x, cheats_y, 12.0, 12.0, [15, 15, 20, 255]);
        ui.rect_border(cheats_x, cheats_y, 12.0, 12.0, 1.0, [60, 60, 70, 255]);
        if state.allow_cheats {
            // Draw checkmark.
            ui.quad(
                cheats_x + 3.0,
                cheats_y + 3.0,
                6.0,
                6.0,
                [74, 127, 212, 255],
            );
        }
        ui.text(
            "Allow Cheats",
            cheats_x + 18.0,
            cheats_y,
            0.8,
            [200, 200, 200, 255],
            &self.render.font,
        );

        // Error message.
        if let Some(ref err) = state.error {
            ui.text(
                err,
                lx,
                cheats_y + 22.0,
                0.7,
                [220, 80, 80, 255],
                &self.render.font,
            );
        }

        // Buttons: Cancel + Create World.
        let btn_y = py + panel_h - 40.0;
        let btn_h = 28.0;
        let cancel_w = 80.0;
        let create_w = 120.0;
        let cancel_x = px + panel_w - cancel_w - create_w - 20.0;
        let create_x = px + panel_w - create_w - 8.0;

        let cancel_hovered = Rect {
            x: cancel_x,
            y: btn_y,
            w: cancel_w,
            h: btn_h,
        }
        .contains(self.gameplay.mouse_pos);
        ui.quad(
            cancel_x,
            btn_y,
            cancel_w,
            btn_h,
            if cancel_hovered {
                [60, 60, 60, 255]
            } else {
                [40, 40, 40, 220]
            },
        );
        ui.rect_border(cancel_x, btn_y, cancel_w, btn_h, 1.0, [80, 80, 80, 255]);
        ui.text(
            "Cancel",
            cancel_x + 16.0,
            btn_y + 6.0,
            0.8,
            [200, 200, 200, 255],
            &self.render.font,
        );

        let name_empty = state.name.trim().is_empty();
        let create_hovered = Rect {
            x: create_x,
            y: btn_y,
            w: create_w,
            h: btn_h,
        }
        .contains(self.gameplay.mouse_pos);
        let create_bg = if name_empty {
            [40, 40, 40, 150]
        } else if create_hovered {
            [50, 90, 50, 255]
        } else {
            [35, 65, 35, 220]
        };
        ui.quad(create_x, btn_y, create_w, btn_h, create_bg);
        ui.rect_border(
            create_x,
            btn_y,
            create_w,
            btn_h,
            1.0,
            if name_empty {
                [60, 60, 60, 150]
            } else {
                [80, 150, 80, 255]
            },
        );
        ui.text(
            "Create World",
            create_x + 10.0,
            btn_y + 6.0,
            0.8,
            if name_empty {
                [100, 100, 100, 150]
            } else {
                [200, 255, 200, 255]
            },
            &self.render.font,
        );

        // Store rects for click handling.
        if let Some(ref mut state) = self.gameplay.create_world_state {
            state.rects = Some(crate::CreateWorldRects {
                name_input: Rect {
                    x: lx,
                    y: name_y,
                    w: input_w,
                    h: input_h,
                },
                seed_input: Rect {
                    x: lx,
                    y: seed_y,
                    w: input_w,
                    h: input_h,
                },
                cancel_btn: Rect {
                    x: cancel_x,
                    y: btn_y,
                    w: cancel_w,
                    h: btn_h,
                },
                create_btn: Rect {
                    x: create_x,
                    y: btn_y,
                    w: create_w,
                    h: btn_h,
                },
                mode_survival: Rect {
                    x: surv_x,
                    y: radio_y,
                    w: 80.0,
                    h: 16.0,
                },
                mode_creative: Rect {
                    x: crea_x,
                    y: radio_y,
                    w: 80.0,
                    h: 16.0,
                },
                cheats_toggle: Rect {
                    x: cheats_x,
                    y: cheats_y,
                    w: 120.0,
                    h: 16.0,
                },
            });
        }
    }

    fn handle_create_world_click(&mut self) {
        let Some(rects) = self
            .gameplay
            .create_world_state
            .as_ref()
            .and_then(|s| s.rects.clone())
        else {
            return;
        };

        // Name input field click.
        if rects.name_input.contains(self.gameplay.mouse_pos) {
            if let Some(ref mut state) = self.gameplay.create_world_state {
                state.active_field = 0;
            }
            return;
        }

        // Seed input field click.
        if rects.seed_input.contains(self.gameplay.mouse_pos) {
            if let Some(ref mut state) = self.gameplay.create_world_state {
                state.active_field = 1;
            }
            return;
        }

        // Survival radio.
        if rects.mode_survival.contains(self.gameplay.mouse_pos) {
            if let Some(ref mut state) = self.gameplay.create_world_state {
                state.game_mode = "survival".into();
            }
            return;
        }

        // Creative radio.
        if rects.mode_creative.contains(self.gameplay.mouse_pos) {
            if let Some(ref mut state) = self.gameplay.create_world_state {
                state.game_mode = "creative".into();
            }
            return;
        }

        // Allow Cheats toggle.
        if rects.cheats_toggle.contains(self.gameplay.mouse_pos) {
            if let Some(ref mut state) = self.gameplay.create_world_state {
                state.allow_cheats = !state.allow_cheats;
            }
            return;
        }

        // Cancel button.
        if rects.cancel_btn.contains(self.gameplay.mouse_pos) {
            self.gameplay.create_world_state = None;
            return;
        }

        // Create World button.
        if rects.create_btn.contains(self.gameplay.mouse_pos) {
            let (name, seed, mode, cheats) =
                if let Some(ref state) = self.gameplay.create_world_state {
                    let name = state.name.trim().to_string();
                    let seed = state.seed.trim().to_string();
                    let mode = state.game_mode.clone();
                    let cheats = state.allow_cheats;
                    if name.is_empty() {
                        if let Some(ref mut s) = self.gameplay.create_world_state {
                            s.error = Some("World name cannot be empty".into());
                        }
                        return;
                    }
                    (name, seed, mode, cheats)
                } else {
                    return;
                };
            self.gameplay.create_world_state = None;
            self.create_world_from_dialog(name, seed, mode, cheats);
        }
    }

    // ── Delete Confirmation Dialog ──────────────────────────────────────

    fn draw_delete_confirm_dialog(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        let Some(idx) = self.gameplay.pending_delete else {
            return;
        };
        let world_name = self
            .gameplay
            .world_list
            .get(idx)
            .map(|w| w.name.as_str())
            .unwrap_or("Unknown");

        // Dimmed overlay.
        ui.quad(0.0, 0.0, w, h, [0, 0, 0, 160]);

        let panel_w = 340.0f32;
        let panel_h = 160.0f32;
        let px = (w - panel_w) * 0.5;
        let py = (h - panel_h) * 0.5;

        ui.quad(px, py, panel_w, panel_h, [30, 25, 25, 245]);
        ui.rect_border(px, py, panel_w, panel_h, 2.0, [100, 60, 60, 255]);

        // Title.
        ui.text(
            "Delete World",
            px + 16.0,
            py + 12.0,
            1.2,
            [220, 100, 100, 255],
            &self.render.font,
        );

        // Message.
        let msg1 = "Are you sure you want to delete".to_string();
        let msg2 = format!("\"{}\"?", world_name);
        let msg3 = "This cannot be undone.".to_string();
        ui.text(
            &msg1,
            px + 16.0,
            py + 38.0,
            0.8,
            [180, 180, 180, 255],
            &self.render.font,
        );
        ui.text(
            &msg2,
            px + 16.0,
            py + 54.0,
            0.8,
            [220, 200, 200, 255],
            &self.render.font,
        );
        ui.text(
            &msg3,
            px + 16.0,
            py + 72.0,
            0.7,
            [160, 120, 120, 200],
            &self.render.font,
        );

        // Buttons: Cancel + Delete World.
        let btn_y = py + panel_h - 36.0;
        let btn_h = 26.0;
        let cancel_w = 70.0;
        let delete_w = 100.0;
        let cancel_x = px + panel_w - cancel_w - delete_w - 16.0;
        let delete_x = px + panel_w - delete_w - 8.0;

        let cancel_hovered = Rect {
            x: cancel_x,
            y: btn_y,
            w: cancel_w,
            h: btn_h,
        }
        .contains(self.gameplay.mouse_pos);
        ui.quad(
            cancel_x,
            btn_y,
            cancel_w,
            btn_h,
            if cancel_hovered {
                [60, 60, 60, 255]
            } else {
                [40, 40, 40, 220]
            },
        );
        ui.rect_border(cancel_x, btn_y, cancel_w, btn_h, 1.0, [80, 80, 80, 255]);
        ui.text(
            "Cancel",
            cancel_x + 10.0,
            btn_y + 5.0,
            0.8,
            [200, 200, 200, 255],
            &self.render.font,
        );

        let delete_hovered = Rect {
            x: delete_x,
            y: btn_y,
            w: delete_w,
            h: btn_h,
        }
        .contains(self.gameplay.mouse_pos);
        ui.quad(
            delete_x,
            btn_y,
            delete_w,
            btn_h,
            if delete_hovered {
                [160, 50, 50, 255]
            } else {
                [100, 35, 35, 220]
            },
        );
        ui.rect_border(delete_x, btn_y, delete_w, btn_h, 1.0, [200, 80, 80, 255]);
        ui.text(
            "Delete World",
            delete_x + 8.0,
            btn_y + 5.0,
            0.8,
            [255, 200, 200, 255],
            &self.render.font,
        );

        // Store rects for click handling (reuse pending_delete for state).
        // We'll check coordinates directly in handle_delete_confirm_click.
    }

    fn handle_delete_confirm_click(&mut self) {
        let Some(idx) = self.gameplay.pending_delete else {
            return;
        };

        // Compute the same layout as draw_delete_confirm_dialog.
        let (w, h) = self.render.logical_size();
        let panel_w = 340.0f32;
        let panel_h = 160.0f32;
        let px = (w - panel_w) * 0.5;
        let py = (h - panel_h) * 0.5;
        let btn_y = py + panel_h - 36.0;
        let btn_h = 26.0;
        let cancel_w = 70.0;
        let delete_w = 100.0;
        let cancel_x = px + panel_w - cancel_w - delete_w - 16.0;
        let delete_x = px + panel_w - delete_w - 8.0;

        // Cancel.
        if (Rect {
            x: cancel_x,
            y: btn_y,
            w: cancel_w,
            h: btn_h,
        })
        .contains(self.gameplay.mouse_pos)
        {
            self.gameplay.pending_delete = None;
            return;
        }

        // Delete World.
        if (Rect {
            x: delete_x,
            y: btn_y,
            w: delete_w,
            h: btn_h,
        })
        .contains(self.gameplay.mouse_pos)
        {
            if idx < self.gameplay.world_list.len() {
                let path = self.gameplay.world_list[idx].path.clone();
                if path.exists() {
                    let _ = std::fs::remove_dir_all(&path);
                    log::info!("deleted world: {}", self.gameplay.world_list[idx].name);
                }
            }
            self.gameplay.pending_delete = None;
            self.enter_world_select(); // refresh list
        }
    }

    // ── Settings Menu ───────────────────────────────────────────────────

    fn draw_settings_menu(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        // Dark overlay.
        ui.quad(0.0, 0.0, w, h, [0, 0, 0, 180]);

        let panel_w = 500.0f32;
        let panel_h = 540.0f32;
        let px = (w - panel_w) * 0.5;
        let py = (h - panel_h) * 0.5;

        ui.quad(px, py, panel_w, panel_h, [25, 25, 30, 240]);
        ui.rect_border(px, py, panel_w, panel_h, 2.0, [80, 80, 100, 255]);

        // Title + Back button
        ui.text(
            "OPTIONS",
            px + 16.0,
            py + 12.0,
            1.5,
            [255, 255, 255, 255],
            &self.render.font,
        );

        let back_x = px + panel_w - 70.0;
        let back_y = py + 8.0;
        let back_hovered = Rect {
            x: back_x,
            y: back_y,
            w: 60.0,
            h: 24.0,
        }
        .contains(self.gameplay.mouse_pos);
        ui.quad(
            back_x,
            back_y,
            60.0,
            24.0,
            if back_hovered {
                [60, 60, 80, 255]
            } else {
                [40, 40, 55, 200]
            },
        );
        ui.rect_border(back_x, back_y, 60.0, 24.0, 1.0, [80, 80, 100, 255]);
        ui.text(
            "Back",
            back_x + 12.0,
            back_y + 5.0,
            0.9,
            [200, 200, 200, 255],
            &self.render.font,
        );

        // Settings sections
        let mut sy = py + 44.0;
        let lx = px + 16.0;

        // Track interactive widget rects for click handling.
        let mut sliders: Vec<crate::SliderWidget> = Vec::new();
        let mut toggles: Vec<crate::ToggleWidget> = Vec::new();

        // GRAPHICS section
        sy = self.draw_section_header(ui, "GRAPHICS", lx, sy, panel_w - 32.0);
        let mut slider_idx: usize = 0;
        let (ny, s_rect) = self.draw_setting_slider_with_rect(
            ui,
            "Render Distance",
            self.config.stream.load_radius as f32,
            SliderRange {
                min: 2.0,
                max: 16.0,
            },
            Row {
                x: lx,
                y: sy,
                w: panel_w - 32.0,
            },
            slider_idx,
        );
        slider_idx += 1;
        sy = ny;
        sliders.push(s_rect);
        let (ny, t_rect) = self.draw_setting_toggle_with_rect(
            ui,
            "VSync",
            self.config.render.vsync,
            lx,
            sy,
            panel_w - 32.0,
        );
        sy = ny;
        toggles.push(t_rect);
        let (ny, s_rect) = self.draw_setting_slider_with_rect(
            ui,
            "Fog Distance",
            self.config.render.fog_distance,
            SliderRange {
                min: 100.0,
                max: 800.0,
            },
            Row {
                x: lx,
                y: sy,
                w: panel_w - 32.0,
            },
            slider_idx,
        );
        slider_idx += 1;
        sy = ny;
        sliders.push(s_rect);
        let (ny, s_rect) = self.draw_setting_slider_with_rect(
            ui,
            "Exposure",
            self.config.exposure,
            SliderRange { min: 0.1, max: 3.0 },
            Row {
                x: lx,
                y: sy,
                w: panel_w - 32.0,
            },
            slider_idx,
        );
        slider_idx += 1;
        sy = ny;
        sliders.push(s_rect);
        let (ny, t_rect) = self.draw_setting_toggle_with_rect(
            ui,
            "Shadows",
            self.config.shadow_enabled,
            lx,
            sy,
            panel_w - 32.0,
        );
        sy = ny;
        toggles.push(t_rect);
        let (ny, t_rect) = self.draw_setting_toggle_with_rect(
            ui,
            "Vignette",
            self.config.vignette_strength > 0.0,
            lx,
            sy,
            panel_w - 32.0,
        );
        sy = ny;
        toggles.push(t_rect);
        let (ny, t_rect) = self.draw_setting_toggle_with_rect(
            ui,
            "SSAO",
            self.config.ssao_enabled,
            lx,
            sy,
            panel_w - 32.0,
        );
        sy = ny;
        toggles.push(t_rect);
        let (ny, s_rect) = self.draw_setting_slider_with_rect(
            ui,
            "SSAO Radius",
            self.config.ssao_radius,
            SliderRange { min: 0.5, max: 5.0 },
            Row {
                x: lx,
                y: sy,
                w: panel_w - 32.0,
            },
            slider_idx,
        );
        slider_idx += 1;
        sy = ny;
        sliders.push(s_rect);
        let (ny, s_rect) = self.draw_setting_slider_with_rect(
            ui,
            "SSAO Bias",
            self.config.ssao_bias,
            SliderRange {
                min: 0.001,
                max: 0.1,
            },
            Row {
                x: lx,
                y: sy,
                w: panel_w - 32.0,
            },
            slider_idx,
        );
        slider_idx += 1;
        sy = ny;
        sliders.push(s_rect);
        let (ny, s_rect) = self.draw_setting_slider_with_rect(
            ui,
            "SSAO Strength",
            self.config.ssao_strength,
            SliderRange { min: 0.0, max: 3.0 },
            Row {
                x: lx,
                y: sy,
                w: panel_w - 32.0,
            },
            slider_idx,
        );
        slider_idx += 1;
        sy = ny;
        sliders.push(s_rect);

        // MSAA Samples: discrete slider mapping 0-3 to [1, 2, 4, 8]
        let msaa_val = match self.config.render.msaa_samples {
            8 => 3.0,
            4 => 2.0,
            2 => 1.0,
            _ => 0.0,
        };
        let (ny, s_rect) = self.draw_setting_slider_with_rect(
            ui,
            "MSAA Samples",
            msaa_val,
            SliderRange { min: 0.0, max: 3.0 },
            Row {
                x: lx,
                y: sy,
                w: panel_w - 32.0,
            },
            slider_idx,
        );
        slider_idx += 1;
        sy = ny;
        sliders.push(s_rect);

        sy += 8.0;

        // PLAYER section
        sy = self.draw_section_header(ui, "PLAYER", lx, sy, panel_w - 32.0);
        let (ny, s_rect) = self.draw_setting_slider_with_rect(
            ui,
            "Mouse Sensitivity",
            self.config.player.mouse_sensitivity * 1000.0,
            SliderRange {
                min: 0.5,
                max: 10.0,
            },
            Row {
                x: lx,
                y: sy,
                w: panel_w - 32.0,
            },
            slider_idx,
        );
        slider_idx += 1;
        sy = ny;
        sliders.push(s_rect);
        let (ny, s_rect) = self.draw_setting_slider_with_rect(
            ui,
            "Walk Speed",
            self.config.player.walk_speed,
            SliderRange {
                min: 1.0,
                max: 10.0,
            },
            Row {
                x: lx,
                y: sy,
                w: panel_w - 32.0,
            },
            slider_idx,
        );
        slider_idx += 1;
        sy = ny;
        sliders.push(s_rect);
        let (ny, s_rect) = self.draw_setting_slider_with_rect(
            ui,
            "Fly Speed",
            self.config.player.fly_speed,
            SliderRange {
                min: 5.0,
                max: 50.0,
            },
            Row {
                x: lx,
                y: sy,
                w: panel_w - 32.0,
            },
            slider_idx,
        );
        sy = ny;
        sliders.push(s_rect);

        sy += 8.0;

        // CONTROLS section
        sy = self.draw_section_header(ui, "CONTROLS", lx, sy, panel_w - 32.0);
        let keybinds = [
            ("Chat", &self.config.keybinds.chat),
            ("Fly", &self.config.keybinds.fly),
            ("Pause", &self.config.keybinds.pause),
            ("Block Picker", &self.config.keybinds.block_picker),
            ("Edit Mode", &self.config.keybinds.edit_mode),
        ];
        for (label, key) in keybinds.iter() {
            sy = self.draw_keybind_row(ui, label, key, lx, sy);
        }

        // Apply / Defaults buttons
        let btn_y = py + panel_h - 40.0;
        let btn_w = 100.0;
        let apply_x = px + panel_w - btn_w * 2.0 - 20.0;
        let defaults_x = px + panel_w - btn_w - 8.0;
        let apply_hovered = Rect {
            x: apply_x,
            y: btn_y,
            w: btn_w,
            h: 28.0,
        }
        .contains(self.gameplay.mouse_pos);
        let defaults_hovered = Rect {
            x: defaults_x,
            y: btn_y,
            w: btn_w,
            h: 28.0,
        }
        .contains(self.gameplay.mouse_pos);
        ui.quad(
            apply_x,
            btn_y,
            btn_w,
            28.0,
            if apply_hovered {
                [50, 80, 50, 255]
            } else {
                [35, 55, 35, 220]
            },
        );
        ui.rect_border(apply_x, btn_y, btn_w, 28.0, 1.0, [80, 140, 80, 255]);
        ui.text(
            "Apply",
            apply_x + 24.0,
            btn_y + 6.0,
            0.8,
            [200, 255, 200, 255],
            &self.render.font,
        );
        ui.quad(
            defaults_x,
            btn_y,
            btn_w,
            28.0,
            if defaults_hovered {
                [60, 60, 60, 255]
            } else {
                [40, 40, 40, 220]
            },
        );
        ui.rect_border(defaults_x, btn_y, btn_w, 28.0, 1.0, [80, 80, 80, 255]);
        ui.text(
            "Defaults",
            defaults_x + 16.0,
            btn_y + 6.0,
            0.8,
            [200, 200, 200, 255],
            &self.render.font,
        );

        // Store rects for click handling.
        self.gameplay.settings_back_btn = Some(Rect {
            x: back_x,
            y: back_y,
            w: 60.0,
            h: 24.0,
        });
        self.gameplay.settings_widgets = Some(crate::SettingsWidgets {
            sliders,
            toggles,
            apply_btn: Rect {
                x: apply_x,
                y: btn_y,
                w: btn_w,
                h: 28.0,
            },
            defaults_btn: Rect {
                x: defaults_x,
                y: btn_y,
                w: btn_w,
                h: 28.0,
            },
        });
    }

    pub(crate) fn handle_settings_click(&mut self) {
        // Back button
        if let Some(back_rect) = self.gameplay.settings_back_btn {
            if back_rect.contains(self.gameplay.mouse_pos) {
                match self.gameplay.settings_previous {
                    GameState::TitleScreen | GameState::WorldSelect => self.enter_title_screen(),
                    GameState::PauseMenu => self.enter_pause(),
                    _ => self.enter_title_screen(),
                }
                self.gameplay.settings_widgets = None;
                return;
            }
        }

        let Some(ref widgets) = self.gameplay.settings_widgets.clone() else {
            return;
        };

        // Toggle clicks.
        for toggle in &widgets.toggles {
            if toggle.rect.contains(self.gameplay.mouse_pos) {
                match toggle.label.as_str() {
                    "VSync" => self.config.render.vsync = !self.config.render.vsync,
                    "Shadows" => self.config.shadow_enabled = !self.config.shadow_enabled,
                    "Vignette" => {
                        if self.config.vignette_strength > 0.0 {
                            self.config.vignette_strength = 0.0;
                        } else {
                            self.config.vignette_strength = 0.15;
                        }
                    }
                    "SSAO" => self.config.ssao_enabled = !self.config.ssao_enabled,
                    _ => {}
                }
                return;
            }
        }

        // Slider clicks -- set value based on click position within the bar.
        // Also start dragging so the user can hold and drag.
        for (i, slider) in widgets.sliders.iter().enumerate() {
            if slider.rect.contains(self.gameplay.mouse_pos) {
                let new_val = slider.value_at(self.gameplay.mouse_pos);
                self.apply_slider_value(&slider.label, new_val);
                self.gameplay.settings_slider_dragging = Some(i);
                return;
            }
        }

        // Apply button: save current settings to config.toml.
        if widgets.apply_btn.contains(self.gameplay.mouse_pos) {
            let path = self.config.config_path.clone();
            let gs = self.current_game_settings();
            match gs.save(&path) {
                Ok(()) => self.gameplay.chat.push_message("[config] saved".into()),
                Err(e) => {
                    log::warn!("save config: {e}");
                    self.gameplay
                        .chat
                        .push_message(format!("[config] save failed: {e}"));
                }
            }
            return;
        }

        // Defaults button: reset all settings to defaults and save.
        if widgets.defaults_btn.contains(self.gameplay.mouse_pos) {
            let defaults = crate::settings::GameSettings::default();
            let new_rc = defaults.to_renderer_config();
            if let Some(r) = self.render.renderer.as_mut() {
                let _ = r.reload_config(&new_rc);
            }
            self.config.render = new_rc;
            self.config.stream = defaults.to_stream_config();
            self.config.player = defaults.to_player_config();
            self.config.keybinds = defaults.keys.clone();
            self.config.world = defaults.world.clone();
            self.config.shadow_enabled = defaults.graphics.shadow_enabled;
            self.config.shadow_resolution = defaults.graphics.shadow_resolution;
            self.config.exposure = defaults.graphics.exposure;
            self.config.vignette_strength = defaults.graphics.vignette_strength;
            self.config.ssao_enabled = defaults.graphics.ssao_enabled;
            self.config.ssao_radius = defaults.graphics.ssao_radius;
            self.config.ssao_bias = defaults.graphics.ssao_bias;
            self.config.ssao_strength = defaults.graphics.ssao_strength;
            let path = self.config.config_path.clone();
            match defaults.save(&path) {
                Ok(()) => self
                    .gameplay
                    .chat
                    .push_message("[config] reset to defaults".into()),
                Err(e) => {
                    log::warn!("save defaults: {e}");
                    self.gameplay
                        .chat
                        .push_message(format!("[config] reset failed: {e}"));
                }
            }
        }
    }

    /// Build a `GameSettings` snapshot from the live engine config.
    fn current_game_settings(&self) -> crate::settings::GameSettings {
        crate::settings::GameSettings {
            graphics: crate::settings::GraphicsSettings {
                width: self.render.window_size.0,
                height: self.render.window_size.1,
                vsync: self.config.render.vsync,
                fog_distance: self.config.render.fog_distance,
                shadow_enabled: self.config.shadow_enabled,
                shadow_resolution: self.config.shadow_resolution,
                exposure: self.config.exposure,
                vignette_strength: self.config.vignette_strength,
                textures_dir: self
                    .config
                    .render
                    .textures_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                texture_packs_dir: self
                    .config
                    .render
                    .texture_packs_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                msaa_samples: self.config.render.msaa_samples,
                occlusion_culling: self.config.render.occlusion_culling,
                ssao_enabled: self.config.ssao_enabled,
                gpu_driven: self.config.render.gpu_driven,
                ssao_radius: self.config.ssao_radius,
                ssao_bias: self.config.ssao_bias,
                ssao_strength: self.config.ssao_strength,
                water_y: self.config.water_y,
                wet_edge_strength: self.config.wet_edge_strength,
                caustics_strength: self.config.caustics_strength,
                leaves_sss_strength: self.config.leaves_sss_strength,
                reflection_strength: self.config.reflection_strength,
            },
            world: self.config.world.clone(),
            player: crate::settings::PlayerSettings {
                walk_speed: self.config.player.walk_speed,
                sprint_speed: self.config.player.sprint_speed,
                sneak_speed: self.config.player.sneak_speed,
                jump_speed: self.config.player.jump_speed,
                gravity: self.config.player.gravity,
                terminal_velocity: self.config.player.terminal_velocity,
                mouse_sensitivity: self.config.player.mouse_sensitivity,
                fly_speed: self.config.player.fly_speed,
            },
            keys: self.config.keybinds.clone(),
            debug: crate::settings::DebugSettings {
                show_overlay: self.gameplay.debug_overlay,
            },
        }
    }

    /// Apply a slider value to the corresponding config field.
    fn apply_slider_value(&mut self, label: &str, new_val: f32) {
        match label {
            "Render Distance" => self.config.stream.load_radius = new_val.round() as i32,
            "Fog Distance" => self.config.render.fog_distance = new_val,
            "Exposure" => self.config.exposure = new_val,
            "Mouse Sensitivity" => self.config.player.mouse_sensitivity = new_val / 1000.0,
            "Walk Speed" => self.config.player.walk_speed = new_val,
            "Fly Speed" => self.config.player.fly_speed = new_val,
            "MSAA Samples" => self.config.render.msaa_samples = 1 << new_val.round() as u32,
            "SSAO Radius" => self.config.ssao_radius = new_val,
            "SSAO Bias" => self.config.ssao_bias = new_val,
            "SSAO Strength" => self.config.ssao_strength = new_val,
            _ => {}
        }
    }

    /// Update the dragged slider value each frame while the mouse is held.
    pub(crate) fn update_settings_slider_drag(&mut self) {
        if !self.gameplay.settings_left_mouse_held {
            return;
        }
        let Some(drag_idx) = self.gameplay.settings_slider_dragging else {
            return;
        };
        let Some(ref widgets) = self.gameplay.settings_widgets else {
            self.gameplay.settings_slider_dragging = None;
            return;
        };
        let Some(slider) = widgets.sliders.get(drag_idx).cloned() else {
            return;
        };
        let new_val = slider.value_at(self.gameplay.mouse_pos);
        self.apply_slider_value(&slider.label, new_val);
    }

    fn draw_section_header(&self, ui: &mut UiDrawData, title: &str, x: f32, y: f32, w: f32) -> f32 {
        ui.text(title, x, y, 0.8, [120, 160, 220, 255], &self.render.font);
        ui.quad(x, y + 14.0, w, 1.0, [50, 50, 60, 255]);
        y + 20.0
    }

    fn draw_setting_slider(
        &self,
        ui: &mut UiDrawData,
        label: &str,
        value: f32,
        range: SliderRange,
        rect: Row,
        is_dragging: bool,
    ) -> f32 {
        let SliderRange { min, max } = range;
        let Row { x, y, w } = rect;
        let bar_x = x + w * 0.55;
        let bar_w = w * 0.35;
        let bar_h = 8.0;
        let bar_y = y + 4.0;

        // Detect hover: mouse over the bar area (with some vertical padding).
        let Point { x: mx, y: my } = self.gameplay.mouse_pos;
        let is_hovered = !is_dragging
            && mx >= bar_x
            && mx <= bar_x + bar_w
            && my >= bar_y - 6.0
            && my <= bar_y + bar_h + 6.0;

        // Label: brighter when hovered or dragging.
        let label_color = if is_dragging {
            [220, 230, 255, 255]
        } else if is_hovered {
            [200, 210, 240, 255]
        } else {
            [180, 180, 180, 255]
        };
        ui.text(label, x, y + 2.0, 0.8, label_color, &self.render.font);

        let pct = ((value - min) / (max - min)).clamp(0.0, 1.0);

        // Track background: slightly brighter when hovered.
        let track_color = if is_hovered || is_dragging {
            [50, 50, 65, 255]
        } else {
            [40, 40, 50, 255]
        };
        ui.quad(bar_x, bar_y, bar_w, bar_h, track_color);

        // Fill: brighter when hovered, even brighter when dragging.
        let fill_color = if is_dragging {
            [100, 160, 255, 240]
        } else if is_hovered {
            [85, 140, 230, 220]
        } else {
            [74, 127, 212, 200]
        };
        ui.quad(bar_x, bar_y, bar_w * pct, bar_h, fill_color);

        // Drag handle: small bright circle at the current value position.
        if is_dragging || is_hovered {
            let handle_x = bar_x + bar_w * pct - 5.0;
            let handle_y = bar_y - 2.0;
            let handle_size = 12.0;
            let handle_color = if is_dragging {
                [140, 190, 255, 255]
            } else {
                [100, 150, 230, 200]
            };
            ui.quad(handle_x, handle_y, handle_size, handle_size, handle_color);
        }

        let val_text = if label == "MSAA Samples" {
            format!("{}x", 1u32 << value.round() as u32)
        } else if label == "SSAO Bias" {
            format!("{:.3}", value)
        } else {
            format!("{:.1}", value)
        };
        // Value text: brighter when active.
        let val_color = if is_dragging {
            [255, 255, 255, 255]
        } else if is_hovered {
            [230, 230, 240, 255]
        } else {
            [200, 200, 200, 255]
        };
        ui.text(
            &val_text,
            bar_x + bar_w + 6.0,
            bar_y - 2.0,
            0.7,
            val_color,
            &self.render.font,
        );

        y + 22.0
    }

    fn draw_setting_toggle(
        &self,
        ui: &mut UiDrawData,
        label: &str,
        on: bool,
        x: f32,
        y: f32,
        _w: f32,
    ) -> f32 {
        ui.text(
            label,
            x,
            y + 2.0,
            0.8,
            [180, 180, 180, 255],
            &self.render.font,
        );

        let toggle_x = x + 280.0;
        let toggle_w = 36.0;
        let toggle_h = 16.0;
        let toggle_y = y + 2.0;

        ui.quad(toggle_x, toggle_y, toggle_w, toggle_h, [40, 40, 50, 255]);
        ui.rect_border(
            toggle_x,
            toggle_y,
            toggle_w,
            toggle_h,
            1.0,
            [60, 60, 70, 255],
        );

        if on {
            ui.quad(
                toggle_x + toggle_w - 16.0,
                toggle_y,
                16.0,
                toggle_h,
                [74, 127, 212, 255],
            );
            ui.text(
                "ON",
                toggle_x + 4.0,
                toggle_y + 1.0,
                0.6,
                [200, 200, 200, 255],
                &self.render.font,
            );
        } else {
            ui.quad(toggle_x, toggle_y, 16.0, toggle_h, [60, 60, 60, 255]);
            ui.text(
                "OFF",
                toggle_x + 18.0,
                toggle_y + 1.0,
                0.6,
                [120, 120, 120, 255],
                &self.render.font,
            );
        }

        y + 22.0
    }

    fn draw_setting_slider_with_rect(
        &mut self,
        ui: &mut UiDrawData,
        label: &str,
        value: f32,
        range: SliderRange,
        rect: Row,
        slider_index: usize,
    ) -> (f32, crate::SliderWidget) {
        let SliderRange { min, max } = range;
        let Row { x, y, w } = rect;
        let is_dragging = self.gameplay.settings_slider_dragging == Some(slider_index);
        let next_y = self.draw_setting_slider(ui, label, value, range, rect, is_dragging);
        let bar_x = x + w * 0.55;
        let bar_w = w * 0.35;
        let bar_y = y + 4.0;
        let bar_h = 8.0;
        (
            next_y,
            crate::SliderWidget {
                rect: Rect {
                    x: bar_x,
                    y: bar_y,
                    w: bar_w,
                    h: bar_h,
                },
                label: label.to_string(),
                range: SliderRange { min, max },
            },
        )
    }

    fn draw_setting_toggle_with_rect(
        &self,
        ui: &mut UiDrawData,
        label: &str,
        on: bool,
        x: f32,
        y: f32,
        w: f32,
    ) -> (f32, crate::ToggleWidget) {
        let next_y = self.draw_setting_toggle(ui, label, on, x, y, w);
        let toggle_x = x + 280.0;
        let toggle_w = 36.0;
        let toggle_h = 16.0;
        let toggle_y = y + 2.0;
        (
            next_y,
            crate::ToggleWidget {
                rect: Rect {
                    x: toggle_x,
                    y: toggle_y,
                    w: toggle_w,
                    h: toggle_h,
                },
                label: label.to_string(),
            },
        )
    }

    fn draw_keybind_row(&self, ui: &mut UiDrawData, label: &str, key: &str, x: f32, y: f32) -> f32 {
        ui.text(
            label,
            x,
            y + 2.0,
            0.8,
            [180, 180, 180, 255],
            &self.render.font,
        );

        let key_x = x + 200.0;
        let key_w = 80.0;
        let key_h = 18.0;
        let key_y = y + 1.0;

        ui.quad(key_x, key_y, key_w, key_h, [40, 40, 50, 255]);
        ui.rect_border(key_x, key_y, key_w, key_h, 1.0, [60, 60, 70, 255]);
        ui.text(
            key,
            key_x + 8.0,
            key_y + 2.0,
            0.7,
            [200, 200, 200, 255],
            &self.render.font,
        );

        // Rebind button
        let rb_x = key_x + key_w + 6.0;
        let rb_hovered = Rect {
            x: rb_x,
            y: key_y,
            w: 48.0,
            h: key_h,
        }
        .contains(self.gameplay.mouse_pos);
        ui.quad(
            rb_x,
            key_y,
            48.0,
            key_h,
            if rb_hovered {
                [60, 60, 80, 255]
            } else {
                [40, 40, 55, 200]
            },
        );
        ui.rect_border(rb_x, key_y, 48.0, key_h, 1.0, [70, 70, 90, 255]);
        ui.text(
            "rebind",
            rb_x + 4.0,
            key_y + 2.0,
            0.6,
            [180, 180, 200, 255],
            &self.render.font,
        );

        y + 22.0
    }

    /// Draw the creative inventory overlay (Builder's Catalog).
    fn draw_block_picker(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        // ── Design tokens (matching the HTML mockup) ──
        let ink = [28, 27, 32, 255]; // #1c1b20
        let slate = [74, 73, 82, 255]; // #4a4952
        let slate_deep = [55, 54, 61, 255]; // #37363d
        let slate_light = [87, 86, 95, 255]; // #57565f
        let bone = [236, 233, 226, 255]; // #ece9e2
        let dust = [165, 162, 173, 255]; // #a5a2ad
        let ember = [224, 168, 62, 255]; // #e0a83e
        let ember_soft = [224, 168, 62, 76]; // rgba(224,168,62,.30)
        let slot_bg = [41, 40, 46, 255]; // #29282e

        let slot_size = 40.0f32;
        let slot_gap = 3.0f32;
        let cols = 9usize;
        let panel_pad = 12.0f32;

        // ── Tab definitions ──
        let tabs = [
            "Blocks",
            "Nature",
            "Building",
            "Ores",
            "Decoration",
            "Liquids",
            "All",
            "Search",
        ];
        let tab_count = tabs.len();

        // ── Filter items by active tab ──
        let active_tab = self.gameplay.creative_tab;
        let is_search = active_tab == tab_count - 1; // last tab is search
        let filtered_items: Vec<&crate::CreativeItem> = if is_search {
            let q = self.gameplay.creative_search.to_ascii_lowercase();
            if q.is_empty() {
                self.gameplay.creative_items.iter().collect()
            } else {
                self.gameplay
                    .creative_items
                    .iter()
                    .filter(|it| it.name.to_ascii_lowercase().contains(&q))
                    .collect()
            }
        } else if active_tab == tab_count - 2 {
            // "All" tab
            self.gameplay.creative_items.iter().collect()
        } else {
            let cat = tabs[active_tab];
            self.gameplay
                .creative_items
                .iter()
                .filter(|it| it.category == cat)
                .collect()
        };

        let item_count = filtered_items.len();
        let rows = item_count.div_ceil(cols);
        let visible_rows = rows.min(5); // Show max 5 rows, scroll for more

        // ── Panel dimensions ──
        let grid_w = cols as f32 * (slot_size + slot_gap) - slot_gap;
        let panel_w = grid_w + panel_pad * 2.0;
        let tab_height = 32.0f32;
        let titlebar_h = 28.0f32;
        let search_h = 28.0f32;
        let grid_h = visible_rows as f32 * (slot_size + slot_gap) - slot_gap;
        let hotbar_h = slot_size;
        let divider_h = 2.0f32;

        let panel_h = panel_pad
            + titlebar_h
            + 6.0
            + if is_search { search_h + 6.0 } else { 0.0 }
            + grid_h
            + panel_pad
            + divider_h
            + 6.0
            + hotbar_h
            + panel_pad;

        let panel_x = (w - panel_w) * 0.5;
        let panel_y = (h - panel_h) * 0.5 - 10.0;

        // ── Dim background ──
        ui.quad(0.0, 0.0, w, h, [0, 0, 0, 160]);

        // ── Tab row (above panel) ──
        let tab_w = 40.0f32;
        let tab_gap = 2.0f32;
        let tab_row_w = tab_count as f32 * tab_w + (tab_count - 1) as f32 * tab_gap;
        let tab_x0 = panel_x + (panel_w - tab_row_w) * 0.5;
        let tab_y = panel_y - tab_height;

        for (i, _label) in tabs.iter().enumerate() {
            let tx = tab_x0 + i as f32 * (tab_w + tab_gap);
            let is_active = i == active_tab;
            let t_h = if is_active {
                tab_height + 6.0
            } else {
                tab_height
            };
            let ty = if is_active { tab_y - 6.0 } else { tab_y };

            // Tab background
            let bg = if is_active { slate } else { slate_deep };
            let alpha = if is_active { 255 } else { 170 };
            ui.quad(tx, ty, tab_w, t_h, [bg[0], bg[1], bg[2], alpha]);

            // Tab icon (simple shape)
            let icon_size = 16.0f32;
            let icon_x = tx + (tab_w - icon_size) * 0.5;
            let icon_y = ty + (t_h - icon_size) * 0.5 + 2.0;
            let icon_color = if is_active { bone } else { dust };
            // Draw simple geometric icon based on tab
            match i {
                0 => {
                    // Blocks - cube
                    ui.quad(
                        icon_x + 2.0,
                        icon_y + 2.0,
                        icon_size - 4.0,
                        icon_size - 4.0,
                        icon_color,
                    );
                }
                1 => {
                    // Nature - circle-ish
                    ui.quad(
                        icon_x + 3.0,
                        icon_y + 1.0,
                        icon_size - 6.0,
                        icon_size - 2.0,
                        icon_color,
                    );
                }
                2 => {
                    // Building - brick
                    ui.quad(icon_x + 1.0, icon_y + 2.0, icon_size - 2.0, 5.0, icon_color);
                    ui.quad(icon_x + 1.0, icon_y + 9.0, icon_size - 2.0, 5.0, icon_color);
                }
                3 => {
                    // Ores - diamond
                    ui.quad(icon_x + 4.0, icon_y + 1.0, 8.0, 14.0, icon_color);
                }
                4 => {
                    // Decoration - lamp
                    ui.quad(icon_x + 5.0, icon_y + 1.0, 6.0, 6.0, icon_color);
                    ui.quad(icon_x + 4.0, icon_y + 8.0, 8.0, 6.0, icon_color);
                }
                5 => {
                    // Liquids - wave
                    ui.quad(icon_x + 1.0, icon_y + 5.0, 14.0, 3.0, icon_color);
                    ui.quad(icon_x + 3.0, icon_y + 9.0, 10.0, 3.0, icon_color);
                }
                6 => {
                    // All - grid
                    ui.quad(icon_x + 1.0, icon_y + 1.0, 6.0, 6.0, icon_color);
                    ui.quad(icon_x + 9.0, icon_y + 1.0, 6.0, 6.0, icon_color);
                    ui.quad(icon_x + 1.0, icon_y + 9.0, 6.0, 6.0, icon_color);
                    ui.quad(icon_x + 9.0, icon_y + 9.0, 6.0, 6.0, icon_color);
                }
                7 => {
                    // Search - magnifier
                    ui.rect_border(icon_x + 1.0, icon_y + 1.0, 10.0, 10.0, 2.0, icon_color);
                    ui.quad(icon_x + 10.0, icon_y + 11.0, 5.0, 2.0, icon_color);
                }
                _ => {}
            }
        }

        // ── Panel background ──
        // Gradient approximation: use the mid-slate color
        ui.quad(panel_x, panel_y, panel_w, panel_h, slate);
        // Border
        ui.rect_border(panel_x, panel_y, panel_w, panel_h, 3.0, ink);
        // Inner highlight
        ui.rect_border(
            panel_x + 1.0,
            panel_y + 1.0,
            panel_w - 2.0,
            panel_h - 2.0,
            1.0,
            slate_light,
        );

        let mut cy = panel_y + panel_pad;

        // ── Title bar ──
        let category_label = if is_search {
            "Search"
        } else {
            tabs[active_tab]
        };
        ui.text(
            "Builder's Catalog",
            panel_x + panel_pad,
            cy,
            0.65,
            dust,
            &self.render.font,
        );
        ui.text(
            category_label,
            panel_x + panel_pad + 100.0,
            cy,
            0.85,
            bone,
            &self.render.font,
        );
        // Close button
        let close_x = panel_x + panel_w - panel_pad - 20.0;
        let close_y = cy;
        ui.quad(close_x, close_y, 20.0, 20.0, slot_bg);
        ui.rect_border(close_x, close_y, 20.0, 20.0, 1.0, slate_light);
        ui.text(
            "X",
            close_x + 5.0,
            close_y + 2.0,
            0.9,
            dust,
            &self.render.font,
        );
        cy += titlebar_h + 6.0;

        // ── Search row (only when search tab active) ──
        if is_search {
            ui.quad(panel_x + panel_pad, cy, grid_w, search_h, slot_bg);
            ui.rect_border(panel_x + panel_pad, cy, grid_w, search_h, 1.0, ink);
            let search_label = if self.gameplay.creative_search.is_empty() {
                "Search catalog...".to_string()
            } else {
                self.gameplay.creative_search.clone()
            };
            ui.text(
                &search_label,
                panel_x + panel_pad + 6.0,
                cy + 6.0,
                0.85,
                dust,
                &self.render.font,
            );
            cy += search_h + 6.0;
        }

        // ── Item grid ──
        let grid_x0 = panel_x + panel_pad;
        let grid_y0 = cy;
        let mut hovered_item_name: Option<&str> = None;
        let mut hovered_item_cat: Option<&str> = None;
        let mut hovered_slot_index: Option<usize> = None;

        // Calculate scroll bounds.
        let total_rows = filtered_items.len().div_ceil(cols);
        let max_scroll = total_rows.saturating_sub(visible_rows);
        let scroll_offset = self.gameplay.creative_scroll.min(max_scroll);

        // Map an item index to its on-screen slot position. Shared by the
        // hover loop and the tooltip so the layout math can't drift.
        let slot_screen_pos = |i: usize| {
            let col = i % cols;
            let row = i / cols;
            let display_row = row - scroll_offset;
            Point::new(
                grid_x0 + col as f32 * (slot_size + slot_gap),
                grid_y0 + display_row as f32 * (slot_size + slot_gap),
            )
        };

        for (i, item) in filtered_items.iter().enumerate() {
            let row = i / cols;
            // Apply scroll offset.
            if row < scroll_offset {
                continue;
            }
            let display_row = row - scroll_offset;
            if display_row >= visible_rows {
                break;
            }
            let Point { x: sx, y: sy } = slot_screen_pos(i);

            // Check if mouse is hovering over this slot.
            let is_hovered = self.gameplay.mouse_pos.x >= sx
                && self.gameplay.mouse_pos.x < sx + slot_size
                && self.gameplay.mouse_pos.y >= sy
                && self.gameplay.mouse_pos.y < sy + slot_size;

            // Slot background
            if is_hovered {
                // Hover highlight (ember glow from mockup)
                ui.quad(sx, sy, slot_size, slot_size, ember_soft);
                ui.rect_border(sx, sy, slot_size, slot_size, 2.0, ember);
                hovered_item_name = Some(&item.name);
                hovered_item_cat = Some(&item.category);
                hovered_slot_index = Some(i);
            } else {
                ui.quad(sx, sy, slot_size, slot_size, slot_bg);
                // 3D inset effect
                ui.quad(sx, sy, slot_size, 2.0, ink); // top shadow
                ui.quad(sx, sy, 2.0, slot_size, ink); // left shadow
                ui.quad(sx + slot_size - 2.0, sy, 2.0, slot_size, slate_light); // right highlight
                ui.quad(sx, sy + slot_size - 2.0, slot_size, 2.0, slate_light); // bottom highlight
            }

            // Block icon
            ui.block_icon(
                sx + 2.0,
                sy + 2.0,
                slot_size - 4.0,
                slot_size - 4.0,
                item.tile,
                [255, 255, 255, 255],
            );
        }

        // ── Tooltip (shown when hovering over an item) ──
        if let (Some(name), Some(cat), Some(slot)) =
            (hovered_item_name, hovered_item_cat, hovered_slot_index)
        {
            let tooltip_pad = 8.0f32;
            let tooltip_w = 140.0f32;
            let tooltip_h = 36.0f32;
            // Position tooltip near the hovered slot, but not off-screen.
            let Point { x: sx, y: sy } = slot_screen_pos(slot);
            let mut tooltip_x = sx + slot_size + 6.0;
            let mut tooltip_y = sy - 4.0;
            if tooltip_x + tooltip_w > w - 10.0 {
                tooltip_x = sx - tooltip_w - 6.0;
            }
            if tooltip_y + tooltip_h > h - 10.0 {
                tooltip_y = h - tooltip_h - 10.0;
            }
            if tooltip_y < 10.0 {
                tooltip_y = 10.0;
            }
            // Tooltip background
            ui.quad(tooltip_x, tooltip_y, tooltip_w, tooltip_h, ink);
            ui.rect_border(tooltip_x, tooltip_y, tooltip_w, tooltip_h, 1.0, slate_light);
            // Item name (ember color, like mockup)
            ui.text(
                name,
                tooltip_x + tooltip_pad,
                tooltip_y + 4.0,
                0.85,
                ember,
                &self.render.font,
            );
            // Category (dust color)
            ui.text(
                cat,
                tooltip_x + tooltip_pad,
                tooltip_y + 18.0,
                0.65,
                dust,
                &self.render.font,
            );
        }
        cy += grid_h + panel_pad;

        // ── Divider ──
        ui.quad(panel_x + panel_pad, cy, grid_w, divider_h, ink);
        ui.quad(panel_x + panel_pad, cy + 1.0, grid_w, 1.0, slate_light);
        cy += divider_h + 6.0;

        // ── Hotbar (bottom section) ──
        let hotbar_x0 = panel_x + panel_pad;
        for i in 0..9 {
            let sx = hotbar_x0 + i as f32 * (slot_size + slot_gap);
            ui.quad(sx, cy, slot_size, slot_size, slot_bg);
            ui.quad(sx, cy, slot_size, 2.0, ink);
            ui.quad(sx, cy, 2.0, slot_size, ink);
            ui.quad(sx + slot_size - 2.0, cy, 2.0, slot_size, slate_light);
            ui.quad(sx, cy + slot_size - 2.0, slot_size, 2.0, slate_light);

            // Highlight selected hotbar slot
            if i == self.gameplay.hotbar.selected {
                ui.rect_border(
                    sx - 1.0,
                    cy - 1.0,
                    slot_size + 2.0,
                    slot_size + 2.0,
                    2.0,
                    ember,
                );
            }

            // Show hotbar item if any
            let block_id = self
                .gameplay
                .hotbar
                .slot(i)
                .unwrap_or(voxel_core::BlockId::AIR);
            if !block_id.is_air() {
                // Try to find the item in our creative cache to get its tile
                if let Some(cached) = self
                    .gameplay
                    .creative_items
                    .iter()
                    .find(|it| it.id == block_id)
                {
                    ui.block_icon(
                        sx + 2.0,
                        cy + 2.0,
                        slot_size - 4.0,
                        slot_size - 4.0,
                        cached.tile,
                        [255, 255, 255, 255],
                    );
                }
            }

            // Slot number
            let num = format!("{}", i + 1);
            ui.text(&num, sx + 1.0, cy + 1.0, 0.6, dust, &self.render.font);
        }
    }

    /// Draw the chat overlay.
    fn draw_chat(&self, ui: &mut UiDrawData, _w: f32, h: f32) {
        let line_h = 16.0;
        let max_visible = 10;
        let pad = 8.0;

        let visible_messages = self.gameplay.chat.messages.len().min(max_visible);
        let input_line = if self.gameplay.chat.open { 1 } else { 0 };
        let total_lines = visible_messages + input_line;

        if total_lines == 0 {
            return;
        }

        let box_h = total_lines as f32 * line_h + pad * 2.0;
        let box_w = 400.0;
        let box_x = pad;
        let box_y = h - box_h - 60.0;

        ui.quad(box_x, box_y, box_w, box_h, [0, 0, 0, 160]);

        let mut y = box_y + pad;
        for i in (0..visible_messages).rev() {
            if let Some(msg) = self.gameplay.chat.messages.get(i) {
                ui.text(
                    msg,
                    box_x + pad,
                    y,
                    1.0,
                    [200, 200, 200, 255],
                    &self.render.font,
                );
                y += line_h;
            }
        }

        if self.gameplay.chat.open {
            let input_text = format!("> {}", self.gameplay.chat.input_buf);
            ui.text(
                &input_text,
                box_x + pad,
                y,
                1.0,
                [255, 255, 100, 255],
                &self.render.font,
            );
        }
    }

    /// Draw the developer console overlay.
    fn draw_console(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let panel_h = h * 0.4;
        let panel_y = h - panel_h;

        ui.quad(0.0, panel_y, w, panel_h, [0, 0, 0, 200]);

        let line_h = 16.0;
        let pad = 8.0;

        let prompt_y = panel_y + panel_h - line_h - pad;
        let input_text = self.gameplay.console.current_line_text();
        let cursor_pos = self.gameplay.console.cursor_pos();

        let before: String = input_text.chars().take(cursor_pos).collect();
        let after: String = input_text.chars().skip(cursor_pos).collect();
        let cursor_char = if self.gameplay.console.cursor_visible() {
            "\u{2588}"
        } else {
            " "
        };
        let display = format!("> {}{}{}", before, cursor_char, after);
        ui.text(
            &display,
            pad,
            prompt_y,
            1.0,
            [100, 255, 100, 255],
            &self.render.font,
        );

        let max_visible = ((panel_h - line_h - pad * 3.0) / line_h) as usize;
        let scrollback = self.gameplay.console.visible_lines(max_visible);
        let mut y = prompt_y - line_h;
        for line in scrollback.iter().rev() {
            ui.text(line, pad, y, 1.0, [200, 200, 200, 230], &self.render.font);
            y -= line_h;
            if y < panel_y {
                break;
            }
        }
    }

    /// Draw the telemetry dashboard overlay.
    fn draw_telemetry_dashboard(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        let collector = &self.telemetry;
        let panel_w = 420.0f32;
        let panel_x = w - panel_w;
        let panel_y = 0.0f32;
        let panel_h = h;

        ui.quad(panel_x, panel_y, panel_w, panel_h, [15, 15, 20, 220]);
        ui.rect_border(panel_x, panel_y, panel_w, panel_h, 1.0, [60, 60, 80, 255]);

        let pad = 8.0f32;
        let font = &self.render.font;

        let last = collector.last();
        let fps = last
            .map(|s| {
                if s.cpu_frame_ms > 0.0 {
                    1000.0 / s.cpu_frame_ms
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        let cpu_ms = last.map(|s| s.cpu_frame_ms).unwrap_or(0.0);
        let gpu_ms = last.map(|s| s.gpu_frame_ms).unwrap_or(0.0);
        let rss_mb = last.map(|s| s.process_rss_mb).unwrap_or(0.0);

        let card_y = panel_y + pad;
        let card_w = 95.0;
        let card_gap = 4.0;
        let cards = [
            ("FPS", format!("{:.0}", fps), [80, 200, 80, 255]),
            ("CPU", format!("{:.1}ms", cpu_ms), [200, 200, 80, 255]),
            ("GPU", format!("{:.1}ms", gpu_ms), [200, 200, 80, 255]),
            ("RAM", format!("{:.0}MB", rss_mb), [80, 180, 220, 255]),
        ];
        for (i, (label, value, color)) in cards.iter().enumerate() {
            let cx = panel_x + pad + i as f32 * (card_w + card_gap);
            ui.quad(cx, card_y, card_w, 40.0, [30, 30, 40, 200]);
            ui.rect_border(cx, card_y, card_w, 40.0, 1.0, [60, 60, 80, 255]);
            ui.text(
                label,
                cx + 4.0,
                card_y + 2.0,
                0.7,
                [160, 160, 180, 255],
                font,
            );
            ui.text(value, cx + 4.0, card_y + 18.0, 1.0, *color, font);
        }

        let graph_x = panel_x + pad;
        let graph_y = card_y + 50.0;
        let graph_w = panel_w - pad * 2.0;
        let graph_h = 100.0;

        ui.text(
            "Frame Time (ms)",
            graph_x,
            graph_y - 12.0,
            0.7,
            [160, 160, 180, 255],
            font,
        );
        ui.grid_h(graph_x, graph_y, graph_w, graph_h, 4, [50, 50, 60, 100]);

        let cpu_samples = collector.extract_f32(crate::telemetry::MetricSelector::CpuFrameMs, 300);
        let gpu_samples = collector.extract_f32(crate::telemetry::MetricSelector::GpuFrameMs, 300);
        ui.area_graph(
            Rect::from_xywh(graph_x, graph_y, graph_w, graph_h),
            &cpu_samples,
            GraphStyle {
                min_y: Some(0.0),
                max_y: Some(50.0),
                color: [80, 200, 80, 60],
            },
        );
        ui.line_graph(
            Rect::from_xywh(graph_x, graph_y, graph_w, graph_h),
            &gpu_samples,
            GraphStyle {
                min_y: Some(0.0),
                max_y: Some(50.0),
                color: [200, 200, 80, 200],
            },
        );

        let stack_y = graph_y + graph_h + 24.0;
        let stack_h = 80.0;
        ui.text(
            "GPU Pass Breakdown",
            graph_x,
            stack_y - 12.0,
            0.7,
            [160, 160, 180, 255],
            font,
        );

        let series_data = [
            collector.extract_f32(crate::telemetry::MetricSelector::GpuShadowMs, 300),
            collector.extract_f32(crate::telemetry::MetricSelector::GpuSkyMs, 300),
            collector.extract_f32(crate::telemetry::MetricSelector::GpuOpaqueMs, 300),
            collector.extract_f32(crate::telemetry::MetricSelector::GpuTransparentMs, 300),
            collector.extract_f32(crate::telemetry::MetricSelector::GpuUiMs, 300),
            collector.extract_f32(crate::telemetry::MetricSelector::GpuPostMs, 300),
        ];
        let series_refs: Vec<&[f32]> = series_data.iter().map(|s| s.as_slice()).collect();
        let colors = [
            [200, 150, 80, 200],
            [150, 200, 80, 200],
            [80, 200, 150, 200],
            [80, 150, 200, 200],
            [150, 80, 200, 200],
            [200, 80, 150, 200],
        ];
        ui.stacked_area(
            Rect::from_xywh(graph_x, stack_y, graph_w, stack_h),
            &series_refs,
            &colors,
        );

        let mem_y = stack_y + stack_h + 24.0;
        let mem_h = 60.0;
        ui.text(
            "GPU Memory (MB)",
            graph_x,
            mem_y - 12.0,
            0.7,
            [160, 160, 180, 255],
            font,
        );
        let alloc_samples =
            collector.extract_f32(crate::telemetry::MetricSelector::GpuAllocatedMb, 300);
        ui.area_graph(
            Rect::from_xywh(graph_x, mem_y, graph_w, mem_h),
            &alloc_samples,
            GraphStyle {
                min_y: Some(0.0),
                max_y: Some(256.0),
                color: [80, 180, 220, 100],
            },
        );

        let chunk_y = mem_y + mem_h + 24.0;
        let chunk_h = 60.0;
        ui.text(
            "Chunks (loaded / GPU)",
            graph_x,
            chunk_y - 12.0,
            0.7,
            [160, 160, 180, 255],
            font,
        );
        let loaded = collector.extract_f32(crate::telemetry::MetricSelector::ChunksLoaded, 300);
        let gpu_chunks = collector.extract_f32(crate::telemetry::MetricSelector::ChunksGpu, 300);
        ui.line_graph(
            Rect::from_xywh(graph_x, chunk_y, graph_w, chunk_h),
            &loaded,
            GraphStyle {
                min_y: None,
                max_y: None,
                color: [80, 200, 80, 200],
            },
        );
        ui.line_graph(
            Rect::from_xywh(graph_x, chunk_y, graph_w, chunk_h),
            &gpu_chunks,
            GraphStyle {
                min_y: None,
                max_y: None,
                color: [200, 200, 80, 200],
            },
        );

        let stats_y = chunk_y + chunk_h + 16.0;
        let mut sy = stats_y;
        let stat_color = [180, 180, 200, 230];
        let line_h = 14.0;

        if let Some(s) = last {
            let stats = [
                format!(
                    "Entities: {}  Archetypes: {}",
                    s.entity_count, s.archetype_count
                ),
                format!(
                    "Chunks: {} loaded, {} meshed, {} GPU",
                    s.chunks_loaded, s.chunks_meshed, s.chunks_gpu
                ),
                format!(
                    "Vertices: {}  Indices: {}",
                    s.chunk_vertices, s.chunk_indices
                ),
                format!(
                    "Streamer: {} gen, {} mesh, {} remesh",
                    s.streamer_gen_queue, s.streamer_mesh_queue, s.streamer_pending_remesh
                ),
                format!(
                    "Gen {:.1}ms  Mesh {:.1}ms",
                    s.streamer_gen_ms, s.streamer_mesh_ms
                ),
                format!(
                    "Water {:.2}ms  Upload {:.2}ms",
                    s.water_tick_ms, s.chunk_upload_ms
                ),
                format!("Water pending: {}", s.water_pending_flow),
                format!(
                    "GPU mem: {:.1} MB alloc / {:.1} MB reserved",
                    s.gpu_allocated_mb, s.gpu_reserved_mb
                ),
            ];
            for line in &stats {
                if sy + line_h > panel_h - 8.0 {
                    break;
                }
                ui.text(line, graph_x, sy, 0.7, stat_color, font);
                sy += line_h;
            }
        }
    }

    /// Draw the F3 debug overlay.
    fn draw_debug_overlay(&mut self, ui: &mut UiDrawData, _w: f32, _h: f32) {
        let x = 8.0;
        let mut y = 8.0;
        let line_h = 16.0;
        let color = [255, 255, 255, 230];

        let panel_w = 320.0;
        let panel_h = 9.0 * line_h + 16.0;
        ui.quad(x - 4.0, y - 4.0, panel_w, panel_h, [0, 0, 0, 150]);

        let pos = self
            .simulation
            .player_pos()
            .unwrap_or(self.gameplay.spawn_pos);
        let flying = self.simulation.player_flying();
        let lines = [
            format!("XYZ: {:.1} / {:.1} / {:.1}", pos.x, pos.y, pos.z),
            format!(
                "Chunk: {} / {} / {}",
                (pos.x as i32) >> 4,
                (pos.y as i32) >> 4,
                (pos.z as i32) >> 4
            ),
            format!(
                "Chunks GPU: {}",
                self.render
                    .renderer
                    .as_ref()
                    .map(|r| r.chunk_count())
                    .unwrap_or(0)
            ),
            format!("Loaded: {}", self.world_state.world.loaded_chunk_count()),
            format!("Meshed: {}", self.world_state.world.meshed_chunk_count()),
            format!(
                "Time: {:.1}s / {:.0}s",
                self.gameplay.game_time, self.gameplay.day_length
            ),
            format!("Fly: {}", if flying { "ON" } else { "OFF" }),
            format!(
                "Wireframe: {}",
                self.render
                    .renderer
                    .as_ref()
                    .map(|r| r.is_wireframe())
                    .unwrap_or(false)
            ),
        ];

        for line in &lines {
            ui.text(line, x, y, 1.0, color, &self.render.font);
            y += line_h;
        }

        if self.gameplay.chunk_debug_enabled {
            self.draw_chunk_debug_minimap(ui, _w, _h);
        }
    }

    /// Draw the terrain minimap HUD overlay in the top-right corner.
    fn draw_minimap(&mut self, ui: &mut UiDrawData, w: f32, _h: f32) {
        if !self.gameplay.map.visible {
            return;
        }
        // Don't show minimap when fullscreen map is open.
        if self.gameplay.map.fullscreen_open {
            return;
        }

        let map_size = 128.0f32;
        let mx = w - map_size - 10.0;
        let my = 10.0;

        // Background + border.
        ui.quad(
            mx - 2.0,
            my - 2.0,
            map_size + 4.0,
            map_size + 4.0,
            [0, 0, 0, 180],
        );
        ui.rect_border(
            mx - 2.0,
            my - 2.0,
            map_size + 4.0,
            map_size + 4.0,
            1.0,
            [80, 80, 80, 200],
        );

        // Terrain quad (tex_id = 2.0 for minimap texture).
        ui.quad_uv(
            mx,
            my,
            map_size,
            map_size,
            0.0,
            0.0,
            1.0,
            1.0,
            [255, 255, 255, 255],
            2.0,
        );

        // Player direction indicator — triangle pointing in facing direction.
        let dot_x = mx + map_size * 0.5;
        let dot_y = my + map_size * 0.5;
        let dot_size = 4.0f32;
        let angle = self
            .simulation
            .player_camera()
            .map(|c| c.yaw)
            .unwrap_or(0.0);
        let tip = [
            dot_x + angle.sin() * dot_size * 2.0,
            dot_y - angle.cos() * dot_size * 2.0,
        ];
        let left = [
            dot_x + (angle + 2.5).sin() * dot_size,
            dot_y - (angle + 2.5).cos() * dot_size,
        ];
        let right = [
            dot_x + (angle - 2.5).sin() * dot_size,
            dot_y - (angle - 2.5).cos() * dot_size,
        ];
        ui.triangle(tip, left, right, [255, 255, 255, 255]);

        // North indicator.
        ui.text(
            "N",
            mx + map_size * 0.5 - 4.0,
            my - 8.0,
            0.8,
            [255, 200, 100, 255],
            &self.render.font,
        );

        // Zoom indicator.
        let zoom_text = format!("{}m", self.gameplay.map.blocks_per_pixel);
        ui.text(
            &zoom_text,
            mx + 2.0,
            my + map_size - 14.0,
            0.6,
            [180, 180, 200, 200],
            &self.render.font,
        );
    }

    /// Draw the fullscreen map overlay.
    fn draw_fullscreen_map(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        if !self.gameplay.map.fullscreen_open {
            return;
        }

        // Full-screen dark overlay.
        ui.quad(0.0, 0.0, w, h, [0, 0, 0, 220]);

        // Large map centered with margin.
        let map_size = w.min(h) * 0.85;
        let mx = (w - map_size) * 0.5;
        let my = (h - map_size) * 0.5;

        // Background.
        ui.quad(
            mx - 4.0,
            my - 4.0,
            map_size + 8.0,
            map_size + 8.0,
            [20, 20, 30, 255],
        );
        ui.rect_border(
            mx - 4.0,
            my - 4.0,
            map_size + 8.0,
            map_size + 8.0,
            1.0,
            [80, 80, 100, 255],
        );

        // Terrain quad (same texture as minimap).
        ui.quad_uv(
            mx,
            my,
            map_size,
            map_size,
            0.0,
            0.0,
            1.0,
            1.0,
            [255, 255, 255, 255],
            2.0,
        );

        // Player position marker.
        let dot_x = mx + map_size * 0.5;
        let dot_y = my + map_size * 0.5;
        let dot_size = 6.0f32;
        let angle = self
            .simulation
            .player_camera()
            .map(|c| c.yaw)
            .unwrap_or(0.0);
        let tip = [
            dot_x + angle.sin() * dot_size * 2.0,
            dot_y - angle.cos() * dot_size * 2.0,
        ];
        let left = [
            dot_x + (angle + 2.5).sin() * dot_size,
            dot_y - (angle + 2.5).cos() * dot_size,
        ];
        let right = [
            dot_x + (angle - 2.5).sin() * dot_size,
            dot_y - (angle - 2.5).cos() * dot_size,
        ];
        ui.triangle(tip, left, right, [255, 255, 255, 255]);

        // Coordinates text at top.
        let pos = self
            .simulation
            .player_pos()
            .unwrap_or(self.gameplay.spawn_pos);
        let coord_text = format!("[{:.0}, {:.0}, {:.0}]", pos.x, pos.y, pos.z);
        ui.text(
            &coord_text,
            mx + 8.0,
            my + 8.0,
            1.0,
            [255, 255, 255, 255],
            &self.render.font,
        );

        // Controls hint at bottom.
        ui.text(
            "M: close | Scroll: zoom",
            mx + 8.0,
            my + map_size - 20.0,
            0.7,
            [180, 180, 200, 200],
            &self.render.font,
        );
    }

    fn draw_chunk_debug_minimap(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        let map_x = w - 200.0;
        let map_y = h - 220.0;
        let map_w = 192.0;
        let map_h = 192.0;

        ui.quad(
            map_x - 4.0,
            map_y - 4.0,
            map_w + 8.0,
            map_h + 8.0,
            [0, 0, 0, 180],
        );
        ui.text(
            "Chunk Debug",
            map_x,
            map_y - 20.0,
            1.0,
            [200, 200, 255, 230],
            &self.render.font,
        );

        let pos = self
            .simulation
            .player_pos()
            .unwrap_or(self.gameplay.spawn_pos);
        let player_chunk_x = (pos.x as i32) >> 4;
        let player_chunk_z = (pos.z as i32) >> 4;
        let half = 6;
        let cell = map_w / (half * 2 + 1) as f32;

        let age_exceeded = self
            .profiler
            .last_minimap_update
            .map(|t| t.elapsed() >= std::time::Duration::from_millis(100))
            .unwrap_or(true);
        let moved = self
            .profiler
            .last_minimap_chunk
            .map(|(x, z)| x != player_chunk_x || z != player_chunk_z)
            .unwrap_or(true);
        let needs_refresh = age_exceeded || moved || self.profiler.cached_minimap_batch.is_empty();

        if needs_refresh {
            let center = voxel_core::math::ChunkPos::new(player_chunk_x, 0, player_chunk_z);
            self.profiler.cached_minimap_batch =
                self.world_state.world.chunk_debug_info_batch(center, half);
            self.profiler.last_minimap_update = Some(std::time::Instant::now());
            self.profiler.last_minimap_chunk = Some((player_chunk_x, player_chunk_z));
        }

        let batch = self.profiler.cached_minimap_batch.clone();

        for (pos, loaded, dirty, palette_mode, has_mesh) in &batch {
            let color = if *loaded {
                if *dirty {
                    [255, 100, 100, 200]
                } else if !*has_mesh {
                    [255, 255, 100, 200]
                } else if *palette_mode {
                    [100, 100, 255, 200]
                } else {
                    [100, 255, 100, 200]
                }
            } else {
                [60, 60, 60, 150]
            };

            let dx = pos.x() - player_chunk_x;
            let dz = pos.z() - player_chunk_z;
            let sx = map_x + (dx + half) as f32 * cell;
            let sy = map_y + (dz + half) as f32 * cell;
            ui.quad(sx, sy, cell - 1.0, cell - 1.0, color);
        }

        let px = map_x + half as f32 * cell + cell * 0.25;
        let py = map_y + half as f32 * cell + cell * 0.25;
        ui.quad(px, py, cell * 0.5, cell * 0.5, [255, 255, 255, 255]);
    }

    /// Draw the runtime ECS inspector panel (F10).
    fn draw_ecs_inspector(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let panel_w = 380.0;
        let panel_x = w - panel_w - 8.0;
        let panel_y = 8.0;
        let line_h = 14.0;
        let pad = 4.0;
        let color_white = [240, 240, 240, 230];
        let color_label = [120, 180, 255, 230];
        let color_value = [200, 200, 200, 220];
        let color_dim = [140, 140, 160, 200];

        let panel_h = h - panel_y - 8.0;
        ui.quad(panel_x, panel_y, panel_w, panel_h, [0, 0, 0, 170]);
        ui.rect_border(panel_x, panel_y, panel_w, panel_h, 1.0, [80, 80, 100, 200]);

        let mut y = panel_y + pad;
        let cutoff = h - 12.0;

        if y + line_h >= cutoff {
            return;
        }
        ui.text(
            "ECS Inspector",
            panel_x + pad,
            y,
            1.0,
            color_white,
            &self.render.font,
        );
        y += line_h;

        for arch in self.simulation.ecs_world().archetypes() {
            if y + line_h >= cutoff {
                break;
            }
            let component_names = arch.component_names.join(", ");
            let truncated = if component_names.len() > 40 {
                format!("{}... (N={})", &component_names[..37], arch.len())
            } else {
                format!("{} (N={})", component_names, arch.len())
            };
            ui.text(
                &truncated,
                panel_x + pad,
                y,
                0.9,
                color_label,
                &self.render.font,
            );
            y += line_h;

            for (row, entity) in arch.entities().iter().enumerate() {
                if y + line_h >= cutoff {
                    break;
                }
                let is_pinned = self.gameplay.pinned_entity == Some(*entity);
                let pin_color = if is_pinned {
                    [255, 220, 80, 230]
                } else {
                    [120, 120, 140, 200]
                };
                let pin_label = if is_pinned { "[PIN]" } else { "[   ]" };
                ui.text(
                    pin_label,
                    panel_x + pad,
                    y,
                    0.85,
                    pin_color,
                    &self.render.font,
                );
                ui.text(
                    &format!("Entity[{}:{}]", entity.index, entity.generation),
                    panel_x + pad + 38.0,
                    y,
                    0.85,
                    color_white,
                    &self.render.font,
                );
                y += line_h;

                for (col_idx, &tid) in arch.component_types.iter().enumerate() {
                    let name = arch.component_names.get(col_idx).copied().unwrap_or("?");
                    if y + line_h >= cutoff {
                        break;
                    }
                    ui.text(
                        name,
                        panel_x + pad + 8.0,
                        y,
                        0.85,
                        color_label,
                        &self.render.font,
                    );
                    y += line_h;

                    if let Some(raw) = arch.columns()[col_idx].value_as_any(row as u32) {
                        if let Some(text) = self.simulation.ecs_world().format_component(tid, raw) {
                            for sub in text.lines() {
                                if y + line_h >= cutoff {
                                    break;
                                }
                                ui.text(
                                    sub,
                                    panel_x + pad + 16.0,
                                    y,
                                    0.8,
                                    color_value,
                                    &self.render.font,
                                );
                                y += line_h;
                            }
                        } else {
                            ui.text(
                                "  <no formatter>",
                                panel_x + pad + 16.0,
                                y,
                                0.8,
                                color_dim,
                                &self.render.font,
                            );
                            y += line_h;
                        }
                    }
                }
            }
        }

        let rids = self.simulation.ecs_world().resource_type_ids();
        if !rids.is_empty() {
            if y + line_h >= cutoff {
                return;
            }
            ui.text(
                "\u{2500}\u{2500} Resources \u{2500}\u{2500}",
                panel_x + pad,
                y,
                0.9,
                color_label,
                &self.render.font,
            );
            y += line_h;
            for tid in rids {
                if y + line_h >= cutoff {
                    break;
                }
                let name = self.simulation.ecs_world().name_for(tid);
                ui.text(
                    name,
                    panel_x + pad + 4.0,
                    y,
                    0.85,
                    color_label,
                    &self.render.font,
                );
                y += line_h;
                if let Some(text) = self.simulation.ecs_world().format_resource(&tid) {
                    for sub in text.lines() {
                        if y + line_h >= cutoff {
                            break;
                        }
                        ui.text(
                            sub,
                            panel_x + pad + 16.0,
                            y,
                            0.8,
                            color_value,
                            &self.render.font,
                        );
                        y += line_h;
                    }
                } else {
                    ui.text(
                        "  <no formatter>",
                        panel_x + pad + 16.0,
                        y,
                        0.8,
                        color_dim,
                        &self.render.font,
                    );
                    y += line_h;
                }
            }
        }
    }

    /// Draw the profiler overlay (F6) on the right side of the screen.
    fn draw_profiler_overlay(&self, ui: &mut UiDrawData, w: f32, _h: f32) {
        let line_h = 16.0;
        let pad = 8.0;
        let panel_w = 340.0;
        let panel_h = 16.0 * line_h + pad * 2.0;
        let panel_x = w - panel_w - pad;
        let panel_y = pad;

        ui.quad(panel_x, panel_y, panel_w, panel_h, [0, 0, 0, 160]);

        let white = [255, 255, 255, 230];
        let green = [100, 255, 100, 230];
        let yellow = [255, 255, 100, 230];
        let red = [255, 100, 100, 230];

        let x = panel_x + pad;
        let mut y = panel_y + pad;

        let avg_cpu = self.profiler.avg_ms();
        let fps = self.profiler.avg_fps();
        let fps_color = if fps >= 55.0 {
            green
        } else if fps >= 30.0 {
            yellow
        } else {
            red
        };
        ui.text(
            &format!("FPS: {:.0}  ({:.1} ms)", fps, avg_cpu),
            x,
            y,
            1.0,
            fps_color,
            &self.render.font,
        );
        y += line_h;

        if let Some(latest) = self.profiler.gpu_timings.back() {
            let gpu_color = if latest.frame_ms < 16.0 {
                green
            } else if latest.frame_ms < 33.0 {
                yellow
            } else {
                red
            };
            ui.text(
                &format!("GPU: {:.2} ms", latest.frame_ms),
                x,
                y,
                1.0,
                gpu_color,
                &self.render.font,
            );
            y += line_h;

            let total = latest.frame_ms.max(0.001);
            let passes = [
                ("Sky", latest.sky_ms),
                ("Opaque", latest.opaque_ms),
                ("Trans.", latest.transparent_ms),
                ("UI", latest.ui_ms),
                ("Shadow", latest.shadow_ms),
                ("Post", latest.post_ms),
            ];
            for (name, ms) in &passes {
                ui.text(
                    &format!(
                        "  {}:       {:.2} ms ({:.0}%)",
                        name,
                        ms,
                        ms / total * 100.0
                    ),
                    x,
                    y,
                    1.0,
                    white,
                    &self.render.font,
                );
                y += line_h;
            }
        } else {
            ui.text(
                "GPU: waiting...",
                x,
                y,
                1.0,
                [150, 150, 150, 200],
                &self.render.font,
            );
            y += line_h;
            y += line_h * 6.0;
        }

        if !self.profiler.gpu_timings.is_empty() {
            let n = self.profiler.gpu_timings.len();
            let avg_gpu: f32 = self
                .profiler
                .gpu_timings
                .iter()
                .map(|t| t.frame_ms)
                .sum::<f32>()
                / n as f32;
            let max_gpu: f32 = self
                .profiler
                .gpu_timings
                .iter()
                .map(|t| t.frame_ms)
                .fold(0.0f32, f32::max);
            ui.text(
                &format!("Avg GPU: {:.2} ms  Max: {:.2} ms", avg_gpu, max_gpu),
                x,
                y,
                1.0,
                [180, 180, 255, 230],
                &self.render.font,
            );
            y += line_h;
        }

        if !self.profiler.system_timings.is_empty() {
            y += line_h;
            ui.text(
                "Systems (per-step):",
                x,
                y,
                1.0,
                [180, 180, 200, 230],
                &self.render.font,
            );
            y += line_h;
            let total: u64 = self.profiler.system_timings.iter().map(|(_, us)| *us).sum();
            for (name, us) in &self.profiler.system_timings {
                let color = if *us < 500 {
                    green
                } else if *us < 2_000 {
                    yellow
                } else {
                    red
                };
                let label = if *us >= 1_000 {
                    format!("  {name}: {:.2} ms", *us as f64 / 1_000.0)
                } else {
                    format!("  {name}: {us} \u{00B5}s")
                };
                ui.text(&label, x, y, 1.0, color, &self.render.font);
                y += line_h;
            }
            ui.text(
                &format!("  total: {total} \u{00B5}s"),
                x,
                y,
                1.0,
                white,
                &self.render.font,
            );
            y += line_h;
        }

        let chart_x = x;
        let chart_y = y + 4.0;
        let chart_w = panel_w - pad * 2.0;
        let chart_h = 40.0;
        ui.quad(chart_x, chart_y, chart_w, chart_h, [30, 30, 30, 200]);
        let bar_count = self.profiler.gpu_timings.len().min(60);
        if bar_count > 0 {
            let bar_w = chart_w / 60.0;
            let start = self.profiler.gpu_timings.len().saturating_sub(60);
            let max_ms = self
                .profiler
                .gpu_timings
                .iter()
                .skip(start)
                .map(|t| t.frame_ms)
                .fold(1.0f32, f32::max);
            for (i, timing) in self.profiler.gpu_timings.iter().skip(start).enumerate() {
                let bar_h = (timing.frame_ms / max_ms * chart_h).max(1.0);
                let bx = chart_x + i as f32 * bar_w;
                let by = chart_y + chart_h - bar_h;
                let c = if timing.frame_ms < 16.0 {
                    [80, 200, 80, 220]
                } else if timing.frame_ms < 33.0 {
                    [220, 220, 80, 220]
                } else {
                    [220, 80, 80, 220]
                };
                ui.quad(bx, by, bar_w - 1.0, bar_h, c);
            }
        }
    }
}

// ── Transform helper functions (Phase 7) ─────────────────────────────

use glam::IVec3;
use std::sync::Arc;
use voxel_core::BlockId;
use voxel_world::World;

/// Move all blocks in selection by delta. Returns affected chunks.
fn apply_transform_move(
    world: &Arc<World>,
    min: IVec3,
    max: IVec3,
    delta: IVec3,
    undo_redo: &mut voxel_game::UndoRedoState,
) -> Vec<voxel_core::math::ChunkPos> {
    // Read all blocks first.
    let mut blocks = Vec::new();
    for y in min.y..=max.y {
        for z in min.z..=max.z {
            for x in min.x..=max.x {
                let block = world.get_block(x, y, z);
                blocks.push((x, y, z, block));
            }
        }
    }

    undo_redo.begin_batch("Move Selection");

    // Clear old positions.
    for &(x, y, z, block) in &blocks {
        if !block.is_air() && world.set_block(x, y, z, BlockId::AIR) {
            let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                x,
                y,
                z,
                old_block: block.0,
                new_block: 0,
            });
        }
    }

    // Place at new positions.
    for &(x, y, z, block) in &blocks {
        if !block.is_air() {
            let nx = x + delta.x;
            let ny = y + delta.y;
            let nz = z + delta.z;
            if (0..256).contains(&ny) {
                let old = world.get_block(nx, ny, nz);
                if world.set_block(nx, ny, nz, block) {
                    let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                        x: nx,
                        y: ny,
                        z: nz,
                        old_block: old.0,
                        new_block: block.0,
                    });
                }
            }
        }
    }

    undo_redo.commit_batch();

    let new_min = min + delta;
    let new_max = max + delta;
    edit::brush::affected_chunks_range(min.min(new_min), max.max(new_max))
}

/// Rotate blocks in selection 90 degrees CW around Y axis.
fn apply_transform_rotate(
    world: &Arc<World>,
    min: IVec3,
    max: IVec3,
    degrees: i32,
    undo_redo: &mut voxel_game::UndoRedoState,
) -> Vec<voxel_core::math::ChunkPos> {
    let center = (min + max) / 2;
    let steps = (degrees.rem_euclid(360) / 90) as usize;

    // Read all blocks.
    let mut blocks = Vec::new();
    for y in min.y..=max.y {
        for z in min.z..=max.z {
            for x in min.x..=max.x {
                let block = world.get_block(x, y, z);
                blocks.push((x, y, z, block));
            }
        }
    }

    undo_redo.begin_batch("Rotate Selection");

    // Clear old positions.
    for &(x, y, z, block) in &blocks {
        if !block.is_air() && world.set_block(x, y, z, BlockId::AIR) {
            let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                x,
                y,
                z,
                old_block: block.0,
                new_block: 0,
            });
        }
    }

    // Apply rotation steps.
    for &(x, y, z, block) in &blocks {
        if !block.is_air() {
            let mut rx = x;
            let mut rz = z;
            for _ in 0..steps {
                let rel_x = rx - center.x;
                let rel_z = rz - center.z;
                // 90 CW: (x, z) -> (z, -x)
                rx = center.x + rel_z;
                rz = center.z - rel_x;
            }
            if (0..256).contains(&y) {
                let old = world.get_block(rx, y, rz);
                if world.set_block(rx, y, rz, block) {
                    let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                        x: rx,
                        y,
                        z: rz,
                        old_block: old.0,
                        new_block: block.0,
                    });
                }
            }
        }
    }

    undo_redo.commit_batch();
    edit::brush::affected_chunks_range(min, max)
}

/// Scale blocks in selection by factor from origin.
fn apply_transform_scale(
    world: &Arc<World>,
    min: IVec3,
    max: IVec3,
    factor: f32,
    undo_redo: &mut voxel_game::UndoRedoState,
) -> Vec<voxel_core::math::ChunkPos> {
    if factor <= 0.0 {
        return Vec::new();
    }

    let origin = min;

    // Read all blocks.
    let mut blocks = Vec::new();
    for y in min.y..=max.y {
        for z in min.z..=max.z {
            for x in min.x..=max.x {
                let block = world.get_block(x, y, z);
                blocks.push((x, y, z, block));
            }
        }
    }

    undo_redo.begin_batch("Scale Selection");

    // Clear old positions.
    for &(x, y, z, block) in &blocks {
        if !block.is_air() && world.set_block(x, y, z, BlockId::AIR) {
            let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                x,
                y,
                z,
                old_block: block.0,
                new_block: 0,
            });
        }
    }

    // Scale and place.
    for &(x, y, z, block) in &blocks {
        if !block.is_air() {
            let rel_x = (x - origin.x) as f32;
            let rel_y = (y - origin.y) as f32;
            let rel_z = (z - origin.z) as f32;
            let nx = origin.x + (rel_x * factor).round() as i32;
            let ny = origin.y + (rel_y * factor).round() as i32;
            let nz = origin.z + (rel_z * factor).round() as i32;
            if (0..256).contains(&ny) {
                let old = world.get_block(nx, ny, nz);
                if world.set_block(nx, ny, nz, block) {
                    let _ = undo_redo.push_edit_batched(voxel_game::BlockEdit {
                        x: nx,
                        y: ny,
                        z: nz,
                        old_block: old.0,
                        new_block: block.0,
                    });
                }
            }
        }
    }

    undo_redo.commit_batch();

    let new_max = IVec3::new(
        origin.x + ((max.x - origin.x) as f32 * factor).round() as i32,
        origin.y + ((max.y - origin.y) as f32 * factor).round() as i32,
        origin.z + ((max.z - origin.z) as f32 * factor).round() as i32,
    );
    edit::brush::affected_chunks_range(origin.min(new_max), origin.max(new_max))
}
