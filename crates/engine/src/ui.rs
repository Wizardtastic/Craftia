//! UI overlay rendering and click-handling.
//!
//! All `draw_*` methods on `EngineApp` plus their paired click handlers
//! (`handle_*_click`) live here. The split keeps the frame loop in
//! `lib.rs` focused on per-frame orchestration rather than HUD layout.

use voxel_core::{Point, Rect};
use voxel_render::{
    GraphStyle, UiDrawData, TILE_BTN, TILE_BTN_DISABLED, TILE_BTN_HOVER, TILE_BUBBLE_BG,
    TILE_BUBBLE_FULL, TILE_CROSSHAIR, TILE_HEART_BG, TILE_HEART_FULL, TILE_HEART_HALF,
    TILE_HUNGER_BG, TILE_HUNGER_FULL, TILE_HUNGER_HALF, TILE_PANEL, TILE_SCROLL_THUMB,
    TILE_SCROLL_TRACK, TILE_SEL_FRAME, TILE_SLOT,
};

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

/// Fixed layout for the survival inventory screen, shared by the draw
/// and click handlers so hit-testing always matches what is rendered.
struct InventoryLayout {
    /// Panel top-left.
    panel_x: f32,
    panel_y: f32,
    panel_w: f32,
    panel_h: f32,
    /// Top-left of each slot grid (armor / main / hotbar), logical px.
    armor_xy: (f32, f32),
    offhand_xy: (f32, f32),
    main_xy: (f32, f32),
    hotbar_xy: (f32, f32),
    slot: f32,
    gap: f32,
}

const INV_COLS: usize = 9;
const INV_ROWS: usize = 3;

impl InventoryLayout {
    fn new(w: f32, h: f32) -> Self {
        let slot = 44.0;
        let gap = 4.0;
        let grid_w = INV_COLS as f32 * slot + (INV_COLS - 1) as f32 * gap;
        let panel_pad = 14.0;
        // Side column (armor/offhand) + label gutter sits left of the grid.
        let side_w = slot * 2.0 + gap + panel_pad;
        let panel_w = side_w + grid_w + panel_pad * 2.0;
        let panel_h = panel_pad * 2.0 + 20.0 + INV_ROWS as f32 * (slot + gap) + 8.0 + slot + 20.0;
        let panel_x = (w - panel_w) * 0.5;
        let panel_y = (h - panel_h) * 0.5;
        let grid_x0 = panel_x + panel_pad + side_w;
        let grid_y0 = panel_y + panel_pad + 20.0;
        Self {
            panel_x,
            panel_y,
            panel_w,
            panel_h,
            armor_xy: (panel_x + panel_pad, grid_y0),
            offhand_xy: (panel_x + panel_pad, grid_y0 + 4.0 * (slot + gap)),
            main_xy: (grid_x0, grid_y0),
            hotbar_xy: (grid_x0, grid_y0 + INV_ROWS as f32 * (slot + gap) + 8.0),
            slot,
            gap,
        }
    }

    /// Rect of a specific slot, or `None` if out of range.
    fn slot_rect(&self, s: voxel_game::InventorySlot) -> Option<Rect> {
        let gx = |x: f32, y: f32| Rect::from_xywh(x, y, self.slot, self.slot);
        match s {
            voxel_game::InventorySlot::Main(i) if i < 27 => {
                let col = (i % INV_COLS) as f32;
                let row = (i / INV_COLS) as f32;
                Some(gx(
                    self.main_xy.0 + col * (self.slot + self.gap),
                    self.main_xy.1 + row * (self.slot + self.gap),
                ))
            }
            voxel_game::InventorySlot::Hotbar(i) if i < 9 => Some(gx(
                self.hotbar_xy.0 + i as f32 * (self.slot + self.gap),
                self.hotbar_xy.1,
            )),
            voxel_game::InventorySlot::Armor(i) if i < 4 => Some(gx(
                self.armor_xy.0,
                self.armor_xy.1 + i as f32 * (self.slot + self.gap),
            )),
            voxel_game::InventorySlot::Offhand => Some(gx(self.offhand_xy.0, self.offhand_xy.1)),
            _ => None,
        }
    }

    /// Slot under the given logical-pixel position, if any.
    fn slot_at(&self, mx: f32, my: f32) -> Option<voxel_game::InventorySlot> {
        let inside =
            |r: &Rect, px: f32, py: f32| px >= r.x && px < r.x + r.w && py >= r.y && py < r.y + r.h;
        let mut hit = None;
        for i in 0..27 {
            let r = self.slot_rect(voxel_game::InventorySlot::Main(i)).unwrap();
            if inside(&r, mx, my) {
                hit = Some(voxel_game::InventorySlot::Main(i));
            }
        }
        for i in 0..9 {
            let r = self
                .slot_rect(voxel_game::InventorySlot::Hotbar(i))
                .unwrap();
            if inside(&r, mx, my) {
                hit = Some(voxel_game::InventorySlot::Hotbar(i));
            }
        }
        for i in 0..4 {
            let r = self.slot_rect(voxel_game::InventorySlot::Armor(i)).unwrap();
            if inside(&r, mx, my) {
                hit = Some(voxel_game::InventorySlot::Armor(i));
            }
        }
        if inside(
            &self.slot_rect(voxel_game::InventorySlot::Offhand).unwrap(),
            mx,
            my,
        ) {
            hit = Some(voxel_game::InventorySlot::Offhand);
        }
        hit
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
                if self.gameplay.inventory_open {
                    self.draw_inventory_screen(&mut ui, w, h);
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
        let size = 20.0;
        // Classic plus-shaped crosshair from the UI atlas.
        ui.sprite(
            TILE_CROSSHAIR,
            cx - size * 0.5,
            cy - size * 0.5,
            size,
            [255, 255, 255, 220],
        );
        match self.gameplay.crosshair_mode {
            crate::CrosshairMode::BlockTarget => {
                // Block targeted: redraw at full opacity.
                ui.sprite(
                    TILE_CROSSHAIR,
                    cx - size * 0.5,
                    cy - size * 0.5,
                    size,
                    [255, 255, 255, 255],
                );
            }
            crate::CrosshairMode::Interact => {
                // Interactable: center square marker.
                let square_size = 5.0;
                ui.quad(
                    cx - square_size * 0.5,
                    cy - square_size * 0.5,
                    square_size,
                    square_size,
                    [255, 255, 255, 200],
                );
            }
            crate::CrosshairMode::Default => {}
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
            ui.sprite_wh(TILE_SLOT, x, y0, slot, slot, [235, 235, 240, 240]);

            let block_id = self
                .simulation
                .inventory()
                .and_then(|inv| inv.hotbar_block(i))
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

            if i == self.simulation.inventory().map_or(0, |inv| inv.selected) {
                ui.nine_slice(
                    TILE_SEL_FRAME,
                    x - 3.0,
                    y0 - 3.0,
                    slot + 6.0,
                    slot + 6.0,
                    3.0,
                    [255, 255, 255, 255],
                );
            }
        }
    }

    /// Draw one inventory slot: frame + block icon for non-empty stacks.
    fn draw_inv_slot(
        &self,
        ui: &mut UiDrawData,
        rect: Rect,
        stack: &voxel_game::ItemStack,
        selected: bool,
        reg: &voxel_world::BlockRegistry,
    ) {
        let _ = selected;
        ui.sprite_wh(
            TILE_SLOT,
            rect.x,
            rect.y,
            rect.w,
            rect.h,
            [235, 235, 240, 240],
        );
        if !stack.is_empty() {
            let def = reg.get(stack.id());
            let tile = def.textures.tile(voxel_world::registry::Face::PosX);
            let pad = 4.0;
            ui.block_icon(
                rect.x + pad,
                rect.y + pad,
                rect.w - pad * 2.0,
                rect.h - pad * 2.0,
                tile,
                [255, 255, 255, 255],
            );
            // Stack count (bottom-right), only when > 1.
            if stack.count > 1 {
                let label = stack.count.to_string();
                let tw = self.render.font.text_width(&label, 1.0);
                ui.text_shadow(
                    &label,
                    rect.x + rect.w - tw - 3.0,
                    rect.y + rect.h - 12.0,
                    1.0,
                    [240, 240, 240, 255],
                    &self.render.font,
                );
            }
        }
    }

    /// Draw the survival inventory screen: main 9×3 grid, hotbar row, armor
    /// column, and offhand slot. Interactions are handled by
    /// [`Self::handle_inventory_click`], which recomputes this layout.
    fn draw_inventory_screen(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let l = InventoryLayout::new(w, h);
        let Some(inv) = self.simulation.inventory() else {
            return;
        };
        let reg = self.world_state.world.registry();

        // Dim the world behind the panel.
        ui.quad(0.0, 0.0, w, h, [0, 0, 0, 140]);

        // Panel + title (atlas chrome).
        ui.nine_slice(
            TILE_PANEL,
            l.panel_x,
            l.panel_y,
            l.panel_w,
            l.panel_h,
            6.0,
            [205, 205, 215, 255],
        );
        ui.text_shadow(
            "Inventory",
            l.panel_x + 14.0,
            l.panel_y + 10.0,
            1.2,
            crate::ui_kit::palette::INK,
            &self.render.font,
        );

        // Armor column.
        for (i, name) in ["Helm", "Chest", "Legs", "Boots"].iter().enumerate() {
            let slot = voxel_game::InventorySlot::Armor(i);
            let rect = l.slot_rect(slot).unwrap();
            self.draw_inv_slot(ui, rect, inv.get_slot(slot), false, &reg);
            ui.text(
                name,
                rect.x,
                rect.y - 13.0,
                1.0,
                crate::ui_kit::palette::INK_MUTED,
                &self.render.font,
            );
        }
        // Offhand slot.
        let off = voxel_game::InventorySlot::Offhand;
        let off_rect = l.slot_rect(off).unwrap();
        self.draw_inv_slot(ui, off_rect, inv.get_slot(off), false, &reg);
        ui.text(
            "Offhand",
            off_rect.x,
            off_rect.y - 13.0,
            1.0,
            crate::ui_kit::palette::INK_MUTED,
            &self.render.font,
        );

        // Main 9×3 grid.
        for i in 0..27 {
            let slot = voxel_game::InventorySlot::Main(i);
            let rect = l.slot_rect(slot).unwrap();
            self.draw_inv_slot(ui, rect, inv.get_slot(slot), false, &reg);
        }

        // Hotbar row (highlight the selected slot).
        for i in 0..9 {
            let slot = voxel_game::InventorySlot::Hotbar(i);
            let rect = l.slot_rect(slot).unwrap();
            self.draw_inv_slot(ui, rect, inv.get_slot(slot), inv.selected == i, &reg);
        }

        // Hint line.
        ui.text_shadow(
            "Left-click: move stack - Right-click: half/one - Shift-click: quick-move - I/Esc: close",
            l.panel_x + 14.0,
            l.panel_y + l.panel_h - 18.0,
            1.0,
            crate::ui_kit::palette::INK_MUTED,
            &self.render.font,
        );
    }

    /// Handle a click on the survival inventory screen. `left`/`right` mirror
    /// the mouse buttons; `shift` enables quick-move. Currently supports
    /// shift-click quick-move (main <-> hotbar) and plain left-click swaps
    /// between two slots; armor/offhand accept swaps so players can equip.
    pub(crate) fn handle_inventory_click(&mut self, left: bool, _right: bool, shift: bool) {
        let (w, h) = self.render.logical_size();
        let l = InventoryLayout::new(w, h);
        let (mx, my) = (self.gameplay.mouse_pos.x, self.gameplay.mouse_pos.y);
        let Some(slot) = l.slot_at(mx, my) else {
            return;
        };
        let Some(inv) = self.simulation.inventory_mut() else {
            return;
        };

        if shift && left {
            inv.shift_click_merge(slot);
            return;
        }
        if !left {
            // Right-click: not wired up yet — treat as a no-op rather than a
            // silent stack swap.
            return;
        }

        // Left-click swap: clicking armor/offhand equips from the storage
        // zones and unequips back; storage↔storage swaps move whole stacks.
        match (self.gameplay.inv_cursor.take(), slot) {
            (None, from) => {
                // First click picks the stack up (only for non-empty slots).
                if !inv.get_slot(from).is_empty() {
                    self.gameplay.inv_cursor = Some(from);
                }
            }
            (Some(from), to) if from == to => {
                // Clicked the same slot again: put it back.
                self.gameplay.inv_cursor = None;
            }
            (Some(from), to) => {
                inv.swap(from, to);
                self.gameplay.inv_cursor = None;
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
            .simulation
            .inventory()
            .and_then(|inv| inv.selected_block())
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

        // Draw 10 hearts from the UI atlas.
        for i in 0..10 {
            let heart_x = x0 + i as f32 * (heart_size + heart_gap);
            let health_value = health.current - i as f32 * 2.0;
            let tile = if health_value >= 2.0 {
                TILE_HEART_FULL
            } else if health_value >= 1.0 {
                TILE_HEART_HALF
            } else {
                TILE_HEART_BG
            };
            ui.sprite(tile, heart_x, bar_y, heart_size, [255, 255, 255, 255]);
        }

        // Draw health text (optional, for debugging).
        let health_text = format!("{}/{}", health.current as i32, health.max as i32);
        let text_x = x0 + 10.0 * (heart_size + heart_gap) + 4.0;
        ui.text_shadow(
            &health_text,
            text_x,
            bar_y + 4.0,
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

        // Draw 10 drumsticks from the right (UI atlas icons).
        for i in 0..10 {
            let drumstick_x = x0 + total - (i as f32 + 1.0) * (drumstick_size + drumstick_gap);
            let food_value = hunger.food - i as f32 * 2.0;
            let tile = if food_value >= 2.0 {
                TILE_HUNGER_FULL
            } else if food_value >= 1.0 {
                TILE_HUNGER_HALF
            } else {
                TILE_HUNGER_BG
            };
            ui.sprite(
                tile,
                drumstick_x,
                bar_y,
                drumstick_size,
                [255, 255, 255, 255],
            );
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

        // Draw 10 bubbles from the right (UI atlas icons).
        for i in 0..10 {
            let bubble_x = x0 + total - (i as f32 + 1.0) * (bubble_size + bubble_gap);
            let air_value = air.current - i as f32 * (air.max / 10.0);
            let tile = if air_value >= air.max / 10.0 {
                TILE_BUBBLE_FULL
            } else if air_value > 0.0 {
                // Partial bubble: draw the full icon dimmed.
                TILE_BUBBLE_FULL
            } else {
                TILE_BUBBLE_BG
            };
            let alpha = if air_value >= air.max / 10.0 {
                255
            } else if air_value > 0.0 {
                140
            } else {
                255
            };
            ui.sprite(tile, bubble_x, bar_y, bubble_size, [255, 255, 255, alpha]);
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
        // Dark red gradient background.
        ui.gradient_v(0.0, 0.0, w, h, [130, 12, 12, 190], [30, 0, 0, 190]);

        // "You died!" title.
        let title = "You died!";
        let title_size = 3.0;
        let title_width = self.render.font.text_width(title, title_size);
        ui.text_shadow(
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
        ui.text_shadow(
            message,
            (w - msg_width) * 0.5,
            h * 0.4,
            msg_size,
            [225, 225, 228, 255],
            &self.render.font,
        );

        // Respawn + Title Screen buttons (atlas chrome, same geometry).
        let btn_w = 200.0;
        let btn_h = 40.0;
        let btn_x = (w - btn_w) * 0.5;
        let btn_y = h * 0.5;
        let btn2_y = btn_y + 60.0;
        let labels = ["RESPAWN", "TITLE SCREEN"];
        let tints = [[225, 240, 225, 255], [250, 220, 215, 255]];
        for (i, label) in labels.iter().enumerate() {
            let r = Rect::from_xywh(btn_x, if i == 0 { btn_y } else { btn2_y }, btn_w, btn_h);
            let hovered = r.contains(self.gameplay.mouse_pos);
            let tile = if hovered { TILE_BTN_HOVER } else { TILE_BTN };
            ui.nine_slice(tile, r.x, r.y, r.w, r.h, 6.0, tints[i]);
            let scale = 1.5;
            let lw = self.render.font.text_width(label, scale);
            ui.text_shadow(
                label,
                r.x + (r.w - lw) * 0.5,
                r.y + (r.h - 14.0 * scale) * 0.5,
                scale,
                crate::ui_kit::palette::INK,
                &self.render.font,
            );
        }
    }

    /// Draw the pause/exit menu (4 buttons: back, options, save & quit, quit).
    fn draw_pause_menu(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        // UiKit path: themed immediate-mode widgets (see ui_kit.rs).
        let mut kit = crate::ui_kit::UiKit::new(
            ui,
            &self.render.font,
            (self.gameplay.mouse_pos.x, self.gameplay.mouse_pos.y),
        );
        kit.dim(0.0, 0.0, w, h, 160);

        // Shared layout (screen_layout.rs) — identical math for the click
        // handler, so no stored rects are needed.
        let lay = crate::screen_layout::PauseLayout::new(w, h);
        kit.panel(
            lay.panel.x,
            lay.panel.y,
            lay.panel.w,
            lay.panel.h,
            [205, 205, 215, 255],
        );
        kit.label_centered(
            "PAUSED",
            lay.panel.x + lay.panel.w * 0.5,
            lay.panel.y + 20.0,
            2.0,
            crate::ui_kit::palette::INK,
        );

        const LABELS: [&str; 4] = ["BACK TO GAME", "OPTIONS", "SAVE & QUIT", "QUIT GAME"];
        for (label, rect) in LABELS.iter().zip(lay.buttons.iter()) {
            kit.ui.nine_slice(
                if kit.hovered(*rect) {
                    TILE_BTN_HOVER
                } else {
                    TILE_BTN
                },
                rect.x,
                rect.y,
                rect.w,
                rect.h,
                6.0,
                [225, 225, 232, 255],
            );
            let scale = 1.5;
            let lw = self.render.font.text_width(label, scale);
            kit.ui.text_shadow(
                label,
                rect.x + (rect.w - lw) * 0.5,
                rect.y + (rect.h - 14.0 * scale) * 0.5,
                scale,
                crate::ui_kit::palette::INK,
                &self.render.font,
            );
        }
        kit.finish();
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
                if let Some(inv) = self.simulation.inventory_mut() {
                    inv.set_hotbar_block(inv.selected, item.id);
                }
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
                if let Some(inv) = self.simulation.inventory_mut() {
                    inv.select(slot_idx);
                }
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
        // Shared layout — the exact rects the draw pass rendered.
        let (w, h) = self.render.logical_size();
        let lay = crate::screen_layout::PauseLayout::new(w, h);
        let mouse = self.gameplay.mouse_pos;

        // Play UI click sound.
        self.audio.push_event(voxel_audio::AudioEvent::PlaySfx {
            sound: "ui.click".into(),
            position: None,
            volume: 1.0,
            pitch: None,
            group: voxel_audio::AudioGroup::Sfx,
        });
        // Back to Game
        if lay.buttons[crate::screen_layout::PauseLayout::BACK].contains(mouse) {
            self.enter_playing();
        }
        // Options
        if lay.buttons[crate::screen_layout::PauseLayout::OPTIONS].contains(mouse) {
            self.enter_settings(GameState::PauseMenu);
        }
        // Save & Quit to Title
        if lay.buttons[crate::screen_layout::PauseLayout::SAVE_QUIT].contains(mouse) {
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
        if lay.buttons[crate::screen_layout::PauseLayout::QUIT].contains(mouse) {
            log::info!("exit game requested");
            self.gameplay.want_exit = true;
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
        // Background: vertical dark gradient (the panorama renders behind;
        // this veil keeps the menu readable on any sky).
        ui.gradient_v(0.0, 0.0, w, h, [14, 16, 26, 235], [4, 4, 8, 235]);

        // Shared layout (screen_layout.rs) — identical math for the click
        // handler, so no stored rects are needed.
        let lay = crate::screen_layout::TitleLayout::new(w, h);

        // Title — drop-shadowed golden logo text.
        let title = "VOXEL ENGINE";
        let tw = self.render.font.text_width(title, 4.0);
        ui.text_shadow(
            title,
            (w - tw) * 0.5,
            h * 0.38 - 70.0,
            4.0,
            [255, 224, 128, 255],
            &self.render.font,
        );

        // Subtitle
        let sub = "voxel engine v0.1";
        let sw = self.render.font.text_width(sub, 1.0);
        ui.text_shadow(
            sub,
            (w - sw) * 0.5,
            h * 0.38 - 36.0,
            1.0,
            [190, 190, 200, 255],
            &self.render.font,
        );

        // Buttons — beveled stone chrome from the procedural UI atlas.
        const LABELS: [&str; 4] = ["SINGLEPLAYER", "MULTIPLAYER", "OPTIONS", "QUIT GAME"];
        const TINTS: [[u8; 4]; 4] = [
            [228, 238, 228, 255],
            [225, 225, 230, 255],
            [226, 226, 240, 255],
            [255, 206, 200, 255],
        ];

        for (i, (label, rect)) in LABELS.iter().zip(lay.buttons.iter()).enumerate() {
            let hovered = rect.contains(self.gameplay.mouse_pos);
            let tile = if i == crate::screen_layout::TitleLayout::MULTIPLAYER {
                TILE_BTN_DISABLED
            } else if hovered {
                TILE_BTN_HOVER
            } else {
                TILE_BTN
            };
            let text = if i == crate::screen_layout::TitleLayout::MULTIPLAYER {
                crate::ui_kit::palette::INK_DISABLED
            } else {
                crate::ui_kit::palette::INK
            };
            ui.nine_slice(tile, rect.x, rect.y, rect.w, rect.h, 6.0, TINTS[i]);
            let scale = 1.5;
            let lw = self.render.font.text_width(label, scale);
            ui.text_shadow(
                label,
                rect.x + (rect.w - lw) * 0.5,
                rect.y + (rect.h - 14.0 * scale) * 0.5,
                scale,
                text,
                &self.render.font,
            );
        }

        // "Multiplayer" is grayed out — draw "Coming Soon" on hover.
        let mp_btn = lay.buttons[crate::screen_layout::TitleLayout::MULTIPLAYER];
        if mp_btn.contains(self.gameplay.mouse_pos) {
            let cs = "Coming Soon";
            let csw = self.render.font.text_width(cs, 0.8);
            ui.text_shadow(
                cs,
                mp_btn.x + (mp_btn.w - csw) * 0.5,
                mp_btn.y + mp_btn.h + 4.0,
                0.8,
                [140, 220, 140, 230],
                &self.render.font,
            );
        }

        // Copyright
        let copyright = "Copyright 2026";
        let cw = self.render.font.text_width(copyright, 0.7);
        ui.text_shadow(
            copyright,
            (w - cw) * 0.5,
            h - 24.0,
            0.7,
            [120, 120, 125, 220],
            &self.render.font,
        );
    }

    pub(crate) fn handle_title_click(&mut self) {
        // Shared layout — the exact rects the draw pass rendered.
        let (w, h) = self.render.logical_size();
        let lay = crate::screen_layout::TitleLayout::new(w, h);
        let mouse = self.gameplay.mouse_pos;

        // Play UI click sound.
        self.audio.push_event(voxel_audio::AudioEvent::PlaySfx {
            sound: "ui.click".into(),
            position: None,
            volume: 1.0,
            pitch: None,
            group: voxel_audio::AudioGroup::Sfx,
        });
        // Singleplayer — show world selection screen.
        if lay.buttons[crate::screen_layout::TitleLayout::SINGLEPLAYER].contains(mouse) {
            self.enter_world_select();
        }
        // Multiplayer (not implemented)
        if lay.buttons[crate::screen_layout::TitleLayout::MULTIPLAYER].contains(mouse) {
            self.gameplay
                .chat
                .push_message("Multiplayer not yet implemented".into());
        }
        // Options
        if lay.buttons[crate::screen_layout::TitleLayout::OPTIONS].contains(mouse) {
            self.enter_settings(GameState::TitleScreen);
        }
        // Quit
        if lay.buttons[crate::screen_layout::TitleLayout::QUIT].contains(mouse) {
            self.gameplay.want_exit = true;
        }
    }

    // ── World Select ────────────────────────────────────────────────────

    fn draw_world_select(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        // Full-screen dimmed overlay.
        ui.quad(0.0, 0.0, w, h, [0, 0, 0, 200]);

        // Shared layout (screen_layout.rs) — identical math for the click
        // handler, so stored rects are unnecessary.
        let lay = crate::screen_layout::WorldSelectLayout::new(
            w,
            h,
            self.gameplay.world_list.len(),
            self.gameplay.world_select_scroll,
        );
        let px = lay.panel.x;
        let py = lay.panel.y;
        let panel_w = lay.panel.w;

        // Scroll window over the world list (draw-side view of the offset
        // the input handlers maintain in `world_select_scroll`).
        let scroll = crate::ui_kit::Scroll::new(
            self.gameplay.world_select_scroll as f32,
            self.gameplay.world_list.len(),
            crate::screen_layout::WorldSelectLayout::capacity(),
        );

        ui.nine_slice(
            TILE_PANEL,
            px,
            py,
            panel_w,
            lay.panel.h,
            6.0,
            [205, 205, 215, 255],
        );

        // Title
        ui.text_shadow(
            "Select World",
            px + 16.0,
            py + 12.0,
            1.5,
            crate::ui_kit::palette::INK,
            &self.render.font,
        );

        // Close button
        let close = lay.close_btn;
        let close_hovered = close.contains(self.gameplay.mouse_pos);
        ui.nine_slice(
            if close_hovered {
                TILE_BTN_HOVER
            } else {
                TILE_BTN
            },
            close.x,
            close.y,
            close.w,
            close.h,
            4.0,
            [255, 210, 205, 255],
        );
        ui.text_shadow(
            "X",
            close.x + 5.0,
            close.y + 3.0,
            1.0,
            crate::ui_kit::palette::INK_DANGER,
            &self.render.font,
        );

        // World rows: the layout's window over the (possibly scrolled)
        // list. `rows[k]` is world `lay.first_index + k`.
        for (k, world) in self
            .gameplay
            .world_list
            .iter()
            .skip(lay.first_index)
            .take(lay.visible_rows())
            .enumerate()
        {
            let i = lay.first_index + k;
            let row = lay.rows[k];
            let del = lay.delete_buttons[k];
            let ry = row.y;
            let selected = self.gameplay.selected_world_index == Some(i);
            let row_hovered = row.contains(self.gameplay.mouse_pos);

            // Row: panel chrome; selected rows get the highlight tint.
            let tint = if selected {
                [235, 240, 255, 255]
            } else if row_hovered {
                [210, 210, 220, 255]
            } else {
                [170, 170, 180, 255]
            };
            ui.nine_slice(TILE_PANEL, row.x, row.y, row.w, row.h, 4.0, tint);

            // World name
            ui.text_shadow(
                &world.name,
                px + 16.0,
                ry + 6.0,
                1.2,
                if selected {
                    [30, 28, 36, 255]
                } else {
                    crate::ui_kit::palette::INK
                },
                &self.render.font,
            );
            // Seed + last played (human-readable relative time)
            let played_str = format_last_played(&world.last_played);
            let info = format!("Seed: {}  |  Played: {}", world.seed, played_str);
            ui.text_shadow(
                &info,
                px + 16.0,
                ry + 24.0,
                0.8,
                crate::ui_kit::palette::INK_MUTED,
                &self.render.font,
            );
            // Game mode
            ui.text_shadow(
                &world.game_mode,
                px + 16.0,
                ry + 38.0,
                0.7,
                crate::ui_kit::palette::INK_DISABLED,
                &self.render.font,
            );

            // Delete button (small beveled chrome, danger tint on hover)
            let del_hovered = del.contains(self.gameplay.mouse_pos);
            ui.nine_slice(
                if del_hovered {
                    TILE_BTN_HOVER
                } else {
                    TILE_BTN
                },
                del.x,
                del.y,
                del.w,
                del.h,
                4.0,
                if del_hovered {
                    [255, 205, 200, 255]
                } else {
                    [215, 215, 220, 255]
                },
            );
            ui.text_shadow(
                "Del",
                del.x + 6.0,
                del.y + 2.0,
                0.7,
                crate::ui_kit::palette::INK_DANGER,
                &self.render.font,
            );
        }

        // Scrollbar along the list's right edge (only when content
        // overflows the window). Same thumb math as `UiKit::scrollbar`.
        if let Some(frac) = scroll.thumb() {
            let bar_w = 8.0;
            let track_x = lay.list.x + lay.list.w - bar_w - 2.0;
            ui.nine_slice(
                TILE_SCROLL_TRACK,
                track_x,
                lay.list.y,
                bar_w,
                lay.list.h,
                3.0,
                [255, 255, 255, 255],
            );
            let thumb_h = lay.list.h * scroll.visible_fraction();
            let thumb_y = lay.list.y + (lay.list.h - thumb_h) * frac.clamp(0.0, 1.0);
            ui.nine_slice(
                TILE_SCROLL_THUMB,
                track_x + 1.0,
                thumb_y,
                bar_w - 2.0,
                thumb_h,
                3.0,
                [255, 255, 255, 255],
            );
        }

        // Create New World button
        let create = lay.create_btn;
        let create_hovered = create.contains(self.gameplay.mouse_pos);
        ui.nine_slice(
            if create_hovered {
                TILE_BTN_HOVER
            } else {
                TILE_BTN
            },
            create.x,
            create.y,
            create.w,
            create.h,
            4.0,
            [228, 240, 228, 255],
        );
        ui.text_shadow(
            "+ Create New World",
            create.x + 12.0,
            create.y + 7.0,
            0.8,
            crate::ui_kit::palette::INK,
            &self.render.font,
        );

        // Play Selected World button
        let play = lay.play_btn;
        let play_enabled = self.gameplay.selected_world_index.is_some();
        let play_hovered = play.contains(self.gameplay.mouse_pos);
        let play_tile = if !play_enabled {
            TILE_BTN_DISABLED
        } else if play_hovered {
            TILE_BTN_HOVER
        } else {
            TILE_BTN
        };
        ui.nine_slice(
            play_tile,
            play.x,
            play.y,
            play.w,
            play.h,
            4.0,
            [228, 240, 228, 255],
        );
        ui.text_shadow(
            "Play Selected World",
            play.x + 12.0,
            play.y + 7.0,
            0.8,
            if play_enabled {
                crate::ui_kit::palette::INK
            } else {
                crate::ui_kit::palette::INK_DISABLED
            },
            &self.render.font,
        );
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

        let (w, h) = self.render.logical_size();
        let lay = crate::screen_layout::WorldSelectLayout::new(
            w,
            h,
            self.gameplay.world_list.len(),
            self.gameplay.world_select_scroll,
        );

        // Close button
        if lay.close_btn.contains(self.gameplay.mouse_pos) {
            self.enter_title_screen();
            return;
        }

        // Delete buttons — check BEFORE row selection (delete sits inside the row rect).
        for (k, &rect) in lay.delete_buttons.iter().enumerate() {
            if rect.contains(self.gameplay.mouse_pos) {
                let idx = lay.first_index + k;
                if idx < self.gameplay.world_list.len() {
                    self.gameplay.pending_delete = Some(idx);
                }
                return;
            }
        }

        // World row selection (with double-click detection)
        for (k, &rect) in lay.rows.iter().enumerate() {
            if rect.contains(self.gameplay.mouse_pos) {
                let i = lay.first_index + k;
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
        if lay.create_btn.contains(self.gameplay.mouse_pos) {
            self.gameplay.create_world_state = Some(crate::CreateWorldState::default());
            return;
        }

        // Play Selected World
        if lay.play_btn.contains(self.gameplay.mouse_pos) {
            if let Some(idx) = self.gameplay.selected_world_index {
                if idx < self.gameplay.world_list.len() {
                    let save_path = self.gameplay.world_list[idx].path.clone();
                    self.load_and_play_world(save_path);
                }
            }
        }
    }

    /// Load a world from a save directory and start playing.
    pub(crate) fn load_and_play_world(&mut self, save_path: std::path::PathBuf) {
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

        let Some(ref state) = self.gameplay.create_world_state else {
            return;
        };
        let lay = crate::screen_layout::CreateWorldLayout::new(w, h, state.name.trim().is_empty());
        let panel = lay.panel;
        let px = panel.x;
        let py = panel.y;

        ui.nine_slice(
            TILE_PANEL,
            panel.x,
            panel.y,
            panel.w,
            panel.h,
            6.0,
            [205, 205, 215, 255],
        );

        // Title + close button.
        ui.text_shadow(
            "Create New World",
            px + 16.0,
            py + 12.0,
            1.3,
            crate::ui_kit::palette::INK,
            &self.render.font,
        );
        let close = lay.close_btn;
        let close_hovered = close.contains(self.gameplay.mouse_pos);
        ui.nine_slice(
            if close_hovered {
                TILE_BTN_HOVER
            } else {
                TILE_BTN
            },
            close.x,
            close.y,
            close.w,
            close.h,
            4.0,
            [255, 210, 205, 255],
        );
        ui.text_shadow(
            "x",
            close.x + 5.0,
            close.y + 2.0,
            0.9,
            crate::ui_kit::palette::INK,
            &self.render.font,
        );

        let lx = panel.x + 16.0;

        // Text fields (inset chrome, blue focus ring on the active one).
        for (label, field, active, value) in [
            (
                "World Name:",
                lay.name_input,
                state.active_field == 0,
                &state.name,
            ),
            (
                "Seed (blank = random):",
                lay.seed_input,
                state.active_field == 1,
                &state.seed,
            ),
        ] {
            let label_y = field.y - 16.0;
            ui.text_shadow(
                label,
                lx,
                label_y,
                0.8,
                crate::ui_kit::palette::INK_MUTED,
                &self.render.font,
            );
            ui.nine_slice(
                voxel_render::TILE_PANEL_INSET,
                field.x,
                field.y,
                field.w,
                field.h,
                3.0,
                if active {
                    [225, 232, 255, 255]
                } else {
                    [170, 170, 180, 255]
                },
            );
            // Draw text with blinking cursor on the active field.
            let display = if active {
                let blink = ((std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
                    / 500)
                    % 2)
                    == 0;
                if blink {
                    format!("{}|", value)
                } else {
                    format!("{} ", value)
                }
            } else {
                value.clone()
            };
            ui.text_shadow(
                &display,
                field.x + 4.0,
                field.y + 5.0,
                0.8,
                crate::ui_kit::palette::TEXT,
                &self.render.font,
            );
        }

        // Game Mode radio buttons.
        ui.text_shadow(
            "Game Mode:",
            lx,
            lay.mode_survival.y - 16.0,
            0.8,
            crate::ui_kit::palette::INK_MUTED,
            &self.render.font,
        );
        let radio_y = lay.mode_survival.y;
        for (r, label, active) in [
            (lay.mode_survival, "Survival", state.game_mode == "survival"),
            (lay.mode_creative, "Creative", state.game_mode == "creative"),
        ] {
            let x = r.x - 10.0;
            ui.quad(x, radio_y, 12.0, 12.0, [15, 15, 20, 255]);
            ui.rect_border(x, radio_y, 12.0, 12.0, 1.0, [70, 70, 85, 255]);
            if active {
                ui.quad(x + 3.0, radio_y + 3.0, 6.0, 6.0, [74, 127, 212, 255]);
            }
            ui.text_shadow(
                label,
                x + 18.0,
                radio_y,
                0.8,
                crate::ui_kit::palette::INK,
                &self.render.font,
            );
        }

        // Allow Cheats checkbox.
        let cb = Rect::from_xywh(lay.cheats_toggle.x - 10.0, lay.cheats_toggle.y, 12.0, 12.0);
        ui.quad(cb.x, cb.y, cb.w, cb.h, [15, 15, 20, 255]);
        ui.rect_border(cb.x, cb.y, cb.w, cb.h, 1.0, [70, 70, 85, 255]);
        if state.allow_cheats {
            ui.quad(cb.x + 3.0, cb.y + 3.0, 6.0, 6.0, [74, 127, 212, 255]);
        }
        ui.text_shadow(
            "Allow Cheats",
            cb.x + 18.0,
            cb.y,
            0.8,
            [220, 220, 220, 255],
            &self.render.font,
        );

        // Error message.
        if let Some(ref err) = state.error {
            ui.text_shadow(
                err,
                lx,
                cb.y + 22.0,
                0.7,
                [240, 100, 100, 255],
                &self.render.font,
            );
        }

        // Buttons: Cancel + Create World.
        let cancel = lay.cancel_btn;
        let cancel_hovered = cancel.contains(self.gameplay.mouse_pos);
        ui.nine_slice(
            if cancel_hovered {
                TILE_BTN_HOVER
            } else {
                TILE_BTN
            },
            cancel.x,
            cancel.y,
            cancel.w,
            cancel.h,
            4.0,
            [225, 225, 230, 255],
        );
        ui.text_shadow(
            "Cancel",
            cancel.x + 16.0,
            cancel.y + 7.0,
            0.8,
            crate::ui_kit::palette::INK,
            &self.render.font,
        );

        let create = lay.create_btn;
        let create_hovered = create.contains(self.gameplay.mouse_pos);
        let (create_tile, create_text) = if lay.create_enabled {
            (
                if create_hovered {
                    TILE_BTN_HOVER
                } else {
                    TILE_BTN
                },
                crate::ui_kit::palette::INK,
            )
        } else {
            (TILE_BTN_DISABLED, crate::ui_kit::palette::INK_DISABLED)
        };
        ui.nine_slice(
            create_tile,
            create.x,
            create.y,
            create.w,
            create.h,
            4.0,
            [228, 240, 228, 255],
        );
        ui.text_shadow(
            "Create World",
            create.x + 10.0,
            create.y + 7.0,
            0.8,
            create_text,
            &self.render.font,
        );
    }

    fn handle_create_world_click(&mut self) {
        let (w, h) = self.render.logical_size();
        let name_is_empty = self
            .gameplay
            .create_world_state
            .as_ref()
            .map(|s| s.name.trim().is_empty())
            .unwrap_or(true);
        let lay = crate::screen_layout::CreateWorldLayout::new(w, h, name_is_empty);

        // Close button.
        if lay.close_btn.contains(self.gameplay.mouse_pos) {
            self.gameplay.create_world_state = None;
            return;
        }

        // Name input field click.
        if lay.name_input.contains(self.gameplay.mouse_pos) {
            if let Some(ref mut state) = self.gameplay.create_world_state {
                state.active_field = 0;
            }
            return;
        }

        // Seed input field click.
        if lay.seed_input.contains(self.gameplay.mouse_pos) {
            if let Some(ref mut state) = self.gameplay.create_world_state {
                state.active_field = 1;
            }
            return;
        }

        // Survival radio.
        if lay.mode_survival.contains(self.gameplay.mouse_pos) {
            if let Some(ref mut state) = self.gameplay.create_world_state {
                state.game_mode = "survival".into();
            }
            return;
        }

        // Creative radio.
        if lay.mode_creative.contains(self.gameplay.mouse_pos) {
            if let Some(ref mut state) = self.gameplay.create_world_state {
                state.game_mode = "creative".into();
            }
            return;
        }

        // Allow Cheats toggle.
        if lay.cheats_toggle.contains(self.gameplay.mouse_pos) {
            if let Some(ref mut state) = self.gameplay.create_world_state {
                state.allow_cheats = !state.allow_cheats;
            }
            return;
        }

        // Cancel button.
        if lay.cancel_btn.contains(self.gameplay.mouse_pos) {
            self.gameplay.create_world_state = None;
            return;
        }

        // Create World button.
        if lay.create_btn.contains(self.gameplay.mouse_pos) {
            let Some(ref state) = self.gameplay.create_world_state else {
                return;
            };
            let name = state.name.trim().to_string();
            if name.is_empty() {
                if let Some(ref mut s) = self.gameplay.create_world_state {
                    s.error = Some("World name cannot be empty".into());
                }
                return;
            }
            let seed = state.seed.trim().to_string();
            let mode = state.game_mode.clone();
            let cheats = state.allow_cheats;
            self.gameplay.create_world_state = None;
            self.create_world_from_dialog(name, seed, mode, cheats);
        }
    }

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

        let panel = crate::screen_layout::DeleteConfirmLayout::panel(w, h);
        let lay = crate::screen_layout::DeleteConfirmLayout::new(w, h);
        ui.nine_slice(
            TILE_PANEL,
            panel.x,
            panel.y,
            panel.w,
            panel.h,
            6.0,
            [225, 205, 205, 255],
        );

        // Title.
        ui.text_shadow(
            "Delete World",
            panel.x + 16.0,
            panel.y + 12.0,
            1.2,
            crate::ui_kit::palette::INK_DANGER,
            &self.render.font,
        );

        // Message.
        let msg1 = "Are you sure you want to delete".to_string();
        let msg2 = format!("\"{}\"?", world_name);
        let msg3 = "This cannot be undone.".to_string();
        ui.text_shadow(
            &msg1,
            panel.x + 16.0,
            panel.y + 38.0,
            0.8,
            crate::ui_kit::palette::INK_MUTED,
            &self.render.font,
        );
        ui.text_shadow(
            &msg2,
            panel.x + 16.0,
            panel.y + 54.0,
            0.8,
            crate::ui_kit::palette::INK,
            &self.render.font,
        );
        ui.text_shadow(
            &msg3,
            panel.x + 16.0,
            panel.y + 72.0,
            0.7,
            [170, 130, 130, 200],
            &self.render.font,
        );

        // Buttons: Cancel + Delete World (shared layout).
        let cancel = lay.cancel_btn;
        let cancel_hovered = cancel.contains(self.gameplay.mouse_pos);
        ui.nine_slice(
            if cancel_hovered {
                TILE_BTN_HOVER
            } else {
                TILE_BTN
            },
            cancel.x,
            cancel.y,
            cancel.w,
            cancel.h,
            4.0,
            [225, 225, 230, 255],
        );
        ui.text_shadow(
            "Cancel",
            cancel.x + 10.0,
            cancel.y + 5.0,
            0.8,
            crate::ui_kit::palette::INK,
            &self.render.font,
        );

        let delete = lay.delete_btn;
        let delete_hovered = delete.contains(self.gameplay.mouse_pos);
        ui.nine_slice(
            if delete_hovered {
                TILE_BTN_HOVER
            } else {
                TILE_BTN
            },
            delete.x,
            delete.y,
            delete.w,
            delete.h,
            4.0,
            [255, 205, 200, 255],
        );
        ui.text_shadow(
            "Delete World",
            delete.x + 8.0,
            delete.y + 5.0,
            0.8,
            crate::ui_kit::palette::INK_DANGER,
            &self.render.font,
        );
    }

    fn handle_delete_confirm_click(&mut self) {
        let Some(idx) = self.gameplay.pending_delete else {
            return;
        };

        let (w, h) = self.render.logical_size();
        let lay = crate::screen_layout::DeleteConfirmLayout::new(w, h);

        // Cancel.
        if lay.cancel_btn.contains(self.gameplay.mouse_pos) {
            self.gameplay.pending_delete = None;
            return;
        }

        // Delete World.
        if lay.delete_btn.contains(self.gameplay.mouse_pos) {
            if idx < self.gameplay.world_list.len() {
                let path = self.gameplay.world_list[idx].path.clone();
                if path.exists() {
                    let _ = std::fs::remove_dir_all(&path);
                    log::info!("deleted world: {}", self.gameplay.world_list[idx].name);
                } else {
                    log::warn!("world path missing, removing entry: {}", path.display());
                }
                self.gameplay.world_list.remove(idx);
                if self.gameplay.selected_world_index == Some(idx) {
                    self.gameplay.selected_world_index = None;
                } else if self.gameplay.selected_world_index.map(|s| s > idx) == Some(true) {
                    self.gameplay.selected_world_index =
                        self.gameplay.selected_world_index.map(|s| s - 1);
                }
            }
            self.gameplay.pending_delete = None;
        }
    }

    fn draw_settings_menu(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        use crate::screen_layout::{
            SettingsRow, KEYBIND_ROW_H, SETTINGS_ROWS, SETTINGS_TOP_PAD, SLIDER_ROW_H, TOGGLE_ROW_H,
        };

        // Dark overlay.
        ui.gradient_v(0.0, 0.0, w, h, [6, 6, 10, 185], [0, 0, 0, 185]);

        // Shared layout: the exact rects `handle_settings_click` hit-tests.
        let lay = crate::screen_layout::SettingsLayout::new(w, h);
        let panel = lay.panel;
        let px = panel.x;
        let py = panel.y;

        let mut kit = crate::ui_kit::UiKit::new(
            ui,
            &self.render.font,
            (self.gameplay.mouse_pos.x, self.gameplay.mouse_pos.y),
        );
        kit.panel(px, py, panel.w, panel.h, [205, 205, 215, 255]);

        // Title + Back button.
        kit.label(
            "OPTIONS",
            px + 16.0,
            py + 12.0,
            1.5,
            crate::ui_kit::palette::INK,
        );
        kit.small_button(lay.back_btn.x, lay.back_btn.y, lay.back_btn.w, "Back");

        // Rows in `SETTINGS_ROWS` order; every hit rect comes from `lay`, so
        // the click handler can never drift from what is drawn.
        let mut si = 0usize;
        let mut ti = 0usize;
        let mut y = py + SETTINGS_TOP_PAD;
        for row in SETTINGS_ROWS {
            match row {
                SettingsRow::Section(title) => {
                    y = self.draw_section_header(kit.ui, title, lay.lx, y, lay.row_w);
                }
                SettingsRow::Slider(..) => {
                    let spec = &lay.sliders[si];
                    let dragging = self.gameplay.settings_slider_dragging == Some(si);
                    let value = self.slider_value(spec.label);
                    self.draw_setting_slider(kit.ui, lay.lx, spec, value, dragging);
                    si += 1;
                    y += SLIDER_ROW_H;
                }
                SettingsRow::Toggle(..) => {
                    let spec = &lay.toggles[ti];
                    let on = self.toggle_on(spec.label);
                    self.draw_setting_toggle(kit.ui, lay.lx, spec, on);
                    ti += 1;
                    y += TOGGLE_ROW_H;
                }
                SettingsRow::Keybind(label) => {
                    let key = self.keybind_label(label);
                    self.draw_keybind_row(kit.ui, label, &key, lay.lx, y);
                    y += KEYBIND_ROW_H;
                }
                SettingsRow::Gap(gap) => y += *gap,
            }
        }

        // Apply / Defaults: atlas chrome sized to the layout rects.
        for (rect, label) in [(lay.apply_btn, "Apply"), (lay.defaults_btn, "Defaults")] {
            let hovered = rect.contains(self.gameplay.mouse_pos);
            kit.ui.nine_slice(
                if hovered { TILE_BTN_HOVER } else { TILE_BTN },
                rect.x,
                rect.y,
                rect.w,
                rect.h,
                4.0,
                [225, 225, 232, 255],
            );
            let lw = self.render.font.text_width(label, 1.0);
            kit.ui.text_shadow(
                label,
                rect.x + (rect.w - lw) * 0.5,
                rect.y + (rect.h - 7.0) * 0.5,
                1.0,
                crate::ui_kit::palette::INK,
                &self.render.font,
            );
        }
        kit.finish();
    }
    pub(crate) fn handle_settings_click(&mut self) {
        let (w, h) = self.render.logical_size();
        let lay = crate::screen_layout::SettingsLayout::new(w, h);
        let mouse = self.gameplay.mouse_pos;

        // Back button.
        if lay.back_btn.contains(mouse) {
            match self.gameplay.settings_previous {
                GameState::TitleScreen | GameState::WorldSelect => self.enter_title_screen(),
                GameState::PauseMenu => self.enter_pause(),
                _ => self.enter_title_screen(),
            }
            return;
        }

        // Toggle clicks.
        for toggle in &lay.toggles {
            if toggle.rect.contains(mouse) {
                match toggle.label {
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

        // Slider clicks -- set value from click position and start dragging.
        for (i, slider) in lay.sliders.iter().enumerate() {
            if slider.rect.contains(mouse) {
                let new_val = slider.value_at(mouse.x);
                self.apply_slider_value(slider.label, new_val);
                self.gameplay.settings_slider_dragging = Some(i);
                return;
            }
        }

        // Apply button: save current settings to config.toml.
        if lay.apply_btn.contains(mouse) {
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
        if lay.defaults_btn.contains(mouse) {
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
                gpu_meshing: self.config.render.gpu_meshing,
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

    /// Current config value for a settings slider (by row label).
    fn slider_value(&self, label: &str) -> f32 {
        match label {
            "Render Distance" => self.config.stream.load_radius as f32,
            "Fog Distance" => self.config.render.fog_distance,
            "Exposure" => self.config.exposure,
            "Mouse Sensitivity" => self.config.player.mouse_sensitivity * 1000.0,
            "Walk Speed" => self.config.player.walk_speed,
            "Fly Speed" => self.config.player.fly_speed,
            "SSAO Radius" => self.config.ssao_radius,
            "SSAO Bias" => self.config.ssao_bias,
            "SSAO Strength" => self.config.ssao_strength,
            "MSAA Samples" => match self.config.render.msaa_samples {
                8 => 3.0,
                4 => 2.0,
                2 => 1.0,
                _ => 0.0,
            },
            _ => 0.0,
        }
    }

    /// Current on/off state for a settings toggle (by row label).
    fn toggle_on(&self, label: &str) -> bool {
        match label {
            "VSync" => self.config.render.vsync,
            "Shadows" => self.config.shadow_enabled,
            "Vignette" => self.config.vignette_strength > 0.0,
            "SSAO" => self.config.ssao_enabled,
            _ => false,
        }
    }

    /// Current key binding for a keybind row (by row label).
    fn keybind_label(&self, label: &str) -> String {
        match label {
            "Chat" => self.config.keybinds.chat.clone(),
            "Fly" => self.config.keybinds.fly.clone(),
            "Pause" => self.config.keybinds.pause.clone(),
            "Block Picker" => self.config.keybinds.block_picker.clone(),
            "Inventory" => self.config.keybinds.inventory.clone(),
            "Edit Mode" => self.config.keybinds.edit_mode.clone(),
            _ => String::new(),
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
        let (w, h) = self.render.logical_size();
        let lay = crate::screen_layout::SettingsLayout::new(w, h);
        let Some(slider) = lay.sliders.get(drag_idx) else {
            self.gameplay.settings_slider_dragging = None;
            return;
        };
        let new_val = slider.value_at(self.gameplay.mouse_pos.x);
        self.apply_slider_value(slider.label, new_val);
    }
    fn draw_section_header(&self, ui: &mut UiDrawData, title: &str, x: f32, y: f32, w: f32) -> f32 {
        ui.text_shadow(
            title,
            x,
            y,
            0.8,
            crate::ui_kit::palette::INK_MUTED,
            &self.render.font,
        );
        ui.quad(x, y + 14.0, w, 1.0, [130, 130, 145, 255]);
        y + 20.0
    }

    /// Draw one settings slider from its shared-layout spec. The label sits
    /// left of the track at `lx`; all track/fill/thumb geometry comes from
    /// `spec`, matching what the click/drag handlers hit-test.
    fn draw_setting_slider(
        &self,
        ui: &mut UiDrawData,
        lx: f32,
        spec: &crate::screen_layout::SliderSpec,
        value: f32,
        is_dragging: bool,
    ) {
        let min = spec.min;
        let max = spec.max;
        let Rect {
            x: bar_x,
            y: bar_y,
            w: bar_w,
            h: bar_h,
        } = spec.rect;

        // Detect hover: mouse over the bar area (with some vertical padding).
        let Point { x: mx, y: my } = self.gameplay.mouse_pos;
        let is_hovered = !is_dragging
            && mx >= bar_x
            && mx <= bar_x + bar_w
            && my >= bar_y - 6.0
            && my <= bar_y + bar_h + 6.0;

        // Label on the light panel: dark ink, brightening slightly on hover.
        let label_color = if is_dragging {
            crate::ui_kit::palette::INK
        } else if is_hovered {
            [72, 68, 82, 255]
        } else {
            crate::ui_kit::palette::INK_MUTED
        };
        ui.text(
            spec.label,
            lx,
            bar_y - 2.0,
            0.8,
            label_color,
            &self.render.font,
        );

        let pct = ((value - min) / (max - min)).clamp(0.0, 1.0);

        // Track background: inset chrome from the UI atlas.
        let track_color = if is_hovered || is_dragging {
            [220, 220, 235, 255]
        } else {
            [180, 180, 195, 255]
        };
        ui.nine_slice(
            voxel_render::TILE_PANEL_INSET,
            bar_x,
            bar_y,
            bar_w,
            bar_h,
            2.0,
            track_color,
        );

        // Fill: brighter when hovered, even brighter when dragging.
        let fill_color = if is_dragging {
            [110, 180, 110, 255]
        } else if is_hovered {
            [95, 165, 95, 255]
        } else {
            [85, 150, 85, 255]
        };
        if bar_w * pct > 3.0 {
            ui.quad(
                bar_x + 1.0,
                bar_y + 1.0,
                (bar_w * pct - 2.0).max(0.0),
                bar_h - 2.0,
                fill_color,
            );
        }

        // Drag handle: small beveled thumb when active.
        if is_dragging || is_hovered {
            let handle_x = bar_x + bar_w * pct - 5.0;
            let handle_y = bar_y - 3.0;
            ui.nine_slice(
                voxel_render::TILE_BTN,
                handle_x,
                handle_y,
                14.0,
                bar_h + 6.0,
                3.0,
                [255, 255, 255, 255],
            );
        }

        let val_text = if spec.label == "MSAA Samples" {
            format!("{}x", 1u32 << value.round() as u32)
        } else if spec.label == "SSAO Bias" {
            format!("{:.3}", value)
        } else {
            format!("{:.1}", value)
        };
        // Value text: dark ink, full-strength while active.
        let val_color = if is_dragging {
            crate::ui_kit::palette::INK
        } else if is_hovered {
            [72, 68, 82, 255]
        } else {
            crate::ui_kit::palette::INK_MUTED
        };
        ui.text(
            &val_text,
            bar_x + bar_w + 6.0,
            bar_y - 2.0,
            0.7,
            val_color,
            &self.render.font,
        );
    }
    /// Draw one settings toggle from its shared-layout spec (inset track +
    /// beveled knob, slid right = on).
    fn draw_setting_toggle(
        &self,
        ui: &mut UiDrawData,
        lx: f32,
        spec: &crate::screen_layout::ToggleSpec,
        on: bool,
    ) {
        ui.text(
            spec.label,
            lx,
            spec.rect.y,
            0.8,
            crate::ui_kit::palette::INK_MUTED,
            &self.render.font,
        );

        let toggle_x = spec.rect.x;
        let toggle_w = spec.rect.w;
        let toggle_h = spec.rect.h;
        let toggle_y = spec.rect.y;

        ui.nine_slice(
            voxel_render::TILE_PANEL_INSET,
            toggle_x,
            toggle_y,
            toggle_w,
            toggle_h,
            3.0,
            if on {
                [170, 220, 170, 255]
            } else {
                [150, 150, 160, 255]
            },
        );
        let knob_w = 14.0;
        let knob_x = if on {
            toggle_x + toggle_w - knob_w
        } else {
            toggle_x
        };
        ui.nine_slice(
            voxel_render::TILE_BTN,
            knob_x,
            toggle_y,
            knob_w,
            toggle_h,
            3.0,
            if on {
                [235, 255, 235, 255]
            } else {
                [200, 200, 205, 255]
            },
        );
    }
    fn draw_keybind_row(&self, ui: &mut UiDrawData, label: &str, key: &str, x: f32, y: f32) -> f32 {
        ui.text(
            label,
            x,
            y + 2.0,
            0.8,
            crate::ui_kit::palette::INK_MUTED,
            &self.render.font,
        );

        let key_x = x + 200.0;
        let key_w = 80.0;
        let key_h = 18.0;
        let key_y = y + 1.0;

        ui.nine_slice(
            voxel_render::TILE_PANEL_INSET,
            key_x,
            key_y,
            key_w,
            key_h,
            3.0,
            [190, 190, 200, 255],
        );
        ui.text_shadow(
            key,
            key_x + 8.0,
            key_y + 3.0,
            0.7,
            crate::ui_kit::palette::TEXT,
            &self.render.font,
        );

        y + crate::screen_layout::KEYBIND_ROW_H
    }
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
            if i == self.simulation.inventory().map_or(0, |inv| inv.selected) {
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
                .simulation
                .inventory()
                .and_then(|inv| inv.hotbar_block(i))
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
                ui.text_shadow(
                    msg,
                    box_x + pad,
                    y,
                    1.0,
                    [225, 225, 225, 255],
                    &self.render.font,
                );
                y += line_h;
            }
        }

        if self.gameplay.chat.open {
            let input_text = format!("> {}", self.gameplay.chat.input_buf);
            ui.text_shadow(
                &input_text,
                box_x + pad,
                y,
                1.0,
                [255, 255, 120, 255],
                &self.render.font,
            );
        }
    }

    /// Draw the developer console overlay.
    fn draw_console(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let panel_h = h * 0.4;
        let panel_y = h - panel_h;

        // Console well: top-to-bottom fade so older scrollback dissolves.
        ui.gradient_v(0.0, panel_y, w, panel_h, [0, 0, 0, 120], [0, 0, 0, 230]);
        ui.quad(0.0, panel_y, w, 2.0, [90, 90, 105, 200]);

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
        ui.text_shadow(
            &display,
            pad,
            prompt_y,
            1.0,
            [130, 255, 130, 255],
            &self.render.font,
        );

        let max_visible = ((panel_h - line_h - pad * 3.0) / line_h) as usize;
        let scrollback = self.gameplay.console.visible_lines(max_visible);
        let mut y = prompt_y - line_h;
        for line in scrollback.iter().rev() {
            ui.text_shadow(line, pad, y, 1.0, [210, 210, 210, 235], &self.render.font);
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
