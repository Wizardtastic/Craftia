//! Block palette: searchable grid of blocks integrated into the left panel.
//!
//! Shows recently used blocks, category tabs, and a searchable grid.
//! Clicking a block sets it as the brush block.

use crate::edit::{BlockCategory, EditState};
use voxel_core::BlockId;
use voxel_render::{FontAtlas, UiDrawData};
use voxel_world::registry::BlockRegistry;

use super::theme;

const PANEL_W: f32 = 320.0;
const CELL_SIZE: f32 = 36.0;
const CELL_GAP: f32 = 2.0;
const COLS: usize = 8;
const HEADER_H: f32 = 28.0;
const SEARCH_H: f32 = 24.0;
const RECENT_H: f32 = 50.0;
const TABS_H: f32 = 24.0;
const SCROLL_SPEED: u32 = 2;

/// Draw the block palette. Returns Some(block_id) if a block was selected.
pub fn draw_block_palette(
    ui: &mut UiDrawData,
    edit: &mut EditState,
    screen_w: f32,
    screen_h: f32,
    mouse: (f32, f32),
    font: &FontAtlas,
    scroll_delta: f32,
    registry: &BlockRegistry,
) -> Option<BlockId> {
    let panel_x = screen_w - PANEL_W;
    let panel_y = 0.0f32;

    ui.quad(panel_x, panel_y, PANEL_W, screen_h, theme::PANEL_BG);
    ui.rect_border(panel_x, panel_y, PANEL_W, screen_h, 1.0, theme::BORDER);

    ui.text("Block Palette", panel_x + 8.0, 6.0, 1.0, theme::TEXT_PRIMARY, font);
    let close_x = panel_x + PANEL_W - 24.0;
    ui.quad(close_x, 4.0, 20.0, 20.0, [80, 30, 30, 200]);
    ui.text("X", close_x + 4.0, 6.0, 0.8, [255, 200, 200, 255], font);
    let (mx, my) = mouse;
    if mx >= close_x && mx <= close_x + 20.0 && my >= 4.0 && my <= 24.0 && edit.ui_click {
        edit.palette_open = false;
        return None;
    }

    let search_y = HEADER_H;
    let search_x = panel_x + 4.0;
    let search_w = PANEL_W - 8.0;
    let is_search_hover =
        mx >= search_x && mx <= search_x + search_w && my >= search_y && my <= search_y + SEARCH_H;

    if is_search_hover && edit.ui_click {
        edit.search_focused = true;
    } else if edit.ui_click {
        edit.search_focused = false;
    }

    let search_border = if edit.search_focused { theme::ACCENT } else { theme::BORDER };
    ui.quad(search_x, search_y, search_w, SEARCH_H, theme::EDITOR_BG);
    ui.rect_border(search_x, search_y, search_w, SEARCH_H, 1.0, search_border);

    let search_display = if edit.palette_search.is_empty() && !edit.search_focused {
        "Search... (click to type)".to_string()
    } else if edit.palette_search.is_empty() && edit.search_focused {
        "|".to_string()
    } else {
        let mut t = edit.palette_search.clone();
        if edit.search_focused { t.push('|'); }
        t
    };
    let search_color = if edit.palette_search.is_empty() && !edit.search_focused {
        theme::TEXT_DIM
    } else {
        theme::TEXT_PRIMARY
    };
    ui.text(&search_display, search_x + 4.0, search_y + 4.0, 0.8, search_color, font);

    let recent_y = HEADER_H + SEARCH_H + 4.0;
    ui.text("Recent", panel_x + 8.0, recent_y, 0.6, theme::TEXT_SECONDARY, font);
    let recent_items_y = recent_y + 14.0;
    for (i, &block_id) in edit.recently_used.iter().take(8).enumerate() {
        let rx = panel_x + 4.0 + i as f32 * (CELL_SIZE + CELL_GAP);
        let ry = recent_items_y;
        if block_id == brush_block(edit) {
            ui.quad(rx, ry, CELL_SIZE, CELL_SIZE, theme::ACTIVE_BG);
        }
        draw_block_cell(ui, block_id, rx, ry, registry);
        if mx >= rx && mx <= rx + CELL_SIZE && my >= ry && my <= ry + CELL_SIZE && edit.ui_click {
            edit.add_recent(block_id);
            return Some(block_id);
        }
    }

    let tab_y = recent_y + RECENT_H;
    let categories = [
        BlockCategory::All, BlockCategory::Stone, BlockCategory::Wood,
        BlockCategory::Glass, BlockCategory::Plant, BlockCategory::Ore,
        BlockCategory::Dirt, BlockCategory::Sand, BlockCategory::Water,
    ];
    let tab_w = (PANEL_W - 8.0) / categories.len() as f32;
    for (i, cat) in categories.iter().enumerate() {
        let tx = panel_x + 4.0 + i as f32 * tab_w;
        let selected = edit.selected_category == Some(*cat)
            || (edit.selected_category.is_none() && *cat == BlockCategory::All);
        let bg = if selected { theme::ACTIVE_BG } else { theme::ELEMENT_BG };
        ui.quad(tx, tab_y, tab_w, TABS_H, bg);
        let label = format!("{:?}", cat);
        let lw = font.text_width(&label, 0.6);
        ui.text(&label, tx + (tab_w - lw) * 0.5, tab_y + 4.0, 0.6, theme::TEXT_PRIMARY, font);
        if mx >= tx && mx <= tx + tab_w && my >= tab_y && my <= tab_y + TABS_H && edit.ui_click {
            edit.selected_category = if *cat == BlockCategory::All { None } else { Some(*cat) };
        }
    }

    if scroll_delta.abs() > 0.01 && mx >= panel_x {
        if scroll_delta > 0.0 {
            edit.palette_scroll = edit.palette_scroll.saturating_add(SCROLL_SPEED);
        } else {
            edit.palette_scroll = edit.palette_scroll.saturating_sub(SCROLL_SPEED);
        }
    }

    let grid_y = tab_y + TABS_H + 8.0;
    let filtered = filter_blocks(edit);
    let mut selected_block = None;

    for (i, (block_id, name)) in filtered.iter().enumerate() {
        let col = i % COLS;
        let row = i / COLS;
        let bx = panel_x + 4.0 + col as f32 * (CELL_SIZE + CELL_GAP);
        let by = grid_y + row as f32 * (CELL_SIZE + CELL_GAP) - edit.palette_scroll as f32;

        if by < grid_y - CELL_SIZE || by > screen_h { continue; }

        if *block_id == brush_block(edit) {
            ui.quad(bx, by, CELL_SIZE, CELL_SIZE, theme::ACTIVE_BG);
        }

        draw_block_cell(ui, *block_id, bx, by, registry);

        if mx >= bx && mx <= bx + CELL_SIZE && my >= by && my <= by + CELL_SIZE {
            ui.quad(bx, by - 14.0, font.text_width(name, 0.7) + 4.0, 14.0, [0, 0, 0, 200]);
            ui.text(name, bx + 2.0, by - 2.0, 0.7, [255, 255, 200, 255], font);
            if edit.ui_click {
                selected_block = Some(*block_id);
            }
        }
    }

    selected_block
}

fn draw_block_cell(ui: &mut UiDrawData, block_id: BlockId, x: f32, y: f32, registry: &BlockRegistry) {
    ui.quad(x, y, CELL_SIZE, CELL_SIZE, theme::ELEMENT_BG);
    ui.rect_border(x, y, CELL_SIZE, CELL_SIZE, 1.0, theme::BORDER);
    if !block_id.is_air() {
        let def = registry.get(block_id);
        let tile = def.textures.tile(voxel_world::registry::Face::PosX);
        let icon_size = CELL_SIZE - 8.0;
        ui.block_icon(x + 4.0, y + 4.0, icon_size, icon_size, tile, [255, 255, 255, 255]);
    }
}

fn brush_block(edit: &EditState) -> BlockId {
    match &edit.mode {
        crate::edit::EditModeState::Active { tool: crate::edit::EditTool::Brush(b) } => b.block,
        _ => BlockId::AIR,
    }
}

fn filter_blocks(edit: &EditState) -> Vec<(BlockId, String)> {
    let mut out = Vec::new();
    for (cat, blocks) in &edit.categories {
        if let Some(selected) = edit.selected_category {
            if *cat != selected { continue; }
        }
        for (id, name) in blocks {
            if !edit.palette_search.is_empty()
                && !name.to_lowercase().contains(&edit.palette_search.to_lowercase())
            {
                continue;
            }
            out.push((*id, name.clone()));
        }
    }
    out
}
