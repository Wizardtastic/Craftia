//! Left tool panel: 185 px wide, sits between the category bar and viewport.
//!
//! Contains (top to bottom):
//!   1. Tool section header + tool grid (4 columns)
//!   2. Active block section
//!   3. Palette grid (6 columns)
//!   4. History section (fills remaining space)

use crate::edit::ctx::PanelCtx;
use crate::edit::{tools_for_category, EditState};
use crate::ui::Row;
use voxel_core::BlockId;
use voxel_render::{FontAtlas, UiDrawData};
use voxel_world::registry::BlockRegistry;

use super::theme;

#[derive(Clone, Debug, PartialEq)]
pub enum LeftPanelAction {
    None,
    SelectTool(String),
    SelectBlock(BlockId),
}

pub fn draw_left_panel(
    ui: &mut UiDrawData,
    edit: &mut EditState,
    screen_h: f32,
    mouse: (f32, f32),
    font: &FontAtlas,
    registry: &BlockRegistry,
) -> LeftPanelAction {
    let mut action = LeftPanelAction::None;
    let panel_x = theme::CAT_BAR_W;
    let panel_y = theme::MENU_BAR_H;
    let panel_w = theme::LEFT_PANEL_W;
    let panel_h = screen_h - theme::MENU_BAR_H - theme::STATUS_BAR_H;

    ui.quad(panel_x, panel_y, panel_w, panel_h, theme::PANEL_BG);
    ui.quad(
        panel_x + panel_w - 1.0,
        panel_y,
        1.0,
        panel_h,
        theme::BORDER,
    );

    let (_mx, _my) = mouse;
    let mut y = panel_y;

    // 1. Tool section
    y = draw_section_header(
        ui,
        &format!("{} Tools", edit.active_category.label()),
        panel_x,
        y,
        panel_w,
        font,
    );
    let tools = tools_for_category(edit.active_category);
    y = draw_tool_grid(
        ui,
        tools,
        &edit.active_tool_id,
        Row {
            x: panel_x,
            y,
            w: panel_w,
        },
        PanelCtx {
            font,
            mx: mouse.0,
            my: mouse.1,
            ui_click: edit.ui_click,
        },
        &mut action,
    );

    // 2. Active block section
    y = draw_section_header(ui, "Active Block", panel_x, y, panel_w, font);
    y = draw_active_block(ui, edit, panel_x, y, panel_w, font, registry);

    // 3. Palette section
    y = draw_palette_section(
        ui,
        edit,
        Row {
            x: panel_x,
            y,
            w: panel_w,
        },
        PanelCtx {
            font,
            mx: mouse.0,
            my: mouse.1,
            ui_click: edit.ui_click,
        },
        registry,
        &mut action,
    );

    // 4. History section
    draw_history_section(
        ui,
        edit,
        panel_x,
        y,
        panel_x + panel_w,
        screen_h - theme::STATUS_BAR_H,
        font,
    );

    action
}

fn draw_section_header(
    ui: &mut UiDrawData,
    title: &str,
    x: f32,
    y: f32,
    w: f32,
    font: &FontAtlas,
) -> f32 {
    let h = theme::SECTION_HEADER_H;
    ui.quad(x, y, w, h, theme::HEADER_BG);
    ui.quad(x, y + h - 1.0, w, 1.0, theme::BORDER);
    ui.text(title, x + 7.0, y + 3.0, 0.65, theme::TEXT_SECONDARY, font);
    ui.text(
        "\u{25BE}",
        x + w - 16.0,
        y + 3.0,
        0.55,
        theme::TEXT_DIM,
        font,
    );
    y + h
}

fn draw_tool_grid(
    ui: &mut UiDrawData,
    tools: &[super::ToolDef],
    active_id: &str,
    rect: Row,
    ctx: PanelCtx<'_>,
    action: &mut LeftPanelAction,
) -> f32 {
    let PanelCtx {
        font,
        mx,
        my,
        ui_click,
    } = ctx;
    let Row { x, y, w } = rect;
    let cols = theme::TOOL_GRID_COLS;
    let pad = 4.0;
    let gap = 2.0;
    let cell_w = (w - pad * 2.0 - gap * (cols - 1) as f32) / cols as f32;
    let cell_h = cell_w;

    let cy = y + pad;
    for (i, tool) in tools.iter().enumerate() {
        let col = i % cols;
        let row = i / cols;
        let bx = x + pad + col as f32 * (cell_w + gap);
        let by = cy + row as f32 * (cell_h + gap);

        let is_active = active_id == tool.id;
        let hovered = mx >= bx && mx <= bx + cell_w && my >= by && my <= by + cell_h;

        let bg = if is_active {
            theme::ACTIVE_BG
        } else if hovered {
            theme::HOVER_BG
        } else {
            theme::ELEMENT_BG
        };
        let border = if is_active {
            theme::ACCENT
        } else {
            theme::BORDER
        };
        let fg = if is_active {
            theme::ACCENT_LIGHT
        } else if hovered {
            theme::TEXT_SECONDARY
        } else {
            theme::TEXT_DIM
        };

        ui.quad(bx, by, cell_w, cell_h, bg);
        ui.rect_border(bx, by, cell_w, cell_h, 1.0, border);

        let icon = &tool.label[..1];
        let iw = font.text_width(icon, 0.9);
        ui.text(icon, bx + (cell_w - iw) * 0.5, by + 4.0, 0.9, fg, font);

        let label = tool.label;
        let lw = font.text_width(label, 0.45);
        if lw <= cell_w {
            ui.text(
                label,
                bx + (cell_w - lw) * 0.5,
                by + cell_h - 10.0,
                0.45,
                fg,
                font,
            );
        }

        if hovered && ui_click {
            *action = LeftPanelAction::SelectTool(tool.id.to_string());
        }
    }

    let rows = tools.len().div_ceil(cols);
    cy + rows as f32 * (cell_h + gap) + pad
}

fn draw_active_block(
    ui: &mut UiDrawData,
    edit: &EditState,
    x: f32,
    y: f32,
    _w: f32,
    font: &FontAtlas,
    registry: &BlockRegistry,
) -> f32 {
    let pad = 7.0;
    let block = edit.brush_ref().map(|b| b.block).unwrap_or(BlockId::AIR);
    let def = registry.get(block);
    let name = def.name.as_ref();

    let icon_size = 34.0;
    let bx = x + pad;
    let by = y + pad;

    ui.quad(bx, by, icon_size, icon_size, theme::ELEMENT_BG);
    ui.rect_border(bx, by, icon_size, icon_size, 1.0, theme::BORDER_LIGHT);
    if !block.is_air() {
        let tile = def.textures.tile(voxel_world::registry::Face::PosX);
        ui.block_icon(
            bx + 2.0,
            by + 2.0,
            icon_size - 4.0,
            icon_size - 4.0,
            tile,
            [255, 255, 255, 255],
        );
    }

    let text_x = bx + icon_size + 6.0;
    ui.text(name, text_x, by + 4.0, 0.75, theme::TEXT_PRIMARY, font);
    let internal = format!("{}", block.0);
    ui.text(&internal, text_x, by + 18.0, 0.6, theme::TEXT_DIM, font);

    y + icon_size + pad * 2.0
}

fn draw_palette_section(
    ui: &mut UiDrawData,
    edit: &mut EditState,
    rect: Row,
    ctx: PanelCtx<'_>,
    registry: &BlockRegistry,
    action: &mut LeftPanelAction,
) -> f32 {
    let PanelCtx {
        font,
        mx,
        my,
        ui_click: _,
    } = ctx;
    let Row { x, mut y, w } = rect;
    let header_h = theme::SECTION_HEADER_H;
    ui.quad(x, y, w, header_h, theme::HEADER_BG);
    ui.quad(x, y + header_h - 1.0, w, 1.0, theme::BORDER);
    ui.text(
        "Palette",
        x + 7.0,
        y + 3.0,
        0.65,
        theme::TEXT_SECONDARY,
        font,
    );

    let plus_x = x + w - 22.0;
    let plus_hovered = mx >= plus_x && mx <= plus_x + 16.0 && my >= y && my <= y + header_h;
    ui.text(
        "+",
        plus_x + 2.0,
        y + 3.0,
        0.75,
        if plus_hovered {
            theme::TEXT_PRIMARY
        } else {
            theme::TEXT_DIM
        },
        font,
    );
    ui.text(
        "\u{25BE}",
        plus_x - 12.0,
        y + 3.0,
        0.55,
        theme::TEXT_DIM,
        font,
    );

    y += header_h;

    let pad = 4.0;
    let cell = theme::PALETTE_CELL;
    let gap = theme::PALETTE_GAP;
    let cols = theme::PALETTE_COLS;

    let blocks: Vec<BlockId> = if edit.recently_used.is_empty() {
        vec![BlockId::AIR; 8]
    } else {
        edit.recently_used.clone()
    };

    let brush_block = edit.brush_ref().map(|b| b.block).unwrap_or(BlockId::AIR);

    for (i, &block_id) in blocks.iter().take(12).enumerate() {
        let col = i % cols;
        let row = i / cols;
        let bx = x + pad + col as f32 * (cell + gap);
        let by = y + pad + row as f32 * (cell + gap);

        let is_selected = block_id == brush_block;
        let hovered = mx >= bx && mx <= bx + cell && my >= by && my <= by + cell;

        let border = if is_selected {
            theme::ACCENT
        } else if hovered {
            theme::BORDER_LIGHT
        } else {
            theme::BORDER
        };

        ui.quad(bx, by, cell, cell, theme::ELEMENT_BG);
        ui.rect_border(bx, by, cell, cell, 1.0, border);

        if !block_id.is_air() {
            let def = registry.get(block_id);
            let tile = def.textures.tile(voxel_world::registry::Face::PosX);
            ui.block_icon(
                bx + 2.0,
                by + 2.0,
                cell - 4.0,
                cell - 4.0,
                tile,
                [255, 255, 255, 255],
            );
        } else {
            ui.text(
                "+",
                bx + cell * 0.5 - 3.0,
                by + cell * 0.5 - 5.0,
                0.6,
                theme::TEXT_DIM,
                font,
            );
        }

        if hovered && edit.ui_click && !block_id.is_air() {
            *action = LeftPanelAction::SelectBlock(block_id);
        }
    }

    let palette_rows = blocks.len().div_ceil(cols);
    y + pad * 2.0 + palette_rows as f32 * (cell + gap)
}

fn draw_history_section(
    ui: &mut UiDrawData,
    edit: &EditState,
    x: f32,
    y: f32,
    right_x: f32,
    bottom_y: f32,
    font: &FontAtlas,
) {
    let w = right_x - x;

    let header_h = theme::SECTION_HEADER_H;
    ui.quad(x, y, w, header_h, theme::HEADER_BG);
    ui.quad(x, y + header_h - 1.0, w, 1.0, theme::BORDER);
    ui.text(
        "History",
        x + 7.0,
        y + 3.0,
        0.65,
        theme::TEXT_SECONDARY,
        font,
    );
    ui.text(
        "\u{25BE}",
        x + w - 16.0,
        y + 3.0,
        0.55,
        theme::TEXT_DIM,
        font,
    );

    let mut hy = y + header_h;
    let line_h = 18.0;

    for entry in &edit.history {
        if hy + line_h > bottom_y {
            break;
        }

        let is_cur = entry.is_current;
        if is_cur {
            ui.quad(x, hy, w, line_h, theme::ACCENT_BG);
            ui.quad(x, hy, 2.0, line_h, theme::ACCENT);
        }

        let icon = if is_cur { "\u{25CF}" } else { "\u{25CB}" };
        let icon_color = if is_cur {
            theme::ACCENT_LIGHT
        } else {
            theme::TEXT_DIM
        };
        ui.text(icon, x + 8.0, hy + 2.0, 0.55, icon_color, font);

        let label_color = if is_cur {
            theme::TEXT_PRIMARY
        } else {
            theme::TEXT_SECONDARY
        };
        ui.text(&entry.label, x + 20.0, hy + 2.0, 0.65, label_color, font);

        hy += line_h;
    }
}
