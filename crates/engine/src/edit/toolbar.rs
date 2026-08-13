//! Category toolbar: 36 px vertical icon strip on the left edge.
//!
//! Each button selects a tool category; the active category is
//! highlighted with the accent color.

use crate::edit::{EditState, ToolCategory};
use voxel_core::Point;
use voxel_render::{FontAtlas, UiDrawData};

use super::theme;

/// Actions returned by category bar clicks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CategoryAction {
    None,
    Select(ToolCategory),
}

/// Draw the category toolbar and return the action if a button was clicked.
pub fn draw_category_bar(
    ui: &mut UiDrawData,
    edit: &EditState,
    screen_h: f32,
    mouse: Point,
    font: &FontAtlas,
) -> CategoryAction {
    let mut action = CategoryAction::None;

    ui.quad(
        0.0,
        theme::MENU_BAR_H,
        theme::CAT_BAR_W,
        screen_h - theme::MENU_BAR_H - theme::STATUS_BAR_H,
        theme::PANEL_BG,
    );
    ui.quad(
        theme::CAT_BAR_W - 1.0,
        theme::MENU_BAR_H,
        1.0,
        screen_h - theme::MENU_BAR_H - theme::STATUS_BAR_H,
        theme::BORDER,
    );

    let btn_size = theme::CAT_BTN_SIZE;
    let gap = 1.0;
    let pad_x = (theme::CAT_BAR_W - btn_size) * 0.5;
    let mut y = theme::MENU_BAR_H + 4.0;

    for &cat in ToolCategory::ALL {
        let is_active = edit.active_category == cat && edit.mode.is_active();
        let Point { x: mx, y: my } = mouse;
        let hovered = mx >= pad_x && mx <= pad_x + btn_size && my >= y && my <= y + btn_size;

        let bg = if is_active {
            theme::ACTIVE_BG
        } else if hovered {
            theme::HOVER_BG
        } else {
            theme::PANEL_BG
        };
        let fg = if is_active {
            theme::ACCENT_LIGHT
        } else if hovered {
            theme::TEXT_SECONDARY
        } else {
            theme::TEXT_DIM
        };

        ui.quad(pad_x, y, btn_size, btn_size, bg);
        let icon = match cat {
            ToolCategory::Selection => "S",
            ToolCategory::Draw => "D",
            ToolCategory::Paint => "P",
            ToolCategory::Heightmap => "H",
            ToolCategory::Manipulation => "M",
            ToolCategory::Utility => "U",
        };
        let lw = font.text_width(icon, 0.9);
        ui.text(
            icon,
            pad_x + (btn_size - lw) * 0.5,
            y + (btn_size - 8.0) * 0.5,
            0.9,
            fg,
            font,
        );

        if hovered && edit.ui_click {
            action = CategoryAction::Select(cat);
        }

        y += btn_size + gap;
    }

    y += 4.0;
    ui.quad(6.0, y, theme::CAT_BAR_W - 12.0, 1.0, theme::BORDER);

    action
}
