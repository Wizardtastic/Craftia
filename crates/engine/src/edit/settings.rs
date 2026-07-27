//! Brush settings panel (legacy, now integrated into right_panel).
//!
//! Kept for backward compatibility; the new VoxEdit-style UI uses
//! `right_panel::draw_right_panel` instead.

use crate::edit::EditState;
use voxel_render::{FontAtlas, UiDrawData};

use super::theme;

const PANEL_X: f32 = 52.0;
const PANEL_Y: f32 = 8.0;
const PANEL_W: f32 = 200.0;
const PANEL_H: f32 = 140.0;

pub fn draw_brush_settings(ui: &mut UiDrawData, edit: &EditState, font: &FontAtlas) {
    let brush = match edit.brush_ref() {
        Some(b) => b,
        None => return,
    };

    ui.quad(PANEL_X, PANEL_Y, PANEL_W, PANEL_H, theme::PANEL_BG);
    ui.rect_border(PANEL_X, PANEL_Y, PANEL_W, PANEL_H, 1.0, theme::BORDER);

    ui.text("Brush Settings", PANEL_X + 8.0, PANEL_Y + 4.0, 0.9, theme::TEXT_PRIMARY, font);

    let mut y = PANEL_Y + 22.0;
    let lx = PANEL_X + 8.0;

    ui.text(&format!("Shape: {:?}", brush.shape), lx, y, 0.7, theme::TEXT_SECONDARY, font);
    y += 18.0;
    ui.text(&format!("Radius: {:.1}", brush.radius), lx, y, 0.7, theme::TEXT_SECONDARY, font);
    y += 18.0;
    ui.text(&format!("Block: {}", brush.block.0), lx, y, 0.7, theme::TEXT_SECONDARY, font);
    y += 18.0;

    let replace_text = if brush.replace { "Replace: ON" } else { "Replace: OFF" };
    ui.text(replace_text, lx, y, 0.7, theme::TEXT_SECONDARY, font);
    y += 18.0;

    if brush.replace {
        let target_text = match brush.target {
            Some(t) => format!("Target: {}", t.0),
            None => "Target: any".to_string(),
        };
        ui.text(&target_text, lx, y, 0.7, theme::TEXT_SECONDARY, font);
    }
}
