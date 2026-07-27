//! Status bar: 22 px strip across the bottom of the editor.
//!
//! Left: keyboard shortcuts. Right: selection summary + flight speed.

use crate::edit::EditState;
use voxel_render::{FontAtlas, UiDrawData};

use super::theme;

pub fn draw_status_bar(
    ui: &mut UiDrawData,
    _edit: &EditState,
    screen_w: f32,
    screen_h: f32,
    font: &FontAtlas,
) {
    let bar_y = screen_h - theme::STATUS_BAR_H;

    ui.quad(0.0, bar_y, screen_w, theme::STATUS_BAR_H, theme::HEADER_BG);
    ui.quad(0.0, bar_y, screen_w, 1.0, theme::BORDER);

    let shortcuts: &[(&str, &str)] = &[
        ("Ctrl+Z", "Undo"),
        ("Ctrl+Y", "Redo"),
        ("Enter", "Confirm"),
        ("Del", "Delete"),
        ("Ctrl+C", "Copy"),
        ("Ctrl+V", "Paste"),
    ];

    let mut sx = 8.0;
    let text_y = bar_y + 4.0;
    for (key, label) in shortcuts {
        let kw = font.text_width(key, 0.6) + 8.0;
        let kh = 14.0;
        ui.quad(sx, text_y - 1.0, kw, kh, theme::ELEMENT_BG);
        ui.rect_border(sx, text_y - 1.0, kw, kh, 1.0, theme::BORDER_LIGHT);
        ui.text(
            key,
            sx + 4.0,
            text_y + 1.0,
            0.6,
            theme::TEXT_SECONDARY,
            font,
        );
        sx += kw + 2.0;
        ui.text(label, sx, text_y + 1.0, 0.65, theme::TEXT_DIM, font);
        sx += font.text_width(label, 0.65) + 10.0;
    }

    let sel_text = "Sel: --";
    let sel_w = font.text_width(sel_text, 0.7);
    let sel_x = screen_w - sel_w - 100.0;
    ui.quad(
        sel_x - 8.0,
        bar_y + 3.0,
        1.0,
        theme::STATUS_BAR_H - 6.0,
        theme::BORDER,
    );
    ui.text(sel_text, sel_x, text_y, 0.7, theme::ACCENT_LIGHT, font);

    let speed_label = "Flight Speed";
    let speed_x = sel_x - font.text_width(speed_label, 0.65) - 60.0;
    ui.quad(
        speed_x - 8.0,
        bar_y + 3.0,
        1.0,
        theme::STATUS_BAR_H - 6.0,
        theme::BORDER,
    );
    ui.text(speed_label, speed_x, text_y, 0.65, theme::TEXT_DIM, font);

    let slider_x = speed_x + font.text_width(speed_label, 0.65) + 6.0;
    let slider_w = 48.0;
    let slider_h = 12.0;
    let slider_y = bar_y + (theme::STATUS_BAR_H - slider_h) * 0.5;
    ui.quad(slider_x, slider_y, slider_w, slider_h, theme::ELEMENT_BG);
    ui.rect_border(slider_x, slider_y, slider_w, slider_h, 1.0, theme::BORDER);
    ui.quad(slider_x, slider_y, slider_w * 0.3, slider_h, theme::ACCENT);
    let speed_val = "3x";
    let vw = font.text_width(speed_val, 0.6);
    ui.text(
        speed_val,
        slider_x + (slider_w - vw) * 0.5,
        slider_y + 1.0,
        0.6,
        theme::TEXT_PRIMARY,
        font,
    );
}
