//! Menu bar: 22 px strip across the top of the editor.
//!
//! Left: logo + menu items. Right: FPS counter + Exit Editor button.

use crate::edit::EditState;
use voxel_render::{FontAtlas, UiDrawData};

use super::theme;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuAction {
    None,
    ExitEditor,
}

pub fn draw_menu_bar(
    ui: &mut UiDrawData,
    edit: &EditState,
    screen_w: f32,
    mouse: (f32, f32),
    font: &FontAtlas,
    fps: f64,
) -> MenuAction {
    let mut action = MenuAction::None;
    let bar_h = theme::MENU_BAR_H;

    ui.quad(0.0, 0.0, screen_w, bar_h, theme::HEADER_BG);
    ui.quad(0.0, bar_h - 1.0, screen_w, 1.0, theme::BORDER);

    let logo = "VoxEdit";
    let logo_x = 8.0;
    ui.text(logo, logo_x, 4.0, 0.85, theme::TEXT_PRIMARY, font);
    let logo_w = font.text_width(logo, 0.85);
    ui.quad(logo_x + logo_w + 8.0, 2.0, 1.0, bar_h - 4.0, theme::BORDER);

    let menus = ["File", "Edit", "Select", "View", "Operations", "Help"];
    let mut mx = logo_x + logo_w + 16.0;
    let (mouse_x, mouse_y) = mouse;

    for label in &menus {
        let lw = font.text_width(label, 0.75);
        let item_w = lw + 18.0;
        let hovered = mouse_x >= mx && mouse_x <= mx + item_w && mouse_y >= 0.0 && mouse_y <= bar_h;

        if hovered {
            ui.quad(mx, 0.0, item_w, bar_h, theme::HOVER_BG);
        }

        ui.text(
            label,
            mx + 9.0,
            4.0,
            0.75,
            if hovered {
                theme::TEXT_PRIMARY
            } else {
                theme::TEXT_SECONDARY
            },
            font,
        );
        mx += item_w;
    }

    let fps_text = format!("{:.0} fps", fps);
    let fps_w = font.text_width(&fps_text, 0.7);
    let fps_x = screen_w - fps_w - 100.0;
    ui.text(&fps_text, fps_x, 4.0, 0.7, theme::TEXT_DIM, font);
    ui.quad(fps_x + fps_w + 6.0, 4.0, 1.0, bar_h - 8.0, theme::BORDER);

    let btn_label = "Exit Editor";
    let btn_w = font.text_width(btn_label, 0.7) + 14.0;
    let btn_x = screen_w - btn_w - 8.0;
    let btn_y = 3.0;
    let btn_h = bar_h - 6.0;
    let btn_hovered = mouse_x >= btn_x
        && mouse_x <= btn_x + btn_w
        && mouse_y >= btn_y
        && mouse_y <= btn_y + btn_h;

    let btn_bg = if btn_hovered {
        theme::ACCENT_LIGHT
    } else {
        theme::ACCENT
    };
    ui.quad(btn_x, btn_y, btn_w, btn_h, btn_bg);
    ui.rect_border(btn_x, btn_y, btn_w, btn_h, 1.0, theme::ACCENT);
    ui.text(
        btn_label,
        btn_x + 7.0,
        btn_y + 3.0,
        0.7,
        [255, 255, 255, 255],
        font,
    );

    if btn_hovered && edit.ui_click {
        action = MenuAction::ExitEditor;
    }

    action
}
