//! Pure layout geometry for the world-select and settings screens.
//!
//! Single source of truth shared by the `draw_*` functions and the click /
//! drag handlers, replacing the old "draw stores rects in GamePlayState"
//! pattern. Everything here is a pure function of the logical window size
//! (+ a couple of flags), so hit-testing can never drift from rendering.

use voxel_core::Rect;

// ── Title screen ─────────────────────────────────────────────────────────

/// Full title-screen layout (menu buttons; title/subtitle are centred text).
#[derive(Clone, Copy, Debug)]
pub(crate) struct TitleLayout {
    /// Buttons in draw order: Singleplayer, Multiplayer (disabled), Options,
    /// Quit.
    pub buttons: [Rect; 4],
}

impl TitleLayout {
    /// Index of the disabled Multiplayer button.
    pub const MULTIPLAYER: usize = 1;
    pub const SINGLEPLAYER: usize = 0;
    pub const OPTIONS: usize = 2;
    pub const QUIT: usize = 3;

    pub fn new(w: f32, h: f32) -> Self {
        let panel_w = 320.0f32;
        let btn_h = 50.0f32;
        let spacing = 70.0f32;
        let btn_start_y = h * 0.38;
        let btn_x = (w - panel_w) * 0.5;
        let mut buttons = [Rect::from_xywh(0.0, 0.0, 0.0, 0.0); 4];
        for (i, b) in buttons.iter_mut().enumerate() {
            *b = Rect::from_xywh(btn_x, btn_start_y + i as f32 * spacing, panel_w, btn_h);
        }
        Self { buttons }
    }
}

// ── Pause menu ───────────────────────────────────────────────────────────

/// Full pause-menu layout (panel + 4 stacked buttons).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PauseLayout {
    pub panel: Rect,
    /// Buttons in draw order: Back to Game, Options, Save & Quit, Quit.
    pub buttons: [Rect; 4],
}

impl PauseLayout {
    pub const BACK: usize = 0;
    pub const OPTIONS: usize = 1;
    pub const SAVE_QUIT: usize = 2;
    pub const QUIT: usize = 3;

    pub fn new(w: f32, h: f32) -> Self {
        let panel_w = 320.0f32;
        let panel_h = 340.0f32;
        let px = (w - panel_w) * 0.5;
        let py = (h - panel_h) * 0.5;
        let btn_w = 240.0f32;
        let btn_x = px + (panel_w - btn_w) * 0.5;
        let spacing = 52.0f32;
        let btn_y0 = py + 60.0;
        let mut buttons = [Rect::from_xywh(0.0, 0.0, 0.0, 0.0); 4];
        for (i, b) in buttons.iter_mut().enumerate() {
            *b = Rect::from_xywh(btn_x, btn_y0 + spacing * i as f32, btn_w, 40.0);
        }
        Self {
            panel: Rect::from_xywh(px, py, panel_w, panel_h),
            buttons,
        }
    }
}

// ── Settings ─────────────────────────────────────────────────────────────

/// One row of the settings panel, in draw order.
pub(crate) enum SettingsRow {
    /// Blue section header.
    Section(&'static str),
    /// Labelled slider with an inclusive value range.
    Slider(&'static str, f32, f32),
    /// Labelled on/off toggle.
    Toggle(&'static str),
    /// Static keybind label (the key comes from the live config at draw time).
    Keybind(&'static str),
    /// Vertical breathing room.
    Gap(f32),
}

/// The settings panel rows. `draw_settings_menu` renders exactly this
/// sequence; `build_settings_layout` computes rects from the same list.
pub(crate) const SETTINGS_ROWS: &[SettingsRow] = &[
    SettingsRow::Section("GRAPHICS"),
    SettingsRow::Slider("Render Distance", 2.0, 16.0),
    SettingsRow::Toggle("VSync"),
    SettingsRow::Slider("Fog Distance", 100.0, 800.0),
    SettingsRow::Slider("Exposure", 0.1, 3.0),
    SettingsRow::Toggle("Shadows"),
    SettingsRow::Toggle("Vignette"),
    SettingsRow::Toggle("SSAO"),
    SettingsRow::Slider("SSAO Radius", 0.5, 5.0),
    SettingsRow::Slider("SSAO Bias", 0.001, 0.1),
    SettingsRow::Slider("SSAO Strength", 0.0, 3.0),
    SettingsRow::Slider("MSAA Samples", 0.0, 3.0),
    SettingsRow::Gap(8.0),
    SettingsRow::Section("PLAYER"),
    SettingsRow::Slider("Mouse Sensitivity", 0.5, 10.0),
    SettingsRow::Slider("Walk Speed", 1.0, 10.0),
    SettingsRow::Slider("Fly Speed", 5.0, 50.0),
    SettingsRow::Gap(8.0),
    SettingsRow::Section("CONTROLS"),
    SettingsRow::Keybind("Chat"),
    SettingsRow::Keybind("Fly"),
    SettingsRow::Keybind("Pause"),
    SettingsRow::Keybind("Block Picker"),
    SettingsRow::Keybind("Inventory"),
    SettingsRow::Keybind("Edit Mode"),
];

/// Row height of a settings slider row.
pub(crate) const SLIDER_ROW_H: f32 = 24.0;
/// Row height of a settings toggle row.
pub(crate) const TOGGLE_ROW_H: f32 = 22.0;
/// Row height of a keybind row.
pub(crate) const KEYBIND_ROW_H: f32 = 22.0;
/// Height of a section header.
const SECTION_H: f32 = 20.0;
/// Top padding (title row) inside the settings panel.
pub(crate) const SETTINGS_TOP_PAD: f32 = 44.0;
/// Bottom button strip height inside the settings panel.
const SETTINGS_BTN_STRIP: f32 = 48.0;

/// One interactive settings slider: track rect + label + range.
#[derive(Clone, Debug)]
pub(crate) struct SliderSpec {
    pub rect: Rect,
    pub label: &'static str,
    pub min: f32,
    pub max: f32,
}

impl SliderSpec {
    /// Value for a pointer position inside the track (clamped to range).
    pub fn value_at(&self, x: f32) -> f32 {
        let pct = ((x - self.rect.x) / self.rect.w).clamp(0.0, 1.0);
        self.min + pct * (self.max - self.min)
    }
}

/// One interactive settings toggle: switch rect + label.
#[derive(Clone, Debug)]
pub(crate) struct ToggleSpec {
    pub rect: Rect,
    pub label: &'static str,
}

/// Full settings-screen layout.
#[derive(Clone, Debug)]
pub(crate) struct SettingsLayout {
    pub panel: Rect,
    pub back_btn: Rect,
    /// Sliders in `SETTINGS_ROWS` order (indices used for drag state).
    pub sliders: Vec<SliderSpec>,
    /// Toggles in `SETTINGS_ROWS` order.
    pub toggles: Vec<ToggleSpec>,
    pub apply_btn: Rect,
    pub defaults_btn: Rect,
    /// Content left edge (for label alignment).
    pub lx: f32,
    /// Content row width.
    pub row_w: f32,
}

impl SettingsLayout {
    pub fn new(w: f32, h: f32) -> Self {
        let panel_w = 500.0;
        // Height from the actual row list so new rows can never overflow.
        let rows_h: f32 = SETTINGS_ROWS
            .iter()
            .map(|r| match r {
                SettingsRow::Section(_) => SECTION_H,
                SettingsRow::Slider(..) => SLIDER_ROW_H,
                SettingsRow::Toggle(_) => TOGGLE_ROW_H,
                SettingsRow::Keybind(_) => KEYBIND_ROW_H,
                SettingsRow::Gap(g) => *g,
            })
            .sum();
        let panel_h = SETTINGS_TOP_PAD + rows_h + SETTINGS_BTN_STRIP;
        let px = (w - panel_w) * 0.5;
        let py = (h - panel_h) * 0.5;
        let lx = px + 16.0;
        let row_w = panel_w - 32.0;

        let mut sliders = Vec::new();
        let mut toggles = Vec::new();
        let mut y = py + SETTINGS_TOP_PAD;
        for row in SETTINGS_ROWS {
            match row {
                SettingsRow::Section(_) => y += SECTION_H,
                SettingsRow::Slider(label, min, max) => {
                    let bar_x = lx + row_w * 0.55;
                    let bar_w = row_w * 0.35;
                    sliders.push(SliderSpec {
                        rect: Rect::from_xywh(bar_x, y + 4.0, bar_w, 8.0),
                        label,
                        min: *min,
                        max: *max,
                    });
                    y += SLIDER_ROW_H;
                }
                SettingsRow::Toggle(label) => {
                    toggles.push(ToggleSpec {
                        rect: Rect::from_xywh(lx + 280.0, y + 2.0, 36.0, 16.0),
                        label,
                    });
                    y += TOGGLE_ROW_H;
                }
                SettingsRow::Keybind(_) => y += KEYBIND_ROW_H,
                SettingsRow::Gap(g) => y += *g,
            }
        }

        let btn_y = py + panel_h - 40.0;
        let btn_w = 100.0;
        Self {
            panel: Rect::from_xywh(px, py, panel_w, panel_h),
            back_btn: Rect::from_xywh(px + panel_w - 70.0, py + 8.0, 60.0, 24.0),
            sliders,
            toggles,
            apply_btn: Rect::from_xywh(px + panel_w - btn_w * 2.0 - 20.0, btn_y, btn_w, 28.0),
            defaults_btn: Rect::from_xywh(px + panel_w - btn_w - 8.0, btn_y, btn_w, 28.0),
            lx,
            row_w,
        }
    }
}

// ── World select ─────────────────────────────────────────────────────────

const WS_PANEL_W: f32 = 600.0;
const WS_PANEL_H: f32 = 500.0;
const WS_LIST_Y: f32 = 40.0; // inside panel
const WS_LIST_H: f32 = 400.0;
const WS_ROW_H: f32 = 60.0;

/// Full world-select screen layout. `rows` holds the entries visible in
/// the scrolled window (content index `first_index + k` for `rows[k]`);
/// `delete_buttons` is parallel to `rows`.
#[derive(Clone, Debug)]
pub(crate) struct WorldSelectLayout {
    pub panel: Rect,
    pub close_btn: Rect,
    /// List content area (scrollbar draws along its right edge).
    pub list: Rect,
    /// Visible row rects (window over the world list).
    pub rows: Vec<Rect>,
    /// Delete button rects parallel to `rows`.
    pub delete_buttons: Vec<Rect>,
    /// World index of `rows[0]` — the current scroll offset.
    pub first_index: usize,
    pub create_btn: Rect,
    pub play_btn: Rect,
}

impl WorldSelectLayout {
    /// Rows that fit the list area, regardless of world count. Scroll
    /// clamping and the wheel/keyboard handlers use this so they agree
    /// with the windowing here.
    pub fn capacity() -> usize {
        (WS_LIST_H / WS_ROW_H) as usize
    }

    pub fn new(w: f32, h: f32, world_count: usize, scroll: usize) -> Self {
        let px = (w - WS_PANEL_W) * 0.5;
        let py = (h - WS_PANEL_H) * 0.5;
        let capacity = Self::capacity();
        let first = scroll.min(world_count.saturating_sub(capacity));
        let shown = world_count.saturating_sub(first).min(capacity);
        let list_y = py + WS_LIST_Y;
        let list = Rect::from_xywh(px + 8.0, list_y, WS_PANEL_W - 16.0, WS_LIST_H);
        let mut rows = Vec::with_capacity(shown);
        let mut delete_buttons = Vec::with_capacity(shown);
        for k in 0..shown {
            let ry = list_y + k as f32 * WS_ROW_H;
            rows.push(Rect::from_xywh(
                px + 8.0,
                ry,
                WS_PANEL_W - 16.0,
                WS_ROW_H - 4.0,
            ));
            delete_buttons.push(Rect::from_xywh(
                px + WS_PANEL_W - 60.0,
                ry + 10.0,
                40.0,
                20.0,
            ));
        }
        let btn_y = list_y + WS_LIST_H + 4.0;
        Self {
            panel: Rect::from_xywh(px, py, WS_PANEL_W, WS_PANEL_H),
            close_btn: Rect::from_xywh(px + WS_PANEL_W - 30.0, py + 8.0, 22.0, 22.0),
            list,
            rows,
            delete_buttons,
            first_index: first,
            create_btn: Rect::from_xywh(px + 8.0, btn_y, 160.0, 28.0),
            play_btn: Rect::from_xywh(px + WS_PANEL_W - 180.0 - 8.0, btn_y, 180.0, 28.0),
        }
    }

    /// How many world rows are actually shown.
    pub fn visible_rows(&self) -> usize {
        self.rows.len()
    }
}

// ── Create-world dialog ──────────────────────────────────────────────────

/// Create-world dialog layout (drawn over world select).
#[derive(Clone, Debug)]
pub(crate) struct CreateWorldLayout {
    pub panel: Rect,
    pub close_btn: Rect,
    pub name_input: Rect,
    pub seed_input: Rect,
    pub mode_survival: Rect,
    pub mode_creative: Rect,
    pub cheats_toggle: Rect,
    pub cancel_btn: Rect,
    pub create_btn: Rect,
    /// Whether the Create button is clickable (non-empty name).
    pub create_enabled: bool,
}

impl CreateWorldLayout {
    pub fn new(w: f32, h: f32, name_is_empty: bool) -> Self {
        let panel_w = 400.0;
        let panel_h = 320.0;
        let px = (w - panel_w) * 0.5;
        let py = (h - panel_h) * 0.5;
        let lx = px + 16.0;
        let input_w = panel_w - 32.0;
        let input_h = 24.0;

        let name_y = py + 42.0 + 16.0;
        let seed_y = name_y + input_h + 8.0 + 16.0;
        let radio_y = seed_y + input_h + 10.0 + 16.0;
        let surv_x = lx + 10.0;
        let cheats_y = radio_y + 22.0;
        let btn_y = py + panel_h - 40.0;
        Self {
            panel: Rect::from_xywh(px, py, panel_w, panel_h),
            close_btn: Rect::from_xywh(px + panel_w - 28.0, py + 8.0, 20.0, 20.0),
            name_input: Rect::from_xywh(lx, name_y, input_w, input_h),
            seed_input: Rect::from_xywh(lx, seed_y, input_w, input_h),
            mode_survival: Rect::from_xywh(surv_x, radio_y, 80.0, 16.0),
            mode_creative: Rect::from_xywh(surv_x + 100.0, radio_y, 80.0, 16.0),
            cheats_toggle: Rect::from_xywh(surv_x, cheats_y, 120.0, 16.0),
            cancel_btn: Rect::from_xywh(px + panel_w - 80.0 - 120.0 - 20.0, btn_y, 80.0, 28.0),
            create_btn: Rect::from_xywh(px + panel_w - 120.0 - 8.0, btn_y, 120.0, 28.0),
            create_enabled: !name_is_empty,
        }
    }
}

// ── Delete-confirm dialog ────────────────────────────────────────────────

/// Delete-confirmation dialog layout (drawn over world select).
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeleteConfirmLayout {
    pub cancel_btn: Rect,
    pub delete_btn: Rect,
}

impl DeleteConfirmLayout {
    pub fn new(w: f32, h: f32) -> Self {
        let panel_w = 340.0;
        let panel_h = 160.0;
        let px = (w - panel_w) * 0.5;
        let py = (h - panel_h) * 0.5;
        let btn_y = py + panel_h - 36.0;
        Self {
            cancel_btn: Rect::from_xywh(px + panel_w - 70.0 - 100.0 - 16.0, btn_y, 70.0, 26.0),
            delete_btn: Rect::from_xywh(px + panel_w - 100.0 - 8.0, btn_y, 100.0, 26.0),
        }
    }

    /// Panel rect (draw uses it for the background).
    pub fn panel(w: f32, h: f32) -> Rect {
        Rect::from_xywh((w - 340.0) * 0.5, (h - 160.0) * 0.5, 340.0, 160.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_sliders_and_toggles_match_row_list() {
        let lay = SettingsLayout::new(1920.0, 1080.0);
        let n_sliders = SETTINGS_ROWS
            .iter()
            .filter(|r| matches!(r, SettingsRow::Slider(..)))
            .count();
        let n_toggles = SETTINGS_ROWS
            .iter()
            .filter(|r| matches!(r, SettingsRow::Toggle(_)))
            .count();
        assert_eq!(lay.sliders.len(), n_sliders);
        assert_eq!(lay.toggles.len(), n_toggles);
        // Names must match apply/lookup in the same order.
        let slider_names: Vec<_> = lay.sliders.iter().map(|s| s.label).collect();
        assert_eq!(
            slider_names,
            vec![
                "Render Distance",
                "Fog Distance",
                "Exposure",
                "SSAO Radius",
                "SSAO Bias",
                "SSAO Strength",
                "MSAA Samples",
                "Mouse Sensitivity",
                "Walk Speed",
                "Fly Speed",
            ]
        );
        let toggle_names: Vec<_> = lay.toggles.iter().map(|t| t.label).collect();
        assert_eq!(toggle_names, vec!["VSync", "Shadows", "Vignette", "SSAO"]);
    }

    #[test]
    fn settings_panel_contains_all_content() {
        let lay = SettingsLayout::new(1280.0, 720.0);
        for s in &lay.sliders {
            assert!(s.rect.y >= lay.panel.y && s.rect.y + s.rect.h <= lay.panel.y + lay.panel.h);
        }
        for t in &lay.toggles {
            assert!(t.rect.y >= lay.panel.y && t.rect.y + t.rect.h <= lay.panel.y + lay.panel.h);
        }
        assert!(lay.apply_btn.y + lay.apply_btn.h <= lay.panel.y + lay.panel.h);
        assert!(lay.defaults_btn.y + lay.defaults_btn.h <= lay.panel.y + lay.panel.h);
    }

    #[test]
    fn slider_value_at_spans_range() {
        let lay = SettingsLayout::new(1920.0, 1080.0);
        let s = &lay.sliders[0]; // Render Distance 2..16
        assert_eq!(s.value_at(s.rect.x), 2.0);
        assert_eq!(s.value_at(s.rect.x + s.rect.w), 16.0);
        // Midpoint is `min + pct * (max - min)`, so compare with an epsilon
        // (exact equality fails on float rounding: 9.000002).
        let mid = s.value_at(s.rect.x + s.rect.w * 0.5);
        assert!((mid - 9.0).abs() < 1e-3, "midpoint {mid} != 9.0");
        // Clamped outside.
        assert_eq!(s.value_at(s.rect.x - 100.0), 2.0);
        assert_eq!(s.value_at(s.rect.x + s.rect.w + 100.0), 16.0);
    }

    #[test]
    fn world_select_visible_rows_cap_at_list_height() {
        let lay = WorldSelectLayout::new(1920.0, 1080.0, 20, 0);
        assert_eq!(lay.visible_rows(), 6);
        assert_eq!(lay.rows.len(), 6);
        assert_eq!(lay.delete_buttons.len(), 6);
        // Buttons sit below the last row, inside the panel.
        let last = lay.rows.last().unwrap();
        assert!(lay.create_btn.y > last.y + last.h);
        assert!(lay.create_btn.y + lay.create_btn.h <= lay.panel.y + lay.panel.h);
    }

    #[test]
    fn world_select_few_worlds_show_all() {
        let lay = WorldSelectLayout::new(1920.0, 1080.0, 2, 0);
        assert_eq!(lay.visible_rows(), 2);
    }

    #[test]
    fn world_select_scroll_windows_the_list() {
        // Offset clamped to what keeps a full window of rows.
        let lay = WorldSelectLayout::new(1920.0, 1080.0, 20, 99);
        assert_eq!(lay.first_index, 20 - 6);
        assert_eq!(lay.visible_rows(), 6);
        assert_eq!(lay.rows[0].y, lay.list.y);
        // First visible content row is the scroll offset; parallel arrays
        // index the same content row.
        let lay = WorldSelectLayout::new(1920.0, 1080.0, 20, 4);
        assert_eq!(lay.first_index, 4);
        assert_eq!(lay.rows.len(), 6);
        // Fewer worlds than capacity: offset clamps to 0 and shows all.
        let lay = WorldSelectLayout::new(1920.0, 1080.0, 3, 2);
        assert_eq!(lay.first_index, 0);
        assert_eq!(lay.visible_rows(), 3);
        // Empty list is safe.
        let lay = WorldSelectLayout::new(1920.0, 1080.0, 0, 0);
        assert_eq!(lay.visible_rows(), 0);
    }

    #[test]
    fn world_select_capacity_matches_layout_window() {
        assert_eq!(WorldSelectLayout::capacity(), 6);
        let lay = WorldSelectLayout::new(1920.0, 1080.0, 100, 0);
        assert_eq!(lay.rows.len(), WorldSelectLayout::capacity());
    }

    #[test]
    fn delete_buttons_inside_rows() {
        let lay = WorldSelectLayout::new(1920.0, 1080.0, 6, 0);
        for (row, del) in lay.rows.iter().zip(lay.delete_buttons.iter()) {
            assert!(row.contains(voxel_core::Point::new(del.x + 1.0, del.y + 1.0)));
        }
    }

    #[test]
    fn create_dialog_buttons_do_not_overlap() {
        let lay = CreateWorldLayout::new(1920.0, 1080.0, false);
        assert!(lay.cancel_btn.x + lay.cancel_btn.w <= lay.create_btn.x);
        assert!(!lay.create_btn.intersects(lay.seed_input));
        assert!(lay.create_btn.y + lay.create_btn.h <= lay.panel.y + lay.panel.h);
    }

    #[test]
    fn title_buttons_are_evenly_spaced() {
        let lay = TitleLayout::new(1920.0, 1080.0);
        for (a, b) in lay.buttons.iter().zip(lay.buttons.iter().skip(1)) {
            let gap = b.y - a.y;
            assert!(
                (gap - 70.0).abs() < 1e-2,
                "button spacing {gap} != 70.0 (float rounding from h * 0.38)"
            );
            assert_eq!((a.x, a.w), (b.x, b.w));
        }
        // All buttons sit on-screen for reasonable window sizes.
        for b in &lay.buttons {
            assert!(b.y >= 0.0 && b.y + b.h <= 1080.0);
        }
    }

    #[test]
    fn pause_buttons_stack_inside_panel() {
        let lay = PauseLayout::new(1280.0, 720.0);
        for b in &lay.buttons {
            assert!(b.x >= lay.panel.x && b.x + b.w <= lay.panel.x + lay.panel.w);
            assert!(b.y >= lay.panel.y && b.y + b.h <= lay.panel.y + lay.panel.h);
        }
        for (a, b) in lay.buttons.iter().zip(lay.buttons.iter().skip(1)) {
            assert!(a.y + a.h <= b.y, "buttons must not overlap");
        }
        assert!(
            lay.buttons[PauseLayout::QUIT].y + lay.buttons[PauseLayout::QUIT].h
                <= lay.panel.y + lay.panel.h
        );
    }

    #[test]
    fn delete_confirm_buttons_do_not_overlap() {
        let lay = DeleteConfirmLayout::new(1920.0, 1080.0);
        assert!(lay.cancel_btn.x + lay.cancel_btn.w <= lay.delete_btn.x);
        let panel = DeleteConfirmLayout::panel(1920.0, 1080.0);
        assert!(lay.delete_btn.y + lay.delete_btn.h <= panel.y + panel.h);
    }
}
