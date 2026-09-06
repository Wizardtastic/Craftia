//! Pure layout geometry for the world-select and settings screens.
//!
//! Single source of truth shared by the `draw_*` functions and the click /
//! drag handlers, replacing the old "draw stores rects in GamePlayState"
//! pattern. Everything here is a pure function of the logical window size
//! (+ a couple of flags), so hit-testing can never drift from rendering.

use voxel_core::Rect;
use voxel_game::InventorySlot;

// ── Creative picker filtering ────────────────────────────────────────────

/// A creative item as seen by the picker: master-list position plus display
/// data. Shared by draw, click, and wheel so all three agree.
pub(crate) struct CreativeItemRef {
    pub block: voxel_core::BlockId,
    pub name: String,
    pub category: String,
    pub tile: u16,
}

/// Filter the creative item list by the active tab (or search query).
/// Mirrors the tab labels in [`PICKER_TABS`]: last two are "All"/"Search".
pub(crate) fn filter_creative_items(
    items: &[crate::CreativeItem],
    active_tab: usize,
    search: &str,
) -> Vec<CreativeItemRef> {
    let tab_count = PICKER_TABS.len();
    let is_search = active_tab == tab_count - 1;
    let is_all = active_tab == tab_count - 2;
    let q = search.to_ascii_lowercase();
    items
        .iter()
        .enumerate()
        .filter(|(_, it)| {
            if is_search {
                q.is_empty() || it.name.to_ascii_lowercase().contains(&q)
            } else if is_all {
                true
            } else {
                active_tab < tab_count && it.category == PICKER_TABS[active_tab]
            }
        })
        .map(|(_, it)| CreativeItemRef {
            block: it.id,
            name: it.name.clone(),
            category: it.category.clone(),
            tile: it.tile,
        })
        .collect()
}

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

// ── Creative block picker ─────────────────────────────────────────────────

/// Tab labels for the creative picker. Index = `creative_tab`; the label IS
/// the category name used for filtering (the last two are the "All" and
/// "Search" pseudo-tabs).
pub(crate) const PICKER_TABS: [&str; 8] = [
    "Blocks",
    "Nature",
    "Building",
    "Ores",
    "Decoration",
    "Liquids",
    "All",
    "Search",
];

const PICKER_SLOT: f32 = 40.0;
const PICKER_GAP: f32 = 3.0;
pub(crate) const PICKER_COLS: usize = 9;
/// Rows visible in the picker grid before it scrolls.
pub(crate) const PICKER_VISIBLE_ROWS: usize = 5;
const PICKER_TAB_W: f32 = 46.0;
const PICKER_TAB_H: f32 = 20.0;

/// Grid metrics for the block picker under the active tab filter:
/// `(total_rows, visible_rows)`. Shared by the draw pass, the click
/// handler, and the wheel handler so all three clamp identically.
pub(crate) fn creative_scroll_bounds(
    items: &[crate::CreativeItem],
    active_tab: usize,
    search: &str,
) -> (usize, usize) {
    let rows = filter_creative_items(items, active_tab, search)
        .len()
        .div_ceil(PICKER_COLS);
    (rows, rows.min(PICKER_VISIBLE_ROWS))
}

/// Creative picker layout (Minecraft-style: tab strip above a light panel,
/// item grid with scrollbar, player hotbar row at the bottom). Shared by the
/// draw pass and the click handler.
#[derive(Clone, Debug)]
pub(crate) struct PickerLayout {
    pub panel: Rect,
    pub close_btn: Rect,
    /// Tab strip above the panel, in `PICKER_TABS` order.
    pub tabs: Vec<(Rect, &'static str)>,
    /// Search text field (only when the Search tab is active).
    pub search_field: Option<Rect>,
    /// Visible grid window (all visible slot cells fit inside).
    pub grid: Rect,
    pub slot: f32,
    pub gap: f32,
    pub cols: usize,
    pub visible_rows: usize,
    /// Scroll offset in rows: display row `k` is content row
    /// `first_index + k`.
    pub first_index: usize,
    /// Scrollbar track rect when the content overflows the window.
    pub scrollbar: Option<Rect>,
    /// Player hotbar row at the bottom (width = grid width).
    pub hotbar: Rect,
}

impl PickerLayout {
    /// Whether the Search tab is active.
    pub fn is_search(tab: usize) -> bool {
        tab == PICKER_TABS.len() - 1
    }

    pub fn new(w: f32, h: f32, active_tab: usize, visible_rows: usize, first_index: usize) -> Self {
        let is_search = Self::is_search(active_tab);
        let pad = 10.0;
        let grid_w = PICKER_COLS as f32 * (PICKER_SLOT + PICKER_GAP) - PICKER_GAP;
        let panel_w = grid_w + pad * 2.0;
        let title_h = 18.0;
        let search_h = 22.0;
        let grid_h = visible_rows as f32 * (PICKER_SLOT + PICKER_GAP) - PICKER_GAP;
        let panel_h = pad
            + title_h
            + if is_search { search_h + 6.0 } else { 0.0 }
            + grid_h
            + 10.0
            + PICKER_SLOT
            + pad;
        let px = (w - panel_w) * 0.5;
        let py = ((h - panel_h) * 0.5).max(36.0); // leave room for the tab strip

        // Tab strip, centered above the panel.
        let tab_row_w = PICKER_TABS.len() as f32 * PICKER_TAB_W + 2.0;
        let tab_x0 = px + (panel_w - tab_row_w) * 0.5;
        let tab_y = py - PICKER_TAB_H - 4.0;
        let tabs = PICKER_TABS
            .iter()
            .enumerate()
            .map(|(i, label)| {
                (
                    Rect::from_xywh(
                        tab_x0 + i as f32 * (PICKER_TAB_W + 2.0),
                        if i == active_tab { tab_y - 3.0 } else { tab_y },
                        PICKER_TAB_W,
                        if i == active_tab {
                            PICKER_TAB_H + 3.0
                        } else {
                            PICKER_TAB_H
                        },
                    ),
                    *label,
                )
            })
            .collect();

        let mut cy = py + pad + title_h;
        let search_field = is_search.then(|| {
            let r = Rect::from_xywh(px + pad, cy, grid_w, search_h);
            cy += search_h + 6.0;
            r
        });
        let grid = Rect::from_xywh(px + pad, cy, grid_w, grid_h);
        let hotbar = Rect::from_xywh(px + pad, grid.y + grid_h + 10.0, grid_w, PICKER_SLOT);

        let scrollable = true; // caller passes first_index; thumb drawn when total > visible
        let bar_w = 6.0;
        let scrollbar = scrollable
            .then(|| Rect::from_xywh(grid.x + grid.w - bar_w - 2.0, grid.y, bar_w, grid.h));

        Self {
            panel: Rect::from_xywh(px, py, panel_w, panel_h),
            close_btn: Rect::from_xywh(px + panel_w - pad - 18.0, py + pad, 18.0, 18.0),
            tabs,
            search_field,
            grid,
            slot: PICKER_SLOT,
            gap: PICKER_GAP,
            cols: PICKER_COLS,
            visible_rows,
            first_index,
            scrollbar,
            hotbar,
        }
    }

    /// Top-left of the visible slot cell for content item `idx`, if that
    /// item is inside the scrolled window.
    pub fn slot_pos(&self, idx: usize) -> Option<(f32, f32)> {
        let col = idx % self.cols;
        let row = idx / self.cols;
        if row < self.first_index {
            return None;
        }
        let display_row = row - self.first_index;
        if display_row >= self.visible_rows {
            return None;
        }
        Some((
            self.grid.x + col as f32 * (self.slot + self.gap),
            self.grid.y + display_row as f32 * (self.slot + self.gap),
        ))
    }

    /// Content item index under a point, if any.
    pub fn item_at(&self, x: f32, y: f32, item_count: usize) -> Option<usize> {
        if !self.grid.contains(voxel_core::Point::new(x, y)) {
            return None;
        }
        let col = ((x - self.grid.x) / (self.slot + self.gap)) as usize;
        let row = ((y - self.grid.y) / (self.slot + self.gap)) as usize;
        if col >= self.cols || row >= self.visible_rows {
            return None;
        }
        let idx = (self.first_index + row) * self.cols + col;
        (idx < item_count).then_some(idx)
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

// ── Survival inventory (Minecraft-style) ─────────────────────────────────

/// Survival inventory layout, matching Minecraft's player-inventory
/// proportions: a top band with the armor rail, player preview, offhand
/// slot, and the 2×2 crafting grid (arrow + output), then the 3×9 main
/// grid and the 9 hotbar slots separated below. The whole panel sits
/// slightly below centre, like Minecraft's. Shared by the draw pass and
/// the click handler.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InventoryLayout {
    pub panel: Rect,
    /// Left rail of the top band: helmet, chestplate, leggings, boots.
    pub armor: [Rect; 4],
    /// Player preview area (paper doll) beside the armor rail.
    pub player: Rect,
    /// Offhand slot, vertically centred in the band right of the preview.
    pub offhand: Rect,
    /// 2×2 crafting input grid.
    pub crafting: [Rect; 4],
    /// Arrow between the crafting grid and its output (draw only).
    pub craft_arrow: Rect,
    /// Crafting result slot.
    pub craft_output: Rect,
    /// 3×9 main storage grid.
    pub main: [Rect; 27],
    /// Hotbar row (separated, like MC's bottom row).
    pub hotbar: [Rect; 9],
}

impl InventoryLayout {
    const SLOT: f32 = 36.0;
    const GAP: f32 = 4.0;
    /// Downward bias from centre (Minecraft sits the inventory a touch low).
    const Y_BIAS: f32 = 16.0;

    pub fn new(w: f32, h: f32) -> Self {
        let slot = Self::SLOT;
        let gap = Self::GAP;
        let grid_w = 9.0f32 * slot + 8.0 * gap; // 356
        let pad = 8.0;
        let title_h = 20.0;
        // Top band: armor rail | preview | offhand | 2×2 | arrow | output.
        let band_h = 4.0f32 * slot + 3.0 * gap; // 156
        let craft_w = 2.0f32 * slot + gap; // 76
        let arrow_w = 20.0;
        // armor + gap + preview + gap + offhand + gap + craft + arrow + out
        // (the arrow's padding absorbs the gaps around itself), so the output
        // slot ends flush with the main grid's right edge.
        let player_w = grid_w - slot - gap - slot - gap - craft_w - arrow_w - slot - gap;
        let panel_w = pad * 2.0 + grid_w;
        let main_h = 3.0f32 * slot + 2.0 * gap; // 116
        let hotbar_sep = 8.0;
        let band_sep = 10.0;
        let panel_h = pad + title_h + band_h + band_sep + main_h + hotbar_sep + slot + pad;
        let px = (w - panel_w) * 0.5;
        let py = (h - panel_h) * 0.5 + Self::Y_BIAS;

        let x0 = px + pad;
        let band_y = py + pad + title_h;
        let armor: [Rect; 4] = core::array::from_fn(|i| {
            Rect::from_xywh(x0, band_y + i as f32 * (slot + gap), slot, slot)
        });
        let player_x = x0 + slot + gap;
        let player = Rect::from_xywh(player_x, band_y, player_w, band_h);
        let offhand = Rect::from_xywh(
            player_x + player_w + gap,
            band_y + (band_h - slot) * 0.5,
            slot,
            slot,
        );
        let craft_x = offhand.x + slot + gap;
        let craft_y = band_y + (band_h - craft_w) * 0.5;
        let crafting: [Rect; 4] = core::array::from_fn(|i| {
            let col = (i % 2) as f32;
            let row = (i / 2) as f32;
            Rect::from_xywh(
                craft_x + col * (slot + gap),
                craft_y + row * (slot + gap),
                slot,
                slot,
            )
        });
        let arrow = Rect::from_xywh(
            craft_x + craft_w + (arrow_w - 14.0) * 0.5,
            band_y + band_h * 0.5 - 6.0,
            14.0,
            12.0,
        );
        let craft_output = Rect::from_xywh(
            craft_x + craft_w + arrow_w,
            band_y + (band_h - slot) * 0.5,
            slot,
            slot,
        );

        let main_y = band_y + band_h + band_sep;
        let main: [Rect; 27] = core::array::from_fn(|i| {
            let col = (i % 9) as f32;
            let row = (i / 9) as f32;
            Rect::from_xywh(
                x0 + col * (slot + gap),
                main_y + row * (slot + gap),
                slot,
                slot,
            )
        });
        let hotbar_y = main_y + main_h + hotbar_sep;
        let hotbar: [Rect; 9] = core::array::from_fn(|i| {
            Rect::from_xywh(x0 + i as f32 * (slot + gap), hotbar_y, slot, slot)
        });
        Self {
            panel: Rect::from_xywh(px, py, panel_w, panel_h),
            armor,
            player,
            offhand,
            crafting,
            craft_arrow: arrow,
            craft_output,
            main,
            hotbar,
        }
    }

    /// Slot under a logical-pixel point, if any.
    pub fn slot_at(&self, x: f32, y: f32) -> Option<InventorySlot> {
        let pt = voxel_core::Point::new(x, y);
        (0..4)
            .find(|&i| self.armor[i].contains(pt))
            .map(InventorySlot::Armor)
            .or_else(|| self.offhand.contains(pt).then_some(InventorySlot::Offhand))
            .or_else(|| {
                (0..4)
                    .find(|&i| self.crafting[i].contains(pt))
                    .map(InventorySlot::CraftingInput)
            })
            .or_else(|| {
                self.craft_output
                    .contains(pt)
                    .then_some(InventorySlot::CraftingOutput)
            })
            .or_else(|| {
                (0..27)
                    .find(|&i| self.main[i].contains(pt))
                    .map(InventorySlot::Main)
            })
            .or_else(|| {
                (0..9)
                    .find(|&i| self.hotbar[i].contains(pt))
                    .map(InventorySlot::Hotbar)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_hit_testing_matches_drawn_geometry() {
        let lay = InventoryLayout::new(1920.0, 1080.0);
        // Every slot's rect, probed at its center, must resolve back to that
        // same slot — draw and click can never disagree.
        for i in 0..27 {
            let r = lay.main[i];
            assert_eq!(
                lay.slot_at(r.x + r.w * 0.5, r.y + r.h * 0.5),
                Some(InventorySlot::Main(i))
            );
        }
        for i in 0..9 {
            let r = lay.hotbar[i];
            assert_eq!(
                lay.slot_at(r.x + r.w * 0.5, r.y + r.h * 0.5),
                Some(InventorySlot::Hotbar(i))
            );
        }
        for i in 0..4 {
            let r = lay.armor[i];
            assert_eq!(
                lay.slot_at(r.x + r.w * 0.5, r.y + r.h * 0.5),
                Some(InventorySlot::Armor(i))
            );
        }
        assert_eq!(
            lay.slot_at(lay.offhand.x + 10.0, lay.offhand.y + 10.0),
            Some(InventorySlot::Offhand)
        );
        // 2×2 crafting grid + output hit-test back too.
        for i in 0..4 {
            let r = lay.crafting[i];
            assert_eq!(
                lay.slot_at(r.x + r.w * 0.5, r.y + r.h * 0.5),
                Some(InventorySlot::CraftingInput(i))
            );
        }
        assert_eq!(
            lay.slot_at(lay.craft_output.x + 10.0, lay.craft_output.y + 10.0),
            Some(InventorySlot::CraftingOutput)
        );
        // Grid rows must not overlap the hotbar row or each other.
        for pair in lay.main.windows(2) {
            assert!(!pair[0].intersects(pair[1]));
        }
        assert!(!lay.main[26].intersects(lay.hotbar[0]));
        // The top band sits fully above the main grid (nothing overlaps the
        // title row — the bug from the first cut of this screen).
        let band_bottom = lay.armor[3].y + lay.armor[3].h;
        assert!(band_bottom <= lay.main[0].y);
        assert!(lay.craft_output.y + lay.craft_output.h <= lay.main[0].y);
        // Everything lives inside the panel.
        for r in lay
            .main
            .iter()
            .chain(lay.hotbar.iter())
            .chain(lay.armor.iter())
            .chain(lay.crafting.iter())
        {
            assert!(lay.panel.contains(voxel_core::Point::new(r.x, r.y)));
        }
        // Every band element (including the crafting output, which ends the
        // row) must fit inside the panel's right edge.
        let band_right = lay.craft_output.x + lay.craft_output.w;
        assert!(band_right <= lay.panel.x + lay.panel.w);
        // The output slot is flush with the main grid's right column.
        let grid_right = lay.main[8].x + lay.main[8].w;
        assert!((band_right - grid_right).abs() < 0.01);
        // The band's rows must all start inside the panel too.
        assert!(lay.player.x >= lay.panel.x);
        assert!(lay.crafting[0].x >= lay.panel.x);
    }

    #[test]
    fn picker_slot_pos_windows_through_scroll() {
        let lay = PickerLayout::new(1920.0, 1080.0, 6, 5, 2);
        // Content rows 0-1 are scrolled above the window.
        assert_eq!(lay.slot_pos(0), None);
        assert_eq!(lay.slot_pos(17), None);
        // Row 2 (first visible) sits at the grid origin.
        let (x, y) = lay.slot_pos(18).unwrap();
        assert_eq!((x, y), (lay.grid.x, lay.grid.y));
        // Row 6 (last visible row, 5 rows window) is inside; row 7 is not.
        assert!(lay.slot_pos(54).is_some());
        assert_eq!(lay.slot_pos(63), None);
        // item_at maps display position back to content index through the
        // same window (row 1 of the window = content row 3 = index 27+4).
        let col_w = lay.slot + lay.gap;
        assert_eq!(
            lay.item_at(lay.grid.x + 4.0 * col_w, lay.grid.y + 1.0 * col_w, 100),
            Some(31)
        );
    }

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
