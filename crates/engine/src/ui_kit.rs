//! ui_kit: themed immediate-mode widgets for the game GUI.
//!
//! Phase 2 of the GUI overhaul. A thin stateless layer over [`UiDrawData`]
//! that draws Minecraft-style chrome (beveled buttons, inset slots, panels)
//! from the procedural UI atlas and exposes hit-test-ready `Rect`s so the
//! existing `handle_*_click` handlers keep working unchanged.
//!
//! Recipe:
//! ```ignore
//! let mut kit = UiKit::new(&mut ui, &self.render.font, self.gameplay.mouse_pos);
//! kit.dim(0.0, 0.0, w, h, 160);
//! let r = kit.button(cx, cy, 240.0, "BACK TO GAME");
//! ```

#![allow(dead_code)] // toolkit surface grows ahead of the screens that use it

use voxel_core::Rect;
use voxel_render::{
    FontAtlas, UiDrawData, TILE_BTN, TILE_BTN_DISABLED, TILE_BTN_HOVER, TILE_PANEL,
    TILE_PANEL_INSET, TILE_SLOT, TILE_SLOT_HIGHLIGHT, TILE_TOOLTIP,
};

/// Colours shared by all game screens. Values are linear-ish grays plus a
/// few accents; chrome tiles are grayscale so tint does the rest.
pub mod palette {
    /// Slot/panel neutral tint (leaves the grayscale tile as-is).
    pub const NEUTRAL: [u8; 4] = [255, 255, 255, 255];
    /// Iron-gray widget tint.
    pub const WIDGET: [u8; 4] = [200, 200, 205, 255];
    /// Standard dark panel tint (Minecraft dialog gray).
    pub const PANEL: [u8; 4] = [70, 70, 80, 255];
    /// Hover highlight (white flash over grayscale chrome).
    pub const HOVER: [u8; 4] = [255, 255, 255, 255];
    /// Disabled text.
    pub const DISABLED: [u8; 4] = [160, 160, 160, 255];
    /// Primary text ON LIGHT CHROME (buttons, panels). The chrome tiles are
    /// bright gray (fill ≈ 0.8), so light text washes out — menus use dark
    /// ink instead, like Minecraft's dark-on-stone buttons.
    pub const INK: [u8; 4] = [56, 52, 64, 255];
    /// Secondary / label text on light chrome.
    pub const INK_MUTED: [u8; 4] = [96, 92, 104, 255];
    /// Disabled text on light chrome.
    pub const INK_DISABLED: [u8; 4] = [126, 122, 134, 255];
    /// Danger text on light chrome (delete buttons, error lines).
    pub const INK_DANGER: [u8; 4] = [150, 48, 42, 255];
    /// Primary text on dark surfaces (inset panels, chat, tooltips).
    pub const TEXT: [u8; 4] = [255, 255, 255, 255];
    /// Secondary / label text.
    pub const MUTED: [u8; 4] = [176, 176, 176, 255];
    /// Title text (soft yellow like MC's splash text).
    pub const TITLE: [u8; 4] = [255, 224, 128, 255];
    /// "Coming soon" style labels.
    pub const TAG: [u8; 4] = [140, 220, 140, 255];
    /// Danger (quit / delete).
    pub const DANGER: [u8; 4] = [255, 110, 100, 255];
    /// Health bar red.
    pub const HEALTH: [u8; 4] = [232, 60, 60, 255];
    /// Hunger bar amber.
    pub const HUNGER: [u8; 4] = [214, 140, 48, 255];
    /// Air bubble blue.
    pub const AIR: [u8; 4] = [90, 160, 255, 255];
    /// Selected-slot gold frame.
    pub const GOLD: [u8; 4] = [224, 168, 62, 255];
}

/// Hover state for the widget currently being drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WidgetState {
    Normal,
    Hover,
    Pressed,
    Disabled,
}

/// Immediate-mode UI helper. Draw order = z order.
pub struct UiKit<'a> {
    pub ui: &'a mut UiDrawData,
    pub font: &'a FontAtlas,
    pub mouse: (f32, f32),
    /// Tooltips are queued (last one wins) and flushed by `finish()`.
    tooltip: Option<(String, f32, f32)>,
}

impl<'a> UiKit<'a> {
    pub fn new(ui: &'a mut UiDrawData, font: &'a FontAtlas, mouse: (f32, f32)) -> Self {
        Self {
            ui,
            font,
            mouse,
            tooltip: None,
        }
    }

    /// Contains-point helper in logical pixels.
    pub fn hovered(&self, r: Rect) -> bool {
        r.contains(voxel_core::Point::new(self.mouse.0, self.mouse.1))
    }

    /// Push a tooltip to be drawn on top of everything at `finish()`.
    pub fn tooltip(&mut self, text: &str, x: f32, y: f32) {
        self.tooltip = Some((text.to_string(), x, y));
    }

    /// Flush queued tooltips. Call once at the end of the frame's UI build.
    pub fn finish(&mut self) {
        if let Some((text, x, y)) = self.tooltip.take() {
            let tw = self.font.text_width(&text, 1.0) + 16.0;
            let th = 22.0;
            self.ui
                .nine_slice(TILE_TOOLTIP, x, y, tw, th, 3.0, [255, 255, 255, 255]);
            self.ui
                .text(&text, x + 8.0, y + 6.0, 1.0, palette::TEXT, self.font);
        }
    }

    // ── Surfaces ─────────────────────────────────────────────────────────

    /// Full-screen (or rect) dim veil.
    pub fn dim(&mut self, x: f32, y: f32, w: f32, h: f32, alpha: u8) {
        self.ui.quad(x, y, w, h, [0, 0, 0, alpha]);
    }

    /// Solid 9-slice panel (dialog background).
    pub fn panel(&mut self, x: f32, y: f32, w: f32, h: f32, tint: [u8; 4]) {
        self.ui.nine_slice(TILE_PANEL, x, y, w, h, 6.0, tint);
    }

    /// Inset panel (text field / list well).
    pub fn panel_inset(&mut self, x: f32, y: f32, w: f32, h: f32, tint: [u8; 4]) {
        self.ui.nine_slice(TILE_PANEL_INSET, x, y, w, h, 6.0, tint);
    }

    // ── Widgets ──────────────────────────────────────────────────────────

    /// Standard button. Returns its rect (already hit-test ready).
    pub fn button(&mut self, x: f32, y: f32, w: f32, label: &str) -> Rect {
        self.button_state(x, y, w, label, WidgetState::Normal)
    }

    /// Button with explicit state (for disabled entries).
    pub fn button_state(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        label: &str,
        state: WidgetState,
    ) -> Rect {
        let h = button_height();
        let r = Rect::from_xywh(x, y, w, h);
        let hovered = self.hovered(r);
        let (tile, tint) = match state {
            WidgetState::Disabled => (TILE_BTN_DISABLED, palette::WIDGET),
            _ if hovered => (TILE_BTN_HOVER, [255, 255, 255, 255]),
            _ => (TILE_BTN, palette::WIDGET),
        };
        // Text tint: dark ink on the bright beveled chrome; the hover
        // state comes from the chrome, not the text.
        let text_color = match state {
            WidgetState::Disabled => palette::INK_DISABLED,
            _ => palette::INK,
        };
        self.ui.nine_slice(tile, x, y, w, h, 6.0, tint);
        let scale = 1.5;
        let lw = self.font.text_width(label, scale);
        self.ui.text_shadow(
            label,
            x + (w - lw) * 0.5,
            y + (h - 14.0 * scale) * 0.5,
            scale,
            text_color,
            self.font,
        );
        r
    }

    /// Small text button (back buttons, dialog rows).
    pub fn small_button(&mut self, x: f32, y: f32, w: f32, label: &str) -> Rect {
        let h = 24.0;
        let r = Rect::from_xywh(x, y, w, h);
        let hovered = self.hovered(r);
        self.ui.nine_slice(
            if hovered { TILE_BTN_HOVER } else { TILE_BTN },
            x,
            y,
            w,
            h,
            4.0,
            palette::WIDGET,
        );
        let lw = self.font.text_width(label, 1.0);
        self.ui.text_shadow(
            label,
            x + (w - lw) * 0.5,
            y + 6.0,
            1.0,
            palette::INK,
            self.font,
        );
        r
    }

    /// Text label with drop shadow; returns end-x.
    pub fn label(&mut self, text: &str, x: f32, y: f32, scale: f32, color: [u8; 4]) -> f32 {
        self.ui.text_shadow(text, x, y, scale, color, self.font)
    }

    /// Centred label.
    pub fn label_centered(&mut self, text: &str, cx: f32, y: f32, scale: f32, color: [u8; 4]) {
        let lw = self.font.text_width(text, scale);
        self.ui
            .text_shadow(text, cx - lw * 0.5, y, scale, color, self.font);
    }

    /// Inset slot with optional block icon and count. Returns the rect.
    #[allow(clippy::too_many_arguments)]
    pub fn slot(
        &mut self,
        x: f32,
        y: f32,
        size: f32,
        icon_tile: Option<u16>,
        count: Option<u32>,
        highlighted: bool,
    ) -> Rect {
        let r = Rect::from_xywh(x, y, size, size);
        let hovered = self.hovered(r);
        let tile = if highlighted || hovered {
            TILE_SLOT_HIGHLIGHT
        } else {
            TILE_SLOT
        };
        self.ui.sprite_wh(tile, x, y, size, size, palette::NEUTRAL);
        if let Some(tile) = icon_tile {
            let pad = size * 0.15;
            self.ui.block_icon(
                x + pad,
                y + pad,
                size - pad * 2.0,
                size - pad * 2.0,
                tile,
                [255, 255, 255, 255],
            );
        }
        if let Some(count) = count {
            if count > 1 {
                let label = count.to_string();
                let tw = self.font.text_width(&label, 1.0);
                self.ui.text_shadow(
                    &label,
                    x + size - tw - 3.0,
                    y + size - 12.0,
                    1.0,
                    palette::TEXT,
                    self.font,
                );
            }
        }
        r
    }

    /// Highlighted outline frame drawn OVER a slot (selection indicator).
    pub fn sel_frame(&mut self, r: Rect) {
        self.ui.nine_slice(
            voxel_render::TILE_SEL_FRAME,
            r.x - 2.0,
            r.y - 2.0,
            r.w + 4.0,
            r.h + 4.0,
            3.0,
            [255, 255, 255, 255],
        );
    }

    /// Settings slider: label row + inset track + beveled thumb. Returns
    /// `(full_row_rect, track_rect)`.
    pub fn slider(&mut self, x: f32, y: f32, w: f32, t: f32, _dragging: bool) -> (Rect, Rect) {
        let row_h = 34.0;
        let track = Rect::from_xywh(x, y + 16.0, w, 14.0);
        let t = t.clamp(0.0, 1.0);
        self.ui.nine_slice(
            TILE_PANEL_INSET,
            track.x,
            track.y,
            track.w,
            track.h,
            3.0,
            palette::WIDGET,
        );
        // Filled portion brightens the left of the track.
        let fw = track.w * t;
        if fw > 2.0 {
            self.ui.quad(
                track.x + 1.0,
                track.y + 1.0,
                fw - 2.0,
                track.h - 2.0,
                [96, 160, 96, 255],
            );
        }
        // Thumb.
        let thumb_x = (track.x + fw - 8.0).max(track.x);
        self.ui.nine_slice(
            if self.hovered(track) {
                TILE_BTN_HOVER
            } else {
                TILE_BTN
            },
            thumb_x,
            track.y - 3.0,
            16.0,
            track.h + 6.0,
            3.0,
            [255, 255, 255, 255],
        );
        (Rect::from_xywh(x, y, w, row_h), track)
    }

    /// Toggle switch: label left, 40×18 switch right. Returns the toggle rect.
    pub fn toggle(&mut self, label: &str, x: f32, y: f32, w: f32, on: bool) -> Rect {
        let h = 24.0;
        let sw = 40.0;
        let r = Rect::from_xywh(x + w - sw, y, sw, h);
        self.label(label, x, y + 5.0, 1.0, palette::TEXT);
        let tint = if on {
            [150, 220, 150, 255]
        } else {
            [120, 120, 125, 255]
        };
        self.ui
            .nine_slice(TILE_PANEL_INSET, r.x, r.y, sw, h, 3.0, tint);
        // Knob.
        let knob_w = 18.0;
        let kx = if on {
            r.x + sw - knob_w - 2.0
        } else {
            r.x + 2.0
        };
        self.ui.nine_slice(
            TILE_BTN,
            kx,
            r.y + 2.0,
            knob_w,
            h - 4.0,
            3.0,
            if on {
                [220, 255, 220, 255]
            } else {
                [180, 180, 185, 255]
            },
        );
        // On/off marker text.
        let marker = if on { "ON" } else { "OFF" };
        self.label(
            marker,
            r.x + 2.0 + (knob_w - self.font.text_width(marker, 0.7)) * 0.5,
            r.y + 7.0,
            0.7,
            [0, 0, 0, 255],
        );
        r
    }

    /// Scrollbar for list areas. `scroll` 0..1, `visible` fraction of the
    /// content shown. Returns the thumb rect (for future drag).
    pub fn scrollbar(&mut self, x: f32, y: f32, w: f32, h: f32, scroll: f32, visible: f32) -> Rect {
        self.ui.nine_slice(
            voxel_render::TILE_SCROLL_TRACK,
            x,
            y,
            w,
            h,
            3.0,
            [255, 255, 255, 255],
        );
        let visible = visible.clamp(0.05, 1.0);
        let thumb_h = h * visible;
        let max_y = h - thumb_h;
        let ty = y + max_y * scroll.clamp(0.0, 1.0);
        let r = Rect::from_xywh(x + 1.0, ty, w - 2.0, thumb_h);
        self.ui.nine_slice(
            voxel_render::TILE_SCROLL_THUMB,
            r.x,
            r.y,
            r.w,
            r.h,
            3.0,
            [255, 255, 255, 255],
        );
        r
    }
}

/// Standard button height for kit buttons.
pub fn button_height() -> f32 {
    40.0
}

/// Lightweight row-list scrolling: converts a fractional scroll offset into
/// the window of visible rows, clamps the offset, and supplies thumb math
/// for [`UiKit::scrollbar`]. Pure geometry — no state, so draw and input
/// paths can each own their copy and stay in sync.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scroll {
    /// Scroll position in row units, clamped to `0..=max` by [`Scroll::new`].
    pub row: f32,
    /// Total content rows (may exceed the visible window).
    pub total: usize,
    /// Rows visible at once (window size).
    pub visible: usize,
}

impl Scroll {
    /// Largest valid top-row index.
    pub fn max_top(&self) -> usize {
        self.total.saturating_sub(self.visible)
    }

    /// Create a scroll state clamped to a valid range.
    pub fn new(row: f32, total: usize, visible: usize) -> Self {
        let visible = visible.max(1);
        let s = Self {
            row,
            total,
            visible,
        };
        Self {
            row: s.row.clamp(0.0, s.max_top() as f32),
            ..s
        }
    }

    /// Apply a wheel delta (positive = scroll down) and clamp.
    pub fn scroll_by(&self, rows: f32) -> Self {
        Self::new(self.row + rows, self.total, self.visible)
    }

    /// Adjust the scroll so the given content row is visible (for
    /// keyboard/selection navigation). No-op when the row is already in view.
    pub fn scrolled_to_show(&self, index: usize) -> Self {
        let idx = index.min(self.total.saturating_sub(1)) as f32;
        let top = self.row.floor();
        let bottom = self.row + self.visible as f32;
        if idx < top {
            Self::new(idx, self.total, self.visible)
        } else if idx + 1.0 > bottom {
            Self::new(idx + 1.0 - self.visible as f32, self.total, self.visible)
        } else {
            *self
        }
    }

    /// Index of the first visible content row.
    pub fn first(&self) -> usize {
        self.row.round() as usize
    }

    /// Number of rows actually shown in the window.
    pub fn shown(&self) -> usize {
        (self.total - self.first()).min(self.visible)
    }

    /// Content index for the k-th drawn row (for hit-testing).
    pub fn content_index(&self, k: usize) -> usize {
        self.first() + k
    }

    /// Thumb scroll fraction 0..1 for [`UiKit::scrollbar`] (`None` when all
    /// rows fit and no bar should be drawn).
    pub fn thumb(&self) -> Option<f32> {
        let max = self.max_top();
        (max > 0).then(|| self.row / max as f32)
    }

    /// Visible fraction of the content for [`UiKit::scrollbar`].
    pub fn visible_fraction(&self) -> f32 {
        if self.total == 0 {
            1.0
        } else {
            (self.visible as f32 / self.total as f32).min(1.0)
        }
    }
}

/// Step size, in rows, for one wheel notch.
pub const SCROLL_STEP: f32 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn button_rect_matches_hover_hitbox() {
        // The returned rect is what the click handler stores, so the drawn
        // hover state and the hit-test rect must be identical.
        let mut ui = UiDrawData::default();
        let font = FontAtlas::new();
        let mut kit = UiKit::new(&mut ui, &font, (50.0, 55.0));
        let r = kit.button(10.0, 35.0, 200.0, "Test");
        assert_eq!((r.x, r.y, r.w, r.h), (10.0, 35.0, 200.0, button_height()));
        assert!(
            kit.hovered(r),
            "mouse at (50, 55) should be inside the button"
        );
    }

    #[test]
    fn slider_track_inside_row() {
        let mut ui = UiDrawData::default();
        let font = FontAtlas::new();
        let mut kit = UiKit::new(&mut ui, &font, (-1.0, -1.0));
        let (row, track) = kit.slider(0.0, 100.0, 300.0, 0.5, false);
        assert!(track.x >= row.x && track.x + track.w <= row.x + row.w);
        assert!(track.y >= row.y && track.y + track.h <= row.y + row.h);
        assert_eq!(track.w, 300.0);
    }

    #[test]
    fn toggle_returns_switch_rect_on_right_edge() {
        let mut ui = UiDrawData::default();
        let font = FontAtlas::new();
        let mut kit = UiKit::new(&mut ui, &font, (-1.0, -1.0));
        let r = kit.toggle("VSync", 0.0, 0.0, 300.0, true);
        assert_eq!(r.x + r.w, 300.0, "toggle hugs the right edge of the row");
        assert_eq!(r.w, 40.0);
    }

    #[test]
    fn scrollbar_thumb_never_exceeds_track() {
        let mut ui = UiDrawData::default();
        let font = FontAtlas::new();
        let mut kit = UiKit::new(&mut ui, &font, (-1.0, -1.0));
        for scroll in [0.0f32, 0.5, 1.0, 2.0, -1.0] {
            let thumb = kit.scrollbar(0.0, 0.0, 14.0, 200.0, scroll, 0.3);
            assert!(thumb.y >= 0.0 && thumb.y + thumb.h <= 200.0 + 0.01);
        }
    }

    #[test]
    fn nine_slice_produces_9_quads_and_covers_rect() {
        let mut ui = UiDrawData::default();
        let font = FontAtlas::new();
        let mut kit = UiKit::new(&mut ui, &font, (-1.0, -1.0));
        kit.panel(10.0, 20.0, 100.0, 80.0, [255, 255, 255, 255]);
        // 3×3 = 9 quads, each 4 vertices.
        assert_eq!(kit.ui.vertices.len(), 36);
        assert_eq!(kit.ui.indices.len(), 54);
        // Bounding box of vertices equals the requested rect.
        let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
        let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
        for v in &kit.ui.vertices {
            min_x = min_x.min(v.pos[0]);
            min_y = min_y.min(v.pos[1]);
            max_x = max_x.max(v.pos[0]);
            max_y = max_y.max(v.pos[1]);
        }
        assert_eq!((min_x, min_y, max_x, max_y), (10.0, 20.0, 110.0, 100.0));
    }

    #[test]
    fn nine_slice_degenerate_small_rect_still_covered() {
        let mut ui = UiDrawData::default();
        let font = FontAtlas::new();
        let mut kit = UiKit::new(&mut ui, &font, (-1.0, -1.0));
        // Smaller than 2× border: must still cover the full rect exactly.
        kit.panel(0.0, 0.0, 6.0, 5.0, [255, 255, 255, 255]);
        let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
        let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
        for v in &kit.ui.vertices {
            min_x = min_x.min(v.pos[0]);
            min_y = min_y.min(v.pos[1]);
            max_x = max_x.max(v.pos[0]);
            max_y = max_y.max(v.pos[1]);
        }
        assert_eq!((min_x, min_y, max_x, max_y), (0.0, 0.0, 6.0, 5.0));
    }

    #[test]
    fn font_supports_lowercase_and_measures() {
        let font = FontAtlas::new();
        assert!(font.char_uv('a').is_some());
        assert!(font.char_uv('Z').is_some());
        assert!(font.char_uv('?').is_some());
        assert!(font.char_uv('{').is_some());
        assert!(font.char_uv('\u{2588}').is_none());
        assert!(font.text_width("hello", 1.0) > 0.0);
    }

    #[test]
    fn ui_atlas_generates_all_tiles_with_opaque_chrome() {
        let atlas = voxel_render::ui_atlas();
        let tile_px = (atlas.width / voxel_render::UI_ATLAS_COLS) as usize;
        let get = |tile: u32, x: usize, y: usize| -> [u8; 4] {
            let tx = (tile % voxel_render::UI_ATLAS_COLS) as usize * tile_px + x;
            let ty = (tile / voxel_render::UI_ATLAS_COLS) as usize * tile_px + y;
            let i = (ty * atlas.width as usize + tx) * 4;
            [
                atlas.rgba[i],
                atlas.rgba[i + 1],
                atlas.rgba[i + 2],
                atlas.rgba[i + 3],
            ]
        };
        // Button fill is opaque gray.
        let btn = get(voxel_render::TILE_BTN, 8, 8);
        assert_eq!(btn[3], 255);
        // Heart icon has red pixels somewhere in its art block.
        let mut found_red = false;
        for y in 3..12 {
            for x in 3..12 {
                let [r, _, b, a] = get(voxel_render::TILE_HEART_FULL, x, y);
                if a == 255 && r > 150 && r > b + 50 {
                    found_red = true;
                }
            }
        }
        assert!(found_red, "heart icon should contain red pixels");
    }

    // ── Scroll ─────────────────────────────────────────────────────────

    #[test]
    fn scroll_clamps_and_reports_window() {
        let s = Scroll::new(-5.0, 20, 6);
        assert_eq!(s.row, 0.0);
        assert_eq!(s.max_top(), 14);
        let s = s.scroll_by(3.0);
        assert_eq!(s.row, 3.0);
        let s = s.scroll_by(100.0);
        assert_eq!(s.row, 14.0, "clamped to max_top");
        assert_eq!(s.first(), 14);
        assert_eq!(s.shown(), 6);
        assert_eq!(s.content_index(0), 14);
    }

    #[test]
    fn scroll_no_bar_when_content_fits() {
        let s = Scroll::new(0.0, 4, 6);
        assert_eq!(s.thumb(), None);
        assert_eq!(s.max_top(), 0);
        // scrolling is a no-op
        assert_eq!(s.scroll_by(10.0).row, 0.0);
        assert_eq!(s.visible_fraction(), 1.0);
    }

    #[test]
    fn scroll_thumb_spans_full_range() {
        let s = Scroll::new(0.0, 20, 6);
        assert_eq!(s.thumb(), Some(0.0));
        let b = Scroll::new(14.0, 20, 6);
        assert_eq!(b.thumb(), Some(1.0));
        assert!((s.visible_fraction() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn scroll_scrolled_to_show_reveals_selection() {
        let s = Scroll::new(5.0, 30, 6); // rows 5..=10 visible
                                         // Selection above the window scrolls up.
        let up = s.scrolled_to_show(2);
        assert_eq!(up.row, 2.0);
        // Selection below the window scrolls down (row at window bottom).
        let down = s.scrolled_to_show(20);
        assert_eq!(down.row, 15.0, "row 20 becomes the last visible row");
        // Selection already visible: unchanged.
        assert_eq!(s.scrolled_to_show(7).row, 5.0);
        // Edge: selection beyond total clamps to the last row, which is not
        // visible, so the view scrolls to the bottom.
        assert_eq!(s.scrolled_to_show(999).row, s.max_top() as f32);
    }

    #[test]
    fn scroll_empty_list_is_safe() {
        let s = Scroll::new(0.0, 0, 6);
        assert_eq!(s.max_top(), 0);
        assert_eq!(s.thumb(), None);
        assert_eq!(s.shown(), 0);
        assert_eq!(s.visible_fraction(), 1.0);
    }
}
