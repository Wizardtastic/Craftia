//! VoxEdit-style visual theme: color palette and layout constants.
//!
//! Colors sourced from the HTML mockup. All are `[u8; 4]` RGBA.

// ── Backgrounds ──────────────────────────────────────────────────────
#[allow(dead_code)]
pub const EDITOR_BG: [u8; 4] = [20, 20, 20, 255];
pub const PANEL_BG: [u8; 4] = [30, 30, 30, 255];
pub const HEADER_BG: [u8; 4] = [37, 37, 37, 255];
pub const ELEMENT_BG: [u8; 4] = [40, 40, 40, 255];
pub const HOVER_BG: [u8; 4] = [50, 50, 50, 255];
pub const ACTIVE_BG: [u8; 4] = [22, 51, 86, 255];

// ── Borders ──────────────────────────────────────────────────────────
pub const BORDER: [u8; 4] = [48, 48, 48, 255];
pub const BORDER_LIGHT: [u8; 4] = [61, 61, 61, 255];

// ── Text ─────────────────────────────────────────────────────────────
pub const TEXT_PRIMARY: [u8; 4] = [204, 204, 204, 255];
pub const TEXT_SECONDARY: [u8; 4] = [136, 136, 136, 255];
pub const TEXT_DIM: [u8; 4] = [80, 80, 80, 255];

// ── Accent ───────────────────────────────────────────────────────────
pub const ACCENT: [u8; 4] = [74, 127, 212, 255];
pub const ACCENT_LIGHT: [u8; 4] = [96, 144, 228, 255];
pub const ACCENT_BG: [u8; 4] = [74, 127, 212, 36];

// ── Coordinate axis colors (future gizmo rendering) ──────────────────
// AXIS_X, AXIS_Y, AXIS_Z reserved for future gizmo rendering.

// ── Layout constants (pixels) ────────────────────────────────────────
pub const MENU_BAR_H: f32 = 22.0;
pub const CAT_BAR_W: f32 = 36.0;
pub const LEFT_PANEL_W: f32 = 185.0;
pub const RIGHT_PANEL_W: f32 = 190.0;
pub const STATUS_BAR_H: f32 = 22.0;
// VIEW_TAB_H reserved for future view tab bar.

// ── Widget sizes ─────────────────────────────────────────────────────
pub const CAT_BTN_SIZE: f32 = 28.0;
pub const TOOL_GRID_COLS: usize = 4;
pub const PALETTE_COLS: usize = 6;
pub const PALETTE_CELL: f32 = 28.0;
pub const PALETTE_GAP: f32 = 2.0;
pub const SLIDER_H: f32 = 14.0;
pub const DROPDOWN_H: f32 = 20.0;
pub const SECTION_HEADER_H: f32 = 20.0;
pub const OPTION_ROW_H: f32 = 22.0;
