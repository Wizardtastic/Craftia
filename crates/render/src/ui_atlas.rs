//! Procedural UI atlas: buttons, slots, panels, and HUD icons.
//!
//! Everything is generated on the CPU once at startup (no asset files) and
//! uploaded as the 4th UI sampler (`tex_id = 3` in `shaders/ui.frag`).
//!
//! Design: chrome tiles are drawn in *grayscale* so widget colours come from
//! the per-vertex tint at draw time (same trick as Minecraft's gray button
//! texture × tint). HUD icons (hearts, hunger, …) carry their own colours.
//!
//! Layout: a 16×2 grid of 16×16 tiles. `ui_tile_uv` converts a tile index to
//! UVs; `nine_slice` splits a tile into a 4px-border 9-patch.

use voxel_core::ATLAS_TILE_SIZE;

use crate::atlas::Atlas;

/// UI atlas tile grid (columns × rows).
pub const UI_ATLAS_COLS: u32 = 16;
pub const UI_ATLAS_ROWS: u32 = 2;
pub const UI_ATLAS_TILES: u32 = UI_ATLAS_COLS * UI_ATLAS_ROWS;

/// Nine-slice border thickness inside a chrome tile, in atlas pixels.
pub const NSLICE: f32 = 4.0;

// ── Tile indices (row 0) ─────────────────────────────────────────────────
pub const TILE_BTN: u32 = 0;
pub const TILE_BTN_HOVER: u32 = 1;
pub const TILE_BTN_PRESSED: u32 = 2;
pub const TILE_BTN_DISABLED: u32 = 3;
pub const TILE_SLOT: u32 = 4;
pub const TILE_SLOT_HIGHLIGHT: u32 = 5;
pub const TILE_PANEL: u32 = 6;
pub const TILE_PANEL_INSET: u32 = 7;
pub const TILE_TOOLTIP: u32 = 8;
pub const TILE_SCROLL_TRACK: u32 = 9;
pub const TILE_SCROLL_THUMB: u32 = 10;
pub const TILE_HUD_FRAME: u32 = 11;
pub const TILE_SEL_FRAME: u32 = 12;
pub const TILE_HEART_BG: u32 = 13;
pub const TILE_HEART_FULL: u32 = 14;
pub const TILE_HEART_HALF: u32 = 15;
// ── Tile indices (row 1) ─────────────────────────────────────────────────
pub const TILE_HUNGER_BG: u32 = 16;
pub const TILE_HUNGER_FULL: u32 = 17;
pub const TILE_HUNGER_HALF: u32 = 18;
pub const TILE_ARMOR_BG: u32 = 19;
pub const TILE_ARMOR_FULL: u32 = 20;
pub const TILE_ARMOR_HALF: u32 = 21;
pub const TILE_BUBBLE_BG: u32 = 22;
pub const TILE_BUBBLE_FULL: u32 = 23;
pub const TILE_CROSSHAIR: u32 = 24;
pub const TILE_ARROW_UP: u32 = 25;
pub const TILE_ARROW_DOWN: u32 = 26;

/// UV rect of a UI-atlas tile as `(u0, v0, u1, v1)`.
pub fn ui_tile_uv(tile: u32) -> (f32, f32, f32, f32) {
    let tx = tile % UI_ATLAS_COLS;
    let ty = tile / UI_ATLAS_COLS;
    let u0 = tx as f32 / UI_ATLAS_COLS as f32;
    let v0 = ty as f32 / UI_ATLAS_ROWS as f32;
    let u1 = (tx + 1) as f32 / UI_ATLAS_COLS as f32;
    let v1 = (ty + 1) as f32 / UI_ATLAS_ROWS as f32;
    (u0, v0, u1, v1)
}

type Px = [u8; 4];

/// Paint a 16×16 tile: `f(x, y) -> color` for each pixel.
fn paint_tile(rgba: &mut [u8], tile: u32, f: impl Fn(u32, u32) -> Px) {
    let tx = (tile % UI_ATLAS_COLS) * ATLAS_TILE_SIZE;
    let ty = (tile / UI_ATLAS_COLS) * ATLAS_TILE_SIZE;
    for y in 0..ATLAS_TILE_SIZE {
        for x in 0..ATLAS_TILE_SIZE {
            let c = f(x, y);
            let idx = (((ty + y) * UI_ATLAS_COLS * ATLAS_TILE_SIZE + tx + x) * 4) as usize;
            rgba[idx] = c[0];
            rgba[idx + 1] = c[1];
            rgba[idx + 2] = c[2];
            rgba[idx + 3] = c[3];
        }
    }
}

/// Grayscale pixel (v = 0..1 brightness, a = alpha 0..255).
fn g(v: f32, a: u8) -> Px {
    let b = (v.clamp(0.0, 1.0) * 255.0) as u8;
    [b, b, b, a]
}

/// Raised bevel button (grayscale; tint at draw time). `pressed` inverts the
/// bevel, `dim` flattens it for disabled buttons.
fn bevel_button(pressed: bool, dim: bool) -> impl Fn(u32, u32) -> Px {
    move |x, y| {
        let edge_dark = x == 15 || y == 15;
        let edge_light = x == 0 || y == 0;
        let fill_top = 0.96f32;
        let fill_bot = 0.80f32;
        let fill = fill_top - (fill_top - fill_bot) * (y as f32 / 15.0);
        if dim {
            let v = if edge_light || edge_dark { 0.42 } else { 0.55 };
            g(v, 255)
        } else if pressed {
            if edge_light {
                g(0.38, 255)
            } else if edge_dark {
                g(0.92, 255)
            } else {
                g(fill - 0.10, 255)
            }
        } else if edge_light {
            g(1.0, 255)
        } else if edge_dark {
            g(0.42, 255)
        } else {
            g(fill, 255)
        }
    }
}

/// Inset slot: dark interior with an inverted bevel ring (grayscale).
fn slot_px(x: u32, y: u32, highlight: bool) -> Px {
    let outer = x == 0 || y == 0 || x == 15 || y == 15;
    let ring = x == 1 || y == 1 || x == 14 || y == 14;
    if outer {
        return g(0.06, 255);
    }
    if ring {
        // Inverted bevel: dark top/left, light bottom/right.
        let v = if x == 1 || y == 1 { 0.38 } else { 0.85 };
        return g(v, 255);
    }
    if highlight {
        return g(0.75, 255);
    }
    g(0.28, 255)
}

/// Solid panel with a black outline and raised bevel edge.
fn panel_px(x: u32, y: u32) -> Px {
    let outer = x == 0 || y == 0 || x == 15 || y == 15;
    let edge = x <= 1 || y <= 1 || x >= 14 || y >= 14;
    if outer {
        return g(0.0, 255);
    }
    if edge {
        // Raised bevel: light top/left, dark bottom/right.
        let v = if x <= 1 || y <= 1 { 0.98 } else { 0.55 };
        return g(v, 255);
    }
    g(0.78, 255)
}

/// Inset panel (text fields, list backgrounds): black outline, inverted bevel.
fn panel_inset_px(x: u32, y: u32) -> Px {
    let outer = x == 0 || y == 0 || x == 15 || y == 15;
    let edge = x <= 1 || y <= 1 || x >= 14 || y >= 14;
    if outer {
        return g(0.0, 255);
    }
    if edge {
        let v = if x <= 1 || y <= 1 { 0.30 } else { 0.90 };
        return g(v, 255);
    }
    g(0.16, 255)
}

/// Tooltip: translucent dark fill with an opaque dark border. Light text
/// stays readable on it, unlike the panel chrome.
fn tooltip_px(x: u32, y: u32) -> Px {
    let border = x == 0 || y == 0 || x == 15 || y == 15;
    if border {
        [46, 44, 54, 255]
    } else {
        [22, 22, 30, 220]
    }
}

/// Scrollbar track / thumb.
fn scroll_px(thumb: bool) -> impl Fn(u32, u32) -> Px {
    move |x, y| {
        let outer = x == 0 || y == 0 || x == 15 || y == 15;
        if thumb {
            if outer {
                g(0.05, 255)
            } else {
                g(0.72, 255)
            }
        } else {
            if outer {
                g(0.0, 255)
            } else {
                g(0.14, 255)
            }
        }
    }
}

/// Translucent hotbar frame: opaque-ish bevel border, dark glassy interior.
fn hud_frame_px(x: u32, y: u32) -> Px {
    let border = x <= 1 || y <= 1 || x >= 14 || y >= 14;
    let outer = x == 0 || y == 0 || x == 15 || y == 15;
    if border {
        g(if outer { 0.28 } else { 0.55 }, 255)
    } else {
        g(0.0, 120)
    }
}

/// Selection frame: bright 1px border, transparent interior.
fn sel_frame_px(x: u32, y: u32) -> Px {
    let border = x == 0 || y == 0 || x == 15 || y == 15;
    if border {
        [255, 255, 255, 255]
    } else {
        [255, 255, 255, 40]
    }
}

// ── Pixel-art HUD icons (9×9 string art inside the 16×16 tile) ───────────

/// Paint a string-art icon in one pass (paint_tile per pixel is wasteful
/// when called per art pixel; this variant writes all pixels of one tile).
fn paint_art_tile<'a>(
    art: &'a [&str; 9],
    palette: &'a [(&str, Px); 4],
) -> impl Fn(u32, u32) -> Px + 'a {
    move |x, y| {
        if !(3..12).contains(&x) || !(3..12).contains(&y) {
            return [0, 0, 0, 0];
        }
        let col = (x - 3) as usize;
        let row = (y - 3) as usize;
        let ch = art[row].as_bytes()[col] as char;
        if ch == '.' {
            return [0, 0, 0, 0];
        }
        palette
            .iter()
            .find(|(k, _)| k.starts_with(ch))
            .map(|&(_, c)| c)
            .unwrap_or([0, 0, 0, 0])
    }
}

/// Overlay the full-colour art onto the left half only — the "half" variants
/// (half heart / half drumstick) of the HUD icons.
fn paint_half_tile<'a>(
    bg_art: &'a [&str; 9],
    full_art: &'a [&str; 9],
    bg_palette: &'a [(&str, Px); 4],
    full_palette: &'a [(&str, Px); 4],
) -> impl Fn(u32, u32) -> Px + 'a {
    move |x, y| {
        let px = paint_art_tile(bg_art, bg_palette)(x, y);
        if x <= 7 {
            let over = paint_art_tile(full_art, full_palette)(x, y);
            if over[3] > 0 {
                return over;
            }
        }
        px
    }
}

const HEART_ART: [&str; 9] = [
    ".........",
    "..##.##..",
    ".#WWRRR#.",
    ".#WRRRR#.",
    ".#RRRRR#.",
    "..#RRR#..",
    "...#R#...",
    "....#....",
    ".........",
];
const HEART_PALETTE: [(&str, Px); 4] = [
    ("#", [40, 10, 10, 255]),
    ("R", [220, 40, 40, 255]),
    ("W", [255, 150, 150, 255]),
    ("r", [150, 20, 20, 255]),
];
const HEART_BG_PALETTE: [(&str, Px); 4] = [
    ("#", [60, 60, 60, 255]),
    ("R", [30, 30, 30, 255]),
    ("W", [45, 45, 45, 255]),
    ("r", [30, 30, 30, 255]),
];

const HUNGER_ART: [&str; 9] = [
    ".........",
    "....####.",
    "...#RRRR#",
    "..#RRRRR#",
    "..#RRRR#.",
    ".#B#RR#..",
    "#BB#.....",
    ".##......",
    ".........",
];
const HUNGER_PALETTE: [(&str, Px); 4] = [
    ("#", [50, 25, 10, 255]),
    ("R", [185, 110, 35, 255]),
    ("B", [230, 220, 200, 255]),
    ("r", [120, 70, 20, 255]),
];
const HUNGER_BG_PALETTE: [(&str, Px); 4] = [
    ("#", [55, 55, 55, 255]),
    ("R", [28, 28, 28, 255]),
    ("B", [45, 45, 45, 255]),
    ("r", [28, 28, 28, 255]),
];

const ARMOR_ART: [&str; 9] = [
    ".........",
    ".#.....#.",
    "###...###",
    "#MMMMMMM#",
    ".#MMMMM#.",
    ".#MMMMM#.",
    ".#MMMMM#.",
    "..#MMM#..",
    ".........",
];
const ARMOR_PALETTE: [(&str, Px); 4] = [
    ("#", [55, 58, 65, 255]),
    ("M", [198, 203, 214, 255]),
    ("W", [240, 244, 250, 255]),
    ("m", [140, 145, 158, 255]),
];
const ARMOR_BG_PALETTE: [(&str, Px); 4] = [
    ("#", [55, 55, 55, 255]),
    ("M", [30, 30, 30, 255]),
    ("W", [30, 30, 30, 255]),
    ("m", [30, 30, 30, 255]),
];

const BUBBLE_ART: [&str; 9] = [
    ".........",
    "..#####..",
    ".#BBBBB#.",
    "#BBWBBBB#",
    "#BBWBBBB#",
    "#BBBBBBB#",
    ".#BBBBB#.",
    "..#####..",
    ".........",
];
const BUBBLE_PALETTE: [(&str, Px); 4] = [
    ("#", [20, 40, 90, 255]),
    ("B", [60, 120, 220, 255]),
    ("W", [200, 230, 255, 255]),
    ("b", [40, 80, 170, 255]),
];
const BUBBLE_BG_PALETTE: [(&str, Px); 4] = [
    ("#", [50, 50, 50, 255]),
    ("B", [28, 28, 28, 255]),
    ("W", [28, 28, 28, 255]),
    ("b", [28, 28, 28, 255]),
];

/// Small white triangles for scroll buttons / dropdown carets.
fn arrow_px(up: bool) -> impl Fn(u32, u32) -> Px {
    move |x, y| {
        // Triangle pointing up: apex at (8, 5), base at y = 10.
        let half = (y as i32 - 5).max(0) as u32;
        let inside = if up {
            (5..=10).contains(&y) && x >= 8 - half.min(4) && x <= 8 + half.min(4)
        } else {
            let yy = 15 - y;
            (5..=10).contains(&yy)
                && x >= 8 - ((yy as i32 - 5).max(0) as u32).min(4)
                && x <= 8 + ((yy as i32 - 5).max(0) as u32).min(4)
        };
        if inside {
            [255, 255, 255, 255]
        } else {
            [0, 0, 0, 0]
        }
    }
}

/// Classic plus-shaped crosshair.
fn crosshair_px(x: u32, y: u32) -> Px {
    let arm =
        (x == 7 || x == 8) && (3..=12).contains(&y) || (y == 7 || y == 8) && (3..=12).contains(&x);
    if arm {
        [255, 255, 255, 210]
    } else {
        [0, 0, 0, 0]
    }
}

/// Build the full UI atlas as an uploadable [`Atlas`].
pub fn ui_atlas() -> Atlas {
    let mut rgba =
        vec![0u8; (UI_ATLAS_COLS * ATLAS_TILE_SIZE * UI_ATLAS_ROWS * ATLAS_TILE_SIZE * 4) as usize];

    // Buttons.
    paint_tile(&mut rgba, TILE_BTN, bevel_button(false, false));
    paint_tile(&mut rgba, TILE_BTN_HOVER, bevel_button(false, false));
    paint_tile(&mut rgba, TILE_BTN_PRESSED, bevel_button(true, false));
    paint_tile(&mut rgba, TILE_BTN_DISABLED, bevel_button(false, true));
    // Slots.
    paint_tile(&mut rgba, TILE_SLOT, |x, y| slot_px(x, y, false));
    paint_tile(&mut rgba, TILE_SLOT_HIGHLIGHT, |x, y| slot_px(x, y, true));
    // Panels.
    paint_tile(&mut rgba, TILE_PANEL, panel_px);
    paint_tile(&mut rgba, TILE_PANEL_INSET, panel_inset_px);
    paint_tile(&mut rgba, TILE_TOOLTIP, tooltip_px);
    // Scrollbar.
    paint_tile(&mut rgba, TILE_SCROLL_TRACK, scroll_px(false));
    paint_tile(&mut rgba, TILE_SCROLL_THUMB, scroll_px(true));
    // Hotbar + selection.
    paint_tile(&mut rgba, TILE_HUD_FRAME, hud_frame_px);
    paint_tile(&mut rgba, TILE_SEL_FRAME, sel_frame_px);
    // HUD icons.
    paint_tile(
        &mut rgba,
        TILE_HEART_BG,
        paint_art_tile(&HEART_ART, &HEART_BG_PALETTE),
    );
    paint_tile(
        &mut rgba,
        TILE_HEART_FULL,
        paint_art_tile(&HEART_ART, &HEART_PALETTE),
    );
    paint_tile(
        &mut rgba,
        TILE_HEART_HALF,
        paint_half_tile(&HEART_ART, &HEART_ART, &HEART_BG_PALETTE, &HEART_PALETTE),
    );
    paint_tile(
        &mut rgba,
        TILE_HUNGER_BG,
        paint_art_tile(&HUNGER_ART, &HUNGER_BG_PALETTE),
    );
    paint_tile(
        &mut rgba,
        TILE_HUNGER_FULL,
        paint_art_tile(&HUNGER_ART, &HUNGER_PALETTE),
    );
    paint_tile(
        &mut rgba,
        TILE_HUNGER_HALF,
        paint_half_tile(
            &HUNGER_ART,
            &HUNGER_ART,
            &HUNGER_BG_PALETTE,
            &HUNGER_PALETTE,
        ),
    );
    paint_tile(
        &mut rgba,
        TILE_ARMOR_BG,
        paint_art_tile(&ARMOR_ART, &ARMOR_BG_PALETTE),
    );
    paint_tile(
        &mut rgba,
        TILE_ARMOR_FULL,
        paint_art_tile(&ARMOR_ART, &ARMOR_PALETTE),
    );
    paint_tile(
        &mut rgba,
        TILE_ARMOR_HALF,
        paint_half_tile(&ARMOR_ART, &ARMOR_ART, &ARMOR_BG_PALETTE, &ARMOR_PALETTE),
    );
    paint_tile(
        &mut rgba,
        TILE_BUBBLE_BG,
        paint_art_tile(&BUBBLE_ART, &BUBBLE_BG_PALETTE),
    );
    paint_tile(
        &mut rgba,
        TILE_BUBBLE_FULL,
        paint_art_tile(&BUBBLE_ART, &BUBBLE_PALETTE),
    );
    paint_tile(&mut rgba, TILE_CROSSHAIR, crosshair_px);
    paint_tile(&mut rgba, TILE_ARROW_UP, arrow_px(true));
    paint_tile(&mut rgba, TILE_ARROW_DOWN, arrow_px(false));

    Atlas {
        width: UI_ATLAS_COLS * ATLAS_TILE_SIZE,
        height: UI_ATLAS_ROWS * ATLAS_TILE_SIZE,
        rgba,
        mip_chain: vec![],
        mip_levels: 1,
    }
}
