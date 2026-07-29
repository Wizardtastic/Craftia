//! Right tool options panel: 190 px wide, scrollable.
//!
//! Changes dynamically depending on the selected tool.
//! Contains: tool options, tool mask, target info, world properties.

use crate::edit::filter::FilterOp;
use crate::edit::paint::{GradientShape, InterpolationMode};
use crate::edit::terrain::TerrainOp;
use crate::edit::{BrushShape, EditState, PaintMode};
use voxel_render::{FontAtlas, UiDrawData};

use super::theme;

#[allow(dead_code)] // Many variants are Phase 1/6/7 infrastructure, matched in ui.rs
#[derive(Clone, Debug, PartialEq)]
pub enum RightPanelAction {
    None,
    SetShape(BrushShape),
    SetPaintMode(PaintMode),
    ToggleHollow,
    ToggleSurfaceOnly,
    ToggleReplace,
    ToggleShowGrid,
    ToggleShowChunks,
    RadiusDelta(f32),
    StrengthDelta(f32),
    // Phase 3: Multi-block palette
    ToggleMultiBlock,
    AddPaletteBlock,
    ClearPalette,
    // Phase 2: Replace target pick
    PickReplaceTarget,
    ClearReplaceTarget,
    // Phase 4: Terrain
    SetTerrainOp(TerrainOp),
    TerrainAmountDelta(f32),
    // Phase 5: Gradient
    SetGradientShape(GradientShape),
    SetInterpolation(InterpolationMode),
    SwapGradientBlocks,
    // Phase 6: Filter
    AddFilter(FilterOp),
    RemoveFilter(usize),
    ClearFilters,
    ApplyFilters,
    // Phase 7: Transform
    TransformMove(glam::IVec3),
    TransformRotate(i32),
    TransformScale(f32),
    // Phase 1: Selection
    SelectionCopy,
    SelectionCut,
    SelectionDelete,
    SelectionClear,
    // Phase 8: Undo
    UndoAction,
    RedoAction,
}

pub fn draw_right_panel(
    ui: &mut UiDrawData,
    edit: &mut EditState,
    screen_w: f32,
    screen_h: f32,
    mouse: (f32, f32),
    font: &FontAtlas,
    _player_pos: (f32, f32, f32),
    cursor_pos: Option<(i32, i32, i32)>,
) -> RightPanelAction {
    let mut action = RightPanelAction::None;
    let panel_x = screen_w - theme::RIGHT_PANEL_W;
    let panel_y = theme::MENU_BAR_H;
    let panel_w = theme::RIGHT_PANEL_W;
    let panel_h = screen_h - theme::MENU_BAR_H - theme::STATUS_BAR_H;

    ui.quad(panel_x, panel_y, panel_w, panel_h, theme::PANEL_BG);
    ui.quad(panel_x, panel_y, 1.0, panel_h, theme::BORDER);

    let (mx, my) = mouse;
    let mut y = panel_y;
    let lx = panel_x + 8.0;

    // ΓöÇΓöÇ Tool Options header ΓöÇΓöÇ
    y = draw_header(
        ui,
        &format!("Tool Options \u{2014} {}", edit.active_tool_label()),
        panel_x,
        y,
        panel_w,
        font,
    );

    // Draw tool-specific options based on active tool.
    match &edit.mode {
        crate::edit::EditModeState::Active { tool } => match tool {
            crate::edit::EditTool::Brush(brush) => {
                y = draw_brush_options(
                    ui,
                    edit,
                    brush.clone(),
                    panel_x,
                    y,
                    panel_w,
                    font,
                    mx,
                    my,
                    &mut action,
                );
            }
            crate::edit::EditTool::Select(sel) => {
                y = draw_select_options(
                    ui,
                    sel.clone(),
                    panel_x,
                    y,
                    panel_w,
                    font,
                    mx,
                    my,
                    &mut action,
                );
            }
            crate::edit::EditTool::Terrain(terrain) => {
                y = draw_terrain_options(
                    ui,
                    edit,
                    terrain.clone(),
                    panel_x,
                    y,
                    panel_w,
                    font,
                    mx,
                    my,
                    &mut action,
                );
            }
            crate::edit::EditTool::Paint(paint) => {
                y = draw_paint_options(
                    ui,
                    edit,
                    paint.clone(),
                    panel_x,
                    y,
                    panel_w,
                    font,
                    mx,
                    my,
                    &mut action,
                );
            }
            crate::edit::EditTool::Filter(filters) => {
                y = draw_filter_options(
                    ui,
                    filters.clone(),
                    panel_x,
                    y,
                    panel_w,
                    font,
                    mx,
                    my,
                    &mut action,
                );
            }
        },
        _ => {}
    }

    // ΓöÇΓöÇ Undo/Redo section (Phase 8) ΓöÇΓöÇ
    y = draw_separator(ui, panel_x, y, panel_w);
    y = draw_header(ui, "Undo / Redo", panel_x, y, panel_w, font);
    let undo_label = format!("Undo ({})", edit.history.len().saturating_sub(1));
    y = draw_button_row(
        ui,
        &undo_label,
        panel_x,
        y,
        panel_w,
        font,
        mx,
        my,
        edit.ui_click,
    );
    if edit.ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
        action = RightPanelAction::UndoAction;
    }
    y = draw_button_row(ui, "Redo", panel_x, y, panel_w, font, mx, my, edit.ui_click);
    if edit.ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
        action = RightPanelAction::RedoAction;
    }

    // ΓöÇΓöÇ Tool Mask section ΓöÇΓöÇ
    y = draw_separator(ui, panel_x, y, panel_w);
    y = draw_header(ui, "Tool Mask", panel_x, y, panel_w, font);
    let (ny, _mask_clicked) = draw_toggle_row(
        ui,
        "Enable Mask",
        false,
        panel_x,
        y,
        panel_w,
        font,
        mx,
        my,
        edit.ui_click,
    );
    y = ny;
    ui.text("No mask active", lx, y + 4.0, 0.6, theme::TEXT_DIM, font);
    y += 22.0;

    // ΓöÇΓöÇ Target Info section ΓöÇΓöÇ
    y = draw_separator(ui, panel_x, y, panel_w);
    y = draw_header(ui, "Target Info", panel_x, y, panel_w, font);

    let block_name = if let Some((cx, cy, cz)) = cursor_pos {
        format!("{}, {}, {}", cx, cy, cz)
    } else {
        "--".to_string()
    };
    y = draw_label_value(ui, "Position", &block_name, panel_x, y, panel_w, font);
    y = draw_label_value(ui, "Face", "+Y (Top)", panel_x, y, panel_w, font);
    y = draw_label_value(ui, "Biome", "Plains", panel_x, y, panel_w, font);
    y = draw_label_value(ui, "Light", "Sky:15 Block:0", panel_x, y, panel_w, font);

    // ΓöÇΓöÇ World Properties section ΓöÇΓöÇ
    y = draw_separator(ui, panel_x, y, panel_w);
    y = draw_header(ui, "World Properties", panel_x, y, panel_w, font);
    y = draw_slider_row(
        ui,
        "Time",
        6000.0,
        0.0,
        24000.0,
        panel_x,
        y,
        panel_w,
        font,
        mx,
        my,
        edit.ui_click,
        &mut action,
    );
    let (ny, grid_clicked) = draw_toggle_row(
        ui,
        "Show Grid",
        edit.show_grid,
        panel_x,
        y,
        panel_w,
        font,
        mx,
        my,
        edit.ui_click,
    );
    y = ny;
    if grid_clicked {
        action = RightPanelAction::ToggleShowGrid;
    }
    let (ny, chunks_clicked) = draw_toggle_row(
        ui,
        "Show Chunks",
        edit.show_chunks,
        panel_x,
        y,
        panel_w,
        font,
        mx,
        my,
        edit.ui_click,
    );
    y = ny;
    if chunks_clicked {
        action = RightPanelAction::ToggleShowChunks;
    }

    action
}

fn draw_brush_options(
    ui: &mut UiDrawData,
    edit: &mut EditState,
    brush: crate::edit::BrushTool,
    x: f32,
    mut y: f32,
    w: f32,
    font: &FontAtlas,
    mx: f32,
    my: f32,
    action: &mut RightPanelAction,
) -> f32 {
    let lx = x + 8.0;

    // Shape dropdown
    y = draw_dropdown_row(
        ui,
        "Shape",
        brush.shape.label(),
        x,
        y,
        w,
        font,
        "shape",
        &edit.ui_click,
        mx,
        my,
        action,
    );
    // Radius slider
    y = draw_slider_row(
        ui,
        "Radius",
        brush.radius,
        1.0,
        25.0,
        x,
        y,
        w,
        font,
        mx,
        my,
        edit.ui_click,
        action,
    );
    // Strength slider
    y = draw_slider_row(
        ui,
        "Strength",
        brush.strength,
        0.0,
        1.0,
        x,
        y,
        w,
        font,
        mx,
        my,
        edit.ui_click,
        action,
    );
    // Paint mode dropdown
    y = draw_dropdown_row(
        ui,
        "Paint Mode",
        brush.paint_mode.label(),
        x,
        y,
        w,
        font,
        "paint_mode",
        &edit.ui_click,
        mx,
        my,
        action,
    );

    y = draw_separator(ui, x, y, w);

    // Toggles
    let (ny, hollow_clicked) = draw_toggle_row(
        ui,
        "Hollow",
        brush.hollow,
        x,
        y,
        w,
        font,
        mx,
        my,
        edit.ui_click,
    );
    y = ny;
    if hollow_clicked {
        *action = RightPanelAction::ToggleHollow;
    }
    let (ny, surface_clicked) = draw_toggle_row(
        ui,
        "Surface Only",
        brush.surface_only,
        x,
        y,
        w,
        font,
        mx,
        my,
        edit.ui_click,
    );
    y = ny;
    if surface_clicked {
        *action = RightPanelAction::ToggleSurfaceOnly;
    }
    let (ny, replace_clicked) = draw_toggle_row(
        ui,
        "Replace Mode",
        brush.replace,
        x,
        y,
        w,
        font,
        mx,
        my,
        edit.ui_click,
    );
    y = ny;
    if replace_clicked {
        *action = RightPanelAction::ToggleReplace;
    }

    y = draw_separator(ui, x, y, w);

    // Block display
    y = draw_label_value(ui, "Block", &format!("{}", brush.block.0), x, y, w, font);

    // Phase 2: Replace target
    if brush.replace {
        let target_str = match brush.target {
            Some(t) => format!("{}", t.0),
            None => "Any non-air".to_string(),
        };
        y = draw_label_value(ui, "Target", &target_str, x, y, w, font);
        y = draw_button_row(
            ui,
            "Pick Target (Shift+RMB)",
            x,
            y,
            w,
            font,
            mx,
            my,
            edit.ui_click,
        );
        if edit.ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
            *action = RightPanelAction::PickReplaceTarget;
        }
        if brush.target.is_some() {
            y = draw_button_row(ui, "Clear Target", x, y, w, font, mx, my, edit.ui_click);
            if edit.ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
                *action = RightPanelAction::ClearReplaceTarget;
            }
        }
    }

    y = draw_separator(ui, x, y, w);

    // Phase 3: Multi-block palette
    let (ny, multi_clicked) = draw_toggle_row(
        ui,
        "Multi-block",
        brush.palette.enabled,
        x,
        y,
        w,
        font,
        mx,
        my,
        edit.ui_click,
    );
    y = ny;
    if multi_clicked {
        *action = RightPanelAction::ToggleMultiBlock;
    }
    if brush.palette.enabled {
        for (_i, entry) in brush.palette.entries.iter().enumerate() {
            let pct = if brush.palette.entries.len() > 1 {
                let total: f32 = brush.palette.entries.iter().map(|e| e.weight).sum();
                if total > 0.0 {
                    entry.weight / total * 100.0
                } else {
                    0.0
                }
            } else {
                100.0
            };
            let label = format!("  {} ({:.0}%)", entry.block.0, pct);
            ui.text(&label, lx, y + 2.0, 0.6, theme::TEXT_SECONDARY, font);
            y += theme::OPTION_ROW_H;
        }
        y = draw_button_row(ui, "+ Add Block", x, y, w, font, mx, my, edit.ui_click);
        if edit.ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
            *action = RightPanelAction::AddPaletteBlock;
        }
        y = draw_button_row(ui, "Clear Palette", x, y, w, font, mx, my, edit.ui_click);
        if edit.ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
            *action = RightPanelAction::ClearPalette;
        }
    }

    y
}

fn draw_select_options(
    ui: &mut UiDrawData,
    sel: crate::edit::select::SelectTool,
    x: f32,
    mut y: f32,
    w: f32,
    font: &FontAtlas,
    mx: f32,
    my: f32,
    action: &mut RightPanelAction,
) -> f32 {
    // Selection info
    y = draw_label_value(ui, "Dimensions", &sel.dimensions_str(), x, y, w, font);
    y = draw_label_value(
        ui,
        "Blocks",
        &format!("{}", sel.block_count()),
        x,
        y,
        w,
        font,
    );

    y = draw_separator(ui, x, y, w);

    // Action buttons
    y = draw_button_row(ui, "Copy (Ctrl+C)", x, y, w, font, mx, my, false);
    if mx >= x + 8.0 && mx <= x + w - 8.0 && my >= y - theme::OPTION_ROW_H && my <= y {
        ui.quad(
            x + 8.0,
            y - theme::OPTION_ROW_H,
            w - 16.0,
            theme::OPTION_ROW_H,
            theme::HOVER_BG,
        );
    }
    y = draw_button_row(ui, "Cut", x, y, w, font, mx, my, false);
    y = draw_button_row(ui, "Delete", x, y, w, font, mx, my, false);
    y = draw_button_row(ui, "Fill...", x, y, w, font, mx, my, false);
    y = draw_button_row(ui, "Replace...", x, y, w, font, mx, my, false);

    y = draw_separator(ui, x, y, w);

    y = draw_button_row(ui, "Clear Selection", x, y, w, font, mx, my, false);

    y
}

fn draw_terrain_options(
    ui: &mut UiDrawData,
    edit: &mut EditState,
    terrain: crate::edit::terrain::TerrainTool,
    x: f32,
    mut y: f32,
    w: f32,
    font: &FontAtlas,
    mx: f32,
    my: f32,
    action: &mut RightPanelAction,
) -> f32 {
    // Shape dropdown
    y = draw_dropdown_row(
        ui,
        "Shape",
        terrain.shape.label(),
        x,
        y,
        w,
        font,
        "terrain_shape",
        &edit.ui_click,
        mx,
        my,
        action,
    );
    // Radius slider
    y = draw_slider_row(
        ui,
        "Radius",
        terrain.radius,
        1.0,
        25.0,
        x,
        y,
        w,
        font,
        mx,
        my,
        edit.ui_click,
        action,
    );

    y = draw_separator(ui, x, y, w);

    // Operation selector
    let ops = [
        ("Raise", TerrainOp::Raise { amount: 1.0 }),
        ("Lower", TerrainOp::Lower { amount: 1.0 }),
        (
            "Flatten",
            TerrainOp::Flatten {
                target_height: None,
            },
        ),
        ("Smooth", TerrainOp::Smooth { iterations: 1 }),
        (
            "Noise",
            TerrainOp::Noise {
                scale: 10.0,
                amplitude: 3.0,
                seed: 0,
            },
        ),
    ];
    for (label, op) in &ops {
        let is_active = terrain.op.label() == *label;
        let bg = if is_active {
            theme::ACTIVE_BG
        } else {
            theme::ELEMENT_BG
        };
        let btn_x = x + 8.0;
        let btn_w = (w - 16.0 - 4.0 * 4.0) / 5.0;
        let btn_idx = ops.iter().position(|(l, _)| l == label).unwrap_or(0);
        let bx = btn_x + btn_idx as f32 * (btn_w + 4.0);
        ui.quad(bx, y, btn_w, theme::OPTION_ROW_H, bg);
        ui.rect_border(
            bx,
            y,
            btn_w,
            theme::OPTION_ROW_H,
            1.0,
            if is_active {
                theme::ACCENT
            } else {
                theme::BORDER
            },
        );
        let lw = font.text_width(label, 0.55);
        ui.text(
            label,
            bx + (btn_w - lw) * 0.5,
            y + 4.0,
            0.55,
            if is_active {
                theme::ACCENT_LIGHT
            } else {
                theme::TEXT_SECONDARY
            },
            font,
        );
        if mx >= bx && mx <= bx + btn_w && my >= y && my <= y + theme::OPTION_ROW_H && edit.ui_click
        {
            *action = RightPanelAction::SetTerrainOp(op.clone());
        }
    }
    y += theme::OPTION_ROW_H + 4.0;

    // Operation-specific params.
    match &terrain.op {
        TerrainOp::Raise { amount } | TerrainOp::Lower { amount } => {
            y = draw_slider_row(
                ui,
                "Amount",
                *amount,
                1.0,
                20.0,
                x,
                y,
                w,
                font,
                mx,
                my,
                edit.ui_click,
                action,
            );
            if edit.ui_click && mx >= x + 58.0 && mx <= x + w - 58.0 {
                let slider_w = w - 116.0;
                let click_pct = ((mx - (x + 58.0)) / slider_w).clamp(0.0, 1.0);
                let new_val = 1.0 + click_pct * 19.0;
                *action = RightPanelAction::TerrainAmountDelta(new_val - amount);
            }
        }
        TerrainOp::Flatten { target_height } => {
            let target_str = match target_height {
                Some(h) => format!("{}", h),
                None => "Pick from world".to_string(),
            };
            y = draw_label_value(ui, "Target Y", &target_str, x, y, w, font);
        }
        TerrainOp::Smooth { iterations } => {
            y = draw_label_value(ui, "Iterations", &format!("{}", iterations), x, y, w, font);
        }
        TerrainOp::Noise {
            scale,
            amplitude,
            seed,
        } => {
            y = draw_slider_row(
                ui,
                "Scale",
                *scale,
                1.0,
                100.0,
                x,
                y,
                w,
                font,
                mx,
                my,
                edit.ui_click,
                action,
            );
            y = draw_slider_row(
                ui,
                "Amplitude",
                *amplitude,
                0.1,
                20.0,
                x,
                y,
                w,
                font,
                mx,
                my,
                edit.ui_click,
                action,
            );
            y = draw_label_value(ui, "Seed", &format!("{}", seed), x, y, w, font);
        }
    }

    y
}

fn draw_paint_options(
    ui: &mut UiDrawData,
    edit: &mut EditState,
    paint: crate::edit::paint::PaintTool,
    x: f32,
    mut y: f32,
    w: f32,
    font: &FontAtlas,
    mx: f32,
    my: f32,
    action: &mut RightPanelAction,
) -> f32 {
    // Shape
    y = draw_dropdown_row(
        ui,
        "Shape",
        paint.shape.label(),
        x,
        y,
        w,
        font,
        "paint_shape",
        &edit.ui_click,
        mx,
        my,
        action,
    );
    // Radius
    y = draw_slider_row(
        ui,
        "Radius",
        paint.radius,
        1.0,
        25.0,
        x,
        y,
        w,
        font,
        mx,
        my,
        edit.ui_click,
        action,
    );

    y = draw_separator(ui, x, y, w);

    // Blocks
    y = draw_label_value(
        ui,
        "Block A",
        &format!("{}", paint.block_a.0),
        x,
        y,
        w,
        font,
    );
    y = draw_label_value(
        ui,
        "Block B",
        &format!("{}", paint.block_b.0),
        x,
        y,
        w,
        font,
    );
    y = draw_button_row(ui, "Swap Blocks", x, y, w, font, mx, my, edit.ui_click);
    if edit.ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
        *action = RightPanelAction::SwapGradientBlocks;
    }

    y = draw_separator(ui, x, y, w);

    // Gradient shape
    let gradients = GradientShape::ALL;
    for gs in gradients {
        let is_active = paint.gradient == *gs;
        let bg = if is_active {
            theme::ACTIVE_BG
        } else {
            theme::ELEMENT_BG
        };
        let btn_x = x + 8.0;
        let btn_w = (w - 16.0 - 4.0 * (gradients.len() - 1) as f32) / gradients.len() as f32;
        let idx = gradients.iter().position(|g| g == gs).unwrap_or(0);
        let bx = btn_x + idx as f32 * (btn_w + 4.0);
        ui.quad(bx, y, btn_w, theme::OPTION_ROW_H, bg);
        ui.rect_border(
            bx,
            y,
            btn_w,
            theme::OPTION_ROW_H,
            1.0,
            if is_active {
                theme::ACCENT
            } else {
                theme::BORDER
            },
        );
        let label = gs.label();
        let lw = font.text_width(label, 0.5);
        ui.text(
            label,
            bx + (btn_w - lw) * 0.5,
            y + 4.0,
            0.5,
            if is_active {
                theme::ACCENT_LIGHT
            } else {
                theme::TEXT_SECONDARY
            },
            font,
        );
        if mx >= bx && mx <= bx + btn_w && my >= y && my <= y + theme::OPTION_ROW_H && edit.ui_click
        {
            *action = RightPanelAction::SetGradientShape(*gs);
        }
    }
    y += theme::OPTION_ROW_H + 4.0;

    // Interpolation
    let interps = InterpolationMode::ALL;
    for im in interps {
        let is_active = paint.interpolation == *im;
        let bg = if is_active {
            theme::ACTIVE_BG
        } else {
            theme::ELEMENT_BG
        };
        let btn_x = x + 8.0;
        let btn_w = (w - 16.0 - 4.0 * (interps.len() - 1) as f32) / interps.len() as f32;
        let idx = interps.iter().position(|i| i == im).unwrap_or(0);
        let bx = btn_x + idx as f32 * (btn_w + 4.0);
        ui.quad(bx, y, btn_w, theme::OPTION_ROW_H, bg);
        ui.rect_border(
            bx,
            y,
            btn_w,
            theme::OPTION_ROW_H,
            1.0,
            if is_active {
                theme::ACCENT
            } else {
                theme::BORDER
            },
        );
        let label = im.label();
        let lw = font.text_width(label, 0.55);
        ui.text(
            label,
            bx + (btn_w - lw) * 0.5,
            y + 4.0,
            0.55,
            if is_active {
                theme::ACCENT_LIGHT
            } else {
                theme::TEXT_SECONDARY
            },
            font,
        );
        if mx >= bx && mx <= bx + btn_w && my >= y && my <= y + theme::OPTION_ROW_H && edit.ui_click
        {
            *action = RightPanelAction::SetInterpolation(*im);
        }
    }
    y += theme::OPTION_ROW_H + 4.0;

    y
}

fn draw_filter_options(
    ui: &mut UiDrawData,
    filters: crate::edit::filter::FilterStack,
    x: f32,
    mut y: f32,
    w: f32,
    font: &FontAtlas,
    mx: f32,
    my: f32,
    action: &mut RightPanelAction,
) -> f32 {
    // List existing filters.
    for (i, op) in filters.filters.iter().enumerate() {
        let label = format!("{}. {}", i + 1, op.label());
        ui.text(&label, x + 8.0, y + 2.0, 0.65, theme::TEXT_PRIMARY, font);
        // Remove button.
        let rm_x = x + w - 24.0;
        ui.text("x", rm_x, y + 2.0, 0.6, theme::TEXT_DIM, font);
        if mx >= rm_x && mx <= rm_x + 12.0 && my >= y && my <= y + theme::OPTION_ROW_H {
            *action = RightPanelAction::RemoveFilter(i);
        }
        y += theme::OPTION_ROW_H;
    }

    y = draw_separator(ui, x, y, w);

    // Add filter buttons.
    y = draw_button_row(ui, "+ Noise Filter", x, y, w, font, mx, my, false);
    // TODO: wire click handler for "+ Noise Filter" button.
    y = draw_button_row(ui, "+ Erode Filter", x, y, w, font, mx, my, false);
    y = draw_button_row(ui, "+ Dilate Filter", x, y, w, font, mx, my, false);
    y = draw_button_row(ui, "+ Smooth Filter", x, y, w, font, mx, my, false);

    y = draw_separator(ui, x, y, w);

    y = draw_button_row(ui, "Apply Filters", x, y, w, font, mx, my, false);
    y = draw_button_row(ui, "Clear All", x, y, w, font, mx, my, false);

    y
}

fn draw_header(ui: &mut UiDrawData, title: &str, x: f32, y: f32, w: f32, font: &FontAtlas) -> f32 {
    let h = theme::SECTION_HEADER_H;
    ui.quad(x, y, w, h, theme::HEADER_BG);
    ui.quad(x, y + h - 1.0, w, 1.0, theme::BORDER);
    ui.text(title, x + 7.0, y + 3.0, 0.65, theme::TEXT_SECONDARY, font);
    y + h
}

fn draw_separator(ui: &mut UiDrawData, x: f32, y: f32, w: f32) -> f32 {
    ui.quad(x + 8.0, y, w - 16.0, 1.0, theme::BORDER);
    y + 4.0
}

fn draw_label_value(
    ui: &mut UiDrawData,
    label: &str,
    value: &str,
    x: f32,
    y: f32,
    w: f32,
    font: &FontAtlas,
) -> f32 {
    let lx = x + 8.0;
    ui.text(label, lx, y + 4.0, 0.65, theme::TEXT_SECONDARY, font);
    let vw = font.text_width(value, 0.65);
    ui.text(
        value,
        x + w - vw - 8.0,
        y + 4.0,
        0.65,
        theme::TEXT_PRIMARY,
        font,
    );
    y + theme::OPTION_ROW_H
}

fn draw_slider_row(
    ui: &mut UiDrawData,
    label: &str,
    value: f32,
    min: f32,
    max: f32,
    x: f32,
    y: f32,
    w: f32,
    font: &FontAtlas,
    mx: f32,
    my: f32,
    ui_click: bool,
    action: &mut RightPanelAction,
) -> f32 {
    let lx = x + 8.0;
    let rw = w - 16.0;

    ui.text(label, lx, y + 2.0, 0.65, theme::TEXT_SECONDARY, font);

    let slider_x = lx + 50.0;
    let slider_w = rw - 100.0;
    let slider_y = y + 3.0;
    let slider_h = theme::SLIDER_H;

    let pct = ((value - min) / (max - min)).clamp(0.0, 1.0);

    ui.quad(slider_x, slider_y, slider_w, slider_h, theme::ELEMENT_BG);
    ui.rect_border(slider_x, slider_y, slider_w, slider_h, 1.0, theme::BORDER);
    ui.quad(
        slider_x,
        slider_y,
        slider_w * pct,
        slider_h,
        [74, 127, 212, 102],
    );

    let val_text = if max <= 1.5 {
        format!("{:.2}", value)
    } else {
        format!("{:.1}", value)
    };
    let vtw = font.text_width(&val_text, 0.6);
    ui.text(
        &val_text,
        slider_x + (slider_w - vtw) * 0.5,
        slider_y + 2.0,
        0.6,
        theme::TEXT_PRIMARY,
        font,
    );

    if mx >= slider_x
        && mx <= slider_x + slider_w
        && my >= slider_y
        && my <= slider_y + slider_h
        && ui_click
    {
        let click_pct = ((mx - slider_x) / slider_w).clamp(0.0, 1.0);
        let new_val = min + click_pct * (max - min);
        let delta = new_val - value;
        if label == "Radius" {
            *action = RightPanelAction::RadiusDelta(delta);
        } else if label == "Strength" {
            *action = RightPanelAction::StrengthDelta(delta);
        }
    }

    let val_disp = if max <= 1.5 {
        format!("{:.2}", value)
    } else {
        format!("{:.0}", value)
    };
    let vdw = font.text_width(&val_disp, 0.65);
    ui.text(
        &val_disp,
        x + w - vdw - 8.0,
        y + 2.0,
        0.65,
        theme::TEXT_PRIMARY,
        font,
    );

    y + theme::OPTION_ROW_H
}

fn draw_toggle_row(
    ui: &mut UiDrawData,
    label: &str,
    on: bool,
    x: f32,
    y: f32,
    _w: f32,
    font: &FontAtlas,
    mx: f32,
    my: f32,
    ui_click: bool,
) -> (f32, bool) {
    let lx = x + 8.0;
    let box_size = 11.0;
    let bx = lx;
    let by = y + 4.0;

    let bg = if on { theme::ACCENT } else { theme::ELEMENT_BG };
    let border = if on {
        theme::ACCENT
    } else {
        theme::BORDER_LIGHT
    };

    ui.quad(bx, by, box_size, box_size, bg);
    ui.rect_border(bx, by, box_size, box_size, 1.0, border);

    if on {
        ui.text(
            "\u{2713}",
            bx + 1.0,
            by - 1.0,
            0.55,
            [255, 255, 255, 255],
            font,
        );
    }

    ui.text(
        label,
        lx + box_size + 6.0,
        y + 2.0,
        0.65,
        theme::TEXT_SECONDARY,
        font,
    );

    let clicked = mx >= bx && mx <= bx + 120.0 && my >= by && my <= by + box_size + 4.0 && ui_click;

    (y + theme::OPTION_ROW_H, clicked)
}

fn draw_dropdown_row(
    ui: &mut UiDrawData,
    label: &str,
    value: &str,
    x: f32,
    y: f32,
    w: f32,
    font: &FontAtlas,
    _id: &str,
    ui_click: &bool,
    mx: f32,
    my: f32,
    action: &mut RightPanelAction,
) -> f32 {
    let lx = x + 8.0;
    let rw = w - 16.0;

    ui.text(label, lx, y + 2.0, 0.65, theme::TEXT_SECONDARY, font);

    let dd_x = lx + 55.0;
    let dd_w = rw - 60.0;
    let dd_y = y + 2.0;
    let dd_h = theme::DROPDOWN_H;

    let hovered = mx >= dd_x && mx <= dd_x + dd_w && my >= dd_y && my <= dd_y + dd_h;

    ui.quad(dd_x, dd_y, dd_w, dd_h, theme::ELEMENT_BG);
    ui.rect_border(
        dd_x,
        dd_y,
        dd_w,
        dd_h,
        1.0,
        if hovered {
            theme::BORDER_LIGHT
        } else {
            theme::BORDER
        },
    );

    ui.text(
        value,
        dd_x + 6.0,
        dd_y + 3.0,
        0.65,
        theme::TEXT_PRIMARY,
        font,
    );
    ui.text(
        "\u{25BE}",
        dd_x + dd_w - 14.0,
        dd_y + 3.0,
        0.5,
        theme::TEXT_DIM,
        font,
    );

    if hovered && *ui_click {
        if label == "Shape" {
            let new_shape = match value {
                "Sphere" => BrushShape::Cylinder,
                "Cylinder" => BrushShape::Box,
                _ => BrushShape::Sphere,
            };
            *action = RightPanelAction::SetShape(new_shape);
        } else if label == "Paint Mode" {
            let new_mode = match value {
                "Fill" => PaintMode::Replace,
                "Replace" => PaintMode::Overlay,
                _ => PaintMode::Fill,
            };
            *action = RightPanelAction::SetPaintMode(new_mode);
        }
    }

    y + theme::OPTION_ROW_H
}

fn draw_button_row(
    ui: &mut UiDrawData,
    label: &str,
    x: f32,
    y: f32,
    w: f32,
    font: &FontAtlas,
    mx: f32,
    my: f32,
    _ui_click: bool,
) -> f32 {
    let bx = x + 8.0;
    let bw = w - 16.0;
    let bh = theme::OPTION_ROW_H - 4.0;
    let by = y + 2.0;

    let hovered = mx >= bx && mx <= bx + bw && my >= by && my <= by + bh;
    let bg = if hovered {
        theme::HOVER_BG
    } else {
        theme::ELEMENT_BG
    };

    ui.quad(bx, by, bw, bh, bg);
    ui.rect_border(bx, by, bw, bh, 1.0, theme::BORDER_LIGHT);

    let lw = font.text_width(label, 0.65);
    ui.text(
        label,
        bx + (bw - lw) * 0.5,
        by + 2.0,
        0.65,
        theme::TEXT_PRIMARY,
        font,
    );

    y + theme::OPTION_ROW_H
}
