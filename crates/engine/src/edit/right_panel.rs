//! Right tool options panel: 190 px wide, scrollable.
//!
//! Changes dynamically depending on the selected tool.
//! Contains: tool options, tool mask, target info, world properties.

use crate::edit::ctx::PanelCtx;
use crate::edit::filter::FilterOp;
use crate::edit::paint::{GradientShape, InterpolationMode};
use crate::edit::terrain::TerrainOp;
use crate::edit::{BrushShape, EditState, PaintMode};
use crate::ui::{Row, SliderRange};
use voxel_core::Point;
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
    screen: (f32, f32),
    mouse: Point,
    font: &FontAtlas,
    cursor_pos: Option<(i32, i32, i32)>,
    pack_infos: &[crate::TexturePackInfo],
) -> RightPanelAction {
    let (screen_w, screen_h) = screen;
    let mut action = RightPanelAction::None;
    let panel_x = screen_w - theme::RIGHT_PANEL_W;
    let panel_y = theme::MENU_BAR_H;
    let panel_w = theme::RIGHT_PANEL_W;
    let panel_h = screen_h - theme::MENU_BAR_H - theme::STATUS_BAR_H;

    ui.quad(panel_x, panel_y, panel_w, panel_h, theme::PANEL_BG);
    ui.quad(panel_x, panel_y, 1.0, panel_h, theme::BORDER);

    let Point { x: mx, y: my } = mouse;
    let ctx = PanelCtx {
        font,
        mx,
        my,
        ui_click: edit.ui_click,
    };
    let mut y = panel_y;
    let lx = panel_x + 8.0;

    // ── Tool Options header ──
    y = draw_header(
        ui,
        &format!("Tool Options \u{2014} {}", edit.active_tool_label()),
        panel_x,
        y,
        panel_w,
        font,
    );

    // Draw tool-specific options based on active tool.
    if let crate::edit::EditModeState::Active { tool } = &edit.mode {
        match tool {
            crate::edit::EditTool::Brush(brush) => {
                y = draw_brush_options(
                    ui,
                    &ctx,
                    &mut action,
                    brush.clone(),
                    Row {
                        x: panel_x,
                        y,
                        w: panel_w,
                    },
                );
            }
            crate::edit::EditTool::Select(sel) => {
                y = draw_select_options(
                    ui,
                    &ctx,
                    sel.clone(),
                    Row {
                        x: panel_x,
                        y,
                        w: panel_w,
                    },
                );
            }
            crate::edit::EditTool::Terrain(terrain) => {
                y = draw_terrain_options(
                    ui,
                    &ctx,
                    &mut action,
                    terrain.clone(),
                    Row {
                        x: panel_x,
                        y,
                        w: panel_w,
                    },
                );
            }
            crate::edit::EditTool::Paint(paint) => {
                y = draw_paint_options(
                    ui,
                    &ctx,
                    &mut action,
                    paint.clone(),
                    Row {
                        x: panel_x,
                        y,
                        w: panel_w,
                    },
                );
            }
            crate::edit::EditTool::Filter(filters) => {
                y = draw_filter_options(
                    ui,
                    &ctx,
                    &mut action,
                    filters.clone(),
                    Row {
                        x: panel_x,
                        y,
                        w: panel_w,
                    },
                );
            }
        }
    }

    // ── Undo/Redo section (Phase 8) ──
    y = draw_separator(ui, panel_x, y, panel_w);
    y = draw_header(ui, "Undo / Redo", panel_x, y, panel_w, font);
    let undo_label = format!("Undo ({})", edit.history.len().saturating_sub(1));
    y = draw_button_row(
        ui,
        &ctx,
        &undo_label,
        Row {
            x: panel_x,
            y,
            w: panel_w,
        },
    );
    if edit.ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
        action = RightPanelAction::UndoAction;
    }
    y = draw_button_row(
        ui,
        &ctx,
        "Redo",
        Row {
            x: panel_x,
            y,
            w: panel_w,
        },
    );
    if edit.ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
        action = RightPanelAction::RedoAction;
    }

    // ── Tool Mask section ──
    y = draw_separator(ui, panel_x, y, panel_w);
    y = draw_header(ui, "Tool Mask", panel_x, y, panel_w, font);
    let (ny, _mask_clicked) =
        draw_toggle_row(ui, &ctx, "Enable Mask", false, Point::new(panel_x, y));
    y = ny;
    ui.text("No mask active", lx, y + 4.0, 0.6, theme::TEXT_DIM, font);
    y += 22.0;

    // ── Target Info section ──
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

    // ── World Properties section ──
    y = draw_separator(ui, panel_x, y, panel_w);
    y = draw_header(ui, "World Properties", panel_x, y, panel_w, font);
    y = draw_slider_row(
        ui,
        &ctx,
        &mut action,
        "Time",
        6000.0,
        SliderRange {
            min: 0.0,
            max: 24000.0,
        },
        Row {
            x: panel_x,
            y,
            w: panel_w,
        },
    );
    let (ny, grid_clicked) = draw_toggle_row(
        ui,
        &ctx,
        "Show Grid",
        edit.show_grid,
        Point::new(panel_x, y),
    );
    y = ny;
    if grid_clicked {
        action = RightPanelAction::ToggleShowGrid;
    }
    let (ny, chunks_clicked) = draw_toggle_row(
        ui,
        &ctx,
        "Show Chunks",
        edit.show_chunks,
        Point::new(panel_x, y),
    );
    y = ny;
    if chunks_clicked {
        action = RightPanelAction::ToggleShowChunks;
    }

    // Texture pack manager section
    draw_pack_manager_section(ui, &ctx, pack_infos, screen_w, y);

    action
}

fn draw_brush_options(
    ui: &mut UiDrawData,
    ctx: &PanelCtx<'_>,
    action: &mut RightPanelAction,
    brush: crate::edit::BrushTool,
    rect: Row,
) -> f32 {
    let PanelCtx {
        font,
        mx: _,
        my,
        ui_click,
    } = *ctx;
    let Row { x, mut y, w } = rect;
    let lx = x + 8.0;

    // Shape dropdown
    y = draw_dropdown_row(
        ui,
        ctx,
        action,
        "Shape",
        brush.shape.label(),
        Row { x, y, w },
    );
    // Radius slider
    y = draw_slider_row(
        ui,
        ctx,
        action,
        "Radius",
        brush.radius,
        SliderRange {
            min: 1.0,
            max: 25.0,
        },
        Row { x, y, w },
    );
    // Strength slider
    y = draw_slider_row(
        ui,
        ctx,
        action,
        "Strength",
        brush.strength,
        SliderRange { min: 0.0, max: 1.0 },
        Row { x, y, w },
    );
    // Paint mode dropdown
    y = draw_dropdown_row(
        ui,
        ctx,
        action,
        "Paint Mode",
        brush.paint_mode.label(),
        Row { x, y, w },
    );

    y = draw_separator(ui, x, y, w);

    // Toggles
    let (ny, hollow_clicked) = draw_toggle_row(ui, ctx, "Hollow", brush.hollow, Point::new(x, y));
    y = ny;
    if hollow_clicked {
        *action = RightPanelAction::ToggleHollow;
    }
    let (ny, surface_clicked) = draw_toggle_row(
        ui,
        ctx,
        "Surface Only",
        brush.surface_only,
        Point::new(x, y),
    );
    y = ny;
    if surface_clicked {
        *action = RightPanelAction::ToggleSurfaceOnly;
    }
    let (ny, replace_clicked) =
        draw_toggle_row(ui, ctx, "Replace Mode", brush.replace, Point::new(x, y));
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
        y = draw_button_row(ui, ctx, "Pick Target (Shift+RMB)", Row { x, y, w });
        if ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
            *action = RightPanelAction::PickReplaceTarget;
        }
        if brush.target.is_some() {
            y = draw_button_row(ui, ctx, "Clear Target", Row { x, y, w });
            if ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
                *action = RightPanelAction::ClearReplaceTarget;
            }
        }
    }

    y = draw_separator(ui, x, y, w);

    // Phase 3: Multi-block palette
    let (ny, multi_clicked) = draw_toggle_row(
        ui,
        ctx,
        "Multi-block",
        brush.palette.enabled,
        Point::new(x, y),
    );
    y = ny;
    if multi_clicked {
        *action = RightPanelAction::ToggleMultiBlock;
    }
    if brush.palette.enabled {
        for entry in brush.palette.entries.iter() {
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
        y = draw_button_row(ui, ctx, "+ Add Block", Row { x, y, w });
        if ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
            *action = RightPanelAction::AddPaletteBlock;
        }
        y = draw_button_row(ui, ctx, "Clear Palette", Row { x, y, w });
        if ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
            *action = RightPanelAction::ClearPalette;
        }
    }

    y
}

fn draw_select_options(
    ui: &mut UiDrawData,
    ctx: &PanelCtx<'_>,
    sel: crate::edit::select::SelectTool,
    rect: Row,
) -> f32 {
    let PanelCtx {
        font,
        mx,
        my,
        ui_click: _,
    } = *ctx;
    let Row { x, mut y, w } = rect;
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
    y = draw_button_row(ui, ctx, "Copy (Ctrl+C)", Row { x, y, w });
    if mx >= x + 8.0 && mx <= x + w - 8.0 && my >= y - theme::OPTION_ROW_H && my <= y {
        ui.quad(
            x + 8.0,
            y - theme::OPTION_ROW_H,
            w - 16.0,
            theme::OPTION_ROW_H,
            theme::HOVER_BG,
        );
    }
    y = draw_button_row(ui, ctx, "Cut", Row { x, y, w });
    y = draw_button_row(ui, ctx, "Delete", Row { x, y, w });
    y = draw_button_row(ui, ctx, "Fill...", Row { x, y, w });
    y = draw_button_row(ui, ctx, "Replace...", Row { x, y, w });

    y = draw_separator(ui, x, y, w);

    y = draw_button_row(ui, ctx, "Clear Selection", Row { x, y, w });

    y
}

fn draw_terrain_options(
    ui: &mut UiDrawData,
    ctx: &PanelCtx<'_>,
    action: &mut RightPanelAction,
    terrain: crate::edit::terrain::TerrainTool,
    rect: Row,
) -> f32 {
    let PanelCtx {
        font,
        mx,
        my,
        ui_click,
    } = *ctx;
    let Row { x, mut y, w } = rect;
    y = draw_dropdown_row(
        ui,
        ctx,
        action,
        "Shape",
        terrain.shape.label(),
        Row { x, y, w },
    );
    // Radius slider
    y = draw_slider_row(
        ui,
        ctx,
        action,
        "Radius",
        terrain.radius,
        SliderRange {
            min: 1.0,
            max: 25.0,
        },
        Row { x, y, w },
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
        if mx >= bx && mx <= bx + btn_w && my >= y && my <= y + theme::OPTION_ROW_H && ui_click {
            *action = RightPanelAction::SetTerrainOp(op.clone());
        }
    }
    y += theme::OPTION_ROW_H + 4.0;

    // Operation-specific params.
    match &terrain.op {
        TerrainOp::Raise { amount } | TerrainOp::Lower { amount } => {
            y = draw_slider_row(
                ui,
                ctx,
                action,
                "Amount",
                *amount,
                SliderRange {
                    min: 1.0,
                    max: 20.0,
                },
                Row { x, y, w },
            );
            if ui_click && mx >= x + 58.0 && mx <= x + w - 58.0 {
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
                ctx,
                action,
                "Scale",
                *scale,
                SliderRange {
                    min: 1.0,
                    max: 100.0,
                },
                Row { x, y, w },
            );
            y = draw_slider_row(
                ui,
                ctx,
                action,
                "Amplitude",
                *amplitude,
                SliderRange {
                    min: 0.1,
                    max: 20.0,
                },
                Row { x, y, w },
            );
            y = draw_label_value(ui, "Seed", &format!("{}", seed), x, y, w, font);
        }
    }

    y
}

fn draw_paint_options(
    ui: &mut UiDrawData,
    ctx: &PanelCtx<'_>,
    action: &mut RightPanelAction,
    paint: crate::edit::paint::PaintTool,
    rect: Row,
) -> f32 {
    let PanelCtx {
        font,
        mx,
        my,
        ui_click,
    } = *ctx;
    let Row { x, mut y, w } = rect;
    y = draw_dropdown_row(
        ui,
        ctx,
        action,
        "Shape",
        paint.shape.label(),
        Row { x, y, w },
    );
    // Radius
    y = draw_slider_row(
        ui,
        ctx,
        action,
        "Radius",
        paint.radius,
        SliderRange {
            min: 1.0,
            max: 25.0,
        },
        Row { x, y, w },
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
    y = draw_button_row(ui, ctx, "Swap Blocks", Row { x, y, w });
    if ui_click && my >= y - theme::OPTION_ROW_H && my <= y {
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
        if mx >= bx && mx <= bx + btn_w && my >= y && my <= y + theme::OPTION_ROW_H && ui_click {
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
        if mx >= bx && mx <= bx + btn_w && my >= y && my <= y + theme::OPTION_ROW_H && ui_click {
            *action = RightPanelAction::SetInterpolation(*im);
        }
    }
    y += theme::OPTION_ROW_H + 4.0;

    y
}

fn draw_filter_options(
    ui: &mut UiDrawData,
    ctx: &PanelCtx<'_>,
    action: &mut RightPanelAction,
    filters: crate::edit::filter::FilterStack,
    rect: Row,
) -> f32 {
    let PanelCtx {
        font,
        mx,
        my,
        ui_click: _,
    } = *ctx;
    let Row { x, mut y, w } = rect;
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
    y = draw_button_row(ui, ctx, "+ Noise Filter", Row { x, y, w });
    // TODO: wire click handler for "+ Noise Filter" button.
    y = draw_button_row(ui, ctx, "+ Erode Filter", Row { x, y, w });
    y = draw_button_row(ui, ctx, "+ Dilate Filter", Row { x, y, w });
    y = draw_button_row(ui, ctx, "+ Smooth Filter", Row { x, y, w });

    y = draw_separator(ui, x, y, w);

    y = draw_button_row(ui, ctx, "Apply Filters", Row { x, y, w });
    y = draw_button_row(ui, ctx, "Clear All", Row { x, y, w });

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
    ctx: &PanelCtx<'_>,
    action: &mut RightPanelAction,
    label: &str,
    value: f32,
    range: SliderRange,
    rect: Row,
) -> f32 {
    let PanelCtx {
        font,
        mx,
        my,
        ui_click,
    } = *ctx;
    let SliderRange { min, max } = range;
    let Row { x, y, w } = rect;
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
    ctx: &PanelCtx<'_>,
    label: &str,
    on: bool,
    pos: Point,
) -> (f32, bool) {
    let PanelCtx {
        font,
        mx,
        my,
        ui_click,
    } = *ctx;
    let Point { x, y } = pos;
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
    ctx: &PanelCtx<'_>,
    action: &mut RightPanelAction,
    label: &str,
    value: &str,
    rect: Row,
) -> f32 {
    let PanelCtx {
        font,
        mx,
        my,
        ui_click,
    } = *ctx;
    let Row { x, y, w } = rect;
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

    if hovered && ui_click {
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

fn draw_button_row(ui: &mut UiDrawData, ctx: &PanelCtx<'_>, label: &str, rect: Row) -> f32 {
    let PanelCtx {
        font,
        mx,
        my,
        ui_click: _,
    } = *ctx;
    let Row { x, y, w } = rect;
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

/// Draw the texture pack manager section in the right panel.
fn draw_pack_manager_section(
    ui: &mut UiDrawData,
    ctx: &PanelCtx<'_>,
    packs: &[crate::TexturePackInfo],
    screen_w: f32,
    y: f32,
) -> f32 {
    let PanelCtx { font, .. } = *ctx;
    let x = screen_w - theme::RIGHT_PANEL_W;
    let w = theme::RIGHT_PANEL_W;
    let header_h = theme::SECTION_HEADER_H;

    // Section header
    ui.quad(x, y, w, header_h, theme::HEADER_BG);
    ui.quad(x, y + header_h - 1.0, w, 1.0, theme::BORDER);
    ui.text(
        "Texture Packs",
        x + 7.0,
        y + 3.0,
        0.65,
        theme::TEXT_SECONDARY,
        font,
    );

    let mut cy = y + header_h + 4.0;

    if packs.is_empty() {
        // No packs loaded: show help text.
        ui.text(
            "Place .zip packs in",
            x + 7.0,
            cy,
            0.55,
            theme::TEXT_DIM,
            font,
        );
        cy += 13.0;
        ui.text(
            "texture_packs_dir to",
            x + 7.0,
            cy,
            0.55,
            theme::TEXT_DIM,
            font,
        );
        cy += 13.0;
        ui.text(
            "load custom textures.",
            x + 7.0,
            cy,
            0.55,
            theme::TEXT_DIM,
            font,
        );
        cy += 16.0;
        ui.text("Supported:", x + 7.0, cy, 0.55, theme::TEXT_SECONDARY, font);
        cy += 13.0;
        for fmt in &[
            "pack.toml (metadata)",
            "textures.toml (tile map)",
            "animations.toml (frames)",
            "*.png (textures)",
        ] {
            ui.text(fmt, x + 14.0, cy, 0.48, theme::TEXT_DIM, font);
            cy += 12.0;
        }
    } else {
        // Show loaded packs with metadata.
        for pack in packs {
            // Enable/disable indicator
            let icon = if pack.enabled { "\u{25CF}" } else { "\u{25CB}" };
            let icon_color = if pack.enabled {
                theme::ACCENT
            } else {
                theme::TEXT_DIM
            };
            ui.text(icon, x + 5.0, cy, 0.55, icon_color, font);
            ui.text(&pack.name, x + 16.0, cy, 0.60, theme::TEXT_PRIMARY, font);
            cy += 14.0;

            // Version + author line
            if !pack.version.is_empty() || !pack.author.is_empty() {
                let detail = match (pack.version.is_empty(), pack.author.is_empty()) {
                    (false, false) => format!("v{} by {}", pack.version, pack.author),
                    (false, true) => format!("v{}", pack.version),
                    (true, false) => format!("by {}", pack.author),
                    (true, true) => String::new(),
                };
                if !detail.is_empty() {
                    ui.text(&detail, x + 16.0, cy, 0.45, theme::TEXT_DIM, font);
                    cy += 12.0;
                }
            }

            // Stats line
            let stats = if pack.animation_count > 0 {
                format!("{} tiles, {} anims", pack.tile_count, pack.animation_count)
            } else {
                format!("{} tiles", pack.tile_count)
            };
            ui.text(&stats, x + 16.0, cy, 0.45, theme::TEXT_DIM, font);
            cy += 12.0;

            // Description (truncated)
            if !pack.description.is_empty() {
                let desc: String = pack.description.chars().take(28).collect();
                let desc = if pack.description.len() > 28 {
                    format!("{}...", desc)
                } else {
                    desc
                };
                ui.text(&desc, x + 16.0, cy, 0.42, theme::TEXT_SECONDARY, font);
                cy += 11.0;
            }
            cy += 4.0;
        }
    }
    cy
}
