//! Brush presets: save/load/delete brush configurations to TOML files.
//!
//! Stored in `assets/brushes/*.toml`. Phase 9 infrastructure ΓÇö not yet
//! wired to the UI but will be once preset dropdown is added to the
//! right panel.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
// BlockId used in PresetWeight resolution, will be needed when wired to UI
use voxel_core::BlockId;

#[allow(unused_imports)] // BrushPalette/WeightedBlock used in preset palette roundtrip
use super::{BrushPalette, BrushShape, BrushTool, PaintMode, WeightedBlock};

/// A saved brush configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrushPreset {
    pub name: String,
    pub shape: String,
    pub radius: f32,
    pub block: String,
    pub replace: bool,
    pub strength: f32,
    pub paint_mode: String,
    pub hollow: bool,
    pub surface_only: bool,
    #[serde(default)]
    pub palette: Vec<PresetWeight>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PresetWeight {
    pub block: String,
    pub weight: f32,
}

/// List all preset files in the given directory.
pub fn list_presets(dir: &Path) -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "toml") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    names.push(stem.to_string());
                }
            }
        }
    }
    names.sort();
    names
}

/// Load a preset by name from the given directory.
pub fn load_preset(dir: &Path, name: &str) -> Option<BrushPreset> {
    let path = dir.join(format!("{}.toml", name));
    let text = std::fs::read_to_string(&path).ok()?;
    toml::from_str(&text).ok()
}

/// Save a preset to the given directory.
pub fn save_preset(dir: &Path, preset: &BrushPreset) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{}.toml", preset.name));
    let text = toml::to_string_pretty(preset).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| e.to_string())
}

/// Delete a preset file.
pub fn delete_preset(dir: &Path, name: &str) -> Result<(), String> {
    let path = dir.join(format!("{}.toml", name));
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| e.to_string())
    } else {
        Err("Preset not found".to_string())
    }
}

/// Convert a BrushTool to a BrushPreset for saving.
pub fn brush_to_preset(
    brush: &BrushTool,
    name: &str,
    registry: &voxel_world::BlockRegistry,
) -> BrushPreset {
    let block_name = registry.get(brush.block).name.to_string();
    let palette = brush
        .palette
        .entries
        .iter()
        .map(|w| PresetWeight {
            block: registry.get(w.block).name.to_string(),
            weight: w.weight,
        })
        .collect();

    BrushPreset {
        name: name.to_string(),
        shape: brush.shape.label().to_string(),
        radius: brush.radius,
        block: block_name,
        replace: brush.replace,
        strength: brush.strength,
        paint_mode: brush.paint_mode.label().to_string(),
        hollow: brush.hollow,
        surface_only: brush.surface_only,
        palette,
    }
}

/// Apply a preset to a BrushTool.
pub fn apply_preset(
    preset: &BrushPreset,
    brush: &mut BrushTool,
    registry: &voxel_world::BlockRegistry,
) {
    brush.shape = match preset.shape.as_str() {
        "Sphere" => BrushShape::Sphere,
        "Cylinder" => BrushShape::Cylinder,
        "Box" => BrushShape::Box,
        _ => BrushShape::Sphere,
    };
    brush.radius = preset.radius;
    brush.replace = preset.replace;
    brush.strength = preset.strength.clamp(0.0, 1.0);
    brush.paint_mode = match preset.paint_mode.as_str() {
        "Replace" => PaintMode::Replace,
        "Overlay" => PaintMode::Overlay,
        _ => PaintMode::Fill,
    };
    brush.hollow = preset.hollow;
    brush.surface_only = preset.surface_only;

    // Resolve block name.
    if let Some(id) = registry.id_of(&preset.block) {
        brush.block = id;
    }

    // Resolve palette.
    brush.palette.clear();
    brush.palette.enabled = !preset.palette.is_empty();
    for pw in &preset.palette {
        if let Some(id) = registry.id_of(&pw.block) {
            brush.palette.add(id, pw.weight);
        }
    }
}

/// Get the default presets directory (assets/brushes/).
pub fn default_presets_dir() -> PathBuf {
    PathBuf::from("assets").join("brushes")
}
