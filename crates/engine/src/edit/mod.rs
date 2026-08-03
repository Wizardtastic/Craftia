//! World editing state machine: edit mode, brush tools, block palette.
//!
//! Toggled by the `EditMode` keybind (default: X). When active, the
//! full VoxEdit-style editor UI is shown with menu bar, category bar,
//! left tool panel, viewport, right options panel, and status bar.

pub mod brush;
pub mod ctx;
pub mod filter;
pub mod left_panel;
pub mod menu_bar;
pub mod paint;
pub mod right_panel;
pub mod select;
pub mod status_bar;
pub mod terrain;
pub mod theme;
pub mod toolbar;

use glam::IVec3;
use voxel_core::BlockId;

// ── Tool categories ──────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ToolCategory {
    Selection,
    Draw,
    Paint,
    Heightmap,
    Manipulation,
    Utility,
}

impl ToolCategory {
    pub fn label(self) -> &'static str {
        match self {
            Self::Selection => "Selection",
            Self::Draw => "Draw",
            Self::Paint => "Paint",
            Self::Heightmap => "Heightmap",
            Self::Manipulation => "Manipulation",
            Self::Utility => "Utility",
        }
    }

    pub const ALL: &[ToolCategory] = &[
        Self::Selection,
        Self::Draw,
        Self::Paint,
        Self::Heightmap,
        Self::Manipulation,
        Self::Utility,
    ];
}

// ── Individual tools ─────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDef {
    pub id: &'static str,
    pub label: &'static str,
    pub category: ToolCategory,
}

pub const TOOL_DEFS: &[ToolDef] = &[
    ToolDef {
        id: "box_select",
        label: "Box Select",
        category: ToolCategory::Selection,
    },
    ToolDef {
        id: "magic_select",
        label: "Magic Select",
        category: ToolCategory::Selection,
    },
    ToolDef {
        id: "freehand_select",
        label: "Freehand",
        category: ToolCategory::Selection,
    },
    ToolDef {
        id: "lasso",
        label: "Lasso",
        category: ToolCategory::Selection,
    },
    ToolDef {
        id: "freehand_draw",
        label: "Freehand",
        category: ToolCategory::Draw,
    },
    ToolDef {
        id: "sculpt",
        label: "Sculpt",
        category: ToolCategory::Draw,
    },
    ToolDef {
        id: "rock",
        label: "Rock",
        category: ToolCategory::Draw,
    },
    ToolDef {
        id: "weld",
        label: "Weld",
        category: ToolCategory::Draw,
    },
    ToolDef {
        id: "melt",
        label: "Melt",
        category: ToolCategory::Draw,
    },
    ToolDef {
        id: "shape",
        label: "Shape",
        category: ToolCategory::Draw,
    },
    ToolDef {
        id: "path",
        label: "Path",
        category: ToolCategory::Draw,
    },
    ToolDef {
        id: "stamp",
        label: "Stamp",
        category: ToolCategory::Draw,
    },
    ToolDef {
        id: "text",
        label: "Text",
        category: ToolCategory::Draw,
    },
    ToolDef {
        id: "painter",
        label: "Painter",
        category: ToolCategory::Paint,
    },
    ToolDef {
        id: "noise_painter",
        label: "Noise Painter",
        category: ToolCategory::Paint,
    },
    ToolDef {
        id: "gradient",
        label: "Gradient",
        category: ToolCategory::Paint,
    },
    ToolDef {
        id: "biome_painter",
        label: "Biome Painter",
        category: ToolCategory::Paint,
    },
    ToolDef {
        id: "script_brush",
        label: "Script Brush",
        category: ToolCategory::Paint,
    },
    ToolDef {
        id: "elevation",
        label: "Elevation",
        category: ToolCategory::Heightmap,
    },
    ToolDef {
        id: "flatten",
        label: "Flatten",
        category: ToolCategory::Heightmap,
    },
    ToolDef {
        id: "slope",
        label: "Slope",
        category: ToolCategory::Heightmap,
    },
    ToolDef {
        id: "smooth",
        label: "Smooth",
        category: ToolCategory::Manipulation,
    },
    ToolDef {
        id: "distort",
        label: "Distort",
        category: ToolCategory::Manipulation,
    },
    ToolDef {
        id: "roughen",
        label: "Roughen",
        category: ToolCategory::Manipulation,
    },
    ToolDef {
        id: "shatter",
        label: "Shatter",
        category: ToolCategory::Manipulation,
    },
    ToolDef {
        id: "extrude",
        label: "Extrude",
        category: ToolCategory::Manipulation,
    },
    ToolDef {
        id: "modify",
        label: "Modify",
        category: ToolCategory::Manipulation,
    },
    ToolDef {
        id: "ruler",
        label: "Ruler",
        category: ToolCategory::Utility,
    },
    ToolDef {
        id: "annotation",
        label: "Annotation",
        category: ToolCategory::Utility,
    },
];

pub fn default_tool(cat: ToolCategory) -> &'static str {
    match cat {
        ToolCategory::Selection => "box_select",
        ToolCategory::Draw => "freehand_draw",
        ToolCategory::Paint => "painter",
        ToolCategory::Heightmap => "elevation",
        ToolCategory::Manipulation => "smooth",
        ToolCategory::Utility => "ruler",
    }
}

pub fn tools_for_category(cat: ToolCategory) -> &'static [ToolDef] {
    let mut start = 0;
    let mut count = 0;
    let mut found = false;
    for (i, t) in TOOL_DEFS.iter().enumerate() {
        if t.category == cat {
            if !found {
                start = i;
                found = true;
            }
            count += 1;
        } else if found {
            break;
        }
    }
    if found {
        &TOOL_DEFS[start..start + count]
    } else {
        &[]
    }
}

// ── Top-level editor state ───────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Default)]
pub enum EditModeState {
    #[default]
    Inactive,
    Active {
        tool: EditTool,
    },
}

impl EditModeState {
    pub fn is_active(&self) -> bool {
        matches!(self, EditModeState::Active { .. })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum EditTool {
    Brush(BrushTool),
    #[allow(dead_code)]
    // Scaffolded tool UIs (right_panel) exist, but the toolbar never constructs these yet
    Select(select::SelectTool),
    #[allow(dead_code)]
    Terrain(terrain::TerrainTool),
    #[allow(dead_code)]
    Paint(paint::PaintTool),
    #[allow(dead_code)]
    Filter(filter::FilterStack),
}

impl EditTool {
    pub fn shape(&self) -> BrushShape {
        match self {
            EditTool::Brush(b) => b.shape,
            EditTool::Terrain(t) => t.shape,
            EditTool::Paint(p) => p.shape,
            _ => BrushShape::Box,
        }
    }

    pub fn radius(&self) -> f32 {
        match self {
            EditTool::Brush(b) => b.radius,
            EditTool::Terrain(t) => t.radius,
            EditTool::Paint(p) => p.radius,
            _ => 3.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushShape {
    Sphere,
    Cylinder,
    Box,
}

impl BrushShape {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sphere => "Sphere",
            Self::Cylinder => "Cylinder",
            Self::Box => "Box",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintMode {
    Fill,
    Replace,
    Overlay,
}

impl PaintMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fill => "Fill",
            Self::Replace => "Replace",
            Self::Overlay => "Overlay",
        }
    }
}

// ── Multi-block brush palette (Phase 3) ──────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct WeightedBlock {
    pub block: BlockId,
    pub weight: f32,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct BrushPalette {
    pub entries: Vec<WeightedBlock>,
    pub enabled: bool,
}

impl BrushPalette {
    /// Pick a block from the weighted palette using a deterministic hash.
    /// The `seed` parameter should vary per-position to get different picks.
    pub fn pick_with_seed(&self, seed: u32) -> BlockId {
        if self.entries.is_empty() {
            return BlockId::AIR;
        }
        let total: f32 = self.entries.iter().map(|e| e.weight).sum();
        if total <= 0.0 {
            return self.entries.last().unwrap().block;
        }
        // Simple hash-based pseudo-random in [0, total).
        let hash = seed.wrapping_mul(2654435761); // Knuth multiplicative hash
        let r = (hash as f32 / u32::MAX as f32) * total;
        let mut accum = 0.0;
        for entry in &self.entries {
            accum += entry.weight;
            if r <= accum {
                return entry.block;
            }
        }
        self.entries.last().unwrap().block
    }

    /// Pick a random block (uses time-based seed for true randomness).
    pub fn pick(&self) -> BlockId {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        self.pick_with_seed(seed)
    }

    pub fn add(&mut self, block: BlockId, weight: f32) {
        self.entries.push(WeightedBlock { block, weight });
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ── Brush state ──────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct BrushTool {
    pub shape: BrushShape,
    pub radius: f32,
    pub block: BlockId,
    pub replace: bool,
    pub target: Option<BlockId>,
    pub strength: f32,
    pub paint_mode: PaintMode,
    pub hollow: bool,
    pub surface_only: bool,
    pub palette: BrushPalette,
}

impl Default for BrushTool {
    fn default() -> Self {
        Self {
            shape: BrushShape::Sphere,
            radius: 3.0,
            block: BlockId::new(1),
            replace: false,
            target: None,
            strength: 1.0,
            paint_mode: PaintMode::Fill,
            hollow: false,
            surface_only: false,
            palette: BrushPalette::default(),
        }
    }
}

// ── Block categories (for palette) ───────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BlockCategory {
    Stone,
    Wood,
    Glass,
    Plant,
    Ore,
    Dirt,
    Sand,
    Water,
    Other,
}

pub fn categorize(name: &str) -> BlockCategory {
    let lower = name.to_lowercase();
    if lower.contains("stone") || lower.contains("cobble") || lower.contains("rock") {
        BlockCategory::Stone
    } else if lower.contains("log") || lower.contains("plank") || lower.contains("wood") {
        BlockCategory::Wood
    } else if lower.contains("glass") {
        BlockCategory::Glass
    } else if lower.contains("leaves")
        || lower.contains("grass")
        || lower.contains("flower")
        || lower.contains("plant")
        || lower.contains("poppy")
        || lower.contains("dandelion")
        || lower.contains("mushroom")
    {
        BlockCategory::Plant
    } else if lower.contains("ore") {
        BlockCategory::Ore
    } else if lower.contains("dirt") || lower.contains("moss") || lower.contains("podzol") {
        BlockCategory::Dirt
    } else if lower.contains("sand") {
        BlockCategory::Sand
    } else if lower.contains("water") || lower.contains("ice") {
        BlockCategory::Water
    } else {
        BlockCategory::Other
    }
}

// ── History ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub label: String,
    pub is_current: bool,
}

// ── Editor state ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct EditState {
    pub mode: EditModeState,

    // Category / tool selection
    pub active_category: ToolCategory,
    pub active_tool_id: String,

    // Palette
    pub palette_open: bool,
    pub palette_search: String,
    pub recently_used: Vec<BlockId>,
    pub categories: Vec<(BlockCategory, Vec<(BlockId, String)>)>,

    // Brush preview
    pub preview_valid: bool,
    pub brush_center: Option<IVec3>,

    // History (UI display only — real undo is in UndoRedoState)
    pub history: Vec<HistoryEntry>,

    // UI interaction state
    pub search_focused: bool,
    pub ui_click: bool,
    pub scroll_delta: f32,

    // Right-panel dropdown state
    pub open_dropdown: Option<String>,

    // World properties
    pub show_grid: bool,
    pub show_chunks: bool,
}

impl Default for EditState {
    fn default() -> Self {
        Self {
            mode: EditModeState::Inactive,
            active_category: ToolCategory::Draw,
            active_tool_id: default_tool(ToolCategory::Draw).to_string(),
            palette_open: true,
            palette_search: String::new(),
            recently_used: Vec::new(),
            categories: Vec::new(),
            preview_valid: false,
            brush_center: None,
            history: Vec::new(),
            search_focused: false,
            ui_click: false,
            scroll_delta: 0.0,
            open_dropdown: None,
            show_grid: true,
            show_chunks: false,
        }
    }
}

impl EditState {
    pub fn toggle(&mut self) {
        self.mode = match self.mode {
            EditModeState::Inactive => EditModeState::Active {
                tool: EditTool::Brush(BrushTool::default()),
            },
            EditModeState::Active { .. } => EditModeState::Inactive,
        };
        if !self.mode.is_active() {
            self.palette_open = false;
            self.brush_center = None;
        }
    }

    pub fn add_recent(&mut self, block: BlockId) {
        self.recently_used.retain(|b| *b != block);
        self.recently_used.insert(0, block);
        if self.recently_used.len() > 10 {
            self.recently_used.pop();
        }
    }

    pub fn brush_mut(&mut self) -> Option<&mut BrushTool> {
        match &mut self.mode {
            EditModeState::Active {
                tool: EditTool::Brush(b),
            } => Some(b),
            _ => None,
        }
    }

    pub fn brush_ref(&self) -> Option<&BrushTool> {
        match &self.mode {
            EditModeState::Active {
                tool: EditTool::Brush(b),
            } => Some(b),
            _ => None,
        }
    }

    pub fn select_mut(&mut self) -> Option<&mut select::SelectTool> {
        match &mut self.mode {
            EditModeState::Active {
                tool: EditTool::Select(s),
            } => Some(s),
            _ => None,
        }
    }

    pub fn select_ref(&self) -> Option<&select::SelectTool> {
        match &self.mode {
            EditModeState::Active {
                tool: EditTool::Select(s),
            } => Some(s),
            _ => None,
        }
    }

    pub fn terrain_mut(&mut self) -> Option<&mut terrain::TerrainTool> {
        match &mut self.mode {
            EditModeState::Active {
                tool: EditTool::Terrain(t),
            } => Some(t),
            _ => None,
        }
    }

    pub fn terrain_ref(&self) -> Option<&terrain::TerrainTool> {
        match &self.mode {
            EditModeState::Active {
                tool: EditTool::Terrain(t),
            } => Some(t),
            _ => None,
        }
    }

    pub fn paint_mut(&mut self) -> Option<&mut paint::PaintTool> {
        match &mut self.mode {
            EditModeState::Active {
                tool: EditTool::Paint(p),
            } => Some(p),
            _ => None,
        }
    }

    pub fn paint_ref(&self) -> Option<&paint::PaintTool> {
        match &self.mode {
            EditModeState::Active {
                tool: EditTool::Paint(p),
            } => Some(p),
            _ => None,
        }
    }

    pub fn filter_mut(&mut self) -> Option<&mut filter::FilterStack> {
        match &mut self.mode {
            EditModeState::Active {
                tool: EditTool::Filter(f),
            } => Some(f),
            _ => None,
        }
    }

    pub fn filter_ref(&self) -> Option<&filter::FilterStack> {
        match &self.mode {
            EditModeState::Active {
                tool: EditTool::Filter(f),
            } => Some(f),
            _ => None,
        }
    }

    pub fn consume_frame(&mut self) {
        self.ui_click = false;
        self.scroll_delta = 0.0;
        self.open_dropdown = None;
    }

    pub fn active_tool_label(&self) -> &str {
        TOOL_DEFS
            .iter()
            .find(|t| t.id == self.active_tool_id)
            .map(|t| t.label)
            .unwrap_or("Unknown")
    }
}
