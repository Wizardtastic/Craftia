//! Shared drawing context for the edit panels.
//!
//! Both the left tool panel and the right options panel draw their rows
//! through small helper functions that need the font, mouse position and
//! click state. A single shared `PanelCtx` keeps those helpers under the
//! `too_many_arguments` threshold and avoids duplicating the struct.

use voxel_render::FontAtlas;

/// Shared drawing context for the edit panels: font, mouse position and click
/// state. Bundled so the per-row helpers stay under the `too_many_arguments`
/// threshold.
#[derive(Clone, Copy)]
pub(crate) struct PanelCtx<'a> {
    pub(crate) font: &'a FontAtlas,
    pub(crate) mx: f32,
    pub(crate) my: f32,
    pub(crate) ui_click: bool,
}
