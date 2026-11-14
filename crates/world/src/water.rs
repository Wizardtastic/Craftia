//! Water bodies and their physical state.
//!
//! Water is no longer simulated as a per-voxel `0..8` flow level. Instead every
//! water body is described by a **surface height field** (one entry per
//! horizontal column) plus the properties of the body that owns that column:
//!
//! * [`WaterKind::Ocean`] — open sea. Wave amplitude grows with depth (and
//!   therefore with distance from the shore); wind-driven swell rolls in from
//!   the deep water and dies down to a ripple at the beach.
//! * [`WaterKind::River`] — a channel with a downstream current. Wide, steep
//!   channels carry rapids (high-frequency chop, foam and a strong push).
//! * [`WaterKind::Lake`] — a deep, enclosed body: completely still.
//! * [`WaterKind::Pond`] — small player-made or worldgen pools; a faint ripple.
//! * [`WaterKind::Aquifer`] — cave pools; still, no surface waves.
//!
//! The wave field is analytic (deep-water dispersion for the ocean) so it can
//! be evaluated identically on the CPU for entity buoyancy and on the GPU for
//! vertex displacement. That keeps a floating object riding exactly the same
//! wave the player sees.
//!
//! [`WaterColumnMeta`] is the compact per-column record stored inside a chunk.
//! The world maintains a global map of [`WaterColumn`]s for each `x/z`, holding
//! every disjoint vertical body (for example a lake and a cave aquifer below
//! it) so physics, currents and caustics can address the correct body.

use glam::Vec2;
use voxel_core::SEA_LEVEL;

// Emitted by crates/world/build.rs from src/water_wave_model.rs. The render
// build script emits the GLSL form of this same expression tree, so the CPU
// and all shader paths share one ocean-wave definition.
include!(concat!(env!("OUT_DIR"), "/water_wave_generated.rs"));

/// Sea-level surface height in world units.
pub const OCEAN_LEVEL: f32 = SEA_LEVEL as f32;
/// Surface height used by generated lakes, above sea level so they are
/// enclosed bodies rather than an extension of the ocean.
pub const LAKE_LEVEL: f32 = SEA_LEVEL as f32 + 10.0;

/// Maximum horizontal current speed of a flowing body, blocks/second.
pub const MAX_FLOW_SPEED: f32 = 2.6;

/// Empty cell in a render-side water-surface grid. Must match the sentinel
/// constant in `shaders/chunk.frag`.
pub const WATER_SURFACE_NONE: f32 = -1000.0;

/// Number of vertically stacked water bodies uploaded per grid cell. The
/// topmost body comes first; a cave aquifer below a lake/ocean is the second.
pub const WATER_SURFACE_LAYERS: usize = 2;
/// Floats per grid cell: `(surface, bed)` for each layer. A negative surface
/// marks a dry wet-edge fringe cell whose bed is the fringe height.
pub const WATER_SURFACE_FLOATS_PER_CELL: usize = WATER_SURFACE_LAYERS * 2;

/// Classification of a water body. Stored as a `u8` in [`WaterColumnMeta`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WaterKind {
    /// Not water (also used for lava, which has no water-body metadata).
    #[default]
    None = 0,
    /// Open sea/ocean.
    Ocean = 1,
    /// Flowing river or stream.
    River = 2,
    /// Still, deep lake.
    Lake = 3,
    /// Small pond / player-placed water.
    Pond = 4,
    /// Enclosed cave pool.
    Aquifer = 5,
}

impl WaterKind {
    pub const ALL: [WaterKind; 5] = [
        WaterKind::Ocean,
        WaterKind::River,
        WaterKind::Lake,
        WaterKind::Pond,
        WaterKind::Aquifer,
    ];

    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Ocean,
            2 => Self::River,
            3 => Self::Lake,
            4 => Self::Pond,
            5 => Self::Aquifer,
            _ => Self::None,
        }
    }

    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Human-readable name for debug overlays and telemetry.
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ocean => "ocean",
            Self::River => "river",
            Self::Lake => "lake",
            Self::Pond => "pond",
            Self::Aquifer => "aquifer",
        }
    }

    /// Bodies that are completely still at the surface.
    pub fn is_still(self) -> bool {
        matches!(self, Self::Lake | Self::Aquifer)
    }
}

/// Compact per-column water metadata stored in a [`crate::Chunk`].
///
/// `fill` is the fraction of the topmost water voxel that is filled with
/// water (`255` = a full voxel). `flow_angle` and the two normalized strengths
/// are quantized to one byte each to keep the record tiny (5 bytes/column,
/// 1.25 KiB per chunk).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaterColumnMeta {
    /// [`WaterKind`] as `u8` (`0` = no water metadata).
    pub kind: u8,
    /// Top-cell fill fraction, `0..=255`.
    pub fill: u8,
    /// Flow direction, `0..=255` mapped to `0..TAU`.
    pub flow_angle: u8,
    /// Normalized current speed, `0..=255` mapped to `0..MAX_FLOW_SPEED`.
    pub flow_speed: u8,
    /// Normalized rapid strength, `0..=255`.
    pub rapids: u8,
}

impl WaterColumnMeta {
    pub const NONE: Self = Self {
        kind: 0,
        fill: 0,
        flow_angle: 0,
        flow_speed: 0,
        rapids: 0,
    };

    pub const BYTES: usize = 5;

    pub fn new(
        kind: WaterKind,
        fill: u8,
        flow_angle: u8,
        flow_speed: u8,
        rapids: u8,
    ) -> Self {
        Self {
            kind: kind.as_u8(),
            fill,
            flow_angle,
            flow_speed,
            rapids,
        }
    }

    #[inline]
    pub fn kind(&self) -> WaterKind {
        WaterKind::from_u8(self.kind)
    }

    #[inline]
    pub fn is_some(&self) -> bool {
        self.kind != 0
    }
}

/// Live top-surface record for one world column.
#[derive(Clone, Copy, Debug)]
pub struct WaterColumn {
    pub kind: WaterKind,
    /// Height of the body surface, in world units (may be fractional).
    pub surface_y: f32,
    /// Height of the lowest water voxel, in world units.
    pub bed_y: f32,
    /// Fill fraction of the top voxel (`0..=1`).
    pub fill: f32,
    /// Flow direction in radians.
    pub flow_angle: f32,
    /// Current speed in blocks/second.
    pub flow_speed: f32,
    /// Rapid strength, `0..=1`.
    pub rapids: f32,
}

impl Default for WaterColumn {
    fn default() -> Self {
        Self {
            kind: WaterKind::None,
            surface_y: 0.0,
            bed_y: 0.0,
            fill: 1.0,
            flow_angle: 0.0,
            flow_speed: 0.0,
            rapids: 0.0,
        }
    }
}

impl WaterColumn {
    #[inline]
    pub fn depth(&self) -> f32 {
        (self.surface_y - self.bed_y).max(0.0)
    }

    #[inline]
    pub fn flow_dir(&self) -> Vec2 {
        Vec2::new(self.flow_angle.cos(), self.flow_angle.sin())
    }

    /// Current velocity vector (blocks/second) at the surface.
    #[inline]
    pub fn flow(&self) -> Vec2 {
        self.flow_dir() * self.flow_speed
    }

    /// Surface wave amplitude, in blocks.
    ///
    /// Ocean waves grow with depth: the deeper the water, the further the
    /// column is from the shore, and the larger the swell. Lakes and cave
    /// pools are flat; rivers only ripple, except in rapids.
    pub fn wave_amplitude(&self) -> f32 {
        match self.kind {
            WaterKind::Ocean => {
                // Asymptotic growth with depth. A 1-block-deep beach column
                // gets a small chop; a 20-block-deep open-ocean column gets
                // ~1.5 blocks of swell. The old 2.25-block ceiling produced
                // metre-high ridges that read as repeating walls when every
                // column shared the same wave direction. Shoaling is stronger
                // in shallow water, so the curve rises faster with depth but
                // tops out lower offshore.
                0.04 + 1.55 * (1.0 - (-self.depth() / 5.0).exp())
            }
            WaterKind::River => 0.025 + 0.32 * self.rapids,
            WaterKind::Pond => 0.012,
            WaterKind::Lake | WaterKind::Aquifer | WaterKind::None => 0.0,
        }
    }

    /// Surface height including the analytic wave field at world `(x, z)` and
    /// time `t`. Shared by the CPU physics and the shader wave model.
    #[inline]
    pub fn surface_at(&self, x: f32, z: f32, t: f32) -> f32 {
        self.surface_y
            + wave_height(
                self.kind,
                self.wave_amplitude(),
                self.flow_angle,
                self.rapids,
                x,
                z,
                t,
            )
    }
}

/// One water sample for entity physics / rendering.
#[derive(Clone, Copy, Debug)]
pub struct WaterSample {
    pub kind: WaterKind,
    /// Wave-adjusted surface height at the sample `x/z`.
    pub surface_y: f32,
    pub bed_y: f32,
    pub depth: f32,
    /// Surface current (blocks/second).
    pub flow: Vec2,
    pub flow_speed: f32,
    pub rapids: f32,
    /// Signed wave height above the base surface at the sample time.
    pub wave: f32,
}

/// Per-vertex water animation data produced by the world for the mesher.
#[derive(Clone, Copy, Debug, Default)]
pub struct WaterVertexInfo {
    /// [`WaterKind`] as `u8` (`0` = not animated water, e.g. lava).
    pub kind: u8,
    /// Height fraction of this voxel's surface (`255` = full voxel).
    pub fill: u8,
    /// Wave amplitude packed as 1/32 blocks (`0..=255`).
    pub amplitude: u8,
    /// Flow direction packed to `0..=255`.
    pub flow_angle: u8,
    /// Normalized flow speed packed to `0..=255`.
    pub flow_speed: u8,
    /// Normalized rapid strength packed to `0..=255`.
    pub rapids: u8,
    /// True when this voxel's top is the body surface and may be displaced.
    pub surface: bool,
}

impl WaterVertexInfo {
    #[inline]
    pub fn height(&self) -> f32 {
        self.fill as f32 / 255.0
    }

    /// Pack into the 32-bit `water` vertex attribute.
    ///
    /// Layout: bits 0..3 kind, bit 3 surface, bits 4..8 rapids,
    /// bits 8..16 amplitude (1/32 block), bits 16..24 flow angle,
    /// bits 24..32 flow speed.
    #[inline]
    pub fn pack(&self) -> u32 {
        (self.kind as u32 & 0x7)
            | ((self.surface as u32) << 3)
            | (((self.rapids as u32) >> 4) << 4)
            | ((self.amplitude as u32) << 8)
            | ((self.flow_angle as u32) << 16)
            | ((self.flow_speed as u32) << 24)
    }
}

/// Pack a `0..1` value into a byte.
#[inline]
pub fn pack_unit(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Unpack a byte into `0..1`.
#[inline]
pub fn unpack_unit(v: u8) -> f32 {
    v as f32 / 255.0
}

/// Pack an angle in radians into a byte (`0..TAU`).
#[inline]
pub fn pack_angle(angle: f32) -> u8 {
    let a = angle.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
    (a * 255.0).round() as u8
}

/// Unpack a byte into an angle in radians.
#[inline]
pub fn unpack_angle(v: u8) -> f32 {
    v as f32 / 255.0 * std::f32::consts::TAU
}

/// Build a live water column from chunk metadata and a scanned surface.
pub fn column_from_meta(
    meta: WaterColumnMeta,
    surface_y: f32,
    bed_y: f32,
    fill: f32,
) -> WaterColumn {
    WaterColumn {
        kind: meta.kind(),
        surface_y,
        bed_y,
        fill: fill.clamp(0.0, 1.0),
        flow_angle: unpack_angle(meta.flow_angle),
        flow_speed: unpack_unit(meta.flow_speed) * MAX_FLOW_SPEED,
        rapids: unpack_unit(meta.rapids),
    }
}

/// Analytic surface wave height shared by the CPU and shader.
///
/// The ocean uses six superposed deep-water swells whose phase speed obeys
/// the dispersion relation `ω = sqrt(g·k)`. A river uses a short, fast ripple
/// train aligned with its current plus high-frequency chop in rapids. Still
/// bodies (lake/aquifer) and ponds never move.
pub fn wave_height(
    kind: WaterKind,
    amplitude: f32,
    angle: f32,
    rapids: f32,
    x: f32,
    z: f32,
    t: f32,
) -> f32 {
    if amplitude <= 0.0 {
        return 0.0;
    }
    water_wave_height(kind.as_u8(), amplitude, angle, rapids, x, z, t)
}

/// Infer a body kind for water that has no stored metadata (old saves and
/// water placed by commands). Surface height is the only reliable clue.
pub fn infer_kind(surface_y: f32) -> WaterKind {
    if surface_y <= OCEAN_LEVEL + 0.75 {
        WaterKind::Ocean
    } else if surface_y >= LAKE_LEVEL - 2.0 {
        WaterKind::Lake
    } else {
        WaterKind::Pond
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_roundtrips_through_u8() {
        for kind in WaterKind::ALL {
            assert_eq!(WaterKind::from_u8(kind.as_u8()), kind);
        }
        assert_eq!(WaterKind::from_u8(99), WaterKind::None);
    }

    #[test]
    fn column_meta_roundtrips() {
        let meta = WaterColumnMeta::new(
            WaterKind::River,
            200,
            pack_angle(1.25),
            pack_unit(0.75),
            pack_unit(0.5),
        );
        let col = column_from_meta(meta, 64.5, 60.0, 200.0 / 255.0);
        assert_eq!(col.kind, WaterKind::River);
        assert!((col.surface_y - 64.5).abs() < 1e-6);
        assert!((col.flow_speed - 0.75 * MAX_FLOW_SPEED).abs() < 0.01);
        assert!((col.rapids - 0.5).abs() < 0.01);
    }

    #[test]
    fn ocean_waves_grow_with_depth() {
        let shallow = WaterColumn {
            kind: WaterKind::Ocean,
            surface_y: 62.0,
            bed_y: 60.0,
            ..Default::default()
        };
        let deep = WaterColumn {
            kind: WaterKind::Ocean,
            surface_y: 62.0,
            bed_y: 30.0,
            ..Default::default()
        };
        assert!(deep.wave_amplitude() > shallow.wave_amplitude() + 1.0);
        assert!(shallow.wave_amplitude() > 0.0);
    }

    #[test]
    fn lakes_and_aquifers_are_still() {
        for kind in [WaterKind::Lake, WaterKind::Aquifer] {
            let col = WaterColumn {
                kind,
                surface_y: 80.0,
                bed_y: 60.0,
                flow_angle: 1.0,
                flow_speed: 1.0,
                rapids: 1.0,
                fill: 1.0,
            };
            assert_eq!(col.wave_amplitude(), 0.0);
            assert_eq!(wave_height(kind, 2.0, 0.4, 1.0, 1.0, 2.0, 3.0), 0.0);
        }
    }

    #[test]
    fn river_rapids_raise_wave_height() {
        let calm = WaterColumn {
            kind: WaterKind::River,
            surface_y: 62.0,
            bed_y: 58.0,
            flow_angle: 0.0,
            flow_speed: 1.0,
            rapids: 0.0,
            fill: 1.0,
        };
        let mut rough = calm;
        rough.rapids = 1.0;
        assert!(rough.wave_amplitude() > calm.wave_amplitude());
        let x = 12.3;
        let z = -4.7;
        let t = 2.1;
        let calm_h = wave_height(calm.kind, calm.wave_amplitude(), 0.0, 0.0, x, z, t);
        let rough_h = wave_height(rough.kind, rough.wave_amplitude(), 0.0, 1.0, x, z, t);
        assert!(rough_h.abs() > calm_h.abs());
    }

    #[test]
    fn wave_field_is_deterministic() {
        let a = wave_height(WaterKind::Ocean, 1.7, 0.9, 0.0, 3.0, 5.0, 11.0);
        let b = wave_height(WaterKind::Ocean, 1.7, 0.9, 0.0, 3.0, 5.0, 11.0);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn water_vertex_pack_roundtrips_fields() {
        let info = WaterVertexInfo {
            kind: WaterKind::Ocean.as_u8(),
            fill: 128,
            amplitude: 64,
            flow_angle: 200,
            flow_speed: 90,
            rapids: 160,
            surface: true,
        };
        let packed = info.pack();
        assert_eq!(packed & 0x7, WaterKind::Ocean.as_u8() as u32);
        assert_eq!((packed >> 3) & 1, 1);
        assert_eq!((packed >> 8) & 0xFF, 64);
        assert_eq!((packed >> 16) & 0xFF, 200);
        assert_eq!((packed >> 24) & 0xFF, 90);
        assert_eq!((packed >> 4) & 0xF, 10);
    }

    #[test]
    fn infer_kind_from_height() {
        assert_eq!(infer_kind(OCEAN_LEVEL), WaterKind::Ocean);
        assert_eq!(infer_kind(LAKE_LEVEL), WaterKind::Lake);
        assert_eq!(infer_kind(68.0), WaterKind::Pond);
    }
}
