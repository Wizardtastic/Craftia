//! Block registry: maps `BlockId` -> properties (solidity, opacity, textures).
//!
//! This is the runtime side of the data-driven content system. In a future
//! iteration `voxel-assets` populates this from JSON; for now we build a
//! hardcoded builtin set so the world has something to generate with.

use std::sync::Arc;

use glam::IVec3;
use voxel_core::BlockId;

/// Default emission color: warm white.
pub const DEFAULT_EMISSION_COLOR: [u8; 3] = [255, 248, 240];

/// The six cube faces, in the order the mesher and shaders expect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Face {
    NegX = 0,
    PosX = 1,
    NegY = 2,
    PosY = 3,
    NegZ = 4,
    PosZ = 5,
}

impl Face {
    pub const ALL: [Face; 6] = [
        Face::NegX,
        Face::PosX,
        Face::NegY,
        Face::PosY,
        Face::NegZ,
        Face::PosZ,
    ];

    /// Outward unit normal for this face.
    pub fn normal(self) -> IVec3 {
        match self {
            Face::NegX => IVec3::new(-1, 0, 0),
            Face::PosX => IVec3::new(1, 0, 0),
            Face::NegY => IVec3::new(0, -1, 0),
            Face::PosY => IVec3::new(0, 1, 0),
            Face::NegZ => IVec3::new(0, 0, -1),
            Face::PosZ => IVec3::new(0, 0, 1),
        }
    }
}

/// Coarse classification used by worldgen and physics. More kinds can be added
/// without breaking the registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum BlockKind {
    #[default]
    Air,
    Solid,
    Liquid,
    Foliage,
    Transparent,
}

/// Per-face texture tile index into the texture atlas.
#[derive(Clone, Copy, Debug)]
pub struct BlockTextures {
    /// One tile index per face; uses a single tile for all faces if `same` was
    /// used at registration.
    pub tiles: [u16; 6],
}

impl BlockTextures {
    /// All six faces share one tile (e.g. stone, planks).
    pub fn same(tile: u16) -> Self {
        Self { tiles: [tile; 6] }
    }
    /// Top/bottom/side variant (e.g. grass: grass_top, dirt, grass_side).
    pub fn top_bottom_side(top: u16, bottom: u16, side: u16) -> Self {
        // Face order: NegX, PosX, NegY, PosY, NegZ, PosZ
        Self {
            tiles: [side, side, bottom, top, side, side],
        }
    }
    pub fn tile(self, face: Face) -> u16 {
        self.tiles[face as usize]
    }
}

/// Tool types that can be used to mine blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ToolType {
    #[default]
    None,
    Pickaxe,
    Axe,
    Shovel,
    Hoe,
    Sword,
}

/// Tool tier levels (0 = hand, 1 = wood, 2 = stone, 3 = iron, 4 = diamond, 5 = netherite).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ToolTier(pub u8);

impl ToolTier {
    pub const HAND: Self = Self(0);
    pub const WOOD: Self = Self(1);
    pub const STONE: Self = Self(2);
    pub const IRON: Self = Self(3);
    pub const DIAMOND: Self = Self(4);
    pub const NETHERITE: Self = Self(5);
}

/// Properties for a tool item.
#[derive(Clone, Copy, Debug)]
pub struct ToolProperties {
    pub tool_type: ToolType,
    pub tier: ToolTier,
    pub mining_speed: f32,
    pub attack_damage: f32,
    pub attack_speed: f32,
    pub max_durability: u16,
}

/// Properties for a food item.
#[derive(Clone, Copy, Debug)]
pub struct FoodProperties {
    pub hunger_restoration: f32,
    pub saturation_modifier: f32,
    pub eat_duration_ticks: u32,
    pub always_edible: bool,
}

/// Armor slot types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArmorSlot {
    Helmet,
    Chestplate,
    Leggings,
    Boots,
}

/// Properties for an armor item.
#[derive(Clone, Copy, Debug)]
pub struct ArmorProperties {
    pub slot: ArmorSlot,
    pub defense_points: u8,
    pub toughness: u8,
    pub max_durability: u16,
}

/// Condition for a block drop to occur.
#[derive(Clone, Copy, Debug)]
pub enum DropCondition {
    /// Always drops regardless of tool.
    Always,
    /// Only drops if the tool has Silk Touch.
    SilkTouchRequired,
    /// Only drops if the tool does NOT have Silk Touch.
    SilkTouchForbidden,
    /// Drop count scales with Fortune level.
    FortuneScaled { base: u16, max_extra: u16 },
    /// Drop this block's own assigned registry ID.
    SelfDrop,
}

/// A single drop entry for a block.
#[derive(Clone, Copy, Debug)]
pub struct BlockDrop {
    /// The block/item ID to drop.
    pub item: BlockId,
    /// Minimum number of items to drop.
    pub min_count: u16,
    /// Maximum number of items to drop.
    pub max_count: u16,
    /// Probability of this drop occurring (0.0 to 1.0).
    pub probability: f32,
    /// Condition that must be met for this drop.
    pub condition: DropCondition,
}

/// How a block animation loops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationMode {
    /// Loop continuously.
    Loop,
    /// Play forward then backward.
    PingPong,
    /// Play once and stop.
    Once,
}

/// Animation definition for a block's texture tiles.
#[derive(Clone, Debug)]
pub struct BlockAnimationDef {
    /// Frame sequence: each entry is (tile_index, duration_seconds).
    pub frames: Vec<(u32, f32)>,
    /// Frame 0 tile index (the canonical tile used during meshing).
    pub canonical_tile: u32,
    /// How the animation loops.
    pub mode: AnimationMode,
    /// Total duration (computed from frames).
    pub total_duration: f32,
}

impl BlockDrop {
    /// Create a simple drop that always drops 1 item.
    pub fn simple(item: BlockId) -> Self {
        Self {
            item,
            min_count: 1,
            max_count: 1,
            probability: 1.0,
            condition: DropCondition::Always,
        }
    }

    /// Create a drop with a count range.
    pub fn range(item: BlockId, min: u16, max: u16) -> Self {
        Self {
            item,
            min_count: min,
            max_count: max,
            probability: 1.0,
            condition: DropCondition::Always,
        }
    }

    /// Create a Fortune-scaled drop.
    pub fn fortune(item: BlockId, base: u16, max_extra: u16) -> Self {
        Self {
            item,
            min_count: base,
            max_count: base + max_extra,
            probability: 1.0,
            condition: DropCondition::FortuneScaled { base, max_extra },
        }
    }

    /// Create an explicit self-drop resolved when the registry assigns an ID.
    pub fn self_drop() -> Self {
        Self {
            item: BlockId::AIR,
            min_count: 1,
            max_count: 1,
            probability: 1.0,
            condition: DropCondition::SelfDrop,
        }
    }
}

/// Static properties of a block type.
#[derive(Clone, Debug)]
pub struct BlockDef {
    pub id: BlockId,
    /// Owned name. `Arc<str>` so the registry can be cheaply cloned without
    /// aliasing a global / leaking the string; also makes runtime-loaded names
    /// from JSON safe.
    pub name: Arc<str>,
    pub kind: BlockKind,
    /// Whether the block blocks entity movement (solids + liquids do, air doesn't).
    pub solid: bool,
    /// Whether the block fully occludes neighbouring faces (air/glass/leaves don't).
    pub opaque: bool,
    /// Whether the block can be broken by the player. False for bedrock etc.
    pub breakable: bool,
    /// Whether a block here can be replaced by placement (air, tall grass, etc.).
    pub replaceable: bool,
    pub textures: BlockTextures,
    /// Light emitted by this block (0–15). 0 for most, 14 for torches.
    pub emission: u8,
    /// RGB color of the emitted light. Default warm white `[255, 248, 240]`.
    /// Only meaningful when `emission > 0`.
    pub emission_color: [u8; 3],
    /// How much light this block absorbs (0–15). 0 = transparent, 15 = fully opaque.
    pub light_absorption: u8,
    /// RGBA tint for minimap display. Assigned by [`default_map_color`] heuristic
    /// or overridden per-block in JSON via `"map_color": [r, g, b, a]`.
    pub map_color: [u8; 4],
    /// Hardness: seconds to break with bare hand (0 = instant, -1 = unbreakable).
    pub hardness: f32,
    /// Blast resistance for TNT calculations.
    pub blast_resistance: f32,
    /// Required tool type to harvest this block (None = any tool).
    pub required_tool: ToolType,
    /// Minimum tool tier required to harvest (0 = any, 1 = wood+, etc.).
    pub required_tier: u8,
    /// Drop table for this block.
    pub drops: Vec<BlockDrop>,
    /// Optional animation definition for animated block textures.
    pub animation: Option<BlockAnimationDef>,
    /// Per-block material data driving the chunk shader's leaves SSS, wet-edge
    /// tint, sun caustics, and glass tinted-absorption effects.
    /// See `BlockMaterial` for the field semantics.
    pub material: BlockMaterial,
}

/// Per-block material data fed to the chunk fragment shader via the
/// `tile_material` UBO. All fields are zero-by-default (opt-in for each block
/// via the constructor helpers); the global uniform/lookup table on the GPU
/// is keyed by atlas tile index.
#[derive(Clone, Copy, Debug)]
pub struct BlockMaterial {
    /// Per-tile flags bitfield. See `BlockMaterialFlag`.
    pub flags: u8,
    /// Scalar roughness proxy (0 = mirror, 255 = matte). Currently used to
    /// modulate the wet-edge add (mirror surfaces don't wet).
    pub roughness: u8,
    /// Tile-local emissive boost (0 = none, 255 = full). Adds a constant to
    /// the surface in addition to torchlight; for now only used by lava.
    pub emissive: u8,
    /// RGBA tint applied to leaves' backlight contribution (subsurface).
    /// Default `[0; 4]` for opaque blocks.
    pub sss_tint: [u8; 4],
    /// RGBA tint for wet-edge additive (default `[0; 4]`).
    /// Only effective when the block is horizontally adjacent to a water source.
    pub wet_tint: [u8; 4],
    /// Absorption coefficients (R,G,B) for water-glass tinted absorption.
    /// Default `[0; 3]`. Active when `flags & TRANSLUCENT_ABSORB` is set.
    pub absorption: [u8; 3],
}

impl Default for BlockMaterial {
    fn default() -> Self {
        Self {
            flags: 0,
            roughness: 192, // default matte (mid-rough)
            emissive: 0,
            sss_tint: [0, 0, 0, 0],
            wet_tint: [0, 0, 0, 0],
            absorption: [0, 0, 0],
        }
    }
}

/// Bitflag constants for `BlockMaterial::flags`.
pub mod material_flag {
    /// Block participates in leaves subsurface backlight contribution.
    pub const LEAVES_SSS: u8 = 1 << 0;
    /// Block participates in glass-tinted refraction absorption path.
    pub const TRANSLUCENT_ABSORB: u8 = 1 << 1;
    /// Block is a water source; flag set on the `water` block so the chunk
    /// shader's "below water level + sunlit" caustics path is cheap to mask.
    pub const WATER: u8 = 1 << 2;
    /// Opaque block receives the sky-reflection + sun-glint glossy coat in
    /// the chunk shader (slice 3). Gloss amount is driven by `roughness`
    /// (0 = mirror … 255 = matte); keep roughness ≥ ~150 for a subtle sheen.
    pub const REFLECTIVE: u8 = 1 << 3;
}

impl BlockMaterial {
    /// Mark this block as a leaves-type (subsurface backlight on sun-facing
    /// surfaces). `tint` is the RGBA color blended INTO lit leaves when the
    /// sun is behind the leaves; default leaves tint is a yellow-green.
    pub fn leaves(mut self, tint: [u8; 4]) -> Self {
        self.flags |= material_flag::LEAVES_SSS;
        // Fall back to a chlorophyll-yellow if no tint supplied.
        self.sss_tint = if tint == [0, 0, 0, 0] {
            [180, 230, 110, 255]
        } else {
            tint
        };
        self
    }

    /// Mark this block as glass / stained-glass for tinted refraction absorption.
    /// `absorption` is the per-channel Beer's-law coefficient (typical
    /// `(46, 18, 10)` for water-blue glass).
    pub fn glass(mut self, absorption: [u8; 3]) -> Self {
        self.flags |= material_flag::TRANSLUCENT_ABSORB;
        self.absorption = absorption;
        self
    }

    /// Mark this block as water source (used to enable cheap caustics path).
    pub fn water() -> Self {
        Self {
            flags: material_flag::WATER,
            roughness: 255,
            emissive: 0,
            sss_tint: [0, 0, 0, 0],
            wet_tint: [0, 0, 0, 0],
            absorption: [46, 18, 10],
        }
    }

    /// Mark this opaque block as glossy-reflective (sky reflection + sun
    /// glint in the chunk shader). `roughness` controls the coat strength:
    /// 0 would be a perfect mirror, 255 fully matte; values around 170–200
    /// read as a subtle sheen (e.g. snow, ice-like surfaces).
    pub fn reflective(roughness: u8) -> Self {
        Self {
            flags: material_flag::REFLECTIVE,
            roughness,
            ..Self::default()
        }
    }
}

impl BlockDef {
    pub fn is_rendered(&self) -> bool {
        !matches!(self.kind, BlockKind::Air)
    }

    /// Set hardness (seconds to break with bare hand).
    pub fn with_hardness(mut self, hardness: f32) -> Self {
        self.hardness = hardness;
        self
    }

    /// Set required tool type and minimum tier.
    pub fn with_tool(mut self, tool: ToolType, tier: u8) -> Self {
        self.required_tool = tool;
        self.required_tier = tier;
        self
    }

    /// Set drop table.
    pub fn with_drops(mut self, drops: Vec<BlockDrop>) -> Self {
        self.drops = drops;
        self
    }

    /// Set a single simple drop (always drops 1 of the given item).
    pub fn with_simple_drop(mut self, item: BlockId) -> Self {
        self.drops = vec![BlockDrop::simple(item)];
        self
    }

    /// Set blast resistance.
    pub fn with_blast_resistance(mut self, resistance: f32) -> Self {
        self.blast_resistance = resistance;
        self
    }
}

/// Central block-definition table. Built once at startup; read by worldgen,
/// the mesher, and physics. Indexed by `BlockId`.
#[derive(Clone)]
pub struct BlockRegistry {
    defs: Vec<BlockDef>,
    /// Owned string keys so the map works for runtime-loaded names from JSON
    /// (we can't use `&'static str` for those, and we don't want a global arena
    /// to leak `Box<str>` to process lifetime).
    by_name: std::collections::HashMap<String, BlockId>,
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

impl BlockRegistry {
    /// Construct the registry populated with the builtin block set used by
    /// worldgen and the player. Tile indices match the atlas layout produced
    /// by `voxel_render::atlas`.
    pub fn with_builtins() -> Self {
        let mut reg = Self {
            defs: Vec::new(),
            by_name: std::collections::HashMap::new(),
        };
        // id 0 must be air.
        reg.add(BlockDef {
            id: BlockId::AIR,
            name: Arc::from("air"),
            kind: BlockKind::Air,
            solid: false,
            opaque: false,
            breakable: false,
            replaceable: true,
            textures: BlockTextures::same(0),
            emission: 0,
            emission_color: DEFAULT_EMISSION_COLOR,
            light_absorption: 0,
            map_color: [0, 0, 0, 0],
            hardness: 0.0,
            blast_resistance: 0.0,
            required_tool: ToolType::None,
            required_tier: 0,
            drops: Vec::new(),
            animation: None,
            material: BlockMaterial::default(),
        });
        // Atlas tile indices (see renderer atlas for the actual PNGs):
        // 0 air, 1 stone, 2 dirt, 3 grass_top, 4 grass_side, 5 sand,
        // 6 water, 7 wood_side, 8 wood_top, 9 leaves, 10 bedrock, 11 coal_ore,
        // 12 iron_ore, 13 gold_ore, 14 diamond_ore, 15 planks, 16 cobblestone,
        // 17 glass, 18 gravel, 19 snow, 20 white, 21 torch, 22 bucket,
        // 23 water_bucket, 24 tall_grass, 25 poppy, 26 dandelion, 27 cactus,
        // 28 mushroom_red, 29 mushroom_brown, 30 birch_log_side, 31 birch_log_top,
        // 32 birch_leaves, 33 spruce_log_side, 34 spruce_log_top, 35 spruce_leaves,
        // 36 mossy_cobblestone, 37 chest.
        reg.add_named("stone", solid_opaque(1));
        reg.add_named("dirt", solid_opaque(2));
        reg.add_named(
            "grass",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("grass"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::top_bottom_side(3, 2, 4),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 15,
                map_color: [200, 160, 200, 255],
                hardness: 0.6,
                blast_resistance: 0.6,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: vec![BlockDrop::simple(BlockId(2))],
                animation: None, // drops dirt
                material: BlockMaterial::default(),
            },
        );
        reg.add_named("sand", solid_opaque(5));
        reg.add_named(
            "water",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("water"),
                kind: BlockKind::Liquid,
                solid: false,
                opaque: false,
                breakable: false,
                replaceable: true,
                textures: BlockTextures::same(6),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 12,
                map_color: [200, 160, 200, 255],
                hardness: 100.0,
                blast_resistance: 500.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None,
                // Water: enables the cheap caustics path in chunk.frag.
                material: BlockMaterial::water(),
            },
        );
        reg.add_named(
            "lava",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("lava"),
                kind: BlockKind::Liquid,
                solid: false,
                opaque: false,
                breakable: false,
                replaceable: true,
                textures: BlockTextures::same(40),
                emission: 15,
                emission_color: [255, 130, 30],
                light_absorption: 15,
                map_color: [255, 100, 0, 255],
                hardness: 100.0,
                blast_resistance: 500.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "wood",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("wood"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::top_bottom_side(8, 8, 7),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 15,
                map_color: [200, 160, 200, 255],
                hardness: 2.0,
                blast_resistance: 10.0,
                required_tool: ToolType::Axe,
                required_tier: 0,
                drops: vec![BlockDrop::self_drop()],
                animation: None, // drops itself (wood)
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "leaves",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("leaves"),
                kind: BlockKind::Transparent,
                solid: true,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(9),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 14,
                map_color: [200, 160, 200, 255],
                hardness: 0.2,
                blast_resistance: 0.2,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None, // drops nothing normally
                // Cherry-yellow chlorophyll backlight on sun-facing surfaces.
                material: BlockMaterial::default().leaves([220, 240, 140, 255]),
            },
        );
        reg.add_named(
            "bedrock",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("bedrock"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: false,
                replaceable: false,
                textures: BlockTextures::same(10),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 15,
                map_color: [200, 160, 200, 255],
                hardness: -1.0, // unbreakable
                blast_resistance: 18000000.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named("coal_ore", solid_opaque(11));
        reg.add_named("iron_ore", solid_opaque(12));
        reg.add_named("gold_ore", solid_opaque(13));
        reg.add_named("diamond_ore", solid_opaque(14));
        reg.add_named("planks", solid_opaque(15));
        reg.add_named("cobblestone", solid_opaque(16));
        reg.add_named(
            "glass",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("glass"),
                kind: BlockKind::Transparent,
                solid: true,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(17),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 14,
                map_color: [200, 160, 200, 255],
                hardness: 0.3,
                blast_resistance: 0.3,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None, // drops nothing without silk touch
                material: BlockMaterial::default().glass([46, 18, 10]),
            },
        );
        reg.add_named("gravel", solid_opaque(18));
        reg.add_named(
            "snow",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("snow"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(19),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 15,
                map_color: [200, 160, 200, 255],
                hardness: 0.2,
                blast_resistance: 0.2,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None,
                // Snow gets a subtle glossy coat (sky reflection + sun
                // glint) — the "other reflecting surfaces" demonstrator for
                // the slice-3 opaque REFLECTIVE path. Roughness 185 keeps it
                // a sheen, not a mirror.
                material: BlockMaterial::reflective(185),
            },
        );
        reg.add_named(
            "torch",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("torch"),
                kind: BlockKind::Transparent,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(21),
                emission: 14,
                emission_color: [255, 200, 100],
                light_absorption: 0,
                map_color: [200, 160, 200, 255],
                hardness: 0.0,
                blast_resistance: 0.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: vec![BlockDrop::self_drop()],
                animation: None, // drops itself
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "tall_grass",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("tall_grass"),
                kind: BlockKind::Foliage,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: true,
                textures: BlockTextures::same(24),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 0,
                map_color: [200, 160, 200, 255],
                hardness: 0.0,
                blast_resistance: 0.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "poppy",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("poppy"),
                kind: BlockKind::Foliage,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: true,
                textures: BlockTextures::same(25),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 0,
                map_color: [200, 160, 200, 255],
                hardness: 0.0,
                blast_resistance: 0.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: vec![BlockDrop::self_drop()],
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "dandelion",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("dandelion"),
                kind: BlockKind::Foliage,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: true,
                textures: BlockTextures::same(26),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 0,
                map_color: [200, 160, 200, 255],
                hardness: 0.0,
                blast_resistance: 0.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: vec![BlockDrop::self_drop()],
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "cactus",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("cactus"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::top_bottom_side(38, 39, 27),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 15,
                map_color: [200, 160, 200, 255],
                hardness: 0.4,
                blast_resistance: 0.4,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: vec![BlockDrop::self_drop()],
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "mushroom_red",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("mushroom_red"),
                kind: BlockKind::Foliage,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: true,
                textures: BlockTextures::same(28),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 0,
                map_color: [200, 160, 200, 255],
                hardness: 0.0,
                blast_resistance: 0.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: vec![BlockDrop::self_drop()],
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "mushroom_brown",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("mushroom_brown"),
                kind: BlockKind::Foliage,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: true,
                textures: BlockTextures::same(29),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 0,
                map_color: [200, 160, 200, 255],
                hardness: 0.0,
                blast_resistance: 0.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: vec![BlockDrop::self_drop()],
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "birch_log",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("birch_log"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::top_bottom_side(31, 31, 30),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 15,
                map_color: [200, 160, 200, 255],
                hardness: 2.0,
                blast_resistance: 10.0,
                required_tool: ToolType::Axe,
                required_tier: 0,
                drops: vec![BlockDrop::self_drop()],
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "birch_leaves",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("birch_leaves"),
                kind: BlockKind::Transparent,
                solid: true,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(32),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 14,
                map_color: [200, 160, 200, 255],
                hardness: 0.2,
                blast_resistance: 0.2,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "spruce_log",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("spruce_log"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::top_bottom_side(34, 34, 33),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 15,
                map_color: [200, 160, 200, 255],
                hardness: 2.0,
                blast_resistance: 10.0,
                required_tool: ToolType::Axe,
                required_tier: 0,
                drops: vec![BlockDrop::self_drop()],
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "spruce_leaves",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("spruce_leaves"),
                kind: BlockKind::Transparent,
                solid: true,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(35),
                emission: 0,
                emission_color: DEFAULT_EMISSION_COLOR,
                light_absorption: 14,
                map_color: [200, 160, 200, 255],
                hardness: 0.2,
                blast_resistance: 0.2,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named("mossy_cobblestone", solid_opaque(36));
        reg.add_named("chest", solid_opaque(37));

        // Colored light blocks for testing colored lighting.
        // Use the white tile (20) as base; emission color tints them visually.
        reg.add_named(
            "light_blue",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("light_blue"),
                kind: BlockKind::Transparent,
                solid: true,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(20),
                emission: 15,
                emission_color: [60, 120, 255],
                light_absorption: 0,
                map_color: [60, 120, 255, 255],
                hardness: 0.0,
                blast_resistance: 0.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "light_red",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("light_red"),
                kind: BlockKind::Transparent,
                solid: true,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(20),
                emission: 15,
                emission_color: [255, 40, 40],
                light_absorption: 0,
                map_color: [255, 40, 40, 255],
                hardness: 0.0,
                blast_resistance: 0.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "light_green",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("light_green"),
                kind: BlockKind::Transparent,
                solid: true,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(20),
                emission: 15,
                emission_color: [40, 255, 80],
                light_absorption: 0,
                map_color: [40, 255, 80, 255],
                hardness: 0.0,
                blast_resistance: 0.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None,
                material: BlockMaterial::default(),
            },
        );
        reg.add_named(
            "light_pink",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("light_pink"),
                kind: BlockKind::Transparent,
                solid: true,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(20),
                emission: 15,
                emission_color: [255, 100, 200],
                light_absorption: 0,
                map_color: [255, 100, 200, 255],
                hardness: 0.0,
                blast_resistance: 0.0,
                required_tool: ToolType::None,
                required_tier: 0,
                drops: Vec::new(),
                animation: None,
                material: BlockMaterial::default(),
            },
        );

        reg
    }

    /// Build a registry from asset-loaded block definitions.
    /// Air (id=0) is always prepended as the first block.
    pub fn from_assets(blocks: &[voxel_assets::BlockData]) -> Self {
        let mut reg = Self {
            defs: Vec::new(),
            by_name: std::collections::HashMap::new(),
        };
        // id 0 is always air (not in by_name).
        reg.add(BlockDef {
            id: BlockId::AIR,
            name: Arc::from("air"),
            kind: BlockKind::Air,
            solid: false,
            opaque: false,
            breakable: false,
            replaceable: true,
            textures: BlockTextures::same(0),
            emission: 0,
            emission_color: DEFAULT_EMISSION_COLOR,
            light_absorption: 0,
            map_color: [0, 0, 0, 0],
            hardness: 0.0,
            blast_resistance: 0.0,
            required_tool: ToolType::None,
            required_tier: 0,
            drops: Vec::new(),
            animation: None,
            material: BlockMaterial::default(),
        });
        for bd in blocks {
            let kind = match bd.kind.as_str() {
                "air" => BlockKind::Air,
                "solid" => BlockKind::Solid,
                "liquid" => BlockKind::Liquid,
                "foliage" => BlockKind::Foliage,
                "transparent" => BlockKind::Transparent,
                _ => BlockKind::Solid,
            };
            let textures = match &bd.textures {
                voxel_assets::BlockTexturesData::Same { same } => BlockTextures::same(*same),
                voxel_assets::BlockTexturesData::PerFace {
                    top,
                    bottom,
                    side,
                    neg_x,
                    pos_x,
                    neg_y,
                    pos_y,
                    neg_z,
                    pos_z,
                } => {
                    let t = top.unwrap_or(0);
                    let b = bottom.unwrap_or(0);
                    let s = side.unwrap_or(0);
                    // If specific faces are given, use them; otherwise fall back to top/bottom/side.
                    if neg_x.is_some()
                        || pos_x.is_some()
                        || neg_y.is_some()
                        || pos_y.is_some()
                        || neg_z.is_some()
                        || pos_z.is_some()
                    {
                        BlockTextures {
                            tiles: [
                                neg_x.unwrap_or(s),
                                pos_x.unwrap_or(s),
                                neg_y.unwrap_or(b),
                                pos_y.unwrap_or(t),
                                neg_z.unwrap_or(s),
                                pos_z.unwrap_or(s),
                            ],
                        }
                    } else {
                        BlockTextures::top_bottom_side(t, b, s)
                    }
                }
            };
            // Arc<str> owns the name; the previous global `NAME_ARENA: Mutex`
            // + `unsafe { &*ptr }` pattern was process-global mutable state
            // and broken for parallel registry construction. Owned
            // `Arc<str>` is safe, cheap to clone, and dropped with the
            // registry.
            let name: Arc<str> = Arc::from(bd.name.as_str());
            let map_color = bd.map_color.unwrap_or_else(|| default_map_color(&bd.name));
            let emission_color = bd
                .emission_color
                .map(|c| [c[0], c[1], c[2]])
                .unwrap_or(DEFAULT_EMISSION_COLOR);
            reg.add_named_owned(
                Arc::clone(&name),
                BlockDef {
                    id: BlockId(0),
                    name,
                    kind,
                    solid: bd.solid,
                    opaque: bd.opaque,
                    breakable: bd.breakable,
                    replaceable: bd.replaceable,
                    textures,
                    emission: bd.emission.min(15),
                    emission_color,
                    light_absorption: bd.light_absorption.min(15),
                    map_color,
                    hardness: bd.hardness.unwrap_or(1.5),
                    blast_resistance: bd.blast_resistance.unwrap_or(6.0),
                    required_tool: bd
                        .required_tool
                        .as_deref()
                        .map(parse_tool_type)
                        .unwrap_or(ToolType::None),
                    required_tier: bd.required_tier.unwrap_or(0),
                    drops: Vec::new(),
                    animation: None,
                    material: BlockMaterial::default(),
                },
            );
        }
        log::info!("built registry with {} blocks from assets", reg.defs.len());
        reg
    }

    fn add(&mut self, def: BlockDef) {
        let mut def = def;
        def.id = BlockId(self.defs.len() as u16);
        self.defs.push(def);
    }

    /// Insert a builtin block whose name is a string literal. The literal is
    /// interned via `Arc::from` so the resulting `BlockDef.name` shares its
    /// storage cheaply. If `map_color` is the fallback pink, it is
    /// auto-filled from `default_map_color(name)`.
    fn add_named(&mut self, name: &str, mut def: BlockDef) {
        def.id = BlockId(self.defs.len() as u16);
        let arc = Arc::from(name);
        def.name = Arc::clone(&arc);
        // Auto-fill map_color from name heuristic if still at fallback pink.
        if def.map_color == [200, 160, 200, 255] {
            def.map_color = default_map_color(name);
        }
        Self::resolve_self_drops(&mut def);
        self.by_name.insert(arc.to_string(), def.id);
        self.defs.push(def);
    }

    /// Rewrite explicit `SelfDrop` entries to the definition's assigned ID.
    /// Builtin self-drops are authored before the final registry ID exists;
    /// resolution happens only after the ID is known. An actually empty drop
    /// table remains an explicit no-drop policy.
    fn resolve_self_drops(def: &mut BlockDef) {
        for drop in &mut def.drops {
            if matches!(drop.condition, DropCondition::SelfDrop) {
                drop.item = def.id;
                drop.condition = DropCondition::Always;
            }
        }
    }

    /// Insert a block whose name string is already owned (used by
    /// `from_assets`). Auto-fills `map_color` from the name heuristic
    /// if it's still the fallback pink.
    fn add_named_owned(&mut self, name: Arc<str>, mut def: BlockDef) {
        def.id = BlockId(self.defs.len() as u16);
        def.name = Arc::clone(&name);
        if def.map_color == [200, 160, 200, 255] {
            def.map_color = default_map_color(&name);
        }
        Self::resolve_self_drops(&mut def);
        self.by_name.insert(name.to_string(), def.id);
        self.defs.push(def);
    }

    pub fn get(&self, id: BlockId) -> &BlockDef {
        let idx = id.0 as usize;
        if idx < self.defs.len() {
            &self.defs[idx]
        } else {
            // Return the first block (air) as fallback for invalid IDs.
            &self.defs[0]
        }
    }

    pub fn id_of(&self, name: &str) -> Option<BlockId> {
        self.by_name.get(name).copied()
    }

    pub fn count(&self) -> usize {
        self.defs.len()
    }

    /// True if the block should be considered for collision.
    pub fn is_solid(&self, id: BlockId) -> bool {
        self.get(id).solid
    }

    /// True if the block fully hides the face of an adjacent solid block.
    pub fn is_opaque(&self, id: BlockId) -> bool {
        self.get(id).opaque
    }

    /// True if the block is a liquid (water, lava, etc.).
    pub fn is_liquid(&self, id: BlockId) -> bool {
        self.get(id).kind == BlockKind::Liquid
    }

    /// Light emission (0–15) for the given block.
    pub fn emission(&self, id: BlockId) -> u8 {
        self.get(id).emission
    }

    /// RGB color of emitted light for the given block.
    pub fn emission_color(&self, id: BlockId) -> [u8; 3] {
        self.get(id).emission_color
    }

    /// Light absorption (0–15) for the given block.
    pub fn light_absorption(&self, id: BlockId) -> u8 {
        self.get(id).light_absorption
    }
}

fn solid_opaque(tile: u16) -> BlockDef {
    BlockDef {
        id: BlockId(0),
        name: Arc::from(""),
        kind: BlockKind::Solid,
        solid: true,
        opaque: true,
        breakable: true,
        replaceable: false,
        textures: BlockTextures::same(tile),
        emission: 0,
        emission_color: DEFAULT_EMISSION_COLOR,
        light_absorption: 15,
        map_color: [200, 160, 200, 255], // fallback pink; overwritten by add_named
        hardness: 1.5,
        blast_resistance: 6.0,
        required_tool: ToolType::None,
        required_tier: 0,
        // Resolve the explicit self-drop after `add_named` assigns the
        // definition's final registry ID.
        drops: vec![BlockDrop::self_drop()],
        animation: None,
        material: BlockMaterial::default(),
    }
}

/// Parse a tool type string from JSON into the ToolType enum.
fn parse_tool_type(s: &str) -> ToolType {
    match s.to_lowercase().as_str() {
        "pickaxe" => ToolType::Pickaxe,
        "axe" => ToolType::Axe,
        "shovel" => ToolType::Shovel,
        "sword" => ToolType::Sword,
        "hoe" => ToolType::Hoe,
        _ => ToolType::None,
    }
}

/// Heuristic map color from block name. Used when no explicit `map_color`
/// is provided in the block data JSON.
pub fn default_map_color(name: &str) -> [u8; 4] {
    let n = name.to_lowercase();
    if n.contains("grass") {
        return [120, 180, 80, 255];
    }
    if n.contains("dirt") {
        return [140, 110, 70, 255];
    }
    if n.contains("stone") || n.contains("cobble") || n.contains("gravel") {
        return [130, 130, 130, 255];
    }
    if n.contains("sand") {
        return [220, 210, 160, 255];
    }
    if n.contains("water") {
        return [60, 100, 220, 180];
    }
    if n.contains("log") || n.contains("wood") || n.contains("plank") {
        return [100, 70, 40, 255];
    }
    if n.contains("leaves") {
        return [60, 140, 60, 255];
    }
    if n.contains("glass") {
        return [180, 220, 255, 120];
    }
    if n.contains("snow") {
        return [240, 245, 250, 255];
    }
    if n.contains("bedrock") {
        return [60, 60, 60, 255];
    }
    if n.contains("ore") {
        return [140, 110, 80, 255];
    }
    if n.contains("mushroom") {
        return [180, 120, 80, 255];
    }
    if n.contains("torch") || n.contains("flower") || n.contains("poppy") || n.contains("dandelion")
    {
        return [200, 180, 60, 255];
    }
    if n.contains("cactus") {
        return [80, 160, 60, 255];
    }
    if n.contains("tall_grass") {
        return [90, 160, 60, 255];
    }
    [200, 160, 200, 255] // fallback pink
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_count() {
        let reg = BlockRegistry::with_builtins();
        assert!(
            reg.count() >= 10,
            "expected at least 10 builtins, got {}",
            reg.count()
        );
    }

    #[test]
    fn air_is_air() {
        let _reg = BlockRegistry::with_builtins();
        assert!(BlockId::AIR.is_air());
    }

    #[test]
    fn air_not_solid() {
        let reg = BlockRegistry::with_builtins();
        assert!(!reg.is_solid(BlockId::AIR));
    }

    #[test]
    fn air_not_opaque() {
        let reg = BlockRegistry::with_builtins();
        assert!(!reg.is_opaque(BlockId::AIR));
    }

    #[test]
    fn air_zero_absorption() {
        let reg = BlockRegistry::with_builtins();
        assert_eq!(reg.light_absorption(BlockId::AIR), 0);
    }

    #[test]
    fn stone_is_solid_opaque() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        assert!(!stone.is_air());
        assert!(reg.is_solid(stone));
        assert!(reg.is_opaque(stone));
        assert_eq!(reg.light_absorption(stone), 15);
    }

    #[test]
    fn builtin_self_drops_resolve_after_id_assignment() {
        let reg = BlockRegistry::with_builtins();
        for name in [
            "wood",
            "torch",
            "poppy",
            "dandelion",
            "cactus",
            "birch_log",
            "spruce_log",
        ] {
            let id = reg.id_of(name).unwrap();
            let def = reg.get(id);
            assert!(!def.drops.is_empty(), "{name} should have an explicit drop");
            assert!(def.drops.iter().all(|drop| !drop.item.is_air()));
            assert_eq!(def.drops[0].item, id, "{name} should drop itself");
        }
    }

    #[test]
    fn builtin_empty_drop_tables_stay_empty() {
        let reg = BlockRegistry::with_builtins();
        for name in [
            "leaves",
            "birch_leaves",
            "spruce_leaves",
            "glass",
            "tall_grass",
        ] {
            let id = reg.id_of(name).unwrap();
            assert!(reg.get(id).drops.is_empty(), "{name} should have no drops");
        }
    }

    #[test]
    fn torch_emits_light() {
        let reg = BlockRegistry::with_builtins();
        let torch = reg.id_of("torch").unwrap();
        assert!(reg.emission(torch) > 0);
    }

    #[test]
    fn torch_zero_absorption() {
        let reg = BlockRegistry::with_builtins();
        let torch = reg.id_of("torch").unwrap();
        assert_eq!(reg.light_absorption(torch), 0);
    }

    #[test]
    fn grass_solid_opaque() {
        let reg = BlockRegistry::with_builtins();
        let grass = reg.id_of("grass").unwrap();
        assert!(reg.is_solid(grass));
        assert!(reg.is_opaque(grass));
    }

    #[test]
    fn water_not_solid() {
        let reg = BlockRegistry::with_builtins();
        let water = reg.id_of("water").unwrap();
        assert!(!reg.is_solid(water));
    }

    #[test]
    fn id_of_unknown_returns_none() {
        let reg = BlockRegistry::with_builtins();
        assert!(reg.id_of("nonexistent_block_xyz").is_none());
    }

    #[test]
    fn block_def_texts() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        let def = reg.get(stone);
        assert_eq!(def.textures.tile(Face::PosY), def.textures.tile(Face::NegY));
    }

    #[test]
    fn is_rendered_air() {
        let reg = BlockRegistry::with_builtins();
        assert!(!reg.get(BlockId::AIR).is_rendered());
    }

    #[test]
    fn is_rendered_stone() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        assert!(reg.get(stone).is_rendered());
    }

    #[test]
    fn face_normal_directions() {
        assert_eq!(Face::PosX.normal(), IVec3::X);
        assert_eq!(Face::NegX.normal(), IVec3::NEG_X);
        assert_eq!(Face::PosY.normal(), IVec3::Y);
        assert_eq!(Face::NegY.normal(), IVec3::NEG_Y);
        assert_eq!(Face::PosZ.normal(), IVec3::Z);
        assert_eq!(Face::NegZ.normal(), IVec3::NEG_Z);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn id_of_get_roundtrip(id in 1u16..21) {
            let reg = BlockRegistry::with_builtins();
            let block_id = BlockId(id);
            let def = reg.get(block_id);
            let looked_up = reg.id_of(def.name.as_ref());
            prop_assert_eq!(looked_up, Some(block_id));
        }

        #[test]
        fn all_ids_in_range(id in 0u16..21) {
            let reg = BlockRegistry::with_builtins();
            let def = reg.get(BlockId(id));
            // Every block should have a non-empty name
            prop_assert!(!def.name.is_empty(), "block {id} has empty name");
        }

        #[test]
        fn emission_in_range(id in 0u16..21) {
            let reg = BlockRegistry::with_builtins();
            let e = reg.emission(BlockId(id));
            prop_assert!(e <= 15, "emission {e} out of range for block {id}");
        }

        #[test]
        fn absorption_in_range(id in 0u16..21) {
            let reg = BlockRegistry::with_builtins();
            let a = reg.light_absorption(BlockId(id));
            prop_assert!(a <= 15, "absorption {a} out of range for block {id}");
        }

        #[test]
        fn id_of_unknown_returns_none(name in "[a-z]{20,30}") {
            let reg = BlockRegistry::with_builtins();
            prop_assert_eq!(reg.id_of(&name), None);
        }
    }
}
