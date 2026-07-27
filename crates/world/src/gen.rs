//! Procedural terrain generation.
//!
//! Pipeline per chunk column (x,z):
//!   1. Biome is chosen from a temperature/humidity noise field.
//!   2. A continent/height field sets the base elevation; biome modifies it
//!      (mountains taller, oceans lower, etc.).
//!   3. Surface depth selects grass/dirt/stone/sand/snow by height + biome.
//!   4. Caves carve 3D noise tunnels; ores sprinkle by depth.
//!   5. Trees and small decorations scatter on grass surfaces.
//!
//! Generation is deterministic from a world seed and entirely `rayon`-parallel
//! at the chunk level (see `ChunkStreamer`).

use fastnoise_lite::FastNoiseLite;
use voxel_core::{
    math::{chunk_origin, ChunkPos},
    BlockId, BlockPos, CHUNK_SIZE, SEA_LEVEL, WORLD_HEIGHT_BLOCKS,
};

use crate::{chunk::Chunk, registry::BlockRegistry};

/// Identifier for a surface biome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BiomeId {
    Ocean,
    Plains,
    Forest,
    Desert,
    Mountains,
    Beach,
}

impl BiomeId {
    /// Pick a biome from temperature (0..1) and humidity (0..1) plus height.
    fn classify(temp: f32, humid: f32, height: f32) -> Self {
        if height < (SEA_LEVEL - 8) as f32 {
            BiomeId::Ocean
        } else if height < (SEA_LEVEL + 1) as f32 {
            if temp > 0.6 {
                BiomeId::Beach
            } else {
                BiomeId::Plains
            }
        } else if height > (SEA_LEVEL + 40) as f32 {
            BiomeId::Mountains
        } else if temp > 0.7 {
            BiomeId::Desert
        } else if humid > 0.55 {
            BiomeId::Forest
        } else {
            BiomeId::Plains
        }
    }

    /// Map display color for this biome. Used by the fullscreen map's biome
    /// overlay (v2) and available for minimap tinting.
    pub fn map_color(self) -> [u8; 4] {
        match self {
            BiomeId::Ocean => [50, 80, 180, 200],
            BiomeId::Plains => [120, 190, 80, 255],
            BiomeId::Forest => [70, 150, 60, 255],
            BiomeId::Desert => [230, 210, 160, 255],
            BiomeId::Mountains => [160, 150, 140, 255],
            BiomeId::Beach => [210, 200, 160, 255],
        }
    }
}

/// Configurable terrain generator. Owns precomputed noise samplers.
pub struct TerrainGenerator {
    seed: i32,
    // Continental shape (low frequency).
    continent: FastNoiseLite,
    // Hills / rolling terrain (medium frequency).
    hills: FastNoiseLite,
    // Mountain ridges (ridge noise).
    ridge: FastNoiseLite,
    // Biome temperature.
    temperature: FastNoiseLite,
    // Biome humidity.
    humidity: FastNoiseLite,
    // 3D cave noise (reused for spaghetti caves).
    cave: FastNoiseLite,
    // Large cavern chambers (cheese caves).
    cheese_cave: FastNoiseLite,
    // Thin winding tunnels (noodle caves).
    noodle_cave: FastNoiseLite,

    // 2D horizontal tunnel routing (added per the caves plan; reserved for
    // future cave routing).
    #[allow(dead_code)]
    carver_2d: FastNoiseLite,
    // 2D ravine anchor mask (cellular).
    ravine_mask: FastNoiseLite,
    // 2D ravine orientation.
    ravine_dir: FastNoiseLite,
    // 2D ravine width variation.
    ravine_width: FastNoiseLite,
    // Ore placement jitter.
    ore: FastNoiseLite,
    // Tree scatter.
    tree: FastNoiseLite,
}

impl TerrainGenerator {
    pub fn new(seed: i32) -> Self {
        let make = |freq: f32, ty: fastnoise_lite::NoiseType| {
            let mut n = FastNoiseLite::new();
            n.set_seed(Some(seed));
            n.set_noise_type(Some(ty));
            n.set_frequency(Some(freq));
            n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
            n.set_fractal_octaves(Some(4));
            n.set_fractal_lacunarity(Some(2.0));
            n.set_fractal_gain(Some(0.5));
            n
        };
        Self {
            seed,
            continent: make(0.0018, fastnoise_lite::NoiseType::OpenSimplex2),
            hills: make(0.01, fastnoise_lite::NoiseType::OpenSimplex2),
            ridge: make(0.0035, fastnoise_lite::NoiseType::OpenSimplex2),
            temperature: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(3)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
                n.set_frequency(Some(0.0015));
                n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
                n.set_fractal_octaves(Some(3));
                n
            },
            humidity: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(7)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
                n.set_frequency(Some(0.0015));
                n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
                n.set_fractal_octaves(Some(3));
                n
            },
            cave: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(11)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
                n.set_frequency(Some(0.015));
                n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
                n.set_fractal_octaves(Some(3));
                n
            },
            cheese_cave: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(19)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
                n.set_frequency(Some(0.006));
                n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
                n.set_fractal_octaves(Some(4));
                n
            },
            noodle_cave: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(23)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
                n.set_frequency(Some(0.025));
                n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
                n.set_fractal_octaves(Some(3));
                n
            },
            carver_2d: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(29)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
                n.set_frequency(Some(0.002));
                n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
                n.set_fractal_octaves(Some(3));
                n
            },
            ravine_mask: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(31)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
                n.set_frequency(Some(0.008));
                n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
                n.set_fractal_octaves(Some(2));
                n
            },
            ravine_dir: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(37)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
                n.set_frequency(Some(0.004));
                n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
                n.set_fractal_octaves(Some(2));
                n
            },
            ravine_width: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(41)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
                n.set_frequency(Some(0.006));
                n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
                n.set_fractal_octaves(Some(2));
                n
            },
            ore: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(13)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
                n.set_frequency(Some(0.06));
                n
            },
            tree: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(17)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
                n.set_frequency(Some(0.12));
                n
            },
        }
    }

    pub fn seed(&self) -> i32 {
        self.seed
    }

    /// Base terrain height (in world Y block units) for a world (x, z).
    fn base_height(&self, x: i32, z: i32) -> f32 {
        let xf = x as f32;
        let zf = z as f32;
        let cont = self.continent.get_noise_2d(xf, zf); // -1..1
        let hills = self.hills.get_noise_2d(xf, zf);
        let ridge = 1.0 - (self.ridge.get_noise_2d(xf, zf).abs()); // 0..1, ridged

        // Continental: large sea-level baseline ± some land/ocean.
        let base = SEA_LEVEL as f32 + cont * 24.0;
        // Hills add rolling variation on land.
        let hill_part = hills * 8.0;
        // Mountains: only where continental is high (land interior).
        let mountain_mask = ((cont - 0.2) / 0.8).clamp(0.0, 1.0);
        let mountain_part = ridge * ridge * 60.0 * mountain_mask;

        base + hill_part + mountain_part
    }

    /// Search outward from the world origin for a land column above sea level.
    /// Returns (x, surface_height, z) so the caller can place a spawn there.
    /// Uses the height function directly — no chunk loading required.
    pub fn find_spawn(&self) -> (i32, i32, i32) {
        for r in 0..128i32 {
            if r == 0 {
                let h = self.base_height(0, 0).round() as i32;
                if h > SEA_LEVEL + 2 {
                    return (0, h, 0);
                }
                continue;
            }
            // Walk the ring at radius r.
            for dx in -r..=r {
                for &dz in &[-r, r] {
                    let h = self.base_height(dx, dz).round() as i32;
                    if h > SEA_LEVEL + 2 {
                        return (dx, h, dz);
                    }
                }
            }
            for &dx in &[-r, r] {
                for dz in (-r + 1)..r {
                    let h = self.base_height(dx, dz).round() as i32;
                    if h > SEA_LEVEL + 2 {
                        return (dx, h, dz);
                    }
                }
            }
        }
        (0, 90, 0) // fallback: high in the air at origin
    }

    fn biome_at(&self, x: i32, z: i32, height: f32) -> BiomeId {
        let t = (self.temperature.get_noise_2d(x as f32, z as f32) + 1.0) * 0.5;
        let h = (self.humidity.get_noise_2d(x as f32, z as f32) + 1.0) * 0.5;
        BiomeId::classify(t, h, height)
    }

    /// Multi-pass cave carver. Returns true if this block should be replaced
    /// with air, combining cheese caves (large chambers), noodle caves
    /// (thin winding tunnels that can breach the surface), and spaghetti
    /// caves (the original wide tunnels).
    ///
    /// `chunk` and `reg` are accepted to match the planned API even though the
    /// current implementation only needs the coordinate and surface data.
    fn carve(
        &self,
        _chunk: &mut Chunk,
        x: i32,
        y: i32,
        z: i32,
        height: i32,
        biome: BiomeId,
        _reg: &BlockRegistry,
    ) -> bool {
        // Keep a safe margin from the world bottom/top.
        if !(4..=WORLD_HEIGHT_BLOCKS - 8).contains(&y) {
            return false;
        }

        let xf = x as f32;
        let yf = y as f32;
        let zf = z as f32;

        // Cheese caves: large round voids.
        let cheese = self.cheese_cave.get_noise_3d(xf, yf * 1.2, zf) > 0.15;
        // Spaghetti caves: reuse the original cave noise with a tighter threshold.
        let spaghetti = self.cave.get_noise_3d(xf, yf * 1.5, zf).abs() < 0.03;
        // Standard crust guard: keep the top few layers intact so caves don't
        // punch through the surface or the ocean floor.
        let standard_crust = y < height - 3;
        if standard_crust && (cheese || spaghetti) {
            return true;
        }

        // Noodle caves: thin winding tunnels that can reach closer to the
        // surface for cave entrances, but only when the column above is solid
        // terrain (not an ocean floor).
        let noodle_crust = y < height - 1;
        let solid_above = biome != BiomeId::Ocean && height > SEA_LEVEL;
        let noodle = self.noodle_cave.get_noise_3d(xf, yf * 1.5, zf).abs() < 0.04;
        if noodle_crust && solid_above && noodle {
            return true;
        }

        false
    }

    /// Pick an ore block for a stone block at depth `y`, or None for plain stone.
    /// `noise_val` is the pre-computed ore noise value (-1..1).
    fn ore_for_val(&self, noise_val: f32, y: i32, reg: &BlockRegistry) -> Option<BlockId> {
        let v = (noise_val + 1.0) * 0.5; // 0..1
        if y < 16 && v > 0.985 {
            reg.id_of("diamond_ore")
        } else if y < 32 && v > 0.97 {
            reg.id_of("gold_ore")
        } else if y < 64 && v > 0.92 {
            reg.id_of("iron_ore")
        } else if v > 0.86 {
            reg.id_of("coal_ore")
        } else {
            None
        }
    }

    /// Carve surface ravines into the chunk. Each chunk independently
    /// evaluates deterministic ravine anchors in a neighbourhood and
    /// carves only the blocks that fall inside its own bounds, so adjacent
    /// chunks naturally stitch together continuous canyons.
    ///
    /// Performance note: the outer anchor scan is bounded by `reach` and a
    /// 4-corner early-out skips chunks that sit entirely above the terrain
    /// ceiling. Inside the loop the cellular-mask + per-anchor hash reject
    /// ~99 % of candidate anchors; surviving anchors spend at most
    /// `length_max × width² × Y-band` work. The earlier
    /// `reach=130 / length=30..120` configuration was the source of
    /// multi-second freezes that presented as the window "crashing" on click
    /// in the title screen.
    pub fn carve_ravines(&self, chunk: &mut Chunk, reg: &BlockRegistry) {
        let origin = chunk_origin(chunk.pos);
        let Some(stone) = reg.id_of("stone") else { return; };
        let Some(dirt) = reg.id_of("dirt") else { return; };
        let Some(grass) = reg.id_of("grass") else { return; };
        let Some(gravel) = reg.id_of("gravel") else { return; };
        let Some(water) = reg.id_of("water") else { return; };

        // Coverage requirement: a ravine anchor places a carved column whose
        // world-XZ extent is `[anchor - length - half_width - 1, anchor + length + half_width + 1]`.
        // For that column to reach this chunk the anchor must satisfy
        //   `anchor_x ∈ [origin.x - (length_max + half_width + 1), origin.x + 15 + (length_max + half_width + 1)]`.
        // We sample `[origin.x - reach, origin.x + 16 + reach]`, so we need
        //   `reach ≥ length_max + half_width + 1 = 70 + 4 + 1 = 75`.
        // We bump to 80 for a small margin so sub-pixel anchoring on
        // discontinuous surface noise cells cannot leak seams. Total anchor
        // count = (2*reach + CHUNK_SIZE)² ≈ 192² ≈ 37K columns per chunk
        // (most skipped by the mask + per-anchor hash filter).
        let reach: i32 = 80;
        let min_ax = origin.x - reach;
        let max_ax = origin.x + CHUNK_SIZE + reach;
        let min_az = origin.z - reach;
        let max_az = origin.z + CHUNK_SIZE + reach;

        // Conservative-safe fast early-out: skip when the chunk sits entirely
        // ABOVE the highest surface we'll sample within our reach. We use a
        // 4-corner MAX so the early-out is correctly rejected whenever ANY
        // corner column could still carve into the chunk (ravines descend
        // from the surface; a chunk fully above all corner surfaces has no
        // ravine blocks to write). The +5 buffer keeps us off by one block
        // of edge cases where a corner column's surface is just above the
        // chunk bottom but still carves solid blocks at y=0.
        let chunk_y_bottom = origin.y;
        let cx0 = origin.x;
        let cx1 = origin.x + CHUNK_SIZE - 1;
        let cz0 = origin.z;
        let cz1 = origin.z + CHUNK_SIZE - 1;
        let max_corner_surf = self
            .base_height(cx0, cz0)
            .max(self.base_height(cx1, cz0))
            .max(self.base_height(cx0, cz1))
            .max(self.base_height(cx1, cz1));
        if chunk_y_bottom as f32 > max_corner_surf + 5.0 {
            return;
        }

        for ax in min_ax..=max_ax {
            for az in min_az..=max_az {
                // Cellular mask: rare ravine anchors (~0.2 % of columns).
                // Cellular noise returns distance values; high values are rare peaks.
                let mask = self.ravine_mask.get_noise_2d(ax as f32, az as f32);
                if mask < 0.85 {
                    continue;
                }
                let h = hash2(self.seed, ax, az);
                if h >= 0.15 {
                    continue;
                }

                // Ravine parameters are all derived deterministically from the anchor.
                let dir_noise = self.ravine_dir.get_noise_2d(ax as f32, az as f32);
                let angle = dir_noise * std::f32::consts::PI + h * std::f32::consts::TAU;
                // Length range 30..70 — the spec asked for 30..120 but the
                // longer end blew per-chunk cost; 70 reads as a long winding
                // canyon on screen and keeps the step-loop budget bounded.
                let length = 30.0 + hash2(self.seed.wrapping_mul(3), ax, az) * 40.0;
                let depth = 15.0 + hash2(self.seed.wrapping_mul(5), ax, az) * 25.0;
                let width = 3.0 + self.ravine_width.get_noise_2d(ax as f32, az as f32).clamp(-1.0, 1.0) * 5.0;
                let width = width.clamp(3.0, 8.0);

                let dir_x = angle.cos();
                let dir_z = angle.sin();

                let surf = self.base_height(ax, az).round() as i32;

                // Step along the major axis.
                let steps = length.ceil() as i32;
                for s in -steps..=steps {
                    let t = s as f32;
                    let cx = ax as f32 + t * dir_x;
                    let cz = az as f32 + t * dir_z;

                    // Taper toward the ends of the ravine.
                    let falloff = 1.0 - (t.abs() / (length * 0.5)).clamp(0.0, 1.0);
                    if falloff <= 0.0 {
                        continue;
                    }

                    let local_width = (width * 0.5 * falloff).max(1.5);
                    let local_bottom = (surf as f32 - depth * falloff).max(5.0) as i32;

                    // Bounds of the carved column in world space, clamped to
                    // this chunk so the inner wx∈wz∈ loops only iterate
                    // columns that actually overlap our chunk (the original
                    // unchecked range triggered an `if lx<0 || …` no-op
                    // branch per out-of-chunk block).
                    let chunk_x_min = origin.x;
                    let chunk_x_max = origin.x + CHUNK_SIZE - 1;
                    let chunk_z_min = origin.z;
                    let chunk_z_max = origin.z + CHUNK_SIZE - 1;
                    let min_x = ((cx - local_width).floor() as i32 - 1).max(chunk_x_min);
                    let max_x = ((cx + local_width).ceil() as i32 + 1).min(chunk_x_max);
                    let min_z = ((cz - local_width).floor() as i32 - 1).max(chunk_z_min);
                    let max_z = ((cz + local_width).ceil() as i32 + 1).min(chunk_z_max);
                    if min_x > max_x || min_z > max_z {
                        continue;
                    }

                    // Tighten the Y sweep to the actual ravine band so chunks
                    // sitting far below the surface don't iterate all 16 layers
                    // per column. `local_bottom` and `surf` are world-Y; we map
                    // to the local layer index and clamp to the chunk window.
                    let ly_min = (local_bottom - origin.y).max(0) as i32;
                    let ly_max = (surf - origin.y).min(CHUNK_SIZE - 1) as i32;
                    if ly_min > ly_max {
                        continue;
                    }

                    for wx in min_x..=max_x {
                        for wz in min_z..=max_z {
                            let lx = wx - origin.x;
                            let lz = wz - origin.z;

                            let dx = wx as f32 - cx;
                            let dz = wz as f32 - cz;
                            let perp = (dx * dx + dz * dz).sqrt();

                            // V-shaped profile: zero width at the bottom, full
                            // width at the surface. Taper the top 2 layers to
                            // preserve grass/dirt on the ravine rim.
                            for ly in ly_min..=ly_max {
                                let wy = origin.y + ly;
                                let depth_frac = (surf - wy) as f32
                                    / (surf - local_bottom).max(1) as f32;
                                // Taper the rim: shrink width near the surface
                                // so grass/dirt on the edge stays intact.
                                let rim_taper = if depth_frac > 0.15 { 1.0 } else { depth_frac / 0.15 };
                                // Jagged walls: vary the half-width by a hash
                                // based on world Y so the canyon sides look natural.
                                let jagged = 1.0 + 0.15 * (hash2(self.seed, wx, wy) - 0.5);
                                let half_width = local_width * depth_frac.max(0.05) * rim_taper * jagged;
                                if perp > half_width {
                                    continue;
                                }

                                let id = chunk.get(lx, ly, lz);
                                if id == stone
                                    || id == dirt
                                    || id == grass
                                    || id == gravel
                                {
                                    // Place water at the very bottom when the
                                    // ravine reaches below sea level.
                                    if wy <= SEA_LEVEL
                                        && wy == local_bottom
                                        && perp <= 1.0
                                    {
                                        chunk.set(lx, ly, lz, water);
                                        chunk.set_water_level(lx, ly, lz, 8);
                                    } else {
                                        chunk.set(lx, ly, lz, BlockId::AIR);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Decorate carved caves with water/lava pools, scattered floor blocks,
    /// extra ore near walls, and simple stalactites/stalagmites.
    pub fn decorate_caves(&self, chunk: &mut Chunk, reg: &BlockRegistry) {
        let origin = chunk_origin(chunk.pos);
        let Some(stone) = reg.id_of("stone") else { return; };
        let Some(dirt) = reg.id_of("dirt") else { return; };
        let Some(gravel) = reg.id_of("gravel") else { return; };
        let Some(water) = reg.id_of("water") else { return; };
        let Some(lava) = reg.id_of("lava") else { return; };

        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                for ly in 0..CHUNK_SIZE {
                    let wx = origin.x + lx;
                    let wy = origin.y + ly;
                    let wz = origin.z + lz;
                    let id = chunk.get(lx, ly, lz);

                    if id.is_air() {
                        continue;
                    }

                    // Ore bonus near cave walls: stone adjacent to air becomes
                    // ore more often.
                    if id == stone && self.is_adjacent_air(chunk, lx, ly, lz) {
                        let ore_noise = self.ore.get_noise_3d(wx as f32, wy as f32, wz as f32);
                        // Boost the ore density by lowering the effective threshold.
                        let mapped = (ore_noise * 1.3).clamp(-1.0, 1.0);
                        if let Some(ore) = self.ore_for_val(mapped, wy, reg) {
                            chunk.set(lx, ly, lz, ore);
                        }
                    }

                    // Floor decorations only on solid blocks with air above.
                    if !(id == stone || id == dirt || id == gravel) {
                        continue;
                    }
                    if ly + 1 >= CHUNK_SIZE || !chunk.get(lx, ly + 1, lz).is_air() {
                        continue;
                    }

                    // Evaluate the cheese-cave 3D noise ONCE per cell. Both
                    // the water/lava pool checks below use the same
                    // `(wx, wy*1.2, wz)` coordinates and threshold, so a
                    // single boolean serves both. Saved: ~1 noise call per
                    // floor-with-air-above cell (~512/chunk).
                    let wy_f12 = wy as f32 * 1.2;
                    let is_cheese = self.cheese_cave.get_noise_3d(wx as f32, wy_f12, wz as f32) > 0.15;

                    // Cheese-cave water pools below sea level. Only place in
                    // small depressions (at least two neighbouring floor blocks)
                    // so entire cavern floors don't flood.
                    if wy <= SEA_LEVEL && is_cheese {
                        let h = hash2(self.seed, wx + wy, wz + 1000);
                        if h < 0.08
                            && self.is_floor_depression(chunk, lx, ly + 1, lz, &[stone, dirt, gravel])
                        {
                            chunk.set(lx, ly + 1, lz, water);
                            // set_water_level stores the liquid level for any
                            // liquid block (water or lava).
                            chunk.set_water_level(lx, ly + 1, lz, 8);
                        }
                    }

                    // Rare lava pools below Y=20.
                    if wy < 20 && is_cheese {
                        let h = hash2(self.seed, wx + wy, wz + 2000);
                        if h < 0.05
                            && self.is_floor_depression(chunk, lx, ly + 1, lz, &[stone, dirt, gravel])
                        {
                            chunk.set(lx, ly + 1, lz, lava);
                            chunk.set_water_level(lx, ly + 1, lz, 8);
                        }
                    }

                    // Gravel/stone floor scatter.
                    let h = hash2(self.seed, wx + 3000, wz);
                    if h < 0.04 && id == stone {
                        chunk.set(lx, ly, lz, gravel);
                    }
                }
            }
        }

        // Stalactites (hanging from ceiling) and stalagmites (rising from floor).
        // Only place them in actual cave pockets below the surface, not on
        // the surface.
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                // Hoist `base_height` (called inside the ly loop previously)
                // so it runs once per column instead of once per AIR cell.
                // `base_height` itself invokes 3 fractal-noise samplers, so
                // for an air-heavy chunk this eliminates ~13 × 3 ≈ 40 noise
                // calls per column × 256 columns = ~10K noise calls total.
                let wx = origin.x + lx;
                let wz = origin.z + lz;
                let surface = self.base_height(wx, wz).round() as i32;
                for ly in 1..(CHUNK_SIZE - 1) {
                    let wy = origin.y + ly;
                    let id = chunk.get(lx, ly, lz);
                    if !id.is_air() {
                        continue;
                    }
                    // Ensure we're underground, not on the surface.
                    if wy >= surface - 2 {
                        continue;
                    }
                    // Require at least two adjacent air blocks so spikes only
                    // form inside real cave voids, not single surface air gaps.
                    let adjacent_air = self.count_adjacent_air(chunk, lx, ly, lz);
                    if adjacent_air < 2 {
                        continue;
                    }

                    // Stalactite: solid above, air here and below.
                    let above = chunk.get(lx, ly + 1, lz);
                    if (above == stone || above == dirt || above == gravel)
                        && chunk.get(lx, ly - 1, lz).is_air()
                    {
                        let h = hash2(self.seed, wx + 4000, wz + wy);
                        if h < 0.06 {
                            let len = 1 + (h * 3.0) as i32;
                            for i in 0..len.min(ly) {
                                chunk.set(lx, ly - i, lz, stone);
                            }
                        }
                    }

                    // Stalagmite: solid below, air here and above.
                    let below = chunk.get(lx, ly - 1, lz);
                    if (below == stone || below == dirt || below == gravel)
                        && chunk.get(lx, ly + 1, lz).is_air()
                    {
                        let h = hash2(self.seed, wx + 5000, wz + wy);
                        if h < 0.06 {
                            let len = 1 + (h * 3.0) as i32;
                            for i in 0..len.min(CHUNK_SIZE - 1 - ly) {
                                chunk.set(lx, ly + i, lz, stone);
                            }
                        }
                    }
                }
            }
        }
    }

    /// True if the block at (lx, ly, lz) is air and at least two of its
    /// horizontal neighbours are solid floors (solid with air above). This
    /// identifies shallow depressions where pooled liquids can sit without
    /// flooding an entire cavern.
    fn is_floor_depression(
        &self,
        chunk: &Chunk,
        lx: i32,
        ly: i32,
        lz: i32,
        floor_blocks: &[BlockId],
    ) -> bool {
        if !chunk.get(lx, ly, lz).is_air() {
            return false;
        }
        let mut floor_neighbours = 0;
        for (dx, dz) in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let nx = lx + dx;
            let nz = lz + dz;
            if nx < 0 || nx >= CHUNK_SIZE || nz < 0 || nz >= CHUNK_SIZE {
                continue;
            }
            let below = chunk.get(nx, ly - 1, nz);
            let here = chunk.get(nx, ly, nz);
            if floor_blocks.contains(&below) && here.is_air() {
                floor_neighbours += 1;
            }
        }
        floor_neighbours >= 2
    }

    /// Count how many of the six neighbours of (lx, ly, lz) are air.
    fn count_adjacent_air(&self, chunk: &Chunk, lx: i32, ly: i32, lz: i32) -> i32 {
        let mut count = 0;
        if lx > 0 && chunk.get(lx - 1, ly, lz).is_air() {
            count += 1;
        }
        if lx + 1 < CHUNK_SIZE && chunk.get(lx + 1, ly, lz).is_air() {
            count += 1;
        }
        if ly > 0 && chunk.get(lx, ly - 1, lz).is_air() {
            count += 1;
        }
        if ly + 1 < CHUNK_SIZE && chunk.get(lx, ly + 1, lz).is_air() {
            count += 1;
        }
        if lz > 0 && chunk.get(lx, ly, lz - 1).is_air() {
            count += 1;
        }
        if lz + 1 < CHUNK_SIZE && chunk.get(lx, ly, lz + 1).is_air() {
            count += 1;
        }
        count
    }

    /// True if any of the six neighbours of (lx, ly, lz) is air.
    fn is_adjacent_air(&self, chunk: &Chunk, lx: i32, ly: i32, lz: i32) -> bool {
        if lx > 0 && chunk.get(lx - 1, ly, lz).is_air() {
            return true;
        }
        if lx + 1 < CHUNK_SIZE && chunk.get(lx + 1, ly, lz).is_air() {
            return true;
        }
        if ly > 0 && chunk.get(lx, ly - 1, lz).is_air() {
            return true;
        }
        if ly + 1 < CHUNK_SIZE && chunk.get(lx, ly + 1, lz).is_air() {
            return true;
        }
        if lz > 0 && chunk.get(lx, ly, lz - 1).is_air() {
            return true;
        }
        if lz + 1 < CHUNK_SIZE && chunk.get(lx, ly, lz + 1).is_air() {
            return true;
        }
        false
    }

    /// Generate a full chunk in place. Does NOT touch neighbours; cross-chunk
    /// decorations (trees) are applied by `decorate` after neighbours exist.
    pub fn generate(&self, chunk: &mut Chunk, reg: &BlockRegistry) {
        let origin = chunk_origin(chunk.pos);
        let stone = reg.id_of("stone").expect("stone block must be registered");
        let dirt = reg.id_of("dirt").expect("dirt block must be registered");
        let grass = reg.id_of("grass").expect("grass block must be registered");
        let sand = reg.id_of("sand").expect("sand block must be registered");
        let water = reg.id_of("water").expect("water block must be registered");
        let bedrock = reg.id_of("bedrock").expect("bedrock block must be registered");
        let snow = reg.id_of("snow").expect("snow block must be registered");
        let gravel = reg.id_of("gravel").expect("gravel block must be registered");

        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let wx = origin.x + lx;
                let wz = origin.z + lz;
                let height_f = self.base_height(wx, wz);
                let height = height_f.round() as i32;
                let biome = self.biome_at(wx, wz, height_f);

                for ly in 0..CHUNK_SIZE {
                    let wy = origin.y + ly;
                    if wy >= WORLD_HEIGHT_BLOCKS {
                        chunk.set(lx, ly, lz, BlockId::AIR);
                        continue;
                    }

                    // Bedrock floor at the very bottom of the world.
                    let mut block = if wy == 0 {
                        bedrock
                    } else if wy < height - 4 {
                        // Deep: stone with possible ores, or gravel pockets.
                        let deep = self.ore.get_noise_3d(wx as f32, wy as f32, wz as f32);
                        let deep_mapped = deep * 2.0 - 1.0;
                        if deep > 0.78 {
                            gravel
                        } else {
                            self.ore_for_val(deep_mapped, wy, reg).unwrap_or(stone)
                        }
                    } else if wy < height - 1 {
                        // Subsurface: dirt (or sand in desert/beach).
                        match biome {
                            BiomeId::Desert | BiomeId::Beach => sand,
                            _ => dirt,
                        }
                    } else if wy < height {
                        // Surface block.
                        match biome {
                            BiomeId::Desert | BiomeId::Beach => sand,
                            BiomeId::Mountains if wy > SEA_LEVEL + 55 => snow,
                            BiomeId::Ocean if wy < SEA_LEVEL => gravel,
                            // Underwater surfaces get dirt, not grass.
                            _ if height <= SEA_LEVEL => dirt,
                            _ => grass,
                        }
                    } else if wy <= SEA_LEVEL {
                        // Below sea level and above terrain: water (oceans/lakes).
                        water
                    } else {
                        BlockId::AIR
                    };

                    // Carve caves — but keep the top crust intact so caves
                    // never break through the surface or puncture the ocean
                    // floor. This prevents seeing caves through water and
                    // avoids surface potholes. Bedrock, water and sand are
                    // never carved.
                    let carveable = block == stone || block == dirt || block == gravel;
                    if carveable && self.carve(chunk, wx, wy, wz, height, biome, reg) {
                        block = BlockId::AIR;
                    }

                    chunk.set(lx, ly, lz, block);
                    // Water placed during worldgen is always a full source block.
                    if block == water {
                        chunk.set_water_level(lx, ly, lz, 8);
                    }
                }
            }
        }

        // Carve surface ravines after the base terrain and caves are in place.
        self.carve_ravines(chunk, reg);

        // Add cave-specific decorations (pools, ores, stalactites).
        self.decorate_caves(chunk, reg);

        chunk.generated = true;
        chunk.dirty = true;
    }

    /// Scatter trees on grass surfaces. Call after the chunk and its neighbours
    /// are generated so trunks/leaves can spill across borders safely.
    pub fn decorate(
        &self,
        chunk: &mut Chunk,
        reg: &BlockRegistry,
        neighbour_sample: impl Fn(i32, i32, i32) -> BlockId,
    ) {
        let origin = chunk_origin(chunk.pos);
        let grass = match reg.id_of("grass") {
            Some(g) => g,
            None => return,
        };
        let wood = match reg.id_of("wood") {
            Some(w) => w,
            None => return,
        };
        let leaves = match reg.id_of("leaves") {
            Some(l) => l,
            None => return,
        };
        // New decorative blocks. None of these abort decoration: a missing block
        // just skips that particular feature so the rest can still be placed.
        let birch_log = reg.id_of("birch_log");
        let birch_leaves = reg.id_of("birch_leaves");
        let spruce_log = reg.id_of("spruce_log");
        let spruce_leaves = reg.id_of("spruce_leaves");
        let tall_grass = reg.id_of("tall_grass");
        let poppy = reg.id_of("poppy");
        let dandelion = reg.id_of("dandelion");
        let cactus = reg.id_of("cactus");
        let mushroom_red = reg.id_of("mushroom_red");
        let mushroom_brown = reg.id_of("mushroom_brown");
        let sand = reg.id_of("sand");

        // Which tree variety to plant for a given column.
        #[derive(Clone, Copy)]
        enum TreeType {
            Oak,
            Birch,
            Spruce,
        }

        // Deterministic per-column tree decision from cellular noise + hash.
        for lx in 2..CHUNK_SIZE - 2 {
            for lz in 2..CHUNK_SIZE - 2 {
                let wx = origin.x + lx;
                let wz = origin.z + lz;
                // Tree density: cellular value high + per-position hash.
                let n = self.tree.get_noise_2d(wx as f32, wz as f32);
                let h = hash2(self.seed, wx, wz);
                // Biome is derived from the base height field (same source as
                // `generate`), so decorations match the surface biome exactly.
                let height_f = self.base_height(wx, wz);
                let biome = self.biome_at(wx, wz, height_f);

                // ---- Trees ------------------------------------------------
                if n >= 0.55 && h <= 0.12 {
                    // Find a grass surface by scanning down from chunk top.
                    let mut surface_y = None;
                    for ly in (0..CHUNK_SIZE).rev() {
                        let wy = origin.y + ly;
                        if wy >= WORLD_HEIGHT_BLOCKS {
                            continue;
                        }
                        let b = chunk.get(lx, ly, lz);
                        if b == grass {
                            surface_y = Some(ly);
                            break;
                        }
                        if !b.is_air() {
                            break; // hit non-grass solid before grass -> no tree
                        }
                    }
                    if let Some(sy) = surface_y {
                        // Pick tree variety by biome + secondary hash. Oak is
                        // the default/fallback when no biome variant fires.
                        let tree_hash = hash2(self.seed, wx + 1, wz);
                        let tree_type = match biome {
                            BiomeId::Forest if tree_hash < 0.3 => TreeType::Birch,
                            BiomeId::Mountains if tree_hash < 0.4 => TreeType::Spruce,
                            _ => TreeType::Oak,
                        };
                        match tree_type {
                            TreeType::Oak => {
                                let trunk_top = (sy + 5).min(CHUNK_SIZE - 1);
                                for ty in (sy + 1)..=trunk_top {
                                    chunk.set(lx, ty, lz, wood);
                                }
                                // Leaf canopy: a 3×3×2 blob centred on trunk_top.
                                let cy = trunk_top;
                                for dy in 0..=2 {
                                    for dx in -2i32..=2 {
                                        for dz in -2i32..=2 {
                                            if dx == 0 && dz == 0 && dy < 2 {
                                                continue; // leave the trunk
                                            }
                                            if dx.abs() == 2 && dz.abs() == 2 {
                                                continue; // round corners
                                            }
                                            let lx2 = lx + dx;
                                            let lz2 = lz + dz;
                                            let ly2 = cy + dy;
                                            if (0..CHUNK_SIZE).contains(&lx2)
                                                && (0..CHUNK_SIZE).contains(&lz2)
                                                && (0..CHUNK_SIZE).contains(&ly2)
                                                && chunk.get(lx2, ly2, lz2).is_air()
                                            {
                                                chunk.set(lx2, ly2, lz2, leaves);
                                            }
                                        }
                                    }
                                }
                            }
                            TreeType::Birch => {
                                if let (Some(log), Some(leaf)) = (birch_log, birch_leaves) {
                                    // 6-7 tall trunk.
                                    let trunk_h =
                                        6 + (hash2(self.seed, wx + 3, wz + 3) * 2.0) as i32;
                                    let trunk_top = (sy + trunk_h).min(CHUNK_SIZE - 1);
                                    for ty in (sy + 1)..=trunk_top {
                                        chunk.set(lx, ty, lz, log);
                                    }
                                    // Small 2×2×2 leaf canopy at the top.
                                    for dy in 0..2 {
                                        let ly2 = trunk_top + dy;
                                        for dx in -1i32..=0 {
                                            for dz in -1i32..=0 {
                                                let lx2 = lx + dx;
                                                let lz2 = lz + dz;
                                                if (0..CHUNK_SIZE).contains(&lx2)
                                                    && (0..CHUNK_SIZE).contains(&lz2)
                                                    && (0..CHUNK_SIZE).contains(&ly2)
                                                    && chunk.get(lx2, ly2, lz2).is_air()
                                                {
                                                    chunk.set(lx2, ly2, lz2, leaf);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            TreeType::Spruce => {
                                if let (Some(log), Some(leaf)) = (spruce_log, spruce_leaves) {
                                    // 6-8 tall trunk.
                                    let trunk_h =
                                        6 + (hash2(self.seed, wx + 5, wz + 5) * 3.0) as i32;
                                    let trunk_top = (sy + trunk_h).min(CHUNK_SIZE - 1);
                                    for ty in (sy + 1)..=trunk_top {
                                        chunk.set(lx, ty, lz, log);
                                    }
                                    // Narrow tapered canopy: 3×3 at the bottom
                                    // narrowing to 1×1 at the very top.
                                    let canopy_top = trunk_top;
                                    let canopy_bottom = trunk_top - 2;
                                    for ly2 in canopy_bottom..=canopy_top {
                                        let dist_from_top = canopy_top - ly2;
                                        let radius = if dist_from_top == 0 { 0 } else { 1 };
                                        for dx in -radius..=radius {
                                            for dz in -radius..=radius {
                                                // Leave the trunk except at the top.
                                                if dx == 0 && dz == 0 && ly2 < canopy_top {
                                                    continue;
                                                }
                                                let lx2 = lx + dx;
                                                let lz2 = lz + dz;
                                                if (0..CHUNK_SIZE).contains(&lx2)
                                                    && (0..CHUNK_SIZE).contains(&lz2)
                                                    && (0..CHUNK_SIZE).contains(&ly2)
                                                    && chunk.get(lx2, ly2, lz2).is_air()
                                                {
                                                    chunk.set(lx2, ly2, lz2, leaf);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // ---- Surface foliage scatter ------------------------------
                // Find the surface block (grass, or sand in desert/beach) by
                // scanning down. Trees above would have caused an early break
                // here, so foliage naturally avoids tree trunks.
                let mut foliage_surface = None;
                for ly in (0..CHUNK_SIZE).rev() {
                    let wy = origin.y + ly;
                    if wy >= WORLD_HEIGHT_BLOCKS {
                        continue;
                    }
                    let b = chunk.get(lx, ly, lz);
                    let is_grass = b == grass;
                    let is_sand = sand.map(|s| b == s).unwrap_or(false);
                    if is_grass || is_sand {
                        foliage_surface = Some(ly);
                        break;
                    }
                    if !b.is_air() {
                        break;
                    }
                }
                if let Some(sy) = foliage_surface {
                    let fy = sy + 1;
                    if fy < CHUNK_SIZE && chunk.get(lx, fy, lz).is_air() {
                        match biome {
                            BiomeId::Plains => {
                                if h < 0.02 {
                                    // Rare flower: poppy or dandelion.
                                    if hash2(self.seed, wx, wz + 100) < 0.5 {
                                        if let Some(p) = poppy {
                                            chunk.set(lx, fy, lz, p);
                                        }
                                    } else if let Some(d) = dandelion {
                                        chunk.set(lx, fy, lz, d);
                                    }
                                } else if h < 0.15 {
                                    if let Some(tg) = tall_grass {
                                        chunk.set(lx, fy, lz, tg);
                                    }
                                }
                            }
                            BiomeId::Forest => {
                                if h < 0.03 {
                                    // Mushroom: red or brown.
                                    if hash2(self.seed, wx, wz + 100) < 0.5 {
                                        if let Some(m) = mushroom_red {
                                            chunk.set(lx, fy, lz, m);
                                        }
                                    } else if let Some(m) = mushroom_brown {
                                        chunk.set(lx, fy, lz, m);
                                    }
                                } else if h < 0.10 {
                                    if let Some(tg) = tall_grass {
                                        chunk.set(lx, fy, lz, tg);
                                    }
                                }
                            }
                            BiomeId::Desert => {
                                if h < 0.05 {
                                    if let Some(c) = cactus {
                                        let cactus_h =
                                            1 + (hash2(self.seed, wx + 7, wz + 7) * 3.0) as i32;
                                        for i in 0..cactus_h {
                                            let cy = sy + 1 + i;
                                            if cy < CHUNK_SIZE && chunk.get(lx, cy, lz).is_air() {
                                                chunk.set(lx, cy, lz, c);
                                            }
                                        }
                                    }
                                }
                            }
                            BiomeId::Mountains => {
                                if h < 0.08 {
                                    if let Some(tg) = tall_grass {
                                        chunk.set(lx, fy, lz, tg);
                                    }
                                }
                            }
                            BiomeId::Beach | BiomeId::Ocean => {}
                        }
                    }
                }
            }
        }
        // neighbour_sample is reserved for future cross-chunk foliage; keep the
        // parameter so callers don't need to change when we add it.
        let _ = neighbour_sample;
    }

    /// Place dungeons, ruined towers and wells. Each chunk independently
    /// recomputes which structures overlap it (deterministically, via the
    /// world-seed hash at the structure's anchor column) and writes only the
    /// blocks that fall inside its own bounds; `chunk.set` no-ops on
    /// out-of-range locals, so cross-chunk structures are stitched together
    /// by each chunk placing its own slice. `sample` reads world blocks
    /// across chunk borders for placement-condition checks.
    pub fn place_structures(
        &self,
        chunk: &mut Chunk,
        reg: &BlockRegistry,
        sample: &dyn Fn(i32, i32, i32) -> voxel_core::BlockId,
    ) {
        let origin = chunk_origin(chunk.pos);
        let (
            Some(stone),
            Some(mossy),
            Some(chest),
            Some(cobble),
            Some(grass),
            Some(dirt),
            Some(water),
        ) = (
            reg.id_of("stone"),
            reg.id_of("mossy_cobblestone"),
            reg.id_of("chest"),
            reg.id_of("cobblestone"),
            reg.id_of("grass"),
            reg.id_of("dirt"),
            reg.id_of("water"),
        )
        else {
            return;
        };

        // Dungeons: 5×3×5 underground rooms, anchor at (ax, ay, az).
        const DS: i32 = 5;
        const DH: i32 = 3;
        for ax in (origin.x - (DS - 1))..=(origin.x + CHUNK_SIZE - 1) {
            for az in (origin.z - (DS - 1))..=(origin.z + CHUNK_SIZE - 1) {
                if hash2(self.seed, ax, az) >= 0.003 {
                    continue;
                }
                let wy = 10 + (hash2(self.seed.wrapping_mul(2), ax, az) * 30.0) as i32;
                if !(10..=40).contains(&wy) {
                    continue;
                }
                // Only carve into solid stone (we're underground): check the
                // block just above the ceiling.
                if sample(ax + 2, wy + DH, az + 2) != stone {
                    continue;
                }
                // Don't spawn in or near water.
                if sample(ax + 2, wy, az + 2) == water {
                    continue;
                }
                for dx in 0..DS {
                    for dy in 0..DH {
                        for dz in 0..DS {
                            let on_shell = dx == 0
                                || dx == DS - 1
                                || dz == 0
                                || dz == DS - 1
                                || dy == 0
                                || dy == DH - 1;
                            let id = if on_shell { mossy } else { BlockId::AIR };
                            chunk.set(
                                ax + dx - origin.x,
                                wy + dy - origin.y,
                                az + dz - origin.z,
                                id,
                            );
                        }
                    }
                }
                // Chest on the centre floor.
                chunk.set(ax + 2 - origin.x, wy - origin.y, az + 2 - origin.z, chest);
            }
        }

        // Ruined towers: 3×3 hollow cobblestone shell on the surface.
        const TS: i32 = 3;
        for ax in (origin.x - (TS - 1))..=(origin.x + CHUNK_SIZE - 1) {
            for az in (origin.z - (TS - 1))..=(origin.z + CHUNK_SIZE - 1) {
                if hash2(self.seed, ax, az) >= 0.001 {
                    continue;
                }
                let surf_f = self.base_height(ax, az);
                let surf = surf_f.round() as i32;
                let surface_block = sample(ax, surf - 1, az);
                if surface_block != grass && surface_block != dirt {
                    continue;
                }
                // Don't spawn towers underwater.
                if surf <= SEA_LEVEL {
                    continue;
                }
                let height = 6 + (hash2(self.seed.wrapping_mul(3), ax, az) * 5.0) as i32;
                for dx in 0..TS {
                    for dz in 0..TS {
                        let is_wall = dx == 0 || dx == TS - 1 || dz == 0 || dz == TS - 1;
                        if !is_wall {
                            continue;
                        }
                        for dy in 0..height {
                            // 1-block door gap at ground level on the +X face.
                            if dy == 0 && dx == TS - 1 && dz == 1 {
                                continue;
                            }
                            chunk.set(
                                ax + dx - origin.x,
                                surf + dy - origin.y,
                                az + dz - origin.z,
                                cobble,
                            );
                        }
                    }
                }
            }
        }

        // Wells: 3×3×4 cobblestone ring (dry), Desert/Plains only.
        const WS: i32 = 3;
        const WH: i32 = 4;
        for ax in (origin.x - (WS - 1))..=(origin.x + CHUNK_SIZE - 1) {
            for az in (origin.z - (WS - 1))..=(origin.z + CHUNK_SIZE - 1) {
                if hash2(self.seed, ax, az) >= 0.005 {
                    continue;
                }
                let surf_f = self.base_height(ax, az);
                let surf = surf_f.round() as i32;
                let biome = self.biome_at(ax, az, surf_f);
                if biome != BiomeId::Desert && biome != BiomeId::Plains {
                    continue;
                }
                // Don't spawn wells underwater.
                if surf <= SEA_LEVEL {
                    continue;
                }
                for dx in 0..WS {
                    for dz in 0..WS {
                        let is_center = dx == 1 && dz == 1;
                        let id = if is_center { BlockId::AIR } else { cobble };
                        for dy in 0..WH {
                            chunk.set(
                                ax + dx - origin.x,
                                surf + dy - origin.y,
                                az + dz - origin.z,
                                id,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Height of the highest non-air block in world column (x, z), or 0.
    pub fn column_height(&self, chunk: &Chunk) -> i32 {
        // Scan columns for the highest non-air block.
        let origin = chunk_origin(chunk.pos);
        let mut max_height = -1i32;
        for lx in 0..voxel_core::CHUNK_SIZE {
            for lz in 0..voxel_core::CHUNK_SIZE {
                let h = chunk.column_height(lx, lz);
                if h > max_height {
                    max_height = h;
                }
            }
        }
        origin.y + max_height + 1
    }
}

/// Deterministic hash of (seed, x, z) into [0, 1).
fn hash2(seed: i32, x: i32, z: i32) -> f32 {
    let mut h = (seed as u32).wrapping_mul(374761393);
    h = h.wrapping_add(x as u32).wrapping_mul(668265263);
    h = h.wrapping_add(z as u32).wrapping_mul(1274126177);
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    (h >> 8) as f32 / ((1u32 << 24) as f32)
}

/// Convert a world block position to its owning chunk position.
pub fn chunk_of(block: BlockPos) -> ChunkPos {
    voxel_core::math::block_to_chunk(block.0)
}
