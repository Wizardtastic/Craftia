//! `World` — the shared facade over chunk storage, block registry, terrain
//! generation, and chunk meshes. Safe to share across threads via `Arc`; all
//! mutation goes through internal `RwLock`s.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};

use parking_lot::RwLock;

use glam::IVec3;
use voxel_core::{
    math::{block_to_chunk, chunk_origin, ChunkPos},
    BlockId, BlockPos, CHUNK_SIZE, WORLD_HEIGHT_BLOCKS,
};

use crate::schematic::{SchematicEntity, SchematicId};
use crate::{chunk::Chunk, gen::TerrainGenerator, registry::BlockRegistry};

/// Number of shards for the chunk storage. Must be a power of two.
const NUM_SHARDS: usize = 16;
const SHARD_MASK: usize = NUM_SHARDS - 1;

/// Sharded chunk storage: splits the chunk map across N independent `RwLock`s
/// so concurrent readers/writers hitting different regions of the world don't
/// contend on the same lock. The shard is selected by hashing the chunk
/// position.
pub struct ShardedChunks {
    shards: [RwLock<HashMap<ChunkPos, Chunk>>; NUM_SHARDS],
}

impl ShardedChunks {
    fn new() -> Self {
        // `HashMap::new()` is const-capable in recent Rust; build the array
        // with a small loop to avoid needing `MaybeUninit`.
        let mut shards: Vec<RwLock<HashMap<ChunkPos, Chunk>>> = Vec::with_capacity(NUM_SHARDS);
        for _ in 0..NUM_SHARDS {
            shards.push(RwLock::new(HashMap::new()));
        }
        // Convert `Vec` → `[T; N]` via `try_into()`.
        let vec: Vec<RwLock<HashMap<ChunkPos, Chunk>>> = shards;
        let arr: [RwLock<HashMap<ChunkPos, Chunk>>; NUM_SHARDS] =
            vec.try_into().unwrap_or_else(|_| {
                panic!("shard count mismatch");
            });
        Self { shards: arr }
    }

    #[inline]
    fn shard_index(pos: ChunkPos) -> usize {
        // Simple hash: XOR the coordinate components and mask to shard count.
        let h = (pos.x() as usize).wrapping_mul(0x9e3779b9)
            ^ (pos.y() as usize).wrapping_mul(0x85ebca6b)
            ^ (pos.z() as usize).wrapping_mul(0xc2b2ae35);
        h & SHARD_MASK
    }

    /// Read-only access to the shard containing `pos`.
    #[inline]
    pub fn read_shard(
        &self,
        pos: ChunkPos,
    ) -> parking_lot::RwLockReadGuard<'_, HashMap<ChunkPos, Chunk>> {
        self.shards[Self::shard_index(pos)].read()
    }

    /// Write access to the shard containing `pos`.
    #[inline]
    pub fn write_shard(
        &self,
        pos: ChunkPos,
    ) -> parking_lot::RwLockWriteGuard<'_, HashMap<ChunkPos, Chunk>> {
        self.shards[Self::shard_index(pos)].write()
    }

    /// Read access to ALL shards (for operations like `all_loaded_chunks`).
    /// Caller must be careful not to deadlock by holding multiple shard reads.
    pub fn read_all(&self) -> Vec<parking_lot::RwLockReadGuard<'_, HashMap<ChunkPos, Chunk>>> {
        self.shards.iter().map(|s| s.read()).collect()
    }

    /// Total number of loaded chunks across all shards.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.read().len()).sum()
    }

    /// True if no chunks are loaded.
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.read().is_empty())
    }
}

pub(crate) enum ConditionalBlockWrite {
    Unloaded,
    Skipped,
    Changed(BlockId),
}

pub struct World {
    seed: i32,
    reg: Arc<BlockRegistry>,
    gen: Arc<TerrainGenerator>,
    pub(crate) chunks: ShardedChunks,
    /// Positions of chunks with a finished mesh. The mesh data itself
    /// travels to the renderer via `ChunkStreamEvent::MeshReady`; the world
    /// only tracks which chunks are meshed (for counts / debug overlays).
    meshes: RwLock<HashSet<ChunkPos>>,
    sun_dir: RwLock<glam::Vec3>,
    /// Positions of water sources/flowing water still spreading. Drained and
    /// refilled each water tick by `tick_water`.
    pending_flow: RwLock<HashSet<IVec3>>,
    /// Set of positions that are water sources (level 8). Used for O(1)
    /// lookups during water simulation instead of scanning all chunks.
    source_water: RwLock<HashSet<IVec3>>,
    /// Chunks whose water mesh changed outside an active simulation tick
    /// (e.g. source removal). Drained by `tick_water` for remeshing.
    water_remesh: RwLock<HashSet<ChunkPos>>,
    /// Accumulated wall-clock seconds since the last water tick.
    water_tick_accumulator: RwLock<f32>,
    /// Reusable scratch buffers for `simulate_flow_step` to avoid per-tick allocations.
    water_sim_buf: RwLock<crate::water::SimulateBuffers>,
    /// Self-reference for safe cross-thread closure capture. Built once in
    /// `new_with_path` via `Arc::new_cyclic`; used by `with_chunk_for_mesh`,
    /// `recompute_lighting_at`, and `set_block` to hand closures an `Arc<World>`
    /// without using raw `*const Self` pointer aliasing.
    self_ref: Weak<World>,
    /// Registry of named schematics pasted into this world. Each entry
    /// is the post-paste metadata (origin / rotation / mirror / bounds)
    /// plus a count of distinct blocks actually written. Lives behind a
    /// `RwLock` because paste operations can come from any system or
    /// network thread and list-snapshot reads are common from the UI.
    ///
    /// `pub(crate)` because sibling modules inside `voxel-world`
    /// (notably `schematic::impl World { paste_schematic, … }`) need to
    /// read/write this map; keeping it private to `world.rs` while
    /// letting methods live in `schematic.rs` would otherwise require
    /// a separate accessor layer for every operation.
    pub(crate) schematic_entities: RwLock<HashMap<SchematicId, SchematicEntity>>,
}

/// Seconds between water simulation steps. Minecraft ticks water every 5 game
/// ticks (~0.25s at 20 TPS).
pub const WATER_TICK_INTERVAL: f32 = 0.25;

impl World {
    pub fn new(seed: i32) -> Arc<Self> {
        Self::new_with_path(seed, None)
    }

    /// Create a world, optionally loading block definitions from a JSON file
    /// at `assets_path/blocks/blocks.json`. Falls back to built-in blocks if
    /// the path is `None` or the file doesn't exist.
    pub fn new_with_path(seed: i32, assets_path: Option<&std::path::Path>) -> Arc<Self> {
        let reg = match assets_path {
            Some(path) => {
                let loader = voxel_assets::AssetLoader::new(path);
                match loader.load_blocks() {
                    Ok(blocks) => {
                        log::info!("loaded {} blocks from {}", blocks.len(), path.display());
                        Arc::new(BlockRegistry::from_assets(&blocks))
                    }
                    Err(e) => {
                        log::warn!(
                            "failed to load blocks from {}: {e}. Using builtins.",
                            path.display()
                        );
                        Arc::new(BlockRegistry::with_builtins())
                    }
                }
            }
            None => Arc::new(BlockRegistry::with_builtins()),
        };
        let gen = Arc::new(TerrainGenerator::new(seed));
        // `Arc::new_cyclic` lets the struct hold a `Weak<Self>` back-reference
        // without a chicken-and-egg bootstrap. The closure receives the Weak
        // during construction; we close over it to seed `self_ref`.
        Arc::new_cyclic(|weak: &Weak<Self>| Self {
            seed,
            reg,
            gen,
            chunks: ShardedChunks::new(),
            meshes: RwLock::new(HashSet::new()),
            sun_dir: RwLock::new(glam::Vec3::new(0.3, 0.9, 0.1).normalize()),
            pending_flow: RwLock::new(HashSet::new()),
            source_water: RwLock::new(HashSet::new()),
            water_remesh: RwLock::new(HashSet::new()),
            water_tick_accumulator: RwLock::new(0.0),
            water_sim_buf: RwLock::new(crate::water::SimulateBuffers::new()),
            self_ref: Weak::clone(weak),
            schematic_entities: RwLock::new(HashMap::new()),
        })
    }

    pub fn seed(&self) -> i32 {
        self.seed
    }
    pub fn registry(&self) -> Arc<BlockRegistry> {
        self.reg.clone()
    }
    pub fn terrain(&self) -> Arc<TerrainGenerator> {
        self.gen.clone()
    }

    pub fn set_sun_dir(&self, dir: glam::Vec3) {
        *self.sun_dir.write() = dir;
    }

    pub fn get_sun_dir(&self) -> glam::Vec3 {
        *self.sun_dir.read()
    }

    // --- chunk storage -----------------------------------------------------

    pub fn insert_chunk(&self, pos: ChunkPos, mut chunk: Chunk) {
        chunk.pos = pos;
        let origin = chunk_origin(pos);
        let water_id = self.reg.id_of("water");
        let mut water_blocks = Vec::new();
        if let Some(water_id) = water_id {
            for ly in 0..CHUNK_SIZE {
                for lz in 0..CHUNK_SIZE {
                    for lx in 0..CHUNK_SIZE {
                        let level = chunk.get_water_level(lx, ly, lz);
                        if chunk.get(lx, ly, lz) == water_id && level > 0 {
                            water_blocks.push((
                                IVec3::new(origin.x + lx, origin.y + ly, origin.z + lz),
                                level,
                            ));
                        }
                    }
                }
            }
        }
        self.chunks.write_shard(pos).insert(pos, chunk);

        // Generated and loaded chunks may already contain water. Register it
        // with the same incremental simulation used for bucket-placed water.
        self.source_water
            .write()
            .retain(|p| block_to_chunk(*p) != pos);
        self.pending_flow
            .write()
            .retain(|p| block_to_chunk(*p) != pos);
        self.water_remesh.write().remove(&pos);
        {
            let mut sources = self.source_water.write();
            let mut pending = self.pending_flow.write();

            for &(p, level) in &water_blocks {
                pending.insert(p);
                if level == 8 {
                    sources.insert(p);
                }
            }
        }

        // A boundary liquid cell may have stopped ticking while its adjacent
        // chunk was unloaded. Resume water immediately across loaded borders.
        let mut edge_water = Vec::new();
        let mut check_edge = |p: IVec3| {
            if let Some(water_id) = water_id {
                if self.get_block(p.x, p.y, p.z) == water_id
                    && self.get_water_level_world(p.x, p.y, p.z) > 0
                {
                    edge_water.push(p);
                }
            }
        };
        for a in 0..CHUNK_SIZE {
            for b in 0..CHUNK_SIZE {
                check_edge(IVec3::new(origin.x - 1, origin.y + a, origin.z + b));
                check_edge(IVec3::new(
                    origin.x + CHUNK_SIZE,
                    origin.y + a,
                    origin.z + b,
                ));
                check_edge(IVec3::new(origin.x + a, origin.y - 1, origin.z + b));
                check_edge(IVec3::new(
                    origin.x + a,
                    origin.y + CHUNK_SIZE,
                    origin.z + b,
                ));
                check_edge(IVec3::new(origin.x + a, origin.y + b, origin.z - 1));
                check_edge(IVec3::new(
                    origin.x + a,
                    origin.y + b,
                    origin.z + CHUNK_SIZE,
                ));
            }
        }
        self.pending_flow.write().extend(edge_water);
    }

    /// Fraction of an axis-aligned box occupied by water, accounting for the
    /// 0–8 fluid surface height. Used by player movement so shallow water only
    /// affects physics when the player's bounds actually intersect it.
    pub fn water_submersion(&self, min: glam::Vec3, max: glam::Vec3) -> f32 {
        let size = max - min;
        let volume = size.x * size.y * size.z;
        if !volume.is_finite() || volume <= 0.0 {
            return 0.0;
        }
        let Some(water_id) = self.reg.id_of("water") else {
            return 0.0;
        };
        let min_block = voxel_core::math::world_to_block(min);
        let max_block = voxel_core::math::world_to_block(max - glam::Vec3::splat(0.001));
        let mut submerged_volume = 0.0;
        for by in min_block.y..=max_block.y {
            for bz in min_block.z..=max_block.z {
                for bx in min_block.x..=max_block.x {
                    if self.get_block(bx, by, bz) != water_id {
                        continue;
                    }
                    let level = self.get_water_level_world(bx, by, bz);
                    if level == 0 {
                        continue;
                    }
                    let overlap_x = (max.x.min(bx as f32 + 1.0) - min.x.max(bx as f32)).max(0.0);
                    let overlap_z = (max.z.min(bz as f32 + 1.0) - min.z.max(bz as f32)).max(0.0);
                    let water_top = by as f32 + level as f32 / 8.0;
                    let overlap_y = (max.y.min(water_top) - min.y.max(by as f32)).max(0.0);
                    submerged_volume += overlap_x * overlap_y * overlap_z;
                }
            }
        }
        (submerged_volume / volume).clamp(0.0, 1.0)
    }

    /// Queue loaded water blocks beside a changed block so openings and newly
    /// loaded chunk boundaries immediately affect nearby fluid flow.
    fn enqueue_water_neighbours(&self, pos: IVec3) {
        let Some(water_id) = self.reg.id_of("water") else {
            return;
        };
        let neighbours: Vec<_> = water_neighbours(pos)
            .into_iter()
            .filter(|p| {
                self.get_block(p.x, p.y, p.z) == water_id
                    && self.get_water_level_world(p.x, p.y, p.z) > 0
            })
            .collect();
        self.pending_flow.write().extend(neighbours);
    }

    pub fn remove_chunk(&self, pos: ChunkPos) {
        self.chunks.write_shard(pos).remove(&pos);
        self.meshes.write().remove(&pos);
        self.water_remesh.write().remove(&pos);
        self.source_water
            .write()
            .retain(|p| block_to_chunk(*p) != pos);
        self.pending_flow
            .write()
            .retain(|p| block_to_chunk(*p) != pos);
        let origin = chunk_origin(pos);
        let mut edge_water = Vec::new();
        let mut check_edge = |p: IVec3| {
            if self.reg.id_of("water").is_some_and(|water_id| {
                self.get_block(p.x, p.y, p.z) == water_id
                    && self.get_water_level_world(p.x, p.y, p.z) > 0
            }) {
                edge_water.push(p);
            }
        };
        for a in 0..CHUNK_SIZE {
            for b in 0..CHUNK_SIZE {
                check_edge(IVec3::new(origin.x - 1, origin.y + a, origin.z + b));
                check_edge(IVec3::new(
                    origin.x + CHUNK_SIZE,
                    origin.y + a,
                    origin.z + b,
                ));
                check_edge(IVec3::new(origin.x + a, origin.y - 1, origin.z + b));
                check_edge(IVec3::new(
                    origin.x + a,
                    origin.y + CHUNK_SIZE,
                    origin.z + b,
                ));
                check_edge(IVec3::new(origin.x + a, origin.y + b, origin.z - 1));
                check_edge(IVec3::new(
                    origin.x + a,
                    origin.y + b,
                    origin.z + CHUNK_SIZE,
                ));
            }
        }
        self.pending_flow.write().extend(edge_water);
    }

    /// Record that the chunk at `pos` has a finished mesh. The world does not
    /// store mesh data itself — the bundle travels to the renderer via
    /// `ChunkStreamEvent::MeshReady`; this map only tracks which positions
    /// are meshed (for counts and debug overlays).
    pub fn insert_mesh(&self, pos: ChunkPos) {
        self.meshes.write().insert(pos);
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.chunks.len()
    }
    pub fn meshed_chunk_count(&self) -> usize {
        self.meshes.read().len()
    }

    pub(crate) fn mark_water_remesh(&self, pos: IVec3) {
        self.water_remesh.write().insert(block_to_chunk(pos));
    }

    pub(crate) fn replace_pending_water(
        &self,
        remove: HashSet<IVec3>,
        insert: impl IntoIterator<Item = IVec3>,
    ) {
        let mut pending = self.pending_flow.write();
        for pos in remove {
            pending.remove(&pos);
        }
        pending.extend(insert);
    }

    /// Number of blocks pending water flow simulation.
    pub fn pending_flow_count(&self) -> usize {
        self.pending_flow.read().len()
    }

    /// Get all loaded chunks (for save).
    pub fn all_loaded_chunks(&self) -> Vec<(ChunkPos, Chunk)> {
        let mut result = Vec::new();
        for shard in self.chunks.shards.iter() {
            let guard = shard.read();
            for (&cp, c) in guard.iter() {
                result.push((cp, c.clone()));
            }
        }
        result
    }

    /// Insert multiple chunks (for load).
    pub fn insert_chunks(&self, chunks: Vec<(ChunkPos, Chunk)>) {
        for (cp, chunk) in chunks {
            self.insert_chunk(cp, chunk);
        }
    }

    /// Get sunlight at a world coordinate (0 if chunk not loaded).
    pub fn get_sunlight_world(&self, x: i32, y: i32, z: i32) -> u8 {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return 0;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let chunks = self.chunks.read_shard(cp);
        let Some(chunk) = chunks.get(&cp) else {
            if y >= voxel_core::SEA_LEVEL {
                return 15;
            }
            return 0;
        };
        let origin = chunk_origin(cp);
        chunk.get_sunlight(x - origin.x, y - origin.y, z - origin.z)
    }

    /// Get torchlight at a world coordinate (0 if chunk not loaded).
    pub fn get_torchlight_world(&self, x: i32, y: i32, z: i32) -> u8 {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return 0;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let chunks = self.chunks.read_shard(cp);
        let Some(chunk) = chunks.get(&cp) else {
            return 0;
        };
        let origin = chunk_origin(cp);
        chunk.get_torchlight(x - origin.x, y - origin.y, z - origin.z)
    }

    /// Set torchlight at a world coordinate (no-op if chunk not loaded).
    pub fn set_torchlight_world(&self, x: i32, y: i32, z: i32, v: u8) {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let mut chunks = self.chunks.write_shard(cp);
        let Some(chunk) = chunks.get_mut(&cp) else {
            return;
        };
        let origin = chunk_origin(cp);
        chunk.set_torchlight(x - origin.x, y - origin.y, z - origin.z, v);
    }

    /// Set torchlight color (packed RGBA8, `R8G8B8A8_UNORM` layout, low byte
    /// = R) at a world coordinate (no-op if chunk not loaded). Pass `0` for
    /// "no color" (sunlight-only cell).
    pub fn set_torchlight_color_world(&self, x: i32, y: i32, z: i32, color: u32) {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let mut chunks = self.chunks.write_shard(cp);
        let Some(chunk) = chunks.get_mut(&cp) else {
            return;
        };
        let origin = chunk_origin(cp);
        chunk.set_torchlight_color(x - origin.x, y - origin.y, z - origin.z, color);
    }

    /// Recompute sunlight and torchlight for a single chunk. Used after water
    /// flow simulation modifies block types/light absorption across chunks.
    pub fn recompute_lighting_at(&self, cp: ChunkPos) {
        let reg = self.reg.clone();
        let chunks = self.chunks.read_shard(cp);
        let Some(chunk) = chunks.get(&cp) else {
            return;
        };
        let mut chunk_copy = chunk.clone();
        drop(chunks);

        let sun_dir = self.get_sun_dir();

        // Upgrade the self-reference once per call (`self_ref` was seeded via
        // `Arc::new_cyclic`); closures capture the resulting `Arc<World>` by
        // move so cross-thread access is safe without raw self-pointers.
        let arc = self.self_ref.upgrade().expect("World outlives use");
        let sample_block = {
            let arc = Arc::clone(&arc);
            move |wx: i32, wy: i32, wz: i32| arc.get_block(wx, wy, wz)
        };
        let sample_torch = {
            let arc = Arc::clone(&arc);
            move |wx: i32, wy: i32, wz: i32| arc.get_torchlight_world(wx, wy, wz)
        };
        let mut cross_updates = Vec::new();
        crate::light::compute_all(
            &mut chunk_copy,
            &reg,
            sun_dir,
            &sample_block,
            &sample_torch,
            &mut |pos, level, color| cross_updates.push((pos, level, color)),
        );

        for (pos, level, color) in cross_updates {
            self.set_torchlight_world(pos.0.x, pos.0.y, pos.0.z, level);
            self.set_torchlight_color_world(pos.0.x, pos.0.y, pos.0.z, color);
        }

        let mut chunks = self.chunks.write_shard(cp);
        if let Some(chunk) = chunks.get_mut(&cp) {
            std::mem::swap(&mut chunk.sunlight, &mut chunk_copy.sunlight);
            std::mem::swap(&mut chunk.torchlight, &mut chunk_copy.torchlight);
            std::mem::swap(
                &mut chunk.torchlight_color,
                &mut chunk_copy.torchlight_color,
            );
            chunk.dirty = true;
            chunk.light_dirty = true;
        }
    }

    /// True if a chunk is loaded (generated) at the given chunk position.
    pub fn is_chunk_loaded(&self, pos: ChunkPos) -> bool {
        self.chunks.read_shard(pos).contains_key(&pos)
    }

    /// Chunk debug info for the minimap visualization.
    /// Returns (loaded, dirty, palette_mode, has_mesh).
    pub fn chunk_debug_info(&self, pos: ChunkPos) -> (bool, bool, bool, bool) {
        let loaded = {
            let chunks = self.chunks.read_shard(pos);
            match chunks.get(&pos) {
                Some(c) => (true, c.dirty, c.is_palette_mode()),
                None => return (false, false, false, false),
            }
        };
        let has_mesh = self.meshes.read().contains(&pos);
        (loaded.0, loaded.1, loaded.2, has_mesh)
    }

    /// Batch version: acquires chunks + meshes locks once for an entire minimap grid.
    pub fn chunk_debug_info_batch(
        &self,
        center: ChunkPos,
        half: i32,
    ) -> Vec<(ChunkPos, bool, bool, bool, bool)> {
        let mut result = Vec::with_capacity(((half * 2 + 1) * (half * 2 + 1)) as usize);
        let meshes = self.meshes.read();
        for dx in -half..=half {
            for dz in -half..=half {
                let pos = ChunkPos::new(center.x() + dx, 0, center.z() + dz);
                let (loaded, dirty, palette_mode) = {
                    let chunks = self.chunks.read_shard(pos);
                    match chunks.get(&pos) {
                        Some(c) => (true, c.dirty, c.is_palette_mode()),
                        None => (false, false, false),
                    }
                };
                let has_mesh = meshes.contains(&pos);
                result.push((pos, loaded, dirty, palette_mode, has_mesh));
            }
        }
        result
    }

    /// True if the chunk containing the given world block position is loaded.
    pub fn is_block_loaded(&self, x: i32, y: i32, z: i32) -> bool {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return false;
        }
        self.is_chunk_loaded(block_to_chunk(IVec3::new(x, y, z)))
    }

    // --- block queries -----------------------------------------------------

    /// Get a block by world coordinate. Returns air for unloaded chunks or Y
    /// out of range (so the world looks "empty" where data isn't loaded yet).
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockId {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return BlockId::AIR;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let chunks = self.chunks.read_shard(cp);
        let Some(chunk) = chunks.get(&cp) else {
            return BlockId::AIR;
        };
        let origin = chunk_origin(cp);
        chunk.get(x - origin.x, y - origin.y, z - origin.z)
    }

    /// Get a read reference to the chunks storage for batch access.
    /// The caller must not hold this longer than necessary to avoid blocking writers.
    pub fn chunks_ref(&self) -> &ShardedChunks {
        &self.chunks
    }

    /// Get a reference to the block registry.
    pub fn registry_ref(&self) -> &BlockRegistry {
        &self.reg
    }
    #[inline]
    pub fn get_block_guarded(
        chunks: &std::collections::HashMap<ChunkPos, Chunk>,
        x: i32,
        y: i32,
        z: i32,
    ) -> BlockId {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return BlockId::AIR;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let Some(chunk) = chunks.get(&cp) else {
            return BlockId::AIR;
        };
        let origin = chunk_origin(cp);
        chunk.get(x - origin.x, y - origin.y, z - origin.z)
    }

    /// Check solidity using a pre-acquired chunks read guard.
    #[inline]
    pub fn is_solid_guarded(
        chunks: &std::collections::HashMap<ChunkPos, Chunk>,
        reg: &BlockRegistry,
        x: i32,
        y: i32,
        z: i32,
    ) -> bool {
        let id = Self::get_block_guarded(chunks, x, y, z);
        reg.is_solid(id)
    }

    /// Set a block without any lighting recomputation. Writes the block,
    /// marks the chunk dirty + light_dirty, and returns true on success.
    ///
    /// Use this when placing many blocks during world generation (e.g.
    /// tree leaves spilling across chunk borders) where the per-block
    /// lighting recomputation cost of [`set_block`] would be prohibitive.
    /// Lighting is deferred to the caller — typically the chunk's own
    /// `compute_all` pass or a future remesh.
    pub fn set_block_no_light(&self, x: i32, y: i32, z: i32, id: BlockId) -> bool {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return false;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let origin = chunk_origin(cp);
        let lx = x - origin.x;
        let ly = y - origin.y;
        let lz = z - origin.z;

        let mut chunks = self.chunks.write_shard(cp);
        let Some(chunk) = chunks.get_mut(&cp) else {
            return false;
        };
        chunk.set(lx, ly, lz, id);
        chunk.light_dirty = true;
        true
    }

    /// Set a block by world coordinate. Returns true if its chunk is loaded.
    /// Also recomputes lighting for the affected chunk and its 6 cardinal
    /// neighbours. Neighbours are processed first to scrub cross-chunk light
    /// decreases before the central chunk is recalculated. Water flow remains
    /// caller-driven because water level may not be set yet (bucket placement
    /// sets the block before its level).
    pub fn set_block(&self, x: i32, y: i32, z: i32, id: BlockId) -> bool {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return false;
        }
        match self.set_block_if(x, y, z, id, crate::volume::BlockPredicate::Any) {
            ConditionalBlockWrite::Unloaded => false,
            ConditionalBlockWrite::Skipped => {
                // A loaded no-op still follows set_block's original lighting
                // refresh behavior.
                let cp = block_to_chunk(IVec3::new(x, y, z));
                self.recompute_lighting_around(cp);
                true
            }
            ConditionalBlockWrite::Changed(_) => true,
        }
    }

    /// Conditionally replace a block, evaluating the built-in predicate
    /// against the current value while holding the owning chunk's write lock.
    /// Returns `Skipped` if the value already equals `id` or the predicate
    /// rejects it.
    pub(crate) fn set_block_if(
        &self,
        x: i32,
        y: i32,
        z: i32,
        id: BlockId,
        predicate: crate::volume::BlockPredicate,
    ) -> ConditionalBlockWrite {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return ConditionalBlockWrite::Unloaded;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let origin = chunk_origin(cp);
        let lx = x - origin.x;
        let ly = y - origin.y;
        let lz = z - origin.z;

        let old = {
            let mut chunks = self.chunks.write_shard(cp);
            let Some(chunk) = chunks.get_mut(&cp) else {
                return ConditionalBlockWrite::Unloaded;
            };
            let old = chunk.get(lx, ly, lz);
            if old == id || !predicate.matches(old) {
                return ConditionalBlockWrite::Skipped;
            }
            chunk.set(lx, ly, lz, id);
            // Water-level metadata belongs to the old block value; never let
            // it survive replacement by another block type.
            chunk.set_water_level(lx, ly, lz, 0);
            old
        };
        let pos = IVec3::new(x, y, z);
        let removed_water = self
            .reg
            .id_of("water")
            .is_some_and(|water_id| old == water_id && id != water_id);
        self.source_water.write().remove(&pos);
        self.pending_flow.write().remove(&pos);

        // Re-light the 6 cardinal neighbours FIRST so they pick up the new
        // central-chunk torchlight via `sample_torchlight`/`sample_block`,
        // and then re-light the central chunk so its BFS sees the freshly
        // re-lit boundary cells at their final values. Doing it this order
        // (rather than central-first) means we never double-write torchlight
        // values on the same cell. Without this re-light, a torch that was
        // emitting across a chunk boundary would leave phantom light on the
        // neighbour side: `cross_chunk_update` only ever writes *higher*
        // values, so a remove never sends a decrease signal. The neighbour
        // re-lights are how we "scrub" that ghost light out.
        //
        // Cost: `compute_torchlight` is O(chunk volume) per chunk and the
        // BFS is cheap at 16³. We gate on `is_chunk_loaded` so unloaded
        // neighbour positions are a single hashmap probe (no chunk copy).
        self.recompute_lighting_around(cp);
        self.enqueue_water_neighbours(pos);
        if removed_water {
            self.mark_water_remesh(pos);
            crate::water::rebuild_flow_after_source_removed(self, pos);
        }
        ConditionalBlockWrite::Changed(old)
    }

    /// Recompute lighting for the changed chunk and its loaded cardinal
    /// neighbours. Recomputing the neighbours first lets them sample the new
    /// center value; the final center pass then clears stale light decreases.
    fn recompute_lighting_around(&self, cp: ChunkPos) {
        const CARDINALS: [glam::IVec3; 6] = [
            glam::IVec3::new(1, 0, 0),
            glam::IVec3::new(-1, 0, 0),
            glam::IVec3::new(0, 1, 0),
            glam::IVec3::new(0, -1, 0),
            glam::IVec3::new(0, 0, 1),
            glam::IVec3::new(0, 0, -1),
        ];
        // Vertical bounds guard: skip neighbour chunk positions outside the
        // legal Y chunk range so we don't acquire the world write lock just
        // to discover the chunk is absent.
        for d in CARDINALS {
            let ny = cp.y() + d.y;
            if !(0..voxel_core::MAX_CHUNK_Y).contains(&ny) {
                continue;
            }
            let ncp = ChunkPos::new(cp.x() + d.x, ny, cp.z() + d.z);
            if self.is_chunk_loaded(ncp) {
                self.recompute_lighting_at(ncp);
            }
        }
        // Finally re-light the central chunk itself.
        self.recompute_lighting_at(cp);

        // NOTE: water flow simulation is NOT called here because the water
        // level may not be set yet (bucket places block then sets level).
        // Callers that need flow should enqueue the position (via
        // `place_water` / `remove_water`) and let `tick_water` drive the
        // incremental spread.
    }

    /// Convenience: set a block and report the owning chunk position, if any.
    pub fn set_block_world(&self, pos: BlockPos, id: BlockId) -> Option<ChunkPos> {
        if self.set_block(pos.0.x, pos.0.y, pos.0.z, id) {
            Some(block_to_chunk(pos.0))
        } else {
            None
        }
    }

    /// True if the block at (x,y,z) is collidable per the registry.
    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        let id = self.get_block(x, y, z);
        self.reg.is_solid(id)
    }

    // --- water queries ---------------------------------------------------

    /// Get water level (0–8) at a world coordinate. Returns 0 if not liquid.
    pub fn get_water_level_world(&self, x: i32, y: i32, z: i32) -> u8 {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return 0;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let chunks = self.chunks.read_shard(cp);
        let Some(chunk) = chunks.get(&cp) else {
            return 0;
        };
        let origin = chunk_origin(cp);
        chunk.get_water_level(x - origin.x, y - origin.y, z - origin.z)
    }

    /// Set water level (0–8) at a world coordinate. No-op if chunk not loaded.
    /// Also sets the block to water if level > 0, or air if level == 0.
    pub fn set_water_level_world(&self, x: i32, y: i32, z: i32, level: u8) {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return;
        }
        let Some(water_id) = self.reg.id_of("water") else {
            return;
        };
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let mut chunks = self.chunks.write_shard(cp);
        let Some(chunk) = chunks.get_mut(&cp) else {
            return;
        };
        let origin = chunk_origin(cp);
        let lx = x - origin.x;
        let ly = y - origin.y;
        let lz = z - origin.z;
        let pos = IVec3::new(x, y, z);
        let old_level = chunk.get_water_level(lx, ly, lz);

        if level > 0 {
            let current = chunk.get(lx, ly, lz);
            if current != water_id && current != BlockId::AIR && self.reg.is_liquid(current) {
                return;
            }
            let can_replace = current.is_air() || self.reg.get(current).replaceable;
            if current != water_id && !can_replace {
                return;
            }
            if current != water_id {
                chunk.set(lx, ly, lz, water_id);
            }
            chunk.set_water_level(lx, ly, lz, level.min(8));
            if old_level != level.min(8) || current != water_id {
                self.water_remesh.write().insert(cp);
            }
            return;
        }

        let current = chunk.get(lx, ly, lz);
        if current == water_id {
            chunk.set(lx, ly, lz, BlockId::AIR);
            chunk.set_water_level(lx, ly, lz, 0);
        } else if !self.reg.is_liquid(current) {
            // Clean stale water metadata without ever deleting another liquid.
            chunk.set_water_level(lx, ly, lz, 0);
        }
        if old_level != 0 || current == water_id {
            self.water_remesh.write().insert(cp);
        }
        drop(chunks);
        self.source_water.write().remove(&pos);
        self.pending_flow.write().remove(&pos);
        self.enqueue_water_neighbours(pos);
    }

    /// True if the block at (x,y,z) is a water source (liquid with level 8).
    /// Uses the source index for O(1) lookup.
    pub fn is_water_source(&self, x: i32, y: i32, z: i32) -> bool {
        self.is_known_water_source(x, y, z)
    }

    /// True if the block at (x,y,z) is any liquid.
    pub fn is_liquid(&self, x: i32, y: i32, z: i32) -> bool {
        let id = self.get_block(x, y, z);
        self.reg.is_liquid(id)
    }

    /// Get the set of all water source positions (read access).
    pub fn water_sources(&self) -> &RwLock<HashSet<IVec3>> {
        &self.source_water
    }

    /// Check if a position is a known water source (O(1) via index).
    pub fn is_known_water_source(&self, x: i32, y: i32, z: i32) -> bool {
        self.source_water.read().contains(&IVec3::new(x, y, z))
    }

    /// Remove a water source block at (x,y,z). Uses `set_block` to set air
    /// so lighting is properly recalculated, then enqueues neighbouring water
    /// positions so the simulation resumes on the next tick. Returns true if
    /// removed.
    pub fn remove_water(&self, x: i32, y: i32, z: i32) -> bool {
        let Some(water_id) = self.reg.id_of("water") else {
            return false;
        };
        if !self.is_water_source(x, y, z) || self.get_block(x, y, z) != water_id {
            return false;
        }
        // Use set_block to remove water so lighting and water indexes update.
        self.set_block(x, y, z, BlockId::AIR)
    }

    /// Place a water source block at (x,y,z). Sets the block + water level 8
    /// and enqueues the position for the simulation. Spread happens over
    /// subsequent ticks via `tick_water` (not synchronously here).
    pub fn place_water(&self, x: i32, y: i32, z: i32) -> bool {
        let water_id = match self.reg.id_of("water") {
            Some(id) => id,
            None => return false,
        };
        let current = self.get_block(x, y, z);
        if !current.is_air()
            && current != water_id
            && (!self.reg.get(current).replaceable || self.reg.is_liquid(current))
        {
            return false;
        }
        if !self.set_block(x, y, z, water_id) {
            return false;
        }
        // Set water level to 8 (source) after the block is placed.
        self.set_water_level_world(x, y, z, 8);
        // Enqueue the source for incremental flow on the next tick.
        self.pending_flow.write().insert(IVec3::new(x, y, z));
        // Track the source position in the O(1) index for fast lookups.
        self.source_water.write().insert(IVec3::new(x, y, z));
        true
    }

    /// Advance the water simulation by `dt` seconds. When the internal
    /// accumulator crosses `WATER_TICK_INTERVAL`, runs one flow step and
    /// returns the chunks modified (so the caller can request remeshes).
    /// Lighting is intentionally NOT recomputed here — water level changes
    /// don't meaningfully alter light absorption, and the cost would add up
    /// for large water regions.
    pub fn tick_water(&self, dt: f32) -> HashSet<ChunkPos> {
        {
            let mut acc = self.water_tick_accumulator.write();
            *acc += dt;
            if *acc < WATER_TICK_INTERVAL {
                return HashSet::new();
            }
            *acc -= WATER_TICK_INTERVAL;
            // Clamp to avoid runaway after long pauses.
            if *acc > WATER_TICK_INTERVAL {
                *acc = WATER_TICK_INTERVAL;
            }
        }
        // Copy-then-swap: take the pending set out from under the lock so the
        // simulation step doesn't block `place_water`/`remove_water` (which
        // briefly lock `pending_flow`), then merge the leftovers back in.
        // Positions added mid-step land in the live set and run next tick;
        // duplicates are harmless (the sim re-checks actual water levels).
        let mut pending = std::mem::take(&mut *self.pending_flow.write());
        let mut buf = self.water_sim_buf.write();
        let mut affected = crate::water::simulate_flow_step(self, &mut pending, &mut buf);
        drop(buf);
        self.pending_flow.write().extend(pending);
        affected.extend(std::mem::take(&mut *self.water_remesh.write()));
        affected
    }

    // --- meshing support ---------------------------------------------------

    /// Run a closure with read access to a chunk and a neighbour-sampling
    /// function that crosses chunk borders (used by the mesher on worker
    /// threads). The samplers read through the shared `RwLock`. The water
    /// sampler returns the level (0-8) at the given world coordinate, or 0
    /// for non-water / unloaded positions; it is used by the mesher to fill
    /// the "step" between adjacent water layers at different levels. The
    /// loaded sampler returns true when the chunk at the given world
    /// coordinate is loaded; the mesher uses it to decide whether to apply
    /// the chunk-border face-ownership rule (which prevents Z-fighting)
    /// without leaving holes at the edge of the loaded area.
    pub fn with_chunk_for_mesh<R>(
        &self,
        pos: ChunkPos,
        f: impl FnOnce(
            &Chunk,
            &dyn Fn(i32, i32, i32) -> BlockId,
            &dyn Fn(i32, i32, i32) -> u8,
            &dyn Fn(i32, i32, i32) -> bool,
        ) -> R,
    ) -> R {
        // Clone the target chunk out from under the lock so meshing is lock-free;
        // neighbours are sampled live (cheap read lock per sample is acceptable).
        let chunk = {
            let chunks = self.chunks.read_shard(pos);
            chunks.get(&pos).cloned()
        };

        // Upgrade the self-reference once per call; the samplers below all
        // borrow the resulting `Arc<World>` (shared read access) for the
        // duration of the synchronous call.
        let world_arc = self
            .self_ref
            .upgrade()
            .expect("`World` outlives `with_chunk_for_mesh` callers");

        let Some(chunk) = chunk else {
            // No chunk: produce an empty result by giving the closure an empty
            // chunk. This path should be rare (mesh requested for unloaded chunk).
            let empty = Chunk::new(pos);
            let sample: &dyn Fn(i32, i32, i32) -> BlockId = &|_, _, _| BlockId::AIR;
            let sample_water: &dyn Fn(i32, i32, i32) -> u8 = &|_, _, _| 0;
            let sample_loaded: &dyn Fn(i32, i32, i32) -> bool = &|_, _, _| false;
            return f(&empty, sample, sample_water, sample_loaded);
        };

        // Plain stack closures borrowing a single shared `Arc<World>` — no
        // per-mesh heap allocations or `Arc` clones. The call is synchronous,
        // so the closures don't need to outlive it.
        let sample = |x: i32, y: i32, z: i32| world_arc.get_block(x, y, z);
        let sample_water = |x: i32, y: i32, z: i32| world_arc.get_water_level_world(x, y, z);
        let sample_loaded = |x: i32, y: i32, z: i32| world_arc.is_block_loaded(x, y, z);
        f(&chunk, &sample, &sample_water, &sample_loaded)
    }

    /// Snapshot a chunk as an 18³ `u16` BlockId volume for the GPU compute
    /// mesher: the 16³ chunk interior plus a 1-voxel border from the 26
    /// neighbouring chunks. Output layout is z-major, then y, then x so it
    /// matches the `cmd_copy_buffer_to_image` upload (`buffer_row_length =
    /// buffer_image_height = 18`): index `(z * 18 + y) * 18 + x` holds the
    /// block at chunk-local `(x - 1, y - 1, z - 1)`. Border voxels (coords
    /// 0 or 17) read the neighbour block, falling back to air when the
    /// neighbour chunk is unloaded or the coordinate is outside world Y.
    pub fn gpu_mesh_voxels(&self, pos: ChunkPos) -> Box<[u16]> {
        const N: usize = 18;
        let origin = chunk_origin(pos);
        let mut out = vec![0u16; N * N * N];
        for oz in 0..N {
            for oy in 0..N {
                for ox in 0..N {
                    let wx = origin.x + ox as i32 - 1;
                    let wy = origin.y + oy as i32 - 1;
                    let wz = origin.z + oz as i32 - 1;
                    out[(oz * N + oy) * N + ox] = self.get_block(wx, wy, wz).0;
                }
            }
        }
        out.into_boxed_slice()
    }
}

/// Six face-adjacent neighbours of `pos`.
fn water_neighbours(pos: IVec3) -> [IVec3; 6] {
    [
        pos + IVec3::new(1, 0, 0),
        pos + IVec3::new(-1, 0, 0),
        pos + IVec3::new(0, 1, 0),
        pos + IVec3::new(0, -1, 0),
        pos + IVec3::new(0, 0, 1),
        pos + IVec3::new(0, 0, -1),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;
    use voxel_core::BlockId;

    #[test]
    fn set_block_returns_false_for_unloaded_chunk() {
        let world = World::new(42);
        // No chunks loaded.
        assert!(!world.set_block(0, 0, 0, BlockId(1)));
    }

    #[test]
    fn set_block_loaded_same_value_reports_success_without_changing_value() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        chunk.set(0, 0, 0, BlockId(2));
        world.insert_chunk(cp, chunk);

        assert!(world.set_block(0, 0, 0, BlockId(2)));
        assert_eq!(world.get_block(0, 0, 0), BlockId(2));
    }

    #[test]
    fn get_block_returns_air_for_unloaded() {
        let world = World::new(42);
        assert_eq!(world.get_block(0, 0, 0), BlockId::AIR);
    }

    #[test]
    fn get_block_out_of_y_range_returns_air() {
        let world = World::new(42);
        assert_eq!(world.get_block(0, 1000, 0), BlockId::AIR);
        assert_eq!(world.get_block(0, -1, 0), BlockId::AIR);
    }

    #[test]
    fn is_block_loaded_false_for_unloaded() {
        let world = World::new(42);
        assert!(!world.is_block_loaded(0, 0, 0));
    }

    #[test]
    fn insert_and_remove_chunk() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let chunk = Chunk::new(cp);
        world.insert_chunk(cp, chunk);
        assert!(world.is_chunk_loaded(cp));
        assert_eq!(world.loaded_chunk_count(), 1);
        world.remove_chunk(cp);
        assert!(!world.is_chunk_loaded(cp));
        assert_eq!(world.loaded_chunk_count(), 0);
    }

    #[test]
    fn set_block_loaded_chunk() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let chunk = Chunk::new(cp);
        world.insert_chunk(cp, chunk);
        assert!(world.set_block(0, 0, 0, BlockId(2)));
        assert_eq!(world.get_block(0, 0, 0), BlockId(2));
    }

    #[test]
    fn is_solid_uses_registry() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        let stone = world.registry().id_of("stone").unwrap();
        chunk.set(0, 0, 0, stone);
        world.insert_chunk(cp, chunk);
        assert!(world.is_solid(0, 0, 0));
        assert!(!world.is_solid(1, 0, 0));
    }

    #[test]
    fn get_torchlight_unloaded_returns_zero() {
        let world = World::new(42);
        assert_eq!(world.get_torchlight_world(0, 0, 0), 0);
    }

    #[test]
    fn set_torchlight_noop_for_unloaded() {
        let world = World::new(42);
        world.set_torchlight_world(0, 0, 0, 10);
        // No panic, no effect.
        assert_eq!(world.get_torchlight_world(0, 0, 0), 0);
    }

    #[test]
    fn chunk_debug_info_unloaded() {
        let world = World::new(42);
        let info = world.chunk_debug_info(ChunkPos::new(0, 0, 0));
        assert!(!info.0);
    }

    #[test]
    fn all_loaded_chunks_roundtrip() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        chunk.set(0, 0, 0, BlockId(5));
        world.insert_chunk(cp, chunk);
        let chunks = world.all_loaded_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, cp);
        assert_eq!(chunks[0].1.get(0, 0, 0), BlockId(5));
    }

    #[test]
    fn inserted_chunk_registers_water_sources_and_replacing_chunk_clears_them() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let water = world.registry().id_of("water").unwrap();
        let mut chunk = Chunk::new(cp);
        chunk.set(2, 3, 4, water);
        chunk.set_water_level(2, 3, 4, 8);
        world.insert_chunk(cp, chunk);
        assert!(world.is_known_water_source(2, 3, 4));
        assert_eq!(world.pending_flow_count(), 1);

        world.insert_chunk(cp, Chunk::new(cp));
        assert!(!world.is_known_water_source(2, 3, 4));
        assert_eq!(world.pending_flow_count(), 0);
    }

    #[test]
    fn water_submersion_uses_partial_fluid_height() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, Chunk::new(cp));
        world.set_water_level_world(0, 1, 0, 4);

        let submerged = world.water_submersion(
            glam::Vec3::new(0.2, 1.0, 0.2),
            glam::Vec3::new(0.8, 1.9, 0.8),
        );
        assert!((submerged - (0.5 / 0.9)).abs() < 0.01);
        assert_eq!(
            world.water_submersion(
                glam::Vec3::new(0.2, 1.6, 0.2),
                glam::Vec3::new(0.8, 2.0, 0.8),
            ),
            0.0
        );
    }

    #[test]
    fn water_source_index_tracking() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, Chunk::new(cp));

        // Place water — should add to index.
        world.place_water(1, 10, 1);
        assert!(world.is_known_water_source(1, 10, 1));
        assert_eq!(world.water_sources().read().len(), 1);

        // Remove water — should remove from index.
        world.remove_water(1, 10, 1);
        assert!(!world.is_known_water_source(1, 10, 1));
        assert_eq!(world.water_sources().read().len(), 0);
    }

    #[test]
    fn water_source_index_multiple() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, Chunk::new(cp));

        world.place_water(0, 5, 0);
        world.place_water(1, 5, 0);
        world.place_water(2, 5, 0);
        assert_eq!(world.water_sources().read().len(), 3);
        assert!(world.is_known_water_source(0, 5, 0));
        assert!(world.is_known_water_source(1, 5, 0));
        assert!(world.is_known_water_source(2, 5, 0));

        // Remove one — index should shrink to 2.
        world.remove_water(1, 5, 0);
        assert_eq!(world.water_sources().read().len(), 2);
        assert!(!world.is_known_water_source(1, 5, 0));
        assert!(world.is_known_water_source(0, 5, 0));
        assert!(world.is_known_water_source(2, 5, 0));
    }

    #[test]
    fn water_source_index_idempotent_place() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, Chunk::new(cp));

        // Place water twice at the same position — index should still have 1.
        world.place_water(0, 5, 0);
        world.place_water(0, 5, 0);
        assert_eq!(world.water_sources().read().len(), 1);
    }
}
