//! `World` — the shared facade over chunk storage, block registry, terrain
//! generation, and chunk meshes. Safe to share across threads via `Arc`; all
//! mutation goes through internal `RwLock`s.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};

use parking_lot::RwLock;

use glam::{IVec2, IVec3};
use voxel_core::{
    math::{block_to_chunk, chunk_origin, ChunkPos},
    BlockId, BlockPos, CHUNK_SIZE, WORLD_HEIGHT_BLOCKS,
};

use crate::schematic::{SchematicEntity, SchematicId};
use crate::water::{
    column_from_meta, infer_kind, WaterColumn, WaterColumnMeta, WaterKind, WaterSample,
    WaterVertexInfo, MAX_FLOW_SPEED,
};
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

/// Two water runs closer than this are contiguous slices of the same body.
const WATER_BODY_EPS: f32 = 0.05;

/// Sort bodies from highest surface to lowest and merge contiguous or
/// overlapping runs. The top run's body metadata wins, which preserves the
/// lake/ocean classification when a lower chunk supplies the rest of a deep
/// body. Separated runs stay independent cave aquifers.
fn normalize_water_bodies(mut bodies: Vec<WaterColumn>) -> Vec<WaterColumn> {
    bodies.sort_by(|a, b| b.surface_y.total_cmp(&a.surface_y));
    let mut out: Vec<WaterColumn> = Vec::with_capacity(bodies.len());
    for body in bodies {
        if let Some(last) = out.last_mut() {
            if body.surface_y >= last.bed_y - WATER_BODY_EPS {
                last.bed_y = last.bed_y.min(body.bed_y);
                continue;
            }
        }
        out.push(body);
    }
    out
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
    /// Live water bodies keyed by world `(x, z)`. Each entry contains every
    /// disjoint body in that vertical column, sorted from highest surface to
    /// lowest. Keeping all bodies (not just the top one) is what makes a cave
    /// aquifer below a lake/ocean visible to physics.
    water_columns: RwLock<HashMap<IVec2, Vec<WaterColumn>>>,
    /// Accumulated time driving the analytic wave/current field. Advanced by
    /// `tick_water` and sampled by both physics and the shader.
    water_time: RwLock<f32>,
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
            water_columns: RwLock::new(HashMap::new()),
            water_time: RwLock::new(0.0),
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

    // --- chunk storage -----------------------------------------------------

    pub fn insert_chunk(&self, pos: ChunkPos, mut chunk: Chunk) {
        chunk.pos = pos;
        let columns = self.scan_chunk_water(pos, &chunk);
        let replaced = self.chunks.write_shard(pos).insert(pos, chunk).is_some();

        if replaced {
            // A replaced chunk can change any body in these columns. Rescan
            // every affected column across all loaded vertical chunks instead
            // of trusting the stale map entry.
            let origin = chunk_origin(pos);
            for lx in 0..CHUNK_SIZE {
                for lz in 0..CHUNK_SIZE {
                    self.refresh_water_column(origin.x + lx, origin.z + lz);
                }
            }
            return;
        }

        // Register every disjoint body in the chunk. Contiguous slices of the
        // same body loaded in adjacent vertical chunks are merged; separated
        // cave pools stay as additional bodies instead of being dropped.
        if !columns.is_empty() {
            let mut map = self.water_columns.write();
            for (key, column) in columns {
                let bodies = map.entry(key).or_default();
                bodies.push(column);
                *bodies = normalize_water_bodies(std::mem::take(bodies));
            }
        }
    }

    /// Fraction of an axis-aligned box occupied by water. The topmost water
    /// voxel in a column uses the body's continuous surface height (and its
    /// analytic wave field) instead of a flow level, so shallow water and
    /// wave troughs affect physics only when the box actually intersects the
    /// moving surface.
    pub fn water_submersion(&self, min: glam::Vec3, max: glam::Vec3) -> f32 {
        let size = max - min;
        let volume = size.x * size.y * size.z;
        if !volume.is_finite() || volume <= 0.0 {
            return 0.0;
        }
        let min_block = voxel_core::math::world_to_block(min);
        let max_block = voxel_core::math::world_to_block(max - glam::Vec3::splat(0.001));
        let time = self.water_time();
        let columns = self.water_columns.read();
        let mut submerged_volume = 0.0;
        // Integrate every body in each horizontal column. A wave crest above
        // the top water voxel still counts, and a cave aquifer below a lake or
        // ocean contributes separately instead of being hidden by the top
        // body's bed.
        for bz in min_block.z..=max_block.z {
            for bx in min_block.x..=max_block.x {
                let Some(bodies) = columns.get(&IVec2::new(bx, bz)) else {
                    continue;
                };
                let overlap_x = (max.x.min(bx as f32 + 1.0) - min.x.max(bx as f32)).max(0.0);
                let overlap_z = (max.z.min(bz as f32 + 1.0) - min.z.max(bz as f32)).max(0.0);
                for column in bodies {
                    let surface = column.surface_at(bx as f32 + 0.5, bz as f32 + 0.5, time);
                    let overlap_y = (max.y.min(surface) - min.y.max(column.bed_y)).max(0.0);
                    submerged_volume += overlap_x * overlap_y * overlap_z;
                }
            }
        }
        (submerged_volume / volume).clamp(0.0, 1.0)
    }

    pub fn remove_chunk(&self, pos: ChunkPos) {
        self.chunks.write_shard(pos).remove(&pos);
        self.meshes.write().remove(&pos);

        // Removing a slice can expose or shorten any body in these columns
        // (a cave pool may sit below an ocean surface). Rebuild all 256
        // horizontal columns; refresh_water_column scans only the loaded
        // vertical chunks at each x/z.
        let origin = chunk_origin(pos);
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                self.refresh_water_column(origin.x + lx, origin.z + lz);
            }
        }
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

    /// Number of live water-surface columns across loaded chunks.
    pub fn water_column_count(&self) -> usize {
        self.water_columns.read().len()
    }

    /// True if a chunk contains any water voxel. Used by the streamer to keep
    /// animated water on the CPU mesher instead of the naive GPU mesher.
    pub fn chunk_has_water(&self, pos: ChunkPos) -> bool {
        let Some(water_id) = self.reg.id_of("water") else {
            return false;
        };
        let chunks = self.chunks.read_shard(pos);
        let Some(chunk) = chunks.get(&pos) else {
            return false;
        };
        for ly in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    if chunk.get(lx, ly, lz) == water_id {
                        return true;
                    }
                }
            }
        }
        false
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

    /// Set a block without extra bookkeeping. Returns true on success.
    ///
    /// Used by world generation when placing many blocks (e.g. tree leaves
    /// spilling across chunk borders).
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
        true
    }

    /// Set a block by world coordinate. Returns true if its chunk is loaded.
    /// Editing water (or ground under it) refreshes that column's registered
    /// surface immediately; there is no flow solver to tick.
    pub fn set_block(&self, x: i32, y: i32, z: i32, id: BlockId) -> bool {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return false;
        }
        match self.set_block_if(x, y, z, id, crate::volume::BlockPredicate::Any) {
            ConditionalBlockWrite::Unloaded => false,
            ConditionalBlockWrite::Skipped => true,
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
            old
        };

        // Water metadata is per column: recompute the registered surface when
        // the edit touches water or an existing water body. This keeps the
        // physics/render queries in sync without a BFS flow pass.
        let water_related = self
            .reg
            .id_of("water")
            .is_some_and(|water_id| old == water_id || id == water_id)
            || self.water_column_at(x, z).is_some();
        if water_related {
            self.refresh_water_column(x, z);
        }
        ConditionalBlockWrite::Changed(old)
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

    /// Current water-field time in seconds (drives waves and currents).
    #[inline]
    pub fn water_time(&self) -> f32 {
        *self.water_time.read()
    }

    /// Advance the analytic water field by `dt`. There is no per-voxel flow
    /// solver anymore: waves and currents are evaluated from the body state,
    /// so this only advances the clock. Returns the set of chunks that need a
    /// remesh (always empty; the API is kept for the engine's fixed-step loop).
    pub fn tick_water(&self, dt: f32) -> HashSet<ChunkPos> {
        if dt.is_finite() && dt > 0.0 {
            let mut t = self.water_time.write();
            *t = (*t + dt) % 100_000.0;
        }
        HashSet::new()
    }

    /// Highest water body in the world column at `(x, z)`.
    pub fn water_column_at(&self, x: i32, z: i32) -> Option<WaterColumn> {
        self.water_bodies_at(x, z).into_iter().next()
    }

    /// Every disjoint water body in the column, sorted highest surface first.
    pub fn water_bodies_at(&self, x: i32, z: i32) -> Vec<WaterColumn> {
        self.water_columns
            .read()
            .get(&IVec2::new(x, z))
            .cloned()
            .unwrap_or_default()
    }

    /// Fill a row-major `size x size` grid of local water bodies for the
    /// chunk shader's caustics / wet-edge shader path.
    ///
    /// Each cell contains [`crate::water::WATER_SURFACE_LAYERS`] layers of
    /// `(surface, bed)` pairs. Positive surface values are real water bodies;
    /// negative surfaces mark dry wet-edge fringe cells (the bed stores the
    /// fringe height). Layers are ordered highest surface first, so a cave
    /// aquifer below a lake/ocean is still present in the second layer.
    ///
    /// `out` must contain at least `size * size *
    /// WATER_SURFACE_FLOATS_PER_CELL` entries.
    pub fn fill_water_surface_grid(
        &self,
        origin_x: i32,
        origin_z: i32,
        size: usize,
        out: &mut [f32],
    ) {
        use crate::water::{WATER_SURFACE_FLOATS_PER_CELL, WATER_SURFACE_LAYERS, WATER_SURFACE_NONE};
        let cells = size.saturating_mul(size);
        let floats = cells.saturating_mul(WATER_SURFACE_FLOATS_PER_CELL);
        if size == 0 || out.len() < floats {
            return;
        }
        out[..floats].fill(WATER_SURFACE_NONE);

        {
            let columns = self.water_columns.read();
            for lz in 0..size {
                for lx in 0..size {
                    let Some(bodies) =
                        columns.get(&IVec2::new(origin_x + lx as i32, origin_z + lz as i32))
                    else {
                        continue;
                    };
                    let base = (lz * size + lx) * WATER_SURFACE_FLOATS_PER_CELL;
                    for (layer, body) in bodies.iter().take(WATER_SURFACE_LAYERS).enumerate() {
                        out[base + layer * 2] = body.surface_y.max(0.01);
                        out[base + layer * 2 + 1] = body.bed_y;
                    }
                }
            }
        }

        // Wet-edge fringe: the first two dry blocks around a shoreline get
        // the neighbouring water height (negative = no caustics). Process
        // radius 1 before radius 2 so the nearest body wins on mixed shores.
        for radius in 1..=2i32 {
            for lz in 0..size {
                for lx in 0..size {
                    let base = (lz * size + lx) * WATER_SURFACE_FLOATS_PER_CELL;
                    for layer in 0..WATER_SURFACE_LAYERS {
                        let surface = out[base + layer * 2];
                        if surface <= 0.0 {
                            continue;
                        }
                        for dz in -radius..=radius {
                            for dx in -radius..=radius {
                                if dx.abs().max(dz.abs()) != radius {
                                    continue;
                                }
                                let nx = lx as i32 + dx;
                                let nz = lz as i32 + dz;
                                if nx < 0 || nz < 0 || nx >= size as i32 || nz >= size as i32 {
                                    continue;
                                }
                                let n = (nz as usize * size + nx as usize)
                                    * WATER_SURFACE_FLOATS_PER_CELL;
                                // Only dry cells get a wet fringe. A cave body
                                // can occupy the top layer under dry land; it
                                // is still real water, so leave it alone.
                                if out[n] > 0.0 {
                                    continue;
                                }
                                // Fill the first free layer; never overwrite a
                                // real water body.
                                for target in 0..WATER_SURFACE_LAYERS {
                                    if out[n + target * 2] <= WATER_SURFACE_NONE + 1.0 {
                                        out[n + target * 2] = -surface;
                                        out[n + target * 2 + 1] = surface;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Sample the water body at a world position. Returns `None` when the
    /// point is not inside water. Waves move the sampled surface, so this is
    /// the correct query for buoyancy, swimming and item drift.
    pub fn water_sample(&self, pos: glam::Vec3) -> Option<WaterSample> {
        let block = voxel_core::math::world_to_block(pos);
        // Query the registered bodies rather than the voxel under `pos`: a
        // wave crest can rise above its top water voxel, and a cave aquifer
        // can be hidden beneath a lake/ocean body in the same column.
        let bodies = self.water_bodies_at(block.x, block.z);
        let time = self.water_time();
        for column in bodies {
            let wave = crate::water::wave_height(
                column.kind,
                column.wave_amplitude(),
                column.flow_angle,
                column.rapids,
                pos.x,
                pos.z,
                time,
            );
            let surface = column.surface_y + wave;
            if pos.y > surface + 0.001 || pos.y < column.bed_y - 0.001 {
                continue;
            }
            return Some(WaterSample {
                kind: column.kind,
                surface_y: surface,
                bed_y: column.bed_y,
                depth: column.depth(),
                flow: column.flow(),
                flow_speed: column.flow_speed,
                rapids: column.rapids,
                wave,
            });
        }
        None
    }

    /// True if the block at (x,y,z) is any liquid.
    pub fn is_liquid(&self, x: i32, y: i32, z: i32) -> bool {
        let id = self.get_block(x, y, z);
        self.reg.is_liquid(id)
    }

    /// Render-side description of a liquid voxel's top/side surface.
    /// Non-water liquids (lava) get a full-height surface with no animation.
    pub fn water_vertex_info(&self, x: i32, y: i32, z: i32) -> WaterVertexInfo {
        let id = self.get_block(x, y, z);
        if id.is_air() {
            return WaterVertexInfo::default();
        }
        let Some(water_id) = self.reg.id_of("water") else {
            return WaterVertexInfo::default();
        };
        if id != water_id {
            return WaterVertexInfo {
                fill: 255,
                surface: true,
                ..Default::default()
            };
        }

        let bodies = self.water_bodies_at(x, z);

        // Cells under another water cell are always full. Pick the body that
        // owns this cell so a cave aquifer does not inherit the lake's kind.
        if self.get_block(x, y + 1, z) == water_id {
            let body = bodies
                .iter()
                .find(|c| y as f32 >= c.bed_y - 0.001 && (y as f32) < c.surface_y - 0.001);
            return WaterVertexInfo {
                kind: body.map(|c| c.kind.as_u8()).unwrap_or(0),
                fill: 255,
                surface: false,
                ..Default::default()
            };
        }

        if bodies.is_empty() {
            return WaterVertexInfo {
                fill: 255,
                surface: true,
                ..Default::default()
            };
        }

        // The top voxel of each registered body gets that body's continuous
        // fill + wave animation. A lower, disconnected pool (a cave aquifer)
        // is its own still body rather than a fabricated aquifer fallback.
        let Some(column) = bodies
            .iter()
            .find(|c| (c.surface_y - 0.001).floor() as i32 == y)
        else {
            return WaterVertexInfo {
                fill: 255,
                surface: true,
                ..Default::default()
            };
        };

        WaterVertexInfo {
            kind: column.kind.as_u8(),
            fill: (column.fill * 255.0).round().clamp(0.0, 255.0) as u8,
            amplitude: (column.wave_amplitude() * 32.0).clamp(0.0, 255.0) as u8,
            flow_angle: crate::water::pack_angle(column.flow_angle),
            flow_speed: crate::water::pack_unit(column.flow_speed / MAX_FLOW_SPEED),
            rapids: crate::water::pack_unit(column.rapids),
            surface: true,
        }
    }

    /// Scan every disjoint water run in one horizontal column of a chunk.
    ///
    /// The chunk stores one metadata record per column, which describes the
    /// topmost run (generation never overwrites an existing record with a
    /// cave pool). Lower runs are disconnected cave aquifers: they are still,
    /// full-height and are always classified as `Aquifer`.
    fn scan_column_bodies_in_chunk(&self, chunk: &Chunk, lx: i32, lz: i32) -> Vec<WaterColumn> {
        let Some(water_id) = self.reg.id_of("water") else {
            return Vec::new();
        };
        let origin = chunk_origin(chunk.pos);
        let meta = chunk.water_meta(lx, lz);
        let mut bodies = Vec::new();
        let mut ly = CHUNK_SIZE as i32 - 1;
        while ly >= 0 {
            if chunk.get(lx, ly, lz) != water_id {
                ly -= 1;
                continue;
            }
            let top_ly = ly;
            while ly > 0 && chunk.get(lx, ly - 1, lz) == water_id {
                ly -= 1;
            }
            let bed_ly = ly;
            let column = if bodies.is_empty() {
                let fill_u8 = if meta.fill == 0 { 255 } else { meta.fill };
                let fill = fill_u8 as f32 / 255.0;
                let surface_y = origin.y as f32 + top_ly as f32 + fill;
                let mut column =
                    column_from_meta(meta, surface_y, origin.y as f32 + bed_ly as f32, fill);
                if !meta.is_some() || column.kind == WaterKind::None {
                    column.kind = infer_kind(surface_y);
                }
                column
            } else {
                WaterColumn {
                    kind: WaterKind::Aquifer,
                    surface_y: (origin.y + top_ly + 1) as f32,
                    bed_y: (origin.y + bed_ly) as f32,
                    fill: 1.0,
                    ..Default::default()
                }
            };
            bodies.push(column);
            ly = bed_ly - 1;
        }
        bodies
    }

    /// All water bodies contained in a chunk (used when registering a newly
    /// loaded chunk).
    fn scan_chunk_water(&self, pos: ChunkPos, chunk: &Chunk) -> Vec<(IVec2, WaterColumn)> {
        let origin = chunk_origin(pos);
        let mut out = Vec::new();
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                for column in self.scan_column_bodies_in_chunk(chunk, lx, lz) {
                    out.push((IVec2::new(origin.x + lx, origin.z + lz), column));
                }
            }
        }
        out
    }

    /// Recompute every water body in one world column by scanning all loaded
    /// vertical chunks at that `(x, z)`. Used after block edits and unloads.
    pub fn refresh_water_column(&self, x: i32, z: i32) {
        let cx = x.div_euclid(CHUNK_SIZE);
        let cz = z.div_euclid(CHUNK_SIZE);
        let mut found = Vec::new();
        for cy in 0..(WORLD_HEIGHT_BLOCKS / CHUNK_SIZE) {
            let cp = ChunkPos::new(cx, cy, cz);
            let bodies = {
                let chunks = self.chunks.read_shard(cp);
                chunks.get(&cp).map_or_else(Vec::new, |chunk| {
                    let origin = chunk_origin(cp);
                    self.scan_column_bodies_in_chunk(chunk, x - origin.x, z - origin.z)
                })
            };
            found.extend(bodies);
        }
        let bodies = normalize_water_bodies(found);
        let mut map = self.water_columns.write();
        let key = IVec2::new(x, z);
        if bodies.is_empty() {
            map.remove(&key);
        } else {
            map.insert(key, bodies);
        }
    }

    /// Fill a single water column from `bottom` up to `surface` and write the
    /// body metadata onto its top chunk. Returns true if anything changed.
    fn fill_water_column(
        &self,
        x: i32,
        z: i32,
        bottom: i32,
        surface: f32,
        kind: WaterKind,
        flow_angle: f32,
        flow_speed: f32,
        rapids: f32,
    ) -> bool {
        let Some(water_id) = self.reg.id_of("water") else {
            return false;
        };
        let top_cell = (surface - 0.001).floor() as i32;
        if top_cell < bottom {
            return false;
        }
        let mut changed = false;
        for y in bottom..=top_cell {
            let cp = block_to_chunk(IVec3::new(x, y, z));
            let mut chunks = self.chunks.write_shard(cp);
            let Some(chunk) = chunks.get_mut(&cp) else {
                continue;
            };
            let origin = chunk_origin(cp);
            let (lx, ly, lz) = (x - origin.x, y - origin.y, z - origin.z);
            let current = chunk.get(lx, ly, lz);
            if current == water_id {
                changed = true;
                continue;
            }
            if current.is_air() || self.reg.get(current).replaceable {
                chunk.set(lx, ly, lz, water_id);
                changed = true;
            } else {
                break;
            }
        }
        if !changed {
            return false;
        }

        let cp = block_to_chunk(IVec3::new(x, top_cell, z));
        let fill = (surface - top_cell as f32).clamp(0.02, 1.0);
        let meta = WaterColumnMeta::new(
            kind,
            (fill * 255.0).round() as u8,
            crate::water::pack_angle(flow_angle),
            crate::water::pack_unit(flow_speed / MAX_FLOW_SPEED),
            crate::water::pack_unit(rapids),
        );
        {
            let mut chunks = self.chunks.write_shard(cp);
            if let Some(chunk) = chunks.get_mut(&cp) {
                let origin = chunk_origin(cp);
                chunk.set_water_meta(x - origin.x, z - origin.z, meta);
            }
        }
        self.refresh_water_column(x, z);
        true
    }

    /// Remove a water voxel. Returns true if a water block was removed.
    pub fn remove_water(&self, x: i32, y: i32, z: i32) -> bool {
        let Some(water_id) = self.reg.id_of("water") else {
            return false;
        };
        if self.get_block(x, y, z) != water_id {
            return false;
        }
        self.set_block(x, y, z, BlockId::AIR)
    }

    /// Place water at `(x,y,z)`. The water falls to the first solid floor and
    /// fills a small puddle at the requested level; it is a still pond body.
    pub fn place_water(&self, x: i32, y: i32, z: i32) -> bool {
        let Some(water_id) = self.reg.id_of("water") else {
            return false;
        };
        let current = self.get_block(x, y, z);
        if !current.is_air()
            && current != water_id
            && (!self.reg.get(current).replaceable || self.reg.is_liquid(current))
        {
            return false;
        }
        let mut bottom = 0;
        for gy in (0..=y).rev() {
            if self.reg.is_solid(self.get_block(x, gy, z)) {
                bottom = gy + 1;
                break;
            }
        }
        let surface = y as f32 + 1.0;
        let mut placed = self.fill_water_column(
            x,
            z,
            bottom,
            surface,
            WaterKind::Pond,
            0.0,
            0.0,
            0.0,
        );
        // Spread one ring so a bucket on flat ground forms a visible puddle
        // rather than a one-block pillar.
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let nx = x + dx;
            let nz = z + dz;
            if self.get_block(nx, y, nz).is_air() && self.is_solid(nx, y - 1, nz) {
                placed |= self.fill_water_column(
                    nx,
                    nz,
                    y,
                    surface,
                    WaterKind::Pond,
                    0.0,
                    0.0,
                    0.0,
                );
            }
        }
        placed
    }

    // --- meshing support ---------------------------------------------------

    /// Run a closure with read access to a chunk and a neighbour-sampling
    /// function that crosses chunk borders (used by the mesher on worker
    /// threads). The samplers read through the shared `RwLock`. The water
    /// sampler returns the continuous surface fill, body kind, wave amplitude
    /// and flow data for a liquid voxel; the loaded sampler returns true when
    /// the chunk at the given world coordinate is loaded (the mesher uses it
    /// to decide whether to apply the chunk-border face-ownership rule without
    /// leaving holes at the edge of the loaded area).
    pub fn with_chunk_for_mesh<R>(
        &self,
        pos: ChunkPos,
        f: impl FnOnce(
            &Chunk,
            &dyn Fn(i32, i32, i32) -> BlockId,
            &dyn Fn(i32, i32, i32) -> WaterVertexInfo,
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
            let sample_water: &dyn Fn(i32, i32, i32) -> WaterVertexInfo =
                &|_, _, _| WaterVertexInfo::default();
            let sample_loaded: &dyn Fn(i32, i32, i32) -> bool = &|_, _, _| false;
            return f(&empty, sample, sample_water, sample_loaded);
        };

        // Plain stack closures borrowing a single shared `Arc<World>` — no
        // per-mesh heap allocations or `Arc` clones. The call is synchronous,
        // so the closures don't need to outlive it.
        let sample = |x: i32, y: i32, z: i32| world_arc.get_block(x, y, z);
        let sample_water = |x: i32, y: i32, z: i32| world_arc.water_vertex_info(x, y, z);
        let sample_loaded = |x: i32, y: i32, z: i32| world_arc.is_block_loaded(x, y, z);
        f(&chunk, &sample, &sample_water, &sample_loaded)
    }

    /// Snapshot a chunk's 16³ block IDs as one byte per voxel for the
    /// renderer's ray-tracing voxel grid. Layout is x fastest, then y, then z
    /// (matching a `VkBufferImageCopy` into a 3D image). Block IDs above 255
    /// are stored as stone so they stay opaque to shadow rays.
    pub fn gpu_light_voxels(&self, pos: ChunkPos) -> Option<Vec<u8>> {
        let chunks = self.chunks.read_shard(pos);
        let chunk = chunks.get(&pos)?;
        let blocks = chunk.snapshot_blocks();
        let mut out = vec![0u8; 16 * 16 * 16];
        for z in 0..16usize {
            for y in 0..16usize {
                for x in 0..16usize {
                    let idx = voxel_core::math::local_index(x as i32, y as i32, z as i32);
                    let id = blocks[idx].0;
                    out[(z * 16 + y) * 16 + x] = if id <= 255 { id as u8 } else { 1 };
                }
            }
        }
        Some(out)
    }

    /// Collect emissive blocks (torches, lava, glowstone) in the loaded chunks
    /// within `chunk_radius` of `center`. Returns `(world position, emission
    /// colour, emission level)`. Used to build the shader's dynamic point-light
    /// list; scanning is O(loaded voxels) so callers should throttle it.
    pub fn emissive_lights_near(
        &self,
        center: IVec3,
        chunk_radius: i32,
    ) -> Vec<(IVec3, [u8; 3], u8)> {
        let center_chunk = block_to_chunk(center);
        let mut out = Vec::new();
        for shard in self.chunks.read_all() {
            for (pos, chunk) in shard.iter() {
                if (pos.x() - center_chunk.x()).abs() > chunk_radius
                    || (pos.y() - center_chunk.y()).abs() > chunk_radius
                    || (pos.z() - center_chunk.z()).abs() > chunk_radius
                {
                    continue;
                }
                let origin = chunk_origin(*pos);
                for ly in 0..CHUNK_SIZE {
                    for lz in 0..CHUNK_SIZE {
                        for lx in 0..CHUNK_SIZE {
                            let id = chunk.get(lx, ly, lz);
                            if id.is_air() {
                                continue;
                            }
                            let def = self.reg.get(id);
                            if def.emission == 0 {
                                continue;
                            }
                            out.push((
                                origin + IVec3::new(lx, ly, lz),
                                def.emission_color,
                                def.emission,
                            ));
                        }
                    }
                }
            }
        }
        out
    }

    /// Positions of every currently loaded chunk (cheap: keys only).
    pub fn loaded_chunk_positions(&self) -> Vec<ChunkPos> {
        let mut out = Vec::new();
        for shard in self.chunks.read_all() {
            out.extend(shard.keys().copied());
        }
        out
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
    fn emissive_lights_near_finds_torch() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        let torch = world.registry().id_of("torch").unwrap();
        chunk.set(1, 2, 3, torch);
        world.insert_chunk(cp, chunk);

        let lights = world.emissive_lights_near(IVec3::new(1, 2, 3), 1);
        assert_eq!(lights.len(), 1);
        assert_eq!(lights[0].0, IVec3::new(1, 2, 3));
        assert_eq!(lights[0].2, 14);
        assert_eq!(lights[0].1, [255, 200, 100]);
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

    fn chunk_with_water(kind: crate::water::WaterKind, fill: u8, surface_cell: i32) -> Chunk {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        let stone = world.registry().id_of("stone").unwrap();
        let water = world.registry().id_of("water").unwrap();
        for x in 0..16 {
            for z in 0..16 {
                chunk.set(x, 0, z, stone);
                if x < 2 && z < 2 {
                    for y in 1..=surface_cell {
                        chunk.set(x, y, z, water);
                    }
                    chunk.set_water_meta(
                        x,
                        z,
                        WaterColumnMeta::new(
                            kind,
                            fill,
                            crate::water::pack_angle(0.5),
                            crate::water::pack_unit(0.5),
                            0,
                        ),
                    );
                }
            }
        }
        chunk
    }

    #[test]
    fn inserted_chunk_registers_water_columns() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, chunk_with_water(WaterKind::Ocean, 255, 4));

        let column = world.water_column_at(0, 0).expect("column registered");
        assert_eq!(column.kind, WaterKind::Ocean);
        assert!((column.surface_y - 5.0).abs() < 1e-5);
        assert!((column.depth() - 4.0).abs() < 1e-5);
        assert_eq!(world.water_column_count(), 4);
        assert!(world.chunk_has_water(cp));
    }

    #[test]
    fn removing_chunk_clears_water_columns() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, chunk_with_water(WaterKind::Lake, 255, 4));
        assert_eq!(world.water_column_count(), 4);
        world.remove_chunk(cp);
        assert_eq!(world.water_column_count(), 0);
        assert!(world.water_column_at(0, 0).is_none());
    }

    #[test]
    fn water_submersion_uses_partial_surface_height() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        // Lake water so the surface never moves; top voxel fill = 128/255.
        world.insert_chunk(cp, {
            let mut chunk = chunk_with_water(WaterKind::Lake, 128, 4);
            // Remove the top full voxel so the partial-fill cell is the top.
            let water = world.registry().id_of("water").unwrap();
            chunk.set(0, 4, 0, water);
            chunk
        });
        let column = world.water_column_at(0, 0).unwrap();
        assert!((column.surface_y - 4.5).abs() < 0.01);

        let submerged = world.water_submersion(
            glam::Vec3::new(0.2, 4.0, 0.2),
            glam::Vec3::new(0.8, 5.0, 0.8),
        );
        assert!((submerged - 0.5).abs() < 0.02, "submersion={submerged}");
    }

    #[test]
    fn deep_water_bed_spans_vertical_chunks() {
        let world = World::new(42);
        let water = world.registry().id_of("water").unwrap();

        // Upper body slice: world y 16..20, one surface column at (0,0).
        let mut upper = Chunk::new(ChunkPos::new(0, 1, 0));
        for y in 0..=3 {
            upper.set(0, y, 0, water);
        }
        upper.set_water_meta(0, 0, WaterColumnMeta::new(WaterKind::Lake, 255, 0, 0, 0));
        world.insert_chunk(ChunkPos::new(0, 1, 0), upper);

        // Lower body slice: world y 1..15. Inserted after the surface chunk so
        // the merge path has to extend the already-known bed.
        let mut lower = Chunk::new(ChunkPos::new(0, 0, 0));
        for y in 1..=15 {
            lower.set(0, y, 0, water);
        }
        lower.set_water_meta(0, 0, WaterColumnMeta::new(WaterKind::Lake, 255, 0, 0, 0));
        world.insert_chunk(ChunkPos::new(0, 0, 0), lower);

        let column = world.water_column_at(0, 0).unwrap();
        assert!((column.surface_y - 20.0).abs() < 1e-4, "surface={}", column.surface_y);
        assert!((column.bed_y - 1.0).abs() < 1e-4, "bed={}", column.bed_y);
        assert!(
            world.water_sample(glam::Vec3::new(0.5, 10.0, 0.5)).is_some(),
            "water below the surface chunk must still be swimmable"
        );

        // Unloading the lower slice must raise the registered bed and stop
        // reporting water in the now-unloaded space.
        world.remove_chunk(ChunkPos::new(0, 0, 0));
        let column = world.water_column_at(0, 0).unwrap();
        assert!((column.bed_y - 16.0).abs() < 1e-4, "bed={}", column.bed_y);
        assert!(world
            .water_sample(glam::Vec3::new(0.5, 10.0, 0.5))
            .is_none());
    }

    #[test]
    fn water_surface_grid_marks_water_and_its_wet_fringe() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, chunk_with_water(WaterKind::Lake, 255, 4));

        let mut grid = vec![0.0; 9 * 9 * crate::water::WATER_SURFACE_FLOATS_PER_CELL];
        world.fill_water_surface_grid(-2, -2, 9, &mut grid);
        let cell = |grid: &[f32], x: usize, z: usize, layer: usize| {
            let base = (z * 9 + x) * crate::water::WATER_SURFACE_FLOATS_PER_CELL + layer * 2;
            (grid[base], grid[base + 1])
        };

        // The 2x2 water columns are positive at the lake surface with their
        // real bed; the second layer is empty.
        for (x, z) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (surface, bed) = cell(&grid, x + 2, z + 2, 0);
            assert!((surface - 5.0).abs() < 1e-4, "water cell ({x},{z}) = {surface}");
            assert!((bed - 1.0).abs() < 1e-4, "water bed ({x},{z}) = {bed}");
            assert_eq!(cell(&grid, x + 2, z + 2, 1).0, crate::water::WATER_SURFACE_NONE);
        }
        // The immediately adjacent dry beach is marked for wet edges, but
        // negative so the shader does not draw caustics on it.
        let (shore, fringe) = cell(&grid, 4, 2, 0); // world (2, 0)
        assert!((shore + 5.0).abs() < 1e-4, "shore cell = {shore}");
        assert!((fringe - 5.0).abs() < 1e-4, "fringe height = {fringe}");
        // Further inland is unmarked in every layer.
        for layer in 0..crate::water::WATER_SURFACE_LAYERS {
            assert_eq!(
                cell(&grid, 6, 6, layer).0,
                crate::water::WATER_SURFACE_NONE
            );
        }
    }

    #[test]
    fn overlapping_cave_aquifer_is_visible_to_physics_and_grid() {
        let world = World::new(42);
        let water = world.registry().id_of("water").unwrap();

        // Lake body in the surface chunk: world y 16..20.
        let mut upper = Chunk::new(ChunkPos::new(0, 1, 0));
        for y in 0..=3 {
            upper.set(0, y, 0, water);
        }
        upper.set_water_meta(0, 0, WaterColumnMeta::new(WaterKind::Lake, 255, 0, 0, 0));
        world.insert_chunk(ChunkPos::new(0, 1, 0), upper);

        // Cave aquifer in the lower chunk: world y 1..5, separated from the
        // lake by eleven blocks of rock/air.
        let mut lower = Chunk::new(ChunkPos::new(0, 0, 0));
        for y in 1..=4 {
            lower.set(0, y, 0, water);
        }
        lower.set_water_meta(0, 0, WaterColumnMeta::new(WaterKind::Aquifer, 255, 0, 0, 0));
        world.insert_chunk(ChunkPos::new(0, 0, 0), lower);

        let bodies = world.water_bodies_at(0, 0);
        assert_eq!(bodies.len(), 2, "top lake + cave aquifer expected: {bodies:?}");
        assert_eq!(bodies[0].kind, WaterKind::Lake);
        assert!((bodies[0].surface_y - 20.0).abs() < 1e-4);
        assert_eq!(bodies[1].kind, WaterKind::Aquifer);
        assert!((bodies[1].surface_y - 5.0).abs() < 1e-4);
        assert!((bodies[1].bed_y - 1.0).abs() < 1e-4);

        assert_eq!(
            world
                .water_sample(glam::Vec3::new(0.5, 3.0, 0.5))
                .expect("cave water")
                .kind,
            WaterKind::Aquifer
        );
        assert!(world.water_sample(glam::Vec3::new(0.5, 10.0, 0.5)).is_none());
        assert_eq!(
            world
                .water_sample(glam::Vec3::new(0.5, 18.0, 0.5))
                .expect("lake water")
                .kind,
            WaterKind::Lake
        );
        let cave_submersion = world.water_submersion(
            glam::Vec3::new(0.2, 2.0, 0.2),
            glam::Vec3::new(0.8, 4.0, 0.8),
        );
        assert!(cave_submersion > 0.9, "cave submersion = {cave_submersion}");

        let mut grid = vec![
            0.0;
            3 * 3 * crate::water::WATER_SURFACE_FLOATS_PER_CELL
        ];
        world.fill_water_surface_grid(-1, -1, 3, &mut grid);
        let base = (1 * 3 + 1) * crate::water::WATER_SURFACE_FLOATS_PER_CELL;
        assert!((grid[base] - 20.0).abs() < 1e-4, "lake grid layer: {}", grid[base]);
        assert!((grid[base + 1] - 16.0).abs() < 1e-4, "lake bed: {}", grid[base + 1]);
        assert!((grid[base + 2] - 5.0).abs() < 1e-4, "cave grid layer: {}", grid[base + 2]);
        assert!((grid[base + 3] - 1.0).abs() < 1e-4, "cave bed: {}", grid[base + 3]);
    }

    #[test]
    fn one_chunk_can_contain_a_surface_body_and_a_cave_pool() {
        let world = World::new(42);
        let water = world.registry().id_of("water").unwrap();
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        for y in 1..=3 {
            chunk.set(0, y, 0, water);
        }
        for y in 10..=12 {
            chunk.set(0, y, 0, water);
        }
        chunk.set_water_meta(0, 0, WaterColumnMeta::new(WaterKind::Lake, 255, 0, 0, 0));
        world.insert_chunk(cp, chunk);

        let bodies = world.water_bodies_at(0, 0);
        assert_eq!(bodies.len(), 2, "both runs expected: {bodies:?}");
        assert_eq!(bodies[0].kind, WaterKind::Lake);
        assert!((bodies[0].surface_y - 13.0).abs() < 1e-4);
        assert_eq!(bodies[1].kind, WaterKind::Aquifer);
        assert!((bodies[1].surface_y - 4.0).abs() < 1e-4);
        assert!(world
            .water_sample(glam::Vec3::new(0.5, 11.0, 0.5))
            .is_some());
        assert!(world
            .water_sample(glam::Vec3::new(0.5, 2.0, 0.5))
            .is_some());
        assert!(world.water_sample(glam::Vec3::new(0.5, 7.0, 0.5)).is_none());
    }

    #[test]
    fn water_submersion_counts_wave_crests_above_the_top_voxel() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        // Use a deeper column than the minimum needed for a surface cell:
        // the six-swell spectrum deliberately spreads crest phases, so a
        // 4-block column no longer reaches the old single-swell peak at this
        // particular sample point. Eight blocks gives the same test a real
        // crest while still exercising the surface-vs-voxel split.
        let surface_cell = 8;
        world.insert_chunk(cp, chunk_with_water(WaterKind::Ocean, 255, surface_cell));
        let flat_top = (surface_cell + 1) as f32;
        let column = world.water_column_at(0, 0).unwrap();
        assert!(column.wave_amplitude() > 0.5,
                "test needs a wave large enough to clear the voxel top");

        // Advance the clock to a crest whose surface is above the flat top
        // voxel. Old block-scan submersion reported zero here.
        let mut surface = 0.0;
        for _ in 0..400 {
            world.tick_water(0.1);
            surface = world
                .water_column_at(0, 0)
                .unwrap()
                .surface_at(0.5, 0.5, world.water_time());
            if surface > flat_top + 0.5 {
                break;
            }
        }
        assert!(surface > flat_top + 0.5, "no crest found (surface={surface})");

        let submerged = world.water_submersion(
            glam::Vec3::new(0.2, flat_top + 0.1, 0.2),
            glam::Vec3::new(0.8, flat_top + 0.9, 0.8),
        );
        assert!(submerged > 0.4, "crest must buoy the box (sub={submerged})");
    }

    #[test]
    fn place_water_creates_a_column_and_remove_clears_it() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, Chunk::new(cp));

        let stone = world.registry().id_of("stone").unwrap();
        world.set_block(1, 0, 1, stone);
        assert!(world.place_water(1, 1, 1));
        let column = world.water_column_at(1, 1).expect("placed column");
        assert_eq!(column.kind, WaterKind::Pond);
        assert!((column.surface_y - 2.0).abs() < 1e-5);
        assert!(world.water_sample(glam::Vec3::new(1.5, 1.5, 1.5)).is_some());

        assert!(world.remove_water(1, 1, 1));
        assert!(world.water_column_at(1, 1).is_none());
    }

    #[test]
    fn water_sample_reports_body_flow_and_wave() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, chunk_with_water(WaterKind::River, 255, 4));
        let sample = world
            .water_sample(glam::Vec3::new(0.5, 2.0, 0.5))
            .expect("inside river");
        assert_eq!(sample.kind, WaterKind::River);
        assert!(sample.flow_speed > 0.0);
        assert!((sample.depth - 4.0).abs() < 1e-5);
        // A point well above the wave crest is not submerged.
        assert!(world
            .water_sample(glam::Vec3::new(0.5, 8.0, 0.5))
            .is_none());
    }

    #[test]
    fn tick_water_advances_the_wave_clock() {
        let world = World::new(42);
        let before = world.water_time();
        world.tick_water(0.25);
        assert!((world.water_time() - before - 0.25).abs() < 1e-5);
        world.tick_water(-1.0);
        assert!((world.water_time() - before - 0.25).abs() < 1e-5);
    }

    #[test]
    fn water_vertex_info_marks_surface_and_amplitude() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, chunk_with_water(WaterKind::Ocean, 255, 4));
        let top = world.water_vertex_info(0, 4, 0);
        assert!(top.surface);
        assert_eq!(top.kind, WaterKind::Ocean.as_u8());
        assert!(top.amplitude > 0);
        let under = world.water_vertex_info(0, 3, 0);
        assert!(!under.surface);
        assert_eq!(under.fill, 255);
    }
}
