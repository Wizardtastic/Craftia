//! Chunk storage: a 16³ block volume with palette-based compression.
//!
//! Blocks are stored using a palette + index mapping:
//! - A palette of unique `BlockId` values (max 256 for u8 indices)
//! - Per-block u8 index into the palette
//! - Falls back to flat `Box<[BlockId]>` storage when palette overflows
//!
//! Layout is y-major, z-middle, x-fastest so that vertical column scans (used
//! during worldgen and meshing) are cache-friendly. Indexing helpers live in
//! `voxel_core::math`.

use voxel_core::{BlockId, ChunkPos, CHUNK_CUBED, CHUNK_SIZE};

use crate::registry::BlockRegistry;
use crate::water::WaterColumnMeta;

/// One cubic chunk section: 16×16×16 blocks.
#[derive(Clone)]
pub struct Chunk {
    pub pos: ChunkPos,
    /// Palette: unique block IDs used in this chunk. Empty when in flat mode.
    palette: Vec<BlockId>,
    /// Index array: for each block position, the index into `palette`.
    /// Only allocated when in palette mode.
    indices: Box<[u8]>,
    /// Flat block storage, used when the palette overflows 255 entries.
    /// Invariant: populated iff `!palette_mode`; freed (empty) in palette mode.
    flat: Box<[BlockId]>,
    /// True when using palette mode (indices is valid).
    palette_mode: bool,
    /// Cached count of non-air blocks, kept O(1) for the mesher's per-remesh
    /// capacity reservation instead of scanning all 4096 voxels each time.
    non_air: usize,
    /// Per-column water-body metadata (16×16 records). Water surfaces are a
    /// continuous height field; there are no per-voxel flow levels anymore.
    pub(crate) water_meta: Box<[WaterColumnMeta]>,
    /// Set whenever a block changes so the mesher/streamer knows to remesh.
    pub dirty: bool,
    /// True once `TerrainGenerator` has populated this chunk.
    pub generated: bool,
}

impl Chunk {
    pub fn new(pos: ChunkPos) -> Self {
        Self {
            pos,
            palette: Vec::new(),
            indices: Box::new([0u8; CHUNK_CUBED]),
            flat: vec![BlockId::AIR; CHUNK_CUBED].into_boxed_slice(),
            palette_mode: false,
            // A fresh chunk is entirely air.
            non_air: 0,
            water_meta: vec![
                WaterColumnMeta::NONE;
                (CHUNK_SIZE * CHUNK_SIZE) as usize
            ]
            .into_boxed_slice(),
            dirty: false,
            generated: false,
        }
    }

    /// Borrow the flat block array. Only valid in flat mode — in palette mode
    /// the flat array is freed (see `consider_palette_mode`). Callers that may
    /// run in either mode should use [`Chunk::snapshot_blocks`] instead.
    #[inline]
    pub fn blocks(&self) -> &[BlockId] {
        debug_assert!(
            !self.palette_mode,
            "blocks() is only valid in flat mode; use snapshot_blocks()"
        );
        &self.flat
    }

    /// Materialize the chunk's blocks into an owned `Vec`, working in either
    /// storage mode. Use this when a flat view is needed regardless of whether
    /// the chunk is palette-compressed.
    pub fn snapshot_blocks(&self) -> Vec<BlockId> {
        if self.palette_mode {
            self.indices
                .iter()
                .map(|&i| self.palette[i as usize])
                .collect()
        } else {
            self.flat.to_vec()
        }
    }

    /// Get a block by local coordinate. Returns air for out-of-range locals.
    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> BlockId {
        if !in_bounds(x, y, z) {
            return BlockId::AIR;
        }
        let idx = voxel_core::math::local_index(x, y, z);
        if self.palette_mode {
            let pal_idx = self.indices[idx] as usize;
            self.palette[pal_idx]
        } else {
            self.flat[idx]
        }
    }

    /// Set a block by local coordinate. Ignores out-of-range writes and marks
    /// the chunk dirty on change.
    #[inline]
    pub fn set(&mut self, x: i32, y: i32, z: i32, id: BlockId) {
        if !in_bounds(x, y, z) {
            return;
        }
        let idx = voxel_core::math::local_index(x, y, z);
        // Read the current block once so we can (a) early-out on no-op writes
        // and (b) update the running non-air count by the air-ness delta.
        let old = if self.palette_mode {
            self.palette[self.indices[idx] as usize]
        } else {
            self.flat[idx]
        };
        if old == id {
            return;
        }
        if self.palette_mode {
            // Try to find or add to palette.
            if let Some(new_pal_idx) = self.find_or_add_palette(id) {
                self.indices[idx] = new_pal_idx as u8;
            } else {
                // Palette overflow: convert to flat mode.
                self.convert_to_flat();
                self.flat[idx] = id;
            }
        } else {
            self.flat[idx] = id;
            // Try to switch to palette mode if beneficial.
            self.consider_palette_mode();
        }
        // Maintain the non-air count incrementally (old != id here).
        match (old.is_air(), id.is_air()) {
            (true, false) => self.non_air += 1,
            (false, true) => self.non_air -= 1,
            _ => {}
        }
        self.dirty = true;
    }

    /// Find `id` in the palette, or add it. Returns the palette index, or None
    /// if palette is full (>255 unique blocks).
    fn find_or_add_palette(&mut self, id: BlockId) -> Option<usize> {
        if let Some(pos) = self.palette.iter().position(|&p| p == id) {
            return Some(pos);
        }
        if self.palette.len() >= 256 {
            return None;
        }
        self.palette.push(id);
        Some(self.palette.len() - 1)
    }

    /// After enough unique blocks accumulate in flat mode, switch to palette mode.
    fn consider_palette_mode(&mut self) {
        if self.palette_mode {
            return;
        }
        // Count unique blocks. If <= 128, switch to palette mode.
        let mut unique = std::collections::HashSet::with_capacity(128);
        let mut unique_vec = Vec::with_capacity(128);
        for &b in self.flat.iter() {
            if unique.insert(b) {
                unique_vec.push(b);
                if unique_vec.len() > 128 {
                    return; // Too many unique blocks, stay in flat mode.
                }
            }
        }
        // Switch to palette mode.
        self.palette = unique_vec;
        let mut lookup: std::collections::HashMap<BlockId, u8> = std::collections::HashMap::new();
        for (i, &p) in self.palette.iter().enumerate() {
            lookup.insert(p, i as u8);
        }
        for i in 0..CHUNK_CUBED {
            let id = self.flat[i];
            let pal_idx = *lookup.get(&id).unwrap();
            self.indices[i] = pal_idx;
        }
        self.palette_mode = true;
        // In palette mode the flat array is redundant; free its ~8 KB.
        // The invariant `flat is populated iff !palette_mode` is restored by
        // `convert_to_flat` before any flat-mode read.
        self.flat = Box::new([]);
    }

    /// Convert from palette mode to flat mode.
    fn convert_to_flat(&mut self) {
        if !self.palette_mode {
            return;
        }
        // `flat` was freed on entering palette mode; reallocate and repopulate.
        let mut flat = vec![BlockId::AIR; CHUNK_CUBED].into_boxed_slice();
        for i in 0..CHUNK_CUBED {
            flat[i] = self.palette[self.indices[i] as usize];
        }
        self.flat = flat;
        self.palette.clear();
        self.palette.shrink_to_fit();
        self.palette_mode = false;
    }

    /// Water metadata for the horizontal column at local `(x, z)`.
    #[inline]
    pub fn water_meta(&self, x: i32, z: i32) -> WaterColumnMeta {
        if !(0..CHUNK_SIZE).contains(&x) || !(0..CHUNK_SIZE).contains(&z) {
            return WaterColumnMeta::NONE;
        }
        self.water_meta[(x as usize) * CHUNK_SIZE as usize + z as usize]
    }

    /// Set water metadata for the horizontal column at local `(x, z)`.
    #[inline]
    pub fn set_water_meta(&mut self, x: i32, z: i32, meta: WaterColumnMeta) {
        if !(0..CHUNK_SIZE).contains(&x) || !(0..CHUNK_SIZE).contains(&z) {
            return;
        }
        let idx = (x as usize) * CHUNK_SIZE as usize + z as usize;
        if self.water_meta[idx] != meta {
            self.water_meta[idx] = meta;
            self.dirty = true;
        }
    }

    /// Clear water metadata for a column (used when the last water voxel is
    /// removed).
    #[inline]
    pub fn clear_water_meta(&mut self, x: i32, z: i32) {
        self.set_water_meta(x, z, WaterColumnMeta::NONE);
    }

    /// Count non-air blocks (used for empty-chunk culling in the mesher).
    /// O(1): returns the cached count maintained by `set` and the bulk paths.
    #[inline]
    pub fn non_air_count(&self) -> usize {
        self.non_air
    }

    /// Recompute the non-air count from current storage. Used by bulk paths
    /// (fill/restore) which are already O(4096), so this adds no asymptotic
    /// cost while keeping the cached `non_air` field authoritative.
    fn recompute_non_air(&mut self) {
        self.non_air = if self.palette_mode {
            // Fast path: check if air is in the palette.
            if let Some(air_idx) = self.palette.iter().position(|b| b.is_air()) {
                let air_idx = air_idx as u8;
                self.indices.iter().filter(|&&i| i != air_idx).count()
            } else {
                // No air in palette, all blocks are non-air.
                CHUNK_CUBED
            }
        } else {
            self.flat.iter().filter(|b| !b.is_air()).count()
        };
    }

    /// Highest non-air block in a local column `(x, z)`, or 0 if empty.
    pub fn column_height(&self, x: i32, z: i32) -> i32 {
        if !in_bounds(x, 0, z) {
            return 0;
        }
        for y in (0..voxel_core::CHUNK_SIZE).rev() {
            if !self.get(x, y, z).is_air() {
                return y + 1;
            }
        }
        0
    }

    /// Fill the chunk from a function keyed by local coordinate. Used by gen.
    pub fn fill_with(&mut self, mut f: impl FnMut(i32, i32, i32) -> BlockId) {
        // Always use flat mode for fill_with to avoid repeated palette lookups.
        self.convert_to_flat();
        for y in 0..voxel_core::CHUNK_SIZE {
            for z in 0..voxel_core::CHUNK_SIZE {
                for x in 0..voxel_core::CHUNK_SIZE {
                    let id = f(x, y, z);
                    let idx = voxel_core::math::local_index(x, y, z);
                    self.flat[idx] = id;
                }
            }
        }
        self.dirty = true;
        self.consider_palette_mode();
        self.recompute_non_air();
    }

    /// Convenience: replace every block with `id`.
    pub fn fill_uniform(&mut self, id: BlockId) {
        if self.palette_mode {
            // Just reset palette to single entry.
            self.palette.clear();
            self.palette.push(id);
            for i in self.indices.iter_mut() {
                *i = 0;
            }
        } else {
            for b in self.flat.iter_mut() {
                *b = id;
            }
            self.consider_palette_mode();
        }
        self.non_air = if id.is_air() { 0 } else { CHUNK_CUBED };
        self.dirty = true;
    }

    /// Number of solid (collidable) blocks per the given registry.
    pub fn solid_count(&self, reg: &BlockRegistry) -> usize {
        if self.palette_mode {
            // Pre-compute solidity per palette entry.
            let solid: Vec<bool> = self.palette.iter().map(|b| reg.is_solid(*b)).collect();
            self.indices.iter().filter(|&&i| solid[i as usize]).count()
        } else {
            self.flat.iter().filter(|b| reg.is_solid(**b)).count()
        }
    }

    /// Returns true if the chunk is in palette mode (for diagnostics).
    pub fn is_palette_mode(&self) -> bool {
        self.palette_mode
    }

    /// Number of unique blocks in the palette (0 if in flat mode).
    pub fn palette_len(&self) -> usize {
        if self.palette_mode {
            self.palette.len()
        } else {
            0
        }
    }

    // --- Save/Load accessors ---

    /// Get palette data (for save).
    pub fn palette_data(&self) -> &[BlockId] {
        &self.palette
    }

    /// Get index data (for save).
    pub fn indices_data(&self) -> &[u8] {
        &self.indices
    }

    /// Get per-column water metadata (for save).
    pub fn water_meta_data(&self) -> &[WaterColumnMeta] {
        &self.water_meta
    }

    /// Restore from palette + indices (for load).
    pub fn restore_palette(&mut self, palette: Vec<BlockId>, indices: Vec<u8>) {
        self.palette = palette;
        self.indices = indices.into_boxed_slice();
        self.palette_mode = true;
        // Palette mode keeps the flat array freed (see `consider_palette_mode`).
        self.flat = Box::new([]);
        self.dirty = true;
        self.recompute_non_air();
    }

    /// Restore from flat block data (for load).
    pub fn restore_flat(&mut self, blocks: Vec<BlockId>) {
        self.flat = blocks.into_boxed_slice();
        self.palette.clear();
        self.palette_mode = false;
        self.dirty = true;
        self.recompute_non_air();
    }

    /// Restore per-column water metadata (for load). Missing entries are
    /// filled with the empty record so older saves still load.
    pub fn restore_water_meta(&mut self, data: Vec<WaterColumnMeta>) {
        let len = (CHUNK_SIZE * CHUNK_SIZE) as usize;
        let mut meta = vec![WaterColumnMeta::NONE; len];
        let copy = data.len().min(len);
        meta[..copy].copy_from_slice(&data[..copy]);
        self.water_meta = meta.into_boxed_slice();
        self.dirty = true;
    }
}

#[inline]
fn in_bounds(x: i32, y: i32, z: i32) -> bool {
    (0..voxel_core::CHUNK_SIZE).contains(&x)
        && (0..voxel_core::CHUNK_SIZE).contains(&y)
        && (0..voxel_core::CHUNK_SIZE).contains(&z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::{ChunkPos, CHUNK_SIZE};

    fn test_chunk() -> Chunk {
        Chunk::new(ChunkPos::new(0, 0, 0))
    }

    #[test]
    fn new_chunk_is_all_air() {
        let c = test_chunk();
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    assert!(c.get(x, y, z).is_air());
                }
            }
        }
    }

    #[test]
    fn set_and_get() {
        let mut c = test_chunk();
        let stone = BlockId(2);
        c.set(3, 5, 7, stone);
        assert_eq!(c.get(3, 5, 7), stone);
    }

    #[test]
    fn out_of_bounds_returns_air() {
        let c = test_chunk();
        assert!(c.get(-1, 0, 0).is_air());
        assert!(c.get(CHUNK_SIZE, 0, 0).is_air());
        assert!(c.get(0, -1, 0).is_air());
        assert!(c.get(0, 0, CHUNK_SIZE).is_air());
    }

    #[test]
    fn non_air_count() {
        let mut c = test_chunk();
        assert_eq!(c.non_air_count(), 0);
        c.set(0, 0, 0, BlockId(1));
        c.set(1, 1, 1, BlockId(2));
        assert_eq!(c.non_air_count(), 2);
    }

    #[test]
    fn fill_uniform() {
        let mut c = test_chunk();
        c.fill_uniform(BlockId(5));
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    assert_eq!(c.get(x, y, z), BlockId(5));
                }
            }
        }
    }

    #[test]
    fn fill_with() {
        let mut c = test_chunk();
        c.fill_with(|x, y, z| BlockId((x + y + z) as u16));
        assert_eq!(c.get(0, 0, 0), BlockId(0));
        assert_eq!(c.get(1, 2, 3), BlockId(6));
    }

    #[test]
    fn column_height_empty() {
        let c = test_chunk();
        assert_eq!(c.column_height(0, 0), 0);
    }

    #[test]
    fn column_height_with_block() {
        let mut c = test_chunk();
        c.set(0, 10, 0, BlockId(1));
        assert_eq!(c.column_height(0, 0), 11); // y + 1
    }

    #[test]
    fn solid_count_empty() {
        let c = test_chunk();
        let reg = crate::registry::BlockRegistry::with_builtins();
        assert_eq!(c.solid_count(&reg), 0);
    }

    // --- Palette mode tests ---

    #[test]
    fn palette_mode_activates_with_few_unique_blocks() {
        let mut c = test_chunk();
        // Set several blocks of the same type.
        for i in 0..20 {
            c.set(i, 0, 0, BlockId(1));
        }
        // 2 unique blocks (air + BlockId(1)) is well under 128, so palette mode activates.
        assert!(c.is_palette_mode());
    }

    #[test]
    fn fill_uniform_uses_palette() {
        let mut c = test_chunk();
        c.fill_uniform(BlockId(5));
        // Single block type should use palette mode.
        assert!(c.is_palette_mode());
        assert_eq!(c.palette_len(), 1);
        assert_eq!(c.get(0, 0, 0), BlockId(5));
        assert_eq!(c.get(15, 15, 15), BlockId(5));
    }

    #[test]
    fn palette_get_set_roundtrip() {
        let mut c = test_chunk();
        c.fill_uniform(BlockId(1));
        // Now set a different block.
        c.set(5, 5, 5, BlockId(2));
        assert_eq!(c.get(5, 5, 5), BlockId(2));
        assert_eq!(c.get(4, 4, 4), BlockId(1));
        // Palette should have grown.
        assert!(c.palette_len() >= 2);
    }

    // --- water metadata tests ---

    #[test]
    fn water_meta_defaults_to_none() {
        let c = test_chunk();
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                assert!(!c.water_meta(x, z).is_some());
            }
        }
    }

    #[test]
    fn set_get_water_meta() {
        let mut c = test_chunk();
        let meta = WaterColumnMeta::new(
            crate::water::WaterKind::River,
            200,
            12,
            180,
            30,
        );
        c.set_water_meta(3, 7, meta);
        assert_eq!(c.water_meta(3, 7), meta);
    }

    #[test]
    fn water_meta_out_of_bounds_returns_none() {
        let c = test_chunk();
        assert!(!c.water_meta(-1, 0).is_some());
        assert!(!c.water_meta(CHUNK_SIZE, 0).is_some());
        assert!(!c.water_meta(0, -1).is_some());
        assert!(!c.water_meta(0, CHUNK_SIZE).is_some());
    }

    #[test]
    fn water_meta_set_out_of_bounds_is_noop() {
        let mut c = test_chunk();
        let meta = WaterColumnMeta::new(crate::water::WaterKind::Pond, 255, 0, 0, 0);
        c.set_water_meta(-1, 0, meta);
        c.set_water_meta(CHUNK_SIZE, 0, meta);
        assert!(!c.water_meta(0, 0).is_some());
    }

    #[test]
    fn water_meta_dirty_on_change() {
        let mut c = test_chunk();
        c.dirty = false;
        c.set_water_meta(0, 0, WaterColumnMeta::new(crate::water::WaterKind::Pond, 255, 0, 0, 0));
        assert!(c.dirty);
    }

    #[test]
    fn water_meta_no_dirty_on_same_value() {
        let mut c = test_chunk();
        let meta = WaterColumnMeta::new(crate::water::WaterKind::Pond, 255, 0, 0, 0);
        c.set_water_meta(0, 0, meta);
        c.dirty = false;
        c.set_water_meta(0, 0, meta);
        assert!(!c.dirty);
    }

    #[test]
    fn water_meta_multiple_columns() {
        let mut c = test_chunk();
        let river = WaterColumnMeta::new(crate::water::WaterKind::River, 255, 1, 2, 3);
        let lake = WaterColumnMeta::new(crate::water::WaterKind::Lake, 128, 0, 0, 0);
        c.set_water_meta(0, 0, river);
        c.set_water_meta(5, 5, lake);
        assert_eq!(c.water_meta(0, 0), river);
        assert_eq!(c.water_meta(5, 5), lake);
        assert!(!c.water_meta(1, 0).is_some());
        assert!(!c.water_meta(5, 6).is_some());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn get_set_roundtrip(x in 0i32..16, y in 0i32..16, z in 0i32..16, id in 1u16..255) {
            let mut c = Chunk::new(ChunkPos::new(0, 0, 0));
            let block = BlockId(id);
            c.set(x, y, z, block);
            prop_assert_eq!(c.get(x, y, z), block);
        }

        #[test]
        fn out_of_bounds_get_is_air(x in -100i32..0, y in 0i32..16, z in 0i32..16) {
            let c = Chunk::new(ChunkPos::new(0, 0, 0));
            prop_assert_eq!(c.get(x, y, z), BlockId::AIR);
        }

        #[test]
        fn out_of_bounds_set_is_noop(x in -100i32..0, y in 0i32..16, z in 0i32..16) {
            let mut c = Chunk::new(ChunkPos::new(0, 0, 0));
            c.set(x, y, z, BlockId(5));
            prop_assert!(c.get(0, 0, 0).is_air());
        }

        #[test]
        fn set_same_value_no_dirty(x in 0i32..16, y in 0i32..16, z in 0i32..16) {
            let mut c = Chunk::new(ChunkPos::new(0, 0, 0));
            c.set(x, y, z, BlockId::AIR); // already air
            prop_assert!(!c.dirty);
        }

        #[test]
        fn water_meta_get_set(x in 0i32..16, z in 0i32..16, fill in 0u8..=255) {
            let mut c = Chunk::new(ChunkPos::new(0, 0, 0));
            let meta = WaterColumnMeta::new(crate::water::WaterKind::Pond, fill, 0, 0, 0);
            c.set_water_meta(x, z, meta);
            prop_assert_eq!(c.water_meta(x, z), meta);
        }
    }
}
