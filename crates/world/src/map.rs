//! Minimap column sampler: bulk-scans world columns for top-block data.
//!
//! Used by the minimap/fullscreen map system to build terrain-colored
//! framebuffer textures. The chunk-acquired approach (`sample_columns_chunked`)
//! locks the chunk map once and iterates local columns via `Chunk::column_height`
//! + `Chunk::get`, avoiding per-block hashmap lookups.

use std::sync::Arc;

use voxel_core::{
    math::{chunk_origin, ChunkPos},
    BlockId, CHUNK_SIZE,
};

use crate::world::World;

/// Result of scanning one world column for map purposes.
#[derive(Clone, Debug)]
pub struct ColumnSample {
    pub block_x: i32,
    pub block_z: i32,
    pub top_block: BlockId,
    /// Y coordinate of the top non-air block.
    pub height: i32,
}

impl World {
    /// Bulk-scan columns in a square radius around `center` (block coords).
    /// Returns one [`ColumnSample`] per column that has a non-air block.
    ///
    /// Uses the chunk-acquired approach: locks the chunk map once, iterates
    /// chunks in radius, then scans local columns via `column_height` + `get`.
    /// For a radius of 128 blocks (8 chunks), this does ~289 chunk lookups
    /// instead of 65,536+ block lookups.
    pub fn sample_columns_chunked(
        self: &Arc<Self>,
        center: (i32, i32),
        radius_blocks: u32,
    ) -> Vec<ColumnSample> {
        let chunks = self.chunks_ref().read();

        let radius_chunks = (radius_blocks as i32 + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let center_cx = center.0 >> 4;
        let center_cz = center.1 >> 4;

        let capacity = (radius_blocks as usize * 2 + 1).pow(2);
        let mut results = Vec::with_capacity(capacity);

        for dcx in -radius_chunks..=radius_chunks {
            for dcz in -radius_chunks..=radius_chunks {
                let cx = center_cx + dcx;
                let cz = center_cz + dcz;
                let pos = ChunkPos::new(cx, 0, cz);

                let Some(chunk) = chunks.get(&pos) else {
                    continue;
                };
                if !chunk.generated {
                    continue;
                }

                let origin = chunk_origin(pos);

                for lx in 0..CHUNK_SIZE {
                    for lz in 0..CHUNK_SIZE {
                        let wx = origin.x + lx;
                        let wz = origin.z + lz;

                        // Skip columns outside the block radius.
                        let dx = wx - center.0;
                        let dz = wz - center.1;
                        if dx * dx + dz * dz > (radius_blocks as i32).pow(2) {
                            continue;
                        }

                        let height = chunk.column_height(lx, lz);
                        if height <= 0 {
                            continue;
                        }
                        let top_y = height - 1;
                        let top_block = chunk.get(lx, top_y, lz);
                        if top_block.is_air() {
                            continue;
                        }

                        results.push(ColumnSample {
                            block_x: wx,
                            block_z: wz,
                            top_block,
                            height: top_y,
                        });
                    }
                }
            }
        }

        results
    }
}
