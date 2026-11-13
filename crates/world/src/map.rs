//! Minimap column sampler: bulk-scans world columns for top-block data.
//!
//! Used by the minimap/fullscreen map system to build terrain-colored
//! framebuffer textures. The chunk-acquired approach (`sample_columns_chunked`)
//! scans local columns and avoids per-block world queries.

use std::sync::Arc;

use voxel_core::{math::chunk_origin, BlockId, ChunkPos, CHUNK_SIZE};

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
    /// Scans each candidate chunk's local columns once, avoiding per-block
    /// world queries.
    pub fn sample_columns_chunked(
        self: &Arc<Self>,
        center: (i32, i32),
        radius_blocks: u32,
    ) -> Vec<ColumnSample> {
        let radius_chunks = (radius_blocks as i32 + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let center_cx = center.0 >> 4;
        let center_cz = center.1 >> 4;

        let capacity = (radius_blocks as usize * 2 + 1).pow(2);
        let mut results = Vec::with_capacity(capacity);

        let radius_squared = (radius_blocks as i32).pow(2);
        for dcx in -radius_chunks..=radius_chunks {
            for dcz in -radius_chunks..=radius_chunks {
                let pos = ChunkPos::new(center_cx + dcx, 0, center_cz + dcz);
                let chunks = self.chunks.read_shard(pos);
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
                        let dx = wx - center.0;
                        let dz = wz - center.1;
                        if dx * dx + dz * dz > radius_squared {
                            continue;
                        }

                        let height = chunk.column_height(lx, lz);
                        if height <= 0 {
                            continue;
                        }
                        let top_y = height - 1;
                        let top_block = chunk.get(lx, top_y, lz);
                        if !top_block.is_air() {
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
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;

    #[test]
    fn sample_columns_includes_generated_chunks_and_filters_radius() {
        let world = World::new(42);
        let stone = world.registry().id_of("stone").expect("builtin stone");

        let mut origin_chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        origin_chunk.generated = true;
        origin_chunk.set(8, 3, 8, stone);
        world.insert_chunk(ChunkPos::new(0, 0, 0), origin_chunk);

        let mut adjacent_chunk = Chunk::new(ChunkPos::new(1, 0, 0));
        adjacent_chunk.generated = true;
        adjacent_chunk.set(0, 5, 8, stone);
        world.insert_chunk(ChunkPos::new(1, 0, 0), adjacent_chunk);

        let samples = world.sample_columns_chunked((8, 8), 7);
        assert!(samples.iter().any(|sample| {
            sample.block_x == 8
                && sample.block_z == 8
                && sample.top_block == stone
                && sample.height == 3
        }));
        assert!(!samples
            .iter()
            .any(|sample| sample.block_x == 16 && sample.block_z == 8));
    }
}
