//! Permanent CPU greedy-mesher benchmark.
//!
//! Run with:
//!   cargo run --release -p voxel-world --example mesher_bench
//!
//! Kept as an example (rather than a `#[bench]` harness) so it needs no extra
//! dependency and works on stable Rust. It reports ms/chunk and a checksum so
//! a silent geometry regression is visible alongside the timing.

use std::time::Instant;

use voxel_core::{BlockId, ChunkPos, CHUNK_SIZE};
use voxel_world::{
    chunk::Chunk,
    mesh::ChunkMesher,
    registry::BlockRegistry,
    water::WaterVertexInfo,
};

fn main() {
    let reg = BlockRegistry::with_builtins();
    let stone = reg.id_of("stone").unwrap();
    let dirt = reg.id_of("dirt").unwrap();
    let grass = reg.id_of("grass").unwrap();
    let sand = reg.id_of("sand").unwrap();
    let water = reg.id_of("water").unwrap();

    let s = CHUNK_SIZE;
    let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
    for x in 0..s {
        for z in 0..s {
            let h = 5 + ((x * 7 + z * 13) % 6);
            for y in 0..h {
                let id = if y + 2 >= h {
                    grass
                } else if y + 4 >= h {
                    dirt
                } else {
                    stone
                };
                chunk.set(x, y, z, id);
            }
            if h < 4 {
                for y in h..4 {
                    chunk.set(x, y, z, water);
                }
            } else {
                chunk.set(x, h, z, sand);
            }
        }
    }

    let sample = |x: i32, y: i32, z: i32| {
        if (0..s).contains(&x) && (0..s).contains(&y) && (0..s).contains(&z) {
            chunk.get(x, y, z)
        } else {
            BlockId::AIR
        }
    };
    // Match the real streamer's water metadata closely enough for the mesher's
    // surface path: every water voxel is part of the same ocean body.
    let sample_water = |x: i32, y: i32, z: i32| {
        if (0..s).contains(&x)
            && (0..s).contains(&y)
            && (0..s).contains(&z)
            && chunk.get(x, y, z) == water
        {
            WaterVertexInfo {
                kind: 1,
                fill: 255,
                amplitude: 32,
                flow_angle: 0,
                flow_speed: 0,
                rapids: 0,
                surface: true,
            }
        } else {
            WaterVertexInfo::default()
        }
    };
    let sample_loaded = |x: i32, y: i32, z: i32| {
        (0..s).contains(&x) && (0..s).contains(&y) && (0..s).contains(&z)
    };

    let mesher = ChunkMesher;
    // Warm up allocators/branch predictors.
    let warm = mesher.build(&chunk, &reg, sample, sample_water, sample_loaded);
    let expected_verts = warm.opaque.vertices.len() + warm.transparent.vertices.len();
    let expected_idxs = warm.opaque.indices.len() + warm.transparent.indices.len();

    let iters = 300u32;
    let t0 = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iters {
        let b = mesher.build(&chunk, &reg, sample, sample_water, sample_loaded);
        checksum = checksum.wrapping_add(
            b.opaque.vertices.len() * 31
                + b.opaque.indices.len()
                + b.transparent.vertices.len() * 17
                + b.transparent.indices.len(),
        );
    }
    let elapsed = t0.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0 / f64::from(iters);
    println!(
        "mesh: {ms:.3} ms/chunk  ({expected_verts} verts, {expected_idxs} indices, checksum {checksum})"
    );
    assert_eq!(checksum % 7, (expected_verts * 31 + expected_idxs) % 7, "geometry changed between iterations");
}
