//! World save/load: binary format with flate2 compression.
//!
//! File layout:
//! ```text
//! [9 bytes]  Magic: b"VOXELSAV2"
//! [4 bytes]  Format version: u32 LE
//! [4 bytes]  Seed: i32 LE
//! --- per chunk (each chunk.bin is independently deflated) ---
//! [12 bytes] Chunk position: 3 × i32 LE (cx, cy, cz) — written under
//!           a directory named `c_<cx>_<cy>_<cz>/chunk.bin`
//! [N bytes]  Compressed chunk data (flate2 deflate)
//!   Original data:
//!     [1 byte]   Palette mode flag (0 = flat, 1 = palette)
//!     If flat (0):
//!       [4096 * 2 bytes] Block IDs as u16 LE × 4096
//!     If palette (1):
//!       [2 bytes]  Palette length p as u16 LE (max 4096)
//!       [2p bytes] Palette entries as u16 LE
//!       [4096 bytes] u8 indices into palette
//!     [4096 bytes] Sunlight u8 × 4096
//!     [4096 bytes] Torchlight u8 × 4096
//!     [4096 bytes] Water level u8 × 4096
//!     [v3 and later only]
//!     [4096 * 4 bytes] Packed RGBA8 torchlight color u32 × 4096 little-endian
//! ```
//!
//! Format versioning
//! -----------------
//! v1: original (pre-coloured-torchlight). No `torchlight_color` field at all.
//! v2: added `torchlight_color` in chunk save? Actually v2 was the version that
//!     did NOT save torchlight colour, which is why reloads reverted all
//!     voxels to warm white.
//! v3: **current**. Persists packed RGBA8 torchlight colour for every voxel
//!     after torchlight/water_level. v2 saves still load cleanly (the new
//!     field defaults to all-zero, which `Chunk::get_combined_light_color`
//!     transparently translates to warm white — same behaviour as before).

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use flate2::Compression;

use crate::chunk::Chunk;
use crate::world::World;

const MAGIC: &[u8; 9] = b"VOXELSAV2";
const FORMAT_VERSION: u32 = 3;

/// Save the entire world to a directory. Each chunk is a separate compressed
/// blob keyed by (cx, cy, cz).
pub fn save_world(world: &World, path: &Path) -> Result<()> {
    fs::create_dir_all(path).context("creating save directory")?;

    let meta_path = path.join("meta.bin");
    let seed = world.seed();

    // Write metadata.
    {
        let mut f = BufWriter::new(File::create(&meta_path)?);
        f.write_all(MAGIC)?;
        f.write_all(&FORMAT_VERSION.to_le_bytes())?;
        f.write_all(&seed.to_le_bytes())?;
    }

    // Save each loaded chunk.
    let chunks = world.all_loaded_chunks();
    let count = chunks.len();
    for (cp, chunk) in &chunks {
        let chunk_path = chunk_dir(path, cp.x(), cp.y(), cp.z());
        fs::create_dir_all(chunk_path.parent().unwrap())?;
        let mut f = BufWriter::new(File::create(&chunk_path)?);
        write_chunk(chunk, &mut f)?;
    }

    log::info!("Saved {} chunks to {}", count, path.display());
    Ok(())
}

/// Load a world from a directory. Returns (seed, chunks).
pub fn load_world(path: &Path) -> Result<(i32, Vec<(voxel_core::ChunkPos, Chunk)>)> {
    let meta_path = path.join("meta.bin");
    if !meta_path.exists() {
        bail!("No save file found at {}", path.display());
    }

    let mut f = BufReader::new(File::open(&meta_path)?);
    let mut magic = [0u8; 9];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("Invalid save file magic");
    }

    let mut version_buf = [0u8; 4];
    f.read_exact(&mut version_buf)?;
    let version = u32::from_le_bytes(version_buf);
    if version != FORMAT_VERSION {
        bail!("Unsupported save version {version}");
    }

    let mut seed_buf = [0u8; 4];
    f.read_exact(&mut seed_buf)?;
    let seed = i32::from_le_bytes(seed_buf);

    // Scan for chunk files.
    let mut chunks = Vec::new();
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let dir_name = entry.file_name();
            let dir_str = dir_name.to_string_lossy();
            // Chunk dirs are named "c_x_y_z".
            if let Some(rest) = dir_str.strip_prefix("c_") {
                let parts: Vec<&str> = rest.split('_').collect();
                if parts.len() == 3 {
                    if let (Ok(cx), Ok(cy), Ok(cz)) = (
                        parts[0].parse::<i32>(),
                        parts[1].parse::<i32>(),
                        parts[2].parse::<i32>(),
                    ) {
                        let chunk_path = entry.path().join("chunk.bin");
                        if chunk_path.exists() {
                            let mut cf = BufReader::new(File::open(&chunk_path)?);
                            let chunk = read_chunk(&mut cf)?;
                            chunks.push((voxel_core::ChunkPos::new(cx, cy, cz), chunk));
                        }
                    }
                }
            }
        }
    }

    log::info!("Loaded {} chunks from {}", chunks.len(), path.display());
    Ok((seed, chunks))
}

fn chunk_dir(base: &Path, cx: i32, cy: i32, cz: i32) -> std::path::PathBuf {
    base.join(format!("c_{cx}_{cy}_{cz}")).join("chunk.bin")
}

/// Write a single chunk to a writer (compressed).
fn write_chunk(chunk: &Chunk, writer: &mut impl Write) -> Result<()> {
    let mut encoder = DeflateEncoder::new(writer, Compression::fast());

    let palette_mode = chunk.is_palette_mode();
    encoder.write_all(&(palette_mode as u8).to_le_bytes())?;

    if palette_mode {
        // Write palette.
        let pal = chunk.palette_data();
        if pal.len() > 4096 {
            bail!("palette size {} exceeds 4096", pal.len());
        }
        encoder.write_all(&(pal.len() as u16).to_le_bytes())?;
        for id in pal {
            encoder.write_all(&id.0.to_le_bytes())?;
        }
        // Write indices.
        let indices = chunk.indices_data();
        encoder.write_all(indices)?;
    } else {
        // Write flat block data as u16 LE.
        let blocks = chunk.blocks();
        for b in blocks {
            encoder.write_all(&b.0.to_le_bytes())?;
        }
    }

    // Write sunlight, torchlight, water_level.
    encoder.write_all(chunk.sunlight_data())?;
    encoder.write_all(chunk.torchlight_data())?;
    encoder.write_all(chunk.water_level_data())?;

    // Write packed RGBA8 torchlight colour (v3+). Saved as little-endian
    // u32 × 4096 so a bytemuck::cast_slice of &[u32] produces the exact
    // byte pattern `R8G8B8A8_UNORM` upload expects.
    let tc_bytes: &[u8] = bytemuck::cast_slice(chunk.torchlight_color_data());
    encoder.write_all(tc_bytes)?;

    encoder.finish()?;
    Ok(())
}

/// Read a single chunk from a reader (compressed).
fn read_chunk(reader: &mut impl Read) -> Result<Chunk> {
    let mut decoder = DeflateDecoder::new(reader);

    let mut mode_buf = [0u8; 1];
    decoder.read_exact(&mut mode_buf)?;
    let palette_mode = match mode_buf[0] {
        0 => false,
        1 => true,
        other => bail!("unknown chunk mode {other}"),
    };

    let mut chunk = Chunk::new(voxel_core::ChunkPos::new(0, 0, 0));

    if palette_mode {
        let mut pal_len_buf = [0u8; 2];
        decoder.read_exact(&mut pal_len_buf)?;
        let pal_len = u16::from_le_bytes(pal_len_buf) as usize;

        let mut palette = Vec::with_capacity(pal_len);
        for _ in 0..pal_len {
            let mut id_buf = [0u8; 2];
            decoder.read_exact(&mut id_buf)?;
            palette.push(voxel_core::BlockId(u16::from_le_bytes(id_buf)));
        }

        let mut indices = vec![0u8; voxel_core::CHUNK_CUBED];
        decoder.read_exact(&mut indices)?;

        chunk.restore_palette(palette, indices);
    } else {
        let mut blocks = vec![voxel_core::BlockId(0); voxel_core::CHUNK_CUBED];
        for b in blocks.iter_mut() {
            let mut id_buf = [0u8; 2];
            decoder.read_exact(&mut id_buf)?;
            *b = voxel_core::BlockId(u16::from_le_bytes(id_buf));
        }
        chunk.restore_flat(blocks);
    }

    // Read sunlight.
    let mut sunlight = vec![0u8; voxel_core::CHUNK_CUBED];
    decoder.read_exact(&mut sunlight)?;
    for v in sunlight.iter_mut() {
        *v = (*v).min(15);
    }
    chunk.restore_sunlight(sunlight);

    // Read torchlight.
    let mut torchlight = vec![0u8; voxel_core::CHUNK_CUBED];
    decoder.read_exact(&mut torchlight)?;
    for v in torchlight.iter_mut() {
        *v = (*v).min(15);
    }
    chunk.restore_torchlight(torchlight);

    // Read water_level.
    let mut water_level = vec![0u8; voxel_core::CHUNK_CUBED];
    decoder.read_exact(&mut water_level)?;
    for v in water_level.iter_mut() {
        *v = (*v).min(8);
    }
    chunk.restore_water_level(water_level);

    // Read packed RGBA8 torchlight colour (added in v3). Older v2 saves
    // don't have this field — when EOF hits before we've read it, we leave
    // the colour at all-zero (warm white via `Chunk::get_combined_light_color`).
    //
    // We use `try_read_exact`-style fallback: read into a stack buffer, only
    // commit if we got all the bytes. This makes v2 loads degrade gracefully
    // instead of erroring on EOF.
    let mut tc_bytes = vec![0u8; voxel_core::CHUNK_CUBED * 4];
    match decoder.read_exact(&mut tc_bytes) {
        Ok(()) => {
            let mut tc = vec![0u32; voxel_core::CHUNK_CUBED];
            bytemuck::cast_slice_mut(&mut tc[..]).copy_from_slice(&tc_bytes);
            chunk.restore_torchlight_color(tc);
        }
        Err(_) => {
            // EOF / partial read: assume v2 layout; leave colour as zero.
            log::debug!(
                "save v2 chunk detected (no torchlight_color field); restoring with warm-white default"
            );
            chunk.restore_torchlight_color(vec![0u32; voxel_core::CHUNK_CUBED]);
        }
    }

    Ok(chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::{BlockId, ChunkPos};

    #[test]
    fn save_load_roundtrip_flat() {
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        chunk.set(0, 0, 0, BlockId(2));
        chunk.set(5, 10, 15, BlockId(5));

        let mut buf = Vec::new();
        write_chunk(&chunk, &mut buf).unwrap();

        let loaded = read_chunk(&mut buf.as_slice()).unwrap();
        assert_eq!(loaded.get(0, 0, 0), BlockId(2));
        assert_eq!(loaded.get(5, 10, 15), BlockId(5));
        assert!(loaded.get(1, 0, 0).is_air());
    }

    #[test]
    fn save_load_roundtrip_palette() {
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        // Fill with a few block types to trigger palette mode.
        for i in 0..20 {
            chunk.set(i, 0, 0, BlockId(3));
        }
        chunk.set(0, 1, 0, BlockId(7));

        let mut buf = Vec::new();
        write_chunk(&chunk, &mut buf).unwrap();

        let loaded = read_chunk(&mut buf.as_slice()).unwrap();
        assert_eq!(loaded.get(0, 0, 0), BlockId(3));
        assert_eq!(loaded.get(5, 0, 0), BlockId(3));
        assert_eq!(loaded.get(0, 1, 0), BlockId(7));
    }

    #[test]
    fn save_load_light_and_water() {
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        chunk.set_sunlight(3, 3, 3, 15);
        chunk.set_torchlight(5, 5, 5, 12);
        chunk.set_water_level(7, 7, 7, 6);

        let mut buf = Vec::new();
        write_chunk(&chunk, &mut buf).unwrap();

        let loaded = read_chunk(&mut buf.as_slice()).unwrap();
        assert_eq!(loaded.get_sunlight(3, 3, 3), 15);
        assert_eq!(loaded.get_torchlight(5, 5, 5), 12);
        assert_eq!(loaded.get_water_level(7, 7, 7), 6);
    }

    #[test]
    fn save_load_torchlight_color_roundtrip_v3() {
        // Regression test for v3 save format: packed RGBA8 torchlight color
        // per voxel must survive write_chunk/read_chunk unchanged so colored
        // lighting actually persists across save/load (previously `torchlight_color`
        // was completely absent from the save file, so reloaded worlds reverted
        // every voxel to warm white).
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        // A few saturated test colors. Each is packed RGBA8 little-endian.
        let blue = crate::chunk::pack_rgb(60, 120, 255);
        let red = crate::chunk::pack_rgb(255, 40, 40);
        let green = crate::chunk::pack_rgb(40, 255, 80);
        chunk.set_torchlight_color(2, 3, 4, blue);
        chunk.set_torchlight_color(5, 6, 7, red);
        chunk.set_torchlight_color(8, 9, 10, green);
        // Pair one cell with a non-zero torchlight so combined-light lookup
        // picks torchlight over the warm-white default.
        chunk.set_torchlight(2, 3, 4, 12);
        chunk.set_torchlight(5, 6, 7, 8);
        chunk.set_torchlight(8, 9, 10, 6);

        let mut buf = Vec::new();
        write_chunk(&chunk, &mut buf).unwrap();

        let loaded = read_chunk(&mut buf.as_slice()).unwrap();
        assert_eq!(
            loaded.get_torchlight_color(2, 3, 4),
            blue,
            "blue torchlight color lost"
        );
        assert_eq!(
            loaded.get_torchlight_color(5, 6, 7),
            red,
            "red torchlight color lost"
        );
        assert_eq!(
            loaded.get_torchlight_color(8, 9, 10),
            green,
            "green torchlight color lost"
        );
        // Untouched cells must still be the 0-sentinel (warm-white fallback).
        assert_eq!(loaded.get_torchlight_color(0, 0, 0), 0);
        // Combined-light lookup should return the explicit color when
        // torchlight dominates, warm white when sunlight dominates.
        assert_eq!(loaded.get_combined_light_color(2, 3, 4), blue);
        assert_eq!(loaded.get_combined_light_color(5, 6, 7), red);
    }

    #[test]
    fn save_load_world_roundtrip() {
        let dir = std::env::temp_dir().join("voxel_save_test");
        let _ = fs::remove_dir_all(&dir);

        let world = World::new(42);
        // Insert a chunk.
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        chunk.set(1, 2, 3, BlockId(5));
        world.insert_chunk(cp, chunk);

        save_world(&world, &dir).unwrap();
        let (seed, chunks) = load_world(&dir).unwrap();

        assert_eq!(seed, 42);
        assert_eq!(chunks.len(), 1);
        let (loaded_cp, loaded_chunk) = &chunks[0];
        assert_eq!(*loaded_cp, cp);
        assert_eq!(loaded_chunk.get(1, 2, 3), BlockId(5));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_load_large_palette() {
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0));
        for i in 0..300u16 {
            chunk.set(
                (i % 16) as i32,
                ((i / 16) % 16) as i32,
                (i / 256) as i32,
                BlockId(i + 1),
            );
        }
        let mut buf = Vec::new();
        write_chunk(&chunk, &mut buf).unwrap();
        let loaded = read_chunk(&mut buf.as_slice()).unwrap();
        for i in 0..300u16 {
            let x = (i % 16) as i32;
            let y = ((i / 16) % 16) as i32;
            let z = (i / 256) as i32;
            assert_eq!(
                loaded.get(x, y, z),
                BlockId(i + 1),
                "mismatch at ({x},{y},{z})"
            );
        }
    }

    #[test]
    fn read_chunk_invalid_mode_errors() {
        // Write a chunk with mode byte = 99 (invalid).
        let buf: Vec<u8> = vec![99u8];
        assert!(read_chunk(&mut buf.as_slice()).is_err());
    }

    #[test]
    fn read_chunk_truncated_errors() {
        // Empty input.
        let buf: Vec<u8> = vec![];
        assert!(read_chunk(&mut buf.as_slice()).is_err());
    }
}
