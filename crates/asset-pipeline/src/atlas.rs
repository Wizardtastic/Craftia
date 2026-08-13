//! Atlas construction: fill every tile with an error pattern, then overlay
//! PNGs loaded from a tile mapping. Adapted from `render/src/atlas.rs`.

use std::collections::HashMap;
use std::path::Path;

use image::imageops::FilterType;
use voxel_core::{ATLAS_PIXELS, ATLAS_TILES, ATLAS_TILE_SIZE};

/// Build the atlas RGBA8 buffer from a textures directory and tile mapping.
/// Returns the raw RGBA8 pixel data (ATLAS_PIXELS x ATLAS_PIXELS x 4 bytes).
pub fn build_atlas(textures_dir: &Path, mapping: &HashMap<u32, String>) -> Vec<u8> {
    let total_tiles = (ATLAS_TILES * ATLAS_TILES) as usize;
    let mut rgba = vec![0u8; (ATLAS_PIXELS * ATLAS_PIXELS * 4) as usize];

    // Fill every tile with the error texture first.
    for tile in 0..total_tiles as u32 {
        fill_error_tile(&mut rgba, tile);
    }

    let mut loaded = 0;
    let mut missing = Vec::new();

    for (tile_index, filename) in mapping {
        if *tile_index >= total_tiles as u32 {
            log::warn!(
                "texture config: tile index {} out of range (0..{}), skipping '{}'",
                tile_index,
                total_tiles,
                filename
            );
            continue;
        }
        let png_path = textures_dir.join(filename);
        match load_png_into_atlas(&mut rgba, *tile_index, &png_path) {
            Ok(()) => loaded += 1,
            Err(e) => {
                log::warn!("failed to load texture {}: {}", png_path.display(), e);
                missing.push(*tile_index);
            }
        }
    }

    if loaded > 0 {
        log::info!(
            "loaded {}/{} textures from {}",
            loaded,
            mapping.len(),
            textures_dir.display()
        );
    }
    if !missing.is_empty() {
        log::warn!(
            "{} texture(s) still missing - will show error pattern: {:?}",
            missing.len(),
            missing
        );
    }
    if mapping.is_empty() {
        log::warn!(
            "no textures.toml found in {} — all tiles show error pattern",
            textures_dir.display()
        );
    }

    rgba
}

/// Write a pixel into a tile at tile-local (tx, ty).
fn put(atlas: &mut [u8], tile: u32, tx: u32, ty: u32, color: [u8; 4]) {
    let tile_x = (tile % ATLAS_TILES) * ATLAS_TILE_SIZE;
    let tile_y = (tile / ATLAS_TILES) * ATLAS_TILE_SIZE;
    let px = tile_x + tx;
    let py = tile_y + ty;
    let idx = ((py * ATLAS_PIXELS + px) * 4) as usize;
    atlas[idx] = color[0];
    atlas[idx + 1] = color[1];
    atlas[idx + 2] = color[2];
    atlas[idx + 3] = color[3];
}

/// Fill a tile with the blue+black checkerboard error pattern.
fn fill_error_tile(atlas: &mut [u8], tile: u32) {
    for ty in 0..ATLAS_TILE_SIZE {
        for tx in 0..ATLAS_TILE_SIZE {
            let checker = ((tx / 2) + (ty / 2)) % 2 == 0;
            let (r, g, b) = if checker {
                (0, 60, 220) // blue
            } else {
                (0, 0, 0) // black
            };
            put(atlas, tile, tx, ty, [r, g, b, 255]);
        }
    }
}

/// Decode a PNG file and write its pixels into the given tile in the atlas.
fn load_png_into_atlas(atlas: &mut [u8], tile: u32, path: &Path) -> Result<(), String> {
    let img = image::open(path).map_err(|e| format!("decode: {e}"))?;
    let resized = img.resize_exact(ATLAS_TILE_SIZE, ATLAS_TILE_SIZE, FilterType::Nearest);
    let rgba_img = resized.to_rgba8();
    let tile_x = (tile % ATLAS_TILES) * ATLAS_TILE_SIZE;
    let tile_y = (tile / ATLAS_TILES) * ATLAS_TILE_SIZE;
    for ty in 0..ATLAS_TILE_SIZE {
        for tx in 0..ATLAS_TILE_SIZE {
            let pixel = rgba_img.get_pixel(tx, ty);
            let px = tile_x + tx;
            let py = tile_y + ty;
            let idx = ((py * ATLAS_PIXELS + px) * 4) as usize;
            atlas[idx] = pixel[0];
            atlas[idx + 1] = pixel[1];
            atlas[idx + 2] = pixel[2];
            atlas[idx + 3] = pixel[3];
        }
    }
    Ok(())
}
