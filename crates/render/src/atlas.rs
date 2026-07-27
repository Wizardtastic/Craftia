//! Name-based texture atlas with PNG file loading and BLAKE3-cached
//! preprocessing.
//!
//! `build_atlas_with_textures()` delegates to `voxel_asset_pipeline::process_assets()`
//! which checks a content-addressed cache and only rebuilds when source PNGs change.
//! On cache hit, the atlas is loaded from disk (~0ms); on miss, it is rebuilt
//! from scratch and the cache is written for next time.

use std::path::Path;

use image::imageops::FilterType;
use voxel_core::ATLAS_TILE_SIZE;

/// Atlas side length in tiles (16x16 = 256 tiles). Single source of truth.
pub use voxel_core::ATLAS_TILES;
/// Atlas side length in pixels.
pub const ATLAS_PIXELS: u32 = ATLAS_TILES * ATLAS_TILE_SIZE;

/// A finished atlas: RGBA8 pixels + mip chain ready to upload to a Vulkan image.
pub struct Atlas {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub mip_chain: Vec<Vec<u8>>,
    pub mip_levels: u32,
}

/// Build the atlas from `textures_dir`, using the asset pipeline cache.
///
/// On cache hit, returns immediately with the cached atlas + mip chain.
/// On cache miss, rebuilds from PNGs, generates mip chain, writes cache.
pub fn build_atlas_with_textures(textures_dir: &Path) -> Atlas {
    match voxel_asset_pipeline::process_assets(textures_dir) {
        Ok(status) => {
            let width = status.width();
            let height = status.height();
            let mip_levels = status.mip_chain().len() as u32;
            let rgba = status.atlas_rgba().to_vec();
            let mip_chain = status.mip_chain().to_vec();

            if status.is_hit() {
                log::info!("atlas loaded from cache ({} mip levels)", mip_levels);
            } else {
                log::info!(
                    "atlas rebuilt from PNGs ({} mip levels, {}x{})",
                    mip_levels,
                    width,
                    height
                );
            }

            Atlas {
                width,
                height,
                rgba,
                mip_chain,
                mip_levels,
            }
        }
        Err(e) => {
            log::warn!("asset pipeline failed: {e}, falling back to direct build");
            build_atlas_fallback(textures_dir)
        }
    }
}

/// Fallback: build atlas directly without the cache (if asset pipeline fails).
fn build_atlas_fallback(textures_dir: &Path) -> Atlas {
    use image::imageops::FilterType;

    let total_tiles = (ATLAS_TILES * ATLAS_TILES) as usize;
    let mut rgba = vec![0u8; (ATLAS_PIXELS * ATLAS_PIXELS * 4) as usize];

    // Fill with error pattern.
    for tile in 0..total_tiles as u32 {
        fill_error_tile(&mut rgba, tile);
    }

    let mapping = load_texture_config(textures_dir);
    for (tile_index, filename) in &mapping {
        if *tile_index >= total_tiles as u32 {
            continue;
        }
        let png_path = textures_dir.join(filename);
        if let Err(e) = load_png_into_atlas(&mut rgba, *tile_index, &png_path, FilterType::Nearest)
        {
            log::warn!("fallback: failed to load {}: {e}", png_path.display());
        }
    }

    // Generate simple mip chain (single-level for fallback).
    let mip_chain = vec![rgba.clone()];
    let mip_levels = 1;

    Atlas {
        width: ATLAS_PIXELS,
        height: ATLAS_PIXELS,
        rgba,
        mip_chain,
        mip_levels,
    }
}

fn fill_error_tile(atlas: &mut [u8], tile: u32) {
    for ty in 0..ATLAS_TILE_SIZE {
        for tx in 0..ATLAS_TILE_SIZE {
            let checker = ((tx / 2) + (ty / 2)) % 2 == 0;
            let (r, g, b) = if checker {
                (0, 60, 220)
            } else {
                (0, 0, 0)
            };
            let tile_x = (tile % ATLAS_TILES) * ATLAS_TILE_SIZE;
            let tile_y = (tile / ATLAS_TILES) * ATLAS_TILE_SIZE;
            let px = tile_x + tx;
            let py = tile_y + ty;
            let idx = ((py * ATLAS_PIXELS + px) * 4) as usize;
            atlas[idx] = r;
            atlas[idx + 1] = g;
            atlas[idx + 2] = b;
            atlas[idx + 3] = 255;
        }
    }
}

fn load_png_into_atlas(
    atlas: &mut [u8],
    tile: u32,
    path: &Path,
    filter: FilterType,
) -> Result<(), String> {
    let img = image::open(path).map_err(|e| format!("decode: {e}"))?;
    let resized = img.resize_exact(ATLAS_TILE_SIZE, ATLAS_TILE_SIZE, filter);
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

fn load_texture_config(textures_dir: &Path) -> std::collections::HashMap<u32, String> {
    use std::collections::HashMap;
    let config_path = textures_dir.join("textures.toml");
    let content = match std::fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(_) => return HashMap::new(),
    };
    let value: toml::Value = match content.parse() {
        Ok(v) => v,
        Err(e) => {
            log::warn!("failed to parse {}: {}", config_path.display(), e);
            return HashMap::new();
        }
    };
    let tiles = match value.get("tiles").and_then(|v| v.as_table()) {
        Some(t) => t,
        None => return HashMap::new(),
    };
    let mut map = HashMap::new();
    for (key, val) in tiles {
        if let (Ok(index), Some(filename)) = (key.parse::<u32>(), val.as_str()) {
            map.insert(index, filename.to_string());
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_constants() {
        assert_eq!(ATLAS_TILES, 16);
        assert_eq!(ATLAS_PIXELS, 256);
    }
}
