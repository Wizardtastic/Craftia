use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::mipmap::generate_mip_chain;
use crate::texture_pack;
use voxel_core::{ATLAS_PIXELS, ATLAS_TILES, ATLAS_TILE_SIZE};

/// Magic bytes identifying the cache format.
const MAGIC: &[u8; 20] = b"VOXEL_ASSET_CACHE_v1";

/// Cache schema version.
const SCHEMA_VERSION: u32 = 1;

/// Atlas format: raw RGBA8.
const FORMAT_RGBA8: u8 = 0;

/// Status returned by `process_assets`.
pub enum CacheStatus {
    /// Cache is up to date. Contains the atlas RGBA8 pixels and mip chain.
    Hit {
        atlas_rgba: Vec<u8>,
        mip_chain: Vec<Vec<u8>>,
        width: u32,
        height: u32,
    },
    /// Cache was stale or missing. Contains the newly built atlas + mip chain.
    Miss {
        atlas_rgba: Vec<u8>,
        mip_chain: Vec<Vec<u8>>,
        width: u32,
        height: u32,
    },
}

impl CacheStatus {
    pub fn atlas_rgba(&self) -> &[u8] {
        match self {
            CacheStatus::Hit { atlas_rgba, .. } => atlas_rgba,
            CacheStatus::Miss { atlas_rgba, .. } => atlas_rgba,
        }
    }

    pub fn mip_chain(&self) -> &[Vec<u8>] {
        match self {
            CacheStatus::Hit { mip_chain, .. } => mip_chain,
            CacheStatus::Miss { mip_chain, .. } => mip_chain,
        }
    }

    pub fn width(&self) -> u32 {
        match self {
            CacheStatus::Hit { width, .. } => *width,
            CacheStatus::Miss { width, .. } => *width,
        }
    }

    pub fn height(&self) -> u32 {
        match self {
            CacheStatus::Hit { height, .. } => *height,
            CacheStatus::Miss { height, .. } => *height,
        }
    }

    pub fn is_hit(&self) -> bool {
        matches!(self, CacheStatus::Hit { .. })
    }
}

/// A single entry in the source manifest: hash of the path + hash of the content.
#[derive(Clone, Debug)]
struct ManifestEntry {
    path_hash: [u8; 32],
    content_hash: [u8; 32],
}

/// Compute BLAKE3 hash of a file's contents.
fn hash_file(path: &Path) -> Result<[u8; 32]> {
    let bytes =
        std::fs::read(path).with_context(|| format!("hash_file: read {}", path.display()))?;
    Ok(blake3::hash(&bytes).into())
}

/// Compute BLAKE3 hash of a string (for path hashing).
fn hash_string(s: &str) -> [u8; 32] {
    blake3::hash(s.as_bytes()).into()
}

/// Compute composite hash over sorted manifest entries.
fn composite_hash(entries: &[ManifestEntry]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    let mut sorted = entries.to_vec();
    sorted.sort_by_key(|e| e.path_hash);
    for entry in &sorted {
        hasher.update(&entry.path_hash);
        hasher.update(&entry.content_hash);
    }
    hasher.finalize().into()
}

/// Build the source manifest from textures.toml + referenced PNGs.
fn build_manifest(textures_dir: &Path) -> Result<(HashMap<u32, String>, Vec<ManifestEntry>)> {
    let mapping = load_texture_config(textures_dir);
    let mut entries = Vec::new();

    // Hash textures.toml itself.
    let config_path = textures_dir.join("textures.toml");
    let config_hash = hash_file(&config_path)?;
    entries.push(ManifestEntry {
        path_hash: hash_string("textures.toml"),
        content_hash: config_hash,
    });

    // Hash each referenced PNG.
    for (tile_index, filename) in &mapping {
        let png_path = textures_dir.join(filename);
        let _ = tile_index; // used for ordering only
        if png_path.exists() {
            entries.push(ManifestEntry {
                path_hash: hash_string(filename),
                content_hash: hash_file(&png_path)?,
            });
        }
    }

    Ok((mapping, entries))
}

/// Cache file path: `<textures_dir>/.cache/atlas.cache`
fn cache_path(textures_dir: &Path) -> PathBuf {
    textures_dir.join(".cache").join("atlas.cache")
}

/// Read and validate an existing cache file. Returns None if missing or invalid.
fn read_cache(
    textures_dir: &Path,
    expected_composite: &[u8; 32],
) -> Option<(Vec<u8>, Vec<Vec<u8>>)> {
    let path = cache_path(textures_dir);
    let bytes = std::fs::read(&path).ok()?;

    let mut cursor = std::io::Cursor::new(bytes.as_slice());

    // Magic.
    let mut magic = [0u8; 20];
    cursor.read_exact(&mut magic).ok()?;
    if magic != *MAGIC {
        return None;
    }

    // Schema version.
    let version = read_u32(&mut cursor)?;
    if version != SCHEMA_VERSION {
        return None;
    }

    // Format.
    let format = read_u8(&mut cursor)?;
    if format != FORMAT_RGBA8 {
        return None;
    }

    // Source manifest.
    let count = read_u32(&mut cursor)?;
    let mut stored_entries = Vec::new();
    for _ in 0..count {
        let mut path_hash = [0u8; 32];
        let mut content_hash = [0u8; 32];
        cursor.read_exact(&mut path_hash).ok()?;
        cursor.read_exact(&mut content_hash).ok()?;
        stored_entries.push(ManifestEntry {
            path_hash,
            content_hash,
        });
    }

    // Composite hash.
    let mut stored_composite = [0u8; 32];
    cursor.read_exact(&mut stored_composite).ok()?;

    if stored_composite != *expected_composite {
        return None;
    }

    // Atlas metadata.
    let _width = read_u32(&mut cursor)?;
    let _height = read_u32(&mut cursor)?;
    let _tile_size = read_u32(&mut cursor)?;
    let _tiles_x = read_u32(&mut cursor)?;
    let _tiles_y = read_u32(&mut cursor)?;
    let mip_count = read_u32(&mut cursor)?;
    let _format = read_u8(&mut cursor)?;

    // Mip data.
    let mut mip_chain = Vec::with_capacity(mip_count as usize);
    for _ in 0..mip_count {
        let size = read_u32(&mut cursor)? as usize;
        let mut data = vec![0u8; size];
        cursor.read_exact(&mut data).ok()?;
        mip_chain.push(data);
    }

    let atlas_rgba = mip_chain.first()?.clone();
    Some((atlas_rgba, mip_chain))
}

/// Write cache file.
fn write_cache(
    textures_dir: &Path,
    composite: &[u8; 32],
    mip_chain: &[Vec<u8>],
    width: u32,
    height: u32,
    entries: &[ManifestEntry],
) -> Result<()> {
    let dir = textures_dir.join(".cache");
    std::fs::create_dir_all(&dir)?;
    let path = cache_path(textures_dir);

    let mut out = Vec::new();

    // Magic.
    out.write_all(MAGIC)?;

    // Schema version.
    write_u32(&mut out, SCHEMA_VERSION);

    // Format.
    write_u8(&mut out, FORMAT_RGBA8);

    // Source manifest.
    write_u32(&mut out, entries.len() as u32);
    for entry in entries {
        out.write_all(&entry.path_hash)?;
        out.write_all(&entry.content_hash)?;
    }

    // Composite hash.
    out.write_all(composite)?;

    // Atlas metadata.
    let tile_size = ATLAS_TILE_SIZE;
    let tiles_x = ATLAS_TILES;
    let tiles_y = ATLAS_TILES;
    write_u32(&mut out, width);
    write_u32(&mut out, height);
    write_u32(&mut out, tile_size);
    write_u32(&mut out, tiles_x);
    write_u32(&mut out, tiles_y);
    write_u32(&mut out, mip_chain.len() as u32);
    write_u8(&mut out, FORMAT_RGBA8);

    // Mip data.
    for mip in mip_chain {
        write_u32(&mut out, mip.len() as u32);
        out.write_all(mip)?;
    }

    std::fs::write(&path, &out).with_context(|| format!("write_cache: {}", path.display()))?;
    log::info!("asset cache written to {}", path.display());
    Ok(())
}

/// Process assets: check cache, rebuild if stale.
pub fn process_assets(textures_dir: &Path) -> Result<CacheStatus> {
    process_assets_with_packs(textures_dir, None, None)
}

/// Process assets with optional texture packs merged in.
///
/// When `pack_mapping` is provided, it is merged on top of the base
/// `textures_dir` mapping so texture pack tiles override the originals.
/// Pack zip file hashes are included in the cache manifest so pack
/// changes trigger a rebuild.
pub fn process_assets_with_packs(
    textures_dir: &Path,
    pack_mapping: Option<&HashMap<u32, String>>,
    packs_dir: Option<&Path>,
) -> Result<CacheStatus> {
    let (mut mapping, mut manifest) = build_manifest(textures_dir)?;
    // Layer texture pack tiles on top of the base mapping.
    if let Some(packs) = pack_mapping {
        for (tile_index, filename) in packs {
            mapping.insert(*tile_index, filename.clone());
        }
    }
    // Include texture pack zip file hashes in the manifest so the cache
    // is invalidated when any pack changes.
    if let Some(pdir) = packs_dir {
        if let Ok(entries) = texture_pack::hash_pack_files(pdir) {
            for (path_hash, content_hash) in entries {
                manifest.push(ManifestEntry {
                    path_hash,
                    content_hash,
                });
            }
        }
    }
    let composite = composite_hash(&manifest);

    // Try cache.
    if let Some((atlas_rgba, mip_chain)) = read_cache(textures_dir, &composite) {
        let width = ATLAS_TILES * ATLAS_TILE_SIZE ;
        let height = width;
        log::info!("asset cache hit ({} tiles)", mapping.len());
        return Ok(CacheStatus::Hit {
            atlas_rgba,
            mip_chain,
            width,
            height,
        });
    }

    // Cache miss: build atlas from scratch.
    log::info!(
        "asset cache miss — rebuilding atlas from {} tiles",
        mapping.len()
    );
    let atlas = crate::atlas::build_atlas(textures_dir, &mapping);
    let mip_chain = generate_mip_chain(&atlas, ATLAS_PIXELS, ATLAS_PIXELS);

    // Write cache.
    if let Err(e) = write_cache(
        textures_dir,
        &composite,
        &mip_chain,
        ATLAS_PIXELS,
        ATLAS_PIXELS,
        &manifest,
    ) {
        log::warn!("failed to write asset cache: {e}");
    }

    let atlas_rgba = mip_chain[0].clone();
    Ok(CacheStatus::Miss {
        atlas_rgba,
        mip_chain,
        width: ATLAS_PIXELS,
        height: ATLAS_PIXELS,
    })
}

/// Validate cache freshness without rebuilding. Returns true if cache is up to date.
pub fn check_cache(textures_dir: &Path) -> Result<bool> {
    let (_, manifest) = build_manifest(textures_dir)?;
    let composite = composite_hash(&manifest);
    Ok(read_cache(textures_dir, &composite).is_some())
}

/// Read textures.toml and return tile_index -> filename mapping.
fn load_texture_config(textures_dir: &Path) -> HashMap<u32, String> {
    texture_pack::load_texture_config(textures_dir)
}

// --- Binary helpers ---

fn read_u32(cursor: &mut std::io::Cursor<&[u8]>) -> Option<u32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf).ok()?;
    Some(u32::from_le_bytes(buf))
}

fn read_u8(cursor: &mut std::io::Cursor<&[u8]>) -> Option<u8> {
    let mut buf = [0u8; 1];
    cursor.read_exact(&mut buf).ok()?;
    Some(buf[0])
}

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_deterministic() {
        let h1 = hash_string("test.png");
        let h2 = hash_string("test.png");
        assert_eq!(h1, h2);
    }

    #[test]
    fn composite_deterministic() {
        let entries = vec![
            ManifestEntry {
                path_hash: hash_string("a.png"),
                content_hash: hash_string("content_a"),
            },
            ManifestEntry {
                path_hash: hash_string("b.png"),
                content_hash: hash_string("content_b"),
            },
        ];
        let c1 = composite_hash(&entries);
        let c2 = composite_hash(&entries);
        assert_eq!(c1, c2);
    }

    #[test]
    fn composite_order_independent() {
        let e1 = vec![
            ManifestEntry {
                path_hash: hash_string("b.png"),
                content_hash: hash_string("cb"),
            },
            ManifestEntry {
                path_hash: hash_string("a.png"),
                content_hash: hash_string("ca"),
            },
        ];
        let mut e2 = e1.clone();
        e2.reverse();
        assert_eq!(composite_hash(&e1), composite_hash(&e2));
    }

    #[test]
    fn roundtrip_cache_format() {
        let mip_chain = vec![vec![1u8, 2, 3, 4], vec![5u8, 6, 7, 8]];
        let entries = vec![ManifestEntry {
            path_hash: [1u8; 32],
            content_hash: [2u8; 32],
        }];
        let composite = composite_hash(&entries);

        // Write to temp dir.
        let dir = std::env::temp_dir().join("voxel_asset_cache_test");
        let _ = std::fs::create_dir_all(dir.join(".cache"));
        let _ = write_cache(&dir, &composite, &mip_chain, 4, 4, &entries);

        // Read back.
        let result = read_cache(&dir, &composite);
        assert!(result.is_some());
        let (atlas, read_mips) = result.unwrap();
        assert_eq!(atlas, vec![1u8, 2, 3, 4]);
        assert_eq!(read_mips.len(), 2);

        // Cleanup.
        let _ = std::fs::remove_dir_all(&dir);
    }
}
