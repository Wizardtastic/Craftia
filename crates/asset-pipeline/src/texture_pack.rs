//! Texture pack support: loads block textures from `.zip` files and merges
//! them with the base `textures_dir` mapping.
//!
//! ## Texture Pack Format
//!
//! A texture pack is a standard `.zip` archive containing:
//!
//! ```text
//! my_texture_pack.zip
//! ├── textures.toml          ← optional; tile→PNG mapping (same format as base)
//! ├── textures/              ← optional; PNG files (flat or sub-dirs)
//! │   ├── stone.png
//!   │   ├── grass_top.png
//!   │   └── ...
//! └── pack.toml              ← optional; pack metadata
//! ```
//!
//! **Resolution order**: texture pack textures override the base `textures_dir`
//! textures on a per-tile basis. When multiple packs are loaded, later packs
//! in the list win.
//!
//! The pack may contain either:
//! 1. A `textures.toml` + `textures/` subdirectory (matches the base layout)
//! 2. PNG files directly in the root or a `textures/` subdirectory
//!    (filenames must match those referenced by the base `textures.toml`)

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Metadata parsed from `pack.toml` inside a texture pack zip.
#[derive(Clone, Debug, Default)]
pub struct PackMetadata {
    /// Human-readable pack name.
    pub name: String,
    /// Pack description.
    pub description: String,
    /// Semantic version string.
    pub version: String,
    /// Author name.
    pub author: String,
    /// Number of overridden tiles.
    pub tile_count: usize,
}

/// An animated texture definition parsed from `animations.toml` in a pack.
#[derive(Clone, Debug)]
pub struct PackAnimationDef {
    /// Tile index that this animation overrides.
    pub tile_index: u32,
    /// Frame tile indices (each entry is a tile in the atlas).
    pub frames: Vec<u32>,
    /// Duration per frame in seconds.
    pub frame_duration: f32,
}

/// A loaded texture pack: owns the extracted tile→filename mapping and
/// a temporary directory containing the extracted PNGs.
pub struct TexturePack {
    /// Human-readable name (derived from the zip filename).
    pub name: String,
    /// Path to the temporary directory where PNGs were extracted.
    extract_dir: PathBuf,
    /// Tile index → extracted PNG filename mapping.
    mapping: HashMap<u32, String>,
    /// Pack metadata from pack.toml (if present).
    pub metadata: PackMetadata,
    /// Animation definitions from animations.toml (if present).
    pub animations: Vec<PackAnimationDef>,
}

impl TexturePack {
    /// The tile→filename mapping for this pack.
    pub fn mapping(&self) -> &HashMap<u32, String> {
        &self.mapping
    }

    /// The directory containing the extracted PNG files.
    pub fn extract_dir(&self) -> &Path {
        &self.extract_dir
    }

    /// The pack metadata (name, description, version, author).
    pub fn metadata(&self) -> &PackMetadata {
        &self.metadata
    }

    /// Animation definitions for this pack.
    pub fn animations(&self) -> &[PackAnimationDef] {
        &self.animations
    }
}

/// Load a texture pack from a `.zip` file.
///
/// Extracts PNG files and the optional `textures.toml` into a temp directory
/// under `<textures_dir>/.texture_packs/<pack_name>/`, then parses the tile
/// mapping.
pub fn load_texture_pack(zip_path: &Path, base_textures_dir: &Path) -> Result<TexturePack> {
    let pack_name = zip_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Ensure the zip exists.
    if !zip_path.is_file() {
        return Err(anyhow::anyhow!(
            "texture pack not found: {}",
            zip_path.display()
        ));
    }

    // Create extraction directory.
    let extract_dir = base_textures_dir.join(".texture_packs").join(&pack_name);

    // Clean previous extraction if it exists.
    if extract_dir.exists() {
        std::fs::remove_dir_all(&extract_dir).with_context(|| {
            format!(
                "failed to clean old extraction at {}",
                extract_dir.display()
            )
        })?;
    }
    std::fs::create_dir_all(&extract_dir).with_context(|| {
        format!(
            "failed to create extraction dir at {}",
            extract_dir.display()
        )
    })?;

    // Open the zip archive.
    let file = std::fs::File::open(zip_path)
        .with_context(|| format!("failed to open zip: {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("failed to read zip archive: {}", zip_path.display()))?;

    // Extract all files.
    let mut toml_content: Option<String> = None;
    let mut pack_toml_content: Option<String> = None;
    let mut animations_toml_content: Option<String> = None;
    let mut extracted_files: Vec<String> = Vec::new();

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .with_context(|| format!("failed to read zip entry {}", i))?;

        let entry_name = entry.name().to_string();

        // Skip directories.
        if entry.is_dir() {
            // Create subdirectories as needed.
            let dir_path = extract_dir.join(&entry_name);
            std::fs::create_dir_all(&dir_path)
                .with_context(|| format!("failed to create dir: {}", dir_path.display()))?;
            continue;
        }

        // Extract the file.
        let out_path = extract_dir.join(&entry_name);

        // Create parent directories.
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create parent dir: {}", parent.display()))?;
        }

        let mut out_file = std::fs::File::create(&out_path)
            .with_context(|| format!("failed to create file: {}", out_path.display()))?;

        let mut content = Vec::new();
        entry
            .read_to_end(&mut content)
            .with_context(|| format!("failed to read zip entry: {}", entry_name))?;

        std::io::Write::write_all(&mut out_file, &content)
            .with_context(|| format!("failed to write file: {}", out_path.display()))?;

        // Check if this is textures.toml (at any depth).
        if entry_name == "textures.toml" || entry_name.ends_with("/textures.toml") {
            toml_content = Some(String::from_utf8_lossy(&content).to_string());
        }

        // Check if this is pack.toml (at any depth).
        if entry_name == "pack.toml" || entry_name.ends_with("/pack.toml") {
            pack_toml_content = Some(String::from_utf8_lossy(&content).to_string());
        }

        // Check if this is animations.toml (at any depth).
        if entry_name == "animations.toml" || entry_name.ends_with("/animations.toml") {
            animations_toml_content = Some(String::from_utf8_lossy(&content).to_string());
        }

        // Track extracted PNG files.
        if entry_name.ends_with(".png") {
            // Get just the filename (not the full path in the zip).
            let filename = Path::new(&entry_name)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(&entry_name)
                .to_string();
            extracted_files.push(filename);
        }
    }

    // Parse the tile mapping.
    let mapping = if let Some(toml_str) = toml_content {
        parse_texture_pack_toml(&toml_str)
    } else {
        // No textures.toml in the pack — try to build mapping from extracted
        // PNGs by matching against the base textures_dir mapping.
        let base_mapping = load_texture_config(base_textures_dir);
        build_mapping_from_extracted(&extracted_files, &base_mapping)
    };

    // Parse pack metadata.
    let metadata = pack_toml_content
        .as_deref()
        .map(parse_pack_metadata)
        .unwrap_or_default();
    let metadata = PackMetadata {
        name: if metadata.name.is_empty() {
            pack_name.clone()
        } else {
            metadata.name
        },
        tile_count: mapping.len(),
        ..metadata
    };

    // Parse animation definitions.
    let animations = animations_toml_content
        .as_deref()
        .map(parse_pack_animations)
        .unwrap_or_default();

    log::info!(
        "loaded texture pack '{}' ({} tiles, {} animations, extracted to {})",
        pack_name,
        mapping.len(),
        animations.len(),
        extract_dir.display()
    );

    Ok(TexturePack {
        name: pack_name,
        extract_dir,
        mapping,
        metadata,
        animations,
    })
}

/// Unload a texture pack: remove the extracted files.
pub fn unload_texture_pack(pack: &TexturePack) -> Result<()> {
    if pack.extract_dir.exists() {
        std::fs::remove_dir_all(&pack.extract_dir).with_context(|| {
            format!("failed to remove pack dir: {}", pack.extract_dir.display())
        })?;
        log::info!("unloaded texture pack '{}'", pack.name);
    }
    Ok(())
}

/// Parse a `textures.toml` from a texture pack into a tile→filename mapping.
fn parse_texture_pack_toml(content: &str) -> HashMap<u32, String> {
    let value: toml::Value = match content.parse() {
        Ok(v) => v,
        Err(e) => {
            log::warn!("failed to parse texture pack textures.toml: {e}");
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

/// Build a mapping from extracted files by matching against the base mapping.
///
/// If a pack contains `stone.png` but no `textures.toml`, we map it to the
/// same tile index that `stone.png` has in the base `textures.toml`.
fn build_mapping_from_extracted(
    extracted: &[String],
    base_mapping: &HashMap<u32, String>,
) -> HashMap<u32, String> {
    let extracted_set: std::collections::HashSet<&str> =
        extracted.iter().map(|s| s.as_str()).collect();

    let mut map = HashMap::new();
    for (tile_index, filename) in base_mapping {
        if extracted_set.contains(filename.as_str()) {
            map.insert(*tile_index, filename.clone());
        }
    }
    map
}

/// Merge multiple texture pack mappings with the base mapping.
///
/// Later packs in the list override earlier ones and the base on a per-tile
/// basis. Returns the merged mapping.
pub fn merge_texture_pack_mappings(
    base_mapping: &HashMap<u32, String>,
    packs: &[&TexturePack],
) -> HashMap<u32, String> {
    // Start with the base mapping.
    let mut merged = base_mapping.clone();

    // Layer each pack on top.
    for pack in packs {
        for (tile_index, filename) in pack.mapping() {
            merged.insert(*tile_index, filename.clone());
        }
    }

    merged
}

/// Discover all `.zip` files in a directory, sorted by filename for deterministic
/// load order. Returns paths to valid zip files.
pub fn discover_texture_packs(packs_dir: &Path) -> Vec<PathBuf> {
    if !packs_dir.is_dir() {
        return Vec::new();
    }

    let mut packs: Vec<PathBuf> = std::fs::read_dir(packs_dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "zip"))
        .map(|entry| entry.path())
        .collect();

    packs.sort();
    packs
}

/// Load all texture packs from a directory and merge with the base mapping.
///
/// Returns the merged mapping and a list of loaded packs (for later cleanup).
pub fn load_all_texture_packs(
    packs_dir: &Path,
    base_textures_dir: &Path,
) -> Result<(HashMap<u32, String>, Vec<TexturePack>)> {
    let base_mapping = load_texture_config(base_textures_dir);
    let pack_paths = discover_texture_packs(packs_dir);

    let mut packs = Vec::new();
    for zip_path in &pack_paths {
        match load_texture_pack(zip_path, base_textures_dir) {
            Ok(pack) => packs.push(pack),
            Err(e) => {
                log::warn!("failed to load texture pack {}: {e}", zip_path.display());
            }
        }
    }

    // Build merged mapping with pack references.
    let pack_refs: Vec<&TexturePack> = packs.iter().collect();
    let merged = merge_texture_pack_mappings(&base_mapping, &pack_refs);

    Ok((merged, packs))
}

/// Hash all zip files in a packs directory for cache invalidation.
/// Returns (path_hash, content_hash) tuples suitable for inclusion in
/// the cache composite hash.
pub fn hash_pack_files(packs_dir: &Path) -> Result<Vec<([u8; 32], [u8; 32])>> {
    let mut entries = Vec::new();
    for pack_path in discover_texture_packs(packs_dir) {
        let filename = pack_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        if let Ok(bytes) = std::fs::read(&pack_path) {
            let path_hash: [u8; 32] = blake3::hash(filename.as_bytes()).into();
            let content_hash: [u8; 32] = blake3::hash(&bytes).into();
            entries.push((path_hash, content_hash));
        }
    }
    Ok(entries)
}

/// Read textures.toml from a directory and return tile_index -> filename mapping.
pub fn load_texture_config(textures_dir: &Path) -> HashMap<u32, String> {
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

/// Parse `pack.toml` content into a `PackMetadata`.
fn parse_pack_metadata(content: &str) -> PackMetadata {
    let value: toml::Value = match content.parse() {
        Ok(v) => v,
        Err(e) => {
            log::warn!("failed to parse pack.toml: {e}");
            return PackMetadata::default();
        }
    };
    let pack = value.get("pack").and_then(|v| v.as_table());
    PackMetadata {
        name: pack
            .and_then(|t| t.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        description: pack
            .and_then(|t| t.get("description"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        version: pack
            .and_then(|t| t.get("version"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        author: pack
            .and_then(|t| t.get("author"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        tile_count: 0,
    }
}

/// Parse `animations.toml` content into a list of `PackAnimationDef`.
///
/// Expected format:
/// ```toml
/// [[animation]]
/// tile = 6          # base tile index (e.g. water)
/// frames = [6, 41, 42, 43]  # tile indices for each frame
/// duration = 0.5    # seconds per frame
/// ```
fn parse_pack_animations(content: &str) -> Vec<PackAnimationDef> {
    let value: toml::Value = match content.parse() {
        Ok(v) => v,
        Err(e) => {
            log::warn!("failed to parse animations.toml: {e}");
            return Vec::new();
        }
    };
    let animations = match value.get("animation").and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut defs = Vec::new();
    for entry in animations {
        let tile_index = entry.get("tile").and_then(|v| v.as_integer()).unwrap_or(0) as u32;
        let frames: Vec<u32> = entry
            .get("frames")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_integer().map(|i| i as u32))
                    .collect()
            })
            .unwrap_or_default();
        let frame_duration = entry
            .get("duration")
            .and_then(|v| v.as_float())
            .unwrap_or(0.5) as f32;
        if !frames.is_empty() {
            defs.push(PackAnimationDef {
                tile_index,
                frames,
                frame_duration,
            });
        }
    }
    defs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_zip(dir: &Path, name: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let zip_path = dir.join(format!("{name}.zip"));
        let file = std::fs::File::create(&zip_path).unwrap();
        let mut zip = zip::ZipWriter::new(file);

        for (filename, content) in files {
            let options = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file(*filename, options).unwrap();
            zip.write_all(content).unwrap();
        }

        zip.finish().unwrap();
        zip_path
    }

    #[test]
    fn test_load_texture_pack_with_toml() {
        let tmp = std::env::temp_dir().join("texture_pack_test_1");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Create a zip with textures.toml + a PNG.
        let png_data = vec![0u8; 1024]; // fake PNG data
        let toml_data = b"[tiles]\n1 = \"custom_stone.png\"\n2 = \"custom_dirt.png\"\n";

        let zip_path = create_test_zip(
            &tmp,
            "test_pack",
            &[
                ("textures.toml", toml_data),
                ("custom_stone.png", &png_data),
                ("custom_dirt.png", &png_data),
            ],
        );

        let pack = load_texture_pack(&zip_path, &tmp).unwrap();
        assert_eq!(pack.name, "test_pack");
        assert_eq!(pack.mapping().len(), 2);
        assert_eq!(
            pack.mapping().get(&1),
            Some(&"custom_stone.png".to_string())
        );
        assert_eq!(pack.mapping().get(&2), Some(&"custom_dirt.png".to_string()));

        // Cleanup.
        let _ = unload_texture_pack(&pack);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_merge_texture_pack_mappings() {
        let mut base = HashMap::new();
        base.insert(1, "stone.png".to_string());
        base.insert(2, "dirt.png".to_string());

        let mut pack_mapping = HashMap::new();
        pack_mapping.insert(1, "custom_stone.png".to_string());

        let pack = TexturePack {
            name: "test".to_string(),
            extract_dir: PathBuf::new(),
            mapping: pack_mapping,
            metadata: PackMetadata::default(),
            animations: Vec::new(),
        };

        let merged = merge_texture_pack_mappings(&base, &[&pack]);
        assert_eq!(merged.get(&1), Some(&"custom_stone.png".to_string())); // overridden
        assert_eq!(merged.get(&2), Some(&"dirt.png".to_string())); // untouched
    }

    #[test]
    fn test_discover_texture_packs() {
        let tmp = std::env::temp_dir().join("texture_pack_test_discover");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Create some files.
        std::fs::File::create(tmp.join("a_pack.zip")).unwrap();
        std::fs::File::create(tmp.join("b_pack.zip")).unwrap();
        std::fs::File::create(tmp.join("not_a_pack.txt")).unwrap();

        let packs = discover_texture_packs(&tmp);
        assert_eq!(packs.len(), 2);
        assert!(packs[0].to_str().unwrap().contains("a_pack.zip"));
        assert!(packs[1].to_str().unwrap().contains("b_pack.zip"));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
