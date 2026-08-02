//! `voxel-asset-pipeline` — build-time / first-run asset preprocessing.
//!
//! Provides `process_assets()` which checks a BLAKE3-content-addressed cache
//! and only rebuilds the texture atlas + mip chain when source PNGs change.
//!
//! The cache file lives at `<textures_dir>/.cache/atlas.cache` and contains:
//! - Source manifest (BLAKE3 hashes of all referenced PNGs)
//! - Composite hash for fast validation
//! - Full mip chain (RGBA8, largest → smallest)

pub mod atlas;
pub mod cache;
pub mod mipmap;
pub mod texture_pack;

pub use cache::{check_cache, process_assets, CacheStatus};
pub use texture_pack::{load_all_texture_packs, load_texture_pack, TexturePack};
