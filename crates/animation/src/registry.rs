//! Animation clip registry: loads `.anim` files and stores all clips.
//!
//! `.anim` files are JSON format defining non-glTF animations.
//! Stored in `assets/animations/*.anim`.

use std::collections::HashMap;
use std::path::Path;

use crate::data::AnimationClip;

/// Registry of loaded animation clips, keyed by name.
#[derive(Clone, Debug, Default)]
pub struct AnimationClipRegistry {
    clips: HashMap<String, AnimationClip>,
}

impl AnimationClipRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            clips: HashMap::new(),
        }
    }

    /// Load all `.anim` files from a directory.
    pub fn load_from_directory(dir: &Path) -> Self {
        let mut registry = Self::new();
        if !dir.exists() {
            log::warn!("Animations directory not found: {}", dir.display());
            return registry;
        }
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return registry,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("anim") {
                match Self::load_file(&path) {
                    Ok(clip) => {
                        log::info!("Loaded animation: {} from {}", clip.name, path.display());
                        registry.clips.insert(clip.name.clone(), clip);
                    }
                    Err(e) => {
                        log::warn!("Failed to load animation {}: {e}", path.display());
                    }
                }
            }
        }
        log::info!(
            "Loaded {} animations from {}",
            registry.clips.len(),
            dir.display()
        );
        registry
    }

    /// Load a single `.anim` file.
    fn load_file(path: &Path) -> anyhow::Result<AnimationClip> {
        let text = std::fs::read_to_string(path)?;
        let clip: AnimationClip = serde_json::from_str(&text)?;
        Ok(clip)
    }

    /// Get a clip by name.
    pub fn get(&self, name: &str) -> Option<&AnimationClip> {
        self.clips.get(name)
    }

    /// Insert a clip manually.
    pub fn insert(&mut self, clip: AnimationClip) {
        self.clips.insert(clip.name.clone(), clip);
    }

    /// Number of loaded clips.
    pub fn len(&self) -> usize {
        self.clips.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }
}
