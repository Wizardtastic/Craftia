//! Sound registry: loads and caches audio files from `assets/audio/`.
//!
//! File naming convention: directory structure maps to dotted keys:
//!   assets/audio/step/grass.ogg  → key "step.grass"
//!   assets/audio/dig/stone.ogg   → key "dig.stone"
//!   assets/audio/ui/click.ogg    → key "ui.click"

use std::collections::HashMap;
use std::path::Path;

use kira::sound::static_sound::StaticSoundData;

/// A loaded sound ready to play.
#[derive(Clone)]
pub struct SoundData {
    pub data: StaticSoundData,
}

/// Registry of loaded sounds, keyed by dotted names.
pub struct SoundRegistry {
    sounds: HashMap<String, SoundData>,
}

impl SoundRegistry {
    /// Load all supported audio files from `assets/audio/`.
    pub fn load_from_directory(audio_dir: &Path) -> Self {
        let mut sounds = HashMap::new();
        if !audio_dir.exists() {
            log::warn!("Audio directory not found: {}", audio_dir.display());
            return Self { sounds };
        }
        Self::load_recursive(audio_dir, audio_dir, &mut sounds);
        log::info!("Loaded {} sounds from {}", sounds.len(), audio_dir.display());
        Self { sounds }
    }

    /// Recursively load audio files, building dotted keys from paths.
    fn load_recursive(base: &Path, dir: &Path, sounds: &mut HashMap<String, SoundData>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::load_recursive(base, &path, sounds);
                continue;
            }
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext != "ogg" && ext != "wav" {
                continue;
            }
            // Build dotted key from relative path.
            let rel = path.strip_prefix(base).unwrap_or(&path);
            let key = rel
                .with_extension("")
                .components()
                .map(|c| c.as_os_str().to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join(".");

            match StaticSoundData::from_file(&path) {
                Ok(data) => {
                    sounds.insert(key, SoundData { data });
                }
                Err(e) => {
                    log::warn!("Failed to load sound {}: {e}", path.display());
                }
            }
        }
    }

    /// Get a sound by dotted key (e.g. "step.grass").
    pub fn get(&self, key: &str) -> Option<&SoundData> {
        self.sounds.get(key)
    }

    /// Number of loaded sounds.
    pub fn len(&self) -> usize {
        self.sounds.len()
    }

    /// Reload a specific sound from disk.
    pub fn reload(&mut self, key: &str, audio_dir: &Path) -> anyhow::Result<()> {
        // Convert dotted key back to path.
        let rel = key.replace('.', std::path::MAIN_SEPARATOR_STR);
        let mut path = audio_dir.join(&rel);
        // Try .ogg first, then .wav.
        if !path.with_extension("ogg").exists() {
            path = audio_dir.join(format!("{rel}.wav"));
        } else {
            path = path.with_extension("ogg");
        }
        let data = StaticSoundData::from_file(&path)?;
        self.sounds.insert(key.to_string(), SoundData { data });
        Ok(())
    }
}
