//! Audio configuration: volume levels for each mixer group.

use serde::{Deserialize, Serialize};

/// Volume settings for the audio system.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Master volume (0.0..1.0). Default 0.8.
    pub master_volume: f32,
    /// SFX volume (0.0..1.0). Default 1.0.
    pub sfx_volume: f32,
    /// Music volume (0.0..1.0). Default 0.5.
    pub music_volume: f32,
    /// Ambient volume (0.0..1.0). Default 0.6.
    pub ambient_volume: f32,
    /// Master mute toggle.
    pub muted: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            master_volume: 0.8,
            sfx_volume: 1.0,
            music_volume: 0.5,
            ambient_volume: 0.6,
            muted: false,
        }
    }
}
