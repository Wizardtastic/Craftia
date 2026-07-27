//! Audio events: commands sent to the audio manager.

/// An audio event to be processed by the AudioManager.
#[derive(Clone, Debug)]
pub enum AudioEvent {
    /// Play a spatial sound effect at a world position.
    PlaySfx {
        /// Key into SoundRegistry (e.g. "step.grass").
        sound: String,
        /// World-space position (None = non-spatial/UI).
        position: Option<[f32; 3]>,
        /// Volume (0.0..1.0), relative to the group volume.
        volume: f32,
        /// Pitch shift (None = 1.0).
        pitch: Option<f32>,
        /// Which mixer group to play in.
        group: AudioGroup,
    },
    /// Play a music track (streaming).
    PlayMusic {
        /// Key into SoundRegistry (e.g. "music.menu").
        track: String,
        /// Volume (0.0..1.0).
        volume: f32,
        /// Whether to loop the track.
        loop_: bool,
    },
    /// Stop current music.
    StopMusic,
    /// Stop all sounds in a given group.
    StopGroup(AudioGroup),
    /// Set the 3D listener position + orientation.
    SetListener {
        pos: [f32; 3],
        forward: [f32; 3],
        up: [f32; 3],
    },
    /// Set master volume (from settings change).
    SetMasterVolume(f32),
    /// Set group volume.
    SetGroupVolume(AudioGroup, f32),
    /// Mute/unmute all.
    SetMuted(bool),
    /// Reload a specific sound (for hot-reload).
    ReloadSound(String),
}

/// Audio mixer group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioGroup {
    Sfx,
    Music,
    Ambient,
}
