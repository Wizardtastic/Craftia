//! Audio manager: owns the kira backend, processes events, manages mixer groups.

use std::path::Path;

use kira::{
    sound::static_sound::StaticSoundSettings, AudioManager as KiraManager, AudioManagerSettings,
    Decibels, DefaultBackend,
};

use crate::config::AudioConfig;
use crate::event::{AudioEvent, AudioGroup};
use crate::registry::SoundRegistry;

/// Convert a linear amplitude (0.0..1.0) to kira Decibels.
fn amplitude_to_db(amplitude: f32) -> Decibels {
    if amplitude <= 0.0 {
        Decibels::SILENCE
    } else if amplitude >= 1.0 {
        Decibels::IDENTITY
    } else {
        Decibels(20.0 * amplitude.log10())
    }
}

/// Main audio manager. Owns the kira backend and processes audio events.
pub struct AudioManager {
    manager: Option<KiraManager<DefaultBackend>>,
    registry: SoundRegistry,
    config: AudioConfig,
    event_sender: flume::Sender<AudioEvent>,
    event_receiver: flume::Receiver<AudioEvent>,
    is_null: bool,
    audio_dir: std::path::PathBuf,
    /// Current listener position for distance-based volume attenuation.
    listener_pos: [f32; 3],
}

impl AudioManager {
    /// Create a new audio manager, loading sounds from `audio_dir`.
    pub fn new(config: AudioConfig, audio_dir: &Path) -> anyhow::Result<Self> {
        let (tx, rx) = flume::unbounded();

        match KiraManager::<DefaultBackend>::new(AudioManagerSettings::default()) {
            Ok(manager) => {
                let registry = SoundRegistry::load_from_directory(audio_dir);
                log::info!("Audio system initialized ({} sounds)", registry.len());

                Ok(Self {
                    manager: Some(manager),
                    registry,
                    config,
                    event_sender: tx,
                    event_receiver: rx,
                    is_null: false,
                    audio_dir: audio_dir.to_path_buf(),
                    listener_pos: [0.0; 3],
                })
            }
            Err(e) => {
                log::warn!("Failed to initialize audio backend: {e}. Running in null mode.");
                Ok(Self::null_inner(audio_dir, config, tx, rx))
            }
        }
    }

    /// Create a null/stub audio manager that does nothing.
    pub fn null() -> Self {
        let (tx, rx) = flume::unbounded();
        Self {
            manager: None,
            registry: SoundRegistry::load_from_directory(Path::new("assets/audio")),
            config: AudioConfig::default(),
            event_sender: tx,
            event_receiver: rx,
            is_null: true,
            audio_dir: std::path::PathBuf::from("assets/audio"),
            listener_pos: [0.0; 3],
        }
    }

    fn null_inner(
        audio_dir: &Path,
        config: AudioConfig,
        tx: flume::Sender<AudioEvent>,
        rx: flume::Receiver<AudioEvent>,
    ) -> Self {
        Self {
            manager: None,
            registry: SoundRegistry::load_from_directory(audio_dir),
            config,
            event_sender: tx,
            event_receiver: rx,
            is_null: true,
            audio_dir: audio_dir.to_path_buf(),
            listener_pos: [0.0; 3],
        }
    }

    /// Get a sender for pushing events from other threads (ECS systems, UI).
    pub fn event_sender(&self) -> flume::Sender<AudioEvent> {
        self.event_sender.clone()
    }

    /// Push an event directly (from the main thread).
    pub fn push_event(&self, event: AudioEvent) {
        if self.is_null {
            return;
        }
        self.event_sender.send(event).ok();
    }

    /// Process all queued audio events. Call once per frame from frame.rs.
    pub fn process_events(&mut self) {
        if self.is_null {
            while self.event_receiver.try_recv().is_ok() {}
            return;
        }
        while let Ok(event) = self.event_receiver.try_recv() {
            self.handle_event(event);
        }
    }

    fn handle_event(&mut self, event: AudioEvent) {
        match event {
            AudioEvent::PlaySfx {
                sound,
                position,
                volume,
                pitch,
                group: _,
            } => {
                let Some(sound_data) = self.registry.get(&sound) else {
                    log::trace!("Sound not found: {sound}");
                    return;
                };
                let Some(ref mut manager) = self.manager else {
                    return;
                };

                let mut settings = StaticSoundSettings::default();

                if let Some(p) = pitch {
                    settings = settings.playback_rate(p as f64);
                }

                let vol = volume * self.config.master_volume * self.config.sfx_volume;

                // Distance-based volume attenuation for 3D positioning.
                let effective_vol = if let Some(pos) = position {
                    let dx = pos[0] - self.listener_pos[0];
                    let dy = pos[1] - self.listener_pos[1];
                    let dz = pos[2] - self.listener_pos[2];
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    // Attenuate volume based on distance: full at 0, silent at 32 blocks.
                    let attenuation = (1.0 - (distance / 32.0).min(1.0)).max(0.0);
                    vol * attenuation
                } else {
                    vol
                };

                settings = settings.volume(amplitude_to_db(effective_vol));

                let _ = manager.play(sound_data.data.clone().with_settings(settings));
            }
            AudioEvent::PlayMusic {
                track,
                volume,
                loop_,
            } => {
                let Some(sound_data) = self.registry.get(&track) else {
                    log::trace!("Music track not found: {track}");
                    return;
                };
                let Some(ref mut manager) = self.manager else {
                    return;
                };

                let vol = volume * self.config.master_volume * self.config.music_volume;
                let mut settings = StaticSoundSettings::default().volume(amplitude_to_db(vol));

                if loop_ {
                    settings = settings.loop_region(..);
                }

                let _ = manager.play(sound_data.data.clone().with_settings(settings));
            }
            AudioEvent::StopMusic => {
                // kira 0.12: would need to track the instance handle to stop it.
            }
            AudioEvent::StopGroup(_group) => {
                // Would need handle tracking. No-op for now.
            }
            AudioEvent::SetListener {
                pos,
                forward: _,
                up: _,
            } => {
                self.listener_pos = pos;
            }
            AudioEvent::SetMasterVolume(v) => {
                self.config.master_volume = v;
            }
            AudioEvent::SetGroupVolume(group, v) => match group {
                AudioGroup::Sfx => self.config.sfx_volume = v,
                AudioGroup::Music => self.config.music_volume = v,
                AudioGroup::Ambient => self.config.ambient_volume = v,
            },
            AudioEvent::SetMuted(muted) => {
                self.config.muted = muted;
            }
            AudioEvent::ReloadSound(key) => {
                if let Err(e) = self.registry.reload(&key, &self.audio_dir) {
                    log::warn!("Failed to reload sound {key}: {e}");
                }
            }
        }
    }
}
