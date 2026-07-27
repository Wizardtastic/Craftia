//! Audio system for the voxel engine.
//!
//! Provides spatial 3D audio, mixer groups (SFX, Music, Ambient),
//! and a sound registry for loading sounds from `assets/audio/`.

pub mod config;
pub mod event;
pub mod manager;
pub mod registry;

pub use config::AudioConfig;
pub use event::{AudioEvent, AudioGroup};
pub use manager::AudioManager;
pub use registry::SoundRegistry;
