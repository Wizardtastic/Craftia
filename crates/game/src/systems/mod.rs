//! Gameplay systems: input -> movement -> hierarchy -> animation -> lifecycle.
//!
//! Each system is a plain `fn(&mut World, f32)` so the engine can wrap it
//! in an `FnSystem` and add it to a `SystemSchedule`.

mod animation_system;
pub mod armor_system;
mod drowning_system;
mod environmental_system;
mod health_system;
mod held_item_system;
mod hierarchy_system;
mod hunger_system;
mod input_system;
mod lifecycle_system;
mod mining_system;
mod movement_system;
mod pickup_system;
pub mod property_animation_system;
mod regen_system;
pub mod xp_system;

pub use animation_system::{
    animation_system, AnimationDataResource, AnimChannel, AnimInterpolation, AnimPath,
    ModelAnimationData, SkinDataResource, ModelSkinData, SkinInfo, AnimationClip,
};
pub use armor_system::armor_system;
pub use drowning_system::{drowning_system, DrowningState};
pub use environmental_system::environmental_damage_system;
pub use health_system::health_system;
pub use held_item_system::{held_item_system, HotbarResource};
pub use hierarchy_system::{hierarchy_system, ChildMap, ChildMapResource};
pub use hunger_system::{hunger_system, DifficultyResource, GameTimeResource};
pub use input_system::{input_system, InputResource, InputSnapshot};
pub use lifecycle_system::lifecycle_system;
pub use mining_system::progressive_mining_system;
pub use movement_system::{
    movement_system, CameraResource, PhysicsWorldRes, EYE_HEIGHT, EYE_HEIGHT_SNEAK, FLY_SPEED,
    GRAVITY, JUMP_SPEED, MOUSE_SENSITIVITY, SNEAK_SPEED, SPRINT_SPEED, SWIM_BASE_FRACTION,
    SWIM_UP_SPEED, TERMINAL_VELOCITY, WALK_SPEED, WATER_DRAG,
};
pub use pickup_system::item_pickup_system;
pub use property_animation_system::property_animation_system;
pub use regen_system::regeneration_system;
pub use xp_system::xp_collection_system;
// `PLAYER_HALF` lives on `player.rs` (next to the AABB math) — re-export
// it through the `systems` umbrella so `crates/game/src/lib.rs` can
// `pub use systems::PLAYER_HALF`.
pub use crate::player::PLAYER_HALF;
