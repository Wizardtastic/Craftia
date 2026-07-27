//! `voxel-game` — gameplay logic: components, systems, player actions.
//!
//! The engine crate wires this into the main loop. The legacy per-struct
//! modules (`block`, `chat`, `input`, `inv`, `player`, `undo`) coexist
//! with the new ECS-based `components` and `systems`; the engine
//! integration agent will retire the legacy ones once everything is
//! ported.

pub mod block;
pub mod chat;
pub mod components;
pub mod console;
pub mod editor_entities;
pub mod input;
pub mod inv;
pub mod inventory;
pub mod item_entity;
pub mod items;
pub mod player;
pub mod systems;
pub mod undo;

pub use block::BlockAction;
pub use chat::{ChatState, CommandResult};
pub use components::{
    update_camera_from_transform, Aabb, AnimationPlayer, Attachment, BoneTransforms,
    CameraOwner, Children, HeldBlock, Mesh, MiningProgress, ModelRef, Parent, PlayerEntity,
    PlayerInput, PlayerLookTarget, PlayerState, Transform, Velocity, ViewMode,
};
pub use console::DeveloperConsole;
pub use editor_entities::{spawn_debug_entity, DebugEntityMarker};
pub use input::InputState;
pub use inv::Hotbar;
pub use inventory::{InventorySlot, SurvivalInventory};
pub use item_entity::ItemEntity;
pub use items::ItemStack;
pub use player::PlayerConfig;
pub use systems::{
    animation_system, held_item_system, hierarchy_system, input_system, lifecycle_system,
    movement_system, health_system, progressive_mining_system, item_pickup_system,
    hunger_system, drowning_system, regeneration_system, environmental_damage_system,
    armor_system, xp_collection_system, property_animation_system,
    AnimationDataResource, AnimationClip, AnimChannel, AnimInterpolation, AnimPath,
    ModelAnimationData, SkinDataResource, ModelSkinData, SkinInfo,
    CameraResource, ChildMap, ChildMapResource, HotbarResource,
    InputResource, InputSnapshot, PhysicsWorldRes, DifficultyResource, GameTimeResource, EYE_HEIGHT,
    EYE_HEIGHT_SNEAK, FLY_SPEED, GRAVITY, JUMP_SPEED, MOUSE_SENSITIVITY, PLAYER_HALF,
    SNEAK_SPEED, SPRINT_SPEED, SWIM_BASE_FRACTION, SWIM_UP_SPEED, TERMINAL_VELOCITY,
    WALK_SPEED, WATER_DRAG, DrowningState,
};
pub use components::property_animator::{PropertyAnimator, PropertyTarget, AnimationClipHandle};
pub use undo::{BlockEdit, EditAction, UndoRedoState};

// Re-export gamemode and combat types for convenience.
pub use voxel_gamemode::{GameMode, HealthRegenMode, InventoryBehavior, ItemConsumptionMode};
pub use voxel_combat::{
    AirSupply, DamageEvent, DamageQueue, DamageSource, DeathEvent, Health, RegenState,
};
pub use voxel_hunger::{Difficulty, EatingState, FoodProperties, Hunger};
pub use systems::xp_system::Experience;
pub use systems::armor_system as armor_module;
