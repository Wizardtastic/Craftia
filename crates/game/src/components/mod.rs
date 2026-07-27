//! ECS components used by gameplay systems.
//!
//! These are small data containers attached to entities. Logic lives in
//! `crate::systems`; data lives here.

mod aabb;
mod animation_player;
mod attachment;
mod bone_transforms;
mod camera_owner;
mod children;
mod held_block;
mod mesh;
mod mining_progress;
mod model_ref;
mod parent;
mod player_input;
mod player_state;
pub mod property_animator;
mod transform;
mod velocity;
mod view_mode;

pub use aabb::Aabb;
pub use animation_player::AnimationPlayer;
pub use attachment::Attachment;
pub use bone_transforms::BoneTransforms;
pub use camera_owner::{update_camera_from_transform, CameraOwner, PlayerEntity};
pub use children::Children;
pub use held_block::HeldBlock;
pub use mesh::Mesh;
pub use mining_progress::{MiningProgress, PlayerLookTarget};
pub use model_ref::ModelRef;
pub use parent::Parent;
pub use player_input::PlayerInput;
pub use player_state::PlayerState;
pub use transform::Transform;
pub use velocity::Velocity;
pub use view_mode::ViewMode;
