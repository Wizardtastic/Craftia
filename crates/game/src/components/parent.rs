use voxel_ecs::Entity;

use super::Transform;

/// Links an entity to its parent in the scene graph.
/// `local` is the child's transform relative to the parent's
/// world-space transform. The hierarchy system computes the
/// child's world `Transform` from this chain.
#[derive(Clone, Copy, Debug)]
pub struct Parent {
    pub entity: Entity,
    pub local: Transform,
}

impl Default for Parent {
    fn default() -> Self {
        Self {
            entity: Entity {
                index: u32::MAX,
                generation: u32::MAX,
            },
            local: Transform::default(),
        }
    }
}
