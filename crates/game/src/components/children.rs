use voxel_ecs::Entity;

/// Maintained by the hierarchy system: the list of entities
/// that have this entity as their parent. Allows fast iteration
/// without scanning all Parent components.
#[derive(Clone, Debug, Default)]
pub struct Children(pub Vec<Entity>);
