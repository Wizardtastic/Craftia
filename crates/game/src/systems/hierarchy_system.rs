//! Hierarchy system: propagates transforms through the parent-child
//! scene graph. Runs AFTER movement_system.
//!
//! Uses a `ChildMapResource` side-table instead of ECS `Children`
//! components for O(1) parent→children lookup without archetype
//! column overhead.
//!
//! Passes:
//!   1. Orphan cleanup — despawn children whose parent is no longer alive.
//!   2. Transform propagation — compute world Transform from parent chain.
//!   3. ChildMap update — maintain parent → children lists.

use std::collections::{HashMap, HashSet, VecDeque};

use voxel_ecs::{Entity, World};

use crate::components::{Parent, Transform};

/// Maximum depth before we bail (cycle safety).
const MAX_DEPTH: u64 = 64;

/// Side-table mapping parent entity → child entity list.
/// Stored as an ECS resource via `ChildMapResource`. Avoids ECS
/// component overhead for entities that have no children.
#[derive(Clone, Debug, Default)]
pub struct ChildMap {
    /// parent → children
    map: HashMap<Entity, Vec<Entity>>,
    /// child → parent (for O(1) parent lookup)
    reverse: HashMap<Entity, Entity>,
}

impl ChildMap {
    pub fn add_child(&mut self, parent: Entity, child: Entity) {
        self.map.entry(parent).or_default().push(child);
        self.reverse.insert(child, parent);
    }

    pub fn remove_child(&mut self, child: Entity) {
        if let Some(parent) = self.reverse.remove(&child) {
            if let Some(kids) = self.map.get_mut(&parent) {
                kids.retain(|e| *e != child);
            }
        }
    }

    /// Remove an entity and return its children for despawn cascade.
    pub fn remove_entity(&mut self, entity: Entity) -> Vec<Entity> {
        let kids = self.map.remove(&entity).unwrap_or_default();
        for child in &kids {
            self.reverse.remove(child);
        }
        kids
    }

    pub fn children_of(&self, parent: Entity) -> &[Entity] {
        self.map.get(&parent).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn parent_of(&self, child: Entity) -> Option<Entity> {
        self.reverse.get(&child).copied()
    }

    pub fn reparent(&mut self, child: Entity, new_parent: Option<Entity>) {
        // Remove from old parent.
        self.remove_child(child);
        // Add to new parent.
        if let Some(parent) = new_parent {
            self.add_child(parent, child);
        }
    }

    /// Clear all entries (for testing).
    pub fn clear(&mut self) {
        self.map.clear();
        self.reverse.clear();
    }
}

/// ECS resource wrapper for ChildMap.
#[derive(Clone, Debug, Default)]
pub struct ChildMapResource(pub ChildMap);

/// System: propagate transforms through the scene graph.
pub fn hierarchy_system(world: &mut World, _dt: f32) {
    // Ensure ChildMapResource exists.
    if !world.resource::<ChildMapResource>().is_some() {
        world.insert_resource(ChildMapResource::default());
    }

    // Single pass: collect Parent entities + detect orphans + build new ChildMap.
    let mut orphans = Vec::new();
    let mut new_child_map = ChildMap::default();
    let mut parent_entities = Vec::new();

    for arch in world.archetypes().iter() {
        // Only scan archetypes that have Parent component.
        if !arch.has::<Parent>() {
            continue;
        }
        for &e in arch.entities() {
            if let Some(parent_comp) = world.get::<Parent>(e) {
                if !world.is_alive(parent_comp.entity) {
                    orphans.push(e);
                } else {
                    new_child_map.add_child(parent_comp.entity, e);
                }
                parent_entities.push(e);
            }
        }
    }

    for orphan in &orphans {
        log::warn!(
            "hierarchy: despawning orphan e[{}:{}] (parent gone)",
            orphan.index,
            orphan.generation
        );
        despawn_recursive(world, *orphan);
    }

    // Update the ChildMapResource.
    if let Some(res) = world.resource_mut::<ChildMapResource>() {
        res.0 = new_child_map;
    }

    // BFS transform propagation from roots (entities WITHOUT Parent).
    // Collect roots in the same scan over archetypes.
    let mut roots = Vec::new();
    for arch in world.archetypes().iter() {
        for &e in arch.entities() {
            if world.has::<Transform>(e) && !world.has::<Parent>(e) {
                roots.push(e);
            }
        }
    }

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    for root in roots {
        visited.insert(root);
        queue.push_back((root, 0u64));
    }

    while let Some((entity, depth)) = queue.pop_front() {
        if depth >= MAX_DEPTH {
            log::warn!(
                "hierarchy: max depth ({}) reached at e[{}:{}], possible cycle",
                MAX_DEPTH,
                entity.index,
                entity.generation
            );
            continue;
        }

        // Read children from the resource without holding a long borrow.
        let kids: Vec<Entity> = world
            .resource::<ChildMapResource>()
            .map(|r| r.0.children_of(entity).to_vec())
            .unwrap_or_default();

        for child in kids {
            if visited.contains(&child) {
                log::warn!(
                    "hierarchy: cycle detected at e[{}:{}], skipping",
                    child.index,
                    child.generation
                );
                continue;
            }
            visited.insert(child);

            // Compute child's world transform = parent.world * child.local.
            if let (Some(parent_transform), Some(parent_comp)) =
                (world.get::<Transform>(entity), world.get::<Parent>(child))
            {
                let local = parent_comp.local;
                let world_pos = parent_transform.pos + parent_transform.rot * local.pos;
                let world_rot = parent_transform.rot * local.rot;
                if let Some(child_transform) = world.get_mut::<Transform>(child) {
                    child_transform.pos = world_pos;
                    child_transform.rot = world_rot;
                }
            }

            queue.push_back((child, depth + 1));
        }
    }
}

/// Recursively despawn an entity and all its children (DFS).
fn despawn_recursive(world: &mut World, entity: Entity) {
    // Collect children from ChildMapResource without cloning the whole map.
    let kids: Vec<Entity> = world
        .resource::<ChildMapResource>()
        .map(|r| r.0.children_of(entity).to_vec())
        .unwrap_or_default();

    for child in kids {
        despawn_recursive(world, child);
    }

    // Remove from child map.
    if let Some(res) = world.resource_mut::<ChildMapResource>() {
        res.0.remove_entity(entity);
    }

    log::info!(
        "hierarchy: despawning e[{}:{}]",
        entity.index,
        entity.generation
    );
    world.despawn(entity);
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    fn parent_child_transform_propagation() {
        let mut world = World::new();
        world.insert_resource(ChildMapResource::default());
        let parent = world.spawn((Transform {
            pos: Vec3::new(10.0, 0.0, 0.0),
            rot: Quat::IDENTITY,
        },));
        let child = world.spawn((
            Transform {
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY,
            },
            Parent {
                entity: parent,
                local: Transform {
                    pos: Vec3::new(5.0, 0.0, 0.0),
                    rot: Quat::IDENTITY,
                },
            },
        ));

        hierarchy_system(&mut world, 0.016);

        let child_pos = world.get::<Transform>(child).unwrap().pos;
        assert!((child_pos.x - 15.0).abs() < f32::EPSILON);
    }

    #[test]
    fn orphan_cleanup() {
        let mut world = World::new();
        world.insert_resource(ChildMapResource::default());
        let parent = world.spawn((Transform {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        },));
        let child = world.spawn((
            Transform {
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY,
            },
            Parent {
                entity: parent,
                local: Transform::default(),
            },
        ));

        world.despawn(parent);
        hierarchy_system(&mut world, 0.016);

        assert!(!world.is_alive(child));
    }

    #[test]
    fn child_map_updated() {
        let mut world = World::new();
        world.insert_resource(ChildMapResource::default());
        let parent = world.spawn((Transform {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        },));
        let _child = world.spawn((
            Transform {
                pos: Vec3::ZERO,
                rot: Quat::IDENTITY,
            },
            Parent {
                entity: parent,
                local: Transform::default(),
            },
        ));

        hierarchy_system(&mut world, 0.016);

        let child_map = world.resource::<ChildMapResource>().unwrap();
        assert_eq!(child_map.0.children_of(parent).len(), 1);
    }

    #[test]
    fn child_map_reparent() {
        let mut cm = ChildMap::default();
        let a = Entity {
            index: 0,
            generation: 0,
        };
        let b = Entity {
            index: 1,
            generation: 0,
        };
        let c = Entity {
            index: 2,
            generation: 0,
        };

        cm.add_child(a, b);
        assert_eq!(cm.children_of(a).len(), 1);
        assert_eq!(cm.parent_of(b), Some(a));

        cm.reparent(b, Some(c));
        assert_eq!(cm.children_of(a).len(), 0);
        assert_eq!(cm.children_of(c).len(), 1);
        assert_eq!(cm.parent_of(b), Some(c));
    }
}
