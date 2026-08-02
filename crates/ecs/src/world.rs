//! The ECS `World`.
//!
//! Owns all entities, archetypes, and resources. Provides the primary
//! spawn/despawn/set/get API plus the resource and query entry points.

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::collections::HashMap;

use crate::archetype::{Archetype, ArchetypeId, ColumnCtor, ErasedColumn, TypedColumn};
use crate::component::{Bundle, Component};
use crate::entity::{Entity, EntityLocation};
use crate::query::{Query, QueryIter};
use crate::resources::Resources;

/// The ECS world. Single owner of all entity, archetype, and resource
/// state.
pub struct World {
    archetypes: Vec<Archetype>,
    /// Sorted unique `TypeId` sets → [`ArchetypeId`].
    archetype_by_components: HashMap<Vec<TypeId>, ArchetypeId>,
    /// `EntityLocation` per entity slot, indexed by `Entity.index`.
    entities: Vec<EntityLocation>,
    /// Current generation per entity slot, indexed by `Entity.index`.
    generations: Vec<u32>,
    /// Recycled entity indices, LIFO.
    free_list: Vec<u32>,
    /// Constructor for each component column type, keyed by `TypeId`.
    /// Populated lazily on first use of a component type.
    column_ctors: HashMap<TypeId, ColumnCtor>,
    /// Per-component-type short name lookup. Populated lazily by
    /// [`World::ensure_registered`] so archetype creation always has a
    /// non-placeholder name. Also populated for resources via
    /// [`World::register_debug_formatter`].
    name_fns: HashMap<TypeId, fn() -> &'static str>,
    /// Per-component-type debug formatter. Optional — types not registered
    /// here are simply omitted from the runtime ECS inspector (rather than
    /// panicking). Used by [`World::format_component`] and
    /// [`World::format_resource`].
    debug_formatters: HashMap<TypeId, fn(&dyn Any) -> String>,
    /// World-wide singleton resources.
    pub resources: Resources,
}

/// Free function used as the column constructor for a component type.
fn make_column<T: crate::component::Component>() -> Box<dyn ErasedColumn> {
    Box::new(TypedColumn::<T>::new())
}

/// Short type name (`Transform`, `Velocity`, ...) derived from
/// `std::any::type_name::<T>()`. Used by the runtime ECS inspector to
/// label archetype columns and resource entries.
pub fn name_of<T: 'static>() -> &'static str {
    let full = std::any::type_name::<T>();
    full.rsplit("::").next().unwrap_or(full)
}

/// Generic helper used as the `HashMap` value for `debug_formatters`.
/// Compiles down to one concrete fn item per monomorphised `T`, so it
/// fits in `fn(&dyn Any) -> String` without any closure captures.
fn debug_format<T: std::fmt::Debug + 'static>(value: &dyn Any) -> String {
    match value.downcast_ref::<T>() {
        Some(v) => format!("{:#?}", v),
        None => String::from("<type mismatch>"),
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// Create an empty world.
    pub fn new() -> Self {
        Self {
            archetypes: Vec::new(),
            archetype_by_components: HashMap::new(),
            entities: Vec::new(),
            generations: Vec::new(),
            free_list: Vec::new(),
            column_ctors: HashMap::new(),
            name_fns: HashMap::new(),
            debug_formatters: HashMap::new(),
            resources: Resources::new(),
        }
    }

    // -----------------------------------------------------------------
    // Entity allocation
    // -----------------------------------------------------------------

    /// Allocate a fresh entity slot. If a slot is on the free list it is
    /// recycled and its generation is bumped; otherwise a new slot is
    /// appended.
    fn alloc_entity(&mut self) -> Entity {
        if let Some(index) = self.free_list.pop() {
            let gen = self.generations[index as usize] + 1;
            self.generations[index as usize] = gen;
            self.entities[index as usize] = EntityLocation {
                archetype: EntityLocation::EMPTY,
                index: 0,
            };
            Entity {
                index,
                generation: gen,
            }
        } else {
            let index = self.entities.len() as u32;
            self.entities.push(EntityLocation {
                archetype: EntityLocation::EMPTY,
                index: 0,
            });
            self.generations.push(0);
            Entity {
                index,
                generation: 0,
            }
        }
    }

    /// Returns true iff the entity's generation matches the current
    /// generation of its slot.
    pub fn is_alive(&self, entity: Entity) -> bool {
        if entity.is_null() {
            return false;
        }
        self.generations.get(entity.index as usize).copied() == Some(entity.generation)
    }

    /// Number of live entities (those whose generation matches their slot).
    pub fn entity_count(&self) -> u32 {
        // Generations match by construction; live count = slot count - free list.
        (self.entities.len() as u32) - (self.free_list.len() as u32)
    }

    /// Number of archetypes in the world.
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

    /// All archetypes currently in the world. Used by
    /// [`QueryIter`](crate::query::QueryIter) and the runtime ECS
    /// inspector in the engine crate. Visibility was widened from
    /// `pub(crate)` to `pub` for the inspector's read-only access.
    pub fn archetypes(&self) -> &[Archetype] {
        &self.archetypes
    }

    // -----------------------------------------------------------------
    // Type registry
    // -----------------------------------------------------------------

    /// Lazily register a column constructor for component type `T` if it
    /// is not already known. Also registers the short `name_of::<T>()` so
    /// subsequent archetype construction can label the column with the
    /// type's short name (e.g. `Transform`) instead of `<component>`.
    fn ensure_registered<T: Component>(&mut self) {
        let id = TypeId::of::<T>();
        // `make_column::<T>` is a fn-item of type `ColumnCtor`.
        self.column_ctors.entry(id).or_insert(make_column::<T>);
        // Short name lookup for archetype labels. `name_of::<T>` is a
        // fn-item; cloning the fn-pointer into the HashMap is cheap.
        self.name_fns.entry(id).or_insert(name_of::<T>);
    }

    /// Register a `Debug`-based formatter for type `T` so the runtime
    /// ECS inspector can display component and resource values. Safe to
    /// call multiple times for the same type — the most recent entry wins.
    ///
    /// `T` must be `Debug + Send + Sync + 'static` to satisfy the
    /// formatter's downcast + the generic resource storage bounds.
    pub fn register_debug_formatter<T: std::fmt::Debug + Send + Sync + 'static>(&mut self) {
        self.debug_formatters
            .insert(TypeId::of::<T>(), debug_format::<T>);
        // Resource type names are not auto-registered (resources don't go
        // through `ensure_registered`), so install the short name here too.
        // For components this is redundant with `ensure_registered`, but
        // harmless — the entry already existed.
        self.name_fns.entry(TypeId::of::<T>()).or_insert(name_of::<T>);
    }

    /// Look up the short name for a registered type id. Falls back to
    /// `"<unknown>"` if the type was never registered.
    pub fn name_for(&self, tid: TypeId) -> &'static str {
        self.name_fns
            .get(&tid)
            .map(|f| f())
            .unwrap_or("<unknown>")
    }

    /// Format a component value through its registered `Debug` formatter.
    /// Returns `None` if no formatter is registered for `type_id`.
    pub fn format_component(
        &self,
        type_id: TypeId,
        value: &dyn Any,
    ) -> Option<String> {
        self.debug_formatters.get(&type_id).map(|fmt| fmt(value))
    }

    /// Enumerate the world-wide singleton resources. Order is unspecified.
    pub fn resource_type_ids(&self) -> Vec<TypeId> {
        self.resources.type_ids()
    }

    /// Format a resource value through its `Debug` formatter. Returns
    /// `None` if the resource is missing or no formatter is registered.
    pub fn format_resource(&self, type_id: &TypeId) -> Option<String> {
        let raw = self.resources.get_raw(type_id)?;
        self.debug_formatters.get(type_id).map(|fmt| fmt(raw))
    }

    /// Look up the constructor for a known type. Panics if the type was
    /// never registered (which means no component of that type was ever
    /// touched).
    fn ctor_for(&self, type_id: TypeId) -> ColumnCtor {
        self.column_ctors
            .get(&type_id)
            .copied()
            .expect("component type used without ensure_registered")
    }

    // -----------------------------------------------------------------
    // Archetype lookup / creation
    // -----------------------------------------------------------------

    /// Look up an archetype by its sorted component type set, creating
    /// it on demand. Uses a stack-allocated buffer for the common case
    /// of ≤16 component types to avoid heap allocation on every call.
    /// Sets larger than the buffer fall back to a heap `Vec` rather than
    /// silently dropping types (which would create an archetype missing
    /// columns and panic later when values are pushed into it).
    fn get_or_create_archetype(&mut self, types: &[TypeId]) -> ArchetypeId {
        // Stack buffer for the sorted, deduplicated key. 16 components
        // covers virtually all real-world archetypes.
        const MAX_STACK: usize = 16;
        let mut buf: [TypeId; MAX_STACK] = [TypeId::of::<()>(); MAX_STACK];
        let mut len = 0usize;
        let mut overflowed = false;
        for &t in types {
            if !buf[..len].contains(&t) {
                if len < MAX_STACK {
                    buf[len] = t;
                    len += 1;
                } else {
                    overflowed = true;
                }
            }
        }

        // Rare: >16 distinct component types. Rebuild the full
        // deduplicated, sorted key on the heap. The stack path must
        // never silently drop types — the created archetype would lack
        // columns for them and `transition` would panic pushing values.
        let key: Cow<[TypeId]> = if overflowed {
            let mut v = types.to_vec();
            v.sort();
            v.dedup();
            Cow::Owned(v)
        } else {
            buf[..len].sort();
            Cow::Borrowed(&buf[..len])
        };

        if let Some(&id) = self.archetype_by_components.get(key.as_ref()) {
            return id;
        }
        // Slow path: only heap-allocates on archetype creation (very rare).
        let key_vec = key.into_owned();
        let id = ArchetypeId(self.archetypes.len() as u32);
        let mut arch = Archetype::new(id);
        let entries: Vec<(TypeId, &'static str, ColumnCtor)> = key_vec
            .iter()
            .map(|&t| (t, self.name_for(t), self.ctor_for(t)))
            .collect();
        arch.set_columns(entries);
        self.archetype_by_components.insert(key_vec, id);
        self.archetypes.push(arch);
        id
    }

    // -----------------------------------------------------------------
    // Archetype transitions
    // -----------------------------------------------------------------

    /// Move `entity` from its current archetype to the archetype with
    /// `to_remove` types removed and `to_add` types added.
    ///
    /// Returns the value taken from the first column in `to_remove` (if
    /// any), as a `Box<dyn Any>`. Panics if more than one type is in
    /// `to_remove`.
    fn transition(
        &mut self,
        entity: Entity,
        to_remove: &[TypeId],
        to_add: Vec<(TypeId, Box<dyn Any>)>,
    ) -> Option<Box<dyn Any>> {
        let loc = self.entities[entity.index as usize];
        let old_arch_id = loc.archetype;
        let old_idx = loc.index;

        // Compute the new archetype's component set without heap allocation.
        // Use a fixed-size buffer for the common case (≤16 component types),
        // falling back to a heap `Vec` on overflow — never silently drop
        // types, or the archetype would miss columns and the push below
        // would panic ("missing column in new archetype after get_or_create").
        const MAX_STACK: usize = 16;
        let mut new_buf: [TypeId; MAX_STACK] = [TypeId::of::<()>(); MAX_STACK];
        let mut new_len = 0usize;
        let mut overflowed = false;

        if old_arch_id != EntityLocation::EMPTY {
            for &t in &self.archetypes[old_arch_id as usize].component_types {
                if !to_remove.contains(&t) && !new_buf[..new_len].contains(&t) {
                    if new_len < MAX_STACK {
                        new_buf[new_len] = t;
                        new_len += 1;
                    } else {
                        overflowed = true;
                    }
                }
            }
        }
        for (t, _) in &to_add {
            if !new_buf[..new_len].contains(t) {
                if new_len < MAX_STACK {
                    new_buf[new_len] = *t;
                    new_len += 1;
                } else {
                    overflowed = true;
                }
            }
        }

        let new_key: Cow<[TypeId]> = if overflowed {
            // Rare: rebuild the full deduplicated, sorted set on the heap.
            let mut v: Vec<TypeId> = Vec::with_capacity(new_len + to_add.len());
            if old_arch_id != EntityLocation::EMPTY {
                for &t in &self.archetypes[old_arch_id as usize].component_types {
                    if !to_remove.contains(&t) && !v.contains(&t) {
                        v.push(t);
                    }
                }
            }
            for (t, _) in &to_add {
                if !v.contains(t) {
                    v.push(*t);
                }
            }
            v.sort();
            Cow::Owned(v)
        } else {
            new_buf[..new_len].sort();
            Cow::Borrowed(&new_buf[..new_len])
        };

        let new_arch_id = self.get_or_create_archetype(&new_key);

        // Same archetype: just overwrite the value. (Only relevant when
        // to_remove and to_add are both empty and the entity is already
        // in the target archetype; this should be handled by callers.)
        if old_arch_id == new_arch_id.0
            && old_arch_id != EntityLocation::EMPTY
            && to_add.is_empty()
            && to_remove.is_empty()
        {
            return None;
        }

        // Helper closure to check if a type is in to_add (avoids HashSet allocation).
        let is_in_to_add = |t: &TypeId| -> bool { to_add.iter().any(|(tt, _)| tt == t) };

        // Capture the last row in the old archetype's entities vec
        // before we touch anything — we need it to identify the entity
        // that will be swap-removed into `old_idx`.
        let old_last: usize = if old_arch_id == EntityLocation::EMPTY {
            0
        } else {
            self.archetypes[old_arch_id as usize]
                .entities
                .len()
                .saturating_sub(1)
        };

        // First pass: drain the old archetype's columns at `old_idx`,
        // collecting (type, value) pairs to push to the new archetype.
        // We use component_types directly (parallel to columns) to avoid
        // allocating a separate Vec for type IDs.
        let mut moves: Vec<(TypeId, Box<dyn Any>)> = Vec::new();
        let mut taken: Option<Box<dyn Any>> = None;
        if old_arch_id != EntityLocation::EMPTY {
            let old_arch = &mut self.archetypes[old_arch_id as usize];
            for (i, &t) in old_arch.component_types.iter().enumerate() {
                if to_remove.contains(&t) {
                    let value = old_arch.columns[i].take_any(old_idx);
                    if taken.is_some() {
                        panic!("transition: multiple types in to_remove");
                    }
                    taken = Some(value);
                } else if !is_in_to_add(&t) {
                    // Type is in both old and new: move the value.
                    moves.push((t, old_arch.columns[i].take_any(old_idx)));
                }
                // else: type is in to_add, we drop the old value (it'll
                // be replaced by the new one).
            }
        }

        // Second pass: push all values into the new archetype.
        {
            let new_arch = &mut self.archetypes[new_arch_id.0 as usize];
            for (t, value) in moves {
                let new_col_idx = new_arch
                    .column_index
                    .get(&t)
                    .copied()
                    .expect("missing column in new archetype");
                new_arch.columns[new_col_idx].push_any(value);
            }
            for (t, value) in to_add {
                let new_col_idx = new_arch
                    .column_index
                    .get(&t)
                    .copied()
                    .expect("missing column in new archetype after get_or_create");
                new_arch.columns[new_col_idx].push_any(value);
            }
        }

        // Remove the entity from the old archetype's `entities` vec.
        // The columns were already drained in the first pass, so we
        // must NOT call `swap_remove_entity` (which would drain them a
        // second time). We just swap-remove from `entities` directly.
        let moved = if old_arch_id != EntityLocation::EMPTY {
            let old_arch = &mut self.archetypes[old_arch_id as usize];
            let moved_entity = if (old_idx as usize) != old_last {
                Some(old_arch.entities[old_last])
            } else {
                None
            };
            old_arch.entities.swap_remove(old_idx as usize);
            moved_entity
        } else {
            None
        };

        // Add the entity to the new archetype.
        let new_idx = self.archetypes[new_arch_id.0 as usize].entities.len() as u32;
        self.archetypes[new_arch_id.0 as usize]
            .entities
            .push(entity);

        // Update the entity's location.
        self.entities[entity.index as usize] = EntityLocation {
            archetype: new_arch_id.0,
            index: new_idx,
        };

        // The "moved" entity was at `old_last` in the old archetype; the
        // swap-remove put it at `old_idx`. Update its location.
        if let Some(moved_entity) = moved {
            self.entities[moved_entity.index as usize].index = old_idx;
        }

        taken
    }

    // -----------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------

    /// Spawn a new entity from a bundle of components.
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.alloc_entity();
        bundle.add_to(self, entity);
        entity
    }

    /// Despawn an entity. Returns `true` if the entity was alive.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let loc = self.entities[entity.index as usize];
        if loc.archetype != EntityLocation::EMPTY {
            let moved = self.archetypes[loc.archetype as usize].swap_remove_entity(loc.index);
            if let Some(moved_entity) = moved {
                self.entities[moved_entity.index as usize].index = loc.index;
            }
        }
        // Bump generation and recycle the slot.
        self.generations[entity.index as usize] += 1;
        self.free_list.push(entity.index);
        true
    }

    /// Set (or insert) component `T` on `entity`. If the entity already
    /// has `T` the value is replaced in place; otherwise the entity is
    /// moved to an archetype that includes `T`.
    pub fn set<T: Component>(&mut self, entity: Entity, value: T) {
        assert!(
            self.is_alive(entity),
            "set: entity {:?} is not alive",
            entity
        );
        self.ensure_registered::<T>();

        let loc = self.entities[entity.index as usize];
        let type_id = TypeId::of::<T>();

        if loc.archetype == EntityLocation::EMPTY {
            // Brand new entity, transition to {T}.
            self.transition(
                entity,
                &[],
                vec![(type_id, Box::new(value) as Box<dyn Any>)],
            );
        } else {
            let arch_id = loc.archetype;
            let col_idx = self.archetypes[arch_id as usize]
                .column_index
                .get(&type_id)
                .copied();
            if let Some(col_idx) = col_idx {
                // Entity has T; replace in place.
                let col = self.archetypes[arch_id as usize].columns[col_idx].as_any();
                let typed = col
                    .downcast_ref::<TypedColumn<T>>()
                    .expect("column downcast mismatch in set");
                typed.set(loc.index, value);
            } else {
                // Entity doesn't have T; transition.
                self.transition(
                    entity,
                    &[],
                    vec![(type_id, Box::new(value) as Box<dyn Any>)],
                );
            }
        }
    }

    /// Borrow component `T` of `entity`, if present.
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        if !self.is_alive(entity) {
            return None;
        }
        let loc = self.entities[entity.index as usize];
        if loc.archetype == EntityLocation::EMPTY {
            return None;
        }
        self.archetypes[loc.archetype as usize].get::<T>(loc.index)
    }

    /// Mutably borrow component `T` of `entity`, if present.
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.is_alive(entity) {
            return None;
        }
        let loc = self.entities[entity.index as usize];
        if loc.archetype == EntityLocation::EMPTY {
            return None;
        }
        self.archetypes[loc.archetype as usize].get_mut::<T>(loc.index)
    }

    /// Remove component `T` from `entity`, returning the removed value.
    /// The entity is moved to the archetype that excludes `T`.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        let loc = self.entities[entity.index as usize];
        if loc.archetype == EntityLocation::EMPTY {
            return None;
        }
        let type_id = TypeId::of::<T>();
        let arch_id = loc.archetype;
        if !self.archetypes[arch_id as usize]
            .column_index
            .contains_key(&type_id)
        {
            return None;
        }
        let taken = self.transition(entity, &[type_id], vec![]);
        taken.and_then(|b| b.downcast::<T>().ok().map(|b| *b))
    }

    /// Returns true iff `entity` has a component of type `T`.
    pub fn has<T: Component>(&self, entity: Entity) -> bool {
        self.get::<T>(entity).is_some()
    }

    /// Begin a query for components matching `Q`.
    pub fn query<Q: Query>(&self) -> QueryIter<'_, Q> {
        QueryIter::new(self)
    }

    // -----------------------------------------------------------------
    // Resource API (forwards to self.resources)
    // -----------------------------------------------------------------

    pub fn insert_resource<T: Send + Sync + 'static>(&mut self, value: T) -> Option<T> {
        self.resources.insert(value)
    }

    pub fn resource<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.resources.get::<T>()
    }

    pub fn resource_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.resources.get_mut::<T>()
    }

    pub fn remove_resource<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.resources.remove::<T>()
    }
}
