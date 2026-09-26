//! Archetype-based storage.
//!
//! Entities that share the same component composition live in the same
//! [`Archetype`]. Each archetype owns a set of typed columns
//! ([`TypedColumn<T>`]) — one per component type — plus a parallel
//! `Vec<Entity>` listing which entity occupies each row.
//!
//! Shared queries borrow columns immutably; mutation requires an exclusive
//! `&mut Archetype` borrow. This keeps the storage itself free of interior
//! mutability and makes Rust enforce the aliasing rules.

use std::any::{Any, TypeId};
use std::collections::BTreeMap;

use crate::component::Component;
use crate::entity::Entity;

/// Unique identifier for an archetype.
///
/// Two archetypes are equal iff they have the same sorted set of component
/// types.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ArchetypeId(pub u32);

/// Type-erased column. Knows how to swap-remove a row, push a boxed value,
/// and produce typed `Any` views.
///
/// Visibility is `pub` (not `pub(crate)`) so external crates — most
/// notably the engine's runtime ECS inspector — can call
/// [`ErasedColumn::value_as_any`] on a `Box<dyn ErasedColumn>` returned
/// from [`Archetype::columns`]. Trait methods default to `pub` when the
/// trait itself is `pub`, so existing internal callers don't need any
/// additional changes.
pub trait ErasedColumn: Send + Sync {
    /// The `TypeId` of the component this column stores.
    #[allow(dead_code)]
    fn type_id(&self) -> TypeId;

    /// Append a boxed value of the column's element type. Panics if the
    /// boxed value's runtime type does not match.
    fn push_any(&mut self, value: Box<dyn Any>);

    /// Remove the row at `index`, swapping the last row into its place.
    /// Returns the removed value as a boxed `Any`.
    fn take_any(&mut self, index: u32) -> Box<dyn Any>;

    /// Borrow the column itself as a typed `Any`.
    fn as_any(&self) -> &dyn Any;

    /// Mutably borrow the column as a typed `Any`, when supported.
    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        None
    }

    /// Borrow the row at `index` as `&dyn Any` so callers (notably the
    /// runtime ECS inspector) can format the value without knowing its
    /// concrete type. The lifetime is tied to `&self`, matching
    /// `Archetype::get`.
    fn value_as_any(&self, index: u32) -> Option<&dyn Any>;
}

/// Typed column for a specific component. Shared reads use `&self`; any
/// mutation requires `&mut self` and is therefore exclusive.
pub(crate) struct TypedColumn<T: Component> {
    data: Vec<T>,
}

impl<T: Component> TypedColumn<T> {
    pub(crate) fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Append a value to the column.
    pub(crate) fn push(&mut self, value: T) {
        self.data.push(value);
    }

    pub(crate) fn get(&self, index: u32) -> Option<&T> {
        self.data.get(index as usize)
    }

    pub(crate) fn get_mut(&mut self, index: u32) -> Option<&mut T> {
        self.data.get_mut(index as usize)
    }

    /// Overwrite the value at `index`.
    pub(crate) fn set(&mut self, index: u32, value: T) {
        self.data[index as usize] = value;
    }
}

impl<T: Component> ErasedColumn for TypedColumn<T> {
    fn type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn push_any(&mut self, value: Box<dyn Any>) {
        let typed = value.downcast::<T>().expect("type mismatch in push_any");
        self.push(*typed);
    }

    fn take_any(&mut self, index: u32) -> Box<dyn Any> {
        Box::new(self.data.swap_remove(index as usize))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }

    fn value_as_any(&self, index: u32) -> Option<&dyn Any> {
        self.data.get(index as usize).map(|v| v as &dyn Any)
    }
}

/// A constructor for an empty [`ErasedColumn`] of a particular concrete
/// component type. Used by the [`World`](crate::World) to materialize
/// columns at archetype-creation time when it only knows a `TypeId`.
pub(crate) type ColumnCtor = fn() -> Box<dyn ErasedColumn>;

/// An archetype: a table of SoA columns for one specific component
/// composition, plus the list of entities currently occupying each row.
pub struct Archetype {
    pub id: ArchetypeId,
    /// Sorted unique `TypeId`s of the components stored by this archetype.
    pub component_types: Vec<TypeId>,
    /// Parallel to `component_types`: pretty type names for diagnostics.
    pub component_names: Vec<&'static str>,
    /// Maps `TypeId` to the index in `columns` (and `component_names`).
    pub(crate) column_index: BTreeMap<TypeId, usize>,
    /// Type-erased columns, one per component type. Indexed by
    /// `column_index[type_id]`.
    ///
    /// Visibility is `pub(crate)`; external consumers (notably the engine's
    /// runtime ECS inspector) go through [`Archetype::columns`] which
    /// returns a slice of the same boxed trait objects. Direct field
    /// access is reserved for internal mutators such as
    /// [`Archetype::swap_remove_entity`].
    pub(crate) columns: Vec<Box<dyn ErasedColumn>>,
    /// Entities occupying each row, in lock-step with each column.
    pub entities: Vec<Entity>,
}

impl Archetype {
    /// Construct a new empty archetype with no columns. Columns are
    /// installed by [`Archetype::set_columns`] immediately after.
    pub fn new(id: ArchetypeId) -> Self {
        Self {
            id,
            component_types: Vec::new(),
            component_names: Vec::new(),
            column_index: BTreeMap::new(),
            columns: Vec::new(),
            entities: Vec::new(),
        }
    }

    /// Install the column set. The entries are sorted by `TypeId` so the
    /// resulting archetype is canonical — two archetypes with the same
    /// component set always agree on the column order.
    pub(crate) fn set_columns(&mut self, mut entries: Vec<(TypeId, &'static str, ColumnCtor)>) {
        entries.sort_by_key(|(t, _, _)| *t);
        self.component_types.clear();
        self.component_names.clear();
        self.columns.clear();
        self.column_index.clear();
        for (i, (t, name, ctor)) in entries.into_iter().enumerate() {
            self.component_types.push(t);
            self.component_names.push(name);
            self.columns.push(ctor());
            self.column_index.insert(t, i);
        }
    }

    /// Number of entities currently in this archetype.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Returns true iff this archetype stores a column for `T`.
    pub fn has<T: Component>(&self) -> bool {
        self.column_index.contains_key(&TypeId::of::<T>())
    }

    /// Shared reference to component `T` of the entity at `index`, if any.
    pub fn get<T: Component>(&self, index: u32) -> Option<&T> {
        let col_idx = *self.column_index.get(&TypeId::of::<T>())?;
        let col = self.columns[col_idx].as_any();
        let typed = col.downcast_ref::<TypedColumn<T>>()?;
        typed.get(index)
    }

    /// Exclusive reference to component `T` of the entity at `index`.
    pub fn get_mut<T: Component>(&mut self, index: u32) -> Option<&mut T> {
        let col_idx = *self.column_index.get(&TypeId::of::<T>())?;
        let typed = self.columns[col_idx]
            .as_any_mut()?
            .downcast_mut::<TypedColumn<T>>()?;
        typed.get_mut(index)
    }

    /// Borrow distinct components with shared access to `A` and mutable
    /// access to `B`, splitting the column slice to prove disjointness.
    pub(crate) fn get_shared_mut<A: Component, B: Component>(
        &mut self,
        index: u32,
    ) -> Option<(&A, &mut B)> {
        let a_idx = *self.column_index.get(&TypeId::of::<A>())?;
        let b_idx = *self.column_index.get(&TypeId::of::<B>())?;
        if a_idx == b_idx {
            return None;
        }
        if a_idx < b_idx {
            let (left, right) = self.columns.split_at_mut(b_idx);
            let a = left[a_idx]
                .as_any()
                .downcast_ref::<TypedColumn<A>>()?
                .get(index)?;
            let b = right[0]
                .as_any_mut()?
                .downcast_mut::<TypedColumn<B>>()?
                .get_mut(index)?;
            Some((a, b))
        } else {
            let (left, right) = self.columns.split_at_mut(a_idx);
            let b = left[b_idx]
                .as_any_mut()?
                .downcast_mut::<TypedColumn<B>>()?
                .get_mut(index)?;
            let a = right[0]
                .as_any()
                .downcast_ref::<TypedColumn<A>>()?
                .get(index)?;
            Some((a, b))
        }
    }

    /// Borrow distinct components with mutable access to `A` and shared
    /// access to `B`.
    pub(crate) fn get_mut_shared<A: Component, B: Component>(
        &mut self,
        index: u32,
    ) -> Option<(&mut A, &B)> {
        let a_idx = *self.column_index.get(&TypeId::of::<A>())?;
        let b_idx = *self.column_index.get(&TypeId::of::<B>())?;
        if a_idx == b_idx {
            return None;
        }
        if a_idx < b_idx {
            let (left, right) = self.columns.split_at_mut(b_idx);
            let a = left[a_idx]
                .as_any_mut()?
                .downcast_mut::<TypedColumn<A>>()?
                .get_mut(index)?;
            let b = right[0]
                .as_any()
                .downcast_ref::<TypedColumn<B>>()?
                .get(index)?;
            Some((a, b))
        } else {
            let (left, right) = self.columns.split_at_mut(a_idx);
            let b = left[b_idx]
                .as_any()
                .downcast_ref::<TypedColumn<B>>()?
                .get(index)?;
            let a = right[0]
                .as_any_mut()?
                .downcast_mut::<TypedColumn<A>>()?
                .get_mut(index)?;
            Some((a, b))
        }
    }

    /// Borrow two distinct components mutably by splitting the column
    /// slice, which makes their non-aliasing explicit to the borrow checker.
    pub(crate) fn get_two_mut<A: Component, B: Component>(
        &mut self,
        index: u32,
    ) -> Option<(&mut A, &mut B)> {
        let a_idx = *self.column_index.get(&TypeId::of::<A>())?;
        let b_idx = *self.column_index.get(&TypeId::of::<B>())?;
        if a_idx == b_idx {
            return None;
        }
        if a_idx < b_idx {
            let (left, right) = self.columns.split_at_mut(b_idx);
            let a = left[a_idx]
                .as_any_mut()?
                .downcast_mut::<TypedColumn<A>>()?
                .get_mut(index)?;
            let b = right[0]
                .as_any_mut()?
                .downcast_mut::<TypedColumn<B>>()?
                .get_mut(index)?;
            Some((a, b))
        } else {
            let (left, right) = self.columns.split_at_mut(a_idx);
            let b = left[b_idx]
                .as_any_mut()?
                .downcast_mut::<TypedColumn<B>>()?
                .get_mut(index)?;
            let a = right[0]
                .as_any_mut()?
                .downcast_mut::<TypedColumn<A>>()?
                .get_mut(index)?;
            Some((a, b))
        }
    }

    /// Entities in this archetype, in row order.
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Type-erased columns, one per component type. Indexed in lock-step
    /// with [`Self::component_types`] and [`Self::component_names`]. The
    /// returned trait objects expose [`crate::archetype::ErasedColumn::value_as_any`]
    /// for the runtime ECS inspector to read components back as `&dyn Any`.
    pub fn columns(&self) -> &[Box<dyn ErasedColumn>] {
        &self.columns
    }

    /// Remove the entity at `index` (swap-remove). Returns the entity
    /// that was moved into the vacated slot, if any — the caller must
    /// update that entity's `EntityLocation` to point at `index`.
    pub(crate) fn swap_remove_entity(&mut self, index: u32) -> Option<Entity> {
        let last = self.entities.len().saturating_sub(1);
        let moved = if (index as usize) != last {
            Some(self.entities[last])
        } else {
            None
        };
        self.entities.swap_remove(index as usize);
        for col in &mut self.columns {
            col.take_any(index);
        }
        moved
    }
}
