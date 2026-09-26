//! Query APIs for iterating entities that match a component pattern.
//!
//! [`Query`] supports shared-reference iteration through [`World::query`](crate::World::query).
//! Queries containing mutable references implement [`QueryMut`] and are available only
//! through [`World::for_each_mut`](crate::World::for_each_mut), whose exclusive world
//! borrow and higher-ranked callback prevent component references from escaping.

use std::any::TypeId;
use std::marker::PhantomData;

use crate::archetype::Archetype;
use crate::component::Component;
use crate::entity::Entity;

/// A read-only query for entities matching a set of component types.
///
/// Implemented for `&A` and tuples of shared references up to length 4.
pub trait Query {
    type Item<'a>;

    /// Returns true iff this query matches `archetype`.
    fn matches(archetype: &Archetype) -> bool;

    /// Fetch the item for the entity at `index` in `archetype`.
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a>;
}

/// A query that may mutably access components.
///
/// This trait is used only by [`World::for_each_mut`](crate::World::for_each_mut).
/// The query fetches from an exclusively borrowed archetype and validates that
/// shared and mutable components in a tuple do not have the same type.
pub trait QueryMut {
    type Item<'a>;

    /// Returns true iff this query matches `archetype`.
    fn matches(archetype: &Archetype) -> bool;

    /// Panic if the query's component references could alias.
    fn assert_no_aliasing();

    /// Fetch a row while the archetype is exclusively borrowed.
    fn fetch<'a>(archetype: &'a mut Archetype, index: usize) -> Self::Item<'a>;
}

impl<A: Component> Query for &A {
    type Item<'a> = &'a A;

    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>()
    }

    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
        archetype
            .get::<A>(index as u32)
            .expect("query fetch: component A not present")
    }
}

macro_rules! impl_read_query_tuple {
    ($(($($component:ident),+)),+ $(,)?) => {
        $(
            impl<$($component: Component),+> Query for ($(& $component,)+) {
                type Item<'a> = ($(&'a $component,)+);

                fn matches(archetype: &Archetype) -> bool {
                    true $(&& archetype.has::<$component>())+
                }

                #[allow(non_snake_case)]
                fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
                    ($(
                        archetype
                            .get::<$component>(index as u32)
                            .expect("query fetch: required component not present"),
                    )+)
                }
            }
        )+
    };
}

impl_read_query_tuple!((A), (A, B), (A, B, C), (A, B, C, D));

impl<A: Component> QueryMut for &mut A {
    type Item<'a> = &'a mut A;

    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>()
    }

    fn assert_no_aliasing() {}

    fn fetch<'a>(archetype: &'a mut Archetype, index: usize) -> Self::Item<'a> {
        archetype
            .get_mut::<A>(index as u32)
            .expect("mutable query fetch: component A not present")
    }
}

impl<A: Component> QueryMut for (&mut A,) {
    type Item<'a> = (&'a mut A,);

    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>()
    }

    fn assert_no_aliasing() {}

    fn fetch<'a>(archetype: &'a mut Archetype, index: usize) -> Self::Item<'a> {
        (archetype
            .get_mut::<A>(index as u32)
            .expect("mutable query fetch: component A not present"),)
    }
}

impl<A: Component, B: Component> QueryMut for (&A, &mut B) {
    type Item<'a> = (&'a A, &'a mut B);

    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>() && archetype.has::<B>()
    }

    fn assert_no_aliasing() {
        assert!(
            TypeId::of::<A>() != TypeId::of::<B>(),
            "query contains aliased shared and mutable component types"
        );
    }

    fn fetch<'a>(archetype: &'a mut Archetype, index: usize) -> Self::Item<'a> {
        archetype
            .get_shared_mut::<A, B>(index as u32)
            .expect("mutable query fetch: required components not present")
    }
}

impl<A: Component, B: Component> QueryMut for (&mut A, &B) {
    type Item<'a> = (&'a mut A, &'a B);

    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>() && archetype.has::<B>()
    }

    fn assert_no_aliasing() {
        assert!(
            TypeId::of::<A>() != TypeId::of::<B>(),
            "query contains aliased mutable and shared component types"
        );
    }

    fn fetch<'a>(archetype: &'a mut Archetype, index: usize) -> Self::Item<'a> {
        archetype
            .get_mut_shared::<A, B>(index as u32)
            .expect("mutable query fetch: required components not present")
    }
}

impl<A: Component, B: Component> QueryMut for (&mut A, &mut B) {
    type Item<'a> = (&'a mut A, &'a mut B);

    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>() && archetype.has::<B>()
    }

    fn assert_no_aliasing() {
        assert!(
            TypeId::of::<A>() != TypeId::of::<B>(),
            "query contains duplicate mutable component types"
        );
    }

    fn fetch<'a>(archetype: &'a mut Archetype, index: usize) -> Self::Item<'a> {
        archetype
            .get_two_mut::<A, B>(index as u32)
            .expect("mutable query fetch: required components not present")
    }
}

/// Lazy iterator over the matching `(Entity, item)` pairs of a read-only [`Query`].
pub struct QueryIter<'w, Q: Query> {
    archetypes: std::iter::Enumerate<std::slice::Iter<'w, Archetype>>,
    current: Option<&'w Archetype>,
    current_index: usize,
    _marker: PhantomData<Q>,
}

impl<'w, Q: Query> QueryIter<'w, Q> {
    pub(crate) fn new(world: &'w crate::World) -> Self {
        Self {
            archetypes: world.archetypes().iter().enumerate(),
            current: None,
            current_index: 0,
            _marker: PhantomData,
        }
    }
}

impl<'w, Q: Query> Iterator for QueryIter<'w, Q> {
    type Item = (Entity, Q::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(arch) = self.current {
                if self.current_index < arch.len() {
                    let entity = arch.entities()[self.current_index];
                    let item = Q::fetch(arch, self.current_index);
                    self.current_index += 1;
                    return Some((entity, item));
                }
            }
            self.current_index = 0;
            self.current = None;
            for (_, arch) in self.archetypes.by_ref() {
                if Q::matches(arch) {
                    self.current = Some(arch);
                    break;
                }
            }
            self.current?;
        }
    }
}
