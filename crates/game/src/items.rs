//! Item system: ItemStack, item types, and item properties.
//!
//! This module provides the foundation for the inventory system. Items can be
//! blocks, tools, food, armor, or misc items. Each item has an ID that maps
//! to either a BlockId (for placeable blocks) or a non-block item ID.

use serde::{Deserialize, Serialize};
use voxel_core::BlockId;

/// Maximum stack size for most items.
pub const MAX_STACK_SIZE: u16 = 64;

/// An item stack with count and damage tracking.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ItemStack {
    /// The block/item ID as raw u16 (0 = empty/air).
    id_raw: u16,
    /// Number of items in this stack.
    pub count: u16,
    /// Damage value for tools/armor (0 = pristine).
    pub damage: u16,
}

impl Default for ItemStack {
    fn default() -> Self {
        Self::empty()
    }
}

impl ItemStack {
    /// Create an empty (air) stack.
    pub fn empty() -> Self {
        Self {
            id_raw: 0,
            count: 0,
            damage: 0,
        }
    }

    /// Create a stack with the given BlockId and count.
    pub fn new(id: BlockId, count: u16) -> Self {
        Self {
            id_raw: id.raw(),
            count,
            damage: 0,
        }
    }

    /// Create a single item stack.
    pub fn single(id: BlockId) -> Self {
        Self::new(id, 1)
    }

    /// Get the BlockId for this stack.
    pub fn id(&self) -> BlockId {
        BlockId::new(self.id_raw)
    }

    /// Set the BlockId for this stack.
    pub fn set_id(&mut self, id: BlockId) {
        self.id_raw = id.raw();
    }

    /// Whether this stack is empty (air or zero count).
    pub fn is_empty(&self) -> bool {
        self.id_raw == 0 || self.count == 0
    }

    /// Whether this stack is a block that can be placed.
    pub fn is_block(&self) -> bool {
        self.id_raw != 0 && self.count > 0
    }

    /// Clear this stack to empty.
    pub fn clear(&mut self) {
        *self = Self::empty();
    }

    /// Try to merge another stack into this one.
    /// Returns the remainder that didn't fit (if any).
    pub fn merge_with(&mut self, other: &ItemStack) -> Option<ItemStack> {
        if self.is_empty() {
            *self = *other;
            return None;
        }
        if self.id_raw != other.id_raw {
            return Some(*other);
        }
        // For now, use a default max stack size. This should be looked up
        // from the item registry based on the item type.
        let max = MAX_STACK_SIZE;
        let space = max.saturating_sub(self.count);
        if space >= other.count {
            self.count += other.count;
            None
        } else {
            self.count = max;
            let mut remainder = *other;
            remainder.count -= space;
            Some(remainder)
        }
    }

    /// Split this stack, taking `n` items. Returns the taken portion.
    pub fn split(&mut self, n: u16) -> ItemStack {
        let taken = n.min(self.count);
        self.count -= taken;
        let result = ItemStack {
            id_raw: self.id_raw,
            count: taken,
            damage: self.damage,
        };
        if self.count == 0 {
            self.clear();
        }
        result
    }

    /// Take one item from this stack. Returns the taken item.
    pub fn take_one(&mut self) -> ItemStack {
        self.split(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stack() {
        let s = ItemStack::empty();
        assert!(s.is_empty());
        assert!(!s.is_block());
    }

    #[test]
    fn single_item() {
        let mut s = ItemStack::single(BlockId(1));
        assert!(!s.is_empty());
        assert!(s.is_block());
        assert_eq!(s.count, 1);

        let taken = s.take_one();
        assert_eq!(taken.count, 1);
        assert!(s.is_empty());
    }

    #[test]
    fn merge_compatible() {
        let mut a = ItemStack::new(BlockId(1), 10);
        let b = ItemStack::new(BlockId(1), 5);
        let remainder = a.merge_with(&b);
        assert!(remainder.is_none());
        assert_eq!(a.count, 15);
    }

    #[test]
    fn merge_incompatible() {
        let mut a = ItemStack::new(BlockId(1), 10);
        let b = ItemStack::new(BlockId(2), 5);
        let remainder = a.merge_with(&b);
        assert!(remainder.is_some());
        assert_eq!(a.count, 10);
    }

    #[test]
    fn split_stack() {
        let mut s = ItemStack::new(BlockId(1), 10);
        let taken = s.split(3);
        assert_eq!(taken.count, 3);
        assert_eq!(s.count, 7);
    }
}
