//! Survival inventory: full inventory with armor slots, offhand, and crafting grid.
//!
//! This component represents the player's complete inventory in survival mode.
//! It contains 27 main slots, 9 hotbar slots, 4 armor slots, 1 offhand slot,
//! and a 2×2 crafting grid with output.

use serde::{Deserialize, Serialize};
use voxel_core::BlockId;

use super::items::{ItemStack, MAX_STACK_SIZE};

/// Armor slot indices.
pub const ARMOR_HELMET: usize = 0;
pub const ARMOR_CHESTPLATE: usize = 1;
pub const ARMOR_LEGGINGS: usize = 2;
pub const ARMOR_BOOTS: usize = 3;

/// Main inventory slots (3 rows × 9 columns).
pub const MAIN_SLOTS: usize = 27;
/// Hotbar slots (1 row × 9 columns).
pub const HOTBAR_SLOTS: usize = 9;
/// Armor slots (4 pieces).
pub const ARMOR_SLOTS: usize = 4;
/// Crafting input slots (2×2 grid).
pub const CRAFTING_INPUT_SLOTS: usize = 4;

/// Full survival inventory component.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurvivalInventory {
    /// Main inventory (27 slots, rows 1-3).
    pub main: [ItemStack; MAIN_SLOTS],
    /// Hotbar (9 slots, row 0).
    pub hotbar: [ItemStack; HOTBAR_SLOTS],
    /// Armor slots (helmet, chestplate, leggings, boots).
    pub armor: [ItemStack; ARMOR_SLOTS],
    /// Offhand slot (shield, torches, etc.).
    pub offhand: ItemStack,
    /// Crafting input (2×2 grid).
    pub crafting_input: [ItemStack; CRAFTING_INPUT_SLOTS],
    /// Crafting output.
    pub crafting_output: ItemStack,
}

impl Default for SurvivalInventory {
    fn default() -> Self {
        Self {
            main: [ItemStack::empty(); MAIN_SLOTS],
            hotbar: [ItemStack::empty(); HOTBAR_SLOTS],
            armor: [ItemStack::empty(); ARMOR_SLOTS],
            offhand: ItemStack::empty(),
            crafting_input: [ItemStack::empty(); CRAFTING_INPUT_SLOTS],
            crafting_output: ItemStack::empty(),
        }
    }
}

impl SurvivalInventory {
    /// Create a new empty inventory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a reference to all 36 storage slots (main + hotbar) as a flat array.
    pub fn all_slots(&self) -> Vec<&ItemStack> {
        let mut slots = Vec::with_capacity(MAIN_SLOTS + HOTBAR_SLOTS);
        for slot in self.main.iter() {
            slots.push(slot);
        }
        for slot in self.hotbar.iter() {
            slots.push(slot);
        }
        slots
    }

    /// Get a mutable reference to all 36 storage slots (main + hotbar).
    pub fn all_slots_mut(&mut self) -> Vec<&mut ItemStack> {
        let mut slots = Vec::with_capacity(MAIN_SLOTS + HOTBAR_SLOTS);
        for slot in self.main.iter_mut() {
            slots.push(slot);
        }
        for slot in self.hotbar.iter_mut() {
            slots.push(slot);
        }
        slots
    }

    /// Try to insert an item stack into the inventory.
    /// Returns the remainder that didn't fit (if any).
    ///
    /// Behavior:
    /// - If a partial merge occurs (any items absorbed by an existing stack),
    ///   the leftover remainder is returned without cascading to empty slots.
    ///   This is the expected behavior when an insert fills a stack exactly:
    ///   the overflow is handed back to the caller, not silently placed in
    ///   another slot.
    /// - If no merge occurs, the stack is placed in the first empty slot
    ///   (hotbar first, then main).
    /// - If the stack is fully consumed by a merge, returns `None`.
    pub fn insert(&mut self, stack: ItemStack) -> Option<ItemStack> {
        if stack.is_empty() {
            return None;
        }

        let mut remainder = stack;
        let mut merged_any = false;

        // First, try to merge into existing stacks. Hotbar first, then main.
        if let Some(r) = self.try_merge_into(&remainder, 0, HOTBAR_SLOTS, true) {
            if r.count < remainder.count {
                merged_any = true;
            }
            remainder = r;
        } else {
            return None;
        }
        if remainder.is_empty() {
            return None;
        }
        if let Some(r) = self.try_merge_into(&remainder, 0, MAIN_SLOTS, false) {
            if r.count < remainder.count {
                merged_any = true;
            }
            remainder = r;
        } else {
            return None;
        }
        if remainder.is_empty() {
            return None;
        }

        // If any merge happened, return the remainder — don't cascade to empty
        // slots. A partial merge means the caller (e.g. a UI drag, a pickup)
        // expects the overflow back rather than seeing it silently land in a
        // different slot.
        if merged_any {
            return Some(remainder);
        }

        // No merge happened, so try to insert into empty slots.
        match self.try_insert_into_empty(&remainder, 0, HOTBAR_SLOTS, true) {
            Some(r) => remainder = r,
            None => return None,
        }
        match self.try_insert_into_empty(&remainder, 0, MAIN_SLOTS, false) {
            Some(r) => remainder = r,
            None => return None,
        }

        Some(remainder)
    }

    /// Try to merge a stack into existing stacks in the specified range.
    fn try_merge_into(&mut self, stack: &ItemStack, start: usize, end: usize, is_hotbar: bool) -> Option<ItemStack> {
        let mut remainder = *stack;

        for i in start..end {
            let slot = if is_hotbar {
                &mut self.hotbar[i]
            } else {
                &mut self.main[i]
            };

            if !slot.is_empty() && slot.id() == remainder.id() {
                let merged = slot.merge_with(&remainder);
                match merged {
                    Some(r) => remainder = r,
                    None => return None,
                }
            }
        }

        Some(remainder)
    }

    /// Try to insert a stack into empty slots in the specified range.
    ///
    /// Places up to `MAX_STACK_SIZE` items in each empty slot and returns the
    /// leftover remainder (if any). This prevents a single slot from exceeding
    /// the max stack size when the inserted stack is larger.
    fn try_insert_into_empty(&mut self, stack: &ItemStack, start: usize, end: usize, is_hotbar: bool) -> Option<ItemStack> {
        let mut remainder = *stack;
        if remainder.is_empty() {
            return None;
        }

        for i in start..end {
            let slot = if is_hotbar {
                &mut self.hotbar[i]
            } else {
                &mut self.main[i]
            };

            if slot.is_empty() {
                let to_place = remainder.count.min(MAX_STACK_SIZE);
                *slot = ItemStack::new(remainder.id(), to_place);
                remainder.count -= to_place;
                if remainder.is_empty() {
                    return None;
                }
            }
        }

        Some(remainder)
    }

    /// Swap two slots in the inventory.
    pub fn swap(&mut self, slot_a: InventorySlot, slot_b: InventorySlot) {
        // Get the values first to avoid borrow conflicts.
        let a_val = *self.get_slot(slot_a);
        let b_val = *self.get_slot(slot_b);
        self.set_slot(slot_a, b_val);
        self.set_slot(slot_b, a_val);
    }

    /// Get a reference to a specific slot.
    pub fn get_slot(&self, slot: InventorySlot) -> &ItemStack {
        match slot {
            InventorySlot::Main(i) => &self.main[i],
            InventorySlot::Hotbar(i) => &self.hotbar[i],
            InventorySlot::Armor(i) => &self.armor[i],
            InventorySlot::Offhand => &self.offhand,
            InventorySlot::CraftingInput(i) => &self.crafting_input[i],
            InventorySlot::CraftingOutput => &self.crafting_output,
        }
    }

    /// Get a mutable reference to a specific slot.
    pub fn get_slot_mut(&mut self, slot: InventorySlot) -> &mut ItemStack {
        match slot {
            InventorySlot::Main(i) => &mut self.main[i],
            InventorySlot::Hotbar(i) => &mut self.hotbar[i],
            InventorySlot::Armor(i) => &mut self.armor[i],
            InventorySlot::Offhand => &mut self.offhand,
            InventorySlot::CraftingInput(i) => &mut self.crafting_input[i],
            InventorySlot::CraftingOutput => &mut self.crafting_output,
        }
    }

    /// Set a specific slot to a stack.
    pub fn set_slot(&mut self, slot: InventorySlot, stack: ItemStack) {
        *self.get_slot_mut(slot) = stack;
    }

    /// Try to shift-click a slot (move item between hotbar and main, or auto-equip armor).
    pub fn shift_click(&mut self, slot: InventorySlot) {
        let stack = *self.get_slot(slot);
        if stack.is_empty() {
            return;
        }

        match slot {
            InventorySlot::Main(i) => {
                // Move from main to hotbar.
                let moved = self.main[i];
                self.main[i] = ItemStack::empty();
                if let Some(remainder) = self.insert_into_hotbar(moved) {
                    // Put remainder back in main.
                    self.main[i] = remainder;
                }
            }
            InventorySlot::Hotbar(i) => {
                // Move from hotbar to main.
                let moved = self.hotbar[i];
                self.hotbar[i] = ItemStack::empty();
                if let Some(remainder) = self.insert_into_main(moved) {
                    // Put remainder back in hotbar.
                    self.hotbar[i] = remainder;
                }
            }
            InventorySlot::Armor(_) => {
                // Move armor to main inventory.
                let moved = stack;
                *self.get_slot_mut(slot) = ItemStack::empty();
                if let Some(remainder) = self.insert_into_main(moved) {
                    *self.get_slot_mut(slot) = remainder;
                }
            }
            InventorySlot::Offhand => {
                // Move offhand to main inventory.
                let moved = self.offhand;
                self.offhand = ItemStack::empty();
                if let Some(remainder) = self.insert_into_main(moved) {
                    self.offhand = remainder;
                }
            }
            _ => {}
        }
    }

    /// Insert a stack into hotbar slots only.
    fn insert_into_hotbar(&mut self, stack: ItemStack) -> Option<ItemStack> {
        let mut remainder = stack;

        // Try to merge first.
        for i in 0..HOTBAR_SLOTS {
            if !self.hotbar[i].is_empty() && self.hotbar[i].id() == remainder.id() {
                let merged = self.hotbar[i].merge_with(&remainder);
                match merged {
                    Some(r) => remainder = r,
                    None => return None,
                }
            }
        }

        // Then try empty slots.
        for i in 0..HOTBAR_SLOTS {
            if self.hotbar[i].is_empty() {
                self.hotbar[i] = remainder;
                return None;
            }
        }

        Some(remainder)
    }

    /// Insert a stack into main slots only.
    fn insert_into_main(&mut self, stack: ItemStack) -> Option<ItemStack> {
        let mut remainder = stack;

        // Try to merge first.
        for i in 0..MAIN_SLOTS {
            if !self.main[i].is_empty() && self.main[i].id() == remainder.id() {
                let merged = self.main[i].merge_with(&remainder);
                match merged {
                    Some(r) => remainder = r,
                    None => return None,
                }
            }
        }

        // Then try empty slots.
        for i in 0..MAIN_SLOTS {
            if self.main[i].is_empty() {
                self.main[i] = remainder;
                return None;
            }
        }

        Some(remainder)
    }

    /// Get the hotbar as a fixed-size array reference (for compatibility with existing Hotbar).
    pub fn hotbar_array(&self) -> &[ItemStack; HOTBAR_SLOTS] {
        &self.hotbar
    }

    /// Set a hotbar slot.
    pub fn set_hotbar_slot(&mut self, index: usize, stack: ItemStack) {
        if index < HOTBAR_SLOTS {
            self.hotbar[index] = stack;
        }
    }

    /// Get a hotbar slot.
    pub fn get_hotbar_slot(&self, index: usize) -> ItemStack {
        if index < HOTBAR_SLOTS {
            self.hotbar[index]
        } else {
            ItemStack::empty()
        }
    }

    /// Check if the player has a specific item in their inventory.
    pub fn has_item(&self, id: BlockId) -> bool {
        for slot in self.all_slots() {
            if slot.id() == id && !slot.is_empty() {
                return true;
            }
        }
        false
    }

    /// Count the total number of a specific item in the inventory.
    pub fn count_item(&self, id: BlockId) -> u16 {
        let mut total = 0;
        for slot in self.all_slots() {
            if slot.id() == id {
                total += slot.count;
            }
        }
        total
    }

    /// Remove a specific number of items from the inventory.
    /// Returns the number actually removed.
    pub fn remove_item(&mut self, id: BlockId, count: u16) -> u16 {
        let mut remaining = count;

        // Remove from hotbar first, then main.
        for i in 0..HOTBAR_SLOTS {
            if remaining == 0 {
                break;
            }
            if self.hotbar[i].id() == id {
                let to_remove = remaining.min(self.hotbar[i].count);
                self.hotbar[i].count -= to_remove;
                remaining -= to_remove;
                if self.hotbar[i].count == 0 {
                    self.hotbar[i].clear();
                }
            }
        }

        for i in 0..MAIN_SLOTS {
            if remaining == 0 {
                break;
            }
            if self.main[i].id() == id {
                let to_remove = remaining.min(self.main[i].count);
                self.main[i].count -= to_remove;
                remaining -= to_remove;
                if self.main[i].count == 0 {
                    self.main[i].clear();
                }
            }
        }

        count - remaining
    }
}

/// An inventory slot identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InventorySlot {
    /// Main inventory slot (0-26).
    Main(usize),
    /// Hotbar slot (0-8).
    Hotbar(usize),
    /// Armor slot (0-3: helmet, chestplate, leggings, boots).
    Armor(usize),
    /// Offhand slot.
    Offhand,
    /// Crafting input slot (0-3).
    CraftingInput(usize),
    /// Crafting output slot.
    CraftingOutput,
}

impl InventorySlot {
    /// Get the display name for this slot type.
    pub fn display_name(&self) -> &'static str {
        match self {
            InventorySlot::Main(_) => "Main",
            InventorySlot::Hotbar(_) => "Hotbar",
            InventorySlot::Armor(i) => match i {
                0 => "Helmet",
                1 => "Chestplate",
                2 => "Leggings",
                3 => "Boots",
                _ => "Armor",
            },
            InventorySlot::Offhand => "Offhand",
            InventorySlot::CraftingInput(_) => "Crafting",
            InventorySlot::CraftingOutput => "Output",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_inventory() {
        let inv = SurvivalInventory::new();
        assert!(inv.main.iter().all(|s| s.is_empty()));
        assert!(inv.hotbar.iter().all(|s| s.is_empty()));
        assert!(inv.armor.iter().all(|s| s.is_empty()));
        assert!(inv.offhand.is_empty());
    }

    #[test]
    fn insert_into_empty() {
        let mut inv = SurvivalInventory::new();
        let stack = ItemStack::new(BlockId(1), 10);
        let remainder = inv.insert(stack);
        assert!(remainder.is_none());
        assert_eq!(inv.hotbar[0].count, 10);
    }

    #[test]
    fn insert_merge_existing() {
        let mut inv = SurvivalInventory::new();
        inv.hotbar[0] = ItemStack::new(BlockId(1), 10);
        let stack = ItemStack::new(BlockId(1), 5);
        let remainder = inv.insert(stack);
        assert!(remainder.is_none());
        assert_eq!(inv.hotbar[0].count, 15);
    }

    #[test]
    fn insert_overflow() {
        let mut inv = SurvivalInventory::new();
        inv.hotbar[0] = ItemStack::new(BlockId(1), 60);
        let stack = ItemStack::new(BlockId(1), 10);
        let remainder = inv.insert(stack);
        assert!(remainder.is_some());
        assert_eq!(inv.hotbar[0].count, 64);
        assert_eq!(remainder.unwrap().count, 6);
    }

    #[test]
    fn insert_large_stack_splits_across_slots() {
        let mut inv = SurvivalInventory::new();
        let stack = ItemStack::new(BlockId(1), 100);
        let remainder = inv.insert(stack);
        assert!(remainder.is_none());
        // 100 items split into 64 + 36 across two slots.
        assert_eq!(inv.hotbar[0].count, 64);
        assert_eq!(inv.hotbar[1].count, 36);
        // No slot exceeds the max stack size.
        for slot in inv.hotbar.iter() {
            assert!(slot.count <= MAX_STACK_SIZE);
        }
    }

    #[test]
    fn shift_click_main_to_hotbar() {
        let mut inv = SurvivalInventory::new();
        inv.main[0] = ItemStack::new(BlockId(1), 10);
        inv.shift_click(InventorySlot::Main(0));
        assert!(inv.main[0].is_empty());
        assert_eq!(inv.hotbar[0].count, 10);
    }

    #[test]
    fn shift_click_hotbar_to_main() {
        let mut inv = SurvivalInventory::new();
        inv.hotbar[0] = ItemStack::new(BlockId(1), 10);
        inv.shift_click(InventorySlot::Hotbar(0));
        assert!(inv.hotbar[0].is_empty());
        assert_eq!(inv.main[0].count, 10);
    }

    #[test]
    fn remove_items() {
        let mut inv = SurvivalInventory::new();
        inv.hotbar[0] = ItemStack::new(BlockId(1), 10);
        inv.main[0] = ItemStack::new(BlockId(1), 5);
        let removed = inv.remove_item(BlockId(1), 12);
        assert_eq!(removed, 12);
        assert!(inv.hotbar[0].is_empty());
        assert_eq!(inv.main[0].count, 3);
    }
}
