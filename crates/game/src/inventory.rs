//! Survival inventory: the single inventory implementation — contents,
//! selection state, and tool-tier metadata in one component.
//!
//! Slot layout: 27 main slots, 9 hotbar slots, 4 armor slots, 1 offhand slot,
//! and a 2×2 crafting grid with output. The hotbar row (slots 0..9) backs the
//! on-screen quick bar; `selected` indexes into it.
//!
//! The engine reads/writes this component on the player entity; gameplay
//! systems (mining, pickup, armor) query it directly, so there is no separate
//! engine-side hotbar struct to keep in sync.

use serde::{Deserialize, Serialize};
use voxel_core::BlockId;
use voxel_world::BlockRegistry;

use super::items::{ItemStack, MAX_STACK_SIZE};

/// Number of hotbar slots (Minecraft-like).
pub const HOTBAR_SLOTS: usize = 9;
/// Main inventory slots (3 rows × 9 columns).
pub const MAIN_SLOTS: usize = 27;
/// Armor slots (4 pieces).
pub const ARMOR_SLOTS: usize = 4;
/// Crafting input slots (2×2 grid).
pub const CRAFTING_INPUT_SLOTS: usize = 4;

/// Armor slot indices.
pub const ARMOR_HELMET: usize = 0;
pub const ARMOR_CHESTPLATE: usize = 1;
pub const ARMOR_LEGGINGS: usize = 2;
pub const ARMOR_BOOTS: usize = 3;

/// Full survival inventory component.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurvivalInventory {
    /// Main inventory (27 slots, rows 1-3).
    pub main: [ItemStack; MAIN_SLOTS],
    /// Hotbar (9 slots, row 0). The on-screen quick bar.
    pub hotbar: [ItemStack; HOTBAR_SLOTS],
    /// Armor slots (helmet, chestplate, leggings, boots).
    pub armor: [ItemStack; ARMOR_SLOTS],
    /// Offhand slot (shield, torches, etc.).
    pub offhand: ItemStack,
    /// Crafting input (2×2 grid).
    pub crafting_input: [ItemStack; CRAFTING_INPUT_SLOTS],
    /// Crafting output.
    pub crafting_output: ItemStack,
    /// Currently selected hotbar slot (0..HOTBAR_SLOTS).
    pub selected: usize,
    /// Minimum tool tier carried per hotbar slot. The inventory is
    /// block-oriented, so this metadata is explicit until tool items receive
    /// their own registry. Reset whenever a slot is replaced.
    pub tool_tiers: [u8; HOTBAR_SLOTS],
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
            selected: 0,
            tool_tiers: [0; HOTBAR_SLOTS],
        }
    }
}

/// Which storage array a slot range lives in.
#[derive(Clone, Copy)]
enum Zone {
    Hotbar,
    Main,
}

impl SurvivalInventory {
    /// Create a new empty inventory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fill the hotbar with a default creative palette from the registry.
    pub fn populate_defaults(&mut self, reg: &BlockRegistry) {
        let names = [
            "grass",
            "dirt",
            "stone",
            "cobblestone",
            "planks",
            "wood",
            "torch",
            "sand",
            "glass",
        ];
        for (i, name) in names.iter().enumerate() {
            if let Some(id) = reg.id_of(name) {
                self.hotbar[i] = ItemStack::new(id, 1);
            }
        }
    }

    // --- Hotbar selection -------------------------------------------------

    /// Select a hotbar slot by index (0..=8). Out-of-range values are ignored.
    pub fn select(&mut self, i: usize) {
        if i < HOTBAR_SLOTS {
            self.selected = i;
        }
    }

    /// Cycle the hotbar selection by a delta (mouse wheel), wrapping.
    pub fn cycle_selection(&mut self, delta: i32) {
        let n = HOTBAR_SLOTS as i32;
        let mut s = self.selected as i32 + delta;
        s = ((s % n) + n) % n;
        self.selected = s as usize;
    }

    /// Block id held in hotbar slot `i`, or `None` if `i` is out of range or
    /// the slot is empty.
    pub fn hotbar_block(&self, i: usize) -> Option<BlockId> {
        let stack = self.hotbar.get(i)?;
        if stack.is_empty() {
            None
        } else {
            Some(stack.id())
        }
    }

    /// Block id in the selected hotbar slot, or `None` if the slot is empty.
    /// (Out-of-range `selected` is impossible through `select`, but the
    /// indexed lookup stays defensive.)
    pub fn selected_block(&self) -> Option<BlockId> {
        self.hotbar_block(self.selected)
    }

    /// Tool tier carried by the selected hotbar slot; zero means hand/no tool.
    pub fn selected_tool_tier(&self) -> u8 {
        self.tool_tiers.get(self.selected).copied().unwrap_or(0)
    }

    /// Set a hotbar slot to hold a single item of `id`, resetting its tool
    /// tier. Out-of-range indices are ignored.
    ///
    /// The hotbar is block-oriented today (counts are not surfaced in the UI),
    /// so this models "one block per slot"; callers that need stacked items
    /// should use `set_slot` with an explicit [`ItemStack`].
    pub fn set_hotbar_block(&mut self, i: usize, id: BlockId) {
        if i < HOTBAR_SLOTS {
            self.hotbar[i] = ItemStack::new(id, 1);
            self.tool_tiers[i] = 0;
        }
    }

    /// Set the minimum tool tier carried by a hotbar slot.
    pub fn set_slot_tool_tier(&mut self, i: usize, tier: u8) {
        if i < HOTBAR_SLOTS {
            self.tool_tiers[i] = tier;
        }
    }

    // --- Generic slot access ----------------------------------------------

    /// Get a reference to all 36 storage slots (main + hotbar) as a flat list.
    pub fn all_slots(&self) -> Vec<&ItemStack> {
        let mut slots = Vec::with_capacity(MAIN_SLOTS + HOTBAR_SLOTS);
        slots.extend(self.main.iter());
        slots.extend(self.hotbar.iter());
        slots
    }

    /// Get a mutable reference to all 36 storage slots (main + hotbar).
    pub fn all_slots_mut(&mut self) -> Vec<&mut ItemStack> {
        let mut slots = Vec::with_capacity(MAIN_SLOTS + HOTBAR_SLOTS);
        slots.extend(self.main.iter_mut());
        slots.extend(self.hotbar.iter_mut());
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
        let r = self.try_merge_into(&remainder, 0..HOTBAR_SLOTS, Zone::Hotbar)?;
        if r.count < remainder.count {
            merged_any = true;
        }
        remainder = r;
        if remainder.is_empty() {
            return None;
        }
        let r = self.try_merge_into(&remainder, 0..MAIN_SLOTS, Zone::Main)?;
        if r.count < remainder.count {
            merged_any = true;
        }
        remainder = r;
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
        remainder = self.try_insert_into_empty(&remainder, 0..HOTBAR_SLOTS, Zone::Hotbar)?;
        remainder = self.try_insert_into_empty(&remainder, 0..MAIN_SLOTS, Zone::Main)?;

        Some(remainder)
    }

    /// Try to merge a stack into existing stacks over `range` of one zone.
    fn try_merge_into(
        &mut self,
        stack: &ItemStack,
        range: std::ops::Range<usize>,
        zone: Zone,
    ) -> Option<ItemStack> {
        let mut remainder = *stack;
        let slots: &mut [ItemStack] = match zone {
            Zone::Hotbar => &mut self.hotbar,
            Zone::Main => &mut self.main,
        };
        for i in range {
            let slot = &mut slots[i];
            if !slot.is_empty() && slot.id() == remainder.id() {
                remainder = slot.merge_with(&remainder)?;
            }
        }
        Some(remainder)
    }

    /// Try to insert a stack into empty slots over `range` of one zone.
    ///
    /// Places up to `MAX_STACK_SIZE` items in each empty slot and returns the
    /// leftover remainder (if any). This prevents a single slot from exceeding
    /// the max stack size when the inserted stack is larger.
    fn try_insert_into_empty(
        &mut self,
        stack: &ItemStack,
        range: std::ops::Range<usize>,
        zone: Zone,
    ) -> Option<ItemStack> {
        let mut remainder = *stack;
        if remainder.is_empty() {
            return None;
        }

        let slots: &mut [ItemStack] = match zone {
            Zone::Hotbar => &mut self.hotbar,
            Zone::Main => &mut self.main,
        };
        for i in range {
            let slot = &mut slots[i];
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

    /// Insert a stack into hotbar slots only (merge first, then empty slots).
    /// Returns the remainder that didn't fit.
    pub fn insert_into_hotbar(&mut self, stack: ItemStack) -> Option<ItemStack> {
        let mut remainder = stack;
        remainder = self.try_merge_into(&remainder, 0..HOTBAR_SLOTS, Zone::Hotbar)?;
        if remainder.is_empty() {
            return None;
        }
        self.try_insert_into_empty(&remainder, 0..HOTBAR_SLOTS, Zone::Hotbar)
    }

    /// Insert a stack into main slots only (merge first, then empty slots).
    /// Returns the remainder that didn't fit.
    pub fn insert_into_main(&mut self, stack: ItemStack) -> Option<ItemStack> {
        let mut remainder = stack;
        remainder = self.try_merge_into(&remainder, 0..MAIN_SLOTS, Zone::Main)?;
        if remainder.is_empty() {
            return None;
        }
        self.try_insert_into_empty(&remainder, 0..MAIN_SLOTS, Zone::Main)
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

    /// Check if the player has a specific item in their inventory.
    pub fn has_item(&self, id: BlockId) -> bool {
        self.all_slots()
            .iter()
            .any(|slot| slot.id() == id && !slot.is_empty())
    }

    /// Count the total number of a specific item in the inventory.
    pub fn count_item(&self, id: BlockId) -> u16 {
        self.all_slots()
            .iter()
            .filter(|slot| slot.id() == id)
            .map(|slot| slot.count)
            .sum()
    }

    /// Remove a specific number of items from the inventory.
    /// Returns the number actually removed.
    pub fn remove_item(&mut self, id: BlockId, count: u16) -> u16 {
        let mut remaining = count;

        // Remove from hotbar first, then main.
        for zone in [&mut self.hotbar[..] as &mut [ItemStack], &mut self.main[..]] {
            if remaining == 0 {
                break;
            }
            for slot in zone.iter_mut() {
                if remaining == 0 {
                    break;
                }
                if slot.id() == id {
                    let to_remove = remaining.min(slot.count);
                    slot.count -= to_remove;
                    remaining -= to_remove;
                    if slot.count == 0 {
                        slot.clear();
                    }
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
        assert_eq!(inv.selected, 0);
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

    // --- Selection / hotbar access (absorbed from the old Hotbar) ---

    fn reg() -> BlockRegistry {
        BlockRegistry::with_builtins()
    }

    #[test]
    fn populate_defaults_fills_all_hotbar_slots() {
        let mut inv = SurvivalInventory::new();
        inv.populate_defaults(&reg());
        for i in 0..HOTBAR_SLOTS {
            assert!(
                matches!(inv.hotbar_block(i), Some(id) if !id.is_air()),
                "slot {i} should be filled"
            );
        }
    }

    #[test]
    fn select_within_range() {
        let mut inv = SurvivalInventory::new();
        inv.select(3);
        assert_eq!(inv.selected, 3);
    }

    #[test]
    fn select_out_of_range_ignored() {
        let mut inv = SurvivalInventory::new();
        inv.select(99);
        assert_eq!(inv.selected, 0);
    }

    #[test]
    fn cycle_selection_wraps_forward() {
        let mut inv = SurvivalInventory::new();
        inv.select(8);
        inv.cycle_selection(1);
        assert_eq!(inv.selected, 0);
    }

    #[test]
    fn cycle_selection_wraps_backward() {
        let mut inv = SurvivalInventory::new();
        inv.cycle_selection(-1);
        assert_eq!(inv.selected, HOTBAR_SLOTS - 1);
    }

    #[test]
    fn set_hotbar_block_and_selection() {
        let mut inv = SurvivalInventory::new();
        inv.set_hotbar_block(2, BlockId(5));
        assert_eq!(inv.hotbar_block(2), Some(BlockId(5)));
        inv.select(2);
        assert_eq!(inv.selected_block(), Some(BlockId(5)));
    }

    #[test]
    fn set_hotbar_block_out_of_range_ignored() {
        let mut inv = SurvivalInventory::new();
        let original = inv.hotbar_block(0);
        inv.set_hotbar_block(99, BlockId(5));
        assert_eq!(inv.hotbar_block(0), original);
    }

    #[test]
    fn hotbar_block_out_of_range_returns_none() {
        let inv = SurvivalInventory::new();
        assert_eq!(inv.hotbar_block(HOTBAR_SLOTS), None);
        assert_eq!(inv.hotbar_block(usize::MAX), None);
    }

    #[test]
    fn hotbar_block_empty_slot_returns_none() {
        let inv = SurvivalInventory::new();
        // Unset slots are empty stacks, so `hotbar_block` is `None` (the old
        // Hotbar returned `Some(BlockId::AIR)` here; the inventory models
        // "no item" explicitly).
        assert_eq!(inv.hotbar_block(0), None);
    }

    #[test]
    fn selected_block_out_of_range_returns_none() {
        // Defensive: even if `selected` gets corrupted, `selected_block()`
        // must never panic — it returns `None` so callers can early-return.
        let mut inv = SurvivalInventory::new();
        inv.select(0);
        assert_eq!(inv.selected_block(), None);
    }

    #[test]
    fn selected_tool_tier_tracks_selected_slot() {
        let mut inv = SurvivalInventory::new();
        inv.set_slot_tool_tier(2, 3);
        inv.select(2);
        assert_eq!(inv.selected_tool_tier(), 3);
        inv.select(0);
        assert_eq!(inv.selected_tool_tier(), 0);
    }

    #[test]
    fn replacing_hotbar_block_resets_tool_tier() {
        let mut inv = SurvivalInventory::new();
        inv.set_slot_tool_tier(2, 3);
        inv.set_hotbar_block(2, BlockId(5));
        assert_eq!(inv.selected_tool_tier(), 0);
        inv.select(2);
        assert_eq!(inv.selected_tool_tier(), 0);
    }
}
