//! Armor system: defense calculation and durability.
//!
//! This system runs each tick and:
//! 1. Computes total defense from equipped armor
//! 2. Applies damage reduction
//! 3. Decrements armor durability on hit

use voxel_combat::DamageQueue;
use voxel_core::BlockId;
use voxel_ecs::World;
use voxel_gamemode::GameMode;

use crate::components::PlayerEntity;
use crate::inventory::SurvivalInventory;
use crate::systems::PhysicsWorldRes;

/// Armor properties for an item.
#[derive(Clone, Copy, Debug)]
pub struct ArmorProperties {
    /// Defense points (1..8).
    pub defense_points: u8,
    /// Toughness (0..4). Reduces high-damage penetration.
    pub toughness: u8,
    /// Maximum durability.
    pub max_durability: u16,
}

/// Compute total defense from armor slots.
pub fn compute_armor_defense(
    inventory: &SurvivalInventory,
    registry: &voxel_world::BlockRegistry,
) -> (u8, u8) {
    let mut total_defense = 0u8;
    let mut total_toughness = 0u8;

    for slot in &inventory.armor {
        if !slot.is_empty() {
            let (defense, toughness) = get_armor_stats(slot.id(), registry);
            total_defense += defense;
            total_toughness += toughness;
        }
    }

    (total_defense, total_toughness)
}

/// Get armor stats for a block ID. Returns (defense_points, toughness).
/// Uses the block name to determine armor properties.
/// This will be replaced by a proper item registry lookup when armor items are added.
fn get_armor_stats(id: BlockId, registry: &voxel_world::BlockRegistry) -> (u8, u8) {
    let def = registry.get(id);
    let name = def.name.as_ref();
    // Map known armor material names to defense values.
    // Format: "material_slot" e.g. "iron_helmet", "diamond_chestplate"
    let (base_defense, toughness) = if name.contains("leather") {
        (1, 0)
    } else if name.contains("chain") {
        (2, 0)
    } else if name.contains("iron") {
        (3, 0)
    } else if name.contains("diamond") {
        (4, 2)
    } else if name.contains("netherite") {
        (4, 3)
    } else if name.contains("gold") {
        (1, 0)
    } else {
        return (0, 0); // Not armor
    };

    // Adjust defense by slot type.
    let slot_defense = if name.contains("helmet") || name.contains("cap") {
        base_defense
    } else if name.contains("chestplate") || name.contains("tunic") {
        base_defense * 2 // chestplate has more defense
    } else if name.contains("leggings") || name.contains("pants") {
        (base_defense as f32 * 1.5) as u8
    } else if name.contains("boots") {
        base_defense
    } else {
        return (0, 0); // Not a recognized armor slot
    };

    (slot_defense, toughness)
}

/// Apply armor reduction to damage.
pub fn apply_armor_reduction(damage: f32, total_defense: u8, _total_toughness: u8) -> f32 {
    // Formula: damage_after = damage * (1 - min(total_defense / 25, 0.8))
    let defense_factor = 1.0 - (total_defense as f32 / 25.0).min(0.8);
    damage * defense_factor
}

/// Armor system entry point. Called each fixed timestep.
pub fn armor_system(world: &mut World, _dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let game_mode = world
        .get::<GameMode>(player_entity)
        .copied()
        .unwrap_or(GameMode::Survival);
    if !game_mode.can_have_armor() {
        return;
    }

    // Get the physics world for registry access.
    let physics = world.resource::<PhysicsWorldRes>().cloned();
    let Some(phys) = physics else { return };
    let registry = phys.0.registry();

    // Get the inventory (clone to avoid borrow issues).
    let inventory = match world.get::<SurvivalInventory>(player_entity) {
        Some(inv) => inv.clone(),
        None => return,
    };

    // Compute armor defense.
    let (total_defense, total_toughness) = compute_armor_defense(&inventory, &registry);

    // Process damage events with armor reduction.
    if let Some(dq) = world.resource_mut::<DamageQueue>() {
        for (entity_opt, event) in dq.events.iter_mut() {
            let Some(entity) = entity_opt else { continue };
            if *entity != player_entity {
                continue;
            }

            // Apply armor reduction.
            if total_defense > 0 {
                event.amount = apply_armor_reduction(event.amount, total_defense, total_toughness);
            }
        }
    }
}
