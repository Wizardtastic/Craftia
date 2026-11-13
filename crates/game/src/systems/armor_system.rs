//! Armor system: defense calculation, vanilla damage reduction, durability.
//!
//! Reduction follows the modern vanilla formula:
//!
//! ```text
//! reduction = min(20, max(defense / 5, defense − damage / (2 + toughness / 4)))
//! final     = damage × (1 − reduction / 25)
//! ```
//!
//! Armor points and toughness match vanilla materials. Each hit damages every
//! worn piece by `max(1, damage / 4)`; pieces break at their vanilla durability.

use voxel_combat::DamageQueue;
use voxel_core::BlockId;
use voxel_ecs::World;
use voxel_gamemode::GameMode;

use crate::components::PlayerEntity;
use crate::inventory::SurvivalInventory;
use crate::systems::PhysicsWorldRes;

/// Armor attributes for one material: defense per slot (helmet, chestplate,
/// leggings, boots), toughness, durability per slot.
struct ArmorMaterial {
    defense: [u8; 4],
    toughness: u8,
    durability: [u16; 4],
}

/// Resolve a material name from an armor item name.
fn armor_material(name: &str) -> Option<ArmorMaterial> {
    let n = name.to_lowercase();
    let material = if n.contains("leather") {
        ArmorMaterial {
            defense: [1, 3, 2, 1],
            toughness: 0,
            durability: [55, 80, 75, 65],
        }
    } else if n.contains("chain") {
        ArmorMaterial {
            defense: [2, 5, 4, 1],
            toughness: 0,
            durability: [165, 240, 225, 195],
        }
    } else if n.contains("netherite") {
        ArmorMaterial {
            defense: [3, 8, 6, 3],
            toughness: 3,
            durability: [407, 592, 555, 481],
        }
    } else if n.contains("diamond") {
        ArmorMaterial {
            defense: [3, 8, 6, 3],
            toughness: 2,
            durability: [363, 528, 495, 429],
        }
    } else if n.contains("gold") {
        ArmorMaterial {
            defense: [2, 5, 3, 1],
            toughness: 0,
            durability: [77, 112, 105, 91],
        }
    } else if n.contains("iron") {
        ArmorMaterial {
            defense: [2, 6, 5, 2],
            toughness: 0,
            durability: [165, 240, 225, 195],
        }
    } else {
        return None;
    };
    Some(material)
}

/// Which armor slot a name refers to.
fn armor_slot(name: &str) -> Option<usize> {
    let n = name.to_lowercase();
    if n.contains("helmet") || n.contains("cap") {
        Some(0)
    } else if n.contains("chestplate") || n.contains("tunic") {
        Some(1)
    } else if n.contains("leggings") || n.contains("pants") {
        Some(2)
    } else if n.contains("boots") {
        Some(3)
    } else {
        None
    }
}

/// Compute total defense and toughness from the four armor slots.
pub fn compute_armor_defense(
    inventory: &SurvivalInventory,
    registry: &voxel_world::BlockRegistry,
) -> (u8, u8) {
    let mut total_defense = 0u8;
    let mut total_toughness = 0u8;

    for (slot, stack) in inventory.armor.iter().enumerate() {
        if stack.is_empty() {
            continue;
        }
        if let Some((defense, toughness)) = armor_stats(stack.id(), registry) {
            // Only count a piece in its proper slot.
            if armor_slot(registry.get(stack.id()).name.as_ref()) == Some(slot) {
                total_defense += defense;
                total_toughness += toughness;
            }
        }
    }

    (total_defense, total_toughness)
}

/// Get (defense, toughness) for an armor block id.
fn armor_stats(
    id: BlockId,
    registry: &voxel_world::BlockRegistry,
) -> Option<(u8, u8)> {
    let name = registry.get(id).name.as_ref().to_string();
    let material = armor_material(&name)?;
    let slot = armor_slot(&name)?;
    Some((material.defense[slot], material.toughness))
}

/// Vanilla armor damage reduction formula.
pub fn apply_armor_reduction(damage: f32, total_defense: u8, total_toughness: u8) -> f32 {
    if damage <= 0.0 || total_defense == 0 {
        return damage.max(0.0);
    }
    let defense = total_defense as f32;
    let toughness = total_toughness as f32;
    let reduction = (defense / 5.0)
        .max(defense - damage / (2.0 + toughness / 4.0))
        .min(20.0);
    (damage * (1.0 - reduction / 25.0)).max(0.0)
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

    let physics = world.resource::<PhysicsWorldRes>().cloned();
    let Some(phys) = physics else { return };
    let registry = phys.0.registry();

    let inventory = match world.get::<SurvivalInventory>(player_entity) {
        Some(inv) => inv.clone(),
        None => return,
    };

    let (total_defense, total_toughness) = compute_armor_defense(&inventory, &registry);
    if total_defense == 0 {
        return;
    }

    let mut updated_inventory = inventory.clone();
    let mut changed = false;

    if let Some(dq) = world.resource_mut::<DamageQueue>() {
        for (entity_opt, event) in dq.events.iter_mut() {
            let Some(entity) = entity_opt else { continue };
            if *entity != player_entity {
                continue;
            }
            // Vanilla armor protects against physical damage, not fire,
            // lava, drowning, starvation or the void.
            if !event.source.is_armor_reducible() {
                continue;
            }

            let original = event.amount;
            event.amount = apply_armor_reduction(original, total_defense, total_toughness);

            // Each worn piece loses max(1, damage / 4) durability per hit.
            let wear = ((original / 4.0).floor() as u16).max(1);
            for (slot, stack) in updated_inventory.armor.iter_mut().enumerate() {
                if stack.is_empty() {
                    continue;
                }
                let name = registry.get(stack.id()).name.as_ref().to_string();
                if armor_slot(&name) != Some(slot) {
                    continue;
                }
                let Some(max) = armor_material(&name).map(|m| m.durability[slot]) else {
                    continue;
                };
                stack.damage = stack.damage.saturating_add(wear);
                if stack.damage >= max {
                    stack.clear();
                }
                changed = true;
            }
        }
    }

    if changed {
        world.set(player_entity, updated_inventory);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_skin_takes_full_damage() {
        assert_eq!(apply_armor_reduction(6.0, 0, 0), 6.0);
    }

    #[test]
    fn full_iron_absorbs_vanilla_amount() {
        // 15 armor points: reduction = max(3, 15 − 6/2) = 12 → 6 × 13/25.
        let out = apply_armor_reduction(6.0, 15, 0);
        assert!((out - 3.12).abs() < 1e-3, "got {out}");
    }

    #[test]
    fn toughness_helps_against_big_hits() {
        let without = apply_armor_reduction(20.0, 20, 0);
        let with = apply_armor_reduction(20.0, 20, 8);
        assert!(with < without, "toughness should reduce more damage");
    }

    #[test]
    fn material_tables_match_vanilla() {
        let iron = armor_material("iron_chestplate").unwrap();
        assert_eq!(iron.defense, [2, 6, 5, 2]);
        assert_eq!(iron.durability, [165, 240, 225, 195]);
        let diamond = armor_material("diamond_leggings").unwrap();
        assert_eq!(diamond.toughness, 2);
        assert_eq!(armor_material("stone").is_none(), true);
    }
}
