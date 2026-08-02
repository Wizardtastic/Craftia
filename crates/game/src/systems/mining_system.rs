//! Progressive mining system: handles block breaking over time.
//!
//! This system:
//! 1. Checks if the player is holding left-click on a block
//! 2. Calculates break time based on hardness and tool
//! 3. Updates mining progress
//! 4. Breaks the block when progress reaches 1.0
//! 5. Handles interruption (releasing mouse, changing target)

use voxel_core::BlockId;
use voxel_ecs::World;
use voxel_world::registry::ToolType;

use crate::components::{MiningProgress, PlayerEntity, PlayerInput, PlayerLookTarget, Transform};
use crate::item_entity::ItemEntity;
use crate::systems::held_item_system::HotbarResource;
use crate::systems::PhysicsWorldRes;

/// Speed multiplier for each tool type when used on the correct block type.
fn tool_speed_for_block(tool: ToolType, required: ToolType) -> f32 {
    if tool == ToolType::None || tool != required {
        return 1.0; // bare hand or wrong tool
    }
    // Correct tool: faster mining. Use 2.0 as a reasonable default speed.
    // A full implementation would look up the tool's speed from an item registry.
    2.0
}

/// Check if the held block is the correct tool type for mining a block.
fn held_is_correct_tool(
    held: BlockId,
    required: ToolType,
    registry: &voxel_world::BlockRegistry,
) -> bool {
    if required == ToolType::None {
        return true; // any tool (or hand) works
    }
    let def = registry.get(held);
    def.required_tool == required
}

/// Calculate break time while enforcing the held tool's minimum tier.
/// Returns the time in seconds to break the block. -1.0 means unbreakable.
pub fn calculate_break_time_with_tier(
    hardness: f32,
    required_tool: ToolType,
    required_tier: u8,
    has_correct_tool: bool,
    held_tool_tier: u8,
    tool_speed: f32,
) -> f32 {
    if hardness < 0.0 {
        return -1.0; // unbreakable
    }
    // A minimum tier applies even to instant-break blocks. Unbreakable blocks
    // are handled above and remain unbreakable regardless of tool metadata.
    if required_tier > held_tool_tier {
        return -1.0;
    }
    if hardness == 0.0 {
        return 0.0; // instant break
    }

    // Base break time: hardness * 1.5 seconds
    let base_time = hardness * 1.5;

    // Apply tool speed multiplier.
    // If the player has the correct tool, divide by tool speed.
    // If not, use bare hand speed (1.0).
    let speed = if has_correct_tool { tool_speed } else { 1.0 };

    // Blocks that require a specific tool can't be harvested without the
    // correct category and minimum tier. Keep the slower wrong-tool path for
    // category mismatches, but reject an insufficient tier entirely: this is
    // the contract exposed by BlockDef::required_tier.
    if required_tool != ToolType::None && !has_correct_tool {
        // Can still mine, but much slower (5x slower).
        base_time * 5.0
    } else {
        base_time / speed
    }
}

/// Evaluate block drops based on the block's drop table.
/// Returns a list of (item_id, count) to spawn.
pub fn evaluate_drops(
    block_def: &voxel_world::registry::BlockDef,
    has_silk_touch: bool,
    fortune_level: u8,
) -> Vec<(BlockId, u16)> {
    let mut drops = Vec::new();

    for drop in &block_def.drops {
        // Check condition and determine item + count range. Fortune changes
        // the range, but must not bypass the entry's probability roll.
        let (item, min_count, max_count) = match drop.condition {
            voxel_world::registry::DropCondition::Always => {
                (drop.item, drop.min_count, drop.max_count)
            }
            voxel_world::registry::DropCondition::SilkTouchRequired => {
                if !has_silk_touch {
                    continue;
                }
                (drop.item, drop.min_count, drop.max_count)
            }
            voxel_world::registry::DropCondition::SilkTouchForbidden => {
                if has_silk_touch {
                    continue;
                }
                (drop.item, drop.min_count, drop.max_count)
            }
            voxel_world::registry::DropCondition::FortuneScaled { base, max_extra } => {
                let max_for_level = base.saturating_add(max_extra.min(fortune_level as u16));
                (drop.item, base, max_for_level)
            }
            voxel_world::registry::DropCondition::SelfDrop => {
                (block_def.id, drop.min_count, drop.max_count)
            }
        };

        // Roll probability after all conditions, including FortuneScaled.
        if drop.probability < 1.0 && rand::random::<f32>() >= drop.probability {
            continue;
        }

        // Roll count.
        let count = if min_count == max_count {
            min_count
        } else {
            let range = max_count - min_count + 1;
            min_count + (rand::random::<u16>() % range)
        };

        if count > 0 {
            drops.push((item, count));
        }
    }

    // An empty table is an explicit no-drop policy. Blocks that should drop
    // themselves declare that through an explicit BlockDrop entry in the
    // registry; do not infer it from `breakable`.
    drops
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use voxel_world::registry::{BlockDef, BlockDrop, BlockKind, BlockTextures, ToolType};

    fn test_block(drops: Vec<BlockDrop>) -> BlockDef {
        BlockDef {
            id: BlockId(7),
            name: Arc::from("test"),
            kind: BlockKind::Solid,
            solid: true,
            opaque: true,
            breakable: true,
            replaceable: false,
            textures: BlockTextures::same(1),
            emission: 0,
            emission_color: [255, 248, 240],
            light_absorption: 15,
            map_color: [255, 255, 255, 255],
            hardness: 1.0,
            blast_resistance: 1.0,
            required_tool: ToolType::None,
            required_tier: 0,
            drops,
            animation: None,
            material: Default::default(),
        }
    }

    #[test]
    fn empty_drop_table_means_no_drop() {
        assert!(evaluate_drops(&test_block(Vec::new()), false, 0).is_empty());
    }

    #[test]
    fn fortune_level_scales_fortune_drop() {
        let item = BlockId(9);
        let block = test_block(vec![BlockDrop::fortune(item, 2, 3)]);
        assert_eq!(evaluate_drops(&block, false, 0), vec![(item, 2)]);
        let drops = evaluate_drops(&block, false, 3);
        assert_eq!(drops.len(), 1);
        assert!((2..=5).contains(&drops[0].1));
    }

    #[test]
    fn self_drop_condition_uses_block_id() {
        let block = test_block(vec![BlockDrop::self_drop()]);
        assert_eq!(evaluate_drops(&block, false, 0), vec![(BlockId(7), 1)]);
    }

    #[test]
    fn builtin_solid_blocks_drop_themselves() {
        let reg = voxel_world::BlockRegistry::with_builtins();
        for name in ["stone", "dirt", "sand", "gravel", "planks", "coal_ore"] {
            let id = reg.id_of(name).unwrap();
            let drops = evaluate_drops(reg.get(id), false, 0);
            assert_eq!(drops, vec![(id, 1)], "{name} should drop itself once");
        }
    }

    #[test]
    fn builtin_no_drop_blocks_drop_nothing() {
        let reg = voxel_world::BlockRegistry::with_builtins();
        for name in ["glass", "leaves", "tall_grass"] {
            let id = reg.id_of(name).unwrap();
            assert!(
                evaluate_drops(reg.get(id), false, 0).is_empty(),
                "{name} should drop nothing"
            );
        }
    }

    #[test]
    fn fortune_drop_still_honors_probability() {
        let block = test_block(vec![BlockDrop {
            item: BlockId(9),
            min_count: 2,
            max_count: 5,
            probability: 0.0,
            condition: voxel_world::registry::DropCondition::FortuneScaled {
                base: 2,
                max_extra: 3,
            },
        }]);
        assert!(evaluate_drops(&block, false, 3).is_empty());
    }

    #[test]
    fn insufficient_tool_tier_is_unharvestable() {
        assert_eq!(
            calculate_break_time_with_tier(1.0, ToolType::Pickaxe, 2, true, 1, 2.0),
            -1.0
        );
        assert!(calculate_break_time_with_tier(1.0, ToolType::Pickaxe, 2, true, 2, 2.0) > 0.0);
    }
}

/// Progressive mining system entry point. Called each fixed timestep.
pub fn progressive_mining_system(world: &mut World, dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    // Get player input to check if mining (left click).
    let input = match world.get::<PlayerInput>(player_entity) {
        Some(i) => *i,
        None => return,
    };

    // Get or create mining progress component.
    let mut mining = world
        .get::<MiningProgress>(player_entity)
        .copied()
        .unwrap_or_default();

    // If not mining (no left click), reset progress.
    if !input.mining {
        if mining.is_mining() {
            mining.reset();
            world.set(player_entity, mining);
        }
        return;
    }

    // Get the block the player is looking at from the ECS resource (updated by the engine).
    let look_target = world
        .resource::<PlayerLookTarget>()
        .copied()
        .unwrap_or_default();

    // If no target block, reset mining.
    let Some(target_pos) = look_target.block else {
        if mining.is_mining() {
            mining.reset();
            world.set(player_entity, mining);
        }
        return;
    };

    // If target changed, reset progress.
    if mining.target_changed(target_pos) {
        mining = MiningProgress::start(target_pos, look_target.block_id);
    }

    // Get the physics world for block queries.
    let physics = world.resource::<PhysicsWorldRes>().cloned();
    let Some(phys) = physics else { return };

    // Check if the block still exists.
    let block_id = phys
        .0
        .get_block(target_pos[0], target_pos[1], target_pos[2]);
    if block_id.is_air() {
        mining.reset();
        world.set(player_entity, mining);
        return;
    }

    // Get block properties.
    let registry = phys.0.registry();
    let block_def = registry.get(block_id);

    // Get the player's held item to check tool type and speed.
    let held = world
        .resource::<HotbarResource>()
        .map(|h| h.selected_block)
        .unwrap_or(BlockId::AIR);
    let has_correct_tool = held_is_correct_tool(held, block_def.required_tool, &registry);
    let held_tool_tier = world
        .resource::<HotbarResource>()
        .map(|h| h.selected_tool_tier)
        .unwrap_or(0);
    let tool_speed =
        tool_speed_for_block(registry.get(held).required_tool, block_def.required_tool);

    // Calculate break time.
    let break_time = calculate_break_time_with_tier(
        block_def.hardness,
        block_def.required_tool,
        block_def.required_tier,
        has_correct_tool,
        held_tool_tier,
        tool_speed,
    );

    if break_time < 0.0 {
        // Unbreakable block.
        mining.reset();
        world.set(player_entity, mining);
        return;
    }

    // Update progress.
    let should_break = mining.update(dt, break_time);
    world.set(player_entity, mining);

    if should_break {
        // Break the block!
        log::info!("Block broken at {:?}", target_pos);

        // Evaluate drops. Enchantment state is not modeled yet, so the
        // current player path supplies the baseline values; the pure drop
        // evaluator already honors these arguments for future item support.
        let drops = evaluate_drops(block_def, false, 0);

        // Get player position for spawning items.
        let _player_pos = world
            .get::<Transform>(player_entity)
            .map(|t| t.pos)
            .unwrap_or(glam::Vec3::ZERO);

        // Spawn item entities for each drop.
        for (item_id, count) in drops {
            let item_entity = ItemEntity::with_velocity(
                item_id,
                count,
                glam::Vec3::new(
                    (rand::random::<f32>() - 0.5) * 2.0,
                    2.0,
                    (rand::random::<f32>() - 0.5) * 2.0,
                ),
            );
            let item_transform = Transform {
                pos: glam::Vec3::new(
                    target_pos[0] as f32 + 0.5,
                    target_pos[1] as f32 + 0.5,
                    target_pos[2] as f32 + 0.5,
                ),
                rot: glam::Quat::IDENTITY,
            };
            world.spawn((item_entity, item_transform));
        }

        // Set the block to air.
        phys.0
            .set_block(target_pos[0], target_pos[1], target_pos[2], BlockId::AIR);

        // Reset mining progress.
        mining.reset();
        world.set(player_entity, mining);
    }
}
