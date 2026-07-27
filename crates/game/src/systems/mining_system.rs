//! Progressive mining system: handles block breaking over time.
//!
//! This system:
//! 1. Checks if the player is holding left-click on a block
//! 2. Calculates break time based on hardness and tool
//! 3. Updates mining progress
//! 4. Breaks the block when progress reaches 1.0
//! 5. Handles interruption (releasing mouse, changing target)

use voxel_ecs::World;
use voxel_core::BlockId;
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
fn held_is_correct_tool(held: BlockId, required: ToolType, registry: &voxel_world::BlockRegistry) -> bool {
    if required == ToolType::None {
        return true; // any tool (or hand) works
    }
    let def = registry.get(held);
    def.required_tool == required
}

/// Calculate the break time for a block given the player's tool.
/// Returns the time in seconds to break the block. -1.0 means unbreakable.
pub fn calculate_break_time(
    hardness: f32,
    required_tool: ToolType,
    _required_tier: u8,
    has_correct_tool: bool,
    tool_speed: f32,
) -> f32 {
    if hardness < 0.0 {
        return -1.0; // unbreakable
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

    // Blocks that require a specific tool can't be mined without it
    // (unless required_tool is None, meaning any tool works).
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
    _has_silk_touch: bool,
    _fortune_level: u8,
) -> Vec<(BlockId, u16)> {
    let mut drops = Vec::new();

    for drop in &block_def.drops {
        // Check condition.
        match drop.condition {
            voxel_world::registry::DropCondition::Always => {}
            voxel_world::registry::DropCondition::SilkTouchRequired => {
                if !_has_silk_touch {
                    continue;
                }
            }
            voxel_world::registry::DropCondition::SilkTouchForbidden => {
                if _has_silk_touch {
                    continue;
                }
            }
            voxel_world::registry::DropCondition::FortuneScaled { base, max_extra } => {
                // For now, use base count. Fortune will be implemented later.
                let _ = (base, max_extra, _fortune_level);
            }
        }

        // Roll probability.
        if drop.probability < 1.0 && rand::random::<f32>() > drop.probability {
            continue;
        }

        // Roll count.
        let count = if drop.min_count == drop.max_count {
            drop.min_count
        } else {
            let range = drop.max_count - drop.min_count + 1;
            drop.min_count + (rand::random::<u16>() % range)
        };

        if count > 0 {
            drops.push((drop.item, count));
        }
    }

    // If no drops specified, drop the block itself.
    if drops.is_empty() && block_def.breakable {
        drops.push((block_def.id, 1));
    }

    drops
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
    let mut mining = world.get::<MiningProgress>(player_entity).copied().unwrap_or_default();

    // If not mining (no left click), reset progress.
    if !input.mining {
        if mining.is_mining() {
            mining.reset();
            world.set(player_entity, mining);
        }
        return;
    }

    // Get the block the player is looking at from the ECS resource (updated by the engine).
    let look_target = world.resource::<PlayerLookTarget>().copied().unwrap_or_default();

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
    let block_id = phys.0.get_block(target_pos[0], target_pos[1], target_pos[2]);
    if block_id.is_air() {
        mining.reset();
        world.set(player_entity, mining);
        return;
    }

    // Get block properties.
    let registry = phys.0.registry();
    let block_def = registry.get(block_id);

    // Get the player's held item to check tool type and speed.
    let held = world.resource::<HotbarResource>()
        .map(|h| h.selected_block)
        .unwrap_or(BlockId::AIR);
    let has_correct_tool = held_is_correct_tool(held, block_def.required_tool, &registry);
    let tool_speed = tool_speed_for_block(
        registry.get(held).required_tool,
        block_def.required_tool,
    );

    // Calculate break time.
    let break_time = calculate_break_time(
        block_def.hardness,
        block_def.required_tool,
        block_def.required_tier,
        has_correct_tool,
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

        // Evaluate drops.
        let drops = evaluate_drops(block_def, false, 0);

        // Get player position for spawning items.
        let _player_pos = world.get::<Transform>(player_entity)
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
        phys.0.set_block(target_pos[0], target_pos[1], target_pos[2], BlockId::AIR);

        // Reset mining progress.
        mining.reset();
        world.set(player_entity, mining);
    }
}
