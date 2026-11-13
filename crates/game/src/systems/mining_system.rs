//! Progressive mining system: handles block breaking over time.
//!
//! Break time follows the vanilla formula:
//!
//! ```text
//! damage per tick = tool_speed / hardness / (can_harvest ? 30 : 100)
//! ```
//!
//! which is `hardness × 1.5 / tool_speed` seconds with the right tool and
//! `hardness × 5` seconds by hand. Tool speeds match vanilla materials
//! (wood 2, stone 4, iron 6, diamond 8, netherite 9, gold 12).
//!
//! Blocks with hardness 0 break instantly; hardness < 0 (bedrock) never does.
//! Progress resets when the target changes or the mouse is released.

use voxel_core::BlockId;
use voxel_ecs::World;
use voxel_gamemode::GameMode;
use voxel_hunger::exhaustion::MINE_BLOCK;
use voxel_hunger::Hunger;
use voxel_world::registry::ToolType;

use crate::components::{MiningProgress, PlayerEntity, PlayerInput, PlayerLookTarget, Transform};
use crate::inventory::SurvivalInventory;
use crate::item_entity::ItemEntity;
use crate::systems::PhysicsWorldRes;

/// Resource: blocks broken by the mining system this frame. The engine drains
/// it to remesh the affected chunks and spawn break particles.
#[derive(Clone, Debug, Default)]
pub struct MinedBlocks(pub Vec<([i32; 3], BlockId)>);

/// Vanilla tool speed for a material prefix.
fn tool_material_speed(name: &str) -> (f32, u8) {
    if name.starts_with("wooden") || name.starts_with("wood_") {
        (2.0, 1)
    } else if name.starts_with("stone") {
        (4.0, 2)
    } else if name.starts_with("iron") {
        (6.0, 3)
    } else if name.starts_with("diamond") {
        (8.0, 4)
    } else if name.starts_with("netherite") {
        (9.0, 5)
    } else if name.starts_with("golden") || name.starts_with("gold_") {
        (12.0, 1)
    } else {
        (1.0, 0)
    }
}

/// Infer the tool category from an item name (`iron_pickaxe` → pickaxe).
fn tool_type_from_name(name: &str) -> ToolType {
    let n = name.to_lowercase();
    if n.ends_with("_pickaxe") {
        ToolType::Pickaxe
    } else if n.ends_with("_axe") {
        ToolType::Axe
    } else if n.ends_with("_shovel") {
        ToolType::Shovel
    } else if n.ends_with("_sword") {
        ToolType::Sword
    } else if n.ends_with("_hoe") {
        ToolType::Hoe
    } else {
        ToolType::None
    }
}

/// Speed multiplier and tier of the held item, if it is a tool.
fn held_tool_speed(
    held: BlockId,
    registry: &voxel_world::BlockRegistry,
) -> (f32, u8, ToolType) {
    if held.is_air() {
        return (1.0, 0, ToolType::None);
    }
    let name = registry.get(held).name.as_ref().to_string();
    let kind = tool_type_from_name(&name);
    let (speed, tier) = tool_material_speed(&name);
    (speed, tier, kind)
}

/// The tool that speeds this block up. Blocks without a hard requirement
/// still have a preferred tool (dirt → shovel, logs → axe, ...).
fn preferred_tool_for_block(name: &str) -> ToolType {
    let n = name.to_lowercase();
    if n.contains("ore")
        || n.contains("stone")
        || n.contains("cobble")
        || n.contains("bedrock")
    {
        ToolType::Pickaxe
    } else if n.contains("dirt")
        || n.contains("sand")
        || n.contains("gravel")
        || n.contains("snow")
    {
        ToolType::Shovel
    } else if n.contains("log")
        || n.contains("wood")
        || n.contains("plank")
        || n.contains("chest")
    {
        ToolType::Axe
    } else if n.contains("leaves") {
        ToolType::Hoe
    } else {
        ToolType::None
    }
}

/// Vanilla break time in seconds.
///
/// `has_correct_tool` selects the 30× divisor (can harvest) instead of the
/// 100× wrong-tool divisor. An insufficient tool tier is reported as
/// unbreakable; builtin blocks all use tier 0.
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
    if required_tier > held_tool_tier {
        return -1.0;
    }
    if hardness == 0.0 {
        return 0.0; // instant break
    }

    let speed = if has_correct_tool {
        tool_speed.max(1.0)
    } else {
        1.0
    };
    let divisor = if required_tool == ToolType::None || has_correct_tool {
        30.0
    } else {
        100.0
    };

    hardness * divisor / (20.0 * speed)
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

    drops
}

/// Progressive mining system entry point. Called each fixed timestep.
pub fn progressive_mining_system(world: &mut World, dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    // Only survival/adventure mine over time; creative breaks instantly
    // through the block-interaction path.
    let game_mode = world
        .get::<GameMode>(player_entity)
        .copied()
        .unwrap_or(GameMode::Survival);
    // Adventure players cannot break blocks in vanilla.
    if game_mode != GameMode::Survival {
        return;
    }

    let input = match world.get::<PlayerInput>(player_entity) {
        Some(i) => *i,
        None => return,
    };

    let mut mining = world
        .get::<MiningProgress>(player_entity)
        .copied()
        .unwrap_or_default();

    // Keep the arm swinging while the button is held.
    if input.mining {
        if let Some(mut state) = world
            .get::<crate::components::PlayerState>(player_entity)
            .copied()
        {
            if state.mining_swing <= 0.0 {
                state.mining_swing = 1.0;
                world.set(player_entity, state);
            }
        }
    }

    if !input.mining {
        if mining.is_mining() {
            mining.reset();
            world.set(player_entity, mining);
        }
        return;
    }

    let look_target = world
        .resource::<PlayerLookTarget>()
        .copied()
        .unwrap_or_default();

    let Some(target_pos) = look_target.block else {
        if mining.is_mining() {
            mining.reset();
            world.set(player_entity, mining);
        }
        return;
    };

    if mining.target_changed(target_pos) {
        mining = MiningProgress::start(target_pos, look_target.block_id);
    }

    let physics = world.resource::<PhysicsWorldRes>().cloned();
    let Some(phys) = physics else { return };

    let block_id = phys
        .0
        .get_block(target_pos[0], target_pos[1], target_pos[2]);
    if block_id.is_air() {
        mining.reset();
        world.set(player_entity, mining);
        return;
    }

    let registry = phys.0.registry();
    let block_def = registry.get(block_id);

    let (held, held_tool_tier) = world
        .get::<SurvivalInventory>(player_entity)
        .map(|inv| (inv.selected_block().unwrap_or(BlockId::AIR), inv.selected_tool_tier()))
        .unwrap_or((BlockId::AIR, 0));
    let (tool_speed, inferred_tier, tool_kind) = held_tool_speed(held, &registry);
    let preferred = if block_def.required_tool != ToolType::None {
        block_def.required_tool
    } else {
        preferred_tool_for_block(block_def.name.as_ref())
    };
    // A matching tool speeds the block up even when the block can be
    // harvested by hand (vanilla preferred-tool behaviour).
    let tool_speed = if preferred != ToolType::None && tool_kind == preferred {
        tool_speed
    } else {
        1.0
    };
    // Per-slot tier metadata wins when present; otherwise the tool name
    // determines its tier.
    let held_tool_tier = held_tool_tier.max(inferred_tier);
    let has_correct_tool = block_def.required_tool == ToolType::None
        || held_tool_speed(held, &registry).2 == block_def.required_tool;

    let break_time = calculate_break_time_with_tier(
        block_def.hardness,
        block_def.required_tool,
        block_def.required_tier,
        has_correct_tool,
        held_tool_tier,
        tool_speed,
    );

    if break_time < 0.0 {
        // Unbreakable block (or insufficient tool tier).
        mining.reset();
        world.set(player_entity, mining);
        return;
    }

    let should_break = mining.update(dt, break_time);
    world.set(player_entity, mining);

    if should_break {
        log::debug!("Block broken at {:?}", target_pos);

        let drops = evaluate_drops(block_def, false, 0);

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

        phys.0
            .set_block(target_pos[0], target_pos[1], target_pos[2], BlockId::AIR);

        // Tell the engine so it can rebuild the chunk mesh and spawn particles.
        if let Some(mined) = world.resource_mut::<MinedBlocks>() {
            mined.0.push((target_pos, block_id));
        }

        // Vanilla charges 0.005 exhaustion per block broken.
        if let Some(mut hunger) = world.get::<Hunger>(player_entity).copied() {
            hunger.add_exhaustion(MINE_BLOCK);
            world.set(player_entity, hunger);
        }

        mining.reset();
        world.set(player_entity, mining);
    }
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
        for name in ["dirt", "sand", "gravel", "planks", "coal_ore"] {
            let id = reg.id_of(name).unwrap();
            let drops = evaluate_drops(reg.get(id), false, 0);
            assert_eq!(drops, vec![(id, 1)], "{name} should drop itself once");
        }
    }

    #[test]
    fn stone_drops_cobblestone_like_vanilla() {
        let reg = voxel_world::BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        let cobble = reg.id_of("cobblestone").unwrap();
        assert_eq!(evaluate_drops(reg.get(stone), false, 0), vec![(cobble, 1)]);
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

    #[test]
    fn break_times_match_vanilla() {
        // Stone (1.5) by hand: 7.5 s. With a wooden pickaxe (speed 2): 1.125 s.
        assert!(
            (calculate_break_time_with_tier(1.5, ToolType::Pickaxe, 0, false, 0, 1.0) - 7.5).abs()
                < 1e-4
        );
        assert!(
            (calculate_break_time_with_tier(1.5, ToolType::Pickaxe, 0, true, 1, 2.0) - 1.125)
                .abs()
                < 1e-4
        );
        // Dirt (0.5) by hand: 0.75 s.
        assert!(
            (calculate_break_time_with_tier(0.5, ToolType::None, 0, true, 0, 1.0) - 0.75).abs()
                < 1e-4
        );
        // Instant-break blocks.
        assert_eq!(
            calculate_break_time_with_tier(0.0, ToolType::None, 0, true, 0, 1.0),
            0.0
        );
        // Bedrock.
        assert_eq!(
            calculate_break_time_with_tier(-1.0, ToolType::None, 0, true, 0, 1.0),
            -1.0
        );
    }
}
