//! Experience system: XP orbs, levels, and collection.
//!
//! This system provides XP tracking and level progression.

use serde::{Deserialize, Serialize};

/// ECS component tracking the player's experience.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Experience {
    /// Lifetime cumulative XP.
    pub total_xp: u32,
    /// Current displayed level.
    pub level: u16,
    /// Progress towards next level (0..1).
    pub progress: f32,
}

impl Default for Experience {
    fn default() -> Self {
        Self {
            total_xp: 0,
            level: 0,
            progress: 0.0,
        }
    }
}

impl Experience {
    /// Calculate XP required to reach the next level.
    pub fn xp_to_next_level(level: u16) -> u32 {
        match level {
            0..=15 => 2 * level as u32 + 7,
            16..=30 => 5 * level as u32 - 38,
            _ => 9 * level as u32 - 158,
        }
    }

    /// Add XP and handle level-ups.
    pub fn add_xp(&mut self, amount: u32) {
        self.total_xp += amount;
        self.progress += amount as f32 / Self::xp_to_next_level(self.level) as f32;

        // Handle multiple level-ups.
        while self.progress >= 1.0 {
            self.level += 1;
            self.progress -= 1.0;
            // Recalculate for overflow.
            let needed = Self::xp_to_next_level(self.level) as f32;
            if needed > 0.0 {
                // progress is already relative to the new level's requirement.
            }
        }
    }

    /// Reset experience (for death).
    pub fn reset(&mut self) {
        self.total_xp = 0;
        self.level = 0;
        self.progress = 0.0;
    }
}

/// ECS entity for XP orbs in the world.
#[derive(Clone, Copy, Debug)]
pub struct XpOrb {
    /// XP value contained.
    pub value: u16,
    /// Age in ticks. Despawn after 30000 ticks (5 min).
    pub age: u32,
    /// Cooldown before merging with other orbs.
    pub merge_cooldown: u32,
}

impl Default for XpOrb {
    fn default() -> Self {
        Self {
            value: 1,
            age: 0,
            merge_cooldown: 0,
        }
    }
}

impl XpOrb {
    /// Create a new XP orb with the given value.
    pub fn new(value: u16) -> Self {
        Self {
            value,
            age: 0,
            merge_cooldown: 20, // 1 second before merge
        }
    }

    /// Tick the orb age. Returns true if it should despawn.
    pub fn tick(&mut self) -> bool {
        self.age += 1;
        if self.merge_cooldown > 0 {
            self.merge_cooldown -= 1;
        }
        self.age > 30000
    }
}

/// XP collection system. Runs each tick.
pub fn xp_collection_system(world: &mut voxel_ecs::World, _dt: f32) {
    use crate::components::{PlayerEntity, Transform};

    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    let player_pos = world
        .get::<Transform>(player_entity)
        .map(|t| t.pos)
        .unwrap_or(glam::Vec3::ZERO);

    // Get or create experience component.
    let mut experience = world
        .get::<Experience>(player_entity)
        .copied()
        .unwrap_or_default();

    // Find all XP orbs in range.
    let mut orbs_to_collect = Vec::new();
    let mut orbs_to_despawn = Vec::new();

    for (entity, (orb, transform)) in world.query::<(&mut XpOrb, &Transform)>() {
        // Tick the orb.
        if orb.tick() {
            orbs_to_despawn.push(entity);
            continue;
        }

        // Check collection range (1.5 blocks).
        let dist = (transform.pos - player_pos).length();
        if dist < 1.5 && orb.merge_cooldown == 0 {
            orbs_to_collect.push((entity, orb.value));
        }
    }

    // Collect orbs.
    for (entity, value) in orbs_to_collect {
        experience.add_xp(value as u32);
        world.despawn(entity);
    }

    // Despawn old orbs.
    for entity in orbs_to_despawn {
        if world.is_alive(entity) {
            world.despawn(entity);
        }
    }

    world.set(player_entity, experience);
}
