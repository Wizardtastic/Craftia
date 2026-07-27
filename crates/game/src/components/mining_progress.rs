//! Mining progress tracking for progressive block breaking.
//!
//! This component tracks the player's current mining state: which block
//! they're breaking, how much progress they've made, and when to break it.

use serde::{Deserialize, Serialize};
use voxel_core::BlockId;

/// ECS resource storing the block the player is currently looking at.
/// Updated each frame by the engine's raycast.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerLookTarget {
    /// The block position under the crosshair (None if looking at nothing).
    pub block: Option<[i32; 3]>,
    /// The block ID at that position.
    pub block_id: BlockId,
}

/// ECS component tracking the player's current mining progress.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct MiningProgress {
    /// The block position being mined (None if not mining).
    pub target_block: Option<[i32; 3]>,
    /// The block type being mined (as raw u16).
    target_block_id_raw: u16,
    /// Mining progress from 0.0 to 1.0.
    pub progress: f32,
    /// The crack overlay stage to display (0-10, 0 = no cracks).
    pub crack_stage: u8,
}

impl MiningProgress {
    /// Create a new mining state for a block.
    pub fn start(block_pos: [i32; 3], block_id: BlockId) -> Self {
        Self {
            target_block: Some(block_pos),
            target_block_id_raw: block_id.raw(),
            progress: 0.0,
            crack_stage: 0,
        }
    }

    /// Reset mining progress (when interrupted or completed).
    pub fn reset(&mut self) {
        self.target_block = None;
        self.target_block_id_raw = 0;
        self.progress = 0.0;
        self.crack_stage = 0;
    }

    /// Get the block ID being mined.
    pub fn target_block_id(&self) -> BlockId {
        BlockId::new(self.target_block_id_raw)
    }

    /// Whether the player is currently mining something.
    pub fn is_mining(&self) -> bool {
        self.target_block.is_some()
    }

    /// Update progress and crack stage. Returns true if the block should break.
    pub fn update(&mut self, dt: f32, break_time: f32) -> bool {
        if !self.is_mining() {
            return false;
        }
        if break_time <= 0.0 {
            return true; // instant break
        }
        self.progress += dt / break_time;
        self.crack_stage = ((self.progress * 10.0) as u8).min(10);
        self.progress >= 1.0
    }

    /// Check if the target has changed (requires reset).
    pub fn target_changed(&self, block_pos: [i32; 3]) -> bool {
        self.target_block.map_or(true, |t| t != block_pos)
    }
}
