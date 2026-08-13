//! Hunger, saturation, and exhaustion systems.
//!
//! This crate provides the `Hunger` component and related types for
//! tracking food levels, saturation, and exhaustion in survival mode.

use serde::{Deserialize, Serialize};

/// ECS component tracking the player's hunger state.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Hunger {
    /// Food level (0..20). Displayed as drumsticks in the HUD.
    pub food: f32,
    /// Saturation (0..food). Hidden buffer that decays before food.
    pub saturation: f32,
    /// Exhaustion accumulator (0..4). Resets on reaching 4.
    pub exhaustion: f32,
}

impl Default for Hunger {
    fn default() -> Self {
        Self {
            food: 20.0,
            saturation: 5.0,
            exhaustion: 0.0,
        }
    }
}

impl Hunger {
    /// Create a new Hunger with the given food level and saturation.
    pub fn new(food: f32, saturation: f32) -> Self {
        Self {
            food: food.clamp(0.0, 20.0),
            saturation: saturation.clamp(0.0, food),
            exhaustion: 0.0,
        }
    }

    /// Add exhaustion from an action. When exhaustion reaches 4.0,
    /// it triggers saturation/food decay.
    pub fn add_exhaustion(&mut self, amount: f32) {
        self.exhaustion = (self.exhaustion + amount).min(4.0);
    }

    /// Tick the hunger system. Returns true if starvation damage should be dealt.
    pub fn tick(&mut self) -> bool {
        if self.exhaustion >= 4.0 {
            self.exhaustion -= 4.0;
            if self.saturation > 0.0 {
                self.saturation = (self.saturation - 1.0).max(0.0);
            } else if self.food > 0.0 {
                self.food = (self.food - 1.0).max(0.0);
            }
        }

        // Starvation: food == 0
        self.food <= 0.0
    }

    /// Restore food and saturation from eating.
    pub fn eat(&mut self, hunger_restore: f32, saturation_modifier: f32) {
        self.food = (self.food + hunger_restore).min(20.0);
        // Saturation can't exceed food level
        let sat_restore = hunger_restore * saturation_modifier * 2.0;
        self.saturation = (self.saturation + sat_restore).min(self.food);
    }

    /// Whether the player can sprint (food > 6).
    pub fn can_sprint(&self) -> bool {
        self.food > 6.0
    }

    /// Food level as a fraction (0.0 ..= 1.0).
    pub fn fraction(&self) -> f32 {
        self.food / 20.0
    }

    /// Reset hunger to full (for respawn).
    pub fn reset(&mut self) {
        self.food = 20.0;
        self.saturation = 5.0;
        self.exhaustion = 0.0;
    }
}

/// Exhaustion costs for various actions.
pub mod exhaustion {
    /// Per meter traveled while sprinting.
    pub const SPRINT_PER_METER: f32 = 0.01;
    /// Per jump.
    pub const JUMP: f32 = 0.05;
    /// Per sprint-jump.
    pub const SPRINT_JUMP: f32 = 0.2;
    /// Per block mined.
    pub const MINE_BLOCK: f32 = 0.005;
    /// Per half-heart of damage taken.
    pub const DAMAGE_TAKEN: f32 = 0.1;
    /// Per meter swum.
    pub const SWIM_PER_METER: f32 = 0.01;
}

/// Food properties for edible items.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FoodProperties {
    /// Food points restored (e.g., 4 for apple).
    pub hunger_restoration: f32,
    /// Saturation modifier (0..2). Multiplied with hunger_restoration.
    pub saturation_modifier: f32,
    /// Eat duration in ticks (default 32 = 1.6s).
    pub eat_duration_ticks: u32,
    /// Whether this food can be eaten when hunger is full.
    pub always_edible: bool,
}

impl FoodProperties {
    pub const fn new(hunger: f32, sat_mod: f32, duration: u32, always: bool) -> Self {
        Self {
            hunger_restoration: hunger,
            saturation_modifier: sat_mod,
            eat_duration_ticks: duration,
            always_edible: always,
        }
    }
}

/// ECS component tracking the player's eating state.
#[derive(Clone, Copy, Debug, Default)]
pub struct EatingState {
    /// Ticks remaining until eating completes.
    pub ticks_remaining: u32,
    /// The food item being eaten (block/item ID).
    pub food_item_id: u16,
    /// Whether the player is currently eating.
    pub active: bool,
}

impl EatingState {
    /// Start eating a food item.
    pub fn start(&mut self, item_id: u16, duration: u32) {
        self.ticks_remaining = duration;
        self.food_item_id = item_id;
        self.active = true;
    }

    /// Cancel eating.
    pub fn cancel(&mut self) {
        self.ticks_remaining = 0;
        self.food_item_id = 0;
        self.active = false;
    }

    /// Tick the eating state. Returns true if eating completes this tick.
    pub fn tick(&mut self) -> bool {
        if !self.active {
            return false;
        }
        if self.ticks_remaining > 0 {
            self.ticks_remaining -= 1;
            if self.ticks_remaining == 0 {
                self.active = false;
                return true;
            }
        }
        false
    }
}

/// Difficulty level affecting hunger and damage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum Difficulty {
    Peaceful,
    Easy,
    #[default]
    Normal,
    Hard,
}


impl Difficulty {
    /// Parse difficulty from string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "peaceful" => Some(Self::Peaceful),
            "easy" => Some(Self::Easy),
            "normal" => Some(Self::Normal),
            "hard" => Some(Self::Hard),
            _ => None,
        }
    }

    /// Display name.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Peaceful => "Peaceful",
            Self::Easy => "Easy",
            Self::Normal => "Normal",
            Self::Hard => "Hard",
        }
    }

    /// Whether hunger depletes in this difficulty.
    pub fn has_hunger_depletion(self) -> bool {
        match self {
            Self::Peaceful => false,
            Self::Easy | Self::Normal | Self::Hard => true,
        }
    }

    /// Whether starvation damage applies.
    pub fn has_starvation(self) -> bool {
        match self {
            Self::Peaceful => false,
            Self::Easy | Self::Normal | Self::Hard => true,
        }
    }

    /// Drowning damage per tick (hearts per second / 20).
    pub fn drowning_damage(self) -> f32 {
        match self {
            Self::Peaceful => 0.0,
            Self::Easy => 1.0,
            Self::Normal => 2.0,
            Self::Hard => 4.0,
        }
    }

    /// Natural regen interval in ticks (lower = faster).
    pub fn regen_interval(self) -> Option<u32> {
        match self {
            Self::Peaceful => Some(20),
            Self::Easy => Some(200),
            Self::Normal => Some(80),
            Self::Hard => None, // No natural regen
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hunger_default() {
        let h = Hunger::default();
        assert_eq!(h.food, 20.0);
        assert_eq!(h.saturation, 5.0);
        assert_eq!(h.exhaustion, 0.0);
    }

    #[test]
    fn exhaustion_triggers_decay() {
        let mut h = Hunger {
            exhaustion: 3.5,
            ..Default::default()
        };
        h.add_exhaustion(0.6);
        let starved = h.tick();
        assert!(!starved);
        assert_eq!(h.food, 20.0);
        assert_eq!(h.saturation, 4.0);
    }

    #[test]
    fn food_decays_when_saturation_empty() {
        let mut h = Hunger::new(10.0, 0.0);
        h.exhaustion = 4.0;
        let starved = h.tick();
        assert!(!starved);
        assert_eq!(h.food, 9.0);
    }

    #[test]
    fn starvation_at_zero_food() {
        let mut h = Hunger::new(0.0, 0.0);
        let starved = h.tick();
        assert!(starved);
    }

    #[test]
    fn eat_restores_food() {
        let mut h = Hunger::new(10.0, 0.0);
        h.eat(4.0, 0.3);
        assert_eq!(h.food, 14.0);
        assert!(h.saturation > 0.0);
    }

    #[test]
    fn can_sprint_threshold() {
        let mut h = Hunger::default();
        assert!(h.can_sprint());
        h.food = 6.0;
        assert!(!h.can_sprint());
    }
}
