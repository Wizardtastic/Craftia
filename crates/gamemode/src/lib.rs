//! Game mode definitions and permission queries.
//!
//! This crate defines the `GameMode` enum and provides methods to query
//! what each mode allows (flight, damage, hunger, etc.).

use serde::{Deserialize, Serialize};

/// The game mode determines player capabilities and survival mechanics.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum GameMode {
    Survival,
    Creative,
    Adventure,
    Spectator,
}

/// How health regeneration works in a given game mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HealthRegenMode {
    /// No regeneration at all.
    None,
    /// Natural regen based on food level (vanilla survival).
    Natural,
    /// Regen only when saturation > 0.
    SaturationOnly,
    /// Always regenerate (creative / peaceful).
    Always,
}

/// How items are consumed (durability, stack counts).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ItemConsumptionMode {
    /// Items are never consumed (creative).
    None,
    /// Items have infinite uses but stacks still track count.
    Infinite,
    /// Normal consumption (survival).
    Normal,
}

/// How the inventory UI behaves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InventoryBehavior {
    /// Full creative inventory with tabs and search.
    CreativeTabs,
    /// Survival inventory with armor slots and crafting grid.
    SurvivalSlots,
}

impl GameMode {
    /// Parse a game mode from a string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "survival" | "surv" | "s" => Some(Self::Survival),
            "creative" | "crea" | "c" => Some(Self::Creative),
            "adventure" | "adv" | "a" => Some(Self::Adventure),
            "spectator" | "spec" | "sp" => Some(Self::Spectator),
            _ => None,
        }
    }

    /// Human-readable name.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Survival => "Survival",
            Self::Creative => "Creative",
            Self::Adventure => "Adventure",
            Self::Spectator => "Spectator",
        }
    }

    /// Whether the player can toggle flight in this mode.
    pub fn allows_flight(self) -> bool {
        match self {
            Self::Survival | Self::Adventure => false,
            Self::Creative | Self::Spectator => true,
        }
    }

    /// Whether the player takes fall damage.
    pub fn takes_fall_damage(self) -> bool {
        match self {
            Self::Survival | Self::Adventure => true,
            Self::Creative | Self::Spectator => false,
        }
    }

    /// Whether the hunger system is active.
    pub fn has_hunger(self) -> bool {
        match self {
            Self::Survival => true,
            Self::Creative | Self::Adventure | Self::Spectator => false,
        }
    }

    /// Whether blocks break instantly (single click).
    pub fn instant_break(self) -> bool {
        match self {
            Self::Creative => true,
            Self::Survival | Self::Adventure | Self::Spectator => false,
        }
    }

    /// Whether the player can place/break blocks.
    pub fn allows_block_interaction(self) -> bool {
        match self {
            Self::Spectator => false,
            Self::Survival | Self::Creative | Self::Adventure => true,
        }
    }

    /// Whether the player can interact with entities (attack, use items on).
    pub fn allows_entity_interaction(self) -> bool {
        match self {
            Self::Spectator => false,
            Self::Survival | Self::Creative | Self::Adventure => true,
        }
    }

    /// How health regeneration works in this mode.
    pub fn health_regeneration(self) -> HealthRegenMode {
        match self {
            Self::Creative | Self::Spectator => HealthRegenMode::Always,
            Self::Survival => HealthRegenMode::Natural,
            Self::Adventure => HealthRegenMode::Natural,
        }
    }

    /// How item consumption works in this mode.
    pub fn item_consumption(self) -> ItemConsumptionMode {
        match self {
            Self::Creative => ItemConsumptionMode::None,
            Self::Spectator => ItemConsumptionMode::None,
            Self::Survival | Self::Adventure => ItemConsumptionMode::Normal,
        }
    }

    /// Which inventory UI to show.
    pub fn inventory_behavior(self) -> InventoryBehavior {
        match self {
            Self::Creative => InventoryBehavior::CreativeTabs,
            Self::Survival | Self::Adventure | Self::Spectator => {
                InventoryBehavior::SurvivalSlots
            }
        }
    }

    /// Whether the player drops their inventory on death.
    pub fn drop_inventory_on_death(self) -> bool {
        match self {
            Self::Survival | Self::Adventure => true,
            Self::Creative | Self::Spectator => false,
        }
    }

    /// Whether the player can take any damage.
    pub fn can_take_damage(self) -> bool {
        match self {
            Self::Spectator => false,
            Self::Survival | Self::Creative | Self::Adventure => true,
        }
    }

    /// Whether the player can equip armor.
    pub fn can_have_armor(self) -> bool {
        match self {
            Self::Survival | Self::Adventure => true,
            Self::Creative | Self::Spectator => false,
        }
    }

    /// Whether the player can eat food.
    pub fn can_use_food(self) -> bool {
        match self {
            Self::Survival | Self::Adventure => true,
            Self::Creative | Self::Spectator => false,
        }
    }

    /// Whether the player can sprint.
    pub fn can_sprint(self) -> bool {
        match self {
            Self::Spectator => false,
            Self::Survival | Self::Creative | Self::Adventure => true,
        }
    }

    /// Attack cooldown in ticks (60 ticks = 1 second).
    pub fn attack_cooldown_ticks(self) -> u32 {
        match self {
            Self::Creative => 0,
            Self::Survival | Self::Adventure => 20,
            Self::Spectator => 0,
        }
    }

    /// Mining speed multiplier (1.0 = normal, higher = faster).
    pub fn mining_speed_multiplier(self) -> f32 {
        match self {
            Self::Creative => 1.0,
            Self::Survival | Self::Adventure => 1.0,
            Self::Spectator => 0.0,
        }
    }
}

impl Default for GameMode {
    fn default() -> Self {
        Self::Survival
    }
}

impl std::fmt::Display for GameMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn survival_restrictions() {
        let m = GameMode::Survival;
        assert!(!m.allows_flight());
        assert!(m.takes_fall_damage());
        assert!(m.has_hunger());
        assert!(!m.instant_break());
        assert!(m.allows_block_interaction());
        assert!(m.can_take_damage());
        assert_eq!(m.item_consumption(), ItemConsumptionMode::Normal);
        assert_eq!(m.inventory_behavior(), InventoryBehavior::SurvivalSlots);
        assert!(m.drop_inventory_on_death());
    }

    #[test]
    fn creative_permissions() {
        let m = GameMode::Creative;
        assert!(m.allows_flight());
        assert!(!m.takes_fall_damage());
        assert!(!m.has_hunger());
        assert!(m.instant_break());
        assert_eq!(m.item_consumption(), ItemConsumptionMode::None);
        assert_eq!(m.inventory_behavior(), InventoryBehavior::CreativeTabs);
        assert!(!m.drop_inventory_on_death());
    }

    #[test]
    fn spectator_permissions() {
        let m = GameMode::Spectator;
        assert!(m.allows_flight());
        assert!(!m.takes_fall_damage());
        assert!(!m.allows_block_interaction());
        assert!(!m.allows_entity_interaction());
        assert!(!m.can_take_damage());
    }

    #[test]
    fn parse_gamemode() {
        assert_eq!(GameMode::from_str_loose("survival"), Some(GameMode::Survival));
        assert_eq!(GameMode::from_str_loose("creative"), Some(GameMode::Creative));
        assert_eq!(GameMode::from_str_loose("adventure"), Some(GameMode::Adventure));
        assert_eq!(GameMode::from_str_loose("spectator"), Some(GameMode::Spectator));
        assert_eq!(GameMode::from_str_loose("surv"), Some(GameMode::Survival));
        assert_eq!(GameMode::from_str_loose("c"), Some(GameMode::Creative));
        assert_eq!(GameMode::from_str_loose("unknown"), None);
    }
}
