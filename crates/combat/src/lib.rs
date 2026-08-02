//! Health, damage, and combat systems.
//!
//! This crate provides the `Health` component, `DamageEvent` / `DamageSource`
//! for tracking how damage occurs, and the core damage processing pipeline.

use serde::{Deserialize, Serialize};
use voxel_ecs::Entity;

// ─── Health Component ────────────────────────────────────────────────────────

/// ECS component attached to any entity that can take damage.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Health {
    /// Current health points. 1 point = half-heart in the HUD.
    pub current: f32,
    /// Maximum health. Default 20.0 (10 full hearts).
    pub max: f32,
    /// Ticks of invulnerability remaining after taking damage.
    pub invulnerability_ticks: u32,
    /// Game time (in seconds) when the entity last took damage.
    pub last_damage_time: f64,
    /// Whether the entity is dead. Set to true when `current <= 0`.
    pub dead: bool,
}

impl Default for Health {
    fn default() -> Self {
        Self {
            current: 20.0,
            max: 20.0,
            invulnerability_ticks: 0,
            last_damage_time: 0.0,
            dead: false,
        }
    }
}

impl Health {
    /// Create a new Health with the given max (full health).
    pub fn new(max: f32) -> Self {
        Self {
            current: max,
            max,
            ..Default::default()
        }
    }

    /// Apply damage, respecting invulnerability. Returns actual damage dealt.
    pub fn apply_damage(&mut self, amount: f32, game_time: f64) -> f32 {
        if self.invulnerability_ticks > 0 || self.dead || amount <= 0.0 {
            return 0.0;
        }
        let actual = amount.min(self.current);
        self.current -= actual;
        self.invulnerability_ticks = 20; // 1 second at 20 tps
        self.last_damage_time = game_time;
        if self.current <= 0.0 {
            self.current = 0.0;
            self.dead = true;
        }
        actual
    }

    /// Heal the entity. Clamps to max.
    pub fn heal(&mut self, amount: f32) {
        if self.dead {
            return;
        }
        self.current = (self.current + amount).min(self.max);
    }

    /// Reset health to full and clear death state (for respawn).
    pub fn reset(&mut self) {
        self.current = self.max;
        self.dead = false;
        self.invulnerability_ticks = 60; // brief invulnerability on respawn
    }

    /// Tick down invulnerability timer. Call once per game tick.
    pub fn tick(&mut self) {
        if self.invulnerability_ticks > 0 {
            self.invulnerability_ticks -= 1;
        }
    }

    /// Whether the entity is currently invulnerable.
    pub fn is_invulnerable(&self) -> bool {
        self.invulnerability_ticks > 0
    }

    /// Health as a fraction (0.0 ..= 1.0).
    pub fn fraction(&self) -> f32 {
        if self.max <= 0.0 {
            0.0
        } else {
            self.current / self.max
        }
    }
}

// ─── Damage Source ───────────────────────────────────────────────────────────

/// Describes what caused damage. Used for death messages and game logic.
#[derive(Clone, Debug)]
pub enum DamageSource {
    /// Fall damage. The f32 is the fall distance in blocks.
    Fall(f32),
    /// Standing in fire. Ticks spent in fire.
    Fire { ticks: u32 },
    /// Swimming in lava. Ticks spent in lava.
    Lava { ticks: u32 },
    /// Ran out of air underwater.
    Drowning,
    /// Head inside a solid block.
    Suffocation,
    /// Touching a cactus block.
    Cactus,
    /// Fell below the world boundary.
    Void,
    /// Starved to death (food = 0).
    Starvation,
    /// Attacked by another entity.
    Entity {
        /// The attacking entity (for death messages).
        attacker_name: String,
    },
    /// Generic / unknown source.
    Generic,
}

impl DamageSource {
    /// Default damage amount for this source (can be overridden).
    pub fn default_amount(&self) -> f32 {
        match self {
            Self::Fall(distance) => {
                // MC formula: damage = distance - 3.0, minimum 0
                (distance - 3.0).max(0.0)
            }
            Self::Fire { .. } => 1.0,   // 1 heart/sec
            Self::Lava { .. } => 4.0,   // 4 hearts/sec
            Self::Drowning => 2.0,      // 2 hearts/sec
            Self::Suffocation => 1.0,   // 1 heart/sec
            Self::Cactus => 0.5,        // half heart per hit
            Self::Void => 1000.0,       // instant kill
            Self::Starvation => 1.0,    // 1 heart per 4 seconds
            Self::Entity { .. } => 1.0, // base, overridden by weapon
            Self::Generic => 1.0,
        }
    }

    /// Death message for this damage source.
    pub fn death_message(&self, player_name: &str) -> String {
        match self {
            Self::Fall(d) if *d >= 18.0 => {
                format!("{player_name} hit the ground too hard")
            }
            Self::Fall(_) => format!("{player_name} fell from a high place"),
            Self::Fire { .. } => format!("{player_name} went up in flames"),
            Self::Lava { .. } => format!("{player_name} tried to swim in lava"),
            Self::Drowning => format!("{player_name} drowned"),
            Self::Suffocation => format!("{player_name} suffocated in a wall"),
            Self::Cactus => format!("{player_name} was pricked to death"),
            Self::Void => format!("{player_name} fell out of the world"),
            Self::Starvation => format!("{player_name} starved to death"),
            Self::Entity { attacker_name } => {
                format!("{player_name} was slain by {attacker_name}")
            }
            Self::Generic => format!("{player_name} died"),
        }
    }

    /// Whether this source ignores invulnerability frames.
    pub fn ignores_invulnerability(&self) -> bool {
        matches!(self, Self::Void | Self::Starvation)
    }
}

// ─── Damage Event ────────────────────────────────────────────────────────────

/// A pending damage event to be processed by the combat system.
#[derive(Clone, Debug)]
pub struct DamageEvent {
    /// The source of the damage.
    pub source: DamageSource,
    /// The damage amount (after armor reduction, etc.).
    pub amount: f32,
}

// ─── Air Supply (for drowning) ───────────────────────────────────────────────

/// ECS component tracking air supply for drowning.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AirSupply {
    /// Maximum air ticks (300 = 15 seconds at 20 tps).
    pub max: f32,
    /// Current air ticks. Drains when head is underwater.
    pub current: f32,
}

impl Default for AirSupply {
    fn default() -> Self {
        Self {
            max: 300.0,
            current: 300.0,
        }
    }
}

impl AirSupply {
    /// Whether the entity is currently drowning (out of air).
    pub fn is_drowning(&self) -> bool {
        self.current <= 0.0
    }

    /// Air as a fraction (0.0 ..= 1.0).
    pub fn fraction(&self) -> f32 {
        if self.max <= 0.0 {
            0.0
        } else {
            self.current / self.max
        }
    }
}

// ─── Death Event ─────────────────────────────────────────────────────────────

/// Event fired when an entity dies.
#[derive(Clone, Debug)]
pub struct DeathEvent {
    /// The damage source that killed the entity.
    pub source: DamageSource,
    /// The death message to display.
    pub message: String,
}

// ─── Pending Damage Queue ────────────────────────────────────────────────────

/// Resource that accumulates damage events during a tick, drained by HealthSystem.
#[derive(Clone, Debug, Default)]
pub struct DamageQueue {
    pub events: Vec<(Option<Entity>, DamageEvent)>,
}

impl DamageQueue {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Queue damage for a specific entity.
    pub fn push(&mut self, entity: Entity, event: DamageEvent) {
        self.events.push((Some(entity), event));
    }

    /// Drain all events, leaving the queue empty.
    pub fn drain(&mut self) -> Vec<(Option<Entity>, DamageEvent)> {
        std::mem::take(&mut self.events)
    }
}

// ─── Health Regen Mode ───────────────────────────────────────────────────────

/// Tracks regeneration delay per entity.
#[derive(Clone, Copy, Debug, Default)]
pub struct RegenState {
    /// Ticks since last damage. Regen only kicks in after 400 ticks (20s).
    pub ticks_since_damage: u32,
}

impl RegenState {
    /// Whether natural regeneration can proceed.
    pub fn can_regen_natural(&self) -> bool {
        self.ticks_since_damage >= 400
    }
}
