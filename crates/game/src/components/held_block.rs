/// Attached to the player entity. Represents the block currently
/// held in the player's hand, visible in first person.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeldBlock {
    /// Atlas tile index (0 = air = invisible).
    pub tile: u32,
    /// Whether to render in first-person view.
    pub in_first_person: bool,
}
