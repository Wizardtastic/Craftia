/// Attached to an entity with a ModelRef. Drives animation state.
#[derive(Clone, Debug)]
pub struct AnimationPlayer {
    /// Index into the model's animation list.
    pub current_animation: u32,
    /// Local time in seconds.
    pub time: f32,
    /// Whether to loop.
    pub looping: bool,
    /// Speed multiplier.
    pub speed: f32,
    /// Whether the animation is currently playing.
    pub playing: bool,
}

impl Default for AnimationPlayer {
    fn default() -> Self {
        Self {
            current_animation: 0,
            time: 0.0,
            looping: true,
            speed: 1.0,
            playing: false,
        }
    }
}
