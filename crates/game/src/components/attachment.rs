/// Attached to a child entity. Specifies which part of the
/// parent entity this child should follow.
#[derive(Clone, Debug)]
pub enum Attachment {
    /// Named attachment point on the parent's skeleton.
    /// "right_hand", "head", "torso", "left_foot", etc.
    Bone(String),
    /// First-person camera offset (for held items in FP view).
    FirstPerson {
        /// Offset relative to camera.
        offset: glam::Vec3,
        /// Scale factor (0.5 = half size).
        scale: f32,
        /// Breathing/sway magnitude.
        bob_amplitude: f32,
    },
}

impl Default for Attachment {
    fn default() -> Self {
        Attachment::FirstPerson {
            offset: glam::Vec3::new(0.4, -0.3, -0.5),
            scale: 0.5,
            bob_amplitude: 0.02,
        }
    }
}
