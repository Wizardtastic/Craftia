/// Visual mesh for an entity. Simple billboard quad for now;
/// could become a cube or model reference later.
#[derive(Clone, Copy, Debug)]
pub struct Mesh {
    /// Which tile index in the texture atlas this entity uses.
    pub tile: u32,
    /// Billboard (always faces camera) vs fixed-orientation quad.
    pub billboard: bool,
    /// Half-size of the quad in world units.
    pub half_size: f32,
    /// Whether this entity uses alpha blending (rendered back-to-front).
    pub transparent: bool,
}

impl Default for Mesh {
    fn default() -> Self {
        Self {
            tile: 0,
            billboard: true,
            half_size: 0.25,
            transparent: false,
        }
    }
}
