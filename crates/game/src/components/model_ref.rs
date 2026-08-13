/// Attached to an entity with a loaded 3D model.
/// References a model in the engine's model registry.
#[derive(Clone, Copy, Debug)]
#[derive(Default)]
pub struct ModelRef {
    /// Index into the engine's model registry.
    pub model_id: u32,
    /// Which node of the model to render.
    pub node_index: usize,
}

