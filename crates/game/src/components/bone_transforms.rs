/// Computed each tick by AnimationSystem. Maps model node index to
/// local transform. Used by the renderer to position model nodes
/// and by the hierarchy system for bone-attached entities.
#[derive(Clone, Debug, Default)]
pub struct BoneTransforms {
    /// Local transforms per node (in model space).
    pub node_transforms: Vec<glam::Mat4>,
    /// Joint matrices for GPU skinning (inverse_bind * global_transform).
    pub skin_matrices: Vec<glam::Mat4>,
}
