//! First-person camera. Lives in `voxel-core` because both the renderer
//! (view-projection) and gameplay (look direction, ray origin) need it.
//! Frustum culling lives in [`crate::culling`].

use glam::{Mat4, Vec3};

/// First-person view camera. Yaw/pitch in radians; position in world space.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Near and far clip plane distances (metres).
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fov_y: std::f32::consts::FRAC_PI_3, // 60°
            aspect: 16.0 / 9.0,
            near: 0.05,
            far: 1024.0,
        }
    }
}

impl Camera {
    /// Unit forward vector derived from yaw/pitch (right-handed, -Z forward).
    #[inline]
    pub fn forward(self) -> Vec3 {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        let (cy, sy) = (self.yaw.cos(), self.yaw.sin());
        Vec3::new(-cp * sy, sp, -cp * cy)
    }

    /// Unit right vector (perpendicular to forward, lying in the XZ plane).
    #[inline]
    pub fn right(self) -> Vec3 {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        Vec3::new(cy, 0.0, -sy).normalize()
    }

    /// World-space view direction with pitch clamped to horizontal (for movement).
    #[inline]
    pub fn forward_flat(self) -> Vec3 {
        let sy = self.yaw.sin();
        let cy = self.yaw.cos();
        Vec3::new(-sy, 0.0, -cy).normalize()
    }

    /// Look-at view matrix (world -> view space).
    pub fn view(self) -> Mat4 {
        // Clamp pitch to avoid gimbal lock (NaN from cross product when
        // forward is parallel to the up vector).
        let pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.001,
            std::f32::consts::FRAC_PI_2 - 0.001,
        );
        let forward = Vec3::new(
            -pitch.cos() * self.yaw.sin(),
            pitch.sin(),
            -pitch.cos() * self.yaw.cos(),
        );
        let target = self.pos + forward;
        Mat4::look_at_rh(self.pos, target, Vec3::Y)
    }

    /// Perspective projection matrix (view -> clip space).
    pub fn projection(self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    /// Combined view-projection matrix.
    pub fn view_projection(self) -> Mat4 {
        self.projection() * self.view()
    }
}
