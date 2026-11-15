//! View-frustum culling primitives shared by the chunk streamer and the
//! renderer.
//!
//! The frustum is extracted once per frame from the camera's view-projection
//! matrix and then queried with cheap batched AABB / bounding-sphere tests.
//! The plane layout and the "outside" sign convention match the GPU cull
//! shader (`shaders/chunk_cull.comp`) exactly, so CPU and GPU visibility
//! decisions agree at the edges of the view.

use glam::{Mat4, Vec3, Vec4};

/// A 6-plane view frustum. Plane `i` is `(a, b, c, d)` in world space with
/// `a*x + b*y + c*z + d >= 0` for points inside; all normals are normalized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extract the six clip-space planes from a view-projection matrix using
    /// the Gribb-Hartmann method. glam matrices are column-major, so row `r`
    /// is `(col0[r], col1[r], col2[r], col3[r])`.
    pub fn from_view_projection(vp: Mat4) -> Self {
        let col0 = vp.col(0);
        let col1 = vp.col(1);
        let col2 = vp.col(2);
        let col3 = vp.col(3);
        let row = |r: usize| Vec4::new(col0[r], col1[r], col2[r], col3[r]);
        let r4 = row(3);
        let r1 = row(0);
        let r2 = row(1);
        let r3 = row(2);

        let mut planes = [
            r4 + r1, // left
            r4 - r1, // right
            r4 + r2, // bottom
            r4 - r2, // top
            r4 + r3, // near
            r4 - r3, // far
        ];
        for p in &mut planes {
            let len = Vec3::new(p.x, p.y, p.z).length();
            if len > 0.0 {
                *p /= len;
            }
        }
        Self { planes }
    }

    /// The normalized planes, in left/right/bottom/top/near/far order.
    #[inline]
    pub fn planes(&self) -> &[Vec4; 6] {
        &self.planes
    }

    /// Conservative AABB test: `true` when any part of the box may be visible.
    /// Uses the box centre / half-extent form of the plane test, which needs
    /// no per-plane corner selection and vectorises well.
    #[inline]
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        let center = (min + max) * 0.5;
        let half = (max - min) * 0.5;
        self.intersects_center_half(center, half)
    }

    /// AABB test with an explicit centre and half-extent. Prefer this when the
    /// caller already keeps its bounds in that form (e.g. the chunk streamer).
    #[inline]
    pub fn intersects_center_half(&self, center: Vec3, half: Vec3) -> bool {
        for p in &self.planes {
            let radius = p.x.abs() * half.x + p.y.abs() * half.y + p.z.abs() * half.z;
            if p.x * center.x + p.y * center.y + p.z * center.z + p.w < -radius {
                return false;
            }
        }
        true
    }

    /// Conservative bounding-sphere test. Cheaper than an AABB test (no
    /// per-axis absolute values) and sufficient for coarse prioritisation.
    #[inline]
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        for p in &self.planes {
            if p.x * center.x + p.y * center.y + p.z * center.z + p.w < -radius {
                return false;
            }
        }
        true
    }

    /// Bitmask of the planes the box is entirely behind. `0` means the box is
    /// potentially visible; any set bit identifies a separating plane. This is
    /// the building block for hierarchical culling, where a group test can be
    /// reused for every child without repeating all six plane evaluations.
    #[inline]
    pub fn outside_mask(&self, center: Vec3, half: Vec3) -> u8 {
        let mut mask = 0u8;
        for (i, p) in self.planes.iter().enumerate() {
            let radius = p.x.abs() * half.x + p.y.abs() * half.y + p.z.abs() * half.z;
            if p.x * center.x + p.y * center.y + p.z * center.z + p.w < -radius {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// `true` when `point` lies inside all six planes.
    #[inline]
    pub fn contains_point(&self, point: Vec3) -> bool {
        self.planes.iter().all(|p| {
            p.x * point.x + p.y * point.y + p.z * point.z + p.w >= 0.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Camera;

    fn camera_looking_forward() -> Camera {
        // Default camera looks down -Z from the origin.
        Camera::default()
    }

    #[test]
    fn box_in_front_is_visible() {
        let f = Frustum::from_view_projection(camera_looking_forward().view_projection());
        assert!(f.intersects_aabb(Vec3::new(-8.0, -8.0, -80.0), Vec3::new(8.0, 8.0, -64.0)));
        assert!(f.intersects_sphere(Vec3::new(0.0, 0.0, -72.0), 12.0));
    }

    #[test]
    fn box_behind_is_culled() {
        let f = Frustum::from_view_projection(camera_looking_forward().view_projection());
        assert!(!f.intersects_aabb(Vec3::new(-8.0, -8.0, 64.0), Vec3::new(8.0, 8.0, 80.0)));
        assert!(!f.intersects_sphere(Vec3::new(0.0, 0.0, 72.0), 12.0));
        assert_ne!(
            f.outside_mask(Vec3::new(0.0, 0.0, 72.0), Vec3::splat(8.0)),
            0
        );
    }

    #[test]
    fn box_far_to_the_side_is_culled() {
        let f = Frustum::from_view_projection(camera_looking_forward().view_projection());
        assert!(!f.intersects_aabb(
            Vec3::new(900.0, -8.0, -80.0),
            Vec3::new(916.0, 8.0, -64.0)
        ));
    }

    #[test]
    fn box_beyond_far_plane_is_culled() {
        let cam = camera_looking_forward();
        let f = Frustum::from_view_projection(cam.view_projection());
        let z = -cam.far - 64.0;
        assert!(!f.intersects_aabb(
            Vec3::new(-8.0, -8.0, z - 8.0),
            Vec3::new(8.0, 8.0, z + 8.0)
        ));
    }

    #[test]
    fn visible_box_has_zero_outside_mask() {
        let f = Frustum::from_view_projection(camera_looking_forward().view_projection());
        assert_eq!(
            f.outside_mask(Vec3::new(0.0, 0.0, -72.0), Vec3::splat(8.0)),
            0
        );
    }

    #[test]
    fn contains_point_matches_aabb_test() {
        let f = Frustum::from_view_projection(camera_looking_forward().view_projection());
        assert!(f.contains_point(Vec3::new(0.0, 0.0, -10.0)));
        assert!(!f.contains_point(Vec3::new(0.0, 0.0, 10.0)));
    }
}
