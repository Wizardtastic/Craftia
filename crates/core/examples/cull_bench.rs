//! Permanent CPU frustum-culling benchmark.
//!
//!   cargo run --release -p voxel-core --example cull_bench
//!
//! Tests a realistic mixed set of in-front and behind-camera AABBs. The check
//! asserts the culling count so a sign/plane regression fails the benchmark
//! instead of silently returning "everything visible".

use std::time::Instant;

use glam::{Mat4, Vec3};
use voxel_core::{Camera, Frustum};

fn main() {
    let cam = Camera::default();
    let vp: Mat4 = cam.view_projection();
    let frustum = Frustum::from_view_projection(vp);

    let mut boxes = Vec::with_capacity(200_000);
    for i in 0..200_000u32 {
        let x = ((i * 37) % 400) as f32 - 200.0;
        let y = ((i * 53) % 160) as f32 - 80.0;
        let z = -(((i * 29) % 400) as f32) + 40.0;
        boxes.push((Vec3::new(x, y, z), Vec3::splat(8.0)));
    }

    let iters = 100u32;
    let t0 = Instant::now();
    let mut visible = 0usize;
    for _ in 0..iters {
        for (c, h) in &boxes {
            if frustum.intersects_center_half(*c, *h) {
                visible += 1;
            }
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
    let per_test = t0.elapsed().as_nanos() as f64 / (f64::from(iters) * boxes.len() as f64);
    println!(
        "cull: {ms:.3} ms for {} AABBs ({per_test:.2} ns/test, {visible} visible hits)",
        boxes.len()
    );
    assert!(visible > 0, "nothing visible: frustum extraction is broken");
    assert!(
        visible < iters as usize * boxes.len(),
        "everything visible: culling is broken"
    );
}
