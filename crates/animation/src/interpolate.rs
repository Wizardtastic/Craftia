//! Interpolation helpers: lerp, slerp, cubic spline.

/// Linear interpolation between two f32 values.
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Linear interpolation between two [f32; 4] values (component-wise).
pub fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
        lerp(a[3], b[3], t),
    ]
}

/// Spherical linear interpolation between two quaternions [x, y, z, w].
/// Falls back to linear interpolation when quaternions are nearly parallel.
pub fn slerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

    // If quaternions are very close, use linear interpolation.
    if dot.abs() > 0.9995 {
        return lerp4(a, b, t);
    }

    // Ensure we take the shortest path.
    let sign = if dot < 0.0 { -1.0 } else { 1.0 };
    let b = [b[0] * sign, b[1] * sign, b[2] * sign, b[3] * sign];
    let dot = dot.abs();

    let theta = dot.acos();
    let sin_theta = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / sin_theta;
    let wb = (t * theta).sin() / sin_theta;

    [
        wa * a[0] + wb * b[0],
        wa * a[1] + wb * b[1],
        wa * a[2] + wb * b[2],
        wa * a[3] + wb * b[3],
    ]
}

/// Cubic Hermite interpolation for one component.
/// `p0`, `m0` = value and tangent at start; `p1`, `m1` = value and tangent at end.
pub fn cubic_hermite(p0: f32, m0: f32, p1: f32, m1: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * p0
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * p1
        + (t3 - t2) * m1
}

/// Cubic Hermite interpolation for [f32; 4].
pub fn cubic_hermite4(p0: [f32; 4], m0: [f32; 4], p1: [f32; 4], m1: [f32; 4], t: f32) -> [f32; 4] {
    [
        cubic_hermite(p0[0], m0[0], p1[0], m1[0], t),
        cubic_hermite(p0[1], m0[1], p1[1], m1[1], t),
        cubic_hermite(p0[2], m0[2], p1[2], m1[2], t),
        cubic_hermite(p0[3], m0[3], p1[3], m1[3], t),
    ]
}
