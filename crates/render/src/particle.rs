//! GPU-resident 3D billboarded particle system.
//!
//! Phase 1 (this file) ships the data structures and CPU simulation:
//! - [`ParticleInstance`] is the layout uploaded each frame as an
//!   instance-rate vertex stream.
//! - [`ParticleManager`] owns the particle pool, exposes spawn APIs
//!   (`emit_break`, `emit_place`), advances physics in [`ParticleManager::update`],
//!   and snapshots into the GPU layout via [`ParticleManager::instances`].
//!
//! Phase 2 will fold in depth-aware softening via subpass input attachments;
//! no struct changes here, only shader/pipeline changes.

use glam::Vec3;
use rand::Rng;

/// Maximum simultaneous particles. 4096 keeps the host-visible VBO at 128 KB.
pub const MAX_PARTICLES: usize = 4096;

/// Soft cap; can be tuned down for low-end hardware. Always clamped down to
/// [`MAX_PARTICLES`].
pub const DEFAULT_MAX_PARTICLES: usize = 2048;

/// GPU-side per-particle data (32 bytes), uploaded as a single vertex stream
/// with `input_rate = INSTANCE`. Bound together with the 6-vertex unit-quad
/// VBO at binding 0 to render N instances of a quad.
///
/// Layout (must stay in sync with `crates/render/src/renderer/pipeline.rs`
/// `create_particle_pipeline` vertex attribute setup, AND with the matching
/// attribute locations in `shaders/particle.vert`):
///   - location 2: `vec3 pos`      (offset 0)
///   - location 3: `float rot`      (offset 12)
///   - location 4: `uint color`     (offset 16, unpacked as R8G8B8A8_UNORM)
///   - location 5: `float size`     (offset 20)
///   - location 6: `float tile`     (offset 24, atlas tile index, 0..255)
///   - unused 4-byte pad           (offset 28)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleInstance {
    /// World position.
    pub pos: [f32; 3], // 12, offset 0
    /// Rotation around the camera-facing axis (radians).
    pub rot: f32, // 4, offset 12
    /// Packed RGBA8 colour (low byte = R, high byte = A — `unpackUnorm4x8`).
    pub color: u32, // 4, offset 16
    /// Per-particle half-extent in world units (e.g. `0.10` = 0.20 m wide).
    pub size: f32, // 4, offset 20
    /// Atlas tile index (the fragment shader does
    /// `ivec2(tile % 16, tile / 16) + fract(uv) / 16` to sample the right
    /// sub-tile of the 16×16 atlas). Phase 1 only ships break particles,
    /// which carry the broken block's top-face tile.
    pub tile: f32, // 4, offset 24
    /// Reserved for Phase 2 (additive blend cue); ignored by the current
    /// alpha-blend pipeline.
    pub _pad: [f32; 1], // 4, offset 28
}
// total: 32 bytes

/// CPU-side per-particle simulation state. Kept private to the renderer;
/// the engine only sees spawn APIs.
#[derive(Clone, Copy, Debug)]
struct ParticleSim {
    pos: Vec3,
    vel: Vec3,
    color: [u8; 4],
    life: f32,
    _max_life: f32,
    size: f32,
    rot: f32,
    rot_rate: f32,
    _additive: bool,
}

/// Owns the per-particle CPU simulation. Allocation is bounded; new spawns
/// beyond the cap silently drop.
#[derive(Default)]
pub struct ParticleManager {
    particles: Vec<ParticleSim>,
    max_particles: usize,
}

impl ParticleManager {
    /// Create a manager with the given cap (clamped down to [`MAX_PARTICLES`]).
    pub fn new(max_particles: usize) -> Self {
        let cap = max_particles.min(MAX_PARTICLES);
        Self {
            particles: Vec::with_capacity(cap),
            max_particles: cap,
        }
    }

    /// Spawn particles for a block break event.
    ///
    /// `pos` is the block centroid, `color` is the colour of the broken
    /// block (RGBA8), `normal` is the face normal that the break came from,
    /// which biases particle ejection.
    pub fn emit_break(&mut self, pos: Vec3, color: [u8; 4], normal: Vec3) {
        let count = 12;
        let mut rng = rand::thread_rng();
        for _ in 0..count {
            if self.particles.len() >= self.max_particles {
                break;
            }
            let speed = rng.gen::<f32>() * 3.0;
            let jx = rng.gen::<f32>() - 0.5;
            let jy = rng.gen::<f32>() - 0.5;
            let jz = rng.gen::<f32>() - 0.5;
            let vel = normal * speed + Vec3::new(jx, jy, jz) * 2.0;
            self.particles.push(ParticleSim {
                pos,
                vel,
                color,
                life: 1.0 + rng.gen::<f32>(),
                _max_life: 1.5,
                size: 0.05 + rng.gen::<f32>() * 0.05,
                rot: rng.gen::<f32>() * std::f32::consts::TAU,
                rot_rate: (rng.gen::<f32>() - 0.5) * 4.0,
                _additive: false,
            });
        }
    }

    /// Spawn particles for a block placement event.
    pub fn emit_place(&mut self, pos: Vec3, color: [u8; 4]) {
        let count = 6;
        let mut rng = rand::thread_rng();
        for _ in 0..count {
            if self.particles.len() >= self.max_particles {
                break;
            }
            let vel = Vec3::new(
                rng.gen::<f32>() - 0.5,
                rng.gen::<f32>() * 0.5,
                rng.gen::<f32>() - 0.5,
            ) * 1.5;
            self.particles.push(ParticleSim {
                pos,
                vel,
                color,
                life: 0.5 + rng.gen::<f32>() * 0.5,
                _max_life: 1.0,
                size: 0.04 + rng.gen::<f32>() * 0.04,
                rot: rng.gen::<f32>() * std::f32::consts::TAU,
                rot_rate: (rng.gen::<f32>() - 0.5) * 4.0,
                _additive: false,
            });
        }
    }

    /// Advance physics: linear velocity, gravity, lifetime decay, angular spin.
    /// Drop dead particles.
    pub fn update(&mut self, dt: f32) {
        for p in &mut self.particles {
            p.pos += p.vel * dt;
            p.vel.y -= 9.8 * dt;
            p.life -= dt;
            p.rot += p.rot_rate * dt;
        }
        self.particles.retain(|p| p.life > 0.0);
    }

    /// Snapshot the current particle set into the GPU-ready
    /// [`ParticleInstance`] layout. Returns a fresh `Vec` each call.
    pub fn instances(&self) -> Vec<ParticleInstance> {
        let mut out = Vec::with_capacity(self.particles.len());
        for p in &self.particles {
            let color = u32::from(p.color[0])
                | (u32::from(p.color[1]) << 8)
                | (u32::from(p.color[2]) << 16)
                | (u32::from(p.color[3]) << 24);
            // Phase 1 placeholder: every particle samples atlas tile 0
            // (a procedural grey "rubble" tile). Phase 2 will let the
            // engine pass a tile index per spawn call so each block's
            // break particles carry that block's top-face texture.
            out.push(ParticleInstance {
                pos: [p.pos.x, p.pos.y, p.pos.z],
                rot: p.rot,
                color,
                size: p.size,
                tile: 0.0,
                _pad: [0.0],
            });
        }
        out
    }

    pub fn len(&self) -> usize {
        self.particles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particle_instance_size_is_32() {
        assert_eq!(
            std::mem::size_of::<ParticleInstance>(),
            32,
            "ParticleInstance stride must be 32 bytes; update pipeline binding + shader attribute offsets."
        );
    }

    #[test]
    fn particle_instance_is_pod_safe() {
        let v = ParticleInstance::default();
        let bytes = bytemuck::bytes_of(&v);
        let back: ParticleInstance = *bytemuck::from_bytes(bytes);
        assert_eq!(back.pos, [0.0, 0.0, 0.0]);
        assert_eq!(back.rot, 0.0);
        assert_eq!(back.color, 0);
        assert_eq!(back.size, 0.0);
        assert_eq!(back.tile, 0.0);
    }

    #[test]
    fn manager_caps_at_max_particles() {
        let mut mgr = ParticleManager::new(2);
        let pos = Vec3::ZERO;
        let color = [255, 200, 100, 255];
        let normal = Vec3::Y;
        mgr.emit_break(pos, color, normal); // requests 12, capped to 2
        assert_eq!(mgr.len(), 2);

        // A second emit_break should still hold the cap.
        mgr.emit_break(pos, color, normal);
        assert_eq!(mgr.len(), 2);
    }

    #[test]
    fn manager_caps_at_max_particles_global() {
        let mgr = ParticleManager::new(MAX_PARTICLES + 1000);
        assert_eq!(
            mgr.max_particles, MAX_PARTICLES,
            "manager should clamp to MAX_PARTICLES even if caller asked for more"
        );
    }

    #[test]
    fn update_drops_dead_particles() {
        let mut mgr = ParticleManager::new(8);
        mgr.emit_break(Vec3::ZERO, [255, 0, 0, 255], Vec3::Y);
        assert!(mgr.len() > 0);
        // Step far past max_life so every particle expires.
        mgr.update(100.0);
        assert!(mgr.is_empty(), "all particles should die after huge dt");
    }

    #[test]
    fn instances_match_particles() {
        let mut mgr = ParticleManager::new(8);
        mgr.emit_break(Vec3::new(1.0, 2.0, 3.0), [10, 20, 30, 40], Vec3::Y);
        let instances = mgr.instances();
        assert_eq!(instances.len(), mgr.len());
        // Packed colour should match the input.
        let expected = 10u32 | (20u32 << 8) | (30u32 << 16) | (40u32 << 24);
        for inst in &instances {
            assert_eq!(inst.color, expected);
            assert_eq!(inst.pos, [1.0, 2.0, 3.0]);
        }
    }
}
