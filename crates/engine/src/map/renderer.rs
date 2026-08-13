//! Framebuffer builder: converts `ColumnSample` data into an RGBA pixel
//! buffer suitable for GPU upload as the minimap texture.

use voxel_world::BlockRegistry;

use super::MapState;

impl MapState {
    /// Rebuild the CPU framebuffer from the latest samples.
    /// Each sample's `top_block` is looked up in the registry to get `map_color`.
    /// Height-based shading is applied: lower blocks are darker.
    pub fn rebuild_framebuffer(&mut self, registry: &BlockRegistry) {
        let size = self.fb_size as usize;
        let mut fb = vec![0u8; size * size * 4]; // RGBA8

        let blocks_per_px = self.blocks_per_pixel as i32;
        let half = (self.fb_size / 2) as i32;

        for sample in &self.samples {
            // Map world (x,z) ΓåÆ framebuffer (px, py).
            let dx = sample.block_x - self.center_x;
            let dz = sample.block_z - self.center_z;
            let px = half + dx / blocks_per_px;
            let py = half + dz / blocks_per_px;
            if px < 0 || px >= size as i32 || py < 0 || py >= size as i32 {
                continue;
            }

            let def = registry.get(sample.top_block);
            let base = def.map_color;

            // Height-based shading: scale brightness by height.
            // Sea level (62) = neutral, above = lighter, below = darker.
            let height_factor = {
                let h = sample.height as f32;
                let sea = 62.0;
                let scale = if h > sea {
                    1.0 + (h - sea) * 0.003
                } else {
                    0.7 + (h / sea) * 0.3
                };
                scale.clamp(0.5, 1.5)
            };

            let r = (base[0] as f32 * height_factor).min(255.0) as u8;
            let g = (base[1] as f32 * height_factor).min(255.0) as u8;
            let b = (base[2] as f32 * height_factor).min(255.0) as u8;
            let a = base[3];

            let idx = ((py as usize) * size + (px as usize)) * 4;
            fb[idx] = r;
            fb[idx + 1] = g;
            fb[idx + 2] = b;
            fb[idx + 3] = a;
        }

        // Draw waypoint markers on the framebuffer.
        for wp in &self.waypoints {
            let dx = wp.x - self.center_x;
            let dz = wp.z - self.center_z;
            let px = half + dx / blocks_per_px;
            let py = half + dz / blocks_per_px;
            if px >= 0 && px < size as i32 && py >= 0 && py < size as i32 {
                // Draw a 3├ù3 dot.
                for dy in -1..=1i32 {
                    for ddx in -1..=1i32 {
                        let fx = px + ddx;
                        let fy = py + dy;
                        if fx >= 0 && fx < size as i32 && fy >= 0 && fy < size as i32 {
                            let idx = ((fy as usize) * size + (fx as usize)) * 4;
                            fb[idx] = wp.color[0];
                            fb[idx + 1] = wp.color[1];
                            fb[idx + 2] = wp.color[2];
                            fb[idx + 3] = wp.color[3];
                        }
                    }
                }
            }
        }

        self.framebuffer = fb;
        self.texture_dirty = true;
        self.dirty = false;
    }
}
