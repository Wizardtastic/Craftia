/// Generate a mip chain by repeatedly downsampling 2x with Nearest filter.
/// Returns Vec<Vec<u8>> where [0] is the original (largest) level.
///
/// Nearest-neighbor downsampling is correct for pixel art -- it keeps colors
/// crisp and avoids blur that would make distant blocks look muddy.
pub fn generate_mip_chain(rgba: &[u8], width: u32, height: u32) -> Vec<Vec<u8>> {
    let max_levels = 1 + (width.max(height) as f64).log2().floor() as u32;
    let mut levels = Vec::with_capacity(max_levels as usize);
    levels.push(rgba.to_vec());

    let mut w = width;
    let mut h = height;
    let mut src = rgba.to_vec();

    for _ in 1..max_levels {
        let nw = (w / 2).max(1);
        let nh = (h / 2).max(1);
        let mut dst = vec![0u8; (nw * nh * 4) as usize];

        for y in 0..nh {
            for x in 0..nw {
                // Nearest-neighbor downscale: sample from 2x2 pixel block,
                // take the top-left pixel (no blending -- matches pixel-art aesthetic).
                let sx = x * 2;
                let sy = y * 2;
                let si = ((sy * w + sx) * 4) as usize;
                let di = ((y * nw + x) * 4) as usize;
                dst[di..di + 4].copy_from_slice(&src[si..si + 4]);
            }
        }

        levels.push(dst.clone());
        src = dst;
        w = nw;
        h = nh;
    }

    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_chain_single_pixel() {
        let rgba = vec![255, 0, 0, 255]; // 1x1 red
        let mips = generate_mip_chain(&rgba, 1, 1);
        assert_eq!(mips.len(), 1);
        assert_eq!(mips[0], rgba);
    }

    #[test]
    fn mip_chain_2x2() {
        let rgba = vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
        ];
        let mips = generate_mip_chain(&rgba, 2, 2);
        assert_eq!(mips.len(), 2);
        assert_eq!(mips[0].len(), 16); // 2x2x4
        assert_eq!(mips[1].len(), 4); // 1x1x4
        // Nearest takes top-left pixel
        assert_eq!(&mips[1][..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn mip_chain_4x4() {
        let rgba = vec![128u8; 4 * 4 * 4];
        let mips = generate_mip_chain(&rgba, 4, 4);
        assert_eq!(mips.len(), 3); // 4x4, 2x2, 1x1
        assert_eq!(mips[0].len(), 64);
        assert_eq!(mips[1].len(), 16);
        assert_eq!(mips[2].len(), 4);
    }

    #[test]
    fn mip_chain_256x256() {
        let rgba = vec![0u8; 256 * 256 * 4];
        let mips = generate_mip_chain(&rgba, 256, 256);
        assert_eq!(mips.len(), 9); // 256, 128, 64, 32, 16, 8, 4, 2, 1
        assert_eq!(mips[0].len(), 256 * 256 * 4);
        assert_eq!(mips[8].len(), 4);
    }
}
