//! A palette-indexed square sprite (-1 = transparent) and helpers shared by the UI, animators and exporters.

use std::sync::Arc;

pub type Palette = Arc<Vec<[u8; 3]>>;

#[derive(Clone, Debug)]
pub struct Sprite {
    pub size: usize,
    /// Row-major palette indices, -1 = transparent.
    pub px: Vec<i16>,
    pub palette: Palette,
}

/// Animation frames are sprites with the reference palette.
pub type Frame = Sprite;

impl Sprite {
    pub fn new(size: usize, px: Vec<i16>, palette: Palette) -> Self {
        debug_assert_eq!(px.len(), size * size);
        Sprite { size, px, palette }
    }

    pub fn blank_like(&self) -> Self {
        Sprite { size: self.size, px: vec![-1; self.px.len()], palette: self.palette.clone() }
    }

    #[inline]
    pub fn get(&self, x: i32, y: i32) -> i16 {
        let s = self.size as i32;
        if x < 0 || y < 0 || x >= s || y >= s { -1 } else { self.px[(y * s + x) as usize] }
    }

    #[inline]
    pub fn set(&mut self, x: i32, y: i32, v: i16) {
        let s = self.size as i32;
        if x >= 0 && y >= 0 && x < s && y < s {
            self.px[(y * s + x) as usize] = v;
        }
    }

    pub fn rgba(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.px.len() * 4];
        for (i, &k) in self.px.iter().enumerate() {
            if k >= 0 {
                if let Some(c) = self.palette.get(k as usize) {
                    out[4 * i..4 * i + 3].copy_from_slice(c);
                    out[4 * i + 3] = 255;
                }
            }
        }
        out
    }

    /// Bounding box of the opaque pixels: (x0, y0, x1, y1) inclusive.
    pub fn bbox(&self) -> Option<(i32, i32, i32, i32)> {
        let s = self.size as i32;
        let (mut x0, mut y0, mut x1, mut y1) = (s, s, -1, -1);
        for y in 0..s {
            for x in 0..s {
                if self.px[(y * s + x) as usize] >= 0 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        (x1 >= 0).then_some((x0, y0, x1, y1))
    }

    pub fn mirrored(&self) -> Self {
        let s = self.size;
        let mut out = self.blank_like();
        for y in 0..s {
            for x in 0..s {
                out.px[y * s + x] = self.px[y * s + (s - 1 - x)];
            }
        }
        out
    }

    pub fn opaque_count(&self) -> usize {
        self.px.iter().filter(|&&k| k >= 0).count()
    }

    /// A pixel-art image (RGBA, row-major) -> a sprite with its own palette (colours in order of first use; alpha
    /// < 128 = transparent). It must be square: 16, 24 or 32 px, or an integer nearest-neighbour upscale of one. The
    /// smallest of 16 / 24 / 32 whose blocks are all one colour wins (a 32 px file that is a 2x upscale of a 16 px
    /// sprite is read as 16 px). Returns the sprite and the upscale factor found.
    pub fn from_rgba(w: usize, h: usize, rgba: &[u8]) -> anyhow::Result<(Sprite, usize)> {
        if w != h || rgba.len() != w * h * 4 {
            anyhow::bail!("the image is {w}x{h}: a sprite must be square (16, 24 or 32 px, or an integer upscale)");
        }
        let px = |x: usize, y: usize| -> [u8; 4] {
            let i = (y * w + x) * 4;
            let a = if rgba[i + 3] < 128 { 0 } else { 255 };
            if a == 0 { [0; 4] } else { [rgba[i], rgba[i + 1], rgba[i + 2], a] }
        };
        let uniform = |s: usize, f: usize| (0..s * s).all(|p| {
            let (x0, y0) = ((p % s) * f, (p / s) * f);
            let c = px(x0, y0);
            (0..f).all(|dy| (0..f).all(|dx| px(x0 + dx, y0 + dy) == c))
        });
        let Some((s, f)) = [16usize, 24, 32].into_iter().filter(|&s| w % s == 0).map(|s| (s, w / s)).find(|&(s, f)| uniform(s, f)) else {
            anyhow::bail!("the image is {w}x{w}: not a 16, 24 or 32 px sprite or an integer upscale of one");
        };
        let mut palette: Vec<[u8; 3]> = Vec::new();
        let mut index: std::collections::HashMap<[u8; 3], i16> = std::collections::HashMap::new();
        let mut out = Vec::with_capacity(s * s);
        for p in 0..s * s {
            let c = px((p % s) * f, (p / s) * f);
            if c[3] == 0 {
                out.push(-1);
                continue;
            }
            let rgb = [c[0], c[1], c[2]];
            let k = *index.entry(rgb).or_insert_with(|| {
                palette.push(rgb);
                (palette.len() - 1) as i16
            });
            out.push(k);
        }
        Ok((Sprite::new(s, out, Arc::new(palette)), f))
    }

    /// [`Sprite::from_rgba`] of a PNG file.
    pub fn from_png(path: &std::path::Path) -> anyhow::Result<(Sprite, usize)> {
        let img = image::open(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?.to_rgba8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        Sprite::from_rgba(w, h, img.as_raw())
    }
}

/// mulberry32, seeded exactly like `makeRng` in the web demo's engine.js, so a seed draws the same random numbers.
#[derive(Clone, Debug)]
pub struct Mulberry32(u32);

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Mulberry32(seed ^ 0x9e37_79b9)
    }
    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b_79f5);
        let a = self.0;
        let mut t = (a ^ (a >> 15)).wrapping_mul(1 | a);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        ((t ^ (t >> 14)) as f64) / 4_294_967_296.0
    }
}

#[cfg(test)]
mod tests {
    use super::{Mulberry32, Sprite};

    #[test]
    fn png_upscale_detected() {
        // a 16 px sprite with two colours, saved 2x (32x32): read back as 16 px
        let mut rgba = vec![0u8; 32 * 32 * 4];
        for y in 0..32 {
            for x in 0..32 {
                let (sx, sy) = (x / 2, y / 2);
                if (4..12).contains(&sx) && (2..14).contains(&sy) {
                    let c = if sy < 6 { [200, 30, 30, 255] } else { [20, 40, 220, 255] };
                    rgba[(y * 32 + x) * 4..(y * 32 + x) * 4 + 4].copy_from_slice(&c);
                }
            }
        }
        let (s, f) = Sprite::from_rgba(32, 32, &rgba).unwrap();
        assert_eq!((s.size, f, s.palette.len(), s.opaque_count()), (16, 2, 2, 8 * 12));
        assert!(Sprite::from_rgba(20, 20, &vec![0u8; 20 * 20 * 4]).is_err());
    }

    #[test]
    fn mulberry_matches_js() {
        // makeRng(seed) from engine.js, first three values, computed with Node 24.
        let want: [(u32, [f64; 3]); 3] = [
            (0, [0.3588899802416563, 0.10590326134115458, 0.675290479324758]),
            (12345, [0.5975837109144777, 0.6314287825953215, 0.6228211561683565]),
            (4_000_000_000, [0.011607177322730422, 0.20093227201141417, 0.12383230426348746]),
        ];
        for (seed, vals) in want {
            let mut r = Mulberry32::new(seed);
            for v in vals {
                assert_eq!(r.next_f64(), v);
            }
        }
    }
}
