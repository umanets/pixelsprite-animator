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

    /// Any image -> a sprite. A proper sprite file ([`Sprite::from_rgba`]) is read as it is; anything else is
    /// fitted ([`Sprite::fit_rgba`]). Returns the sprite, the upscale factor found and whether it was fitted.
    pub fn from_image_file(path: &std::path::Path) -> anyhow::Result<(Sprite, usize, bool)> {
        let img = image::open(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?.to_rgba8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        if let Ok((s, f)) = Sprite::from_rgba(w, h, img.as_raw()) {
            return Ok((s, f, false));
        }
        Ok((Sprite::fit_rgba(w, h, img.as_raw())?, 1, true))
    }

    /// An arbitrary image (RGBA, row-major) -> a 16 / 24 / 32 px sprite.
    ///
    /// 1. A picture without transparency gets its background removed: while the outer edge of what is left is one
    ///    colour, that colour is flood-filled away from the edge (a frame first, then the paper behind the
    ///    character). Parts of the character with the background's colour survive unless they touch it.
    /// 2. The opaque part is cropped.
    /// 3. Pixel art drawn with blocks of n x n screen pixels (n need not be whole: a 32 px sprite saved at 1200 px
    ///    has 37.5 px blocks) is reduced to one pixel per block; the block size is the largest one that nearly all
    ///    colour runs are multiples of.
    /// 4. If the longer side is still over 30 px it is shrunk with nearest-neighbour sampling.
    /// 5. The result is centred on the smallest of the three canvases that holds it with a free pixel all around
    ///    (the sprites the animation model learned from do not touch the canvas edge, and an outline needs room).
    pub fn fit_rgba(w: usize, h: usize, rgba: &[u8]) -> anyhow::Result<Sprite> {
        if w == 0 || h == 0 || rgba.len() != w * h * 4 {
            anyhow::bail!("the image is empty");
        }
        let rgb = |k: usize| [rgba[4 * k], rgba[4 * k + 1], rgba[4 * k + 2]];
        let near = |a: [u8; 3], b: [u8; 3]| (0..3).all(|c| a[c].abs_diff(b[c]) <= 12);
        let mut solid: Vec<bool> = (0..w * h).map(|k| rgba[4 * k + 3] >= 128).collect();

        // ---- 1. background removal (only for pictures without any transparency)
        if solid.iter().all(|&o| o) {
            for layer in 0..4 {
                // the outer edge of what is left
                let edge: Vec<usize> = (0..w * h)
                    .filter(|&k| {
                        let (x, y) = (k % w, k / w);
                        solid[k] && (x == 0 || y == 0 || x == w - 1 || y == h - 1 || !solid[k - 1] || !solid[k + 1] || !solid[k - w] || !solid[k + w])
                    })
                    .collect();
                if edge.is_empty() {
                    break;
                }
                // its most common colour (sampled), and how much of the edge has it
                let mut votes: std::collections::HashMap<[u8; 3], usize> = std::collections::HashMap::new();
                for &k in edge.iter().step_by((edge.len() / 4096).max(1)) {
                    *votes.entry(rgb(k)).or_default() += 1;
                }
                let bg = votes.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c).unwrap_or([0; 3]);
                let same = edge.iter().filter(|&&k| near(rgb(k), bg)).count();
                if same * 10 < edge.len() * 9 {
                    break; // a mixed edge: this is the character (or a photo)
                }
                // flood from the edge over that colour
                let mut gone = vec![false; w * h];
                let mut stack: Vec<usize> = edge.into_iter().filter(|&k| near(rgb(k), bg)).collect();
                for &k in &stack {
                    gone[k] = true;
                }
                let mut n_gone = stack.len();
                while let Some(k) = stack.pop() {
                    let (x, y) = (k % w, k / w);
                    let mut visit = |q: usize| {
                        if solid[q] && !gone[q] && near(rgb(q), bg) {
                            gone[q] = true;
                            n_gone += 1;
                            stack.push(q);
                        }
                    };
                    if x > 0 {
                        visit(k - 1);
                    }
                    if x + 1 < w {
                        visit(k + 1);
                    }
                    if y > 0 {
                        visit(k - w);
                    }
                    if y + 1 < h {
                        visit(k + w);
                    }
                }
                let left = solid.iter().filter(|&&o| o).count();
                // after the first layer only something as large as a background goes (not a sprite's outline),
                // and something must stay
                if n_gone == left || (layer > 0 && n_gone * 2 < left) {
                    break;
                }
                for k in 0..w * h {
                    if gone[k] {
                        solid[k] = false;
                    }
                }
            }
        }

        // ---- 2. crop
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
        for y in 0..h {
            for x in 0..w {
                if solid[y * w + x] {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        if x0 > x1 || y0 > y1 {
            anyhow::bail!("the image is fully transparent");
        }
        let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
        let cell = |x: usize, y: usize| -> Option<[u8; 3]> {
            let k = (y0 + y) * w + x0 + x;
            solid[k].then(|| rgb(k))
        };

        // ---- 3. the block size: every run of one colour along the rows and the columns
        let mut runs = vec![0usize; bw.max(bh) + 1]; // runs[len] = how many
        for y in 0..bh {
            let mut run = 1;
            for x in 1..=bw {
                if x < bw && cell(x, y) == cell(x - 1, y) {
                    run += 1;
                } else {
                    runs[run] += 1;
                    run = 1;
                }
            }
        }
        for x in 0..bw {
            let mut run = 1;
            for y in 1..=bh {
                if y < bh && cell(x, y) == cell(x, y - 1) {
                    run += 1;
                } else {
                    runs[run] += 1;
                    run = 1;
                }
            }
        }
        let total: usize = runs.iter().enumerate().map(|(len, n)| len * n).sum();
        let fits = |len: usize, b: usize| {
            let m = (len as f64 / b as f64).round();
            m >= 1.0 && (len as f64 - m * b as f64).abs() <= (0.04 * len as f64).max(1.0)
        };
        let mut block = 1.0f64;
        for b in (2..=bw.max(bh) / 8).rev() {
            let ok: usize = runs.iter().enumerate().filter(|&(len, _)| fits(len, b)).map(|(len, n)| len * n).sum();
            // single blocks must be common too (a flat shape has only long runs and no block size at all)
            let singles: usize = runs.iter().enumerate().filter(|&(len, _)| fits(len, b) && (len as f64 / b as f64).round() == 1.0).map(|(_, n)| *n).sum();
            let n_runs: usize = runs.iter().sum();
            if ok * 100 >= total * 92 && singles * 20 >= n_runs {
                // refine: the mean run length per block, over the runs that fit
                let (mut px_sum, mut blocks) = (0.0f64, 0.0f64);
                for (len, &n) in runs.iter().enumerate().filter(|&(len, _)| fits(len, b)) {
                    px_sum += (len * n) as f64;
                    blocks += (len as f64 / b as f64).round() * n as f64;
                }
                block = px_sum / blocks.max(1.0);
                break;
            }
        }
        let (nw, nh) = (((bw as f64 / block).round() as usize).max(1), ((bh as f64 / block).round() as usize).max(1)); // the native size

        // ---- 4. the target box: the native sprite, or scaled so that its longer side is 30 px
        let long = nw.max(nh);
        let (tw, th) = if long <= 30 { (nw, nh) } else { ((nw * 30).div_ceil(long).clamp(1, 30), (nh * 30).div_ceil(long).clamp(1, 30)) };

        // ---- 5. the canvas
        let size = [16usize, 24, 32].into_iter().find(|&s| tw + 2 <= s && th + 2 <= s).unwrap_or(32);
        let (ox, oy) = ((size - tw) / 2, (size - th) / 2);
        let mut palette: Vec<[u8; 3]> = Vec::new();
        let mut index: std::collections::HashMap<[u8; 3], i16> = std::collections::HashMap::new();
        let mut px = vec![-1i16; size * size];
        for ty in 0..th {
            for tx in 0..tw {
                // the centre of the target pixel, in source pixels
                let sx = ((2 * tx + 1) * bw / (2 * tw)).min(bw - 1);
                let sy = ((2 * ty + 1) * bh / (2 * th)).min(bh - 1);
                let Some(c) = cell(sx, sy) else { continue };
                let k = *index.entry(c).or_insert_with(|| {
                    palette.push(c);
                    (palette.len() - 1) as i16
                });
                px[(oy + ty) * size + ox + tx] = k;
            }
        }
        Ok(Sprite::new(size, px, Arc::new(palette)))
    }

    /// Most of the silhouette's edge is dark: the sprite is drawn with an outline, like the sprites the animation
    /// model was trained on.
    pub fn has_dark_outline(&self) -> bool {
        let s = self.size as i32;
        let (mut edge, mut dark) = (0usize, 0usize);
        for y in 0..s {
            for x in 0..s {
                let k = self.get(x, y);
                if k < 0 {
                    continue;
                }
                if [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| self.get(x + dx, y + dy) < 0) {
                    edge += 1;
                    if self.palette.get(k as usize).is_some_and(|c| luma(*c) < 70.0) {
                        dark += 1;
                    }
                }
            }
        }
        edge > 0 && dark * 10 >= edge * 6
    }

    /// The sprite with a one-pixel dark outline around its silhouette (in the free pixels next to it; the sprite's
    /// own pixels are not changed). The colour is the sprite's darkest one if that is near-black, else the ink of
    /// the animation model's training sprites.
    pub fn outlined(&self) -> Sprite {
        let s = self.size as i32;
        let (palette, k) = self.with_ink();
        let mut px = self.px.clone();
        for y in 0..s {
            for x in 0..s {
                if self.get(x, y) < 0 && [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| self.get(x + dx, y + dy) >= 0) {
                    px[(y * s + x) as usize] = k;
                }
            }
        }
        Sprite::new(self.size, px, Arc::new(palette))
    }

    /// The sprite with the edge of its silhouette repainted dark (an "inline": the outline is drawn on the sprite's
    /// own outermost pixels, so the silhouette keeps its size). Same ink as [`Sprite::outlined`].
    pub fn inlined(&self) -> Sprite {
        let s = self.size as i32;
        let (palette, k) = self.with_ink();
        let mut px = self.px.clone();
        for y in 0..s {
            for x in 0..s {
                if self.get(x, y) >= 0 && [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| self.get(x + dx, y + dy) < 0) {
                    px[(y * s + x) as usize] = k;
                }
            }
        }
        Sprite::new(self.size, px, Arc::new(palette))
    }

    /// The palette with the outline colour in it, and that colour's index: the sprite's darkest colour if that is
    /// near-black, else the ink of the animation model's training sprites.
    fn with_ink(&self) -> (Vec<[u8; 3]>, i16) {
        let darkest = self.palette.iter().copied().min_by(|a, b| luma(*a).total_cmp(&luma(*b)));
        let ink = darkest.filter(|c| luma(*c) < 32.0).unwrap_or([28, 24, 30]);
        let mut palette = (*self.palette).clone();
        let k = match palette.iter().position(|c| *c == ink) {
            Some(i) => i,
            None => {
                palette.push(ink);
                palette.len() - 1
            }
        };
        (palette, k as i16)
    }

    /// [`Sprite::from_rgba`] of a PNG file.
    pub fn from_png(path: &std::path::Path) -> anyhow::Result<(Sprite, usize)> {
        let img = image::open(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?.to_rgba8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        Sprite::from_rgba(w, h, img.as_raw())
    }
}

fn luma(c: [u8; 3]) -> f32 {
    0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32
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
    fn any_image_is_fitted() {
        // a 10 x 20 opaque block inside a 40 x 50 image: cropped, put on a 24 px canvas, centred, on the floor
        let (w, h) = (40usize, 50usize);
        let mut rgba = vec![0u8; w * h * 4];
        for y in 5..25 {
            for x in 7..17 {
                rgba[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[10, 200, 30, 255]);
            }
        }
        let s = Sprite::fit_rgba(w, h, &rgba).unwrap();
        assert_eq!((s.size, s.opaque_count(), s.bbox()), (24, 200, Some((7, 2, 16, 21))));
        // a 100 x 200 block: shrunk to 15 x 30 on a 32 px canvas
        let (w, h) = (120usize, 220usize);
        let mut rgba = vec![0u8; w * h * 4];
        for y in 10..210 {
            for x in 10..110 {
                rgba[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[1, 2, 3, 255]);
            }
        }
        let s = Sprite::fit_rgba(w, h, &rgba).unwrap();
        assert_eq!((s.size, s.bbox()), (32, Some((8, 1, 22, 30))));
        assert!(Sprite::fit_rgba(4, 4, &vec![0u8; 64]).is_err());
        // pixel art drawn with 12 px blocks on an odd canvas: read back at one pixel per block (10 x 12 -> 16 px)
        let (w, h) = (300usize, 450usize);
        let mut rgba = vec![0u8; w * h * 4];
        for by in 0..12 {
            for bx in 0..10 {
                let c = if (bx + by) % 2 == 0 { [200, 10, 10, 255] } else { [10, 10, 200, 255] };
                for y in 0..12 {
                    for x in 0..12 {
                        let i = ((91 + by * 12 + y) * w + 37 + bx * 12 + x) * 4;
                        rgba[i..i + 4].copy_from_slice(&c);
                    }
                }
            }
        }
        let s = Sprite::fit_rgba(w, h, &rgba).unwrap();
        assert_eq!((s.size, s.opaque_count(), s.palette.len(), s.bbox()), (16, 120, 2, Some((3, 2, 12, 13))));

        // an opaque picture: a frame, white paper, and a 9 x 12 character drawn with 37.5 px blocks
        let (w, h) = (1200usize, 1200usize);
        let mut rgba = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                if x < 16 || y < 16 || x >= w - 16 || y >= h - 16 {
                    rgba[(y * w + x) * 4..(y * w + x) * 4 + 3].copy_from_slice(&[222, 35, 120]);
                }
            }
        }
        for by in 0..12usize {
            for bx in 0..9usize {
                if (bx == 0 || bx == 8) && by < 3 {
                    continue; // a narrower head
                }
                let c = [(40 + 20 * ((bx * 3 + by * 5) % 7)) as u8, (60 + 9 * by) as u8, (90 + 11 * bx) as u8];
                let (xa, xb) = ((400.0 + bx as f64 * 37.5) as usize, (400.0 + (bx + 1) as f64 * 37.5) as usize);
                let (ya, yb) = ((300.0 + by as f64 * 37.5) as usize, (300.0 + (by + 1) as f64 * 37.5) as usize);
                for y in ya..yb {
                    for x in xa..xb {
                        rgba[(y * w + x) * 4..(y * w + x) * 4 + 3].copy_from_slice(&c);
                    }
                }
            }
        }
        let s = Sprite::fit_rgba(w, h, &rgba).unwrap();
        assert_eq!((s.size, s.opaque_count(), s.bbox()), (16, 9 * 12 - 6, Some((3, 2, 11, 13))));
        // no outline yet; with one, the silhouette grows by a pixel on every side and its edge is dark
        assert!(!s.has_dark_outline());
        let o = s.outlined();
        assert!(o.has_dark_outline());
        assert_eq!((o.bbox(), o.palette.len()), (Some((2, 1, 12, 14)), s.palette.len() + 1));
        assert!((0..o.px.len()).all(|i| s.px[i] < 0 || s.px[i] == o.px[i]));
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
