//! Saving generated sprites with the web demo's file names (docs/demo/app.js `finishCard` / the sheet button):
//!
//! * one sprite: `<slug>_<size>_s<seed>_<kk>.png` (kk = 00, 01, … in the run; `src` for the source sprite of
//!   "one shape"), at native size with transparency, like the demo's per-sprite PNG;
//! * the run: `<slug>_<size>_s<seed>_sheet.png`, the sprites in rows of up to 8 (the source first), transparent.
//!
//! `scale` > 1 upscales (nearest neighbour) and adds `_x<scale>` before `.png`, so a native file and an upscaled one
//! never overwrite each other. The demo's own sheet is a preview (x8 / x5 / x4 with 8 px gaps); here the sheet is a
//! plain grid of sprite cells, so it can be cut back into sprites. Both load back through [`Sprite::from_png`]
//! (an upscaled file is recognised as its native size).

use std::path::Path;

use anyhow::{Result, bail};

use crate::config::slug;
use crate::export::sheet_png;
use crate::sprite::Sprite;

/// Columns of a sheet (the demo: `Math.min(8, cards.length)`).
pub const SHEET_COLS: usize = 8;

fn scale_tag(scale: u32) -> String {
    if scale > 1 { format!("_x{scale}") } else { String::new() }
}

/// Tag of sprite k of a run: "00", "01", … (the demo's `String(k).padStart(2, "0")`).
pub fn sprite_tag(k: usize) -> String {
    format!("{k:02}")
}

/// `<slug>_<size>_s<seed>_<tag>.png` (+ `_x<scale>`).
pub fn sprite_file_name(caption: &str, size: usize, seed: u32, tag: &str, scale: u32) -> String {
    format!("{}_{size}_s{seed}_{tag}{}.png", slug(caption), scale_tag(scale))
}

/// `<slug>_<size>_s<seed>_sheet.png` (+ `_x<scale>`).
pub fn sheet_file_name(caption: &str, size: usize, seed: u32, scale: u32) -> String {
    format!("{}_{size}_s{seed}_sheet{}.png", slug(caption), scale_tag(scale))
}

/// One sprite as an RGBA PNG (alpha 0 or 255), `scale` x nearest.
pub fn save_sprite_png(sprite: &Sprite, scale: u32, path: &Path) -> Result<()> {
    sheet_png(&[vec![sprite.clone()]], scale.max(1), 0, path)
}

/// The sprites of a run in rows of [`SHEET_COLS`], no gaps, transparent, `scale` x nearest.
pub fn save_sheet_png(sprites: &[Sprite], scale: u32, path: &Path) -> Result<()> {
    if sprites.is_empty() {
        bail!("no sprites to save");
    }
    let rows: Vec<Vec<Sprite>> = sprites.chunks(SHEET_COLS).map(<[Sprite]>::to_vec).collect();
    sheet_png(&rows, scale.max(1), 0, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn names_like_the_demo() {
        assert_eq!(sprite_file_name("Burger skull", 16, 3, &sprite_tag(0), 1), "burger_skull_16_s3_00.png");
        assert_eq!(sprite_file_name("  рыжий кот! ", 32, 12, "src", 4), "рыжий_кот_32_s12_src_x4.png");
        assert_eq!(sheet_file_name("", 24, 0, 1), "no_text_24_s0_sheet.png");
        assert_eq!(sheet_file_name("a teal cookie jar", 16, 7, 2), "a_teal_cookie_jar_16_s7_sheet_x2.png");
    }

    #[test]
    fn png_round_trip() {
        let pal = Arc::new(vec![[200u8, 30, 30], [20, 40, 220], [250, 250, 250]]);
        let dir = std::env::temp_dir().join(format!("animator_ui_save_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for size in [16usize, 24, 32] {
            // a pattern with no 2x2 uniformity, so a native 32 px file cannot pass for a 2x 16 px one
            let px: Vec<i16> = (0..size * size).map(|p| if (p * 7 + p / size) % 5 == 0 { -1 } else { ((p * 3 + p / size) % 3) as i16 }).collect();
            let s = Sprite::new(size, px, pal.clone());
            for scale in [1u32, 2, 4] {
                let p = dir.join(sprite_file_name("round trip", size, 1, "00", scale));
                save_sprite_png(&s, scale, &p).unwrap();
                let (back, f) = Sprite::from_png(&p).unwrap();
                assert_eq!((back.size, f), (size, scale as usize));
                assert_eq!(back.rgba(), s.rgba(), "{size} px x{scale}");
            }
        }
        let sprites: Vec<Sprite> = (0..10).map(|k| Sprite::new(16, (0..256).map(|p| if (p + k) % 4 == 0 { -1 } else { (p % 3) as i16 }).collect(), pal.clone())).collect();
        let p = dir.join(sheet_file_name("round trip", 16, 1, 1));
        save_sheet_png(&sprites, 1, &p).unwrap();
        let img = image::open(&p).unwrap().to_rgba8();
        assert_eq!((img.width(), img.height()), (8 * 16, 2 * 16));
        let cell = |k: usize| -> Vec<u8> {
            let (ox, oy) = ((k % 8) as u32 * 16, (k / 8) as u32 * 16);
            (0..16).flat_map(|y| (0..16).map(move |x| (ox + x, oy + y))).flat_map(|(x, y)| img.get_pixel(x, y).0).collect()
        };
        for (k, s) in sprites.iter().enumerate() {
            let (back, _) = Sprite::from_rgba(16, 16, &cell(k)).unwrap();
            assert_eq!(back.rgba(), s.rgba());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
