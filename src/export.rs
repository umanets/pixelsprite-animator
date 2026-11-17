//! Sprite sheets (PNG) and animated GIFs, nearest-neighbour scaled, transparent background.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Result, bail};

use crate::sprite::Sprite;

fn put(img: &mut image::RgbaImage, s: &Sprite, ox: u32, oy: u32, scale: u32) {
    let rgba = s.rgba();
    for y in 0..s.size as u32 {
        for x in 0..s.size as u32 {
            let i = ((y * s.size as u32 + x) * 4) as usize;
            if rgba[i + 3] == 0 {
                continue;
            }
            let c = image::Rgba([rgba[i], rgba[i + 1], rgba[i + 2], 255]);
            for dy in 0..scale {
                for dx in 0..scale {
                    img.put_pixel(ox + x * scale + dx, oy + y * scale + dy, c);
                }
            }
        }
    }
}

/// Rows of frames -> one PNG (row `r` = `rows[r]`, left to right), `scale` x nearest, `gap` px between cells.
pub fn sheet_png(rows: &[Vec<Sprite>], scale: u32, gap: u32, path: &Path) -> Result<()> {
    let cols = rows.iter().map(Vec::len).max().unwrap_or(0) as u32;
    if cols == 0 {
        bail!("nothing to export");
    }
    let cell = rows.iter().flatten().map(|s| s.size).max().unwrap_or(16) as u32 * scale;
    let w = cols * cell + (cols + 1) * gap;
    let h = rows.len() as u32 * cell + (rows.len() as u32 + 1) * gap;
    let mut img = image::RgbaImage::new(w, h);
    for (r, row) in rows.iter().enumerate() {
        for (c, s) in row.iter().enumerate() {
            put(&mut img, s, gap + c as u32 * (cell + gap), gap + r as u32 * (cell + gap), scale);
        }
    }
    img.save(path)?;
    Ok(())
}

/// Animated GIF: one global palette (index 0 = transparent), `fps` frames per second, endless loop or once.
pub fn gif(frames: &[Sprite], scale: u32, fps: f32, looping: bool, path: &Path) -> Result<()> {
    if frames.is_empty() {
        bail!("nothing to export");
    }
    let size = frames.iter().map(|f| f.size).max().unwrap_or(16);
    // smaller frames sit in the top-left corner of the largest one
    let rgba: Vec<Vec<u8>> = frames
        .iter()
        .map(|f| {
            let src = f.rgba();
            let mut out = vec![0u8; size * size * 4];
            for y in 0..f.size {
                out[y * size * 4..(y * size + f.size) * 4].copy_from_slice(&src[y * f.size * 4..(y + 1) * f.size * 4]);
            }
            out
        })
        .collect();
    let delay = (100.0 / fps.max(0.5)).round().clamp(2.0, 6000.0) as u16;
    gif_rgba(&rgba, size, size, &vec![delay; frames.len()], scale, looping, path)
}

/// Animated GIF of `w` x `h` RGBA pictures (alpha 0 = transparent), each with its own delay in centiseconds,
/// `scale` x nearest. Same palette rules as [`gif()`]. Every frame must hold `w * h * 4` bytes; a frame without its own
/// delay reuses the last one (10 cs when `delays_cs` is empty); delays are at least 2 cs.
pub fn gif_rgba(frames: &[Vec<u8>], w: usize, h: usize, delays_cs: &[u16], scale: u32, looping: bool, path: &Path) -> Result<()> {
    if frames.is_empty() {
        bail!("nothing to export");
    }
    if w == 0 || h == 0 || scale == 0 {
        bail!("empty picture size {w}x{h} at scale {scale}");
    }
    if let Some(i) = frames.iter().position(|f| f.len() != w * h * 4) {
        bail!("frame {i} has {} bytes, a {w}x{h} RGBA picture needs {}", frames[i].len(), w * h * 4);
    }
    let (sw, sh) = (w as u32 * scale, h as u32 * scale);
    if sw > u16::MAX as u32 || sh > u16::MAX as u32 {
        bail!("{sw}x{sh} is too large for a GIF");
    }
    let mut colours: Vec<[u8; 3]> = vec![[0, 0, 0]];
    let mut index: HashMap<[u8; 3], u8> = HashMap::new();
    let mut indexed: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    for rgba in frames {
        let mut buf = vec![0u8; (sw * sh) as usize];
        for y in 0..h as u32 {
            for x in 0..w as u32 {
                let i = ((y * w as u32 + x) * 4) as usize;
                if rgba[i + 3] == 0 {
                    continue;
                }
                let c = [rgba[i], rgba[i + 1], rgba[i + 2]];
                let k = match index.get(&c) {
                    Some(&k) => k,
                    None if colours.len() < 256 => {
                        let k = colours.len() as u8;
                        colours.push(c);
                        index.insert(c, k);
                        k
                    }
                    None => nearest(&colours, c),
                };
                for dy in 0..scale {
                    for dx in 0..scale {
                        buf[((y * scale + dy) * sw + x * scale + dx) as usize] = k;
                    }
                }
            }
        }
        indexed.push(buf);
    }
    let flat: Vec<u8> = colours.iter().flatten().copied().collect();
    let mut enc = gif::Encoder::new(BufWriter::new(File::create(path)?), sw as u16, sh as u16, &flat)?;
    if looping {
        enc.set_repeat(gif::Repeat::Infinite)?;
    }
    for (i, buf) in indexed.into_iter().enumerate() {
        let mut fr = gif::Frame::from_indexed_pixels(sw as u16, sh as u16, buf, Some(0));
        fr.delay = delays_cs.get(i).or(delays_cs.last()).copied().unwrap_or(10).max(2);
        fr.dispose = gif::DisposalMethod::Background;
        enc.write_frame(&fr)?;
    }
    Ok(())
}

fn nearest(pal: &[[u8; 3]], c: [u8; 3]) -> u8 {
    let d = |p: &[u8; 3]| (0..3).map(|i| (p[i] as i32 - c[i] as i32).pow(2)).sum::<i32>();
    pal.iter().enumerate().skip(1).min_by_key(|(_, p)| d(p)).map(|(i, _)| i as u8).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_gif(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("animator_ui_test_{}_{name}.gif", std::process::id()))
    }

    #[test]
    fn gif_rgba_rejects_a_frame_of_the_wrong_length() {
        let err = gif_rgba(&[vec![0u8; 4 * 4 * 4], vec![0u8; 10]], 4, 4, &[10], 1, true, &temp_gif("short")).unwrap_err();
        assert!(err.to_string().contains("frame 1"), "{err}");
    }

    #[test]
    fn gif_rgba_rejects_empty_sizes_and_no_frames() {
        assert!(gif_rgba(&[], 4, 4, &[10], 1, true, &temp_gif("none")).is_err());
        assert!(gif_rgba(&[vec![0u8; 16]], 2, 2, &[10], 0, true, &temp_gif("scale0")).is_err());
        assert!(gif_rgba(&[vec![]], 0, 2, &[10], 1, true, &temp_gif("w0")).is_err());
    }

    #[test]
    fn gif_rgba_writes_a_gif_with_a_missing_delay_reusing_the_last() {
        let path = temp_gif("ok");
        let opaque_red = [255u8, 0, 0, 255].repeat(4);
        gif_rgba(&[opaque_red.clone(), opaque_red], 2, 2, &[7], 3, false, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(&bytes[..6], b"GIF89a");
        // logical screen size = 2 * 3 pixels each way
        assert_eq!(u16::from_le_bytes([bytes[6], bytes[7]]), 6);
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 6);
    }
}
