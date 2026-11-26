//! Free helpers of the app: textures, small drawing utilities, the file picker and the demo silhouettes.

use crate::*;

/// The animation note, shortened to what fits the Preview header
/// ("model anim39 · T 0.80 · seed 1 · ref walk front").
pub(crate) fn note_short(note: &str) -> String {
    // "anim39: walk at 270°, 8 of 8 frames, 32px, T 0.80, seed 1, ref walk front"
    //   -> "model anim39 · T 0.80 · seed 1 · ref walk front"
    let mut parts: Vec<String> = Vec::new();
    for seg in note.split(',') {
        let s = seg.trim();
        match s.split_once(':') {
            Some((head, tail)) => {
                let head = head.trim();
                if head.to_lowercase().contains("model") || head.to_lowercase().contains("anim39") {
                    parts.push(head.to_string());
                }
                if tail.trim().starts_with('T') {
                    parts.push(tail.trim().to_string());
                }
            }
            None => {
                if s.starts_with('T') || s.starts_with("seed") || s.starts_with("ref") {
                    parts.push(s.to_string());
                }
            }
        }
    }
    if parts.is_empty() { note.to_string() } else { parts.join(" · ") }
}


/// Silhouettes for the scripted demo: "potion" (cork, neck, round body) or "disc".
pub(crate) fn demo_mask(name: &str, s: usize) -> Vec<u8> {
    let cx = (s as f64 - 1.0) / 2.0;
    (0..s * s)
        .map(|p| {
            let (u, v) = (((p % s) as f64 - cx) / s as f64, ((p / s) as f64 + 0.5) / s as f64);
            let on = match name {
                "disc" => u.hypot(v - 0.5) < 0.4,
                _ if v > 0.12 && v < 0.22 => u.abs() < 0.1,
                _ if (0.22..0.42).contains(&v) => u.abs() < 0.08,
                _ if (0.42..0.92).contains(&v) => u.hypot((v - 0.66) * 1.05) < 0.30,
                _ => false,
            };
            u8::from(on)
        })
        .collect()
}

/// The demo allows up to 32 sprites per run.
pub(crate) const MAX_COUNT: usize = 32;

/// "How many" until the user sets it: 8 at 16x16, 4 at 24 / 32 (the demo on a desktop).
pub(crate) fn default_count(size: usize) -> usize {
    if size == 16 { 8 } else { 4 }
}

/// A native file picker for the dropzone (falls back to the path field when the dialog fails).
/// Opens the Windows file dialog in a helper process and returns at once; the chosen path (or None when the
/// dialog was cancelled) arrives on the receiver. The dialog must not block the UI thread: the user may take as
/// long as they like.
pub(crate) fn pick_png(wake: impl Fn() + Send + 'static) -> std::sync::mpsc::Receiver<Option<PathBuf>> {
    let (tx, rx) = std::sync::mpsc::channel::<Option<PathBuf>>();
    std::thread::spawn(move || {
        use std::process::{Command, Stdio};
        // a hidden top-most owner keeps the dialog in front of the app window
        let ps = "Add-Type -AssemblyName System.Windows.Forms;                   $owner = New-Object System.Windows.Forms.Form -Property @{TopMost = $true; ShowInTaskbar = $false; Opacity = 0};                   $o = New-Object System.Windows.Forms.OpenFileDialog;                   $o.Filter = 'PNG images|*.png'; $o.Title = 'Pick a sprite PNG';                   [Console]::OutputEncoding = [System.Text.Encoding]::UTF8;                   if ($o.ShowDialog($owner) -eq 'OK') { Write-Output $o.FileName }";
        let out = Command::new("powershell").args(["-NoProfile", "-STA", "-Command", ps]).creation_flags(0x0800_0000).stdin(Stdio::null()).stdout(Stdio::piped()).output();
        let _ = tx.send(match out {
            Ok(o) if o.status.success() => Some(PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string())).filter(|p| p.is_file()),
            _ => None,
        });
        wake();
    });
    rx
}

/// The demo's small variant diagrams (mix.js `icon`): the first object in the accent colour, the second in grey.
pub(crate) fn paint_variant_icon(ui: &egui::Ui, r: Rect, kind: VariantKind, active: bool) {
    let p = ui.painter().clone();
    let a = if active { theme::ACCENT } else { Color32::from_gray(150) };
    let b = Color32::from_gray(if active { 110 } else { 80 });
    let band = |y0: f32, y1: f32, c: Color32| p.rect_filled(Rect::from_min_max(pos2(r.min.x, r.min.y + y0 * r.height()), pos2(r.max.x, r.min.y + y1 * r.height())), CornerRadius::ZERO, c);
    match kind {
        VariantKind::Inside => {
            band(0.0, 1.0 / 3.0, b);
            band(1.0 / 3.0, 2.0 / 3.0, a);
            band(2.0 / 3.0, 1.0, b);
        }
        VariantKind::Over => {
            band(0.0, 0.5, a);
            band(0.5, 1.0, b);
        }
        VariantKind::Centre => {
            band(0.0, 1.0, b);
            p.circle_filled(r.center(), r.width() * 0.27, a);
        }
        VariantKind::Blend => {
            band(0.0, 1.0, b);
            p.add(egui::Shape::convex_polygon(vec![r.left_bottom(), r.right_top(), pos2(r.max.x, r.min.y + r.height() / 3.0), pos2(r.min.x + r.width() / 3.0, r.max.y)], a, Stroke::NONE));
            p.add(egui::Shape::convex_polygon(vec![pos2(r.min.x, r.min.y + r.height() / 3.0), pos2(r.min.x + r.width() / 3.0, r.min.y), pos2(r.min.x + 2.0 * r.width() / 3.0, r.min.y), pos2(r.min.x, r.min.y + 2.0 * r.height() / 3.0)], a, Stroke::NONE));
        }
        VariantKind::Caption => {
            p.rect_stroke(r.shrink(1.0), CornerRadius::same(2), Stroke::new(1.0, a), StrokeKind::Inside);
            p.line_segment([pos2(r.min.x + 3.0, r.center().y - 1.0), pos2(r.max.x - 3.0, r.center().y - 1.0)], Stroke::new(1.0, a));
            p.line_segment([pos2(r.min.x + 3.0, r.center().y + 1.5), pos2(r.center().x + 1.0, r.center().y + 1.5)], Stroke::new(1.0, a));
        }
    }
}

pub(crate) fn color_image(s: &Sprite) -> ColorImage {
    ColorImage::from_rgba_unmultiplied([s.size, s.size], &s.rgba())
}

pub(crate) fn texture(ctx: &egui::Context, name: &str, s: &Sprite) -> TextureHandle {
    ctx.load_texture(name, color_image(s), TextureOptions::NEAREST)
}

pub(crate) fn rand_u32() -> u32 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let mut x = (t as u64) ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x as u32
}

/// A row of chips that wraps at the panel's width. egui's `horizontal_wrapped` does not wrap inside the scroll
/// panels (their content ui is unbounded), so the wrap width is pinned explicitly.
pub(crate) fn wrapped_row<R>(ui: &mut egui::Ui, f: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let w = ui.available_width().clamp(120.0, 2000.0);
    ui.allocate_ui_with_layout(vec2(w, 0.0), egui::Layout::left_to_right(egui::Align::Center).with_main_wrap(true), f)
        .inner
}
