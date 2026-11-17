use eframe::egui::{self, Color32, Response, RichText, Stroke, Ui, Vec2, vec2};

use crate::theme::{ACCENT, ACCENT_SOFT, BG_RAISED, BORDER_STRONG, CTRL_H, DANGER, ON_ACCENT, R_CTRL, TEXT_PRIMARY, lighten};
use super::*;

// ----------------------------------------------------------------------------------------------------------------- buttons
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Btn {
    /// Accent fill, white label (Generate, Export, GIF, Play).
    Primary,
    /// Raised surface, strong border (Settings, Restart, Sheet PNG, Save all).
    Ghost,
    /// No fill, danger border and label (Stop).
    Danger,
    /// Accent-soft fill, accent border and label (a chosen option in a row).
    Soft,
}

fn button_impl(ui: &mut Ui, text: RichText, kind: Btn, h: f32, w: Option<f32>) -> Response {
    let (fill, border, fg) = match kind {
        Btn::Primary => (ACCENT, ACCENT, ON_ACCENT),
        Btn::Ghost => (BG_RAISED, BORDER_STRONG, TEXT_PRIMARY),
        Btn::Danger => (Color32::TRANSPARENT, DANGER, DANGER),
        Btn::Soft => (ACCENT_SOFT, ACCENT, TEXT_PRIMARY),
    };
    let saved = ui.style().clone();
    {
        let style = ui.style_mut();
        let hover_fill = match kind {
            Btn::Danger => Color32::from_rgba_unmultiplied(0xF8, 0x71, 0x71, 24),
            _ => lighten(fill, 0.06),
        };
        let press_fill = match kind {
            Btn::Danger => Color32::from_rgba_unmultiplied(0xF8, 0x71, 0x71, 40),
            _ => lighten(fill, 0.12),
        };
        for (state, f) in [
            (&mut style.visuals.widgets.inactive, fill),
            (&mut style.visuals.widgets.hovered, hover_fill),
            (&mut style.visuals.widgets.active, press_fill),
        ] {
            state.bg_fill = f;
            state.weak_bg_fill = f;
            state.bg_stroke = Stroke::new(1.0, border);
            state.fg_stroke = Stroke::new(1.0, fg);
            state.corner_radius = R_CTRL;
        }
        style.spacing.button_padding = Vec2::new(12.0, (h - 17.0).max(2.0) / 2.0);
    }
    let text = text.color(fg);
    let mut b = egui::Button::new(text).min_size(vec2(w.unwrap_or(0.0), h));
    if let Some(w) = w {
        b = b.min_size(vec2(w, h));
    }
    let r = ui.add(b);
    ui.set_style(saved);
    r
}

/// A button of the mock's height (32 by default) in one of its four styles.
pub fn button(ui: &mut Ui, label: impl Into<String>, kind: Btn) -> Response {
    let size = if kind == Btn::Primary { 13.0 } else { 13.0 };
    button_impl(ui, med(label, size), kind, CTRL_H, None)
}

/// A button with an explicit height (the 44 px Generate / Stop, the 28 px dock buttons).
pub fn button_h(ui: &mut Ui, label: impl Into<String>, kind: Btn, h: f32) -> Response {
    button_impl(ui, med(label, if h >= 40.0 { 14.0 } else { 13.0 }), kind, h, None)
}

/// A button with an explicit width and height (the top bar's Export, the gallery's Save all).
pub fn button_wh(ui: &mut Ui, label: impl Into<String>, kind: Btn, w: f32, h: f32) -> Response {
    button_impl(ui, med(label, 13.0), kind, h, Some(w))
}


