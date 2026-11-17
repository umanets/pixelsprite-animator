use eframe::egui::{self, Color32, CornerRadius, Rect, Response, RichText, Sense, Stroke, StrokeKind, Ui, Vec2, pos2, vec2};

use crate::theme::{
    ACCENT, ACCENT_SOFT, ACTION_CHIP_H, BG_INPUT, BG_RAISED, BORDER_SUBTLE, CHIP_H, DANGER, ON_ACCENT, R_CHIP, R_CTRL,
    R_TRACK, SUCCESS, SUCCESS_SOFT, TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY, WARN, font_med, font_mono_med, lighten,
};
use super::*;

// ------------------------------------------------------------------------------------------------------------------- chips
/// A chip of the mock (24 px, radius 8, 10 px side padding): raised when off, accent-soft + accent border when on.
pub fn chip_h(ui: &mut Ui, label: impl Into<String>, active: bool, h: f32) -> Response {
    let text = if active {
        med(label.into(), 12.5).color(TEXT_PRIMARY)
    } else {
        txt(label.into(), 12.5).color(TEXT_SECONDARY)
    };
    chip_rt(ui, text, active, h)
}

/// A chip with a prepared label (e.g. italics for an attribute word, a "?" marker for an unknown one).
pub fn chip_rt(ui: &mut Ui, text: RichText, active: bool, h: f32) -> Response {
    let (fill, border) = if active { (ACCENT_SOFT, ACCENT) } else { (BG_RAISED, BORDER_SUBTLE) };
    let saved = ui.style().clone();
    {
        let style = ui.style_mut();
        for state in [
            &mut style.visuals.widgets.inactive,
            &mut style.visuals.widgets.hovered,
            &mut style.visuals.widgets.active,
        ] {
            state.bg_fill = fill;
            state.weak_bg_fill = fill;
            state.bg_stroke = Stroke::new(1.0, border);
            state.fg_stroke = Stroke::new(1.0, if active { ACCENT } else { BORDER_SUBTLE });
            state.corner_radius = R_CTRL;
        }
        style.spacing.button_padding = Vec2::new(10.0, ((h - 16.0) / 2.0).max(2.0));
    }
    let r = ui.add(egui::Button::new(text).min_size(vec2(0.0, h)));
    ui.set_style(saved);
    r
}

/// A 24 px chip (the table's examples, the kind filter, the action chips' shorter siblings).
pub fn chip(ui: &mut Ui, label: &str, active: bool) -> Response {
    chip_h(ui, label, active, CHIP_H)
}

/// An action chip: 26 px, the same look.
pub fn action_chip(ui: &mut Ui, label: &str, active: bool) -> Response {
    chip_h(ui, label, active, ACTION_CHIP_H)
}

// -------------------------------------------------------------------------------------------------------- segmented
/// The mock's segmented control: an input-coloured track (radius 9, 4 px inset, 2 px gaps) whose active item is a
/// filled accent segment (radius 6, white label). Returns true when the selection changed.
pub fn segmented_h(ui: &mut Ui, items: &[&str], sel: &mut usize, h: f32, mono: bool) -> bool {
    let font = if mono { font_mono_med(12.5) } else { font_med(12.5) };
    let text_w = items
        .iter()
        .map(|i| ui.painter().layout_no_wrap((*i).to_owned(), font.clone(), TEXT_PRIMARY).size().x)
        .fold(0.0f32, f32::max);
    let item_w = (text_w + 24.0).ceil();
    let n = items.len().max(1) as f32;
    let width = item_w * n + 2.0 * (n - 1.0) + 8.0;
    let (rect, track) = ui.allocate_exact_size(vec2(width, h), Sense::click());
    let track = track.on_hover_cursor(egui::CursorIcon::PointingHand);
    let p = ui.painter();
    p.rect_filled(rect, R_TRACK, BG_INPUT);
    p.rect_stroke(rect, R_TRACK, Stroke::new(1.0, BORDER_SUBTLE), StrokeKind::Inside);
    let mut changed = false;
    let mut x = rect.left() + 4.0;
    let inner_h = h - 8.0;
    for (i, item) in items.iter().enumerate() {
        let seg = Rect::from_min_size(pos2(x, rect.top() + 4.0), vec2(item_w, inner_h));
        let active = i == *sel;
        let hovered = track.hovered() && ui.input(|s| s.pointer.latest_pos()).is_some_and(|q| seg.contains(q));
        if active {
            p.rect_filled(seg, R_CHIP, ACCENT);
        } else if hovered {
            p.rect_filled(seg, R_CHIP, lighten(BG_INPUT, 0.06));
        }
        let color = if active { ON_ACCENT } else { TEXT_SECONDARY };
        let gal = p.layout_no_wrap((*item).to_owned(), font.clone(), color);
        p.galley(pos2(seg.center().x - gal.size().x / 2.0, seg.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
        if track.clicked() && ui.input(|s| s.pointer.interact_pos()).is_some_and(|q| seg.contains(q)) && !active {
            *sel = i;
            changed = true;
        }
        x = seg.right() + 2.0;
    }
    changed
}

/// The 34 px segmented control (Size, Silhouette), labels in Geist Mono like the mock.
pub fn segmented(ui: &mut Ui, items: &[&str], sel: &mut usize) -> bool {
    segmented_h(ui, items, sel, 34.0, true)
}

/// The 32 px segmented control (zoom ×1…×8, export scale), monospace labels.
pub fn segmented_small(ui: &mut Ui, items: &[&str], sel: &mut usize) -> bool {
    segmented_h(ui, items, sel, 32.0, true)
}

// ------------------------------------------------------------------------------------------------------------------ badges
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BadgeKind {
    Neutral,
    Accent,
    Success,
    Warn,
    Danger,
}

/// A 21 px label with a soft background (kind tags, HUD chips), radius 6.
pub fn badge(ui: &mut Ui, text: &str, kind: BadgeKind) -> Response {
    let (fg, bg) = match kind {
        BadgeKind::Neutral => (TEXT_SECONDARY, BG_RAISED),
        BadgeKind::Accent => (TEXT_PRIMARY, ACCENT_SOFT),
        BadgeKind::Success => (SUCCESS, SUCCESS_SOFT),
        BadgeKind::Warn => (WARN, BG_RAISED),
        BadgeKind::Danger => (DANGER, BG_RAISED),
    };
    let frame = egui::Frame::NONE
        .fill(bg)
        .inner_margin(egui::Margin { left: 8, right: 8, top: 3, bottom: 3 })
        .corner_radius(R_CHIP);
    frame.show(ui, |ui| ui.add(egui::Label::new(med(text, 11.5).color(fg)).selectable(false))).inner
}

/// The panels' section header: a 20 px accent-soft step chip with a mono digit, then the title in caps.
pub fn section_header(ui: &mut Ui, step: &str, title: &str) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(vec2(20.0, 20.0), Sense::hover());
        ui.painter().rect_filled(rect, R_CHIP, ACCENT_SOFT);
        let gal = ui.painter().layout_no_wrap(step.to_owned(), font_mono_med(11.0), TEXT_PRIMARY);
        ui.painter().galley(pos2(rect.center().x - gal.size().x / 2.0, rect.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
        ui.add_space(3.0);
        ui.add(egui::Label::new(semi_txt(title.to_uppercase(), 12.0).color(TEXT_SECONDARY)).selectable(false));
    });
}

/// A keyboard hint pill, e.g. "Ctrl ↵" inside the caption field.
pub fn kbd(ui: &mut Ui, text: &str) -> Response {
    let frame = egui::Frame::NONE
        .fill(Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 46))
        .inner_margin(egui::Margin { left: 6, right: 6, top: 2, bottom: 2 })
        .corner_radius(CornerRadius::same(5));
    frame.show(ui, |ui| ui.add(egui::Label::new(mono_txt(text, 11.0).color(ON_ACCENT)).selectable(false))).inner
}

/// A keyboard hint on a dark pill ("Esc" in the Stop button).
pub fn kbd_dark(ui: &mut Ui, text: &str) -> Response {
    let frame = egui::Frame::NONE
        .fill(BG_INPUT)
        .stroke(Stroke::new(1.0, BORDER_SUBTLE))
        .inner_margin(egui::Margin { left: 6, right: 6, top: 2, bottom: 2 })
        .corner_radius(CornerRadius::same(5));
    frame.show(ui, |ui| ui.add(egui::Label::new(mono_txt(text, 11.0).color(TEXT_MUTED)).selectable(false))).inner
}

pub fn kbd_hint(ui: &mut Ui, text: &str) {
    kbd_dark(ui, text);
}

/// A status dot (execution provider, ONNX animator): 8 px.
pub fn dot(ui: &mut Ui, color: Color32) {
    let (r, _) = ui.allocate_exact_size(vec2(8.0, 12.0), Sense::hover());
    ui.painter().circle_filled(r.center(), 4.0, color);
}

/// A small round dot used inline (7 px, the Animator dropdown).
pub fn dot_at(painter: &egui::Painter, center: egui::Pos2, d: f32, color: Color32) {
    painter.circle_filled(center, d / 2.0, color);
}

/// A painted check mark (the ✓ glyph is not in every Geist face): two strokes inside `rect`.
pub fn check_mark(p: &egui::Painter, rect: Rect, color: Color32) {
    let c = rect.center();
    let s = rect.width().min(rect.height()) * 0.30;
    let a = c + vec2(-s, 0.0);
    let b = c + vec2(-s * 0.25, s * 0.85);
    let d = c + vec2(s * 1.15, -s * 0.9);
    let w = (rect.width() * 0.13).max(1.4);
    p.line_segment([a, b], Stroke::new(w, color));
    p.line_segment([b, d], Stroke::new(w, color));
}
