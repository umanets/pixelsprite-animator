use std::ops::RangeInclusive;

use eframe::egui::{self, Align, Color32, CornerRadius, Layout, Rect, Response, Sense, Stroke, StrokeKind, Ui, Vec2, pos2, vec2};

use crate::theme::{ACCENT, BG_INPUT, BORDER_STRONG, ON_ACCENT, TEXT_PRIMARY, TEXT_SECONDARY, font};
use super::*;

// ----------------------------------------------------------------------------------------------------------- checkbox
/// A 16 px checkbox with a 12.5 px label, as in the mock. Returns true when clicked.
pub fn checkbox(ui: &mut Ui, checked: &mut bool, label: &str) -> bool {
    let label_gal = ui.painter().layout_no_wrap(label.to_owned(), font(12.5), TEXT_SECONDARY);
    let h = label_gal.size().y.max(16.0);
    let w = 16.0 + 6.0 + label_gal.size().x;
    let (rect, resp) = ui.allocate_exact_size(vec2(w, h), Sense::click());
    let cb = Rect::from_min_size(pos2(rect.left(), rect.center().y - 8.0), Vec2::splat(16.0));
    let p = ui.painter();
    let (fill, stroke) = if *checked { (ACCENT, ACCENT) } else { (BG_INPUT, BORDER_STRONG) };
    p.rect(cb, CornerRadius::same(4), fill, Stroke::new(1.0, stroke), StrokeKind::Inside);
    if *checked {
        check_mark(p, cb.shrink(3.0), ON_ACCENT);
    }
    p.galley(pos2(cb.right() + 6.0, rect.center().y - label_gal.size().y / 2.0), label_gal, Color32::WHITE);
    if resp.clicked() {
        *checked = !*checked;
        true
    } else {
        false
    }
}

/// A toggle switch of the mock: 34 × 20, radius 10, a 16 px white knob.
pub fn switch(ui: &mut Ui, on: &mut bool) -> Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(34.0, 20.0), Sense::click());
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(10), if *on { ACCENT } else { BORDER_STRONG });
    let cx = if *on { rect.right() - 10.0 } else { rect.left() + 10.0 };
    p.circle_filled(pos2(cx, rect.center().y), 8.0, Color32::WHITE);
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    if resp.clicked() {
        *on = !*on;
    }
    resp
}

// ------------------------------------------------------------------------------------------------------------ slider
/// The mock's slider: a 100 px rail (4 px, radius 2) with an accent fill and a 14 px white knob. `h` is the row's
/// height (16). Returns true while the value changes.
pub fn slider(ui: &mut Ui, value: &mut f32, range: RangeInclusive<f32>, w: f32) -> bool {
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 16.0), Sense::click_and_drag());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    let (lo, hi) = (*range.start(), *range.end());
    let mut v = *value;
    if (resp.dragged() || resp.clicked()) && hi > lo {
        if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
            let t = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            v = lo + t * (hi - lo);
        }
    }
    let changed = v != *value;
    *value = v;
    let t = if hi > lo { ((v - lo) / (hi - lo)).clamp(0.0, 1.0) } else { 0.0 };
    let p = ui.painter();
    let rail = Rect::from_min_size(pos2(rect.left(), rect.center().y - 2.0), vec2(rect.width(), 4.0));
    p.rect_filled(rail, CornerRadius::same(2), BORDER_STRONG);
    let filled = Rect::from_min_size(rail.min, vec2(rail.width() * t, 4.0));
    p.rect_filled(filled, CornerRadius::same(2), ACCENT);
    let knob = pos2(rect.left() + (rect.width() - 14.0) * t + 7.0, rect.center().y);
    p.circle(knob, 7.0, Color32::WHITE, Stroke::new(1.5, ACCENT));
    changed
}

/// A labelled slider row: "Frames" on the left, the value in mono on the right, then the rail.
pub fn slider_row(ui: &mut Ui, label: &str, value: &mut f32, range: RangeInclusive<f32>, w: f32, decimals: usize) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.add(egui::Label::new(txt(label, 12.5).color(TEXT_SECONDARY)).selectable(false));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let shown = if decimals == 0 { format!("{:.0}", value) } else { format!("{value:.decimals$}") };
            ui.add(egui::Label::new(mono_med_txt(shown, 12.0).color(TEXT_PRIMARY)).selectable(false));
        });
    });
    changed |= slider(ui, value, range, w);
    changed
}

