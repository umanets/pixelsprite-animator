use eframe::egui::{self, Align, Color32, CornerRadius, Layout, Rect, Sense, Stroke, StrokeKind, Ui, Vec2, pos2, vec2};

use crate::theme::{
    ACCENT, ACCENT_SELECT, BG_INPUT, BG_RAISED, BORDER_STRONG, BORDER_SUBTLE, ON_ACCENT, ON_SELECT, R_CARD, R_FIELD,
    TEXT_MUTED, TEXT_PRIMARY, font_med, font_mono, font_semi,
};
use super::*;

// ------------------------------------------------------------------------------------------------------------- cards
/// A raised card (radius 12, subtle border) with the mock's 12 px inner padding.
pub fn card<R>(ui: &mut Ui, f: impl FnOnce(&mut Ui) -> R) -> R {
    egui::Frame::NONE
        .fill(BG_RAISED)
        .stroke(Stroke::new(1.0, BORDER_SUBTLE))
        .corner_radius(R_CARD)
        .inner_margin(egui::Margin::same(12))
        .show(ui, f)
        .inner
}

/// A card with a fixed outer size; the contents get the 12 px padding.
pub fn card_sized<R>(ui: &mut Ui, w: f32, h: f32, f: impl FnOnce(&mut Ui) -> R) -> (Rect, R) {
    let (rect, _) = ui.allocate_exact_size(vec2(w, h), Sense::hover());
    ui.painter().rect(rect, R_CARD, BG_RAISED, Stroke::new(1.0, BORDER_SUBTLE), StrokeKind::Inside);
    let inner = rect.shrink(12.0);
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(inner).layout(Layout::left_to_right(Align::Min)));
    let out = f(&mut child);
    (rect, out)
}

/// The "Build everything …" option card: accent checkbox, wrapping title, mono subtitle. 70 px tall in the mock.
pub fn checkbox_card(ui: &mut Ui, checked: &mut bool, title: &str, sub: &str, disabled_hint: Option<&str>) -> bool {
    let enabled = disabled_hint.is_none();
    let w = ui.available_width().clamp(120.0, 2000.0);
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 70.0), Sense::click());
    let resp = if enabled {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        resp.on_disabled_hover_text(disabled_hint.unwrap_or_default())
    };
    let p = ui.painter();
    p.rect(rect, R_FIELD, BG_RAISED, Stroke::new(1.0, BORDER_SUBTLE), StrokeKind::Inside);
    // checkbox at 13, 11
    let cb = Rect::from_min_size(pos2(rect.left() + 13.0, rect.top() + 11.0), Vec2::splat(16.0));
    let (fill, stroke) = if *checked && enabled { (ACCENT, ACCENT) } else { (BG_INPUT, BORDER_STRONG) };
    p.rect(cb, CornerRadius::same(4), fill, Stroke::new(1.0, stroke), StrokeKind::Inside);
    if *checked && enabled {
        check_mark(p, cb.shrink(3.0), ON_ACCENT);
    }
    let text_x = rect.left() + 39.0;
    let text_w = rect.right() - text_x - 12.0;
    let title_color = if enabled { TEXT_PRIMARY } else { TEXT_MUTED };
    let title_gal = p.layout(title.to_owned(), font_med(12.5), title_color, text_w.max(40.0));
    p.galley(pos2(text_x, rect.top() + 9.0), title_gal, Color32::WHITE);
    let sub_gal = p.layout_no_wrap(sub.to_owned(), font_mono(11.0), TEXT_MUTED);
    p.galley(pos2(text_x, rect.top() + 45.0), sub_gal, Color32::WHITE);
    if enabled && resp.clicked() {
        *checked = !*checked;
        true
    } else {
        false
    }
}

/// Mutes a whole area while a generation runs: no interaction, the mock's ~45 % dim.
pub fn dimmed<R>(ui: &mut Ui, f: impl FnOnce(&mut Ui) -> R) -> R {
    let out = ui.scope(|ui| {
        ui.disable();
        f(ui)
    })
    .inner;
    let rect = ui.min_rect();
    ui.painter().rect_filled(rect, R_CARD, Color32::from_rgba_unmultiplied(0x0D, 0x0E, 0x13, 115));
    out
}

/// The amber frame of the selected gallery card / frame strip: 2 px.
pub fn select_frame(ui: &Ui, rect: Rect, radius: CornerRadius) {
    ui.painter().rect_stroke(rect, radius, Stroke::new(2.0, ACCENT_SELECT), StrokeKind::Inside);
}

/// A subtle 1 px frame for unselected tiles.
pub fn tile_frame(ui: &Ui, rect: Rect) {
    ui.painter().rect_stroke(rect, CornerRadius::same(4), Stroke::new(1.0, BORDER_SUBTLE), StrokeKind::Inside);
}

/// A tag on a filled background, laid out inside `rect` (the gallery's «selected»).
pub fn tag_at(ui: &Ui, rect: Rect, text: &str, fill: Color32, fg: Color32, radius: CornerRadius) {
    ui.painter().rect_filled(rect, radius, fill);
    let gal = ui.painter().layout_no_wrap(text.to_owned(), font_semi(10.5), fg);
    ui.painter().galley(pos2(rect.center().x - gal.size().x / 2.0, rect.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
}

/// The `on` colour of the selected tag's text.
pub const TAG_TEXT: Color32 = ON_SELECT;

