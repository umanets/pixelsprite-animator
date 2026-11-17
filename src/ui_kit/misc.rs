use eframe::egui::{self, Align, Color32, CornerRadius, FontId, Layout, Rect, Response, RichText, Sense, Stroke, Ui, Vec2, pos2, vec2};

use crate::theme::{ACCENT, BG_APP, BG_INPUT, BORDER_STRONG, BORDER_SUBTLE, R_CTRL, R_FIELD, TEXT_MUTED, TEXT_PRIMARY, font};

// -------------------------------------------------------------------------------------------------------------- misc
/// A 1 px horizontal divider in the border colour.
pub fn divider(ui: &mut Ui) {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(w, 1.0), Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::ZERO, BORDER_SUBTLE);
}

/// A filled progress bar of the mock: 6 px tall, radius 3, accent fill.
pub fn progress(ui: &mut Ui, width: f32, frac: f32) {
    let w = width.max(40.0);
    let (rect, _) = ui.allocate_exact_size(vec2(w, 6.0), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(3), BORDER_SUBTLE);
    let f = Rect::from_min_size(rect.min, vec2(rect.width() * frac.clamp(0.0, 1.0), 6.0));
    p.rect_filled(f, CornerRadius::same(3), ACCENT);
}

/// The checkerboard behind sprites (canvas/check-a / check-b).
pub fn checker(ui: &Ui, rect: Rect, cell: f32) {
    let cell = cell.max(3.0);
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::ZERO, crate::theme::CHECK_A);
    let cols = (rect.width() / cell).ceil() as i32 + 1;
    let rows = (rect.height() / cell).ceil() as i32 + 1;
    for r in 0..rows {
        for c in 0..cols {
            if (r + c) % 2 == 0 {
                continue;
            }
            let min = pos2(rect.left() + c as f32 * cell, rect.top() + r as f32 * cell);
            let max = pos2((min.x + cell).min(rect.right()), (min.y + cell).min(rect.bottom()));
            if max.x > min.x && max.y > min.y {
                p.rect_filled(Rect::from_min_max(min, max), CornerRadius::ZERO, crate::theme::CHECK_B);
            }
        }
    }
}

/// A dashed rounded frame (the PNG dropzone).
pub fn dashed_rect(ui: &Ui, rect: Rect, color: Color32) {
    let p = ui.painter();
    p.rect_filled(rect, R_FIELD, Color32::TRANSPARENT);
    let step = 6.0;
    let r = 10.0;
    let (l, t, rr, b) = (rect.left(), rect.top(), rect.right(), rect.bottom());
    let dash = |a: egui::Pos2, b: egui::Pos2| {
        let n = ((a - b).length() / (step * 2.0)).ceil().max(1.0);
        for i in 0..n as i32 {
            let t0 = i as f32 / n;
            let t1 = (i as f32 + 0.5) / n;
            p.line_segment([a.lerp(b, t0), a.lerp(b, t1)], Stroke::new(1.0, color));
        }
    };
    dash(pos2(l + r, t), pos2(rr - r, t));
    dash(pos2(l + r, b), pos2(rr - r, b));
    dash(pos2(l, t + r), pos2(l, b - r));
    dash(pos2(rr, t + r), pos2(rr, b - r));
    // corners
    for (cx, cy, a0) in [(l + r, t + r, 180.0), (rr - r, t + r, 270.0), (rr - r, b - r, 0.0), (l + r, b - r, 90.0)] {
        let center = pos2(cx, cy);
        let n = 5;
        for i in 0..n {
            let a1 = (a0 + (i as f32) * 90.0 / n as f32).to_radians();
            let a2 = (a0 + (i as f32 + 0.5) * 90.0 / n as f32).to_radians();
            p.line_segment(
                [center + vec2(a1.cos(), a1.sin()) * r, center + vec2(a2.cos(), a2.sin()) * r],
                Stroke::new(1.0, color),
            );
        }
    }
}

/// A surface that looks like an input field (radius 8, input fill, strong border) around `f`.
pub fn field<R>(ui: &mut Ui, h: f32, f: impl FnOnce(&mut Ui) -> R) -> R {
    egui::Frame::NONE
        .fill(BG_INPUT)
        .stroke(Stroke::new(1.0, BORDER_STRONG))
        .corner_radius(R_CTRL)
        .inner_margin(egui::Margin { left: 11, right: 11, top: 0, bottom: 0 })
        .show(ui, |ui| {
            ui.set_height(h);
            ui.with_layout(Layout::left_to_right(Align::Center), f).inner
        })
        .inner
}

/// A text input styled like the mock's fields. Returns the response (for Enter handling).
pub fn text_input(ui: &mut Ui, text: &mut String, hint: &str, w: f32, h: f32) -> Response {
    let saved = ui.style().clone();
    {
        let style = ui.style_mut();
        for state in [
            &mut style.visuals.widgets.inactive,
            &mut style.visuals.widgets.hovered,
            &mut style.visuals.widgets.active,
            &mut style.visuals.widgets.open,
        ] {
            state.bg_fill = BG_INPUT;
            state.weak_bg_fill = BG_INPUT;
            state.bg_stroke = Stroke::new(1.0, BORDER_STRONG);
            state.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
            state.corner_radius = R_CTRL;
        }
        style.spacing.button_padding = Vec2::new(10.0, (h - 17.0) / 2.0);
    }
    let r = ui.add(
        egui::TextEdit::singleline(text)
            .hint_text(RichText::new(hint).font(font(12.5)).color(TEXT_MUTED))
            .font(font(13.0))
            .text_color(TEXT_PRIMARY)
            .desired_width(w)
            .margin(egui::Margin { left: 10, right: 10, top: ((h - 17.0) / 2.0) as i8, bottom: 0 })
            .align(egui::Align2::CENTER_CENTER),
    );
    ui.set_style(saved);
    r
}

/// An empty 32 px row reserved for a right-aligned group (used to line up panel rows).
pub fn spacer(ui: &mut Ui, w: f32) {
    ui.add_space(w);
}

/// Text drawn at a position with a painter (panels' meta lines that must not wrap).
pub fn text_at(ui: &Ui, pos: egui::Pos2, text: &str, font: FontId, color: Color32) {
    ui.painter().galley(pos, ui.painter().layout_no_wrap(text.to_owned(), font, color), Color32::WHITE);
}

/// The app's background for a panel that is not an egui `Panel` (the dock, the stage).
pub const SURFACE: Color32 = BG_APP;
