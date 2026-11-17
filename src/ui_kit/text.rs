use eframe::egui::RichText;

use crate::theme::{TEXT_MUTED, font, font_med, font_mono, font_mono_med, font_semi};

// -------------------------------------------------------------------------------------------------------------------- text
/// Geist Regular at the mock's size.
pub fn txt(text: impl Into<String>, size: f32) -> RichText {
    RichText::new(text).font(font(size))
}

/// Geist Medium at the mock's size.
pub fn med(text: impl Into<String>, size: f32) -> RichText {
    RichText::new(text).font(font_med(size))
}

/// Geist SemiBold at the mock's size.
pub fn semi_txt(text: impl Into<String>, size: f32) -> RichText {
    RichText::new(text).font(font_semi(size))
}

/// Geist Mono at the mock's size (numbers, ids, timings, seeds, fps).
pub fn mono_txt(text: impl Into<String>, size: f32) -> RichText {
    RichText::new(text).font(font_mono(size))
}

/// Geist Mono Medium at the mock's size.
pub fn mono_med_txt(text: impl Into<String>, size: f32) -> RichText {
    RichText::new(text).font(font_mono_med(size))
}

/// A small mono label in the muted colour (meta lines).
pub fn meta(text: impl Into<String>) -> RichText {
    mono_txt(text, 11.5).color(TEXT_MUTED)
}

