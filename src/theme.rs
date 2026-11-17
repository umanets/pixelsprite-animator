//! Design tokens of the Figma design ("Screens" page:
//! "Studio / Main" 1440×1060 and "Studio / Generating"), and the egui [`egui::Visuals`], style and fonts built from them.
//!
//! Every colour / radius / spacing / type size in the UI comes from here, never ad hoc. The numbers are the mock's
//! absolute ones (a 1440×1060 frame): top bar 52, left panel 320, inspector 340, dock 200, status bar 28; panel
//! content sits 16 px from the panel edge (17 in the inspector, 20 in the centre); cards are 12 px, the stage 14.
//!
//! Fonts: Geist (Regular / Medium / SemiBold) and Geist Mono (Regular / Medium), instanced from Google Fonts'
//! variable files into `fonts/` and embedded with `include_bytes!`. Ubuntu-Light (egui's default) stays in every
//! family after Geist as the Cyrillic / Greek fallback — Geist has no Cyrillic, and captions are EN / RU / UK.
//! Numbers, ids, timings, seeds and fps are always shown in Monospace (Geist Mono).

use eframe::egui::{
    self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Margin, Stroke, TextStyle, Vec2,
    Theme,
};

// ----------------------------------------------------------------------------------------------------- colours
pub const BG_APP: Color32 = Color32::from_rgb(0x0D, 0x0E, 0x13); // bg/app
pub const BG_PANEL: Color32 = Color32::from_rgb(0x14, 0x16, 0x1D); // bg/panel: top / side / status / dock
pub const BG_RAISED: Color32 = Color32::from_rgb(0x1B, 0x1E, 0x27); // bg/raised: cards, buttons, chips
pub const BG_INPUT: Color32 = Color32::from_rgb(0x0F, 0x11, 0x16); // bg/input: fields, wells, tracks
pub const BORDER_SUBTLE: Color32 = Color32::from_rgb(0x23, 0x26, 0x34); // border/subtle
pub const BORDER_STRONG: Color32 = Color32::from_rgb(0x33, 0x38, 0x49); // border/strong
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xEC, 0xEE, 0xF5); // text/primary
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(0xA5, 0xAB, 0xC0); // text/secondary
pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x6B, 0x71, 0x88); // text/muted
pub const ACCENT: Color32 = Color32::from_rgb(0x7C, 0x6B, 0xFF); // accent/primary
pub const ACCENT_SOFT: Color32 = Color32::from_rgb(0x27, 0x22, 0x50); // accent/primary-soft
pub const ACCENT_SELECT: Color32 = Color32::from_rgb(0xFF, 0xC2, 0x47); // amber: selection
pub const ON_ACCENT: Color32 = Color32::from_rgb(0xFF, 0xFF, 0xFF);
pub const SUCCESS: Color32 = Color32::from_rgb(0x4A, 0xDE, 0x80); // status/success
pub const SUCCESS_SOFT: Color32 = Color32::from_rgb(0x15, 0x30, 0x1F); // status/success-soft
pub const WARN: Color32 = Color32::from_rgb(0xFB, 0xBF, 0x24); // status/warn
pub const DANGER: Color32 = Color32::from_rgb(0xF8, 0x71, 0x71); // status/danger
pub const CHECK_A: Color32 = Color32::from_rgb(0x1E, 0x21, 0x2B); // canvas/check-a
pub const CHECK_B: Color32 = Color32::from_rgb(0x25, 0x29, 0x36); // canvas/check-b
/// The near-black text on the amber "selected" tag.
pub const ON_SELECT: Color32 = Color32::from_rgb(0x2B, 0x21, 0x00);
/// The blue frame of a "one shape" run's source sprite (not a token of the mock, kept from the old build).
pub const SOURCE_BLUE: Color32 = Color32::from_rgb(0x6E, 0x96, 0xF0);

/// Lighten `c` towards white by `f` (0..1); hover / active states of raised surfaces.
pub fn lighten(c: Color32, f: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(
        c.r() + ((255 - c.r()) as f32 * f) as u8,
        c.g() + ((255 - c.g()) as f32 * f) as u8,
        c.b() + ((255 - c.b()) as f32 * f) as u8,
        c.a(),
    )
}

// ----------------------------------------------------------------------------------------------------- metrics
/// Radii of the mock: 6 (step chips, badges, segments, frame strip) / 8 (buttons, fields, chips, dropdowns,
/// compass cells) / 9 (segmented track) / 10 (textarea, dropzone, option card, clips track) / 12 (cards) /
/// 14 (stage) / 18 (the round play button).
pub const R_CHIP: CornerRadius = CornerRadius::same(6);
pub const R_CTRL: CornerRadius = CornerRadius::same(8);
pub const R_TRACK: CornerRadius = CornerRadius::same(9);
pub const R_FIELD: CornerRadius = CornerRadius::same(10);
pub const R_CARD: CornerRadius = CornerRadius::same(12);
pub const R_PANEL: CornerRadius = CornerRadius::same(14);

/// Spacing grid: 4 / 8 / 12 / 16 / 24.
pub const SPACE: f32 = 4.0;

/// Heights of the mock: top bar 52, status bar 28, dock 200, controls 32, chips 24, action chips 26,
/// the Generate / Stop button 44, panels 320 (left) / 340 (inspector).
pub const TOP_BAR_H: f32 = 52.0;
pub const STATUS_H: f32 = 28.0;
pub const DOCK_H: f32 = 200.0;
pub const CTRL_H: f32 = 32.0;
pub const CHIP_H: f32 = 24.0;
pub const ACTION_CHIP_H: f32 = 26.0;
pub const GEN_H: f32 = 44.0;
pub const LEFT_W: f32 = 320.0;
pub const RIGHT_W: f32 = 340.0;

/// Padding inside the panels: content starts 16 px from the left / right edge (17 in the inspector), the centre
/// column 20 px from the sides of its 780 px area.
pub const PAD: f32 = 16.0;
pub const PAD_INSPECTOR: f32 = 17.0;
pub const PAD_CENTRE: f32 = 20.0;
/// Vertical gaps of the mock.
pub const GAP_XS: f32 = 4.0;
pub const GAP_SM: f32 = 6.0;
pub const GAP: f32 = 8.0;
pub const GAP_MD: f32 = 12.0;
pub const GAP_LG: f32 = 16.0;

// ------------------------------------------------------------------------------------------------------ fonts
pub const FONT_REGULAR: &str = "Geist";
pub const FONT_MEDIUM: &str = "GeistMedium";
pub const FONT_SEMIBOLD: &str = "GeistSemiBold";
pub const FONT_MONO: &str = "GeistMono";
pub const FONT_MONO_MEDIUM: &str = "GeistMonoMedium";

/// The five embedded faces; `(name, bytes)`.
const FONTS: [(&str, &[u8]); 5] = [
    (FONT_REGULAR, include_bytes!("../fonts/Geist-Regular.ttf")),
    (FONT_MEDIUM, include_bytes!("../fonts/Geist-Medium.ttf")),
    (FONT_SEMIBOLD, include_bytes!("../fonts/Geist-SemiBold.ttf")),
    (FONT_MONO, include_bytes!("../fonts/GeistMono-Regular.ttf")),
    (FONT_MONO_MEDIUM, include_bytes!("../fonts/GeistMono-Medium.ttf")),
];

fn font_definitions() -> FontDefinitions {
    let mut defs = FontDefinitions::default();
    for (name, bytes) in FONTS {
        defs.font_data.insert(name.to_owned(), std::sync::Arc::new(FontData::from_static(bytes)));
    }
    // Geist first, then egui's defaults (Ubuntu-Light keeps Cyrillic; the emoji fonts stay).
    let prepend = |defs: &mut FontDefinitions, family: FontFamily, faces: &[&str]| {
        let list = defs.families.entry(family).or_default();
        for face in faces {
            list.retain(|f| f != face);
            list.insert(0, (*face).to_owned());
        }
    };
    // Named families get the *whole* proportional list behind the weight face, so any glyph the weight lacks
    // (Cyrillic, Greek, emoji) falls back instead of rendering as a box.
    let fallback: Vec<String> = defs
        .families
        .get(&FontFamily::Proportional)
        .cloned()
        .unwrap_or_else(|| vec![FONT_REGULAR.to_owned()]);
    for (name, face) in [(FONT_MEDIUM, FONT_MEDIUM), (FONT_SEMIBOLD, FONT_SEMIBOLD)] {
        let mut list = vec![face.to_owned()];
        list.extend(fallback.iter().filter(|f| f.as_str() != face).cloned());
        defs.families.insert(FontFamily::Name(name.into()), list);
    }
    let mono_fallback: Vec<String> = defs
        .families
        .get(&FontFamily::Monospace)
        .cloned()
        .unwrap_or_else(|| vec![FONT_MONO.to_owned()]);
    let mut mono_list = vec![FONT_MONO_MEDIUM.to_owned()];
    mono_list.extend(mono_fallback.iter().filter(|f| f.as_str() != FONT_MONO_MEDIUM).cloned());
    defs.families.insert(FontFamily::Name(FONT_MONO_MEDIUM.into()), mono_list);
    prepend(&mut defs, FontFamily::Proportional, &[FONT_REGULAR]);
    prepend(&mut defs, FontFamily::Monospace, &[FONT_MONO]);
    defs
}

// -------------------------------------------------------------------------------------------------- type scale
/// Geist Regular at `size` (body text of the mock: 11 / 11.5 / 12 / 12.5 / 13 / 15 / 18).
pub fn font(size: f32) -> FontId {
    FontId::new(size, FontFamily::Proportional)
}

/// Geist Medium at `size` (buttons, chips, labels, small titles).
pub fn font_med(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(FONT_MEDIUM.into()))
}

/// Geist SemiBold at `size` (the brand, the gallery caption, the "selected" tag).
pub fn font_semi(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(FONT_SEMIBOLD.into()))
}

/// Geist Mono Regular at `size` (numbers, ids, timings, seeds, fps: 10 … 12).
pub fn font_mono(size: f32) -> FontId {
    FontId::new(size, FontFamily::Monospace)
}

/// Geist Mono Medium at `size` (segment labels, step digits).
pub fn font_mono_med(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(FONT_MONO_MEDIUM.into()))
}

/// The mock's type scale as egui text styles (so `.small()`, `.body()` and friends stay on scale).
fn text_styles() -> Vec<(TextStyle, FontId)> {
    vec![
        (TextStyle::Heading, font_semi(18.0)),
        (TextStyle::Body, font(13.0)),
        (TextStyle::Button, font_med(13.0)),
        (TextStyle::Small, font(12.0)),
        (TextStyle::Monospace, font_mono(12.0)),
        (TextStyle::Name("medium".into()), font_med(13.0)),
        (TextStyle::Name("semibold".into()), font_semi(13.0)),
        (TextStyle::Name("mono-medium".into()), font_mono_med(12.0)),
    ]
}

/// Applies the dark theme (colours, radii, spacing, fonts) to the context. Call once at startup.
pub fn apply(ctx: &egui::Context) {
    let mut style = (*ctx.style_of(Theme::Dark)).clone();
    let v = &mut style.visuals;
    v.dark_mode = true;
    v.panel_fill = BG_APP;
    v.window_fill = BG_PANEL;
    v.extreme_bg_color = BG_INPUT;
    v.faint_bg_color = BG_INPUT;
    v.hyperlink_color = ACCENT;
    v.window_stroke = Stroke::new(1.0, BORDER_STRONG);
    v.window_corner_radius = R_PANEL;
    v.menu_corner_radius = R_CTRL;
    v.selection.bg_fill = ACCENT_SOFT;
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.warn_fg_color = WARN;
    v.error_fg_color = DANGER;
    // widget states: raised surfaces, subtle borders, no egui-blue
    let mut w = v.widgets.clone();
    w.noninteractive.bg_fill = BG_PANEL;
    w.noninteractive.weak_bg_fill = BG_APP;
    w.noninteractive.bg_stroke = Stroke::new(1.0, BORDER_SUBTLE);
    w.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    w.noninteractive.corner_radius = R_CTRL;
    w.inactive.bg_fill = BG_RAISED;
    w.inactive.weak_bg_fill = BG_RAISED;
    w.inactive.bg_stroke = Stroke::new(1.0, BORDER_STRONG);
    w.inactive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    w.inactive.corner_radius = R_CTRL;
    w.hovered.bg_fill = lighten(BG_RAISED, 0.05);
    w.hovered.weak_bg_fill = lighten(BG_RAISED, 0.05);
    w.hovered.bg_stroke = Stroke::new(1.0, BORDER_STRONG);
    w.hovered.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    w.hovered.corner_radius = R_CTRL;
    w.active.bg_fill = lighten(BG_RAISED, 0.09);
    w.active.weak_bg_fill = lighten(BG_RAISED, 0.09);
    w.active.bg_stroke = Stroke::new(1.0, BORDER_STRONG);
    w.active.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    w.active.corner_radius = R_CTRL;
    w.open.bg_fill = BG_INPUT;
    w.open.weak_bg_fill = BG_INPUT;
    w.open.bg_stroke = Stroke::new(1.0, BORDER_STRONG);
    w.open.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    w.open.corner_radius = R_CTRL;
    v.widgets = w;
    // spacing on the 4/8 grid: 32 px controls, 8 px item gaps, 14x6 button padding
    let s = &mut style.spacing;
    s.item_spacing = Vec2::new(GAP, GAP);
    s.button_padding = Vec2::new(14.0, 6.0);
    s.interact_size = Vec2::new(40.0, CTRL_H);
    s.slider_width = 100.0;
    s.window_margin = Margin::same(12);
    s.menu_margin = Margin::same(8);
    s.combo_width = 200.0;
    s.text_edit_width = 200.0;
    style.text_styles = text_styles().into_iter().collect();
    ctx.set_style_of(Theme::Dark, style);
    ctx.set_fonts(font_definitions());
}
