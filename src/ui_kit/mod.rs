//! UI components of the Figma mock, built on [`crate::theme`] tokens. Every control the panels show goes through
//! these instead of ad-hoc `ui.button` / `selectable_label`, so the app matches the design's metrics: 20 px step
//! chips, 24 / 26 px chips, 32 px buttons and fields, 44 px primary action, 6 / 8 / 9 / 10 / 12 / 14 px radii.
//!
//! Old call sites (Settings window, small hints) keep the named wrappers at the bottom (`primary_button`,
//! `secondary_button`, `medium`, `mono`, …).

mod buttons;
mod cards;
mod chips;
mod controls;
mod misc;
mod text;

pub use buttons::{Btn, button, button_h, button_wh};
pub use cards::{TAG_TEXT, card, card_sized, checkbox_card, dimmed, select_frame, tag_at, tile_frame};
pub use chips::{
    BadgeKind, action_chip, badge, check_mark, chip, chip_h, chip_rt, dot, dot_at, kbd, kbd_dark, kbd_hint, section_header,
    segmented, segmented_h, segmented_small,
};
pub use controls::{checkbox, slider, slider_row, switch};
pub use misc::{SURFACE, checker, dashed_rect, divider, field, progress, spacer, text_at, text_input};
pub use text::{meta, med, mono_med_txt, mono_txt, semi_txt, txt};
