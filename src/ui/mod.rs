//! The egui panels of [`crate::App`], one file per screen area. Each file adds an `impl App` block; the `App` state
//! types live in `state.rs` and the shared helpers in `helpers.rs` (both next to `main.rs`).

mod animate;
mod center;
mod chrome;
mod demo;
mod export_ui;
mod inspector;
mod pad;
mod left;
mod poll;
mod scene_ui;
mod settings;
