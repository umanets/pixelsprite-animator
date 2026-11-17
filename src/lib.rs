//! animator_ui: a draft desktop app around pixelsprite canvas v2 (text -> sprite) and the sprite animation model
//! (AnimGPT, `animator::onnx`).
//! See docs/DEVELOPMENT.md for the architecture.

pub mod anchor_eval;
pub mod cancel;
pub mod animator;
pub mod classify;
pub mod config;
pub mod ep;
pub mod export;
pub mod fetch;
pub mod files;
pub mod mix;
pub mod theme;
pub mod ui_kit;
pub mod model;
pub mod save;
pub mod sprite;
pub mod text;
pub mod worker;
