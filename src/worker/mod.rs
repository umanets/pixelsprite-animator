//! The background thread that owns the models: loads them (with the CUDA -> DirectML -> CPU fallback), encodes
//! captions, draws sprites and streams progress back to the UI. The UI only sends commands and reads messages.
//!
//! A generation request follows the web demo's `run()` (docs/demo/app.js): in the "new shapes" mode a caption of
//! two or three words is mixed (`prepare_mix`, mix.rs: each sprite draws its own variant); otherwise the caption's
//! vector (the demo's precomputed one for its example captions, else the text encoder) is drawn with split
//! guidance. "One shape" first draws one sprite and then colours its silhouette `n` times (seed + 1); "draw" colours
//! the user's silhouette.

mod classify;
mod generate;
mod handle;
/// Headless helpers for the self-test binary (same code paths as the worker).
pub mod headless;
mod messages;
mod presets;
mod state;

pub use classify::MAP_THRESHOLD;
pub use handle::Worker;
pub use messages::{Cmd, GenRequest, GenResult, MixOptions, Mode, Msg, VariantInfo};
pub use presets::b64_floats;
