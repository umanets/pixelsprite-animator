//! The animation model in ONNX Runtime: AnimGPT (`pilot3d/train/anim_model.py`, run `anim39`), exported by
//! `pilot3d/train/export_anim_onnx.py` to `<app>/models/animator/`.
//!
//! # The model (sprite "image-to-video")
//!
//! A decoder-only transformer of the same family as the text-to-sprite model v2 (it was warm-started from v2: same
//! 12 blocks, colour / row / column embeddings and head). It draws one frame of a character pixel by pixel, raster
//! order, given the character's reference sprite and the previous frame. One network row = one direction of one frame:
//!
//! 1. *v2 prefix*: the "no text" vector, the size token, S unknown-silhouette row tokens (2 + S tokens);
//! 2. *condition token*: kind (human / quadruped / bird) + action (`view` or one of 19 actions) + azimuth (8) +
//!    elevation (0 / 20, the app uses 0) + frame index + frame count of the action + reference type
//!    (0 = the original v2 sprite, 1 = the pixelated front render of the 3D farm);
//! 3. *reference*: the reference sprite, S*S palette tokens (v2's global palette, 1024 colours, EMPTY = 1024);
//! 4. *previous*: S*S tokens of the previous canvas;
//! 5. *target*: the frame, v2 pixel tokens shifted by one (BOS, then the colour of pixel t-1); logits over the 1024
//!    colours + EMPTY. S is 16, 24 or 32 (a real 24 x 24 canvas here, not v2's window in the 32 canvas).
//!
//! Two tasks, as in training (`train_anim.py`): **view** (action `view`, previous = reference: the character turned
//! to an azimuth at elevation 0) and **animation** (frame 0: previous = the view at that azimuth; frame k: previous =
//! frame k-1). The app draws the views first, then the action frames, and samples only colours of the reference
//! (+ EMPTY). Frame counts are the dataset's (`embeddings.json` `frames`, same as [`crate::animator::standard_actions`]).
//!
//! # ONNX interface (only the transformer; input vectors are sums of table rows built here, see `embeddings.json`)
//!
//! * `prefill.onnx` (v2's growing-cache decoder): `x` (B,T,512), `past_k_i` / `past_v_i` (B,8,P,64) -> `logits` of
//!   the last token (B,1025), `present_k_i` / `present_v_i` (B,8,P+T,64). The *context* (prefix, condition,
//!   reference, previous canvas and the target's BOS token: C = 2 + S + 1 + 2 S² + 1 tokens, 2084 at 32) goes in
//!   as one call per row (P = 0, T = C) instead of 2000 single-token calls; the presents are bound to row j of the
//!   preallocated context cache, and the logits are pixel 0's.
//! * `step.onnx` (GPU): one target token per call: `x` (B,1,512), `pos` (1,) int64, `ctx_k_i` / `ctx_v_i`
//!   (B,8,C,64) read only, `past_k_i` / `past_v_i` (B,8,M,64) with M = S² - 1 -> `logits` (B,1025), `present_*`
//!   (B,8,M,64), new key / value written at `pos`. The target cache is two fixed sets swapped every step through an
//!   `IoBinding` (as v2's `decoder_static.onnx`), so nothing is allocated per step and only `x`, `pos` and the logits
//!   cross the bus; the context cache is bound once per frame and never copied. On CUDA the step is replayed as a
//!   CUDA graph (a session per padded batch / size with fixed dimensions, two captured graphs for the two cache
//!   directions): ~7 ms -> 1.6-5 ms per step, same output; see [`Engine::run`].
//! * `step_grow.onnx` (CPU): the same with a growing target cache (`past` (B,8,P,64) -> `present` (B,8,P+1,64)).
//! * `anim.data`: the float32 weights shared by the three graphs.
//!
//! # Rust side
//!
//! [`OnnxAnimator`] loads lazily on the execution provider of the sprite model ([`OnnxAnimator::set_providers`]),
//! maps a [`crate::sprite::Sprite`] to v2 palette tokens (exact colour, else the nearest; frames come back in the sprite's own
//! palette), draws all requested clips as one batch (views of every azimuth first, then the frames, at most
//! [`Engine::max_batch`] rows), caches each clip per (sprite, size, kind, reference type, action, azimuth, seed,
//! temperature), and reports progress / stops through a [`Ctl`]. Sampling: softmax(logits / T) over the reference's
//! colours + EMPTY, inverse CDF with one mulberry32 number per pixel, one random stream per clip ([`row_seed`]), so a
//! clip does not depend on what else is in its batch. `pilot3d/train/export_anim_onnx.py --reference` runs the same
//! sampler in PyTorch; `selftest --onnx-anim` compares.

mod animator;
mod clips;
mod engine;
mod reference;
mod runners;
mod run;
mod settings;
mod tables;

#[cfg(test)]
mod tests;

pub use self::animator::{OnnxAnimator, OnnxWith};
pub use self::engine::{Engine, RowSpec, RowsOut, RunStats};
pub use self::reference::{ANCHOR_PALETTE, ANCHOR_REF, AnchorOpts, AnchorPalette, AnchorRef, RefChoice, Reference, anchor_action, split_anchor_clips};
pub use self::settings::{AnimProgress, Ctl, OnnxSettings, Totals};
pub use self::tables::{AnimTables, Cond, bits_of, draw, row_seed};
