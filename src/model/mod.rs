//! pixelsprite canvas v2 (text -> sprite), a port of the web demo's engine.js to Rust + ONNX Runtime.
//!
//! Two graphs of the release hold the 12 transformer blocks, the final norm and the head; input vectors are sums of
//! rows of small tables (`embeddings.bin`), built exactly as `writePrefix` / `writePixel` in engine.js.
//!
//! * GPU (CUDA, DirectML when it works): `decoder_static.onnx`, one token per call with a key/value cache of fixed
//!   length M = 2 + S + S*S (inputs `x` (B,1,512), `pos` (1,) int64, `past_k_i` / `past_v_i` (B,8,M,64); outputs
//!   `logits` (B,1025) and `present_k_i` / `present_v_i` of the same shape, new key / value written at `pos`).
//!   The cache never leaves the device: two sets of cache tensors are allocated once per batch on the provider's
//!   memory and swapped every step through an `IoBinding` (inputs = set A, outputs = set B, then the other way
//!   round); only `x`, `pos` and `logits` (B x 1025 floats) cross the bus. The first step reads a zero-filled CPU
//!   cache (fresh device memory is not guaranteed to be zero, and masked slots are multiplied by 0, so NaN garbage
//!   there would poison the output).
//! * CPU (and DirectML's last resort): `decoder.onnx`, growing cache (`present` = concat(past, new)), the prefix in
//!   one call; the fixed graph's full-cache `Where` per step is much slower than that on a CPU.
//!
//! Sampling is the same code as engine.js, so a seed gives the web demo's sprites (checked against the demo's own
//! engine.js run headless, see docs/DEVELOPMENT.md *Parity with the web demo*):
//! * strategies ([`Strategy`], engine.js `Strategy`): plain (one caption), split classifier-free guidance (shape and
//!   colour strengths), mix (several captions at once, AND / product of experts, a weight map per caption: the
//!   demo's caption mixing, `mix.rs`); each sprite owns its strategy's rows of the batch, the runner knows nothing
//!   about guidance;
//! * silhouette masks (the demo's "draw" and "one shape" modes): known-row prefix, edge flags, forced empty / filled
//!   pixels;
//! * the row budget: sprites are packed into chunks while their rows fit `rows` (the demo's `rowsFor`), so the
//!   random numbers are consumed in the demo's order;
//! * temperature, top-p, the empty / 1-2 px filter, mulberry32.

mod engine;
mod filter;
mod masks;
mod params;
mod runners;
mod sampling;
mod strategy;
mod tables;

pub use engine::SpriteModel;
pub use filter::{crop_view, pieces, rejected};
pub use masks::{demo_rows_cpu, demo_rows_gpu, edge_flags};
pub use params::{GenOutput, GenParams, GenStats, Progress, View};
pub use strategy::Strategy;
pub use tables::{Kinds, ModelCore};

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session, SessionInputValue};
use ort::value::{DynTensor, DynValue, Tensor};
use serde::Deserialize;

use crate::ep::{Ep, SessionOpts, build_session};
use crate::files::ModelFiles;
use crate::sprite::{Mulberry32, Palette};
