use std::sync::atomic::{AtomicBool, Ordering};

use crate::ep::{Ep, SessionOpts};

use super::engine::{Engine, RunStats};
use super::reference::{AnchorOpts, RefChoice};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnnxSettings {
    pub temperature: f64,
    pub seed: u32,
    pub reference: RefChoice,
    pub anchor: AnchorOpts,
}

impl Default for OnnxSettings {
    /// Temperature 0.8: what the training's sample sheets use.
    fn default() -> Self {
        OnnxSettings { temperature: 0.8, seed: 1, reference: RefChoice::Auto, anchor: AnchorOpts::OFF }
    }
}

/// Progress of a request (for the UI's bar).
#[derive(Debug, Clone, Default)]
pub struct AnimProgress {
    /// "views 90°, 270°" / "walk 270°, run 4 dirs: frame 3/8" / "loading the model on CUDA".
    pub stage: String,
    /// 0..1 over the whole request.
    pub frac: f32,
    pub secs: f64,
}

/// Stop flag and progress callback of one request.
#[derive(Clone, Copy, Default)]
pub struct Ctl<'a> {
    pub cancel: Option<&'a AtomicBool>,
    pub progress: Option<&'a (dyn Fn(&AnimProgress) + Send + Sync)>,
}

impl Ctl<'_> {
    pub(super) fn stopped(&self) -> bool {
        self.cancel.is_some_and(|c| c.load(Ordering::Relaxed))
    }
    pub(super) fn report(&self, p: AnimProgress) {
        if let Some(f) = self.progress {
            f(&p);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct ClipKey {
    pub(super) sprite: u64,
    pub(super) size: usize,
    pub(super) kind: usize,
    pub(super) refk: usize,
    /// 0 = the view.
    pub(super) action: usize,
    pub(super) az: u16,
    pub(super) seed: u32,
    pub(super) temp: u64,
}

/// Providers to try (best first) and the session options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Wanted {
    pub(super) order: [Option<Ep>; 3],
    cuda_mem_limit_mb: usize,
    cpu_threads: usize,
}

impl Wanted {
    pub(super) fn new(order: &[Ep], opts: SessionOpts) -> Self {
        let mut o = [None; 3];
        for (i, ep) in order.iter().take(3).enumerate() {
            o[i] = Some(*ep);
        }
        Wanted { order: o, cuda_mem_limit_mb: opts.cuda_mem_limit_mb, cpu_threads: opts.cpu_threads }
    }
    pub(super) fn opts(&self) -> SessionOpts {
        SessionOpts { cuda_mem_limit_mb: self.cuda_mem_limit_mb, cpu_threads: self.cpu_threads }
    }
}

pub(super) struct Slot {
    pub(super) engine: Option<Engine>,
    /// What the loaded engine was loaded for.
    pub(super) loaded_for: Option<Wanted>,
    /// Why the providers before the one in use were skipped.
    pub(super) fallbacks: Vec<String>,
}

/// Totals of all requests (for the self-test's timing lines).
#[derive(Debug, Clone, Default)]
pub struct Totals {
    pub batches: Vec<RunStats>,
}
