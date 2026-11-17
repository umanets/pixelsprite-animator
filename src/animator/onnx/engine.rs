use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session};
use ort::value::{DynTensor, Tensor};

use crate::ep::{Ep, SessionOpts, build_session};

use super::tables::{AnimTables, Cond};

// ------------------------------------------------------------------------------------------------------ engine
/// How the target cache is handled.
#[derive(Debug, Clone)]
pub(super) enum StepMode {
    /// `step.onnx`, two fixed target-cache sets swapped through an IoBinding; Some = on that device memory (the
    /// context cache from the prefill stays there too).
    Fixed(Option<MemoryInfo<'static>>),
    /// `step_grow.onnx`, growing target cache; Some = kept on that device memory through an IoBinding.
    Grow(Option<MemoryInfo<'static>>),
}

impl StepMode {
    fn label(&self) -> &'static str {
        match self {
            StepMode::Fixed(Some(_)) => "fixed cache on the device",
            StepMode::Fixed(None) => "fixed cache in RAM",
            StepMode::Grow(Some(_)) => "growing cache on the device",
            StepMode::Grow(None) => "growing cache in RAM",
        }
    }
    pub(super) fn device(&self) -> Option<&MemoryInfo<'static>> {
        match self {
            StepMode::Fixed(m) | StepMode::Grow(m) => m.as_ref(),
        }
    }
}

/// Timing of one batch of rows (one frame per row).
#[derive(Debug, Clone, Copy, Default)]
pub struct RunStats {
    pub prefill_secs: f64,
    pub step_secs: f64,
    pub steps: usize,
    pub rows: usize,
    pub size: usize,
}

impl RunStats {
    pub fn add(&mut self, o: &RunStats) {
        self.prefill_secs += o.prefill_secs;
        self.step_secs += o.step_secs;
        self.steps += o.steps;
        self.rows = self.rows.max(o.rows);
        self.size = o.size;
    }
}

/// One row of a batch: what to condition on and, for teacher forcing, the frame to score.
pub struct RowSpec<'a> {
    pub cond: Cond,
    pub reference: &'a [u16],
    pub prev: &'a [u16],
    /// Colours that may be drawn (ascending, with EMPTY).
    pub allowed: &'a [usize],
    /// Teacher forcing: feed these tokens instead of sampling and score them.
    pub force: Option<&'a [u16]>,
}

pub struct RowsOut {
    pub tokens: Vec<Vec<u16>>,
    /// Bits of the forced tokens (0 for sampled rows).
    pub bits: Vec<f64>,
    pub stats: RunStats,
}

/// The two sessions of one execution provider.
pub struct Engine {
    pub tables: Arc<AnimTables>,
    pub(super) prefill: Session,
    pub(super) step: Session,
    pub(super) mode: StepMode,
    pub ep: Ep,
    pub(super) dir: PathBuf,
    pub(super) opts: SessionOpts,
    /// Context cache (+ target-cache sets) of the last batch shape, reused across frames.
    pub(super) kv: Option<KvCache>,
    /// CUDA: `step.onnx` replayed as CUDA graphs, one slot per (padded batch, size), most recent first.
    pub(super) graphs: Vec<GraphSlot>,
    /// CUDA graphs are tried (CUDA with the cache on the GPU) and have not failed.
    pub use_graphs: bool,
    /// Why CUDA graphs are off (a failure), for the status line.
    pub graph_note: Option<String>,
}

/// `step.onnx` captured as CUDA graphs for one (batch, context, target) shape (only the latest shape is kept, to
/// keep VRAM down; a new shape costs ~1 s to set up): its own session (symbolic
/// dimensions fixed), its own caches, fixed device buffers for `x`, `pos` and the logits, and two graphs (cache set
/// A -> B and B -> A, run options `gpu_graph_id` 1 and 2). A step = copy x and pos in, one graph launch, copy the
/// logits out: no per-kernel launch overhead, which is what a step costs otherwise (~7 ms at any batch / size).
pub(super) struct GraphSlot {
    pub(super) b: usize,
    pub(super) c: usize,
    pub(super) m: usize,
    pub(super) session: Session,
    pub(super) kv: KvCache,
    pub(super) x_dev: Tensor<f32>,
    pub(super) pos_dev: Tensor<i64>,
    pub(super) logits_dev: Tensor<f32>,
    pub(super) x_host: Tensor<f32>,
    pub(super) pos_host: Tensor<i64>,
    pub(super) logits_host: Tensor<f32>,
    pub(super) binds: [IoBinding; 2],
    pub(super) ids: [ort::session::RunOptions; 2],
}

/// Caches allocated once per (batch, size) on the provider's memory (RAM for the CPU).
pub(super) struct KvCache {
    pub(super) b: usize,
    pub(super) c: usize,
    pub(super) m: usize,
    /// (b, 8, C, 64) per layer and key / value: written by the prefill, read by every step.
    pub(super) ctx: Vec<DynTensor>,
    /// The two target-cache sets of `step.onnx` (b, 8, M, 64).
    pub(super) sets: Option<(Vec<DynTensor>, Vec<DynTensor>)>,
    /// Never written yet: the first step reads zeros instead (fresh memory may hold NaN).
    pub(super) fresh: bool,
}

pub(super) fn cache_names(n_layers: usize, a: &str, b: &str) -> Vec<(String, String)> {
    (0..n_layers).flat_map(|i| [("k", i), ("v", i)]).map(|(kv, i)| (format!("{a}_{kv}_{i}"), format!("{b}_{kv}_{i}"))).collect()
}

pub(super) fn er(e: ort::Error) -> anyhow::Error {
    anyhow!("{e}")
}

fn short(s: &str) -> String {
    let one = s.replace('\n', " ");
    if one.chars().count() > 300 { format!("{}…", one.chars().take(300).collect::<String>()) } else { one }
}

impl Engine {
    /// Loads the graphs on `ep` and proves them (a short prefill + two steps, finite logits). Tries, in order:
    /// CUDA: fixed cache on the GPU, then in RAM; DirectML: fixed on the GPU, growing on the GPU, growing in RAM;
    /// CPU: growing cache.
    pub fn load(dir: &Path, tables: Arc<AnimTables>, ep: Ep, opts: SessionOpts) -> Result<Self> {
        let dev = |d: AllocationDevice<'static>| MemoryInfo::new(d, 0, AllocatorType::Device, MemoryType::Default).ok();
        let attempts: Vec<StepMode> = match ep {
            Ep::Cuda => vec![StepMode::Fixed(dev(AllocationDevice::CUDA)), StepMode::Fixed(None)],
            Ep::DirectMl => vec![StepMode::Fixed(dev(AllocationDevice::DIRECTML)), StepMode::Grow(dev(AllocationDevice::DIRECTML)), StepMode::Grow(None)],
            Ep::Cpu => vec![StepMode::Grow(None)],
        };
        let mut errors = Vec::new();
        let mut prefill: Option<Session> = None;
        let mut steps: HashMap<&'static str, Session> = HashMap::new();
        for mode in attempts {
            let pre = match prefill.take() {
                Some(s) => s,
                None => match build_session(ep, &dir.join("prefill.onnx"), opts) {
                    Ok(s) => s,
                    Err(e) => {
                        errors.push(format!("prefill.onnx: {}", short(&format!("{e:#}"))));
                        break;
                    }
                },
            };
            let graph = if matches!(mode, StepMode::Fixed(_)) { "step.onnx" } else { "step_grow.onnx" };
            let step = match steps.remove(graph) {
                Some(s) => s,
                None => match build_session(ep, &dir.join(graph), opts) {
                    Ok(s) => s,
                    Err(e) => {
                        errors.push(format!("{}: {}", mode.label(), short(&format!("{e:#}"))));
                        prefill = Some(pre);
                        continue;
                    }
                },
            };
            let use_graphs = ep == Ep::Cuda && matches!(&mode, StepMode::Fixed(Some(_))) && std::env::var_os("ANIMATOR_UI_NO_CUDA_GRAPH").is_none();
            let mut eng = Engine { tables: tables.clone(), prefill: pre, step, mode: mode.clone(), ep, kv: None, graphs: Vec::new(), use_graphs, graph_note: None, dir: dir.to_path_buf(), opts };
            match eng.warm_up() {
                Ok(()) => return Ok(eng),
                Err(e) => {
                    errors.push(format!("{}: {}", mode.label(), short(&format!("{e:#}"))));
                    eng.kv = None;
                    prefill = Some(eng.prefill);
                    steps.insert(graph, eng.step);
                }
            }
        }
        bail!("{} failed: {}", ep.label(), errors.join(" | "))
    }

    pub fn describe(&self) -> String {
        let graphs = match (&self.graph_note, self.use_graphs) {
            (Some(n), _) => format!(", CUDA graphs off: {}", short(n)),
            (None, true) => ", CUDA graphs".to_string(),
            _ => String::new(),
        };
        format!("{} ({}{graphs})", self.ep.label(), self.mode.label())
    }

    /// Rows per batch: 8 directions on CUDA; on the CPU fewer at 24 / 32 to keep RAM modest; 1 on DirectML, whose
    /// device pointers are opaque handles (the prefill writes row j at an offset of the context cache).
    pub fn max_batch(&self, size: usize) -> usize {
        match (self.ep, size) {
            (Ep::DirectMl, _) => 1,
            (Ep::Cuda, _) | (Ep::Cpu, 16) => 8,
            (Ep::Cpu, _) => 4,
        }
    }

    /// Proves the plain graphs (CUDA graphs are set up on first use and fall back to these if they fail).
    fn warm_up(&mut self) -> Result<()> {
        let graphs = std::mem::replace(&mut self.use_graphs, false);
        let r = self.warm_up_plain();
        self.use_graphs = graphs;
        self.kv = None;
        r
    }

    fn warm_up_plain(&mut self) -> Result<()> {
        let t = self.tables.clone();
        let (d, c, n) = (t.d, 5, 3);
        let mut x0 = vec![0f32; c * d];
        for i in 0..c {
            t.write_target(&mut x0[i * d..(i + 1) * d], 16, t.empty, i);
        }
        let mut ok = true;
        self.run(&x0, 1, c, n, &mut |_, logits, x| {
            ok &= logits.iter().all(|v| v.is_finite());
            t.write_target(x, 16, t.empty, 1);
            Ok(())
        })?;
        if !ok {
            bail!("the backend returned non-finite logits");
        }
        Ok(())
    }
}
