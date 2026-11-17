use std::borrow::Cow;

use anyhow::{Result, anyhow, bail};
use ort::memory::{Allocator, MemoryInfo};
use ort::session::{IoBinding, Session, SessionInputValue};
use ort::value::{DynTensor, DynValue, Tensor};

use super::engine::{cache_names, er};
use super::tables::AnimTables;

/// A CUDA-graph failure while setting up (fall back to the plain path) or while drawing (report it).
pub(super) enum GraphErr {
    Setup(anyhow::Error),
    Run(anyhow::Error),
}

pub(super) fn copy_logits(val: &DynValue, dst: &mut [f32], b: usize, v: usize) -> Result<()> {
    let (shape, l) = val.try_extract_tensor::<f32>().map_err(er)?;
    if shape.len() != 2 || shape[0] as usize != b || shape[1] as usize != v {
        bail!("unexpected logits shape {shape:?}");
    }
    dst.copy_from_slice(l);
    Ok(())
}

pub(super) enum StepRun<'s> {
    Fixed(FixedRun<'s>),
    Grow(GrowRun<'s>),
}

impl StepRun<'_> {
    pub(super) fn step(&mut self, x: &[f32], pos: i64, ctx: &[DynTensor]) -> Result<()> {
        match self {
            StepRun::Fixed(r) => r.step(x, pos),
            StepRun::Grow(r) => r.step(x, ctx),
        }
    }
    pub(super) fn logits(&self) -> &[f32] {
        match self {
            StepRun::Fixed(r) => &r.logits,
            StepRun::Grow(r) => &r.logits,
        }
    }
}

/// `step.onnx`: the two target-cache sets of the engine's cache swapped every step through an IoBinding (cf.
/// `StaticRun` in model.rs); the context cache is bound once. On a fresh cache the first step reads a zero-filled CPU
/// cache (fresh device memory may hold NaN, and masked slots are multiplied by 0); later frames reuse the sets, whose
/// stale slots are finite and masked.
pub(super) struct FixedRun<'s> {
    session: &'s mut Session,
    binding: IoBinding,
    set_a: &'s mut Vec<DynTensor>,
    set_b: &'s mut Vec<DynTensor>,
    zero: Option<DynTensor>,
    a_is_input: bool,
    first: bool,
    batch: usize,
    d: usize,
    v: usize,
    logits: Vec<f32>,
    names: Vec<(String, String)>,
}

impl<'s> FixedRun<'s> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        session: &'s mut Session,
        t: &AnimTables,
        batch: usize,
        m: usize,
        ctx: &[DynTensor],
        set_a: &'s mut Vec<DynTensor>,
        set_b: &'s mut Vec<DynTensor>,
        fresh: bool,
    ) -> Result<Self> {
        let shape = [batch, t.n_heads, m, t.hd];
        let zero = if fresh { Some(Tensor::from_array((shape, vec![0f32; batch * t.n_heads * m * t.hd])).map_err(er)?.upcast()) } else { None };
        let mut binding = session.create_binding().map_err(er)?;
        binding.bind_output_to_device("logits", &MemoryInfo::default()).map_err(er)?;
        for ((c, _), val) in cache_names(t.n_layers, "ctx", "").iter().zip(ctx) {
            binding.bind_input(c.as_str(), val).map_err(er)?;
        }
        Ok(FixedRun {
            session,
            binding,
            set_a,
            set_b,
            first: zero.is_some(),
            zero,
            a_is_input: false,
            batch,
            d: t.d,
            v: t.k + 1,
            logits: vec![0f32; batch * (t.k + 1)],
            names: cache_names(t.n_layers, "past", "present"),
        })
    }

    fn step(&mut self, x: &[f32], pos: i64) -> Result<()> {
        let xt = Tensor::from_array(([self.batch, 1, self.d], x.to_vec())).map_err(er)?;
        let pt = Tensor::from_array(([1usize], vec![pos])).map_err(er)?;
        self.binding.bind_input("x", &xt).map_err(er)?;
        self.binding.bind_input("pos", &pt).map_err(er)?;
        let (src, dst) = if self.a_is_input { (&*self.set_a, &mut *self.set_b) } else { (&*self.set_b, &mut *self.set_a) };
        for (i, (past, present)) in self.names.iter().enumerate() {
            match (&self.zero, self.first) {
                (Some(z), true) => self.binding.bind_input(past.as_str(), z).map_err(er)?,
                _ => self.binding.bind_input(past.as_str(), &src[i]).map_err(er)?,
            }
            self.binding.bind_output(present.as_str(), &mut dst[i]).map_err(er)?;
        }
        {
            let outs = self.session.run_binding(&self.binding).map_err(er)?;
            copy_logits(&outs["logits"], &mut self.logits, self.batch, self.v)?;
        }
        if self.first {
            self.first = false;
            self.zero = None; // the binding keeps it alive until the inputs are re-bound next step
        }
        self.a_is_input = !self.a_is_input;
        Ok(())
    }
}

/// `step_grow.onnx`: the target cache grows by one per call; the context cache is passed by reference (CPU) or bound
/// once (device).
pub(super) struct GrowRun<'s> {
    session: &'s mut Session,
    past: Vec<DynValue>,
    device: Option<(MemoryInfo<'static>, IoBinding)>,
    names: Vec<(String, String)>,
    ctx_names: Vec<String>,
    batch: usize,
    d: usize,
    v: usize,
    logits: Vec<f32>,
}

impl<'s> GrowRun<'s> {
    pub(super) fn new(session: &'s mut Session, mem: Option<MemoryInfo<'static>>, t: &AnimTables, batch: usize, ctx: &[DynTensor]) -> Result<Self> {
        let names = cache_names(t.n_layers, "past", "present");
        let ctx_names: Vec<String> = cache_names(t.n_layers, "ctx", "").into_iter().map(|(a, _)| a).collect();
        let past = (0..names.len())
            .map(|_| Tensor::<f32>::new(&Allocator::default(), [batch, t.n_heads, 0, t.hd]).map(|x| x.into_dyn()).map_err(er))
            .collect::<Result<Vec<_>>>()?;
        let device = match mem {
            Some(mem) => {
                let mut b = session.create_binding().map_err(er)?;
                for (c, val) in ctx_names.iter().zip(ctx) {
                    b.bind_input(c.as_str(), val).map_err(er)?;
                }
                Some((mem, b))
            }
            None => None,
        };
        Ok(GrowRun { session, past, device, names, ctx_names, batch, d: t.d, v: t.k + 1, logits: vec![0f32; batch * (t.k + 1)] })
    }

    fn step(&mut self, x: &[f32], ctx: &[DynTensor]) -> Result<()> {
        let xt = Tensor::from_array(([self.batch, 1, self.d], x.to_vec())).map_err(er)?;
        let mut next = Vec::with_capacity(self.names.len());
        if let Some((mem, binding)) = &mut self.device {
            binding.clear_outputs();
            binding.bind_output_to_device("logits", &MemoryInfo::default()).map_err(er)?;
            for (_, present) in &self.names {
                binding.bind_output_to_device(present.as_str(), mem).map_err(er)?;
            }
            binding.bind_input("x", &xt).map_err(er)?;
            for ((p, _), val) in self.names.iter().zip(&self.past) {
                binding.bind_input(p.as_str(), val).map_err(er)?;
            }
            let mut outs = self.session.run_binding(binding).map_err(er)?;
            copy_logits(&outs["logits"], &mut self.logits, self.batch, self.v)?;
            for (_, present) in &self.names {
                next.push(outs.remove(present.as_str()).ok_or_else(|| anyhow!("step: no output {present}"))?);
            }
        } else {
            let mut inputs: Vec<(Cow<'static, str>, SessionInputValue<'_>)> = Vec::with_capacity(1 + 2 * self.names.len());
            inputs.push(("x".into(), xt.into()));
            for (c, val) in self.ctx_names.iter().zip(ctx) {
                inputs.push((c.clone().into(), val.into()));
            }
            for ((p, _), val) in self.names.iter().zip(self.past.drain(..)) {
                inputs.push((p.clone().into(), val.into()));
            }
            let mut outs = self.session.run(inputs).map_err(er)?;
            copy_logits(&outs["logits"], &mut self.logits, self.batch, self.v)?;
            for (_, present) in &self.names {
                next.push(outs.remove(present.as_str()).ok_or_else(|| anyhow!("step: no output {present}"))?);
            }
        }
        self.past = next;
        Ok(())
    }
}
