use super::*;

pub(super) enum Runner<'s> {
    Fixed(StaticRun<'s>),
    Growing(GrowRun<'s>),
}

impl Runner<'_> {
    /// The prefix tokens x0 (batch, t0, d): one call (growing cache) or one token at a time (fixed cache).
    pub(super) fn prefix(&mut self, x0: &[f32], batch: usize, t0: usize, d: usize) -> Result<()> {
        match self {
            Runner::Growing(r) => r.run(x0, t0),
            Runner::Fixed(r) => {
                let mut xt = vec![0f32; batch * d];
                for i in 0..t0 {
                    for j in 0..batch {
                        xt[j * d..(j + 1) * d].copy_from_slice(&x0[(j * t0 + i) * d..(j * t0 + i + 1) * d]);
                    }
                    r.step(&xt)?;
                }
                Ok(())
            }
        }
    }
    /// `t` new tokens per row (always 1 for the fixed cache).
    pub(super) fn step(&mut self, x: &[f32], t: usize) -> Result<()> {
        match self {
            Runner::Growing(r) => r.run(x, t),
            Runner::Fixed(r) => r.step(x),
        }
    }
    pub(super) fn logits(&self) -> &[f32] {
        match self {
            Runner::Growing(r) => &r.logits,
            Runner::Fixed(r) => &r.logits,
        }
    }
    pub(super) fn steps(&self) -> usize {
        match self {
            Runner::Growing(r) => r.steps,
            Runner::Fixed(r) => r.steps,
        }
    }
}

pub(super) fn cache_names(n_layers: usize) -> Vec<(String, String)> {
    (0..n_layers)
        .flat_map(|i| [("k", i), ("v", i)])
        .map(|(kv, i)| (format!("past_{kv}_{i}"), format!("present_{kv}_{i}")))
        .collect()
}

/// Growing cache with `decoder.onnx`: every call returns the longer cache, which goes straight back in.
pub(super) struct GrowRun<'s> {
    session: &'s mut Session,
    past: Vec<DynValue>,
    /// Some: outputs stay on this device memory (IoBinding).
    device: Option<(MemoryInfo<'static>, IoBinding)>,
    names: Vec<(String, String)>,
    batch: usize,
    d: usize,
    v: usize,
    logits: Vec<f32>,
    steps: usize,
}

impl<'s> GrowRun<'s> {
    pub(super) fn new(session: &'s mut Session, device: Option<MemoryInfo<'static>>, core: &ModelCore, batch: usize) -> Result<Self> {
        let names = cache_names(core.n_layers);
        let past = (0..names.len())
            .map(|_| {
                Tensor::<f32>::new(&Allocator::default(), [batch, core.n_heads, 0, core.hd]).map(|t| t.into_dyn()).map_err(|e| anyhow!("empty cache: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let device = match device {
            Some(mem) => Some((mem, session.create_binding().map_err(|e| anyhow!("{e}"))?)),
            None => None,
        };
        Ok(GrowRun { session, past, device, names, batch, d: core.d, v: core.k + 1, logits: vec![0f32; batch * (core.k + 1)], steps: 0 })
    }

    pub(super) fn run(&mut self, x: &[f32], t: usize) -> Result<()> {
        let er = |e: ort::Error| anyhow!("{e}");
        let xt = Tensor::from_array(([self.batch, t, self.d], x.to_vec())).map_err(er)?;
        if let Some((mem, binding)) = &mut self.device {
            binding.clear_outputs(); // fresh outputs every step: the cache grows by one
            binding.bind_output_to_device("logits", &MemoryInfo::default()).map_err(er)?;
            for (_, present) in &self.names {
                binding.bind_output_to_device(present.as_str(), mem).map_err(er)?;
            }
            binding.bind_input("x", &xt).map_err(er)?;
            for ((past, _), val) in self.names.iter().zip(&self.past) {
                binding.bind_input(past.as_str(), val).map_err(er)?;
            }
            let mut outs = self.session.run_binding(binding).map_err(er)?;
            {
                let (shape, l) = outs["logits"].try_extract_tensor::<f32>().map_err(er)?;
                if shape.len() != 2 || shape[0] as usize != self.batch || shape[1] as usize != self.v {
                    bail!("unexpected logits shape {shape:?}");
                }
                self.logits.copy_from_slice(l);
            }
            let mut next = Vec::with_capacity(self.names.len());
            for (_, present) in &self.names {
                next.push(outs.remove(present.as_str()).ok_or_else(|| anyhow!("missing output {present}"))?);
            }
            drop(outs);
            self.past = next;
            self.steps += 1;
            return Ok(());
        }
        let mut inputs: Vec<(Cow<'static, str>, SessionInputValue<'static>)> = Vec::with_capacity(1 + self.names.len());
        inputs.push(("x".into(), xt.into()));
        for ((past, _), val) in self.names.iter().zip(self.past.drain(..)) {
            inputs.push((past.clone().into(), val.into()));
        }
        let mut outs = self.session.run(inputs).map_err(er)?;
        {
            let (shape, l) = outs["logits"].try_extract_tensor::<f32>().map_err(er)?;
            if shape.len() != 2 || shape[0] as usize != self.batch || shape[1] as usize != self.v {
                bail!("unexpected logits shape {shape:?}");
            }
            self.logits.copy_from_slice(l);
        }
        for (_, present) in &self.names {
            self.past.push(outs.remove(present.as_str()).ok_or_else(|| anyhow!("missing output {present}"))?);
        }
        self.steps += 1;
        Ok(())
    }
}

/// Fixed-size cache stepping with an IoBinding and two swapped cache sets (see module docs).
pub(super) struct StaticRun<'s> {
    session: &'s mut Session,
    binding: IoBinding,
    set_a: Vec<DynTensor>,
    set_b: Vec<DynTensor>,
    zero: Option<DynTensor>,
    a_is_input: bool,
    first: bool,
    pos: i64,
    batch: usize,
    d: usize,
    v: usize,
    logits: Vec<f32>,
    names: Vec<(String, String)>,
    steps: usize,
}

/// The two cache sets of a fixed-cache run (on `cache_mem`, else in RAM).
pub(super) fn cache_sets(session: &Session, cache_mem: Option<&MemoryInfo<'static>>, core: &ModelCore, batch: usize, m: usize) -> Result<(Vec<DynTensor>, Vec<DynTensor>)> {
    let shape = [batch, core.n_heads, m, core.hd];
    let alloc = match cache_mem {
        Some(mi) => Allocator::new(session, mi.clone()).map_err(|e| anyhow!("cache allocator: {e}"))?,
        None => Allocator::default(),
    };
    let mk = || -> Result<Vec<DynTensor>> {
        (0..2 * core.n_layers).map(|_| Tensor::<f32>::new(&alloc, shape).map(|t| t.upcast()).map_err(|e| anyhow!("cache tensor: {e}"))).collect()
    };
    let set_a = mk()?;
    Ok((set_a, mk()?))
}

impl<'s> StaticRun<'s> {
    pub(super) fn new(session: &'s mut Session, sets: (Vec<DynTensor>, Vec<DynTensor>), core: &ModelCore, batch: usize, m: usize) -> Result<Self> {
        let shape = [batch, core.n_heads, m, core.hd];
        let numel = batch * core.n_heads * m * core.hd;
        let (set_a, set_b) = sets;
        let zero = Tensor::from_array((shape, vec![0f32; numel])).map_err(|e| anyhow!("{e}"))?.upcast();
        let mut binding = session.create_binding().map_err(|e| anyhow!("{e}"))?;
        binding.bind_output_to_device("logits", &MemoryInfo::default()).map_err(|e| anyhow!("{e}"))?;
        Ok(StaticRun {
            session,
            binding,
            set_a,
            set_b,
            zero: Some(zero),
            a_is_input: false,
            first: true,
            pos: 0,
            batch,
            d: core.d,
            v: core.k + 1,
            logits: vec![0f32; batch * (core.k + 1)],
            names: cache_names(core.n_layers),
            steps: 0,
        })
    }

    /// One token per row: x is (batch * d); afterwards `logits` holds the next pixel's (batch * (K + 1)).
    pub(super) fn step(&mut self, x: &[f32]) -> Result<()> {
        let er = |e: ort::Error| anyhow!("{e}");
        let xt = Tensor::from_array(([self.batch, 1, self.d], x.to_vec())).map_err(er)?;
        let pt = Tensor::from_array(([1usize], vec![self.pos])).map_err(er)?;
        self.binding.bind_input("x", &xt).map_err(er)?;
        self.binding.bind_input("pos", &pt).map_err(er)?;
        let (src, dst) = if self.a_is_input { (&self.set_a, &mut self.set_b) } else { (&self.set_b, &mut self.set_a) };
        for (i, (past, present)) in self.names.iter().enumerate() {
            match (&self.zero, self.first) {
                (Some(z), true) => self.binding.bind_input(past.as_str(), z).map_err(er)?,
                _ => self.binding.bind_input(past.as_str(), &src[i]).map_err(er)?,
            }
            self.binding.bind_output(present.as_str(), &mut dst[i]).map_err(er)?;
        }
        {
            let outs = self.session.run_binding(&self.binding).map_err(er)?;
            let (shape, l) = outs["logits"].try_extract_tensor::<f32>().map_err(er)?;
            if shape.len() != 2 || shape[0] as usize != self.batch || shape[1] as usize != self.v {
                bail!("unexpected logits shape {shape:?}");
            }
            self.logits.copy_from_slice(l);
        }
        if self.first {
            self.first = false;
            self.zero = None; // the binding keeps it alive until the inputs are re-bound next step
        }
        // the set just written becomes the next input
        self.a_is_input = !self.a_is_input;
        self.pos += 1;
        self.steps += 1;
        Ok(())
    }
}
