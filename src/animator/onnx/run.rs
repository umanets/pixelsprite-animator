use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session};
use ort::value::{DynTensor, Tensor};

use crate::ep::{Ep, build_cuda_graph_session};
use crate::sprite::Mulberry32;

use super::engine::{Engine, GraphSlot, KvCache, RowSpec, RowsOut, RunStats, StepMode, cache_names, er};
use super::runners::{FixedRun, GraphErr, GrowRun, StepRun, copy_logits};
use super::tables::{AnimTables, bits_of, draw};

impl Engine {
    /// The context of every row, one prefill call per row (T = C, no past), written straight into row j of the
    /// preallocated context cache (the presents are bound to views of it). One call per row instead of chunks of a
    /// batch: the cache is allocated once per (batch, size) and reused frame after frame, so nothing grows during a
    /// request and the arena does not fragment; the per-call attention scores (8 x C x C floats, 139 MB at 32 px)
    /// are the only large temporaries. Returns pixel 0's logits (b x V).
    fn prefill(session: &mut Session, t: &AnimTables, ep: Ep, kv: &mut KvCache, x0: &[f32], b: usize, c: usize) -> Result<Vec<f32>> {
        let (d, v, h, hd) = (t.d, t.k + 1, t.n_heads, t.hd);
        let names = cache_names(t.n_layers, "past", "present");
        let mut logits = vec![0f32; b * v];
        let mut binding = session.create_binding().map_err(er)?;
        let row_floats = h * c * hd;
        if b > 1 && ep == Ep::DirectMl {
            bail!("DirectML: one row per batch (its device pointers cannot be offset)");
        }
        for j in 0..b {
            let xt = Tensor::from_array(([1, c, d], x0[j * c * d..(j + 1) * c * d].to_vec())).map_err(er)?;
            binding.clear_outputs();
            binding.bind_output_to_device("logits", &MemoryInfo::default()).map_err(er)?;
            let mut views = Vec::with_capacity(names.len());
            for ctx in kv.ctx.iter_mut() {
                let mem = ctx.memory_info().to_owned();
                let base = ctx.data_ptr_mut().cast::<f32>();
                // SAFETY: row j of a (b, h, c, hd) float tensor that outlives the call (owned by self.kv)
                let view = unsafe { ort::value::TensorRefMut::<f32>::from_raw(mem, base.add(j * row_floats).cast(), ort::value::Shape::new([1i64, h as i64, c as i64, hd as i64])) }
                    .map_err(er)?;
                views.push(view);
            }
            for ((_, present), view) in names.iter().zip(views) {
                binding.bind_output(present.as_str(), view).map_err(er)?;
            }
            binding.bind_input("x", &xt).map_err(er)?;
            for (p, _) in &names {
                let empty = Tensor::<f32>::new(&Allocator::default(), [1, h, 0, hd]).map_err(er)?;
                binding.bind_input(p.as_str(), &empty).map_err(er)?;
            }
            let outs = session.run_binding(&binding).map_err(er)?;
            copy_logits(&outs["logits"], &mut logits[j * v..(j + 1) * v], 1, v)?;
            drop(outs);
            binding.clear_outputs(); // the views die here
        }
        Ok(logits)
    }

    /// (Re)allocates the context cache and, for the fixed graph, the two target-cache sets when the batch or the
    /// size changes; kept otherwise.
    fn ensure_cache(&mut self, b: usize, c: usize, m: usize) -> Result<()> {
        if self.kv.as_ref().is_some_and(|k| k.b == b && k.c == c && k.m == m) {
            return Ok(());
        }
        self.kv = None; // free first
        let t = self.tables.clone();
        let mem = self.mode.device().cloned();
        let alloc = match &mem {
            Some(mi) => Allocator::new(&self.step, mi.clone()).map_err(|e| anyhow!("cache allocator: {e}"))?,
            None => Allocator::default(),
        };
        let mk = |len: usize| -> Result<Vec<DynTensor>> {
            (0..2 * t.n_layers).map(|_| Tensor::<f32>::new(&alloc, [b, t.n_heads, len, t.hd]).map(|x| x.upcast()).map_err(|e| anyhow!("cache tensor: {e}"))).collect()
        };
        let ctx = mk(c)?;
        let sets = match self.mode {
            StepMode::Fixed(_) => Some((mk(m.max(1))?, mk(m.max(1))?)),
            StepMode::Grow(_) => None,
        };
        self.kv = Some(KvCache { b, c, m, ctx, sets, fresh: true });
        Ok(())
    }

    /// Runs b rows: the context x0 (b x c x d), then n target pixels. `next(t, logits, x)` gets the logits of pixel
    /// t (b x V) and writes the input vectors of target token t + 1 into x (b x d); it is called for t = 0 .. n-1.
    /// On CUDA the steps go through CUDA graphs (`GraphSlot`) unless setting one up failed (then, for good, the
    /// plain IoBinding stepping; `ANIMATOR_UI_NO_CUDA_GRAPH` forces that).
    pub fn run(&mut self, x0: &[f32], b: usize, c: usize, n: usize, next: &mut dyn FnMut(usize, &[f32], &mut [f32]) -> Result<()>) -> Result<RunStats> {
        let t0 = Instant::now();
        let t = self.tables.clone();
        let mut stats = RunStats { rows: b, ..Default::default() };
        let m = n.saturating_sub(1);
        if self.use_graphs && m > 0 {
            match self.run_graph(x0, b, c, n, next, &mut stats) {
                Ok(()) => return Ok(stats),
                // a failure before the first step: CUDA graphs off for good, the plain path takes over
                Err(GraphErr::Setup(e)) => {
                    self.use_graphs = false;
                    self.graphs.clear();
                    self.graph_note = Some(format!("{e:#}"));
                }
                Err(GraphErr::Run(e)) => return Err(e),
            }
        }
        self.ensure_cache(b, c, m)?;
        let kv = self.kv.as_mut().ok_or_else(|| anyhow!("no cache"))?;
        let logits = Self::prefill(&mut self.prefill, &t, self.ep, kv, x0, b, c)?;
        stats.prefill_secs = t0.elapsed().as_secs_f64();
        let mut x = vec![0f32; b * t.d];
        next(0, &logits, &mut x)?;
        if n == 1 {
            return Ok(stats);
        }
        let t1 = Instant::now();
        let kv = self.kv.as_mut().ok_or_else(|| anyhow!("no cache"))?;
        let mut runner = match (&self.mode, kv.sets.as_mut()) {
            (StepMode::Fixed(_), Some((a, bb))) => {
                let fresh = std::mem::replace(&mut kv.fresh, false);
                StepRun::Fixed(FixedRun::new(&mut self.step, &t, b, m, &kv.ctx, a, bb, fresh)?)
            }
            (StepMode::Grow(mem), _) => StepRun::Grow(GrowRun::new(&mut self.step, mem.clone(), &t, b, &kv.ctx)?),
            _ => bail!("cache sets missing"),
        };
        for pos in 0..m {
            runner.step(&x, pos as i64, &kv.ctx)?;
            next(pos + 1, runner.logits(), &mut x)?;
        }
        stats.steps = m;
        stats.step_secs = t1.elapsed().as_secs_f64();
        Ok(stats)
    }

    /// The slot for a padded batch, creating it (session, caches, buffers, bindings) when missing; one kept.
    fn graph_slot(&mut self, b: usize, c: usize, m: usize) -> Result<usize> {
        if let Some(i) = self.graphs.iter().position(|g| g.b == b && g.c == c && g.m == m) {
            let g = self.graphs.remove(i);
            self.graphs.insert(0, g);
            return Ok(0);
        }
        self.graphs.clear(); // one shape at a time: free the old session and caches before allocating (VRAM)
        self.kv = None;
        let t = self.tables.clone();
        let dims = [("batch", b as i64), ("context", c as i64), ("target", m as i64)];
        let session = build_cuda_graph_session(&self.dir.join("step.onnx"), self.opts, &dims)?;
        let mem = MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default).map_err(er)?;
        let alloc = Allocator::new(&session, mem).map_err(|e| anyhow!("cache allocator: {e}"))?;
        let mk = |len: usize| -> Result<Vec<DynTensor>> {
            (0..2 * t.n_layers).map(|_| Tensor::<f32>::new(&alloc, [b, t.n_heads, len, t.hd]).map(|x| x.upcast()).map_err(|e| anyhow!("cache tensor: {e}"))).collect()
        };
        let mut sets = (mk(m)?, mk(m)?);
        // zero both sets once (fresh device memory may hold NaN; masked slots are multiplied by 0)
        let zero = Tensor::from_array(([b, t.n_heads, m, t.hd], vec![0f32; b * t.n_heads * m * t.hd])).map_err(er)?.upcast();
        for x in sets.0.iter_mut().chain(sets.1.iter_mut()) {
            zero.copy_into(x).map_err(er)?;
        }
        let kv = KvCache { b, c, m, ctx: mk(c)?, sets: Some(sets), fresh: false };
        let v = t.k + 1;
        let x_dev = Tensor::<f32>::new(&alloc, [b, 1, t.d]).map_err(er)?;
        let pos_dev = Tensor::<i64>::new(&alloc, [1usize]).map_err(er)?;
        let logits_dev = Tensor::<f32>::new(&alloc, [b, v]).map_err(er)?;
        let x_host = Tensor::<f32>::from_array(([b, 1, t.d], vec![0f32; b * t.d])).map_err(er)?;
        let pos_host = Tensor::<i64>::from_array(([1usize], vec![0i64])).map_err(er)?;
        let logits_host = Tensor::<f32>::from_array(([b, v], vec![0f32; b * v])).map_err(er)?;
        let names = cache_names(t.n_layers, "past", "present");
        let ctx_names = cache_names(t.n_layers, "ctx", "");
        let (set_a, set_b) = kv.sets.as_ref().unwrap();
        let make = |src: &Vec<DynTensor>, dst: &Vec<DynTensor>| -> Result<IoBinding> {
            let mut bind = session.create_binding().map_err(er)?;
            bind.bind_input("x", &x_dev).map_err(er)?;
            bind.bind_input("pos", &pos_dev).map_err(er)?;
            for ((cn, _), val) in ctx_names.iter().zip(&kv.ctx) {
                bind.bind_input(cn.as_str(), val).map_err(er)?;
            }
            for (i, (past, present)) in names.iter().enumerate() {
                bind.bind_input(past.as_str(), &src[i]).map_err(er)?;
                // SAFETY: `dst[i]` is a device tensor of shape [b, n_heads, m, hd] (allocated by `mk` above). The ref
                // built here only aliases its buffer so the binding can write the next cache into it; `dst` is owned
                // by `kv.sets`, which moves into the same `GraphSlot` as `binds`, so the memory outlives the binding.
                // Nothing else touches the tensor while a graph run is in flight (`run_graph` takes `&mut self`).
                let out = unsafe { ort::value::TensorRefMut::<f32>::from_raw(dst[i].memory_info().to_owned(), dst[i].data_ptr().cast_mut(), ort::value::Shape::new([b as i64, t.n_heads as i64, m as i64, t.hd as i64])) }
                    .map_err(er)?;
                bind.bind_output(present.as_str(), out).map_err(er)?;
            }
            // SAFETY: same as above for `logits_dev` ([b, v], allocated just before), which is also a field of the `GraphSlot`.
            let lg = unsafe { ort::value::TensorRefMut::<f32>::from_raw(logits_dev.memory_info().to_owned(), logits_dev.data_ptr().cast_mut(), ort::value::Shape::new([b as i64, v as i64])) }.map_err(er)?;
            bind.bind_output("logits", lg).map_err(er)?;
            Ok(bind)
        };
        let binds = [make(set_a, set_b)?, make(set_b, set_a)?];
        let mut ids = [ort::session::RunOptions::new().map_err(er)?, ort::session::RunOptions::new().map_err(er)?];
        ids[0].set("gpu_graph_id", "1").map_err(er)?;
        ids[1].set("gpu_graph_id", "2").map_err(er)?;
        self.graphs.insert(0, GraphSlot { b, c, m, session, kv, x_dev, pos_dev, logits_dev, x_host, pos_host, logits_host, binds, ids });
        Ok(0)
    }

    /// [`Engine::run`] with CUDA graphs: the batch is padded to 1 / 2 / 4 / 8 rows (copies of row 0) so a few
    /// captured shapes serve every request.
    fn run_graph(&mut self, x0: &[f32], b: usize, c: usize, n: usize, next: &mut dyn FnMut(usize, &[f32], &mut [f32]) -> Result<()>, stats: &mut RunStats) -> std::result::Result<(), GraphErr> {
        let t0 = Instant::now();
        let t = self.tables.clone();
        let (d, v, m) = (t.d, t.k + 1, n - 1);
        let bp = [1usize, 2, 4, 8].into_iter().find(|&p| p >= b).unwrap_or(b);
        let i = self.graph_slot(bp, c, m).map_err(GraphErr::Setup)?;
        let mut xp = x0.to_vec();
        for _ in b..bp {
            xp.extend_from_slice(&x0[..c * d]);
        }
        let g = &mut self.graphs[i];
        let logits = Self::prefill(&mut self.prefill, &t, self.ep, &mut g.kv, &xp, bp, c).map_err(GraphErr::Setup)?;
        stats.prefill_secs = t0.elapsed().as_secs_f64();
        let mut x = vec![0f32; b * d];
        next(0, &logits[..b * v], &mut x).map_err(GraphErr::Run)?;
        let t1 = Instant::now();
        let mut lg = vec![0f32; b * v];
        for pos in 0..m {
            {
                let (_, xh) = g.x_host.extract_tensor_mut();
                for j in 0..bp {
                    let src = if j < b { j } else { 0 };
                    xh[j * d..(j + 1) * d].copy_from_slice(&x[src * d..(src + 1) * d]);
                }
                g.pos_host.extract_tensor_mut().1[0] = pos as i64;
            }
            let setup = pos < 2; // the first call of each graph id captures it
            let wrap = |e: ort::Error| if setup { GraphErr::Setup(anyhow!("{e}")) } else { GraphErr::Run(anyhow!("{e}")) };
            g.x_host.copy_into(&mut g.x_dev).map_err(wrap)?;
            g.pos_host.copy_into(&mut g.pos_dev).map_err(wrap)?;
            let k = pos % 2;
            g.session.run_binding_with_options(&g.binds[k], &g.ids[k]).map_err(wrap)?;
            g.logits_dev.copy_into(&mut g.logits_host).map_err(wrap)?;
            lg.copy_from_slice(&g.logits_host.extract_tensor().1[..b * v]);
            if setup && !lg.iter().all(|x| x.is_finite()) {
                return Err(GraphErr::Setup(anyhow!("non-finite logits from the CUDA graph")));
            }
            next(pos + 1, &lg, &mut x).map_err(GraphErr::Run)?;
        }
        stats.steps = m;
        stats.step_secs = t1.elapsed().as_secs_f64();
        Ok(())
    }

    /// One frame per row, sampled (or scored with `force`) with the app's sampler.
    #[allow(clippy::too_many_arguments)]
    pub fn run_rows(
        &mut self,
        s: usize,
        rows: &[RowSpec],
        rngs: &mut [Mulberry32],
        temperature: f64,
        cancel: Option<&AtomicBool>,
        on_pixel: &mut dyn FnMut(usize),
    ) -> Result<RowsOut> {
        let t = self.tables.clone();
        if !t.sizes.contains(&s) {
            bail!("the model draws 16, 24 or 32 px sprites, not {s}");
        }
        let (d, v, b, n, c) = (t.d, t.k + 1, rows.len(), s * s, t.context_len(s));
        if b == 0 {
            return Ok(RowsOut { tokens: Vec::new(), bits: Vec::new(), stats: RunStats::default() });
        }
        if rngs.len() != b {
            bail!("one random stream per row");
        }
        for r in rows {
            if r.reference.len() != n || r.prev.len() != n || r.force.is_some_and(|f| f.len() != n) {
                bail!("canvases must be {s}x{s}");
            }
        }
        let mb = self.max_batch(s);
        if b > mb {
            // more rows than one batch holds: in parts (each row has its own random stream, so nothing changes)
            let mut all = RowsOut { tokens: Vec::new(), bits: Vec::new(), stats: RunStats::default() };
            for (part, rg) in rows.chunks(mb).zip(rngs.chunks_mut(mb)) {
                let o = self.run_rows(s, part, rg, temperature, cancel, on_pixel)?;
                all.tokens.extend(o.tokens);
                all.bits.extend(o.bits);
                all.stats.add(&o.stats);
            }
            return Ok(all);
        }
        let mut x0 = vec![0f32; b * c * d];
        for (j, r) in rows.iter().enumerate() {
            t.write_context(&mut x0[j * c * d..(j + 1) * c * d], s, &r.cond, r.reference, r.prev);
        }
        let mut tokens = vec![vec![0u16; n]; b];
        let mut bits = vec![0f64; b];
        let mut cum = Vec::with_capacity(t.k + 1);
        let inv_t = 1.0 / temperature.max(1e-3);
        let mut stats = self.run(&x0, b, c, n, &mut |px, logits, x| {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Err(crate::cancel::Cancelled.into());
            }
            for (i, r) in rows.iter().enumerate() {
                let l = &logits[i * v..(i + 1) * v];
                let tok = match r.force {
                    Some(f) => {
                        bits[i] += bits_of(l, f[px] as usize);
                        f[px] as usize
                    }
                    None => draw(l, r.allowed, inv_t, rngs[i].next_f64(), &mut cum),
                };
                tokens[i][px] = tok as u16;
                if px + 1 < n {
                    t.write_target(&mut x[i * d..(i + 1) * d], s, tok, px + 1);
                }
            }
            on_pixel(px);
            Ok(())
        })?;
        stats.size = s;
        Ok(RowsOut { tokens, bits, stats })
    }
}
