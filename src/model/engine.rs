use super::*;
use super::filter::crop;
use super::masks::EDGE_UNKNOWN;
use super::runners::{GrowRun, Runner, StaticRun, cache_names, cache_sets};
use super::sampling::draw;
use super::strategy::Ctx;

/// How the key/value cache is handled (see module docs).
#[derive(Debug, Clone)]
enum CacheMode {
    /// `decoder_static.onnx`, fixed-length cache swapped through an IoBinding; Some = on that device memory.
    Fixed(Option<MemoryInfo<'static>>),
    /// `decoder.onnx`, growing cache (concat), prefix in one call (best on the CPU, where the static graph's
    /// full-cache `Where` per step costs more than the whole transformer). Some = the cache stays on that device
    /// memory through an IoBinding (outputs re-bound every step, since their length grows); None = plain `run`.
    Growing(Option<MemoryInfo<'static>>),
}

impl CacheMode {
    fn label(&self) -> &'static str {
        match self {
            CacheMode::Fixed(Some(_)) => "fixed cache on the device",
            CacheMode::Fixed(None) => "fixed cache in RAM",
            CacheMode::Growing(Some(_)) => "growing cache on the device",
            CacheMode::Growing(None) => "growing cache in RAM",
        }
    }
}

/// The decoder session plus the core.
pub struct SpriteModel {
    pub core: Arc<ModelCore>,
    session: Session,
    pub ep: Ep,
    mode: CacheMode,
    /// (rows, cache length) of the last fixed-cache chunk on the device: a new shape first frees the arena.
    last_shape: Option<(usize, usize)>,
}

impl SpriteModel {
    /// Loads the decoder on `ep` and proves it works (two steps, finite logits). Tries, in order:
    /// CUDA: static graph with the cache on the GPU, then in RAM; DirectML: the same, then the dynamic graph;
    /// CPU: the dynamic graph.
    pub fn load(files: &ModelFiles, ep: Ep, opts: SessionOpts) -> Result<Self> {
        let core = Arc::new(ModelCore::load(files)?);
        let dev = |d: AllocationDevice<'static>| MemoryInfo::new(d, 0, AllocatorType::Device, MemoryType::Default).ok();
        let fixed = files.decoder_static.as_path();
        let growing = files.decoder.as_path();
        let attempts: Vec<(&Path, CacheMode)> = match ep {
            Ep::Cuda => vec![(fixed, CacheMode::Fixed(dev(AllocationDevice::CUDA))), (fixed, CacheMode::Fixed(None))],
            Ep::DirectMl => vec![
                (fixed, CacheMode::Fixed(dev(AllocationDevice::DIRECTML))),
                (growing, CacheMode::Growing(dev(AllocationDevice::DIRECTML))),
                (fixed, CacheMode::Fixed(None)),
                (growing, CacheMode::Growing(None)),
            ],
            Ep::Cpu => vec![(growing, CacheMode::Growing(None))],
        };
        let mut errors = Vec::new();
        let mut cached: Option<(PathBuf, Session)> = None;
        for (graph, mode) in attempts {
            let session = match cached.take() {
                Some((p, s)) if p == graph => s,
                _ => match build_session(ep, graph, opts) {
                    Ok(s) => s,
                    Err(e) => {
                        errors.push(format!("{}: {e:#}", mode.label()));
                        continue;
                    }
                },
            };
            let mut m = SpriteModel { core: core.clone(), session, ep, mode: mode.clone(), last_shape: None };
            match m.warm_up() {
                Ok(()) => return Ok(m),
                Err(e) => {
                    errors.push(format!("{}: {}", mode.label(), short(&format!("{e:#}"))));
                    cached = Some((graph.to_path_buf(), m.session));
                }
            }
        }
        bail!("{} failed: {}", ep.label(), errors.join(" | "))
    }

    pub fn describe(&self) -> String {
        format!("{} ({})", self.ep.label(), self.mode.label())
    }

    fn runner(&mut self, batch: usize, m: usize) -> Result<Runner<'_>> {
        Ok(match self.mode.clone() {
            CacheMode::Fixed(mem) => {
                if mem.is_some() && self.last_shape != Some((batch, m)) {
                    self.last_shape = Some((batch, m));
                    self.shrink_arena()?;
                }
                // The provider's arena never hands freed regions back to the device, and every chunk shape (rows x
                // cache length) leaves regions of its own size; with the row budget (22..24 rows at 16 px, 4..8 at
                // 32) they add up past the arena's cap. When a cache does not fit, release the free regions (a tiny
                // run with arena shrinkage) and try again.
                let sets = match cache_sets(&self.session, mem.as_ref(), &self.core, batch, m) {
                    Ok(s) => s,
                    Err(first) if mem.is_some() => {
                        self.shrink_arena().map_err(|e| anyhow!("{first}; freeing the arena failed: {e}"))?;
                        cache_sets(&self.session, mem.as_ref(), &self.core, batch, m).map_err(|e| anyhow!("{e} (also after freeing the arena's unused memory)"))?
                    }
                    Err(e) => return Err(e),
                };
                Runner::Fixed(StaticRun::new(&mut self.session, sets, &self.core, batch, m)?)
            }
            CacheMode::Growing(mem) => Runner::Growing(GrowRun::new(&mut self.session, mem, &self.core, batch)?),
        })
    }

    /// One 1-row step of the static graph (CPU tensors) with `memory.enable_memory_arena_shrinkage`: at the end of
    /// the run ONNX Runtime returns the arena's completely free regions to the device.
    fn shrink_arena(&mut self) -> Result<()> {
        let er = |e: ort::Error| anyhow!("{e}");
        let c = &self.core;
        let mut opts = ort::session::RunOptions::new().map_err(er)?;
        opts.set("memory.enable_memory_arena_shrinkage", "gpu:0").map_err(er)?;
        let mut x = vec![0f32; c.d];
        c.write_pixel(&mut x, c.bos, 0, 0, EDGE_UNKNOWN);
        let mut inputs: Vec<(Cow<'static, str>, SessionInputValue<'static>)> = vec![
            ("x".into(), Tensor::from_array(([1usize, 1, c.d], x)).map_err(er)?.into()),
            ("pos".into(), Tensor::from_array(([1usize], vec![0i64])).map_err(er)?.into()),
        ];
        for (past, _) in cache_names(c.n_layers) {
            inputs.push((past.into(), Tensor::from_array(([1usize, c.n_heads, 4, c.hd], vec![0f32; c.n_heads * 4 * c.hd])).map_err(er)?.into()));
        }
        self.session.run_with_options(inputs, &opts).map_err(er)?;
        Ok(())
    }

    fn warm_up(&mut self) -> Result<()> {
        let core = self.core.clone();
        let mut x = vec![0f32; core.d];
        let mut run = self.runner(1, 4)?;
        core.write_pixel(&mut x, core.bos, 0, 0, EDGE_UNKNOWN);
        run.step(&x, 1)?;
        core.write_pixel(&mut x, core.empty, 0, 1, EDGE_UNKNOWN);
        run.step(&x, 1)?;
        if !run.logits().iter().all(|v| v.is_finite()) {
            bail!("the backend returned non-finite logits");
        }
        Ok(())
    }

    /// Default sprites per network batch for this provider and size (all of a normal request on a GPU).
    pub fn default_batch(&self, size: usize) -> usize {
        default_batch_for(self.ep.is_gpu(), size)
    }

    /// n sprites of size 16, 24 or 32 with split guidance on `p.vector` (`g_shape` / `g_colour`; plain when both
    /// are 1 or there is no caption) for every sprite. `cancel` stops between pixels; `progress` is called after
    /// every pixel.
    pub fn generate(&mut self, p: &GenParams, cancel: &AtomicBool, progress: &mut dyn FnMut(&Progress)) -> Result<GenOutput> {
        if let Some(v) = &p.vector {
            if v.len() != self.core.text_dim {
                bail!("caption vector has {} numbers, the model wants {}", v.len(), self.core.text_dim);
            }
        }
        let shared = Arc::new(self.core.strategy_split(p.vector.as_deref(), p.g_shape, p.g_colour));
        self.generate_with(p, &|_, _| shared.clone(), cancel, progress)
    }

    /// `generate` of engine.js: n sprites, sprite `id` drawn with `strategy_for(id, view)` (`p.vector`,
    /// `g_shape` and `g_colour` are not used here). Chunks hold as many sprites as fit in `p.rows` network rows
    /// (0: `p.batch` sprites of the first sprite's strategy). Sprites the filter rejects (empty canvas, a detached
    /// piece of 1-2 px; off with a mask) are drawn again in the next round, as in the demo.
    pub fn generate_with(
        &mut self,
        p: &GenParams,
        strategy_for: &dyn Fn(usize, View) -> Arc<Strategy>,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(&Progress),
    ) -> Result<GenOutput> {
        let t0 = Instant::now();
        if ![16, 24, 32].contains(&p.size) {
            bail!("size must be 16, 24 or 32");
        }
        let s = if p.size == 24 { 32 } else { p.size };
        let w = p.size;
        let o = (s - w) / 2;
        let view = View { s, o, w };
        let mut win = vec![false; s * s];
        for y in o..o + w {
            for x in o..o + w {
                win[y * s + x] = true;
            }
        }
        let mask: Option<Vec<u8>> = match &p.mask {
            Some(m) => {
                if m.len() != w * w {
                    bail!("the silhouette has {} pixels, a {w}x{w} sprite needs {}", m.len(), w * w);
                }
                let mut c = vec![0u8; s * s];
                for y in 0..w {
                    for x in 0..w {
                        c[(y + o) * s + x + o] = u8::from(m[y * w + x] != 0);
                    }
                }
                Some(c)
            }
            None => None,
        };
        let edge = edge_flags(mask.as_deref(), s);
        let use_filter = p.filter && mask.is_none();
        let chunk = if p.batch == 0 { 8 } else { p.batch };
        let mut rng = Mulberry32::new(p.seed);
        let mut images: Vec<Option<Vec<i16>>> = vec![None; p.n];
        let mut stats = GenStats::default();
        let mut pending: Vec<usize> = (0..p.n).collect();
        let mut done = 0;
        while !pending.is_empty() && stats.rounds < 50 {
            stats.rounds += 1;
            let mut again = Vec::new();
            let mut i = 0;
            while i < pending.len() {
                // pack sprites while their rows fit the budget (at least one sprite per chunk)
                let (mut ids, mut sts) = (Vec::new(), Vec::new());
                let (mut budget, mut used) = (0usize, 0usize);
                while i < pending.len() {
                    let st = strategy_for(pending[i], view);
                    if budget == 0 {
                        budget = if p.rows > 0 { p.rows } else { chunk * st.rows.len() };
                    }
                    if !ids.is_empty() && used + st.rows.len() > budget {
                        break;
                    }
                    used += st.rows.len();
                    ids.push(pending[i]);
                    sts.push(st);
                    i += 1;
                }
                let rng_before = rng.clone();
                let mut chunk = |this: &mut Self, rng: &mut Mulberry32, stats: &mut GenStats| {
                    this.draw_chunk(&sts, view, &win, mask.as_deref(), &edge, p, rng, cancel, stats, &mut |t, cv| {
                        progress(&Progress { ids: &ids, t, canvases: cv, view, done, total: p.n })
                    })
                };
                let got = match chunk(self, &mut rng, &mut stats) {
                    // the device arena ran out in the middle of a chunk: free it and draw the chunk again with the
                    // same random numbers (the result is the same as if it had fitted)
                    Err(e) if is_arena_error(&e) && matches!(self.mode, CacheMode::Fixed(Some(_))) => {
                        self.shrink_arena()?;
                        rng = rng_before;
                        chunk(self, &mut rng, &mut stats).map_err(|e2| anyhow!("{e2} (also after freeing the arena's unused memory; first: {e})"))?
                    }
                    r => r?,
                };
                for (k, &id) in ids.iter().enumerate() {
                    stats.sampled += 1;
                    if use_filter && rejected(&got[k], s) {
                        stats.rejected += 1;
                        again.push(id);
                    } else {
                        images[id] = Some(crop(&got[k], view));
                        done += 1;
                    }
                }
            }
            pending = again;
        }
        if !pending.is_empty() {
            bail!("{} sprites still rejected after 50 rounds", pending.len());
        }
        Ok(GenOutput { images: images.into_iter().map(Option::unwrap).collect(), size: p.size, stats, secs: t0.elapsed().as_secs_f64() })
    }

    /// `drawChunk` of engine.js: one sprite per strategy on the S x S canvas, in parallel; every strategy owns
    /// `rows.len()` contiguous rows of the batch.
    #[allow(clippy::too_many_arguments)]
    fn draw_chunk(
        &mut self,
        strategies: &[Arc<Strategy>],
        view: View,
        win: &[bool],
        mask: Option<&[u8]>,
        edge: &[u8],
        p: &GenParams,
        rng: &mut Mulberry32,
        cancel: &AtomicBool,
        stats: &mut GenStats,
        on_pixel: &mut dyn FnMut(usize, &[Vec<i16>]),
    ) -> Result<Vec<Vec<i16>>> {
        let c = self.core.clone();
        let (d, kk, e) = (c.d, c.k, c.empty);
        let s = view.s;
        let n = s * s;
        let b = strategies.len();
        let mut bases = Vec::with_capacity(b);
        let mut owner = Vec::new(); // row -> sprite
        for (i, st) in strategies.iter().enumerate() {
            bases.push(owner.len());
            owner.extend(std::iter::repeat_n(i, st.rows.len()));
        }
        let bb = owner.len();
        let t0 = 2 + s + 1;
        // prefix + BOS pixel for every row: (bb, t0, d)
        let mut x0 = vec![0f32; bb * t0 * d];
        for j in 0..bb {
            let base = j * t0 * d;
            let i = owner[j];
            c.write_prefix(&mut x0[base..base + (2 + s) * d], &strategies[i].rows[j - bases[i]], s, mask);
            c.write_pixel(&mut x0[base + (t0 - 1) * d..base + t0 * d], c.bos, 0, 0, edge[0] as usize);
        }
        let ts = Instant::now();
        let mut run = self.runner(bb, 2 + s + n)?;
        run.prefix(&x0, bb, t0, d)?;
        drop(x0);
        let mut xt = vec![0f32; bb * d];
        let mut out = vec![vec![-1i16; n]; b];
        let mut tok = vec![0usize; b];
        let mut ctx = Ctx { k: kk, temperature: p.temperature, top_p: p.top_p, col: vec![0f64; kk], order: Vec::with_capacity(kk), w: Vec::new() };
        for t in 0..n {
            if cancel.load(Ordering::Relaxed) {
                return Err(crate::cancel::Cancelled.into());
            }
            let logits = run.logits();
            let force_empty = !win[t] || mask.is_some_and(|m| m[t] == 0);
            let force_fill = mask.is_some_and(|m| m[t] != 0);
            for i in 0..b {
                let pe = if force_empty {
                    1.0
                } else {
                    let pe = strategies[i].pixel(logits, bases[i], t, &mut ctx);
                    if force_fill { 0.0 } else { pe }
                };
                tok[i] = if rng.next_f64() < pe { e } else { draw(&ctx.col, rng.next_f64()) };
                out[i][t] = if tok[i] == e { -1 } else { tok[i] as i16 };
            }
            on_pixel(t, &out);
            if t + 1 == n {
                break;
            }
            let (r, cc) = ((t + 1) / s, (t + 1) % s);
            for j in 0..bb {
                c.write_pixel(&mut xt[j * d..(j + 1) * d], tok[owner[j]], r, cc, edge[t + 1] as usize);
            }
            run.step(&xt, 1)?;
        }
        stats.steps += run.steps();
        stats.step_secs += ts.elapsed().as_secs_f64();
        stats.chunks += 1;
        stats.max_rows = stats.max_rows.max(bb);
        Ok(out)
    }
}

/// An allocation failure of the provider's memory arena (the CUDA arena is capped, see `cuda_mem_limit_mb`).
fn is_arena_error(e: &anyhow::Error) -> bool {
    let s = format!("{e:#}");
    s.contains("BFCArena") || s.contains("Available memory of") || s.contains("Failed to allocate memory")
}

fn short(s: &str) -> String {
    let one = s.replace('\n', " ");
    if one.chars().count() > 400 { format!("{}…", one.chars().take(400).collect::<String>()) } else { one }
}

/// Default sprites per network batch: 8 at size 16, else 4 on a GPU and 2 on the CPU.
fn default_batch_for(is_gpu: bool, size: usize) -> usize {
    match (is_gpu, size) {
        (_, 16) => 8,
        (true, _) => 4,
        (false, _) => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::default_batch_for;

    #[test]
    fn default_batch_by_provider_and_size() {
        for gpu in [true, false] {
            assert_eq!(default_batch_for(gpu, 16), 8, "size 16 is the same on every provider (gpu={gpu})");
        }
        assert_eq!(default_batch_for(true, 24), 4);
        assert_eq!(default_batch_for(true, 32), 4);
        assert_eq!(default_batch_for(false, 24), 2);
        assert_eq!(default_batch_for(false, 32), 2);
    }
}
