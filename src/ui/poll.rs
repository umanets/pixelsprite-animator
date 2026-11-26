//! Messages from the model worker and the animation thread, starting a generation, saving.

use crate::*;

impl App {
    // ------------------------------------------------------------------------------------------- messages
    pub(crate) fn poll(&mut self, ctx: &egui::Context) {
        self.poll_library_scan(ctx);
        self.poll_png_pick();
        self.poll_jobs(ctx);
        while let Ok(msg) = self.worker.rx.try_recv() {
            match msg {
                Msg::Status(s) => self.status = s,
                Msg::Loaded { ep, describe, fallbacks, secs, palette, lexicon, presets, outline_words } => {
                    self.outline_words = outline_words;
                    self.lexicon = lexicon;
                    self.presets = presets;
                    // the animation model follows the sprite model's provider (with the CPU as the fallback)
                    let order = if ep == Ep::Cpu { vec![Ep::Cpu] } else { vec![ep, Ep::Cpu] };
                    let limit = self.cfg.anim_cuda_mem_limit_mb.unwrap_or(ANIM_CUDA_MEM_LIMIT_MB);
                    self.jobs.anim.set_providers(order, SessionOpts { cuda_mem_limit_mb: limit, ..Default::default() });
                    self.loading = false;
                    self.ep_active = Some(ep);
                    self.ep_describe = describe;
                    self.ep_fallbacks = fallbacks;
                    self.palette = Some(palette);
                    self.load_error = None;
                    self.status = format!("Ready ({secs:.1} s to load).");
                }
                Msg::LoadFailed(e) => {
                    self.loading = false;
                    self.ep_active = None;
                    self.load_error = Some(e.clone());
                    self.status = "The model could not be loaded.".into();
                }
                Msg::Planned { id, variants, learned } => {
                    self.learned_kind.extend(learned);
                    if let Some(run) = self.runs.iter_mut().find(|r| r.id == id) {
                        for (c, v) in run.cards.iter_mut().zip(variants) {
                            c.variant = Some(v);
                        }
                    }
                }
                Msg::Progress { id, done, total, t, n_px, first_card, partial } => {
                    self.progress = (t + 1, n_px, done, total);
                    if let Some(run) = self.runs.iter_mut().find(|r| r.id == id) {
                        for (i, px) in partial {
                            if let Some(c) = run.cards.get_mut(first_card + i) {
                                if c.sprite.is_none() {
                                    c.partial = Some(px);
                                    c.dirty = true;
                                }
                            }
                        }
                    }
                }
                Msg::Generated(r) => {
                    self.running = None;
                    self.learned_kind.extend(r.learned.clone());
                    let pal = self.palette.clone().unwrap_or_default();
                    if let Some(run) = self.runs.iter_mut().find(|x| x.id == r.id) {
                        let first = usize::from(r.source.is_some());
                        let all = r.source.clone().into_iter().chain(r.images.iter().cloned());
                        for (i, px) in all.enumerate() {
                            if let Some(c) = run.cards.get_mut(i) {
                                c.sprite = Some(Sprite::new(r.params.size, px, pal.clone()));
                                c.partial = None;
                                c.dirty = true;
                            }
                        }
                        if let Some(v) = &r.variants {
                            for (c, v) in run.cards.iter_mut().skip(first).zip(v) {
                                c.variant = Some(v.clone());
                            }
                        }
                        let per_step = if r.stats.steps > 0 { r.stats.step_secs * 1e3 / r.stats.steps as f64 } else { 0.0 };
                        self.ms_per_step = Some(per_step);
                        run.summary = format!("{} · {:.1} s on {}", r.summary, r.secs, r.ep.label());
                        run.meta = format!(
                            "{} chunk(s), up to {} rows per step · {:.1} ms/step · encode {:.0} ms{}",
                            r.stats.chunks,
                            r.stats.max_rows,
                            per_step,
                            r.encode_secs * 1e3,
                            match r.caption_exact {
                                Some(false) => " · \u{201c}one caption\u{201d} = averaged word vectors (as a fresh demo page)",
                                _ => "",
                            }
                        );
                        run.class = Some(r.class);
                        run.done = true;
                        self.status = format!("Drew {} sprites in {:.1} s.", run.cards.len(), r.secs);
                        if self.selected.is_none() {
                            self.selected = Some(Selection::Generated { run: run.id, idx: 0 });
                            self.anim.kind = KindChoice::Auto;
                            self.anim.animator = if self.jobs.anim.available() { AnimatorChoice::Onnx } else { AnimatorChoice::Mock };
                        }
                    }
                }
                Msg::GenFailed { id, error } => {
                    self.running = None;
                    if let Some(run) = self.runs.iter_mut().find(|x| x.id == id) {
                        run.done = true;
                        run.error = Some(error.to_string());
                        // cards that were never drawn would stay "queued" for ever
                        run.cards.retain(|c| c.sprite.is_some());
                    }
                    self.runs.retain(|r| !(r.id == id && r.cards.is_empty()));
                    self.status = match error {
                        Failure::Cancelled => "Stopped.".into(),
                        Failure::Failed(why) => {
                            self.gen_error = Some(explain_gen_failure(&why, self.size, self.count));
                            format!("Generation failed: {why}")
                        }
                    };
                }
                Msg::CudnnDone(r) => {
                    self.cudnn_busy = false;
                    self.cudnn_done = Some(match r {
                        Ok(p) => format!("CUDA libraries installed in {}. Restart the app to use CUDA.", p.display()),
                        Err(e) => format!("CUDA libraries download failed: {e}"),
                    });
                }
            }
        }
    }

    /// Results and progress of the animation thread (the ONNX animator). Messages of requests the UI no longer
    /// waits for are dropped (their clips are in the model's cache anyway).
    pub(crate) fn poll_jobs(&mut self, ctx: &egui::Context) {
        while let Ok(m) = self.jobs.rx.try_recv() {
            let mine = |t: &Option<Ticket>| t.as_ref().is_some_and(|t| t.id == m.id);
            let dirs_mine = self.dirs_pending.as_ref().is_some_and(|(t, _)| t.id == m.id);
            match m.event {
                // ids are unique across kinds, so progress is routed by whose request it belongs to
                JobEvent::Progress(p) if mine(&self.anim.pending) => self.anim.progress = Some(p),
                JobEvent::Progress(p) if mine(&self.scene.pending) => self.scene.progress = Some(p),
                JobEvent::Progress(p) if dirs_mine => {
                    self.last_export = Some((format!("Drawing the 8-direction sheet: {} ({:.0} %)", p.stage, p.frac * 100.0), exports_dir()));
                }
                JobEvent::Done(JobResult::Preview(r)) if mine(&self.anim.pending) => {
                    self.anim.pending = None;
                    self.anim.progress = None;
                    match r {
                        Ok(a) => {
                            self.anim.textures = a.frames.iter().enumerate().map(|(i, f)| texture(ctx, &format!("frame{i}"), f)).collect();
                            self.anim.result = Some(a);
                            self.anim.t0 = ctx.input(|i| i.time);
                        }
                        Err(Failure::Cancelled) => self.anim.stopped = true,
                        Err(Failure::Failed(why)) => self.anim.error = Some(why),
                    }
                }
                JobEvent::Done(JobResult::Scene(r)) if mine(&self.scene.pending) => {
                    self.scene.pending = None;
                    self.scene.progress = None;
                    match r {
                        Ok(s) => self.install_scene(ctx, s),
                        Err(Failure::Cancelled) => {
                            self.scene.stopped = true;
                            self.scene.note = Some("stopped".into());
                        }
                        Err(Failure::Failed(why)) => self.scene.note = Some(why),
                    }
                }
                JobEvent::Done(JobResult::Directions(r)) if dirs_mine => self.finish_directions(r),
                _ => {}
            }
        }
    }

    pub(crate) fn start_generation(&mut self) {
        self.gen_error = None;
        if self.auto_seed {
            self.seed = rand_u32() % 1_000_000;
        }
        self.count = self.count.clamp(1, MAX_COUNT);
        if self.mode == Mode::Draw && self.pad_size != self.size {
            self.new_pad(self.size);
        }
        let id = self.next_id;
        self.next_id += 1;
        // a generator with the outline words gets the command from the checkbox; the same words typed in the
        // caption are taken out (they would be mixed as if they were things to draw)
        let outline = self.outline_words.then_some(self.outline);
        let caption = if outline.is_some() { strip_outline_words(&self.caption) } else { self.caption.trim().to_string() };
        let params = GenParams {
            vector: None,
            n: self.count,
            size: self.size,
            seed: self.seed,
            g_shape: self.g_shape,
            g_colour: self.g_colour,
            temperature: self.temperature,
            top_p: self.top_p,
            filter: self.filter,
            batch: 0,
            rows: self.batching.rows(self.size),
            mask: (self.mode == Mode::Draw).then(|| self.pad.clone()),
        };
        let mut cards: Vec<Card> = Vec::new();
        if self.mode == Mode::Same {
            cards.push(Card::new(CardTag::Source));
        }
        cards.extend((0..self.count).map(|k| Card::new(CardTag::Index(k))));
        self.runs.insert(
            0,
            Run {
                id,
                caption: format!("{caption}{}", animator_ui::worker::outline_suffix(outline)),
                size: self.size,
                seed: self.seed,
                cards,
                class: None,
                summary: String::new(),
                meta: "drawing\u{2026}".into(),
                done: false,
                error: None,
            },
        );
        self.runs.truncate(12);
        self.progress = (0, 0, 0, self.count);
        self.running = Some((id, Instant::now()));
        let mix = MixOptions { enabled: self.variants_on.clone(), overrides: self.kind_override.clone(), learned: self.learned_kind.clone(), encoder_ready: self.encoder_ready, outline };
        self.worker.send(Cmd::Generate(GenRequest { id, caption, params, mode: self.mode, mix }));
    }

    pub(crate) fn new_pad(&mut self, size: usize) {
        self.pad_size = size;
        self.pad = vec![0; size * size];
        self.pad_stroke = None;
    }

    /// The demo's "shape ->" button: colour this sprite's silhouette again (draw mode at its size).
    pub(crate) fn use_shape(&mut self, s: &Sprite) {
        self.size = s.size;
        self.mode = Mode::Draw;
        self.pad_size = s.size;
        self.pad = s.px.iter().map(|&v| u8::from(v >= 0)).collect();
        if !self.count_touched {
            self.count = default_count(s.size);
        }
    }

    /// Saves the selected generated sprite ([`SaveTarget::Sprite`]) or a run's sheet ([`SaveTarget::Sheet`]: the
    /// given run, or the selected sprite's run), with the demo's file names, into exports/.
    pub(crate) fn save_selected(&mut self, target: SaveTarget) {
        let sheet_run = match target {
            SaveTarget::Sheet(r) => r,
            SaveTarget::Sprite => None,
        };
        let run_id = match (sheet_run, self.selected) {
            (Some(r), _) => r,
            (None, Some(Selection::Generated { run, .. })) => run,
            _ => {
                self.last_save = Some(("Select a generated sprite first.".into(), exports_dir()));
                return;
            }
        };
        let Some(run) = self.runs.iter().find(|r| r.id == run_id) else { return };
        let dir = exports_dir();
        let scale = self.save_scale;
        let selected = self.selected;
        let result: anyhow::Result<PathBuf> = (|| {
            std::fs::create_dir_all(&dir)?;
            if target == SaveTarget::Sprite {
                let Some(Selection::Generated { idx, .. }) = selected else { anyhow::bail!("no sprite selected") };
                let card = run.cards.get(idx).ok_or_else(|| anyhow::anyhow!("no such sprite"))?;
                let s = card.sprite.as_ref().ok_or_else(|| anyhow::anyhow!("the sprite is still being drawn"))?;
                let p = dir.join(save::sprite_file_name(&run.caption, run.size, run.seed, &card.tag.file_tag(), scale));
                save::save_sprite_png(s, scale, &p)?;
                Ok(p)
            } else {
                let sprites: Vec<Sprite> = run.cards.iter().filter_map(|c| c.sprite.clone()).collect();
                if sprites.len() < run.cards.len() {
                    anyhow::bail!("the run is still being drawn");
                }
                let p = dir.join(save::sheet_file_name(&run.caption, run.size, run.seed, scale));
                save::save_sheet_png(&sprites, scale, &p)?;
                Ok(p)
            }
        })();
        self.last_save = Some(match result {
            Ok(p) => (format!("Saved {}", p.display()), p),
            Err(e) => (format!("Save failed: {e:#}"), dir),
        });
    }
}

/// A failed generation in words the user can act on (the raw error stays in the status bar).
pub(crate) fn explain_gen_failure(why: &str, size: usize, count: usize) -> String {
    let short: String = why.lines().next().unwrap_or(why).chars().take(160).collect();
    if why.contains("BFCArena") || why.contains("Available memory") || why.to_lowercase().contains("out of memory") {
        format!(
            "Nothing was drawn: not enough GPU memory for {count} sprites at {size}\u{d7}{size}. Lower Count or the size, or raise \
             cuda_mem_limit_mb in animator_ui.json (and restart)."
        )
    } else {
        format!("Nothing was drawn: {short}")
    }
}

/// The caption without outline commands typed by hand ("..., no outline", "with outline", "without outline").
pub(crate) fn strip_outline_words(caption: &str) -> String {
    let mut out = caption.to_string();
    for phrase in ["without outline", "with outline", "no outline"] {
        while let Some(i) = out.to_lowercase().find(phrase) {
            // to_lowercase keeps byte offsets for the ASCII phrase only if the text before it kept its length
            if !out.is_char_boundary(i) || !out.is_char_boundary(i + phrase.len()) || !out[i..i + phrase.len()].eq_ignore_ascii_case(phrase) {
                break;
            }
            out.replace_range(i..i + phrase.len(), "");
        }
    }
    let cleaned: Vec<&str> = out.split(',').map(str::trim).filter(|p| !p.is_empty()).collect();
    cleaned.join(", ")
}

#[cfg(test)]
mod outline_tests {
    use super::strip_outline_words;

    #[test]
    fn typed_outline_commands_are_removed() {
        assert_eq!(strip_outline_words("a knight, no outline"), "a knight");
        assert_eq!(strip_outline_words("a knight with a sword, With Outline "), "a knight with a sword");
        assert_eq!(strip_outline_words("red potion without outline"), "red potion");
        assert_eq!(strip_outline_words("лицар, no outline, щит"), "лицар, щит");
        assert_eq!(strip_outline_words("outline of a map"), "outline of a map");
    }
}
