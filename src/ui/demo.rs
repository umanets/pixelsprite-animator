//! The scripted demo mode (screenshots, driving the UI through a script) used for the Figma comparison runs.

use crate::*;

impl App {
    pub(crate) fn screenshot(&self, ctx: &egui::Context, name: String, saved: Arc<AtomicBool>) {
        let path = exports_dir().join(name);
        ctx.request_screenshot(move |img| {
            if let Some(dir) = path.parent() {
                if let Err(e) = std::fs::create_dir_all(dir) {
                    eprintln!("screenshot: cannot create {}: {e}", dir.display());
                }
            }
            let mut buf = image::RgbaImage::new(img.size[0] as u32, img.size[1] as u32);
            for (i, c) in img.pixels.iter().enumerate() {
                buf.put_pixel((i % img.size[0]) as u32, (i / img.size[0]) as u32, image::Rgba(c.to_array()));
            }
            // the demo script waits for `saved`, so it only moves on after a file really exists
            match buf.save(&path) {
                Ok(()) => saved.store(true, Ordering::SeqCst),
                Err(e) => eprintln!("screenshot {}: {e}", path.display()),
            }
        });
        ctx.request_repaint();
    }

    pub(crate) fn drive_demo(&mut self, ctx: &egui::Context) {
        let Some(mut d) = self.demo.take() else { return };
        let out = d.get("out").unwrap_or("ui_demo").to_string();
        if let Some(fps) = d.get("fps").and_then(|s| s.parse::<f32>().ok()) {
            self.anim.fps = fps;
        }
        match d.stage {
            0 if d.get("png").is_some() && !self.loading => {
                self.png_path = d.get("png").unwrap_or_default().to_string();
                // prep=asis|outline|inline (the recommended one without it), prep=ask leaves the dialog open
                match d.get("prep") {
                    Some("ask") => self.load_png(None),
                    Some("asis") => self.load_png(Some(Some(PngPrep::AsIs))),
                    Some("outline") => self.load_png(Some(Some(PngPrep::Outline))),
                    Some("inline") => self.load_png(Some(Some(PngPrep::Inline))),
                    _ => self.load_png(Some(None)),
                }
                if let Some(e) = &self.png_error {
                    eprintln!("demo: {e}");
                }
                self.demo_animator(&d);
                if let Some(a) = d.get("action") {
                    self.anim.action = a.to_string();
                    self.anim.n_frames = d.get("frames").and_then(|s| s.parse().ok()).unwrap_or(8);
                }
                self.anim.azimuth = d.get("az").and_then(|s| s.parse().ok()).unwrap_or(90);
                d.stage = 2;
                d.t = Instant::now();
            }
            0 if d.get("gen") == Some("0") => {
                d.saved.store(true, Ordering::SeqCst);
                d.stage = 3;
            }
            0 if !self.loading && self.ep_active.is_some() => {
                self.caption = d.get("caption").unwrap_or("knight").to_string();
                self.size = d.get("size").and_then(|s| s.parse().ok()).unwrap_or(16);
                self.count = d.get("count").and_then(|s| s.parse().ok()).unwrap_or(default_count(self.size));
                self.seed = d.get("seed").and_then(|s| s.parse().ok()).unwrap_or(1);
                // stage-1 keys: the demo's controls
                self.mode = match d.get("mode") {
                    Some("same") => Mode::Same,
                    Some("draw") => Mode::Draw,
                    _ => Mode::New,
                };
                if let Some(m) = d.get("mask") {
                    self.pad_size = self.size;
                    self.pad = demo_mask(m, self.size);
                }
                if let Some(v) = d.get("variants") {
                    self.variants_on = v.split(',').filter_map(VariantKind::parse).collect();
                }
                for w in d.get("flip").unwrap_or("").split(',').map(str::trim).filter(|w| !w.is_empty()) {
                    let k = self.lexicon.as_ref().and_then(|l| l.kind.get(w).copied()).unwrap_or(ConceptKind::Object);
                    self.kind_override.insert(w.to_string(), k.flipped());
                }
                if let Some(r) = d.get("rows").and_then(|s| s.parse().ok()) {
                    self.batching = Batching::Custom(r);
                }
                if d.get("more") == Some("1") {
                    self.more_open = Some(true);
                }
                self.encoder_ready = d.get("exact") == Some("1");
                if let Some(o) = d.get("outline") {
                    self.outline = o == "1";
                }
                if let Some(s) = d.get("scale").and_then(|s| s.parse().ok()) {
                    self.save_scale = s;
                }
                self.start_generation();
                d.stage = 1;
            }
            // screenshot the "Generating" state while the worker draws, then stop
            1 if d.get("genshot") == Some("1") && self.running.is_some() && d.t.elapsed().as_secs_f32() > 2.0 => {
                self.screenshot(ctx, format!("{out}_gen.png"), d.saved.clone());
                d.stage = 11;
            }
            11 if d.saved.load(Ordering::SeqCst) => {
                self.worker.stop();
                d.stage = 5;
            }
            1 if self.running.is_none() && self.runs.first().is_some_and(|r| r.done) => {
                self.selected = Some(Selection::Generated { run: self.runs[0].id, idx: d.get("idx").and_then(|s| s.parse().ok()).unwrap_or(0) });
                if let Some(e) = &self.runs[0].error {
                    eprintln!("demo: generation failed: {e}");
                }
                eprintln!("demo run: {} · {}", self.runs[0].summary, self.runs[0].meta);
                if d.get("save") == Some("1") {
                    for what in [SaveTarget::Sprite, SaveTarget::Sheet(None)] {
                        self.save_selected(what);
                        if let Some((msg, _)) = &self.last_save {
                            eprintln!("demo save: {msg}");
                        }
                    }
                }
                if let Some(a) = d.get("action") {
                    self.anim.action = a.to_string();
                    self.anim.n_frames = d.get("frames").and_then(|s| s.parse().ok()).unwrap_or(8);
                }
                self.anim.azimuth = d.get("az").and_then(|s| s.parse().ok()).unwrap_or(90);
                self.demo_animator(&d);
                d.stage = 2;
                d.t = Instant::now();
            }
            // the model draws in the background: wait for the preview and the scene (up to `wait` seconds)
            2 if d.t.elapsed().as_secs_f32() > 1.5
                && (d.t.elapsed().as_secs_f32() > d.get("wait").and_then(|s| s.parse().ok()).unwrap_or(600.0)
                    || (self.anim.pending.is_none() && self.scene.pending.is_none())) =>
            {
                if self.anim.pending.is_some() || self.scene.pending.is_some() {
                    eprintln!("demo: the animation model is still drawing after the wait");
                }
                match (&self.scene.scene, &self.scene.note) {
                    (Some(s), _) => eprintln!("demo scene: {} shots, {:.1} ticks", s.shots.len(), s.total_ticks()),
                    (None, note) => eprintln!("demo scene: none ({note:?})"),
                }
                if d.get("export") == Some("1") {
                    for what in [ExportKind::Sheet, ExportKind::Gif, ExportKind::Directions] {
                        self.export_selected(what);
                        if let Some((msg, _)) = &self.last_export {
                            eprintln!("demo export: {msg}");
                        }
                    }
                    self.demo_scene_export();
                }
                self.screenshot(ctx, format!("{out}_1.png"), d.saved.clone());
                d.stage = 3;
            }
            3 if d.saved.load(Ordering::SeqCst) => match d.get("lib").map(str::to_string) {
                Some(name) => {
                    let found = self.lib_entries.iter().chain(self.lib_next.iter().flatten()).find(|e| e.name == name).cloned();
                    if let Some(e) = found {
                        d.saved.store(false, Ordering::SeqCst);
                        match LibraryData::load(&e) {
                            Ok(data) => {
                                if let Some(z) = d.get("libsize").and_then(|s| s.parse().ok()).filter(|z| data.sizes().contains(z)) {
                                    self.anim.lib_size = z;
                                }
                                self.anim.action = d.get("libaction").unwrap_or("walk").to_string();
                                self.anim.n_frames = data.entry.actions.iter().find(|a| a.name == self.anim.action).map(|a| a.frames).unwrap_or(8);
                                self.lib_data = Some(Arc::new(data));
                                self.selected = Some(Selection::Library);
                                self.anim.animator = AnimatorChoice::Library;
                                self.anim.kind = KindChoice::Auto;
                                self.anim.azimuth = d.get("libaz").and_then(|s| s.parse().ok()).unwrap_or(90);
                            }
                            Err(err) => eprintln!("demo: {name}: {err:#}"),
                        }
                        d.stage = 4;
                        d.t = Instant::now();
                    } else if self.lib_scan.is_none() {
                        eprintln!("demo: library entry {name} not found");
                        d.stage = 5;
                    }
                }
                None => d.stage = 6,
            },
            4 if d.t.elapsed().as_secs_f32() > 1.5 => {
                match (&self.scene.scene, &self.scene.note) {
                    (Some(s), _) => eprintln!("demo scene: {} shots, {:.1} ticks; qc {:?}", s.shots.len(), s.total_ticks(), s.qc.iter().map(|q| q.describe()).collect::<Vec<_>>()),
                    (None, note) => eprintln!("demo scene: none ({note:?})"),
                }
                if d.get("export") == Some("1") {
                    self.demo_scene_export();
                }
                self.screenshot(ctx, format!("{out}_2.png"), d.saved.clone());
                d.stage = 6;
            }
            6 if d.saved.load(Ordering::SeqCst) => {
                d.k = 0;
                d.stage = 8;
            }
            8 => {
                let times: Vec<f64> = d.get("scenet").unwrap_or("").split(',').filter_map(|s| s.trim().parse().ok()).collect();
                match (times.get(d.k), &self.scene.scene) {
                    (Some(&t), Some(s)) => {
                        self.scene.shown = s.shot_at(t * self.anim.fps as f64);
                        self.scene.playing = false;
                        d.saved.store(false, Ordering::SeqCst);
                        d.t = Instant::now();
                        d.stage = 9;
                    }
                    _ => d.stage = 5,
                }
            }
            9 if d.t.elapsed().as_secs_f32() > 0.4 => {
                self.screenshot(ctx, format!("{out}_scene_{}.png", d.k + 1), d.saved.clone());
                d.stage = 10;
            }
            10 if d.saved.load(Ordering::SeqCst) => {
                d.k += 1;
                d.stage = 8;
            }
            5 => {
                if d.get("exit") == Some("1") {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                d.stage = 7;
            }
            _ => {}
        }
        if d.stage != 7 {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        self.demo = Some(d);
    }

    /// Demo keys `animator=`, `kind=` and `refwalk=` (after the sprite is selected).
    pub(crate) fn demo_animator(&mut self, d: &Demo) {
        match d.get("animator") {
            Some("onnx") => {
                self.anim.animator = AnimatorChoice::Onnx;
                // the script stands in for the Animate and Build scene buttons
                self.anim.go = d.get("go") != Some("0");
                self.scene.go = d.get("go") != Some("0") && d.get("scene") != Some("0");
            }
            Some("mock") => self.anim.animator = AnimatorChoice::Mock,
            Some("library") if self.lib_data.is_some() => self.anim.animator = AnimatorChoice::Library,
            _ => {}
        }
        if let Some(k) = d.get("kind").and_then(Kind::parse) {
            self.anim.kind = KindChoice::Force(k);
        }
        if let Some(v) = d.get("ref") {
            let mut st = self.jobs.anim.settings();
            st.reference = match v {
                "front" => RefChoice::FrontRender,
                "original" => RefChoice::Original,
                _ => RefChoice::Auto,
            };
            self.jobs.anim.set_settings(st);
        }
        if let Some(v) = d.get("refwalk") {
            self.anim.ref_from_walk_front = v != "0";
            self.apply_anim_ref();
        }
    }
}
