//! The fight scene: building it when its inputs change, exporting it, and the dock that shows it.

use crate::*;

impl App {
    // ------------------------------------------------------------------------------------------ fight scene
    /// Rebuilds the fight scene when the character, animator, kind, size or QC switch changed; re-lays out the
    /// timeline (same clips) when only the speeds changed.
    pub(crate) fn refresh_scene(&mut self, ctx: &egui::Context) {
        let Some((sprite, _cap, class)) = self.selected_sprite() else {
            self.scene.key = None;
            self.scene.scene = None;
            self.scene.textures.clear();
            self.scene.note = None;
            self.scene.show = false;
            return;
        };
        let kind = self.effective_kind(&class);
        let speeds = (self.scene.walk_speed, self.scene.run_speed);
        let script = scene::fight_with(speeds.0, speeds.1);
        let key = SceneKey {
            sel: self.selected,
            lib_name: self.lib_data.as_ref().map(|d| d.entry.name.clone()),
            loaded: self.loaded_id,
            animator: self.anim.animator,
            kind,
            size: sprite.size,
            qc: self.scene.qc,
            onnx: self.onnx_tag(),
        };
        if self.scene.key.as_ref() == Some(&key) {
            if self.scene.laid_out != speeds {
                if let Some(s) = &mut self.scene.scene {
                    s.relayout(&script);
                    self.scene.shown = self.scene.shown.min(s.shots.len().saturating_sub(1));
                }
                self.scene.laid_out = speeds;
            }
            return;
        }
        self.scene.key = Some(key);
        self.scene.scene = None;
        self.scene.textures.clear();
        self.scene.note = None;
        self.scene.show = false;
        self.scene.last_export = None;
        self.scene.laid_out = speeds;
        self.scene.stopped = false;
        self.scene.progress = None;
        if let Some(t) = self.scene.pending.take() {
            t.stop();
        }
        let Some(kind) = kind else { return };
        if self.anim.animator == AnimatorChoice::Onnx {
            // every action of the scene exists for humans (no dataset QC); built on the animation thread
            self.scene.is_mock = false;
            self.scene.show = kind == Kind::Human;
            if kind == Kind::Human {
                let job = Job::Scene { sprite: sprite.clone(), kind, size: sprite.size, script, refk: self.onnx_refk() };
                self.scene.pending = Some(self.jobs.submit(job));
            }
            return;
        }
        let (is_mock, result) = {
            let animator = self.animator();
            let is_mock = animator.is_mock();
            // the dataset QC judges real motion; the mock only pushes the reference sprite around
            (is_mock, scene::build(&*animator, &sprite, kind, sprite.size, &script, self.scene.qc && !is_mock))
        };
        self.scene.is_mock = is_mock;
        match result {
            Ok(s) => self.install_scene(ctx, s),
            Err(why) => {
                // humans get a one-line reason; other kinds simply have no scene
                self.scene.show = kind == Kind::Human;
                self.scene.note = Some(why.summary());
            }
        }
    }

    pub(crate) fn install_scene(&mut self, ctx: &egui::Context, s: Scene) {
        self.scene.textures = s
            .clips
            .iter()
            .enumerate()
            .map(|(c, clip)| clip.frames.iter().enumerate().map(|(f, fr)| texture(ctx, &format!("scene{c}_{f}"), fr)).collect())
            .collect();
        self.scene.scene = Some(s);
        self.scene.show = true;
        self.scene.shown = 0;
        self.scene.t0 = ctx.input(|i| i.time);
    }

    pub(crate) fn export_scene(&mut self) {
        let Some(sc) = &self.scene.scene else { return };
        let Some((_, caption, class)) = self.selected_sprite() else { return };
        let kind = self.effective_kind(&class).map(Kind::label).unwrap_or("none");
        let dir = exports_dir();
        let tag = if self.scene.is_mock { "_MOCK" } else { "" };
        let reftag = self.ref_tag();
        let who = match (self.selected, &self.lib_data) {
            (Some(Selection::Library), Some(d)) => format!("{}_{}", d.entry.name, slug(&caption)),
            _ => slug(&caption),
        };
        let p = dir.join(format!("{who}_{}_{kind}_fight_scene_{}f{tag}{reftag}.gif", sc.size, sc.shots.len()));
        let result = std::fs::create_dir_all(&dir).map_err(anyhow::Error::from).and_then(|_| {
            let frames: Vec<Vec<u8>> = (0..sc.shots.len()).map(|i| sc.rgba(i)).collect();
            export::gif_rgba(&frames, sc.stage_w, sc.size, &sc.delays_cs(self.anim.fps), self.export_scale, true, &p)
        });
        self.scene.last_export = Some(match result {
            Ok(()) => (format!("Saved {}", p.display()), p),
            Err(e) => (format!("Export failed: {e}"), dir),
        });
    }

    pub(crate) fn demo_scene_export(&mut self) {
        if self.scene.scene.is_some() {
            self.export_scene();
            if let Some((msg, _)) = &self.scene.last_export {
                eprintln!("demo export: {msg}");
            }
        }
    }

    // ------------------------------------------------------------------------------------ fight scene dock
    /// The bottom dock of the mock: header, seconds ruler, clip track with the amber playhead, the two speed
    /// fields and the 250 × 168 stage preview on the right.
    pub(crate) fn scene_panel(&mut self, ui: &mut egui::Ui) {
        let p = ui.painter().clone();
        let ctx = ui.ctx().clone();
        let fps = self.anim.fps.max(1.0);
        let qc_hint = "Offer the scene only if walk, punch, spin_kick and run pass the dataset check of \
                       pilot3d/dataset/pack.py: an action is dropped when more than 30 % of its 32 px frames (all 8 \
                       directions) touch the border, jump in area outside 0.4–2.5x of the first frame, or fall into \
                       pieces. spin_kick fails it often (the leg leaves the canvas). Not applied to the mock.";
        // the dock's own 20 px inner padding
        ui.add_space(1.0);
        if let Some(t) = self.scene.pending.clone() {
            let p = self.scene.progress.clone().unwrap_or_default();
            ui.horizontal(|ui| {
                ui.add(egui::Label::new(kit::semi_txt(scene::fight().title, 14.0).color(theme::TEXT_PRIMARY)));
                ui.add(egui::Label::new(kit::mono_txt(p.stage, 12.0).color(theme::TEXT_MUTED)));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if kit::button_h(ui, "Stop", kit::Btn::Danger, 28.0).clicked() {
                        t.stop();
                    }
                    kit::progress(ui, ui.available_width(), p.frac);
                });
            });
            return;
        }
        if self.scene.stopped && self.scene.scene.is_none() {
            ui.horizontal(|ui| {
                ui.add(egui::Label::new(kit::semi_txt("Fight scene", 14.0).color(theme::TEXT_PRIMARY)));
                ui.add(egui::Label::new(kit::txt("stopped · finished clips are kept", 12.0).color(theme::TEXT_SECONDARY)));
                if kit::button(ui, "Draw the scene", kit::Btn::Ghost).clicked() {
                    self.scene.key = None;
                }
            });
            return;
        }
        let Some(sc) = &self.scene.scene else {
            ui.horizontal_wrapped(|ui| {
                ui.add(egui::Label::new(kit::semi_txt("Fight scene", 14.0).color(theme::TEXT_PRIMARY)));
                let note = self.scene.note.clone().unwrap_or_default();
                ui.add(egui::Label::new(kit::txt(format!("not available for this character: {note}"), 12.0).color(theme::WARN)));
                if self.anim.animator != AnimatorChoice::Onnx && kit::chip(ui, "dataset QC", self.scene.qc).on_hover_text(qc_hint).clicked() {
                    self.scene.qc = !self.scene.qc;
                }
            });
            return;
        };
        let total = sc.total_ticks();
        let now = ctx.input(|i| i.time);
        if self.scene.playing {
            self.scene.shown = sc.shot_at((now - self.scene.t0) * fps as f64);
            let next = sc.start_of(self.scene.shown) + sc.shots[self.scene.shown].ticks;
            let wait = ((next as f64 - ((now - self.scene.t0) * fps as f64).rem_euclid(total as f64)) / fps as f64).clamp(0.005, 1.0);
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(wait));
        }
        let shown = self.scene.shown.min(sc.shots.len().saturating_sub(1));
        let mut restart = false;
        let mut export = false;
        // ---- the right-hand stage preview first, so the left column knows its width
        let left_w = (ui.available_width() - 250.0 - 20.0).max(320.0);
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(vec2(left_w, ui.available_height()), egui::Layout::top_down(egui::Align::Min), |ui| {
                // ---- header (28 px)
                ui.horizontal(|ui| {
                    ui.add(egui::Label::new(kit::semi_txt(sc.title, 14.0).color(theme::TEXT_PRIMARY)));
                    ui.add_space(2.0);
                    ui.add(egui::Label::new(kit::mono_txt(format!("{} frames · {:.1} s", sc.shots.len(), total / fps), 12.0).color(theme::TEXT_MUTED)));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if kit::button_h(ui, if self.scene.playing { "Pause" } else { "Play" }, kit::Btn::Primary, 28.0).clicked() {
                            self.scene.playing = !self.scene.playing;
                            self.scene.t0 = now - sc.start_of(shown) as f64 / fps as f64;
                        }
                        if kit::button_wh(ui, format!("GIF ×{}", self.export_scale), kit::Btn::Ghost, 69.0, 28.0)
                            .on_hover_text("the whole loop, timed by the fps slider; scale = Export × in the right panel")
                            .clicked()
                        {
                            export = true;
                        }
                        if kit::button_wh(ui, "Restart", kit::Btn::Ghost, 76.0, 28.0).clicked() {
                            restart = true;
                        }
                        if self.anim.animator != AnimatorChoice::Onnx {
                            kit::checkbox(ui, &mut self.scene.qc, "dataset QC");
                            ui.add_space(4.0);
                        }
                    });
                });
                // ---- ruler (14 px): a tick a second, a label every 2 s
                let (ruler, _) = ui.allocate_exact_size(vec2(ui.available_width(), 14.0), Sense::hover());
                {
                    let secs = (total / fps).ceil().max(1.0);
                    let per = ruler.width() / secs as f32;
                    let mut s = 0;
                    while s as f32 <= secs {
                        let x = ruler.left() + per * s as f32;
                        let tall = s % 2 == 0;
                        p.line_segment([pos2(x, ruler.top()), pos2(x, ruler.top() + if tall { 7.0 } else { 4.0 })], Stroke::new(1.0, theme::BORDER_STRONG));
                        if tall {
                            kit::text_at(ui, pos2(x + 3.0, ruler.top()), &format!("{s}s"), theme::font_mono(10.0), theme::TEXT_MUTED);
                        }
                        s += 1;
                    }
                }
                // ---- clip track (56 px) with the amber playhead
                let (track, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 56.0), Sense::click());
                p.rect(track, theme::R_FIELD, theme::BG_INPUT, Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
                let inner = track.shrink(4.0);
                let fade = 1.0f32;
                let mut groups: Vec<(usize, usize, usize)> = Vec::new(); // (clip, first shot, len)
                for (i, shot) in sc.shots.iter().enumerate() {
                    match shot.sprite {
                        Some((c, _)) => match groups.last_mut() {
                            Some((gc, _, len)) if *gc == c => *len += 1,
                            _ => groups.push((c, i, 1)),
                        },
                        None => groups.push((usize::MAX, i, 1)),
                    }
                }
                let mut seen: HashMap<usize, usize> = HashMap::new();
                for (clip, first, len) in groups.iter() {
                    let x0 = inner.left() + inner.width() * (sc.start_of(*first) as f32) / total.max(1.0);
                    let x1 = inner.left() + inner.width() * ((sc.start_of(*first) as f32) + (*len as f32) * sc.shots[*first].ticks) / total.max(1.0);
                    let r = Rect::from_min_max(pos2(x0 + 0.5, inner.top()), pos2((x1 - 0.5).max(x0 + 2.0), inner.bottom()));
                    let (action, secs): (String, f32) = if *clip == usize::MAX {
                        ("pause".to_string(), (*len as f32) / fps)
                    } else {
                        let c = &sc.clips[*clip];
                        (String::from(c.action), (*len as f32) / fps)
                    };
                    let count = seen.entry(*clip).or_insert(0);
                    *count += 1;
                    let label = if *clip != usize::MAX && *count > 1 { format!("{action} ×{count}") } else { action.clone() };
                    let (fill, border) = match action.as_str() {
                        "walk" | "run" => (theme::ACCENT_SOFT, theme::ACCENT),
                        "punch" => (theme::BG_RAISED, theme::WARN),
                        "spin_kick" => (theme::BG_RAISED, theme::ACCENT_SELECT),
                        "turn" => (theme::BG_RAISED, theme::TEXT_SECONDARY),
                        _ => (theme::BG_RAISED, theme::BORDER_SUBTLE),
                    };
                    let fill = Color32::from_rgba_unmultiplied(fill.r(), fill.g(), fill.b(), (fill.a() as f32 * fade) as u8);
                    p.rect(r, theme::R_CTRL, fill, Stroke::new(1.0, border), StrokeKind::Inside);
                    if r.width() > 52.0 {
                        kit::text_at(ui, pos2(r.left() + 10.0, r.top() + 6.0), &label, theme::font_med(12.0), theme::TEXT_PRIMARY);
                        kit::text_at(ui, pos2(r.left() + 10.0, r.top() + 24.0), &format!("{secs:.1}s"), theme::font_mono(10.5), theme::TEXT_MUTED);
                    }
                }
                // playhead: 2 px, 64 tall, above the track
                let px = inner.left() + inner.width() * ((sc.start_of(shown) as f32) / total.max(1.0)).clamp(0.0, 1.0);
                let head = Rect::from_min_max(pos2(px - 1.0, track.top() - 4.0), pos2(px + 1.0, track.bottom() + 4.0));
                p.rect_filled(head, CornerRadius::same(1), theme::ACCENT_SELECT);
                if resp.clicked() {
                    if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                        let frac = ((pos.x - inner.left()) / inner.width()).clamp(0.0, 1.0);
                        let ticks = frac as f64 * total as f64;
                        self.scene.shown = sc.shot_at(ticks);
                        self.scene.t0 = now - ticks / fps as f64;
                    }
                }
                // ---- speeds + the current shot line (24 px)
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    ui.add(egui::Label::new(kit::txt("walk", 12.0).color(theme::TEXT_SECONDARY)));
                    let mut v = self.scene.walk_speed;
                    if ui.add(egui::DragValue::new(&mut v).range(0.25..=6.0).speed(0.05).fixed_decimals(2)).changed() {
                        self.scene.walk_speed = v;
                    }
                    ui.add_space(4.0);
                    ui.add(egui::Label::new(kit::txt("run", 12.0).color(theme::TEXT_SECONDARY)));
                    let mut v = self.scene.run_speed;
                    if ui.add(egui::DragValue::new(&mut v).range(0.25..=8.0).speed(0.05).fixed_decimals(2)).changed() {
                        self.scene.run_speed = v;
                    }
                    ui.add(egui::Label::new(kit::txt("px / frame", 11.5).color(theme::TEXT_MUTED)));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let what = match sc.shots[shown].sprite {
                            Some((c, f)) => format!(
                                "{} {}° · frame {}/{} · shot {}/{}",
                                sc.clips[c].action,
                                sc.clips[c].azimuth,
                                f + 1,
                                sc.clips[c].frames.len(),
                                shown + 1,
                                sc.shots.len()
                            ),
                            None => format!("pause · shot {}/{}", shown + 1, sc.shots.len()),
                        };
                        ui.add(egui::Label::new(kit::mono_txt(what, 11.5).color(theme::TEXT_MUTED)));
                    });
                });
            });
            ui.add_space(20.0);
            // ---- the stage preview, 250 x 168, inside a bordered card
            let (rect, _) = ui.allocate_exact_size(vec2(250.0, 168.0), Sense::hover());
            let p = ui.painter().clone();
            p.rect(rect, theme::R_CARD, theme::BG_APP, Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
            let inner = rect.shrink(1.0);
            let size = sc.size as f32;
            let scale = (inner.height() * 0.78 / size).floor().clamp(1.0, 8.0);
            let sprite_h = size * scale;
            kit::checker(ui, inner, (scale * 2.0).max(5.0));
            let shot = sc.shots[shown];
            if let Some((c, f)) = shot.sprite {
                if let Some(t) = self.scene.textures.get(c).and_then(|v| v.get(f)) {
                    // the stage is wider than the card: the shot's x maps onto the card, clipped to it
                    let travel = (sc.stage_w as f32 - size).max(1.0);
                    let frac = (shot.x as f32 / travel).clamp(0.0, 1.0);
                    let floor = inner.center().y + sprite_h * 0.5;
                    let r = Rect::from_min_size(
                        pos2(inner.left() + (inner.width() - sprite_h) * frac, floor - sprite_h),
                        Vec2::splat(sprite_h),
                    );
                    p.with_clip_rect(inner).image(t.id(), r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
            }
            if self.scene.is_mock {
                p.text(inner.left_top() + vec2(6.0, 4.0), egui::Align2::LEFT_TOP, "MOCK", egui::FontId::proportional(13.0), Color32::from_rgba_unmultiplied(255, 70, 70, 210));
            }
        });
        if restart {
            self.scene.t0 = now;
            self.scene.shown = 0;
            self.scene.playing = true;
        }
        if export {
            self.export_scene();
        }
    }
}
