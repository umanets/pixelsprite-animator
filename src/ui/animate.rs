//! The selected sprite and its animation: which animator, which kind, refreshing the loop.

use crate::*;

impl App {
    // ------------------------------------------------------------------------------------------- selection
    pub(crate) fn selected_sprite(&self) -> Option<(Sprite, String, Option<Classification>)> {
        match self.selected? {
            Selection::Generated { run, idx } => {
                let r = self.runs.iter().find(|r| r.id == run)?;
                let s = r.cards.get(idx)?.sprite.clone()?;
                let class = r.class.clone().or_else(|| Some(classify_english(&r.caption)));
                Some((s, r.caption.clone(), class))
            }
            Selection::Library => {
                let d = self.lib_data.as_ref()?;
                let s = d.base_sprite(self.anim.lib_size).or_else(|| d.sizes().first().and_then(|&z| d.base_sprite(z)))?;
                let class = Some(match d.entry.kind {
                    Some(k) => Classification { kind: Some(k), reason: "kind of the 3D-farm job".into() },
                    None => classify_english(&d.entry.caption),
                });
                Some((s, d.entry.caption.clone(), class))
            }
            Selection::Loaded => {
                let (s, name, _) = self.loaded.as_ref()?;
                let words = name.trim_end_matches(".png").trim_end_matches(".PNG").replace(['_', '-', '.'], " ");
                let class = match classify_english(&words) {
                    c @ Classification { kind: Some(_), .. } => c,
                    _ => Classification { kind: Some(Kind::Human), reason: "loaded PNG: no kind word in the file name, assumed human (override below)".into() },
                };
                Some((s.clone(), name.clone(), Some(class)))
            }
        }
    }

    /// Opens the file dialog (once); the answer is picked up by [`Self::poll_png_pick`].
    pub(crate) fn start_png_pick(&mut self, ctx: &egui::Context) {
        if self.png_pick.is_none() {
            let ctx = ctx.clone();
            self.png_pick = Some(pick_png(move || ctx.request_repaint()));
        }
    }

    pub(crate) fn poll_png_pick(&mut self) {
        let Some(rx) = &self.png_pick else { return };
        match rx.try_recv() {
            Ok(Some(path)) => {
                self.png_pick = None;
                self.png_path = path.display().to_string();
                self.load_png(None);
            }
            Ok(None) | Err(std::sync::mpsc::TryRecvError::Disconnected) => self.png_pick = None,
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
    }

    /// Reads `png_path`: a 16 / 24 / 32 px sprite (or an integer upscale of one) as it is, any other image fitted
    /// to the model's sizes. Then the sprite is prepared: `prep` says how; without it the user is asked (the
    /// dialog of [`Self::prep_window`]). `Some(None)` takes the recommended preparation without asking.
    pub(crate) fn load_png(&mut self, prep: Option<Option<PngPrep>>) {
        let path = PathBuf::from(self.png_path.trim().trim_matches('"'));
        match Sprite::from_image_file(&path) {
            Ok((sprite, factor, fitted)) => {
                let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "sprite.png".into());
                // the animation model was trained on sprites with a dark outline: a sprite that has one is fine as
                // it is, any other gets a line around it
                let recommended = if sprite.has_dark_outline() { PngPrep::AsIs } else { PngPrep::Outline };
                let dialog = PrepDialog {
                    name,
                    factor,
                    fitted,
                    variants: PngPrep::ALL.iter().map(|&p| (p, p.apply(&sprite), None)).collect(),
                    recommended,
                };
                self.png_error = None;
                match prep {
                    Some(p) => self.finish_load(dialog, p.unwrap_or(recommended)),
                    None => self.png_prep = Some(dialog),
                }
            }
            Err(e) => self.png_error = Some(format!("{e:#}")),
        }
    }

    /// Selects the loaded sprite, prepared as chosen.
    pub(crate) fn finish_load(&mut self, d: PrepDialog, prep: PngPrep) {
        let Some((_, sprite, _)) = d.variants.into_iter().find(|v| v.0 == prep) else { return };
        let mut notes = Vec::new();
        if d.fitted {
            notes.push(format!("fitted to {} px", sprite.size));
        }
        match prep {
            PngPrep::AsIs => {}
            PngPrep::Outline => notes.push("a dark outline was added".to_owned()),
            PngPrep::Inline => notes.push("the edge was painted dark (inline)".to_owned()),
        }
        if !notes.is_empty() {
            self.status = format!("{}: {}.", d.name, notes.join(", "));
        }
        self.loaded = Some((sprite, d.name, d.factor));
        self.loaded_id += 1;
        self.png_prep_used = Some(prep);
        self.selected = Some(Selection::Loaded);
        self.anim.kind = KindChoice::Auto;
        // a loaded sprite goes to the animation model when it is there (the mock only when it is not)
        if self.anim.animator != AnimatorChoice::Onnx {
            self.anim.animator = if self.jobs.anim.available() { AnimatorChoice::Onnx } else { AnimatorChoice::Mock };
        }
    }

    /// The question of a load: the image as it is, with an outline or with an inline, each shown as it will go to
    /// the animation model. A click on a card loads that one.
    pub(crate) fn prep_window(&mut self, ctx: &egui::Context) {
        let Some(mut d) = self.png_prep.take() else { return };
        let mut chosen = None;
        let mut cancel = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        // the rest of the window is dimmed and does not take clicks
        egui::Area::new(egui::Id::unique("prep_dim")).order(egui::Order::Middle).fixed_pos(pos2(0.0, 0.0)).show(ctx, |ui| {
            let screen = ctx.content_rect();
            ui.allocate_rect(screen, Sense::click());
            ui.painter().rect_filled(screen, CornerRadius::ZERO, Color32::from_black_alpha(150));
        });
        egui::Window::new("Prepare the sprite")
            .order(egui::Order::Foreground)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.add(egui::Label::new(kit::semi_txt(&d.name, 14.0)));
                ui.add(egui::Label::new(
                    kit::txt("The animation model draws best from a sprite with a dark outline. Pick how this one goes to it.", 12.5)
                        .color(theme::TEXT_SECONDARY),
                ));
                ui.add_space(10.0);
                let side = 148.0;
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 12.0;
                    for (prep, sprite, tex) in &mut d.variants {
                        let tex = tex.get_or_insert_with(|| texture(ctx, &format!("prep_{prep:?}"), sprite));
                        let (rect, resp) = ui.allocate_exact_size(vec2(side, side + 52.0), Sense::click());
                        let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
                        let p = ui.painter();
                        let pic = Rect::from_min_size(rect.min, Vec2::splat(side));
                        kit::checker(ui, pic, side / sprite.size as f32 * 2.0);
                        p.image(tex.id(), pic.shrink(2.0), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                        let rec = *prep == d.recommended;
                        let stroke = if resp.hovered() {
                            Stroke::new(2.0, theme::ACCENT)
                        } else if rec {
                            Stroke::new(2.0, theme::ACCENT_SELECT)
                        } else {
                            Stroke::new(1.0, theme::BORDER_STRONG)
                        };
                        p.rect_stroke(pic, theme::R_CARD, stroke, StrokeKind::Inside);
                        if rec {
                            let tag = Rect::from_min_size(pic.min + vec2(6.0, 6.0), vec2(86.0, 18.0));
                            kit::tag_at(ui, tag, "recommended", theme::ACCENT_SELECT, theme::ON_SELECT, theme::R_CHIP);
                        }
                        let l1 = p.layout_no_wrap(prep.label().into(), theme::font_semi(13.0), theme::TEXT_PRIMARY);
                        let l2 = p.layout(prep.describe().into(), theme::font(11.5), theme::TEXT_MUTED, side);
                        p.galley(pos2(rect.left(), pic.bottom() + 6.0), l1, Color32::WHITE);
                        p.galley(pos2(rect.left(), pic.bottom() + 25.0), l2, Color32::WHITE);
                        if resp.clicked() {
                            chosen = Some(*prep);
                        }
                    }
                });
                ui.add_space(8.0);
                if kit::button(ui, "Cancel", kit::Btn::Ghost).clicked() {
                    cancel = true;
                }
            });
        match chosen {
            Some(p) => self.finish_load(d, p),
            None if cancel => {}
            None => self.png_prep = Some(d),
        }
    }

    /// The selected sprite's texture (the "original" card, the still preview), made once per selection.
    pub(crate) fn reference_tex(&mut self, ctx: &egui::Context) -> Option<(TextureHandle, usize)> {
        let (sprite, _, _) = self.selected_sprite()?;
        let key = format!("{:?}{}/{}", self.selected, self.anim.lib_size, self.loaded_id);
        match &self.lib_tex {
            Some((k, size, t)) if *k == key => Some((t.clone(), *size)),
            _ => {
                let t = texture(ctx, "reference", &sprite);
                self.lib_tex = Some((key, sprite.size, t.clone()));
                Some((t, sprite.size))
            }
        }
    }

    /// The reference type for the model: a library entry's sprite is the farm's front render.
    pub(crate) fn onnx_refk(&self) -> Option<usize> {
        match (self.selected, self.jobs.anim.settings().reference) {
            (Some(Selection::Library), RefChoice::Auto) => Some(1),
            _ => None,
        }
    }

    /// What the ONNX animator's output depends on besides the request (empty for the other animators). Includes
    /// the walk-front reference option: toggling it draws the preview and the scene again.
    pub(crate) fn onnx_tag(&self) -> String {
        if self.anim.animator == AnimatorChoice::Onnx { format!("{:?} {:?}", self.jobs.anim.settings(), self.onnx_refk()) } else { String::new() }
    }

    pub(crate) fn effective_kind(&self, class: &Option<Classification>) -> Option<Kind> {
        match self.anim.kind {
            KindChoice::Auto => class.as_ref().and_then(|c| c.kind),
            KindChoice::Force(k) => Some(k),
            KindChoice::NotAnimatable => None,
        }
    }

    pub(crate) fn animator(&self) -> Box<dyn Animator + '_> {
        match self.anim.animator {
            AnimatorChoice::Mock => Box::new(&self.mock),
            AnimatorChoice::Library => match &self.lib_data {
                Some(d) => Box::new(LibraryAnimator { data: d.clone() }),
                None => Box::new(MissingLibrary),
            },
            AnimatorChoice::Onnx => Box::new(&*self.jobs.anim),
        }
    }

    pub(crate) fn actions_for(&self, kind: Kind) -> Vec<ActionSpec> {
        self.animator().actions(kind)
    }

    /// Recomputes the animation when anything it depends on changed.
    pub(crate) fn refresh_animation(&mut self, ctx: &egui::Context) {
        let Some((sprite, _cap, class)) = self.selected_sprite() else {
            self.anim.result = None;
            self.anim.textures.clear();
            self.anim.key = None;
            return;
        };
        let kind = self.effective_kind(&class);
        let key = AnimKey {
            sel: self.selected,
            lib_name: self.lib_data.as_ref().map(|d| d.entry.name.clone()),
            loaded: self.loaded_id,
            animator: self.anim.animator,
            kind,
            action: self.anim.action.clone(),
            azimuth: self.anim.azimuth,
            n_frames: self.anim.n_frames,
            size: sprite.size,
            onnx: self.onnx_tag(),
        };
        if self.anim.key.as_ref() == Some(&key) {
            return;
        }
        self.anim.key = Some(key);
        self.anim.result = None;
        self.anim.textures.clear();
        self.anim.error = None;
        self.anim.stopped = false;
        self.anim.progress = None;
        self.anim.idle = false;
        let go = std::mem::take(&mut self.anim.go);
        if let Some(t) = self.anim.pending.take() {
            t.stop();
        }
        let Some(kind) = kind else { return };
        let req = AnimRequest { kind, action: self.anim.action.clone(), azimuth: self.anim.azimuth, n_frames: self.anim.n_frames, size: sprite.size };
        if self.anim.animator == AnimatorChoice::Onnx {
            // The model never starts by itself: a clip that is already in the cache is shown at once, anything
            // else waits for the Animate button (`go`). Drawn on the animation thread; see poll_jobs.
            self.anim.is_mock = false;
            let refk = self.onnx_refk();
            let cached = self.jobs.anim.cached(&sprite, kind, &[(req.action.clone(), req.azimuth)], refk).is_some();
            if go || cached {
                self.anim.pending = Some(self.jobs.submit(Job::Preview { sprite, req, refk }));
            } else {
                self.anim.idle = true;
            }
            return;
        }
        let (is_mock, result) = {
            let animator = self.animator();
            (animator.is_mock(), animator.animate(&sprite, &req))
        };
        self.anim.is_mock = is_mock;
        match result {
            Ok(a) => {
                self.anim.textures = a.frames.iter().enumerate().map(|(i, f)| texture(ctx, &format!("frame{i}"), f)).collect();
                self.anim.result = Some(a);
                self.anim.t0 = ctx.input(|i| i.time);
            }
            Err(e) => self.anim.error = Some(format!("{e:#}")),
        }
    }

    /// Keeps the action valid for the current kind / animator (and applies its default frame count).
    pub(crate) fn fix_action(&mut self, kind: Option<Kind>) {
        let Some(kind) = kind else { return };
        let acts = self.actions_for(kind);
        if acts.is_empty() {
            return;
        }
        if !acts.iter().any(|a| a.name == self.anim.action) {
            let a = &acts[0];
            self.anim.action = a.name.clone();
            self.anim.n_frames = a.frames.max(1);
            self.anim.looping = a.looping;
        }
    }
}
