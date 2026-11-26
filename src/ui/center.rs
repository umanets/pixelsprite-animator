//! The centre: gallery of generated sprites and the animation preview card.

use crate::*;

impl App {
    // --------------------------------------------------------------------------------------------- panels
    // -------------------------------------------------------------------------------------------- centre
    pub(crate) fn center_panel(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.center_header(ui);
        ui.add_space(16.0);
        let gallery_h = 180.0f32.min((ui.available_height() * 0.30).max(96.0));
        egui::ScrollArea::vertical().id_salt("runs").auto_shrink([false, false]).max_height(gallery_h).show(ui, |ui| self.gallery(ui, &ctx));
        ui.add_space(16.0);
        self.preview_card(ui, &ctx);
    }

    /// The centre header: «“caption”», the mono meta line, the ×1…×8 zoom and Save all.
    pub(crate) fn center_header(&mut self, ui: &mut egui::Ui) {
        let sel = match self.selected {
            Some(Selection::Generated { run, idx }) => self
                .runs
                .iter()
                .find(|r| r.id == run)
                .and_then(|r| r.cards.get(idx).and_then(|c| c.sprite.clone()).map(|s| (r, s, idx))),
            _ => None,
        };
        // while a generation runs, the header describes that run ("… drawing 1 of 4…"), as the mock does
        let running = self.running.is_some();
        let latest = self.runs.last();
        let (caption, meta) = match (sel.as_ref(), if running { latest } else { None }) {
            (_, Some(r)) => {
                let (_, total, done, _) = self.progress;
                let n = r.cards.len().max(1);
                let cap = if r.caption.is_empty() { "no text".to_string() } else { format!("\u{201c}{}\u{201d}", r.caption) };
                let drawn = if total > 1 { format!("drawing {} of {}\u{2026}", (done + 1).min(n), n) } else { "drawing\u{2026}".to_string() };
                (cap, format!("{0}×{0} · seed {1} · {2}", r.size, r.seed, drawn))
            }
            (Some((r, _, _)), None) => (
                if r.caption.is_empty() { "no text".to_string() } else { format!("\u{201c}{}\u{201d}", r.caption) },
                format!("{0}×{0} · seed {1} · {2} sprites", r.size, r.seed, r.cards.len()),
            ),
            (None, None) => match (self.selected, self.selected_sprite()) {
                (Some(Selection::Loaded), Some((s, name, _))) => (name, format!("{0}×{0} · loaded image", s.size)),
                (Some(_), Some((s, cap, _))) => (cap, format!("{0}×{0} · library", s.size)),
                _ => ("No sprite selected yet".to_string(), "generate or pick one".into()),
            },
        };
        let (header, _) = ui.allocate_exact_size(vec2(ui.available_width(), 41.0), Sense::hover());
        let p = ui.painter().clone();
        p.galley(pos2(header.left(), header.top()), p.layout_no_wrap(caption, theme::font_semi(18.0), theme::TEXT_PRIMARY), Color32::WHITE);
        let meta_gal = p.layout_no_wrap(meta, theme::font_mono(12.0), theme::TEXT_MUTED);
        p.with_clip_rect(Rect::from_min_max(header.left_top(), pos2(header.right() - 220.0, header.bottom()))).galley(pos2(header.left(), header.bottom() - 16.0), meta_gal, Color32::WHITE);
        // the right cluster: ×1 … ×8 then Save all, both 32 px tall, centred on the header
        let save_w = 78.0;
        let seg_w = 117.0;
        let x_seg = header.right() - 16.0 - save_w - 8.0 - seg_w;
        let y = header.center().y - 16.0;
        let seg_rect = Rect::from_min_size(pos2(x_seg, y), vec2(seg_w, 32.0));
        let save_rect = Rect::from_min_size(pos2(header.right() - save_w, y), vec2(save_w, 32.0));
        let mut save = false;
        ui.scope_builder(egui::UiBuilder::new().max_rect(seg_rect).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
            let mut idx = [1u32, 2, 4, 8].iter().position(|&s| s == self.save_scale).unwrap_or(0);
            if kit::segmented_small(ui, &["×1", "×2", "×4", "×8"], &mut idx) {
                self.save_scale = [1u32, 2, 4, 8][idx.min(3)];
            }
        });
        ui.scope_builder(egui::UiBuilder::new().max_rect(save_rect).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
            if kit::button_wh(ui, "Save all", kit::Btn::Ghost, save_w, 32.0)
                .on_hover_text("a sheet of the selected sprite's run (the source first), 8 per row — the demo's \"download sheet\"")
                .clicked()
                && sel.is_some()
            {
                save = true;
            }
        });
        if save {
            self.save_selected(SaveTarget::Sheet(None));
        }
    }

    /// The gallery: 1:1 cards of every run on the checkerboard, four per row. The selected card gets the amber
    /// frame and a «selected» tag; cards still being drawn say «drawing…» / «queued».
    pub(crate) fn gallery(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let p = ui.painter().clone();
        if self.runs.is_empty() && self.loaded.is_none() {
            ui.add(egui::Label::new(kit::txt("Nothing yet: type a caption and press Generate.", 13.0).color(theme::TEXT_SECONDARY)));
            return;
        }
        let mut select: Option<Selection> = None;
        let mut ctx_action: Option<(CardAction, u64, usize)> = None;
        let gap = 12.0;
        let cell = ((ui.available_width() - 3.0 * gap) / 4.0).floor().max(96.0);
        // the loaded image is the gallery's first card
        if let Some((sprite, name, _)) = &self.loaded {
            let tex = match &self.loaded_tex {
                Some((id, t)) if *id == self.loaded_id => t.clone(),
                _ => {
                    let t = texture(ctx, "loaded", sprite);
                    self.loaded_tex = Some((self.loaded_id, t.clone()));
                    t
                }
            };
            let (rect, resp) = ui.allocate_exact_size(Vec2::splat(cell), Sense::click());
            let resp = resp.on_hover_text(name);
            kit::checker(ui, rect, (cell / sprite.size as f32).max(6.0));
            p.image(tex.id(), rect.shrink(1.0), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            let sel = self.selected == Some(Selection::Loaded);
            let tag = Rect::from_min_size(rect.min + vec2(6.0, 6.0), vec2(58.0, 18.0));
            if sel {
                kit::select_frame(ui, rect, theme::R_CARD);
                kit::tag_at(ui, tag, "selected", theme::ACCENT_SELECT, theme::ON_SELECT, theme::R_CHIP);
            } else {
                p.rect_stroke(rect, theme::R_CARD, Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
                kit::tag_at(ui, tag, "loaded", theme::BG_PANEL, theme::TEXT_SECONDARY, theme::R_CHIP);
            }
            if resp.clicked() {
                select = Some(Selection::Loaded);
            }
            ui.add_space(gap);
        }
        for run in &mut self.runs {
            let n = run.cards.len();
            let mut i = 0;
            while i < n {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = gap;
                    for k in 0..4 {
                        let idx = i + k;
                        if idx >= n {
                            break;
                        }
                        let card = &mut run.cards[idx];
                        if card.dirty {
                            let s = card.sprite.clone().or_else(|| {
                                card.partial.as_ref().map(|p| Sprite::new(run.size, p.clone(), self.palette.clone().unwrap_or_default()))
                            });
                            if let Some(s) = s {
                                let img = color_image(&s);
                                match &mut card.tex {
                                    Some(t) => t.set(img, TextureOptions::NEAREST),
                                    None => card.tex = Some(ctx.load_texture(format!("run{}_{}", run.id, idx), img, TextureOptions::NEAREST)),
                                }
                            }
                            card.dirty = false;
                        }
                        let sel = self.selected == Some(Selection::Generated { run: run.id, idx });
                        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(cell), Sense::click());
                        kit::checker(ui, rect, (cell / run.size as f32).max(6.0));
                        if let Some(t) = &card.tex {
                            p.image(t.id(), rect.shrink(1.0), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                        }
                        if sel {
                            kit::select_frame(ui, rect, theme::R_CARD);
                            let tag = Rect::from_min_size(rect.min + vec2(6.0, 6.0), vec2(58.0, 18.0));
                            kit::tag_at(ui, tag, "selected", theme::ACCENT_SELECT, theme::ON_SELECT, theme::R_CHIP);
                        } else if card.tag == CardTag::Source {
                            p.rect_stroke(rect, theme::R_CARD, Stroke::new(2.0, theme::SOURCE_BLUE), StrokeKind::Inside);
                        } else if resp.hovered() {
                            p.rect_stroke(rect, theme::R_CARD, Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
                        } else {
                            p.rect_stroke(rect, theme::R_CARD, Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
                        }
                        if card.sprite.is_none() {
                            let label = if card.partial.is_some() { "drawing\u{2026}" } else { "queued" };
                            let gal = p.layout_no_wrap(label.to_owned(), theme::font_mono(11.5), theme::TEXT_SECONDARY);
                            p.galley(pos2(rect.center().x - gal.size().x / 2.0, rect.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
                        }
                        // the variant of a mix, as a small tag in the card's corner
                        if let Some(v) = &card.variant {
                            let gal = p.layout_no_wrap(v.label.clone(), theme::font(11.0), theme::TEXT_SECONDARY);
                            let tw = gal.size().x + 14.0;
                            let tr = Rect::from_min_size(pos2(rect.left() + 6.0, rect.bottom() - 24.0), vec2(tw, 18.0));
                            p.rect_filled(tr, theme::R_CHIP, Color32::from_rgba_unmultiplied(0x0D, 0x0E, 0x13, 220));
                            p.galley(pos2(tr.center().x - gal.size().x / 2.0, tr.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
                        }
                        let resp = match &card.variant {
                            Some(v) => resp.on_hover_text(format!("{}\n{}", v.label, v.describe)),
                            None => resp,
                        };
                        if card.sprite.is_some() && resp.clicked() {
                            select = Some(Selection::Generated { run: run.id, idx });
                        }
                        if card.sprite.is_some() {
                            resp.context_menu(|ui| {
                                if ui.button("save this sprite (PNG)").clicked() {
                                    ctx_action = Some((CardAction::SavePng, run.id, idx));
                                    ui.close();
                                }
                                if ui.button("use its shape").clicked() {
                                    ctx_action = Some((CardAction::UseShape, run.id, idx));
                                    ui.close();
                                }
                            });
                        }
                    }
                });
                i += 4;
            }
        }
        if let Some((what, run_id, idx)) = ctx_action {
            if let Some(sprite) = self.runs.iter().find(|r| r.id == run_id).and_then(|r| r.cards.get(idx)).and_then(|c| c.sprite.clone()) {
                match what {
                    CardAction::SavePng => {
                        self.selected = Some(Selection::Generated { run: run_id, idx });
                        self.save_selected(SaveTarget::Sprite);
                    }
                    CardAction::UseShape => self.use_shape(&sprite),
                }
            }
        }
        if let Some(s) = select {
            if self.selected != Some(s) {
                self.selected = Some(s);
                self.anim.kind = KindChoice::Auto;
                if self.anim.animator == AnimatorChoice::Library {
                    self.anim.animator = if self.jobs.anim.available() { AnimatorChoice::Onnx } else { AnimatorChoice::Mock };
                }
            }
        }
    }

    /// The Preview stage of the mock: a 44 px header, the big checkerboard canvas (the sprite, the «original»
    /// reference card and the frame HUD) and the 60 px transport with the frame strip and the loop switch.
    pub(crate) fn preview_card(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let anim = self.anim.result.clone();
        let n = anim.as_ref().map(|a| a.frames.len().max(1)).unwrap_or(1);
        let now = ctx.input(|i| i.time);
        if anim.is_some() && self.anim.playing {
            let f = ((now - self.anim.t0) * self.anim.fps as f64).floor().max(0.0) as usize;
            self.anim.shown = if self.anim.looping { f % n } else { f.min(n - 1) };
            ctx.request_repaint_after(std::time::Duration::from_secs_f32(1.0 / self.anim.fps.max(1.0)));
        }
        let h = ui.available_height().max(220.0);
        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::hover());
        let p = ui.painter().clone();
        p.rect(rect, theme::R_PANEL, theme::BG_PANEL, Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
        let header_h = 44.0;
        let transport_h = 60.0;
        let header = Rect::from_min_max(rect.min, pos2(rect.right(), rect.top() + header_h));
        let transport = Rect::from_min_max(pos2(rect.left(), rect.bottom() - transport_h), rect.max);
        let canvas = Rect::from_min_max(pos2(rect.left() + 1.0, header.bottom()), pos2(rect.right() - 1.0, transport.top()));

        // ---- header, laid out by hand so the mono note never runs into the badge
        p.line_segment([header.left_bottom(), header.right_bottom()], Stroke::new(1.0, theme::BORDER_SUBTLE));
        {
            let mut x = header.left() + 16.0;
            let chip = Rect::from_min_size(pos2(x, header.center().y - 10.0), vec2(20.0, 20.0));
            p.rect_filled(chip, theme::R_CHIP, theme::ACCENT_SOFT);
            let digit = p.layout_no_wrap("2".into(), theme::font_mono_med(11.0), theme::TEXT_PRIMARY);
            p.galley(pos2(chip.center().x - digit.size().x / 2.0, chip.center().y - digit.size().y / 2.0), digit, Color32::WHITE);
            x = chip.right() + 8.0;
            let title = p.layout_no_wrap("PREVIEW".into(), theme::font_semi(12.0), theme::TEXT_SECONDARY);
            p.galley(pos2(x, header.center().y - title.size().y / 2.0), title.clone(), Color32::WHITE);
            x += title.size().x + 8.0;
            if let Some(a) = &anim {
                let dir = format!("{}°", self.anim.azimuth);
                let side = match self.anim.azimuth {
                    0 => "front",
                    90 | 270 => "side",
                    180 => "back",
                    _ => "three-quarter",
                };
                let badge_txt = format!("{} · {} {} · {} frames", self.anim.action, dir, side, n);
                let gal = p.layout_no_wrap(badge_txt, theme::font_med(11.5), theme::TEXT_SECONDARY);
                let br = Rect::from_min_size(pos2(x, header.center().y - 10.5), vec2(gal.size().x + 16.0, 21.0));
                p.rect_filled(br, theme::R_CHIP, theme::BG_RAISED);
                p.galley(pos2(br.left() + 8.0, br.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
                x = br.right() + 8.0;
                if self.anim.is_mock {
                    let gal = p.layout_no_wrap("MOCK".into(), theme::font_med(11.5), theme::DANGER);
                    let br = Rect::from_min_size(pos2(x, header.center().y - 10.5), vec2(gal.size().x + 16.0, 21.0));
                    p.rect_filled(br, theme::R_CHIP, theme::BG_RAISED);
                    p.galley(pos2(br.left() + 8.0, br.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
                    x = br.right() + 8.0;
                }
                // the note: right aligned, clipped to what is left of the badge
                let note = note_short(&a.note);
                let gal = p.layout_no_wrap(note.clone(), theme::font_mono(11.5), theme::TEXT_MUTED);
                let right = header.right() - 16.0;
                let room = right - x;
                if gal.size().x <= room {
            p.galley(pos2(right - gal.size().x, header.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
            }
            } else {
                let gal = p.layout_no_wrap("no clip yet".into(), theme::font(12.0), theme::TEXT_MUTED);
                p.galley(pos2(x, header.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
                if let Some(t) = &self.anim.pending {
                    let pr = self.anim.progress.clone().unwrap_or_default();
                    let stop_rect = Rect::from_min_size(pos2(header.right() - 16.0 - 62.0, header.center().y - 14.0), vec2(62.0, 28.0));
                    ui.scope_builder(egui::UiBuilder::new().max_rect(stop_rect).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
                        if kit::button_h(ui, "Stop", kit::Btn::Danger, 28.0).clicked() {
                            t.stop();
                        }
                    });
                    let pct = p.layout_no_wrap(format!("{:.0} % · {:.0} s", pr.frac * 100.0, pr.secs), theme::font_mono(11.5), theme::TEXT_MUTED);
                    let bar_w = (header.right() - 16.0 - 62.0 - 12.0 - pct.size().x - 12.0 - x).max(60.0);
                    let bar = Rect::from_min_size(pos2(x, header.center().y - 3.0), vec2(bar_w, 6.0));
                    p.rect_filled(bar, CornerRadius::same(3), theme::BORDER_SUBTLE);
                    p.rect_filled(Rect::from_min_size(bar.min, vec2(bar.width() * pr.frac.clamp(0.0, 1.0), 6.0)), CornerRadius::same(3), theme::ACCENT);
                    p.galley(pos2(bar.right() + 12.0, header.center().y - pct.size().y / 2.0), pct, Color32::WHITE);
                } else if self.anim.stopped {
                    let gal = p.layout_no_wrap("stopped · finished clips are kept".into(), theme::font(12.0), theme::TEXT_MUTED);
                    p.galley(pos2(x, header.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
                    let btn = Rect::from_min_size(pos2(header.right() - 16.0 - 90.0, header.center().y - 14.0), vec2(90.0, 28.0));
                    ui.scope_builder(egui::UiBuilder::new().max_rect(btn).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
                        if kit::button_h(ui, "Draw again", kit::Btn::Ghost, 28.0).clicked() {
                            self.anim.go = true;
                            self.anim.key = None;
                        }
                    });
                }
            }
        }

        // ---- the canvas
        kit::checker(ui, canvas, 12.0);
        let mut jump = None;
        let mut restart = false;
        let mut toggle = false;
        if let Some(a) = &anim {
            let size = a.frames.first().map(|f| f.size).unwrap_or(32) as f32;
            let scale = (canvas.height() * 0.72 / size).min(canvas.width() * 0.45 / size).floor().clamp(1.0, 16.0);
            let side = size * scale;
            let center = pos2(canvas.center().x, canvas.top() + canvas.height() * 0.56);
            let spr = Rect::from_center_size(center, Vec2::splat(side));
            let sh = Rect::from_center_size(pos2(center.x, spr.bottom() + 30.0), vec2(120.0, 8.0));
            p.rect_filled(sh, CornerRadius::same(4), Color32::from_rgba_unmultiplied(0, 0, 0, 64));
            if let Some(t) = self.anim.textures.get(self.anim.shown) {
                p.image(t.id(), spr, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            }
            if self.anim.is_mock {
                p.text(spr.left_top() + vec2(6.0, 4.0), egui::Align2::LEFT_TOP, "MOCK", egui::FontId::proportional(14.0), Color32::from_rgba_unmultiplied(255, 70, 70, 210));
            }
            // the «original» reference card, top left
            let card = Rect::from_min_size(canvas.min + vec2(14.0, 14.0), vec2(78.0, 96.0));
            p.rect(card, theme::R_CTRL, theme::BG_PANEL, Stroke::new(1.0, theme::BORDER_STRONG), StrokeKind::Inside);
            let thumb = Rect::from_min_size(card.min + vec2(7.0, 7.0), Vec2::splat(64.0));
            if let Some((sprite, _, _)) = self.selected_sprite() {
                let key = format!("{:?}{}/{}", self.selected, self.anim.lib_size, self.loaded_id);
                let tex = match &self.lib_tex {
                    Some((k, _, t)) if *k == key => t.clone(),
                    _ => {
                        let t = texture(ctx, "reference", &sprite);
                        self.lib_tex = Some((key, sprite.size, t.clone()));
                        t
                    }
                };
                kit::checker(ui, thumb, 64.0 / sprite.size as f32);
                p.image(tex.id(), thumb, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                p.rect_stroke(thumb, CornerRadius::same(4), Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
            }
            let gal = p.layout_no_wrap("original".into(), theme::font_mono(10.5), theme::TEXT_MUTED);
            p.galley(pos2(card.center().x - gal.size().x / 2.0, card.bottom() - gal.size().y - 5.0), gal, Color32::WHITE);
            // the frame HUD, top right
            let hud_txt = format!("frame {} / {} · {:.0} fps", self.anim.shown + 1, n, self.anim.fps);
            let gal = p.layout_no_wrap(hud_txt, theme::font_mono(11.5), theme::TEXT_SECONDARY);
            let hud = Rect::from_min_size(pos2(canvas.right() - 14.0 - gal.size().x - 20.0, canvas.top() + 14.0), vec2(gal.size().x + 20.0, 25.0));
            p.rect(hud, theme::R_CTRL, theme::BG_PANEL, Stroke::new(1.0, theme::BORDER_STRONG), StrokeKind::Inside);
            p.galley(pos2(hud.left() + 10.0, hud.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
        } else {
            let have_sprite = self.selected_sprite().is_some();
            let msg = if self.anim.pending.is_some() {
                "the model draws the first view at this direction\u{2026}".to_owned()
            } else if self.anim.idle {
                format!("{} · {}° is not drawn yet", self.anim.action, self.anim.azimuth)
            } else if have_sprite {
                "this sprite has no animation with these settings".to_owned()
            } else {
                "generate a sprite on the left, or load your own image to animate it".to_owned()
            };
            // the sprite itself stands in the preview until its animation is drawn
            let still = if self.anim.pending.is_none() { self.reference_tex(ctx) } else { None };
            let mid = match &still {
                Some((tex, size)) => {
                    let size = *size as f32;
                    let scale = (canvas.height() * 0.5 / size).min(canvas.width() * 0.45 / size).floor().clamp(1.0, 16.0);
                    let spr = Rect::from_center_size(pos2(canvas.center().x, canvas.top() + canvas.height() * 0.38), Vec2::splat(size * scale));
                    let sh = Rect::from_center_size(pos2(spr.center().x, spr.bottom() + 12.0), vec2(120.0, 8.0));
                    p.rect_filled(sh, CornerRadius::same(4), Color32::from_rgba_unmultiplied(0, 0, 0, 64));
                    p.image(tex.id(), spr, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                    canvas.bottom() - 78.0
                }
                None => canvas.center().y,
            };
            let gal = p.layout_no_wrap(msg, theme::font(13.0), theme::TEXT_SECONDARY);
            p.galley(pos2(canvas.center().x - gal.size().x / 2.0, mid - gal.size().y / 2.0 - 30.0), gal, Color32::WHITE);
            // the one button of this state: Animate (the model never starts by itself) or Load image
            let label = if self.anim.pending.is_some() {
                None
            } else if self.anim.idle {
                Some("Animate")
            } else if !have_sprite {
                Some(if self.png_pick.is_some() { "Pick a file in the dialog\u{2026}" } else { "Load image\u{2026}" })
            } else {
                None
            };
            if let Some(label) = label {
                let bw = if self.anim.idle { 132.0 } else { 220.0 };
                let btn = Rect::from_center_size(pos2(canvas.center().x, mid + 8.0), vec2(bw, 36.0));
                let clicked = ui
                    .scope_builder(egui::UiBuilder::new().max_rect(btn).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
                        kit::button_wh(ui, label, kit::Btn::Primary, bw, 36.0).clicked()
                    })
                    .inner;
                if clicked {
                    if self.anim.idle {
                        self.anim.go = true;
                        self.anim.key = None;
                    } else {
                        self.start_png_pick(ctx);
                    }
                }
            }
            if let Some(e) = &self.anim.error {
                let gal = p.layout(e.clone(), theme::font(12.0), theme::DANGER, canvas.width() - 80.0);
                p.galley(pos2(canvas.center().x - gal.size().x / 2.0, mid + 36.0), gal, Color32::WHITE);
            }
        }

        // ---- the transport: play, Restart, the frame strip and the loop switch, on fixed rects
        p.line_segment([transport.left_top(), transport.right_top()], Stroke::new(1.0, theme::BORDER_SUBTLE));
        let inner = Rect::from_min_max(pos2(transport.left() + 16.0, transport.top()), pos2(transport.right() - 16.0, transport.bottom()));
        let loop_rect = Rect::from_min_size(pos2(inner.right() - 88.0, inner.center().y - 10.0), vec2(88.0, 20.0));
        {
            ui.scope_builder(egui::UiBuilder::new().max_rect(loop_rect).layout(egui::Layout::right_to_left(egui::Align::Center)), |ui| {
                let mut looping = self.anim.looping;
                if kit::switch(ui, &mut looping).clicked() {
                    self.anim.looping = looping;
                }
                ui.add_space(6.0);
                ui.add(egui::Label::new(kit::txt("loop", 12.0).color(theme::TEXT_SECONDARY)));
            });
        }
        if anim.is_some() {
            let play = Rect::from_min_size(pos2(inner.left(), inner.center().y - 18.0), Vec2::splat(36.0));
            let rr = ui.interact(play, ui.scope_id().with("play"), Sense::click());
            let fill = if rr.hovered() { theme::lighten(theme::ACCENT, 0.06) } else { theme::ACCENT };
            p.circle_filled(play.center(), 18.0, fill);
            if self.anim.playing {
                p.rect_filled(Rect::from_center_size(play.center() + vec2(-3.0, 0.0), vec2(3.0, 12.0)), CornerRadius::same(1), Color32::WHITE);
                p.rect_filled(Rect::from_center_size(play.center() + vec2(3.0, 0.0), vec2(3.0, 12.0)), CornerRadius::same(1), Color32::WHITE);
            } else {
                let c = play.center() + vec2(1.0, 0.0);
                p.add(egui::Shape::convex_polygon(vec![c + vec2(-4.0, -6.5), c + vec2(-4.0, 6.5), c + vec2(6.5, 0.0)], Color32::WHITE, Stroke::NONE));
            }
            if rr.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                toggle = true;
            }
            let restart_rect = Rect::from_min_size(pos2(play.right() + 12.0, inner.center().y - 15.0), vec2(76.0, 30.0));
            ui.scope_builder(egui::UiBuilder::new().max_rect(restart_rect).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
                if kit::button_wh(ui, "Restart", kit::Btn::Ghost, 76.0, 30.0).on_hover_text("from the first frame").clicked() {
                    restart = true;
                }
            });
            // the frame strip: 42 px cells with a 4 px gap, centred on the transport row
            let strip = Rect::from_min_max(pos2(restart_rect.right() + 12.0, transport.top()), pos2(loop_rect.left() - 12.0, transport.bottom()));
            let cell = (((strip.width() - 4.0 * (n as f32 - 1.0)) / n as f32).min(42.0)).max(8.0);
            let total_w = cell * n as f32 + 4.0 * (n as f32 - 1.0);
            let max_cells = (((strip.width() + 4.0) / (cell + 4.0)).floor() as usize).max(1);
            let mut x = strip.center().x - total_w / 2.0;
            for (i, t) in self.anim.textures.iter().enumerate() {
                let sel = i == self.anim.shown;
                let sz = if sel { cell + 2.0 } else { cell };
                let r = Rect::from_center_size(pos2(x + sz / 2.0, strip.center().y), Vec2::splat(sz));
                let resp = ui.interact(r, ui.scope_id().with(("frame", i)), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
                kit::checker(ui, r.shrink(1.0), (cell / 16.0).max(4.0));
                p.image(t.id(), r.shrink(1.0), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                if sel {
                    kit::select_frame(ui, r, CornerRadius::same(7));
                } else {
                    p.rect_stroke(r, CornerRadius::same(7), Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
                }
                if resp.clicked() {
                    jump = Some(i);
                }
                x += sz + 4.0;
                if i + 1 >= max_cells { break; }
            }
        }
        if let Some(e) = &self.anim.error {
            if anim.is_some() {
                let gal = p.layout(e.clone(), theme::font(12.0), theme::DANGER, canvas.width() - 40.0);
                p.galley(pos2(canvas.left() + 16.0, transport.top() - gal.size().y - 6.0), gal, Color32::WHITE);
            }
        }
        if toggle {
            self.anim.playing = !self.anim.playing;
            self.anim.t0 = now - self.anim.shown as f64 / self.anim.fps as f64;
        }
        if restart {
            self.anim.t0 = now;
            self.anim.shown = 0;
        }
        if let Some(i) = jump {
            self.anim.playing = false;
            self.anim.shown = i;
        }
    }
}
