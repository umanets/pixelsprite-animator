//! The right panel (Animate inspector), the settings window, caption word chips and the drawing pad.

use crate::*;

impl App {
    // ------------------------------------------------------------------------------------------ right panel
    pub(crate) fn right_panel(&mut self, ui: &mut egui::Ui) {
        let p = ui.painter().clone();
    ui.spacing_mut().interact_size.y = 18.0;
    ui.spacing_mut().item_spacing.y = 0.0;
        let ctx = ui.ctx().clone();
        kit::section_header(ui, "2", "Animate");
        ui.add_space(14.0);
        let Some((sprite, caption, class)) = self.selected_sprite() else {
            ui.add(egui::Label::new(kit::txt("Select a generated sprite (or a library entry).", 13.0).color(theme::TEXT_SECONDARY)));
            return;
        };
        // ---- the selected sprite card (307 × 90)
        let w = ui.available_width();
        let (card_rect, _) = ui.allocate_exact_size(vec2(w, 90.0), Sense::hover());
        {
            p.rect(card_rect, theme::R_CARD, theme::BG_RAISED, Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
            let thumb = Rect::from_min_size(card_rect.min + vec2(11.0, 11.0), Vec2::splat(68.0));
            let key = format!("{:?}{}/{}", self.selected, self.anim.lib_size, self.loaded_id);
            let tex = match &self.lib_tex {
                Some((k, _, t)) if *k == key => t.clone(),
                _ => {
                    let t = texture(&ctx, "reference", &sprite);
                    self.lib_tex = Some((key, sprite.size, t.clone()));
                    t
                }
            };
            p.rect(thumb, theme::R_CTRL, theme::BG_INPUT, Stroke::new(1.0, theme::ACCENT_SELECT), StrokeKind::Inside);
            let inner = thumb.shrink(2.0);
            kit::checker(ui, inner, 64.0 / sprite.size as f32);
            p.image(tex.id(), inner, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            let x = thumb.right() + 12.0;
            kit::text_at(ui, pos2(x, card_rect.top() + 13.0), if caption.is_empty() { "no text" } else { &caption }, theme::font_semi(15.0), theme::TEXT_PRIMARY);
            kit::text_at(ui, pos2(x, card_rect.top() + 37.0), &format!("{0}×{0} · seed {1}", sprite.size, self.current_seed()), theme::font_mono(11.5), theme::TEXT_MUTED);
            // the kind badge
            let (txt, kind) = match class.as_ref().and_then(|c| c.kind) {
                Some(k) => (format!("{} · {}", k.label(), class.as_ref().map(|c| c.reason.clone()).unwrap_or_default()), kit::BadgeKind::Success),
                None => ("not animatable".to_string(), kit::BadgeKind::Warn),
            };
            let gal = p.layout_no_wrap(txt.clone(), theme::font_med(11.5), if kind == kit::BadgeKind::Success { theme::SUCCESS } else { theme::WARN });
            let bw = (gal.size().x + 16.0).min(card_rect.right() - x - 12.0);
            let badge = Rect::from_min_size(pos2(x, card_rect.top() + 56.0), vec2(bw, 21.0));
            p.rect_filled(badge, theme::R_CHIP, if kind == kit::BadgeKind::Success { theme::SUCCESS_SOFT } else { theme::BG_RAISED });
            let gal = p.layout_no_wrap(txt, theme::font_med(11.5), if kind == kit::BadgeKind::Success { theme::SUCCESS } else { theme::WARN });
            let clipped = ui.painter().with_clip_rect(badge);
            clipped.galley(pos2(badge.left() + 8.0, badge.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
        }
        ui.add_space(14.0);
        // ---- Kind
        let auto_label = match class.as_ref().and_then(|c| c.kind) {
            Some(k) => format!("Auto ({})", k.label()),
            None => "Auto (not animatable)".into(),
        };
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(vec2(62.0, 32.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.add(egui::Label::new(kit::txt("Kind", 12.5).color(theme::TEXT_SECONDARY)));
            });
            ui.scope(|ui| {
                ui.style_mut().spacing.interact_size.y = 32.0;
                egui::ComboBox::from_id_salt("kind")
                    .selected_text(kit::txt(
                        match self.anim.kind {
                            KindChoice::Auto => auto_label.clone(),
                            KindChoice::Force(k) => k.label().to_string(),
                            KindChoice::NotAnimatable => "not animatable".into(),
                        },
                        13.0,
                    ))
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.anim.kind, KindChoice::Auto, auto_label.clone());
                        for k in Kind::ALL {
                            ui.selectable_value(&mut self.anim.kind, KindChoice::Force(k), k.label());
                        }
                        ui.selectable_value(&mut self.anim.kind, KindChoice::NotAnimatable, "not animatable");
                    });
            });
        });
        ui.add_space(14.0);
        // ---- Animator
        let have = self.jobs.anim.available();
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(vec2(62.0, 32.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.add(egui::Label::new(kit::txt("Animator", 12.5).color(theme::TEXT_SECONDARY)));
            });
            ui.scope(|ui| {
                ui.style_mut().spacing.interact_size.y = 32.0;
                let label = match self.anim.animator {
                    AnimatorChoice::Onnx => format!("\u{25CF} ONNX model · anim39"),
                    AnimatorChoice::Mock => "Mock · procedural".to_string(),
                    AnimatorChoice::Library => "Library · 3D farm".to_string(),
                };
                egui::ComboBox::from_id_salt("animator")
                    .selected_text(kit::txt(label, 13.0))
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.anim.animator, AnimatorChoice::Mock, AnimatorChoice::Mock.label());
                        ui.add_enabled_ui(self.lib_data.is_some(), |ui| {
                            ui.selectable_value(&mut self.anim.animator, AnimatorChoice::Library, AnimatorChoice::Library.label());
                        });
                        ui.add_enabled_ui(have, |ui| {
                            ui.selectable_value(&mut self.anim.animator, AnimatorChoice::Onnx, AnimatorChoice::Onnx.label())
                                .on_disabled_hover_text(format!("model files missing in {}", self.jobs.anim.dir.display()));
                        });
                    });
            });
        });
        ui.add_space(14.0);
        // ---- the walk-front reference option
        {
            let mut value = self.anim.ref_from_walk_front;
            let hint = if self.anim.animator == AnimatorChoice::Onnx {
                None
            } else if self.anim.animator == AnimatorChoice::Mock {
                Some("The mock only moves the reference sprite — no reference needed. Switch the animator to the ONNX model.")
            } else {
                Some("The library plays precomputed 3D-farm renders — no reference needed. Switch the animator to the ONNX model.")
            };
            if kit::checkbox_card(
                ui,
                &mut value,
                "Build everything from the first walk frame (front view)",
                "experimental · reference: walk · 0° front · frame 1",
                hint,
            ) {
                self.anim.ref_from_walk_front = value;
                self.apply_anim_ref();
                if let Err(e) = animator_ui::config::save_flag("ref_from_walk_front", value) {
                    self.status = format!("Could not save the setting: {e}");
                }
            }
        }
        ui.add_space(14.0);
        let kind = self.effective_kind(&class);
        let Some(kind) = kind else {
            ui.add(egui::Label::new(kit::txt("Not animatable (for now). Override the kind above if the caption words misled the detector.", 12.5).color(theme::WARN)));
            return;
        };
        self.fix_action(Some(kind));
        let actions = self.actions_for(kind);
        // ---- actions
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(kit::med("Action", 12.5).color(theme::TEXT_SECONDARY)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add(egui::Label::new(kit::mono_txt(format!("{} · {}", kind.label(), actions.len()), 11.5).color(theme::TEXT_MUTED)));
            });
        });
        ui.add_space(14.0);
        wrapped_row(ui, |ui| {
            for a in &actions {
                if kit::action_chip(ui, &a.name, self.anim.action == a.name).on_hover_text(format!("{} frames by default", a.frames)).clicked()
                    && self.anim.action != a.name
                {
                    self.anim.action = a.name.clone();
                    self.anim.n_frames = a.frames.max(1);
                    self.anim.looping = a.looping;
                }
            }
            if actions.is_empty() {
                ui.add(egui::Label::new(kit::txt("this animator has no actions for this kind", 12.0).color(theme::TEXT_SECONDARY)));
            }
        });
        ui.add_space(14.0);
        // ---- direction compass + the frames / fps sliders
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.add(egui::Label::new(kit::med("Direction", 12.5).color(theme::TEXT_SECONDARY)));
                ui.add_space(6.0);
                let key = format!("{:?}{}/{}", self.selected, self.anim.lib_size, self.loaded_id);
                let tex = match &self.lib_tex {
                    Some((k, _, t)) if *k == key => t.clone(),
                    _ => texture(&ctx, "compass", &sprite),
                };
                let grid: [[Option<u16>; 3]; 3] = [[Some(315), Some(0), Some(45)], [Some(270), None, Some(90)], [Some(225), Some(180), Some(135)]];
                for row in grid {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for cell in row {
                            match cell {
                                Some(a) => {
                                    let active = self.anim.azimuth == a;
                                    let (rect, resp) = ui.allocate_exact_size(vec2(58.0, 40.0), Sense::click());
                                    let (fill, stroke, fg) = if active {
                                        (theme::ACCENT, theme::ACCENT, Color32::WHITE)
                                    } else {
                                        (theme::BG_RAISED, theme::BORDER_SUBTLE, theme::TEXT_SECONDARY)
                                    };
                                    p.rect(rect, theme::R_CTRL, fill, Stroke::new(1.0, stroke), StrokeKind::Inside);
                                    let gal = p.layout_no_wrap(format!("{a}°"), theme::font_mono_med(11.0), fg);
                                    p.galley(pos2(rect.center().x - gal.size().x / 2.0, rect.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
                                    let resp = resp.on_hover_text(match a {
                                        0 => "0° — front (facing the viewer)".to_string(),
                                        90 => "90° — side / profile, facing screen-left".to_string(),
                                        180 => "180° — back".to_string(),
                                        270 => "270° — side, facing screen-right".to_string(),
                                        _ => format!("{a}°"),
                                    });
                                    if resp.clicked() {
                                        self.anim.azimuth = a;
                                    }
                                }
                                None => {
                                    let (rect, _) = ui.allocate_exact_size(vec2(58.0, 40.0), Sense::hover());
                                    ui.painter().rect(rect, theme::R_CTRL, theme::BG_INPUT, Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
                                    let inner = Rect::from_center_size(rect.center(), Vec2::splat(36.0));
                                    kit::checker(ui, inner, 36.0 / sprite.size as f32);
                                    ui.painter().image(tex.id(), inner, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                                    ui.painter().rect_stroke(inner, CornerRadius::same(4), Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
                                }
                            }
                        }
                    });
                    ui.add_space(4.0);
                }
            });
            ui.add_space(14.0);
            ui.vertical(|ui| {
                let sl = ui.available_width().min(100.0);
                let mut frames = self.anim.n_frames as f32;
                if kit::slider_row(ui, "Frames", &mut frames, 1.0..=32.0, sl, 0) {
                    self.anim.n_frames = frames.round().max(1.0) as usize;
                }
                ui.add_space(12.0);
                let mut fps = self.anim.fps;
                if kit::slider_row(ui, "FPS", &mut fps, 1.0..=30.0, sl, 0) {
                    self.anim.fps = fps.round().max(1.0);
                }
            });
        });

        self.refresh_animation(&ctx);
        // ---- exports
        ui.add_space(14.0);
        kit::divider(ui);
        ui.add_space(15.0);
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(kit::semi_txt("EXPORT", 12.0).color(theme::TEXT_SECONDARY)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mut idx = [1u32, 2, 4, 8].iter().position(|&s| s == self.export_scale).unwrap_or(3);
                if kit::segmented_small(ui, &["×1", "×2", "×4", "×8"], &mut idx) {
                    self.export_scale = [1u32, 2, 4, 8][idx.min(3)];
                }
            });
        });
        ui.add_space(14.0);
        let bw = ((ui.available_width() - 16.0) / 3.0).floor();
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            if kit::button_wh(ui, "Sheet PNG", kit::Btn::Ghost, bw, 32.0).clicked() {
                self.export_selected(ExportKind::Sheet);
            }
            if kit::button_wh(ui, "GIF", kit::Btn::Primary, bw, 32.0).clicked() {
                self.export_selected(ExportKind::Gif);
            }
            if kit::button_wh(ui, "8-dir sheet", kit::Btn::Ghost, bw, 32.0).on_hover_text("rows = 0°, 45°, …, 315°").clicked() {
                self.export_selected(ExportKind::Directions);
            }
        });
        if let Some((msg, path)) = self.last_export.clone() {
            ui.add_space(6.0);
            ui.add(egui::Label::new(kit::mono_txt(msg, 11.0).color(theme::TEXT_MUTED)));
            if ui.link(kit::txt("open folder", 11.0).color(theme::ACCENT_SELECT)).clicked() {
                let dir = if path.is_dir() { path.clone() } else { path.parent().map(|p| p.to_path_buf()).unwrap_or_default() };
                let _ = std::process::Command::new("explorer").arg(dir).spawn();
            }
        }
    }

    /// The seed of the selected run (for the inspector's meta line).
    pub(crate) fn current_seed(&self) -> u32 {
        match self.selected {
            Some(Selection::Generated { run, .. }) => self.runs.iter().find(|r| r.id == run).map(|r| r.seed).unwrap_or(0),
            _ => 0,
        }
    }
}
