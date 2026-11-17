//! The top bar and the status bar.

use crate::*;

impl App {
    // -------------------------------------------------------------------------------------------------- top bar
    /// The 52 px bar: the pixel logo + brand, the three-step stepper in its pill, the provider chip, Settings and
    /// Export.
    pub(crate) fn top_bar(&mut self, ui: &mut egui::Ui) {
        let (bar, _) = ui.allocate_exact_size(vec2(ui.available_width(), theme::TOP_BAR_H), Sense::hover());
        let p = ui.painter().clone();
        // ---- brand: 28 px accent square with the pixel cross, the name, then "animator" in mono
        let logo = Rect::from_min_size(pos2(bar.left() + theme::PAD, bar.center().y - 14.0), Vec2::splat(28.0));
        p.rect_filled(logo, theme::R_CTRL, theme::ACCENT);
        for (dx, dy) in [(12.0, 8.0), (8.0, 12.0), (12.0, 12.0), (16.0, 12.0), (12.0, 16.0)] {
            p.rect_filled(Rect::from_min_size(logo.min + vec2(dx, dy), Vec2::splat(4.0)), CornerRadius::ZERO, Color32::WHITE);
        }
        let name = p.layout_no_wrap("Pixel Sprite Studio".into(), theme::font_semi(14.0), theme::TEXT_PRIMARY);
        let name_pos = pos2(logo.right() + 10.0, bar.center().y - name.size().y / 2.0);
        p.galley(name_pos, name.clone(), Color32::WHITE);
        let tag = p.layout_no_wrap("animator".into(), theme::font_mono(12.0), theme::TEXT_MUTED);
        p.galley(pos2(name_pos.x + name.size().x + 10.0, bar.center().y - tag.size().y / 2.0), tag.clone(), Color32::WHITE);
        let brand_right = name_pos.x + name.size().x + 10.0 + tag.size().x;

        // ---- right cluster: provider chip 163, Settings 81, Export 69, 16 px gaps
        let (w_settings, w_export) = (81.0, 69.0);
        let x_export = bar.right() - theme::PAD - w_export;
        let x_settings = x_export - theme::GAP_LG - w_settings;
        let y = bar.center().y - 16.0;
        let (col, ep_name) = match self.ep_active {
            Some(ep) => (
                match ep {
                    Ep::Cuda => theme::SUCCESS,
                    Ep::DirectMl => theme::WARN,
                    Ep::Cpu => theme::TEXT_MUTED,
                },
                ep.label().to_string(),
            ),
            None => (theme::DANGER, "no model".into()),
        };
        // the chip grows with its text ("DirectML" is wider than "CUDA"), never below the mock's 163 px
        let ms_text = self.ms_per_step.map(|ms| format!("{ms:.1} ms/step"));
        let w_name = p.layout_no_wrap(ep_name.clone(), theme::font_med(12.5), theme::TEXT_PRIMARY).size().x;
        let w_ms = ms_text.as_ref().map_or(0.0, |t| p.layout_no_wrap(t.clone(), theme::font_mono(12.0), theme::TEXT_SECONDARY).size().x + 10.0);
        let w_chip = (27.0 + w_name + w_ms + 14.0).max(163.0);
        let x_chip = x_settings - theme::GAP_LG - w_chip;
        let chip_rect = Rect::from_min_size(pos2(x_chip, y), vec2(w_chip, 32.0));
        p.rect(chip_rect, theme::R_CTRL, theme::BG_RAISED, Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
        p.circle_filled(pos2(chip_rect.left() + 15.0, chip_rect.center().y), 4.0, col);
        kit::text_at(ui, pos2(chip_rect.left() + 27.0, chip_rect.center().y - 8.0), &ep_name, theme::font_med(12.5), theme::TEXT_PRIMARY);
        if let Some(t) = &ms_text {
            kit::text_at(ui, pos2((chip_rect.left() + 27.0 + w_name + 10.0).max(chip_rect.left() + 70.0), chip_rect.center().y - 8.0), t, theme::font_mono(12.0), theme::TEXT_SECONDARY);
        }
        let mut tip = self.ep_describe.clone();
        if !self.ep_fallbacks.is_empty() {
            tip.push_str(&format!("\n\n(!) {} provider(s) skipped:\n{}", self.ep_fallbacks.len(), self.ep_fallbacks.join("\n\n")));
        }
        ui.interact(chip_rect, ui.scope_id().with("epchip"), Sense::hover()).on_hover_text(tip);

        let settings_rect = Rect::from_min_size(pos2(x_settings, y), vec2(w_settings, 32.0));
        let export_rect = Rect::from_min_size(pos2(x_export, y), vec2(w_export, 32.0));
        ui.scope_builder(egui::UiBuilder::new().max_rect(settings_rect).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
            if kit::button_wh(ui, "Settings", kit::Btn::Ghost, w_settings, 32.0).clicked() {
                self.settings_open = true;
            }
        });
        ui.scope_builder(egui::UiBuilder::new().max_rect(export_rect).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
            if kit::button_wh(ui, "Export", kit::Btn::Primary, w_export, 32.0)
                .on_hover_text("Export the selected animation as a GIF (the right panel has the other formats)")
                .clicked()
            {
                self.export_selected(ExportKind::Gif);
            }
        });

        // ---- the stepper pill, centred between the brand and the right cluster
        let gen_done = self.selected.is_some();
        let export_done = self.last_export.is_some() || self.scene.last_export.is_some();
        let step2 = if export_done { 2 } else if gen_done { 1 } else { 0 };
        let steps: [(&str, &str, u8); 3] = [("1", "Generate", if gen_done { 2 } else { 1 }), ("2", "Animate", step2), ("3", "Export", if export_done { 2 } else { 0 })];
        let label_w: Vec<f32> = steps
            .iter()
            .map(|(_, label, _)| p.layout_no_wrap((*label).to_owned(), theme::font(13.0), theme::TEXT_PRIMARY).size().x)
            .collect();
        let widths: Vec<f32> = label_w.iter().map(|w| 46.0 + w).collect();
        let track_w: f32 = 8.0 + widths.iter().sum::<f32>() + 2.0 * (steps.len() as f32 - 1.0);
        let free_l = brand_right + theme::GAP_LG;
        let free_r = x_chip - theme::GAP_LG;
        let track_x = (free_l + ((free_r - free_l) - track_w) / 2.0).round();
        let track = Rect::from_min_size(pos2(track_x, bar.center().y - 19.0), vec2(track_w, 38.0));
        p.rect(track, theme::R_FIELD, theme::BG_INPUT, Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
        let mut x = track.left() + 4.0;
        for (i, (n, label, state)) in steps.iter().enumerate() {
            let w = widths[i];
            let r = Rect::from_min_size(pos2(x, track.top() + 4.0), vec2(w, 30.0));
            if *state == 1 {
                p.rect(r, theme::R_CTRL, theme::ACCENT_SOFT, Stroke::new(1.0, theme::ACCENT), StrokeKind::Inside);
            }
            let (dot_fill, digit_color, label_color) = match state {
                2 => (theme::SUCCESS, Color32::WHITE, theme::TEXT_PRIMARY),
                1 => (theme::ACCENT, Color32::WHITE, theme::TEXT_PRIMARY),
                _ => (theme::BG_RAISED, theme::TEXT_MUTED, theme::TEXT_MUTED),
            };
            let dot = pos2(r.left() + 12.0 + 9.0, r.center().y);
            p.circle_filled(dot, 9.0, dot_fill);             if *state == 2 {                 kit::check_mark(&p, Rect::from_center_size(dot, Vec2::splat(9.0)), Color32::WHITE);             } else {                 let gal = p.layout_no_wrap(n.to_string(), theme::font_mono_med(10.5), digit_color);                 p.galley(pos2(dot.x - gal.size().x / 2.0, dot.y - gal.size().y / 2.0), gal, Color32::WHITE);             }
            let font = if *state == 1 { theme::font_med(13.0) } else { theme::font(13.0) };
            kit::text_at(ui, pos2(dot.x + 9.0 + 8.0, r.center().y - 8.0), label, font, label_color);
            x += w + 2.0;
        }
        // the bar's bottom border
        p.line_segment([bar.left_bottom(), bar.right_bottom()], Stroke::new(1.0, theme::BORDER_SUBTLE));
    }

    // ------------------------------------------------------------------------------------------------- statusbar
    /// The 28 px status bar: the execution provider selector, the active provider, the latest message and the last
    /// saved file.
    pub(crate) fn status_bar(&mut self, ui: &mut egui::Ui) {
        let (bar, _) = ui.allocate_exact_size(vec2(ui.available_width(), theme::STATUS_H), Sense::hover());
        let p = ui.painter().clone();
        let cy = bar.center().y;
        p.line_segment([bar.left_top(), bar.right_top()], Stroke::new(1.0, theme::BORDER_SUBTLE));
        // ---- the right end: the saved file, then "Open folder"
        let mut x_max = bar.right() - theme::PAD;
        let last = self.scene.last_export.clone().or_else(|| self.last_export.clone()).or_else(|| self.last_save.clone());
        if let Some((msg, path)) = last {
            let open = p.layout_no_wrap("Open folder".into(), theme::font_med(12.0), theme::ACCENT_SELECT);
            let open_rect = Rect::from_min_size(pos2(x_max - open.size().x, cy - 9.0), vec2(open.size().x, 18.0));
            let rr = ui.interact(open_rect, ui.scope_id().with("openfolder"), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
            p.galley(open_rect.min, open, Color32::WHITE);
            if rr.clicked() {
                let dir = if path.is_dir() { path.clone() } else { path.parent().map(|q| q.to_path_buf()).unwrap_or_default() };
                let _ = std::process::Command::new("explorer").arg(dir).spawn();
            }
            let short = match path.strip_prefix(exports_dir()) {
                Ok(rel) => format!("exports/{}", rel.display()),
                Err(_) => path.display().to_string(),
            };
            let gal = p.layout_no_wrap(short, theme::font_mono(12.0), theme::TEXT_MUTED);
            let gx = open_rect.left() - 16.0 - gal.size().x;
            p.galley(pos2(gx, cy - gal.size().y / 2.0), gal.clone(), Color32::WHITE);
            ui.interact(Rect::from_min_size(pos2(gx, cy - 9.0), gal.size()), ui.scope_id().with("savedtip"), Sense::hover()).on_hover_text(msg);
            x_max = gx - 20.0;
        }
        // ---- the left: selector, active provider, message
        let mut x = bar.left() + theme::PAD;
        kit::text_at(ui, pos2(x, cy - 7.5), "Execution provider", theme::font(12.0), theme::TEXT_MUTED);
        x += 111.0;
        let label = self.ep_choice.label().to_string();
        let gal = p.layout_no_wrap(label, theme::font_med(12.0), theme::TEXT_SECONDARY);
        let sel_rect = Rect::from_min_size(pos2(x - 2.0, cy - 10.0), vec2(gal.size().x + 6.0, 20.0));
        let r = ui.interact(sel_rect, ui.scope_id().with("epsel"), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
        if r.hovered() {
            p.rect_filled(sel_rect, theme::R_CHIP, theme::BG_RAISED);
        }
        p.galley(pos2(x, cy - gal.size().y / 2.0), gal.clone(), Color32::WHITE);
        let before = self.ep_choice;
        egui::Popup::menu(&r).close_behavior(egui::PopupCloseBehavior::CloseOnClick).show(|ui| {
            ui.set_min_width(240.0);
            for c in [EpChoice::Auto, EpChoice::Only(Ep::Cuda), EpChoice::Only(Ep::DirectMl), EpChoice::Only(Ep::Cpu)] {
                ui.selectable_value(&mut self.ep_choice, c, c.label());
            }
        });
        if before != self.ep_choice && self.running.is_none() {
            self.loading = true;
            self.status = format!("Switching to {}…", self.ep_choice.label());
            self.worker.send(Cmd::Load(self.ep_choice));
        } else if before != self.ep_choice {
            self.ep_choice = before; // not while drawing
        }
        x += gal.size().x + 18.0;
        if self.loading {
            ui.add(egui::Spinner::new().size(12.0));
        }
        if !self.ep_fallbacks.is_empty() {
            let txt = format!("(!) {} provider(s) skipped", self.ep_fallbacks.len());
            let gal = p.layout_no_wrap(txt.clone(), theme::font(12.0), theme::WARN);
            let rr = ui.interact(Rect::from_min_size(pos2(x, cy - 8.0), gal.size()), ui.scope_id().with("skipped"), Sense::hover());
            rr.on_hover_text(self.ep_fallbacks.join("\n\n"));
            p.galley(pos2(x, cy - gal.size().y / 2.0), gal.clone(), Color32::WHITE);
            x += gal.size().x + 16.0;
        }
        if !self.ep_describe.is_empty() && x < x_max - 220.0 {
            p.circle_filled(pos2(x + 3.5, cy), 3.5, theme::SUCCESS);
            let txt = format!("active: {}", self.ep_describe.chars().take(48).collect::<String>());
            let gal = p.layout_no_wrap(txt, theme::font(12.0), theme::SUCCESS);
            p.galley(pos2(x + 12.0, cy - gal.size().y / 2.0), gal.clone(), Color32::WHITE);
            x += 12.0 + gal.size().x + 20.0;
        }
        // the latest message (or the load error)
        // the CUDA libraries download, when an NVIDIA driver is there but the libraries are not
        let cuda_mb = animator_ui::ep::missing_cuda_mb(&self.dll);
        if cuda_mb > 0 && animator_ui::ep::has_nvidia_driver() && self.cudnn_done.is_none() {
            let txt = if self.cudnn_busy { "Downloading CUDA libraries…".to_owned() } else { format!("Download CUDA libraries (~{cuda_mb} MB)") };
            let gal = p.layout_no_wrap(txt, theme::font_med(12.0), theme::ACCENT_SELECT);
            if x + gal.size().x < x_max {
                let rr = ui
                    .interact(Rect::from_min_size(pos2(x, cy - 9.0), gal.size()), ui.scope_id().with("cudnn"), Sense::click())
                    .on_hover_text(self.dll.summary())
                    .on_hover_cursor(egui::CursorIcon::PointingHand);
                p.galley(pos2(x, cy - gal.size().y / 2.0), gal.clone(), Color32::WHITE);
                if rr.clicked() && !self.cudnn_busy {
                    self.cudnn_busy = true;
                    self.worker.send(Cmd::DownloadCudnn);
                }
                x += gal.size().x + 20.0;
            }
        }
        let msg = match &self.load_error {
            Some(e) => (e.clone(), theme::DANGER),
            None => (self.status.clone(), theme::TEXT_MUTED),
        };
        let gal = p.layout_no_wrap(msg.0, theme::font(12.0), msg.1);
        if x + gal.size().x < x_max {
            p.galley(pos2(x, cy - gal.size().y / 2.0), gal.clone(), Color32::WHITE);
            x += gal.size().x + 20.0;
        }
        if let Some(s) = &self.cudnn_done {
            let gal = p.layout_no_wrap(s.clone(), theme::font(12.0), theme::SUCCESS);
            if x + gal.size().x < x_max {
                p.galley(pos2(x, cy - gal.size().y / 2.0), gal, Color32::WHITE);
            }
        }
    }
}
