//! The left panel: the Generate form, the PNG drop zone and the 3D-farm library list.

use crate::*;

impl App {
    // ------------------------------------------------------------------------------------------- left panel
    pub(crate) fn left_panel(&mut self, ui: &mut egui::Ui) {
        let p = ui.painter().clone();
    ui.spacing_mut().interact_size.y = 18.0;
    ui.spacing_mut().item_spacing.y = 0.0;
        let w = ui.available_width();
        kit::section_header(ui, "1", "Generate sprites");
        ui.add_space(14.0);
        // ---- caption label row
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(kit::med("Caption", 12.5).color(theme::TEXT_SECONDARY)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add(egui::Label::new(kit::mono_txt("EN · RU · UA", 11.5).color(theme::TEXT_MUTED)));
            });
        });
        ui.add_space(8.0);
        // ---- the caption field: a 84 px well with the text and a mono foot line
        let (well, _) = ui.allocate_exact_size(vec2(w, 84.0), Sense::hover());
        let inner = well.shrink2(vec2(13.0, 9.0));
        let mut enter = false;
        let mut focused = false;
        let p0 = ui.painter().clone();
        p0.rect(well, theme::R_FIELD, theme::BG_INPUT, Stroke::new(1.0, theme::BORDER_STRONG), StrokeKind::Inside);
        ui.scope_builder(egui::UiBuilder::new().max_rect(inner).layout(egui::Layout::top_down(egui::Align::Min)), |ui| {
            let edit = ui.add(
                egui::TextEdit::multiline(&mut self.caption)
                    .desired_rows(2)
                    .desired_width(inner.width())
                    .frame(egui::Frame::NONE)
                    .hint_text(RichText::new("e.g. burger skull / рыжий кот / сова").font(theme::font(15.0)).color(theme::TEXT_MUTED))
                    .font(theme::font(15.0))
                    .text_color(theme::TEXT_PRIMARY),
            );
            focused = edit.has_focus();
            enter = edit.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.ctrl);
            ui.with_layout(egui::Layout::left_to_right(egui::Align::BOTTOM).with_main_wrap(false), |ui| {
                ui.add(egui::Label::new(kit::mono_txt("Mix words: \u{201c}burger skull\u{201d}", 11.5).color(theme::TEXT_MUTED)));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add(egui::Label::new(kit::mono_txt("Ctrl Enter", 11.0).color(theme::TEXT_MUTED)));
                });
            });
        });
        if focused {
            // the mock accent frame while the caption is focused (stroke only, so the text stays on top)
            p0.rect_stroke(well, theme::R_FIELD, Stroke::new(1.0, theme::ACCENT), StrokeKind::Inside);
        }
        ui.add_space(14.0);
        // the mix chips (words + variants): the mock's panel shows none for a one-word caption
        if self.caption.split_whitespace().count() > 1 {
            self.word_chips(ui);
            ui.add_space(6.0);
        }
        // ---- example captions
        if !self.presets.is_empty() {
            let examples: Vec<String> = self.presets.iter().take(5).cloned().collect();
            wrapped_row(ui, |ui| {
                for t in examples {
                    if kit::chip(ui, &t, self.caption.trim() == t)
                        .on_hover_text("The demo's example captions (their caption vectors are the demo's precomputed ones).")
                        .clicked()
                    {
                        self.caption = t;
                    }
                }
            });
            ui.add_space(14.0);
        }
        // ---- size
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(vec2(44.0, 34.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.add(egui::Label::new(kit::txt("Size", 12.5).color(theme::TEXT_SECONDARY)));
            });
            let mut idx = [16usize, 24, 32].iter().position(|&s| s == self.size).unwrap_or(0);
            let items = ["16×16", "24×24", "32×32"];
            if kit::segmented(ui, &items, &mut idx) {
                self.size = [16usize, 24, 32][idx.min(2)];
                if !self.count_touched {
                    self.count = default_count(self.size);
                }
                if self.mode == Mode::Draw {
                    self.new_pad(self.size);
                }
            }
        });
        // ---- the outline command (only a generator trained with the outline words has it)
        if self.outline_words {
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                ui.allocate_ui_with_layout(vec2(44.0, 20.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.add(egui::Label::new(kit::txt("Outline", 12.5).color(theme::TEXT_SECONDARY)));
                });
                let mut on = self.outline;
                let label = if on { "dark outline around the sprite" } else { "no outline" };
                if kit::checkbox(ui, &mut on, label) {
                    self.outline = on;
                }
            });
        }
        ui.add_space(14.0);
        // ---- seed + random + each time
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(vec2(44.0, 32.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.add(egui::Label::new(kit::txt("Seed", 12.5).color(theme::TEXT_SECONDARY)));
            });
            ui.scope(|ui| {
                ui.style_mut().spacing.interact_size.y = 32.0;
                let mut s = self.seed as i64;
                if ui.add(egui::DragValue::new(&mut s).range(0..=999_999).speed(1.0)).changed() {
                    self.seed = s.max(0) as u32;
                    self.auto_seed = false;
                }
            });
            if kit::button_wh(ui, "Random", kit::Btn::Ghost, 73.0, 32.0).on_hover_text("random seed").clicked() {
                self.seed = rand_u32() % 1_000_000;
                self.auto_seed = false;
            }
            let mut each = self.auto_seed;
            if kit::checkbox(ui, &mut each, "each time") {
                self.auto_seed = each;
            }
        });
        ui.add_space(14.0);
        // ---- count stepper
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(vec2(44.0, 32.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.add(egui::Label::new(kit::txt("Count", 12.5).color(theme::TEXT_SECONDARY)));
            });
            let (stepper, _) = ui.allocate_exact_size(vec2(98.0, 32.0), Sense::hover());
            ui.painter().rect(stepper, theme::R_CTRL, theme::BG_INPUT, Stroke::new(1.0, theme::BORDER_STRONG), StrokeKind::Inside);
            let minus = Rect::from_min_size(stepper.min + vec2(1.0, 1.0), vec2(30.0, 30.0));
            let plus = Rect::from_min_size(pos2(stepper.right() - 31.0, stepper.top() + 1.0), vec2(30.0, 30.0));
            let rm = ui.interact(minus, ui.scope_id().with("count-"), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
            let rp = ui.interact(plus, ui.scope_id().with("count+"), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
            for (r, hovered) in [(minus, rm.hovered()), (plus, rp.hovered())] {
                if hovered {
                    p.rect_filled(r, theme::R_CHIP, theme::lighten(theme::BG_INPUT, 0.06));
                }
            }
            kit::text_at(ui, pos2(minus.center().x - 4.0, minus.center().y - 9.0), "\u{2212}", theme::font(15.0), theme::TEXT_SECONDARY);
            kit::text_at(ui, pos2(plus.center().x - 4.5, plus.center().y - 9.0), "+", theme::font(15.0), theme::TEXT_SECONDARY);
            let gal = p.layout_no_wrap(self.count.to_string(), theme::font_mono_med(13.0), theme::TEXT_PRIMARY);
            p.galley(pos2(stepper.center().x - gal.size().x / 2.0, stepper.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
            if rm.clicked() && self.count > 1 {
                self.count -= 1;
                self.count_touched = true;
            }
            if rp.clicked() && self.count < MAX_COUNT {
                self.count += 1;
                self.count_touched = true;
            }
            ui.add_space(8.0);
            let hint = format!("\u{2264} {} · one batch", self.batching.rows(self.size));
            ui.add(egui::Label::new(kit::txt(hint, 11.5).color(theme::TEXT_MUTED)));
        });
        ui.add_space(14.0);
        // ---- sampling (collapsed): the demo's sliders, the row budget and the silhouette mode
        let mut open = self.more_open.unwrap_or(false);
        {
            let (row, resp) = ui.allocate_exact_size(vec2(w, 18.0), Sense::click());
            let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
            let chev = if open { "\u{02C5}" } else { "\u{203A}" };
            kit::text_at(ui, pos2(row.left(), row.top()), chev, theme::font(14.0), theme::TEXT_MUTED);
            kit::text_at(ui, pos2(row.left() + 13.0, row.top() + 1.0), "Sampling", theme::font_med(12.5), theme::TEXT_SECONDARY);
            kit::text_at(ui, pos2(row.left() + 77.0, row.top() + 1.0), if open { "" } else { "demo defaults" }, theme::font(12.0), theme::TEXT_MUTED);
            if resp.clicked() {
                open = !open;
                self.more_open = Some(open);
            }
        }
        if open {
            self.more_open = Some(true);
            ui.add_space(6.0);
            let sw = (w - 10.0).min(150.0);
            let mut v = self.g_shape as f32;
            if kit::slider_row(ui, "shape guidance", &mut v, 0.0..=4.0, sw, 2) {
                self.g_shape = ((v as f64) * 4.0).round() / 4.0;
            }
            let mut v = self.g_colour as f32;
            if kit::slider_row(ui, "colour guidance", &mut v, 0.0..=4.0, sw, 2) {
                self.g_colour = ((v as f64) * 4.0).round() / 4.0;
            }
            let mut v = self.temperature as f32;
            if kit::slider_row(ui, "colour temperature", &mut v, 0.3..=1.5, sw, 2) {
                self.temperature = ((v as f64) * 20.0).round() / 20.0;
            }
            let mut v = self.top_p as f32;
            if kit::slider_row(ui, "top-p", &mut v, 0.5..=1.0, sw, 2) {
                self.top_p = ((v as f64) * 100.0).round() / 100.0;
            }
            let mut filter = self.filter;
            if kit::checkbox(ui, &mut filter, "redraw sprites with stray 1–2 px pieces") {
                self.filter = filter;
            }
            ui.add_space(4.0);
            // silhouette: hidden in the mock's compact panel, kept here
            ui.horizontal(|ui| {
                ui.add(egui::Label::new(kit::txt("Silhouette", 12.5).color(theme::TEXT_SECONDARY)));
                let mut idx = match self.mode {
                    Mode::New => 0,
                    Mode::Same => 1,
                    Mode::Draw => 2,
                };
                let items = ["new", "one shape", "draw"];
                if kit::segmented(ui, &items, &mut idx) {
                    self.mode = [Mode::New, Mode::Same, Mode::Draw][idx.min(2)];
                    if self.mode == Mode::Draw && self.pad_size != self.size {
                        self.new_pad(self.size);
                    }
                }
            });
            if self.mode == Mode::Draw {
                self.draw_pad(ui);
            }
            ui.horizontal(|ui| {
                ui.add(egui::Label::new(kit::txt("Rows per step", 12.5).color(theme::TEXT_SECONDARY)));
                egui::ComboBox::from_id_salt("batching").selected_text(self.batching.label()).show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.batching, Batching::DemoGpu, Batching::DemoGpu.label());
                    ui.selectable_value(&mut self.batching, Batching::DemoCpu, Batching::DemoCpu.label());
                    let custom = match self.batching {
                        Batching::Custom(n) => n,
                        _ => self.batching.rows(self.size),
                    };
                    ui.selectable_value(&mut self.batching, Batching::Custom(custom), "custom");
                });
                if let Batching::Custom(n) = &mut self.batching {
                    ui.add(egui::DragValue::new(n).range(1..=64));
                }
            });
            let mut exact = self.encoder_ready;
            if kit::checkbox(ui, &mut exact, "mix: exact caption vectors") {
                self.encoder_ready = exact;
            }
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                if kit::chip(ui, "reset to the demo's defaults", false).clicked() {
                    (self.g_shape, self.g_colour, self.temperature, self.top_p, self.filter) = (1.0, 2.0, 1.0, 1.0, true);
                    self.batching = Batching::DemoGpu;
                    self.encoder_ready = false;
                }
            });
        }
        ui.add_space(14.0);
        // ---- the big Generate / Stop button (44 px)
        let ready = !self.loading && self.ep_active.is_some();
        let (btn_rect, _) = ui.allocate_exact_size(vec2(w, theme::GEN_H), Sense::hover());
        if self.running.is_some() {
            let r = ui.interact(btn_rect, ui.scope_id().with("stop"), Sense::click());
            p.rect(btn_rect, theme::R_FIELD, Color32::TRANSPARENT, Stroke::new(1.0, theme::DANGER), StrokeKind::Inside);
            let label = p.layout_no_wrap("Stop".into(), theme::font_med(14.0), theme::DANGER);
            let kbd_w = 34.0;
            let total = label.size().x + 8.0 + kbd_w;
            let lx = btn_rect.center().x - total / 2.0;
            p.galley(pos2(lx, btn_rect.center().y - label.size().y / 2.0), label.clone(), Color32::WHITE);
            let kbd = Rect::from_min_size(pos2(lx + label.size().x + 8.0, btn_rect.center().y - 9.0), vec2(kbd_w, 18.0));
            p.rect(kbd, CornerRadius::same(5), theme::BG_INPUT, Stroke::new(1.0, theme::BORDER_STRONG), StrokeKind::Inside);
            kit::text_at(ui, pos2(kbd.center().x - 9.0, kbd.center().y - 7.0), "Esc", theme::font_mono(11.0), theme::TEXT_MUTED);
            if r.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                self.worker.stop();
            }
        } else {
            let r = ui.interact(btn_rect, ui.scope_id().with("generate"), Sense::click());
            let fill = if ready { theme::ACCENT } else { theme::lighten(theme::ACCENT_SOFT, 0.06) };
            let fill = if r.hovered() && ready { theme::lighten(fill, 0.06) } else { fill };
            p.rect_filled(btn_rect, theme::R_FIELD, fill);
            let label = p.layout_no_wrap("Generate".into(), theme::font_med(14.0), theme::ON_ACCENT);
            let kbd_w = 52.0;
            let total = label.size().x + 8.0 + kbd_w;
            let lx = btn_rect.center().x - total / 2.0;
            p.galley(pos2(lx, btn_rect.center().y - label.size().y / 2.0), label.clone(), Color32::WHITE);
            let kbd = Rect::from_min_size(pos2(lx + label.size().x + 8.0, btn_rect.center().y - 9.0), vec2(kbd_w, 18.0));
            p.rect(kbd, CornerRadius::same(5), Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 46), Stroke::NONE, StrokeKind::Inside);
            kit::text_at(ui, pos2(kbd.center().x - 20.0, kbd.center().y - 7.0), "Ctrl Enter", theme::font_mono(11.0), theme::ON_ACCENT);
            let r = r.on_hover_cursor(egui::CursorIcon::PointingHand);
            if (r.clicked() || enter) && ready {
                self.start_generation();
            }
        }
        // the "Generating" state: the progress bar and the batch / step line
        if let Some((_, t0)) = self.running {
            let (t, n, done, total) = self.progress;
            let frac = if n > 0 { t as f32 / n as f32 } else { 0.0 };
            let ms = if t > 0 { t0.elapsed().as_secs_f64() * 1e3 / t as f64 } else { 0.0 };
            ui.add_space(10.0);
            kit::progress(ui, w, frac);
            ui.add_space(2.0);
            ui.add(egui::Label::new(
                kit::mono_txt(format!("batch {}/{total} · step {t}/{n} · {ms:.1} ms/step", done + 1), 11.5).color(theme::TEXT_MUTED),
            ));
        }
        if let Some(e) = &self.gen_error {
            ui.add_space(8.0);
            ui.add(egui::Label::new(kit::txt(e, 12.5).color(theme::DANGER)).wrap());
        }
        ui.add_space(14.0);
        // ---- the PNG dropzone
        self.png_dropzone(ui);
        ui.add_space(8.0);
        if let (Some(prep), true) = (self.png_prep_used, self.loaded.is_some() && !self.png_path.is_empty()) {
            let tip = "The animation model was trained on sprites with a dark outline; a thin character without one is \
                       often drawn as an object. Click to pick again: as it is, an outline around the sprite, or an \
                       inline on its own edge.";
            if kit::chip(ui, &format!("prepared: {} \u{b7} change\u{2026}", prep.label().to_lowercase()), false).on_hover_text(tip).clicked() {
                self.load_png(None);
            }
        }
        if let Some(e) = &self.png_error {
            ui.add(egui::Label::new(kit::txt(e, 12.0).color(theme::DANGER)));
        }
        ui.add_space(14.0);
        kit::divider(ui);
        ui.add_space(14.0);
        self.library_panel(ui);
    }

    /// The dashed "drop a PNG" card: a click opens the file dialog (a drop on the window works too).
    pub(crate) fn png_dropzone(&mut self, ui: &mut egui::Ui) {
        let p = ui.painter().clone();
        let w = ui.available_width();
        let (rect, resp) = ui.allocate_exact_size(vec2(w, 54.0), Sense::click());
        let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
        kit::dashed_rect(ui, rect, theme::BORDER_STRONG);
        let picking = self.png_pick.is_some();
        let l1 = if picking { "Pick a file in the dialog\u{2026}" } else { "Drop a PNG sprite here, or Load\u{2026}" };
        let l1 = p.layout_no_wrap(l1.into(), theme::font_med(12.5), theme::TEXT_SECONDARY);
        let l2 = p.layout_no_wrap("any size: fitted to 16 / 24 / 32 px".into(), theme::font(11.0), theme::TEXT_MUTED);
        let total = l1.size().y + 2.0 + l2.size().y;
        p.galley(pos2(rect.center().x - l1.size().x / 2.0, rect.center().y - total / 2.0), l1.clone(), Color32::WHITE);
        p.galley(
            pos2(rect.center().x - l2.size().x / 2.0, rect.center().y - total / 2.0 + l1.size().y + 2.0),
            l2,
            Color32::WHITE,
        );
        let tip = if self.png_path.is_empty() {
            "Animate your own sprite without generating one. A 16 / 24 / 32 px sprite (or an integer upscale of one) is \
             read as it is; any other image is cropped to its opaque part and fitted to 32 px. Or just drop the file on \
             the window."
                .to_string()
        } else {
            format!("{}\n(click to pick another file)", self.png_path)
        };
        let resp = resp.on_hover_text(tip);
        if resp.clicked() {
            self.start_png_pick(ui.ctx());
        }
    }

    /// The Library: search, kind filter chips, and the list with 36 px thumbnails.
    pub(crate) fn library_panel(&mut self, ui: &mut egui::Ui) {
        let w = ui.available_width();
        // ---- header: LIBRARY · N renders · Rescan
        let (head, _) = ui.allocate_exact_size(vec2(w, 24.0), Sense::hover());
        let p = ui.painter().clone();
        let title = p.layout_no_wrap("LIBRARY".into(), theme::font_semi(12.0), theme::TEXT_SECONDARY);
        p.galley(pos2(head.left(), head.top() + 4.0), title, Color32::WHITE);
        let rescan_w = 62.0;
        let rescan = Rect::from_min_size(pos2(head.right() - rescan_w, head.top()), vec2(rescan_w, 24.0));
        let n_txt = if self.lib_entries.is_empty() { String::new() } else { format!("{} renders", self.lib_entries.len()) };
        let n_gal = p.layout_no_wrap(n_txt, theme::font_mono(11.5), theme::TEXT_MUTED);
        p.galley(pos2(rescan.left() - 10.0 - n_gal.size().x, head.top() + 5.0), n_gal, Color32::WHITE);
        ui.scope_builder(egui::UiBuilder::new().max_rect(rescan).layout(egui::Layout::left_to_right(egui::Align::Center)), |ui| {
            ui.add_enabled_ui(self.lib_scan.is_none(), |ui| {
                if kit::chip_h(ui, "Rescan", false, 24.0)
                    .on_hover_text(self.cfg.library_roots().iter().map(|q| q.display().to_string()).collect::<Vec<_>>().join("\n"))
                    .clicked()
                {
                    self.rescan_library(ui.ctx());
                }
            });
        });
        ui.add_space(12.0);
        // ---- search
        let (search, _) = ui.allocate_exact_size(vec2(w, 32.0), Sense::hover());
        p.rect(search, theme::R_CTRL, theme::BG_INPUT, Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
        let lens = pos2(search.left() + 13.0, search.center().y - 1.0);
        p.circle_stroke(lens, 4.0, Stroke::new(1.2, theme::TEXT_MUTED));
        p.line_segment([lens + vec2(2.9, 2.9), lens + vec2(6.4, 6.4)], Stroke::new(1.2, theme::TEXT_MUTED));
        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(Rect::from_min_max(pos2(search.left() + 28.0, search.top() + 6.0), pos2(search.right() - 8.0, search.bottom() - 4.0)))
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
            |ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.lib_filter)
                        .frame(egui::Frame::NONE)
                        .hint_text(RichText::new("Filter name, kind, caption").font(theme::font(12.5)).color(theme::TEXT_MUTED))
                        .font(theme::font(12.5)),
                );
            },
        );
        ui.add_space(10.0);
        // ---- kind filter
        ui.horizontal(|ui| {
            let kinds = [(None, "All"), (Some(Kind::Human), "human"), (Some(Kind::Quadruped), "quadruped"), (Some(Kind::Bird), "bird")];
            for (k, label) in kinds {
                if kit::chip(ui, label, self.lib_kind == k).clicked() {
                    self.lib_kind = k;
                }
            }
        });
        if self.lib_scan.is_some() || !self.lib_scan_note.is_empty() || self.lib_error.is_some() {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if self.lib_scan.is_some() {
                    ui.add(egui::Spinner::new().size(10.0));
                }
                let msg = self.lib_error.clone().unwrap_or_else(|| self.lib_scan_note.clone());
                let color = if self.lib_error.is_some() { theme::DANGER } else { theme::TEXT_MUTED };
                ui.add(egui::Label::new(kit::mono_txt(msg, 11.0).color(color)));
            });
        }
        ui.add_space(10.0);
        let line = |e: &LibraryEntry| format!("{} · {}", e.name, e.caption.trim_start_matches("pixel art icon of "));
        // filter once per change of the text / kind / list (the big farm folder has over 10 000 entries)
        let filter_key = format!("{}|{:?}", self.lib_filter, self.lib_kind);
        if self.lib_view.0 != filter_key || self.lib_view.1 != self.lib_entries.len() {
            let filter = self.lib_filter.to_lowercase();
            let idx = (0..self.lib_entries.len())
                .filter(|&i| {
                    let e = &self.lib_entries[i];
                    (self.lib_kind.is_none() || e.kind == self.lib_kind)
                        && (filter.is_empty() || line(e).to_lowercase().contains(&filter) || e.root.to_lowercase().contains(&filter))
                })
                .collect();
            self.lib_view = (filter_key, self.lib_entries.len(), idx);
        }
        let mut clicked: Option<LibraryEntry> = None;
        let mut want_thumbs: Vec<(String, PathBuf)> = Vec::new();
        let row_h = 56.0;
        egui::ScrollArea::vertical().id_salt("lib").auto_shrink([false, false]).show_rows(ui, row_h, self.lib_view.2.len(), |ui, range| {
            for &i in &self.lib_view.2[range] {
                let e = &self.lib_entries[i];
                let sel = self.selected == Some(Selection::Library) && self.lib_data.as_ref().is_some_and(|d| d.entry.name == e.name && d.entry.dir == e.dir);
                let key = format!("{}/{}", e.root, e.name);
                let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), row_h - 6.0), Sense::click());
                if sel {
                    ui.painter().rect_filled(rect, theme::R_CTRL, theme::ACCENT_SOFT);
                }
                let th = Rect::from_min_size(rect.min + vec2(6.0, 3.0), Vec2::splat(36.0));
                if let Some(Some(t)) = self.thumb_cache.get(&key) {
                    kit::checker(ui, th, 36.0 / 8.0);
                    ui.painter().image(t.id(), th, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                    ui.painter().rect_stroke(th, theme::R_CHIP, Stroke::new(1.0, theme::BORDER_SUBTLE), StrokeKind::Inside);
                } else {
                    ui.painter().rect_filled(th, theme::R_CHIP, theme::BG_INPUT);
                    want_thumbs.push((key, e.dir.clone()));
                }
                // the kind badge first, so the caption knows how much room it has
                let mut badge_w = 0.0;
                if let Some(k) = e.kind {
                    let gal = ui.painter().layout_no_wrap(k.label().to_string(), theme::font_med(11.5), theme::TEXT_SECONDARY);
                    let bw = gal.size().x + 16.0;
                    let br = Rect::from_min_size(pos2(rect.right() - bw - 6.0, rect.center().y - 10.5), vec2(bw, 21.0));
                    ui.painter().rect_filled(br, theme::R_CHIP, theme::BG_RAISED);
                    ui.painter().galley(pos2(br.center().x - gal.size().x / 2.0, br.center().y - gal.size().y / 2.0), gal, Color32::WHITE);
                    badge_w = bw + 10.0;
                }
                let text_x = th.right() + 10.0;
                let text_w = (rect.right() - text_x - badge_w - 6.0).max(40.0);
                let caption = e.caption.trim_start_matches("pixel art icon of ").to_string();
                let gal = ui.painter().layout(caption, theme::font_med(12.5), theme::TEXT_PRIMARY, text_w);
                let clip = Rect::from_min_max(pos2(text_x, rect.top() + 3.0), pos2(text_x + text_w, rect.top() + 3.0 + gal.size().y.min(34.0)));
                ui.painter().with_clip_rect(clip).galley(pos2(text_x, rect.top() + 3.0), gal, Color32::WHITE);
                let id = ui.painter().layout_no_wrap(e.name.clone(), theme::font_mono(11.0), theme::TEXT_MUTED);
                ui.painter().with_clip_rect(rect).galley(pos2(text_x, rect.bottom() - 16.0), id, Color32::WHITE);
                let resp = resp.on_hover_text(format!("{}\nroot: {}", e.dir.display(), e.root));
                if resp.clicked() {
                    clicked = Some(e.clone());
                }
            }
        });
        self.request_thumbs(&want_thumbs);
        if let Some(e) = clicked {
            match LibraryData::load(&e) {
                Ok(d) => {
                    if !d.sizes().contains(&self.anim.lib_size) {
                        self.anim.lib_size = d.sizes().last().copied().unwrap_or(32);
                    }
                    self.lib_data = Some(Arc::new(d));
                    self.lib_error = None;
                    self.selected = Some(Selection::Library);
                    self.anim.animator = AnimatorChoice::Library;
                    self.anim.kind = KindChoice::Auto;
                    self.anim.action.clear();
                }
                Err(err) => self.lib_error = Some(format!("{err:#}")),
            }
        }
    }
}
