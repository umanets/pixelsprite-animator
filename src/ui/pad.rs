//! Caption word chips and the drawing pad.

use crate::*;

impl App {
    /// Under the caption: its words (the strongest solid; attributes in italics; "?" = not in the lexicon, its kind
    /// is guessed when generating; a click flips object / attribute) and the variant chips, as the demo.
    pub(crate) fn word_chips(&mut self, ui: &mut egui::Ui) {
        let Some(lex) = self.lexicon.clone() else { return };
        let text = self.caption.trim().to_string();
        if text.is_empty() {
            return;
        }
        let kinds: HashMap<String, ConceptKind> = self.learned_kind.iter().chain(self.kind_override.iter()).map(|(k, v)| (k.clone(), *v)).collect();
        let parsed = mix::parse(&text, &lex, &kinds);
        let concepts = parsed.concepts;
        if concepts.len() < 2 {
            return;
        }
        let pmax = concepts.iter().map(|c| c.p).fold(0.0, f64::max);
        let mut flip = None;
        wrapped_row(ui, |ui| {
            for c in &concepts {
                let unknown = !c.known && !self.learned_kind.contains_key(&c.text);
                let label = format!("{}{}", c.text, if unknown { "?" } else { "" });
                let alpha = (0.35 + 0.65 * c.p / pmax) as f32;
                let col = if c.kind == ConceptKind::Attribute { theme::TEXT_SECONDARY } else { theme::TEXT_PRIMARY }.gamma_multiply(alpha);
                let mut tip = format!(
                    "{} · {:.0}% · click to make it {}",
                    c.kind.label(),
                    c.p * 100.0,
                    if c.kind == ConceptKind::Object { "an attribute" } else { "an object" }
                );
                if unknown {
                    tip.push_str("\nnot in the lexicon: its kind comes from the nearest lexicon word when you generate");
                }
                let mut rt = RichText::new(label).color(col);
                if c.kind == ConceptKind::Attribute {
                    rt = rt.italics();
                }
                if kit::chip_rt(ui, rt, c.kind == ConceptKind::Object, theme::CHIP_H)
                    .on_hover_text(tip)
                    .clicked()
                {
                    flip = Some((c.text.clone(), c.kind.flipped()));
                }
            }
        });
        let preset_needs_encoder = self.presets.contains(&text) && !self.encoder_ready && concepts.iter().any(|c| !lex.index.contains_key(&c.text));
        let mut note = if self.mode != Mode::New {
            "mixing works with new shapes".to_string()
        } else if preset_needs_encoder {
            "one caption (as the demo before its text encoder loads; Sampling: exact caption vectors)".to_string()
        } else {
            match mix::plan(&concepts, &self.variants_on) {
                Some(pl) => format!("mix · {} variant{}", pl.count, if pl.count == 1 { "" } else { "s" }),
                None => "one caption".into(),
            }
        };
        if parsed.dropped > 0 {
            note.push_str(&format!(" · only {} words used", mix::MAX));
        }
        ui.label(RichText::new(note).small().color(theme::TEXT_MUTED));
        if let Some((t, k)) = flip {
            self.kind_override.insert(t, k);
        }
        if self.mode == Mode::New {
            wrapped_row(ui, |ui| {
                for k in VariantKind::ALL {
                    let on = self.variants_on.contains(&k);
                    let why = mix::unfit(k, &concepts);
                    let (r, _) = ui.allocate_exact_size(Vec2::splat(12.0), Sense::hover());
                    paint_variant_icon(ui, r, k, on && why.is_none());
                    let mut label = RichText::new(k.title()).small();
                    if why.is_some() {
                        label = label.color(theme::TEXT_MUTED).strikethrough();
                    }
                    let tip = match why {
                        Some(w) => format!("{} (not for this caption: {w})", k.hint()),
                        None => k.hint().to_string(),
                    };
                    if kit::chip_rt(ui, label, on, theme::CHIP_H).on_hover_text(tip).clicked() {
                        if on && self.variants_on.len() > 1 {
                            self.variants_on.retain(|x| *x != k);
                        } else if !on {
                            self.variants_on.push(k);
                        }
                    }
                }
            });
        }
    }

    /// The silhouette pad: paint with the left button, erase with the right button or Shift.
    pub(crate) fn draw_pad(&mut self, ui: &mut egui::Ui) {
        if self.pad_size != self.size {
            self.new_pad(self.size);
        }
        let n = self.pad_size;
        let side = ui.available_width().clamp(160.0, 272.0);
        let cell = side / n as f32;
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(side), Sense::click_and_drag());
        if resp.is_pointer_button_down_on() {
            if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                let at = |v: f32| ((v / cell).floor().max(0.0) as usize).min(n - 1);
                let here = (at(pos.x - rect.min.x), at(pos.y - rect.min.y));
                let erase = ui.input(|i| i.pointer.secondary_down() || i.modifiers.shift);
                let (paint, from) = match self.pad_stroke {
                    Some((paint, last)) => (paint, last),
                    None => (!erase, here),
                };
                // every cell between two pointer positions, so fast strokes leave no gaps (the demo's paintLine)
                let steps = from.0.abs_diff(here.0).max(from.1.abs_diff(here.1)).max(1);
                for k in 0..=steps {
                    let f = k as f32 / steps as f32;
                    let x = (from.0 as f32 + (here.0 as f32 - from.0 as f32) * f).round() as usize;
                    let y = (from.1 as f32 + (here.1 as f32 - from.1 as f32) * f).round() as usize;
                    self.pad[y * n + x] = u8::from(paint);
                }
                self.pad_stroke = Some((paint, here));
            }
        } else {
            self.pad_stroke = None;
        }
        let p = ui.painter_at(rect);
        p.rect_filled(rect, CornerRadius::ZERO, theme::BG_INPUT);
        for y in 0..n {
            for x in 0..n {
                if self.pad[y * n + x] != 0 {
                    let r = Rect::from_min_size(rect.min + vec2(x as f32 * cell, y as f32 * cell), Vec2::splat(cell));
                    p.rect_filled(r, CornerRadius::ZERO, theme::TEXT_PRIMARY);
                }
            }
        }
        for i in 0..=n {
            let o = i as f32 * cell;
            let col = Stroke::new(1.0, if i == n / 2 { theme::BORDER_STRONG } else { theme::BORDER_SUBTLE });
            p.line_segment([pos2(rect.min.x + o, rect.min.y), pos2(rect.min.x + o, rect.max.y)], col);
            p.line_segment([pos2(rect.min.x, rect.min.y + o), pos2(rect.max.x, rect.min.y + o)], col);
        }
        ui.horizontal(|ui| {
            if kit::chip(ui, "clear", false).clicked() {
                self.new_pad(n);
            }
            if kit::chip(ui, "fill all", false).clicked() {
                self.pad.fill(1);
            }
            if kit::chip(ui, "invert", false).clicked() {
                for v in &mut self.pad {
                    *v = 1 - *v;
                }
            }
            ui.label(kit::mono_txt(format!("{} px · right button or Shift erases", self.pad.iter().filter(|&&v| v != 0).count()), 12.0).small().color(theme::TEXT_MUTED));
        });
    }
}
