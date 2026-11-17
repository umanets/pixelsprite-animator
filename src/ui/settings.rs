//! The Settings window and the animation model's advanced settings.

use crate::*;

impl App {
    /// The Settings window: the execution provider's state, cuDNN, and the animation model's advanced settings.
    pub(crate) fn settings_window(&mut self, ctx: &egui::Context) {
        if !self.settings_open {
            return;
        }
        let mut open = self.settings_open;
        egui::Window::new("Settings")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(400.0)
            .show(ctx, |ui| {
                ui.label(kit::semi_txt("Execution provider", 13.0));
                match (self.loading, self.ep_active) {
                    (true, _) => {
                        ui.spinner();
                    }
                    (false, Some(ep)) => {
                        let col = match ep {
                            Ep::Cuda => theme::SUCCESS,
                            Ep::DirectMl => theme::WARN,
                            Ep::Cpu => theme::TEXT_MUTED,
                        };
                        ui.horizontal(|ui| {
                            kit::dot(ui, col);
                            ui.label(RichText::new(format!("active: {}", self.ep_describe)).color(col));
                        });
                    }
                    (false, None) => {
                        ui.label(RichText::new("no model").color(theme::DANGER));
                    }
                }
                if !self.ep_fallbacks.is_empty() {
                    ui.label(RichText::new(format!("(!) {} provider(s) skipped: hover in the status bar for why", self.ep_fallbacks.len())).color(theme::WARN).small());
                }
                if let Some(e) = &self.load_error {
                    ui.label(RichText::new(e).color(theme::DANGER).small());
                }
                let cuda_mb = animator_ui::ep::missing_cuda_mb(&self.dll);
                if cuda_mb > 0 && animator_ui::ep::has_nvidia_driver() && self.cudnn_done.is_none() {
                    ui.add_enabled_ui(!self.cudnn_busy, |ui| {
                        if kit::button(ui, &format!("Download CUDA libraries (~{cuda_mb} MB)"), kit::Btn::Ghost).on_hover_text(self.dll.summary()).clicked() {
                            self.cudnn_busy = true;
                            self.worker.send(Cmd::DownloadCudnn);
                        }
                    });
                }
                if let Some(s) = &self.cudnn_done {
                    ui.label(RichText::new(s).italics().small());
                }
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(kit::semi_txt("Animation model", 13.0));
                self.anim_model_settings(ui);
            });
        self.settings_open = open;
    }

    /// The animation model's advanced settings (temperature, seed, reference type, cache). The walk-front option
    /// itself is the checkbox card in the Animate panel.
    pub(crate) fn anim_model_settings(&mut self, ui: &mut egui::Ui) {
        let anim = self.jobs.anim.clone();
        let mut st = anim.settings();
        let mut t = st.temperature as f32;
        if ui.add(egui::Slider::new(&mut t, 0.3..=1.5).text("temperature").fixed_decimals(2)).on_hover_text("0.8 = the training's sample sheets").changed() {
            st.temperature = (t as f64 * 100.0).round() / 100.0;
        }
        ui.horizontal(|ui| {
            ui.label("seed");
            ui.add(egui::DragValue::new(&mut st.seed).range(0..=999_999));
            if kit::chip(ui, "new", false).clicked() {
                st.seed = (rand_u32() % 1_000_000).max(1);
            }
        });
        ui.horizontal(|ui| {
            ui.label("reference");
            egui::ComboBox::from_id_salt("refk").selected_text(st.reference.label()).width(250.0).show_ui(ui, |ui| {
                for c in [RefChoice::Auto, RefChoice::Original, RefChoice::FrontRender] {
                    ui.selectable_value(&mut st.reference, c, c.label());
                }
            });
        })
        .response
        .on_hover_text("What the model is told the reference is: an original v2 sprite (16 / 32 px, what the app draws), or the pixelated front render of the 3D farm (the only kind it saw at 24 px; library entries are renders).");
        ui.horizontal_wrapped(|ui| {
            ui.label(kit::mono_txt(anim.status(), 12.0).small().color(theme::TEXT_MUTED));
            if kit::chip(ui, "clear cache", false).on_hover_text("forget the clips drawn so far").clicked() {
                anim.clear_cache();
                self.anim.key = None;
                self.scene.key = None;
            }
        });
        ui.label(
            RichText::new("Animation is the model's strong part; turning to a new direction (the view drawn before every clip) is weaker, especially for animals.")
                .small()
                .color(theme::TEXT_MUTED),
        );
        if st != anim.settings() {
            anim.set_settings(st);
        }
    }
}
