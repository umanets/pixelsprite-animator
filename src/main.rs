//! Pixel Sprite Studio: generate sprites from a caption (pixelsprite canvas v2 on CUDA / DirectML / CPU) or load
//! a PNG, pick one, choose how it should move and watch the loop. Animators: the animation model (AnimGPT in ONNX
//! Runtime, drawn on a background thread), the 3D-farm library and a procedural mock (see animator/).
//!
//! The layout follows the Figma mock ("Studio / Main"): a 52 px top bar, a 320 px Generate + Library panel on the
//! left, the gallery + preview in the centre, a 340 px Animate panel on the right, a 200 px Fight scene dock and a
//! 28 px status bar. Colours / radii / spacing / fonts come from [`theme`]; every control is a [`ui_kit`] component.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use animator_ui::cancel::Failure;
use animator_ui::animator::jobs::{AnimJobs, Job, JobEvent, JobResult, Ticket};
use animator_ui::animator::library::{LibraryAnimator, LibraryData, LibraryEntry, MissingLibrary, ScanMsg, Scanner};
use animator_ui::animator::mock::MockAnimator;
use animator_ui::animator::onnx::{AnimProgress, AnchorOpts, OnnxAnimator, RefChoice};
use animator_ui::animator::scene::{self, Scene};
use animator_ui::animator::{AZIMUTHS, ActionSpec, AnimRequest, Animation, Animator, Kind};
use animator_ui::classify::{Classification, classify_english};
use animator_ui::config::{ANIM_CUDA_MEM_LIMIT_MB, Config, exports_dir, slug};
use animator_ui::ep::{DllReport, Ep, EpChoice, SessionOpts, setup_dll_paths};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use animator_ui::export;
use animator_ui::mix::{self, ConceptKind, Lexicon, VariantKind};
use animator_ui::model::{GenParams, demo_rows_cpu, demo_rows_gpu};
use animator_ui::save;
use animator_ui::sprite::{Palette, Sprite};
use animator_ui::theme;
use animator_ui::ui_kit as kit;
use animator_ui::worker::{Cmd, GenRequest, MixOptions, Mode, Msg, VariantInfo, Worker};
use eframe::egui::{
    self, Color32, ColorImage, CornerRadius, Margin, Rect, RichText, Sense, Stroke, StrokeKind, TextureHandle,
    TextureOptions, Vec2, pos2, vec2,
};

mod helpers;
mod state;
mod ui;

use helpers::*;
use state::*;

fn main() -> eframe::Result {
    let cfg = Config::load();
    let dll = setup_dll_paths(&cfg); // ep::environment() errors if this has not run
    // ANIMATOR_UI_WIN="1440x1060" (the screenshot runs) overrides the default window size
    let size = std::env::var("ANIMATOR_UI_WIN")
        .ok()
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some([a.parse().ok()?, b.parse().ok()?])
        })
        .unwrap_or([1440.0, 1060.0]);
    let [w, h] = size;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Pixel Sprite Studio")
            .with_inner_size([w, h])
            .with_min_inner_size([1100.0, 760.0]),
        ..Default::default()
    };
    eframe::run_native("animator_ui", options, Box::new(move |cc| Ok(Box::new(App::new(cc, cfg, dll)))))
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, cfg: Config, dll: DllReport) -> Self {
        theme::apply(&cc.egui_ctx);
        let ctx = cc.egui_ctx.clone();
        let worker = Worker::spawn(cfg.clone(), dll.clone(), Box::new(move || ctx.request_repaint()));
        let demo = Demo::from_env();
        let ep_name = demo.as_ref().and_then(|d| d.get("ep").map(str::to_string)).or_else(|| cfg.ep.clone());
        let ep_choice = EpChoice::parse(ep_name.as_deref().unwrap_or("auto"));
        worker.send(Cmd::Load(ep_choice));
        let ctx = cc.egui_ctx.clone();
        let lib_scan = Some(Scanner::start(cfg.library_roots(), Box::new(move || ctx.request_repaint())));
        let ctx = cc.egui_ctx.clone();
        let jobs = AnimJobs::spawn(Arc::new(OnnxAnimator::new()), Box::new(move || ctx.request_repaint()));
        let ref_from_walk_front = cfg.ref_from_walk_front.unwrap_or(false);
        let mut app = App {
            cfg,
            dll,
            worker,
            ep_choice,
            ep_active: None,
            ep_describe: String::new(),
            ep_fallbacks: Vec::new(),
            loading: true,
            palette: None,
            status: "Preparing the model…".into(),
            load_error: None,
            cudnn_busy: false,
            cudnn_done: None,
            ms_per_step: None,
            caption: "knight".into(),
            size: 16,
            seed: 1,
            auto_seed: false,
            count: 8,
            count_touched: false,
            g_shape: 1.0,
            g_colour: 2.0,
            temperature: 1.0,
            top_p: 1.0,
            filter: true,
            batching: Batching::DemoGpu,
            mode: Mode::New,
            pad: vec![0; 16 * 16],
            pad_size: 16,
            pad_stroke: None,
            lexicon: None,
            presets: Vec::new(),
            variants_on: VariantKind::ALL.to_vec(),
            kind_override: HashMap::new(),
            learned_kind: HashMap::new(),
            encoder_ready: false,
            more_open: None,
            save_scale: 1,
            last_save: None,
            next_id: 1,
            running: None,
            progress: (0, 0, 0, 0),
            runs: Vec::new(),
            selected: None,
            lib_entries: Vec::new(),
            lib_scan,
            lib_next: None,
            lib_scan_note: "scanning…".into(),
            lib_filter: String::new(),
            lib_kind: None,
            lib_view: (String::new(), usize::MAX, Vec::new()),
            lib_data: None,
            lib_error: None,
            lib_tex: None,
            thumbs: Some(Thumbs::start()),
            thumb_cache: HashMap::new(),
            anim: AnimState {
                animator: AnimatorChoice::Mock,
                kind: KindChoice::Auto,
                action: "idle".into(),
                azimuth: 90,
                n_frames: 4,
                fps: 8.0,
                looping: true,
                playing: true,
                shown: 0,
                t0: 0.0,
                key: None,
                result: None,
                is_mock: false,
                textures: Vec::new(),
                error: None,
                lib_size: 32,
                ref_from_walk_front,
                pending: None,
                progress: None,
                stopped: false,
            },
            scene: SceneState {
                key: None,
                scene: None,
                note: None,
                show: false,
                is_mock: false,
                textures: Vec::new(),
                playing: true,
                t0: 0.0,
                shown: 0,
                qc: true,
                walk_speed: scene::WALK_SPEED,
                run_speed: scene::RUN_SPEED,
                laid_out: (scene::WALK_SPEED, scene::RUN_SPEED),
                last_export: None,
                pending: None,
                progress: None,
                stopped: false,
            },
            export_scale: 4,
            last_export: None,
            mock: MockAnimator,
            jobs,
            dirs_pending: None,
            loaded: None,
            loaded_id: 0,
            png_path: String::new(),
            png_error: None,
            settings_open: false,
            demo,
        };
        app.apply_anim_ref();
        app
    }

    /// Pushes the "build everything from the first walk frame" checkbox into the animation model's settings
    /// (the anchor). Toggling it changes [`Self::onnx_tag`], so the preview and the scene are drawn again.
    fn apply_anim_ref(&mut self) {
        let mut st = self.jobs.anim.settings();
        st.anchor = if self.anim.ref_from_walk_front { AnchorOpts::ON } else { AnchorOpts::OFF };
        self.jobs.anim.set_settings(st);
    }

    fn rescan_library(&mut self, ctx: &egui::Context) {
        let ctx = ctx.clone();
        self.lib_scan = Some(Scanner::start(self.cfg.library_roots(), Box::new(move || ctx.request_repaint())));
        self.lib_next = (!self.lib_entries.is_empty()).then(Vec::new);
        self.lib_scan_note = "rescanning…".into();
    }

    fn poll_library_scan(&mut self, ctx: &egui::Context) {
        let mut done = None;
        if let Some(sc) = &self.lib_scan {
            while let Ok(msg) = sc.rx.try_recv() {
                match msg {
                    ScanMsg::Batch(b) => match &mut self.lib_next {
                        Some(next) => next.extend(b),
                        None => self.lib_entries.extend(b),
                    },
                    ScanMsg::Done { entries, secs } => done = Some((entries, secs)),
                }
            }
        }
        if let Some((n, secs)) = done {
            if let Some(next) = self.lib_next.take() {
                self.lib_entries = next;
            }
            self.lib_scan = None;
            self.lib_scan_note = format!("{n} finished entries, listed in {secs:.1} s");
            self.lib_view.1 = usize::MAX;
        } else if self.lib_scan.is_some() && self.lib_next.is_none() {
            self.lib_scan_note = format!("{} entries so far, scanning…", self.lib_entries.len());
        }
        // thumbnails that arrived
        if let Some(th) = &self.thumbs {
            while let Ok((name, sprite)) = th.rx.try_recv() {
                let tex = sprite.map(|s| texture(ctx, &format!("thumb{name}"), &s));
                self.thumb_cache.insert(name, tex);
            }
            if self.thumb_cache.len() > 600 {
                self.thumb_cache.clear();
            }
        }
    }

    /// Asks the thumbnail thread for the visible entries that are not cached yet.
    fn request_thumbs(&mut self, names: &[(String, PathBuf)]) {
        if let Some(th) = &self.thumbs {
            for (name, dir) in names {
                if !self.thumb_cache.contains_key(name) {
                    let _ = th.tx.send((name.clone(), dir.clone()));
                }
            }
        }
    }

}

// -------------------------------------------------------------------------------------------------- the window
impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        if let Some(path) = ctx.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf()).filter(|p| !p.as_os_str().is_empty())) {
            self.png_path = path.display().to_string();
            self.load_png();
        }
        self.drive_demo(&ctx);
        // Esc stops a running generation (the Stop button's kbd hint)
        if self.running.is_some() && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.worker.stop();
        }
        let bg = |fill: Color32, l: i8, r: i8, t: i8, b: i8| egui::Frame::NONE.fill(fill).inner_margin(Margin { left: l, right: r, top: t, bottom: b });

        // ---- the 52 px top bar
        let top_r = egui::Panel::top("top").exact_size(theme::TOP_BAR_H).frame(bg(theme::BG_PANEL, 0, 0, 0, 0)).show(ui, |ui| self.top_bar(ui)).response.rect;
        self.settings_window(&ctx);
        // ---- the 28 px status bar
        let status_r = egui::Panel::bottom("status").exact_size(theme::STATUS_H).frame(bg(theme::BG_PANEL, 0, 0, 0, 0)).show(ui, |ui| self.status_bar(ui)).response.rect;
        // ---- the 320 px Generate + Library column, the full height of the body
        let left_r = egui::Panel::left("generate")
            .exact_size(theme::LEFT_W)
            .frame(bg(theme::BG_PANEL, 16, 14, 16, 14))
            .show(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("leftscroll").auto_shrink([false, false]).show(ui, |ui| {
                    self.left_panel(ui);
                    ui.add_space(8.0);
                });
            })
            .response
            .rect;
        // ---- the 200 px Fight scene dock: it spans the workspace (centre + inspector), not the left panel
        self.refresh_scene(&ctx);
        let mut dock_top: Option<f32> = None;
        if self.scene.show {
            let dim = self.running.is_some();
            let r = egui::Panel::bottom("scene")
                .exact_size(theme::DOCK_H)
                .frame(bg(theme::BG_PANEL, 20, 20, 14, 14))
                .show(ui, |ui| {
                    if dim {
                        kit::dimmed(ui, |ui| self.scene_panel(ui));
                    } else {
                        self.scene_panel(ui);
                    }
                })
                .response
                .rect;
            dock_top = Some(r.top());
        }
        // ---- the 340 px Animate inspector
        let right_r = egui::Panel::right("animate")
            .exact_size(theme::RIGHT_W)
            .frame(bg(theme::BG_PANEL, 17, 16, 16, 14))
            .show(ui, |ui| {
                let body = |ui: &mut egui::Ui, app: &mut App| {
                    egui::ScrollArea::vertical().id_salt("animscroll").auto_shrink([false, false]).show(ui, |ui| {
                        app.right_panel(ui);
                        ui.add_space(10.0);
                    });
                };
                if self.running.is_some() {
                    kit::dimmed(ui, |ui| body(ui, self));
                } else {
                    body(ui, self);
                }
            })
            .response
            .rect;
        // ---- the centre: gallery + preview stage
        egui::CentralPanel::default().frame(bg(theme::BG_APP, 20, 20, 20, 20)).show(ui, |ui| self.center_panel(ui));
        // ---- the panels' 1 px separators, on the panels' own edges (the mock: x 319 / 1100, y 51 / 832 / 1032)
        {
            let p = ui.painter();
            let line = Stroke::new(1.0, theme::BORDER_SUBTLE);
            let w = ui.max_rect().width();
            let y = top_r.bottom() - 0.5;
            p.line_segment([pos2(0.0, y), pos2(w, y)], line);
            let x = left_r.right() - 0.5;
            p.line_segment([pos2(x, left_r.top()), pos2(x, left_r.bottom())], line);
            let x = right_r.left() + 0.5;
            p.line_segment([pos2(x, right_r.top()), pos2(x, right_r.bottom())], line);
            if let Some(y0) = dock_top {
                let y = y0 - 0.5;
                p.line_segment([pos2(left_r.right(), y), pos2(w, y)], line);
            }
            let y = status_r.top() - 0.5;
            p.line_segment([pos2(0.0, y), pos2(w, y)], line);
        }
        if self.running.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

