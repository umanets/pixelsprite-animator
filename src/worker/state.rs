use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::time::Instant;

use anyhow::Result;

use crate::cancel::Failure;
use crate::config::Config;
use crate::ep::{DllReport, Ep, EpChoice, SessionOpts, download_cuda_libs};
use crate::files::{ModelFiles, prepare};
use crate::mix::{Lexicon, Plan};
use crate::model::SpriteModel;
use crate::text::TextEncoder;
use super::presets::load_presets;
use super::{Cmd, Msg};

pub(super) struct State {
    pub(super) cfg: Config,
    pub(super) dll: DllReport,
    pub(super) files: Option<ModelFiles>,
    pub(super) model: Option<SpriteModel>,
    pub(super) enc: Option<TextEncoder>,
    pub(super) vocab: Option<Vec<(&'static str, Vec<f32>, f32)>>,
    pub(super) lexicon: Option<Arc<Lexicon>>,
    /// The demo's example captions with their precomputed (Python) vectors.
    pub(super) presets: HashMap<String, Vec<f32>>,
    pub(super) preset_order: Vec<String>,
    pub(super) tx: Sender<Msg>,
    pub(super) repaint: Box<dyn Fn() + Send>,
    pub(super) cancel: Arc<AtomicBool>,
}

/// A mix ready to draw (`prepareMix` of app.js).
pub(super) struct Prepared {
    pub(super) pl: Plan,
    pub(super) vectors: Vec<Vec<f32>>,
    pub(super) caption: Option<Vec<f32>>,
    pub(super) caption_exact: bool,
}

impl State {
    pub(super) fn new(cfg: Config, dll: DllReport, tx: Sender<Msg>, repaint: Box<dyn Fn() + Send>, cancel: Arc<AtomicBool>) -> Self {
        State { cfg, dll, files: None, model: None, enc: None, vocab: None, lexicon: None, presets: HashMap::new(), preset_order: Vec::new(), tx, repaint, cancel }
    }

    /// A State with no model, no encoder and no lexicon, for tests (the receiver is returned to keep the channel open).
    #[cfg(test)]
    pub(super) fn for_test() -> (Self, std::sync::mpsc::Receiver<Msg>) {
        let (tx, rx) = std::sync::mpsc::channel();
        (State::new(Config::default(), DllReport::default(), tx, Box::new(|| {}), Arc::new(AtomicBool::new(false))), rx)
    }

    pub(super) fn send(&self, m: Msg) {
        let _ = self.tx.send(m);
        (self.repaint)();
    }

    pub(super) fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Load(choice) => {
                if let Err(e) = self.load(choice) {
                    self.send(Msg::LoadFailed(format!("{e:#}")));
                }
            }
            Cmd::Generate(req) => {
                let id = req.id;
                match self.generate(req) {
                    Ok(r) => self.send(Msg::Generated(r)),
                    Err(e) => self.send(Msg::GenFailed { id, error: Failure::from(e) }),
                }
            }
            Cmd::DownloadCudnn => {
                let tx = self.tx.clone();
                let r = download_cuda_libs(&self.dll, &mut |s| {
                    let _ = tx.send(Msg::Status(s));
                });
                self.send(Msg::CudnnDone(r.map_err(Failure::from)));
            }
        }
    }

    pub(super) fn load(&mut self, choice: EpChoice) -> Result<()> {
        let t0 = Instant::now();
        if self.files.is_none() {
            let tx = self.tx.clone();
            let files = prepare(&self.cfg, &mut |s| {
                let _ = tx.send(Msg::Status(s));
            })?;
            self.files = Some(files);
        }
        let files = self.files.clone().unwrap();
        self.model = None; // free the old session (and its GPU memory) first
        let opts = SessionOpts { cuda_mem_limit_mb: self.cfg.cuda_mem_limit_mb.unwrap_or(2048), ..Default::default() };
        let mut fallbacks = files.notes.clone();
        for ep in choice.order() {
            self.send(Msg::Status(format!("Loading the sprite model on {}…", ep.label())));
            match SpriteModel::load(&files, ep, opts) {
                Ok(m) => {
                    self.model = Some(m);
                    break;
                }
                Err(e) => {
                    let mut why = format!("{}: {e:#}", ep.label());
                    if ep == Ep::Cuda && (self.dll.cudnn.is_none() || self.dll.cudart.is_none()) {
                        why.push_str(" [CUDA 13 libraries not found: use 'Download CUDA libraries' and restart]");
                    }
                    fallbacks.push(why);
                }
            }
        }
        let Some(m) = &self.model else {
            anyhow::bail!("no execution provider worked:\n{}", fallbacks.join("\n"));
        };
        let (ep, describe, palette) = (m.ep, m.describe(), m.core.palette.clone());
        if self.enc.is_none() {
            self.send(Msg::Status("Loading the text encoder (CPU)…".into()));
            self.enc = Some(TextEncoder::load(&files)?);
        }
        if self.lexicon.is_none() && files.lexicon_json.is_file() {
            match Lexicon::load(&files.lexicon_json, &files.lexicon_f16) {
                Ok(l) => self.lexicon = Some(Arc::new(l)),
                Err(e) => fallbacks.push(format!("caption mixing is off (no lexicon): {e:#}")),
            }
        }
        if self.presets.is_empty() && files.presets.is_file() {
            match load_presets(&files.presets) {
                Ok(p) => {
                    self.preset_order = p.iter().map(|(t, _)| t.clone()).collect();
                    self.presets = p.into_iter().collect();
                }
                Err(e) => fallbacks.push(format!("no example captions: {e:#}")),
            }
        }
        self.send(Msg::Loaded { ep, describe, fallbacks, secs: t0.elapsed().as_secs_f64(), palette, lexicon: self.lexicon.clone(), presets: self.preset_order.clone() });
        Ok(())
    }

    pub(super) fn encoder(&mut self) -> Result<&mut TextEncoder> {
        self.enc.as_mut().ok_or_else(|| anyhow::anyhow!("the text encoder is not loaded"))
    }

    /// `vectorFor` of app.js: the precomputed vector of an example caption, else the text encoder; None = no text.
    pub(super) fn vector_for(&mut self, caption: &str) -> Result<Option<Vec<f32>>> {
        if caption.is_empty() {
            return Ok(None);
        }
        if let Some(v) = self.presets.get(caption) {
            return Ok(Some(v.clone()));
        }
        Ok(Some(self.encoder()?.encode(caption)?))
    }
}
