use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, channel};

use anyhow::Result;

use crate::classify::{Classification, MappedWord};
use crate::config::Config;
use crate::ep::{DllReport, EpChoice};
use crate::mix::Lexicon;
use crate::model::GenParams;
use crate::sprite::Palette;
use super::state::State;
use super::{GenRequest, GenResult, Msg};


pub struct Headless {
    st: State,
    _rx: Receiver<Msg>,
}

impl Headless {
    pub fn new(cfg: Config, dll: DllReport) -> Self {
        let (tx, rx) = channel();
        let st = State::new(cfg, dll, tx, Box::new(|| {}), Arc::new(AtomicBool::new(false)));
        Headless { st, _rx: rx }
    }
    pub fn load(&mut self, choice: EpChoice) -> Result<(String, Vec<String>)> {
        self.st.load(choice)?;
        let m = self.st.model.as_ref().unwrap();
        // drain messages to collect the fallbacks
        let mut fb = Vec::new();
        while let Ok(msg) = self._rx.try_recv() {
            if let Msg::Loaded { fallbacks, .. } = msg {
                fb = fallbacks;
            }
        }
        Ok((m.describe(), fb))
    }
    /// A "new shapes" request with the demo's mixing defaults (a caption of two or three words is mixed).
    pub fn generate(&mut self, caption: &str, params: GenParams) -> Result<GenResult> {
        self.request(GenRequest::new(0, caption, params))
    }
    /// Any request (mode, mixing options), as the UI sends it.
    pub fn request(&mut self, req: GenRequest) -> Result<GenResult> {
        let r = self.st.generate(req);
        while self._rx.try_recv().is_ok() {}
        r
    }
    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>> {
        self.st.enc.as_mut().ok_or_else(|| anyhow::anyhow!("no encoder"))?.encode(text)
    }
    pub fn token_ids(&self, text: &str) -> Result<Vec<u32>> {
        self.st.enc.as_ref().ok_or_else(|| anyhow::anyhow!("no encoder"))?.ids(text)
    }
    pub fn map_word(&mut self, w: &str) -> Option<MappedWord> {
        let r = self.st.map_word(w);
        while self._rx.try_recv().is_ok() {}
        r
    }
    pub fn classify(&mut self, caption: &str) -> Classification {
        let r = self.st.classify(caption);
        while self._rx.try_recv().is_ok() {}
        r
    }
    pub fn palette(&self) -> Option<Palette> {
        self.st.model.as_ref().map(|m| m.core.palette.clone())
    }
    pub fn lexicon(&self) -> Option<Arc<Lexicon>> {
        self.st.lexicon.clone()
    }
    pub fn presets(&self) -> Vec<String> {
        self.st.preset_order.clone()
    }
    /// Sprites per chunk the model uses by default for this provider and size (`SpriteModel::default_batch`).
    pub fn default_batch(&self, size: usize) -> usize {
        self.st.model.as_ref().map(|m| m.default_batch(size)).unwrap_or(8)
    }
}
