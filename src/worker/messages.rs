use std::collections::HashMap;
use std::sync::Arc;


use crate::cancel::Failure;
use crate::classify::Classification;
use crate::ep::Ep;
use crate::ep::EpChoice;
use crate::mix::{ConceptKind, Lexicon, VariantKind};
use crate::model::{GenParams, GenStats};
use crate::sprite::Palette;

pub enum Cmd {
    Load(EpChoice),
    Generate(GenRequest),
    DownloadCudnn,
}

/// The demo's "Silhouette" choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// The model draws every sprite from scratch (captions of two or three words are mixed).
    #[default]
    New,
    /// One sprite first, then new colourings of its exact silhouette (seed + 1).
    Same,
    /// The user's silhouette (`GenParams::mask`) coloured from the caption.
    Draw,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::New => "new shapes",
            Mode::Same => "one shape",
            Mode::Draw => "your shape",
        }
    }
}

/// Caption mixing settings (the demo's word chips and variant chips).
#[derive(Debug, Clone)]
pub struct MixOptions {
    /// Variants a batch may use (the demo's chips; those that do not fit the caption are skipped).
    pub enabled: Vec<VariantKind>,
    /// Kinds set by clicks on the word chips (win over everything).
    pub overrides: HashMap<String, ConceptKind>,
    /// Kinds learned for words outside the lexicon (nearest lexicon word); the worker adds to them.
    pub learned: HashMap<String, ConceptKind>,
    /// As the demo once its text encoder has loaded: the "one caption" variant uses the exact caption vector (and
    /// the Russian / Ukrainian example captions mix). Off = a fresh demo page: the normalised sum of the word
    /// vectors when every word is in the lexicon.
    pub encoder_ready: bool,
}

impl Default for MixOptions {
    fn default() -> Self {
        MixOptions { enabled: VariantKind::ALL.to_vec(), overrides: HashMap::new(), learned: HashMap::new(), encoder_ready: false }
    }
}

#[derive(Debug, Clone)]
pub struct GenRequest {
    pub id: u64,
    pub caption: String,
    pub params: GenParams,
    pub mode: Mode,
    pub mix: MixOptions,
}

impl GenRequest {
    /// A "new shapes" request with the demo's mixing defaults.
    pub fn new(id: u64, caption: &str, params: GenParams) -> Self {
        GenRequest { id, caption: caption.into(), params, mode: Mode::New, mix: MixOptions::default() }
    }
}

/// How one sprite of a mixed batch was drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct VariantInfo {
    pub kind: VariantKind,
    /// "skull / burger" (A / B), "skull + burger" (blend), the caption in quotes.
    pub label: String,
    /// A sentence for the tooltip.
    pub describe: String,
}

pub enum Msg {
    Status(String),
    Loaded { ep: Ep, describe: String, fallbacks: Vec<String>, secs: f64, palette: Palette, lexicon: Option<Arc<Lexicon>>, presets: Vec<String> },
    LoadFailed(String),
    /// The variant of every sprite, before drawing (mixed captions only); `learned` = kinds of unknown words.
    Planned { id: u64, variants: Vec<VariantInfo>, learned: HashMap<String, ConceptKind> },
    /// `first_card`: index of the run's card of sprite 0 (1 for the main pass of "one shape", whose card 0 is the
    /// source sprite).
    Progress { id: u64, done: usize, total: usize, t: usize, n_px: usize, first_card: usize, partial: Vec<(usize, Vec<i16>)> },
    Generated(GenResult),
    GenFailed { id: u64, error: Failure },
    CudnnDone(std::result::Result<std::path::PathBuf, Failure>),
}

#[derive(Debug, Clone)]
pub struct GenResult {
    pub id: u64,
    pub caption: String,
    pub params: GenParams,
    pub mode: Mode,
    pub images: Vec<Vec<i16>>,
    /// "One shape": the sprite whose silhouette the others colour.
    pub source: Option<Vec<i16>>,
    /// Mixed captions: the variant of every sprite.
    pub variants: Option<Vec<VariantInfo>>,
    /// Concepts of a mix ("burger:object"), in caption order.
    pub concepts: Vec<String>,
    /// The "one caption" variant used the exact caption vector (else the normalised sum of the word vectors).
    pub caption_exact: Option<bool>,
    /// Kinds learned for unknown words (to keep, as the demo keeps them for the page session).
    pub learned: HashMap<String, ConceptKind>,
    /// The demo's summary line ("mix of burger, skull · 5 variants · seed 3" / "new shapes · seed 3 · shape g 1 ...").
    pub summary: String,
    pub stats: GenStats,
    pub secs: f64,
    pub encode_secs: f64,
    pub ep: Ep,
    pub class: Classification,
}
