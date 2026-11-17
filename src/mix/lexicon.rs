use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::files::half_to_f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConceptKind {
    Object,
    Attribute,
}

impl ConceptKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "object" => Some(ConceptKind::Object),
            "attribute" => Some(ConceptKind::Attribute),
            _ => None,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            ConceptKind::Object => "object",
            ConceptKind::Attribute => "attribute",
        }
    }
    pub fn flipped(self) -> Self {
        match self {
            ConceptKind::Object => ConceptKind::Attribute,
            ConceptKind::Attribute => ConceptKind::Object,
        }
    }
}

/// The variants, in the order of the demo's chips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VariantKind {
    Inside,
    Over,
    Centre,
    Blend,
    Caption,
}

impl VariantKind {
    pub const ALL: [VariantKind; 5] = [VariantKind::Inside, VariantKind::Over, VariantKind::Centre, VariantKind::Blend, VariantKind::Caption];

    pub fn title(self) -> &'static str {
        match self {
            VariantKind::Inside => "inside",
            VariantKind::Over => "on top",
            VariantKind::Centre => "centre",
            VariantKind::Blend => "blend",
            VariantKind::Caption => "one caption",
        }
    }
    pub fn hint(self) -> &'static str {
        match self {
            VariantKind::Inside => "the first object in a middle band, the second above and below",
            VariantKind::Over => "the first object on top, the second below",
            VariantKind::Centre => "the first object in the centre, the second around it",
            VariantKind::Blend => "both objects everywhere, equal weights",
            VariantKind::Caption => "the whole caption as one vector: the plain way",
        }
    }
    /// A and B play different parts (2 orders for two objects).
    pub fn ordered(self) -> bool {
        matches!(self, VariantKind::Inside | VariantKind::Over | VariantKind::Centre)
    }
    /// Short name (settings, demo keys).
    pub fn key(self) -> &'static str {
        match self {
            VariantKind::Inside => "inside",
            VariantKind::Over => "over",
            VariantKind::Centre => "centre",
            VariantKind::Blend => "blend",
            VariantKind::Caption => "caption",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        VariantKind::ALL.into_iter().find(|k| k.key() == s.trim())
    }
}

/// `lexicon.json` + `lexicon.f16` of the demo (release/tools/build_lexicon.py): stop words, known multi-word
/// subjects, the kind of 3600 words and phrases, and their encoder vectors (float16).
pub struct Lexicon {
    pub stop: HashSet<String>,
    pub phrases: HashSet<String>,
    pub kind: HashMap<String, ConceptKind>,
    pub index: HashMap<String, usize>,
    pub entries: Vec<(String, ConceptKind)>,
    pub dim: usize,
    pub(super) vectors: Vec<f32>,
}

impl Lexicon {
    pub fn load(json: &Path, f16: &Path) -> Result<Self> {
        #[derive(serde::Deserialize)]
        struct Lx {
            dim: usize,
            stop: Vec<String>,
            phrases: Vec<String>,
            entries: Vec<(String, String)>,
        }
        let lx: Lx = serde_json::from_str(&std::fs::read_to_string(json).with_context(|| format!("reading {}", json.display()))?)
            .with_context(|| format!("parsing {}", json.display()))?;
        let raw = std::fs::read(f16).with_context(|| format!("reading {}", f16.display()))?;
        if raw.len() != lx.entries.len() * lx.dim * 2 {
            bail!("{}: {} bytes, expected {} entries x {} float16", f16.display(), raw.len(), lx.entries.len(), lx.dim);
        }
        let vectors = raw.chunks_exact(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        let mut entries = Vec::with_capacity(lx.entries.len());
        let mut kind = HashMap::new();
        let mut index = HashMap::new();
        for (i, (t, k)) in lx.entries.into_iter().enumerate() {
            let k = ConceptKind::parse(&k).unwrap_or(ConceptKind::Object);
            kind.insert(t.clone(), k); // later duplicates win, as `new Map(entries)`
            index.insert(t.clone(), i);
            entries.push((t, k));
        }
        Ok(Lexicon { stop: lx.stop.into_iter().collect(), phrases: lx.phrases.into_iter().collect(), kind, index, entries, dim: lx.dim, vectors })
    }

    /// float32 vector of entry `i` (expanded from float16 exactly as `halfRows`).
    pub fn vector(&self, i: usize) -> &[f32] {
        &self.vectors[i * self.dim..(i + 1) * self.dim]
    }

    /// Vector of a word or phrase when the lexicon has it.
    pub fn vector_of(&self, text: &str) -> Option<&[f32]> {
        self.index.get(text).map(|&i| self.vector(i))
    }
}

/// One concept of a caption.
#[derive(Debug, Clone, PartialEq)]
pub struct Concept {
    pub text: String,
    pub kind: ConceptKind,
    /// In the lexicon, or a Russian / Ukrainian adjective by its ending.
    pub known: bool,
    pub p: f64,
    pub rank: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parsed {
    pub concepts: Vec<Concept>,
    pub cyr: bool,
    /// Words beyond [`super::MAX`] that were left out.
    pub dropped: usize,
}
