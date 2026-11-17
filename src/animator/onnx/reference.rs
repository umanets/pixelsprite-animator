use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::animator::{Kind, standard_actions};
use crate::sprite::{Frame, Palette, Sprite};

// ----------------------------------------------------------------------------------------------- the animator
/// A sprite as the model sees it: v2 palette tokens, the colours it may use, and the way back to its own palette.
#[derive(Clone)]
pub struct Reference {
    pub size: usize,
    /// S*S tokens, EMPTY = 1024.
    pub tokens: Vec<u16>,
    /// Tokens of the reference's colours + EMPTY, ascending.
    pub allowed: Vec<usize>,
    /// A hash of (size, tokens): the cache key of everything drawn from it.
    pub hash: u64,
    /// Colours not in v2's palette that were mapped to the nearest one.
    pub approximated: usize,
    pub(super) palette: Palette,
    /// token -> index into `palette`; None = the sprite already uses v2's palette.
    pub(super) back: Option<HashMap<u16, i16>>,
    pub(super) empty: u16,
}

impl Reference {
    /// Model tokens -> a frame in the sprite's own palette.
    pub fn to_frame(&self, tokens: &[u16]) -> Frame {
        let px = tokens
            .iter()
            .map(|&t| {
                if t == self.empty {
                    -1
                } else {
                    match &self.back {
                        None => t as i16,
                        Some(b) => b.get(&t).copied().unwrap_or(-1),
                    }
                }
            })
            .collect();
        Sprite::new(self.size, px, self.palette.clone())
    }
}

impl Reference {
    /// The same sprite family (palette, way back, size) with other tokens, e.g. an anchor frame. `keep_allowed`:
    /// sampling stays restricted to this reference's colours; otherwise to the new tokens' colours.
    pub fn with_tokens(&self, tokens: Vec<u16>, keep_allowed: bool) -> Reference {
        let mut r = self.clone();
        if !keep_allowed {
            let mut allowed: Vec<usize> = tokens.iter().map(|&x| x as usize).collect();
            allowed.push(self.empty as usize);
            allowed.sort_unstable();
            allowed.dedup();
            r.allowed = allowed;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.size.hash(&mut h);
        tokens.hash(&mut h);
        keep_allowed.hash(&mut h);
        r.hash = h.finish();
        r.tokens = tokens;
        r
    }
}

/// Which kind of reference the model is told it gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefChoice {
    /// The original v2 sprite at 16 / 32 px (what the app draws), the front render at 24 (the model never saw an
    /// original 24 px sprite).
    Auto,
    Original,
    FrontRender,
}

impl RefChoice {
    pub fn label(self) -> &'static str {
        match self {
            RefChoice::Auto => "auto (original sprite; front render at 24)",
            RefChoice::Original => "original sprite",
            RefChoice::FrontRender => "3D-farm front render",
        }
    }
    pub fn refk(self, size: usize) -> usize {
        match self {
            RefChoice::Auto => (size == 24) as usize,
            RefChoice::Original => 0,
            RefChoice::FrontRender => 1,
        }
    }
}

/// Which reference-type token the model gets for the anchor frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorRef {
    Original,
    FrontRender,
    /// The one the sprite itself was drawn with.
    Same,
}

/// Which colours the sampler may use for clips drawn from the anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorPalette {
    /// The input sprite's colours.
    Input,
    /// Only the colours the anchor frame itself uses.
    Anchor,
}

/// The experimental "build everything from the first frame of walk (front)" option: the sprite draws `walk` (`fly`
/// for birds) at 0 deg as now; its first frame, the anchor, is the reference of every other view and clip (and of
/// the fight scene). The clip of the anchor's own action and direction is the first clip itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorOpts {
    pub enabled: bool,
    pub refk: AnchorRef,
    pub palette: AnchorPalette,
}

impl AnchorOpts {
    pub const OFF: AnchorOpts = AnchorOpts { enabled: false, refk: ANCHOR_REF, palette: ANCHOR_PALETTE };
    pub const ON: AnchorOpts = AnchorOpts { enabled: true, refk: ANCHOR_REF, palette: ANCHOR_PALETTE };
}

/// Defaults of the option, chosen with `selftest --anchor-eval` (see docs/DEVELOPMENT.md).
pub const ANCHOR_REF: AnchorRef = AnchorRef::FrontRender;
pub const ANCHOR_PALETTE: AnchorPalette = AnchorPalette::Input;

/// The action whose 0 deg clip is drawn first: `fly` for birds, `walk` otherwise (falls back to the first action the
/// model has for the kind).
pub fn anchor_action(kind: Kind, has: impl Fn(&str) -> bool) -> String {
    let want = if kind == Kind::Bird { "fly" } else { "walk" };
    if has(want) {
        return want.into();
    }
    standard_actions(kind).into_iter().map(|a| a.name).find(|a| has(a)).unwrap_or_else(|| want.into())
}

/// The clips that still need drawing from the anchor (all but the anchor's own action at 0 deg) and their indices
/// in `clips`.
pub fn split_anchor_clips(action: &str, clips: &[(String, u16)]) -> (Vec<(String, u16)>, Vec<usize>) {
    let idx: Vec<usize> = (0..clips.len()).filter(|&i| !(clips[i].0 == action && clips[i].1 == 0)).collect();
    (idx.iter().map(|&i| clips[i].clone()).collect(), idx)
}

pub(super) struct Anchor {
    /// The anchor as a reference (its first frame's tokens).
    pub(super) r: Reference,
    /// Reference-type token for clips drawn from it.
    pub(super) refk: usize,
    pub(super) action: String,
    /// The first clip (all frames, tokens).
    pub(super) first: Arc<Vec<Vec<u16>>>,
}
