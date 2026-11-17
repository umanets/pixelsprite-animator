//! Mixing concepts without sliders, a port of the web demo's `mix.js` (docs/demo, v=12): the words of the caption
//! become concepts (object or attribute, from the lexicon), and every sprite of a batch is drawn with its own
//! variant of how they combine. The sampler side is `model::Strategy::mix` (AND / product-of-experts guidance with
//! a weight map per caption).
//!
//! Rules (mix.js, decided with the author on 15.11.2026):
//! * concepts = words of the caption without stop words, known multi-word subjects kept together ("treasure
//!   chest"), at most three; each is an object or an attribute (lexicon; a click on the word chip flips it);
//! * probability by position 1/2, 1/4, 1/8 ... normalised: in Latin script the LAST word is the strongest, in
//!   Cyrillic the first;
//! * mixing only with two or more words and a variant that fits; otherwise the plain caption;
//! * each sprite draws a variant uniformly from the enabled ones: three layouts with an order (main object A ~ p,
//!   second object B among the rest ~ p): inside / over / centre; plus "blend" (both captions everywhere, equal
//!   weights) and "caption" (the whole caption as one vector). Zone weights 1.6 / 0.4; a third object and every
//!   attribute act on the whole sprite; attributes get strength ATTR * (0.5 + 0.5 p / max p).
//!
//! Every number is computed as in JavaScript (f64 arithmetic, weight maps stored as f32 like the demo's
//! Float32Arrays, V8's `Math.hypot`), so a seed draws the demo's variants and the demo's weights.

mod lexicon;
mod parse;
mod plan;
#[cfg(test)]
mod tests;
mod weights;

pub use lexicon::{Concept, ConceptKind, Lexicon, Parsed, VariantKind};
pub use parse::{is_cyrillic, looks_like_adjective, parse, unfit, words};
pub use plan::{Plan, Variant, describe, draw_variant, draw_variants, label, plan};
pub use weights::{js_hypot, mean_caption, nearest_kind, variant_maps};

pub const HI: f64 = 1.6;
pub const LO: f64 = 0.4;
pub const BLEND: f64 = 1.0;
pub const ATTR: f64 = 1.0;
pub const THIRD: f64 = 0.6;
/// At most this many concepts (words) are used.
pub const MAX: usize = 3;
/// Seed of the variant draws: `makeRng((seed ^ 0x5bd1e995) >>> 0)` in app.js.
pub const VARIANT_SEED_XOR: u32 = 0x5bd1_e995;
