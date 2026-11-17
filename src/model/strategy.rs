use super::*;
use super::sampling::{guided, guided_mix};

/// How one sprite turns the logits of its rows into its next pixel (engine.js `Strategy`). It owns `rows.len()`
/// contiguous rows of the batch (condition vectors); sprites with different strategies share a chunk. Cost: one
/// network row per entry of `rows`.
#[derive(Debug)]
pub struct Strategy {
    pub name: &'static str,
    pub rows: Vec<Arc<Vec<f32>>>,
    rule: Rule,
}

#[derive(Debug)]
enum Rule {
    Plain,
    Split { g_shape: f64, g_colour: f64 },
    /// k captions + the "no text" row; maps[g][t] = caption g's weight at canvas pixel t.
    Mix { maps: Vec<Vec<f32>> },
}

impl Strategy {
    /// Several captions at once (AND / product of experts) with a weight per pixel: `conds` = condition vectors of
    /// the captions ([`ModelCore::condition`]), `null` = the "no text" one, `maps[g][t]` for caption g on the
    /// S x S canvas.
    pub fn mix(conds: Vec<Arc<Vec<f32>>>, null: Arc<Vec<f32>>, maps: Vec<Vec<f32>>) -> Self {
        assert_eq!(conds.len(), maps.len(), "one weight map per caption");
        let mut rows = conds;
        rows.push(null);
        Strategy { name: "mix", rows, rule: Rule::Mix { maps } }
    }

    /// p(EMPTY) of pixel t; fills `c.col` with the colour probabilities. `l`: logits of the chunk (row r starts at
    /// r * (K + 1)); `base`: this sprite's first row.
    pub(super) fn pixel(&self, l: &[f32], base: usize, t: usize, c: &mut Ctx) -> f64 {
        let v = c.k + 1;
        match &self.rule {
            Rule::Plain => guided(&l[base * v..(base + 1) * v], None, c.k, 1.0, 1.0, c.temperature, c.top_p, &mut c.col, &mut c.order),
            Rule::Split { g_shape, g_colour } => {
                guided(&l[base * v..(base + 1) * v], Some(&l[(base + 1) * v..(base + 2) * v]), c.k, *g_shape, *g_colour, c.temperature, c.top_p, &mut c.col, &mut c.order)
            }
            Rule::Mix { maps } => {
                c.w.clear();
                c.w.extend(maps.iter().map(|m| m[t] as f64));
                guided_mix(l, base, &c.w, c.k, c.temperature, c.top_p, &mut c.col, &mut c.order)
            }
        }
    }
}

/// Per-chunk sampling scratch (engine.js `ctx`).
pub(super) struct Ctx {
    pub(super) k: usize,
    pub(super) temperature: f64,
    pub(super) top_p: f64,
    pub(super) col: Vec<f64>,
    pub(super) order: Vec<usize>,
    pub(super) w: Vec<f64>,
}

impl ModelCore {
    /// Plain strategy: one caption (or none), the model's own distribution.
    pub fn strategy_plain(&self, vector: Option<&[f32]>) -> Strategy {
        Strategy { name: "plain", rows: vec![Arc::new(self.condition(vector))], rule: Rule::Plain }
    }

    /// Split guidance; without a caption, or with both strengths 1, it is plain (as engine.js).
    pub fn strategy_split(&self, vector: Option<&[f32]>, g_shape: f64, g_colour: f64) -> Strategy {
        if vector.is_none() || (g_shape == 1.0 && g_colour == 1.0) {
            return self.strategy_plain(vector);
        }
        Strategy { name: "split", rows: vec![Arc::new(self.condition(vector)), Arc::new(self.condition(None))], rule: Rule::Split { g_shape, g_colour } }
    }
}
