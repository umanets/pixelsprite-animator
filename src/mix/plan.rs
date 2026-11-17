use crate::sprite::Mulberry32;
use super::*;


/// How a batch mixes the concepts. Concepts are referred to by their index in `concepts` (caption order).
#[derive(Debug, Clone)]
pub struct Plan {
    pub concepts: Vec<Concept>,
    pub objects: Vec<usize>,
    pub attributes: Vec<usize>,
    pub kinds: Vec<VariantKind>,
    /// Number of distinct variants (for the note under the words).
    pub count: usize,
    pmax: f64,
}

impl Plan {
    pub fn attr_strength(&self, c: usize) -> f64 {
        ATTR * (0.5 + 0.5 * self.concepts[c].p / self.pmax)
    }
}

/// None = draw the plain caption (fewer than two words, or only "one caption" fits / is enabled).
pub fn plan(concepts: &[Concept], enabled: &[VariantKind]) -> Option<Plan> {
    if concepts.len() < 2 {
        return None;
    }
    let objects: Vec<usize> = (0..concepts.len()).filter(|&i| concepts[i].kind == ConceptKind::Object).collect();
    let attributes: Vec<usize> = (0..concepts.len()).filter(|&i| concepts[i].kind == ConceptKind::Attribute).collect();
    let kinds: Vec<VariantKind> = VariantKind::ALL.into_iter().filter(|k| enabled.contains(k) && unfit(*k, concepts).is_none()).collect();
    if kinds.is_empty() || (kinds.len() == 1 && kinds[0] == VariantKind::Caption) {
        return None;
    }
    let pmax = concepts.iter().map(|c| c.p).fold(f64::NEG_INFINITY, f64::max);
    let n = objects.len();
    let count = kinds.iter().map(|k| if k.ordered() { n * (n - 1) } else { 1 }).sum();
    Some(Plan { concepts: concepts.to_vec(), objects, attributes, kinds, count, pmax })
}

/// One variant: a / b = concept indices (b = None without a second object).
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub kind: VariantKind,
    pub a: Option<usize>,
    pub b: Option<usize>,
    pub others: Vec<usize>,
    pub attributes: Vec<usize>,
}

fn choose(items: &[usize], pl: &Plan, rng: &mut Mulberry32) -> usize {
    let total = items.iter().fold(0.0, |s, &i| s + pl.concepts[i].p);
    let mut u = rng.next_f64() * total;
    for &i in items {
        u -= pl.concepts[i].p;
        if u <= 0.0 {
            return i;
        }
    }
    *items.last().expect("choose from nothing")
}

/// `drawVariant`: one rng number for the kind, two more for an ordered pair.
pub fn draw_variant(pl: &Plan, rng: &mut Mulberry32) -> Variant {
    let kind = pl.kinds[(rng.next_f64() * pl.kinds.len() as f64).floor() as usize];
    if pl.objects.len() < 2 {
        return Variant { kind, a: pl.objects.first().copied(), b: None, others: pl.objects.iter().skip(1).copied().collect(), attributes: pl.attributes.clone() };
    }
    let (a, b) = if kind.ordered() {
        let a = choose(&pl.objects, pl, rng);
        let rest: Vec<usize> = pl.objects.iter().copied().filter(|&o| o != a).collect();
        (a, choose(&rest, pl, rng))
    } else {
        let mut by_p = pl.objects.clone();
        by_p.sort_by(|&x, &y| pl.concepts[y].p.partial_cmp(&pl.concepts[x].p).unwrap_or(std::cmp::Ordering::Equal));
        (by_p[0], by_p[1])
    };
    Variant { kind, a: Some(a), b: Some(b), others: pl.objects.iter().copied().filter(|&o| o != a && o != b).collect(), attributes: pl.attributes.clone() }
}

/// The variants of a batch of `n` sprites for `seed` (the demo's order of random numbers).
pub fn draw_variants(pl: &Plan, n: usize, seed: u32) -> Vec<Variant> {
    let mut rng = Mulberry32::new(seed ^ VARIANT_SEED_XOR);
    (0..n).map(|_| draw_variant(pl, &mut rng)).collect()
}

/// Text under a sprite: "skull / burger + fried" (A / B), "skull + burger" for a blend, the caption in quotes.
pub fn label(v: &Variant, pl: &Plan, caption: &str) -> String {
    if v.kind == VariantKind::Caption {
        return format!("“{caption}”");
    }
    let t = |i: usize| pl.concepts[i].text.as_str();
    let head = match (v.a, v.b) {
        (Some(a), Some(b)) => format!("{} {} {}", t(a), if v.kind.ordered() { "/" } else { "+" }, t(b)),
        (Some(a), None) => t(a).to_string(),
        _ => String::new(),
    };
    std::iter::once(head).chain(v.others.iter().map(|&i| t(i).to_string())).chain(v.attributes.iter().map(|&i| t(i).to_string())).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" + ")
}

/// A sentence for the tooltip of a sprite.
pub fn describe(v: &Variant, pl: &Plan) -> String {
    let t = |i: Option<usize>| i.map(|i| pl.concepts[i].text.clone()).unwrap_or_default();
    match (v.kind, v.b) {
        (VariantKind::Caption, _) => "the whole caption as one vector (the plain way)".into(),
        (_, None) => "every word as its own caption over the whole sprite".into(),
        (VariantKind::Inside, Some(_)) => format!("{} in a middle band, {} above and below", t(v.a), t(v.b)),
        (VariantKind::Over, Some(_)) => format!("{} on top, {} below", t(v.a), t(v.b)),
        (VariantKind::Centre, Some(_)) => format!("{} in the centre, {} around", t(v.a), t(v.b)),
        (VariantKind::Blend, Some(_)) => format!("{} and {} everywhere, equal weights", t(v.a), t(v.b)),
    }
}
