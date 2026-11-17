use super::*;


/// V8's `Math.hypot` (builtins/math.tq): scale by the largest magnitude, Kahan-summed squares.
pub fn js_hypot(values: &[f64]) -> f64 {
    let mut max = 0.0f64;
    let mut nan = false;
    for &v in values {
        if v.is_nan() {
            nan = true;
        } else if v.abs() > max {
            max = v.abs();
        }
    }
    if max == f64::INFINITY {
        return f64::INFINITY;
    }
    if nan {
        return f64::NAN;
    }
    if max == 0.0 {
        return 0.0;
    }
    let (mut sum, mut comp) = (0.0f64, 0.0f64);
    for &v in values {
        let n = v.abs() / max;
        let summand = n * n - comp;
        let prelim = sum + summand;
        comp = (prelim - sum) - summand;
        sum = prelim;
    }
    sum.sqrt() * max
}

/// Layout value m(y, x) in [0, 1] (1 = A dominates) at window coordinates in [0, 1], with a one-pixel ramp.
fn layout_value(kind: VariantKind, y: f64, x: f64, w: f64) -> f64 {
    let soft = |d: f64| (d * w + 0.5).max(0.0).min(1.0);
    match kind {
        VariantKind::Inside => soft((y - 0.35).min(0.65 - y)),
        VariantKind::Over => soft(0.5 - y),
        _ => soft(0.3 - js_hypot(&[x - 0.5, y - 0.5])),
    }
}

/// Weight maps over the S x S canvas, one per concept of the plan (caption order; zeros for concepts the variant
/// does not use): `order.map((c) => maps[concepts.indexOf(c)] || zeros)` of app.js over `variantMaps`. The layout
/// is laid over the visible w x w window at offset o (size 24 = a window on the 32 canvas). Stored as f32.
pub fn variant_maps(v: &Variant, pl: &Plan, s: usize, o: usize, w: usize) -> Vec<Vec<f32>> {
    let mut maps = vec![Vec::new(); pl.concepts.len()];
    let fill = |val: f64| vec![val as f32; s * s];
    match (v.a, v.b) {
        (Some(a), Some(b)) if v.kind == VariantKind::Blend => {
            maps[a] = fill(BLEND);
            maps[b] = fill(BLEND);
        }
        (Some(a), Some(b)) => {
            let (mut ma, mut mb) = (vec![0f32; s * s], vec![0f32; s * s]);
            for r in 0..s {
                for c in 0..s {
                    let m = layout_value(v.kind, (r as f64 - o as f64 + 0.5) / w as f64, (c as f64 - o as f64 + 0.5) / w as f64, w as f64);
                    ma[r * s + c] = (LO + (HI - LO) * m) as f32;
                    mb[r * s + c] = (HI + (LO - HI) * m) as f32;
                }
            }
            maps[a] = ma;
            maps[b] = mb;
        }
        (Some(a), None) => maps[a] = fill(1.0),
        _ => {}
    }
    for &c in &v.others {
        maps[c] = fill(THIRD);
    }
    for &c in &v.attributes {
        maps[c] = fill(pl.attr_strength(c));
    }
    for m in &mut maps {
        if m.is_empty() {
            *m = vec![0f32; s * s];
        }
    }
    maps
}

/// The "one caption" vector without the text encoder: the normalised sum of the word vectors (f32 sums,
/// `Math.hypot` norm, as app.js).
pub fn mean_caption(vectors: &[Vec<f32>]) -> Vec<f32> {
    let mut c = vec![0f32; vectors.first().map(Vec::len).unwrap_or(0)];
    for v in vectors {
        for (a, b) in c.iter_mut().zip(v) {
            *a += *b;
        }
    }
    let norm = js_hypot(&c.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let norm = if norm == 0.0 || norm.is_nan() { 1.0 } else { norm };
    c.iter().map(|&x| (x as f64 / norm) as f32).collect()
}

/// Kind of an unknown word from its nearest single-word lexicon entry (dot product of the encoder vectors;
/// phrases skipped). Returns (kind, nearest entry, cosine).
pub fn nearest_kind(vec: &[f32], lex: &Lexicon) -> (ConceptKind, String, f64) {
    let (mut best, mut bi) = (-2.0f64, 0usize);
    for (i, (t, _)) in lex.entries.iter().enumerate() {
        if t.contains(' ') {
            continue;
        }
        let d = lex.vector(i).iter().zip(vec).fold(0.0f64, |d, (a, b)| d + *a as f64 * *b as f64);
        if d > best {
            best = d;
            bi = i;
        }
    }
    (lex.entries[bi].1, lex.entries[bi].0.clone(), best)
}
