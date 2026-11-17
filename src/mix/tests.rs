use super::*;
use std::collections::HashMap;

fn lex() -> Lexicon {
    let entries = [("burger", ConceptKind::Object), ("skull", ConceptKind::Object), ("fried", ConceptKind::Attribute), ("treasure chest", ConceptKind::Object), ("sword", ConceptKind::Object), ("frost", ConceptKind::Attribute)];
    Lexicon {
        stop: ["a", "of", "pixel", "art", "icon"].iter().map(|s| s.to_string()).collect(),
        phrases: ["treasure chest"].iter().map(|s| s.to_string()).collect(),
        kind: entries.iter().map(|(t, k)| (t.to_string(), *k)).collect(),
        index: entries.iter().enumerate().map(|(i, (t, _))| (t.to_string(), i)).collect(),
        entries: entries.iter().map(|(t, k)| (t.to_string(), *k)).collect(),
        dim: 2,
        vectors: vec![0.0; entries.len() * 2],
    }
}

#[test]
fn words_like_the_regex() {
    assert_eq!(words("rock'n'roll  м'ясо, burger-skull 42x"), ["rock'n", "roll", "м'ясо", "burger", "skull", "x"]);
    assert_eq!(words("it’s"), ["it’s"]);
    assert_eq!(words("a' b"), ["a", "b"]);
}

#[test]
fn parse_order_and_kinds() {
    let lx = lex();
    let p = parse("A fried Burger skull of treasure chest", &lx, &HashMap::new());
    let t: Vec<&str> = p.concepts.iter().map(|c| c.text.as_str()).collect();
    assert_eq!(t, ["fried", "burger", "skull"]);
    assert_eq!(p.dropped, 1);
    // Latin script: the last word is the strongest (1/2), then 1/4, 1/8, normalised
    let ps: Vec<f64> = p.concepts.iter().map(|c| c.p).collect();
    assert_eq!(ps, [0.125 / 0.875, 0.25 / 0.875, 0.5 / 0.875]);
    assert_eq!(p.concepts[0].kind, ConceptKind::Attribute);
    assert!(!p.cyr);
    let p = parse("treasure chest sword", &lx, &HashMap::from([("sword".to_string(), ConceptKind::Attribute)]));
    assert_eq!(p.concepts.iter().map(|c| (c.text.as_str(), c.kind)).collect::<Vec<_>>(), [("treasure chest", ConceptKind::Object), ("sword", ConceptKind::Attribute)]);
    // Cyrillic: the first word is the strongest; adjectives by ending
    let p = parse("жареный череп", &lx, &HashMap::new());
    assert!(p.cyr);
    assert_eq!(p.concepts[0].kind, ConceptKind::Attribute);
    assert!(p.concepts[0].known && !p.concepts[1].known);
    assert!(p.concepts[0].p > p.concepts[1].p);
}

#[test]
fn plan_and_variants() {
    let lx = lex();
    let c = parse("burger skull", &lx, &HashMap::new()).concepts;
    let pl = plan(&c, &VariantKind::ALL).unwrap();
    assert_eq!(pl.count, 2 + 2 + 2 + 1 + 1);
    assert!(plan(&c, &[VariantKind::Caption]).is_none());
    let c1 = parse("frost sword", &lx, &HashMap::new()).concepts;
    let p1 = plan(&c1, &VariantKind::ALL).unwrap();
    assert_eq!(p1.kinds, [VariantKind::Blend, VariantKind::Caption]);
    assert!(unfit(VariantKind::Inside, &c1).is_some());
    // same seed, same variants; the label format of mix.js
    let v = draw_variants(&pl, 8, 3);
    assert_eq!(v, draw_variants(&pl, 8, 3));
    let blend = Variant { kind: VariantKind::Blend, a: Some(1), b: Some(0), others: vec![], attributes: vec![] };
    assert_eq!(label(&blend, &pl, "burger skull"), "skull + burger");
    let ins = Variant { kind: VariantKind::Inside, a: Some(1), b: Some(0), others: vec![], attributes: vec![] };
    assert_eq!(label(&ins, &pl, "burger skull"), "skull / burger");
    // maps: A high in the middle band, B high above and below (16 px)
    let m = variant_maps(&ins, &pl, 16, 0, 16);
    assert_eq!(m[1][8 * 16], 1.6f32);
    assert_eq!(m[0][8 * 16], 0.4f32);
    assert_eq!(m[1][0], 0.4f32);
    assert_eq!(m[0][0], 1.6f32);
}

#[test]
fn hypot_matches_v8() {
    // values computed with Node 24 (Math.hypot)
    assert_eq!(js_hypot(&[3.0, 4.0]), 5.0);
    assert_eq!(js_hypot(&[0.1, 0.2, 0.3]), 0.37416573867739417);
    assert_eq!(js_hypot(&[-0.21875, 0.46875]), 0.517279542413964);
    assert_eq!(js_hypot(&[0.36691597986741337, -0.4064283729806926]), 0.5475503252175065);
    assert_eq!(js_hypot(&[-0.342551432461261, 0.497592311076468]), 0.604102302531353);
    // 384 numbers, as the mean caption vector: Math.fround(((i * 37) % 101) * 0.0013 - 0.06)
    let v: Vec<f64> = (0..384).map(|i| (((i * 37) % 101) as f64 * 0.0013 - 0.06) as f32 as f64).collect();
    assert_eq!(js_hypot(&v), 0.7498968681635478);
}

fn variant(kind: VariantKind, a: Option<usize>, b: Option<usize>, others: &[usize], attributes: &[usize]) -> Variant {
    Variant { kind, a, b, others: others.to_vec(), attributes: attributes.to_vec() }
}

fn plan_of(caption: &str) -> Plan {
    let c = parse(caption, &lex(), &HashMap::new()).concepts;
    plan(&c, &VariantKind::ALL).unwrap()
}

#[test]
fn blend_and_single_caption_maps_are_flat() {
    let pl = plan_of("burger skull");
    let blend = variant_maps(&variant(VariantKind::Blend, Some(0), Some(1), &[], &[]), &pl, 4, 0, 4);
    assert!(blend.iter().all(|m| m.len() == 16 && m.iter().all(|&v| v == 1.0)));
    // one caption: its map is full strength, the other concept is switched off
    let single = variant_maps(&variant(VariantKind::Caption, Some(1), None, &[], &[]), &pl, 4, 0, 4);
    assert!(single[1].iter().all(|&v| v == 1.0));
    assert!(single[0].iter().all(|&v| v == 0.0));
}

#[test]
fn third_objects_and_attributes_get_their_constant_weights() {
    let pl = plan_of("burger skull sword");
    let m = variant_maps(&variant(VariantKind::Inside, Some(1), Some(2), &[0], &[]), &pl, 4, 0, 4);
    assert!(m[0].iter().all(|&v| v == THIRD as f32));
    let pl = plan_of("fried burger skull");
    let m = variant_maps(&variant(VariantKind::Inside, Some(1), Some(2), &[], &[0]), &pl, 4, 0, 4);
    let want = pl.attr_strength(0) as f32;
    assert!(m[0].iter().all(|&v| v == want));
    // an attribute is never silenced (>= half strength) and never louder than ATTR
    assert!(want >= 0.5 * ATTR as f32 && want <= ATTR as f32);
}

#[test]
fn layout_weights_of_two_objects_always_sum_to_lo_plus_hi() {
    let pl = plan_of("burger skull");
    // full canvas, and a 24 window in the middle of a 32 canvas
    for (s, o, w) in [(16, 0, 16), (32, 4, 24)] {
        for kind in [VariantKind::Inside, VariantKind::Over, VariantKind::Centre] {
            let m = variant_maps(&variant(kind, Some(0), Some(1), &[], &[]), &pl, s, o, w);
            for p in 0..s * s {
                assert!((0.4 - 1e-6..=1.6 + 1e-6).contains(&m[0][p]), "{kind:?} a at {p}: {}", m[0][p]);
                assert!((m[0][p] + m[1][p] - 2.0).abs() < 1e-5, "{kind:?} at {p}: {} + {}", m[0][p], m[1][p]);
            }
        }
    }
}

#[test]
fn mean_caption_is_normalised() {
    assert_eq!(mean_caption(&[vec![3.0, 0.0], vec![0.0, 4.0]]), vec![0.6, 0.8]);
    // a zero sum is left as it is instead of dividing by zero
    assert_eq!(mean_caption(&[vec![0.0, 0.0]]), vec![0.0, 0.0]);
    assert!(mean_caption(&[]).is_empty());
}

#[test]
fn nearest_kind_uses_single_words_only() {
    let mut lx = lex();
    // burger, skull, fried, treasure chest, sword, frost
    lx.vectors = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.6, 0.8, 0.0, -1.0, -0.6, -0.8];
    let (kind, word, cos) = nearest_kind(&[-0.95, 0.0], &lx);
    assert_eq!((kind, word.as_str()), (ConceptKind::Attribute, "fried"));
    assert!((cos - 0.95).abs() < 1e-6);
    // (0.6, 0.8) is exactly the phrase "treasure chest", which is skipped: skull (0.8) wins over burger (0.6)
    let (kind, word, _) = nearest_kind(&[0.6, 0.8], &lx);
    assert_eq!((kind, word.as_str()), (ConceptKind::Object, "skull"));
}
