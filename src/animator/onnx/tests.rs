use std::sync::Arc;

use crate::animator::{Kind, standard_actions};
use crate::sprite::{Palette, Sprite};

use super::*;

#[test]
fn seed_matches_python() {
    // row_seed(1, "human", "walk", 90, 32) etc. from export_anim_onnx.py
    assert_eq!(row_seed(1, 0, 2, 2, 32), ((2_654_435_761u64 + 2 * 40_503 + 2 * 9_973 + 32 * 31) % (1 << 32)) as u32);
    assert_eq!(row_seed(4_000_000_000, 2, 18, 7, 16), ((4_000_000_000u64 * 2_654_435_761 + 18 * 40_503 + 7 * 9_973 + 2 * 7_919 + 16 * 31) % (1 << 32)) as u32);
}

/// Own-palette sprite -> v2 tokens -> frame in the own palette (needs the exported tables: ignored by
/// default, run with `cargo test -- --ignored`; returns early if the tables are still missing).
#[test]
#[ignore = "needs the exported anim39 tables"]
fn reference_round_trip() {
    let anim = OnnxAnimator::new();
    let Ok(t) = anim.tables() else { return };
    // a farm-like palette: two exact v2 colours (in another order) and one that is not in v2's palette
    let (a, b) = (t.palette[17], t.palette[900]);
    let odd = [a[0].wrapping_add(1), a[1], a[2]];
    let odd = if t.rgb.contains_key(&odd) { [a[0], a[1].wrapping_add(1), a[2]] } else { odd };
    let pal: Palette = Arc::new(vec![b, a, odd]);
    let mut px = vec![-1i16; 16 * 16];
    px[0] = 0;
    px[1] = 1;
    px[2] = 2;
    let s = Sprite::new(16, px.clone(), pal);
    let r = anim.reference(&s).unwrap();
    assert_eq!((r.tokens[0], r.tokens[1], r.tokens[3]), (900, 17, t.empty as u16));
    assert_eq!(r.approximated, 1);
    assert!(r.allowed.contains(&900) && r.allowed.contains(&17) && r.allowed.contains(&t.empty));
    let back = r.to_frame(&r.tokens);
    assert_eq!(&back.px[..2], &px[..2]);
    assert_eq!(back.px[3], -1);
    // a sprite already in v2's palette maps 1:1
    let v2 = Sprite::new(16, (0..256).map(|i| if i % 3 == 0 { -1 } else { i as i16 }).collect(), Arc::new(t.palette.clone()));
    let r = anim.reference(&v2).unwrap();
    assert!(r.tokens.iter().zip(&v2.px).all(|(&tk, &p)| if p < 0 { tk == t.empty as u16 } else { tk == p as u16 }));
    assert_eq!(r.to_frame(&r.tokens).px, v2.px);
    // the context has the length the graphs expect
    let c = t.cond(Kind::Human, 0, "walk", 270, 3).unwrap();
    assert_eq!((c.nfr, c.frame), (8, 3));
    let mut x = vec![0f32; t.context_len(16) * t.d];
    t.write_context(&mut x, 16, &c, &r.tokens, &r.tokens);
    assert!(x.iter().all(|v| v.is_finite()) && t.context_len(16) == 2 + 16 + 1 + 512 + 1);
}

#[test]
fn anchor_action_and_split() {
    let has = |a: &str| standard_actions(Kind::Human).iter().any(|x| x.name == a);
    assert_eq!(anchor_action(Kind::Human, has), "walk");
    assert_eq!(anchor_action(Kind::Quadruped, |a| a == "walk"), "walk");
    assert_eq!(anchor_action(Kind::Bird, |a| a == "fly" || a == "glide"), "fly");
    // a model without walk: the first action it has
    assert_eq!(anchor_action(Kind::Human, |a| a == "run"), "run");
    let clips: Vec<(String, u16)> = vec![("run".into(), 90), ("walk".into(), 0), ("walk".into(), 90), ("punch".into(), 0)];
    let (rest, idx) = split_anchor_clips("walk", &clips);
    assert_eq!(idx, vec![0, 2, 3]);
    assert_eq!(rest.len(), 3);
    assert!(rest.iter().all(|c| !(c.0 == "walk" && c.1 == 0)));
}

/// An anchor reference keeps the sprite's way back to its own palette; its allowed colours follow the option and
/// the cache key (hash) differs between the options and from the sprite's.
/// Needs the exported tables: ignored by default (`cargo test -- --ignored`).
#[test]
#[ignore = "needs the exported anim39 tables"]
fn anchor_reference_variants() {
    let anim = OnnxAnimator::new();
    let Ok(t) = anim.tables() else { return };
    let pal: Palette = Arc::new(vec![t.palette[5], t.palette[600], t.palette[700]]);
    let mut px = vec![-1i16; 256];
    px[0] = 0;
    px[1] = 1;
    px[2] = 2;
    let s = Sprite::new(16, px, pal);
    let r = anim.reference(&s).unwrap();
    let empty = t.empty as u16;
    let mut tk = vec![empty; 256];
    tk[10] = 5;
    tk[11] = 5;
    let keep = r.with_tokens(tk.clone(), true);
    let own = r.with_tokens(tk.clone(), false);
    assert_eq!(keep.allowed, r.allowed);
    assert_eq!(own.allowed, vec![5, t.empty]);
    assert_ne!(keep.hash, own.hash);
    assert_ne!(keep.hash, r.hash);
    // frames come back in the sprite's own palette, also for colours the anchor itself does not use
    let f = keep.to_frame(&[600, 700, empty, 5].repeat(64));
    assert_eq!(&f.px[..4], &[1, 2, -1, 0]);
}

#[test]
fn draw_is_inverse_cdf() {
    let logits = [0.0f32, (2.0f32).ln(), 5.0, 0.0];
    let allowed = [0usize, 1, 3];
    let mut cum = Vec::new();
    // weights 1, 2, 1 of 4 (token 2 is not allowed)
    assert_eq!(draw(&logits, &allowed, 1.0, 0.0, &mut cum), 0);
    assert_eq!(draw(&logits, &allowed, 1.0, 0.24, &mut cum), 0);
    assert_eq!(draw(&logits, &allowed, 1.0, 0.26, &mut cum), 1);
    assert_eq!(draw(&logits, &allowed, 1.0, 0.74, &mut cum), 1);
    assert_eq!(draw(&logits, &allowed, 1.0, 0.76, &mut cum), 3);
    assert_eq!(draw(&logits, &allowed, 1.0, 0.999_999, &mut cum), 3);
    assert!((bits_of(&[0.0, 0.0, 0.0, 0.0], 2) - 2.0).abs() < 1e-12);
}
