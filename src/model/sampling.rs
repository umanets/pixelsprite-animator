// ------------------------------------------------------------------------------------------------- sampling
fn log_sum_exp(a: &[f32]) -> f64 {
    let m = a.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x as f64));
    let s: f64 = a.iter().map(|&x| (x as f64 - m).exp()).sum();
    m + s.ln()
}

/// log p(EMPTY) - log(1 - p(EMPTY)) of one row of V = K + 1 logits.
fn empty_log_odds(l: &[f32], e: usize) -> f64 {
    let lp = l[e] as f64 - log_sum_exp(l);
    lp - (-(lp.exp().min(1.0 - 1e-6))).ln_1p()
}

/// `guided` of engine.js: fills `col` with colour probabilities, returns p(EMPTY). lu = None: no guidance.
#[allow(clippy::too_many_arguments)]
pub(super) fn guided(lc: &[f32], lu: Option<&[f32]>, k: usize, g_shape: f64, g_colour: f64, temperature: f64, top_p: f64, col: &mut [f64], order: &mut Vec<usize>) -> f64 {
    let e = k;
    let s = match lu {
        Some(lu) => {
            let sc = empty_log_odds(lc, e);
            let su = empty_log_odds(lu, e);
            let ls_c = log_sum_exp(&lc[..k]);
            let ls_u = log_sum_exp(&lu[..k]);
            for j in 0..k {
                let u = lu[j] as f64 - ls_u;
                col[j] = u + g_colour * ((lc[j] as f64 - ls_c) - u);
            }
            su + g_shape * (sc - su)
        }
        None => {
            for j in 0..k {
                col[j] = lc[j] as f64;
            }
            empty_log_odds(lc, e)
        }
    };
    finish_colour(col, temperature, top_p, order);
    1.0 / (1.0 + (-s).exp())
}

/// `guidedMix` of engine.js: rows base .. base+k-1 hold the k captions, row base+k the "no text" condition; w[g]
/// is caption g's weight at this pixel. Shape log-odds and colour log-probabilities s = s_u + sum_g w_g (s_g - s_u),
/// then temperature and top-p as in [`guided`]. Fills `col`, returns p(EMPTY).
#[allow(clippy::too_many_arguments)]
pub(super) fn guided_mix(l: &[f32], base: usize, w: &[f64], k: usize, temperature: f64, top_p: f64, col: &mut [f64], order: &mut Vec<usize>) -> f64 {
    let v = k + 1;
    let kc = w.len();
    let lu = &l[(base + kc) * v..(base + kc + 1) * v];
    let su = empty_log_odds(lu, k);
    let ls_u = log_sum_exp(&lu[..k]);
    let mut s = su;
    for j in 0..k {
        col[j] = lu[j] as f64 - ls_u;
    }
    for (g, &wg) in w.iter().enumerate() {
        if wg == 0.0 {
            continue;
        }
        let lg = &l[(base + g) * v..(base + g + 1) * v];
        s += wg * (empty_log_odds(lg, k) - su);
        let ls = log_sum_exp(&lg[..k]);
        for j in 0..k {
            col[j] += wg * ((lg[j] as f64 - ls) - (lu[j] as f64 - ls_u));
        }
    }
    finish_colour(col, temperature, top_p, order);
    1.0 / (1.0 + (-s).exp())
}

/// Temperature, softmax, then top-p (keeps the most likely colours whose mass before them is < top_p).
fn finish_colour(col: &mut [f64], temperature: f64, top_p: f64, order: &mut Vec<usize>) {
    let inv = 1.0 / temperature.max(1e-6);
    let mut m = f64::NEG_INFINITY;
    for c in col.iter_mut() {
        *c *= inv;
        m = m.max(*c);
    }
    let mut sum = 0.0;
    for c in col.iter_mut() {
        *c = (*c - m).exp();
        sum += *c;
    }
    for c in col.iter_mut() {
        *c /= sum;
    }
    if top_p < 1.0 {
        order.clear();
        order.extend(0..col.len());
        order.sort_by(|&a, &b| col[b].partial_cmp(&col[a]).unwrap_or(std::cmp::Ordering::Equal));
        let (mut cum, mut kept) = (0.0, 0.0);
        for &j in order.iter() {
            if cum < top_p {
                cum += col[j];
                kept += col[j];
            } else {
                col[j] = 0.0;
            }
        }
        for c in col.iter_mut() {
            *c /= kept;
        }
    }
}

pub(super) fn draw(col: &[f64], u: f64) -> usize {
    let mut acc = 0.0;
    for (j, &p) in col.iter().enumerate() {
        acc += p;
        if u < acc {
            return j;
        }
    }
    (0..col.len()).rev().find(|&j| col[j] > 0.0).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn log_sum_exp_is_stable_for_large_logits() {
        assert!(close(log_sum_exp(&[0.0, 0.0]), 2f64.ln()));
        assert!(close(log_sum_exp(&[1000.0, 1000.0]), 1000.0 + 2f64.ln()));
    }

    #[test]
    fn empty_log_odds_of_uniform_logits() {
        // three classes, EMPTY is class 2: p = 1/3, log-odds = ln(1/3) - ln(2/3) = -ln 2
        assert!(close(empty_log_odds(&[0.0, 0.0, 0.0], 2), -(2f64.ln())));
    }

    #[test]
    fn finish_colour_normalises_and_applies_top_p() {
        let probs = [0.6f64, 0.3, 0.1];
        let mut order = Vec::new();
        let mut col: Vec<f64> = probs.iter().map(|p| p.ln()).collect();
        finish_colour(&mut col, 1.0, 1.0, &mut order);
        for (c, p) in col.iter().zip(probs) {
            assert!(close(*c, p));
        }
        // top-p 0.5: the most likely colour already holds 0.6 of the mass, so only it stays
        let mut col: Vec<f64> = probs.iter().map(|p| p.ln()).collect();
        finish_colour(&mut col, 1.0, 0.5, &mut order);
        assert!(close(col[0], 1.0) && col[1] == 0.0 && col[2] == 0.0);
        // top-p 0.7: 0.6 < 0.7 so the second colour is kept as well, renormalised to 2/3 and 1/3
        let mut col: Vec<f64> = probs.iter().map(|p| p.ln()).collect();
        finish_colour(&mut col, 1.0, 0.7, &mut order);
        assert!(close(col[0], 2.0 / 3.0) && close(col[1], 1.0 / 3.0) && col[2] == 0.0);
    }

    #[test]
    fn top_p_stops_once_the_kept_mass_reaches_it() {
        // two equally likely colours, top-p 0.5: the first already holds exactly 0.5, so the second is dropped
        let mut col = vec![0.5f64.ln(), 0.5f64.ln()];
        finish_colour(&mut col, 1.0, 0.5, &mut Vec::new());
        assert!(close(col[0], 1.0) && col[1] == 0.0);
    }

    #[test]
    fn low_temperature_concentrates_on_the_best_colour() {
        let mut col = vec![1.0, 2.0, 0.5];
        finish_colour(&mut col, 0.01, 1.0, &mut Vec::new());
        assert!(col[1] > 0.999);
    }

    #[test]
    fn guided_without_guidance_reads_the_conditional_row() {
        // two colours + EMPTY; uniform logits: colours 1/2 each, p(EMPTY) = 1/3
        let mut col = [0.0; 2];
        let p = guided(&[0.0, 0.0, 0.0], None, 2, 1.0, 1.0, 1.0, 1.0, &mut col, &mut Vec::new());
        assert!(close(p, 1.0 / 3.0));
        assert!(close(col[0], 0.5) && close(col[1], 0.5));
    }

    #[test]
    fn guided_scale_zero_is_unconditional_and_one_is_conditional() {
        let lc = [2.0f32, 0.0, -1.0];
        let lu = [0.0f32, 1.0, 0.5];
        let run = |l: &[f32], lu: Option<&[f32]>, g: f64| {
            let mut col = [0.0; 2];
            let p = guided(l, lu, 2, g, g, 1.0, 1.0, &mut col, &mut Vec::new());
            (p, col)
        };
        let (p0, c0) = run(&lc, Some(&lu), 0.0);
        let (pu, cu) = run(&lu, None, 1.0);
        assert!(close(p0, pu) && close(c0[0], cu[0]) && close(c0[1], cu[1]));
        let (p1, c1) = run(&lc, Some(&lu), 1.0);
        let (pc, cc) = run(&lc, None, 1.0);
        assert!(close(p1, pc) && close(c1[0], cc[0]) && close(c1[1], cc[1]));
    }

    #[test]
    fn guided_mix_with_one_caption_equals_guided_at_full_weight() {
        // rows: caption 0, then the "no text" row
        let lc = [2.0f32, 0.0, -1.0];
        let lu = [0.0f32, 1.0, 0.5];
        let l: Vec<f32> = lc.iter().chain(&lu).copied().collect();
        let mut a = [0.0; 2];
        let pa = guided_mix(&l, 0, &[1.0], 2, 1.0, 1.0, &mut a, &mut Vec::new());
        let mut b = [0.0; 2];
        let pb = guided(&lc, Some(&lu), 2, 1.0, 1.0, 1.0, 1.0, &mut b, &mut Vec::new());
        assert!(close(pa, pb) && close(a[0], b[0]) && close(a[1], b[1]));
    }

    #[test]
    fn guided_mix_with_zero_weight_ignores_the_caption() {
        let lc = [5.0f32, -5.0, 3.0];
        let lu = [0.0f32, 1.0, 0.5];
        let l: Vec<f32> = lc.iter().chain(&lu).copied().collect();
        let mut a = [0.0; 2];
        let pa = guided_mix(&l, 0, &[0.0], 2, 1.0, 1.0, &mut a, &mut Vec::new());
        let mut b = [0.0; 2];
        let pb = guided(&lu, None, 2, 1.0, 1.0, 1.0, 1.0, &mut b, &mut Vec::new());
        assert!(close(pa, pb) && close(a[0], b[0]) && close(a[1], b[1]));
    }

    #[test]
    fn draw_is_an_inverse_cdf_with_a_fallback_to_the_last_possible_colour() {
        let col = [0.2, 0.3, 0.5];
        assert_eq!(draw(&col, 0.1), 0);
        assert_eq!(draw(&col, 0.25), 1);
        assert_eq!(draw(&col, 0.99), 2);
        // rounding can leave u >= the summed mass: the last colour with probability stays valid
        assert_eq!(draw(&[0.5, 0.5, 0.0], 1.0), 1);
    }
}
