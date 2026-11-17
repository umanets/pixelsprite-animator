//! Metrics and tables of `selftest --anchor-eval` (the "build everything from the first frame of walk" option).
//!
//! Colour drift of a drawn frame is measured against the colours of the sprite it was drawn from ("input"):
//!
//! * `foreign`  share of drawn pixels whose colour is not used by the input sprite (the sampler is restricted to the
//!   input's palette, so this is 0 by construction for the model; it is kept to prove that, and as a guard);
//! * `fdist`    mean CIE76 distance of the drawn pixels to the nearest colour of the input (0 when `foreign` is 0);
//! * `tv`       total-variation distance between the frame's colour histogram and the input's (0 = same mix of
//!   colours, 1 = disjoint); side views legitimately change the mix a little, so the ground truth's own `tv` is the
//!   floor to compare with;
//! * `mde`      CIE76 distance between the frame's mean colour and the input's mean colour (a global tone / hue shift);
//! * `spread`   inside a clip: root mean square distance (Lab) of each frame's mean colour to the clip's mean colour
//!   (flicker of the body colour from frame to frame);
//! * `agree`, `iou`  pixel agreement and silhouette IoU with the farm's ground-truth render (val characters);
//! * `gde`      CIE76 distance between the mean colours of the frame and of the ground-truth frame.

use std::collections::BTreeMap;

use crate::sprite::Sprite;

pub const METRICS: [&str; 9] = ["foreign", "fdist", "tv", "mde", "spread", "agree", "iou", "gde", "n"];
pub const N_METRICS: usize = METRICS.len();
/// Metrics compared pairwise (on vs off) with the direction in which they are better (-1 = lower, +1 = higher).
pub const PAIRED: [(usize, i8); 7] = [(0, -1), (1, -1), (2, -1), (3, -1), (4, -1), (5, 1), (6, 1)];

pub fn lab(c: [u8; 3]) -> [f32; 3] {
    let lin = |v: u8| {
        let v = v as f32 / 255.0;
        if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
    };
    let (r, g, b) = (lin(c[0]), lin(c[1]), lin(c[2]));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: f32| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

pub fn de(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn opaque(s: &Sprite) -> impl Iterator<Item = [u8; 3]> + '_ {
    s.px.iter().filter(|&&p| p >= 0).filter_map(|&p| s.palette.get(p as usize).copied())
}

/// The colours of a sprite: distinct colours (RGB and Lab), their shares and the mean colour.
pub struct Colours {
    pub rgb: Vec<[u8; 3]>,
    pub lab: Vec<[f32; 3]>,
    pub share: Vec<f64>,
    pub count: Vec<usize>,
    pub mean: [f32; 3],
}

impl Colours {
    pub fn of(s: &Sprite) -> Option<Colours> {
        let mut count: BTreeMap<[u8; 3], usize> = BTreeMap::new();
        for c in opaque(s) {
            *count.entry(c).or_default() += 1;
        }
        let n: usize = count.values().sum();
        if n == 0 {
            return None;
        }
        let rgb: Vec<[u8; 3]> = count.keys().copied().collect();
        let labs: Vec<[f32; 3]> = rgb.iter().map(|&c| lab(c)).collect();
        let share: Vec<f64> = count.values().map(|&k| k as f64 / n as f64).collect();
        let mut mean = [0f32; 3];
        for (l, w) in labs.iter().zip(&share) {
            for i in 0..3 {
                mean[i] += l[i] * *w as f32;
            }
        }
        Some(Colours { rgb, lab: labs, share, count: count.values().copied().collect(), mean })
    }

    fn share_of(&self, c: [u8; 3]) -> f64 {
        self.rgb.binary_search(&c).map(|i| self.share[i]).unwrap_or(0.0)
    }

    /// Total-variation distance of two colour histograms.
    pub fn tv(&self, o: &Colours) -> f64 {
        let mut all: Vec<[u8; 3]> = self.rgb.iter().chain(&o.rgb).copied().collect();
        all.sort_unstable();
        all.dedup();
        0.5 * all.iter().map(|&c| (self.share_of(c) - o.share_of(c)).abs()).sum::<f64>()
    }
}

/// One frame against the input sprite's colours.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameM {
    pub px: usize,
    pub foreign_px: usize,
    /// Sum over the pixels of the distance to the nearest input colour.
    pub dist_sum: f64,
    pub tv: f64,
    pub mean_de: f64,
    pub mean: [f32; 3],
}

pub fn frame_metrics(f: &Sprite, input: &Colours) -> Option<FrameM> {
    let c = Colours::of(f)?;
    let (mut px, mut foreign_px, mut dist_sum) = (0usize, 0usize, 0f64);
    for ((rgb, l), &k) in c.rgb.iter().zip(&c.lab).zip(&c.count) {
        px += k;
        if input.rgb.binary_search(rgb).is_err() {
            foreign_px += k;
            let d = input.lab.iter().map(|il| de(*l, *il)).fold(f32::MAX, f32::min);
            dist_sum += d as f64 * k as f64;
        }
    }
    Some(FrameM { px, foreign_px, dist_sum, tv: c.tv(input), mean_de: de(c.mean, input.mean) as f64, mean: c.mean })
}

/// Root mean square distance of the frames' mean colours to their average (flicker of the body colour in a clip).
pub fn clip_spread(means: &[[f32; 3]]) -> f64 {
    if means.len() < 2 {
        return 0.0;
    }
    let n = means.len() as f32;
    let avg: [f32; 3] = std::array::from_fn(|i| means.iter().map(|m| m[i]).sum::<f32>() / n);
    (means.iter().map(|m| de(*m, avg).powi(2) as f64).sum::<f64>() / means.len() as f64).sqrt()
}

/// Accumulates frames of one character in one mode into the metric vector (order of [`METRICS`]).
#[derive(Default)]
pub struct Acc {
    px: usize,
    foreign: usize,
    dist: f64,
    tv: Vec<f64>,
    mde: Vec<f64>,
    spread: Vec<f64>,
    agree: Vec<f64>,
    iou: Vec<f64>,
    gde: Vec<f64>,
    frames: usize,
}

impl Acc {
    /// A set of frames that belong together (a clip: spread is taken over it; a pose list of views: pass false).
    pub fn add_frames(&mut self, frames: &[Sprite], input: &Colours, as_clip: bool) {
        let ms: Vec<FrameM> = frames.iter().filter_map(|f| frame_metrics(f, input)).collect();
        for m in &ms {
            self.px += m.px;
            self.foreign += m.foreign_px;
            self.dist += m.dist_sum;
            self.tv.push(m.tv);
            self.mde.push(m.mean_de);
            self.frames += 1;
        }
        if as_clip && ms.len() > 1 {
            self.spread.push(clip_spread(&ms.iter().map(|m| m.mean).collect::<Vec<_>>()));
        }
    }

    /// Frame against its ground truth (same tokens = same palette).
    pub fn add_gt(&mut self, f: &Sprite, gt: &Sprite) {
        let n = f.px.len().max(1) as f64;
        let ident = |s: &Sprite, i: usize| -> i32 { s.px[i] as i32 };
        let same = (0..f.px.len()).filter(|&i| ident(f, i) == ident(gt, i)).count() as f64 / n;
        let (mut inter, mut uni) = (0usize, 0usize);
        for i in 0..f.px.len() {
            let (p, q) = (f.px[i] >= 0, gt.px[i] >= 0);
            inter += (p && q) as usize;
            uni += (p || q) as usize;
        }
        self.agree.push(same);
        self.iou.push(if uni == 0 { 1.0 } else { inter as f64 / uni as f64 });
        if let (Some(a), Some(b)) = (Colours::of(f), Colours::of(gt)) {
            self.gde.push(de(a.mean, b.mean) as f64);
        }
    }

    pub fn finish(&self) -> [f64; N_METRICS] {
        let mean = |v: &[f64]| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };
        let px = self.px.max(1) as f64;
        [
            if self.frames == 0 { f64::NAN } else { 100.0 * self.foreign as f64 / px },
            if self.frames == 0 { f64::NAN } else { self.dist / px },
            mean(&self.tv),
            mean(&self.mde),
            mean(&self.spread),
            mean(&self.agree),
            mean(&self.iou),
            mean(&self.gde),
            self.frames as f64,
        ]
    }
}

/// One character in one mode.
#[derive(Debug, Clone)]
pub struct Row {
    pub set: String,
    pub size: usize,
    pub kind: String,
    pub name: String,
    pub mode: String,
    pub v: [f64; N_METRICS],
}

impl Row {
    pub fn csv(&self) -> String {
        let v: Vec<String> = self.v.iter().map(|x| if x.is_nan() { "nan".into() } else { format!("{x:.5}") }).collect();
        format!("{},{},{},{},{},{}", self.set, self.size, self.kind, self.name, self.mode, v.join(","))
    }

    pub fn parse(line: &str) -> Option<Row> {
        let f: Vec<&str> = line.trim().split(',').collect();
        if f.len() != 5 + N_METRICS {
            return None;
        }
        let mut v = [f64::NAN; N_METRICS];
        for (i, x) in f[5..].iter().enumerate() {
            v[i] = x.parse().ok()?;
        }
        Some(Row { set: f[0].into(), size: f[1].parse().ok()?, kind: f[2].into(), name: f[3].into(), mode: f[4].into(), v })
    }

    pub fn csv_header() -> String {
        format!("set,size,kind,name,mode,{}", METRICS.join(","))
    }
}

fn nan_mean(v: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = v.filter(|x| !x.is_nan()).collect();
    if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
}

fn f(x: f64, d: usize) -> String {
    if x.is_nan() { "-".into() } else { format!("{x:.d$}") }
}

/// Mean and standard error of paired differences (mode - base) and how many characters moved in the better /
/// worse direction (`dir` -1 = lower is better).
pub fn paired(base: &[f64], other: &[f64], dir: i8) -> Option<(f64, f64, usize, usize, usize)> {
    let d: Vec<f64> = base.iter().zip(other).filter(|(a, b)| !a.is_nan() && !b.is_nan()).map(|(a, b)| b - a).collect();
    if d.is_empty() {
        return None;
    }
    let n = d.len() as f64;
    let m = d.iter().sum::<f64>() / n;
    let var = if d.len() > 1 { d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0) } else { 0.0 };
    let eps = 1e-9;
    let better = d.iter().filter(|&&x| x * (dir as f64) > eps).count();
    let worse = d.iter().filter(|&&x| x * (dir as f64) < -eps).count();
    Some((m, (var / n).sqrt(), better, d.len() - better - worse, worse))
}

/// The report: per (set, size, kind) the mean of each metric per mode, then the paired comparison with `base`.
pub fn tables(rows: &[Row], base: &str) -> String {
    let mut out = String::new();
    let mut modes: Vec<String> = Vec::new();
    for r in rows {
        if !modes.contains(&r.mode) {
            modes.push(r.mode.clone());
        }
    }
    // groups in a stable order
    let mut groups: Vec<(String, usize, String)> = rows.iter().map(|r| (r.set.clone(), r.size, r.kind.clone())).collect();
    groups.sort();
    groups.dedup();
    let mut with_all: Vec<(String, usize, String)> = Vec::new();
    for (set, size, _) in &groups {
        let g = (set.clone(), *size, "all".to_string());
        if !with_all.contains(&g) {
            with_all.push(g);
        }
    }
    for (set, size, kind) in groups.iter().cloned().chain(with_all) {
        let sel = |mode: &str| -> BTreeMap<String, &Row> {
            rows.iter().filter(|r| r.set == set && r.size == size && (kind == "all" || r.kind == kind) && r.mode == mode).map(|r| (r.name.clone(), r)).collect()
        };
        let nb = sel(base);
        if nb.is_empty() && !modes.iter().any(|m| !sel(m).is_empty()) {
            continue;
        }
        let n_chars = modes.iter().map(|m| sel(m).len()).max().unwrap_or(0);
        out.push_str(&format!("\n### {set} {size}px {kind} ({n_chars} characters)\n\n"));
        out.push_str("| mode | n | foreign % | fdist dE | hist TV | mean-colour dE | clip spread dE | pixels = GT | IoU | GT mean dE |\n|---|---|---|---|---|---|---|---|---|---|\n");
        for m in &modes {
            let s = sel(m);
            if s.is_empty() {
                continue;
            }
            let col = |i: usize| nan_mean(s.values().map(|r| r.v[i]));
            out.push_str(&format!(
                "| {m} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                s.len(),
                f(col(0), 2),
                f(col(1), 2),
                f(col(2), 3),
                f(col(3), 2),
                f(col(4), 2),
                f(col(5), 3),
                f(col(6), 3),
                f(col(7), 2)
            ));
        }
        for m in modes.iter().filter(|m| *m != base && *m != "gt") {
            let (s, b) = (sel(m), &nb);
            let names: Vec<&String> = s.keys().filter(|k| b.contains_key(*k)).collect();
            if names.is_empty() {
                continue;
            }
            let mut cells = Vec::new();
            for (i, dir) in PAIRED {
                let (x, y): (Vec<f64>, Vec<f64>) = names.iter().map(|k| (b[*k].v[i], s[*k].v[i])).unzip();
                if let Some((mean, se, better, tie, worse)) = paired(&x, &y, dir) {
                    cells.push(format!("{} {:+.3} +- {:.3} (better {better} / tie {tie} / worse {worse})", METRICS[i], mean, se));
                }
            }
            out.push_str(&format!("\npaired {m} - {base}, {} characters: {}\n", names.len(), cells.join("; ")));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn sprite(px: &[i16], pal: &[[u8; 3]]) -> Sprite {
        Sprite::new(2, px.to_vec(), Arc::new(pal.to_vec()))
    }

    #[test]
    fn identical_colours_have_no_drift() {
        let pal = [[200, 30, 30], [30, 30, 200], [10, 200, 10]];
        let input = Colours::of(&sprite(&[0, 1, 0, -1], &pal)).unwrap();
        let same = frame_metrics(&sprite(&[1, 0, -1, 0], &pal), &input).unwrap();
        assert_eq!((same.px, same.foreign_px), (3, 0));
        assert!(same.dist_sum == 0.0 && same.tv < 1e-9 && same.mean_de < 1e-3); // same mix 2:1
        let drift = frame_metrics(&sprite(&[2, 2, 2, 0], &pal), &input).unwrap();
        assert_eq!((drift.px, drift.foreign_px), (4, 3));
        assert!(drift.dist_sum > 0.0 && drift.tv > 0.5 && drift.mean_de > 10.0);
    }

    #[test]
    fn tv_and_spread() {
        let pal = [[0, 0, 0], [255, 255, 255]];
        let a = Colours::of(&sprite(&[0, 0, 0, 0], &pal)).unwrap();
        let b = Colours::of(&sprite(&[1, 1, 1, 1], &pal)).unwrap();
        let h = Colours::of(&sprite(&[0, 0, 1, 1], &pal)).unwrap();
        assert!((a.tv(&b) - 1.0).abs() < 1e-9 && (a.tv(&h) - 0.5).abs() < 1e-9 && a.tv(&a) == 0.0);
        assert_eq!(clip_spread(&[a.mean, a.mean, a.mean]), 0.0);
        assert!(clip_spread(&[a.mean, b.mean]) > 40.0);
        assert!((lab([255, 255, 255])[0] - 100.0).abs() < 0.1 && lab([0, 0, 0])[0].abs() < 0.1);
    }

    #[test]
    fn rows_round_trip_and_paired() {
        let r = Row { set: "val".into(), size: 16, kind: "human".into(), name: "s16_1".into(), mode: "on".into(), v: [0.0, 0.0, 0.2, 3.0, 1.5, f64::NAN, 0.8, 4.0, 40.0] };
        let back = Row::parse(&r.csv()).unwrap();
        assert_eq!((back.mode.as_str(), back.size), ("on", 16));
        assert!(back.v[5].is_nan() && (back.v[3] - 3.0).abs() < 1e-9);
        let (m, _, better, tie, worse) = paired(&[1.0, 2.0, 3.0], &[0.5, 2.0, 4.0], -1).unwrap();
        assert!((m - 0.1666).abs() < 1e-3 && (better, tie, worse) == (1, 1, 1));
        let rows = vec![r.clone(), Row { mode: "off".into(), ..r }];
        assert!(tables(&rows, "off").contains("val 16px human"));
    }
}
