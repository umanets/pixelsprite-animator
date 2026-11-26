use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

use crate::animator::Kind;

// ------------------------------------------------------------------------------------------------------ tables
#[derive(Debug, Deserialize)]
struct TableInfo {
    offset: usize,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct TokenKinds {
    pixel: usize,
    text: usize,
    size: usize,
    row_unknown: usize,
}

#[derive(Debug, Deserialize)]
struct Info {
    format: String,
    tables: HashMap<String, TableInfo>,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    n_colors: usize,
    max_size: usize,
    sizes: Vec<usize>,
    empty: usize,
    bos: usize,
    kinds_of_token: TokenKinds,
    edge_unknown: usize,
    actions: Vec<String>,
    kinds: Vec<String>,
    azimuths: Vec<u16>,
    elevations: Vec<u16>,
    max_frames: usize,
    frames: HashMap<String, HashMap<String, usize>>,
    palette: Vec<[u8; 3]>,
    #[serde(default)]
    step: Option<u64>,
    #[serde(default)]
    checkpoint: Option<String>,
}

/// The condition token's ids (indices into the embedding tables).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cond {
    pub kind: usize,
    pub refk: usize,
    pub action: usize,
    pub az: usize,
    pub el: usize,
    pub frame: usize,
    pub nfr: usize,
}

/// Embedding tables + metadata of `embeddings.json`; builds the input vectors exactly as `NpAnim` of the exporter
/// (sums in the same order, float32).
pub struct AnimTables {
    pub d: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub hd: usize,
    /// Number of colours K; logits have K + 1 entries (the last = EMPTY).
    pub k: usize,
    pub empty: usize,
    pub bos: usize,
    pub sizes: Vec<usize>,
    pub max_size: usize,
    pub actions: Vec<String>,
    pub kinds: Vec<String>,
    pub azimuths: Vec<u16>,
    pub max_frames: usize,
    pub frames: HashMap<String, HashMap<String, usize>>,
    /// v2's global palette (token -> RGB).
    pub palette: Vec<[u8; 3]>,
    pub step: Option<u64>,
    pub checkpoint: String,
    tk: TokenKinds,
    edge_unknown: usize,
    el0: usize,
    data: Vec<f32>,
    idx: HashMap<String, (usize, usize)>,
    pub(super) rgb: HashMap<[u8; 3], u16>,
}

impl AnimTables {
    pub fn load(dir: &Path) -> Result<Self> {
        let info: Info = serde_json::from_str(&std::fs::read_to_string(dir.join("embeddings.json")).with_context(|| format!("{}", dir.join("embeddings.json").display()))?)
            .context("embeddings.json")?;
        if info.format != "pixelsprite-anim" {
            bail!("embeddings.json: format {:?}, expected pixelsprite-anim", info.format);
        }
        let raw = std::fs::read(dir.join("embeddings.bin")).context("embeddings.bin")?;
        let data: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut idx = HashMap::new();
        for (name, t) in &info.tables {
            let n: usize = t.shape.iter().product();
            let rows = if t.shape.len() == 2 { t.shape[0] } else { 1 };
            let width = t.shape.last().copied().unwrap_or(0);
            if t.offset % 4 != 0 || t.offset / 4 + n > data.len() || width != info.d_model {
                bail!("embeddings.bin: table {name} out of range or not {} wide", info.d_model);
            }
            idx.insert(name.clone(), (t.offset / 4, rows));
        }
        for need in ["null_cond", "size_emb", "mask_proj.bias", "tok", "row", "col", "edge", "kind", "act_emb", "kind_emb", "az_emb", "el_emb", "frame_emb", "nfr_emb", "refk_emb", "seg"] {
            if !idx.contains_key(need) {
                bail!("embeddings.bin: no table {need}");
            }
        }
        if info.palette.len() != info.n_colors || info.empty != info.n_colors {
            bail!("embeddings.json: palette / EMPTY do not match n_colors");
        }
        let rgb = info.palette.iter().enumerate().map(|(i, c)| (*c, i as u16)).collect();
        let el0 = info.elevations.iter().position(|&e| e == 0).unwrap_or(0);
        Ok(AnimTables {
            d: info.d_model,
            n_layers: info.n_layers,
            n_heads: info.n_heads,
            hd: info.d_model / info.n_heads,
            k: info.n_colors,
            empty: info.empty,
            bos: info.bos,
            sizes: info.sizes,
            max_size: info.max_size,
            actions: info.actions,
            kinds: info.kinds,
            azimuths: info.azimuths,
            max_frames: info.max_frames,
            frames: info.frames,
            palette: info.palette,
            step: info.step,
            checkpoint: info.checkpoint.unwrap_or_default(),
            tk: info.kinds_of_token,
            edge_unknown: info.edge_unknown,
            el0,
            data,
            idx,
            rgb,
        })
    }

    fn row(&self, name: &str, i: usize) -> &[f32] {
        let (o, rows) = self.idx[name];
        debug_assert!(i < rows, "{name}[{i}]");
        &self.data[o + i * self.d..o + (i + 1) * self.d]
    }

    fn add(&self, dst: &mut [f32], name: &str, i: usize) {
        for (a, b) in dst.iter_mut().zip(self.row(name, i)) {
            *a += *b;
        }
    }

    fn set(&self, dst: &mut [f32], name: &str, i: usize) {
        dst.copy_from_slice(self.row(name, i));
    }

    /// Tokens before target pixel 1: 2 + S + 1 + 2 S² + 1.
    pub fn context_len(&self, s: usize) -> usize {
        2 + s + 1 + 2 * s * s + 1
    }

    /// A pixel token: tok[colour] + row + col + edge[unknown] + kind[pixel] + seg (the order of `AnimGPT.tokens`).
    fn write_px(&self, dst: &mut [f32], colour: usize, r: usize, c: usize, seg: usize) {
        self.set(dst, "tok", colour);
        self.add(dst, "row", r);
        self.add(dst, "col", c);
        self.add(dst, "edge", self.edge_unknown);
        self.add(dst, "kind", self.tk.pixel);
        self.add(dst, "seg", seg);
    }

    /// Target token t: the colour of pixel t-1 (BOS for t = 0) at pixel t's row / column.
    pub fn write_target(&self, dst: &mut [f32], s: usize, colour: usize, t: usize) {
        self.write_px(dst, colour, t / s, t % s, 3);
    }

    /// The whole context of one row into dst (context_len(s) * d floats).
    pub fn write_context(&self, dst: &mut [f32], s: usize, c: &Cond, reference: &[u16], prev: &[u16]) {
        let d = self.d;
        let mut o = 0;
        let mut next = || -> std::ops::Range<usize> {
            let r = o..o + d;
            o += d;
            r
        };
        let r = next();
        self.set(&mut dst[r.clone()], "null_cond", 0);
        self.add(&mut dst[r], "kind", self.tk.text);
        let r = next();
        let si = self.sizes.iter().position(|&x| x == s).expect("size checked by the caller");
        self.set(&mut dst[r.clone()], "size_emb", si);
        self.add(&mut dst[r], "kind", self.tk.size);
        for row in 0..s {
            let r = next();
            self.set(&mut dst[r.clone()], "mask_proj.bias", 0);
            self.add(&mut dst[r.clone()], "row", row);
            self.add(&mut dst[r], "kind", self.tk.row_unknown);
        }
        let r = next();
        let x = &mut dst[r];
        self.set(x, "kind_emb", c.kind);
        self.add(x, "refk_emb", c.refk);
        self.add(x, "act_emb", c.action);
        self.add(x, "az_emb", c.az);
        self.add(x, "el_emb", c.el);
        self.add(x, "frame_emb", c.frame.min(self.max_frames - 1));
        self.add(x, "nfr_emb", c.nfr.min(self.max_frames));
        self.add(x, "seg", 0);
        if self.idx.contains_key("ol_emb") {
            self.add(x, "ol_emb", self.outline_of(s, reference));
        }
        for (canvas, seg) in [(reference, 1usize), (prev, 2)] {
            for p in 0..s * s {
                let r = next();
                self.write_px(&mut dst[r], canvas[p] as usize, p / s, p % s, seg);
            }
        }
        let r = next();
        self.write_px(&mut dst[r], self.bos, 0, 0, 3);
        debug_assert_eq!(o, self.context_len(s) * d);
    }

    /// True for a model with the outline condition (AnimGPT3, table `ol_emb`: 0 no outline, 1 outlined).
    pub fn has_outline_condition(&self) -> bool {
        self.idx.contains_key("ol_emb")
    }

    /// The outline id of a canvas of tokens, by the rule the training data was labelled with (`dark_outline` of
    /// export_anim3_onnx.py): the commonest colour of the silhouette's edge ring is dark (luma < 110) and at least
    /// 60 % of the ring is within 36 (RGB distance) of it; fewer than 20 filled pixels = no outline.
    pub fn outline_of(&self, s: usize, canvas: &[u16]) -> usize {
        let filled = |x: i32, y: i32| x >= 0 && y >= 0 && (x as usize) < s && (y as usize) < s && (canvas[y as usize * s + x as usize] as usize) < self.empty;
        let (mut n, mut ring) = (0usize, Vec::new());
        for y in 0..s as i32 {
            for x in 0..s as i32 {
                if !filled(x, y) {
                    continue;
                }
                n += 1;
                if !(filled(x - 1, y) && filled(x + 1, y) && filled(x, y - 1) && filled(x, y + 1)) {
                    ring.push(canvas[y as usize * s + x as usize] as usize);
                }
            }
        }
        if n < 20 || ring.is_empty() {
            return 0;
        }
        let mut count: HashMap<usize, usize> = HashMap::new();
        for &t in &ring {
            *count.entry(t).or_default() += 1;
        }
        // the commonest token, the smallest one on a tie (numpy's bincount().argmax())
        let top = count.iter().map(|(&t, &c)| (c, std::cmp::Reverse(t))).max().map(|(_, t)| t.0).unwrap_or(0);
        let rgb = |t: usize| self.palette[t].map(f64::from);
        let oc = rgb(top);
        if 0.3 * oc[0] + 0.59 * oc[1] + 0.11 * oc[2] >= 110.0 {
            return 0;
        }
        let near = ring.iter().filter(|&&t| rgb(t).iter().zip(&oc).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt() < 36.0).count();
        usize::from(near as f64 >= 0.6 * ring.len() as f64)
    }

    pub fn kind_index(&self, kind: Kind) -> Result<usize> {
        self.kinds.iter().position(|k| k == kind.label()).ok_or_else(|| anyhow!("the model has no kind {}", kind.label()))
    }
    pub fn action_index(&self, action: &str) -> Result<usize> {
        self.actions.iter().position(|a| a == action).ok_or_else(|| anyhow!("the model has no action {action}"))
    }
    /// The azimuths the model knows, in its order (0, 45 ... 315).
    pub fn all_azimuths(&self) -> &[u16] {
        &self.azimuths
    }
    pub fn az_index(&self, az: u16) -> Result<usize> {
        self.azimuths.iter().position(|&a| a == az).ok_or_else(|| anyhow!("the model has no azimuth {az}"))
    }

    /// Native frame count of an action for a kind (the dataset's), None if the model never saw it.
    pub fn n_frames(&self, kind: Kind, action: &str) -> Option<usize> {
        self.frames.get(kind.label()).and_then(|m| m.get(action)).copied()
    }

    /// Condition of a view (frame 0 of 1, elevation 0) or of frame `frame` of an action.
    pub fn cond(&self, kind: Kind, refk: usize, action: &str, az: u16, frame: usize) -> Result<Cond> {
        let nfr = if action == "view" { 1 } else { self.n_frames(kind, action).ok_or_else(|| anyhow!("{} has no action {action}", kind.label()))? };
        Ok(Cond { kind: self.kind_index(kind)?, refk, action: self.action_index(action)?, az: self.az_index(az)?, el: self.el0, frame, nfr })
    }
}

/// The per-clip random seed (the same formula as `row_seed` in export_anim_onnx.py): a view at an azimuth, or all
/// frames of an action at an azimuth, draw from one mulberry32 stream.
pub fn row_seed(base: u32, kind: usize, action: usize, az: usize, size: usize) -> u32 {
    (base as u64)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(action as u64 * 40_503)
        .wrapping_add(az as u64 * 9_973)
        .wrapping_add(kind as u64 * 7_919)
        .wrapping_add(size as u64 * 31) as u32
}

/// softmax(logits / T) over `allowed` (ascending token ids), inverse CDF at u; `cum` is scratch space.
/// Mirrors `draw` of export_anim_onnx.py (float64, sequential sums).
pub fn draw(logits: &[f32], allowed: &[usize], inv_t: f64, u: f64, cum: &mut Vec<f64>) -> usize {
    let mut m = f64::NEG_INFINITY;
    for &j in allowed {
        m = m.max(logits[j] as f64 * inv_t);
    }
    cum.clear();
    let mut acc = 0.0;
    for &j in allowed {
        acc += (logits[j] as f64 * inv_t - m).exp();
        cum.push(acc);
    }
    let target = u * acc;
    let i = cum.partition_point(|&c| c <= target);
    allowed[i.min(allowed.len() - 1)]
}

/// -log2 p(token) under softmax(logits) (all K + 1 classes).
pub fn bits_of(logits: &[f32], token: usize) -> f64 {
    let m = logits.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x as f64));
    let s: f64 = logits.iter().map(|&x| (x as f64 - m).exp()).sum();
    -((logits[token] as f64 - m) - s.ln()) / std::f64::consts::LN_2
}
