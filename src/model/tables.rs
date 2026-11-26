use super::*;

#[derive(Debug, Deserialize)]
struct TableInfo {
    offset: usize,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct Kinds {
    pub pixel: usize,
    pub text: usize,
    pub size: usize,
    pub row_known: usize,
    pub row_unknown: usize,
}

#[derive(Debug, Deserialize)]
struct EmbInfo {
    tables: HashMap<String, TableInfo>,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    n_colors: usize,
    max_size: usize,
    sizes: Vec<usize>,
    empty: usize,
    bos: usize,
    kinds: Kinds,
    /// The generator was trained with ", with outline" / ", no outline" at the end of its captions.
    #[serde(default)]
    outline_words: bool,
}

/// Everything except the session: tables, sizes, palette. Builds the input vectors.
pub struct ModelCore {
    pub d: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub hd: usize,
    /// Number of colours K; the head has K + 1 outputs (the last = EMPTY).
    pub k: usize,
    pub empty: usize,
    pub bos: usize,
    pub sizes: Vec<usize>,
    pub max_size: usize,
    pub kinds: Kinds,
    pub text_dim: usize,
    /// The model understands ", with outline" / ", no outline" at the end of a caption (`outline_words` of
    /// embeddings.json): the UI then offers the Outline checkbox.
    pub outline_words: bool,
    pub palette: Palette,
    data: Vec<f32>,
    idx: HashMap<String, (usize, Vec<usize>)>,
}

impl ModelCore {
    pub fn load(files: &ModelFiles) -> Result<Self> {
        let info: EmbInfo = serde_json::from_str(&std::fs::read_to_string(&files.emb_json)?).context("embeddings.json")?;
        let raw = std::fs::read(&files.emb_bin).context("embeddings.bin")?;
        let data: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut idx = HashMap::new();
        for (name, t) in &info.tables {
            let n: usize = t.shape.iter().product();
            if t.offset % 4 != 0 || t.offset / 4 + n > data.len() {
                bail!("embeddings.bin: table {name} out of range");
            }
            idx.insert(name.clone(), (t.offset / 4, t.shape.clone()));
        }
        #[derive(Deserialize)]
        struct Pal {
            colors: Vec<[u8; 3]>,
        }
        let pal: Pal = serde_json::from_str(&std::fs::read_to_string(&files.palette)?).context("palette.json")?;
        if pal.colors.len() != info.n_colors {
            bail!("palette.json has {} colours, the model {}", pal.colors.len(), info.n_colors);
        }
        let text_dim = idx.get("cond_proj.weight").map(|(_, s)| s[1]).ok_or_else(|| anyhow!("no cond_proj.weight"))?;
        Ok(ModelCore {
            outline_words: info.outline_words,
            d: info.d_model,
            n_layers: info.n_layers,
            n_heads: info.n_heads,
            hd: info.d_model / info.n_heads,
            k: info.n_colors,
            empty: info.empty,
            bos: info.bos,
            sizes: info.sizes.clone(),
            max_size: info.max_size,
            kinds: info.kinds,
            text_dim,
            palette: Arc::new(pal.colors),
            data,
            idx,
        })
    }

    fn table(&self, name: &str) -> &[f32] {
        let (o, shape) = &self.idx[name];
        &self.data[*o..*o + shape.iter().product::<usize>()]
    }

    fn row(&self, name: &str, i: usize) -> &[f32] {
        &self.table(name)[i * self.d..(i + 1) * self.d]
    }

    fn add(&self, dst: &mut [f32], name: &str, i: usize) {
        for (a, b) in dst.iter_mut().zip(self.row(name, i)) {
            *a += *b;
        }
    }

    /// Caption vector (L2-normalised, `text_dim` long) -> condition vector; None -> the learned "no text" vector.
    pub fn condition(&self, text: Option<&[f32]>) -> Vec<f32> {
        let Some(text) = text else { return self.table("null_cond").to_vec() };
        let w = self.table("cond_proj.weight");
        let b = self.table("cond_proj.bias");
        let dd = self.text_dim;
        (0..self.d)
            .map(|o| {
                let mut s = b[o] as f64;
                for i in 0..dd {
                    s += w[o * dd + i] as f64 * text[i] as f64;
                }
                s as f32
            })
            .collect()
    }

    /// The 2 + S prefix vectors (text, size, one per silhouette row) into dst. `mask`: the silhouette on the S x S
    /// canvas (1 = inside), or None (rows "unknown"), as `writePrefix` in engine.js.
    pub fn write_prefix(&self, dst: &mut [f32], cond: &[f32], s: usize, mask: Option<&[u8]>) {
        let d = self.d;
        let k = self.kinds;
        dst[..d].copy_from_slice(cond);
        self.add(&mut dst[..d], "kind", k.text);
        let size_i = self.sizes.iter().position(|&x| x == s).expect("size");
        let o = d;
        dst[o..o + d].fill(0.0);
        self.add(&mut dst[o..o + d], "size_emb", size_i);
        self.add(&mut dst[o..o + d], "kind", k.size);
        let bias = self.table("mask_proj.bias").to_vec();
        let w = self.table("mask_proj.weight");
        let mm = self.max_size;
        for r in 0..s {
            let o = (2 + r) * d;
            dst[o..o + d].copy_from_slice(&bias);
            self.add(&mut dst[o..o + d], "row", r);
            self.add(&mut dst[o..o + d], "kind", if mask.is_some() { k.row_known } else { k.row_unknown });
            if let Some(m) = mask {
                for c in 0..s {
                    if m[r * s + c] != 0 {
                        for j in 0..d {
                            dst[o + j] += w[j * mm + c];
                        }
                    }
                }
            }
        }
    }


    /// One pixel token: previous pixel's token + row + column + edge flag (3 = unknown) of this pixel.
    pub fn write_pixel(&self, dst: &mut [f32], prev: usize, r: usize, c: usize, edge: usize) {
        dst.fill(0.0);
        self.add(dst, "tok", prev);
        self.add(dst, "row", r);
        self.add(dst, "col", c);
        self.add(dst, "edge", edge);
        self.add(dst, "kind", self.kinds.pixel);
    }
}
