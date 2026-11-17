//! Caption -> 384-number vector: paraphrase-multilingual-MiniLM-L12-v2 (Xenova's int8 ONNX export, the revision the
//! web demo pins), mean pooling over the tokens + L2 normalisation, as text.js and sentence-transformers.
//! Always on the CPU: the int8 graph (MatMulInteger, DynamicQuantizeLinear) has no GPU kernels worth the copies.

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::ep::{Ep, SessionOpts, build_session};
use crate::files::ModelFiles;

pub const MAX_TOKENS: usize = 128;

pub struct TextEncoder {
    tokenizer: Tokenizer,
    session: Session,
    token_type: bool,
    out_name: String,
    cache: HashMap<String, Vec<f32>>,
}

impl TextEncoder {
    pub fn load(files: &ModelFiles) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(&files.encoder_tokenizer).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        let opts = SessionOpts { cpu_threads: 4, ..Default::default() };
        let session = build_session(Ep::Cpu, &files.encoder_model, opts)?;
        let token_type = session.inputs().iter().any(|i| i.name() == "token_type_ids");
        let out_name = session.outputs().first().map(|o| o.name().to_string()).ok_or_else(|| anyhow!("encoder has no outputs"))?;
        Ok(TextEncoder { tokenizer, session, token_type, out_name, cache: HashMap::new() })
    }

    /// Token ids of a text (with the special tokens), as the encoder sees them.
    pub fn ids(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text, true).map_err(|e| anyhow!("tokenizer: {e}"))?.get_ids().to_vec())
    }

    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>> {
        if let Some(v) = self.cache.get(text) {
            return Ok(v.clone());
        }
        let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        if ids.len() > MAX_TOKENS {
            let last = *ids.last().unwrap();
            ids.truncate(MAX_TOKENS - 1);
            ids.push(last);
        }
        let n = ids.len();
        if n == 0 {
            bail!("empty tokenisation");
        }
        let er = |e: ort::Error| anyhow!("{e}");
        let mut feeds = ort::inputs![
            "input_ids" => Tensor::from_array(([1usize, n], ids)).map_err(er)?,
            "attention_mask" => Tensor::from_array(([1usize, n], vec![1i64; n])).map_err(er)?,
        ];
        if self.token_type {
            feeds.push(("token_type_ids".into(), Tensor::from_array(([1usize, n], vec![0i64; n])).map_err(er)?.into()));
        }
        let out = self.session.run(feeds).map_err(er)?;
        let (shape, h) = out[self.out_name.as_str()].try_extract_tensor::<f32>().map_err(er)?;
        let dim = *shape.last().unwrap_or(&0) as usize;
        if dim == 0 || h.len() != n * dim {
            bail!("unexpected encoder output shape {shape:?}");
        }
        // as text.js: a Float32Array accumulator updated with float64 arithmetic, a float64 norm
        let mut v = vec![0f32; dim];
        for t in 0..n {
            for j in 0..dim {
                v[j] = (v[j] as f64 + h[t * dim + j] as f64 / n as f64) as f32;
            }
        }
        let norm = v.iter().fold(0f64, |a, &x| a + x as f64 * x as f64).sqrt();
        let norm = if norm > 0.0 { norm } else { 1.0 };
        for x in &mut v {
            *x = (*x as f64 / norm) as f32;
        }
        self.cache.insert(text.to_string(), v.clone());
        Ok(v)
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}
