//! Model files: copies the released v2 graphs next to a float32 `decoder.data` in the app's own cache.
//!
//! Layout (as in `release/tools/export_onnx.py`): `decoder.onnx` and `decoder_static.onnx` keep every large weight
//! as external data at a float32 offset in a file named `decoder.data`; `decoder.f16` is that file as float16, in the
//! same order. The web demo expands the float16 bytes in memory and mounts them as `decoder.data`; here they are
//! expanded once to `<app>/models/v2/decoder.data` and the graphs are loaded from files in that folder.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::config::{Config, models_dir};
use crate::fetch;

pub const HF_REPO: &str = "ihor-umanets/pixelsprite-canvas-v2";
/// The commit the web demo pins (docs/demo/app.js, MODEL_REVISION).
pub const HF_REVISION: &str = "0a29494e9fc020a26feb6f97ec2047fc207ce898";
pub const ENCODER_REPO: &str = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
/// The commit the web demo pins (docs/demo/text.js, ENCODER.revision).
pub const ENCODER_REVISION: &str = "2c4055b12046f11709e9df2c122e59ffbdc2f900";
/// The animation model (AnimGPT): `onnx/*` goes to `<app>/models/animator`.
pub const ANIM_REPO: &str = "ihor-umanets/pixelsprite-animgpt";
pub const ANIM_REVISION: &str = "3c540730cc96d86bafdfc91db2e38073ac587f66";
pub const ANIM_FILES: [&str; 6] = ["prefill.onnx", "step.onnx", "step_grow.onnx", "anim.data", "embeddings.json", "embeddings.bin"];
/// The web demo's caption-mixing lexicon and example captions (github.com/umanets/pixelsprite, `docs/demo`).
pub const DEMO_BASE: &str = "https://raw.githubusercontent.com/umanets/pixelsprite/9be92c73597f53d85d7f6f5019266c4f0d14bd50/docs/demo";

#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub dir: PathBuf,
    pub decoder: PathBuf,
    pub decoder_static: PathBuf,
    pub emb_json: PathBuf,
    pub emb_bin: PathBuf,
    pub palette: PathBuf,
    pub encoder_model: PathBuf,
    pub encoder_tokenizer: PathBuf,
    /// The web demo's caption-mixing lexicon (`lexicon.json` + `lexicon.f16`) and example captions with their
    /// precomputed vectors (`presets.json`), copied from the demo folder; optional (without the lexicon, mixing
    /// is off, as in the demo; without the presets, every caption goes through the encoder).
    pub lexicon_json: PathBuf,
    pub lexicon_f16: PathBuf,
    pub presets: PathBuf,
    /// Optional parts that could not be fetched (shown with the skipped providers); the app works without them.
    pub notes: Vec<String>,
}

/// Makes sure every file is in the cache; `say` gets one-line progress messages.
pub fn prepare(cfg: &Config, say: &mut dyn FnMut(String)) -> Result<ModelFiles> {
    let dir = models_dir().join("v2");
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let mut src = cfg.model_src();
    if !src.join("onnx").join("decoder.f16").is_file() {
        let dl = models_dir().join("v2_download");
        if !dl.join("onnx").join("decoder.f16").is_file() {
            fetch::hf_files(
                HF_REPO,
                HF_REVISION,
                &["onnx/decoder.onnx", "onnx/decoder_static.onnx", "onnx/decoder.f16", "onnx/embeddings.json",
                  "onnx/embeddings.bin", "palette.json"],
                &dl,
                say,
            )?;
        }
        src = dl;
    }
    let onnx = src.join("onnx");
    for (from, name) in [
        (onnx.join("decoder.onnx"), "decoder.onnx"),
        (onnx.join("decoder_static.onnx"), "decoder_static.onnx"),
        (onnx.join("embeddings.json"), "embeddings.json"),
        (onnx.join("embeddings.bin"), "embeddings.bin"),
        (src.join("palette.json"), "palette.json"),
    ] {
        copy_if_changed(&from, &dir.join(name))?;
    }
    let f16 = onnx.join("decoder.f16");
    let data = dir.join("decoder.data");
    let want = fs::metadata(&f16).with_context(|| format!("reading {}", f16.display()))?.len() * 2;
    if fs::metadata(&data).map(|m| m.len()).unwrap_or(0) != want {
        say(format!("Expanding decoder.f16 to float32 ({:.0} MB, once)…", want as f64 / 1e6));
        half_file_to_float(&f16, &data)?;
    }

    // the demo's lexicon and presets: from the demo folder of a development checkout, else downloaded once
    let mut notes = Vec::new();
    let demo = cfg.demo_src();
    for name in ["lexicon.json", "lexicon.f16", "presets.json"] {
        if demo.join(name).is_file() {
            copy_if_changed(&demo.join(name), &dir.join(name))?;
        } else if !dir.join(name).is_file() {
            if let Err(e) = fetch::download(&format!("{DEMO_BASE}/{name}"), &dir.join(name), name, say) {
                notes.push(format!("{name} is missing (caption mixing / examples are off): {e:#}"));
            }
        }
    }

    let enc = models_dir().join("encoder");
    let encoder_model = enc.join("onnx").join("model_quantized.onnx");
    let encoder_tokenizer = enc.join("tokenizer.json");
    if !encoder_model.is_file() || !encoder_tokenizer.is_file() {
        fetch::hf_files(ENCODER_REPO, ENCODER_REVISION, &["onnx/model_quantized.onnx", "tokenizer.json", "tokenizer_config.json"], &enc, say)?;
    }
    if let Err(e) = prepare_animator(say) {
        notes.push(format!("the animation model is missing (only the mock animator works): {e:#}"));
    }
    Ok(ModelFiles {
        decoder: dir.join("decoder.onnx"),
        decoder_static: dir.join("decoder_static.onnx"),
        emb_json: dir.join("embeddings.json"),
        emb_bin: dir.join("embeddings.bin"),
        palette: dir.join("palette.json"),
        lexicon_json: dir.join("lexicon.json"),
        lexicon_f16: dir.join("lexicon.f16"),
        presets: dir.join("presets.json"),
        notes,
        dir,
        encoder_model,
        encoder_tokenizer,
    })
}

fn copy_if_changed(from: &Path, to: &Path) -> Result<()> {
    let a = fs::metadata(from).with_context(|| format!("missing model file {}", from.display()))?;
    if fs::metadata(to).map(|b| b.len() == a.len()).unwrap_or(false) {
        return Ok(());
    }
    fs::copy(from, to).with_context(|| format!("copying {} -> {}", from.display(), to.display()))?;
    Ok(())
}

/// Makes sure the animation model is in `<app>/models/animator` (about 157 MB, downloaded once).
pub fn prepare_animator(say: &mut dyn FnMut(String)) -> Result<()> {
    let dir = models_dir().join("animator");
    for name in ANIM_FILES {
        let to = dir.join(name);
        if fs::metadata(&to).map(|m| m.len() > 0).unwrap_or(false) {
            continue;
        }
        fetch::download(&fetch::hf_url(ANIM_REPO, ANIM_REVISION, &format!("onnx/{name}")), &to, &format!("{ANIM_REPO} · {name}"), say)?;
    }
    Ok(())
}

/// IEEE half -> single, exactly as `halfToFloat` in engine.js (subnormals, infinities, NaN).
pub fn half_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 31) as i32;
    let f = (h & 1023) as f32;
    if e == 0 {
        s * f * 2f32.powi(-24)
    } else if e == 31 {
        if f != 0.0 { f32::NAN } else { s * f32::INFINITY }
    } else {
        s * (1.0 + f / 1024.0) * 2f32.powi(e - 15)
    }
}

fn half_file_to_float(src: &Path, dst: &Path) -> Result<()> {
    let table: Vec<f32> = (0..=u16::MAX).map(half_to_f32).collect();
    let tmp = dst.with_extension("data.tmp");
    {
        let mut r = BufReader::with_capacity(1 << 20, File::open(src)?);
        let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
        let mut buf = vec![0u8; 1 << 20];
        let mut out = Vec::with_capacity(2 << 20);
        loop {
            let n = read_full(&mut r, &mut buf)?;
            if n == 0 {
                break;
            }
            if n % 2 != 0 {
                bail!("{} has an odd length", src.display());
            }
            out.clear();
            for c in buf[..n].chunks_exact(2) {
                out.extend_from_slice(&table[u16::from_le_bytes([c[0], c[1]]) as usize].to_le_bytes());
            }
            w.write_all(&out)?;
        }
        w.flush()?;
    }
    fs::rename(&tmp, dst)?;
    Ok(())
}

fn read_full(r: &mut impl Read, buf: &mut [u8]) -> Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        let n = r.read(&mut buf[got..])?;
        if n == 0 {
            break;
        }
        got += n;
    }
    Ok(got)
}

#[cfg(test)]
mod tests {
    use super::half_to_f32;

    #[test]
    fn halves() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
        assert!(half_to_f32(0x7e00).is_nan());
        assert_eq!(half_to_f32(0x7c00), f32::INFINITY);
    }
}
