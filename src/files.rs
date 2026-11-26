//! Model files: copies the released v2 graphs next to a float32 `decoder.data` in the app's own cache.
//!
//! Layout (as in `release/tools/export_onnx.py`): `decoder.onnx` and `decoder_static.onnx` keep every large weight
//! as external data at a float32 offset in a file named `decoder.data`; `decoder.f16` is that file as float16, in the
//! same order. The web demo expands the float16 bytes in memory and mounts them as `decoder.data`; here they are
//! expanded once to `<app>/models/v3/decoder.data` and the graphs are loaded from files in that folder.
//!
//! The built-in models are pixelsprite canvas v3 and AnimGPT v3. Their cache folders (`v3`, `v3_download`,
//! `animator-v3`) are new, so a cache left by v0.1.0 (the v2 models in `v2`, `v2_download`, `animator`) is never
//! taken for them: files already in a download folder are not fetched again.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use sha2::{Digest, Sha256};

use crate::config::{Config, ModelRef, app_root, models_dir, slug};
use crate::fetch;

pub const HF_REPO: &str = "ihor-umanets/pixelsprite-canvas-v3";
pub const HF_REVISION: &str = "4f6a1ab61ee248b4ee23f612e875ffcc66f4bc94";
pub const ENCODER_REPO: &str = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
/// The commit the web demo pins (docs/demo/text.js, ENCODER.revision).
pub const ENCODER_REVISION: &str = "2c4055b12046f11709e9df2c122e59ffbdc2f900";
/// The animation model (AnimGPT v3): `onnx/*` goes to [`builtin_animator_dir`].
pub const ANIM_REPO: &str = "ihor-umanets/pixelsprite-animgpt-v3";
pub const ANIM_REVISION: &str = "e5a4ad7c9d5a84efa6732cb8765f244b4584cfc7";
pub const ANIM_FILES: [&str; 6] = ["prefill.onnx", "step.onnx", "step_grow.onnx", "anim.data", "embeddings.json", "embeddings.bin"];
/// The web demo's caption-mixing lexicon and example captions (github.com/umanets/pixelsprite, `docs/demo`).
pub const DEMO_BASE: &str = "https://raw.githubusercontent.com/umanets/pixelsprite/9be92c73597f53d85d7f6f5019266c4f0d14bd50/docs/demo";

const GENERATOR_FILES: [&str; 6] =
    ["onnx/decoder.onnx", "onnx/decoder_static.onnx", "onnx/decoder.f16", "onnx/embeddings.json", "onnx/embeddings.bin", "palette.json"];

/// Where a model of `models` in `animator_ui.json` comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelSource {
    /// A folder laid out as the repo.
    Local(PathBuf),
    Hf { repo: String, revision: String },
}

impl ModelSource {
    /// `dir` wins over `repo`; without `repo` the built-in repo is meant; without `revision` the built-in repo gets
    /// its pinned commit and any other repo `main`.
    pub fn resolve(m: &ModelRef, default_repo: &str, default_revision: &str) -> Result<Self> {
        if let Some(d) = &m.dir {
            return Ok(ModelSource::Local(if d.is_absolute() { d.clone() } else { app_root().join(d) }));
        }
        let repo = match &m.repo {
            Some(r) => clean_repo(r)?,
            None => default_repo.to_owned(),
        };
        let revision = match &m.revision {
            Some(r) => r.trim().to_owned(),
            None if repo == default_repo => default_revision.to_owned(),
            None => "main".to_owned(),
        };
        if revision.is_empty() || !revision.chars().all(|c| c.is_ascii_alphanumeric() || "._-/".contains(c)) || revision.contains("..") {
            bail!("revision {revision:?} is not a commit, tag or branch name");
        }
        Ok(ModelSource::Hf { repo, revision })
    }

    pub fn describe(&self) -> String {
        match self {
            ModelSource::Local(d) => d.display().to_string(),
            ModelSource::Hf { repo, revision } => format!("{repo} @ {}", &revision[..revision.len().min(12)]),
        }
    }

    /// Name part of this source's own cache folder: two different sources never share files (two models of one
    /// architecture have files of the same size, so sizes cannot tell them apart).
    fn tag(&self) -> String {
        let (name, key) = match self {
            ModelSource::Local(d) => (d.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), format!("dir:{}", d.display())),
            ModelSource::Hf { repo, revision } => (repo.clone(), format!("hf:{repo}@{revision}")),
        };
        let hash: String = Sha256::digest(key.as_bytes()).iter().take(4).map(|b| format!("{b:02x}")).collect();
        format!("{}-{hash}", slug(&name))
    }

    fn is(&self, repo: &str, revision: &str) -> bool {
        matches!(self, ModelSource::Hf { repo: r, revision: v } if r == repo && v == revision)
    }
}

/// `owner/name` out of `owner/name` or `https://huggingface.co/owner/name`.
fn clean_repo(text: &str) -> Result<String> {
    let t = text.trim().trim_end_matches('/');
    let t = t.strip_prefix("https://huggingface.co/").or_else(|| t.strip_prefix("hf://")).unwrap_or(t);
    let parts: Vec<&str> = t.split('/').collect();
    let ok = |p: &str| !p.is_empty() && !p.starts_with('.') && p.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c));
    if parts.len() != 2 || !parts.iter().all(|p| ok(p)) {
        bail!("{text:?} is not a Hugging Face model repo (expected owner/name or https://huggingface.co/owner/name)");
    }
    Ok(t.to_owned())
}

/// The generator of `models.generator`; None = not configured (the built-in model, a development checkout's
/// `model_src` first).
pub fn generator_source(cfg: &Config) -> Result<Option<ModelSource>> {
    let m = cfg.models.as_ref().and_then(|m| m.generator.as_ref());
    m.map(|m| ModelSource::resolve(m, HF_REPO, HF_REVISION).context("models.generator in animator_ui.json")).transpose()
}

/// The animator of `models.animator`; None = not configured (the built-in model).
pub fn animator_source(cfg: &Config) -> Result<Option<ModelSource>> {
    let m = cfg.models.as_ref().and_then(|m| m.animator.as_ref());
    m.map(|m| ModelSource::resolve(m, ANIM_REPO, ANIM_REVISION).context("models.animator in animator_ui.json")).transpose()
}

/// `<app>/models/animator-v3`: the cache of the built-in animation model.
pub fn builtin_animator_dir() -> PathBuf {
    models_dir().join("animator-v3")
}

/// The folder the animation model is loaded from: [`builtin_animator_dir`] for the built-in model, a folder of its
/// own for another repo, the folder itself (or its `onnx` subfolder) for a local one. A broken entry gives a folder
/// that does not exist, so the built-in model is never used silently instead (see [`prepare_animator`]).
pub fn animator_dir(cfg: &Config) -> PathBuf {
    match animator_source(cfg) {
        Ok(None) => builtin_animator_dir(),
        Ok(Some(s)) if s.is(ANIM_REPO, ANIM_REVISION) => builtin_animator_dir(),
        Ok(Some(ModelSource::Local(d))) => {
            if d.join("onnx").join("step.onnx").is_file() { d.join("onnx") } else { d }
        }
        Ok(Some(s)) => models_dir().join(format!("animator-{}", s.tag())),
        Err(_) => models_dir().join("animator-invalid"),
    }
}

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
    let mut notes = Vec::new();
    let has_weights = |d: &Path| d.join("onnx").join("decoder.f16").is_file();
    let builtin = |say: &mut dyn FnMut(String)| -> Result<PathBuf> {
        let dl = models_dir().join("v3_download");
        fetch::hf_files(HF_REPO, HF_REVISION, &GENERATOR_FILES, &dl, say)?;
        Ok(dl)
    };
    let (dir, src) = match generator_source(cfg)? {
        None => {
            let src = cfg.model_src();
            (models_dir().join("v3"), if has_weights(&src) { src } else { builtin(say)? })
        }
        Some(s) if s.is(HF_REPO, HF_REVISION) => (models_dir().join("v3"), builtin(say)?),
        Some(s) => {
            let dir = models_dir().join(format!("v2-{}", s.tag()));
            notes.push(format!("sprite model from animator_ui.json: {}", s.describe()));
            let src = match &s {
                ModelSource::Local(d) => {
                    if !has_weights(d) {
                        bail!("models.generator.dir in animator_ui.json: {} has no onnx/decoder.f16", d.display());
                    }
                    d.clone()
                }
                ModelSource::Hf { repo, revision } => {
                    let dl = dir.join("download");
                    fetch::hf_files(repo, revision, &GENERATOR_FILES, &dl, say)?;
                    dl
                }
            };
            (dir, src)
        }
    };
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
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
    if fs::metadata(&data).map(|m| m.len()).unwrap_or(0) != want || newer(&f16, &data) {
        say(format!("Expanding decoder.f16 to float32 ({:.0} MB, once)…", want as f64 / 1e6));
        half_file_to_float(&f16, &data)?;
    }

    // the demo's lexicon and presets: from the demo folder of a development checkout, else downloaded once
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
    match animator_source(cfg) {
        Ok(Some(s)) if !s.is(ANIM_REPO, ANIM_REVISION) => notes.push(format!("animation model from animator_ui.json: {}", s.describe())),
        _ => {}
    }
    if let Err(e) = prepare_animator(cfg, say) {
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
    if fs::metadata(to).map(|b| b.len() == a.len()).unwrap_or(false) && !newer(from, to) {
        return Ok(());
    }
    fs::copy(from, to).with_context(|| format!("copying {} -> {}", from.display(), to.display()))?;
    Ok(())
}

/// True when `from` was written after `to` (a model exported again into the same folder has files of the same size).
fn newer(from: &Path, to: &Path) -> bool {
    match (fs::metadata(from).and_then(|m| m.modified()), fs::metadata(to).and_then(|m| m.modified())) {
        (Ok(a), Ok(b)) => a > b,
        _ => false,
    }
}

/// Makes sure the animation model is in [`animator_dir`] (the built-in one: about 157 MB, downloaded once). A local
/// folder of `models.animator.dir` is used in place and only checked.
pub fn prepare_animator(cfg: &Config, say: &mut dyn FnMut(String)) -> Result<()> {
    let dir = animator_dir(cfg);
    let (repo, revision) = match animator_source(cfg)? {
        Some(ModelSource::Local(_)) => {
            return match ANIM_FILES.iter().find(|f| !dir.join(f).is_file()) {
                Some(f) => bail!("models.animator.dir in animator_ui.json: {} has no {f}", dir.display()),
                None => Ok(()),
            };
        }
        Some(ModelSource::Hf { repo, revision }) => (repo, revision),
        None => (ANIM_REPO.to_owned(), ANIM_REVISION.to_owned()),
    };
    for name in ANIM_FILES {
        let to = dir.join(name);
        if fs::metadata(&to).map(|m| m.len() > 0).unwrap_or(false) {
            continue;
        }
        fetch::hf_file(&repo, &revision, &format!("onnx/{name}"), &to, say)?;
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
    use super::*;
    use crate::config::Models;

    fn cfg(generator: Option<ModelRef>, animator: Option<ModelRef>) -> Config {
        Config { models: Some(Models { generator, animator }), ..Default::default() }
    }

    fn hf(repo: &str, revision: Option<&str>) -> ModelRef {
        ModelRef { repo: Some(repo.into()), revision: revision.map(str::to_owned), dir: None }
    }

    #[test]
    fn no_models_section_means_the_builtin_pair() {
        let c = Config::default();
        assert_eq!(generator_source(&c).unwrap(), None);
        assert_eq!(animator_dir(&c), builtin_animator_dir());
        // never the folder v0.1.0 filled with the v2 animator
        assert!(builtin_animator_dir() != models_dir().join("animator"));
    }

    #[test]
    fn repo_forms_and_default_revisions() {
        let s = ModelSource::resolve(&hf("https://huggingface.co/me/gen/", None), HF_REPO, HF_REVISION).unwrap();
        assert_eq!(s, ModelSource::Hf { repo: "me/gen".into(), revision: "main".into() });
        let s = ModelSource::resolve(&ModelRef::default(), HF_REPO, HF_REVISION).unwrap();
        assert!(s.is(HF_REPO, HF_REVISION));
        let s = ModelSource::resolve(&ModelRef { revision: Some("v2".into()), ..Default::default() }, HF_REPO, HF_REVISION).unwrap();
        assert_eq!(s, ModelSource::Hf { repo: HF_REPO.into(), revision: "v2".into() });
    }

    #[test]
    fn bad_repo_or_revision_is_an_error() {
        for r in ["gen", "a/b/c", "a/b?x=1", "../b", "a b/c"] {
            assert!(ModelSource::resolve(&hf(r, None), HF_REPO, HF_REVISION).is_err(), "{r}");
        }
        assert!(ModelSource::resolve(&hf("a/b", Some("x y")), HF_REPO, HF_REVISION).is_err());
        assert!(ModelSource::resolve(&hf("a/b", Some("../../x")), HF_REPO, HF_REVISION).is_err());
    }

    #[test]
    fn every_source_has_its_own_cache_folder() {
        let a = animator_dir(&cfg(None, Some(hf("me/anim", Some("aaa")))));
        let b = animator_dir(&cfg(None, Some(hf("me/anim", Some("bbb")))));
        let builtin = builtin_animator_dir();
        assert!(a != b && a != builtin && b != builtin);
        assert!(a.starts_with(models_dir()));
        // the built-in repo at its pinned commit keeps the folder that is already filled
        assert_eq!(animator_dir(&cfg(None, Some(hf(ANIM_REPO, None)))), builtin);
        // a broken entry never falls back to the built-in model
        assert!(animator_dir(&cfg(None, Some(hf("nonsense", None)))) != builtin);
    }

    #[test]
    fn local_dir_wins_and_is_used_in_place() {
        let d = std::env::temp_dir().join(format!("animator_ui_anim_{}", std::process::id()));
        fs::create_dir_all(d.join("onnx")).unwrap();
        let c = cfg(None, Some(ModelRef { repo: Some("me/anim".into()), revision: None, dir: Some(d.clone()) }));
        assert_eq!(animator_source(&c).unwrap(), Some(ModelSource::Local(d.clone())));
        assert_eq!(animator_dir(&c), d);
        fs::write(d.join("onnx").join("step.onnx"), b"x").unwrap();
        assert_eq!(animator_dir(&c), d.join("onnx"));
        assert!(prepare_animator(&c, &mut |_| {}).is_err()); // the other files are missing: reported, nothing downloaded
    }

    #[test]
    fn halves() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
        assert!(half_to_f32(0x7e00).is_nan());
        assert_eq!(half_to_f32(0x7c00), f32::INFINITY);
    }
}
