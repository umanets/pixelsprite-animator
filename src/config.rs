//! Where things live: the app folder, the model cache, the source model files and the 3D-farm library roots.
//!
//! Everything is resolved from the app root (the `animator_ui` folder, found by walking up from the executable
//! until a `Cargo.toml` next to `src/main.rs` appears; override with the `ANIMATOR_UI_HOME` environment variable).
//! An optional `animator_ui.json` in the app root overrides the defaults (see docs/DEVELOPMENT.md).

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// One model of `models` in `animator_ui.json`: a Hugging Face repo (`repo`, optional `revision`) or a local folder
/// (`dir`, which wins when both are given). See `files::ModelSource` for how it is resolved.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct ModelRef {
    /// `owner/name` or `https://huggingface.co/owner/name`. Default: the built-in repo of that model.
    pub repo: Option<String>,
    /// Commit, tag or branch. Default: the pinned commit for the built-in repo, `main` for any other repo.
    pub revision: Option<String>,
    /// Local folder laid out as the repo (`onnx/...`); a relative path is taken from the app folder.
    pub dir: Option<PathBuf>,
}

/// The pair of models the app works with (`models` in `animator_ui.json`); a missing entry = the built-in model.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct Models {
    /// Text -> sprite.
    pub generator: Option<ModelRef>,
    /// Sprite -> views / animation frames.
    pub animator: Option<ModelRef>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    /// The generator and the animator to use (Hugging Face repos or local folders). Default: the built-in pair.
    pub models: Option<Models>,
    /// Folder with the released v2 model (`onnx/decoder.onnx`, `onnx/decoder.f16`, ..., `palette.json`).
    /// Default: `<app root>/../release/hf`; if it does not exist the files are downloaded from Hugging Face.
    pub model_src: Option<PathBuf>,
    /// The web demo's folder (`lexicon.json`, `lexicon.f16`, `presets.json` are copied from it into `models/v2`).
    /// Default: `<app root>/../release/github/docs/demo`.
    pub demo_src: Option<PathBuf>,
    /// Folders scanned for precomputed 3D-farm animations (`<root>/<name>/pix/sprites.npz`). Default:
    /// `pilot3d/farm_out_trial`, `farm_out_test` and `farm_out_full` (the big one, ~11k entries and growing).
    pub library_roots: Option<Vec<PathBuf>>,
    /// Extra folders prepended to the DLL search path before ONNX Runtime loads its CUDA provider
    /// (for example a cuDNN 9 for CUDA 13 `bin` folder).
    pub cuda_dll_dirs: Option<Vec<PathBuf>>,
    /// "auto" (CUDA -> DirectML -> CPU), "cuda", "directml" or "cpu".
    pub ep: Option<String>,
    /// CUDA arena limit in MB (the GPU is shared with other work). Default 2048.
    pub cuda_mem_limit_mb: Option<usize>,
    /// CUDA arena limit of the animation model's sessions in MB (its own arena next to the sprite model's).
    /// Default [`ANIM_CUDA_MEM_LIMIT_MB`].
    pub anim_cuda_mem_limit_mb: Option<usize>,
    /// UI: build every animation from the first frame of walk at 0° (front) instead of the original sprite.
    /// Default false (experimental); written back by the app when the checkbox changes.
    pub ref_from_walk_front: Option<bool>,
}

/// Default CUDA arena of the animation model: 8 rows at 32 px keep a 2084-token context cache (0.8 GB) and two
/// target-cache sets (0.8 GB); see docs/DEVELOPMENT.md.
pub const ANIM_CUDA_MEM_LIMIT_MB: usize = 3072;

impl Config {
    /// The settings of `animator_ui.json`, or the defaults when it is absent; the caller may still be told when the
    /// file exists but cannot be read or parsed (see [`Config::load`] for the lenient variant).
    pub fn try_load() -> Result<Self, ConfigError> {
        Self::try_load_from(&app_root().join("animator_ui.json"))
    }

    /// As [`Config::try_load`] for an explicit file.
    pub fn try_load_from(path: &Path) -> Result<Self, ConfigError> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).map_err(|e| ConfigError::Parse(path.to_path_buf(), e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(ConfigError::Io(path.to_path_buf(), e)),
        }
    }

    /// [`Config::try_load`], but a broken file is reported on stderr and the defaults are used (the app must start).
    pub fn load() -> Self {
        Self::try_load().unwrap_or_else(|e| {
            eprintln!("animator_ui.json ignored: {e}");
            Self::default()
        })
    }

    pub fn model_src(&self) -> PathBuf {
        self.model_src.clone().unwrap_or_else(|| app_root().join("..").join("release").join("hf"))
    }

    pub fn demo_src(&self) -> PathBuf {
        self.demo_src.clone().unwrap_or_else(|| app_root().join("..").join("release").join("github").join("docs").join("demo"))
    }

    pub fn library_roots(&self) -> Vec<PathBuf> {
        self.library_roots.clone().unwrap_or_else(|| {
            let p = app_root().join("..").join("pilot3d");
            vec![p.join("farm_out_trial"), p.join("farm_out_test"), p.join("farm_out_full")]
        })
    }
}

/// Persists one UI setting in `animator_ui.json`, keeping every other key (also unknown ones) as it is. A file that
/// exists but is not valid JSON is left untouched and reported, so a typo never costs the user the other settings.
pub fn save_flag(key: &str, value: bool) -> Result<(), ConfigError> {
    save_flag_at(&app_root().join("animator_ui.json"), key, value)
}

/// [`save_flag`] for an explicit file.
pub fn save_flag_at(path: &Path, key: &str, value: bool) -> Result<(), ConfigError> {
    let mut root: serde_json::Value = match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).map_err(|e| ConfigError::Parse(path.to_path_buf(), e))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_json::Value::Object(serde_json::Map::new()),
        Err(e) => return Err(ConfigError::Io(path.to_path_buf(), e)),
    };
    let Some(object) = root.as_object_mut() else {
        return Err(ConfigError::NotAnObject(path.to_path_buf()));
    };
    object.insert(key.to_owned(), value.into());
    let text = serde_json::to_string_pretty(&root).expect("a JSON value always serialises");
    std::fs::write(path, text).map_err(|e| ConfigError::Io(path.to_path_buf(), e))
}

/// Why `animator_ui.json` could not be read or written.
#[derive(Debug)]
pub enum ConfigError {
    Io(PathBuf, std::io::Error),
    Parse(PathBuf, serde_json::Error),
    NotAnObject(PathBuf),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(p, e) => write!(f, "{}: {e}", p.display()),
            ConfigError::Parse(p, e) => write!(f, "{}: not valid settings JSON ({e})", p.display()),
            ConfigError::NotAnObject(p) => write!(f, "{}: expected a JSON object", p.display()),
        }
    }
}

impl std::error::Error for ConfigError {}

/// The `animator_ui` folder (see module docs).
pub fn app_root() -> PathBuf {
    if let Ok(p) = std::env::var("ANIMATOR_UI_HOME") {
        return PathBuf::from(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(Path::to_path_buf);
        while let Some(d) = dir {
            if d.join("Cargo.toml").is_file() && d.join("src").join("main.rs").is_file() {
                return d;
            }
            dir = d.parent().map(Path::to_path_buf);
        }
        if let Some(p) = exe.parent() {
            return p.to_path_buf();
        }
    }
    std::env::current_dir().unwrap_or_default()
}

/// `<app root>/models`: converted decoder weights, copies of the graphs, the text encoder.
pub fn models_dir() -> PathBuf {
    app_root().join("models")
}

/// `<app root>/exports`: sprite sheets and GIFs written by the UI.
pub fn exports_dir() -> PathBuf {
    app_root().join("exports")
}

/// Lower-case slug for file names (keeps Cyrillic letters).
pub fn slug(text: &str) -> String {
    let mut out = String::new();
    let mut last_us = true;
    for ch in text.trim().to_lowercase().chars() {
        if ch.is_alphanumeric() {
            out.push(ch);
            last_us = false;
        } else if !last_us {
            out.push('_');
            last_us = true;
        }
    }
    let s: String = out.trim_matches('_').chars().take(40).collect();
    if s.is_empty() { "no_text".into() } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("animator_ui_cfg_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("animator_ui.json")
    }

    #[test]
    fn missing_file_gives_defaults() {
        let path = temp_file("missing");
        assert!(Config::try_load_from(&path).is_ok());
    }

    #[test]
    fn models_pair_is_read() {
        let path = temp_file("models");
        std::fs::write(&path, r#"{"models": {"generator": {"repo": "a/b", "revision": "abc"}, "animator": {"dir": "D:/m/anim"}}}"#).unwrap();
        let m = Config::try_load_from(&path).unwrap().models.unwrap();
        assert_eq!(m.generator.unwrap(), ModelRef { repo: Some("a/b".into()), revision: Some("abc".into()), dir: None });
        assert_eq!(m.animator.unwrap().dir, Some(PathBuf::from("D:/m/anim")));
    }

    #[test]
    fn corrupt_file_is_reported_not_swallowed() {
        let path = temp_file("corrupt");
        std::fs::write(&path, "{ not json").unwrap();
        assert!(matches!(Config::try_load_from(&path), Err(ConfigError::Parse(..))));
    }

    #[test]
    fn save_flag_keeps_other_keys_and_unknown_ones() {
        let path = temp_file("keep");
        std::fs::write(&path, r#"{"unknown_key": 7, "ref_from_walk_front": false}"#).unwrap();
        save_flag_at(&path, "ref_from_walk_front", true).unwrap();
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["unknown_key"], 7);
        assert_eq!(v["ref_from_walk_front"], true);
    }

    #[test]
    fn save_flag_does_not_overwrite_a_corrupt_file() {
        let path = temp_file("nooverwrite");
        std::fs::write(&path, "{ not json").unwrap();
        assert!(save_flag_at(&path, "x", true).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{ not json");
    }

    #[test]
    fn save_flag_creates_the_file() {
        let path = temp_file("create");
        let _ = std::fs::remove_file(&path);
        save_flag_at(&path, "x", true).unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().contains("\"x\": true"));
    }
}
