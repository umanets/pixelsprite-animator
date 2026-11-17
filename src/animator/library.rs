//! Precomputed animations from the 3D farm (`pilot3d/farm_out_*/<name>/`):
//!
//! * `job.json`        {"kind": "human" | "quadruped" | "bird", "caption": ...}
//! * `pix/meta.json`   {"actions": {"walk": 8, ...}, ...}
//! * `pix/sprites.npz` numpy zip: `palette` (P,3) uint8 and int16 palette-index images (-1 = transparent), keys
//!   `{S}__views__e{elev:02}_a{az:03}` and `{S}__anim__{action}__a{az:03}_f{frame:02}` for S in 16 / 24 / 32.
//!
//! A farm run may still be writing new folders: [`scan`] / [`Scanner`] list only folders whose `pix/sprites.npz`
//! exists; a half-written npz fails in [`LibraryData::load`] with an error the UI shows.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

use super::{ActionSpec, AnimRequest, Animation, Animator, Kind, resample, standard_actions};
use crate::sprite::{Palette, Sprite};

#[derive(Debug, Clone)]
pub struct LibraryEntry {
    pub name: String,
    pub dir: PathBuf,
    pub kind: Option<Kind>,
    pub caption: String,
    pub actions: Vec<ActionSpec>,
    /// Which farm folder it came from (farm_out_trial, farm_out_test, ...).
    pub root: String,
}

#[derive(Deserialize)]
struct Job {
    kind: Option<String>,
    caption: Option<String>,
}

#[derive(Deserialize)]
struct Meta {
    #[serde(default)]
    actions: serde_json::Map<String, serde_json::Value>,
}

/// Finished farm entries under the given roots, sorted by root then name (with their actions).
pub fn scan(roots: &[PathBuf]) -> Vec<LibraryEntry> {
    let mut out = Vec::new();
    for root in roots {
        let root_name = root_label(root);
        for dir in finished_dirs(root) {
            if let Some(e) = read_entry(dir, &root_name, true) {
                out.push(e);
            }
        }
    }
    out
}

fn root_label(root: &Path) -> String {
    root.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
}

/// Folders of a root that have `pix/sprites.npz`, sorted.
fn finished_dirs(root: &Path) -> Vec<PathBuf> {
    let Ok(rd) = fs::read_dir(root) else { return Vec::new() };
    let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.join("pix").join("sprites.npz").is_file()).collect();
    dirs.sort();
    dirs
}

/// One farm folder: `job.json` (kind, caption) and, with `with_actions`, `pix/meta.json`. A folder the farm is
/// still writing simply has no kind / caption / actions yet: a missing or unparsable `job.json` is read as "not
/// written yet" and gives an entry with an empty caption and no kind, never an error. `None` only when `dir` has no
/// final path component (so there is no name to show).
pub fn read_entry(dir: PathBuf, root_name: &str, with_actions: bool) -> Option<LibraryEntry> {
    let job: Option<Job> = fs::read_to_string(dir.join("job.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
    let kind = job.as_ref().and_then(|j| j.kind.as_deref()).and_then(Kind::parse);
    Some(LibraryEntry {
        name: dir.file_name()?.to_string_lossy().to_string(),
        caption: job.and_then(|j| j.caption).unwrap_or_default(),
        kind,
        actions: if with_actions { read_actions(&dir, kind) } else { Vec::new() },
        dir,
        root: root_name.to_string(),
    })
}

/// The actions of `pix/meta.json`, in the farm's order (idle, walk, run, ...). A missing or unparsable file gives no
/// actions, and an entry whose frame count is not a whole number is skipped rather than shown with 0 frames.
pub fn read_actions(dir: &Path, kind: Option<Kind>) -> Vec<ActionSpec> {
    let meta: Option<Meta> = fs::read_to_string(dir.join("pix").join("meta.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
    let std_actions = kind.map(standard_actions).unwrap_or_default();
    let looping = |a: &str| std_actions.iter().find(|s| s.name == a).map(|s| s.looping).unwrap_or(true);
    let mut actions: Vec<ActionSpec> = meta
        .map(|m| m.actions.iter().filter_map(|(k, v)| Some(ActionSpec { name: k.clone(), frames: v.as_u64()? as usize, looping: looping(k) })).collect())
        .unwrap_or_default();
    // whatever order the JSON map came back in
    actions.sort_by_cached_key(|s| std_actions.iter().position(|x| x.name == s.name).unwrap_or(usize::MAX));
    actions
}

pub enum ScanMsg {
    /// More entries, in order (root by root, sorted by name).
    Batch(Vec<LibraryEntry>),
    Done { entries: usize, secs: f64 },
}

/// The library scan on a background thread: `pilot3d/farm_out_full` alone has over 10 000 folders and grows by
/// thousands an hour, so listing it takes seconds (minutes on a cold, busy disk). Entries come in batches without
/// their actions; [`LibraryData::load`] reads those. Dropping the scanner stops the thread.
pub struct Scanner {
    pub rx: std::sync::mpsc::Receiver<ScanMsg>,
    stop: Arc<std::sync::atomic::AtomicBool>,
}

impl Scanner {
    pub fn start(roots: Vec<PathBuf>, wake: Box<dyn Fn() + Send>) -> Self {
        use std::sync::atomic::Ordering;
        let (tx, rx) = std::sync::mpsc::channel();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let halt = stop.clone();
        std::thread::Builder::new()
            .name("library-scan".into())
            .spawn(move || {
                let t = std::time::Instant::now();
                let mut n = 0;
                for root in &roots {
                    let root_name = root_label(root);
                    let mut batch = Vec::new();
                    let mut last = std::time::Instant::now();
                    for dir in finished_dirs(root) {
                        if halt.load(Ordering::Relaxed) {
                            return;
                        }
                        if let Some(e) = read_entry(dir, &root_name, false) {
                            batch.push(e);
                        }
                        if batch.len() >= 2000 || (last.elapsed().as_millis() > 250 && !batch.is_empty()) {
                            n += batch.len();
                            if tx.send(ScanMsg::Batch(std::mem::take(&mut batch))).is_err() {
                                return;
                            }
                            wake();
                            last = std::time::Instant::now();
                        }
                    }
                    n += batch.len();
                    if !batch.is_empty() && tx.send(ScanMsg::Batch(batch)).is_err() {
                        return;
                    }
                }
                let _ = tx.send(ScanMsg::Done { entries: n, secs: t.elapsed().as_secs_f64() });
                wake();
            })
            .expect("spawning the library scan");
        Scanner { rx, stop }
    }
}

impl Drop for Scanner {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// One `.npy` array: int16 or uint8 data with its shape.
#[derive(Debug, Clone)]
pub enum Npy {
    I16(Vec<usize>, Vec<i16>),
    U8(Vec<usize>, Vec<u8>),
}

/// Parses a version 1 / 2 / 3 `.npy` file holding `<i2` or `|u1` C-ordered data.
pub fn parse_npy(bytes: &[u8]) -> Result<Npy> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not an .npy file");
    }
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen])?;
    let data = &bytes[start + hlen..];
    let descr = field(header, "descr").ok_or_else(|| anyhow!("npy: no descr"))?;
    if header.contains("'fortran_order': True") {
        bail!("npy: fortran order not supported");
    }
    let shape_s = header.split("'shape':").nth(1).and_then(|s| s.split(')').next()).ok_or_else(|| anyhow!("npy: no shape"))?;
    let shape: Vec<usize> = shape_s.trim_start_matches([' ', '(']).split(',').filter_map(|t| t.trim().parse().ok()).collect();
    let n: usize = shape.iter().product();
    match descr.as_str() {
        "<i2" => {
            if data.len() < 2 * n {
                bail!("npy: short data");
            }
            Ok(Npy::I16(shape, data[..2 * n].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()))
        }
        "|u1" | "<u1" => {
            if data.len() < n {
                bail!("npy: short data");
            }
            Ok(Npy::U8(shape, data[..n].to_vec()))
        }
        d => bail!("npy: dtype {d} not supported"),
    }
}

fn field(header: &str, key: &str) -> Option<String> {
    let after = header.split(&format!("'{key}':")).nth(1)?;
    let q = after.find('\'')?;
    let rest = &after[q + 1..];
    Some(rest[..rest.find('\'')?].to_string())
}

/// All arrays of an `.npz` file, by name without the `.npy` suffix.
pub fn read_npz(path: &Path) -> Result<HashMap<String, Npy>> {
    let f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(f).with_context(|| format!("reading {}", path.display()))?;
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut e = zip.by_index(i)?;
        let name = e.name().trim_end_matches(".npy").to_string();
        let mut buf = Vec::with_capacity(e.size() as usize);
        e.read_to_end(&mut buf)?;
        out.insert(name.clone(), parse_npy(&buf).with_context(|| format!("{name} in {}", path.display()))?);
    }
    Ok(out)
}

/// One small thumbnail of an entry for the Library list: the front view (32 px, else the first size there is)
/// read straight out of the npz zip — only two members, not the whole file. None = it could not be read.
pub fn read_thumb(dir: &Path) -> Option<Sprite> {
    let f = fs::File::open(dir.join("pix").join("sprites.npz")).ok()?;
    let mut zip = zip::ZipArchive::new(f).ok()?;
    let names: Vec<String> = zip.file_names().map(str::to_string).collect();
    let pick = |size: usize| {
        names.iter().find(|n| n.as_str() == format!("{size}__views__e00_a000.npy")).cloned().or_else(|| {
            names.iter().find(|n| n.starts_with(&format!("{size}__views__"))).cloned()
        })
    };
    let key = pick(32).or_else(|| pick(16)).or_else(|| pick(24))?;
    let mut buf = Vec::new();
    zip.by_name(&key).ok()?.read_to_end(&mut buf).ok()?;
    let pal = {
        let mut b = Vec::new();
        zip.by_name("palette.npy").ok()?.read_to_end(&mut b).ok()?;
        match parse_npy(&b).ok()? {
            Npy::U8(_, d) => d.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            _ => return None,
        }
    };
    match parse_npy(&buf).ok()? {
        Npy::I16(shape, d) if shape.len() == 2 && shape[0] == shape[1] => Some(Sprite::new(shape[0], d, Arc::new(pal))),
        _ => None,
    }
}

/// A loaded farm entry: palette + every image.
pub struct LibraryData {
    pub entry: LibraryEntry,
    pub palette: Palette,
    images: HashMap<String, (usize, Vec<i16>)>,
}

impl LibraryData {
    /// Reads `pix/sprites.npz` (and `pix/meta.json` when the entry came from the background scan without actions).
    pub fn load(entry: &LibraryEntry) -> Result<Self> {
        let mut entry = entry.clone();
        if entry.actions.is_empty() {
            entry.actions = read_actions(&entry.dir, entry.kind);
        }
        let arrays = read_npz(&entry.dir.join("pix").join("sprites.npz"))?;
        let palette = match arrays.get("palette") {
            Some(Npy::U8(shape, d)) if shape.len() == 2 && shape[1] == 3 => d.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect::<Vec<_>>(),
            _ => bail!("{}: no (P,3) uint8 palette", entry.name),
        };
        let mut images = HashMap::new();
        for (k, v) in arrays {
            if let Npy::I16(shape, d) = v {
                if shape.len() == 2 && shape[0] == shape[1] {
                    images.insert(k, (shape[0], d));
                }
            }
        }
        Ok(LibraryData { entry, palette: Arc::new(palette), images })
    }

    fn sprite(&self, key: &str) -> Option<Sprite> {
        self.images.get(key).map(|(s, d)| Sprite::new(*s, d.clone(), self.palette.clone()))
    }

    /// The front view (elevation 0, azimuth 0) at size 16 / 24 / 32, or the first view there is.
    pub fn base_sprite(&self, size: usize) -> Option<Sprite> {
        self.sprite(&format!("{size}__views__e00_a000")).or_else(|| {
            let mut keys: Vec<&String> = self.images.keys().filter(|k| k.starts_with(&format!("{size}__views__"))).collect();
            keys.sort();
            keys.first().and_then(|k| self.sprite(k))
        })
    }

    /// The static view at eye level (elevation 0) and an azimuth.
    pub fn view(&self, size: usize, azimuth: u16) -> Option<Sprite> {
        self.sprite(&format!("{size}__views__e00_a{azimuth:03}"))
    }

    pub fn sizes(&self) -> Vec<usize> {
        [16, 24, 32].into_iter().filter(|s| self.images.keys().any(|k| k.starts_with(&format!("{s}__")))).collect()
    }

    /// Frames of an action at one azimuth, in order.
    pub fn frames(&self, size: usize, action: &str, azimuth: u16) -> Vec<Sprite> {
        let prefix = format!("{size}__anim__{action}__a{azimuth:03}_f");
        let mut keys: Vec<&String> = self.images.keys().filter(|k| k.starts_with(&prefix)).collect();
        keys.sort();
        keys.into_iter().filter_map(|k| self.sprite(k)).collect()
    }
}

/// Plays the precomputed animation of one farm entry. The sprite passed to `animate` is only used for its size:
/// the frames are the farm's own renders of that entry.
pub struct LibraryAnimator {
    pub data: Arc<LibraryData>,
}

/// Stands in for the library animator while no farm entry is loaded: it offers the standard actions and explains how to
/// pick an entry when asked to animate.
pub struct MissingLibrary;

impl Animator for MissingLibrary {
    fn name(&self) -> &str {
        "Library"
    }
    fn actions(&self, kind: Kind) -> Vec<ActionSpec> {
        standard_actions(kind)
    }
    fn animate(&self, _: &Sprite, _: &AnimRequest) -> Result<Animation> {
        bail!("The library animator plays precomputed 3D-farm animations: pick an entry in the Library list (left).")
    }
}

impl Animator for LibraryAnimator {
    fn name(&self) -> &str {
        "Library (3D farm renders)"
    }
    fn actions(&self, kind: Kind) -> Vec<ActionSpec> {
        if self.data.entry.kind == Some(kind) && !self.data.entry.actions.is_empty() {
            return self.data.entry.actions.clone();
        }
        Vec::new()
    }
    fn animate(&self, sprite: &Sprite, req: &AnimRequest) -> Result<Animation> {
        let size = if self.data.sizes().contains(&req.size) { req.size } else { sprite.size };
        let frames = self.data.frames(size, &req.action, req.azimuth);
        if frames.is_empty() {
            bail!("{}: no frames for '{}' at {}° (size {size})", self.data.entry.name, req.action, req.azimuth);
        }
        let native = frames.len();
        let frames = resample(&frames, req.n_frames);
        Ok(Animation {
            note: format!(
                "3D farm render {}: {} at {}°, {} of {native} frames",
                self.data.entry.name,
                req.action,
                req.azimuth,
                frames.len()
            ),
            frames,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_npy, read_actions, read_entry};
    use crate::animator::Kind;

    fn farm_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("animator_ui_lib_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("pix")).unwrap();
        dir
    }

    #[test]
    fn the_placeholder_library_explains_itself_instead_of_animating() {
        use super::MissingLibrary;
        use crate::animator::{AnimRequest, Animator, standard_actions};
        use crate::sprite::Sprite;
        let placeholder = MissingLibrary;
        assert_eq!(placeholder.name(), "Library");
        assert_eq!(placeholder.actions(Kind::Bird), standard_actions(Kind::Bird));
        let sprite = Sprite::new(16, vec![-1; 256], Default::default());
        let req = AnimRequest { kind: Kind::Human, action: "idle".into(), azimuth: 0, n_frames: 0, size: 16 };
        let err = placeholder.animate(&sprite, &req).unwrap_err();
        assert!(err.to_string().contains("pick an entry in the Library list"), "{err}");
    }

    #[test]
    fn an_entry_without_job_json_is_not_written_yet() {
        let dir = farm_dir("nojob");
        let e = read_entry(dir.clone(), "farm_out_test", true).unwrap();
        assert!(e.caption.is_empty() && e.kind.is_none() && e.actions.is_empty());
        assert_eq!(e.root, "farm_out_test");
        std::fs::write(dir.join("job.json"), "{ broken").unwrap();
        assert!(read_entry(dir.clone(), "r", false).unwrap().caption.is_empty());
        std::fs::write(dir.join("job.json"), r#"{"caption": "a knight", "kind": "human"}"#).unwrap();
        let e = read_entry(dir.clone(), "r", false).unwrap();
        assert_eq!((e.caption.as_str(), e.kind), ("a knight", Some(Kind::Human)));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn actions_follow_the_farm_order_and_skip_malformed_counts() {
        let dir = farm_dir("actions");
        std::fs::write(dir.join("pix").join("meta.json"), r#"{"actions": {"walk": 8, "idle": 4, "bogus": "many", "spin_kick": 8}}"#).unwrap();
        let names: Vec<(String, usize)> = read_actions(&dir, Some(Kind::Human)).into_iter().map(|a| (a.name, a.frames)).collect();
        assert_eq!(names, [("idle".to_string(), 4), ("walk".to_string(), 8), ("spin_kick".to_string(), 8)]);
        std::fs::write(dir.join("pix").join("meta.json"), "nope").unwrap();
        assert!(read_actions(&dir, Some(Kind::Human)).is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn npy_int16() {
        let header = "{'descr': '<i2', 'fortran_order': False, 'shape': (2, 2), }";
        let mut h = header.to_string();
        while (10 + h.len() + 1) % 64 != 0 {
            h.push(' ');
        }
        h.push('\n');
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        b.extend_from_slice(&(h.len() as u16).to_le_bytes());
        b.extend_from_slice(h.as_bytes());
        for v in [-1i16, 3, 7, -1] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        match parse_npy(&b).unwrap() {
            super::Npy::I16(shape, d) => {
                assert_eq!(shape, vec![2, 2]);
                assert_eq!(d, vec![-1, 3, 7, -1]);
            }
            _ => panic!(),
        }
    }
}
