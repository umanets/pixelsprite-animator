use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Result, bail};

use crate::animator::{ActionSpec, AnimRequest, Animation, Animator, Kind, standard_actions};
use crate::ep::{Ep, SessionOpts};
use crate::sprite::{Frame, Sprite};

use super::engine::Engine;
use super::reference::{Reference, anchor_action};
use super::settings::{AnimProgress, ClipKey, Ctl, OnnxSettings, Slot, Totals, Wanted};
use super::tables::AnimTables;

/// The animation model behind the [`Animator`] trait. Thread-safe: the UI's job thread draws while the UI reads the
/// cache and the status.
pub struct OnnxAnimator {
    pub dir: PathBuf,
    tables: Mutex<Option<Arc<AnimTables>>>,
    slot: Mutex<Slot>,
    /// Set by the UI thread without waiting for a running request; applied on the next request.
    wanted: Mutex<Wanted>,
    pub(super) cache: Mutex<HashMap<ClipKey, Arc<Vec<Vec<u16>>>>>,
    settings: Mutex<OnnxSettings>,
    status: Mutex<String>,
    pub totals: Mutex<Totals>,
}

impl Default for OnnxAnimator {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxAnimator {
    pub fn new() -> Self {
        Self::with_dir(crate::files::builtin_animator_dir())
    }

    pub fn with_dir(dir: PathBuf) -> Self {
        OnnxAnimator {
            dir,
            tables: Mutex::new(None),
            slot: Mutex::new(Slot { engine: None, loaded_for: None, fallbacks: Vec::new() }),
            wanted: Mutex::new(Wanted::new(&[Ep::Cuda, Ep::DirectMl, Ep::Cpu], SessionOpts::default())),
            cache: Mutex::new(HashMap::new()),
            settings: Mutex::new(OnnxSettings::default()),
            status: Mutex::new("not loaded yet (loads on first use)".into()),
            totals: Mutex::new(Totals::default()),
        }
    }

    /// Model files present?
    pub fn available(&self) -> bool {
        ["prefill.onnx", "step.onnx", "step_grow.onnx", "anim.data", "embeddings.json", "embeddings.bin"].iter().all(|f| self.dir.join(f).is_file())
    }

    pub fn settings(&self) -> OnnxSettings {
        *self.settings.lock().unwrap()
    }
    pub fn set_settings(&self, s: OnnxSettings) {
        *self.settings.lock().unwrap() = s;
    }

    /// One line for the UI: "CUDA (fixed cache on the device), loaded in 1.2 s" / an error / "not loaded yet".
    pub fn status(&self) -> String {
        self.status.lock().unwrap().clone()
    }

    /// Providers to try (best first) and session options. Does not wait for a running request: the sessions are
    /// reloaded on the next request if this changed what they were loaded for.
    pub fn set_providers(&self, order: Vec<Ep>, opts: SessionOpts) {
        *self.wanted.lock().unwrap() = Wanted::new(&order, opts);
    }

    pub fn tables(&self) -> Result<Arc<AnimTables>> {
        let mut t = self.tables.lock().unwrap();
        if let Some(t) = t.as_ref() {
            return Ok(t.clone());
        }
        if !self.available() {
            bail!(
                "the animation model is missing in {} (it is downloaded on the first start; restart the app with a network connection)",
                self.dir.display()
            );
        }
        let loaded = Arc::new(AnimTables::load(&self.dir)?);
        *t = Some(loaded.clone());
        Ok(loaded)
    }

    /// Runs `f` with the loaded engine (loading it first on the configured providers).
    pub fn with_engine<R>(&self, ctl: &Ctl, f: impl FnOnce(&mut Engine) -> Result<R>) -> Result<R> {
        let tables = self.tables()?;
        let mut slot = self.slot.lock().unwrap();
        let wanted = *self.wanted.lock().unwrap();
        if slot.loaded_for != Some(wanted) {
            slot.engine = None; // free the old sessions (and their GPU memory) first
        }
        if slot.engine.is_none() {
            let t0 = Instant::now();
            let mut fallbacks = Vec::new();
            slot.loaded_for = Some(wanted);
            for ep in wanted.order.into_iter().flatten() {
                ctl.report(AnimProgress { stage: format!("loading the animation model on {}", ep.label()), frac: 0.0, secs: 0.0 });
                *self.status.lock().unwrap() = format!("loading on {}…", ep.label());
                match Engine::load(&self.dir, tables.clone(), ep, wanted.opts()) {
                    Ok(e) => {
                        *self.status.lock().unwrap() = format!(
                            "{}, loaded in {:.1} s{}",
                            e.describe(),
                            t0.elapsed().as_secs_f64(),
                            if fallbacks.is_empty() { String::new() } else { format!(" ({} provider(s) skipped)", fallbacks.len()) }
                        );
                        slot.engine = Some(e);
                        break;
                    }
                    Err(e) => fallbacks.push(format!("{e:#}")),
                }
            }
            slot.fallbacks = fallbacks.clone();
            if slot.engine.is_none() {
                slot.loaded_for = None;
                let msg = format!("the animation model could not be loaded: {}", fallbacks.join(" | "));
                *self.status.lock().unwrap() = msg.clone();
                bail!(msg);
            }
        }
        let eng = slot.engine.as_mut().unwrap();
        let had_note = eng.graph_note.is_some();
        let r = f(eng);
        if !had_note && eng.graph_note.is_some() {
            *self.status.lock().unwrap() = eng.describe(); // CUDA graphs just fell back to plain steps
        }
        r
    }

    /// Why providers were skipped on the last load.
    pub fn fallbacks(&self) -> Vec<String> {
        self.slot.lock().unwrap().fallbacks.clone()
    }

    /// The sprite as model tokens (v2 palette).
    pub fn reference(&self, sprite: &Sprite) -> Result<Reference> {
        let t = self.tables()?;
        if !t.sizes.contains(&sprite.size) {
            bail!("the animation model draws 16, 24 or 32 px sprites (this one is {0}x{0})", sprite.size);
        }
        let identity = sprite.palette.len() == t.palette.len() && sprite.palette.iter().zip(&t.palette).all(|(a, b)| a == b);
        let empty = t.empty as u16;
        let mut back = HashMap::new();
        let mut approximated = 0;
        let mut map: HashMap<i16, u16> = HashMap::new();
        // The ink. The sprites the model learned from are drawn with a near-black outline (the darkest colour of
        // 62 % of them is (20, 15, 11), nearly all others are within a few units of black), and the model draws
        // only with the reference's colours: without a near-black one it draws nothing, or an object. So a sprite
        // whose darkest colour is dark but not near-black (a purple or navy outline) is shown to the model with
        // that colour as the ink token, and the ink is turned back into the sprite's own colour in the frames:
        // the sprite looks the same, the model sees the outline it knows.
        let luma = |c: [u8; 3]| 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32;
        let mut ink_for: Option<i16> = None;
        if !identity {
            let mut used: Vec<i16> = sprite.px.iter().copied().filter(|&p| p >= 0 && (p as usize) < sprite.palette.len()).collect();
            used.sort_unstable();
            used.dedup();
            let model_colour = |p: i16| {
                let c = sprite.palette[p as usize];
                t.palette[t.rgb.get(&c).copied().unwrap_or_else(|| nearest(&t.palette, c)) as usize]
            };
            let has_ink = used.iter().any(|&p| luma(model_colour(p)) < INK_LUMA);
            let darkest = used.iter().copied().min_by(|&a, &b| luma(sprite.palette[a as usize]).total_cmp(&luma(sprite.palette[b as usize])));
            if let Some(d) = darkest.filter(|&d| !has_ink && luma(sprite.palette[d as usize]) < DARK_LUMA) {
                ink_for = Some(d);
            }
        }
        let ink = nearest(&t.palette, INK);
        let tokens: Vec<u16> = sprite
            .px
            .iter()
            .map(|&p| {
                if p < 0 || p as usize >= sprite.palette.len() {
                    return empty;
                }
                if identity {
                    return p as u16;
                }
                *map.entry(p).or_insert_with(|| {
                    let c = sprite.palette[p as usize];
                    let tok = if ink_for == Some(p) {
                        ink
                    } else {
                        t.rgb.get(&c).copied().unwrap_or_else(|| {
                            approximated += 1;
                            nearest(&t.palette, c)
                        })
                    };
                    back.entry(tok).or_insert(p);
                    tok
                })
            })
            .collect();
        let mut allowed: Vec<usize> = tokens.iter().map(|&x| x as usize).collect();
        allowed.push(t.empty);
        allowed.sort_unstable();
        allowed.dedup();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        sprite.size.hash(&mut h);
        tokens.hash(&mut h);
        Ok(Reference { size: sprite.size, tokens, allowed, hash: h.finish(), approximated, palette: sprite.palette.clone(), back: (!identity).then_some(back), empty })
    }

    pub(super) fn key(&self, r: &Reference, kind: usize, refk: usize, action: usize, az: u16, st: &OnnxSettings) -> ClipKey {
        ClipKey { sprite: r.hash, size: r.size, kind, refk, action, az, seed: st.seed, temp: st.temperature.to_bits() }
    }

    /// Frames of each clip from the cache, without drawing; None if any is missing.
    pub fn cached(&self, sprite: &Sprite, kind: Kind, clips: &[(String, u16)], refk: Option<usize>) -> Option<Vec<Vec<Frame>>> {
        let t = self.tables().ok()?;
        let r_in = self.reference(sprite).ok()?;
        let st = self.settings();
        let refk = refk.unwrap_or_else(|| st.reference.refk(r_in.size));
        let ki = t.kind_index(kind).ok()?;
        let cache = self.cache.lock().unwrap();
        let (r, refk, anchor) = if st.anchor.enabled {
            let action = anchor_action(kind, |a| t.n_frames(kind, a).is_some());
            let first = cache.get(&self.key(&r_in, ki, refk, t.action_index(&action).ok()?, 0, &st))?.clone();
            let an = self.anchor_from(&r_in, refk, action, first, &st);
            (an.r.clone(), an.refk, Some(an))
        } else {
            (r_in.clone(), refk, None)
        };
        clips
            .iter()
            .map(|(a, az)| match &anchor {
                Some(an) if *a == an.action && *az == 0 => Some(an.first.iter().map(|x| r_in.to_frame(x)).collect()),
                _ => {
                    let key = self.key(&r, ki, refk, t.action_index(a).ok()?, *az, &st);
                    cache.get(&key).map(|f| f.iter().map(|x| r.to_frame(x)).collect())
                }
            })
            .collect()
    }

    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
    }
}

pub(super) fn describe_group(g: &[(String, u16, usize, usize)]) -> String {
    let mut parts: Vec<(String, Vec<u16>)> = Vec::new();
    for (a, az, _, _) in g {
        match parts.iter_mut().find(|p| p.0 == *a) {
            Some(p) => p.1.push(*az),
            None => parts.push((a.clone(), vec![*az])),
        }
    }
    parts
        .into_iter()
        .map(|(a, az)| if az.len() == 1 { format!("{a} {}°", az[0]) } else { format!("{a} {} dirs", az.len()) })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The outline colour of the training sprites (see [`OnnxAnimator::reference`]).
const INK: [u8; 3] = [28, 24, 30];
/// A model colour below this luma counts as ink.
const INK_LUMA: f32 = 32.0;
/// A sprite colour below this luma is dark enough to stand in for the ink.
const DARK_LUMA: f32 = 90.0;

fn nearest(palette: &[[u8; 3]], c: [u8; 3]) -> u16 {
    let d = |p: &[u8; 3]| (0..3).map(|i| (p[i] as i32 - c[i] as i32).pow(2)).sum::<i32>();
    palette.iter().enumerate().min_by_key(|(_, p)| d(p)).map(|(i, _)| i as u16).unwrap_or(0)
}

/// The animator with a request's stop flag / progress and reference type: what the UI's job thread hands to
/// [`crate::animator::scene::build`] and friends.
pub struct OnnxWith<'a> {
    pub anim: &'a OnnxAnimator,
    pub ctl: Ctl<'a>,
    pub refk: Option<usize>,
}

impl Animator for OnnxWith<'_> {
    fn name(&self) -> &str {
        self.anim.name()
    }
    fn actions(&self, kind: Kind) -> Vec<ActionSpec> {
        self.anim.actions(kind)
    }
    fn animate(&self, sprite: &Sprite, req: &AnimRequest) -> Result<Animation> {
        Ok(self.anim.animate_ctl(sprite, req, &[req.azimuth], self.refk, &self.ctl)?.remove(0))
    }
    fn animate_directions(&self, sprite: &Sprite, req: &AnimRequest, azimuths: &[u16]) -> Result<Vec<Animation>> {
        self.anim.animate_ctl(sprite, req, azimuths, self.refk, &self.ctl)
    }
    fn prefetch(&self, sprite: &Sprite, kind: Kind, clips: &[(String, u16)]) -> Result<()> {
        self.anim.draw_clips(sprite, kind, clips, self.refk, &self.ctl).map(|_| ())
    }
    fn applies_dataset_qc(&self) -> bool {
        false
    }
}

impl Animator for OnnxAnimator {
    fn name(&self) -> &str {
        "ONNX model (animgpt-v3)"
    }
    /// The dataset's actions per kind (the model was trained on all of them). Objects only with a model that
    /// knows the kind "object" (AnimGPT3).
    fn actions(&self, kind: Kind) -> Vec<ActionSpec> {
        standard_actions(kind)
    }
    fn animate(&self, sprite: &Sprite, req: &AnimRequest) -> Result<Animation> {
        OnnxWith { anim: self, ctl: Ctl::default(), refk: None }.animate(sprite, req)
    }
    fn animate_directions(&self, sprite: &Sprite, req: &AnimRequest, azimuths: &[u16]) -> Result<Vec<Animation>> {
        self.animate_ctl(sprite, req, azimuths, None, &Ctl::default())
    }
    fn prefetch(&self, sprite: &Sprite, kind: Kind, clips: &[(String, u16)]) -> Result<()> {
        self.draw_clips(sprite, kind, clips, None, &Ctl::default()).map(|_| ())
    }
    fn applies_dataset_qc(&self) -> bool {
        false
    }
}
