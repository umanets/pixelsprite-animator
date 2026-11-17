use std::collections::HashMap;

use super::super::{AZIMUTHS, AnimRequest, Animator, Kind};
use super::qc::QcVerdict;
use super::script::Script;
use super::timeline::{Clip, Scene};
use crate::sprite::{Frame, Sprite};

/// Why a scene cannot be shown.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Unavailable {
    /// Actions the animator does not offer for this kind.
    pub missing: Vec<String>,
    /// The animator failed on a clip (no frames in that direction, model error, ...).
    pub errors: Vec<String>,
    /// The user stopped the animator while it drew the clips; nothing is wrong with the scene itself.
    pub cancelled: bool,
    /// Actions the dataset quality check drops.
    pub failed_qc: Vec<QcVerdict>,
}

impl Unavailable {
    /// One line for the UI.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.missing.is_empty() {
            parts.push(format!("no {}", self.missing.join(", ")));
        }
        if !self.failed_qc.is_empty() {
            parts.push(format!("dropped by the dataset QC: {}", self.failed_qc.iter().map(QcVerdict::describe).collect::<Vec<_>>().join("; ")));
        }
        if !self.errors.is_empty() {
            parts.push(self.errors.join("; "));
        }
        parts.join(" · ")
    }
}

/// Clips + quality check + timeline. `size` is the display size. All clips are first requested together
/// ([`Animator::prefetch`]: one model batch for the ONNX animator, which can take a while; the UI calls this on its
/// animation thread). The quality check asks for all 8 directions at 32 px (the dataset's rule) when `qc` is on and
/// the animator applies it ([`Animator::applies_dataset_qc`]: the 3D-farm library only).
pub fn build(animator: &dyn Animator, sprite: &Sprite, kind: Kind, size: usize, script: &Script, qc: bool) -> Result<Scene, Unavailable> {
    let mut why = Unavailable::default();
    let offered = animator.actions(kind);
    for a in script.actions() {
        if !offered.iter().any(|s| s.name == a) {
            why.missing.push(a.to_string());
        }
    }
    if !why.missing.is_empty() {
        return Err(why);
    }
    // one call for every clip (the ONNX model draws them as one batch); `animate` then reads its cache
    let needs: Vec<(String, u16)> = script.needs().into_iter().map(|(a, az)| (a.to_string(), az)).collect();
    if let Err(e) = animator.prefetch(sprite, kind, &needs) {
        why.cancelled = crate::cancel::is_cancelled(&e);
        if !why.cancelled {
            why.errors.push(format!("{e:#}"));
        }
        return Err(why);
    }
    let mut clips = Vec::new();
    for (action, az) in script.needs() {
        let req = AnimRequest { kind, action: action.to_string(), azimuth: az, n_frames: 0, size };
        match animator.animate(sprite, &req) {
            Ok(a) if !a.frames.is_empty() => clips.push(Clip { action, azimuth: az, frames: a.frames }),
            Ok(_) => why.errors.push(format!("{action} at {az}°: no frames")),
            Err(e) if crate::cancel::is_cancelled(&e) => {
                why.cancelled = true;
                return Err(why);
            }
            Err(e) => {
                let e = format!("{e:#}");
                let e: String = e.chars().take(140).collect();
                if !why.errors.iter().any(|x| x.ends_with(&e)) {
                    why.errors.push(format!("{action} at {az}°: {e}"));
                }
            }
        }
    }
    let mut verdicts = Vec::new();
    if qc && animator.applies_dataset_qc() && why.errors.is_empty() {
        let mut cache: HashMap<&str, QcVerdict> = HashMap::new();
        for action in script.actions() {
            let req = AnimRequest { kind, action: action.to_string(), azimuth: 0, n_frames: 0, size: 32 };
            match animator.animate_directions(sprite, &req, &AZIMUTHS) {
                Ok(dirs) => {
                    let frames: Vec<Vec<Frame>> = dirs.into_iter().map(|a| a.frames).collect();
                    cache.insert(action, QcVerdict::of(action, &frames));
                }
                Err(e) if crate::cancel::is_cancelled(&e) => {
                    why.cancelled = true;
                    return Err(why);
                }
                Err(e) => why.errors.push(format!("{action} (quality check): {e:#}")),
            }
        }
        for action in script.actions() {
            if let Some(v) = cache.remove(action) {
                if v.dropped() {
                    why.failed_qc.push(v.clone());
                }
                verdicts.push(v);
            }
        }
    }
    if !why.missing.is_empty() || !why.errors.is_empty() || !why.failed_qc.is_empty() {
        return Err(why);
    }
    Ok(Scene::new(script, size, clips, verdicts))
}
