//! Sprite animation ("image-to-video" for sprites) behind one trait, so the UI does not care where frames come from.
//!
//! * [`mock::MockAnimator`]  — procedural placeholder motion made from the sprite itself (used now).
//! * [`library::LibraryAnimator`] — plays precomputed animations of the 3D farm (`pilot3d/farm_out_*`).
//! * [`onnx::OnnxAnimator`] — the animation model (AnimGPT, run anim39) in ONNX Runtime; [`jobs`] runs it on a
//!   background thread for the UI.
//!
//! [`scene`] strings several actions of any animator into a scripted scene on a wide stage (the fight scene).

pub mod jobs;
pub mod library;
pub mod mock;
pub mod onnx;
pub mod scene;

use anyhow::Result;

pub use crate::classify::Kind;
use crate::sprite::{Frame, Sprite};

/// The 8 directions: 0 = front (facing the viewer), 90 = side / profile, 180 = back, 270 = the other side.
pub const AZIMUTHS: [u16; 8] = [0, 45, 90, 135, 180, 225, 270, 315];

/// An action and its frame count in the 3D farm (pilot3d `pix/meta.json`), used as the default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionSpec {
    pub name: String,
    pub frames: usize,
    /// Loops naturally (walk) or plays once (death); only a hint for the default of the loop checkbox.
    pub looping: bool,
}

fn spec(name: &str, frames: usize, looping: bool) -> ActionSpec {
    ActionSpec { name: name.into(), frames, looping }
}

/// The action set the farm renders per kind (17 human, 3 quadruped, 2 bird actions).
pub fn standard_actions(kind: Kind) -> Vec<ActionSpec> {
    match kind {
        Kind::Human => vec![
            spec("idle", 4, true),
            spec("walk", 8, true),
            spec("run", 8, true),
            spec("jump", 6, true),
            spec("crouch", 2, true),
            spec("crouch_walk", 6, true),
            spec("roll", 6, true),
            spec("punch", 4, true),
            spec("kick", 5, true),
            spec("spin_kick", 8, true),
            spec("slash", 6, true),
            spec("cast", 6, true),
            spec("block", 2, true),
            spec("hurt", 3, false),
            spec("death", 6, false),
            spec("cheer", 4, true),
            spec("wave", 6, true),
        ],
        Kind::Quadruped => vec![spec("idle", 4, true), spec("walk", 8, true), spec("run", 8, true)],
        Kind::Bird => vec![spec("fly", 8, true), spec("glide", 4, true)],
    }
}

/// What to animate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimRequest {
    pub kind: Kind,
    pub action: String,
    /// One of [`AZIMUTHS`].
    pub azimuth: u16,
    /// Frames to produce (0 = the animator's native count).
    pub n_frames: usize,
    /// Output size (16, 24 or 32); normally the sprite's own size.
    pub size: usize,
}

#[derive(Debug, Clone)]
pub struct Animation {
    pub frames: Vec<Frame>,
    /// Short provenance line for the UI ("mock: walk, 8 frames, mirrored" ...).
    pub note: String,
}

/// An animator turns a reference sprite + conditions into frames that use the reference palette.
pub trait Animator: Send + Sync {
    /// Short name for menus.
    fn name(&self) -> &str;
    /// True for placeholders whose output must be labelled as such in the UI.
    fn is_mock(&self) -> bool {
        false
    }
    /// Actions available for a body kind (with default frame counts).
    fn actions(&self, kind: Kind) -> Vec<ActionSpec>;
    /// Frames for one direction.
    fn animate(&self, sprite: &Sprite, req: &AnimRequest) -> Result<Animation>;
    /// Several directions of the same action. The ONNX model draws these as one batch (one row per direction);
    /// the default just loops.
    fn animate_directions(&self, sprite: &Sprite, req: &AnimRequest, azimuths: &[u16]) -> Result<Vec<Animation>> {
        azimuths.iter().map(|&a| self.animate(sprite, &AnimRequest { azimuth: a, ..req.clone() })).collect()
    }
    /// Prepares every (action, azimuth) clip a caller is about to ask for with `animate`, in as few batches as it
    /// can (the ONNX model draws them together and caches them); the default does nothing.
    fn prefetch(&self, _sprite: &Sprite, _kind: Kind, _clips: &[(String, u16)]) -> Result<()> {
        Ok(())
    }
    /// Whether the fight scene applies the dataset's quality check (pack.py) to this animator's clips. True for
    /// the 3D-farm renders; the mock only moves the sprite and the model's actions all exist, so neither is checked.
    fn applies_dataset_qc(&self) -> bool {
        !self.is_mock()
    }
}

/// Borrowed animators are animators too, so callers can hand out `Box<dyn Animator + '_>` without cloning. Every
/// method is forwarded (overrides included), so a wrapper can never drop a behaviour of the animator it borrows.
impl<A: Animator + ?Sized> Animator for &A {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn is_mock(&self) -> bool {
        (**self).is_mock()
    }
    fn actions(&self, kind: Kind) -> Vec<ActionSpec> {
        (**self).actions(kind)
    }
    fn animate(&self, sprite: &Sprite, req: &AnimRequest) -> Result<Animation> {
        (**self).animate(sprite, req)
    }
    fn animate_directions(&self, sprite: &Sprite, req: &AnimRequest, azimuths: &[u16]) -> Result<Vec<Animation>> {
        (**self).animate_directions(sprite, req, azimuths)
    }
    fn prefetch(&self, sprite: &Sprite, kind: Kind, clips: &[(String, u16)]) -> Result<()> {
        (**self).prefetch(sprite, kind, clips)
    }
    fn applies_dataset_qc(&self) -> bool {
        (**self).applies_dataset_qc()
    }
}

/// Picks `n` frames out of `frames` by nearest index (keeps the loop's timing roughly).
pub fn resample(frames: &[Frame], n: usize) -> Vec<Frame> {
    if n == 0 || n == frames.len() || frames.is_empty() {
        return frames.to_vec();
    }
    (0..n).map(|i| frames[(i * frames.len()) / n].clone()).collect()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// An animator that overrides every defaulted method, to see that borrowing it keeps the overrides.
    #[derive(Default)]
    struct Overrides {
        prefetched: AtomicUsize,
    }

    impl Animator for Overrides {
        fn name(&self) -> &str {
            "overrides"
        }
        fn is_mock(&self) -> bool {
            true
        }
        fn actions(&self, _: Kind) -> Vec<ActionSpec> {
            vec![spec("only", 3, false)]
        }
        fn animate(&self, _: &Sprite, req: &AnimRequest) -> Result<Animation> {
            Ok(Animation { frames: Vec::new(), note: format!("single {}", req.azimuth) })
        }
        fn animate_directions(&self, _: &Sprite, _: &AnimRequest, azimuths: &[u16]) -> Result<Vec<Animation>> {
            Ok(azimuths.iter().map(|a| Animation { frames: Vec::new(), note: format!("batch {a}") }).collect())
        }
        fn prefetch(&self, _: &Sprite, _: Kind, clips: &[(String, u16)]) -> Result<()> {
            self.prefetched.fetch_add(clips.len(), Ordering::SeqCst);
            Ok(())
        }
        fn applies_dataset_qc(&self) -> bool {
            true // the default would be `!is_mock()` = false
        }
    }

    fn request() -> AnimRequest {
        AnimRequest { kind: Kind::Human, action: "only".into(), azimuth: 90, n_frames: 0, size: 16 }
    }

    #[test]
    fn a_borrowed_animator_keeps_every_override() {
        let owned = Overrides::default();
        let borrowed: Box<dyn Animator + '_> = Box::new(&owned);
        let sprite = Sprite::new(16, vec![-1; 256], Default::default());
        assert_eq!(borrowed.name(), "overrides");
        assert!(borrowed.is_mock());
        assert!(borrowed.applies_dataset_qc(), "the default `!is_mock()` must not replace the override");
        assert_eq!(borrowed.actions(Kind::Bird).len(), 1);
        assert_eq!(borrowed.animate(&sprite, &request()).unwrap().note, "single 90");
        let dirs = borrowed.animate_directions(&sprite, &request(), &[0, 180]).unwrap();
        assert_eq!(dirs.iter().map(|a| a.note.as_str()).collect::<Vec<_>>(), ["batch 0", "batch 180"]);
        borrowed.prefetch(&sprite, Kind::Human, &[("only".into(), 0), ("only".into(), 45)]).unwrap();
        assert_eq!(owned.prefetched.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn the_default_directions_loop_over_animate() {
        struct Plain;
        impl Animator for Plain {
            fn name(&self) -> &str {
                "plain"
            }
            fn actions(&self, kind: Kind) -> Vec<ActionSpec> {
                standard_actions(kind)
            }
            fn animate(&self, _: &Sprite, req: &AnimRequest) -> Result<Animation> {
                Ok(Animation { frames: Vec::new(), note: req.azimuth.to_string() })
            }
        }
        let sprite = Sprite::new(16, vec![-1; 256], Default::default());
        let out = Plain.animate_directions(&sprite, &request(), &AZIMUTHS).unwrap();
        assert_eq!(out.len(), 8);
        assert_eq!(out[3].note, "135");
        assert!(!Plain.is_mock() && Plain.applies_dataset_qc());
    }

    #[test]
    fn the_farm_action_set_per_kind() {
        let names = |k| standard_actions(k).into_iter().map(|a| a.name).collect::<Vec<_>>();
        assert_eq!(standard_actions(Kind::Human).len(), 17);
        assert_eq!(names(Kind::Quadruped), ["idle", "walk", "run"]);
        assert_eq!(names(Kind::Bird), ["fly", "glide"]);
        let death = standard_actions(Kind::Human).into_iter().find(|a| a.name == "death").unwrap();
        assert_eq!((death.frames, death.looping), (6, false));
    }

    #[test]
    fn resample_picks_nearest_indices() {
        let frames: Vec<Frame> = (0..4).map(|i| Sprite::new(1, vec![i], Default::default())).collect();
        let px = |v: &[Frame]| v.iter().map(|f| f.px[0]).collect::<Vec<_>>();
        assert_eq!(px(&resample(&frames, 2)), [0, 2]);
        assert_eq!(px(&resample(&frames, 8)), [0, 0, 1, 1, 2, 2, 3, 3]);
        // 0 = keep the native count; same count and an empty list are returned as they are
        assert_eq!(px(&resample(&frames, 0)), [0, 1, 2, 3]);
        assert_eq!(px(&resample(&frames, 4)), [0, 1, 2, 3]);
        assert!(resample(&[], 5).is_empty());
    }
}
