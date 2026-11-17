//! Scripted scenes on a wide stage: a character walks in, fights and runs away.
//!
//! A scene is built only from per-action frame sequences that an [`crate::animator::Animator`] returns (`animate` for the
//! directions the scene draws, `animate_directions` for the quality check), so it works the same for the 3D-farm
//! library, the mock and the ONNX model. The frames are "in place" (root motion removed); the timeline
//! moves the sprite across the stage itself.
//!
//! * [`fight`]: enter from the left walking to the centre, `punch`, `spin_kick`, `punch` x2, then `run` on
//!   in the same direction off the right edge (no turn), a short empty pause, loop.
//! * [`build`]: asks the animator for the clips, applies the dataset quality check ([`QcVerdict`], the rule of
//!   `pilot3d/dataset/pack.py`) and lays out the timeline ([`timeline`]), or says why the scene is unavailable.
//!
//! Timing is in *ticks*: one tick = one frame of the single-action preview (1 / fps). Run frames last
//! [`RUN_TEMPO`] ticks, as in the farm's preview GIFs (70 ms run frames vs 110 ms for every other action).


mod build;
mod qc;
mod script;
#[cfg(test)]
mod tests;
mod timeline;

pub use build::{Unavailable, build};
pub use qc::{QcFlags, QcVerdict, qc_frame};
pub use script::{AZ_LEFT, AZ_RIGHT, Mark, RUN_SPEED, RUN_TEMPO, Script, Step, WALK_SPEED, fight, fight_with};
pub use timeline::{Clip, Scene, Shot, timeline};
