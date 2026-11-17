use super::qc::{QcVerdict, qc_frame};
use super::script::{Mark, Script, Step};
use crate::sprite::Frame;

// --------------------------------------------------------------------------------------------------- the scene
/// The frames of one action in one direction.
#[derive(Clone, Debug)]
pub struct Clip {
    pub action: &'static str,
    pub azimuth: u16,
    pub frames: Vec<Frame>,
}

/// One frame of the scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shot {
    /// (clip, frame) on the stage; None = empty stage.
    pub sprite: Option<(usize, usize)>,
    /// Left edge of the sprite's cell in stage pixels (may be off the stage).
    pub x: i32,
    /// How long it stays, in ticks (1 tick = 1 / fps).
    pub ticks: f32,
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub title: &'static str,
    pub describe: String,
    /// Sprite size; the stage is `stage_w` x `size`.
    pub size: usize,
    pub stage_w: usize,
    pub clips: Vec<Clip>,
    pub shots: Vec<Shot>,
    /// Start of each shot in ticks.
    starts: Vec<f32>,
    /// Dataset QC verdict per action (empty when the check was off).
    pub qc: Vec<QcVerdict>,
}

impl Scene {
    pub fn new(script: &Script, size: usize, clips: Vec<Clip>, qc: Vec<QcVerdict>) -> Self {
        let mut s = Scene { title: script.title, describe: String::new(), size, stage_w: 0, clips, shots: Vec::new(), starts: Vec::new(), qc };
        s.relayout(script);
        s
    }

    /// Lays the same clips out for another script (other speeds); the script must need the same clips.
    pub fn relayout(&mut self, script: &Script) {
        self.title = script.title;
        self.describe = script.describe();
        self.stage_w = script.width * self.size;
        self.shots = timeline(script, &self.clips, self.size);
        let mut t = 0.0;
        self.starts = self
            .shots
            .iter()
            .map(|s| {
                let st = t;
                t += s.ticks;
                st
            })
            .collect();
    }

    pub fn total_ticks(&self) -> f32 {
        self.shots.iter().map(|s| s.ticks).sum()
    }

    /// The shot on screen `ticks` after the start, looping.
    pub fn shot_at(&self, ticks: f64) -> usize {
        let total = self.total_ticks() as f64;
        if self.shots.is_empty() || total <= 0.0 {
            return 0;
        }
        let t = ticks.rem_euclid(total) as f32;
        self.starts.partition_point(|&s| s <= t).saturating_sub(1)
    }

    pub fn start_of(&self, shot: usize) -> f32 {
        self.starts.get(shot).copied().unwrap_or(0.0)
    }

    /// The sprite of a shot and where its cell goes.
    pub fn sprite_of(&self, shot: usize) -> Option<(&Frame, i32)> {
        let s = self.shots.get(shot)?;
        let (c, f) = s.sprite?;
        Some((self.clips.get(c)?.frames.get(f)?, s.x))
    }

    /// The stage at one shot as RGBA (`stage_w` x `size`, transparent background).
    pub fn rgba(&self, shot: usize) -> Vec<u8> {
        let (w, h) = (self.stage_w, self.size);
        let mut out = vec![0u8; w * h * 4];
        if let Some((f, x0)) = self.sprite_of(shot) {
            let px = f.rgba();
            for y in 0..f.size.min(h) {
                for x in 0..f.size {
                    let sx = x0 + x as i32;
                    if sx < 0 || sx >= w as i32 {
                        continue;
                    }
                    let i = (y * f.size + x) * 4;
                    if px[i + 3] > 0 {
                        let o = (y * w + sx as usize) * 4;
                        out[o..o + 4].copy_from_slice(&px[i..i + 4]);
                    }
                }
            }
        }
        out
    }

    /// GIF delays (centiseconds) at `fps` ticks per second, rounded on the running clock so the total stays right.
    pub fn delays_cs(&self, fps: f32) -> Vec<u16> {
        let fps = fps.max(0.5) as f64;
        let mut t = 0.0f64;
        let mut prev = 0i64;
        self.shots
            .iter()
            .map(|s| {
                t += s.ticks as f64;
                let now = (t / fps * 100.0).round() as i64;
                let d = (now - prev).max(2);
                prev = now.max(prev + 2);
                d.min(6000) as u16
            })
            .collect()
    }

    /// Frames that touch the canvas border, per clip the scene shows ("spin_kick 270°: 2/8").
    pub fn clipped_frames(&self) -> Vec<String> {
        self.clips
            .iter()
            .filter_map(|c| {
                let n = c.frames.iter().filter(|f| qc_frame(f, f.opaque_count()).clipped).count();
                (n > 0).then(|| format!("{} {}°: {n}/{}", c.action, c.azimuth, c.frames.len()))
            })
            .collect()
    }
}

/// Horizontal extents of the opaque pixels of a clip's frames: (smallest left edge, largest left edge, smallest
/// right edge, largest right edge), inclusive; the whole cell for an empty clip.
fn extent(frames: &[Frame], size: usize) -> (i32, i32, i32, i32) {
    let mut ext: Option<(i32, i32, i32, i32)> = None;
    for f in frames {
        if let Some((x0, _, x1, _)) = f.bbox() {
            ext = Some(match ext {
                Some((a, b, c, d)) => (a.min(x0), b.max(x0), c.min(x1), d.max(x1)),
                None => (x0, x0, x1, x1),
            });
        }
    }
    ext.unwrap_or((0, 0, size as i32 - 1, size as i32 - 1))
}

/// Lays out the shots of a script for clips of sprite size `size`. Clips the script needs but `clips` lacks are
/// skipped (with [`crate::animator::scene::build`] they are always there).
pub fn timeline(script: &Script, clips: &[Clip], size: usize) -> Vec<Shot> {
    let stage_w = (script.width * size) as f32;
    let find = |action: &str, az: u16| clips.iter().position(|c| c.action == action && c.azimuth == az && !c.frames.is_empty());
    // Off-stage marks: an exit goes until every frame is hidden (widest frame); an entrance starts where the
    // narrowest frame is just hidden, so a swinging arm or weapon does not leave the first frames empty.
    let mark_x = |m: Mark, clip: usize, entering: bool| -> f32 {
        let (x0_min, x0_max, x1_min, x1_max) = extent(&clips[clip].frames, size);
        match (m, entering) {
            (Mark::OffLeft, true) => -(x1_min as f32 + 1.0),
            (Mark::OffLeft, false) => -(x1_max as f32 + 1.0),
            (Mark::OffRight, true) => stage_w - x0_max as f32,
            (Mark::OffRight, false) => stage_w - x0_min as f32,
            (Mark::Centre, _) => ((stage_w - size as f32) / 2.0).round(),
        }
    };
    let scale = size as f32 / 32.0;
    let mut shots = Vec::new();
    let mut pos: Option<f32> = None;
    // the cycle continues when the next step uses the same action (walk -> walk, turn -> run)
    let mut phase: (&str, usize) = ("", 0);
    for step in &script.steps {
        match step {
            Step::Move { action, az, to, speed, tempo } => {
                let Some(c) = find(action, *az) else { continue };
                let from = pos.unwrap_or_else(|| mark_x(script.start, c, true));
                let target = mark_x(*to, c, false);
                let d = target - from;
                let v = (speed * scale).max(0.1);
                let n = ((d.abs() / v).ceil() as usize).max(1);
                let len = clips[c].frames.len();
                let first = if phase.0 == *action { phase.1 } else { 0 };
                for i in 1..=n {
                    let x = from + d * i as f32 / n as f32;
                    shots.push(Shot { sprite: Some((c, (first + i - 1) % len)), x: x.round() as i32, ticks: *tempo });
                }
                phase = (action, first + n);
                pos = Some(target);
            }
            Step::Play { action, az, times, tempo } => {
                let Some(c) = find(action, *az) else { continue };
                let x = pos.unwrap_or_else(|| mark_x(script.start, c, true));
                for _ in 0..*times {
                    for f in 0..clips[c].frames.len() {
                        shots.push(Shot { sprite: Some((c, f)), x: x.round() as i32, ticks: *tempo });
                    }
                }
                phase = (action, 0);
                pos = Some(x);
            }
            Step::Turn { action, azimuths, tempo } => {
                let mut k = if phase.0 == *action { phase.1 } else { 0 };
                for &az in azimuths {
                    let Some(c) = find(action, az) else { continue };
                    let x = pos.unwrap_or_else(|| mark_x(script.start, c, true));
                    shots.push(Shot { sprite: Some((c, k % clips[c].frames.len())), x: x.round() as i32, ticks: *tempo });
                    k += 1;
                    pos = Some(x);
                }
                phase = (action, k);
            }
            Step::Pause { ticks } => shots.push(Shot { sprite: None, x: 0, ticks: *ticks }),
        }
    }
    shots
}
