/// The azimuth whose renders face screen-right. Checked on 120 random farm humans: in `punch` the arm reaches to
/// the right at 270° in 105 of them (to the left in 13), and to the left at 90° in 105.
pub const AZ_RIGHT: u16 = 270;
/// The azimuth whose renders face screen-left.
pub const AZ_LEFT: u16 = 90;
/// Run frames last 70 ms in the farm's GIFs, every other action 110 ms.
pub const RUN_TEMPO: f32 = 70.0 / 110.0;
/// Walking speed in stage pixels per walk frame at 32 px (scaled with the sprite size). Measured on farm humans
/// that pass the QC (about 25 px tall, chibi legs): a step is about 5 px heel to heel, so 2 steps per 8-frame
/// cycle give ~1.25 px per frame, and the planted foot then stays put.
pub const WALK_SPEED: f32 = 1.25;
/// Running speed in stage pixels per run frame at 32 px. The mocap run covers 1.56x the walk's distance per frame
/// (2.46 vs 1.57 leg lengths per 8-frame cycle); the run strides of the same characters agree (1.6-2 px).
pub const RUN_SPEED: f32 = 2.0;

/// A place on the stage. Off-stage marks are just outside the edge for the clip that moves there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    OffLeft,
    Centre,
    OffRight,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// Cycle `action` at `az` while moving to `to`; `speed` in stage px per frame at 32 px, frames last `tempo` ticks.
    Move { action: &'static str, az: u16, to: Mark, speed: f32, tempo: f32 },
    /// Play `action` at `az` from its first frame, `times` times, in place.
    Play { action: &'static str, az: u16, times: usize, tempo: f32 },
    /// One frame of `action` per azimuth (continuing its cycle), in place: a turn.
    Turn { action: &'static str, azimuths: Vec<u16>, tempo: f32 },
    /// The empty stage for `ticks`.
    Pause { ticks: f32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Script {
    pub title: &'static str,
    /// Stage width in sprite widths (the stage is one sprite high).
    pub width: usize,
    /// Where the character is before the first step.
    pub start: Mark,
    pub steps: Vec<Step>,
}

impl Script {
    /// The (action, azimuth) clips the scene draws, in order of first use.
    pub fn needs(&self) -> Vec<(&'static str, u16)> {
        let mut out: Vec<(&'static str, u16)> = Vec::new();
        let mut add = |a: &'static str, az: u16| {
            if !out.contains(&(a, az)) {
                out.push((a, az));
            }
        };
        for s in &self.steps {
            match s {
                Step::Move { action, az, .. } | Step::Play { action, az, .. } => add(action, *az),
                Step::Turn { action, azimuths, .. } => azimuths.iter().for_each(|&az| add(action, az)),
                Step::Pause { .. } => {}
            }
        }
        out
    }

    /// The distinct actions, in order of first use.
    pub fn actions(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        for (a, _) in self.needs() {
            if !out.contains(&a) {
                out.push(a);
            }
        }
        out
    }

    /// "walk in · punch · spin_kick · punch x2 · run off" (egui's default font has no arrows).
    pub fn describe(&self) -> String {
        self.steps
            .iter()
            .filter_map(|s| match s {
                Step::Move { action, to: Mark::Centre, .. } => Some(format!("{action} in")),
                Step::Move { action, .. } => Some(format!("{action} off")),
                Step::Play { action, times: 1, .. } => Some(action.to_string()),
                Step::Play { action, times, .. } => Some(format!("{action} x{times}")),
                Step::Turn { .. } => Some("turn".into()),
                Step::Pause { .. } => None,
            })
            .collect::<Vec<_>>()
            .join(" · ")
    }
}

/// The fight scene: walk in from the left to the centre (facing right), punch, spin kick, punch twice, then run
/// on to the right, off the stage (the way it came, no turn); then the empty stage for 5 ticks and loop.
pub fn fight() -> Script {
    fight_with(WALK_SPEED, RUN_SPEED)
}

/// [`fight`] with other walk / run speeds (stage px per frame at 32 px).
pub fn fight_with(walk_speed: f32, run_speed: f32) -> Script {
    Script {
        title: "Fight scene",
        width: 4,
        start: Mark::OffLeft,
        steps: vec![
            Step::Move { action: "walk", az: AZ_RIGHT, to: Mark::Centre, speed: walk_speed, tempo: 1.0 },
            Step::Play { action: "punch", az: AZ_RIGHT, times: 1, tempo: 1.0 },
            Step::Play { action: "spin_kick", az: AZ_RIGHT, times: 1, tempo: 1.0 },
            Step::Play { action: "punch", az: AZ_RIGHT, times: 2, tempo: 1.0 },
            Step::Move { action: "run", az: AZ_RIGHT, to: Mark::OffRight, speed: run_speed, tempo: RUN_TEMPO },
            Step::Pause { ticks: 5.0 },
        ],
    }
}
