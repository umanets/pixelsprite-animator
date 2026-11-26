//! The app's state types: cards and runs of generated sprites, the animation / scene state, selections and the
//! `App` struct itself. Fields are `pub(crate)` because the panels in `ui/` read and write them.

use crate::*;

// ------------------------------------------------------------------------------------------------------ state
/// Which sprite of a run a card holds: the source of "one shape", or the k-th drawn sprite.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CardTag {
    Source,
    Index(usize),
}

impl CardTag {
    /// The file-name tag: "src" or "00", "01", … (see [`save::sprite_tag`]).
    pub(crate) fn file_tag(self) -> String {
        match self {
            CardTag::Source => "src".to_string(),
            CardTag::Index(k) => save::sprite_tag(k),
        }
    }
}

/// What `App::save_selected` writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SaveTarget {
    /// The selected generated sprite.
    Sprite,
    /// A sheet of a run: the given run, or (None) the selected sprite's run.
    Sheet(Option<u64>),
}

/// What `App::export_selected` writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ExportKind {
    Sheet,
    Gif,
    Directions,
}

/// An entry of a card's context menu.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CardAction {
    SavePng,
    UseShape,
}

pub(crate) struct Card {
    pub(crate) sprite: Option<Sprite>,
    pub(crate) partial: Option<Vec<i16>>,
    pub(crate) tex: Option<TextureHandle>,
    pub(crate) dirty: bool,
    /// Which sprite of the run this is (gives the file-name tag).
    pub(crate) tag: CardTag,
    /// Mixed captions: how this sprite was drawn (shown under it).
    pub(crate) variant: Option<VariantInfo>,
}

impl Card {
    pub(crate) fn new(tag: CardTag) -> Self {
        Card { sprite: None, partial: None, tex: None, dirty: false, tag, variant: None }
    }
}

pub(crate) struct Run {
    pub(crate) id: u64,
    pub(crate) caption: String,
    pub(crate) size: usize,
    pub(crate) seed: u32,
    pub(crate) cards: Vec<Card>,
    pub(crate) class: Option<Classification>,
    /// The demo's line ("mix of burger, skull · 5 variants · seed 3") and the timing details.
    pub(crate) summary: String,
    pub(crate) meta: String,
    pub(crate) done: bool,
    pub(crate) error: Option<String>,
}

/// Rows per network step (the demo's row budget, `rowsFor` in app.js). The random numbers are drawn chunk by
/// chunk, so the sprites of a seed depend on it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Batching {
    /// The demo on a desktop with WebGPU: 24 rows at 16 px, 8 at 24 / 32.
    DemoGpu,
    /// The demo on the CPU (WebAssembly): 16 / 4.
    DemoCpu,
    Custom(usize),
}

impl Batching {
    pub(crate) fn rows(self, size: usize) -> usize {
        match self {
            Batching::DemoGpu => demo_rows_gpu(size),
            Batching::DemoCpu => demo_rows_cpu(size),
            Batching::Custom(n) => n.max(1),
        }
    }
    pub(crate) fn label(self) -> String {
        match self {
            Batching::DemoGpu => "as the demo with WebGPU (24 / 8)".into(),
            Batching::DemoCpu => "as the demo on the CPU (16 / 4)".into(),
            Batching::Custom(n) => format!("custom: {n} rows"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Selection {
    Generated { run: u64, idx: usize },
    Library,
    /// A PNG loaded from disk (`App::loaded`).
    Loaded,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum KindChoice {
    Auto,
    Force(Kind),
    NotAnimatable,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AnimatorChoice {
    Mock,
    Library,
    Onnx,
}

impl AnimatorChoice {
    pub(crate) fn label(self) -> &'static str {
        match self {
            AnimatorChoice::Mock => "Mock (procedural placeholder)",
            AnimatorChoice::Library => "Library (3D farm renders)",
            AnimatorChoice::Onnx => "ONNX model (animgpt-v3)",
        }
    }
}

/// Everything that determines the animation; recomputed when it changes.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct AnimKey {
    pub(crate) sel: Option<Selection>,
    pub(crate) lib_name: Option<String>,
    /// Which loaded PNG (counts loads).
    pub(crate) loaded: u64,
    pub(crate) animator: AnimatorChoice,
    pub(crate) kind: Option<Kind>,
    pub(crate) action: String,
    pub(crate) azimuth: u16,
    pub(crate) n_frames: usize,
    pub(crate) size: usize,
    /// ONNX animator: settings + reference type (another value draws again; empty for the other animators).
    pub(crate) onnx: String,
}

pub(crate) struct AnimState {
    pub(crate) animator: AnimatorChoice,
    pub(crate) kind: KindChoice,
    pub(crate) action: String,
    pub(crate) azimuth: u16,
    pub(crate) n_frames: usize,
    pub(crate) fps: f32,
    pub(crate) looping: bool,
    pub(crate) playing: bool,
    pub(crate) shown: usize,
    pub(crate) t0: f64,
    pub(crate) key: Option<AnimKey>,
    pub(crate) result: Option<Animation>,
    pub(crate) is_mock: bool,
    pub(crate) textures: Vec<TextureHandle>,
    pub(crate) error: Option<String>,
    pub(crate) lib_size: usize,
    /// Build every animation from the first frame of walk at 0° (front) instead of the original sprite.
    pub(crate) ref_from_walk_front: bool,
    /// ONNX animator: the request being drawn on the animation thread, its progress, and whether the user stopped it.
    /// The animation model draws only when asked: `go` is set by the Animate / Build button (and consumed by the
    /// next refresh), `idle` means "nothing drawn for the current settings yet, waiting for the button".
    pub(crate) go: bool,
    pub(crate) idle: bool,
    pub(crate) pending: Option<Ticket>,
    pub(crate) progress: Option<AnimProgress>,
    pub(crate) stopped: bool,
}

/// What the fight scene depends on (not the single preview's action / direction / frame count).
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct SceneKey {
    pub(crate) sel: Option<Selection>,
    pub(crate) lib_name: Option<String>,
    /// Which loaded PNG (counts loads).
    pub(crate) loaded: u64,
    pub(crate) animator: AnimatorChoice,
    pub(crate) kind: Option<Kind>,
    pub(crate) size: usize,
    pub(crate) qc: bool,
    pub(crate) onnx: String,
}

/// The second, wide preview: a scripted scene (walk in, punch, spin kick, punch x2, run off) built from the
/// selected animator's clips. Shown when the character has those actions (passing the dataset QC).
pub(crate) struct SceneState {
    pub(crate) key: Option<SceneKey>,
    pub(crate) scene: Option<Scene>,
    /// Why there is no scene (one line), for humans.
    pub(crate) note: Option<String>,
    /// The bottom panel is shown (a scene, or a human with a reason why not).
    pub(crate) show: bool,
    pub(crate) is_mock: bool,
    pub(crate) textures: Vec<Vec<TextureHandle>>,
    pub(crate) playing: bool,
    pub(crate) t0: f64,
    pub(crate) shown: usize,
    /// Apply the dataset quality check (pack.py) before offering the scene.
    pub(crate) qc: bool,
    /// Speeds in stage px per frame at 32 px, and the ones the current timeline was laid out with.
    pub(crate) walk_speed: f32,
    pub(crate) run_speed: f32,
    pub(crate) laid_out: (f32, f32),
    pub(crate) last_export: Option<(String, PathBuf)>,
    /// ONNX animator: the scene being built on the animation thread.
    /// The animation model draws only when asked: `go` is set by the Animate / Build button (and consumed by the
    /// next refresh), `idle` means "nothing drawn for the current settings yet, waiting for the button".
    pub(crate) go: bool,
    pub(crate) idle: bool,
    pub(crate) pending: Option<Ticket>,
    pub(crate) progress: Option<AnimProgress>,
    pub(crate) stopped: bool,
}

/// Library thumbnails on a background thread: the UI asks for an entry's folder, the thread reads one small
/// array out of the entry's npz (not the whole file) and answers with the sprite (None = failed).
pub(crate) struct Thumbs {
    pub(crate) tx: Sender<(String, PathBuf)>,
    pub(crate) rx: Receiver<(String, Option<Sprite>)>,
}

impl Thumbs {
    pub(crate) fn start() -> Self {
        let (treq, rreq) = channel::<(String, PathBuf)>();
        let (tdone, rdone) = channel::<(String, Option<Sprite>)>();
        std::thread::Builder::new()
            .name("library-thumbs".into())
            .spawn(move || {
                while let Ok((name, dir)) = rreq.recv() {
                    // drop stale requests: only the newest ask matters
                    let mut last = (name, dir);
                    while let Ok(n) = rreq.try_recv() {
                        last = n;
                    }
                    let _ = tdone.send((last.0, animator_ui::animator::library::read_thumb(&last.1)));
                }
            })
            .expect("spawning the thumbnail thread");
        Thumbs { tx: treq, rx: rdone }
    }
}

pub(crate) struct App {
    pub(crate) cfg: Config,
    pub(crate) dll: DllReport,
    pub(crate) worker: Worker,
    pub(crate) ep_choice: EpChoice,
    pub(crate) ep_active: Option<Ep>,
    pub(crate) ep_describe: String,
    pub(crate) ep_fallbacks: Vec<String>,
    pub(crate) loading: bool,
    pub(crate) palette: Option<Palette>,
    pub(crate) status: String,
    pub(crate) load_error: Option<String>,
    pub(crate) cudnn_busy: bool,
    pub(crate) cudnn_done: Option<String>,
    /// ms/step of the last finished generation (the top bar's provider chip).
    pub(crate) ms_per_step: Option<f64>,

    pub(crate) caption: String,
    pub(crate) size: usize,
    pub(crate) seed: u32,
    pub(crate) auto_seed: bool,
    /// The Outline checkbox (only for a generator that knows the outline words, see `outline_words`).
    pub(crate) outline: bool,
    /// The loaded generator understands ", with outline" / ", no outline".
    pub(crate) outline_words: bool,
    pub(crate) count: usize,
    /// "How many" follows the size (8 / 4) until the user sets it.
    pub(crate) count_touched: bool,
    /// Sampling as the demo's sliders (f64, snapped to their steps, so 0.8 is the demo's 0.8).
    pub(crate) g_shape: f64,
    pub(crate) g_colour: f64,
    pub(crate) temperature: f64,
    pub(crate) top_p: f64,
    pub(crate) filter: bool,
    pub(crate) batching: Batching,
    pub(crate) mode: Mode,
    /// The silhouette pad (pad_size x pad_size, 1 = inside); a stroke paints (Some(true)) or erases.
    pub(crate) pad: Vec<u8>,
    pub(crate) pad_size: usize,
    pub(crate) pad_stroke: Option<(bool, (usize, usize))>,
    /// Caption mixing: the lexicon (from the worker), the enabled variants, the kinds set by clicks on the word
    /// chips and the kinds learned for unknown words (kept for the session, as the demo's page state).
    pub(crate) lexicon: Option<Arc<Lexicon>>,
    pub(crate) presets: Vec<String>,
    pub(crate) variants_on: Vec<VariantKind>,
    pub(crate) kind_override: HashMap<String, ConceptKind>,
    pub(crate) learned_kind: HashMap<String, ConceptKind>,
    pub(crate) encoder_ready: bool,
    /// The demo's `more=1` key: "Sampling" starts open.
    pub(crate) more_open: Option<bool>,
    /// Saving generated sprites: scale and the last message / path.
    pub(crate) save_scale: u32,
    pub(crate) last_save: Option<(String, PathBuf)>,

    pub(crate) next_id: u64,
    pub(crate) running: Option<(u64, Instant)>,
    pub(crate) progress: (usize, usize, usize, usize), // pixel, of, sprites done, of
    pub(crate) runs: Vec<Run>,

    pub(crate) selected: Option<Selection>,
    pub(crate) lib_entries: Vec<LibraryEntry>,
    /// Background scan in progress; on a rescan the new list builds up in `lib_next` and replaces the old one.
    pub(crate) lib_scan: Option<Scanner>,
    pub(crate) lib_next: Option<Vec<LibraryEntry>>,
    pub(crate) lib_scan_note: String,
    pub(crate) lib_filter: String,
    /// Kind filter chip of the Library (None = all kinds).
    pub(crate) lib_kind: Option<Kind>,
    /// Indices of `lib_entries` that pass the filter, for (filter, kind, number of entries).
    pub(crate) lib_view: (String, usize, Vec<usize>),
    pub(crate) lib_data: Option<Arc<LibraryData>>,
    pub(crate) lib_error: Option<String>,
    pub(crate) lib_tex: Option<(String, usize, TextureHandle)>,
    pub(crate) thumbs: Option<Thumbs>,
    /// name -> thumbnail (None = failed to read); bounded, cleared when it grows too big.
    pub(crate) thumb_cache: HashMap<String, Option<TextureHandle>>,

    pub(crate) anim: AnimState,
    pub(crate) scene: SceneState,
    pub(crate) export_scale: u32,
    pub(crate) last_export: Option<(String, PathBuf)>,
    pub(crate) mock: MockAnimator,
    /// The animation model and its background thread (the model itself is `jobs.anim`).
    pub(crate) jobs: AnimJobs,
    /// An 8-direction sheet being drawn by the model, and where it goes.
    pub(crate) dirs_pending: Option<(Ticket, PathBuf)>,
    /// A PNG sprite loaded from disk: sprite, file name, upscale factor found.
    pub(crate) loaded: Option<(Sprite, String, usize)>,
    pub(crate) loaded_id: u64,
    pub(crate) png_path: String,
    pub(crate) png_error: Option<String>,
    /// Why the last generation failed (shown under the Generate button until the next one starts).
    pub(crate) gen_error: Option<String>,
    /// The question asked when an image is loaded: as it is, with an outline, or with an inline.
    pub(crate) png_prep: Option<PrepDialog>,
    /// How the loaded sprite was prepared.
    pub(crate) png_prep_used: Option<PngPrep>,
    /// The loaded sprite's gallery card.
    pub(crate) loaded_tex: Option<(u64, TextureHandle)>,
    /// The file dialog of "Load…" runs in its own process; its answer arrives here.
    pub(crate) png_pick: Option<std::sync::mpsc::Receiver<Option<std::path::PathBuf>>>,
    pub(crate) settings_open: bool,
    pub(crate) demo: Option<Demo>,
}

/// Scripted run for checking the UI without touching the mouse: `ANIMATOR_UI_DEMO="caption=knight;size=16;
/// action=walk;az=90;lib=s32_4900;libaction=walk;out=ui_demo;exit=1"` generates, selects the first sprite, sets the
/// action, saves a screenshot to exports/<out>_1.png, then (with lib=) selects that library entry and saves
/// <out>_2.png, and closes the window when exit=1.
///
/// More keys: `ep=cpu` (load the sprite model on the CPU), `gen=0` (skip generating, go straight to the library
/// entry; waits for the background scan to reach it), `libsize=16|24|32`, `fps=..`, `export=1` (also exports the
/// fight scene GIF when there is one), `scenet=1.5,7,9` (pauses the fight scene at these seconds and saves
/// <out>_scene_<i>.png for each), `animator=onnx|mock|library` (the animator for the generated sprite),
/// `png=<path>` (load this PNG instead of generating), `kind=human|quadruped|bird`, `wait=600` (seconds to wait
/// for the animation model's preview and scene before the screenshots), `refwalk=0|1` (the "build everything
/// from the first walk frame" option), `genshot=1` (screenshot <out>_gen.png while generating, then exit),
/// `win=1440x1060` (handled in main via ANIMATOR_UI_WIN).
///
/// Stage-1 keys: `count=N`, `seed=N`, `mode=new|same|draw`, `mask=potion|disc` (the silhouette pad), `variants=
/// inside,over,centre,blend,caption`, `flip=word,word` (click those word chips), `rows=N` (custom row budget),
/// `exact=1` (exact caption vectors), `more=1` (open "Sampling"), `save=1` (save the selected sprite and its
/// run's sheet after generating), `scale=1|2|4|8`, `idx=K` (which sprite to select).
pub(crate) struct Demo {
    pub(crate) opts: HashMap<String, String>,
    pub(crate) stage: u8,
    pub(crate) t: Instant,
    pub(crate) saved: Arc<AtomicBool>,
    /// Index of the next scene screenshot.
    pub(crate) k: usize,
}

impl Demo {
    pub(crate) fn from_env() -> Option<Self> {
        let v = std::env::var("ANIMATOR_UI_DEMO").ok()?;
        let opts = v.split(';').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (k.trim().to_string(), v.trim().to_string())).collect();
        Some(Demo { opts, stage: 0, t: Instant::now(), saved: Arc::new(AtomicBool::new(false)), k: 0 })
    }
    pub(crate) fn get(&self, k: &str) -> Option<&str> {
        self.opts.get(k).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_tag_renders_as_the_demo_file_tag() {
        assert_eq!(CardTag::Source.file_tag(), "src");
        assert_eq!(CardTag::Index(0).file_tag(), "00");
        assert_eq!(CardTag::Index(7).file_tag(), "07");
        assert_eq!(CardTag::Index(12).file_tag(), "12");
    }

    #[test]
    fn new_card_keeps_its_tag() {
        assert_eq!(Card::new(CardTag::Source).tag, CardTag::Source);
        assert_eq!(Card::new(CardTag::Index(3)).tag.file_tag(), "03");
    }
}

/// How a loaded image is prepared for the animation model, which was trained on sprites drawn with a dark outline.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PngPrep {
    /// Nothing is changed.
    AsIs,
    /// A one-pixel dark line around the silhouette, in the free pixels next to it.
    Outline,
    /// The silhouette's own edge pixels repainted dark.
    Inline,
}

impl PngPrep {
    pub(crate) const ALL: [PngPrep; 3] = [PngPrep::AsIs, PngPrep::Outline, PngPrep::Inline];

    pub(crate) fn label(self) -> &'static str {
        match self {
            PngPrep::AsIs => "As it is",
            PngPrep::Outline => "Outline",
            PngPrep::Inline => "Inline",
        }
    }

    pub(crate) fn describe(self) -> &'static str {
        match self {
            PngPrep::AsIs => "nothing is changed",
            PngPrep::Outline => "a dark line around the sprite",
            PngPrep::Inline => "the sprite's own edge is painted dark",
        }
    }

    pub(crate) fn apply(self, s: &Sprite) -> Sprite {
        match self {
            PngPrep::AsIs => s.clone(),
            PngPrep::Outline => s.outlined(),
            PngPrep::Inline => s.inlined(),
        }
    }
}

/// An image that was read and waits for the user's choice of [`PngPrep`].
pub(crate) struct PrepDialog {
    pub(crate) name: String,
    pub(crate) factor: usize,
    pub(crate) fitted: bool,
    /// The three results, in the order of [`PngPrep::ALL`], and their previews (made on first show).
    pub(crate) variants: Vec<(PngPrep, Sprite, Option<TextureHandle>)>,
    pub(crate) recommended: PngPrep,
}
