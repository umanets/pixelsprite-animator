//! The selected sprite and its animation: which animator, which kind, refreshing the loop.

use crate::*;

impl App {
    // ------------------------------------------------------------------------------------------- selection
    pub(crate) fn selected_sprite(&self) -> Option<(Sprite, String, Option<Classification>)> {
        match self.selected? {
            Selection::Generated { run, idx } => {
                let r = self.runs.iter().find(|r| r.id == run)?;
                let s = r.cards.get(idx)?.sprite.clone()?;
                let class = r.class.clone().or_else(|| Some(classify_english(&r.caption)));
                Some((s, r.caption.clone(), class))
            }
            Selection::Library => {
                let d = self.lib_data.as_ref()?;
                let s = d.base_sprite(self.anim.lib_size).or_else(|| d.sizes().first().and_then(|&z| d.base_sprite(z)))?;
                let class = Some(match d.entry.kind {
                    Some(k) => Classification { kind: Some(k), reason: "kind of the 3D-farm job".into() },
                    None => classify_english(&d.entry.caption),
                });
                Some((s, d.entry.caption.clone(), class))
            }
            Selection::Loaded => {
                let (s, name, _) = self.loaded.as_ref()?;
                let words = name.trim_end_matches(".png").trim_end_matches(".PNG").replace(['_', '-', '.'], " ");
                let class = match classify_english(&words) {
                    c @ Classification { kind: Some(_), .. } => c,
                    _ => Classification { kind: Some(Kind::Human), reason: "loaded PNG: no kind word in the file name, assumed human (override below)".into() },
                };
                Some((s.clone(), name.clone(), Some(class)))
            }
        }
    }

    /// Loads `png_path` (a 16 / 24 / 32 px sprite or an integer upscale of one) and selects it.
    pub(crate) fn load_png(&mut self) {
        let path = PathBuf::from(self.png_path.trim().trim_matches('"'));
        match Sprite::from_png(&path) {
            Ok((sprite, f)) => {
                let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "sprite.png".into());
                self.loaded = Some((sprite, name, f));
                self.loaded_id += 1;
                self.selected = Some(Selection::Loaded);
                self.anim.kind = KindChoice::Auto;
                if self.anim.animator == AnimatorChoice::Library {
                    self.anim.animator = if self.jobs.anim.available() { AnimatorChoice::Onnx } else { AnimatorChoice::Mock };
                }
                self.png_error = None;
            }
            Err(e) => self.png_error = Some(format!("{e:#}")),
        }
    }

    /// The reference type for the model: a library entry's sprite is the farm's front render.
    pub(crate) fn onnx_refk(&self) -> Option<usize> {
        match (self.selected, self.jobs.anim.settings().reference) {
            (Some(Selection::Library), RefChoice::Auto) => Some(1),
            _ => None,
        }
    }

    /// What the ONNX animator's output depends on besides the request (empty for the other animators). Includes
    /// the walk-front reference option: toggling it draws the preview and the scene again.
    pub(crate) fn onnx_tag(&self) -> String {
        if self.anim.animator == AnimatorChoice::Onnx { format!("{:?} {:?}", self.jobs.anim.settings(), self.onnx_refk()) } else { String::new() }
    }

    pub(crate) fn effective_kind(&self, class: &Option<Classification>) -> Option<Kind> {
        match self.anim.kind {
            KindChoice::Auto => class.as_ref().and_then(|c| c.kind),
            KindChoice::Force(k) => Some(k),
            KindChoice::NotAnimatable => None,
        }
    }

    pub(crate) fn animator(&self) -> Box<dyn Animator + '_> {
        match self.anim.animator {
            AnimatorChoice::Mock => Box::new(&self.mock),
            AnimatorChoice::Library => match &self.lib_data {
                Some(d) => Box::new(LibraryAnimator { data: d.clone() }),
                None => Box::new(MissingLibrary),
            },
            AnimatorChoice::Onnx => Box::new(&*self.jobs.anim),
        }
    }

    pub(crate) fn actions_for(&self, kind: Kind) -> Vec<ActionSpec> {
        self.animator().actions(kind)
    }

    /// Recomputes the animation when anything it depends on changed.
    pub(crate) fn refresh_animation(&mut self, ctx: &egui::Context) {
        let Some((sprite, _cap, class)) = self.selected_sprite() else {
            self.anim.result = None;
            self.anim.textures.clear();
            self.anim.key = None;
            return;
        };
        let kind = self.effective_kind(&class);
        let key = AnimKey {
            sel: self.selected,
            lib_name: self.lib_data.as_ref().map(|d| d.entry.name.clone()),
            loaded: self.loaded_id,
            animator: self.anim.animator,
            kind,
            action: self.anim.action.clone(),
            azimuth: self.anim.azimuth,
            n_frames: self.anim.n_frames,
            size: sprite.size,
            onnx: self.onnx_tag(),
        };
        if self.anim.key.as_ref() == Some(&key) {
            return;
        }
        self.anim.key = Some(key);
        self.anim.result = None;
        self.anim.textures.clear();
        self.anim.error = None;
        self.anim.stopped = false;
        self.anim.progress = None;
        if let Some(t) = self.anim.pending.take() {
            t.stop();
        }
        let Some(kind) = kind else { return };
        let req = AnimRequest { kind, action: self.anim.action.clone(), azimuth: self.anim.azimuth, n_frames: self.anim.n_frames, size: sprite.size };
        if self.anim.animator == AnimatorChoice::Onnx {
            // drawn on the animation thread (answered at once from the cache); see poll_jobs
            self.anim.is_mock = false;
            self.anim.pending = Some(self.jobs.submit(Job::Preview { sprite, req, refk: self.onnx_refk() }));
            return;
        }
        let (is_mock, result) = {
            let animator = self.animator();
            (animator.is_mock(), animator.animate(&sprite, &req))
        };
        self.anim.is_mock = is_mock;
        match result {
            Ok(a) => {
                self.anim.textures = a.frames.iter().enumerate().map(|(i, f)| texture(ctx, &format!("frame{i}"), f)).collect();
                self.anim.result = Some(a);
                self.anim.t0 = ctx.input(|i| i.time);
            }
            Err(e) => self.anim.error = Some(format!("{e:#}")),
        }
    }

    /// Keeps the action valid for the current kind / animator (and applies its default frame count).
    pub(crate) fn fix_action(&mut self, kind: Option<Kind>) {
        let Some(kind) = kind else { return };
        let acts = self.actions_for(kind);
        if acts.is_empty() {
            return;
        }
        if !acts.iter().any(|a| a.name == self.anim.action) {
            let a = &acts[0];
            self.anim.action = a.name.clone();
            self.anim.n_frames = a.frames.max(1);
            self.anim.looping = a.looping;
        }
    }
}
