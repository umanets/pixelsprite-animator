//! Exporting the selected sprite / loop as PNG sheets and GIFs.

use crate::*;

impl App {
    // ---------------------------------------------------------------------------------------------- export
    /// `_refwalk` is added to export names only when the experimental walk-front reference option is ON, so the
    /// default names stay as they were.
    pub(crate) fn ref_tag(&self) -> &'static str {
        if self.anim.animator == AnimatorChoice::Onnx && self.anim.ref_from_walk_front { "_refwalk" } else { "" }
    }

    pub(crate) fn export_selected(&mut self, what: ExportKind) {
        let Some((sprite, caption, class)) = self.selected_sprite() else { return };
        let Some(kind) = self.effective_kind(&class) else { return };
        let dir = exports_dir();
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.last_export = Some((format!("Export failed: {e}"), dir));
            return;
        }
        let tag = if self.anim.is_mock { "_MOCK" } else { "" };
        let stem = format!("{}_{}_{}_{}{}", slug(&caption), sprite.size, kind.label(), self.anim.action, self.ref_tag());
        let result: anyhow::Result<PathBuf> = (|| {
            let anim = self.anim.result.as_ref().ok_or_else(|| anyhow::anyhow!("no animation"))?;
            match what {
                ExportKind::Sheet => {
                    let p = dir.join(format!("{stem}_a{:03}_{}f{tag}.png", self.anim.azimuth, anim.frames.len()));
                    export::sheet_png(&[anim.frames.clone()], self.export_scale, 0, &p)?;
                    Ok(p)
                }
                ExportKind::Gif => {
                    let p = dir.join(format!("{stem}_a{:03}_{}f{tag}.gif", self.anim.azimuth, anim.frames.len()));
                    export::gif(&anim.frames, self.export_scale, self.anim.fps, self.anim.looping, &p)?;
                    Ok(p)
                }
                ExportKind::Directions if self.anim.animator == AnimatorChoice::Onnx => {
                    // drawn on the animation thread; poll_jobs writes the sheet ({n} = frames)
                    let req = AnimRequest { kind, action: self.anim.action.clone(), azimuth: 0, n_frames: self.anim.n_frames, size: sprite.size };
                    let p = dir.join(format!("{stem}_8dirs_{{n}}f.png"));
                    let t = self.jobs.submit(Job::Directions { sprite: sprite.clone(), req, refk: self.onnx_refk() });
                    self.dirs_pending = Some((t, p.clone()));
                    Ok(dir.clone())
                }
                ExportKind::Directions => {
                    let req = AnimRequest { kind, action: self.anim.action.clone(), azimuth: 0, n_frames: self.anim.n_frames, size: sprite.size };
                    let all = self.animator().animate_directions(&sprite, &req, &AZIMUTHS)?;
                    let rows: Vec<Vec<Sprite>> = all.into_iter().map(|a| a.frames).collect();
                    let p = dir.join(format!("{stem}_8dirs_{}f{tag}.png", rows.first().map(Vec::len).unwrap_or(0)));
                    export::sheet_png(&rows, self.export_scale, 0, &p)?;
                    Ok(p)
                }
            }
        })();
        if self.dirs_pending.is_some() && what == ExportKind::Directions {
            self.last_export = Some(("Drawing the 8-direction sheet with the model…".into(), dir));
            return;
        }
        self.last_export = Some(match result {
            Ok(p) => (format!("Saved {}", p.display()), p),
            Err(e) => (format!("Export failed: {e:#}"), dir),
        });
    }

    /// The animation thread drew the 8 directions: writes the sheet (`{n}` in the pending name = frames per row).
    pub(crate) fn finish_directions(&mut self, r: Result<Vec<Animation>, Failure>) {
        let Some((_, path)) = self.dirs_pending.take() else { return };
        let result = r.map_err(|e| anyhow::anyhow!(e)).and_then(|all| {
            let rows: Vec<Vec<Sprite>> = all.into_iter().map(|a| a.frames).collect();
            let path = path.with_file_name(path.file_name().unwrap_or_default().to_string_lossy().replace("{n}", &rows.first().map(Vec::len).unwrap_or(0).to_string()));
            export::sheet_png(&rows, self.export_scale, 0, &path)?;
            Ok(path)
        });
        self.last_export = Some(match result {
            Ok(p) => (format!("Saved {}", p.display()), p),
            Err(e) => (format!("Export failed: {e:#}"), exports_dir()),
        });
    }
}
