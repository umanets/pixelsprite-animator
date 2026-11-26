use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};

use crate::animator::{AnimRequest, Animation, Kind, resample};
use crate::sprite::{Frame, Mulberry32, Sprite};

use super::animator::{OnnxAnimator, describe_group};
use super::engine::RowSpec;
use super::reference::{Anchor, AnchorPalette, AnchorRef, Reference, anchor_action, split_anchor_clips};
use super::settings::{AnimProgress, Ctl, OnnxSettings};
use super::tables::row_seed;

impl OnnxAnimator {
    /// Draws (or takes from the cache) every requested (action, azimuth) clip of a sprite at its own size: the
    /// views of all their azimuths as one batch first, then the frames of all clips as one batch per frame index
    /// (at most [`super::Engine::max_batch`] rows; clips with fewer frames leave the batch early). `refk` None = the
    /// setting. Finished clips stay cached even when the request is stopped.
    pub fn draw_clips(&self, sprite: &Sprite, kind: Kind, clips: &[(String, u16)], refk: Option<usize>, ctl: &Ctl) -> Result<Vec<Vec<Frame>>> {
        let r = self.reference(sprite)?;
        let st = self.settings();
        let refk = refk.unwrap_or_else(|| st.reference.refk(r.size));
        if !st.anchor.enabled {
            return self.draw_clips_ref(&r, kind, clips, refk, ctl);
        }
        // anchored: the anchor action at 0 deg is drawn from the sprite, everything else from its first frame
        let an = self.anchor(&r, kind, refk, ctl)?;
        let (rest, rest_idx) = split_anchor_clips(&an.action, clips);
        let mut out: Vec<Option<Vec<Frame>>> = vec![None; clips.len()];
        for i in 0..clips.len() {
            if !rest_idx.contains(&i) {
                out[i] = Some(an.first.iter().map(|x| r.to_frame(x)).collect());
            }
        }
        if !rest.is_empty() {
            for (i, f) in rest_idx.iter().zip(self.draw_clips_ref(&an.r, kind, &rest, an.refk, ctl)?) {
                out[*i] = Some(f);
            }
        }
        out.into_iter().enumerate().map(|(i, f)| f.ok_or_else(|| anyhow!("clip {i} was not produced"))).collect()
    }

    /// The anchor of a sprite (see [`AnchorOpts`]): draws the anchor action at 0 deg from the sprite (cached like any
    /// clip) and takes its first frame as the reference of everything else.
    fn anchor(&self, r_in: &Reference, kind: Kind, refk_in: usize, ctl: &Ctl) -> Result<Anchor> {
        let t = self.tables()?;
        let st = self.settings();
        let action = anchor_action(kind, |a| t.n_frames(kind, a).is_some());
        self.draw_clips_ref(r_in, kind, &[(action.clone(), 0)], refk_in, ctl)?;
        let ki = t.kind_index(kind)?;
        let first = self
            .cache
            .lock()
            .unwrap()
            .get(&self.key(r_in, ki, refk_in, t.action_index(&action)?, 0, &st))
            .cloned()
            .ok_or_else(|| anyhow!("the anchor clip {action} 0 deg was not drawn"))?;
        Ok(self.anchor_from(r_in, refk_in, action, first, &st))
    }

    /// The anchor reference from the drawn first clip (no drawing).
    pub(super) fn anchor_from(&self, r_in: &Reference, refk_in: usize, action: String, first: Arc<Vec<Vec<u16>>>, st: &OnnxSettings) -> Anchor {
        let tokens = first[0].clone();
        let empty = r_in.empty;
        let r = if tokens.iter().all(|&x| x == empty) {
            r_in.clone() // an empty anchor tells nothing: fall back to the sprite itself
        } else {
            r_in.with_tokens(tokens, st.anchor.palette == AnchorPalette::Input)
        };
        let refk = match st.anchor.refk {
            AnchorRef::Original => 0,
            AnchorRef::FrontRender => 1,
            AnchorRef::Same => refk_in,
        };
        Anchor { r, refk, action, first }
    }

    fn draw_clips_ref(&self, r: &Reference, kind: Kind, clips: &[(String, u16)], refk: usize, ctl: &Ctl) -> Result<Vec<Vec<Frame>>> {
        let t0 = Instant::now();
        let t = self.tables()?;
        let st = self.settings();
        let s = r.size;
        let r = r.clone();
        let ki = t.kind_index(kind)?;
        let view_i = t.action_index("view")?;
        // what is missing
        let mut todo: Vec<(String, u16, usize, usize)> = Vec::new(); // action, az, action index, frames
        {
            let cache = self.cache.lock().unwrap();
            for (a, az) in clips {
                let ai = t.action_index(a)?;
                t.az_index(*az)?;
                let nf = t.n_frames(kind, a).ok_or_else(|| anyhow!("the model has no '{a}' for a {}", kind.label()))?;
                if !cache.contains_key(&self.key(&r, ki, refk, ai, *az, &st)) && !todo.iter().any(|x| x.0 == *a && x.1 == *az) {
                    todo.push((a.clone(), *az, ai, nf));
                }
            }
        }
        if !todo.is_empty() {
            let views: Vec<u16> = {
                let cache = self.cache.lock().unwrap();
                let mut v: Vec<u16> = todo.iter().map(|x| x.1).filter(|az| !cache.contains_key(&self.key(&r, ki, refk, view_i, *az, &st))).collect();
                v.sort_unstable();
                v.dedup();
                v
            };
            self.with_engine(ctl, |eng| {
                let mb = eng.max_batch(s);
                // the plan, for the progress bar: (label, rows) per network batch
                let mut plan: Vec<String> = Vec::new();
                for g in views.chunks(mb) {
                    plan.push(format!("views {}", g.iter().map(|a| format!("{a}°")).collect::<Vec<_>>().join(", ")));
                }
                let groups: Vec<&[(String, u16, usize, usize)]> = todo.chunks(mb).collect();
                for g in &groups {
                    let nmax = g.iter().map(|x| x.3).max().unwrap_or(0);
                    for f in 0..nmax {
                        plan.push(format!("{}: frame {}/{nmax}", describe_group(g), f + 1));
                    }
                }
                let total = plan.len().max(1);
                let n = s * s;
                let mut done = 0usize;
                let mut last = Instant::now();
                let progress = |stage: &str, done: usize, px: usize, last: &mut Instant| {
                    if px == 0 || px + 1 == n || last.elapsed().as_millis() >= 100 {
                        *last = Instant::now();
                        ctl.report(AnimProgress { stage: stage.to_string(), frac: (done as f32 + (px + 1) as f32 / n as f32) / total as f32, secs: t0.elapsed().as_secs_f64() });
                    }
                };
                // 1. views (previous = the reference)
                for g in views.chunks(mb) {
                    let label = plan[done].clone();
                    let conds = g.iter().map(|&az| t.cond(kind, refk, "view", az, 0)).collect::<Result<Vec<_>>>()?;
                    let rows: Vec<RowSpec> = conds.iter().map(|&c| RowSpec { cond: c, reference: &r.tokens, prev: &r.tokens, allowed: &r.allowed, force: None }).collect();
                    let mut rngs: Vec<Mulberry32> = g.iter().map(|&az| Mulberry32::new(row_seed(st.seed, ki, view_i, t.az_index(az).unwrap_or(0), s))).collect();
                    let out = eng.run_rows(s, &rows, &mut rngs, st.temperature, ctl.cancel, &mut |px| progress(&label, done, px, &mut last))?;
                    self.totals.lock().unwrap().batches.push(out.stats);
                    let mut cache = self.cache.lock().unwrap();
                    for (&az, tok) in g.iter().zip(out.tokens) {
                        cache.insert(self.key(&r, ki, refk, view_i, az, &st), Arc::new(vec![tok]));
                    }
                    done += 1;
                }
                // 2. the frames, clip by clip in batches; frame 0 starts from the view at its azimuth
                for g in groups {
                    let mut prev: Vec<Vec<u16>> = {
                        let cache = self.cache.lock().unwrap();
                        g.iter()
                            .map(|x| cache.get(&self.key(&r, ki, refk, view_i, x.1, &st)).map(|v| v[0].clone()).ok_or_else(|| anyhow!("view {}° missing", x.1)))
                            .collect::<Result<_>>()?
                    };
                    let mut rngs: Vec<Mulberry32> = g.iter().map(|x| Mulberry32::new(row_seed(st.seed, ki, x.2, t.az_index(x.1).unwrap_or(0), s))).collect();
                    let mut frames: Vec<Vec<Vec<u16>>> = vec![Vec::new(); g.len()];
                    let nmax = g.iter().map(|x| x.3).max().unwrap_or(0);
                    for f in 0..nmax {
                        if ctl.stopped() {
                            return Err(crate::cancel::Cancelled.into());
                        }
                        let label = plan[done].clone();
                        let active: Vec<usize> = (0..g.len()).filter(|&i| f < g[i].3).collect();
                        let conds = active.iter().map(|&i| t.cond(kind, refk, &g[i].0, g[i].1, f)).collect::<Result<Vec<_>>>()?;
                        let rows: Vec<RowSpec> =
                            active.iter().zip(&conds).map(|(&i, &c)| RowSpec { cond: c, reference: &r.tokens, prev: &prev[i], allowed: &r.allowed, force: None }).collect();
                        // the streams of the active rows (moved out and back so each clip keeps its own)
                        let mut sub: Vec<Mulberry32> = active.iter().map(|&i| std::mem::replace(&mut rngs[i], Mulberry32::new(0))).collect();
                        let out = eng.run_rows(s, &rows, &mut sub, st.temperature, ctl.cancel, &mut |px| progress(&label, done, px, &mut last));
                        for (k, &i) in active.iter().enumerate() {
                            rngs[i] = std::mem::replace(&mut sub[k], Mulberry32::new(0));
                        }
                        let out = out?;
                        self.totals.lock().unwrap().batches.push(out.stats);
                        for (&i, tok) in active.iter().zip(out.tokens) {
                            frames[i].push(tok.clone());
                            prev[i] = tok;
                            if frames[i].len() == g[i].3 {
                                self.cache.lock().unwrap().insert(self.key(&r, ki, refk, g[i].2, g[i].1, &st), Arc::new(std::mem::take(&mut frames[i])));
                            }
                        }
                        done += 1;
                    }
                }
                ctl.report(AnimProgress { stage: "done".into(), frac: 1.0, secs: t0.elapsed().as_secs_f64() });
                Ok(())
            })?;
        }
        let cache = self.cache.lock().unwrap();
        clips
            .iter()
            .map(|(a, az)| {
                let key = self.key(&r, ki, refk, t.action_index(a)?, *az, &st);
                let f = cache.get(&key).ok_or_else(|| anyhow!("{a} at {az}° was not drawn"))?;
                Ok(f.iter().map(|x| r.to_frame(x)).collect())
            })
            .collect()
    }

    /// The views (turns) of a sprite at the given azimuths, one frame each (cached like clips).
    pub fn draw_views(&self, sprite: &Sprite, kind: Kind, azimuths: &[u16], refk: Option<usize>, ctl: &Ctl) -> Result<Vec<Frame>> {
        let r = self.reference(sprite)?;
        let st = self.settings();
        let refk = refk.unwrap_or_else(|| st.reference.refk(r.size));
        if st.anchor.enabled {
            let an = self.anchor(&r, kind, refk, ctl)?;
            return self.draw_views_ref(&an.r, kind, azimuths, an.refk, ctl);
        }
        self.draw_views_ref(&r, kind, azimuths, refk, ctl)
    }

    fn draw_views_ref(&self, r: &Reference, kind: Kind, azimuths: &[u16], refk: usize, ctl: &Ctl) -> Result<Vec<Frame>> {
        let t = self.tables()?;
        let r = r.clone();
        let st = self.settings();
        let ki = t.kind_index(kind)?;
        let view_i = t.action_index("view")?;
        let missing: Vec<u16> = {
            let cache = self.cache.lock().unwrap();
            azimuths.iter().copied().filter(|&az| !cache.contains_key(&self.key(&r, ki, refk, view_i, az, &st))).collect()
        };
        if !missing.is_empty() {
            let t0 = Instant::now();
            self.with_engine(ctl, |eng| {
                let groups: Vec<&[u16]> = missing.chunks(eng.max_batch(r.size)).collect();
                let (total, n) = (groups.len() as f32, r.size * r.size);
                for (gi, g) in groups.into_iter().enumerate() {
                    let label = format!("views {}", g.iter().map(|a| format!("{a}°")).collect::<Vec<_>>().join(", "));
                    let mut last = Instant::now();
                    let mut progress = |px: usize| {
                        if px == 0 || px + 1 == n || last.elapsed().as_millis() >= 100 {
                            last = Instant::now();
                            ctl.report(AnimProgress { stage: label.clone(), frac: (gi as f32 + (px + 1) as f32 / n as f32) / total, secs: t0.elapsed().as_secs_f64() });
                        }
                    };
                    let conds = g.iter().map(|&az| t.cond(kind, refk, "view", az, 0)).collect::<Result<Vec<_>>>()?;
                    let rows: Vec<RowSpec> = conds.iter().map(|&c| RowSpec { cond: c, reference: &r.tokens, prev: &r.tokens, allowed: &r.allowed, force: None }).collect();
                    let mut rngs: Vec<Mulberry32> = g.iter().map(|&az| Mulberry32::new(row_seed(st.seed, ki, view_i, t.az_index(az).unwrap_or(0), r.size))).collect();
                    let out = eng.run_rows(r.size, &rows, &mut rngs, st.temperature, ctl.cancel, &mut progress)?;
                    let mut cache = self.cache.lock().unwrap();
                    for (&az, tok) in g.iter().zip(out.tokens) {
                        cache.insert(self.key(&r, ki, refk, view_i, az, &st), Arc::new(vec![tok]));
                    }
                }
                Ok(())
            })?;
        }
        let cache = self.cache.lock().unwrap();
        azimuths.iter().map(|&az| Ok(r.to_frame(&cache.get(&self.key(&r, ki, refk, view_i, az, &st)).ok_or_else(|| anyhow!("view {az}° missing"))?[0]))).collect()
    }

    pub(super) fn animate_ctl(&self, sprite: &Sprite, req: &AnimRequest, azimuths: &[u16], refk: Option<usize>, ctl: &Ctl) -> Result<Vec<Animation>> {
        let clips: Vec<(String, u16)> = azimuths.iter().map(|&a| (req.action.clone(), a)).collect();
        let all = self.draw_clips(sprite, req.kind, &clips, refk, ctl)?;
        let st = self.settings();
        let r = self.reference(sprite)?;
        let refk = refk.unwrap_or_else(|| st.reference.refk(r.size));
        // the note under the preview: "ref walk front" when the anchor option builds everything from the first
        // walk frame, otherwise what the reference really is
        let reftype = if st.anchor.enabled { "walk front" } else if refk == 0 { "original" } else { "front render" };
        Ok(all
            .into_iter()
            .zip(azimuths)
            .map(|(frames, az)| {
                let native = frames.len();
                let frames = resample(&frames, req.n_frames);
                Animation {
                    note: format!(
                        "model animgpt-v3: {} at {az}°, {} of {native} frames, {}px, T {:.2}, seed {}, ref {}{}",
                        req.action,
                        frames.len(),
                        r.size,
                        st.temperature,
                        st.seed,
                        reftype,
                        if r.approximated > 0 { format!(", {} colour(s) mapped to the nearest v2 colour", r.approximated) } else { String::new() }
                    ),
                    frames,
                }
            })
            .collect())
    }
}
