use std::path::PathBuf;
use std::sync::Arc;
use animator_ui::animator::{AZIMUTHS, AnimRequest, Kind};
use animator_ui::config::{Config, exports_dir, slug};
use animator_ui::ep::{EpChoice, setup_dll_paths};
use animator_ui::export::{gif, sheet_png};
use animator_ui::model::{GenParams};
use animator_ui::sprite::Sprite;
use animator_ui::worker::headless::Headless;
use super::*;

pub(super) fn b64_floats(s: &str) -> Vec<f32> {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut val = [255u8; 256];
    for (i, &c) in T.iter().enumerate() {
        val[c as usize] = i as u8;
    }
    let mut bytes = Vec::new();
    let mut acc = 0u32;
    let mut bits = 0;
    for &c in s.as_bytes() {
        let v = val[c as usize];
        if v == 255 {
            continue;
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push((acc >> bits) as u8);
        }
    }
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Share of opaque rows identical to the row above (horizontal stripes = a broken cache).
pub(super) fn stripe_score(s: &Sprite) -> f32 {
    let n = s.size;
    let (mut same, mut rows) = (0, 0);
    for y in 1..n {
        let a = &s.px[(y - 1) * n..y * n];
        let b = &s.px[y * n..(y + 1) * n];
        if b.iter().any(|&k| k >= 0) {
            rows += 1;
            if a == b {
                same += 1;
            }
        }
    }
    if rows == 0 { 0.0 } else { same as f32 / rows as f32 }
}

// ---------------------------------------------------------------------------------------- the animation model
/// Lines printed and kept for `selftest_onnx_<ep>.txt`.
pub(super) struct Report(pub(super) Vec<String>);

impl Report {
    pub(super) fn line(&mut self, s: impl Into<String>) {
        let s = s.into();
        println!("{s}");
        self.0.push(s);
    }
}

/// Farm entry of a character (the Library animator's data: ground truth of held-out characters).
pub(super) fn farm(cfg: &Config, name: &str) -> anyhow::Result<Arc<animator_ui::animator::library::LibraryData>> {
    use animator_ui::animator::library::{LibraryData, read_entry};
    let roots = cfg.library_roots();
    let root = roots.iter().find(|r| r.join(name).join("pix").join("sprites.npz").is_file()).ok_or_else(|| anyhow::anyhow!("{name} is not in the library roots"))?;
    let e = read_entry(root.join(name), &root.file_name().unwrap_or_default().to_string_lossy(), true).ok_or_else(|| anyhow::anyhow!("{name}: bad entry"))?;
    Ok(Arc::new(LibraryData::load(&e)?))
}

/// Share of pixels with the same token, and IoU of the silhouettes.
pub(super) fn agreement(a: &[u16], b: &[u16], empty: u16) -> (f64, f64) {
    let same = a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len().max(1) as f64;
    let (mut inter, mut uni) = (0usize, 0usize);
    for (x, y) in a.iter().zip(b) {
        let (p, q) = (*x != empty, *y != empty);
        inter += (p && q) as usize;
        uni += (p || q) as usize;
    }
    (same, if uni == 0 { 1.0 } else { inter as f64 / uni as f64 })
}

pub(super) fn training_val_bpp(s: usize, what: &str) -> f64 {
    // train_log.json of anim39 (best.pt): val{S}_{view|anim}_bpp
    match (s, what) {
        (16, "view") => 0.719,
        (16, _) => 0.328,
        (24, "view") => 0.606,
        (24, _) => 0.324,
        (32, "view") => 0.598,
        _ => 0.306,
    }
}

/// `--onnx-anim`: the animation model on held-out (split "val") characters of the 3D farm.
///
/// 1. Parity with PyTorch (`models/animator/parity_ref.json`, written by `export_anim_onnx.py --reference`):
///    teacher-forced bits of the farm frames (8 views, the walk / fly frames at 90°) and frames sampled with the same
///    random numbers (views at 90 / 180 / 270, then 4 frames at 90 from the sampled 90° view).
/// 2. Quality against the farm (ground truth): bits per pixel (teacher forced), and pixel agreement / silhouette IoU
///    of sampled views and 8-direction walks; GT-vs-model sheets and GIFs in exports/onnx/.
/// 3. Timings: one walk direction and a full 8-direction walk at each size (views batch + frame batches).
/// 4. The fight scene drawn by the model (GIF), for a held-out human at each size and, with `--caption`, a sprite
///    drawn by the app's text-to-sprite model.
pub(super) fn onnx_anim(cfg: &Config) -> anyhow::Result<()> {
    use animator_ui::animator::onnx::{Ctl, OnnxAnimator, OnnxSettings, OnnxWith, RowSpec, row_seed};
    use animator_ui::animator::scene::{build, fight};
    use animator_ui::animator::standard_actions;
    use animator_ui::config::ANIM_CUDA_MEM_LIMIT_MB;
    use animator_ui::ep::SessionOpts;
    use animator_ui::sprite::Mulberry32;
    use std::time::Instant;

    let dll = setup_dll_paths(cfg);
    let choice = EpChoice::parse(&arg("--ep").unwrap_or_else(|| "auto".into()));
    let out = arg("--out").map(PathBuf::from).unwrap_or_else(|| exports_dir().join("onnx"));
    std::fs::create_dir_all(&out)?;
    let has = |f: &str| std::env::args().any(|a| a == f);
    let sizes: Vec<usize> = arg("--sizes").unwrap_or_else(|| "32,16".into()).split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let mut rep = Report(Vec::new());
    rep.line(format!("DLL search: {}", dll.summary()));
    let anim = OnnxAnimator::with_dir(animator_ui::files::animator_dir(&cfg));
    let opts = SessionOpts { cuda_mem_limit_mb: cfg.anim_cuda_mem_limit_mb.unwrap_or(ANIM_CUDA_MEM_LIMIT_MB), ..Default::default() };
    anim.set_providers(choice.order(), opts);
    let t = anim.tables()?;
    let t0 = Instant::now();
    let ep_tag = anim.with_engine(&Ctl::default(), |e| Ok(e.ep.label().to_lowercase()))?;
    rep.line(format!("animation model: {} step {:?}; {} (first load {:.1} s)", t.checkpoint, t.step, anim.status(), t0.elapsed().as_secs_f64()));
    for f in anim.fallbacks() {
        rep.line(format!("  skipped: {f}"));
    }
    for kind in Kind::ALL {
        for a in standard_actions(kind) {
            if t.n_frames(kind, &a.name) != Some(a.frames) {
                rep.line(format!("WARNING: {} {}: the app says {} frames, the model {:?}", kind.label(), a.name, a.frames, t.n_frames(kind, &a.name)));
            }
        }
    }
    let empty = t.empty as u16;
    let v2pal: animator_ui::sprite::Palette = Arc::new(t.palette.clone());
    let sprite_of = |tokens: &[u16], s: usize| Sprite::new(s, tokens.iter().map(|&x| if x == empty { -1 } else { x as i16 }).collect(), v2pal.clone());
    let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(anim.dir.join("parity_ref.json"))?)?;
    let cases = doc["cases"].as_array().cloned().unwrap_or_default();
    let seed = doc["seed"].as_u64().unwrap_or(1) as u32;
    let temp = doc["temperature"].as_f64().unwrap_or(0.8);
    anim.set_settings(OnnxSettings { temperature: temp, seed, ..Default::default() });
    let toks = |v: &serde_json::Value| -> Vec<u16> { v.as_array().into_iter().flatten().map(|x| x.as_u64().unwrap_or(0) as u16).collect() };
    let only: Option<Vec<usize>> = arg("--case-sizes").map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    let cases: Vec<serde_json::Value> = cases.into_iter().filter(|c| only.as_ref().is_none_or(|o| o.contains(&(c["size"].as_u64().unwrap_or(0) as usize)))).collect();

    // ---------------------------------------------------------------- 1 + 2: parity and teacher-forced quality
    let mut bpp: std::collections::BTreeMap<(usize, &str), Vec<f64>> = Default::default();
    if !has("--skip-parity") {
        rep.line(format!("--- parity with PyTorch (parity_ref.json: seed {seed}, temperature {temp}) and quality vs the farm"));
        let (mut max_bits, mut n_frames_same, mut n_frames, mut px_same, mut px_all) = (0f64, 0usize, 0usize, 0usize, 0usize);
        for case in &cases {
            let name = case["character"].as_str().unwrap_or_default();
            let kind = Kind::parse(case["kind"].as_str().unwrap_or_default()).unwrap_or(Kind::Human);
            let s = case["size"].as_u64().unwrap_or(32) as usize;
            let action = case["action"].as_str().unwrap_or("walk").to_string();
            let nfr = case["n_frames"].as_u64().unwrap_or(8) as usize;
            let reference = toks(&case["reference"]);
            let data = farm(cfg, name)?;
            let gt = |sp: Option<Sprite>| -> anyhow::Result<Vec<u16>> { Ok(anim.reference(&sp.ok_or_else(|| anyhow::anyhow!("{name}: missing farm frame"))?)?.tokens) };
            let gt_views: Vec<Vec<u16>> = AZIMUTHS.iter().map(|&az| gt(data.view(s, az))).collect::<anyhow::Result<_>>()?;
            let gt_frames: Vec<Vec<u16>> = data.frames(s, &action, 90).into_iter().map(|f| gt(Some(f))).collect::<anyhow::Result<_>>()?;
            let mut allowed: Vec<usize> = reference.iter().map(|&x| x as usize).collect();
            allowed.push(t.empty);
            allowed.sort_unstable();
            allowed.dedup();
            let ki = t.kind_index(kind)?;
            // (a) teacher forced: the 8 views (previous = reference) and the action at 90° (previous = the farm frame)
            let (bits_v, bits_a) = anim.with_engine(&Ctl::default(), |eng| {
                let conds: Vec<_> = AZIMUTHS.iter().map(|&az| t.cond(kind, 0, "view", az, 0)).collect::<anyhow::Result<_>>()?;
                let rows: Vec<RowSpec> = conds.iter().zip(&gt_views).map(|(&c, g)| RowSpec { cond: c, reference: &reference, prev: &reference, allowed: &allowed, force: Some(g) }).collect();
                let mut rngs: Vec<Mulberry32> = rows.iter().map(|_| Mulberry32::new(0)).collect();
                let v = eng.run_rows(s, &rows, &mut rngs, 1.0, None, &mut |_| {})?.bits;
                let conds: Vec<_> = (0..nfr).map(|f| t.cond(kind, 0, &action, 90, f)).collect::<anyhow::Result<_>>()?;
                let prevs: Vec<&[u16]> = (0..nfr).map(|f| if f == 0 { gt_views[2].as_slice() } else { gt_frames[f - 1].as_slice() }).collect();
                let rows: Vec<RowSpec> = (0..nfr).map(|f| RowSpec { cond: conds[f], reference: &reference, prev: prevs[f], allowed: &allowed, force: Some(&gt_frames[f]) }).collect();
                let mut rngs: Vec<Mulberry32> = rows.iter().map(|_| Mulberry32::new(0)).collect();
                let a = eng.run_rows(s, &rows, &mut rngs, 1.0, None, &mut |_| {})?.bits;
                Ok((v, a))
            })?;
            let want: Vec<f64> = case["forced"].as_array().into_iter().flatten().map(|r| r["bits"].as_f64().unwrap_or(f64::NAN)).collect();
            let got: Vec<f64> = bits_v.iter().chain(&bits_a).copied().collect();
            let d = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            max_bits = max_bits.max(d);
            let px = (s * s) as f64;
            let mv = bits_v.iter().sum::<f64>() / bits_v.len() as f64 / px;
            let ma = bits_a.iter().sum::<f64>() / bits_a.len() as f64 / px;
            bpp.entry((s, "view")).or_default().push(mv);
            bpp.entry((s, "anim")).or_default().push(ma);
            // (b) sampled with the reference's random numbers
            let view_i = t.action_index("view")?;
            let views_az: Vec<u16> = case["sampled"]["views_az"].as_array().into_iter().flatten().map(|x| x.as_u64().unwrap_or(0) as u16).collect();
            let want_views: Vec<Vec<u16>> = case["sampled"]["views"].as_array().into_iter().flatten().map(toks).collect();
            let want_frames: Vec<Vec<u16>> = case["sampled"]["frames"].as_array().into_iter().flatten().map(toks).collect();
            let (views, frames) = anim.with_engine(&Ctl::default(), |eng| {
                let conds: Vec<_> = views_az.iter().map(|&az| t.cond(kind, 0, "view", az, 0)).collect::<anyhow::Result<_>>()?;
                let rows: Vec<RowSpec> = conds.iter().map(|&c| RowSpec { cond: c, reference: &reference, prev: &reference, allowed: &allowed, force: None }).collect();
                let mut rngs: Vec<Mulberry32> = views_az.iter().map(|&az| Mulberry32::new(row_seed(seed, ki, view_i, t.az_index(az).unwrap_or(0), s))).collect();
                let views = eng.run_rows(s, &rows, &mut rngs, temp, None, &mut |_| {})?.tokens;
                let mut rng = [Mulberry32::new(row_seed(seed, ki, t.action_index(&action)?, t.az_index(90)?, s))];
                let mut prev = views[0].clone();
                let mut frames = Vec::new();
                for f in 0..want_frames.len() {
                    let c = t.cond(kind, 0, &action, 90, f)?;
                    let row = [RowSpec { cond: c, reference: &reference, prev: &prev, allowed: &allowed, force: None }];
                    let fr = eng.run_rows(s, &row, &mut rng, temp, None, &mut |_| {})?.tokens.remove(0);
                    prev = fr.clone();
                    frames.push(fr);
                }
                Ok((views, frames))
            })?;
            let mut same = Vec::new();
            for (a, b) in views.iter().chain(&frames).zip(want_views.iter().chain(&want_frames)) {
                let k = a.iter().zip(b).filter(|(x, y)| x == y).count();
                px_same += k;
                px_all += a.len();
                n_frames += 1;
                n_frames_same += (k == a.len()) as usize;
                same.push(match a.iter().zip(b).position(|(x, y)| x != y) {
                    None => "=".to_string(),
                    Some(p) => format!("{:.0}%@{p}", 100.0 * k as f64 / a.len() as f64),
                });
            }
            let gt_chain: Vec<&Vec<u16>> = views_az
                .iter()
                .map(|&az| &gt_views[AZIMUTHS.iter().position(|&x| x == az).unwrap_or(0)])
                .chain(gt_frames.iter().take(frames.len()))
                .collect();
            let agree: Vec<String> = views.iter().chain(&frames).zip(&gt_chain).map(|(a, g)| format!("{:.2}", agreement(a, g, empty).0)).collect();
            rep.line(format!(
                "{name} ({}, {s}px): teacher-forced bits/px views {mv:.3}, {action} {ma:.3} (max |bits - PyTorch| per frame {d:.4}); sampled vs PyTorch [{}]; pixel agreement with the farm: Rust [{}] PyTorch {}",
                kind.label(),
                same.join(" "),
                agree.join(" "),
                case["sampled"]["pixel_agreement_with_farm"]
            ));
        }
        rep.line(format!(
            "parity summary: teacher-forced bits per frame within {max_bits:.4} of PyTorch; sampled frames identical to PyTorch: {n_frames_same}/{n_frames} (pixels {:.2} %; '=' identical, 'x%@p' = share equal, first difference at pixel p)",
            100.0 * px_same as f64 / px_all.max(1) as f64
        ));
        for ((s, what), v) in &bpp {
            rep.line(format!(
                "  teacher-forced bits/pixel, {what} at {s}px: {:.3} (mean of {} characters; training's val set: {:.3})",
                v.iter().sum::<f64>() / v.len() as f64,
                v.len(),
                training_val_bpp(*s, what)
            ));
        }
    }

    // ---------------------------------------------------------------- 2: sheets / GIFs through the animator API
    let refk0 = Some(0);
    let ctl = Ctl::default();
    if !has("--skip-gifs") {
        rep.line("--- farm (GT) vs model: sheets and GIFs through the animator API (reference = the original sprite)");
        for case in &cases {
            let name = case["character"].as_str().unwrap_or_default();
            let kind = Kind::parse(case["kind"].as_str().unwrap_or_default()).unwrap_or(Kind::Human);
            let s = case["size"].as_u64().unwrap_or(32) as usize;
            let action = case["action"].as_str().unwrap_or("walk").to_string();
            let sprite = sprite_of(&toks(&case["reference"]), s);
            let data = farm(cfg, name)?;
            let ts = Instant::now();
            let views = anim.draw_views(&sprite, kind, &AZIMUTHS, refk0, &ctl)?;
            let req = AnimRequest { kind, action: action.clone(), azimuth: 0, n_frames: 0, size: s };
            let dirs = OnnxWith { anim: &anim, ctl, refk: refk0 }.animate_directions(&sprite, &req, &AZIMUTHS)?;
            let secs = ts.elapsed().as_secs_f64();
            let gt_views: Vec<Sprite> = AZIMUTHS.iter().filter_map(|&az| data.view(s, az)).collect();
            let tk = |sp: &Sprite| anim.reference(sp).map(|r| r.tokens);
            let mut v_ag = Vec::new();
            for (m, g) in views.iter().zip(&gt_views) {
                v_ag.push(agreement(&tk(m)?, &tk(g)?, empty));
            }
            let mut a_ag = Vec::new();
            let mut rows = Vec::new();
            for (az, a) in AZIMUTHS.iter().zip(&dirs) {
                let g = data.frames(s, &action, *az);
                for (m, gg) in a.frames.iter().zip(&g) {
                    a_ag.push(agreement(&tk(m)?, &tk(gg)?, empty));
                }
                rows.push(g);
                rows.push(a.frames.clone());
            }
            let mean = |v: &[(f64, f64)]| (v.iter().map(|x| x.0).sum::<f64>() / v.len().max(1) as f64, v.iter().map(|x| x.1).sum::<f64>() / v.len().max(1) as f64);
            let (vp, vi) = mean(&v_ag);
            let (ap, ai) = mean(&a_ag);
            rep.line(format!(
                "{name} ({}, {s}px): views vs farm: pixels {vp:.2}, silhouette IoU {vi:.2} (IoU at 0/90/180/270°: {:.2} {:.2} {:.2} {:.2}); {action} in 8 directions: pixels {ap:.2}, IoU {ai:.2}; drawn in {secs:.1} s",
                kind.label(),
                v_ag[0].1,
                v_ag[2].1,
                v_ag[4].1,
                v_ag[6].1
            ));
            let mut first = vec![sprite.clone()];
            first.extend(std::iter::repeat_n(sprite.blank_like(), 7));
            sheet_png(&[first, gt_views.clone(), views.clone()], 4, 4, &out.join(format!("{name}_{s}_views_ref_farm_model.png")))?;
            sheet_png(&rows, 4, 4, &out.join(format!("{name}_{s}_{action}_8dirs_farm_model.png")))?;
            gif(&dirs[2].frames, 8, 8.0, true, &out.join(format!("{name}_{s}_{action}_a090_model.gif")))?;
            gif(&dirs[6].frames, 8, 8.0, true, &out.join(format!("{name}_{s}_{action}_a270_model.gif")))?;
            gif(&data.frames(s, &action, 90), 8, 8.0, true, &out.join(format!("{name}_{s}_{action}_a090_farm.gif")))?;
        }
    }

    // ---------------------------------------------------------------- 3: timings
    if !has("--skip-timing") {
        rep.line(format!("--- timings ({})", anim.status()));
        for &s in &sizes {
            let Some(case) = cases.iter().find(|c| c["size"].as_u64() == Some(s as u64) && c["kind"] == "human") else { continue };
            let base = sprite_of(&toks(&case["reference"]), s);
            let st0 = anim.settings();
            let dirs8: Vec<usize> = arg("--dirs8-sizes").map(|x| x.split(',').filter_map(|v| v.trim().parse().ok()).collect()).unwrap_or_else(|| sizes.clone());
            let mut runs = vec![("1 direction", vec![90u16])];
            if dirs8.contains(&s) {
                runs.push(("8 directions", AZIMUTHS.to_vec()));
            }
            for (label, azs) in runs {
                // another seed: nothing comes from the cache
                anim.set_settings(OnnxSettings { seed: st0.seed + 1000 + azs.len() as u32, ..st0 });
                anim.totals.lock().unwrap().batches.clear();
                let ts = Instant::now();
                let clips: Vec<(String, u16)> = azs.iter().map(|&a| ("walk".to_string(), a)).collect();
                anim.draw_clips(&base, Kind::Human, &clips, refk0, &ctl)?;
                let wall = ts.elapsed().as_secs_f64();
                let b = anim.totals.lock().unwrap().batches.clone();
                let (views, frames) = (&b[0], &b[1..]);
                let nf = frames.len().max(1) as f64;
                let fp = frames.iter().map(|x| x.prefill_secs).sum::<f64>() / nf;
                let fs = frames.iter().map(|x| x.step_secs).sum::<f64>() / nf;
                let steps = frames.iter().map(|x| x.steps).sum::<usize>() as f64 / nf;
                rep.line(format!(
                    "{s}px walk, {label} (batch {}): {wall:.1} s = views {:.2} s + {} frames x {:.2} s (prefill {:.0} ms + {steps:.0} steps x {:.2} ms)",
                    views.rows,
                    views.prefill_secs + views.step_secs,
                    frames.len(),
                    fp + fs,
                    fp * 1e3,
                    fs / steps.max(1.0) * 1e3
                ));
            }
            anim.set_settings(st0);
        }
    }

    // ---------------------------------------------------------------- 4: the fight scene by the model
    if !has("--skip-scene") {
        rep.line("--- fight scene drawn by the model");
        let script = fight();
        let mut scenes: Vec<(String, Sprite)> = Vec::new();
        for &s in &sizes {
            if let Some(case) = cases.iter().find(|c| c["size"].as_u64() == Some(s as u64) && c["kind"] == "human") {
                scenes.push((case["character"].as_str().unwrap_or_default().to_string(), sprite_of(&toks(&case["reference"]), s)));
            }
        }
        if let Some(caption) = arg("--caption") {
            let mut h = Headless::new(cfg.clone(), setup_dll_paths(cfg));
            h.load(choice)?;
            let palette = h.palette().unwrap();
            for &s in &sizes {
                let r = h.generate(&caption, GenParams { n: 1, size: s, seed: 3, ..Default::default() })?;
                scenes.push((format!("{}_v2", slug(&caption)), Sprite::new(s, r.images[0].clone(), palette.clone())));
            }
        }
        for (name, sprite) in scenes {
            let s = sprite.size;
            let ts = Instant::now();
            match build(&anim, &sprite, Kind::Human, s, &script, false) {
                Ok(scene) => {
                    let frames: Vec<Vec<u8>> = (0..scene.shots.len()).map(|i| scene.rgba(i)).collect();
                    let p = out.join(format!("{name}_{s}_fight_scene_model.gif"));
                    animator_ui::export::gif_rgba(&frames, scene.stage_w, scene.size, &scene.delays_cs(8.0), 4, true, &p)?;
                    let clips: Vec<Vec<Sprite>> = scene.clips.iter().map(|c| c.frames.clone()).collect();
                    sheet_png(&clips, 4, 2, &out.join(format!("{name}_{s}_fight_scene_clips_model.png")))?;
                    rep.line(format!("{name} {s}px: scene built in {:.1} s ({} clips, {} shots) -> {}", ts.elapsed().as_secs_f64(), scene.clips.len(), scene.shots.len(), p.display()));
                    if name.ends_with("_v2") {
                        sheet_png(&[vec![sprite.clone()]], 8, 0, &out.join(format!("{name}_{s}_sprite.png")))?;
                        let walk = anim.animate(&sprite, &AnimRequest { kind: Kind::Human, action: "walk".into(), azimuth: 270, n_frames: 0, size: s })?;
                        gif(&walk.frames, 8, 8.0, true, &out.join(format!("{name}_{s}_walk_a270_model.gif")))?;
                    }
                }
                Err(why) => rep.line(format!("{name} {s}px: no scene: {}", why.summary())),
            }
        }
    }
    let p = out.join(format!("selftest_onnx_{ep_tag}.txt"));
    std::fs::write(&p, rep.0.join("\n"))?;
    println!("report: {}", p.display());
    Ok(())
}
