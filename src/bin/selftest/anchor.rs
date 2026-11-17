use std::path::PathBuf;
use std::sync::Arc;
use animator_ui::animator::{Kind};
use animator_ui::config::{Config, exports_dir, slug};
use animator_ui::ep::{EpChoice, setup_dll_paths};
use animator_ui::export::sheet_png;
use animator_ui::model::{GenParams};
use animator_ui::sprite::Sprite;
use animator_ui::worker::headless::Headless;
use super::*;


// ------------------------------------------------------------------------------------------------ --anchor-eval
/// The fixed set drawn per character: actions at 90 and 270 degrees and the views at 45 / 90 / 180 / 270.
pub(super) fn anchor_actions(kind: Kind) -> &'static [&'static str] {
    match kind {
        Kind::Human => &["walk", "run", "punch"],
        Kind::Quadruped => &["walk", "run"],
        Kind::Bird => &["fly", "glide"],
    }
}

pub(super) const ANCHOR_VIEWS: [u16; 4] = [45, 90, 180, 270];
pub(super) const ANCHOR_AZ: [u16; 2] = [90, 270];

pub(super) struct EvalChar {
    set: &'static str,
    name: String,
    kind: Kind,
    sprite: Sprite,
    /// Ground truth (val characters).
    gt: Option<Arc<animator_ui::animator::library::LibraryData>>,
}

pub(super) struct Drawn {
    views: Vec<Sprite>,
    clips: Vec<(String, u16, Vec<Sprite>)>,
}

pub(super) fn anchor_label(o: animator_ui::animator::onnx::AnchorOpts) -> String {
    use animator_ui::animator::onnx::{AnchorPalette, AnchorRef};
    if !o.enabled {
        return "off".into();
    }
    let r = match o.refk {
        AnchorRef::Original => "orig",
        AnchorRef::FrontRender => "front",
        AnchorRef::Same => "same",
    };
    let p = match o.palette {
        AnchorPalette::Input => "inpal",
        AnchorPalette::Anchor => "anchpal",
    };
    format!("on-{r}-{p}")
}

/// `--anchor-eval`: the experimental anchor option (off vs on, same seeds) on held-out val characters of the farm
/// (ground truth available) and on sprites drawn by the app's text-to-sprite model. See `animator_ui::anchor_eval`
/// for the metrics. Writes anchor_metrics.csv (resumable with `--resume`), anchor_table.md, sheets and GIFs.
///
/// ```text
///  selftest --anchor-eval [--ep cuda|cpu] [--sizes 16,32] [--variants] [--n-app 8] [--scenes 2] [--limit N]
///                         [--skip-val] [--skip-app] [--resume] [--from-csv] [--out exports\anchor]
///                         [--seed 1] [--gen-ep cpu] [--anim-mem MB] [--no-media] [--only-modes off,on-front-inpal]
/// ```
pub(super) fn anchor_eval(cfg: &Config) -> anyhow::Result<()> {
    use animator_ui::anchor_eval::{Acc, Colours, Row, tables};
    use animator_ui::animator::onnx::{AnchorOpts, AnchorPalette, AnchorRef, Ctl, OnnxAnimator, OnnxSettings};
    use animator_ui::animator::scene::{build, fight};
    use animator_ui::config::ANIM_CUDA_MEM_LIMIT_MB;
    use animator_ui::ep::SessionOpts;
    use std::io::Write;

    let has = |f: &str| std::env::args().any(|a| a == f);
    let out = arg("--out").map(PathBuf::from).unwrap_or_else(|| exports_dir().join("anchor"));
    std::fs::create_dir_all(&out)?;
    let csv_path = out.join("anchor_metrics.csv");
    let read_rows = |p: &std::path::Path| -> Vec<Row> { std::fs::read_to_string(p).unwrap_or_default().lines().filter_map(Row::parse).collect() };
    // --only-modes off,on-front-inpal: a table of just these modes (anchor_table_final.md), e.g. after a variants run
    let only: Option<Vec<String>> = arg("--only-modes").map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let write_tables = |rows: &[Row], out: &std::path::Path| -> anyhow::Result<()> {
        let rows: Vec<Row> = rows.iter().filter(|r| only.as_ref().is_none_or(|o| o.contains(&r.mode) || r.mode == "gt")).cloned().collect();
        let t = tables(&rows, "off");
        std::fs::write(out.join(if only.is_some() { "anchor_table_final.md" } else { "anchor_table.md" }), &t)?;
        println!("{t}");
        Ok(())
    };
    if has("--from-csv") {
        return write_tables(&read_rows(&csv_path), &out);
    }
    let sizes: Vec<usize> = arg("--sizes").unwrap_or_else(|| "16,32".into()).split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let seed: u32 = arg("--seed").and_then(|s| s.parse().ok()).unwrap_or(1);
    let n_app: usize = arg("--n-app").and_then(|s| s.parse().ok()).unwrap_or(8);
    let n_scenes: usize = arg("--scenes").and_then(|s| s.parse().ok()).unwrap_or(2);
    let limit: usize = arg("--limit").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let media = !has("--no-media");
    let dll = setup_dll_paths(cfg);
    println!("DLL search: {}", dll.summary());
    let choice = EpChoice::parse(&arg("--ep").unwrap_or_else(|| "auto".into()));

    // the modes: off, then the option with the defaults (or all reference-type x palette variants)
    let mut modes: Vec<AnchorOpts> = vec![AnchorOpts::OFF];
    if has("--variants") {
        for refk in [AnchorRef::Original, AnchorRef::FrontRender] {
            for palette in [AnchorPalette::Input, AnchorPalette::Anchor] {
                modes.push(AnchorOpts { enabled: true, refk, palette });
            }
        }
    } else {
        modes.push(AnchorOpts::ON);
    }

    let mut done: std::collections::HashSet<(String, String, String)> = Default::default();
    if has("--resume") {
        for r in read_rows(&csv_path) {
            done.insert((r.set, r.name, r.mode));
        }
    } else {
        std::fs::write(&csv_path, format!("# {}\n", Row::csv_header()))?;
    }
    let mut csv = std::fs::OpenOptions::new().append(true).open(&csv_path)?;

    // ---- the input sprites: app sprites first (the text-to-sprite model is released before the animation model loads)
    let mut chars: Vec<EvalChar> = Vec::new();
    if !has("--skip-app") {
        let gen_choice = EpChoice::parse(&arg("--gen-ep").unwrap_or_else(|| "auto".into()));
        let mut h = Headless::new(cfg.clone(), setup_dll_paths(cfg));
        h.load(gen_choice)?;
        let v2pal = h.palette().unwrap();
        let captions = ["knight", "ninja", "wizard", "robot", "cat", "dog", "parrot", "owl", "pirate", "warrior", "horse", "eagle", "mage", "archer"];
        for &s in &sizes {
            let mut got = 0;
            for cap in captions {
                if got >= n_app {
                    break;
                }
                let r = h.generate(cap, GenParams { n: 1, size: s, seed: 3, ..Default::default() })?;
                let Some(kind) = r.class.kind else { continue };
                chars.push(EvalChar { set: "app", name: format!("{}_{s}", slug(cap)), kind, sprite: Sprite::new(s, r.images[0].clone(), v2pal.clone()), gt: None });
                got += 1;
            }
            let kinds: Vec<&str> = chars.iter().filter(|c| c.sprite.size == s).map(|c| c.kind.label()).collect();
            println!("app sprites {s}px: {got} ({})", kinds.join(","));
        }
        drop(h);
    }

    // ---- the animation model
    let anim = OnnxAnimator::new();
    let opts = SessionOpts { cuda_mem_limit_mb: arg("--anim-mem").and_then(|s| s.parse().ok()).or(cfg.anim_cuda_mem_limit_mb).unwrap_or(ANIM_CUDA_MEM_LIMIT_MB), ..Default::default() };
    anim.set_providers(choice.order(), opts);
    let t = anim.tables()?;
    let ep_tag = anim.with_engine(&Ctl::default(), |e| Ok(e.ep.label().to_lowercase()))?;
    println!("animation model: {}", anim.status());
    let empty = t.empty as u16;
    let v2: animator_ui::sprite::Palette = Arc::new(t.palette.clone());
    let sprite_of = |tokens: &[u16], s: usize| Sprite::new(s, tokens.iter().map(|&x| if x == empty { -1 } else { x as i16 }).collect(), v2.clone());
    if !has("--skip-val") {
        let p = animator_ui::config::app_root().join("anchor_eval").join("val_chars.json");
        let doc: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
        for c in doc {
            let (name, size) = (c["name"].as_str().unwrap_or_default().to_string(), c["size"].as_u64().unwrap_or(0) as usize);
            if !sizes.contains(&size) {
                continue;
            }
            let kind = Kind::parse(c["kind"].as_str().unwrap_or_default()).unwrap_or(Kind::Human);
            let tokens: Vec<u16> = c["tokens"].as_array().into_iter().flatten().map(|x| x.as_u64().unwrap_or(0) as u16).collect();
            chars.push(EvalChar { set: "val", sprite: sprite_of(&tokens, size), gt: Some(farm(cfg, &name)?), name, kind });
        }
    }
    // --limit N: at most N characters per (set, size, kind)
    if limit != usize::MAX {
        let mut seen: std::collections::HashMap<(String, usize, &str), usize> = Default::default();
        chars.retain(|c| {
            let n = seen.entry((c.set.to_string(), c.sprite.size, c.kind.label())).or_default();
            *n += 1;
            *n <= limit
        });
    }
    anim.set_settings(OnnxSettings { seed, ..Default::default() });
    let ctl = Ctl::default();
    let tk = |sp: &Sprite| anim.reference(sp).map(|r| r.tokens);
    let mut rep = Report(Vec::new());
    rep.line(format!(
        "anchor eval on {ep_tag}: {} characters, modes {:?}, seed {seed}, T {}",
        chars.len(),
        modes.iter().map(|m| anchor_label(*m)).collect::<Vec<_>>(),
        anim.settings().temperature
    ));
    let ts_all = std::time::Instant::now();

    for (ci, c) in chars.iter().enumerate() {
        let s = c.sprite.size;
        let Some(input) = Colours::of(&c.sprite) else { continue };
        let todo = modes.iter().any(|m| !done.contains(&(c.set.to_string(), c.name.clone(), anchor_label(*m))));
        // ground truth frames (and their metrics as the "gt" row: the floor of the drift metrics)
        let gt_views_v2: Vec<Sprite> = c
            .gt
            .as_ref()
            .map(|g| ANCHOR_VIEWS.iter().filter_map(|&az| g.view(s, az)).filter_map(|f| tk(&f).ok()).map(|t| sprite_of(&t, s)).collect())
            .unwrap_or_default();
        let gt_sprites = |action: &str, az: u16| -> Vec<Sprite> {
            c.gt.as_ref().map(|g| g.frames(s, action, az).iter().filter_map(|f| tk(f).ok()).map(|t| sprite_of(&t, s)).collect()).unwrap_or_default()
        };
        if c.gt.is_some() && !done.contains(&(c.set.to_string(), c.name.clone(), "gt".to_string())) {
            let mut acc = Acc::default();
            acc.add_frames(&gt_views_v2, &input, false);
            for &az in &ANCHOR_AZ {
                for a in anchor_actions(c.kind) {
                    acc.add_frames(&gt_sprites(a, az), &input, true);
                }
            }
            let row = Row { set: c.set.into(), size: s, kind: c.kind.label().into(), name: c.name.clone(), mode: "gt".into(), v: acc.finish() };
            writeln!(csv, "{}", row.csv())?;
        }
        if !todo {
            continue;
        }
        let ts = std::time::Instant::now();
        let mut drawn: Vec<(String, Drawn)> = Vec::new();
        for m in &modes {
            let lab = anchor_label(*m);
            if done.contains(&(c.set.to_string(), c.name.clone(), lab.clone())) {
                continue;
            }
            anim.set_settings(OnnxSettings { seed, anchor: *m, ..Default::default() });
            let mut clips: Vec<(String, u16)> = Vec::new();
            for &az in &ANCHOR_AZ {
                for a in anchor_actions(c.kind) {
                    clips.push((a.to_string(), az));
                }
            }
            let views = anim.draw_views(&c.sprite, c.kind, &ANCHOR_VIEWS, None, &ctl)?;
            let frames = anim.draw_clips(&c.sprite, c.kind, &clips, None, &ctl)?;
            let d = Drawn { views, clips: clips.into_iter().zip(frames).map(|((a, az), f)| (a, az, f)).collect() };
            let mut acc = Acc::default();
            acc.add_frames(&d.views, &input, false);
            for (_, _, f) in &d.clips {
                acc.add_frames(f, &input, true);
            }
            if c.gt.is_some() {
                for (m, g) in d.views.iter().zip(&gt_views_v2) {
                    acc.add_gt(m, g);
                }
                for (a, az, f) in &d.clips {
                    for (m, g) in f.iter().zip(&gt_sprites(a, *az)) {
                        acc.add_gt(m, g);
                    }
                }
            }
            let row = Row { set: c.set.into(), size: s, kind: c.kind.label().into(), name: c.name.clone(), mode: lab.clone(), v: acc.finish() };
            writeln!(csv, "{}", row.csv())?;
            csv.flush()?;
            drawn.push((lab, d));
        }
        rep.line(format!("[{}/{}] {} {} {s}px ({}) drawn in {:.1} s", ci + 1, chars.len(), c.set, c.name, c.kind.label(), ts.elapsed().as_secs_f64()));
        if media && !drawn.is_empty() {
            // sheet rows, top to bottom: [GT views], every mode's views, then per action [GT], every mode (all at 90 deg)
            let mut rows: Vec<Vec<Sprite>> = Vec::new();
            if c.gt.is_some() {
                let mut first = vec![c.sprite.clone()];
                first.extend(gt_views_v2.iter().cloned());
                rows.push(first);
            }
            for (_, d) in &drawn {
                let mut r = vec![c.sprite.clone()];
                r.extend(d.views.iter().cloned());
                rows.push(r);
            }
            for a in anchor_actions(c.kind) {
                if c.gt.is_some() {
                    rows.push(gt_sprites(a, 90));
                }
                for (_, d) in &drawn {
                    if let Some((_, _, f)) = d.clips.iter().find(|x| x.0 == *a && x.1 == 90) {
                        rows.push(f.clone());
                    }
                }
            }
            let scale = if s == 16 { 5 } else { 3 };
            sheet_png(&rows, scale, 3, &out.join(format!("{}_{}_sheet.png", c.set, c.name)))?;
            // side-by-side GIFs of the first two actions at 90 deg: [GT] off on ...
            for a in anchor_actions(c.kind).iter().take(2) {
                let mut cols: Vec<Vec<Sprite>> = Vec::new();
                if c.gt.is_some() {
                    cols.push(gt_sprites(a, 90));
                }
                for (_, d) in &drawn {
                    if let Some((_, _, f)) = d.clips.iter().find(|x| x.0 == *a && x.1 == 90) {
                        cols.push(f.clone());
                    }
                }
                let n = cols.iter().map(Vec::len).min().unwrap_or(0);
                if n == 0 {
                    continue;
                }
                let pw = s + 2;
                let w = cols.len() * pw;
                let frames: Vec<Vec<u8>> = (0..n)
                    .map(|i| {
                        let mut img = vec![0u8; w * s * 4];
                        for (k, col) in cols.iter().enumerate() {
                            let rgba = col[i].rgba();
                            for y in 0..s {
                                let o = (y * w + k * pw) * 4;
                                img[o..o + s * 4].copy_from_slice(&rgba[y * s * 4..(y + 1) * s * 4]);
                            }
                        }
                        img
                    })
                    .collect();
                animator_ui::export::gif_rgba(&frames, w, s, &[12], if s == 16 { 8 } else { 5 }, true, &out.join(format!("{}_{}_{a}_a090.gif", c.set, c.name)))?;
            }
        }
    }

    // ---- fight scenes (humans): clip metrics of the scene's own clips, one GIF per mode
    if n_scenes > 0 {
        let script = fight();
        for &s in &sizes {
            for set in ["val", "app"] {
                let mut n = 0;
                for c in chars.iter().filter(|c| c.set == set && c.sprite.size == s && c.kind == Kind::Human) {
                    if n >= n_scenes {
                        break;
                    }
                    n += 1;
                    let Some(input) = Colours::of(&c.sprite) else { continue };
                    for m in &modes {
                        let lab = anchor_label(*m);
                        if done.contains(&(format!("scene-{set}"), c.name.clone(), lab.clone())) {
                            continue;
                        }
                        anim.set_settings(OnnxSettings { seed, anchor: *m, ..Default::default() });
                        let ts = std::time::Instant::now();
                        match build(&anim, &c.sprite, Kind::Human, s, &script, false) {
                            Ok(scene) => {
                                let mut acc = Acc::default();
                                for clip in &scene.clips {
                                    acc.add_frames(&clip.frames, &input, true);
                                }
                                let row = Row { set: format!("scene-{set}"), size: s, kind: c.kind.label().into(), name: c.name.clone(), mode: lab.clone(), v: acc.finish() };
                                writeln!(csv, "{}", row.csv())?;
                                csv.flush()?;
                                rep.line(format!("scene {} {s}px {lab}: {:.1} s, {} clips", c.name, ts.elapsed().as_secs_f64(), scene.clips.len()));
                                if media {
                                    let frames: Vec<Vec<u8>> = (0..scene.shots.len()).map(|i| scene.rgba(i)).collect();
                                    animator_ui::export::gif_rgba(&frames, scene.stage_w, scene.size, &scene.delays_cs(8.0), 4, true, &out.join(format!("scene_{set}_{}_{lab}.gif", c.name)))?;
                                }
                            }
                            Err(why) => rep.line(format!("scene {} {s}px: {}", c.name, why.summary())),
                        }
                    }
                }
            }
        }
    }
    anim.set_settings(OnnxSettings::default());
    rep.line(format!("done in {:.0} s", ts_all.elapsed().as_secs_f64()));
    drop(csv);
    std::fs::write(out.join(format!("anchor_eval_{ep_tag}.log")), rep.0.join("\n"))?;
    write_tables(&read_rows(&csv_path), &out)?;
    println!("outputs in {}", out.display());
    Ok(())
}
