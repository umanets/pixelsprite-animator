//! Headless check of the whole pipeline (same code as the app's worker):
//!
//! ```text
//!  target\release\selftest.exe [--ep auto|cuda|directml|cpu] [--captions "knight,cat,parrot"] [--sizes 16,32]
//!                              [--seed 1] [--n16 8] [--n32 4] [--out exports\selftest]
//! ```
//!
//! Prints the execution provider (and why others were skipped), timings per caption and size, the encoder's
//! agreement with the demo's precomputed caption vectors, and writes one PNG sheet per size (8x nearest).
//!
//! Then the stage-1 checks (`--skip-stage1` to leave them out): caption mixing (variants, determinism, an
//! attribute, a word-kind override), silhouettes ("draw" and "one shape": every sprite fills exactly the
//! silhouette), saving with the demo's file names and loading the PNGs back, and mixed vs plain timings.
//!
//! ```text
//!  target\release\selftest.exe --parity parity\ref_node_growing.json [--cases parity\cases.json] [--only a,b]
//!                              [--ep cuda|cpu] [--out exports\parity]
//! ```
//!
//! Draws the cases of `parity/cases.json` exactly as the web demo's reference run (`parity/demo_parity.mjs`, the
//! demo's own engine.js / mix.js under Node) and compares the sprites pixel by pixel.
//!
//! ```text
//!  target\release\selftest.exe --onnx-anim [--ep cuda|cpu|...] [--sizes 32,16] [--caption knight]
//!                              [--skip-parity] [--skip-gifs] [--skip-timing] [--skip-scene] [--out exports\onnx]
//!                              [--dirs8-sizes 16] [--case-sizes 16]
//! ```
//!
//! The animation model (see `onnx_anim`): parity with PyTorch (models\animator\parity_ref.json), quality against
//! the 3D-farm frames of held-out characters, GIFs / sheets / fight scenes in exports\onnx\, timings.

mod anchor;
mod onnx_anim;
mod parity;
mod scene;
mod stage1;

use anchor::*;
use onnx_anim::*;
use parity::*;
use scene::*;
use stage1::*;

use std::path::PathBuf;
use std::sync::Arc;

use animator_ui::animator::mock::MockAnimator;
use animator_ui::animator::{AZIMUTHS, AnimRequest, Animator, Kind};
use animator_ui::classify::classify_english;
use animator_ui::config::{Config, app_root, exports_dir, slug};
use animator_ui::ep::{EpChoice, setup_dll_paths};
use animator_ui::export::{gif, sheet_png};
use animator_ui::model::{GenParams, demo_rows_gpu};
use animator_ui::sprite::Sprite;
use animator_ui::text::cosine;
use animator_ui::worker::headless::Headless;
use animator_ui::worker::GenRequest;

pub(crate) fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1).cloned())
}

fn main() -> anyhow::Result<()> {
    let cfg = Config::load();
    if std::env::args().any(|a| a == "--scene-scan") {
        return scene_scan(&cfg);
    }
    if std::env::args().any(|a| a == "--anchor-eval") {
        return anchor_eval(&cfg);
    }
    if std::env::args().any(|a| a == "--onnx-anim") {
        return onnx_anim(&cfg);
    }
    if std::env::args().any(|a| a == "--download-cudnn") {
        let dir = animator_ui::ep::download_cuda_libs(&setup_dll_paths(&cfg), &mut |s| println!("{s}"))?;
        println!("CUDA DLLs in {}", dir.display());
        return Ok(());
    }
    let dll = setup_dll_paths(&cfg);
    println!("DLL search: {}", dll.summary());
    let choice = EpChoice::parse(&arg("--ep").unwrap_or_else(|| "auto".into()));
    let captions: Vec<String> = arg("--captions").unwrap_or_else(|| "knight,cat,parrot".into()).split(',').map(|s| s.trim().to_string()).collect();
    let sizes: Vec<usize> = arg("--sizes").unwrap_or_else(|| "16,32".into()).split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let seed: u32 = arg("--seed").and_then(|s| s.parse().ok()).unwrap_or(1);
    let n16: usize = arg("--n16").and_then(|s| s.parse().ok()).unwrap_or(8);
    let n32: usize = arg("--n32").and_then(|s| s.parse().ok()).unwrap_or(4);
    let out = arg("--out").map(PathBuf::from).unwrap_or_else(|| exports_dir().join("selftest"));
    std::fs::create_dir_all(&out)?;

    let mut h = Headless::new(cfg.clone(), dll);
    let t = std::time::Instant::now();
    let (describe, fallbacks) = h.load(choice)?;
    println!("loaded in {:.1} s on {describe}", t.elapsed().as_secs_f64());
    for f in &fallbacks {
        println!("  fallback: {f}");
    }

    // encoder vs the demo's precomputed vectors (presets.json, computed in Python)
    let presets = app_root().join("..").join("release").join("github").join("docs").join("demo").join("presets.json");
    if let Ok(text) = std::fs::read_to_string(&presets) {
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let mut sims = Vec::new();
        for it in v["items"].as_array().into_iter().flatten().take(6) {
            let (Some(tx), Some(b64)) = (it["text"].as_str(), it["vector"].as_str()) else { continue };
            let want = b64_floats(b64);
            let got = h.encode(tx)?;
            sims.push(format!("{tx:?} {:.4}", cosine(&want, &got)));
        }
        println!("encoder vs presets.json (cosine): {}", sims.join(", "));
    }

    if let Some(file) = arg("--compare-vectors") {
        // vectors of parity/encode.mjs (the demo's text.js on onnxruntime-web) vs the app's encoder
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&file)?)?;
        for (text, it) in v.as_object().into_iter().flatten() {
            let want = animator_ui::worker::b64_floats(it["vector"].as_str().unwrap_or_default());
            let got = h.encode(text)?;
            let ids_js: Vec<u64> = it["ids"].as_array().into_iter().flatten().filter_map(|x| x.as_u64()).collect();
            let ids: Vec<u64> = h.token_ids(text)?.into_iter().map(u64::from).collect();
            let same = want.iter().zip(&got).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
            let maxd = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            println!("{text:?}: token ids {} · {same}/{} numbers bit-identical, max |diff| {maxd:.2e}, cosine {:.9}", if ids == ids_js { "identical" } else { "DIFFER" }, want.len(), cosine(&want, &got));
        }
        return Ok(());
    }
    if let Some(reference) = arg("--parity") {
        let o = arg("--out").map(PathBuf::from).unwrap_or_else(|| exports_dir().join("parity"));
        return parity(&mut h, &PathBuf::from(reference), &o, &describe);
    }

    let palette = h.palette().unwrap();
    for &size in &sizes {
        let mut rows = Vec::new();
        for cap in &captions {
            let n = if size == 16 { n16 } else { n32 };
            let p = GenParams { n, size, seed, ..Default::default() };
            let r = h.generate(cap, p)?;
            let per_step = if r.stats.steps > 0 { r.stats.step_secs / r.stats.steps as f64 * 1e3 } else { 0.0 };
            println!(
                "{size}x{size} {cap:?}: {} sprites in {:.2} s (encode {:.3} s, {} steps, {:.2} ms/step, batch {}, redrawn {}) on {} · class: {:?} ({})",
                r.images.len(),
                r.secs,
                r.encode_secs,
                r.stats.steps,
                per_step,
                r.params.batch,
                r.stats.rejected,
                r.ep.label(),
                r.class.kind.map(Kind::label),
                r.class.reason
            );
            let sprites: Vec<Sprite> = r.images.iter().map(|px| Sprite::new(size, px.clone(), palette.clone())).collect();
            for (i, s) in sprites.iter().enumerate() {
                let stripes = stripe_score(s);
                if stripes > 0.6 {
                    println!("  WARNING sprite {i}: {:.0}% of rows repeat the row above (stripes?)", stripes * 100.0);
                }
            }
            rows.push(sprites);
        }
        let path = out.join(format!("selftest_{size}_s{seed}.png"));
        sheet_png(&rows, 8, 8, &path)?;
        println!("  sheet: {}", path.display());
        // mock animation of the first sprite, to exercise the exporters
        if let Some(first) = rows.first().and_then(|r| r.first()) {
            let kind = classify_english(&captions[0]).kind.unwrap_or(Kind::Human);
            let anim = MockAnimator.animate(first, &AnimRequest { kind, action: if kind == Kind::Bird { "fly".into() } else { "walk".into() }, azimuth: 90, n_frames: 8, size })?;
            let g = out.join(format!("mock_{}_{size}.gif", slug(&captions[0])));
            gif(&anim.frames, 8, 8.0, true, &g)?;
            let dirs = MockAnimator.animate_directions(first, &AnimRequest { kind, action: "idle".into(), azimuth: 0, n_frames: 4, size }, &AZIMUTHS)?;
            let rows: Vec<Vec<Sprite>> = dirs.into_iter().map(|a| a.frames).collect();
            sheet_png(&rows, 4, 4, &out.join(format!("mock_dirs_{}_{size}.png", slug(&captions[0]))))?;
            println!("  mock: {} ({} frames), {}", g.display(), anim.frames.len(), anim.note);
        }
    }

    if let Some(words) = arg("--map") {
        for w in words.split(',') {
            println!("map {w:?} -> {:?}", h.map_word(w.trim()).map(|m| (m.english, m.score)));
        }
    }
    if let Some(caps) = arg("--classify") {
        for c in caps.split('|') {
            let k = h.classify(c.trim());
            println!("classify {c:?} -> {:?} ({})", k.kind.map(Kind::label), k.reason);
        }
    }
    if !std::env::args().any(|a| a == "--skip-stage1") {
        stage1(&mut h, &out.join("stage1"), &describe)?;
    }
    let _ = Arc::new(());
    Ok(())
}

/// A request as the UI sends it: the demo's defaults (split guidance 1 / 2, T 1, top-p 1, the filter) and the
/// demo's row budget with WebGPU.
pub(crate) fn req(caption: &str, size: usize, n: usize, seed: u32) -> GenRequest {
    GenRequest::new(0, caption, GenParams { n, size, seed, rows: demo_rows_gpu(size), ..Default::default() })
}
