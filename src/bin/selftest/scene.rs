use std::path::PathBuf;
use std::sync::Arc;
use animator_ui::animator::{Kind};
use animator_ui::config::{Config, exports_dir};

use super::*;

/// `--scene-scan [--roots a;b] [--limit N] [--list file]`: which library humans get the fight scene (the app's own
/// check: the four actions present + the dataset QC), CPU only, no model. With `--gif name,name [--fps 8]` it
/// instead writes the scene GIFs (x4, 16 / 24 / 32 px) of those entries to exports\scene\ (no full scan).
pub(super) fn scene_scan(cfg: &Config) -> anyhow::Result<()> {
    use animator_ui::animator::library::{LibraryAnimator, LibraryData, scan};
    use animator_ui::animator::scene::{build, fight};
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    let roots: Vec<PathBuf> = arg("--roots").map(|r| r.split(';').map(PathBuf::from).collect()).unwrap_or_else(|| cfg.library_roots());
    let script = fight();
    if let Some(names) = arg("--gif") {
        // named entries only: no full scan
        let out = exports_dir().join("scene");
        std::fs::create_dir_all(&out)?;
        let fps: f32 = arg("--fps").and_then(|s| s.parse().ok()).unwrap_or(8.0);
        for name in names.split(',').map(str::trim) {
            let Some(e) = roots.iter().filter(|r| r.join(name).join("pix").join("sprites.npz").is_file()).find_map(|r| {
                animator_ui::animator::library::read_entry(r.join(name), &r.file_name().unwrap_or_default().to_string_lossy(), true)
            }) else {
                println!("{name}: not found in the library roots");
                continue;
            };
            let data = Arc::new(LibraryData::load(&e)?);
            for size in [16usize, 24, 32] {
                let Some(sprite) = data.base_sprite(size) else { continue };
                match build(&LibraryAnimator { data: data.clone() }, &sprite, Kind::Human, size, &script, true) {
                    Ok(scene) => {
                        let frames: Vec<Vec<u8>> = (0..scene.shots.len()).map(|i| scene.rgba(i)).collect();
                        let p = out.join(format!("{}_{size}_fight_scene.gif", e.name));
                        animator_ui::export::gif_rgba(&frames, scene.stage_w, scene.size, &scene.delays_cs(fps), 4, true, &p)?;
                        println!(
                            "scene gif: {} ({} frames, {:.1} s at {fps} fps; shown clips touching the border: {:?})",
                            p.display(),
                            scene.shots.len(),
                            scene.total_ticks() / fps,
                            scene.clipped_frames()
                        );
                    }
                    Err(why) => println!("{name} {size}: no scene: {}", why.summary()),
                }
            }
        }
        return Ok(());
    }
    let t = std::time::Instant::now();
    let entries = scan(&roots);
    println!("library scan: {} entries in {:.2} s ({})", entries.len(), t.elapsed().as_secs_f64(), roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", "));
    let limit: usize = arg("--limit").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let humans: Vec<_> = entries.iter().filter(|e| e.kind == Some(Kind::Human)).take(limit).collect();
    #[derive(Default)]
    struct Tally {
        ok: Vec<String>,
        missing: usize,
        load_errors: usize,
        failed: BTreeMap<String, usize>,
        failed_only: BTreeMap<String, usize>,
        clipped_shown: usize,
        clipped_examples: Vec<String>,
        ticks: Vec<f32>,
    }
    let tally = Mutex::new(Tally::default());
    let t = std::time::Instant::now();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    let chunk = humans.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for part in humans.chunks(chunk) {
            let (tally, script) = (&tally, &script);
            s.spawn(move || {
                for e in part {
                    let Ok(data) = LibraryData::load(e) else {
                        tally.lock().unwrap().load_errors += 1;
                        continue;
                    };
                    let data = Arc::new(data);
                    let Some(sprite) = data.base_sprite(32) else { continue };
                    let r = build(&LibraryAnimator { data: data.clone() }, &sprite, Kind::Human, 32, script, true);
                    let mut t = tally.lock().unwrap();
                    match r {
                        Ok(scene) => {
                            t.ok.push(format!("{}/{}", e.root, e.name));
                            t.ticks.push(scene.total_ticks());
                            let c = scene.clipped_frames();
                            if !c.is_empty() {
                                t.clipped_shown += 1;
                                if t.clipped_examples.len() < 8 {
                                    t.clipped_examples.push(format!("{} [{}]", e.name, c.join(", ")));
                                }
                            }
                        }
                        Err(why) => {
                            if !why.missing.is_empty() || !why.errors.is_empty() {
                                t.missing += 1;
                            }
                            for q in &why.failed_qc {
                                *t.failed.entry(q.action.clone()).or_default() += 1;
                            }
                            if why.failed_qc.len() == 1 {
                                *t.failed_only.entry(why.failed_qc[0].action.clone()).or_default() += 1;
                            }
                        }
                    }
                }
            });
        }
    });
    let mut t2 = tally.into_inner().unwrap();
    t2.ok.sort();
    let n = humans.len();
    println!("humans checked: {n} in {:.1} s ({threads} threads), load errors (half-written?): {}", t.elapsed().as_secs_f64(), t2.load_errors);
    println!("fight scene available: {} ({:.1} %)", t2.ok.len(), 100.0 * t2.ok.len() as f64 / n.max(1) as f64);
    println!("missing an action / frames: {}", t2.missing);
    println!("dropped by the dataset QC, per action: {:?}", t2.failed);
    println!("  ... and the only reason: {:?}", t2.failed_only);
    println!("available but a shown clip touches the border in some frame: {} e.g. {:?}", t2.clipped_shown, t2.clipped_examples);
    if !t2.ticks.is_empty() {
        let mut tk = t2.ticks.clone();
        tk.sort_by(|a, b| a.total_cmp(b));
        println!("scene length in ticks: min {:.1} median {:.1} max {:.1} (8 fps: {:.1} s median)", tk[0], tk[tk.len() / 2], tk[tk.len() - 1], tk[tk.len() / 2] / 8.0);
    }
    println!("first available: {:?}", t2.ok.iter().take(12).collect::<Vec<_>>());
    if let Some(p) = arg("--list") {
        std::fs::write(&p, t2.ok.join("\n"))?;
        println!("list: {p}");
    }
    Ok(())
}
