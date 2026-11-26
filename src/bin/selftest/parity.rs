use std::path::PathBuf;
use animator_ui::config::{app_root};
use animator_ui::export::{sheet_png};
use animator_ui::mix::{ConceptKind, VariantKind};
use animator_ui::model::{GenParams};
use animator_ui::sprite::Sprite;
use animator_ui::worker::headless::Headless;
use animator_ui::worker::{GenRequest, MixOptions, Mode};
use super::*;

/// `--parity <reference.json>`: the cases of parity/cases.json drawn by the app (the worker's code path) and
/// compared with the web demo's sprites from `parity/demo_parity.mjs`.
pub(super) fn parity(h: &mut Headless, reference: &std::path::Path, out: &std::path::Path, describe: &str) -> anyhow::Result<()> {
    std::fs::create_dir_all(out)?;
    let refj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(reference)?)?;
    let cases_path = arg("--cases").map(PathBuf::from).unwrap_or_else(|| app_root().join("parity").join("cases.json"));
    let cases: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(&cases_path)?)?;
    let only: Option<Vec<String>> = arg("--only").map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let palette = h.palette().unwrap();
    let mut lines = vec![format!("app: {describe}; reference: {} ({}, {} cache)", reference.display(), refj["ort"].as_str().unwrap_or("?"), refj["cache"].as_str().unwrap_or("?"))];
    println!("{}", lines[0]);
    let (mut same_sprites, mut all_sprites, mut same_cases, mut n_cases) = (0usize, 0usize, 0usize, 0usize);
    let px = |v: &serde_json::Value| -> Vec<i16> { v.as_array().into_iter().flatten().map(|x| x.as_i64().unwrap_or(-2) as i16).collect() };
    for rc in refj["cases"].as_array().into_iter().flatten() {
        let name = rc["name"].as_str().unwrap_or_default();
        if only.as_ref().is_some_and(|o| !o.iter().any(|x| x == name)) {
            continue;
        }
        let Some(c) = cases.iter().find(|c| c["name"] == name) else {
            println!("{name}: not in {}", cases_path.display());
            continue;
        };
        let f = |k: &str, d: f64| c[k].as_f64().unwrap_or(d);
        let size = c["size"].as_u64().unwrap_or(16) as usize;
        let mode = match c["mode"].as_str().unwrap_or("new") {
            "same" => Mode::Same,
            "draw" => Mode::Draw,
            _ => Mode::New,
        };
        let mask = c["mask"].as_array().map(|rows| {
            let rows: Vec<&str> = rows.iter().map(|r| r.as_str().unwrap_or_default()).collect();
            (0..size * size).map(|p| u8::from(rows.get(p / size).and_then(|r| r.chars().nth(p % size)) == Some('#'))).collect::<Vec<u8>>()
        });
        let params = GenParams {
            n: c["n"].as_u64().unwrap_or(8) as usize,
            size,
            seed: c["seed"].as_u64().unwrap_or(0) as u32,
            g_shape: f("gShape", 1.0),
            g_colour: f("gColour", 2.0),
            temperature: f("temperature", 1.0),
            top_p: f("topP", 1.0),
            filter: c["filter"].as_bool().unwrap_or(true),
            rows: rc["rows"].as_u64().unwrap_or(0) as usize,
            mask,
            ..Default::default()
        };
        let mix = MixOptions {
            outline: None,
            enabled: c["variants"].as_array().map(|v| v.iter().filter_map(|k| VariantKind::parse(k.as_str().unwrap_or_default())).collect()).unwrap_or_else(|| VariantKind::ALL.to_vec()),
            overrides: c["overrides"].as_object().map(|o| o.iter().filter_map(|(k, v)| Some((k.clone(), ConceptKind::parse(v.as_str()?)?))).collect()).unwrap_or_default(),
            learned: Default::default(),
            encoder_ready: c["encoderReady"].as_bool().unwrap_or(false),
        };
        let r = h.request(GenRequest { id: 0, caption: c["text"].as_str().unwrap_or_default().to_string(), params, mode, mix })?;
        let want: Vec<Vec<i16>> = rc["images"].as_array().into_iter().flatten().map(px).collect();
        let mut got_all: Vec<&Vec<i16>> = r.images.iter().collect();
        let mut want_all: Vec<Vec<i16>> = want.clone();
        if let (Some(s), Some(ws)) = (&r.source, rc["source"].as_array()) {
            got_all.insert(0, s);
            want_all.insert(0, ws.iter().map(|x| x.as_i64().unwrap_or(-2) as i16).collect());
        }
        let per: Vec<String> = got_all
            .iter()
            .zip(&want_all)
            .map(|(g, w)| match g.iter().zip(w).position(|(a, b)| a != b) {
                None if g.len() == w.len() => "=".to_string(),
                None => "len".to_string(),
                Some(p) => format!("{:.0}%@{p}", 100.0 * g.iter().zip(w).filter(|(a, b)| a == b).count() as f64 / g.len() as f64),
            })
            .collect();
        let k = per.iter().filter(|x| *x == "=").count();
        let labels_ref: Vec<String> = rc["variants"].as_array().into_iter().flatten().map(|v| v.as_str().unwrap_or_default().to_string()).collect();
        let labels_app: Vec<String> = r.variants.iter().flatten().map(|v| format!("{}: {}", v.kind.key(), v.label)).collect();
        let labels_ok = labels_ref == labels_app;
        same_sprites += k;
        all_sprites += per.len();
        n_cases += 1;
        let identical = k == per.len() && got_all.len() == want_all.len() && labels_ok;
        same_cases += usize::from(identical);
        let line = format!(
            "{name}: {}/{} sprites identical [{}]{} · variants {} · {:.1} s, {} chunk(s) of up to {} rows{}",
            k,
            want_all.len(),
            per.join(" "),
            if r.source.is_some() { " (source first)" } else { "" },
            if labels_ref.is_empty() && labels_app.is_empty() { "none (one caption)".to_string() } else if labels_ok { "identical".to_string() } else { format!("DIFFER: app {labels_app:?} vs demo {labels_ref:?}") },
            r.secs,
            r.stats.chunks,
            r.stats.max_rows,
            if r.caption_exact == Some(false) { " · one caption = averaged word vectors" } else { "" }
        );
        println!("{line}");
        lines.push(line);
        if !identical || arg("--sheets").is_some() {
            let to = |v: &Vec<i16>| Sprite::new(size, v.clone(), palette.clone());
            let rows = vec![want_all.iter().map(|v| to(v)).collect::<Vec<_>>(), got_all.iter().map(|v| to(*v)).collect()];
            sheet_png(&rows, if size == 16 { 6 } else { 3 }, 4, &out.join(format!("{name}_demo_vs_app.png")))?;
        }
    }
    let sum = format!("parity summary: {same_cases}/{n_cases} cases identical, {same_sprites}/{all_sprites} sprites pixel-identical ('=' identical, 'x%@p' = share of equal pixels, first difference at pixel p)");
    println!("{sum}");
    lines.push(sum);
    let tag = describe.split_whitespace().next().unwrap_or("ep").to_lowercase();
    let stem = reference.file_stem().unwrap_or_default().to_string_lossy().to_string();
    let p = out.join(format!("parity_{tag}_vs_{stem}.txt"));
    std::fs::write(&p, lines.join("\n"))?;
    println!("report: {}", p.display());
    Ok(())
}

