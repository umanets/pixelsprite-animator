use animator_ui::export::{sheet_png};
use animator_ui::mix::{ConceptKind, VariantKind};
use animator_ui::save;
use animator_ui::sprite::Sprite;
use animator_ui::worker::headless::Headless;
use animator_ui::worker::{GenResult, MixOptions, Mode};
use super::*;


/// A potion-bottle silhouette (cork, neck, round body) for the mask checks.
pub(super) fn potion_mask(s: usize) -> Vec<u8> {
    let cx = (s as f64 - 1.0) / 2.0;
    let mut m = vec![0u8; s * s];
    for y in 0..s {
        for x in 0..s {
            let (u, v) = ((x as f64 - cx) / s as f64, (y as f64 + 0.5) / s as f64);
            let on = if v > 0.12 && v < 0.22 {
                u.abs() < 0.1
            } else if (0.22..0.42).contains(&v) {
                u.abs() < 0.08
            } else if (0.42..0.92).contains(&v) {
                u.hypot((v - 0.66) * 1.05) < 0.30
            } else {
                false
            };
            m[y * s + x] = u8::from(on);
        }
    }
    m
}

pub(super) fn opaque(px: &[i16]) -> Vec<u8> {
    px.iter().map(|&v| u8::from(v >= 0)).collect()
}

/// Stage 1 against the web demo's features: mixing, silhouettes, saving (+ load back), timings.
pub(super) fn stage1(h: &mut Headless, out: &std::path::Path, describe: &str) -> anyhow::Result<()> {
    std::fs::create_dir_all(out)?;
    let palette = h.palette().unwrap();
    let mut fails: Vec<String> = Vec::new();
    let mut check = |ok: bool, what: String| {
        println!("  [{}] {what}", if ok { "ok" } else { "FAIL" });
        if !ok {
            fails.push(what);
        }
    };
    println!("--- stage 1: mixing, silhouettes, saving ({describe})");
    if h.lexicon().is_none() {
        anyhow::bail!("no lexicon in models/v2 (lexicon.json / lexicon.f16): caption mixing is off");
    }
    println!("  lexicon: {} entries; example captions: {}", h.lexicon().unwrap().entries.len(), h.presets().len());
    let show = |r: &GenResult| r.variants.as_ref().map(|v| v.iter().map(|x| format!("{}: {}", x.kind.key(), x.label)).collect::<Vec<_>>().join(" | ")).unwrap_or_default();

    // 1. a two-object mix: variants, determinism, more than one layout, different from the plain caption
    let mixed = h.request(req("burger skull", 16, 8, 3))?;
    println!("  burger skull 16 s3: {} · {} · {:.2} s, {} chunk(s) of up to {} rows", mixed.summary, show(&mixed), mixed.secs, mixed.stats.chunks, mixed.stats.max_rows);
    let labels: std::collections::HashSet<String> = mixed.variants.iter().flatten().map(|v| v.label.clone()).collect();
    check(mixed.concepts == ["burger:object", "skull:object"], format!("concepts {:?}", mixed.concepts));
    check(mixed.variants.as_ref().map(Vec::len) == Some(8) && labels.len() >= 3, format!("8 variants, {} distinct", labels.len()));
    check(mixed.caption_exact == Some(false), "the 'one caption' variant averages the word vectors (fresh demo page)".into());
    let again = h.request(req("burger skull", 16, 8, 3))?;
    check(again.images == mixed.images, "the same seed draws the same mixed sprites".into());
    let mut plain_req = req("burger skull", 16, 8, 3);
    plain_req.mix.enabled = vec![VariantKind::Caption];
    let plain = h.request(plain_req)?;
    check(plain.variants.is_none() && plain.images != mixed.images, "only 'one caption' enabled = the plain caption (no mix), other sprites".into());
    // 2. three concepts with an attribute (4 rows per sprite), and a word-kind override
    let attr = h.request(req("fried burger skull", 16, 8, 5))?;
    println!("  fried burger skull 16 s5: {} · {}", attr.summary, show(&attr));
    check(attr.concepts == ["fried:attribute", "burger:object", "skull:object"], format!("concepts {:?}", attr.concepts));
    check(attr.stats.max_rows <= 24, format!("chunks within the 24-row budget (max {})", attr.stats.max_rows));
    let fs = h.request(req("frost sword", 16, 8, 0))?;
    let mut fo = req("frost sword", 16, 8, 0);
    fo.mix = MixOptions { overrides: [("frost".to_string(), ConceptKind::Object)].into_iter().collect(), ..MixOptions::default() };
    let fo = h.request(fo)?;
    println!("  frost sword 16 s0: {} · {}", fs.summary, show(&fs));
    println!("  frost (object) sword 16 s0: {} · {}", fo.summary, show(&fo));
    let kinds = |r: &GenResult| r.variants.iter().flatten().map(|v| v.kind).collect::<std::collections::HashSet<_>>();
    check(kinds(&fs).iter().all(|k| matches!(k, VariantKind::Blend | VariantKind::Caption)) && fs.caption_exact == Some(true), "attribute + object: blend / one caption only (the example's own vector)".into());
    check(kinds(&fo).iter().any(|k| k.ordered()), "frost flipped to an object: layouts appear".into());
    // 3. silhouettes: "draw" fills exactly the mask; "one shape" keeps the source's silhouette
    let mask = potion_mask(16);
    let mut d = req("red potion", 16, 8, 5);
    d.mode = Mode::Draw;
    d.params.mask = Some(mask.clone());
    let drawn = h.request(d)?;
    check(drawn.images.iter().all(|im| opaque(im) == mask), format!("draw: all 8 sprites fill exactly the {} px silhouette", mask.iter().filter(|&&v| v != 0).count()));
    let mut d = req("red potion", 16, 1, 5);
    d.mode = Mode::Draw;
    check(h.request(d).is_err(), "draw without a silhouette is refused".into());
    let mut sm = req("golden crown", 16, 6, 6);
    sm.mode = Mode::Same;
    let same = h.request(sm)?;
    let src = same.source.clone().unwrap_or_default();
    check(!src.is_empty() && same.images.iter().all(|im| opaque(im) == opaque(&src)), "one shape: 6 colourings of the source's exact silhouette".into());
    // 24 px: the mask on the 32 canvas window
    let mut d24 = req("blue mushroom", 24, 2, 1);
    d24.mode = Mode::Draw;
    d24.params.mask = Some(potion_mask(24));
    let r24 = h.request(d24)?;
    check(r24.images.iter().all(|im| opaque(im) == potion_mask(24)), "draw at 24 px (window of the 32 canvas)".into());

    // 4. saving with the demo's names; the files load back as the same sprites
    let sprites: Vec<Sprite> = mixed.images.iter().map(|px| Sprite::new(16, px.clone(), palette.clone())).collect();
    let mut ok = true;
    for (k, s) in sprites.iter().enumerate() {
        for scale in [1u32, 4] {
            let p = out.join(save::sprite_file_name("burger skull", 16, 3, &save::sprite_tag(k), scale));
            save::save_sprite_png(s, scale, &p)?;
            let (back, f) = Sprite::from_png(&p)?;
            ok &= back.size == 16 && f == scale as usize && back.rgba() == s.rgba();
        }
    }
    let sheet = out.join(save::sheet_file_name("burger skull", 16, 3, 1));
    save::save_sheet_png(&sprites, 1, &sheet)?;
    let img = image::open(&sheet)?.to_rgba8();
    for (k, s) in sprites.iter().enumerate() {
        let cell: Vec<u8> = (0..16u32).flat_map(|y| (0..16u32).map(move |x| (x, y))).flat_map(|(x, y)| img.get_pixel(k as u32 * 16 + x, y).0).collect();
        ok &= Sprite::from_rgba(16, 16, &cell).map(|(b, _)| b.rgba() == s.rgba()).unwrap_or(false);
    }
    check(ok, format!("saved 8 sprites (x1, x4) and the sheet as {} …; all load back pixel-identical", out.join(save::sprite_file_name("burger skull", 16, 3, "00", 1)).display()));
    let src_sprite = Sprite::new(16, src.clone(), palette.clone());
    let p = out.join(save::sprite_file_name("golden crown", 16, 6, "src", 1));
    save::save_sprite_png(&src_sprite, 1, &p)?;
    check(Sprite::from_png(&p).map(|(b, _)| b.rgba() == src_sprite.rgba()).unwrap_or(false), format!("source sprite saved as {}", p.file_name().unwrap_or_default().to_string_lossy()));
    let mut rows = vec![sprites.clone()];
    rows.push(plain.images.iter().map(|px| Sprite::new(16, px.clone(), palette.clone())).collect());
    rows.push(attr.images.iter().map(|px| Sprite::new(16, px.clone(), palette.clone())).collect());
    rows.push(fo.images.iter().map(|px| Sprite::new(16, px.clone(), palette.clone())).collect());
    rows.push(drawn.images.iter().map(|px| Sprite::new(16, px.clone(), palette.clone())).collect());
    rows.push(std::iter::once(&src).chain(&same.images).map(|px| Sprite::new(16, px.clone(), palette.clone())).collect());
    sheet_png(&rows, 6, 6, &out.join("stage1_overview_16.png"))?;
    println!("  overview: {} (rows: burger skull mixed / one caption / fried burger skull / frost(object) sword / drawn potion / one shape)", out.join("stage1_overview_16.png").display());

    // 5. timings: mixed vs the plain caption
    if !std::env::args().any(|a| a == "--skip-stage1-timing") {
        for (size, n) in [(16usize, 8usize), (32, 4)] {
            let mut line = format!("  timing {size}x{size}, {n} sprites:");
            for (label, enabled) in [("plain caption", vec![VariantKind::Caption]), ("mixed", VariantKind::ALL.to_vec())] {
                let mut r = req("burger skull", size, n, 11);
                r.mix.enabled = enabled;
                let g = h.request(r)?;
                line += &format!(
                    " {label} {:.2} s ({} chunk(s), up to {} rows, {} steps, {:.2} ms/step{});",
                    g.secs,
                    g.stats.chunks,
                    g.stats.max_rows,
                    g.stats.steps,
                    g.stats.step_secs / g.stats.steps.max(1) as f64 * 1e3,
                    if g.stats.rejected > 0 { format!(", redrew {}", g.stats.rejected) } else { String::new() }
                );
                if size == 32 && label == "mixed" {
                    sheet_png(&[g.images.iter().map(|px| Sprite::new(32, px.clone(), palette.clone())).collect()], 4, 4, &out.join("stage1_burger_skull_32_s11.png"))?;
                }
            }
            println!("{line}");
        }
    }
    if fails.is_empty() {
        println!("stage 1: all checks passed");
        Ok(())
    } else {
        anyhow::bail!("stage 1: {} check(s) failed: {}", fails.len(), fails.join("; "))
    }
}

