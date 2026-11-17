use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use crate::mix::{self, ConceptKind, VariantKind};
use crate::model::{GenParams, GenStats, Progress, Strategy, View, crop_view};
use super::state::{Prepared, State};
use super::{GenRequest, GenResult, MixOptions, Mode, Msg, VariantInfo};

impl State {
    /// `prepareMix` of app.js: the caption as a mix plan with its vectors, or None to draw it as one caption.
    /// Unknown words get the kind of their nearest lexicon entry (added to `mo.learned`).
    pub(super) fn prepare_mix(&mut self, text: &str, mo: &mut MixOptions) -> Result<Option<Prepared>> {
        let Some(lex) = self.lexicon.clone() else { return Ok(None) };
        if text.is_empty() {
            return Ok(None);
        }
        let mut ready = mo.encoder_ready;
        let kinds: HashMap<String, ConceptKind> = mo.learned.iter().chain(mo.overrides.iter()).map(|(k, v)| (k.clone(), *v)).collect();
        let mut concepts = mix::parse(text, &lex, &kinds).concepts;
        let preset_needs_encoder = self.presets.contains_key(text) && !ready && concepts.iter().any(|c| !lex.index.contains_key(&c.text));
        if concepts.len() < 2 || preset_needs_encoder {
            return Ok(None);
        }
        let mut vecs: HashMap<String, Vec<f32>> = HashMap::new();
        for c in concepts.iter_mut().filter(|c| !c.known) {
            ready = true;
            let v = self.encoder()?.encode(&c.text)?;
            if !mo.learned.contains_key(&c.text) {
                mo.learned.insert(c.text.clone(), mix::nearest_kind(&v, &lex).0);
            }
            if !mo.overrides.contains_key(&c.text) {
                c.kind = mo.learned[&c.text];
            }
            vecs.insert(c.text.clone(), v);
        }
        let Some(pl) = mix::plan(&concepts, &mo.enabled) else { return Ok(None) };
        let mut vectors = Vec::with_capacity(pl.concepts.len());
        for c in &pl.concepts {
            vectors.push(match (vecs.get(&c.text), lex.vector_of(&c.text)) {
                (Some(v), _) => v.clone(),
                (None, Some(v)) => v.to_vec(),
                (None, None) => {
                    ready = true;
                    self.encoder()?.encode(&c.text)?
                }
            });
        }
        let (mut caption, mut caption_exact) = (None, false);
        if pl.kinds.contains(&VariantKind::Caption) {
            if let Some(v) = self.presets.get(text) {
                (caption, caption_exact) = (Some(v.clone()), true);
            } else if ready {
                (caption, caption_exact) = (Some(self.encoder()?.encode(text)?), true);
            } else {
                caption = Some(mix::mean_caption(&vectors));
            }
        }
        Ok(Some(Prepared { pl, vectors, caption, caption_exact }))
    }

    pub(super) fn generate(&mut self, req: GenRequest) -> Result<GenResult> {
        let caption = req.caption.trim().to_string();
        let t0 = Instant::now();
        let mut params = req.params.clone();
        let mut mo = req.mix.clone();
        if req.mode == Mode::Draw && !params.mask.as_ref().is_some_and(|m| m.iter().any(|&v| v != 0)) {
            bail!("Draw a silhouette first.");
        }
        if req.mode != Mode::Draw {
            params.mask = None;
        }
        let prepared = if req.mode == Mode::New { self.prepare_mix(&caption, &mut mo)? } else { None };
        params.vector = if prepared.is_some() { None } else { self.vector_for(&caption)? };
        let encode_secs = t0.elapsed().as_secs_f64();
        let model = self.model.as_mut().ok_or_else(|| anyhow::anyhow!("the model is not loaded"))?;
        if params.batch == 0 {
            params.batch = model.default_batch(params.size);
        }
        let core = model.core.clone();
        if let Some(v) = &params.vector {
            if v.len() != core.text_dim {
                bail!("caption vector has {} numbers, the model wants {}", v.len(), core.text_dim);
            }
        }
        // strategies: the demo's strategyFor (a mix per variant, split guidance for "one caption") or split guidance
        let shared = Arc::new(core.strategy_split(params.vector.as_deref(), params.g_shape, params.g_colour));
        let mut variants: Option<Vec<mix::Variant>> = None;
        let mut mix_rows: Option<(Vec<Arc<Vec<f32>>>, Arc<Vec<f32>>, Option<Arc<Strategy>>)> = None;
        if let Some(m) = &prepared {
            let v = mix::draw_variants(&m.pl, params.n, params.seed);
            let infos: Vec<VariantInfo> = v.iter().map(|x| VariantInfo { kind: x.kind, label: mix::label(x, &m.pl, &caption), describe: mix::describe(x, &m.pl) }).collect();
            self.tx.send(Msg::Planned { id: req.id, variants: infos, learned: mo.learned.clone() }).ok();
            (self.repaint)();
            let conds = m.vectors.iter().map(|v| Arc::new(core.condition(Some(v)))).collect();
            let as_caption = m.caption.as_ref().map(|c| Arc::new(core.strategy_split(Some(c), params.g_shape, params.g_colour)));
            mix_rows = Some((conds, Arc::new(core.condition(None)), as_caption));
            variants = Some(v);
        }
        let strategy_for = |id: usize, view: View| -> Arc<Strategy> {
            match (&prepared, &variants, &mix_rows) {
                (Some(m), Some(v), Some((conds, null, as_caption))) => {
                    if v[id].kind == VariantKind::Caption {
                        return as_caption.clone().expect("the caption variant has a vector");
                    }
                    Arc::new(Strategy::mix(conds.clone(), null.clone(), mix::variant_maps(&v[id], &m.pl, view.s, view.o, view.w)))
                }
                _ => shared.clone(),
            }
        };
        let ep = model.ep;
        let tx = self.tx.clone();
        let repaint = &self.repaint;
        let id = req.id;
        let last = std::cell::Cell::new(Instant::now() - Duration::from_secs(1));
        let progress = |first_card: usize, p: &Progress| {
            let n_px = p.view.s * p.view.s;
            if last.get().elapsed() >= Duration::from_millis(50) || p.t + 1 == n_px {
                last.set(Instant::now());
                let partial = p.ids.iter().zip(p.canvases).map(|(&i, cv)| (i, crop_view(cv, p.view))).collect();
                let _ = tx.send(Msg::Progress { id, done: p.done, total: p.total, t: p.t, n_px, first_card, partial });
                repaint();
            }
        };
        let mut source = None;
        let mut stats = GenStats::default();
        let mut seed = params.seed;
        if req.mode == Mode::Same {
            let p1 = GenParams { n: 1, batch: 1, mask: None, ..params.clone() };
            let r = model.generate_with(&p1, &strategy_for, &self.cancel, &mut |p| progress(0, p))?;
            params.mask = Some(r.images[0].iter().map(|&v| u8::from(v >= 0)).collect());
            source = r.images.into_iter().next();
            stats = r.stats;
            seed += 1;
        }
        let p2 = GenParams { seed, ..params.clone() };
        let first = usize::from(req.mode == Mode::Same);
        let out = model.generate_with(&p2, &strategy_for, &self.cancel, &mut |p| progress(first, p))?;
        stats.sampled += out.stats.sampled;
        stats.rejected += out.stats.rejected;
        stats.rounds += out.stats.rounds;
        stats.steps += out.stats.steps;
        stats.step_secs += out.stats.step_secs;
        stats.chunks += out.stats.chunks;
        stats.max_rows = stats.max_rows.max(out.stats.max_rows);
        let class = self.classify(&caption);
        let infos = prepared.as_ref().zip(variants.as_ref()).map(|(m, v)| {
            v.iter().map(|x| VariantInfo { kind: x.kind, label: mix::label(x, &m.pl, &caption), describe: mix::describe(x, &m.pl) }).collect::<Vec<_>>()
        });
        // the demo's line under the run
        let mut summary = match (&prepared, &infos) {
            (Some(m), Some(inf)) => {
                let distinct: std::collections::HashSet<&str> = inf.iter().map(|i| i.label.as_str()).collect();
                format!("mix of {} · {} variants · seed {}", m.pl.concepts.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(", "), distinct.len(), params.seed)
            }
            _ => format!("{} · seed {} · shape g {} · colour g {}", req.mode.label(), params.seed, params.g_shape, params.g_colour),
        };
        if params.temperature != 1.0 {
            summary += &format!(" · T {}", params.temperature);
        }
        if params.top_p != 1.0 {
            summary += &format!(" · top-p {}", params.top_p);
        }
        if params.mask.is_none() && params.filter && stats.rejected > 0 {
            summary += &format!(" · filter redrew {}", stats.rejected);
        }
        params.vector = None;
        Ok(GenResult {
            id: req.id,
            caption,
            params,
            mode: req.mode,
            images: out.images,
            source,
            variants: infos,
            concepts: prepared.as_ref().map(|m| m.pl.concepts.iter().map(|c| format!("{}:{}", c.text, c.kind.label())).collect()).unwrap_or_default(),
            caption_exact: prepared.as_ref().filter(|m| m.caption.is_some()).map(|m| m.caption_exact),
            learned: mo.learned,
            summary,
            stats,
            secs: t0.elapsed().as_secs_f64(),
            encode_secs,
            ep,
            class,
        })
    }

}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::mix::{Lexicon, VariantKind};
    use super::{MixOptions, State};

    /// A tiny lexicon (dim 2) written in the demo's file format: burger (1,0), skull (0,1) objects, fried (1,1) attribute.
    fn lexicon(tag: &str) -> Arc<Lexicon> {
        let dir = std::env::temp_dir().join(format!("animator_ui_prepare_mix_{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (json, f16) = (dir.join("lexicon.json"), dir.join("lexicon.f16"));
        std::fs::write(
            &json,
            r#"{"dim":2,"stop":["a","of"],"phrases":[],"entries":[["burger","object"],["skull","object"],["fried","attribute"]]}"#,
        )
        .unwrap();
        // float16 little-endian: 1.0 = 0x3C00, 0.0 = 0x0000
        let (one, zero) = ([0x00u8, 0x3C], [0x00u8, 0x00]);
        let bytes: Vec<u8> = [one, zero, zero, one, one, one].concat();
        std::fs::write(&f16, bytes).unwrap();
        let lx = Lexicon::load(&json, &f16).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        Arc::new(lx)
    }

    fn state(tag: &str) -> State {
        let (mut st, rx) = State::for_test();
        std::mem::forget(rx); // keep the channel open for the test's lifetime
        st.lexicon = Some(lexicon(tag));
        st
    }

    #[test]
    fn no_lexicon_empty_or_single_word_is_one_caption() {
        let mut mo = MixOptions::default();
        let (mut bare, _rx) = State::for_test();
        assert!(bare.prepare_mix("burger skull", &mut mo).unwrap().is_none());
        let mut st = state("single");
        assert!(st.prepare_mix("", &mut mo).unwrap().is_none());
        assert!(st.prepare_mix("burger", &mut mo).unwrap().is_none());
        assert!(st.prepare_mix("a of", &mut mo).unwrap().is_none(), "only stop words");
    }

    #[test]
    fn known_words_mix_without_an_encoder() {
        let mut st = state("known");
        let mut mo = MixOptions::default();
        let p = st.prepare_mix("burger skull", &mut mo).unwrap().expect("two known objects mix");
        assert_eq!(p.vectors, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert!(p.pl.kinds.contains(&VariantKind::Caption) && p.pl.kinds.contains(&VariantKind::Inside));
        // fresh demo page: the caption is the normalised sum of the word vectors, not the exact encoder vector
        assert!(!p.caption_exact);
        let c = p.caption.expect("caption variant is enabled");
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert!((c[0] - h).abs() < 1e-6 && (c[1] - h).abs() < 1e-6, "{c:?}");
        assert!(mo.learned.is_empty(), "nothing unknown to learn");
        assert!(!mo.encoder_ready);
    }

    #[test]
    fn example_caption_uses_its_precomputed_vector() {
        let mut st = state("preset");
        st.presets.insert("burger skull".into(), vec![0.25, 0.75]);
        let p = st.prepare_mix("burger skull", &mut MixOptions::default()).unwrap().unwrap();
        assert!(p.caption_exact);
        assert_eq!(p.caption, Some(vec![0.25, 0.75]));
    }

    #[test]
    fn caption_variant_disabled_gives_no_caption_vector() {
        let mut st = state("nocaption");
        let mut mo = MixOptions { enabled: vec![VariantKind::Blend], ..MixOptions::default() };
        let p = st.prepare_mix("burger skull", &mut mo).unwrap().unwrap();
        assert!(p.caption.is_none() && !p.pl.kinds.contains(&VariantKind::Caption));
        // only "one caption" enabled: nothing to mix
        let mut mo = MixOptions { enabled: vec![VariantKind::Caption], ..MixOptions::default() };
        assert!(st.prepare_mix("burger skull", &mut mo).unwrap().is_none());
    }

    #[test]
    fn unknown_word_needs_the_encoder() {
        let mut st = state("unknown");
        let mut mo = MixOptions::default();
        let err = st.prepare_mix("burger zzzz", &mut mo).err().expect("no encoder to embed 'zzzz'");
        assert!(err.to_string().contains("text encoder is not loaded"), "{err:#}");
        assert!(mo.learned.is_empty());
        // an example caption with an unknown word, before the encoder has loaded: drawn as one caption, no error
        st.presets.insert("burger zzzz".into(), vec![1.0, 0.0]);
        assert!(st.prepare_mix("burger zzzz", &mut mo).unwrap().is_none());
        // once the encoder is ready it is needed again
        mo.encoder_ready = true;
        assert!(st.prepare_mix("burger zzzz", &mut mo).is_err());
    }
}
