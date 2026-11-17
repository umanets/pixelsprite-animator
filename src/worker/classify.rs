use crate::classify::{Classification, MappedWord, classify_mapped, ru_uk_lookup, vocabulary, words};
use crate::text::cosine;
use super::Msg;
use super::state::State;

/// Minimum cosine similarity for mapping a Russian / Ukrainian word onto an English list word.
pub const MAP_THRESHOLD: f32 = 0.93;

impl State {
    /// Rules of classify.rs; non-English words are first mapped to the nearest English list word.
    pub(super) fn classify(&mut self, caption: &str) -> Classification {
        let mut mapped = Vec::new();
        for w in words(caption) {
            if w.is_ascii() {
                mapped.push(MappedWord { english: w.clone(), original: w, score: 1.0 });
                continue;
            }
            if let Some(en) = ru_uk_lookup(&w) {
                mapped.push(MappedWord { original: w, english: en.to_string(), score: 1.0 });
                continue;
            }
            if w.chars().count() < 3 {
                continue;
            }
            if let Some(m) = self.map_word(&w) {
                mapped.push(m);
            }
        }
        classify_mapped(&mapped)
    }

    /// Nearest English list word by CSLS (cross-domain similarity local scaling): 2 cos(q, w) - hub(w), where
    /// hub(w) = mean cosine of w to its 10 nearest vocabulary words. Without it, rare words that sit near the
    /// centre of the embedding space ("tenrec", "pelt") attract every unknown word.
    pub fn map_word(&mut self, w: &str) -> Option<MappedWord> {
        let enc = self.enc.as_mut()?;
        if self.vocab.is_none() {
            let _ = self.tx.send(Msg::Status("Embedding the English word lists (once)…".into()));
            let mut v = Vec::new();
            for word in vocabulary() {
                v.push((word, enc.encode(word).ok()?));
            }
            let hubs: Vec<f32> = v
                .iter()
                .map(|(_, a)| {
                    let mut sims: Vec<f32> = v.iter().map(|(_, b)| cosine(a, b)).collect();
                    sims.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
                    sims.iter().skip(1).take(10).sum::<f32>() / 10.0
                })
                .collect();
            self.vocab = Some(v.into_iter().zip(hubs).map(|((w, e), h)| (w, e, h)).collect());
        }
        let q = enc.encode(w).ok()?;
        let (best, score, _) = self
            .vocab
            .as_ref()?
            .iter()
            .map(|(word, v, hub)| {
                let c = cosine(&q, v);
                (*word, c, 2.0 * c - hub)
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))?;
        (score >= MAP_THRESHOLD).then(|| MappedWord { original: w.to_string(), english: best.to_string(), score })
    }
}

#[cfg(test)]
mod tests {
    use crate::classify::Kind;
    use super::State;

    fn kind_of(caption: &str) -> (Option<Kind>, String) {
        let (mut st, _rx) = State::for_test();
        let c = st.classify(caption);
        (c.kind, c.reason)
    }

    #[test]
    fn english_words_go_straight_to_the_rules() {
        assert_eq!(kind_of("a cat").0, Some(Kind::Quadruped));
        assert_eq!(kind_of("an eagle").0, Some(Kind::Bird));
        // a role word wins over an animal word
        let (k, why) = kind_of("cat warrior");
        assert_eq!(k, Some(Kind::Human));
        assert!(why.contains("warrior"), "{why}");
        // object made after an animal, body plan we cannot animate, no creature at all
        assert_eq!(kind_of("eagle feather").0, None);
        assert_eq!(kind_of("a big snake").0, None);
        assert_eq!(kind_of("red potion").0, None);
        assert_eq!(kind_of("").0, None);
    }

    #[test]
    fn russian_and_ukrainian_use_the_built_in_stems() {
        let (k, why) = kind_of("кошка");
        assert_eq!(k, Some(Kind::Quadruped));
        assert!(why.contains("'кошка'") && why.contains("'cat'"), "{why}");
        assert_eq!(kind_of("ведмідь").0, Some(Kind::Quadruped));
        // the object made after the animal wins, as in English
        assert_eq!(kind_of("череп собаки").0, None);
    }

    #[test]
    fn unmappable_words_are_ignored_without_an_encoder() {
        // no stem for "зелье" and no encoder to map it: it is dropped, the English word still counts
        assert_eq!(kind_of("зелье dog").0, Some(Kind::Quadruped));
        assert_eq!(kind_of("зелье").0, None);
    }
}
