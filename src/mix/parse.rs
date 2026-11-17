use std::collections::HashMap;

use super::*;


fn is_cyr_char(c: char) -> bool {
    ('а'..='я').contains(&c) || matches!(c, 'ё' | 'і' | 'ї' | 'є' | 'ґ')
}

/// `/[а-яёіїєґ]/i`.
pub fn is_cyrillic(text: &str) -> bool {
    text.chars().flat_map(char::to_lowercase).any(is_cyr_char)
}

/// Russian / Ukrainian adjectives by their ending (the encoder knows some of them badly).
pub fn looks_like_adjective(w: &str) -> bool {
    const ENDS: [&str; 10] = ["ый", "ий", "ій", "ой", "ая", "яя", "ое", "ее", "ые", "ие"];
    let lw = w.to_lowercase();
    is_cyrillic(w) && ENDS.iter().any(|e| lw.ends_with(e))
}

/// `\p{L}` for the scripts the model reads (char::is_alphabetic also counts combining marks: left out).
fn is_letter(c: char) -> bool {
    c.is_alphabetic() && !('\u{300}'..='\u{36f}').contains(&c)
}

/// `text.match(/[\p{L}]+(?:['’][\p{L}]+)?/gu)`.
pub fn words(text: &str) -> Vec<String> {
    let ch: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < ch.len() {
        if !is_letter(ch[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < ch.len() && is_letter(ch[i]) {
            i += 1;
        }
        if i + 1 < ch.len() && (ch[i] == '\'' || ch[i] == '’') && is_letter(ch[i + 1]) {
            i += 1;
            while i < ch.len() && is_letter(ch[i]) {
                i += 1;
            }
        }
        out.push(ch[start..i].iter().collect());
    }
    out
}

/// Caption -> concepts (at most [`MAX`]). `overrides`: text -> kind, from clicks on the word chips (and kinds
/// learned for unknown words), as `kindsFor()` in app.js.
pub fn parse(text: &str, lex: &Lexicon, overrides: &HashMap<String, ConceptKind>) -> Parsed {
    let cyr = is_cyrillic(text);
    let ws: Vec<String> = words(&text.to_lowercase()).into_iter().filter(|w| !lex.stop.contains(w)).collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < ws.len() {
        let mut n = 1;
        for k in [3usize, 2] {
            if i + k <= ws.len() && lex.phrases.contains(&ws[i..i + k].join(" ")) {
                n = k;
                break;
            }
        }
        out.push(ws[i..i + n].join(" "));
        i += n;
    }
    let dropped = out.len().saturating_sub(MAX);
    out.truncate(MAX);
    let len = out.len();
    let ranks: Vec<usize> = (0..len).map(|i| if cyr { i } else { len - 1 - i }).collect();
    let raw: Vec<f64> = ranks.iter().map(|&r| 0.5f64.powi(r as i32 + 1)).collect();
    let sum = raw.iter().fold(0.0, |a, b| a + b);
    let concepts = out
        .into_iter()
        .enumerate()
        .map(|(i, t)| {
            let adj = looks_like_adjective(&t);
            let kind = overrides.get(&t).copied().or_else(|| lex.kind.get(&t).copied()).unwrap_or(if adj { ConceptKind::Attribute } else { ConceptKind::Object });
            Concept { known: lex.kind.contains_key(&t) || adj, kind, p: raw[i] / sum, rank: ranks[i], text: t }
        })
        .collect();
    Parsed { concepts, cyr, dropped }
}

/// None when the variant fits the caption, else the reason (for the disabled chip).
pub fn unfit(kind: VariantKind, concepts: &[Concept]) -> Option<&'static str> {
    let objects = concepts.iter().filter(|c| c.kind == ConceptKind::Object).count();
    match kind {
        VariantKind::Caption => None,
        VariantKind::Blend => (concepts.len() < 2).then_some("needs two words"),
        _ => (objects < 2).then_some("needs two objects, e.g. burger skull"),
    }
}
