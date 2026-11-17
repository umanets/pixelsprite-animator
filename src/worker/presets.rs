use anyhow::Result;


/// The demo's example captions (presets.json: base64 little-endian float32 vectors, computed in Python).
pub(super) fn load_presets(path: &std::path::Path) -> Result<Vec<(String, Vec<f32>)>> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let mut out = Vec::new();
    for it in v["items"].as_array().into_iter().flatten() {
        let (Some(t), Some(b)) = (it["text"].as_str(), it["vector"].as_str()) else { continue };
        out.push((t.to_string(), b64_floats(b)));
    }
    Ok(out)
}

/// Standard base64 -> little-endian float32 values.
pub fn b64_floats(s: &str) -> Vec<f32> {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut val = [255u8; 256];
    for (i, &c) in T.iter().enumerate() {
        val[c as usize] = i as u8;
    }
    let mut bytes = Vec::new();
    let (mut acc, mut bits) = (0u32, 0);
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
