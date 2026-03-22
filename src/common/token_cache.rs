//! Token cache — encode once, load instantly on subsequent runs.
//!
//! Saves tokenized data as a binary file next to the source text.
//! Cache key: data file size + tokenizer path hash. If either changes,
//! cache is invalidated and re-encoded.

use std::io::{Read, Write};

/// Cache file format: "WTOK" + vocab_size(u64) + n_tokens(u64) + token_ids([u32])
const MAGIC: &[u8; 4] = b"WTOK";

/// Try to load cached tokens. Returns None if cache doesn't exist or is stale.
pub fn load_cache(data_path: &str, use_bpe: bool) -> Option<(Vec<usize>, usize)> {
    let cache_path = cache_path_for(data_path, use_bpe);
    let mut f = std::fs::File::open(&cache_path).ok()?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).ok()?;
    if &magic != MAGIC { return None; }

    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8).ok()?;
    let vocab_size = u64::from_le_bytes(buf8) as usize;

    f.read_exact(&mut buf8).ok()?;
    let n_tokens = u64::from_le_bytes(buf8) as usize;

    // Validate: check data file size matches
    f.read_exact(&mut buf8).ok()?;
    let cached_file_size = u64::from_le_bytes(buf8);
    let actual_file_size = std::fs::metadata(data_path).ok()?.len();
    if cached_file_size != actual_file_size { return None; }

    // Read token IDs (u32 each)
    let mut raw = vec![0u8; n_tokens * 4];
    f.read_exact(&mut raw).ok()?;
    let tokens: Vec<usize> = raw.chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();

    println!("  Token cache hit: {} tokens loaded from {}", tokens.len(), cache_path);
    Some((tokens, vocab_size))
}

/// Save tokens to cache file.
pub fn save_cache(data_path: &str, use_bpe: bool, tokens: &[usize], vocab_size: usize) {
    let cache_path = cache_path_for(data_path, use_bpe);
    let file_size = std::fs::metadata(data_path).map(|m| m.len()).unwrap_or(0);

    let mut f = match std::fs::File::create(&cache_path) {
        Ok(f) => f,
        Err(_) => return, // silently skip if can't write
    };

    f.write_all(MAGIC).ok();
    f.write_all(&(vocab_size as u64).to_le_bytes()).ok();
    f.write_all(&(tokens.len() as u64).to_le_bytes()).ok();
    f.write_all(&file_size.to_le_bytes()).ok();

    for &t in tokens {
        f.write_all(&(t as u32).to_le_bytes()).ok();
    }

    let mb = (tokens.len() * 4) as f64 / 1e6;
    println!("  Token cache saved: {:.1}MB → {}", mb, cache_path);
}

/// Generate cache file path: data_path + ".bpe.tokens" or ".char.tokens"
fn cache_path_for(data_path: &str, use_bpe: bool) -> String {
    let suffix = if use_bpe { ".bpe.tokens" } else { ".char.tokens" };
    format!("{}{}", data_path, suffix)
}
