//! Wave packet embedding — tokens as phase positions on a harmonic circle.
//!
//! Each token is encoded as cos(n * θ) / sin(n * θ) where θ = token_id * 2π / vocab_size.
//! Positional encoding adds a phase offset per position on the same circle.
//! This is the same Phase mechanics validated in wave-test (Tests 1-25).

use std::f32::consts::PI;

/// Build harmonic embedding table: token_id → [cos(1·θ), sin(1·θ), cos(2·θ), sin(2·θ), ...]
/// Identical to kerr-engine's frozen harmonic embeddings.
pub fn build_harmonic_table(vocab_size: usize, n_bands: usize) -> Vec<Vec<f32>> {
    let n_embd = n_bands * 2;
    (0..vocab_size).map(|tok| {
        let theta = tok as f32 * 2.0 * PI / vocab_size as f32;
        let mut emb = vec![0.0f32; n_embd];
        for n in 0..n_bands {
            let phase = (n + 1) as f32 * theta;
            emb[n * 2] = phase.cos();
            emb[n * 2 + 1] = phase.sin();
        }
        emb
    }).collect()
}

/// Build positional encoding table: position → phase offset.
/// Uses sinusoidal encoding (standard transformer) on the harmonic circle.
pub fn build_positional_table(block_size: usize, n_bands: usize) -> Vec<Vec<f32>> {
    let n_embd = n_bands * 2;
    (0..block_size).map(|pos| {
        let mut pe = vec![0.0f32; n_embd];
        for n in 0..n_bands {
            let freq = 1.0 / (10000.0f32).powf(2.0 * n as f32 / n_embd as f32);
            pe[n * 2] = (pos as f32 * freq).sin();
            pe[n * 2 + 1] = (pos as f32 * freq).cos();
        }
        pe
    }).collect()
}

/// Embed tokens: look up harmonic table + add positional encoding.
pub fn embed_tokens(
    tokens: &[usize],
    wte: &[Vec<f32>],
    wpe: &[Vec<f32>],
    n_embd: usize,
) -> Vec<Vec<f32>> {
    tokens.iter().enumerate().map(|(pos, &tok)| {
        let mut h = vec![0.0f32; n_embd];
        for i in 0..n_embd {
            h[i] = wte[tok][i] + wpe[pos][i];
        }
        h
    }).collect()
}
