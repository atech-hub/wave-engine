//! Phase-native decoder v2 — per-channel harmonic coherence.
//!
//! Instead of collapsing all bands into one scalar (dot product),
//! evaluates cos(n·Δθ) at each harmonic separately per band, then
//! aggregates. Addresses Proposition 3.5 blind spots where individual
//! harmonics are 1.0 but the aggregate is 0.0.
//!
//! Three scoring modes:
//!   1. max-harmonic: score = max_n(mean_k cos(n·Δθ_k))
//!   2. sum-harmonic: score = sum_n(mean_k cos(n·Δθ_k)) / N_harmonics
//!   3. mag-weighted: like max-harmonic but weights by per-band magnitude

/// Harmonics to evaluate — the framework's core relationship detectors.
const DECODE_HARMONICS: &[usize] = &[1, 2, 3, 4, 6];

/// Scoring mode for the v2 decoder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecodeMode {
    /// Score = max across harmonics of mean per-band coherence
    MaxHarmonic,
    /// Score = mean across all harmonics of mean per-band coherence
    SumHarmonic,
    /// Like MaxHarmonic but each band weighted by hidden state magnitude
    MagWeighted,
}

/// Compute per-token logits using per-harmonic coherence.
///
/// For each vocab token v:
///   For each harmonic n in {1,2,3,4,6}:
///     coherence_n = mean_k cos(n * (θ_hidden[k] - θ_emb[v][k]))
///   score[v] = aggregate(coherence_1, ..., coherence_6)
///
/// The output corrector is applied before phase extraction.
pub fn decode_v2(
    hidden: &[f32],
    embeddings: &[Vec<f32>],
    output_corrector: &[f32],
    n_bands: usize,
    vocab_size: usize,
    mode: DecodeMode,
) -> Vec<f32> {
    // Apply corrector rotation to hidden state
    let mut corrected = vec![0.0f32; n_bands * 2];
    for k in 0..n_bands {
        let (sin_c, cos_c) = output_corrector[k].sin_cos();
        let r = hidden[k * 2];
        let s = hidden[k * 2 + 1];
        corrected[k * 2]     = r * cos_c - s * sin_c;
        corrected[k * 2 + 1] = r * sin_c + s * cos_c;
    }

    // Extract per-band phase and magnitude from hidden state
    let h_phase: Vec<f32> = (0..n_bands)
        .map(|k| corrected[k * 2 + 1].atan2(corrected[k * 2]))
        .collect();
    let h_mag: Vec<f32> = (0..n_bands)
        .map(|k| (corrected[k*2]*corrected[k*2] + corrected[k*2+1]*corrected[k*2+1]).sqrt())
        .collect();

    // For each vocab token, compute per-harmonic coherence
    (0..vocab_size).map(|v| {
        let emb = &embeddings[v];

        // Extract embedding phases
        let e_phase: Vec<f32> = (0..n_bands)
            .map(|k| emb[k * 2 + 1].atan2(emb[k * 2]))
            .collect();

        // Compute coherence at each harmonic
        let coherences: Vec<f32> = DECODE_HARMONICS.iter().map(|&n| {
            match mode {
                DecodeMode::MaxHarmonic | DecodeMode::SumHarmonic => {
                    // Unweighted mean per-band coherence
                    let sum: f32 = (0..n_bands).map(|k| {
                        let diff = h_phase[k] - e_phase[k];
                        (n as f32 * diff).cos()
                    }).sum();
                    sum / n_bands as f32
                }
                DecodeMode::MagWeighted => {
                    // Magnitude-weighted mean per-band coherence
                    let mut weighted_sum = 0.0f32;
                    let mut weight_total = 0.0f32;
                    for k in 0..n_bands {
                        let diff = h_phase[k] - e_phase[k];
                        let coh = (n as f32 * diff).cos();
                        let w = h_mag[k];
                        weighted_sum += w * coh;
                        weight_total += w;
                    }
                    if weight_total > 1e-8 { weighted_sum / weight_total } else { 0.0 }
                }
            }
        }).collect();

        // Aggregate across harmonics
        match mode {
            DecodeMode::MaxHarmonic | DecodeMode::MagWeighted => {
                // Max coherence across harmonics — catches the best relationship
                coherences.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            }
            DecodeMode::SumHarmonic => {
                // Mean across harmonics — rewards tokens coherent at multiple harmonics
                coherences.iter().sum::<f32>() / coherences.len() as f32
            }
        }
    }).collect()
}

/// Compare v1 (dot product) vs v2 (per-harmonic) decoder on a set of prompts.
/// Returns (v1_correct, v2_correct, total, per_prompt_details).
pub fn compare_decoders(
    model: &crate::WavePacketModel,
    tokens: &[usize],
    dims: crate::Dims,
    stencil: &crate::fft_ode::StencilFft,
    mode: DecodeMode,
) -> (Vec<f32>, Vec<f32>) {
    // Run forward pass
    let cache = crate::cpu::forward::forward_with_cache(
        model, tokens, dims, None, None, None, Some(stencil), None, None, None,
    );

    let n_bands = dims.n_bands;
    let last = cache.post_ln_f.len() - 1;
    let hidden = &cache.post_ln_f[last];

    // V1: dot product (current)
    let mut v1_logits = vec![0.0f32; model.vocab_size];
    let mut corrected = vec![0.0f32; n_bands * 2];
    for k in 0..n_bands {
        let (sin_c, cos_c) = model.output_corrector[k].sin_cos();
        let r = hidden[k * 2];
        let s = hidden[k * 2 + 1];
        corrected[k * 2]     = r * cos_c - s * sin_c;
        corrected[k * 2 + 1] = r * sin_c + s * cos_c;
    }
    for v in 0..model.vocab_size {
        let emb = &model.wte[v];
        let mut score = 0.0f32;
        for j in 0..(n_bands * 2) { score += corrected[j] * emb[j]; }
        v1_logits[v] = score;
    }

    // V2: per-harmonic coherence
    let v2_logits = decode_v2(hidden, &model.wte, &model.output_corrector, n_bands, model.vocab_size, mode);

    (v1_logits, v2_logits)
}
