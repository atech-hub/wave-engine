//! Phase vocab diagnostic: compare ODE output against embedding table
//! using wave-native similarity instead of the lm_head linear projection.

use crate::common::wave_model::WavePacketModel;
use crate::common::dims::Dims;
use crate::cpu::forward::forward_with_cache;

/// Compare the ODE's final hidden state against each token's embedding
/// using phase coherence (wave-native) instead of the lm_head (linear projection).
/// Returns (phase_decode_token, lm_head_token, per-token coherences).
pub fn phase_decode_compare(
    model: &WavePacketModel,
    tokens: &[usize],
    dims: Dims,
    stencil: &crate::fft_ode::StencilFft,
) -> (usize, usize, Vec<(usize, f32, f32)>) {
    let n_embd = dims.n_embd;
    let n_bands = dims.n_bands;

    // Forward pass — same as training
    let cache = forward_with_cache(model, tokens, dims, None, None, None, Some(stencil), None, None);

    // Get the final hidden state (post ln_f) for the last position
    let last_hidden = &cache.post_ln_f[cache.post_ln_f.len() - 1];

    // lm_head decode: standard linear projection
    let mut logits = vec![0.0f32; model.vocab_size];
    for v in 0..model.vocab_size {
        let mut sum = 0.0f32;
        for j in 0..n_embd { sum += model.lm_head[v][j] * last_hidden[j]; }
        logits[v] = sum;
    }
    let lm_head_token = logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i).unwrap();

    // Phase coherence decode: compare hidden state against lm_head ROWS (output space)
    // NOT against embeddings (input space) — the ODE transforms into a different space.
    // The lm_head rows ARE the output space dictionary.
    let mut coherences: Vec<(usize, f32, f32)> = Vec::new(); // (token_id, phase_coherence, logit)

    for v in 0..model.vocab_size {
        // Phase coherence between hidden state and lm_head row v (output space)
        let lm_row = &model.lm_head[v];
        let mut phase_coh = 0.0f32;
        for k in 0..n_bands {
            let r1 = last_hidden[k * 2];
            let s1 = last_hidden[k * 2 + 1];
            let r2 = lm_row[k * 2];
            let s2 = lm_row[k * 2 + 1];
            let phase1 = s1.atan2(r1);
            let phase2 = s2.atan2(r2);
            phase_coh += (phase1 - phase2).cos();
        }
        phase_coh /= n_bands as f32;
        coherences.push((v, phase_coh, logits[v]));
    }

    // Phase decode: pick token with highest phase coherence
    let phase_token = coherences.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(id, _, _)| *id).unwrap();

    (phase_token, lm_head_token, coherences)
}
