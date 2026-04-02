//! Phase-native loss: ODE output compared directly against embedding table.
//! The ODE learns to produce outputs in the same space as its inputs.
//! No lm_head needed — the embedding table IS the decoder.

/// Compute phase-native loss and gradient.
/// Compares the final hidden state against all token embeddings using phase coherence,
/// then applies cross-entropy over the coherence-derived probabilities.
///
/// Returns (loss, d_hidden) where d_hidden is the gradient w.r.t. the hidden state.
pub fn phase_native_loss(
    hidden: &[f32],           // [n_embd] — final hidden state (post ln_f)
    embeddings: &[Vec<f32>],  // [vocab_size][n_embd] — the embedding table (wte)
    target: usize,            // target token index
    n_bands: usize,
    temperature: f32,
) -> (f32, Vec<f32>) {
    let vocab_size = embeddings.len();
    let n_embd = n_bands * 2;

    // Compute phase coherence with every token's embedding
    let mut coherences = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let emb = &embeddings[v];
        let mut coh = 0.0f32;
        for k in 0..n_bands {
            let r1 = hidden[k * 2];
            let s1 = hidden[k * 2 + 1];
            let r2 = emb[k * 2];
            let s2 = emb[k * 2 + 1];
            let dot = r1 * r2 + s1 * s2;
            let mag1 = (r1 * r1 + s1 * s1).sqrt().max(1e-8);
            let mag2 = (r2 * r2 + s2 * s2).sqrt().max(1e-8);
            coh += dot / (mag1 * mag2);
        }
        coherences[v] = coh / n_bands as f32 / temperature;
    }

    // Softmax over coherences → probabilities
    let max_coh = coherences.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_cohs: Vec<f32> = coherences.iter().map(|&c| (c - max_coh).exp()).collect();
    let sum_exp: f32 = exp_cohs.iter().sum();
    let probs: Vec<f32> = exp_cohs.iter().map(|&e| e / sum_exp).collect();

    // Cross-entropy loss
    let loss = -probs[target].max(1e-10).ln();

    // Gradient: d_loss/d_hidden
    // Chain: loss → softmax → coherences → hidden
    // d_loss/d_coherence[v] = probs[v] - (v == target)
    let mut d_hidden = vec![0.0f32; n_embd];
    for v in 0..vocab_size {
        let weight = probs[v] - if v == target { 1.0 } else { 0.0 };
        let emb = &embeddings[v];
        for k in 0..n_bands {
            let r1 = hidden[k * 2];
            let s1 = hidden[k * 2 + 1];
            let r2 = emb[k * 2];
            let s2 = emb[k * 2 + 1];
            let mag1 = (r1 * r1 + s1 * s1).sqrt().max(1e-8);
            let mag2 = (r2 * r2 + s2 * s2).sqrt().max(1e-8);
            let dot = r1 * r2 + s1 * s2;
            // d(coh_k)/d(r1) = r2/(mag1*mag2) - r1*dot/(mag1^3 * mag2)
            let d_r = (r2 / (mag1 * mag2) - r1 * dot / (mag1.powi(3) * mag2))
                      / n_bands as f32 / temperature;
            let d_s = (s2 / (mag1 * mag2) - s1 * dot / (mag1.powi(3) * mag2))
                      / n_bands as f32 / temperature;
            d_hidden[k * 2] += weight * d_r;
            d_hidden[k * 2 + 1] += weight * d_s;
        }
    }

    (loss, d_hidden)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_loss_gradient() {
        // Simple test: 4 bands, 3 tokens
        let n_bands = 4;
        let hidden: Vec<f32> = vec![0.5, 0.3, -0.2, 0.8, 0.1, -0.4, 0.7, 0.2];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            vec![-1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0],
        ];
        let target = 1;
        let temp = 1.0;

        let (loss, grad) = phase_native_loss(&hidden, &embeddings, target, n_bands, temp);
        assert!(loss > 0.0, "Loss should be positive");

        // Finite difference check
        let eps = 1e-4;
        for i in 0..hidden.len() {
            let mut h_plus = hidden.clone();
            let mut h_minus = hidden.clone();
            h_plus[i] += eps;
            h_minus[i] -= eps;
            let (l_plus, _) = phase_native_loss(&h_plus, &embeddings, target, n_bands, temp);
            let (l_minus, _) = phase_native_loss(&h_minus, &embeddings, target, n_bands, temp);
            let fd = (l_plus - l_minus) / (2.0 * eps);
            let rel_err = if grad[i].abs() > 1e-6 {
                (fd - grad[i]).abs() / grad[i].abs()
            } else {
                (fd - grad[i]).abs()
            };
            assert!(rel_err < 0.05, "Gradient mismatch at {}: fd={:.6} analytical={:.6} rel_err={:.4}",
                i, fd, grad[i], rel_err);
        }
    }
}
