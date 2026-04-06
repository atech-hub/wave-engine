//! Phase-native loss: ODE output compared directly against embedding table.
//! The ODE learns to produce outputs in the same space as its inputs.
//! No lm_head needed — the embedding table IS the decoder.

/// Compute phase-native loss and gradient.
/// Applies output corrector (per-band phase rotation) before comparing against embeddings.
///
/// Returns (loss, d_hidden, d_corrector) where:
/// - d_hidden: gradient w.r.t. the hidden state (for backward through the model)
/// - d_corrector: gradient w.r.t. the output corrector angles
pub fn phase_native_loss(
    hidden: &[f32],           // [n_embd] — final hidden state (post ln_f)
    embeddings: &[Vec<f32>],  // [vocab_size][n_embd] — the embedding table (wte)
    target: usize,            // target token index
    n_bands: usize,
    temperature: f32,
    output_corrector: &[f32], // [n_bands] — per-band phase rotation angles
) -> (f32, Vec<f32>, Vec<f32>) {
    let vocab_size = embeddings.len();
    let n_embd = n_bands * 2;

    // Output corrector: per-band phase rotation only. 84 params.
    // Rotates each band's phase to align with embedding space.
    // The ODE handles magnitude. The corrector handles phase alignment.
    let mut corrected = vec![0.0f32; n_embd];
    for k in 0..n_bands {
        let (sin_c, cos_c) = output_corrector[k].sin_cos();
        let r = hidden[k * 2];
        let s = hidden[k * 2 + 1];
        corrected[k * 2]     = r * cos_c - s * sin_c;
        corrected[k * 2 + 1] = r * sin_c + s * cos_c;
    }

    // Scaled dot product against embeddings (1/sqrt(n_embd) prevents logit explosion at high dims).
    // Same principle as Vaswani et al. scaled dot-product attention.
    let scale = 1.0 / (n_embd as f32).sqrt();
    let mut coherences = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let emb = &embeddings[v];
        let mut score = 0.0f32;
        for j in 0..n_embd {
            score += corrected[j] * emb[j];
        }
        coherences[v] = score * scale / temperature;
    }

    // Softmax over coherences → probabilities
    let max_coh = coherences.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_cohs: Vec<f32> = coherences.iter().map(|&c| (c - max_coh).exp()).collect();
    let sum_exp: f32 = exp_cohs.iter().sum();
    let probs: Vec<f32> = exp_cohs.iter().map(|&e| e / sum_exp).collect();

    // Cross-entropy loss
    let loss = -probs[target].max(1e-10).ln();

    // Gradient: d_loss/d_corrected — chain through scale and temperature
    // d_loss/d_score[v] = (probs[v] - (v == target)) * scale / temperature
    let mut d_corrected = vec![0.0f32; n_embd];
    for v in 0..vocab_size {
        let weight = (probs[v] - if v == target { 1.0 } else { 0.0 }) * scale / temperature;
        let emb = &embeddings[v];
        for j in 0..n_embd {
            d_corrected[j] += weight * emb[j];
        }
    }

    // Chain through output corrector rotation to get d_hidden and d_corrector
    let mut d_hidden = vec![0.0f32; n_embd];
    let mut d_corrector = vec![0.0f32; n_bands];
    for k in 0..n_bands {
        let (sin_c, cos_c) = output_corrector[k].sin_cos();
        let dc_r = d_corrected[k * 2];
        let dc_s = d_corrected[k * 2 + 1];
        let r = hidden[k * 2];
        let s = hidden[k * 2 + 1];

        // d_hidden: inverse rotation
        d_hidden[k * 2]     =  cos_c * dc_r + sin_c * dc_s;
        d_hidden[k * 2 + 1] = -sin_c * dc_r + cos_c * dc_s;

        // d_corrector[k]: gradient of rotation angle
        d_corrector[k] = dc_r * (-r * sin_c - s * cos_c)
                       + dc_s * ( r * cos_c - s * sin_c);
    }

    (loss, d_hidden, d_corrector)
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

        let corrector = vec![0.1, -0.2, 0.05, 0.3]; // 4 phase rotations
        let (loss, grad, d_corr) = phase_native_loss(&hidden, &embeddings, target, n_bands, temp, &corrector);
        assert!(loss > 0.0, "Loss should be positive");

        // Finite difference check
        let eps = 1e-4;
        for i in 0..hidden.len() {
            let mut h_plus = hidden.clone();
            let mut h_minus = hidden.clone();
            h_plus[i] += eps;
            h_minus[i] -= eps;
            let (l_plus, _, _) = phase_native_loss(&h_plus, &embeddings, target, n_bands, temp, &corrector);
            let (l_minus, _, _) = phase_native_loss(&h_minus, &embeddings, target, n_bands, temp, &corrector);
            let fd = (l_plus - l_minus) / (2.0 * eps);
            let rel_err = if grad[i].abs() > 1e-6 {
                (fd - grad[i]).abs() / grad[i].abs()
            } else {
                (fd - grad[i]).abs()
            };
            assert!(rel_err < 0.05, "Hidden grad mismatch at {}: fd={:.6} analytical={:.6} rel_err={:.4}",
                i, fd, grad[i], rel_err);
        }

        // Check corrector gradients too
        for i in 0..corrector.len() {
            let mut c_plus = corrector.clone();
            let mut c_minus = corrector.clone();
            c_plus[i] += eps;
            c_minus[i] -= eps;
            let (l_plus, _, _) = phase_native_loss(&hidden, &embeddings, target, n_bands, temp, &c_plus);
            let (l_minus, _, _) = phase_native_loss(&hidden, &embeddings, target, n_bands, temp, &c_minus);
            let fd = (l_plus - l_minus) / (2.0 * eps);
            let rel_err = if d_corr[i].abs() > 1e-6 {
                (fd - d_corr[i]).abs() / d_corr[i].abs()
            } else {
                (fd - d_corr[i]).abs()
            };
            assert!(rel_err < 0.05, "Corrector grad mismatch at {}: fd={:.6} analytical={:.6} rel_err={:.4}",
                i, fd, d_corr[i], rel_err);
        }
    }
}
