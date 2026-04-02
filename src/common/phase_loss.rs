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

    // Phase-magnitude comparison: phase coherence for discrimination, magnitude match for confidence.
    // Phase score: cos(Δθ) per band — equal weight, discriminates tokens.
    // Magnitude match: 1 - |mag1-mag2|/(mag1+mag2) per band — confirms energy profile.
    // Combined: phase_score * (1 + λ * mag_match) — phase tells WHICH, magnitude tells HOW SURE.
    let mag_weight = 0.5f32; // λ: how much magnitude confirmation matters
    let mut coherences = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let emb = &embeddings[v];
        let mut score = 0.0f32;
        for k in 0..n_bands {
            let r1 = corrected[k * 2];
            let s1 = corrected[k * 2 + 1];
            let r2 = emb[k * 2];
            let s2 = emb[k * 2 + 1];
            let dot = r1 * r2 + s1 * s2;
            let mag1 = (r1 * r1 + s1 * s1).sqrt().max(1e-8);
            let mag2 = (r2 * r2 + s2 * s2).sqrt().max(1e-8);
            // Phase: cos(Δθ) — equal band weight
            let phase_score = dot / (mag1 * mag2);
            // Magnitude match: 0 to 1 (1 = perfect match)
            let mag_match = 1.0 - (mag1 - mag2).abs() / (mag1 + mag2).max(1e-8);
            score += phase_score * (1.0 + mag_weight * mag_match);
        }
        coherences[v] = score / n_bands as f32 / temperature;
    }

    // Softmax over coherences → probabilities
    let max_coh = coherences.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_cohs: Vec<f32> = coherences.iter().map(|&c| (c - max_coh).exp()).collect();
    let sum_exp: f32 = exp_cohs.iter().sum();
    let probs: Vec<f32> = exp_cohs.iter().map(|&e| e / sum_exp).collect();

    // Cross-entropy loss
    let loss = -probs[target].max(1e-10).ln();

    // Gradient: d_loss/d_corrected through phase-mag metric
    // d_loss/d_score[v] = probs[v] - (v == target)
    let mut d_corrected = vec![0.0f32; n_embd];
    for v in 0..vocab_size {
        let softmax_weight = probs[v] - if v == target { 1.0 } else { 0.0 };
        let emb = &embeddings[v];
        for k in 0..n_bands {
            let r1 = corrected[k * 2];
            let s1 = corrected[k * 2 + 1];
            let r2 = emb[k * 2];
            let s2 = emb[k * 2 + 1];
            let dot = r1 * r2 + s1 * s2;
            let mag1 = (r1 * r1 + s1 * s1).sqrt().max(1e-8);
            let mag2 = (r2 * r2 + s2 * s2).sqrt().max(1e-8);
            let phase_score = dot / (mag1 * mag2);
            let mag_diff = mag1 - mag2;
            let mag_sum = (mag1 + mag2).max(1e-8);
            let mag_match = 1.0 - mag_diff.abs() / mag_sum;

            // d(phase_score)/d(r1) = r2/(mag1*mag2) - r1*dot/(mag1^3*mag2)
            let d_phase_r = r2 / (mag1 * mag2) - r1 * dot / (mag1.powi(3) * mag2);
            let d_phase_s = s2 / (mag1 * mag2) - s1 * dot / (mag1.powi(3) * mag2);

            // d(mag_match)/d(r1): mag_match = 1 - |mag1-mag2|/(mag1+mag2)
            // d(mag1)/d(r1) = r1/mag1, d(mag1)/d(s1) = s1/mag1
            // If mag1 > mag2: ratio = (mag1-mag2)/(mag1+mag2), d_ratio/d_mag1 = 2*mag2/(mag1+mag2)²
            // If mag1 < mag2: ratio = (mag2-mag1)/(mag1+mag2), d_ratio/d_mag1 = -2*mag2/(mag1+mag2)² ... wait
            // Cleaner: use subgradient. d|x|/dx = sign(x).
            // ratio = |mag1-mag2|/(mag1+mag2)
            // d_ratio/d_mag1 = sign(mag1-mag2)/(mag1+mag2) - |mag1-mag2|/(mag1+mag2)²
            //                = (sign(mag1-mag2)*(mag1+mag2) - |mag1-mag2|) / (mag1+mag2)²
            let sign = if mag1 >= mag2 { 1.0 } else { -1.0 };
            let d_ratio_dmag1 = (sign * mag_sum - mag_diff.abs()) / (mag_sum * mag_sum);
            // d_mag_match/d_mag1 = -d_ratio/d_mag1
            let d_mm_dmag1 = -d_ratio_dmag1;
            let d_mag_r = d_mm_dmag1 * r1 / mag1;
            let d_mag_s = d_mm_dmag1 * s1 / mag1;

            // Combined: d(score_k)/d(r1) = d_phase*(1+λ*mag) + phase*λ*d_mag
            let combined = (1.0 + mag_weight * mag_match);
            let d_r = (d_phase_r * combined + phase_score * mag_weight * d_mag_r)
                      / n_bands as f32 / temperature;
            let d_s = (d_phase_s * combined + phase_score * mag_weight * d_mag_s)
                      / n_bands as f32 / temperature;

            d_corrected[k * 2] += softmax_weight * d_r;
            d_corrected[k * 2 + 1] += softmax_weight * d_s;
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
