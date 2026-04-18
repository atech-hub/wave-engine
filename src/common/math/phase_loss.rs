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

    // Dot product against embeddings — same metric as lm_head but against fixed embeddings.
    // Magnitude matters: the ODE gets gradient through both phase AND magnitude.
    let mut coherences = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let emb = &embeddings[v];
        let mut score = 0.0f32;
        for j in 0..n_embd {
            score += corrected[j] * emb[j];
        }
        coherences[v] = score / temperature;
    }

    // Softmax over coherences → probabilities
    let max_coh = coherences.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_cohs: Vec<f32> = coherences.iter().map(|&c| (c - max_coh).exp()).collect();
    let sum_exp: f32 = exp_cohs.iter().sum();
    let probs: Vec<f32> = exp_cohs.iter().map(|&e| e / sum_exp).collect();

    // Cross-entropy loss
    let loss = -probs[target].max(1e-10).ln();

    // Gradient: d_loss/d_corrected — dot product gradient is simple: d_score/d_corrected[j] = emb[j]
    // d_loss/d_score[v] = probs[v] - (v == target)
    let mut d_corrected = vec![0.0f32; n_embd];
    for v in 0..vocab_size {
        let weight = (probs[v] - if v == target { 1.0 } else { 0.0 }) / temperature;
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

/// Forward-only phase-native loss at f64 precision. Used by the J1 gradient
/// check to defeat f32 catastrophic cancellation in the FD subtraction.
/// Mirrors `phase_native_loss` exactly — same formula, same inputs — but
/// accumulates dot products, softmax, and ln at f64 throughout. Model weights
/// stay f32; only the loss arithmetic is lifted. Returns only the scalar loss
/// (no gradients) — verification doesn't need them from this path.
pub fn phase_native_loss_value_f64(
    hidden: &[f32],
    embeddings: &[Vec<f32>],
    target: usize,
    n_bands: usize,
    temperature: f32,
    output_corrector: &[f32],
) -> f64 {
    let vocab_size = embeddings.len();
    let n_embd = n_bands * 2;
    let temperature = temperature as f64;

    // Apply output corrector at f64.
    let mut corrected = vec![0.0f64; n_embd];
    for k in 0..n_bands {
        let angle = output_corrector[k] as f64;
        let (sin_c, cos_c) = angle.sin_cos();
        let r = hidden[k * 2] as f64;
        let s = hidden[k * 2 + 1] as f64;
        corrected[k * 2]     = r * cos_c - s * sin_c;
        corrected[k * 2 + 1] = r * sin_c + s * cos_c;
    }

    // Dot product against embeddings at f64.
    let mut coherences = vec![0.0f64; vocab_size];
    for v in 0..vocab_size {
        let emb = &embeddings[v];
        let mut score = 0.0f64;
        for j in 0..n_embd {
            score += corrected[j] * (emb[j] as f64);
        }
        coherences[v] = score / temperature;
    }

    // Softmax at f64.
    let max_coh = coherences.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum_exp = 0.0f64;
    let mut target_exp = 0.0f64;
    for (v, &c) in coherences.iter().enumerate() {
        let e = (c - max_coh).exp();
        sum_exp += e;
        if v == target { target_exp = e; }
    }
    let prob_target = (target_exp / sum_exp).max(1e-30_f64);
    -prob_target.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_loss_matches_f32_loss_within_tolerance() {
        // The f64 variant must agree with the f32 variant to within ~1e-5
        // at typical magnitudes — they are the same math, just different precision.
        let n_bands = 4;
        let hidden: Vec<f32> = vec![0.5, 0.3, -0.2, 0.8, 0.1, -0.4, 0.7, 0.2];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            vec![-1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0],
        ];
        let corrector = vec![0.1, -0.2, 0.05, 0.3];
        let (loss_f32, _, _) = phase_native_loss(&hidden, &embeddings, 1, n_bands, 1.0, &corrector);
        let loss_f64 = phase_native_loss_value_f64(&hidden, &embeddings, 1, n_bands, 1.0, &corrector);
        let diff = (loss_f32 as f64 - loss_f64).abs();
        assert!(diff < 1e-5, "f32 and f64 loss should agree: f32={} f64={} diff={}",
            loss_f32, loss_f64, diff);
    }

    #[test]
    fn test_f64_fd_resolves_where_f32_cancels() {
        // Param perturbation is small; f32 FD cancels, f64 FD doesn't.
        let n_bands = 4;
        let hidden_base: Vec<f32> = vec![0.5, 0.3, -0.2, 0.8, 0.1, -0.4, 0.7, 0.2];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            vec![-1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0],
        ];
        let corrector = vec![0.1, -0.2, 0.05, 0.3];
        let eps = 1e-6_f32; // deep in f32 cancellation territory

        // Perturb one hidden component.
        let mut h_plus = hidden_base.clone();
        let mut h_minus = hidden_base.clone();
        h_plus[0] += eps;
        h_minus[0] -= eps;

        let (l_plus_f32, _, _) = phase_native_loss(&h_plus, &embeddings, 1, n_bands, 1.0, &corrector);
        let (l_minus_f32, _, _) = phase_native_loss(&h_minus, &embeddings, 1, n_bands, 1.0, &corrector);
        let fd_f32 = (l_plus_f32 - l_minus_f32) / (2.0 * eps);

        let l_plus_f64 = phase_native_loss_value_f64(&h_plus, &embeddings, 1, n_bands, 1.0, &corrector);
        let l_minus_f64 = phase_native_loss_value_f64(&h_minus, &embeddings, 1, n_bands, 1.0, &corrector);
        let fd_f64 = (l_plus_f64 - l_minus_f64) / (2.0_f64 * eps as f64);

        // At eps=1e-6 the f32 loss difference cancels to zero for many inputs.
        // The f64 difference should be finite (nonzero) because f64 has the
        // precision to resolve the delta.
        if fd_f32 == 0.0 {
            assert!(fd_f64 != 0.0, "f64 FD should not cancel where f32 does: fd_f64={}", fd_f64);
        }
        // Either way, f64 FD should match analytical gradient (computed at f32,
        // but within tolerance at this scale) better than f32 FD.
        let (_, grad_f32, _) = phase_native_loss(&hidden_base, &embeddings, 1, n_bands, 1.0, &corrector);
        let an = grad_f32[0] as f64;
        let err_f64 = (fd_f64 - an).abs() / an.abs().max(1e-8);
        assert!(err_f64 < 0.1, "f64 FD should agree with analytical: fd={} an={} err={}",
            fd_f64, an, err_f64);
    }

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
