//! Phase-native loss: ODE output compared directly against embedding table.
//! One-sided normalisation: normalise EMBEDDING to remove magnitude bias,
//! leave ODE output UNTOUCHED. score = Σ mag_ode × cos(Δθ).
//! The ODE speaks naturally. The embedding provides pure phase targets.

/// Compute phase-native loss and gradient.
/// One-sided normalisation: embedding normalised per-band, ODE output as-is.
/// Returns (loss, d_hidden, d_corrector).
pub fn phase_native_loss(
    hidden: &[f32],
    embeddings: &[Vec<f32>],
    target: usize,
    n_bands: usize,
    temperature: f32,
    output_corrector: &[f32],
) -> (f32, Vec<f32>, Vec<f32>) {
    let vocab_size = embeddings.len();
    let n_embd = n_bands * 2;

    // Apply output corrector: per-band phase rotation only.
    let mut corrected = vec![0.0f32; n_embd];
    for k in 0..n_bands {
        let (sin_c, cos_c) = output_corrector[k].sin_cos();
        let r = hidden[k * 2];
        let s = hidden[k * 2 + 1];
        corrected[k * 2]     = r * cos_c - s * sin_c;
        corrected[k * 2 + 1] = r * sin_c + s * cos_c;
    }

    // One-sided normalisation: normalise EMBEDDING, leave ODE untouched.
    // score = Σ corrected × (emb / |emb|) = Σ mag_ode × cos(Δθ)
    // With flat embeddings: identical to dot product (|emb|=1.0).
    // With Pythagorean: removes decay bias, keeps phase structure.
    let mut coherences = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let emb = &embeddings[v];
        let mut score = 0.0f32;
        for k in 0..n_bands {
            let r1 = corrected[k * 2];
            let s1 = corrected[k * 2 + 1];
            let r2 = emb[k * 2];
            let s2 = emb[k * 2 + 1];
            let emb_mag = (r2 * r2 + s2 * s2).sqrt().max(1e-8);
            score += r1 * (r2 / emb_mag) + s1 * (s2 / emb_mag);
        }
        coherences[v] = score / temperature;
    }

    // Softmax → cross-entropy
    let max_coh = coherences.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_cohs: Vec<f32> = coherences.iter().map(|&c| (c - max_coh).exp()).collect();
    let sum_exp: f32 = exp_cohs.iter().sum();
    let probs: Vec<f32> = exp_cohs.iter().map(|&e| e / sum_exp).collect();
    let loss = -probs[target].max(1e-10).ln();

    // Gradient: d_score/d_corrected[j] = emb_normalised[j] (simple)
    let mut d_corrected = vec![0.0f32; n_embd];
    for v in 0..vocab_size {
        let weight = (probs[v] - if v == target { 1.0 } else { 0.0 }) / temperature;
        let emb = &embeddings[v];
        for k in 0..n_bands {
            let r2 = emb[k * 2];
            let s2 = emb[k * 2 + 1];
            let emb_mag = (r2 * r2 + s2 * s2).sqrt().max(1e-8);
            d_corrected[k * 2] += weight * r2 / emb_mag;
            d_corrected[k * 2 + 1] += weight * s2 / emb_mag;
        }
    }

    // Chain through output corrector rotation
    let mut d_hidden = vec![0.0f32; n_embd];
    let mut d_corrector = vec![0.0f32; n_bands];
    for k in 0..n_bands {
        let (sin_c, cos_c) = output_corrector[k].sin_cos();
        let dc_r = d_corrected[k * 2];
        let dc_s = d_corrected[k * 2 + 1];
        let r = hidden[k * 2];
        let s = hidden[k * 2 + 1];

        d_hidden[k * 2]     =  cos_c * dc_r + sin_c * dc_s;
        d_hidden[k * 2 + 1] = -sin_c * dc_r + cos_c * dc_s;

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
        let n_bands = 4;
        let hidden: Vec<f32> = vec![0.5, 0.3, -0.2, 0.8, 0.1, -0.4, 0.7, 0.2];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            vec![-1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0],
        ];
        let target = 1;
        let temp = 1.0;
        let corrector = vec![0.1, -0.2, 0.05, 0.3];
        let (loss, grad, d_corr) = phase_native_loss(&hidden, &embeddings, target, n_bands, temp, &corrector);
        assert!(loss > 0.0);

        let eps = 1e-4;
        // Check hidden gradients
        for i in 0..hidden.len() {
            let mut h_plus = hidden.clone();
            let mut h_minus = hidden.clone();
            h_plus[i] += eps;
            h_minus[i] -= eps;
            let (l_plus, _, _) = phase_native_loss(&h_plus, &embeddings, target, n_bands, temp, &corrector);
            let (l_minus, _, _) = phase_native_loss(&h_minus, &embeddings, target, n_bands, temp, &corrector);
            let fd = (l_plus - l_minus) / (2.0 * eps);
            let rel_err = if grad[i].abs() > 1e-6 { (fd - grad[i]).abs() / grad[i].abs() } else { (fd - grad[i]).abs() };
            assert!(rel_err < 0.05, "Hidden grad mismatch at {}: fd={:.6} analytical={:.6} rel_err={:.4}", i, fd, grad[i], rel_err);
        }
        // Check corrector gradients
        for i in 0..corrector.len() {
            let mut c_plus = corrector.clone();
            let mut c_minus = corrector.clone();
            c_plus[i] += eps;
            c_minus[i] -= eps;
            let (l_plus, _, _) = phase_native_loss(&hidden, &embeddings, target, n_bands, temp, &c_plus);
            let (l_minus, _, _) = phase_native_loss(&hidden, &embeddings, target, n_bands, temp, &c_minus);
            let fd = (l_plus - l_minus) / (2.0 * eps);
            let rel_err = if d_corr[i].abs() > 1e-6 { (fd - d_corr[i]).abs() / d_corr[i].abs() } else { (fd - d_corr[i]).abs() };
            assert!(rel_err < 0.05, "Corrector grad mismatch at {}: fd={:.6} analytical={:.6} rel_err={:.4}", i, fd, d_corr[i], rel_err);
        }
    }
}
