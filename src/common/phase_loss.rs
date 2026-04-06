//! Phase-native loss: ODE output compared directly against embedding table.
//! The ODE learns to produce outputs in the same space as its inputs.
//! No lm_head needed — the embedding table IS the decoder.

/// Detection mode for phase-native decoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DetectMode {
    /// I-channel only: standard dot product (current default)
    I,
    /// Q-channel only: quadrature detection — reads phase modulation
    Q,
    /// Learned I/Q mix: 2 global weights (w_I, w_Q)
    IQ,
}

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
    output_scale: &[f32],     // [n_bands] — per-band amplitude scale (init 1.0)
    wave_translator: &[[f32; 4]], // [n_bands] per-band 2x2 [a,b,c,d], identity init
    input_embedding: Option<&[f32]>,  // delta decode: current token's embedding to subtract
    detect_mode: DetectMode,          // I, Q, or IQ detection
    iq_weights: Option<&[f32; 2]>,    // [w_I, w_Q] for IQ mode
) -> (f32, Vec<f32>, Vec<f32>, Vec<f32>, Vec<[f32; 4]>, Option<[f32; 2]>) {  // + d_wave_translator
    let vocab_size = embeddings.len();
    let n_embd = n_bands * 2;

    // Output corrector: per-band phase rotation
    let mut corrected = vec![0.0f32; n_embd];
    for k in 0..n_bands {
        let (sin_c, cos_c) = output_corrector[k].sin_cos();
        let r = hidden[k * 2];
        let s = hidden[k * 2 + 1];
        corrected[k * 2]     = r * cos_c - s * sin_c;
        corrected[k * 2 + 1] = r * sin_c + s * cos_c;
    }

    // Output adapter: per-band amplitude scale
    let pre_scale = corrected.clone();
    for k in 0..n_bands {
        corrected[k * 2]     *= output_scale[k];
        corrected[k * 2 + 1] *= output_scale[k];
    }

    // Wave translator: per-band 2×2 transform (the actual decoder)
    // Gives the ODE complete freedom — translates its output back to embedding space
    let pre_translator = corrected.clone();
    for k in 0..n_bands {
        let r = corrected[k * 2];
        let s = corrected[k * 2 + 1];
        let [a, b, c, d] = wave_translator[k];
        corrected[k * 2]     = a * r + b * s;
        corrected[k * 2 + 1] = c * r + d * s;
    }

    // Delta decode: subtract input token's embedding to strip the residual echo.
    let decode_signal = if let Some(inp_emb) = input_embedding {
        let mut delta = vec![0.0f32; n_embd];
        for j in 0..n_embd {
            delta[j] = corrected[j] - inp_emb[j];
        }
        delta
    } else {
        corrected.clone()
    };

    // Coherent detection against embeddings
    let scale = 1.0 / (n_embd as f32).sqrt();
    let mut coherences = vec![0.0f32; vocab_size];
    let mut i_sums = vec![0.0f32; vocab_size]; // stored for IQ gradient
    let mut q_sums = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let emb = &embeddings[v];
        let mut i_sum = 0.0f32;
        let mut q_sum = 0.0f32;
        for k in 0..n_bands {
            let r_out = decode_signal[k * 2];
            let s_out = decode_signal[k * 2 + 1];
            let r_emb = emb[k * 2];
            let s_emb = emb[k * 2 + 1];
            i_sum += r_out * r_emb + s_out * s_emb;  // Re(z_out * conj(z_emb))
            q_sum += s_out * r_emb - r_out * s_emb;  // Im(z_out * conj(z_emb))
        }
        i_sums[v] = i_sum;
        q_sums[v] = q_sum;
        let score = match detect_mode {
            DetectMode::I => i_sum,
            DetectMode::Q => q_sum,
            DetectMode::IQ => {
                let w = iq_weights.unwrap_or(&[1.0, 0.0]);
                w[0] * i_sum + w[1] * q_sum
            }
        };
        coherences[v] = score * scale / temperature;
    }

    // Softmax over coherences → probabilities
    let max_coh = coherences.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_cohs: Vec<f32> = coherences.iter().map(|&c| (c - max_coh).exp()).collect();
    let sum_exp: f32 = exp_cohs.iter().sum();
    let probs: Vec<f32> = exp_cohs.iter().map(|&e| e / sum_exp).collect();

    // Cross-entropy loss
    let loss = -probs[target].max(1e-10).ln();

    // Gradient: d_loss/d_decode_signal — depends on detection mode
    // d_loss/d_score[v] = (probs[v] - (v == target)) * scale / temperature
    let mut d_decode = vec![0.0f32; n_embd];
    let mut d_iq: Option<[f32; 2]> = None;
    if detect_mode == DetectMode::IQ {
        let mut d_wi = 0.0f32;
        let mut d_wq = 0.0f32;
        let w = iq_weights.unwrap_or(&[1.0, 0.0]);
        for v in 0..vocab_size {
            let sv = (probs[v] - if v == target { 1.0 } else { 0.0 }) * scale / temperature;
            d_wi += sv * i_sums[v];
            d_wq += sv * q_sums[v];
            let emb = &embeddings[v];
            for k in 0..n_bands {
                let r_emb = emb[k * 2];
                let s_emb = emb[k * 2 + 1];
                // d_score/d_r_out = w_I * r_emb - w_Q * s_emb
                // d_score/d_s_out = w_I * s_emb + w_Q * r_emb
                d_decode[k * 2]     += sv * (w[0] * r_emb - w[1] * s_emb);
                d_decode[k * 2 + 1] += sv * (w[0] * s_emb + w[1] * r_emb);
            }
        }
        d_iq = Some([d_wi, d_wq]);
    } else {
        for v in 0..vocab_size {
            let sv = (probs[v] - if v == target { 1.0 } else { 0.0 }) * scale / temperature;
            let emb = &embeddings[v];
            for k in 0..n_bands {
                let r_emb = emb[k * 2];
                let s_emb = emb[k * 2 + 1];
                match detect_mode {
                    DetectMode::I => {
                        // d_score/d_r_out = r_emb, d_score/d_s_out = s_emb
                        d_decode[k * 2]     += sv * r_emb;
                        d_decode[k * 2 + 1] += sv * s_emb;
                    }
                    DetectMode::Q => {
                        // d_score/d_r_out = -s_emb, d_score/d_s_out = r_emb
                        d_decode[k * 2]     += sv * (-s_emb);
                        d_decode[k * 2 + 1] += sv * r_emb;
                    }
                    DetectMode::IQ => unreachable!(),
                }
            }
        }
    }
    // d_decode is w.r.t. decode_signal; for delta decode, d_corrected = d_decode (linear)
    let d_corrected = d_decode;

    // Chain through wave translator: d_corrected → d_translator + d_pre_translator
    let mut d_wave_translator = vec![[0.0f32; 4]; n_bands];
    let mut d_post_scale = vec![0.0f32; n_embd]; // gradient w.r.t. pre-translator (post-scale) values
    for k in 0..n_bands {
        let dc_r = d_corrected[k * 2];
        let dc_s = d_corrected[k * 2 + 1];
        let r_pre = pre_translator[k * 2];
        let s_pre = pre_translator[k * 2 + 1];
        let [a, _, c, _] = wave_translator[k];
        // d_translator params
        d_wave_translator[k] = [dc_r * r_pre, dc_r * s_pre, dc_s * r_pre, dc_s * s_pre];
        // d_pre_translator (chain back to scale/corrector)
        d_post_scale[k * 2]     = dc_r * a + dc_s * c;
        d_post_scale[k * 2 + 1] = dc_r * wave_translator[k][1] + dc_s * wave_translator[k][3];
    }

    // Chain through output_scale
    let mut d_output_scale = vec![0.0f32; n_bands];
    let mut d_pre_scale = vec![0.0f32; n_embd];
    for k in 0..n_bands {
        d_output_scale[k] = d_post_scale[k * 2] * pre_scale[k * 2]
                          + d_post_scale[k * 2 + 1] * pre_scale[k * 2 + 1];
        d_pre_scale[k * 2]     = d_post_scale[k * 2]     * output_scale[k];
        d_pre_scale[k * 2 + 1] = d_post_scale[k * 2 + 1] * output_scale[k];
    }

    // Chain through output corrector rotation to get d_hidden and d_corrector
    let mut d_hidden = vec![0.0f32; n_embd];
    let mut d_corrector = vec![0.0f32; n_bands];
    for k in 0..n_bands {
        let (sin_c, cos_c) = output_corrector[k].sin_cos();
        let dc_r = d_pre_scale[k * 2];
        let dc_s = d_pre_scale[k * 2 + 1];
        let r = hidden[k * 2];
        let s = hidden[k * 2 + 1];

        // d_hidden: inverse rotation
        d_hidden[k * 2]     =  cos_c * dc_r + sin_c * dc_s;
        d_hidden[k * 2 + 1] = -sin_c * dc_r + cos_c * dc_s;

        // d_corrector[k]: gradient of rotation angle
        d_corrector[k] = dc_r * (-r * sin_c - s * cos_c)
                       + dc_s * ( r * cos_c - s * sin_c);
    }

    (loss, d_hidden, d_corrector, d_output_scale, d_wave_translator, d_iq)
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
        let (loss, grad, d_corr, _d_scale, _d_wt, _d_iq) = phase_native_loss(&hidden, &embeddings, target, n_bands, temp, &corrector, &vec![1.0; n_bands], &vec![[1.0, 0.0, 0.0, 1.0]; n_bands], None, DetectMode::I, None);
        assert!(loss > 0.0, "Loss should be positive");

        // Finite difference check
        let eps = 1e-4;
        for i in 0..hidden.len() {
            let mut h_plus = hidden.clone();
            let mut h_minus = hidden.clone();
            h_plus[i] += eps;
            h_minus[i] -= eps;
            let (l_plus, _, _, _, _, _) = phase_native_loss(&h_plus, &embeddings, target, n_bands, temp, &corrector, &vec![1.0; n_bands], &vec![[1.0, 0.0, 0.0, 1.0]; n_bands], None, DetectMode::I, None);
            let (l_minus, _, _, _, _, _) = phase_native_loss(&h_minus, &embeddings, target, n_bands, temp, &corrector, &vec![1.0; n_bands], &vec![[1.0, 0.0, 0.0, 1.0]; n_bands], None, DetectMode::I, None);
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
            let (l_plus, _, _, _, _, _) = phase_native_loss(&hidden, &embeddings, target, n_bands, temp, &c_plus, &vec![1.0; n_bands], &vec![[1.0, 0.0, 0.0, 1.0]; n_bands], None, DetectMode::I, None);
            let (l_minus, _, _, _, _, _) = phase_native_loss(&hidden, &embeddings, target, n_bands, temp, &c_minus, &vec![1.0; n_bands], &vec![[1.0, 0.0, 0.0, 1.0]; n_bands], None, DetectMode::I, None);
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
