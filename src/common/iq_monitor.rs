//! I/Q channel monitor — measures whether ODE computation lives in phase or amplitude.
//! Run at health intervals to diagnose the detection bottleneck.

/// I/Q channel analysis results for one position.
pub struct IqAnalysis {
    pub i_discrim: f32,      // I-channel discrimination (correct - mean wrong)
    pub q_discrim: f32,      // Q-channel discrimination
    pub iq_ratio: f32,       // Q power / I power
    pub phase_mean: f32,     // mean phase shift (rad) for correct token
    pub phase_std: f32,      // std of phase shifts across bands
    pub i_correct_rank: usize, // rank of correct token by I score (1 = best)
    pub q_correct_rank: usize, // rank of correct token by Q score
}

/// Analyze I/Q channels for one position's output against embedding table.
pub fn analyze_iq(
    hidden: &[f32],           // [n_embd] post-LN output
    embeddings: &[Vec<f32>],  // [vocab][n_embd] embedding table
    target: usize,
    n_bands: usize,
    output_corrector: &[f32],
    output_scale: &[f32],
) -> IqAnalysis {
    let n_embd = n_bands * 2;
    let vocab_size = embeddings.len();

    // Apply corrector + scale (same as phase_loss)
    let mut corrected = vec![0.0f32; n_embd];
    for k in 0..n_bands {
        let (sin_c, cos_c) = output_corrector[k].sin_cos();
        let r = hidden[k * 2];
        let s = hidden[k * 2 + 1];
        corrected[k * 2]     = (r * cos_c - s * sin_c) * output_scale[k];
        corrected[k * 2 + 1] = (r * sin_c + s * cos_c) * output_scale[k];
    }

    // Compute I and Q scores for each token
    let mut i_scores = vec![0.0f32; vocab_size];
    let mut q_scores = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let emb = &embeddings[v];
        let mut i_sum = 0.0f32;
        let mut q_sum = 0.0f32;
        for k in 0..n_bands {
            let r_out = corrected[k * 2];
            let s_out = corrected[k * 2 + 1];
            let r_emb = emb[k * 2];
            let s_emb = emb[k * 2 + 1];
            i_sum += r_out * r_emb + s_out * s_emb;
            q_sum += s_out * r_emb - r_out * s_emb;
        }
        i_scores[v] = i_sum;
        q_scores[v] = q_sum;
    }

    // Discrimination: correct - mean(wrong)
    let i_correct = i_scores[target];
    let i_wrong_sum: f32 = i_scores.iter().sum::<f32>() - i_correct;
    let i_discrim = i_correct - i_wrong_sum / (vocab_size - 1).max(1) as f32;

    let q_correct = q_scores[target];
    let q_wrong_sum: f32 = q_scores.iter().sum::<f32>() - q_correct;
    let q_discrim = q_correct - q_wrong_sum / (vocab_size - 1).max(1) as f32;

    // I/Q power ratio
    let i_power: f32 = i_scores.iter().map(|s| s * s).sum();
    let q_power: f32 = q_scores.iter().map(|s| s * s).sum();
    let iq_ratio = q_power / (i_power + 1e-10);

    // Rank of correct token
    let i_correct_rank = i_scores.iter().filter(|&&s| s > i_correct).count() + 1;
    let q_correct_rank = q_scores.iter().filter(|&&s| s > q_correct).count() + 1;

    // Per-band phase shift for correct token
    let emb_target = &embeddings[target];
    let mut phase_shifts = Vec::with_capacity(n_bands);
    for k in 0..n_bands {
        let i_k = corrected[k*2] * emb_target[k*2] + corrected[k*2+1] * emb_target[k*2+1];
        let q_k = corrected[k*2+1] * emb_target[k*2] - corrected[k*2] * emb_target[k*2+1];
        phase_shifts.push(q_k.atan2(i_k));
    }
    let n = phase_shifts.len() as f32;
    let phase_mean = phase_shifts.iter().sum::<f32>() / n;
    let phase_std = (phase_shifts.iter().map(|&p| (p - phase_mean) * (p - phase_mean)).sum::<f32>() / n).sqrt();

    IqAnalysis {
        i_discrim, q_discrim, iq_ratio,
        phase_mean, phase_std,
        i_correct_rank, q_correct_rank,
    }
}

/// Run I/Q analysis on multiple positions, return averaged results.
pub fn analyze_iq_batch(
    hidden_states: &[Vec<f32>],  // [n_pos][n_embd]
    embeddings: &[Vec<f32>],
    targets: &[usize],
    n_bands: usize,
    output_corrector: &[f32],
    output_scale: &[f32],
    max_positions: usize,
) -> IqAnalysis {
    let n_pos = hidden_states.len().min(targets.len()).min(max_positions);
    if n_pos == 0 {
        return IqAnalysis {
            i_discrim: 0.0, q_discrim: 0.0, iq_ratio: 0.0,
            phase_mean: 0.0, phase_std: 0.0,
            i_correct_rank: 0, q_correct_rank: 0,
        };
    }

    let mut sum_i = 0.0f32;
    let mut sum_q = 0.0f32;
    let mut sum_iq = 0.0f32;
    let mut sum_pm = 0.0f32;
    let mut sum_ps = 0.0f32;
    let mut sum_ir = 0usize;
    let mut sum_qr = 0usize;

    for pos in 0..n_pos {
        let a = analyze_iq(&hidden_states[pos], embeddings, targets[pos],
                           n_bands, output_corrector, output_scale);
        sum_i += a.i_discrim;
        sum_q += a.q_discrim;
        sum_iq += a.iq_ratio;
        sum_pm += a.phase_mean;
        sum_ps += a.phase_std;
        sum_ir += a.i_correct_rank;
        sum_qr += a.q_correct_rank;
    }

    let n = n_pos as f32;
    IqAnalysis {
        i_discrim: sum_i / n,
        q_discrim: sum_q / n,
        iq_ratio: sum_iq / n,
        phase_mean: sum_pm / n,
        phase_std: sum_ps / n,
        i_correct_rank: sum_ir / n_pos,
        q_correct_rank: sum_qr / n_pos,
    }
}
