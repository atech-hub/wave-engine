//! Harmonic coherence attention (CPU, frozen) + manual harmonic backward.

#[cfg(feature = "candle-backend")]
pub mod attention {
    use candle_core::{Result, Tensor};

    use crate::candle_tier::candle_model::model::CandleBlock;

    // ─── Harmonic Coherence Attention (CPU, frozen) ───

    pub fn wave_attention(
        x: &Tensor,
        pp_ws_cpu: &[Vec<Vec<f32>>],  // pre-cached on CPU
        pp_bs_cpu: &[Vec<f32>],
        vw_cpu: &[Vec<Vec<f32>>],
        vb_cpu: &[Vec<f32>],
        harmonic_ns: &[f32],
        out_proj_w: &Tensor,
        out_proj_b: &Tensor,
        store_attn_weights: bool,     // true when --harmonics dyn
    ) -> Result<(Tensor, Option<Vec<Vec<Vec<f32>>>>)> {
        let (n_pos, n_embd) = x.dims2()?;
        let n_head = harmonic_ns.len();
        let head_dim = n_embd / n_head;

        // Only ONE GPU→CPU transfer: the input activations (these change every call)
        let x_data = x.to_vec2::<f32>()?;
        let mut out_data = vec![0.0f32; n_pos * n_embd];

        // Optional attention weight storage for harmonic backward
        let mut all_att_weights: Option<Vec<Vec<Vec<f32>>>> = if store_attn_weights {
            Some(vec![vec![vec![0.0; n_pos]; n_pos]; n_head])
        } else {
            None
        };

        for head in 0..n_head {
            let offset = head * head_dim;
            let harmonic_n = crate::common::math::softplus(harmonic_ns[head]);

            // Phase projection — from CPU cache, zero GPU transfers
            let pp_w = &pp_ws_cpu[head];
            let pp_b = &pp_bs_cpu[head];
            let phases: Vec<f32> = (0..n_pos).map(|pos| {
                let mut r = pp_b[0];
                let mut s = pp_b[1];
                for j in 0..n_embd { r += pp_w[0][j] * x_data[pos][j]; s += pp_w[1][j] * x_data[pos][j]; }
                s.atan2(r)
            }).collect();

            // Value projection — from CPU cache, zero GPU transfers
            let vw = &vw_cpu[head];
            let vb = &vb_cpu[head];
            let v_all: Vec<Vec<f32>> = (0..n_pos).map(|pos| {
                let mut v = vec![0.0f32; head_dim];
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..head_dim { sum += vw[d][j] * x_data[pos][offset + j]; }
                    v[d] = sum + vb[d];
                }
                v
            }).collect();

            // Scoring + weighted sum
            for qi in 0..n_pos {
                let mut scores = vec![f32::NEG_INFINITY; n_pos];
                for ki in 0..=qi {
                    let delta = phases[qi] - phases[ki];
                    scores[ki] = (harmonic_n * delta).cos();
                }
                let max_s = scores[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for ki in 0..=qi { scores[ki] = (scores[ki] - max_s).exp(); exp_sum += scores[ki]; }
                if exp_sum > 0.0 { for ki in 0..=qi { scores[ki] /= exp_sum; } }

                // Store the softmax weights for backward
                if let Some(ref mut aw) = all_att_weights {
                    for ki in 0..=qi {
                        aw[head][qi][ki] = scores[ki];
                    }
                }

                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for ki in 0..=qi { sum += scores[ki] * v_all[ki][d]; }
                    out_data[qi * n_embd + offset + d] = sum;
                }
            }
        }

        // Back to tensor, then out_proj through Candle (GPU, frozen but on grad graph for residual)
        let out_tensor = Tensor::from_vec(out_data, (n_pos, n_embd), x.device())?;
        let projected = out_tensor.matmul(&out_proj_w.t()?)?.broadcast_add(out_proj_b)?;
        Ok((projected, all_att_weights))
    }

    // ─── Harmonic backward (manual chain rule — attention runs on CPU, outside autograd) ───

    /// Compute harmonic gradients for one block.
    /// Called AFTER candle backward, using d_out from the grad accumulator.
    /// d_out is [t][n_embd] — the gradient of the block's contribution tensor.
    /// This equals d_attn_out because contribution = attn_out + ffn_out (sum splits gradient equally).
    pub fn harmonic_backward(
        block: &CandleBlock,
        d_out: &[Vec<f32>],        // [t][n_embd] — gradient of contribution (= d_attn_out)
        n_embd: usize,
    ) -> Vec<f32> {                 // [n_head] — d_loss/d_harmonic_raw per head
        let t = d_out.len();
        let n_head = block.harmonic_ns.len();
        let head_dim = n_embd / n_head;

        fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

        let att_w = block.cached_att_weights.as_ref()
            .expect("harmonic backward requires cached attention weights");
        let input = block.cached_normed_cpu.as_ref()
            .expect("harmonic backward requires cached normed input");

        let mut d_harmonic_raws = vec![0.0f32; n_head];

        for h in 0..n_head {
            let harmonic_n = crate::common::math::softplus(block.harmonic_ns[h]);
            let offset = h * head_dim;

            // Recompute phases (same as forward)
            let pp_w = &block.phase_proj_ws_cpu[h];
            let pp_b = &block.phase_proj_bs_cpu[h];
            let phases: Vec<f32> = (0..t).map(|pos| {
                let mut r = pp_b[0];
                let mut s = pp_b[1];
                for j in 0..n_embd { r += pp_w[0][j] * input[pos][j]; s += pp_w[1][j] * input[pos][j]; }
                s.atan2(r)
            }).collect();

            // Recompute value projections (same as forward)
            let vw = &block.v_proj_ws_cpu[h];
            let vb = &block.v_proj_bs_cpu[h];
            let v_all: Vec<Vec<f32>> = (0..t).map(|pos| {
                let mut v = vec![0.0f32; head_dim];
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..head_dim { sum += vw[d][j] * input[pos][offset + j]; }
                    v[d] = sum + vb[d];
                }
                v
            }).collect();

            // Accumulate d_h across all query positions
            let mut d_h = 0.0f32;
            for qi in 0..t {
                // d_weight from d_output: d_w[qi][ki] = sum_d d_out[qi][offset+d] * v_all[ki][d]
                let mut d_w_qi = vec![0.0f32; t];
                for ki in 0..=qi {
                    if att_w[h][qi][ki] > 0.0 {
                        let mut dw = 0.0f32;
                        for d in 0..head_dim {
                            dw += d_out[qi][offset + d] * v_all[ki][d];
                        }
                        d_w_qi[ki] = dw;
                    }
                }

                // Softmax backward
                let weighted_sum: f32 = (0..=qi)
                    .map(|ki| att_w[h][qi][ki] * d_w_qi[ki])
                    .sum();

                // Accumulate through cosine derivative
                for ki in 0..=qi {
                    if att_w[h][qi][ki] > 0.0 {
                        let d_score = att_w[h][qi][ki] * (d_w_qi[ki] - weighted_sum);
                        let delta = phases[qi] - phases[ki];
                        let d_score_d_h = -(harmonic_n * delta).sin() * delta;
                        d_h += d_score * d_score_d_h;
                    }
                }
            }

            // Chain through softplus: d_loss/d_harmonic_raw = d_h * sigmoid(harmonic_raw)
            d_harmonic_raws[h] = d_h * sigmoid(block.harmonic_ns[h]);
        }

        d_harmonic_raws
    }
}
