//! Learnable attention — parallel path to frozen attn.rs.
//!
//! Forward math is identical to attn.rs (harmonic coherence scoring).
//! The difference: this path's weights receive gradients from backward()
//! and are updated by Adam. The frozen path in attn.rs stays untouched.
//!
//! Gradients computed for: phase_proj_w/b, v_proj_w/b, harmonic_raw, out_proj_w/b.
//! Content projection (content_proj_w/b) stays frozen even in this path —
//! isolates "does learnable attention help" from "does learnable content routing help."
//!
//! Created per LEARNABLE-ATTENTION-PARALLEL-SPEC.md (step 4).

use super::attn::{WaveAttnWeights, WaveAttnHeadWeights, project_phase};

/// Gradients for one attention head.
#[derive(Clone)]
pub struct AttnHeadGrads {
    pub d_phase_proj_w: Vec<Vec<f32>>,  // [2][n_embd]
    pub d_phase_proj_b: Vec<f32>,        // [2]
    pub d_v_proj_w: Vec<Vec<f32>>,       // [head_dim][head_dim]
    pub d_v_proj_b: Vec<f32>,            // [head_dim]
    pub d_harmonic_raw: f32,             // scalar (pre-softplus)
}

/// Gradients for the full multi-head attention layer.
#[derive(Clone)]
pub struct AttnLayerGrads {
    pub heads: Vec<AttnHeadGrads>,
    pub d_out_proj_w: Vec<Vec<f32>>,  // [n_embd][n_embd]
    pub d_out_proj_b: Vec<f32>,        // [n_embd]
}

/// Create zero-initialised attention gradients for one layer.
pub fn zero_attn_grads(n_embd: usize, n_head: usize) -> AttnLayerGrads {
    let head_dim = n_embd / n_head;
    AttnLayerGrads {
        heads: (0..n_head).map(|_| AttnHeadGrads {
            d_phase_proj_w: vec![vec![0.0; n_embd]; 2],
            d_phase_proj_b: vec![0.0; 2],
            d_v_proj_w: vec![vec![0.0; head_dim]; head_dim],
            d_v_proj_b: vec![0.0; head_dim],
            d_harmonic_raw: 0.0,
        }).collect(),
        d_out_proj_w: vec![vec![0.0; n_embd]; n_embd],
        d_out_proj_b: vec![0.0; n_embd],
    }
}

/// Flatten attention gradients into a flat vector matching the serialization order
/// in flatten_params_ex (wave_model.rs). Order per head:
///   phase_proj_w[2][n_embd], phase_proj_b[2], v_proj_w[hd][hd], v_proj_b[hd],
///   harmonic_raw (only if dyn_harmonics is NOT separately active)
/// Then: out_proj_w[n_embd][n_embd], out_proj_b[n_embd]
pub fn flatten_attn_grads(grads: &AttnLayerGrads, skip_harmonic: bool) -> Vec<f32> {
    let mut g = Vec::new();
    for head in &grads.heads {
        for row in &head.d_phase_proj_w { g.extend_from_slice(row); }
        g.extend_from_slice(&head.d_phase_proj_b);
        for row in &head.d_v_proj_w { g.extend_from_slice(row); }
        g.extend_from_slice(&head.d_v_proj_b);
        if !skip_harmonic {
            g.push(head.d_harmonic_raw);
        }
    }
    for row in &grads.d_out_proj_w { g.extend_from_slice(row); }
    g.extend_from_slice(&grads.d_out_proj_b);
    g
}

/// Backward pass for learnable attention on one layer.
///
/// Takes:
///   - weights: the attention weights for this block
///   - normed: the layer-normed input [t][n_embd] (same input as forward)
///   - att_weights: the cached attention weights [n_head][t][t] from forward
///   - d_attn_out: gradient of loss w.r.t. the attention output [t][n_embd]
///   - n_bands: number of frequency bands
///
/// Returns: AttnLayerGrads with gradients for all attention parameters.
pub fn wave_attention_backward(
    weights: &WaveAttnWeights,
    normed: &[Vec<f32>],
    att_weights: &[Vec<Vec<f32>>],
    d_attn_out: &[Vec<f32>],
    n_bands: usize,
) -> AttnLayerGrads {
    let t = normed.len();
    let n_embd = n_bands * 2;
    let n_head = weights.heads.len();
    let head_dim = n_embd / n_head;

    let mut grads = zero_attn_grads(n_embd, n_head);

    // ─── Step 1: Backward through output projection ───
    // result[pos][i] = sum_j out_proj_w[i][j] * concat[pos][j] + out_proj_b[i]
    // d_concat[pos][j] = sum_i d_attn_out[pos][i] * out_proj_w[i][j]
    let d_concat: Vec<Vec<f32>> = (0..t).map(|pos| {
        let mut dc = vec![0.0f32; n_embd];
        for j in 0..n_embd {
            let mut sum = 0.0f32;
            for i in 0..n_embd { sum += d_attn_out[pos][i] * weights.out_proj_w[i][j]; }
            dc[j] = sum;
        }
        dc
    }).collect();

    // We need per-head pre-out_proj outputs for the out_proj_w gradient.
    // These aren't cached, so we recompute per head below and accumulate.

    // ─── Step 2-5: Per-head backward ───
    for h in 0..n_head {
        let harmonic_n = super::math::softplus(weights.heads[h].harmonic_raw);
        let offset = h * head_dim;

        // Recompute phases with raw (r, s) for atan2 backward
        let phase_rs: Vec<(f32, f32, f32)> = normed.iter().map(|x| {
            let pw = &weights.heads[h].phase_proj_w;
            let pb = &weights.heads[h].phase_proj_b;
            let mut r = pb[0];
            let mut s = pb[1];
            for j in 0..n_embd { r += pw[0][j] * x[j]; s += pw[1][j] * x[j]; }
            (s.atan2(r), r, s)
        }).collect();
        let phases: Vec<f32> = phase_rs.iter().map(|&(p, _, _)| p).collect();

        // Recompute value projections
        let v_all: Vec<Vec<f32>> = (0..t).map(|pos| {
            let mut v = vec![0.0f32; head_dim];
            for dd in 0..head_dim {
                let mut sum = 0.0f32;
                for j in 0..head_dim { sum += weights.heads[h].v_proj_w[dd][j] * normed[pos][offset + j]; }
                v[dd] = sum + weights.heads[h].v_proj_b[dd];
            }
            v
        }).collect();

        let att_w = &att_weights[h];

        // Accumulators
        let mut d_h = 0.0f32;
        let mut d_phase_per_pos = vec![0.0f32; t];
        let mut d_v_all = vec![vec![0.0f32; head_dim]; t];
        let mut head_out = vec![vec![0.0f32; head_dim]; t]; // for out_proj gradient

        for qi in 0..t {
            let d_head: Vec<f32> = (0..head_dim).map(|dd| d_concat[qi][offset + dd]).collect();

            // Recompute head_out[qi] = sum_ki att_w[qi][ki] * v[ki]
            for dd in 0..head_dim {
                let mut sum = 0.0f32;
                for ki in 0..=qi {
                    if att_w[qi][ki] > 0.0 { sum += att_w[qi][ki] * v_all[ki][dd]; }
                }
                head_out[qi][dd] = sum;
            }

            // d_att_w[qi][ki] = sum_d d_head[d] * v[ki][d]
            let mut d_w_qi = vec![0.0f32; t];
            for ki in 0..=qi {
                if att_w[qi][ki] > 0.0 {
                    let mut dw = 0.0f32;
                    for dd in 0..head_dim { dw += d_head[dd] * v_all[ki][dd]; }
                    d_w_qi[ki] = dw;
                }
            }

            // d_v[ki][d] += att_w[qi][ki] * d_head[d]
            for ki in 0..=qi {
                if att_w[qi][ki] > 0.0 {
                    for dd in 0..head_dim { d_v_all[ki][dd] += att_w[qi][ki] * d_head[dd]; }
                }
            }

            // Softmax backward: d_score[ki] = w[ki] * (d_w[ki] - sum_j w[j] * d_w[j])
            let weighted_sum: f32 = (0..=qi).map(|ki| att_w[qi][ki] * d_w_qi[ki]).sum();
            let d_scores: Vec<f32> = (0..=qi).map(|ki| att_w[qi][ki] * (d_w_qi[ki] - weighted_sum)).collect();

            // score[ki] = cos(n * delta), delta = phase[qi] - phase[ki]
            for ki in 0..d_scores.len() {
                if att_w[qi][ki] > 0.0 {
                    let delta = phases[qi] - phases[ki];
                    let sin_nd = (harmonic_n * delta).sin();
                    // d_harmonic: d_score/d_n = -sin(n*delta) * delta
                    d_h += d_scores[ki] * (-sin_nd * delta);
                    // d_phase: d_score/d_phase_qi = -n * sin(n*delta)
                    let d_phase = d_scores[ki] * (-harmonic_n * sin_nd);
                    d_phase_per_pos[qi] += d_phase;
                    d_phase_per_pos[ki] -= d_phase;
                }
            }
        }

        // ─── Harmonic gradient ───
        grads.heads[h].d_harmonic_raw = d_h * super::math::softplus_derivative(weights.heads[h].harmonic_raw);

        // ─── Phase projection backward ───
        // phase = atan2(s, r); d_atan2/d_r = -s/(r²+s²), d_atan2/d_s = r/(r²+s²)
        for pos in 0..t {
            let dp = d_phase_per_pos[pos];
            if dp.abs() < 1e-12 { continue; }
            let (_, r, s) = phase_rs[pos];
            let r2s2 = r * r + s * s + 1e-8; // epsilon near origin
            let d_r = dp * (-s / r2s2);
            let d_s = dp * (r / r2s2);
            for j in 0..n_embd {
                grads.heads[h].d_phase_proj_w[0][j] += d_r * normed[pos][j];
                grads.heads[h].d_phase_proj_w[1][j] += d_s * normed[pos][j];
            }
            grads.heads[h].d_phase_proj_b[0] += d_r;
            grads.heads[h].d_phase_proj_b[1] += d_s;
        }

        // ─── Value projection backward ───
        // v[d] = v_proj_w[d] @ x[offset..offset+hd] + v_proj_b[d]
        for pos in 0..t {
            for dd in 0..head_dim {
                let dv = d_v_all[pos][dd];
                if dv.abs() < 1e-12 { continue; }
                for j in 0..head_dim {
                    grads.heads[h].d_v_proj_w[dd][j] += dv * normed[pos][offset + j];
                }
                grads.heads[h].d_v_proj_b[dd] += dv;
            }
        }

        // ─── Output projection gradient (accumulate per head) ───
        // d_out_proj_w[i][j] += sum_pos d_attn_out[pos][i] * concat[pos][j]
        // concat[pos][offset+d] = head_out[pos][d]
        for pos in 0..t {
            for i in 0..n_embd {
                let dai = d_attn_out[pos][i];
                if dai.abs() < 1e-12 { continue; }
                for dd in 0..head_dim {
                    grads.d_out_proj_w[i][offset + dd] += dai * head_out[pos][dd];
                }
            }
        }
    }

    // Output projection bias (once, all heads combined)
    for pos in 0..t {
        for i in 0..n_embd {
            grads.d_out_proj_b[i] += d_attn_out[pos][i];
        }
    }

    grads
}
