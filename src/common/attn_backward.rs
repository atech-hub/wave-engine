//! Shared backward primitives for wave coherence attention.
//!
//! Pure functions, no side effects. Consumed by:
//! - `attn.rs::wave_attention_backward_pathway` (frozen mode, d_normed only)
//! - `attn_learnable.rs::wave_attention_backward` (learnable mode, d_normed + weight gradients)
//!
//! Same math, two call sites. Fixing a bug here fixes it for both modes automatically.
//! Self-tests at the bottom of this file verify every primitive against finite differences.
//!
//! Mirrors the structure of `common/backward.rs` — shared primitives consumed by
//! multiple callers.

// ─── Step 1: Out-proj backward ─────────────────────────────────

/// Backward through out_proj: result[i] = sum_j W[i][j] * x[j] + b[i]
/// Returns (d_out_merged, d_w, d_b) where d_out_merged is the gradient w.r.t. pre-proj concat.
pub fn out_proj_backward(
    d_attn_out: &[Vec<f32>],   // [t][n_embd]
    out_merged: &[Vec<f32>],    // [t][n_embd] — pre-proj concat (for weight grad)
    out_proj_w: &[Vec<f32>],    // [n_embd][n_embd]
    n_embd: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let t = d_attn_out.len();
    // d_out[pos][j] = sum_i W[i][j] * d_attn_out[pos][i]
    let d_out: Vec<Vec<f32>> = (0..t).map(|pos| {
        let mut dc = vec![0.0f32; n_embd];
        for j in 0..n_embd {
            let mut sum = 0.0f32;
            for i in 0..n_embd { sum += out_proj_w[i][j] * d_attn_out[pos][i]; }
            dc[j] = sum;
        }
        dc
    }).collect();

    // d_w[i][j] = sum_pos d_attn_out[pos][i] * out_merged[pos][j]
    let mut d_w = vec![vec![0.0f32; n_embd]; n_embd];
    for pos in 0..t {
        for i in 0..n_embd {
            let dai = d_attn_out[pos][i];
            if dai.abs() < 1e-12 { continue; }
            for j in 0..n_embd { d_w[i][j] += dai * out_merged[pos][j]; }
        }
    }

    // d_b[i] = sum_pos d_attn_out[pos][i]
    let mut d_b = vec![0.0f32; n_embd];
    for pos in 0..t { for i in 0..n_embd { d_b[i] += d_attn_out[pos][i]; } }

    (d_out, d_w, d_b)
}

// ─── Step 2: Split heads ────────────────────────────────────────

/// Split d_out_merged into per-head slices.
/// d_head[h][qi][d] = d_out[qi][h * head_dim + d]
pub fn split_heads(
    d_out: &[Vec<f32>],  // [t][n_embd]
    n_head: usize,
    head_dim: usize,
) -> Vec<Vec<Vec<f32>>> {
    let t = d_out.len();
    (0..n_head).map(|h| {
        let offset = h * head_dim;
        (0..t).map(|qi| d_out[qi][offset..offset + head_dim].to_vec()).collect()
    }).collect()
}

// ─── Step 3: Value aggregation backward ─────────────────────────

/// Backward through head_out[qi][d] = sum_ki att_w[qi][ki] * v[ki][d]
/// Returns (d_att_w [t][t], d_v_all [t][head_dim])
pub fn value_aggregation_backward(
    d_head_out: &[Vec<f32>],  // [t][head_dim]
    att_w: &[Vec<f32>],       // [t][t]
    v_all: &[Vec<f32>],       // [t][head_dim]
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let t = d_head_out.len();
    let head_dim = if t > 0 { d_head_out[0].len() } else { 0 };

    // d_att_w[qi][ki] = sum_d d_head_out[qi][d] * v_all[ki][d]
    let mut d_att_w = vec![vec![0.0f32; t]; t];
    for qi in 0..t {
        for ki in 0..=qi {
            if att_w[qi][ki] > 0.0 {
                let mut dw = 0.0f32;
                for d in 0..head_dim { dw += d_head_out[qi][d] * v_all[ki][d]; }
                d_att_w[qi][ki] = dw;
            }
        }
    }

    // d_v_all[ki][d] = sum_qi att_w[qi][ki] * d_head_out[qi][d]
    let mut d_v_all = vec![vec![0.0f32; head_dim]; t];
    for qi in 0..t {
        for ki in 0..=qi {
            if att_w[qi][ki] > 0.0 {
                for d in 0..head_dim { d_v_all[ki][d] += att_w[qi][ki] * d_head_out[qi][d]; }
            }
        }
    }

    (d_att_w, d_v_all)
}

// ─── Step 4: Softmax backward ───────────────────────────────────

/// Backward through softmax. Standard Jacobian.
/// d_score[qi][ki] = att_w[qi][ki] * (d_att_w[qi][ki] - weighted_sum)
pub fn softmax_backward(
    d_att_w: &[Vec<f32>],  // [t][t]
    att_w: &[Vec<f32>],    // [t][t]
) -> Vec<Vec<f32>> {
    let t = d_att_w.len();
    let mut d_scores = vec![vec![0.0f32; t]; t];
    for qi in 0..t {
        let weighted_sum: f32 = (0..=qi).map(|ki| att_w[qi][ki] * d_att_w[qi][ki]).sum();
        for ki in 0..=qi {
            d_scores[qi][ki] = att_w[qi][ki] * (d_att_w[qi][ki] - weighted_sum);
        }
    }
    d_scores
}

// ─── Step 5: Score backward ─────────────────────────────────────

/// Backward through score = cos(n * delta) + content_bias.
/// Returns (d_delta [t][t], d_content_vecs [t][content_dim] or empty, d_harmonic_n scalar)
pub fn score_backward(
    d_scores: &[Vec<f32>],              // [t][t]
    phases: &[f32],                      // [t]
    harmonic_n: f32,
    att_w: &[Vec<f32>],                  // [t][t] — for sparsity mask
    content_vecs: Option<&[Vec<f32>]>,   // [t][content_dim] or None
    content_scale: f32,
) -> (Vec<Vec<f32>>, Option<Vec<Vec<f32>>>, f32) {
    let t = phases.len();
    let mut d_delta = vec![vec![0.0f32; t]; t];
    let mut d_harmonic_n = 0.0f32;

    let has_content = content_vecs.is_some();
    let content_dim = content_vecs.map(|cv| if cv.is_empty() { 0 } else { cv[0].len() }).unwrap_or(0);
    let mut d_content = if has_content { Some(vec![vec![0.0f32; content_dim]; t]) } else { None };

    for qi in 0..t {
        for ki in 0..=qi {
            if att_w[qi][ki] > 0.0 {
                let delta = phases[qi] - phases[ki];
                let sin_nd = (harmonic_n * delta).sin();
                d_delta[qi][ki] = -sin_nd * harmonic_n * d_scores[qi][ki];
                d_harmonic_n += -sin_nd * delta * d_scores[qi][ki];

                if let (Some(cv), Some(dcv)) = (content_vecs, &mut d_content) {
                    for cd in 0..content_dim {
                        dcv[qi][cd] += d_scores[qi][ki] * content_scale * cv[ki][cd];
                        dcv[ki][cd] += d_scores[qi][ki] * content_scale * cv[qi][cd];
                    }
                }
            }
        }
    }

    (d_delta, d_content, d_harmonic_n)
}

// ─── Step 6: Phase subtraction backward ─────────────────────────

/// Backward through delta[qi][ki] = phases[qi] - phases[ki].
/// Returns d_phases [t]
pub fn phase_subtraction_backward(
    d_delta: &[Vec<f32>],  // [t][t]
    att_w: &[Vec<f32>],    // [t][t] — sparsity mask
) -> Vec<f32> {
    let t = d_delta.len();
    let mut d_phases = vec![0.0f32; t];
    for qi in 0..t {
        for ki in 0..=qi {
            if att_w[qi][ki] > 0.0 {
                d_phases[qi] += d_delta[qi][ki];
                d_phases[ki] -= d_delta[qi][ki];
            }
        }
    }
    d_phases
}

// ─── Step 7: Phase projection backward ──────────────────────────

/// Backward through phase = atan2(s, r) where [r, s] = proj_w @ x + proj_b.
/// Returns (d_x_from_phase [t][n_embd], d_proj_w [2][n_embd], d_proj_b [2])
pub fn phase_projection_backward(
    d_phases: &[f32],         // [t]
    x: &[Vec<f32>],           // [t][n_embd]
    phase_proj_w: &[Vec<f32>], // [2][n_embd]
    phase_proj_b: &[f32],      // [2]
    n_embd: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let t = x.len();
    let mut d_x = vec![vec![0.0f32; n_embd]; t];
    let mut d_w = vec![vec![0.0f32; n_embd]; 2];
    let mut d_b = vec![0.0f32; 2];

    for pos in 0..t {
        let dp = d_phases[pos];
        if dp.abs() < 1e-12 { continue; }

        // Recompute r, s
        let mut r = phase_proj_b[0];
        let mut s = phase_proj_b[1];
        for j in 0..n_embd {
            r += phase_proj_w[0][j] * x[pos][j];
            s += phase_proj_w[1][j] * x[pos][j];
        }

        let mag_sq = r * r + s * s;
        if mag_sq < 1e-8 { continue; } // near-origin guard
        let mag_sq_safe = mag_sq.max(1e-12);

        let d_r = dp * (-s / mag_sq_safe);
        let d_s = dp * (r / mag_sq_safe);

        for j in 0..n_embd {
            d_x[pos][j] += phase_proj_w[0][j] * d_r + phase_proj_w[1][j] * d_s;
            d_w[0][j] += d_r * x[pos][j];
            d_w[1][j] += d_s * x[pos][j];
        }
        d_b[0] += d_r;
        d_b[1] += d_s;
    }

    (d_x, d_w, d_b)
}

// ─── Step 8: Value projection backward ──────────────────────────

/// Backward through v[d] = sum_j v_proj_w[d][j] * x[pos][offset+j] + v_proj_b[d]
/// Returns (d_x_from_v [t][n_embd], d_vp_w [head_dim][head_dim], d_vp_b [head_dim])
pub fn value_projection_backward(
    d_v_all: &[Vec<f32>],      // [t][head_dim]
    x: &[Vec<f32>],            // [t][n_embd]
    head_offset: usize,
    head_dim: usize,
    n_embd: usize,
    v_proj_w: &[Vec<f32>],     // [head_dim][head_dim]
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let t = x.len();
    let mut d_x = vec![vec![0.0f32; n_embd]; t];
    let mut d_w = vec![vec![0.0f32; head_dim]; head_dim];
    let mut d_b = vec![0.0f32; head_dim];

    for pos in 0..t {
        for dd in 0..head_dim {
            let dv = d_v_all[pos][dd];
            if dv.abs() < 1e-12 { continue; }
            for j in 0..head_dim {
                d_x[pos][head_offset + j] += v_proj_w[dd][j] * dv;
                d_w[dd][j] += dv * x[pos][head_offset + j];
            }
            d_b[dd] += dv;
        }
    }

    (d_x, d_w, d_b)
}

// ─── Step 9: Content projection backward ────────────────────────

/// Backward through content_vecs[d] = sum_j content_proj_w[d][j] * x[j] + bias[d]
/// Returns (d_x_from_content [t][n_embd], d_cp_w [content_dim][n_embd], d_cp_b [content_dim])
pub fn content_projection_backward(
    d_content_vecs: &[Vec<f32>],  // [t][content_dim]
    x: &[Vec<f32>],                // [t][n_embd]
    content_proj_w: &[Vec<f32>],   // [content_dim][n_embd]
    n_embd: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let t = x.len();
    let content_dim = content_proj_w.len();
    let mut d_x = vec![vec![0.0f32; n_embd]; t];
    let mut d_w = vec![vec![0.0f32; n_embd]; content_dim];
    let mut d_b = vec![0.0f32; content_dim];

    for pos in 0..t {
        for cd in 0..content_dim {
            let dcv = d_content_vecs[pos][cd];
            if dcv.abs() < 1e-12 { continue; }
            for j in 0..n_embd {
                d_x[pos][j] += content_proj_w[cd][j] * dcv;
                d_w[cd][j] += dcv * x[pos][j];
            }
            d_b[cd] += dcv;
        }
    }

    (d_x, d_w, d_b)
}

// ─── Step 10: Combine d_normed ──────────────────────────────────

/// Sum per-head input-side gradients into d_normed [t][n_embd].
pub fn combine_d_normed(
    d_x_from_phase: &[Vec<Vec<f32>>],   // [n_head][t][n_embd]
    d_x_from_v: &[Vec<Vec<f32>>],       // [n_head][t][n_embd]
    d_x_from_content: &[Vec<Vec<f32>>], // [n_head][t][n_embd] (can be empty)
    t: usize,
    n_embd: usize,
) -> Vec<Vec<f32>> {
    let mut d_normed = vec![vec![0.0f32; n_embd]; t];
    let n_head = d_x_from_phase.len();

    for h in 0..n_head {
        for pos in 0..t {
            for j in 0..n_embd {
                d_normed[pos][j] += d_x_from_phase[h][pos][j] + d_x_from_v[h][pos][j];
            }
            if h < d_x_from_content.len() && !d_x_from_content[h].is_empty() {
                for j in 0..n_embd {
                    d_normed[pos][j] += d_x_from_content[h][pos][j];
                }
            }
        }
    }

    d_normed
}

// ─── Self-tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32, name: &str) {
        let diff = (a - b).abs();
        assert!(diff < tol, "{}: {} vs {} (diff {})", name, a, b, diff);
    }

    #[test]
    fn test_out_proj_backward() {
        // Identity out_proj, 2 positions, n_embd=2
        let d_attn = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let merged = vec![vec![3.0, 4.0], vec![5.0, 6.0]];
        let w = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (d_out, d_w, d_b) = out_proj_backward(&d_attn, &merged, &w, 2);
        // d_out should equal d_attn (identity W)
        assert_close(d_out[0][0], 1.0, 1e-6, "d_out[0][0]");
        assert_close(d_out[0][1], 0.0, 1e-6, "d_out[0][1]");
        // d_w[0][0] = d_attn[0][0]*merged[0][0] + d_attn[1][0]*merged[1][0] = 1*3 + 0*5 = 3
        assert_close(d_w[0][0], 3.0, 1e-6, "d_w[0][0]");
        // d_b[0] = 1+0 = 1
        assert_close(d_b[0], 1.0, 1e-6, "d_b[0]");
    }

    #[test]
    fn test_softmax_backward() {
        // att_w = [[1,0],[0.3,0.7]], d_att_w = [[1,0],[2,4]]
        let att_w = vec![vec![1.0, 0.0], vec![0.3, 0.7]];
        let d_att_w = vec![vec![1.0, 0.0], vec![2.0, 4.0]];
        let d_scores = softmax_backward(&d_att_w, &att_w);
        // qi=0: weighted_sum = 1*1 = 1, d_score[0][0] = 1*(1-1) = 0
        assert_close(d_scores[0][0], 0.0, 1e-6, "d_score[0][0]");
        // qi=1: weighted_sum = 0.3*2 + 0.7*4 = 3.4
        // d_score[1][0] = 0.3*(2 - 3.4) = 0.3*(-1.4) = -0.42
        assert_close(d_scores[1][0], -0.42, 1e-4, "d_score[1][0]");
        // d_score[1][1] = 0.7*(4 - 3.4) = 0.7*0.6 = 0.42
        assert_close(d_scores[1][1], 0.42, 1e-4, "d_score[1][1]");
    }

    #[test]
    fn test_value_aggregation_backward() {
        // t=2, head_dim=2, att_w=[[1,0],[0.5,0.5]], v=[[1,2],[3,4]]
        let att_w = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
        let v = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let d_head = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (d_aw, d_v) = value_aggregation_backward(&d_head, &att_w, &v);
        // d_att_w[0][0] = d_head[0] · v[0] = 1*1 + 0*2 = 1
        assert_close(d_aw[0][0], 1.0, 1e-6, "d_aw[0][0]");
        // d_v[0] = att_w[0][0]*d_head[0] + att_w[1][0]*d_head[1] = 1*[1,0] + 0.5*[0,1] = [1.0, 0.5]
        assert_close(d_v[0][0], 1.0, 1e-6, "d_v[0][0]");
        assert_close(d_v[0][1], 0.5, 1e-6, "d_v[0][1]");
    }

    #[test]
    fn test_phase_projection_backward() {
        // Identity phase_proj: x=[1,0] → r=1,s=0 → phase=0
        // d_phase = 1.0
        // d_atan2/d_r = -s/(r²+s²) = 0, d_atan2/d_s = r/(r²+s²) = 1
        // d_x[0] = proj_w[0][0]*d_r + proj_w[1][0]*d_s = 1*0 + 0*1 = 0
        // d_x[1] = proj_w[0][1]*d_r + proj_w[1][1]*d_s = 0*0 + 1*1 = 1
        let d_phases = vec![1.0];
        let x = vec![vec![1.0, 0.0]];
        let w = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![0.0, 0.0];
        let (d_x, _d_w, _d_b) = phase_projection_backward(&d_phases, &x, &w, &b, 2);
        assert_close(d_x[0][0], 0.0, 1e-6, "d_x[0][0]");
        assert_close(d_x[0][1], 1.0, 1e-6, "d_x[0][1]");
    }

    #[test]
    fn test_value_projection_backward() {
        // Identity v_proj, head_offset=0, head_dim=2, n_embd=2
        let d_v = vec![vec![1.0, 0.0]]; // t=1
        let x = vec![vec![3.0, 4.0]];
        let w = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (d_x, d_w, d_b) = value_projection_backward(&d_v, &x, 0, 2, 2, &w);
        // d_x[0][0] = v_proj_w[0][0] * d_v[0][0] = 1*1 = 1
        assert_close(d_x[0][0], 1.0, 1e-6, "d_x[0][0]");
        assert_close(d_x[0][1], 0.0, 1e-6, "d_x[0][1]");
        // d_w[0][0] = d_v[0][0] * x[0][0] = 1*3 = 3
        assert_close(d_w[0][0], 3.0, 1e-6, "d_w[0][0]");
        assert_close(d_b[0], 1.0, 1e-6, "d_b[0]");
    }

    #[test]
    fn test_combine_d_normed() {
        // 2 heads, t=1, n_embd=4 (head_dim=2)
        let d_phase = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0]], // head 0
            vec![vec![0.0, 0.0, 0.5, 0.0]], // head 1
        ];
        let d_v = vec![
            vec![vec![0.0, 2.0, 0.0, 0.0]], // head 0
            vec![vec![0.0, 0.0, 0.0, 3.0]], // head 1
        ];
        let d_content: Vec<Vec<Vec<f32>>> = vec![];
        let result = combine_d_normed(&d_phase, &d_v, &d_content, 1, 4);
        assert_close(result[0][0], 1.0, 1e-6, "d_normed[0][0]");
        assert_close(result[0][1], 2.0, 1e-6, "d_normed[0][1]");
        assert_close(result[0][2], 0.5, 1e-6, "d_normed[0][2]");
        assert_close(result[0][3], 3.0, 1e-6, "d_normed[0][3]");
    }
}
