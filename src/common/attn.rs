//! Harmonic coherence attention — replaces dot-product with wave coherence.
//!
//! Instead of score(q, k) = q · k / sqrt(d), we use:
//!   score(q, k) = sum_n |cos(n * (θ_q - θ_k))| * amplitude_weight
//!
//! Each attention head learns WHICH harmonic to attend to (harmonic number n).
//! This is the same cos(n * Δθ) function from the core framework (Test 9).
//!
//! Key difference from the failed Harmonic Attention Q/K experiment:
//! that added coherence ON TOP of dot product. This REPLACES it entirely.

use std::f32::consts::PI;

/// Weights for harmonic coherence attention (one head).
#[derive(Clone)]
pub struct WaveAttnHeadWeights {
    /// Learnable harmonic number per head (continuous, via softplus → positive).
    /// Controls which relationship type this head attends to.
    pub harmonic_raw: f32,
    /// Phase projection: [2, n_embd] — projects hidden state to single (r, s) phase angle.
    /// This is the key scaling fix: O(T × n_embd) precompute instead of O(T² × n_bands).
    pub phase_proj_w: Vec<Vec<f32>>,
    pub phase_proj_b: Vec<f32>,
    /// Output projection weights for this head: [head_dim, head_dim]
    pub v_proj_w: Vec<Vec<f32>>,
    pub v_proj_b: Vec<f32>,
}

/// Weights for full multi-head wave attention.
#[derive(Clone)]
pub struct WaveAttnWeights {
    pub heads: Vec<WaveAttnHeadWeights>,
    /// Combined output projection: [n_embd, n_embd]
    pub out_proj_w: Vec<Vec<f32>>,
    pub out_proj_b: Vec<f32>,
}

/// Precompute phase angle for a position via learned projection.
/// Projects [n_embd] → [2] (r, s), returns atan2(s, r) as scalar phase.
fn project_phase(x: &[f32], proj_w: &[Vec<f32>], proj_b: &[f32]) -> f32 {
    let n_embd = x.len();
    let mut r = proj_b[0];
    let mut s = proj_b[1];
    for j in 0..n_embd {
        r += proj_w[0][j] * x[j];
        s += proj_w[1][j] * x[j];
    }
    s.atan2(r)
}

/// Forward pass for wave coherence attention (scaled version).
///
/// Phase projection: O(T × n_embd) precompute per head.
/// Attention scoring: O(T² × n_heads) — same scaling as dot product.
/// Total: O(T × n_embd × n_heads + T² × n_heads) — linear in n_embd, not quadratic.
pub fn wave_attention_forward(
    weights: &WaveAttnWeights,
    x: &[Vec<f32>],
    n_bands: usize,
    backend: Option<&(dyn crate::backend::ComputeBackend + Send + Sync)>,
) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
    let t = x.len();
    let n_embd = n_bands * 2;
    let n_head = weights.heads.len();
    let head_dim = n_embd / n_head;

    fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }

    let mut att_weights_all: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0f32; t]; t]; n_head];
    let mut out = vec![vec![0.0f32; n_embd]; t];

    for head in 0..n_head {
        let harmonic_n = softplus(weights.heads[head].harmonic_raw);
        let offset = head * head_dim;

        // Phase 1: Precompute phase angles — CPU (scalar, fast)
        let phases: Vec<f32> = (0..t).map(|pos| {
            project_phase(&x[pos], &weights.heads[head].phase_proj_w, &weights.heads[head].phase_proj_b)
        }).collect();

        // Phase 2: Batch value projection — CPU (small matrices, dispatch overhead > compute)
        let v_all: Vec<Vec<f32>> = (0..t).map(|pos| {
            let mut v = vec![0.0f32; head_dim];
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for j in 0..head_dim {
                    sum += weights.heads[head].v_proj_w[d][j] * x[pos][offset + j];
                }
                v[d] = sum + weights.heads[head].v_proj_b[d];
            }
            v
        }).collect();

        // Phase 3: Phase-hashed sparse attention — O(T × T/B) instead of O(T²)
        // Hash phase angles into B buckets. Only attend within same + adjacent buckets.
        // Coherence between distant phase buckets is near-zero by definition.
        const N_BUCKETS: usize = 8;
        let bucket_width = std::f32::consts::TAU / N_BUCKETS as f32;

        // Assign each position to a phase bucket
        let buckets: Vec<usize> = phases.iter().map(|&p| {
            // Normalize phase to [0, 2π) then bucket
            let normalized = ((p % std::f32::consts::TAU) + std::f32::consts::TAU) % std::f32::consts::TAU;
            ((normalized / bucket_width) as usize).min(N_BUCKETS - 1)
        }).collect();

        // Build per-bucket position lists for fast lookup
        let mut bucket_positions: Vec<Vec<usize>> = vec![Vec::new(); N_BUCKETS];
        for (pos, &b) in buckets.iter().enumerate() {
            bucket_positions[b].push(pos);
        }

        for qi in 0..t {
            let qi_bucket = buckets[qi];
            let mut scores = vec![f32::NEG_INFINITY; t];

            // Score only positions in same bucket + adjacent buckets (causal)
            for db in 0..=2 {
                // Buckets: qi_bucket-1, qi_bucket, qi_bucket+1 (wrapping)
                let target_bucket = if db == 0 {
                    (qi_bucket + N_BUCKETS - 1) % N_BUCKETS
                } else if db == 1 {
                    qi_bucket
                } else {
                    (qi_bucket + 1) % N_BUCKETS
                };

                for &ki in &bucket_positions[target_bucket] {
                    if ki > qi { continue; } // causal mask
                    let delta = phases[qi] - phases[ki];
                    scores[ki] = (harmonic_n * delta).cos();
                }
            }

            // Always attend to self and immediate neighbours (locality bias)
            if qi > 0 && scores[qi - 1] == f32::NEG_INFINITY {
                let delta = phases[qi] - phases[qi - 1];
                scores[qi - 1] = (harmonic_n * delta).cos();
            }
            if scores[qi] == f32::NEG_INFINITY {
                scores[qi] = 1.0; // self-attention always on
            }

            // Softmax over scored positions only
            let max_s = scores[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for ki in 0..=qi {
                if scores[ki] > f32::NEG_INFINITY {
                    scores[ki] = (scores[ki] - max_s).exp();
                    exp_sum += scores[ki];
                } else {
                    scores[ki] = 0.0;
                }
            }
            if exp_sum > 0.0 {
                for ki in 0..=qi { scores[ki] /= exp_sum; }
            }

            att_weights_all[head][qi] = scores.clone();

            // Weighted sum — only non-zero scores contribute
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for ki in 0..=qi {
                    if scores[ki] > 0.0 { sum += scores[ki] * v_all[ki][d]; }
                }
                out[qi][offset + d] = sum;
            }
        }
    }

    // Output projection — batched through GPU when available
    let result = if let Some(be) = backend {
        be.linear_batch(&weights.out_proj_w, &weights.out_proj_b, &out)
    } else {
        out.iter().map(|o| {
            let mut projected = vec![0.0f32; n_embd];
            for i in 0..n_embd {
                let mut sum = 0.0f32;
                for j in 0..n_embd { sum += weights.out_proj_w[i][j] * o[j]; }
                projected[i] = sum + weights.out_proj_b[i];
            }
            projected
        }).collect()
    };

    (result, att_weights_all)
}
