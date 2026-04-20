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

use rayon::prelude::*;

/// Content projection dimension for learnable attention routing.
pub const CONTENT_DIM: usize = 16;

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
    /// Frozen content projection — deterministic-random symmetry-breaker.
    /// Projects input wave to a content vector; dot product between content vectors
    /// adds a content-dependent bias to the harmonic coherence score.
    /// Never serialized or trained — initialized once from RNG seed at model creation.
    /// When empty, attention is pure harmonic coherence (backward-compatible).
    pub content_proj_w: Vec<Vec<f32>>, // [CONTENT_DIM, n_embd]
    pub content_proj_b: Vec<f32>,      // [CONTENT_DIM]
}

/// Weights for full multi-head wave attention.
#[derive(Clone)]
pub struct WaveAttnWeights {
    pub heads: Vec<WaveAttnHeadWeights>,
    /// Combined output projection: [n_embd, n_embd]
    pub out_proj_w: Vec<Vec<f32>>,
    pub out_proj_b: Vec<f32>,
}

/// Forward intermediates for pathway-only backward.
/// Populated only when dims.attention_pathway is true.
#[derive(Clone)]
pub struct WaveAttnCache {
    /// Per-head phase angles per position: [n_head][t]
    pub phases: Vec<Vec<f32>>,
    /// Per-head raw (r, s) for atan2 backward: [n_head][t] of (r, s)
    pub phase_rs: Vec<Vec<(f32, f32)>>,
    /// Per-head value projections per position: [n_head][t][head_dim]
    pub v_all: Vec<Vec<Vec<f32>>>,
    /// Per-head content vectors per position (empty vec if no content_proj): [n_head][t][content_dim]
    pub content_vecs: Vec<Vec<Vec<f32>>>,
    /// Content scale per head: [n_head]
    pub content_scale: Vec<f32>,
    /// Per-head attention weights post-softmax: [n_head][t][t]
    pub att_w: Vec<Vec<Vec<f32>>>,
    /// Merged head outputs before out_proj: [t][n_embd]
    pub out_merged: Vec<Vec<f32>>,
    /// The normed input that attention read from: [t][n_embd]
    pub normed: Vec<Vec<f32>>,
    /// n_bands for dimension info
    pub n_bands: usize,
}

/// Precompute phase angle for a position via learned projection.
/// Projects [n_embd] → [2] (r, s), returns atan2(s, r) as scalar phase.
pub fn project_phase(x: &[f32], proj_w: &[Vec<f32>], proj_b: &[f32]) -> f32 {
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
    return_pathway_cache: bool,
) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>, Option<WaveAttnCache>) {
    let t = x.len();
    let n_embd = n_bands * 2;
    let n_head = weights.heads.len();
    let head_dim = n_embd / n_head;

    // Parallel over attention heads — each head is fully independent.
    // 12 heads on 28 threads: ~4-6x speedup at 24 layers.
    // Per-head result type depends on whether we need the pathway cache
    struct HeadResult {
        head_out: Vec<Vec<f32>>,
        att_w: Vec<Vec<f32>>,
        // Pathway cache intermediates (only populated when return_pathway_cache is true)
        phases: Vec<f32>,
        phase_rs: Vec<(f32, f32)>,
        v_all: Vec<Vec<f32>>,
        content_vecs: Vec<Vec<f32>>,
        content_scale: f32,
    }

    let head_results: Vec<HeadResult> = (0..n_head).into_par_iter().map(|head| {
        let harmonic_n = super::math::softplus(weights.heads[head].harmonic_raw);
        let offset = head * head_dim;

        // Phase 1: Precompute phase angles (and raw r,s for backward when caching)
        let phase_data: Vec<(f32, f32, f32)> = (0..t).map(|pos| {
            let pw = &weights.heads[head].phase_proj_w;
            let pb = &weights.heads[head].phase_proj_b;
            let mut r = pb[0];
            let mut s = pb[1];
            for j in 0..n_embd { r += pw[0][j] * x[pos][j]; s += pw[1][j] * x[pos][j]; }
            (s.atan2(r), r, s)
        }).collect();
        let phases: Vec<f32> = phase_data.iter().map(|&(p, _, _)| p).collect();
        let phase_rs: Vec<(f32, f32)> = phase_data.iter().map(|&(_, r, s)| (r, s)).collect();

        // Phase 1b: Frozen content projection (symmetry-breaking)
        // When content_proj_w is non-empty, project each position to a content vector
        // for content-dependent attention bias. Empty = backward-compatible (no bias).
        let has_content = !weights.heads[head].content_proj_w.is_empty();
        let content_dim = if has_content { weights.heads[head].content_proj_w.len() } else { 0 };
        let content_vecs: Vec<Vec<f32>> = if has_content {
            (0..t).map(|pos| {
                let mut cv = vec![0.0f32; content_dim];
                for d in 0..content_dim {
                    let mut sum = weights.heads[head].content_proj_b[d];
                    for j in 0..n_embd {
                        sum += weights.heads[head].content_proj_w[d][j] * x[pos][j];
                    }
                    cv[d] = sum;
                }
                cv
            }).collect()
        } else {
            vec![]
        };
        let content_scale = if content_dim > 0 { 1.0 / (content_dim as f32).sqrt() } else { 0.0 };

        // Phase 2: Batch value projection
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

        // Phase 3: Phase-hashed sparse attention
        const N_BUCKETS: usize = 8;
        let bucket_width = std::f32::consts::TAU / N_BUCKETS as f32;

        let buckets: Vec<usize> = phases.iter().map(|&p| {
            let normalized = ((p % std::f32::consts::TAU) + std::f32::consts::TAU) % std::f32::consts::TAU;
            ((normalized / bucket_width) as usize).min(N_BUCKETS - 1)
        }).collect();

        let mut bucket_positions: Vec<Vec<usize>> = vec![Vec::new(); N_BUCKETS];
        for (pos, &b) in buckets.iter().enumerate() {
            bucket_positions[b].push(pos);
        }

        // Per-head outputs: head_out[qi][d] and att_w[qi][ki]
        let mut head_out = vec![vec![0.0f32; head_dim]; t];
        let mut att_w = vec![vec![0.0f32; t]; t];

        for qi in 0..t {
            let qi_bucket = buckets[qi];
            let mut scores = vec![f32::NEG_INFINITY; t];

            for db in 0..=2 {
                let target_bucket = if db == 0 {
                    (qi_bucket + N_BUCKETS - 1) % N_BUCKETS
                } else if db == 1 {
                    qi_bucket
                } else {
                    (qi_bucket + 1) % N_BUCKETS
                };

                for &ki in &bucket_positions[target_bucket] {
                    if ki > qi { continue; }
                    let delta = phases[qi] - phases[ki];
                    let mut score = (harmonic_n * delta).cos();
                    // Add content-dependent bias if frozen content projection is active
                    if has_content {
                        let dot: f32 = content_vecs[qi].iter().zip(content_vecs[ki].iter())
                            .map(|(&a, &b)| a * b).sum();
                        score += dot * content_scale;
                    }
                    scores[ki] = score;
                }
            }

            if qi > 0 && scores[qi - 1] == f32::NEG_INFINITY {
                let delta = phases[qi] - phases[qi - 1];
                let mut score = (harmonic_n * delta).cos();
                if has_content {
                    let dot: f32 = content_vecs[qi].iter().zip(content_vecs[qi - 1].iter())
                        .map(|(&a, &b)| a * b).sum();
                    score += dot * content_scale;
                }
                scores[qi - 1] = score;
            }
            if scores[qi] == f32::NEG_INFINITY {
                scores[qi] = 1.0;
            }

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

            att_w[qi] = scores.clone();

            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for ki in 0..=qi {
                    if scores[ki] > 0.0 { sum += scores[ki] * v_all[ki][d]; }
                }
                head_out[qi][d] = sum;
            }
        }

        let cs = if content_dim > 0 { 1.0 / (content_dim as f32).sqrt() } else { 0.0 };
        HeadResult {
            head_out, att_w,
            phases, phase_rs, v_all, content_vecs, content_scale: cs,
        }
    }).collect();

    // Merge head outputs into combined arrays
    let mut att_weights_all: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0f32; t]; t]; n_head];
    let mut out = vec![vec![0.0f32; n_embd]; t];
    for head in 0..n_head {
        let offset = head * head_dim;
        let hr = &head_results[head];
        for qi in 0..t {
            for d in 0..head_dim {
                out[qi][offset + d] = hr.head_out[qi][d];
            }
            att_weights_all[head][qi] = hr.att_w[qi].clone();
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

    // Build pathway cache if requested
    let pathway_cache = if return_pathway_cache {
        Some(WaveAttnCache {
            phases: head_results.iter().map(|hr| hr.phases.clone()).collect(),
            phase_rs: head_results.iter().map(|hr| hr.phase_rs.clone()).collect(),
            v_all: head_results.iter().map(|hr| hr.v_all.clone()).collect(),
            content_vecs: head_results.iter().map(|hr| hr.content_vecs.clone()).collect(),
            content_scale: head_results.iter().map(|hr| hr.content_scale).collect(),
            att_w: head_results.iter().map(|hr| hr.att_w.clone()).collect(),
            out_merged: out.clone(),
            normed: x.to_vec(),
            n_bands,
        })
    } else {
        None
    };

    (result, att_weights_all, pathway_cache)
}

/// Pathway-only backward for wave coherence attention.
///
/// Computes d_normed_from_attention without accumulating attention weight gradients.
/// Used when attention is frozen (default) but gradient pathway correctness is required
/// (flag --attention-pathway). Calls shared primitives from attn_backward.rs.
///
/// Returns d_normed_from_attention shaped [t][n_embd].
/// Per-head attention weight gradients. Matches the layout flatten_params_ex
/// uses when `learnable_attn` is true, so the training loop can drop these
/// straight into the flat gradient vector.
#[derive(Clone)]
pub struct WaveAttnHeadGrads {
    pub phase_proj_w: Vec<Vec<f32>>,
    pub phase_proj_b: Vec<f32>,
    pub v_proj_w: Vec<Vec<f32>>,
    pub v_proj_b: Vec<f32>,
    pub content_proj_w: Vec<Vec<f32>>, // empty when content projection absent
    pub content_proj_b: Vec<f32>,
    pub d_harmonic_raw: f32,
}

/// Full attention gradient bundle — per-head weight grads plus block-level
/// out_proj grads plus `d_normed` (the pathway contribution to the input).
#[derive(Clone)]
pub struct WaveAttnGrads {
    pub heads: Vec<WaveAttnHeadGrads>,
    pub out_proj_w: Vec<Vec<f32>>,
    pub out_proj_b: Vec<f32>,
    pub d_normed: Vec<Vec<f32>>,
}

/// Full attention backward: computes `d_normed` AND all attention weight
/// gradients. Use this path when `dims.learnable_attn` is true so Adam has
/// gradients to update attention with. For the frozen-attention default,
/// `wave_attention_backward_pathway` stays the cheaper option (weight grads
/// are discarded anyway). The math is identical — every step the pathway
/// version runs, this one runs too; the only extra work is keeping the
/// weight-grad tensors the pathway version was already computing and
/// ignoring via `_`.
pub fn wave_attention_backward_full(
    weights: &WaveAttnWeights,
    cache: &WaveAttnCache,
    d_attn_out: &[Vec<f32>],
) -> WaveAttnGrads {
    let n_bands = cache.n_bands;
    let n_embd = n_bands * 2;
    let n_head = weights.heads.len();
    let head_dim = n_embd / n_head;
    let t = d_attn_out.len();

    use super::attn_backward as ab;

    let (d_out, d_op_w, d_op_b) = ab::out_proj_backward(
        d_attn_out, &cache.out_merged, &weights.out_proj_w, n_embd,
    );
    let d_heads = ab::split_heads(&d_out, n_head, head_dim);

    let mut d_x_from_phase_all = Vec::with_capacity(n_head);
    let mut d_x_from_v_all = Vec::with_capacity(n_head);
    let mut d_x_from_content_all = Vec::with_capacity(n_head);
    let mut head_grads = Vec::with_capacity(n_head);

    for h in 0..n_head {
        let raw = weights.heads[h].harmonic_raw;
        let harmonic_n = super::math::softplus(raw);
        let sigmoid_raw = 1.0 / (1.0 + (-raw).exp()); // d softplus(raw) / d raw
        let offset = h * head_dim;

        let (d_att_w, d_v_all) = ab::value_aggregation_backward(
            &d_heads[h], &cache.att_w[h], &cache.v_all[h],
        );
        let d_scores = ab::softmax_backward(&d_att_w, &cache.att_w[h]);

        let cv_opt = if !cache.content_vecs[h].is_empty() {
            Some(cache.content_vecs[h].as_slice())
        } else {
            None
        };
        let (d_delta, d_content_opt, d_harmonic_n) = ab::score_backward(
            &d_scores, &cache.phases[h], harmonic_n, &cache.att_w[h], cv_opt, cache.content_scale[h],
        );

        let d_phases = ab::phase_subtraction_backward(&d_delta, &cache.att_w[h]);

        let (d_x_phase, d_pp_w, d_pp_b) = ab::phase_projection_backward(
            &d_phases, &cache.normed,
            &weights.heads[h].phase_proj_w, &weights.heads[h].phase_proj_b, n_embd,
        );
        let (d_x_v, d_vp_w, d_vp_b) = ab::value_projection_backward(
            &d_v_all, &cache.normed, offset, head_dim, n_embd, &weights.heads[h].v_proj_w,
        );
        let (d_x_content, d_cp_w, d_cp_b) = if let Some(ref dcv) = d_content_opt {
            let (d_x_c, d_cw, d_cb) = ab::content_projection_backward(
                dcv, &cache.normed, &weights.heads[h].content_proj_w, n_embd,
            );
            (d_x_c, d_cw, d_cb)
        } else {
            (vec![vec![0.0f32; n_embd]; t], vec![], vec![])
        };

        d_x_from_phase_all.push(d_x_phase);
        d_x_from_v_all.push(d_x_v);
        d_x_from_content_all.push(d_x_content);

        head_grads.push(WaveAttnHeadGrads {
            phase_proj_w: d_pp_w,
            phase_proj_b: d_pp_b,
            v_proj_w: d_vp_w,
            v_proj_b: d_vp_b,
            content_proj_w: d_cp_w,
            content_proj_b: d_cp_b,
            d_harmonic_raw: d_harmonic_n * sigmoid_raw,
        });
    }

    let d_normed = ab::combine_d_normed(&d_x_from_phase_all, &d_x_from_v_all, &d_x_from_content_all, t, n_embd);

    WaveAttnGrads {
        heads: head_grads,
        out_proj_w: d_op_w,
        out_proj_b: d_op_b,
        d_normed,
    }
}

pub fn wave_attention_backward_pathway(
    weights: &WaveAttnWeights,
    cache: &WaveAttnCache,
    d_attn_out: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let n_bands = cache.n_bands;
    let n_embd = n_bands * 2;
    let n_head = weights.heads.len();
    let head_dim = n_embd / n_head;
    let t = d_attn_out.len();

    use super::attn_backward as ab;

    // Step 1: out_proj backward
    let (d_out, _d_op_w, _d_op_b) = ab::out_proj_backward(
        d_attn_out, &cache.out_merged, &weights.out_proj_w, n_embd,
    );

    // Step 2: split into per-head
    let d_heads = ab::split_heads(&d_out, n_head, head_dim);

    // Per-head backward, accumulate d_normed
    let mut d_x_from_phase_all = Vec::with_capacity(n_head);
    let mut d_x_from_v_all = Vec::with_capacity(n_head);
    let mut d_x_from_content_all = Vec::with_capacity(n_head);

    for h in 0..n_head {
        let harmonic_n = super::math::softplus(weights.heads[h].harmonic_raw);
        let offset = h * head_dim;

        // Step 3: value aggregation backward
        let (d_att_w, d_v_all) = ab::value_aggregation_backward(
            &d_heads[h], &cache.att_w[h], &cache.v_all[h],
        );

        // Step 4: softmax backward
        let d_scores = ab::softmax_backward(&d_att_w, &cache.att_w[h]);

        // Step 5: score backward
        let cv_opt = if !cache.content_vecs[h].is_empty() { Some(cache.content_vecs[h].as_slice()) } else { None };
        let (d_delta, d_content_opt, _d_harmonic_n) = ab::score_backward(
            &d_scores, &cache.phases[h], harmonic_n, &cache.att_w[h], cv_opt, cache.content_scale[h],
        );

        // Step 6: phase subtraction backward
        let d_phases = ab::phase_subtraction_backward(&d_delta, &cache.att_w[h]);

        // Step 7: phase projection backward (d_normed contribution from phases)
        let (d_x_phase, _d_pp_w, _d_pp_b) = ab::phase_projection_backward(
            &d_phases, &cache.normed,
            &weights.heads[h].phase_proj_w, &weights.heads[h].phase_proj_b, n_embd,
        );

        // Step 8: value projection backward (d_normed contribution from values)
        let (d_x_v, _d_vp_w, _d_vp_b) = ab::value_projection_backward(
            &d_v_all,
            &cache.normed,
            offset, head_dim, n_embd, &weights.heads[h].v_proj_w,
        );

        // Step 9: content projection backward (when active)
        let d_x_content = if let Some(ref dcv) = d_content_opt {
            let (d_x_c, _d_cp_w, _d_cp_b) = ab::content_projection_backward(
                dcv,
                &cache.normed,
                &weights.heads[h].content_proj_w, n_embd,
            );
            d_x_c
        } else {
            vec![vec![0.0f32; n_embd]; t]
        };

        d_x_from_phase_all.push(d_x_phase);
        d_x_from_v_all.push(d_x_v);
        d_x_from_content_all.push(d_x_content);
    }

    // Step 10: combine
    ab::combine_d_normed(&d_x_from_phase_all, &d_x_from_v_all, &d_x_from_content_all, t, n_embd)
}
