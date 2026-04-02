//! KV-cache for fast autoregressive generation.
//!
//! Harmonic coherence attention KV-cache is simpler than standard transformers:
//! phases are scalars (not vectors), so the cache is compact.
//!
//! Prefill: process all prompt tokens, cache phases + values per layer per head.
//! Forward one: process one new token using cached state — O(T) per layer, not O(T²).

use crate::common::attn::{WaveAttnWeights, WaveAttnHeadWeights};
use crate::common::model::layer_norm;
use crate::common::wave_model::WavePacketModel;
use crate::common::dims::Dims;

const N_BUCKETS: usize = 8;

/// Per-head cached attention state.
struct HeadCache {
    phases: Vec<f32>,
    values: Vec<Vec<f32>>,
    buckets: Vec<usize>,
    bucket_positions: Vec<Vec<usize>>,
}

/// Per-layer cached state.
struct LayerCache {
    heads: Vec<HeadCache>,
    /// Hidden state after this layer (for residual path on next forward_one)
    hidden: Vec<Vec<f32>>,
}

/// Full KV-cache across all layers.
pub struct KvCache {
    layers: Vec<LayerCache>,
    n_cached: usize,
}

fn project_phase(x: &[f32], proj_w: &[Vec<f32>], proj_b: &[f32]) -> f32 {
    let mut r = proj_b[0];
    let mut s = proj_b[1];
    for j in 0..x.len() {
        r += proj_w[0][j] * x[j];
        s += proj_w[1][j] * x[j];
    }
    s.atan2(r)
}

fn phase_to_bucket(phase: f32) -> usize {
    let bucket_width = std::f32::consts::TAU / N_BUCKETS as f32;
    let normalized = ((phase % std::f32::consts::TAU) + std::f32::consts::TAU) % std::f32::consts::TAU;
    ((normalized / bucket_width) as usize).min(N_BUCKETS - 1)
}

fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }

impl KvCache {
    pub fn new(n_layers: usize, n_head: usize) -> Self {
        let layers = (0..n_layers).map(|_| {
            LayerCache {
                heads: (0..n_head).map(|_| HeadCache {
                    phases: Vec::new(),
                    values: Vec::new(),
                    buckets: Vec::new(),
                    bucket_positions: vec![Vec::new(); N_BUCKETS],
                }).collect(),
                hidden: Vec::new(),
            }
        }).collect();
        KvCache { layers, n_cached: 0 }
    }
}

/// Prefill: run full forward on prompt tokens, populate KV-cache.
/// Returns logits for the last token.
pub fn prefill(
    model: &WavePacketModel,
    tokens: &[usize],
    dims: Dims,
    cache: &mut KvCache,
    stencil: &crate::fft_ode::StencilFft,
) -> Vec<f32> {
    let t = tokens.len();
    let n_embd = dims.n_embd;
    let n_head = dims.n_head;

    // Embed
    let mut hidden: Vec<Vec<f32>> = tokens.iter().map(|&tok| {
        let mut h = model.wte[tok].clone();
        if tok < model.wpe.len() {
            for j in 0..n_embd { h[j] += model.wpe[tok][j]; }
        }
        h
    }).collect();

    for (layer_idx, block) in model.blocks.iter().enumerate() {
        let normed: Vec<Vec<f32>> = hidden.iter()
            .map(|h| layer_norm(h, &block.ln.weight, &block.ln.bias))
            .collect();

        let head_dim = n_embd / n_head;

        // Attention with caching
        let mut attn_out = vec![vec![0.0f32; n_embd]; t];
        for head in 0..n_head {
            let harmonic_n = softplus(model.blocks[layer_idx].attn.heads[head].harmonic_raw);
            let hw = &model.blocks[layer_idx].attn.heads[head];
            let offset = head * head_dim;

            // Compute and cache phases + values
            for pos in 0..t {
                let phase = project_phase(&normed[pos], &hw.phase_proj_w, &hw.phase_proj_b);
                let mut v = vec![0.0f32; head_dim];
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..head_dim { sum += hw.v_proj_w[d][j] * normed[pos][offset + j]; }
                    v[d] = sum + hw.v_proj_b[d];
                }
                let bucket = phase_to_bucket(phase);
                cache.layers[layer_idx].heads[head].phases.push(phase);
                cache.layers[layer_idx].heads[head].values.push(v);
                cache.layers[layer_idx].heads[head].buckets.push(bucket);
                cache.layers[layer_idx].heads[head].bucket_positions[bucket].push(pos);
            }

            // Score and combine (same as attn.rs but reading from cache)
            let hc = &cache.layers[layer_idx].heads[head];
            for qi in 0..t {
                let qi_bucket = hc.buckets[qi];
                let mut scores = vec![f32::NEG_INFINITY; t];

                for db in 0..=2 {
                    let target = if db == 0 { (qi_bucket + N_BUCKETS - 1) % N_BUCKETS }
                                 else if db == 1 { qi_bucket }
                                 else { (qi_bucket + 1) % N_BUCKETS };
                    for &ki in &hc.bucket_positions[target] {
                        if ki > qi { continue; }
                        scores[ki] = (harmonic_n * (hc.phases[qi] - hc.phases[ki])).cos();
                    }
                }
                if qi > 0 && scores[qi - 1] == f32::NEG_INFINITY {
                    scores[qi - 1] = (harmonic_n * (hc.phases[qi] - hc.phases[qi - 1])).cos();
                }
                if scores[qi] == f32::NEG_INFINITY { scores[qi] = 1.0; }

                let max_s = scores[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for ki in 0..=qi {
                    if scores[ki] > f32::NEG_INFINITY {
                        scores[ki] = (scores[ki] - max_s).exp();
                        exp_sum += scores[ki];
                    } else { scores[ki] = 0.0; }
                }
                if exp_sum > 0.0 { for ki in 0..=qi { scores[ki] /= exp_sum; } }

                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for ki in 0..=qi {
                        if scores[ki] > 0.0 { sum += scores[ki] * hc.values[ki][d]; }
                    }
                    attn_out[qi][offset + d] = sum;
                }
            }
        }

        // Out projection
        let attn_proj: Vec<Vec<f32>> = attn_out.iter().map(|o| {
            let mut proj = vec![0.0f32; n_embd];
            for i in 0..n_embd {
                let mut sum = 0.0f32;
                for j in 0..n_embd { sum += block.attn.out_proj_w[i][j] * o[j]; }
                proj[i] = sum + block.attn.out_proj_b[i];
            }
            proj
        }).collect();

        // FFN (normed through ln_ffn)
        let normed_ffn: Vec<Vec<f32>> = hidden.iter()
            .map(|h| layer_norm(h, &block.ln_ffn.weight, &block.ln_ffn.bias))
            .collect();
        let (ffn_out, _) = crate::ffn_backend::ffn_forward_via_backend(
            &block.ffn, &normed_ffn, &crate::backend::CpuBackend,
            Some(stencil), None, None, true, dims.use_corrector, None, None,
        );

        // Residual
        for pos in 0..t {
            for j in 0..n_embd {
                hidden[pos][j] += attn_proj[pos][j] + ffn_out[pos][j];
            }
        }

        cache.layers[layer_idx].hidden = hidden.clone();
    }

    cache.n_cached = t;

    // Final LN + lm_head for last position
    let final_normed = layer_norm(&hidden[t - 1], &model.ln_f.weight, &model.ln_f.bias);
    let mut logits = vec![0.0f32; model.vocab_size];
    for v in 0..model.vocab_size {
        let mut sum = 0.0f32;
        for j in 0..n_embd { sum += model.lm_head[v][j] * final_normed[j]; }
        logits[v] = sum;
    }
    logits
}

/// Forward one token using cached state. O(T) per layer instead of O(T²).
/// Returns logits for the new token.
pub fn forward_one(
    model: &WavePacketModel,
    token: usize,
    pos: usize,
    dims: Dims,
    cache: &mut KvCache,
    stencil: &crate::fft_ode::StencilFft,
) -> Vec<f32> {
    let n_embd = dims.n_embd;
    let n_head = dims.n_head;
    let head_dim = n_embd / n_head;

    // Embed single token
    let mut hidden = model.wte[token].clone();
    if pos < model.wpe.len() {
        for j in 0..n_embd { hidden[j] += model.wpe[pos][j]; }
    }

    for (layer_idx, block) in model.blocks.iter().enumerate() {
        let normed = layer_norm(&hidden, &block.ln.weight, &block.ln.bias);

        // Attention: score new token against all cached positions
        let mut attn_out = vec![0.0f32; n_embd];
        for head in 0..n_head {
            let harmonic_n = softplus(block.attn.heads[head].harmonic_raw);
            let hw = &block.attn.heads[head];
            let offset = head * head_dim;

            // New token's phase and value
            let phase = project_phase(&normed, &hw.phase_proj_w, &hw.phase_proj_b);
            let mut v_new = vec![0.0f32; head_dim];
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for j in 0..head_dim { sum += hw.v_proj_w[d][j] * normed[offset + j]; }
                v_new[d] = sum + hw.v_proj_b[d];
            }
            let bucket = phase_to_bucket(phase);

            // Score against cached positions (bucket-sparse)
            let hc = &cache.layers[layer_idx].heads[head];
            let n_cached = hc.phases.len();
            let mut scores = vec![f32::NEG_INFINITY; n_cached + 1]; // +1 for self

            for db in 0..=2 {
                let target = if db == 0 { (bucket + N_BUCKETS - 1) % N_BUCKETS }
                             else if db == 1 { bucket }
                             else { (bucket + 1) % N_BUCKETS };
                for &ki in &hc.bucket_positions[target] {
                    scores[ki] = (harmonic_n * (phase - hc.phases[ki])).cos();
                }
            }
            // Self-attention
            scores[n_cached] = 1.0;
            // Always include previous position
            if n_cached > 0 && scores[n_cached - 1] == f32::NEG_INFINITY {
                scores[n_cached - 1] = (harmonic_n * (phase - hc.phases[n_cached - 1])).cos();
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                if *s > f32::NEG_INFINITY {
                    *s = (*s - max_s).exp();
                    exp_sum += *s;
                } else { *s = 0.0; }
            }
            if exp_sum > 0.0 { for s in &mut scores { *s /= exp_sum; } }

            // Weighted sum of values
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for ki in 0..n_cached {
                    if scores[ki] > 0.0 { sum += scores[ki] * hc.values[ki][d]; }
                }
                // Self value
                sum += scores[n_cached] * v_new[d];
                attn_out[offset + d] = sum;
            }

            // Append to cache
            cache.layers[layer_idx].heads[head].phases.push(phase);
            cache.layers[layer_idx].heads[head].values.push(v_new);
            cache.layers[layer_idx].heads[head].buckets.push(bucket);
            cache.layers[layer_idx].heads[head].bucket_positions[bucket].push(n_cached);
        }

        // Out projection
        let mut attn_proj = vec![0.0f32; n_embd];
        for i in 0..n_embd {
            let mut sum = 0.0f32;
            for j in 0..n_embd { sum += block.attn.out_proj_w[i][j] * attn_out[j]; }
            attn_proj[i] = sum + block.attn.out_proj_b[i];
        }

        // FFN (single position)
        let normed_ffn = layer_norm(&hidden, &block.ln_ffn.weight, &block.ln_ffn.bias);
        let (ffn_out, _) = crate::ffn_backend::ffn_forward_via_backend(
            &block.ffn, &[normed_ffn], &crate::backend::CpuBackend,
            Some(stencil), None, None, true, dims.use_corrector, None, None,
        );

        // Residual
        for j in 0..n_embd {
            hidden[j] += attn_proj[j] + ffn_out[0][j];
        }
    }

    cache.n_cached += 1;

    // Final LN + lm_head
    let final_normed = layer_norm(&hidden, &model.ln_f.weight, &model.ln_f.bias);
    let mut logits = vec![0.0f32; model.vocab_size];
    for v in 0..model.vocab_size {
        let mut sum = 0.0f32;
        for j in 0..n_embd { sum += model.lm_head[v][j] * final_normed[j]; }
        logits[v] = sum;
    }
    logits
}
