//! Wave Packet Engine — proof of concept
//!
//! New architecture: parallel attention + FFN with harmonic coherence scoring.
//! Tests whether wave packet mechanics can serve as the core computation primitive.

// ─── Module tree ─────────────────────────────────────────────
// Physical layout: src/common/, src/cpu/, src/wgpu_tier/, src/candle_tier/
// Re-exports below keep old crate:: paths working (shim layer).

#[allow(dead_code)]
mod common;
#[allow(dead_code)]
mod cpu;
#[allow(dead_code)]
mod wgpu_tier;
#[allow(dead_code)]
mod candle_tier;

// Re-export shim — existing code uses crate::model, crate::backend, etc.
// These map to the new physical locations without changing any imports.
#[allow(unused_imports)]
pub use common::model;
#[allow(unused_imports)]
pub use common::embed as wave_embed;
#[allow(unused_imports)]
pub use common::attn as wave_attn;
#[allow(unused_imports)]
pub use common::block as wave_block;
#[allow(unused_imports)]
pub use common::ffn as ffn_backend;
#[allow(unused_imports)]
pub use common::checkpoint as wave_checkpoint;
#[allow(unused_imports)]
pub use common::rng;
#[allow(unused_imports)]
pub use common::bpe;
#[allow(unused_imports)]
pub use common::token_cache;
#[allow(unused_imports)]
pub use common::monitor;
#[allow(unused_imports)]
pub use common::data;
#[allow(unused_imports)]
pub use common::fft_ode;

#[allow(unused_imports)]
pub use cpu::train;
#[allow(unused_imports)]
pub use cpu::backward;

#[allow(unused_imports)]
pub use wgpu_tier::backend;
#[allow(unused_imports)]
pub use wgpu_tier::device as gpu;
#[allow(unused_imports)]
pub use wgpu_tier::gpu_backend;
#[allow(unused_imports)]
pub use wgpu_tier::buffers as gpu_buffers;
#[allow(unused_imports)]
pub use wgpu_tier::dispatch as gpu_dispatch;
#[allow(unused_imports)]
pub use wgpu_tier::ops_forward as gpu_ops_forward;
#[allow(unused_imports)]
pub use wgpu_tier::ops_backward as gpu_ops_backward;
#[allow(unused_imports)]
pub use wgpu_tier::pipelines as gpu_pipelines;
#[allow(unused_imports)]
pub use wgpu_tier::resident as gpu_resident;
#[allow(unused_imports)]
pub use wgpu_tier::validate as gpu_validate;
#[allow(unused_imports)]
pub use wgpu_tier::ffn_gpu;
#[allow(unused_imports)]
pub use wgpu_tier::ffn_full_gpu;

#[allow(unused_imports)]
pub use candle_tier::engine as candle_engine;
#[allow(unused_imports)]
pub use candle_tier::ode as gpu_ode;
#[allow(unused_imports)]
pub use candle_tier::block_diag as block_diagonal;

use wave_embed::*;
use wave_attn::*;
use wave_block::*;
use rng::Rng;
use rayon::prelude::*;

static PROFILE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// ─── Config ──────────────────────────────────────────────────────

// Compile-time defaults (used as fallbacks when CLI flags not provided)
const N_BANDS: usize = 384;
const N_EMBD: usize = N_BANDS * 2;
const N_HEAD: usize = 12;
const N_LAYERS: usize = 24;
const MAESTRO_DIM: usize = 16;
const BLOCK_SIZE: usize = 256;
const RK4_STEPS: usize = 16;

/// Runtime model dimensions — replaces compile-time constants.
/// Passed through init_model, forward, backward, and analyze.
#[derive(Clone, Copy)]
pub struct Dims {
    pub n_bands: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub maestro_dim: usize,
    pub block_size: usize,
    pub rk4_steps: usize,
}

impl Dims {
    pub fn from_cli(n_bands: usize, n_head: usize, maestro_dim: usize, block_size: usize, rk4_steps: usize) -> Self {
        Self { n_bands, n_embd: n_bands * 2, n_head, maestro_dim, block_size, rk4_steps }
    }
    pub fn defaults() -> Self {
        Self::from_cli(N_BANDS, N_HEAD, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
    }
}

// ─── Model ──────────────────────────────────────────────────────

struct WavePacketModel {
    wte: Vec<Vec<f32>>,
    wpe: Vec<Vec<f32>>,
    blocks: Vec<WaveBlockWeights>,
    ln_f: LayerNormWeights,
    lm_head: Vec<Vec<f32>>,
    vocab_size: usize,
}

fn init_linear(rng: &mut Rng, out_dim: usize, in_dim: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let limit = 1.0 / (in_dim as f32).sqrt();
    let w: Vec<Vec<f32>> = (0..out_dim)
        .map(|_| (0..in_dim).map(|_| rng.uniform(limit)).collect())
        .collect();
    let b = vec![0.0f32; out_dim];
    (w, b)
}

fn init_model(vocab_size: usize, seed: u64, n_layers: usize, out_proj_groups: usize, d: Dims) -> WavePacketModel {
    let mut rng = Rng::new(seed);

    let wte = build_harmonic_table(vocab_size, d.n_bands);
    let wpe = build_positional_table(d.block_size, d.n_bands);

    let mut blocks = Vec::new();
    for _ in 0..n_layers {
        let ln = LayerNormWeights { weight: vec![1.0f32; d.n_embd], bias: vec![0.0f32; d.n_embd] };

        let head_dim = d.n_embd / d.n_head;
        let heads: Vec<WaveAttnHeadWeights> = (0..d.n_head).map(|h| {
            let (phase_w, phase_b) = init_linear(&mut rng, 2, d.n_embd);
            let (v_w, v_b) = init_linear(&mut rng, head_dim, head_dim);
            WaveAttnHeadWeights {
                harmonic_raw: ((h + 1) as f32 * 0.5f32).ln(),
                phase_proj_w: phase_w,
                phase_proj_b: phase_b,
                v_proj_w: v_w,
                v_proj_b: v_b,
            }
        }).collect();
        let (out_w, out_b) = init_linear(&mut rng, d.n_embd, d.n_embd);
        let attn = WaveAttnWeights { heads, out_proj_w: out_w, out_proj_b: out_b };

        let gamma_raw_val = ((0.1f32).exp() - 1.0).ln();
        let kerr = KerrWeights {
            gamma_raw: vec![gamma_raw_val; d.n_bands],
            omega: (0..d.n_bands).map(|k| (k + 1) as f32 / d.n_bands as f32).collect(),
            // ODE coupling: scales with band count. Proven values:
            //   84 bands (168-dim): 0.01 (2K BPE)
            //   384 bands (768-dim): 0.1 (50K BPE)
            alpha: if d.n_bands <= 128 { 0.01 } else { 0.1 },
            beta: if d.n_bands <= 128 { 0.01 } else { 0.1 },
            rk4_n_steps: d.rk4_steps,
        };
        let (sq_w, sq_b) = init_linear(&mut rng, d.maestro_dim, d.n_embd);
        let (pr_w, pr_b) = init_linear(&mut rng, d.n_embd, d.maestro_dim);
        let maestro_in = MaestroWeights {
            squeeze: LinearWeights { w: sq_w, b: sq_b },
            process_1: LinearWeights { w: pr_w, b: pr_b },
        };
        let (sq_w2, sq_b2) = init_linear(&mut rng, d.maestro_dim, d.n_embd);
        let (pr_w2, pr_b2) = init_linear(&mut rng, d.n_embd, d.maestro_dim);
        let maestro_out = MaestroWeights {
            squeeze: LinearWeights { w: sq_w2, b: sq_b2 },
            process_1: LinearWeights { w: pr_w2, b: pr_b2 },
        };
        let out_proj = if out_proj_groups <= 1 {
            let (op_w, op_b) = init_linear(&mut rng, d.n_embd, d.n_embd);
            OutProjWeights::dense(op_w, op_b)
        } else {
            let group_size = d.n_embd / out_proj_groups;
            let groups: Vec<LinearWeights> = (0..out_proj_groups).map(|_| {
                let (w, b) = init_linear(&mut rng, group_size, group_size);
                LinearWeights { w, b }
            }).collect();
            OutProjWeights::BlockDiagonal(BlockDiagonalWeights {
                groups, n_groups: out_proj_groups, group_size,
            })
        };
        let ffn = KerrDualMaestroWeights { kerr, maestro_in, maestro_out, out_proj };

        let ln_ffn = LayerNormWeights { weight: vec![1.0f32; d.n_embd], bias: vec![0.0f32; d.n_embd] };
        blocks.push(WaveBlockWeights { ln, ln_ffn, attn, ffn });
    }

    let ln_f = LayerNormWeights { weight: vec![1.0f32; d.n_embd], bias: vec![0.0f32; d.n_embd] };
    let limit = 1.0 / (d.n_embd as f32).sqrt();
    let lm_head: Vec<Vec<f32>> = (0..vocab_size)
        .map(|_| (0..d.n_embd).map(|_| rng.uniform(limit)).collect())
        .collect();

    WavePacketModel { wte, wpe, blocks, ln_f, lm_head, vocab_size }
}

// ─── Forward with cache ─────────────────────────────────────────

struct BlockCache {
    input: Vec<Vec<f32>>,
    normed: Vec<Vec<f32>>,
    normed_ffn: Vec<Vec<f32>>,
    attn_out: Vec<Vec<f32>>,
    ffn_out: Vec<Vec<f32>>,
    att_weights: Vec<Vec<Vec<f32>>>,
    // FFN intermediates — two paths:
    // 1. Backend cache (new: all ops through ComputeBackend, self-consistent)
    ffn_backend_cache: Option<ffn_backend::FfnCache>,
    // 2. Legacy cache (old: hand-wired, for fallback)
    ffn_mae_in_sq: Vec<Vec<f32>>,
    ffn_mae_in_act: Vec<Vec<f32>>,
    ffn_precond: Vec<Vec<f32>>,
    ffn_kerr_out: Vec<Vec<f32>>,
    ffn_mae_out_sq: Vec<Vec<f32>>,
    ffn_mae_out_act: Vec<Vec<f32>>,
    ffn_regulated: Vec<Vec<f32>>,
}

struct ForwardCache {
    block_caches: Vec<BlockCache>,
    pre_ln_f: Vec<Vec<f32>>,
    post_ln_f: Vec<Vec<f32>>,
    logits: Vec<Vec<f32>>,
}

fn forward_with_cache(
    model: &WavePacketModel,
    tokens: &[usize],
    d: Dims,
    gpu: Option<&(dyn backend::ComputeBackend + Send + Sync)>,
    ping_pong: Option<(&ffn_gpu::FfnGpuBuffers, &gpu_pipelines::GpuBackend)>,
    full_gpu: Option<(&ffn_full_gpu::FfnFullBuffers, &gpu_pipelines::GpuBackend)>,
    stencil: Option<&fft_ode::StencilFft>,
    gpu_kernel: Option<(&fft_ode::GpuKernelFft, &gpu_pipelines::GpuBackend)>,
) -> ForwardCache {
    let profile = PROFILE.load(std::sync::atomic::Ordering::Relaxed);
    let t = tokens.len();
    let _t0 = std::time::Instant::now();
    let mut hidden = embed_tokens(tokens, &model.wte, &model.wpe, d.n_embd);
    let mut block_caches = Vec::new();
    let mut _attn_total = std::time::Duration::ZERO;
    let mut _ffn_total = std::time::Duration::ZERO;
    let mut _ln_total = std::time::Duration::ZERO;

    for block in &model.blocks {
        let _tln = std::time::Instant::now();
        let normed: Vec<Vec<f32>> = hidden.iter()
            .map(|h| layer_norm(h, &block.ln.weight, &block.ln.bias))
            .collect();
        _ln_total += _tln.elapsed();

        // FFN + Attention: parallel dispatch through ComputeBackend
        let _tpar = std::time::Instant::now();

        // Select backend: GPU if available, otherwise CPU
        let be: &dyn backend::ComputeBackend = match gpu {
            Some(g) => g,
            None => &backend::CpuBackend,
        };

        // FFN forward via backend (kerr-engine pattern: all ops through same device)
        let _tf = std::time::Instant::now();
        let (ffn_out, ffn_be_cache) = ffn_backend::ffn_forward_via_backend(&block.ffn, &normed, be, stencil, ping_pong, gpu_kernel);
        let ffn_dur = _tf.elapsed();

        // Attention (CPU — frozen, harmonic coherence scoring)
        let _ta = std::time::Instant::now();
        let (attn_out, att_weights) = wave_attention_forward(&block.attn, &normed, d.n_bands, gpu);
        let attn_dur = _ta.elapsed();
        _attn_total += attn_dur;
        _ffn_total += ffn_dur;

        let output: Vec<Vec<f32>> = (0..t).map(|i| {
            let mut v = vec![0.0f32; d.n_embd];
            for j in 0..d.n_embd { v[j] = hidden[i][j] + attn_out[i][j] + ffn_out[i][j]; }
            v
        }).collect();

        block_caches.push(BlockCache {
            input: hidden,
            normed: normed.clone(),
            normed_ffn: normed,
            attn_out,
            ffn_out,
            att_weights,
            ffn_backend_cache: Some(ffn_be_cache),
            // Legacy fields empty — backend cache handles everything
            ffn_mae_in_sq: vec![], ffn_mae_in_act: vec![], ffn_precond: vec![],
            ffn_kerr_out: vec![], ffn_mae_out_sq: vec![], ffn_mae_out_act: vec![],
            ffn_regulated: vec![],
        });

        hidden = output;
    }

    let post_ln_f: Vec<Vec<f32>> = hidden.iter()
        .map(|h| layer_norm(h, &model.ln_f.weight, &model.ln_f.bias))
        .collect();

    let logits: Vec<Vec<f32>> = post_ln_f.par_iter().map(|normed| {
        let mut l = vec![0.0f32; model.vocab_size];
        for v in 0..model.vocab_size {
            let mut sum = 0.0f32;
            for j in 0..d.n_embd { sum += model.lm_head[v][j] * normed[j]; }
            l[v] = sum;
        }
        l
    }).collect();

    if profile {
        let total = _t0.elapsed();
        eprintln!("    [profile fwd] LN: {:?}  Attn: {:?}  FFN: {:?}  Total: {:?}",
            _ln_total, _attn_total, _ffn_total, total);
    }

    ForwardCache { block_caches, pre_ln_f: hidden, post_ln_f, logits }
}

// FFN forward — now routes through GPU backend for the full FFN path
fn dual_maestro_forward(
    weights: &KerrDualMaestroWeights,
    x: &[Vec<f32>],
    gpu: Option<&(dyn backend::ComputeBackend + Send + Sync)>,
) -> (Vec<Vec<f32>>, wave_block::FfnForwardCache) {
    wave_block::dual_maestro_forward_cached(weights, x, gpu, None)
}

// ─── Backward pass ──────────────────────────────────────────────
// Frozen attention: only train FFN + LN + lm_head.
// Attention backward is skipped (gradients don't flow through attention weights).
// Gradients DO flow through attention output to the input (for LN backward).

fn cross_entropy_backward(logits: &[f32], target: usize) -> Vec<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_l: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum_exp: f32 = exp_l.iter().sum();
    let mut d = exp_l.iter().map(|&e| e / sum_exp).collect::<Vec<f32>>();
    d[target] -= 1.0;
    d
}

fn layer_norm_backward(d_y: &[f32], x: &[f32], weight: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();

    let x_hat: Vec<f32> = x.iter().map(|&v| (v - mean) * inv_std).collect();
    let d_weight: Vec<f32> = (0..n).map(|i| d_y[i] * x_hat[i]).collect();
    let d_bias: Vec<f32> = d_y.to_vec();

    let d_x_hat: Vec<f32> = (0..n).map(|i| d_y[i] * weight[i]).collect();
    let sum_dx_hat: f32 = d_x_hat.iter().sum();
    let sum_dx_hat_xhat: f32 = d_x_hat.iter().zip(&x_hat).map(|(&a, &b)| a * b).sum();
    let d_x: Vec<f32> = (0..n).map(|i| {
        inv_std / n as f32 * (n as f32 * d_x_hat[i] - sum_dx_hat - x_hat[i] * sum_dx_hat_xhat)
    }).collect();

    (d_x, d_weight, d_bias)
}

struct Gradients {
    // Per-block FFN gradients
    block_ln_w: Vec<Vec<f32>>,
    block_ln_b: Vec<Vec<f32>>,
    block_ln_ffn_w: Vec<Vec<f32>>,
    block_ln_ffn_b: Vec<Vec<f32>>,
    block_ffn_kerr_gamma_raw: Vec<Vec<f32>>,
    block_ffn_kerr_omega: Vec<Vec<f32>>,
    block_ffn_kerr_alpha: Vec<f32>,
    block_ffn_kerr_beta: Vec<f32>,
    block_ffn_mae_in_sq_w: Vec<Vec<Vec<f32>>>,
    block_ffn_mae_in_sq_b: Vec<Vec<f32>>,
    block_ffn_mae_in_pr_w: Vec<Vec<Vec<f32>>>,
    block_ffn_mae_in_pr_b: Vec<Vec<f32>>,
    block_ffn_mae_out_sq_w: Vec<Vec<Vec<f32>>>,
    block_ffn_mae_out_sq_b: Vec<Vec<f32>>,
    block_ffn_mae_out_pr_w: Vec<Vec<Vec<f32>>>,
    block_ffn_mae_out_pr_b: Vec<Vec<f32>>,
    block_ffn_out_proj_w: Vec<Vec<Vec<f32>>>,
    block_ffn_out_proj_b: Vec<Vec<f32>>,
    // Final
    ln_f_w: Vec<f32>,
    ln_f_b: Vec<f32>,
    lm_head: Vec<Vec<f32>>,
}

fn backward(model: &WavePacketModel, cache: &ForwardCache, targets: &[usize], d: Dims, gpu: Option<&(dyn backend::ComputeBackend + Send + Sync)>, ping_pong: Option<(&ffn_gpu::FfnGpuBuffers, &gpu_pipelines::GpuBackend)>, full_gpu: Option<(&ffn_full_gpu::FfnFullBuffers, &gpu_pipelines::GpuBackend)>) -> (f32, Gradients) {
    let t = cache.logits.len();
    let vocab_size = model.vocab_size;

    // Init gradients
    let n_blocks = model.blocks.len();
    let mut grads = Gradients {
        block_ln_w: vec![vec![0.0; d.n_embd]; n_blocks],
        block_ln_b: vec![vec![0.0; d.n_embd]; n_blocks],
        block_ln_ffn_w: vec![vec![0.0; d.n_embd]; n_blocks],
        block_ln_ffn_b: vec![vec![0.0; d.n_embd]; n_blocks],
        block_ffn_kerr_gamma_raw: vec![vec![0.0; d.n_bands]; n_blocks],
        block_ffn_kerr_omega: vec![vec![0.0; d.n_bands]; n_blocks],
        block_ffn_kerr_alpha: vec![0.0; n_blocks],
        block_ffn_kerr_beta: vec![0.0; n_blocks],
        block_ffn_mae_in_sq_w: vec![vec![vec![0.0; d.n_embd]; d.maestro_dim]; n_blocks],
        block_ffn_mae_in_sq_b: vec![vec![0.0; d.maestro_dim]; n_blocks],
        block_ffn_mae_in_pr_w: vec![vec![vec![0.0; d.maestro_dim]; d.n_embd]; n_blocks],
        block_ffn_mae_in_pr_b: vec![vec![0.0; d.n_embd]; n_blocks],
        block_ffn_mae_out_sq_w: vec![vec![vec![0.0; d.n_embd]; d.maestro_dim]; n_blocks],
        block_ffn_mae_out_sq_b: vec![vec![0.0; d.maestro_dim]; n_blocks],
        block_ffn_mae_out_pr_w: vec![vec![vec![0.0; d.maestro_dim]; d.n_embd]; n_blocks],
        block_ffn_mae_out_pr_b: vec![vec![0.0; d.n_embd]; n_blocks],
        block_ffn_out_proj_w: vec![vec![vec![0.0; d.n_embd]; d.n_embd]; n_blocks],
        block_ffn_out_proj_b: vec![vec![0.0; d.n_embd]; n_blocks],
        ln_f_w: vec![0.0; d.n_embd],
        ln_f_b: vec![0.0; d.n_embd],
        lm_head: vec![vec![0.0; d.n_embd]; vocab_size],
    };

    // Loss + d_logits
    let mut total_loss = 0.0f32;
    let mut d_logits: Vec<Vec<f32>> = Vec::with_capacity(t);
    for pos in 0..t {
        let max_l = cache.logits[pos].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_l: Vec<f32> = cache.logits[pos].iter().map(|&l| (l - max_l).exp()).collect();
        let sum_exp: f32 = exp_l.iter().sum();
        total_loss += -(exp_l[targets[pos]] / sum_exp).ln();
        let mut dl = cross_entropy_backward(&cache.logits[pos], targets[pos]);
        for v in &mut dl { *v /= t as f32; }
        d_logits.push(dl);
    }
    total_loss /= t as f32;

    // Backward through LM head (no bias)
    // d_hidden: parallelised over positions (each independent, ~768 floats output)
    let n_embd = d.n_embd; // local alias for closures
    let mut d_hidden: Vec<Vec<f32>> = (0..t).into_par_iter().map(|pos| {
        let mut d_h = vec![0.0f32; n_embd];
        for j in 0..n_embd {
            for v in 0..vocab_size {
                d_h[j] += model.lm_head[v][j] * d_logits[pos][v];
            }
        }
        d_h
    }).collect();
    // lm_head weight gradients — sequential (shared accumulator, no temp allocation)
    for pos in 0..t {
        for v in 0..vocab_size {
            for j in 0..d.n_embd {
                grads.lm_head[v][j] += d_logits[pos][v] * cache.post_ln_f[pos][j];
            }
        }
    }

    // Backward through final LN
    let mut d_pre_ln_f: Vec<Vec<f32>> = Vec::with_capacity(t);
    for pos in 0..t {
        let (d_x, d_w, d_b) = layer_norm_backward(&d_hidden[pos], &cache.pre_ln_f[pos], &model.ln_f.weight);
        for i in 0..d.n_embd {
            grads.ln_f_w[i] += d_w[i];
            grads.ln_f_b[i] += d_b[i];
        }
        d_pre_ln_f.push(d_x);
    }
    d_hidden = d_pre_ln_f;

    // Backward through blocks in reverse
    for (block_idx, block) in model.blocks.iter().enumerate().rev() {
        let bc = &cache.block_caches[block_idx];

        // d_output = d_hidden (from above)
        // output = input + attn_out + ffn_out (parallel residual)
        // d_input = d_output, d_attn_out = d_output, d_ffn_out = d_output
        let d_ffn_out: Vec<Vec<f32>> = d_hidden.clone();

        // ─── FFN backward via ComputeBackend (kerr-engine pattern) ───
        let be: &dyn backend::ComputeBackend = match gpu {
            Some(g) => g,
            None => &backend::CpuBackend,
        };
        let d_normed_from_ffn: Vec<Vec<f32>> = if let Some(ref fc) = bc.ffn_backend_cache {
            let (d_input, fg) = ffn_backend::ffn_backward_via_backend(
                &block.ffn, &d_ffn_out, fc, be, ping_pong,
            );
            // Accumulate weight gradients
            for i in 0..fg.d_out_proj_w.len() { for j in 0..fg.d_out_proj_w[i].len() { grads.block_ffn_out_proj_w[block_idx][i][j] += fg.d_out_proj_w[i][j]; } }
            for i in 0..fg.d_out_proj_b.len() { grads.block_ffn_out_proj_b[block_idx][i] += fg.d_out_proj_b[i]; }
            for i in 0..fg.d_mae_out_pr_w.len() { for j in 0..fg.d_mae_out_pr_w[i].len() { grads.block_ffn_mae_out_pr_w[block_idx][i][j] += fg.d_mae_out_pr_w[i][j]; } }
            for i in 0..fg.d_mae_out_pr_b.len() { grads.block_ffn_mae_out_pr_b[block_idx][i] += fg.d_mae_out_pr_b[i]; }
            for i in 0..fg.d_mae_out_sq_w.len() { for j in 0..fg.d_mae_out_sq_w[i].len() { grads.block_ffn_mae_out_sq_w[block_idx][i][j] += fg.d_mae_out_sq_w[i][j]; } }
            for i in 0..fg.d_mae_out_sq_b.len() { grads.block_ffn_mae_out_sq_b[block_idx][i] += fg.d_mae_out_sq_b[i]; }
            for i in 0..fg.d_mae_in_pr_w.len() { for j in 0..fg.d_mae_in_pr_w[i].len() { grads.block_ffn_mae_in_pr_w[block_idx][i][j] += fg.d_mae_in_pr_w[i][j]; } }
            for i in 0..fg.d_mae_in_pr_b.len() { grads.block_ffn_mae_in_pr_b[block_idx][i] += fg.d_mae_in_pr_b[i]; }
            for i in 0..fg.d_mae_in_sq_w.len() { for j in 0..fg.d_mae_in_sq_w[i].len() { grads.block_ffn_mae_in_sq_w[block_idx][i][j] += fg.d_mae_in_sq_w[i][j]; } }
            for i in 0..fg.d_mae_in_sq_b.len() { grads.block_ffn_mae_in_sq_b[block_idx][i] += fg.d_mae_in_sq_b[i]; }
            d_input
        } else {
            ffn_backward(&block.ffn, &bc.normed, &d_ffn_out, &mut grads, block_idx, bc, d, gpu, ping_pong)
        };

        // ─── LN backward (shared LN, FFN gradients only — attention frozen) ───
        let mut d_input = Vec::with_capacity(t);
        for pos in 0..t {
            let (d_x, d_w, d_b) = layer_norm_backward(
                &d_normed_from_ffn[pos], &bc.input[pos], &block.ln.weight,
            );
            for i in 0..d.n_embd {
                grads.block_ln_w[block_idx][i] += d_w[i];
                grads.block_ln_b[block_idx][i] += d_b[i];
            }
            // Residual: d_hidden passes through to input
            let mut d_h = d_x;
            for i in 0..d.n_embd { d_h[i] += d_hidden[pos][i]; }
            d_input.push(d_h);
        }
        d_hidden = d_input;
    }

    (total_loss, grads)
}

fn ffn_backward(
    weights: &KerrDualMaestroWeights,
    normed: &[Vec<f32>],
    d_ffn_out: &[Vec<f32>],
    grads: &mut Gradients,
    block_idx: usize,
    bc: &BlockCache,
    d: Dims,
    gpu: Option<&(dyn backend::ComputeBackend + Send + Sync)>,
    ping_pong: Option<(&ffn_gpu::FfnGpuBuffers, &gpu_pipelines::GpuBackend)>,
) -> Vec<Vec<f32>> {
    let t = normed.len();

    // Use cached forward intermediates — exact same values the forward produced
    let precond_all = &bc.ffn_precond;
    let mae_in_sq_all = &bc.ffn_mae_in_sq;
    let mae_in_act_all = &bc.ffn_mae_in_act;
    let kerr_out_all = &bc.ffn_kerr_out;
    let mae_out_sq_all = &bc.ffn_mae_out_sq;
    let mae_out_act_all = &bc.ffn_mae_out_act;
    let regulated_all = &bc.ffn_regulated;

    // ─── Backward diagnostic: compare cached vs CPU recompute at each stage ───
    if block_idx == 1 && PROFILE.load(std::sync::atomic::Ordering::Relaxed) {
        let pos = 0;
        // CPU recompute from normed[0]
        let cpu_sq = linear_forward(&weights.maestro_in.squeeze.w, &weights.maestro_in.squeeze.b, &normed[pos]);
        let cpu_act: Vec<f32> = cpu_sq.iter().map(|&v| wave_block::gelu(v)).collect();
        let cpu_mae_in = linear_forward(&weights.maestro_in.process_1.w, &weights.maestro_in.process_1.b, &cpu_act);
        let mut cpu_precond = vec![0.0f32; d.n_embd];
        for i in 0..d.n_embd { cpu_precond[i] = normed[pos][i] + cpu_mae_in[i]; }
        let cpu_kerr = kerr_ode_forward_cpu_standalone(&weights.kerr, &cpu_precond);
        let cpu_sq2 = linear_forward(&weights.maestro_out.squeeze.w, &weights.maestro_out.squeeze.b, &cpu_kerr);
        let cpu_act2: Vec<f32> = cpu_sq2.iter().map(|&v| wave_block::gelu(v)).collect();
        let cpu_mae_out = linear_forward(&weights.maestro_out.process_1.w, &weights.maestro_out.process_1.b, &cpu_act2);
        let mut cpu_regulated = vec![0.0f32; d.n_embd];
        for i in 0..d.n_embd { cpu_regulated[i] = cpu_kerr[i] + cpu_mae_out[i]; }
        // CPU out_proj
        let cpu_output = weights.out_proj.forward(&cpu_regulated);

        fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        }
        eprintln!("  [bwd diag block {}] mae_in_sq:   {:.2e}", block_idx, maxdiff(&cpu_sq, &mae_in_sq_all[pos]));
        eprintln!("  [bwd diag block {}] mae_in_act:  {:.2e}", block_idx, maxdiff(&cpu_act, &mae_in_act_all[pos]));
        eprintln!("  [bwd diag block {}] precond:     {:.2e}", block_idx, maxdiff(&cpu_precond, &precond_all[pos]));
        eprintln!("  [bwd diag block {}] kerr_out:    {:.2e}", block_idx, maxdiff(&cpu_kerr, &kerr_out_all[pos]));
        eprintln!("  [bwd diag block {}] mae_out_sq:  {:.2e}", block_idx, maxdiff(&cpu_sq2, &mae_out_sq_all[pos]));
        eprintln!("  [bwd diag block {}] mae_out_act: {:.2e}", block_idx, maxdiff(&cpu_act2, &mae_out_act_all[pos]));
        eprintln!("  [bwd diag block {}] regulated:   {:.2e}", block_idx, maxdiff(&cpu_regulated, &regulated_all[pos]));
    }

    // Backward through out_proj — ping-pong: reads Buffer A (same bits as forward)
    // Backward through out_proj using enum methods (works for Dense or BlockDiagonal)
    let d_regulated: Vec<Vec<f32>> = d_ffn_out.iter().map(|dy| {
        weights.out_proj.backward_dx(dy)
    }).collect();
    // Accumulate d_W and d_b
    let (d_w, d_b) = weights.out_proj.backward_dw_db(d_ffn_out, regulated_all);
    for i in 0..d_w.len() {
        for j in 0..d_w[i].len() { grads.block_ffn_out_proj_w[block_idx][i][j] += d_w[i][j]; }
        grads.block_ffn_out_proj_b[block_idx][i] += d_b[i];
    }

    // Backward through maestro_out (d_regulated → d_kerr_out + maestro_out grads)
    let mut d_kerr_out = Vec::with_capacity(t);
    for pos in 0..t {
        // d_regulated flows to kerr_out (residual) and to maestro_out
        let d_mae_out = &d_regulated[pos]; // same gradient (additive)

        // process backward
        let mut d_act2 = vec![0.0f32; d.maestro_dim];
        for i in 0..d.n_embd {
            for j in 0..d.maestro_dim {
                d_act2[j] += weights.maestro_out.process_1.w[i][j] * d_mae_out[i];
                grads.block_ffn_mae_out_pr_w[block_idx][i][j] += d_mae_out[i] * mae_out_act_all[pos][j];
            }
            grads.block_ffn_mae_out_pr_b[block_idx][i] += d_mae_out[i];
        }

        // GELU backward
        let d_sq2: Vec<f32> = (0..d.maestro_dim).map(|i| {
            let x = mae_out_sq_all[pos][i];
            let c = 0.7978845608_f32;
            let inner = c * (x + 0.044715 * x * x * x);
            let tanh_val = inner.tanh();
            let sech2 = 1.0 - tanh_val * tanh_val;
            let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
            d_act2[i] * (0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner)
        }).collect();

        // squeeze backward
        let mut d_kerr = vec![0.0f32; d.n_embd];
        for i in 0..d.maestro_dim {
            for j in 0..d.n_embd {
                d_kerr[j] += weights.maestro_out.squeeze.w[i][j] * d_sq2[i];
                grads.block_ffn_mae_out_sq_w[block_idx][i][j] += d_sq2[i] * kerr_out_all[pos][j];
            }
            grads.block_ffn_mae_out_sq_b[block_idx][i] += d_sq2[i];
        }

        // d_kerr_out = d_regulated (residual) + d_kerr (from maestro_out squeeze backward)
        for i in 0..d.n_embd { d_kerr[i] += d_regulated[pos][i]; }
        d_kerr_out.push(d_kerr);
    }

    // Skip Kerr-ODE backward for PoC (freeze ODE params, only train maestro + out_proj)
    // d_precond = d_kerr_out (pass through ODE as identity for gradient purposes)
    let d_precond = d_kerr_out;

    // Backward through maestro_in
    let mut d_normed = Vec::with_capacity(t);
    for pos in 0..t {
        let d_mae_in = &d_precond[pos]; // residual from precond = normed + mae_in

        // process backward
        let mut d_act = vec![0.0f32; d.maestro_dim];
        for i in 0..d.n_embd {
            for j in 0..d.maestro_dim {
                d_act[j] += weights.maestro_in.process_1.w[i][j] * d_mae_in[i];
                grads.block_ffn_mae_in_pr_w[block_idx][i][j] += d_mae_in[i] * mae_in_act_all[pos][j];
            }
            grads.block_ffn_mae_in_pr_b[block_idx][i] += d_mae_in[i];
        }

        // GELU backward
        let d_sq: Vec<f32> = (0..d.maestro_dim).map(|i| {
            let x = mae_in_sq_all[pos][i];
            let c = 0.7978845608_f32;
            let inner = c * (x + 0.044715 * x * x * x);
            let tanh_val = inner.tanh();
            let sech2 = 1.0 - tanh_val * tanh_val;
            let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
            d_act[i] * (0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner)
        }).collect();

        // squeeze backward → d_normed
        let mut d_n = vec![0.0f32; d.n_embd];
        for i in 0..d.maestro_dim {
            for j in 0..d.n_embd {
                d_n[j] += weights.maestro_in.squeeze.w[i][j] * d_sq[i];
                grads.block_ffn_mae_in_sq_w[block_idx][i][j] += d_sq[i] * normed[pos][j];
            }
            grads.block_ffn_mae_in_sq_b[block_idx][i] += d_sq[i];
        }

        // d_normed = d_precond (residual from precond = normed + mae_in) + d_squeeze_input
        for i in 0..d.n_embd { d_n[i] += d_precond[pos][i]; }
        d_normed.push(d_n);
    }

    d_normed
}

fn linear_forward(w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
    let out_dim = w.len();
    let in_dim = x.len();
    let mut y = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut sum = 0.0f32;
        for j in 0..in_dim { sum += w[i][j] * x[j]; }
        y[i] = sum + b[i];
    }
    y
}

fn kerr_ode_forward_cpu_standalone(weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
    // Same as wave_block's implementation, duplicated to avoid visibility issues
    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;
    let n_steps = weights.rk4_n_steps;
    let dt = 1.0 / n_steps as f32;
    fn softplus(v: f32) -> f32 { if v > 20.0 { v } else { (1.0 + v.exp()).ln() } }
    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();
    let mut r: Vec<f32> = (0..n_bands).map(|k| x[k * 2]).collect();
    let mut s: Vec<f32> = (0..n_bands).map(|k| x[k * 2 + 1]).collect();
    for _ in 0..n_steps {
        let (r_new, s_new) = rk4_step_standalone(&r, &s, dt, &gamma, &weights.omega, weights.alpha, weights.beta);
        r = r_new; s = s_new;
    }
    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands { out[k * 2] = r[k]; out[k * 2 + 1] = s[k]; }
    out
}

fn rk4_step_standalone(r: &[f32], s: &[f32], dt: f32, gamma: &[f32], omega: &[f32], alpha: f32, beta: f32) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();
    let deriv = |r: &[f32], s: &[f32]| -> (Vec<f32>, Vec<f32>) {
        let mut dr = vec![0.0f32; n]; let mut ds = vec![0.0f32; n];
        for k in 0..n {
            let mag_sq = r[k]*r[k] + s[k]*s[k];
            let mut ns = 0.0f32;
            if k >= 2 { ns += r[k-2]*r[k-2] + s[k-2]*s[k-2]; }
            if k >= 1 { ns += r[k-1]*r[k-1] + s[k-1]*s[k-1]; }
            if k+1 < n { ns += r[k+1]*r[k+1] + s[k+1]*s[k+1]; }
            if k+2 < n { ns += r[k+2]*r[k+2] + s[k+2]*s[k+2]; }
            let phi = omega[k] + alpha * mag_sq + beta * ns;
            dr[k] = -gamma[k] * r[k] - phi * s[k];
            ds[k] = -gamma[k] * s[k] + phi * r[k];
        }
        (dr, ds)
    };
    let (k1r, k1s) = deriv(r, s);
    let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&a,&b)| a+0.5*dt*b).collect();
    let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&a,&b)| a+0.5*dt*b).collect();
    let (k2r, k2s) = deriv(&r2, &s2);
    let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&a,&b)| a+0.5*dt*b).collect();
    let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&a,&b)| a+0.5*dt*b).collect();
    let (k3r, k3s) = deriv(&r3, &s3);
    let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&a,&b)| a+dt*b).collect();
    let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&a,&b)| a+dt*b).collect();
    let (k4r, k4s) = deriv(&r4, &s4);
    let rn: Vec<f32> = (0..n).map(|i| r[i]+dt/6.0*(k1r[i]+2.0*k2r[i]+2.0*k3r[i]+k4r[i])).collect();
    let sn: Vec<f32> = (0..n).map(|i| s[i]+dt/6.0*(k1s[i]+2.0*k2s[i]+2.0*k3s[i]+k4s[i])).collect();
    (rn, sn)
}

// ─── Flatten / Unflatten (trainable params only) ────────────────

fn count_trainable(model: &WavePacketModel) -> usize {
    // Derive dimensions from model structure — no constants needed
    let n_embd = model.ln_f.weight.len();
    let maestro_dim = model.blocks[0].ffn.maestro_in.squeeze.w.len();
    let mut n = 0;
    for block in &model.blocks {
        n += n_embd * 2; // LN
        n += n_embd * 2; // LN_FFN
        n += maestro_dim * n_embd + maestro_dim; // mae_in squeeze
        n += n_embd * maestro_dim + n_embd;       // mae_in process
        n += maestro_dim * n_embd + maestro_dim; // mae_out squeeze
        n += n_embd * maestro_dim + n_embd;       // mae_out process
        n += block.ffn.out_proj.param_count();      // out_proj (dense or block-diagonal)
    }
    n += n_embd * 2; // ln_f
    n += model.vocab_size * n_embd; // lm_head
    n
}

fn flatten_params(model: &WavePacketModel) -> Vec<f32> {
    let mut p = Vec::new();
    for block in &model.blocks {
        p.extend_from_slice(&block.ln.weight);
        p.extend_from_slice(&block.ln.bias);
        p.extend_from_slice(&block.ln_ffn.weight);
        p.extend_from_slice(&block.ln_ffn.bias);
        for row in &block.ffn.maestro_in.squeeze.w { p.extend_from_slice(row); }
        p.extend_from_slice(&block.ffn.maestro_in.squeeze.b);
        for row in &block.ffn.maestro_in.process_1.w { p.extend_from_slice(row); }
        p.extend_from_slice(&block.ffn.maestro_in.process_1.b);
        for row in &block.ffn.maestro_out.squeeze.w { p.extend_from_slice(row); }
        p.extend_from_slice(&block.ffn.maestro_out.squeeze.b);
        for row in &block.ffn.maestro_out.process_1.w { p.extend_from_slice(row); }
        p.extend_from_slice(&block.ffn.maestro_out.process_1.b);
        block.ffn.out_proj.flatten_into(&mut p);
    }
    p.extend_from_slice(&model.ln_f.weight);
    p.extend_from_slice(&model.ln_f.bias);
    for row in &model.lm_head { p.extend_from_slice(row); }
    p
}

fn flatten_grads(grads: &Gradients) -> Vec<f32> {
    let mut g = Vec::new();
    for b in 0..grads.block_ln_w.len() {
        g.extend_from_slice(&grads.block_ln_w[b]);
        g.extend_from_slice(&grads.block_ln_b[b]);
        g.extend_from_slice(&grads.block_ln_ffn_w[b]);
        g.extend_from_slice(&grads.block_ln_ffn_b[b]);
        for row in &grads.block_ffn_mae_in_sq_w[b] { g.extend_from_slice(row); }
        g.extend_from_slice(&grads.block_ffn_mae_in_sq_b[b]);
        for row in &grads.block_ffn_mae_in_pr_w[b] { g.extend_from_slice(row); }
        g.extend_from_slice(&grads.block_ffn_mae_in_pr_b[b]);
        for row in &grads.block_ffn_mae_out_sq_w[b] { g.extend_from_slice(row); }
        g.extend_from_slice(&grads.block_ffn_mae_out_sq_b[b]);
        for row in &grads.block_ffn_mae_out_pr_w[b] { g.extend_from_slice(row); }
        g.extend_from_slice(&grads.block_ffn_mae_out_pr_b[b]);
        for row in &grads.block_ffn_out_proj_w[b] { g.extend_from_slice(row); }
        g.extend_from_slice(&grads.block_ffn_out_proj_b[b]);
    }
    g.extend_from_slice(&grads.ln_f_w);
    g.extend_from_slice(&grads.ln_f_b);
    for row in &grads.lm_head { g.extend_from_slice(row); }
    g
}

fn unflatten_params(model: &mut WavePacketModel, params: &[f32]) {
    let n_embd = model.ln_f.weight.len();
    let maestro_dim = model.blocks[0].ffn.maestro_in.squeeze.w.len();
    let mut idx = 0;
    for block in &mut model.blocks {
        block.ln.weight.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        block.ln.bias.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        block.ln_ffn.weight.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        block.ln_ffn.bias.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        for row in &mut block.ffn.maestro_in.squeeze.w { row.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd; }
        block.ffn.maestro_in.squeeze.b.copy_from_slice(&params[idx..idx+maestro_dim]); idx += maestro_dim;
        for row in &mut block.ffn.maestro_in.process_1.w { row.copy_from_slice(&params[idx..idx+maestro_dim]); idx += maestro_dim; }
        block.ffn.maestro_in.process_1.b.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        for row in &mut block.ffn.maestro_out.squeeze.w { row.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd; }
        block.ffn.maestro_out.squeeze.b.copy_from_slice(&params[idx..idx+maestro_dim]); idx += maestro_dim;
        for row in &mut block.ffn.maestro_out.process_1.w { row.copy_from_slice(&params[idx..idx+maestro_dim]); idx += maestro_dim; }
        block.ffn.maestro_out.process_1.b.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        block.ffn.out_proj.unflatten_from(params, &mut idx);
    }
    model.ln_f.weight.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
    model.ln_f.bias.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
    for row in &mut model.lm_head { row.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd; }
    assert_eq!(idx, params.len());
}

// Adam, CurriculumSchedule, clip_grad_norm moved to train.rs
// Checkpoint save/load moved to wave_checkpoint.rs

// ─── Main ───────────────────────────────────────────────────────

fn print_help() { common::help::print_help(); }

fn main() {
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    // Rayon thread pool — configurable via --threads (default: half available cores)
    fn parse_flag_early<T: std::str::FromStr>(name: &str, default: T) -> T {
        std::env::args().skip_while(|a| a != name).nth(1)
            .and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads: usize = parse_flag_early("--threads", available / 2);
    let n_threads = n_threads.max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok();

    println!("wave-engine v0.1.0  ({n_threads} threads, {available} available)\n");

    // Check for --candle flag first — routes to entirely different training path
    if std::env::args().any(|a| a == "--candle") {
        fn pflag<T: std::str::FromStr>(name: &str, default: T) -> T {
            std::env::args().skip_while(|a| a != name).nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let data_path = std::env::args().nth(1).unwrap_or("data/input.txt".to_string());
        let n_iters: usize = pflag("--iters", 200);
        let n_bands: usize = pflag("--n-bands", N_BANDS);
        let n_head: usize = pflag("--n-head", N_HEAD);
        let n_layers: usize = pflag("--layers", N_LAYERS);
        let maestro_dim: usize = pflag("--maestro-dim", MAESTRO_DIM);
        let rk4_steps: usize = pflag("--rk4-steps", RK4_STEPS);
        let out_proj_groups: usize = pflag("--out-proj-groups", 6);

        let debug_nan = std::env::args().any(|a| a == "--debug-nan");
        match candle_engine::engine::train_candle(
            &data_path, n_iters, n_bands, n_head, n_layers, maestro_dim, rk4_steps, out_proj_groups, debug_nan,
        ) {
            Ok(()) => return,
            Err(e) => { eprintln!("Candle error: {e:?}"); std::process::exit(1); }
        }
    }

    // ─── CLI flag parser ───
    fn parse_flag<T: std::str::FromStr>(name: &str, default: T) -> T {
        std::env::args().skip_while(|a| a != name).nth(1)
            .and_then(|s| s.parse().ok()).unwrap_or(default)
    }

    // ─── Analyze mode: forward pass + wave structure diagnostics ───
    if std::env::args().any(|a| a == "--analyze") {
        use common::wave_analysis as wa;

        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--analyze requires --resume <checkpoint>");
        let n_layers: usize = parse_flag("--layers", N_LAYERS);
        let out_proj_groups: usize = parse_flag("--out-proj-groups", 6);
        let use_bpe = std::env::args().any(|a| a == "--bpe");
        let tokenizer_path: String = parse_flag("--tokenizer", "data/tokenizer.json".to_string());

        println!("Analyze mode: harmonic coherence diagnostics\n");

        // Curated test sentences — covering semantics, grammar, and registers
        let test_text = concat!(
            "The cat sat on the mat. ",
            "The dog sat on the rug. ",
            "A noun is the name of something. ",
            "A verb is a word for action. ",
            "The boy kicked the ball. ",
            "The ball was kicked by the boy. ",
            "To be or not to be that is the question. ",
            "The contract shall be binding upon execution. ",
            "The rate of change increases with temperature. ",
            "How are you doing today my friend. ",
            "Love is patient and kind. ",
            "War brings destruction and death. ",
        );

        // Tokenize — BPE or char-level
        let (token_ids, vocab_size, token_strings) = if use_bpe {
            let bpe = bpe::BpeTokenizer::from_file(&tokenizer_path);
            let ids: Vec<usize> = bpe.encode(test_text);
            let strings: Vec<String> = ids.iter().map(|&id| bpe.decode(&[id])).collect();
            let vs: usize = ids.iter().max().copied().unwrap_or(0) + 1; // conservative vocab bound
            (ids, vs, strings)
        } else {
            let chars: Vec<char> = test_text.chars().collect();
            let mut vc: Vec<char> = chars.iter().cloned()
                .collect::<std::collections::BTreeSet<_>>().into_iter().collect();
            vc.sort();
            let c2i: std::collections::HashMap<char, usize> = vc.iter()
                .enumerate().map(|(i, &c)| (c, i)).collect();
            let ids: Vec<usize> = chars.iter().map(|c| c2i[c]).collect();
            let strings: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
            (ids, vc.len(), strings)
        };

        // Find word spans — a word may be 1 token (char-level) or multiple (BPE sub-tokens).
        // Concatenate adjacent token strings and find where the target word appears.
        // Find the token span for a word. BPE tokens may have leading space (Ġ → " ").
        // "cat" could be [" c", "at"] or [" cat"] depending on vocab size.
        let find_word_span = |word: &str| -> Option<Vec<usize>> {
            let word_lower = word.to_lowercase();
            for start in 0..token_strings.len() {
                let mut concat = String::new();
                for end in start..token_strings.len().min(start + 5) {
                    concat.push_str(&token_strings[end]);
                    // Clean: strip leading BPE space marker, lowercase, trim
                    let clean = concat.replace('\u{0120}', " ").to_lowercase();
                    let clean = clean.trim();
                    // Exact match: the concatenated tokens form exactly this word
                    if clean == word_lower || clean == format!(" {word_lower}") {
                        return Some((start..=end).collect());
                    }
                }
            }
            None
        };

        println!("  Test text: {} tokens", token_ids.len());
        println!("  Tokenizer: {}", if use_bpe { "BPE" } else { "char-level" });

        // Build semantic pairs — works with both single-token and multi-token words
        let mut related_span_pairs: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        let mut related_labels: Vec<(String, String)> = Vec::new();
        let mut random_span_pairs: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();

        let semantic_pairs = [
            ("cat", "dog"),         // same category (animals)
            ("sat", "kicked"),      // same category (verbs)
            ("boy", "ball"),        // subject-object in same sentence
            ("noun", "verb"),       // same category (grammar terms)
            ("love", "war"),        // semantic opposites
            ("mat", "rug"),         // synonyms
        ];
        let random_pair_words = [
            ("cat", "contract"),
            ("verb", "temperature"),
            ("boy", "question"),
            ("dog", "execution"),
            ("sat", "binding"),
            ("mat", "change"),
        ];

        for (w1, w2) in &semantic_pairs {
            if let (Some(span_a), Some(span_b)) = (find_word_span(w1), find_word_span(w2)) {
                println!("    Related: ({w1}@{:?}, {w2}@{:?})", span_a, span_b);
                related_labels.push((w1.to_string(), w2.to_string()));
                related_span_pairs.push((span_a, span_b));
            }
        }
        for (w1, w2) in &random_pair_words {
            if let (Some(span_a), Some(span_b)) = (find_word_span(w1), find_word_span(w2)) {
                random_span_pairs.push((span_a, span_b));
            }
        }

        // Also build single-position pairs for backward compatibility (band census, clustering)
        let mut related_pairs: Vec<(usize, usize)> = related_span_pairs.iter()
            .map(|(a, b)| (a[0], b[0])).collect();
        let mut random_pairs: Vec<(usize, usize)> = random_span_pairs.iter()
            .map(|(a, b)| (a[0], b[0])).collect();

        if related_span_pairs.is_empty() {
            println!("  WARNING: No semantic pairs found in tokens. Using positional fallback.");
            let t = token_ids.len();
            for i in (0..t.min(20)).step_by(2) {
                if i + 1 < t {
                    related_pairs.push((i, i + 1));
                    related_span_pairs.push((vec![i], vec![i + 1]));
                }
            }
            for i in 0..t.min(10).min(t / 2) {
                random_pairs.push((i, (i + t / 2) % t));
                random_span_pairs.push((vec![i], vec![(i + t / 2) % t]));
            }
        }

        println!("  Pairs: {} related, {} random", related_pairs.len(), random_pairs.len());

        // Load model from checkpoint
        let (params, ck_vocab, _ck_iter, _ck_lr, _ck_rng, _adam_t, _adam_m, _adam_v, _ck_groups) =
            wave_checkpoint::load_checkpoint(&resume_path);
        let effective_vocab = vocab_size.max(ck_vocab);
        let n_bands_cli: usize = parse_flag("--n-bands", N_BANDS);
        let n_head_cli: usize = parse_flag("--n-head", N_HEAD);
        let dims = Dims::from_cli(n_bands_cli, n_head_cli, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS);
        let mut model = init_model(effective_vocab, 42, n_layers, out_proj_groups, dims);
        unflatten_params(&mut model, &params);
        println!("  Loaded {} params from {}", params.len(), resume_path);

        // Forward pass
        let stencil = fft_ode::StencilFft::new(dims.n_bands);
        let cache = forward_with_cache(&model, &token_ids, dims, None, None, None, Some(&stencil), None);

        // Extract per-layer phases
        let mut per_layer_phases: Vec<Vec<Vec<f32>>> = Vec::new();
        for bc in &cache.block_caches {
            per_layer_phases.push(wa::extract_all_phases(&bc.input, dims.n_bands));
        }
        per_layer_phases.push(wa::extract_all_phases(&cache.post_ln_f, dims.n_bands));

        // Build token labels for display (use related pair words where available)
        let display_labels: Vec<String> = token_strings.iter()
            .map(|s| s.trim().replace('\n', "\\n").chars().take(12).collect())
            .collect();

        // Run full report (uses span-based discrimination for proper multi-token words)
        {
            let deep = per_layer_phases.last().unwrap();
            let disc = wa::semantic_discrimination_spans(deep, &related_span_pairs, &random_span_pairs, 12);
            let verdict = if disc.ratio > 2.0 { "STRONG SEMANTIC STRUCTURE" }
                else if disc.ratio > 1.5 { "EMERGING STRUCTURE" }
                else { "NOT YET" };
            println!("\n=== Wave Structure Report ===");
            println!("Checkpoint: {resume_path}");
            println!("Layers: {n_layers}, Bands: {}, Tokens: {}", dims.n_bands, token_ids.len());
            println!("\n1. Semantic Discrimination (span-averaged for multi-token words)");
            println!("   Related: {:.3}    Random: {:.3}    Ratio: {:.1}x    {verdict}",
                disc.related_mean, disc.random_mean, disc.ratio);
            // Print matched pairs
            for (label, (span_a, span_b)) in related_labels.iter().zip(&related_span_pairs) {
                let avg_a = wa::average_phases_over_span(deep, span_a);
                let avg_b = wa::average_phases_over_span(deep, span_b);
                let spectrum = wa::harmonic_spectrum(&avg_a, &avg_b, 12);
                let (best_coh, best_n) = wa::best_harmonic_coherence(&avg_a, &avg_b, 12);
                println!("   ({}, {}): peak n={best_n} ({best_coh:.2})  spans {:?} {:?}",
                    label.0, label.1, span_a, span_b);
            }
        }
        // Rest of report (band census, clustering, depth curve use positional pairs)
        wa::print_report(
            &resume_path, n_layers, dims.n_bands, token_ids.len(),
            &per_layer_phases, &related_pairs, &random_pairs, &display_labels,
        );

        // Save JSON report
        std::fs::create_dir_all("analysis").ok();
        let deep = per_layer_phases.last().unwrap();
        let disc = wa::semantic_discrimination_spans(deep, &related_span_pairs, &random_span_pairs, 12);
        let census = wa::band_census(deep, dims.n_bands);
        let clustering = wa::phase_clustering(deep, dims.n_bands);
        let curve = wa::depth_curve(&per_layer_phases, &related_pairs, &random_pairs, 12);

        let report = serde_json::json!({
            "checkpoint": resume_path,
            "n_layers": n_layers,
            "n_bands": dims.n_bands,
            "n_tokens": token_ids.len(),
            "semantic_discrimination": {
                "related_mean": disc.related_mean,
                "random_mean": disc.random_mean,
                "ratio": disc.ratio,
            },
            "band_census": {
                "universal": census.universal,
                "word_specific": census.word_specific,
                "bimodal": census.bimodal,
                "mean_cv": census.mean_circular_variance,
            },
            "phase_clustering": clustering,
            "depth_curve": curve,
            "related_pairs": related_labels.iter().map(|(a, b)| format!("{a}/{b}")).collect::<Vec<_>>(),
        });
        std::fs::write("analysis/wave_report.json",
            serde_json::to_string_pretty(&report).unwrap()).unwrap();
        println!("\nSaved: analysis/wave_report.json");
        return;
    }

    train::run_training(train::TrainConfig {
        data_path: std::env::args().nth(1).unwrap_or("data/input.txt".to_string()),
        n_iters: parse_flag("--iters", 500),
        batch_size: parse_flag("--batch", 4),
        seq_len: parse_flag("--seq", 256),
        n_layers: parse_flag("--layers", N_LAYERS),
        lr: parse_flag("--lr", if N_BANDS > 256 { 1e-4 } else { 3e-4 }),
        use_bpe: std::env::args().any(|a| a == "--bpe"),
        tokenizer_path: parse_flag("--tokenizer", "data/tokenizer.json".to_string()),
        resume_path: std::env::args().skip_while(|a| a != "--resume").nth(1),
        use_curriculum: !std::env::args().any(|a| a == "--no-curriculum"),
        use_gpu: std::env::args().any(|a| a == "--gpu"),
        use_monitor: std::env::args().any(|a| a == "--monitor"),
        out_proj_groups: parse_flag("--out-proj-groups", 6),
        checkpoint_name: parse_flag("--checkpoint-name", "checkpoint.bin".to_string()),
        n_bands: parse_flag("--n-bands", N_BANDS),
        n_head: parse_flag("--n-head", N_HEAD),
    });
}

// Diagnostic: compare GPU vs CPU ODE
#[allow(dead_code)]
fn diagnose_ode_gpu_vs_cpu(gpu_be: &gpu_pipelines::GpuBackend) {
    let gpu: &dyn backend::ComputeBackend = gpu_be;
    use wave_block::*;
    let n_bands = N_BANDS;
    let n_embd = N_EMBD;
    
    // Create test input
    let x: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.01).sin()).collect();
    
    let gamma_raw_val = ((0.1f32).exp() - 1.0).ln();
    let weights = KerrWeights {
        gamma_raw: vec![gamma_raw_val; n_bands],
        omega: (0..n_bands).map(|k| (k + 1) as f32 / n_bands as f32).collect(),
        alpha: 0.1, beta: 0.1, rk4_n_steps: RK4_STEPS,
    };
    
    // CPU ODE
    let cpu_out = kerr_ode_forward_cpu_standalone(&weights, &x);
    
    // GPU ODE (batched, single position)
    let gpu_out = gpu.kerr_ode_batch(&weights, &[x.clone()]);
    let gpu_out = &gpu_out[0];
    
    let max_diff = cpu_out.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_diff: f32 = cpu_out.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs()).sum::<f32>() / n_embd as f32;
    
    eprintln!("ODE diagnostic: max_diff={:.2e}, mean_diff={:.2e}", max_diff, mean_diff);
    eprintln!("  CPU[0..5]: {:?}", &cpu_out[..5]);
    eprintln!("  GPU[0..5]: {:?}", &gpu_out[..5]);

    // Also test linear_batch (out_proj equivalent)
    let w: Vec<Vec<f32>> = (0..n_embd).map(|i| {
        (0..n_embd).map(|j| ((i * n_embd + j) as f32 * 0.001).cos()).collect()
    }).collect();
    let b: Vec<f32> = (0..n_embd).map(|i| i as f32 * 0.01).collect();
    let inputs = vec![x.clone(); 64];
    let gpu_linear = gpu.linear_batch(&w, &b, &inputs);
    // CPU reference
    let cpu_linear: Vec<Vec<f32>> = inputs.iter().map(|xi| {
        let mut y = vec![0.0f32; n_embd];
        for i in 0..n_embd {
            let mut sum = 0.0f32;
            for j in 0..n_embd { sum += w[i][j] * xi[j]; }
            y[i] = sum + b[i];
        }
        y
    }).collect();
    let linear_max = gpu_linear[0].iter().zip(cpu_linear[0].iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let linear_mean = gpu_linear[0].iter().zip(cpu_linear[0].iter())
        .map(|(a, b)| (a - b).abs()).sum::<f32>() / n_embd as f32;
    let max_mag = cpu_linear[0].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rel_err = if max_mag > 0.0 { linear_max / max_mag } else { 0.0 };
    eprintln!("Linear diagnostic: max_diff={:.2e}, mean_diff={:.2e}, max_mag={:.1}, rel_err={:.2e}",
        linear_max, linear_mean, max_mag, rel_err);

    // ─── GPU FFT convolution validation ───
    validate_gpu_fft(gpu_be);
}

fn validate_gpu_fft(gpu: &gpu_pipelines::GpuBackend) {
    use rustfft::num_complex::Complex;

    let n_bands = N_BANDS;
    let n_positions = 4; // test with 4 positions

    // Create test mag_sq data
    let mag_sq: Vec<f32> = (0..n_positions * n_bands)
        .map(|i| ((i as f32 * 0.1).sin()).abs())
        .collect();

    // CPU reference: use StencilFft from fft_ode
    let stencil = fft_ode::StencilFft::new(n_bands);
    let cpu_results: Vec<f32> = (0..n_positions).flat_map(|pos| {
        let slice = &mag_sq[pos * n_bands..(pos + 1) * n_bands];
        stencil.convolve(slice)
    }).collect();

    // Precompute kernel FFT for GPU (same kernel as StencilFft)
    let fft_len = n_bands.next_power_of_two(); // 512
    let mut kernel = vec![Complex::new(0.0f32, 0.0); fft_len];
    kernel[1] = Complex::new(1.0, 0.0);
    kernel[2] = Complex::new(1.0, 0.0);
    if fft_len >= 2 {
        kernel[fft_len - 1] = Complex::new(1.0, 0.0);
        kernel[fft_len - 2] = Complex::new(1.0, 0.0);
    }
    let mut planner = rustfft::FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(fft_len);
    fft_fwd.process(&mut kernel);

    let kernel_re: Vec<f32> = kernel.iter().map(|c| c.re).collect();
    let kernel_im: Vec<f32> = kernel.iter().map(|c| c.im).collect();

    // GPU FFT convolution
    let gpu_results = gpu.gpu_fft_convolve(&mag_sq, &kernel_re, &kernel_im, n_positions, n_bands);

    // Compare
    let max_diff = cpu_results.iter().zip(gpu_results.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_diff = cpu_results.iter().zip(gpu_results.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / cpu_results.len() as f32;

    eprintln!("GPU FFT validation ({n_bands} bands, {n_positions} positions): max_diff={:.2e}, mean_diff={:.2e}",
        max_diff, mean_diff);
    if max_diff < 1e-3 {
        eprintln!("  GPU FFT: PASS");
    } else {
        eprintln!("  GPU FFT: FAIL — check shader");
        // Print first few for debugging
        eprintln!("  CPU[0..5]: {:?}", &cpu_results[..5]);
        eprintln!("  GPU[0..5]: {:?}", &gpu_results[..5]);
    }
}
