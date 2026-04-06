//! Forward pass with cache — extracted from main.rs.
//! BlockCache, ForwardCache, forward_with_cache, dual_maestro_forward.

use crate::model::*;
use crate::wave_embed::*;
use crate::wave_attn::*;
use crate::wave_block::*;
use crate::ffn_backend;
use crate::backend;
use crate::ffn_gpu;
use crate::ffn_full_gpu;
use crate::gpu_pipelines;
use crate::fft_ode;
use crate::common::dims::PROFILE;
use crate::Dims;
use crate::WavePacketModel;
use rayon::prelude::*;

pub struct BlockCache {
    pub input: Vec<Vec<f32>>,
    pub normed: Vec<Vec<f32>>,
    pub normed_ffn: Vec<Vec<f32>>,
    pub attn_out: Vec<Vec<f32>>,
    pub ffn_out: Vec<Vec<f32>>,
    pub att_weights: Vec<Vec<Vec<f32>>>,
    // FFN intermediates — two paths:
    // 1. Backend cache (new: all ops through ComputeBackend, self-consistent)
    pub ffn_backend_cache: Option<ffn_backend::FfnCache>,
    // 2. Legacy cache (old: hand-wired, for fallback)
    pub ffn_mae_in_sq: Vec<Vec<f32>>,
    pub ffn_mae_in_act: Vec<Vec<f32>>,
    pub ffn_precond: Vec<Vec<f32>>,
    pub ffn_kerr_out: Vec<Vec<f32>>,
    pub ffn_mae_out_sq: Vec<Vec<f32>>,
    pub ffn_mae_out_act: Vec<Vec<f32>>,
    pub ffn_regulated: Vec<Vec<f32>>,
}

pub struct ForwardCache {
    pub block_caches: Vec<BlockCache>,
    pub pre_ln_f: Vec<Vec<f32>>,
    pub post_ln_f: Vec<Vec<f32>>,
    pub logits: Vec<Vec<f32>>,
}

pub fn forward_with_cache(
    model: &WavePacketModel,
    tokens: &[usize],
    d: Dims,
    gpu: Option<&(dyn backend::ComputeBackend + Send + Sync)>,
    ping_pong: Option<(&ffn_gpu::FfnGpuBuffers, &gpu_pipelines::GpuBackend)>,
    full_gpu: Option<(&ffn_full_gpu::FfnFullBuffers, &gpu_pipelines::GpuBackend)>,
    stencil: Option<&fft_ode::StencilFft>,
    gpu_kernel: Option<(&fft_ode::GpuKernelFft, &gpu_pipelines::GpuBackend)>,
    layer_agcs: Option<&mut [crate::common::agc::OdeAgc]>,
) -> ForwardCache {
    let profile = PROFILE.load(std::sync::atomic::Ordering::Relaxed);
    let t = tokens.len();
    let _t0 = std::time::Instant::now();
    let mut hidden = embed_tokens(tokens, &model.wte, &model.wpe, d.n_embd);
    let mut block_caches = Vec::new();
    let mut _attn_total = std::time::Duration::ZERO;
    let mut _ffn_total = std::time::Duration::ZERO;
    let mut _ln_total = std::time::Duration::ZERO;

    // Track layer index for per-layer AGC
    let mut layer_agcs_ref = layer_agcs;
    for (block_idx, block) in model.blocks.iter().enumerate() {
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

        // Per-layer AGC: borrow one element from the mutable slice
        let agc_for_layer = if let Some(ref mut agcs) = layer_agcs_ref {
            Some(&mut agcs[block_idx])
        } else {
            None
        };

        // FFN forward via backend (kerr-engine pattern: all ops through same device)
        let _tf = std::time::Instant::now();
        let freeze_ode = !d.learnable_ode;
        let use_corrector = d.use_corrector;
        let (ffn_out, ffn_be_cache) = ffn_backend::ffn_forward_via_backend(&block.ffn, &normed, be, stencil, ping_pong, gpu_kernel, freeze_ode, use_corrector, agc_for_layer);
        let ffn_dur = _tf.elapsed();

        // Attention (CPU — frozen, harmonic coherence scoring)
        let _ta = std::time::Instant::now();
        let (attn_out, att_weights) = wave_attention_forward(&block.attn, &normed, d.n_bands, gpu);
        let attn_dur = _ta.elapsed();
        _attn_total += attn_dur;
        _ffn_total += ffn_dur;

        let scale = if model.use_layer_scale { model.layer_scale[block_idx] } else { 1.0 };
        let output: Vec<Vec<f32>> = (0..t).map(|i| {
            let mut v = vec![0.0f32; d.n_embd];
            for j in 0..d.n_embd { v[j] = hidden[i][j] + scale * (attn_out[i][j] + ffn_out[i][j]); }
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

    let logits: Vec<Vec<f32>> = if model.phase_native {
        // Phase-native: dot product against embeddings (replaces lm_head)
        let n_bands = d.n_bands;
        post_ln_f.par_iter().map(|hidden| {
            // Apply output corrector: per-band phase rotation
            let mut corrected = vec![0.0f32; n_bands * 2];
            for k in 0..n_bands {
                let (sin_c, cos_c) = model.output_corrector[k].sin_cos();
                let r = hidden[k * 2];
                let s = hidden[k * 2 + 1];
                corrected[k * 2]     = r * cos_c - s * sin_c;
                corrected[k * 2 + 1] = r * sin_c + s * cos_c;
            }
            // Apply output_scale (prism layer)
            for k in 0..n_bands {
                corrected[k * 2]     *= model.output_scale[k];
                corrected[k * 2 + 1] *= model.output_scale[k];
            }
            // Coherent detection: I/Q scoring
            let scale = 1.0 / ((n_bands * 2) as f32).sqrt();
            let detect = model.detect_mode;
            let iq_w = model.iq_weights;
            (0..model.vocab_size).map(|v| {
                let emb = &model.wte[v];
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
                let score = match detect {
                    crate::common::phase_loss::DetectMode::I => i_sum,
                    crate::common::phase_loss::DetectMode::Q => q_sum,
                    crate::common::phase_loss::DetectMode::IQ => {
                        let w = iq_w.unwrap_or([1.0, 0.0]);
                        w[0] * i_sum + w[1] * q_sum
                    }
                };
                score * scale
            }).collect()
        }).collect()
    } else if let Some(ref wds) = model.wd_state {
        crate::common::wave_decode::forward(&post_ln_f, wds)
    } else if model.lm_rank > 0 {
        // Low-rank: hidden → lm_down → lm_up → logits
        let rank = model.lm_rank;
        post_ln_f.par_iter().map(|normed| {
            let mut bottleneck = vec![0.0f32; rank];
            for r in 0..rank {
                let mut sum = 0.0f32;
                for j in 0..d.n_embd { sum += model.lm_down[r][j] * normed[j]; }
                bottleneck[r] = sum;
            }
            let mut l = vec![0.0f32; model.vocab_size];
            for v in 0..model.vocab_size {
                let mut sum = 0.0f32;
                for r in 0..rank { sum += model.lm_up[v][r] * bottleneck[r]; }
                l[v] = sum;
            }
            l
        }).collect()
    } else {
        let decode_table = if d.tied { &model.wte } else { &model.lm_head };
        let tied_temp = if d.tied { model.tied_temperature } else { 1.0 };
        post_ln_f.par_iter().map(|normed| {
            let mut l = vec![0.0f32; model.vocab_size];
            for v in 0..model.vocab_size {
                let mut sum = 0.0f32;
                for j in 0..d.n_embd { sum += decode_table[v][j] * normed[j]; }
                l[v] = sum * tied_temp;
            }
            l
        }).collect()
    };

    if profile {
        let total = _t0.elapsed();
        eprintln!("    [profile fwd] LN: {:?}  Attn: {:?}  FFN: {:?}  Total: {:?}",
            _ln_total, _attn_total, _ffn_total, total);
    }

    ForwardCache { block_caches, pre_ln_f: hidden, post_ln_f, logits }
}

// FFN forward — now routes through GPU backend for the full FFN path
pub fn dual_maestro_forward(
    weights: &KerrDualMaestroWeights,
    x: &[Vec<f32>],
    gpu: Option<&(dyn backend::ComputeBackend + Send + Sync)>,
) -> (Vec<Vec<f32>>, crate::wave_block::FfnForwardCache) {
    crate::wave_block::dual_maestro_forward_cached(weights, x, gpu, None)
}
