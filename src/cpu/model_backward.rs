//! Model-level backward pass — extracted from main.rs.
//! Gradients struct, backward(), ffn_backward(), flatten_grads().
//! Also includes diagnostic helpers: linear_forward, kerr_ode_forward_cpu_standalone, rk4_step_standalone.

use crate::model::*;
use crate::wave_block::*;
use crate::ffn_backend;
use crate::backend;
use crate::ffn_gpu;
use crate::ffn_full_gpu;
use crate::gpu_pipelines;
use crate::common::dims::PROFILE;
use crate::Dims;
use crate::WavePacketModel;
use crate::cpu::forward::{BlockCache, ForwardCache};
use crate::cpu::backward::layer_norm_backward;
use rayon::prelude::*;

fn cross_entropy_backward(logits: &[f32], target: usize) -> Vec<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_l: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum_exp: f32 = exp_l.iter().sum();
    let mut d = exp_l.iter().map(|&e| e / sum_exp).collect::<Vec<f32>>();
    d[target] -= 1.0;
    d
}

pub struct Gradients {
    // Per-block FFN gradients
    pub block_ln_w: Vec<Vec<f32>>,
    pub block_ln_b: Vec<Vec<f32>>,
    pub block_ln_ffn_w: Vec<Vec<f32>>,
    pub block_ln_ffn_b: Vec<Vec<f32>>,
    pub block_ffn_kerr_gamma_raw: Vec<Vec<f32>>,
    pub block_ffn_kerr_omega: Vec<Vec<f32>>,
    pub block_ffn_kerr_alpha: Vec<f32>,
    pub block_ffn_kerr_beta: Vec<f32>,
    pub block_ffn_phase_correction: Vec<Vec<f32>>,
    pub block_ffn_mae_in_sq_w: Vec<Vec<Vec<f32>>>,
    pub block_ffn_mae_in_sq_b: Vec<Vec<f32>>,
    pub block_ffn_mae_in_pr_w: Vec<Vec<Vec<f32>>>,
    pub block_ffn_mae_in_pr_b: Vec<Vec<f32>>,
    pub block_ffn_mae_out_sq_w: Vec<Vec<Vec<f32>>>,
    pub block_ffn_mae_out_sq_b: Vec<Vec<f32>>,
    pub block_ffn_mae_out_pr_w: Vec<Vec<Vec<f32>>>,
    pub block_ffn_mae_out_pr_b: Vec<Vec<f32>>,
    pub block_ffn_out_proj_w: Vec<Vec<Vec<f32>>>,
    pub block_ffn_out_proj_b: Vec<Vec<f32>>,
    // Final
    pub ln_f_w: Vec<f32>,
    pub ln_f_b: Vec<f32>,
    pub lm_head: Vec<Vec<f32>>,
    pub lm_down: Vec<Vec<f32>>,
    pub lm_up: Vec<Vec<f32>>,
    pub tied_temperature: f32,
    // Layer scale gradients
    pub layer_scale: Vec<f32>,
    // Output corrector gradients (phase-native only)
    pub d_output_corrector: Vec<f32>,
    // Wave transduction gradients (self-contained)
    pub wd_grads: Option<crate::common::wave_decode::WaveDecodeGrads>,
}

pub fn backward(model: &WavePacketModel, cache: &ForwardCache, targets: &[usize], d: Dims, gpu: Option<&(dyn backend::ComputeBackend + Send + Sync)>, ping_pong: Option<(&ffn_gpu::FfnGpuBuffers, &gpu_pipelines::GpuBackend)>, full_gpu: Option<(&ffn_full_gpu::FfnFullBuffers, &gpu_pipelines::GpuBackend)>) -> (f32, Gradients) {
    let t = cache.logits.len();
    let vocab_size = model.vocab_size;

    // Init gradients
    let n_blocks = model.blocks.len();
    let mut grads = Gradients {
        block_ln_w: vec![vec![0.0; d.n_embd]; n_blocks],
        block_ln_b: vec![vec![0.0; d.n_embd]; n_blocks],
        block_ln_ffn_w: vec![vec![0.0; d.n_embd]; n_blocks],
        block_ln_ffn_b: vec![vec![0.0; d.n_embd]; n_blocks],
        block_ffn_kerr_gamma_raw: if d.learnable_ode { vec![vec![0.0; d.n_bands]; n_blocks] } else { vec![vec![]; n_blocks] },
        block_ffn_kerr_omega: vec![vec![0.0; d.n_bands]; n_blocks],
        block_ffn_kerr_alpha: vec![0.0; n_blocks],
        block_ffn_kerr_beta: vec![0.0; n_blocks],
        block_ffn_phase_correction: if d.learnable_ode { vec![vec![0.0; d.n_bands]; n_blocks] } else { vec![vec![]; n_blocks] },
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
        lm_head: if d.wave_decode || d.lm_rank > 0 || d.tied || model.phase_native { vec![] } else { vec![vec![0.0; d.n_embd]; vocab_size] },
        lm_down: if d.lm_rank > 0 { vec![vec![0.0; d.n_embd]; d.lm_rank] } else { vec![] },
        lm_up: if d.lm_rank > 0 { vec![vec![0.0; d.lm_rank]; vocab_size] } else { vec![] },
        layer_scale: if d.use_layer_scale { vec![0.0; n_blocks] } else { vec![] },
        d_output_corrector: vec![],
        tied_temperature: 0.0,
        wd_grads: None, // populated by wave_decode::backward when active
    };

    let n_embd = d.n_embd;
    let mut total_loss = 0.0f32;
    let mut d_hidden: Vec<Vec<f32>> = vec![vec![0.0f32; n_embd]; t];

    if model.phase_native {
        // Phase-native loss: compare post_ln_f against embeddings using phase coherence.
        // No lm_head involved. The ODE learns to output in embedding space.
        let temp = if d.phase_temp > 0.0 { d.phase_temp } else { 1.0 };
        let mut d_output_corrector = vec![0.0f32; d.n_bands];
        for pos in 0..t {
            let (loss, d_h, d_oc) = crate::common::phase_loss::phase_native_loss(
                &cache.post_ln_f[pos], &model.wte, targets[pos], d.n_bands, temp,
                &model.output_corrector,
            );
            total_loss += loss;
            for j in 0..n_embd { d_hidden[pos][j] = d_h[j] / t as f32; }
            for k in 0..d.n_bands { d_output_corrector[k] += d_oc[k] / t as f32; }
        }
        total_loss /= t as f32;
        // Store output corrector gradient for flatten_grads
        grads.d_output_corrector = d_output_corrector;
    } else {
    // Standard loss + d_logits through lm_head
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

    // Backward through output decoder
    if let Some(ref wds) = model.wd_state {
        let (dh, wg) = crate::common::wave_decode::backward(&d_logits, &cache.post_ln_f, wds);
        d_hidden = dh;
        grads.wd_grads = Some(wg);
    } else if d.lm_rank > 0 {
        // Low-rank backward: logits = lm_up @ (lm_down @ hidden)
        let rank = d.lm_rank;
        for pos in 0..t {
            let normed = &cache.post_ln_f[pos];
            // Recompute bottleneck
            let mut bottleneck = vec![0.0f32; rank];
            for r in 0..rank {
                let mut sum = 0.0f32;
                for j in 0..n_embd { sum += model.lm_down[r][j] * normed[j]; }
                bottleneck[r] = sum;
            }
            // d_bottleneck from d_logits through lm_up
            let mut d_bottleneck = vec![0.0f32; rank];
            for r in 0..rank {
                for v in 0..vocab_size {
                    d_bottleneck[r] += model.lm_up[v][r] * d_logits[pos][v];
                }
            }
            // d_hidden from d_bottleneck through lm_down
            for j in 0..n_embd {
                for r in 0..rank {
                    d_hidden[pos][j] += model.lm_down[r][j] * d_bottleneck[r];
                }
            }
            // Accumulate lm_up gradients
            for v in 0..vocab_size {
                for r in 0..rank {
                    grads.lm_up[v][r] += d_logits[pos][v] * bottleneck[r];
                }
            }
            // Accumulate lm_down gradients
            for r in 0..rank {
                for j in 0..n_embd {
                    grads.lm_down[r][j] += d_bottleneck[r] * normed[j];
                }
            }
        }
    } else {
        // Full-rank backward (existing)
        let decode_table = if d.tied { &model.wte } else { &model.lm_head };
        d_hidden = (0..t).into_par_iter().map(|pos| {
            let mut d_h = vec![0.0f32; n_embd];
            for j in 0..n_embd {
                for v in 0..vocab_size {
                    d_h[j] += decode_table[v][j] * d_logits[pos][v];
                }
            }
            d_h
        }).collect();
        if !d.tied {
            for pos in 0..t {
                for v in 0..vocab_size {
                    for j in 0..d.n_embd {
                        grads.lm_head[v][j] += d_logits[pos][v] * cache.post_ln_f[pos][j];
                    }
                }
            }
        }
    }
    } // end else (standard lm_head path)

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
        // output = input + scale * (attn_out + ffn_out) (parallel residual with layer scale)
        // d_contribution = scale * d_output, d_scale = dot(d_output, attn_out + ffn_out)
        let scale = if d.use_layer_scale { model.layer_scale[block_idx] } else { 1.0 };
        let d_ffn_out: Vec<Vec<f32>> = if scale != 1.0 {
            d_hidden.iter().map(|dh| dh.iter().map(|&v| v * scale).collect()).collect()
        } else {
            d_hidden.clone()
        };
        // Layer scale gradient: d_scale = sum_pos sum_dim d_hidden[pos][dim] * (attn_out[pos][dim] + ffn_out[pos][dim])
        if !grads.layer_scale.is_empty() {
            let mut d_s = 0.0f32;
            for pos in 0..t {
                for j in 0..d.n_embd {
                    d_s += d_hidden[pos][j] * (bc.attn_out[pos][j] + bc.ffn_out[pos][j]);
                }
            }
            grads.layer_scale[block_idx] += d_s;
        }

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
            // ODE param gradients
            if let Some(ref d_gr) = fg.d_kerr_gamma_raw {
                for k in 0..d_gr.len() { grads.block_ffn_kerr_gamma_raw[block_idx][k] += d_gr[k]; }
            }
            if let Some(d_a) = fg.d_kerr_alpha { grads.block_ffn_kerr_alpha[block_idx] += d_a; }
            if let Some(d_b) = fg.d_kerr_beta { grads.block_ffn_kerr_beta[block_idx] += d_b; }
            if let Some(ref d_pc) = fg.d_phase_correction {
                for k in 0..d_pc.len() { grads.block_ffn_phase_correction[block_idx][k] += d_pc[k]; }
            }
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
        let cpu_act: Vec<f32> = cpu_sq.iter().map(|&v| gelu(v)).collect();
        let cpu_mae_in = linear_forward(&weights.maestro_in.process_1.w, &weights.maestro_in.process_1.b, &cpu_act);
        let mut cpu_precond = vec![0.0f32; d.n_embd];
        for i in 0..d.n_embd { cpu_precond[i] = normed[pos][i] + cpu_mae_in[i]; }
        let cpu_kerr = kerr_ode_forward_cpu_standalone(&weights.kerr, &cpu_precond);
        let cpu_sq2 = linear_forward(&weights.maestro_out.squeeze.w, &weights.maestro_out.squeeze.b, &cpu_kerr);
        let cpu_act2: Vec<f32> = cpu_sq2.iter().map(|&v| gelu(v)).collect();
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

pub fn flatten_grads(grads: &Gradients) -> Vec<f32> {
    flatten_grads_ex(grads, false)
}

pub fn flatten_grads_ex(grads: &Gradients, tied: bool) -> Vec<f32> {
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
        // ODE param gradients (when learnable)
        if !grads.block_ffn_kerr_gamma_raw[b].is_empty() {
            g.extend_from_slice(&grads.block_ffn_kerr_gamma_raw[b]);
            g.push(grads.block_ffn_kerr_alpha[b]);
            g.push(grads.block_ffn_kerr_beta[b]);
            g.extend_from_slice(&grads.block_ffn_phase_correction[b]);
        }
    }
    if !grads.layer_scale.is_empty() {
        g.extend_from_slice(&grads.layer_scale);
    }
    g.extend_from_slice(&grads.ln_f_w);
    g.extend_from_slice(&grads.ln_f_b);
    if !grads.d_output_corrector.is_empty() {
        g.extend_from_slice(&grads.d_output_corrector);
    }
    if let Some(ref wg) = grads.wd_grads {
        g.extend_from_slice(&crate::common::wave_decode::flatten_grads(wg));
    } else if !grads.lm_down.is_empty() {
        // Low-rank
        for row in &grads.lm_down { g.extend_from_slice(row); }
        for row in &grads.lm_up { g.extend_from_slice(row); }
    } else if !tied {
        for row in &grads.lm_head { g.extend_from_slice(row); }
    } else {
        g.push(grads.tied_temperature);
    }
    g
}
