//! Training health monitors, spring regulation, and JSONL telemetry.
//!
//! Extracted from the monolithic run_training() in train_loop.rs.
//! Functions here are called from the training iteration loop.

use crate::*;
use crate::cpu::train::TrainConfig;

// ─── Spring regulation ─────────────────────────────────────────

/// Apply all spring restoring forces to dynamic parameters.
/// Called after optimizer step, before next iteration.
pub fn apply_springs(
    model: &mut WavePacketModel,
    config: &TrainConfig,
    current_lr: f32,
) {
    // Layer scale spring: restoring force toward equilibrium.
    // Spring is in the optimizer flow (like weight decay), not bolted onto loss.
    // param -= lr * k * (param - eq)
    if config.layer_scale.is_dynamic() && config.spring_k > 0.0 {
        let active = config.active_layers.unwrap_or(model.blocks.len());
        for l in 0..model.layer_scale.len() {
            let eq = if l < active { 1.0 } else { 0.0 };
            model.layer_scale[l] -= current_lr * config.spring_k * (model.layer_scale[l] - eq);
            if model.layer_scale[l] < 0.0 { model.layer_scale[l] = 0.0; }
        }
    }

    // WD spring: stiff restoring force toward uniform (1.0).
    // param -= lr * k * (param - eq), eq=1.0, k=1.0 (stiff)
    if config.wd.is_dynamic() && config.spring_k > 0.0 {
        let k_wd = config.spring_k * 1.0; // stiff
        for s in &mut model.wd_scale {
            *s -= current_lr * k_wd * (*s - 1.0);
            *s = s.clamp(0.01, 10.0); // don't let WD go negative or extreme
        }
    }

    // RK4 weights spring: very stiff restoring force toward standard [1/6, 1/3, 1/3, 1/6].
    // Spring k=2.0 (relative to global spring_k). Standard RK4 is mathematically motivated.
    if config.rk4_weights.is_dynamic() && config.spring_k > 0.0 {
        let k_rk4 = config.spring_k * 2.0; // very stiff
        let eq: [f32; 4] = [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0];
        for block in &mut model.blocks {
            for w in 0..4 {
                block.ffn.kerr.rk4_weights[w] -= current_lr * k_rk4 * (block.ffn.kerr.rk4_weights[w] - eq[w]);
            }
        }
    }

    // Harmonics spring: very stiff restoring force toward initial integer-ish harmonics.
    // Equilibrium = ln((h+1)*0.5) which softplus-> approximately 0.5, 1.0, 1.5, 2.0
    if config.harmonics.is_dynamic() && config.spring_k > 0.0 {
        let k_harm = config.spring_k * 2.0; // very stiff — integer harmonics theoretically motivated
        let n_head = model.blocks[0].attn.heads.len();
        for block in &mut model.blocks {
            for h in 0..n_head {
                let eq = ((h + 1) as f32 * 0.5f32).ln(); // same as init_model
                block.attn.heads[h].harmonic_raw -= current_lr * k_harm * (block.attn.heads[h].harmonic_raw - eq);
            }
        }
    }

    // Corrector plate spring: very loose restoring force toward 0.0 (transparent).
    // k=0.01 relative to global spring — corrections earned easily.
    if config.corrector.is_dynamic() && config.spring_k > 0.0 {
        let k_corr = config.spring_k * 0.01; // very loose
        for block in &mut model.blocks {
            for pc in &mut block.ffn.kerr.phase_correction {
                *pc -= current_lr * k_corr * *pc; // eq = 0.0, so (pc - 0.0) = pc
            }
        }
    }

    // AGC headroom spring: stiff restoring force toward 3.0 (3-sigma default).
    if config.agc_headroom.is_dynamic() && config.spring_k > 0.0 {
        let k_agc = config.spring_k * 1.0; // stiff — safety motivated
        for hr in &mut model.agc_headroom {
            *hr -= current_lr * k_agc * (*hr - 3.0);
            *hr = hr.clamp(1.0, 6.0); // don't go below 1-sigma or above 6-sigma
        }
    }

    // Dynamic AGC: update ceiling from learned coupling constants.
    // Uses min ceiling across all layers (most conservative — prevents divergence).
    if !config.freeze_ode {
        let mut min_ceiling = f32::MAX;
        for block in &model.blocks {
            let a = block.ffn.kerr.alpha;
            let b = block.ffn.kerr.beta;
            let c = (std::f32::consts::FRAC_PI_2 / (a + 4.0 * b)).sqrt().max(0.5);
            if c < min_ceiling { min_ceiling = c; }
        }
        // Apply CLI override as maximum
        let effective = match config.agc_ceiling {
            Some(cli) => min_ceiling.min(cli),
            None => min_ceiling,
        };
        if let Some(agc_lock) = crate::ffn_backend::AGC.get() {
            let mut agc = agc_lock.lock().unwrap();
            agc.update_ceiling_with_max(
                model.blocks[0].ffn.kerr.alpha,
                model.blocks[0].ffn.kerr.beta,
                Some(effective),
            );
        }
    }
}

// ─── LR scale + per-group WD ───────────────────────────────────

/// Apply LR scale: per-group gradient scaling before optimizer step.
/// Returns true if dynamic lr spring/hypergradient was applied.
pub fn apply_lr_scale(
    model: &mut WavePacketModel,
    config: &TrainConfig,
    dims: Dims,
    total_grads: &mut [f32],
    current_lr: f32,
) {
    if !config.lr_scale.is_active() { return; }

    // Fixed mode: apply prescribed scales. Dynamic mode: hypergradient adjusts.
    let is_dynamic_lr = config.lr_scale.is_dynamic();
    let n_layers = model.blocks.len();
    let n_embd = dims.n_embd;
    let maestro_dim = crate::MAESTRO_DIM;
    let per_block = n_embd * 4
        + maestro_dim * n_embd + maestro_dim + n_embd * maestro_dim + n_embd
        + maestro_dim * n_embd + maestro_dim + n_embd * maestro_dim + n_embd
        + model.blocks[0].ffn.out_proj.param_count();
    let ode_per = if model.learnable_ode {
        model.blocks[0].ffn.kerr.gamma_raw.len() + 1 + 1 + model.blocks[0].ffn.kerr.phase_correction.len()
    } else { 0 };
    let block_total = per_block + ode_per;
    let ls_count = if model.use_layer_scale { n_layers } else { 0 };

    // Scale per-layer gradients
    for l in 0..n_layers {
        let start = l * block_total;
        let end = start + block_total;
        let s = model.lr_scale[l];
        for i in start..end.min(total_grads.len()) {
            total_grads[i] *= s;
        }
    }
    // Scale lm_head gradients (last group)
    let head_start = n_layers * block_total + ls_count + n_embd * 2;
    let s_head = model.lr_scale[n_layers];
    for i in head_start..total_grads.len() {
        total_grads[i] *= s_head;
    }

    // Spring + hypergradient only in dynamic mode (not when human prescribed values)
    if is_dynamic_lr {
        // Spring on lr_scale: pull toward 1.0
        let k_lr = config.spring_k * 0.5;
        for s in &mut model.lr_scale {
            *s -= current_lr * k_lr * (*s - 1.0);
            *s = s.clamp(0.1, 5.0);
        }

        // Hypergradient: adjust lr_scale based on gradient magnitude per group
        for l in 0..n_layers {
            let start = l * block_total;
            let end = (start + block_total).min(total_grads.len());
            let gn: f32 = total_grads[start..end].iter().map(|g| g * g).sum::<f32>().sqrt();
            let avg_gn: f32 = total_grads.iter().map(|g| g * g).sum::<f32>().sqrt() / (n_layers as f32 + 1.0);
            if avg_gn > 0.001 {
                // Nudge scale toward where gradients are larger
                model.lr_scale[l] += current_lr * 0.01 * (gn / avg_gn - 1.0);
                model.lr_scale[l] = model.lr_scale[l].clamp(0.1, 5.0);
            }
        }
    } // end is_dynamic_lr
}

/// Apply per-group weight decay (when --wd is active).
/// Returns modified params with per-group WD applied.
pub fn apply_per_group_wd(
    model: &WavePacketModel,
    config: &TrainConfig,
    dims: Dims,
    params: &mut [f32],
    current_lr: f32,
) {
    if !config.wd.is_active() { return; }

    let base_wd = 0.01f32;
    let n_layers = model.blocks.len();
    let n_embd = dims.n_embd;
    let maestro_dim = crate::MAESTRO_DIM;
    let per_block = n_embd * 4
        + maestro_dim * n_embd + maestro_dim + n_embd * maestro_dim + n_embd
        + maestro_dim * n_embd + maestro_dim + n_embd * maestro_dim + n_embd
        + model.blocks[0].ffn.out_proj.param_count();
    let ode_per = if model.learnable_ode {
        model.blocks[0].ffn.kerr.gamma_raw.len() + 1 + 1 + model.blocks[0].ffn.kerr.phase_correction.len()
        + if model.use_rk4_weights { 4 } else { 0 }
    } else { 0 };
    let block_total = per_block + ode_per;
    let ls_count = if model.use_layer_scale { n_layers } else { 0 };

    // Apply per-layer WD
    for l in 0..n_layers {
        let start = l * block_total;
        let end = (start + block_total).min(params.len());
        let wd_eff = base_wd * model.wd_scale[l];
        for i in start..end {
            params[i] -= current_lr * wd_eff * params[i];
        }
    }
    // Apply lm_head group WD
    let head_start = n_layers * block_total + ls_count + n_embd * 2;
    let wd_head = base_wd * model.wd_scale[n_layers];
    for i in head_start..params.len() {
        params[i] -= current_lr * wd_head * params[i];
    }
}

// ─── First-10 health check ─────────────────────────────────────

/// Early health check: per-component gradient norms + weight growth (first 10 iters).
pub fn first10_health_check(
    iter: usize,
    model: &WavePacketModel,
    total_grads: &[f32],
    total_loss: f32,
    n_trainable: usize,
    grad_norm: f32,
) {
    if iter >= 10 { return; }

    let lm_head_size = model.vocab_size * model.ln_f.weight.len();
    let lm_start = n_trainable.saturating_sub(lm_head_size);
    let model_gn: f32 = total_grads[..lm_start].iter().map(|g| g * g).sum::<f32>().sqrt();
    let head_gn: f32 = total_grads[lm_start..].iter().map(|g| g * g).sum::<f32>().sqrt();
    let total_gn = grad_norm.max(0.001);
    let alpha = model.blocks[0].ffn.kerr.alpha;
    let beta = model.blocks[0].ffn.kerr.beta;
    eprintln!("  [health {}] loss={:.2} model_gn={:.2} head_gn={:.2} head%={:.1} alpha={:.4} beta={:.4}",
        iter, total_loss, model_gn, head_gn, head_gn / total_gn * 100.0, alpha, beta);
    if head_gn / total_gn > 0.95 {
        eprintln!("  [health {}] ALERT: lm_head gradient dominance {:.1}%", iter, head_gn / total_gn * 100.0);
    }
}

// ─── JSONL telemetry ────────────────────────────────────────────

/// Write per-iteration JSONL telemetry line (compact every iter, detailed every 100).
pub fn write_jsonl_telemetry(
    log_writer: &mut std::io::BufWriter<std::fs::File>,
    iter: usize,
    total_loss: f32,
    current_lr: f32,
    iter_elapsed_ms: u128,
    nan_skip_count: usize,
    model: &WavePacketModel,
    config: &TrainConfig,
    total_grads: &[f32],
    grad_norm: f32,
    n_trainable: usize,
    dims: Dims,
) {
    use std::io::Write;
    let lm_head_size = model.vocab_size * model.ln_f.weight.len();
    let lm_start = n_trainable.saturating_sub(lm_head_size);
    if iter % 100 == 0 {
        // Gradient balance: model vs lm_head
        let model_gn: f32 = total_grads[..lm_start].iter().map(|g| g * g).sum::<f32>().sqrt();
        let head_gn: f32 = total_grads[lm_start..].iter().map(|g| g * g).sum::<f32>().sqrt();
        let head_pct = head_gn / grad_norm.max(0.001) * 100.0;
        // Per-layer gradient norms (model params only, split by layer)
        let n_layers = model.blocks.len();
        let model_params = lm_start;
        let per_layer = model_params / n_layers.max(1);
        let layer_gns: Vec<f32> = (0..n_layers).map(|l| {
            let start = l * per_layer;
            let end = ((l + 1) * per_layer).min(lm_start);
            total_grads[start..end].iter().map(|g| g * g).sum::<f32>().sqrt()
        }).collect();
        let layer_str: String = layer_gns.iter().map(|g| format!("{:.3}", g)).collect::<Vec<_>>().join(",");
        // ODE clamp stats + AGC state from FFN forward
        let clamp_count = crate::ffn_backend::ODE_CLAMP_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        let max_mag = f32::from_bits(crate::ffn_backend::ODE_MAX_MAG.load(std::sync::atomic::Ordering::Relaxed));
        let agc = crate::ffn_backend::agc_stats();
        // ODE param values and gradient norms (when learnable)
        let ode_str = if !config.freeze_ode {
            let mut parts = Vec::new();
            for (l, block) in model.blocks.iter().enumerate() {
                let a = block.ffn.kerr.alpha;
                let b = block.ffn.kerr.beta;
                let g_mean: f32 = block.ffn.kerr.gamma_raw.iter().map(|&g| {
                    if g > 20.0 { g } else { (1.0 + g.exp()).ln() } // softplus
                }).sum::<f32>() / block.ffn.kerr.gamma_raw.len() as f32;
                // Gradient norms for ODE params
                let _a_gn = total_grads.get(l * per_layer + per_layer - 2).copied().unwrap_or(0.0).abs();
                let _b_gn = total_grads.get(l * per_layer + per_layer - 1).copied().unwrap_or(0.0).abs();
                parts.push(format!(r#"{{"a":{:.4},"b":{:.4},"g":{:.4}}}"#, a, b, g_mean));
            }
            format!(r#","ode_params":[{}]"#, parts.join(","))
        } else {
            String::new()
        };
        let ls_str = if config.layer_scale.is_active() {
            let vals: Vec<String> = model.layer_scale.iter().map(|s| format!("{:.4}", s)).collect();
            format!(r#","layer_scale":[{}]"#, vals.join(","))
        } else {
            String::new()
        };
        let lrs_str = if config.lr_scale.is_active() {
            let vals: Vec<String> = model.lr_scale.iter().map(|s| format!("{:.4}", s)).collect();
            format!(r#","lr_scale":[{}]"#, vals.join(","))
        } else {
            String::new()
        };
        let rk4w_str = if config.rk4_weights.is_active() {
            let mut parts = Vec::new();
            for (l, block) in model.blocks.iter().enumerate() {
                let w = &block.ffn.kerr.rk4_weights;
                parts.push(format!(r#"{{"L{}": [{:.4},{:.4},{:.4},{:.4}]}}"#, l, w[0], w[1], w[2], w[3]));
            }
            format!(r#","rk4_weights":[{}]"#, parts.join(","))
        } else {
            String::new()
        };
        let wd_str = if config.wd.is_active() {
            let vals: Vec<String> = model.wd_scale.iter().map(|s| format!("{:.4}", s)).collect();
            format!(r#","wd_scale":[{}]"#, vals.join(","))
        } else {
            String::new()
        };
        let harm_str = if config.harmonics.is_active() {
            let mut parts = Vec::new();
            for (l, block) in model.blocks.iter().enumerate() {
                let vals: Vec<String> = block.attn.heads.iter().map(|h| format!("{:.4}", crate::common::math::softplus(h.harmonic_raw))).collect();
                parts.push(format!(r#"{{"L{}": [{}]}}"#, l, vals.join(",")));
            }
            format!(r#","harmonics":[{}]"#, parts.join(","))
        } else {
            String::new()
        };
        let agc_hr_str = if config.agc_headroom.is_active() {
            let vals: Vec<String> = model.agc_headroom.iter().map(|h| format!("{:.2}", h)).collect();
            format!(r#","agc_headroom":[{}]"#, vals.join(","))
        } else {
            String::new()
        };
        writeln!(log_writer,
            r#"{{"iter":{},"loss":{:.4},"lr":{:.6},"time_ms":{},"nan_skips":{},"model_gn":{:.4},"head_gn":{:.4},"head_pct":{:.1},"layer_gn":[{}],"ode_clamps":{},"ode_max_mag":{:.2},"agc_threshold":{:.3},"agc_mean":{:.3},"agc_std":{:.3}{}{}{}{}{}{}{}}}"#,
            iter, total_loss, current_lr, iter_elapsed_ms, nan_skip_count,
            model_gn, head_gn, head_pct, layer_str, clamp_count, max_mag,
            agc.threshold, agc.ema_mean, agc.ema_std, ode_str, ls_str, lrs_str, rk4w_str, wd_str, harm_str, agc_hr_str
        ).ok();
    } else {
        writeln!(log_writer,
            r#"{{"iter":{},"loss":{:.4},"lr":{:.6},"time_ms":{},"nan_skips":{}}}"#,
            iter, total_loss, current_lr, iter_elapsed_ms, nan_skip_count
        ).ok();
    }
    log_writer.flush().ok();
}

/// Collected health data from first batch element at health intervals.
pub struct BatchHealthData {
    pub distortion: Option<Vec<crate::common::ode_distortion::LayerDistortionSummary>>,
    pub grad_flow: Option<Vec<crate::common::gradient_monitor::GradientFlowStats>>,
    pub attn_stats: Option<Vec<crate::common::attn_monitor::AttentionHeadStats>>,
    pub flow_stats: Option<Vec<crate::common::layer_flow_monitor::LayerFlowStats>>,
    pub output_stats: Option<crate::common::output_monitor::OutputDistStats>,
    pub ode_dynamics: Option<Vec<crate::common::ode_dynamics_monitor::OdeDynamicsStats>>,
    pub iq_analysis: Option<crate::common::iq_monitor::IqAnalysis>,
}

/// Write all health monitor data to JSONL at health intervals.
pub fn write_health_monitors(
    log_writer: &mut std::io::BufWriter<std::fs::File>,
    iter: usize,
    iters_into_run: usize,
    model: &WavePacketModel,
    config: &TrainConfig,
    dims: Dims,
    stencil: &crate::fft_ode::StencilFft,
    batch_distortion_data: &Option<Vec<crate::common::ode_distortion::LayerDistortionSummary>>,
    batch_grad_flow: &Option<Vec<crate::common::gradient_monitor::GradientFlowStats>>,
    batch_attn_stats: &Option<Vec<crate::common::attn_monitor::AttentionHeadStats>>,
    batch_flow_stats: &Option<Vec<crate::common::layer_flow_monitor::LayerFlowStats>>,
    batch_output_stats: &Option<crate::common::output_monitor::OutputDistStats>,
    batch_ode_dynamics: &Option<Vec<crate::common::ode_dynamics_monitor::OdeDynamicsStats>>,
    batch_iq_analysis: &Option<crate::common::iq_monitor::IqAnalysis>,
    prev_dyn_snap: &mut Option<crate::common::dyn_param_monitor::DynParamSnapshot>,
    batch_size: usize,
    seq_len: usize,
    iter_elapsed_secs: f32,
    fwd_bwd_elapsed_secs: f32,
    optim_elapsed_secs: f32,
) {
    use std::io::Write;

    // Encoding health sample
    if let Some(h) = crate::common::encoding_health::sample(
        model, dims, config.use_bpe, &config.tokenizer_path, stencil,
        config.alpha, config.beta,
    ) {
        let health_json = crate::common::encoding_health::to_json(&h);
        writeln!(log_writer, r#"{{"iter":{},"type":"health",{}}}"#, iter, health_json).ok();
        log_writer.flush().ok();
        // Console warning on drift
        if h.entropy > 0.60 && (h.theta_disc > 2.0 * h.delta_theta_disc || h.delta_theta_disc > 2.0 * h.theta_disc) {
            eprintln!("  [enc-health {}] WARNING: entropy={:.3} θ={:.2}x Δθ={:.2}x — encoding drift",
                iter, h.entropy, h.theta_disc, h.delta_theta_disc);
        } else if iters_into_run % (config.health_interval * 5) == 0 {
            let thd_str = if let Some(ref d) = h.distortion {
                format!(" THD={:.3} gain={:.2}", d.thd_total, d.gain_max)
            } else { String::new() };
            eprintln!("  [enc-health {}] θ={:.2}x Δθ={:.2}x entropy={:.3} top=band{} ({:.1}x){}",
                iter, h.theta_disc, h.delta_theta_disc, h.entropy, h.top_band, h.concentration, thd_str);
        }
    }

    // Batch distortion: measured on actual training data (not reference sentence)
    if let Some(layers) = batch_distortion_data {
        let json = crate::common::ode_distortion::batch_to_json(iter, layers);
        writeln!(log_writer, "{}", json).ok();
        log_writer.flush().ok();
        // Console summary: show per-layer THD and gain
        let layer_strs: Vec<String> = layers.iter().map(|l| {
            format!("L{}:THD={:.3}/gain={:.2}/comp={}", l.layer, l.thd_avg, l.gain_max, l.n_compressed)
        }).collect();
        eprintln!("  [batch-distortion {}] {}", iter, layer_strs.join(" | "));
    }

    // --- Monitor suite (batch 1) ---

    // Gradient flow per component (#3)
    if let Some(gf_stats) = batch_grad_flow {
        let gf_json = crate::common::gradient_monitor::to_json(gf_stats);
        if !gf_json.is_empty() {
            writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, gf_json).ok();
            log_writer.flush().ok();
        }
    }

    // Dynamic parameter evolution (#7)
    {
        let snap = crate::common::dyn_param_monitor::snapshot(model, prev_dyn_snap.as_ref());
        let dp_json = crate::common::dyn_param_monitor::to_json(&snap);
        if !dp_json.is_empty() {
            writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, dp_json).ok();
            log_writer.flush().ok();
        }
        *prev_dyn_snap = Some(snap);
    }

    // Throughput (#10)
    {
        let tp_stats = crate::common::throughput_monitor::compute(
            batch_size,
            seq_len,
            iter_elapsed_secs * 1000.0,
            fwd_bwd_elapsed_secs * 1000.0,
            optim_elapsed_secs * 1000.0,
        );
        let tp_json = crate::common::throughput_monitor::to_json(&tp_stats);
        writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, tp_json).ok();
        log_writer.flush().ok();
    }

    // --- Monitor suite (batch 2) ---

    // Attention head activity (#1)
    if let Some(attn_stats) = batch_attn_stats {
        let attn_json = crate::common::attn_monitor::to_json(attn_stats);
        if !attn_json.is_empty() {
            writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, attn_json).ok();
            log_writer.flush().ok();
        }
    }

    // Layer signal flow (#2)
    if let Some(flow_stats) = batch_flow_stats {
        let flow_json = crate::common::layer_flow_monitor::to_json(flow_stats);
        if !flow_json.is_empty() {
            writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, flow_json).ok();
            log_writer.flush().ok();
        }
    }

    // Output distribution (#5)
    if let Some(output_stats) = batch_output_stats {
        let out_json = crate::common::output_monitor::to_json(output_stats);
        writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, out_json).ok();
        log_writer.flush().ok();
    }

    // --- Monitor suite (batch 3) ---

    // Embedding space (#4) — frozen embeddings, only changes at iter 0
    if iters_into_run == 0 {
        let embed_stats = crate::common::embedding_monitor::analyze_embeddings(model);
        let embed_json = crate::common::embedding_monitor::to_json(&embed_stats);
        writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, embed_json).ok();
        log_writer.flush().ok();
    }

    // ODE dynamics deep (#6)
    if let Some(ode_dyn) = batch_ode_dynamics {
        let ode_json = crate::common::ode_dynamics_monitor::to_json(ode_dyn);
        if !ode_json.is_empty() {
            writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, ode_json).ok();
            log_writer.flush().ok();
        }
    }

    // I/Q channel analysis — now persisted to JSONL (was eprintln-only)
    if let Some(iq) = batch_iq_analysis {
        writeln!(log_writer,
            r#"{{"iter":{},"type":"monitor","iq":{{"i_discrim":{:.4},"q_discrim":{:.4},"iq_ratio":{:.4},"phase_mean":{:.4},"phase_std":{:.4},"i_correct_rank":{},"q_correct_rank":{}}}}}"#,
            iter, iq.i_discrim, iq.q_discrim, iq.iq_ratio, iq.phase_mean, iq.phase_std,
            iq.i_correct_rank, iq.q_correct_rank
        ).ok();
        log_writer.flush().ok();
    }
}

/// Write curriculum transition event to JSONL.
pub fn write_curriculum_event(
    log_writer: &mut std::io::BufWriter<std::fs::File>,
    iter: usize,
    event: &crate::common::curriculum_monitor::CurriculumStats,
) {
    use std::io::Write;
    let cur_json = crate::common::curriculum_monitor::to_json(event);
    writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, cur_json).ok();
    log_writer.flush().ok();
    eprintln!("  [curriculum {}] stage {} → {} bands, loss jump {:.4}",
        iter, event.stage, event.active_bands, event.loss_jump);
}

/// Write checkpoint drift event to JSONL.
pub fn write_checkpoint_drift(
    log_writer: &mut std::io::BufWriter<std::fs::File>,
    iter: usize,
    drift: &crate::common::checkpoint_monitor::CheckpointDrift,
) {
    use std::io::Write;
    let drift_json = crate::common::checkpoint_monitor::to_json(drift);
    writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, drift_json).ok();
    log_writer.flush().ok();
    eprintln!("  [drift {}] total={:.4} relative={:.6} ode={:.4}",
        iter, drift.total_drift, drift.relative_drift, drift.ode_drift);
}

/// Write training summary to JSONL and console.
pub fn write_training_summary(
    log_writer: &mut std::io::BufWriter<std::fs::File>,
    config: &TrainConfig,
    start_iter: usize,
    total_iters: usize,
    best_loss: f32,
    best_iter: usize,
    loss_history: &[f32],
    nan_skip_count: usize,
    train_start: std::time::Instant,
    vocab_size: usize,
) {
    use std::io::Write;
    let total_time = train_start.elapsed();
    let ms_per_iter = if config.n_iters > 0 { total_time.as_millis() as f64 / config.n_iters as f64 } else { 0.0 };
    let final_avg = if loss_history.len() >= 100 {
        loss_history[loss_history.len()-100..].iter().sum::<f32>() / 100.0
    } else if !loss_history.is_empty() {
        loss_history.iter().sum::<f32>() / loss_history.len() as f32
    } else { 0.0 };

    println!("\n=== Training Summary ===");
    println!("  Iters: {} → {} ({} steps)", start_iter, total_iters, config.n_iters);
    println!("  Best loss: {:.4} @ iter {}", best_loss, best_iter);
    println!("  Final loss (last 100 avg): {:.4}", final_avg);
    println!("  NaN skips: {}", nan_skip_count);
    println!("  Time: {:.1?}", total_time);
    println!("  Speed: {:.0}ms/iter", ms_per_iter);

    // Rolling averages (2000-iter windows)
    if loss_history.len() > 2000 {
        println!("\n  Rolling averages (2000-iter windows):");
        let mut start_i = 0;
        while start_i < loss_history.len() {
            let end_i = (start_i + 2000).min(loss_history.len());
            let avg: f32 = loss_history[start_i..end_i].iter().sum::<f32>() / (end_i - start_i) as f32;
            println!("    {}-{}: avg {:.3}", start_iter + start_i, start_iter + end_i - 1, avg);
            start_i = end_i;
        }
    }

    let ceiling_str = match config.agc_ceiling {
        Some(c) => format!("{:.2}", c),
        None => format!("auto"),
    };
    println!("\n  Config: {}L, {}b, {}v, α={}, ceiling={}",
        config.n_layers, config.n_bands, vocab_size, config.alpha, ceiling_str);
    println!("  Checkpoint: {}", config.checkpoint_name);

    // Summary line to JSONL
    let summary = format!(
        r#"{{"type":"summary","best_loss":{:.4},"best_iter":{},"final_avg":{:.4},"nan_skips":{},"total_iters":{},"time_secs":{},"ms_per_iter":{:.0}}}"#,
        best_loss, best_iter, final_avg, nan_skip_count, config.n_iters, total_time.as_secs(), ms_per_iter
    );
    writeln!(log_writer, "{}", summary).ok();
    log_writer.flush().ok();
}
