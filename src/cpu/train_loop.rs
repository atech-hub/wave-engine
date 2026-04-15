//! Training loop — the main run_training() function.
//!
//! Orchestrates data loading, model init/resume, the core iteration loop,
//! and final checkpoint saving. Health monitoring and spring regulation
//! are delegated to train_health.rs.

use crate::*;
use crate::cpu::model_backward::backward;
use crate::cpu::train::{Adam, TrainConfig, DynParam, clip_grad_norm};
use crate::cpu::curriculum::CurriculumSchedule;
use crate::cpu::train_health::{self, BatchHealthData};
use crate::wave_checkpoint;
use crate::rng::Rng;

pub fn run_training(config: TrainConfig) {
    // FFT stencil
    fft_ode::validate_fft_derivative(N_BANDS);
    let stencil = fft_ode::StencilFft::new(N_BANDS);
    let gpu_kernel = fft_ode::GpuKernelFft::new(N_BANDS);
    println!("  FFT stencil precomputed (pad to {})", N_BANDS.next_power_of_two());

    // Load data (with token cache — encode once, load instantly on repeat runs)
    println!("Loading dataset from {}...", config.data_path);

    let tok_path = if config.use_bpe { Some(config.tokenizer_path.as_str()) } else { None };
    let (tokens, vocab_size) = crate::common::data_loader::load_data(&config.data_path, config.use_bpe, tok_path);
    let split = (tokens.len() as f32 * 0.9) as usize;
    let train_data = &tokens[..split];
    println!("  Train tokens: {}", train_data.len());

    // Initialize or resume
    let (mut model, start_iter, mut optimizer, mut rng);
    let mut checkpoint_chi = 0.0f32; // FWM strength from checkpoint (0.0 if fresh init)
    if let Some(ref ckpt) = config.resume_path {
        println!("Resuming from checkpoint: {ckpt}");
        let (params, ck_vocab, ck_iter, _ck_lr, ck_rng, adam_t, adam_m, adam_v, _ck_groups, _ck_flags, ck_chi) = wave_checkpoint::load_checkpoint(ckpt);
        checkpoint_chi = ck_chi;
        assert_eq!(ck_vocab, vocab_size, "Vocab size mismatch: checkpoint={ck_vocab}, data={vocab_size}");
        let mut m = init_model(vocab_size, 42, config.n_layers, config.out_proj_groups, crate::Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, crate::BLOCK_SIZE, crate::RK4_STEPS).with_moduli(config.m1, config.m2).with_tied(config.tied).with_lm_rank(config.lm_rank).with_wave_decode(config.wave_decode).with_unfreeze_phases(config.unfreeze_phases).with_learnable_ode(!config.freeze_ode).with_corrector(config.corrector.is_active() && !config.freeze_ode).with_layer_scale(config.layer_scale.is_active()).with_lr_scale(config.lr_scale.is_active()).with_pythagorean(config.pythagorean).with_rk4_weights(config.rk4_weights.is_active()).with_dyn_harmonics(config.harmonics.is_active()), config.alpha, config.beta);
        m.phase_native = config.phase_native; // Must set before count_trainable for correct param count
        let ext_count = count_trainable_ex(&m, config.tied);
        if params.len() == ext_count {
            unflatten_params_ex(&mut m, &params, config.tied);
            println!("  Loaded {} params (with ODE/corrector)", params.len());
        } else {
            // Old checkpoint without ODE/corrector — load base params, ODE starts fresh
            let base_dims = crate::Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, crate::BLOCK_SIZE, crate::RK4_STEPS).with_moduli(config.m1, config.m2).with_tied(config.tied).with_lm_rank(config.lm_rank).with_wave_decode(config.wave_decode).with_unfreeze_phases(config.unfreeze_phases).with_learnable_ode(false).with_corrector(false);
            m = init_model(vocab_size, 42, config.n_layers, config.out_proj_groups, base_dims, config.alpha, config.beta);
            unflatten_params(&mut m, &params);
            // Re-enable learnable ODE and phase-native on the loaded model
            m.learnable_ode = !config.freeze_ode;
            m.phase_native = config.phase_native;
            println!("  Loaded {} params (base — ODE/corrector start fresh)", params.len());
        }
        model = m;
        start_iter = ck_iter;
        let n_ext = count_trainable_ex(&model, config.tied);
        if adam_m.len() == n_ext {
            optimizer = Adam::from_checkpoint(config.lr, adam_t, adam_m, adam_v);
        } else {
            // Old optimizer state doesn't match new param count — fresh optimizer
            eprintln!("  Adam state size mismatch ({} vs {}), starting fresh optimizer", adam_m.len(), n_ext);
            optimizer = Adam::new(config.lr, n_ext);
        }
        rng = Rng::from_state(ck_rng);
        println!("  Resuming from iter {start_iter}");
        if config.m1.is_some() || config.m2.is_some() {
            eprintln!("  WARNING: Custom moduli with --resume will change embeddings but not trained weights");
        }
    } else {
        println!("Initializing model (seed=42)...");
        model = init_model(vocab_size, 42, config.n_layers, config.out_proj_groups, crate::Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, crate::BLOCK_SIZE, crate::RK4_STEPS).with_moduli(config.m1, config.m2).with_tied(config.tied).with_lm_rank(config.lm_rank).with_wave_decode(config.wave_decode).with_unfreeze_phases(config.unfreeze_phases).with_learnable_ode(!config.freeze_ode).with_corrector(config.corrector.is_active() && !config.freeze_ode).with_layer_scale(config.layer_scale.is_active()).with_lr_scale(config.lr_scale.is_active()).with_pythagorean(config.pythagorean).with_rk4_weights(config.rk4_weights.is_active()).with_dyn_harmonics(config.harmonics.is_active()), config.alpha, config.beta);
        model.phase_native = config.phase_native; // Must set before count_trainable
        start_iter = 0;
        let n_t = count_trainable_ex(&model, config.tied);
        optimizer = Adam::new(config.lr, n_t);
        rng = Rng::new(1337);
    }

    let n_trainable = count_trainable_ex(&model, config.tied);
    if config.tied {
        println!("  Trainable parameters: {n_trainable} (TIED — lm_head=wte, all gradient to ODE)");
    } else {
        println!("  Trainable parameters: {n_trainable} (attention frozen, FFN+LN+lm_head trainable)");
    }
    println!("  Architecture: {} parallel blocks, {} harmonic heads, {} bands ({}-dim)", config.n_layers, config.n_head, config.n_bands, config.n_bands * 2);

    // Runtime dimensions (needed by pre-flight, forward, backward)
    let mut dims = crate::Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, crate::BLOCK_SIZE, crate::RK4_STEPS).with_moduli(config.m1, config.m2).with_tied(config.tied).with_lm_rank(config.lm_rank).with_wave_decode(config.wave_decode).with_unfreeze_phases(config.unfreeze_phases).with_learnable_ode(!config.freeze_ode).with_corrector(config.corrector.is_active() && !config.freeze_ode).with_layer_scale(config.layer_scale.is_active()).with_lr_scale(config.lr_scale.is_active()).with_rk4_weights(config.rk4_weights.is_active()).with_dyn_harmonics(config.harmonics.is_active());
    dims.phase_temp = config.phase_temp;
    dims.pythagorean = config.pythagorean;
    dims.fwm_strength = config.fwm_strength;

    // Phase-native mode: ODE learns to output in embedding space, no lm_head
    model.phase_native = config.phase_native;

    // Four-wave mixing: cubic amplitude coupling inside ODE
    // CLI --fwm-strength overrides; checkpoint chi is fallback on resume
    let effective_chi = if config.fwm_strength > 0.0 { config.fwm_strength } else { checkpoint_chi };
    if effective_chi > 0.0 {
        for block in &mut model.blocks {
            block.ffn.kerr.chi = effective_chi;
        }
        dims.fwm_strength = effective_chi;
        println!("  FWM: chi={:.3} (four-wave mixing enabled)", effective_chi);
    }

    // Quantum ladder operators: creation + annihilation inside ODE
    // From Wang & Kang unified matrix/wave mechanics
    // No matrix needed — O(n) tridiagonal, strength √k, computed on the fly

    // Apply fixed values from CLI for dynamic params
    if let DynParam::Fixed(ref vals) = config.layer_scale {
        for (i, &v) in vals.iter().enumerate() {
            if i < model.layer_scale.len() { model.layer_scale[i] = v; }
        }
    }
    if let DynParam::Fixed(ref vals) = config.lr_scale {
        for (i, &v) in vals.iter().enumerate() {
            if i < model.lr_scale.len() { model.lr_scale[i] = v; }
        }
    }
    if let DynParam::Fixed(ref vals) = config.wd {
        for (i, &v) in vals.iter().enumerate() {
            if i < model.wd_scale.len() { model.wd_scale[i] = v; }
        }
    }
    if let DynParam::Fixed(ref vals) = config.agc_headroom {
        for (i, &v) in vals.iter().enumerate() {
            if i < model.agc_headroom.len() { model.agc_headroom[i] = v; }
        }
    }

    // ─── Pre-flight: out_proj_groups must divide n_embd ──────────────
    if config.out_proj_groups > 1 && dims.n_embd % config.out_proj_groups != 0 {
        let n_embd = dims.n_embd;
        let groups = config.out_proj_groups;
        let covered = (n_embd / groups) * groups;
        eprintln!("  [preflight] FATAL: n_embd={} not divisible by out_proj_groups={}",
            n_embd, groups);
        eprintln!("              out_proj covers {}/{} dims — {} orphaned dims cause crash",
            covered, n_embd, n_embd - covered);
        // Suggest valid group sizes
        let valid: Vec<usize> = (1..=n_embd).filter(|g| n_embd % g == 0 && *g <= 32).collect();
        eprintln!("              Valid groups for {}-dim: {:?}", n_embd, valid);
        std::process::exit(1);
    }

    // ─── Pre-flight diagnostics ──────────────────────────────────────
    {
        let n_embd = dims.n_embd;

        // Check 1: Embedding geometric separation
        let self_dot: f32 = model.wte[0].iter().map(|v| v * v).sum();
        let adj_dot: f32 = if model.wte.len() > 1 {
            model.wte[0].iter().zip(&model.wte[1]).map(|(a, b)| a * b).sum()
        } else { self_dot };
        let separation = self_dot - adj_dot;
        let tokens_per_band = vocab_size as f32 / dims.n_bands as f32;
        if separation < 0.01 {
            eprintln!("  [preflight] WARNING: Embedding separation {:.6} < 0.01", separation);
            eprintln!("              {} vocab at {} bands is geometrically degenerate.", vocab_size, dims.n_bands);
            eprintln!("              Reduce vocab or increase bands. Min bands: ~{}", (vocab_size as f32 * 0.008) as usize);
        } else if separation < 0.1 {
            eprintln!("  [preflight] CAUTION: Embedding separation {:.6} < 0.1 ({:.1} tokens/band)", separation, tokens_per_band);
        } else {
            println!("  [preflight] Embedding separation: {:.4} OK ({:.1} tokens/band)", separation, tokens_per_band);
        }

        // Check 2: Parameter balance
        let lm_head_params = vocab_size * n_embd;
        let model_params = n_trainable.saturating_sub(lm_head_params);
        let lm_pct = lm_head_params as f32 / n_trainable as f32 * 100.0;
        if lm_pct > 95.0 {
            eprintln!("  [preflight] WARNING: lm_head is {:.1}% of params — ODE gets <{:.1}% gradient", lm_pct, 100.0 - lm_pct);
        } else if lm_pct > 90.0 {
            eprintln!("  [preflight] CAUTION: lm_head is {:.1}% of params", lm_pct);
        } else {
            println!("  [preflight] Parameter balance: {:.1}% model, {:.1}% lm_head — OK", 100.0 - lm_pct, lm_pct);
        }

        // Check 3: ODE stability estimate
        let alpha = model.blocks[0].ffn.kerr.alpha;
        let beta = model.blocks[0].ffn.kerr.beta;
        let delta_phi = (alpha + 4.0 * beta) * 4.0; // phase shift at M=2.0
        let degrees = delta_phi * 180.0 / std::f32::consts::PI;
        if degrees > 90.0 {
            let safe = 0.5 / 5.0 / 4.0;
            eprintln!("  [preflight] WARNING: ODE phase shift {:.0}° at M=2.0 — chaotic regime", degrees);
            eprintln!("              Reduce alpha/beta. Suggested: {:.3}", safe);
        } else {
            println!("  [preflight] ODE stability: {:.0}° at M=2.0, alpha={:.4} beta={:.4} — OK", degrees, alpha, beta);
        }
    }

    // Initialize AGC with coupling-derived ceiling (or manual override)
    if let Some(ceiling) = config.agc_ceiling {
        crate::ffn_backend::init_agc_ceiling(ceiling);
        println!("  [preflight] AGC ceiling: {:.2} (manual override)", ceiling);
    } else {
        crate::ffn_backend::init_agc(config.alpha, config.beta);
        let derived = (std::f32::consts::FRAC_PI_2 / (config.alpha + 4.0 * config.beta)).sqrt();
        println!("  [preflight] AGC ceiling: {:.2} (derived from α={:.2})", derived, config.alpha);
    }

    // GPU
    let mut monitor = monitor::PipelineMonitor::new(config.use_monitor);
    if config.use_monitor { PROFILE.store(true, std::sync::atomic::Ordering::Relaxed); }
    let gpu_backend: Option<gpu_pipelines::GpuBackend> = if config.use_gpu {
        println!("  GPU: initializing...");
        let be = gpu_pipelines::GpuBackend::new();
        println!("  GPU: ready");
        Some(be)
    } else { None };
    let ffn_bufs: Option<ffn_gpu::FfnGpuBuffers> = gpu_backend.as_ref().map(|be| {
        ffn_gpu::FfnGpuBuffers::new(&be.device, config.seq_len, N_EMBD)
    });
    let ffn_full_bufs: Option<ffn_full_gpu::FfnFullBuffers> = None;

    if let Some(ref be) = gpu_backend {
        diagnose_ode_gpu_vs_cpu(be);
    }

    // Curriculum
    let curriculum = if config.use_curriculum {
        CurriculumSchedule::default_4stage(dims.n_bands)
    } else {
        CurriculumSchedule::none(dims.n_bands)
    };

    let total_iters = start_iter + config.n_iters;
    let batch_size = config.batch_size;
    let seq_len = config.seq_len;
    let lr = config.lr;

    // Tier identification for monitor tagging
    let compute_tier = if config.use_gpu { "wgpu" } else { "cpu" };

    // Framework monitor — build canonical pairs once (reused at every health interval)
    let fw_test_tokens = {
        let (ids, _vs) = crate::monitors::framework_monitor::tokenize_test_text(vocab_size);
        ids
    };
    let fw_test_strings: Vec<String> = crate::monitors::framework_monitor::FRAMEWORK_TEST_TEXT
        .chars().map(|c| c.to_string()).collect();
    let (fw_related, fw_random, fw_labels) =
        crate::monitors::framework_monitor::build_canonical_pairs(&fw_test_strings);

    // JSONL telemetry — derive from checkpoint name or use explicit --log-name
    let log_name = config.log_name.clone().unwrap_or_else(|| {
        let tier = if config.use_gpu { "wgpu" } else { "cpu" };
        // If checkpoint has a custom name, derive log name from it
        let base = &config.checkpoint_name;
        if base != "checkpoint.bin" {
            let stem = base.strip_suffix(".bin").unwrap_or(base);
            format!("training_log_{}.jsonl", stem)
        } else {
            format!("training_log_{}.jsonl", tier)
        }
    });
    let log_file = std::fs::File::create(&log_name)
        .expect(&format!("Failed to create {log_name}"));
    println!("  Telemetry: {log_name}");
    let mut log_writer = std::io::BufWriter::new(log_file);
    let mut nan_skip_count = 0usize;

    // Training summary tracking (#48)
    let mut best_loss = f32::MAX;
    let mut best_iter = 0usize;
    let mut loss_history: Vec<f32> = Vec::new();

    println!("\nTraining for {} iterations (batch={batch_size}, seq={seq_len}, lr={lr})", config.n_iters);
    if config.resume_path.is_some() { println!("  Resuming from iter {start_iter}, target iter {total_iters}"); }
    curriculum.describe(total_iters);
    println!("{:>6} {:>10} {:>10}", "Iter", "Loss", "Time");
    println!("{}", "-".repeat(30));

    let train_start = std::time::Instant::now();
    let mut recent_losses: Vec<f32> = Vec::with_capacity(100);

    let warmup_iters = 100usize;
    let min_lr_ratio = 0.1f32; // decay to 10% of base LR

    // Monitor state: dynamic parameter evolution tracking
    let mut prev_dyn_snap: Option<crate::monitors::dyn_param_monitor::DynParamSnapshot> = None;

    // Monitor state: curriculum transition tracking (#8)
    let mut curriculum_tracker = crate::monitors::curriculum_monitor::CurriculumTracker::new();

    // Monitor state: checkpoint drift tracking (#9)
    let mut checkpoint_tracker = crate::monitors::checkpoint_monitor::CheckpointTracker::new(model.blocks.len());

    for iter in start_iter..total_iters {
        let iter_start = std::time::Instant::now();

        // Cosine LR schedule with warmup — matches Candle tier
        let current_lr = if iter < warmup_iters {
            lr * (iter as f32 + 1.0) / warmup_iters as f32
        } else {
            let progress = (iter - warmup_iters) as f32 / (total_iters - warmup_iters).max(1) as f32;
            let min_lr = lr * min_lr_ratio;
            min_lr + 0.5 * (lr - min_lr) * (1.0 + (progress * std::f32::consts::PI).cos())
        };
        optimizer.lr = current_lr;

        let starts: Vec<usize> = (0..batch_size)
            .map(|_| (rng.next_u64() as usize) % (train_data.len() - seq_len - 1))
            .collect();

        // Should we measure batch distortion this iteration?
        let measure_batch_distortion = config.health_interval > 0 && iter % config.health_interval == 0;

        let t_fwd = monitor.start();
        let batch_results: Vec<(f32, Vec<f32>, BatchHealthData)> = std::thread::scope(|s| {
            let handles: Vec<_> = starts.iter().enumerate().map(|(batch_idx, &start)| {
                let model_ref = &model;
                let gpu_ref: Option<&(dyn backend::ComputeBackend + Send + Sync)> = gpu_backend.as_ref().map(|be| be as &(dyn backend::ComputeBackend + Send + Sync));
                let pp_ref: Option<(&ffn_gpu::FfnGpuBuffers, &gpu_pipelines::GpuBackend)> =
                    match (&ffn_bufs, &gpu_backend) { (Some(b), Some(g)) => Some((b, g)), _ => None };
                let fg_ref: Option<(&ffn_full_gpu::FfnFullBuffers, &gpu_pipelines::GpuBackend)> =
                    match (&ffn_full_bufs, &gpu_backend) { (Some(b), Some(g)) => Some((b, g)), _ => None };
                let input = &train_data[start..start + seq_len];
                let target = &train_data[start + 1..start + seq_len + 1];
                let st_ref = Some(&stencil);
                let gk_ref: Option<(&fft_ode::GpuKernelFft, &gpu_pipelines::GpuBackend)> =
                    gpu_backend.as_ref().map(|be| (&gpu_kernel, be));
                let agc_headrooms = if config.agc_headroom.is_active() { Some(model.agc_headroom.clone()) } else { None };
                s.spawn(move || {
                    // Per-layer AGC with per-layer headroom (when --agc-headroom is active)
                    let mut layer_agcs_storage: Option<Vec<crate::common::agc::OdeAgc>> = agc_headrooms.map(|headrooms| {
                        headrooms.iter().map(|&hr| {
                            let alpha = model_ref.blocks[0].ffn.kerr.alpha;
                            let beta = model_ref.blocks[0].ffn.kerr.beta;
                            let ceiling = (std::f32::consts::FRAC_PI_2 / (alpha + 4.0 * beta)).sqrt().max(0.5);
                            crate::common::agc::OdeAgc::with_ceiling_headroom(ceiling, hr)
                        }).collect()
                    });
                    let layer_agcs_ref = layer_agcs_storage.as_deref_mut();
                    let cache = forward_with_cache(model_ref, input, dims, gpu_ref, pp_ref, fg_ref, st_ref, gk_ref, layer_agcs_ref, None);

                    // Health monitors on first batch element only (before backward)
                    let is_health = measure_batch_distortion && batch_idx == 0;

                    let distortion = if is_health {
                        let mut layer_summaries = Vec::new();
                        for (li, bc) in cache.block_caches.iter().enumerate() {
                            if let Some(ref fc) = bc.ffn_backend_cache {
                                if let Some(summary) = crate::common::ode_distortion::measure_layer(
                                    &fc.precond, &fc.kerr_out, dims.n_bands, li,
                                ) {
                                    layer_summaries.push(summary);
                                }
                            }
                        }
                        if layer_summaries.is_empty() { None } else { Some(layer_summaries) }
                    } else {
                        None
                    };

                    // Attention head activity (#1)
                    let attn_stats = if is_health {
                        Some(crate::monitors::attn_monitor::analyze_attention(model_ref, &cache))
                    } else {
                        None
                    };

                    // Layer signal flow (#2)
                    let flow_stats = if is_health {
                        Some(crate::monitors::layer_flow_monitor::analyze_flow(&cache, dims))
                    } else {
                        None
                    };

                    // Output distribution (#5)
                    let output_stats = if is_health {
                        let targets_vec: Vec<usize> = target.to_vec();
                        Some(crate::monitors::output_monitor::analyze_output(&cache.logits, &targets_vec))
                    } else {
                        None
                    };

                    // ODE dynamics deep (#6)
                    let ode_dynamics = if is_health {
                        Some(crate::monitors::ode_dynamics_monitor::analyze_ode_dynamics(&cache, dims))
                    } else {
                        None
                    };

                    // I/Q channel monitor — observation only, no learnable params
                    let iq_analysis = if is_health {
                        Some(crate::monitors::iq_monitor::analyze_iq_batch(
                            &cache.post_ln_f, &model_ref.wte, target,
                            dims.n_bands, &model_ref.output_corrector, &vec![1.0; dims.n_bands], 10,
                        ))
                    } else {
                        None
                    };

                    let (loss, grads) = backward(model_ref, &cache, target, dims, gpu_ref, pp_ref, fg_ref);
                    // Gradient flow analysis on first batch element at health intervals
                    let grad_flow = if is_health {
                        Some(crate::monitors::gradient_monitor::analyze_gradients(&grads, dims))
                    } else {
                        None
                    };
                    (loss, flatten_grads_ex(&grads, dims.tied), BatchHealthData {
                        distortion, grad_flow, attn_stats, flow_stats, output_stats, ode_dynamics, iq_analysis,
                    })
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let fwd_bwd_elapsed = t_fwd.elapsed();
        monitor.record("fwd+bwd (batch)", t_fwd);

        let t_reduce = monitor.start();
        let mut total_loss = 0.0f32;
        let mut total_grads = vec![0.0f32; n_trainable];
        let mut batch_distortion_data: Option<Vec<crate::common::ode_distortion::LayerDistortionSummary>> = None;
        let mut batch_grad_flow: Option<Vec<crate::monitors::gradient_monitor::GradientFlowStats>> = None;
        let mut batch_attn_stats: Option<Vec<crate::monitors::attn_monitor::AttentionHeadStats>> = None;
        let mut batch_flow_stats: Option<Vec<crate::monitors::layer_flow_monitor::LayerFlowStats>> = None;
        let mut batch_output_stats: Option<crate::monitors::output_monitor::OutputDistStats> = None;
        let mut batch_ode_dynamics: Option<Vec<crate::monitors::ode_dynamics_monitor::OdeDynamicsStats>> = None;
        let mut batch_iq_analysis: Option<crate::monitors::iq_monitor::IqAnalysis> = None;
        for (loss, fg, health) in &batch_results {
            total_loss += loss;
            for (a, g) in total_grads.iter_mut().zip(fg.iter()) { *a += g; }
            if health.distortion.is_some() && batch_distortion_data.is_none() {
                batch_distortion_data = health.distortion.clone();
            }
            if batch_grad_flow.is_none() {
                if let Some(ref gf) = health.grad_flow {
                    batch_grad_flow = Some(gf.iter().map(|s| crate::monitors::gradient_monitor::GradientFlowStats {
                        layer: s.layer,
                        ln_grad_norm: s.ln_grad_norm,
                        maestro_in_grad_norm: s.maestro_in_grad_norm,
                        ode_grad_norm: s.ode_grad_norm,
                        maestro_out_grad_norm: s.maestro_out_grad_norm,
                        out_proj_grad_norm: s.out_proj_grad_norm,
                        alpha_grad: s.alpha_grad,
                        beta_grad: s.beta_grad,
                        corrector_grad_norm: s.corrector_grad_norm,
                        rk4_grad_norm: s.rk4_grad_norm,
                    }).collect());
                }
            }
            // Extract batch 2 monitor data from first element that has it
            if batch_attn_stats.is_none() {
                if let Some(ref stats) = health.attn_stats {
                    batch_attn_stats = Some(stats.iter().map(|s| crate::monitors::attn_monitor::AttentionHeadStats {
                        layer: s.layer, head: s.head, harmonic: s.harmonic,
                        entropy: s.entropy, max_weight: s.max_weight,
                        top_position: s.top_position, self_attn_frac: s.self_attn_frac,
                    }).collect());
                }
            }
            if batch_flow_stats.is_none() {
                if let Some(ref stats) = health.flow_stats {
                    batch_flow_stats = Some(stats.iter().map(|s| crate::monitors::layer_flow_monitor::LayerFlowStats {
                        layer: s.layer, input_norm: s.input_norm,
                        attn_output_norm: s.attn_output_norm, ffn_output_norm: s.ffn_output_norm,
                        output_norm: s.output_norm, attn_ratio: s.attn_ratio,
                        ffn_ratio: s.ffn_ratio, residual_ratio: s.residual_ratio,
                        cosine_in_out: s.cosine_in_out,
                        band_amp_min: s.band_amp_min, band_amp_max: s.band_amp_max,
                        band_amp_mean: s.band_amp_mean, band_amp_std: s.band_amp_std,
                    }).collect());
                }
            }
            if batch_output_stats.is_none() {
                if let Some(ref stats) = health.output_stats {
                    batch_output_stats = Some(crate::monitors::output_monitor::OutputDistStats {
                        avg_entropy: stats.avg_entropy, avg_margin: stats.avg_margin,
                        avg_correct_rank: stats.avg_correct_rank, worst_margin: stats.worst_margin,
                        worst_prompt_pos: stats.worst_prompt_pos, mode_collapse: stats.mode_collapse,
                    });
                }
            }
            if batch_ode_dynamics.is_none() {
                if let Some(ref stats) = health.ode_dynamics {
                    batch_ode_dynamics = Some(stats.iter().map(|s| crate::monitors::ode_dynamics_monitor::OdeDynamicsStats {
                        layer: s.layer, phase_velocity: s.phase_velocity,
                        energy_in: s.energy_in, energy_out: s.energy_out,
                        energy_ratio: s.energy_ratio, band_energy_std: s.band_energy_std,
                        damping_effective: s.damping_effective,
                    }).collect());
                }
            }
            if batch_iq_analysis.is_none() {
                if let Some(ref iq) = health.iq_analysis {
                    batch_iq_analysis = Some(crate::monitors::iq_monitor::IqAnalysis {
                        i_discrim: iq.i_discrim, q_discrim: iq.q_discrim,
                        iq_ratio: iq.iq_ratio, phase_mean: iq.phase_mean,
                        phase_std: iq.phase_std, i_correct_rank: iq.i_correct_rank,
                        q_correct_rank: iq.q_correct_rank,
                    });
                }
            }
            // I/Q + corrector logging (after loop — no break, no contamination)
            if let Some(ref iq) = health.iq_analysis {
                eprintln!("  [I/Q] I_disc={:.3} Q_disc={:.3} IQ_ratio={:.3} phase_std={:.3} I_rank={} Q_rank={}",
                    iq.i_discrim, iq.q_discrim, iq.iq_ratio, iq.phase_std, iq.i_correct_rank, iq.q_correct_rank);
                if !model.output_corrector.is_empty() {
                    let n = model.output_corrector.len() as f32;
                    let mean = model.output_corrector.iter().sum::<f32>() / n;
                    let std = (model.output_corrector.iter().map(|&a| (a - mean) * (a - mean)).sum::<f32>() / n).sqrt();
                    let max_abs = model.output_corrector.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
                    eprintln!("  [corrector] mean={:.4} std={:.4} max_abs={:.4} (radians)", mean, std, max_abs);
                }
            }
        }
        total_loss /= batch_size as f32;
        for g in total_grads.iter_mut() { *g /= batch_size as f32; }
        monitor.record("reduce", t_reduce);

        // FWM stability scan at iter 0 (one-shot) — uses real activations
        if iter == 0 && config.fwm_strength > 0.0 {
            let scan_start = (crate::rng::Rng::new(42).next_u64() as usize) % (train_data.len() - seq_len - 1);
            let scan_tokens = &train_data[scan_start..scan_start + seq_len];
            let scan_cache = crate::cpu::forward::forward_with_cache(
                &model, scan_tokens, dims, None, None, None, Some(&stencil), None, None, None,
            );
            let scan_precond = if let Some(ref fc) = scan_cache.block_caches[0].ffn_backend_cache {
                fc.precond[0].clone()
            } else {
                scan_cache.block_caches[0].ffn_precond[0].clone()
            };
            let scan = crate::monitors::fwm_monitor::fwm_stability_scan(
                &scan_precond,
                &model.blocks[0].ffn.kerr, config.n_bands,
                &[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            );
            eprintln!("  [FWM stability scan (real activations)]");
            for (chi, diag, stable) in &scan {
                eprintln!("    chi={:.2}: fwm_ratio={:.4} rk4_ratio={:.2} triple={:.2} max_amp={:.3} {}",
                    chi, diag.fwm_ratio, diag.rk4_step_ratio, diag.triple_ratio, diag.max_band_amp,
                    if *stable { "STABLE" } else { "UNSTABLE" });
            }
        }

        // ODE decomposition monitor at health intervals — damping/phase/FWM ratios
        if config.health_interval > 0 && iter % config.health_interval == 0 {
            // Grab real precond from the first batch's forward cache
            let cache_ref = &batch_results[0].2; // BatchHealthData from first batch
            // We need the actual hidden state — use the training data directly
            let sample_start = (crate::rng::Rng::new(iter as u64).next_u64() as usize) % (train_data.len() - seq_len - 1);
            let sample_tokens = &train_data[sample_start..sample_start + seq_len];
            // Run a quick forward to get block caches with real activations
            let sample_cache = crate::cpu::forward::forward_with_cache(
                &model, sample_tokens, dims, None, None, None, Some(&stencil), None, None, None,
            );
            use std::io::Write;
            let mut fwm_layers = Vec::new();
            for (layer_idx, block) in model.blocks.iter().enumerate() {
                // Use actual ODE precond from FFN backend cache (not normed_ffn!)
                let precond_data = if let Some(ref fc) = sample_cache.block_caches[layer_idx].ffn_backend_cache {
                    fc.precond[0].clone()
                } else {
                    // Fallback to legacy cache
                    sample_cache.block_caches[layer_idx].ffn_precond[0].clone()
                };
                let diag = crate::monitors::fwm_monitor::measure_fwm(
                    &precond_data, &block.ffn.kerr, config.n_bands, layer_idx,
                );
                fwm_layers.push(format!(
                    r#"{{"layer":{},"fwm_ratio":{:.4},"fwm_vs_phase":{:.4},"damping_ratio":{:.4},"phase_ratio":{:.4},"triple_ratio":{:.2},"max_amp":{:.4},"mean_amp":{:.4},"rk4_ratio":{:.3},"flux_max":{:.6},"top_bands":[{},{},{}]}}"#,
                    layer_idx, diag.fwm_ratio, diag.fwm_vs_phase,
                    diag.damping_ratio, diag.phase_ratio,
                    diag.triple_ratio,
                    diag.max_band_amp, diag.mean_band_amp, diag.rk4_step_ratio,
                    diag.flux_max, diag.top_3_bands[0], diag.top_3_bands[1], diag.top_3_bands[2]
                ));
                eprintln!("  [ODE L{}] damp={:.3} phase={:.3} fwm={:.4} max_amp={:.3}",
                    layer_idx, diag.damping_ratio, diag.phase_ratio, diag.fwm_ratio,
                    diag.max_band_amp);
                // Sanity check: fwm_ratio should be in [0, 1]
                if diag.fwm_ratio < 0.0 || diag.fwm_ratio > 1.0 || diag.fwm_ratio.is_nan() {
                    eprintln!("  WARNING: L{} fwm_ratio={:.4} out of range [0,1] — activation cache may not reflect {} tier",
                        layer_idx, diag.fwm_ratio, compute_tier);
                }
            }
            let _ = writeln!(log_writer, r#"{{"iter":{},"type":"ode_decomposition","tier":"{}","chi":{},"layers":[{}]}}"#,
                iter, compute_tier, config.fwm_strength, fwm_layers.join(","));
            let _ = log_writer.flush();

            // Backward decomposition — per-layer gradient flow through physics terms
            let mut bwd_stats = Vec::new();
            for (layer_idx, block) in model.blocks.iter().enumerate() {
                let precond_data = if let Some(ref fc) = sample_cache.block_caches[layer_idx].ffn_backend_cache {
                    fc.precond[0].clone()
                } else {
                    sample_cache.block_caches[layer_idx].ffn_precond[0].clone()
                };
                let bwd = crate::monitors::ode_backward_monitor::measure_layer_backward(
                    &precond_data, &block.ffn.kerr, layer_idx,
                );
                eprintln!("  [BWD L{}] damp={:.3} phase={:.3} fwm={:.3} d_chi={:.6}",
                    layer_idx, bwd.damping_frac, bwd.phase_frac, bwd.fwm_frac, bwd.d_chi_norm);
                bwd_stats.push(bwd);
            }
            let bwd_json = crate::monitors::ode_backward_monitor::to_json(&bwd_stats, compute_tier);
            let _ = writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, bwd_json);
            let _ = log_writer.flush();

            // Framework monitor — harmonic coherence + band census + phase clustering
            // Uses training data sample positions (works for any task/vocab)
            {
                let all_layer_hidden: Vec<Vec<Vec<f32>>> = sample_cache.block_caches.iter()
                    .map(|bc| bc.input.clone()).collect();
                let t = sample_cache.post_ln_f.len();
                // Adjacent positions as "related", distant as "random"
                let mut rel = Vec::new();
                let mut labs = Vec::new();
                for i in (0..t.min(10)).step_by(2) {
                    if i + 1 < t {
                        rel.push((vec![i], vec![i + 1]));
                        labs.push(format!("pos{}/{}", i, i + 1));
                    }
                }
                let mut rand = Vec::new();
                for i in 0..t.min(5) {
                    let j = (i + t / 2) % t;
                    if j != i { rand.push((vec![i], vec![j])); }
                }

                let fw_report = crate::monitors::framework_monitor::run_framework_scan(
                    &all_layer_hidden, &sample_cache.post_ln_f,
                    config.n_bands, &rel, &rand, &labs,
                );
                let fw_json = crate::monitors::framework_monitor::to_json(&fw_report);
                let _ = writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, fw_json);
                let _ = log_writer.flush();
                if let Some(final_stats) = fw_report.per_layer.last() {
                    eprintln!("  [FW] disc={:.1}x clustering={:.3} peak_layer={}",
                        final_stats.discrimination_ratio, final_stats.phase_clustering,
                        fw_report.dominant_depth_peak);
                }
            }
        }

        // NaN skip with post-mortem: discard batch, diagnose cause
        if total_loss.is_nan() || total_loss.is_infinite() {
            nan_skip_count += 1;
            // Post-mortem: which batch elements caused it?
            let nan_elements: Vec<usize> = batch_results.iter().enumerate()
                .filter(|(_, (loss, _, _))| loss.is_nan() || loss.is_infinite())
                .map(|(i, _)| i).collect();
            let has_nan_grad = total_grads.iter().any(|g| g.is_nan());
            if iter % 100 == 0 || iter < 10 {
                eprintln!("  [NaN skip iter {}] {}/{} bad elements, nan_grad={}, total_skips={}",
                    iter, nan_elements.len(), batch_size, has_nan_grad, nan_skip_count);
            }
            continue;
        }

        // Track for summary (#48)
        loss_history.push(total_loss);
        if total_loss < best_loss {
            best_loss = total_loss;
            best_iter = iter;
        }

        // Curriculum transition tracking (#8) — every iteration
        let curriculum_event = if config.use_curriculum {
            let active = curriculum.active_bands(iter, total_iters);
            curriculum_tracker.update(iter, total_loss, active)
        } else {
            None
        };

        let t_optim = monitor.start();
        clip_grad_norm(&mut total_grads, 1.0);

        // Head LR floor: boost head gradients when effective LR drops too low.
        // The lm_head gradient starves during sustained training (71% -> 9%).
        // Only activate after warmup — cosine peaks within first 500 iters.
        let iters_into_run = iter - start_iter;
        if config.head_lr_floor > 0.0 && iters_into_run > 500 && current_lr < config.head_lr_floor {
            let boost = config.head_lr_floor / current_lr.max(1e-8);
            let lm_start = n_trainable.saturating_sub(
                if model.learnable_ode { 0 } else { 0 } + // ODE params are before lm_head
                if model.lm_rank > 0 { model.lm_rank * model.ln_f.weight.len() + model.vocab_size * model.lm_rank }
                else if model.wd_state.is_some() { crate::common::wave_decode::param_count(model.wd_state.as_ref().unwrap()) }
                else { model.vocab_size * model.ln_f.weight.len() }
            );
            for i in lm_start..total_grads.len() {
                total_grads[i] *= boost;
            }
        }

        // LR scale: per-group gradient scaling before optimizer step.
        train_health::apply_lr_scale(&mut model, &config, dims, &mut total_grads, current_lr);

        let mut params = flatten_params_ex(&model, config.tied);
        if config.wd.is_active() {
            // Per-group weight decay: apply WD manually, then Adam without WD
            train_health::apply_per_group_wd(&model, &config, dims, &mut params, current_lr);
            optimizer.step_wd(&mut params, &total_grads, 0.0); // Adam without WD (already applied)
        } else {
            optimizer.step(&mut params, &total_grads);
        }
        unflatten_params_ex(&mut model, &params, config.tied);

        // Spring regulation for all dynamic parameters
        train_health::apply_springs(&mut model, &config, current_lr);

        let optim_elapsed = t_optim.elapsed();
        monitor.record("optimizer", t_optim);

        let grad_norm: f32 = total_grads.iter().map(|g| g * g).sum::<f32>().sqrt();

        // First-10 health check: per-component gradient norms + weight growth
        train_health::first10_health_check(iter, &model, &total_grads, total_loss, n_trainable, grad_norm);

        // JSONL telemetry — every iteration, with gradient diagnostics every 100
        train_health::write_jsonl_telemetry(
            &mut log_writer, iter, total_loss, current_lr,
            iter_start.elapsed().as_millis(), nan_skip_count,
            &model, &config, &total_grads, grad_norm, n_trainable, dims,
        );

        // Rolling avg100 for honest loss reporting
        recent_losses.push(total_loss);
        if recent_losses.len() > 100 { recent_losses.remove(0); }

        if iter % 10 == 0 || iter == total_iters - 1 {
            let avg100 = if recent_losses.len() >= 10 {
                recent_losses.iter().sum::<f32>() / recent_losses.len() as f32
            } else { total_loss };
            println!("{:>6} {:>10.4} {:>10.1?}  lr={:.6}  gnorm={:.2}  avg100={:.4}", iter, total_loss, iter_start.elapsed(), current_lr, grad_norm, avg100);
            if monitor.enabled() {
                monitor.report(if iter == 0 { 1 } else { 50.min(iter) });
                monitor.reset();
            }
        }

        // Encoding health sample (opt-in via --health-interval)
        if config.health_interval > 0 && iter % config.health_interval == 0 {
            train_health::write_health_monitors(
                &mut log_writer, iter, iters_into_run, &model, &config, dims, &stencil,
                &batch_distortion_data, &batch_grad_flow, &batch_attn_stats,
                &batch_flow_stats, &batch_output_stats, &batch_ode_dynamics,
                &batch_iq_analysis,
                &mut prev_dyn_snap, batch_size, seq_len,
                iter_start.elapsed().as_secs_f32(),
                fwd_bwd_elapsed.as_secs_f32(),
                optim_elapsed.as_secs_f32(),
            );
        }

        // Curriculum transition (#8) — emits when transition detected (every iter tracking)
        if let Some(ref event) = curriculum_event {
            train_health::write_curriculum_event(&mut log_writer, iter, event);
        }

        // Periodic checkpoint: save every 500 iters (all tiers, always untied format)
        // Uses flatten_params_ex to include ODE params + corrector plate (matches optimizer state)
        if (iter + 1) % 500 == 0 {
            let save_p = flatten_params_ex(&model, false);

            // Checkpoint drift (#9) — measure before saving
            if let Some(drift) = checkpoint_tracker.measure(&save_p) {
                train_health::write_checkpoint_drift(&mut log_writer, iter, &drift);
            }

            let path = format!("checkpoint_iter{}.bin", iter + 1);
            let groups = model.blocks[0].ffn.out_proj.n_groups();
            if config.tied {
                let dummy_opt = Adam::new(config.lr, save_p.len());
                wave_checkpoint::save_checkpoint(&save_p, vocab_size, model.blocks.len(), groups, iter + 1, lr, &dummy_opt, rng.state(), &path, dims);
            } else {
                wave_checkpoint::save_checkpoint(&save_p, vocab_size, model.blocks.len(), groups, iter + 1, lr, &optimizer, rng.state(), &path, dims);
            }
            println!("  Checkpoint: {path}");
        }
    }

    println!("\nTraining complete. Total time: {:.1?}", train_start.elapsed());

    // Training summary (#48)
    train_health::write_training_summary(
        &mut log_writer, &config, start_iter, total_iters,
        best_loss, best_iter, &loss_history, nan_skip_count,
        train_start, vocab_size,
    );

    // Final checkpoint (always saves full untied params + ODE/corrector for compatibility)
    let save_params = flatten_params_ex(&model, false);
    let groups = model.blocks[0].ffn.out_proj.n_groups();
    if config.tied {
        // Optimizer is tied-size; checkpoint needs untied-size. Save dummy optimizer.
        let dummy_opt = Adam::new(config.lr, save_params.len());
        wave_checkpoint::save_checkpoint(&save_params, vocab_size, model.blocks.len(), groups, total_iters, lr, &dummy_opt, rng.state(), &config.checkpoint_name, dims);
    } else {
        wave_checkpoint::save_checkpoint(&save_params, vocab_size, model.blocks.len(), groups, total_iters, lr, &optimizer, rng.state(), &config.checkpoint_name, dims);
    }
    println!("Checkpoint saved to {}", config.checkpoint_name);

    // Auto-trigger galaxy map scan on final checkpoint
    {
        let galaxy_dir = std::path::PathBuf::from(
            config.checkpoint_name.replace(".bin", "_galaxy")
        );
        // Use first N tokens of training data as test corpus
        let scan_len = train_data.len().min(200);
        let scan_tokens = &train_data[..scan_len];
        // Forward pass on CPU for phase extraction
        let scan_cache = crate::cpu::forward::forward_with_cache(
            &model, scan_tokens, dims, None, None, None, Some(&stencil), None, None, None,
        );
        let all_layer_hidden: Vec<Vec<Vec<f32>>> = scan_cache.block_caches.iter()
            .map(|bc| bc.input.clone()).collect();
        // Per-layer AGC ceilings from learned alpha/beta
        let per_layer_ceilings: Vec<f32> = model.blocks.iter()
            .map(|b| (std::f32::consts::FRAC_PI_2 / (b.ffn.kerr.alpha + 4.0 * b.ffn.kerr.beta)).sqrt().max(0.5))
            .collect();
        let m1 = config.m1.unwrap_or(5);
        let m2 = config.m2.unwrap_or(7);
        match crate::common::galaxy_scan::run_and_write_full_scan(
            &all_layer_hidden, &scan_cache.post_ln_f,
            config.n_bands, &per_layer_ceilings, m1, m2, &galaxy_dir,
        ) {
            Ok(scan) => {
                eprintln!("Galaxy map: {}", galaxy_dir.display());
                crate::common::galaxy_scan::print_summary(&scan);
            }
            Err(e) => {
                eprintln!("Warning: galaxy scan failed: {}. Training output is unaffected.", e);
            }
        }
    }
}
