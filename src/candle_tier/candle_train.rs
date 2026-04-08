//! Training loop for candle backend.

#[cfg(feature = "candle-backend")]
pub mod train {
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::VarMap;
    use std::time::Instant;

    use crate::candle_tier::candle_model::model::CandleWaveModel;
    use crate::candle_tier::candle_attention::attention::harmonic_backward;
    use crate::candle_tier::candle_checkpoint::checkpoint::{load_wchk_params_into_varmap, extract_wchk_params};
    use crate::candle_tier::candle_monitors::monitors::{
        CandleMonitorData, CandleOutputDist, CandleGradientFlow,
        compute_output_dist, compute_gradient_flow,
        output_dist_json, gradient_flow_json,
    };

    // ─── Training loop ───

    pub fn train_candle(
        data_path: &str, n_iters: usize,
        n_bands: usize, n_head: usize, n_layers: usize,
        maestro_dim: usize, _rk4_steps: usize, out_proj_groups: usize,
        debug_nan: bool, alpha: f32, beta: f32, chi: f32, phase_native: bool,
    ) -> Result<()> {
        // Runtime config — lowercase variables used throughout
        let n_embd = n_bands * 2;
        let block_size = 256usize; // positional table size

        println!("Candle backend — wave-engine\n");
        println!("  Config: {n_bands} bands, {n_head} heads, {n_layers} layers, {maestro_dim} maestro, {out_proj_groups} out_proj groups");

        // Device
        let device = Device::cuda_if_available(0)?;
        println!("  Device: {:?}", device);

        // Load data + tokenize (with token cache — 3min encode → instant reload)
        let use_bpe = std::env::args().any(|a| a == "--bpe");
        let tokenizer_path = std::env::args().skip_while(|a| a != "--tokenizer").nth(1)
            .unwrap_or("data/tokenizer.json".to_string());

        let tok_path_opt = if use_bpe { Some(tokenizer_path.as_str()) } else { None };
        let (tokens, vocab_size) = crate::common::data_loader::load_data(data_path, use_bpe, tok_path_opt);
        let split = (tokens.len() as f32 * 0.9) as usize;
        let train_data = &tokens[..split];
        println!("  Train tokens: {}", train_data.len());

        // Parse dynamic param flags early (needed for model construction + optimizer config)
        let use_rk4_dyn = std::env::args().any(|a| a == "--rk4-weights") &&
            std::env::args().skip_while(|a| a != "--rk4-weights").nth(1).map_or(false, |s| s == "dyn");
        let use_layer_scale_dyn = std::env::args().any(|a| a == "--layer-scale") &&
            std::env::args().skip_while(|a| a != "--layer-scale").nth(1).map_or(false, |s| s == "dyn");
        let use_wd_dyn = std::env::args().any(|a| a == "--wd") &&
            std::env::args().skip_while(|a| a != "--wd").nth(1).map_or(false, |s| s == "dyn");
        let use_agc_headroom_dyn = std::env::args().any(|a| a == "--agc-headroom") &&
            std::env::args().skip_while(|a| a != "--agc-headroom").nth(1).map_or(false, |s| s == "dyn");
        let use_cuda_kernel = std::env::args().any(|a| a == "--cuda-kernel");
        let use_custom_op = use_cuda_kernel || std::env::args().any(|a| a == "--custom-op");

        // Model
        let mut varmap = VarMap::new();
        let mut model = CandleWaveModel::new(&varmap, vocab_size, &device,
            n_bands, n_head, n_layers, maestro_dim, _rk4_steps, out_proj_groups, alpha, beta,
            chi, phase_native)?;
        model.debug_nan = debug_nan;
        model.use_custom_op = use_custom_op;
        model.use_cuda_kernel = use_cuda_kernel;
        if use_custom_op {
            model.ode_param_grads = Some(crate::candle_tier::custom_ode::custom_ode::create_param_grad_storage(n_layers));
            if use_cuda_kernel {
                println!("  CUDA kernel: fused AGC+RK4 on GPU, backward via CPU");
            } else {
                println!("  CustomOp: ODE backward via CPU (no autograd graph)");
            }
        }
        // Wire dynamic params
        if use_rk4_dyn {
            for i in 0..model.blocks.len() {
                let key = format!("block.{i}.ode");
                model.blocks[i].gpu_ode_params.set_rk4_learnable(&varmap, &key, &device)?;
            }
        }
        if use_layer_scale_dyn {
            for i in 0..model.blocks.len() {
                let key = format!("block.{i}.layer_scale");
                // Init at 1.0 (no scaling = default behavior)
                let _t = varmap.get((1,), &key, candle_nn::Init::Const(1.0), DType::F32, &device)?;
                // Re-get for the model to use in forward
                let t = varmap.get((1,), &key, candle_nn::Init::Const(1.0), DType::F32, &device)?;
                model.blocks[i].layer_scale = Some(t);
            }
        }
        if phase_native {
            println!("  Phase-native: dot product against embeddings (zero decoder params)");
        }
        if debug_nan { println!("  [debug-nan] Per-layer NaN detection ENABLED (~6x slower)"); }
        let n_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
        println!("  Trainable params: {n_params}");
        println!("  Architecture: {n_layers} layers, {n_head} heads, {n_bands} bands");

        // Resume from checkpoint if --resume flag
        let resume_path: Option<String> = std::env::args().skip_while(|a| a != "--resume").nth(1);
        let mut start_iter = 0usize;
        if let Some(ref ckpt) = resume_path {
            println!("  Resuming from: {ckpt}");
            if ckpt.ends_with(".safetensors") {
                // Native candle checkpoint
                varmap.load(ckpt)?;
            } else if ckpt.ends_with(".bin") {
                // CPU/wgpu WCHK checkpoint — load and populate VarMap
                let (params, _ck_vocab, ck_iter, _lr, _rng, _at, _am, _av, _groups, ck_flags) =
                    crate::wave_checkpoint::load_checkpoint(ckpt);
                start_iter = ck_iter;
                // Map flat params into VarMap keys (reverse of extract_wchk_params)
                let has_ode = ck_flags & 1 != 0 || ck_flags == 0; // v2 checkpoints have no flags
                let has_ls = ck_flags & 2 != 0;
                let has_rk4 = ck_flags & 4 != 0;
                load_wchk_params_into_varmap(&varmap, &params, n_layers, n_embd, maestro_dim,
                    vocab_size, out_proj_groups, n_bands, has_ode, has_ls, has_rk4, phase_native, &device)?;
                println!("  Loaded {} WCHK params (flags=0x{:02x}) into candle VarMap", params.len(), ck_flags);
            } else {
                eprintln!("  WARNING: unknown checkpoint format: {ckpt}");
            }
            // Read iter from .meta file (try exact match, then strip loss suffix)
            let meta_path = ckpt.replace(".safetensors", ".meta");
            let meta_content = std::fs::read_to_string(&meta_path)
                .or_else(|_| {
                    // Try stripping _lossN.NN from filename
                    let stripped = meta_path.split("_loss").next().unwrap_or(&meta_path);
                    std::fs::read_to_string(format!("{stripped}.meta"))
                })
                .or_else(|_| std::fs::read_to_string("candle_checkpoint_latest.meta"));
            if let Ok(meta) = meta_content {
                for line in meta.lines() {
                    if let Some(v) = line.strip_prefix("iter=") {
                        start_iter = v.parse().unwrap_or(0);
                    }
                }
            }
            println!("  Resumed at iter {start_iter}");
        }

        // CLI flag parsing for Candle path
        fn parse_flag_c<T: std::str::FromStr>(name: &str, default: T) -> T {
            std::env::args().skip_while(|a| a != name).nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let batch_size: usize = parse_flag_c("--batch", 4);
        let seq_len: usize = parse_flag_c("--seq", 256);
        let lr: f64 = parse_flag_c("--lr", if n_bands > 256 { 1e-4 } else { 3e-4 });
        let spring_k: f64 = parse_flag_c("--spring", 0.1);
        let gpu_duty: usize = parse_flag_c("--gpu-duty", 100).clamp(1, 100);
        if gpu_duty < 100 {
            println!("  GPU duty cycle: {}% (sleep between iterations to reduce temperature)", gpu_duty);
        }
        let use_rk4_dyn = std::env::args().any(|a| a == "--rk4-weights") &&
            std::env::args().skip_while(|a| a != "--rk4-weights").nth(1).map_or(false, |s| s == "dyn");
        let use_harmonics_dyn = std::env::args().any(|a| a == "--harmonics") &&
            std::env::args().skip_while(|a| a != "--harmonics").nth(1).map_or(false, |s| s == "dyn");
        let use_wd_dyn = std::env::args().any(|a| a == "--wd") &&
            std::env::args().skip_while(|a| a != "--wd").nth(1).map_or(false, |s| s == "dyn");
        let use_layer_scale_dyn = std::env::args().any(|a| a == "--layer-scale") &&
            std::env::args().skip_while(|a| a != "--layer-scale").nth(1).map_or(false, |s| s == "dyn");
        // Wire harmonic_dyn flag on blocks
        if use_harmonics_dyn {
            for block in &mut model.blocks {
                block.harmonic_dyn = true;
            }
        }
        if spring_k > 0.0 {
            let mut dyn_flags = Vec::new();
            if use_rk4_dyn { dyn_flags.push("rk4-weights"); }
            if use_harmonics_dyn { dyn_flags.push("harmonics"); }
            if use_wd_dyn { dyn_flags.push("wd"); }
            if use_layer_scale_dyn { dyn_flags.push("layer-scale"); }
            if use_agc_headroom_dyn { dyn_flags.push("agc-headroom"); }
            if !dyn_flags.is_empty() {
                println!("  Dynamic params: {} (spring k={:.2})", dyn_flags.join(", "), spring_k);
            }
        }

        // Optimizer — when WD is dynamic, disable built-in WD (we apply per-group manually)
        use candle_nn::Optimizer;
        let wd_builtin = if use_wd_dyn { 0.0 } else { 0.01 };
        let mut optimizer = candle_nn::AdamW::new(
            varmap.all_vars(),
            candle_nn::ParamsAdamW { lr, weight_decay: wd_builtin, ..Default::default() },
        )?;
        // Per-group WD scale: [n_layers + 1] (layers + lm_head), init at 1.0 (uniform)
        let mut wd_scale: Vec<f32> = vec![1.0; n_layers + 1];
        // Per-layer AGC headroom: init at 3.0 (3-sigma default)
        let mut agc_headroom: Vec<f32> = vec![3.0; n_layers];
        // Per-layer AGC instances when --agc-headroom dyn (stored on model)
        if use_agc_headroom_dyn {
            let ceiling = (std::f32::consts::FRAC_PI_2 / (alpha + 4.0 * beta)).sqrt().max(0.5);
            model.layer_agcs = Some((0..n_layers).map(|_| crate::common::agc::OdeAgc::with_ceiling_headroom(ceiling, 3.0)).collect());
        }
        let mut rng = crate::rng::Rng::new(1337);

        // Curriculum: soft-mask inactive bands (0.01 scale, not zero)
        let use_curriculum = !std::env::args().any(|a| a == "--no-curriculum");
        let curriculum = if use_curriculum {
            crate::train::CurriculumSchedule::default_4stage(n_bands)
        } else {
            crate::train::CurriculumSchedule::none(n_bands)
        };

        // ─── Pre-flight diagnostics (must match CPU tier) ───────────
        {
            // Check 1: Embedding separation (rebuild table to check)
            let pf_wte = crate::wave_embed::build_harmonic_table(vocab_size, n_bands);
            let self_dot: f32 = pf_wte[0].iter().map(|v| v * v).sum();
            let adj_dot: f32 = if pf_wte.len() > 1 {
                pf_wte[0].iter().zip(&pf_wte[1]).map(|(a, b)| a * b).sum()
            } else { self_dot };
            let separation = self_dot - adj_dot;
            if separation < 0.01 {
                eprintln!("  [preflight] WARNING: Embedding separation {:.6} — geometrically degenerate", separation);
            } else {
                println!("  [preflight] Embedding separation: {:.4} OK", separation);
            }

            // Check 2: Parameter balance
            let lm_head_params = vocab_size * n_embd;
            let total_params = n_params;
            let lm_pct = lm_head_params as f32 / total_params.max(1) as f32 * 100.0;
            if lm_pct > 95.0 {
                eprintln!("  [preflight] WARNING: lm_head is {:.1}% of params — ODE gets <{:.1}% gradient", lm_pct, 100.0 - lm_pct);
            } else {
                println!("  [preflight] Parameter balance: {:.1}% model, {:.1}% lm_head — OK", 100.0 - lm_pct, lm_pct);
            }

            // Check 3: ODE stability
            let alpha = if n_bands <= 128 { 0.01f32 } else { 0.1 };
            let degrees = (alpha + 4.0 * alpha) * 4.0 * 180.0 / std::f32::consts::PI;
            if degrees > 90.0 {
                eprintln!("  [preflight] WARNING: ODE phase shift {:.0}° at M=2.0", degrees);
            } else {
                println!("  [preflight] ODE stability: {:.0}° at M=2.0, alpha={:.4} — OK", degrees, alpha);
            }
        }

        // Initialize AGC with coupling-derived ceiling (matches CPU tier)
        crate::ffn_backend::init_agc(alpha, beta);
        let derived_ceiling = (std::f32::consts::FRAC_PI_2 / (alpha + 4.0 * beta)).sqrt();
        println!("  [preflight] AGC ceiling: {:.2} (derived from α={:.2})", derived_ceiling, alpha);

        let total_iters = start_iter + n_iters;
        println!("\nTraining for {n_iters} iters (batch={batch_size}, seq={seq_len}, lr={lr})");
        if start_iter > 0 { println!("  Resuming from iter {start_iter}, target {total_iters}"); }
        curriculum.describe(total_iters);
        println!("{:>6} {:>10} {:>10}", "Iter", "Loss", "Time");
        println!("{}", "-".repeat(35));

        // JSONL telemetry — use --log-name if provided, else derive from checkpoint name
        let log_name: String = std::env::args().skip_while(|a| a != "--log-name").nth(1)
            .unwrap_or_else(|| {
                let ckpt = std::env::args().skip_while(|a| a != "--checkpoint-name").nth(1)
                    .unwrap_or_else(|| "checkpoint.bin".to_string());
                let stem = ckpt.strip_suffix(".bin").unwrap_or(&ckpt);
                format!("training_log_{}.jsonl", stem)
            });
        let log_file = std::fs::File::create(&log_name).ok();
        let mut log_writer = log_file.map(|f| std::io::BufWriter::new(f));
        println!("  Telemetry: {log_name}");
        let mut nan_skip_count = 0usize;

        // Cosine LR schedule with warmup
        let warmup_iters = 100usize;
        let min_lr_ratio = 0.1;
        let cosine_lr = |iter: usize| -> f64 {
            if iter < warmup_iters {
                lr * (iter + 1) as f64 / warmup_iters as f64
            } else {
                let progress = (iter - warmup_iters) as f64 / (total_iters - warmup_iters).max(1) as f64;
                let decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                lr * (min_lr_ratio + (1.0 - min_lr_ratio) * decay)
            }
        };

        let train_start = Instant::now();
        let health_interval: usize = parse_flag_c("--health-interval", 0);

        for iter in start_iter..total_iters {
            let band_masks = curriculum.band_masks(iter, total_iters, n_bands);
            let iter_start = Instant::now();
            let mut total_loss = 0.0f32;
            let measure_monitors = health_interval > 0 && iter % health_interval == 0;

            let current_lr = cosine_lr(iter);
            optimizer.set_learning_rate(current_lr);

            // Monitor data: captured on last batch of health-interval iterations
            let mut fwd_monitor: Option<CandleMonitorData> = None;
            let mut output_dist: Option<CandleOutputDist> = None;
            let mut grad_flow: Option<Vec<CandleGradientFlow>> = None;

            for _b in 0..batch_size {
                let start = (rng.next_u64() as usize) % (train_data.len() - seq_len - 1);
                let input = &train_data[start..start + seq_len];
                let target = &train_data[start + 1..start + seq_len + 1];
                let is_monitor_batch = measure_monitors && _b == batch_size - 1;

                // Use monitor-instrumented forward on last batch of health intervals
                let (logits, monitor_opt) = if is_monitor_batch {
                    let (l, m) = model.forward_with_monitors(input, &band_masks)?;
                    (l, Some(m))
                } else {
                    (model.forward_with_curriculum(input, &band_masks)?, None)
                };

                // Output distribution monitor (from logits + targets, before loss)
                if is_monitor_batch {
                    output_dist = Some(compute_output_dist(&logits, target));
                }

                let target_tensor = Tensor::from_vec(
                    target.to_vec().iter().map(|&t| t as u32).collect::<Vec<u32>>(),
                    (seq_len,), &device,
                )?;
                let loss = candle_nn::loss::cross_entropy(&logits, &target_tensor)?;
                let loss_val = loss.to_scalar::<f32>()?;

                if loss_val.is_nan() || loss_val.is_infinite() {
                    nan_skip_count += 1;
                    eprintln!("  [NaN skip] iter {iter} batch {_b} (total skips: {nan_skip_count})");
                } else {
                    let mut grads = loss.backward()?;

                    // Gradient flow monitor (from grads, before optimizer step)
                    if is_monitor_batch {
                        grad_flow = Some(compute_gradient_flow(&grads, &varmap, n_layers));
                    }

                    // ── Harmonic backward (manual, outside autograd) ──
                    // Extract d_contribution from grad graph, compute d_harmonic_raw per head,
                    // apply gradient + spring to harmonic_raws, sync harmonic_ns.
                    if use_harmonics_dyn {
                        let eq_fn = |h: usize| -> f32 { ((h + 1) as f32 * 0.5f32).ln() };
                        let spring_k_harm = 2.0f32; // very stiff — integer harmonics theoretically motivated

                        for block in model.blocks.iter_mut() {
                            if !block.harmonic_dyn { continue; }

                            // Extract gradient of contribution tensor from GradStore
                            let d_out_cpu = if let Some(ref layer_out) = block.cached_layer_output {
                                grads.get(layer_out).map(|g| g.to_vec2::<f32>().ok()).flatten()
                            } else {
                                None
                            };

                            if let Some(d_out) = d_out_cpu {
                                let d_hr = harmonic_backward(block, &d_out, n_embd);

                                for h in 0..block.harmonic_ns.len() {
                                    // Gradient step
                                    block.harmonic_ns[h] -= (current_lr as f32) * d_hr[h];
                                    // Spring pull toward equilibrium
                                    let eq = eq_fn(h);
                                    block.harmonic_ns[h] -= (current_lr as f32) * spring_k_harm * (block.harmonic_ns[h] - eq);
                                }

                                // Sync: harmonic_ns = softplus(harmonic_raws)
                                // Since harmonic_ns stores the raw values (confusing name, but matches CPU tier),
                                // and softplus is applied at use-time in wave_attention, no sync needed here.
                                // The update above directly modifies the raw values.
                            }

                            // Clear caches to free memory
                            block.cached_att_weights = None;
                            block.cached_normed_cpu = None;
                            block.cached_layer_output = None;
                        }
                    }

                    // Gradient clipping: scale gradients directly (matches CPU tier).
                    // CPU clips grads BEFORE Adam, so Adam's m/v accumulators see
                    // clipped values. LR scaling is NOT equivalent — it feeds full
                    // unclipped grads to Adam, poisoning the velocity accumulator.
                    let mut gnorm_sq = 0.0f64;
                    for var in &varmap.all_vars() {
                        if let Some(grad) = grads.get(var) {
                            let g: Vec<f32> = grad.flatten_all()?.to_vec1::<f32>()?;
                            for &v in &g { gnorm_sq += (v as f64) * (v as f64); }
                        }
                    }
                    let gnorm = gnorm_sq.sqrt();
                    if gnorm > 1.0 {
                        // Scale gradients in the GradStore (clip to max_norm=1.0)
                        let scale = 1.0 / gnorm;
                        let all_vars = varmap.all_vars();
                        for var in &all_vars {
                            if let Some(grad) = grads.get(var) {
                                let scaled = (grad * scale)?;
                                grads.insert(var, scaled);
                            }
                        }
                    }
                    optimizer.step(&grads)?;
                    drop(grads);
                    device.synchronize()?;
                }

                if !loss_val.is_nan() {
                    total_loss += loss_val;
                }

                // Stash forward monitor data
                if let Some(m) = monitor_opt {
                    fwd_monitor = Some(m);
                }
            }

            total_loss /= batch_size as f32;

            // CustomOp: apply ODE param gradients manually (they bypass autograd)
            if use_custom_op {
                let data = varmap.data().lock().unwrap();
                let mut ode_grads_applied = 0usize;
                for layer in 0..n_layers {
                    let grads_store = model.ode_param_grads.as_ref().unwrap();
                    if let Some(og) = crate::candle_tier::custom_ode::custom_ode::take_param_grads(grads_store, layer) {
                        ode_grads_applied += 1;
                        // Apply gradient to gamma_raw
                        let key = format!("block.{layer}.ode.gamma_raw");
                        if let Some(var) = data.get(&key) {
                            let current = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                            let updated: Vec<f32> = current.iter().zip(&og.d_gamma_raw)
                                .map(|(&c, &g)| c - current_lr as f32 * g)
                                .collect();
                            let _ = var.set(&Tensor::from_vec(updated, var.shape(), var.device()).unwrap());
                        }
                        // Apply gradient to alpha
                        let key = format!("block.{layer}.ode.alpha");
                        if let Some(var) = data.get(&key) {
                            let c = var.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
                            let updated = (c - current_lr as f32 * og.d_alpha).clamp(0.01, 0.5);
                            let _ = var.set(&Tensor::from_slice(&[updated], (1, 1), var.device()).unwrap());
                        }
                        // Apply gradient to beta
                        let key = format!("block.{layer}.ode.beta");
                        if let Some(var) = data.get(&key) {
                            let c = var.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
                            let updated = (c - current_lr as f32 * og.d_beta).clamp(0.01, 1.0);
                            let _ = var.set(&Tensor::from_slice(&[updated], (1, 1), var.device()).unwrap());
                        }
                        // Apply gradient to rk4_weights (if dynamic)
                        if use_rk4_dyn {
                            let key = format!("block.{layer}.ode.rk4_weights");
                            if let Some(var) = data.get(&key) {
                                let current = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                                let updated: Vec<f32> = current.iter().zip(&og.d_rk4_weights)
                                    .map(|(&c, &g)| c - current_lr as f32 * g)
                                    .collect();
                                let _ = var.set(&Tensor::from_vec(updated, var.shape(), var.device()).unwrap());
                            }
                        }
                    }
                }
                drop(data);
                // Health check: log if CustomOp backward produced gradients
                if iter < 10 || (iter % 500 == 0) {
                    if ode_grads_applied == 0 {
                        eprintln!("  [CustomOp health {}] WARNING: 0/{} layers got ODE gradients", iter, n_layers);
                    } else if iter < 10 {
                        eprintln!("  [CustomOp health {}] ODE gradients applied: {}/{} layers", iter, ode_grads_applied, n_layers);
                    }
                }
            }

            // Spring regulation on dynamic params (after optimizer step, like CPU tier)
            // param -= lr * k * (param - equilibrium)
            if spring_k > 0.0 {
                let clr = current_lr;
                let data = varmap.data().lock().unwrap();

                // ODE alpha/beta springs (free, k=0 — self-regulating via AGC)
                // No spring needed — matches CPU tier behavior

                // Corrector plate spring: very loose (k=0.01), eq=0.0 (transparent)
                if phase_native {
                    let k_corr = clr * spring_k * 0.01;
                    for layer in 0..n_layers {
                        let key = format!("block.{layer}.phase_correction");
                        if let Some(var) = data.get(&key) {
                            let vals: Vec<f32> = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                            let new_vals: Vec<f32> = vals.iter().map(|&v| v - (k_corr as f32) * v).collect();
                            let new_tensor = Tensor::from_vec(new_vals, var.shape(), var.device()).unwrap();
                            var.set(&new_tensor).unwrap();
                        }
                    }
                }

                // RK4 weights spring: eq=[1/6,1/3,1/3,1/6], k=2.0 (very stiff)
                if use_rk4_dyn {
                    let k_rk4 = clr * spring_k * 2.0;
                    let eq = [1.0f32/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0];
                    for layer in 0..n_layers {
                        let key = format!("block.{layer}.ode.rk4_weights");
                        if let Some(var) = data.get(&key) {
                            let vals: Vec<f32> = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                            let new_vals: Vec<f32> = vals.iter().enumerate()
                                .map(|(i, &v)| v - (k_rk4 as f32) * (v - eq[i]))
                                .collect();
                            let new_tensor = Tensor::from_vec(new_vals, var.shape(), var.device()).unwrap();
                            var.set(&new_tensor).unwrap();
                        }
                    }
                }

                // Harmonics: gradient + spring already applied in the backward section above
                // (after loss.backward(), before optimizer.step()).
                // The spring here is redundant — harmonic spring is applied per-batch in the backward block.

                // Per-group weight decay (when --wd dyn)
                if use_wd_dyn {
                    let base_wd = 0.01f32;
                    // Apply WD per layer group: param -= lr * base_wd * wd_scale * param
                    for layer in 0..n_layers {
                        let wd_eff = base_wd * wd_scale[layer];
                        let prefix = format!("block.{layer}.");
                        for (key, var) in data.iter() {
                            if key.starts_with(&prefix) {
                                let vals: Vec<f32> = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                                let new_vals: Vec<f32> = vals.iter()
                                    .map(|&v| v - (clr as f32) * wd_eff * v)
                                    .collect();
                                let new_t = Tensor::from_vec(new_vals, var.shape(), var.device()).unwrap();
                                var.set(&new_t).unwrap();
                            }
                        }
                    }
                    // lm_head group (last wd_scale entry)
                    let wd_head = base_wd * wd_scale[n_layers];
                    for (key, var) in data.iter() {
                        if key == "lm_head" || key == "output_corrector" || key.starts_with("ln_f") {
                            let vals: Vec<f32> = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                            let new_vals: Vec<f32> = vals.iter()
                                .map(|&v| v - (clr as f32) * wd_head * v)
                                .collect();
                            let new_t = Tensor::from_vec(new_vals, var.shape(), var.device()).unwrap();
                            var.set(&new_t).unwrap();
                        }
                    }
                }

                // WD spring: eq=1.0, k=1.0 (stiff — uniform regularisation well-motivated)
                if use_wd_dyn {
                    let k_wd = clr * spring_k * 1.0;
                    for s in &mut wd_scale {
                        *s -= (k_wd as f32) * (*s - 1.0);
                        *s = s.clamp(0.01, 10.0);
                    }
                }

                // AGC headroom spring: eq=3.0, k=1.0 (stiff — safety motivated)
                if use_agc_headroom_dyn {
                    let k_agc = clr * spring_k * 1.0;
                    for hr in &mut agc_headroom {
                        *hr -= (k_agc as f32) * (*hr - 3.0);
                        *hr = hr.clamp(1.0, 6.0);
                    }
                    // Update per-layer AGC instances with new headroom
                    if let Some(ref mut agcs) = model.layer_agcs {
                        let ceiling = (std::f32::consts::FRAC_PI_2 / (alpha + 4.0 * beta)).sqrt().max(0.5);
                        for (i, agc) in agcs.iter_mut().enumerate() {
                            *agc = crate::common::agc::OdeAgc::with_ceiling_headroom(ceiling, agc_headroom[i]);
                        }
                    }
                }

                // Layer scale spring: eq=1.0, k=1.0 (moderate)
                if use_layer_scale_dyn {
                    let k_ls = clr * spring_k * 1.0;
                    for layer in 0..n_layers {
                        let key = format!("block.{layer}.layer_scale");
                        if let Some(var) = data.get(&key) {
                            let vals: Vec<f32> = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                            let new_vals: Vec<f32> = vals.iter()
                                .map(|&v| (v - (k_ls as f32) * (v - 1.0)).max(0.0)) // soft floor at 0
                                .collect();
                            let new_tensor = Tensor::from_vec(new_vals, var.shape(), var.device()).unwrap();
                            var.set(&new_tensor).unwrap();
                        }
                    }
                }

                drop(data);
            }
            let iter_time = iter_start.elapsed();

            // VRAM monitoring via cudarc (direct CUDA query — shows ALL GPU memory)
            let vram_used_mb = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                .map(|(free, total)| (total - free) / (1024 * 1024))
                .unwrap_or(0);

            // JSONL telemetry — with AGC diagnostics every 100 iters + monitors at health interval
            if let Some(ref mut writer) = log_writer {
                use std::io::Write;
                if iter % 100 == 0 {
                    // AGC + ODE stats
                    let clamp_count = crate::ffn_backend::ODE_CLAMP_COUNT.load(std::sync::atomic::Ordering::Relaxed);
                    let max_mag = f32::from_bits(crate::ffn_backend::ODE_MAX_MAG.load(std::sync::atomic::Ordering::Relaxed));
                    let agc = crate::ffn_backend::agc_stats();

                    // ODE coupling values from VarMap
                    let ode_str = {
                        let data = varmap.data().lock().unwrap();
                        let mut parts = Vec::new();
                        for l in 0..n_layers {
                            let a = data.get(&format!("block.{l}.ode.alpha")).map(|v| v.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]).unwrap_or(alpha);
                            let b = data.get(&format!("block.{l}.ode.beta")).map(|v| v.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]).unwrap_or(beta);
                            let g = data.get(&format!("block.{l}.ode.gamma_raw")).map(|v| {
                                let vals = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                                let sp = |x: f32| -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } };
                                vals.iter().map(|&x| sp(x)).sum::<f32>() / vals.len() as f32
                            }).unwrap_or(0.1);
                            parts.push(format!(r#"{{"a":{:.4},"b":{:.4},"g":{:.4}}}"#, a, b, g));
                        }
                        format!(r#","ode_params":[{}]"#, parts.join(","))
                    };

                    // Dynamic param values
                    let dyn_str = {
                        let mut s = String::new();
                        if use_layer_scale_dyn {
                            let data = varmap.data().lock().unwrap();
                            let vals: Vec<String> = (0..n_layers).map(|l| {
                                data.get(&format!("block.{l}.layer_scale")).map(|v| format!("{:.4}", v.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0])).unwrap_or("1.0000".to_string())
                            }).collect();
                            s += &format!(r#","layer_scale":[{}]"#, vals.join(","));
                        }
                        if use_rk4_dyn {
                            let data = varmap.data().lock().unwrap();
                            let mut parts = Vec::new();
                            for l in 0..n_layers {
                                if let Some(v) = data.get(&format!("block.{l}.ode.rk4_weights")) {
                                    let w = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                                    parts.push(format!(r#"{{"L{}": [{:.4},{:.4},{:.4},{:.4}]}}"#, l, w[0], w[1], w[2], w[3]));
                                }
                            }
                            if !parts.is_empty() { s += &format!(r#","rk4_weights":[{}]"#, parts.join(",")); }
                        }
                        if use_wd_dyn {
                            let vals: Vec<String> = wd_scale.iter().map(|v| format!("{:.4}", v)).collect();
                            s += &format!(r#","wd_scale":[{}]"#, vals.join(","));
                        }
                        if use_agc_headroom_dyn {
                            let vals: Vec<String> = agc_headroom.iter().map(|v| format!("{:.2}", v)).collect();
                            s += &format!(r#","agc_headroom":[{}]"#, vals.join(","));
                        }
                        if use_harmonics_dyn {
                            let mut parts = Vec::new();
                            for (l, block) in model.blocks.iter().enumerate() {
                                let vals: Vec<String> = block.harmonic_ns.iter().map(|&h| format!("{:.4}", crate::common::math::softplus(h))).collect();
                                parts.push(format!(r#"{{"L{}": [{}]}}"#, l, vals.join(",")));
                            }
                            s += &format!(r#","harmonics":[{}]"#, parts.join(","));
                        }
                        s
                    };

                    let _ = writeln!(writer,
                        "{{\"iter\":{},\"loss\":{:.4},\"lr\":{:.6},\"time_ms\":{},\"vram_mb\":{},\"nan_skips\":{},\"ode_clamps\":{},\"ode_max_mag\":{:.2},\"agc_threshold\":{:.3},\"agc_mean\":{:.3},\"agc_std\":{:.3}{}{}}}",
                        iter, total_loss, current_lr, iter_time.as_millis(), vram_used_mb, nan_skip_count,
                        clamp_count, max_mag, agc.threshold, agc.ema_mean, agc.ema_std, ode_str, dyn_str
                    );
                } else {
                    let _ = writeln!(writer,
                        "{{\"iter\":{},\"loss\":{:.4},\"lr\":{:.6},\"time_ms\":{},\"vram_mb\":{},\"nan_skips\":{}}}",
                        iter, total_loss, current_lr, iter_time.as_millis(), vram_used_mb, nan_skip_count
                    );
                }

                // Monitor suite at health intervals
                if measure_monitors {
                    // Throughput
                    let tok_s = (batch_size * seq_len) as f32 / iter_time.as_secs_f32().max(0.001);
                    let iter_s = 1.0 / iter_time.as_secs_f32().max(0.001);
                    let _ = writeln!(writer,
                        r#"{{"iter":{},"type":"monitor","throughput":{{"tok_s":{:.0},"iter_s":{:.1},"fwd_ms":{},"vram_mb":{}}}}}"#,
                        iter, tok_s, iter_s, iter_time.as_millis(), vram_used_mb
                    );

                    // Embedding space (static — same analysis as CPU)
                    let embed_stats = crate::common::embedding_monitor::analyze_embeddings(&crate::WavePacketModel {
                        wte: model.wte.to_vec2::<f32>().unwrap_or_default(),
                        wpe: vec![], blocks: vec![], ln_f: crate::model::LayerNormWeights { weight: vec![], bias: vec![] },
                        lm_head: vec![], lm_down: vec![], lm_up: vec![], lm_rank: 0, vocab_size,
                        tied_temperature: 1.0, wd_state: None, learnable_ode: false,
                        use_rk4_weights: false, use_dyn_harmonics: false, layer_scale: vec![], use_layer_scale: false,
                        lr_scale: vec![], use_lr_scale: false, wd_scale: vec![], agc_headroom: vec![],
                        phase_native: false, output_corrector: vec![],
                    });
                    let _ = writeln!(writer, r#"{{"iter":{},"type":"monitor",{}}}"#,
                        iter, crate::common::embedding_monitor::to_json(&embed_stats));

                    // Output distribution (#5)
                    if let Some(ref od) = output_dist {
                        let _ = writeln!(writer, r#"{{"iter":{},"type":"monitor",{}}}"#,
                            iter, output_dist_json(od));
                    }

                    // Layer flow (#2)
                    if let Some(ref fm) = fwd_monitor {
                        let lf_json = fm.layer_flow_json();
                        if !lf_json.is_empty() {
                            let _ = writeln!(writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, lf_json);
                        }

                        // Attention heads (#1)
                        let ah_json = fm.attn_heads_json();
                        if !ah_json.is_empty() {
                            let _ = writeln!(writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, ah_json);
                        }

                        // ODE dynamics (#6)
                        let od_json = fm.ode_dynamics_json();
                        if !od_json.is_empty() {
                            let _ = writeln!(writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, od_json);
                        }
                    }

                    // Gradient flow (#3)
                    if let Some(ref gf) = grad_flow {
                        let gf_json = gradient_flow_json(gf);
                        if !gf_json.is_empty() {
                            let _ = writeln!(writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, gf_json);
                        }
                    }
                }

                let _ = writer.flush();
            }

            if iter % 50 == 0 || iter == total_iters - 1 {
                println!("{:>6} {:>10.4} {:>10.1?}  lr={:.6}  vram={}MB", iter, total_loss, iter_time, current_lr, vram_used_mb);
            }

            // GPU duty cycle throttle: sleep between iterations to reduce temperature.
            // --gpu-duty 50 = work one batch, sleep same duration (GPU drops to ~50%).
            if gpu_duty < 100 && gpu_duty > 0 {
                let work_ms = iter_time.as_millis() as u64;
                let sleep_ms = work_ms * (100 - gpu_duty as u64) / gpu_duty as u64;
                std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
            }

            // Periodic checkpoint: save every 500 iters (leak is fixed, 100 was debug)
            if (iter + 1) % 500 == 0 || iter == total_iters - 1 {
                // NaN guard: never overwrite good checkpoints with corrupted weights
                if total_loss.is_nan() || total_loss == 0.0 || total_loss.is_infinite() {
                    eprintln!("  WARNING: loss={total_loss} — skipping checkpoint (corrupted)");
                } else {
                let st_path = format!("candle_checkpoint_iter{}_loss{:.2}.safetensors", iter + 1, total_loss);
                let meta = format!("iter={}\nloss={}\nlr={}\nvocab_size={}\n", iter + 1, total_loss, current_lr, vocab_size);
                if varmap.save(&st_path).is_ok() {
                    std::fs::write(format!("candle_checkpoint_iter{}_loss{:.2}.meta", iter + 1, total_loss), &meta).ok();
                    println!("  Checkpoint: {st_path}");
                }
                let _ = varmap.save("candle_checkpoint_latest.safetensors");
                std::fs::write("candle_checkpoint_latest.meta", &meta).ok();

                let params = extract_wchk_params(&varmap, &model, n_layers, n_embd, maestro_dim,
                    vocab_size, out_proj_groups, n_bands, phase_native,
                    use_rk4_dyn, use_layer_scale_dyn, use_harmonics_dyn);
                let dummy_adam = crate::train::Adam::new(lr as f32, params.len());
                let mut ck_dims = crate::Dims::from_cli(n_bands, n_head, maestro_dim, 256, _rk4_steps)
                    .with_learnable_ode(true).with_corrector(true)
                    .with_rk4_weights(use_rk4_dyn).with_layer_scale(use_layer_scale_dyn);
                ck_dims.use_dyn_harmonics = use_harmonics_dyn;
                crate::wave_checkpoint::save_checkpoint(
                    &params, vocab_size, n_layers, out_proj_groups, iter + 1, lr as f32,
                    &dummy_adam, 0, "checkpoint.bin", ck_dims,
                );
                } // end NaN guard else
            }
        }

        if nan_skip_count > 0 {
            println!("  Warning: {nan_skip_count} NaN steps skipped during training");
        }
        println!("\nTraining complete. Total: {:.1?}", train_start.elapsed());
        Ok(())
    }
}
