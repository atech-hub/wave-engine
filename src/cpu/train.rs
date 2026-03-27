//! Training loop — batch parallel forward/backward, Adam optimizer, curriculum.
//!
//! Extracted from main.rs. Handles the training iteration loop, gradient
//! accumulation, optimizer step, and checkpoint saving.

use crate::*;
use crate::cpu::model_backward::backward;
use crate::wave_checkpoint;
use crate::rng::Rng;

// ─── Adam optimizer ─────────────────────────────────────────────

pub struct Adam {
    pub lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Adam {
    pub fn new(lr: f32, n: usize) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, m: vec![0.0; n], v: vec![0.0; n] }
    }
    pub fn from_checkpoint(lr: f32, t: usize, m: Vec<f32>, v: Vec<f32>) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t, m, v }
    }
    pub fn checkpoint_state(&self) -> (usize, &[f32], &[f32]) {
        (self.t, &self.m, &self.v)
    }
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step_wd(params, grads, 0.01);
    }
    /// AdamW: weight decay applied before momentum update.
    pub fn step_wd(&mut self, params: &mut [f32], grads: &[f32], wd: f32) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            if wd > 0.0 { params[i] -= self.lr * wd * params[i]; }
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) {
    let norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() { *g *= scale; }
    }
}

// ─── Curriculum schedule ────────────────────────────────────────

/// Progressive band curriculum — starts with fewer bands, opens progressively.
pub struct CurriculumSchedule {
    stages: Vec<(usize, f32)>,
}

impl CurriculumSchedule {
    pub fn default_4stage(n_bands: usize) -> Self {
        // Scale stages proportionally: 12.5%, 25%, 50%, 100% of bands
        // At 64 bands: 8, 16, 32, 64. At 384 bands: 48, 96, 192, 384.
        let s1 = (n_bands / 8).max(8);
        let s2 = (n_bands / 4).max(s1);
        let s3 = (n_bands / 2).max(s2);
        Self { stages: vec![(s1, 0.20), (s2, 0.25), (s3, 0.25), (n_bands, 0.30)] }
    }

    pub fn none(n_bands: usize) -> Self {
        Self { stages: vec![(n_bands, 1.0)] }
    }

    pub fn active_bands(&self, iter: usize, n_iters: usize) -> usize {
        let mut cumulative = 0.0f32;
        for &(bands, frac) in &self.stages {
            cumulative += frac;
            if iter < (cumulative * n_iters as f32) as usize { return bands; }
        }
        self.stages.last().unwrap().0
    }

    /// Compute per-band mask values with gradual ramp at stage transitions.
    /// Returns [n_bands] with values in [0.01, 1.0].
    /// Active bands = 1.0, suppressed = 0.01, ramping bands interpolate linearly.
    pub fn band_masks(&self, iter: usize, n_iters: usize, n_bands: usize) -> Vec<f32> {
        let ramp_iters = 200usize; // linear ramp over 200 iterations
        let mut masks = vec![0.01f32; n_bands];

        // Find each stage's start iter and band range
        let mut stage_start = 0usize;
        let mut prev_bands = 0usize;
        for &(bands, frac) in &self.stages {
            let stage_end = stage_start + (frac * n_iters as f32) as usize;

            // Bands from prev_bands..bands are activated at this stage
            for k in prev_bands..bands.min(n_bands) {
                if iter >= stage_start + ramp_iters {
                    // Fully active (past ramp)
                    masks[k] = 1.0;
                } else if iter >= stage_start {
                    // Ramping: linear from 0.01 to 1.0
                    let progress = (iter - stage_start) as f32 / ramp_iters as f32;
                    masks[k] = 0.01 + progress * 0.99;
                }
                // else: still suppressed (0.01)
            }

            prev_bands = bands;
            stage_start = stage_end;
        }

        masks
    }

    pub fn describe(&self, n_iters: usize) {
        let ramp = 200;
        print!("  Curriculum: ");
        let mut start = 0;
        for &(bands, frac) in &self.stages {
            let end = start + (frac * n_iters as f32) as usize;
            print!("{bands} bands (iters {start}-{end}, ramp {ramp})  ");
            start = end;
        }
        println!();
    }
}

// ─── Training loop ──────────────────────────────────────────────

pub struct TrainConfig {
    pub data_path: String,
    pub n_iters: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub n_layers: usize,
    pub lr: f32,
    pub use_bpe: bool,
    pub tokenizer_path: String,
    pub resume_path: Option<String>,
    pub use_curriculum: bool,
    pub use_gpu: bool,
    pub use_monitor: bool,
    pub out_proj_groups: usize,
    pub checkpoint_name: String,
    pub n_bands: usize,
    pub n_head: usize,
    pub alpha: f32,
    pub beta: f32,
    pub agc_ceiling: Option<f32>, // None = auto-derive from alpha
    pub log_name: Option<String>, // Custom log filename (default: training_log_{tier}.jsonl)
    pub m1: Option<usize>,
    pub m2: Option<usize>,
    pub tied: bool,
}

pub fn run_training(config: TrainConfig) {
    // FFT stencil
    fft_ode::validate_fft_derivative(N_BANDS);
    let stencil = fft_ode::StencilFft::new(N_BANDS);
    let gpu_kernel = fft_ode::GpuKernelFft::new(N_BANDS);
    println!("  FFT stencil precomputed (pad to {})", N_BANDS.next_power_of_two());

    // Load data (with token cache — encode once, load instantly on repeat runs)
    println!("Loading dataset from {}...", config.data_path);

    let tok_path = if config.use_bpe { Some(config.tokenizer_path.as_str()) } else { None };
    let (tokens, vocab_size) = if let Some((cached_toks, cached_vs)) = token_cache::load_cache(&config.data_path, config.use_bpe, tok_path) {
        (cached_toks, cached_vs)
    } else {
        let raw = std::fs::read_to_string(&config.data_path).expect("Failed to read data file");
        let (toks, vs) = if config.use_bpe {
            let tokenizer = bpe::BpeTokenizer::from_file(&config.tokenizer_path);
            let t = tokenizer.encode(&raw);
            let v = tokenizer.vocab_size;
            println!("  BPE tokens: {}, vocab: {}", t.len(), v);
            (t, v)
        } else {
            let chars: Vec<char> = raw.chars().collect();
            let mut vocab: Vec<char> = chars.clone();
            vocab.sort();
            vocab.dedup();
            let v = vocab.len();
            let char_to_idx: std::collections::HashMap<char, usize> = vocab.iter().enumerate().map(|(i, &c)| (c, i)).collect();
            let t: Vec<usize> = chars.iter().map(|c| *char_to_idx.get(c).unwrap_or(&0)).collect();
            println!("  Char-level tokens: {}, vocab: {}", t.len(), v);
            (t, v)
        };
        token_cache::save_cache(&config.data_path, config.use_bpe, tok_path, &toks, vs);
        (toks, vs)
    };
    let split = (tokens.len() as f32 * 0.9) as usize;
    let train_data = &tokens[..split];
    println!("  Train tokens: {}", train_data.len());

    // Initialize or resume
    let (mut model, start_iter, mut optimizer, mut rng);
    if let Some(ref ckpt) = config.resume_path {
        println!("Resuming from checkpoint: {ckpt}");
        let (params, ck_vocab, ck_iter, _ck_lr, ck_rng, adam_t, adam_m, adam_v, _ck_groups) = wave_checkpoint::load_checkpoint(ckpt);
        assert_eq!(ck_vocab, vocab_size, "Vocab size mismatch: checkpoint={ck_vocab}, data={vocab_size}");
        let mut m = init_model(vocab_size, 42, config.n_layers, config.out_proj_groups, crate::Dims::from_cli(config.n_bands, config.n_head, crate::MAESTRO_DIM, crate::BLOCK_SIZE, crate::RK4_STEPS).with_moduli(config.m1, config.m2).with_tied(config.tied), config.alpha, config.beta);
        unflatten_params(&mut m, &params);
        model = m;
        start_iter = ck_iter;
        optimizer = Adam::from_checkpoint(config.lr, adam_t, adam_m, adam_v);
        rng = Rng::from_state(ck_rng);
        println!("  Resuming from iter {start_iter}");
        if config.m1.is_some() || config.m2.is_some() {
            eprintln!("  WARNING: Custom moduli with --resume will change embeddings but not trained weights");
        }
    } else {
        println!("Initializing model (seed=42)...");
        model = init_model(vocab_size, 42, config.n_layers, config.out_proj_groups, crate::Dims::from_cli(config.n_bands, config.n_head, crate::MAESTRO_DIM, crate::BLOCK_SIZE, crate::RK4_STEPS).with_moduli(config.m1, config.m2).with_tied(config.tied), config.alpha, config.beta);
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
    let dims = crate::Dims::from_cli(config.n_bands, config.n_head, crate::MAESTRO_DIM, crate::BLOCK_SIZE, crate::RK4_STEPS).with_moduli(config.m1, config.m2).with_tied(config.tied);

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

    let warmup_iters = 100usize;
    let min_lr_ratio = 0.1f32; // decay to 10% of base LR

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

        let t_fwd = monitor.start();
        let batch_results: Vec<(f32, Vec<f32>)> = std::thread::scope(|s| {
            let handles: Vec<_> = starts.iter().map(|&start| {
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
                s.spawn(move || {
                    let cache = forward_with_cache(model_ref, input, dims, gpu_ref, pp_ref, fg_ref, st_ref, gk_ref);
                    let (loss, grads) = backward(model_ref, &cache, target, dims, gpu_ref, pp_ref, fg_ref);
                    (loss, flatten_grads_ex(&grads, dims.tied))
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        monitor.record("fwd+bwd (batch)", t_fwd);

        let t_reduce = monitor.start();
        let mut total_loss = 0.0f32;
        let mut total_grads = vec![0.0f32; n_trainable];
        for (loss, fg) in &batch_results {
            total_loss += loss;
            for (a, g) in total_grads.iter_mut().zip(fg.iter()) { *a += g; }
        }
        total_loss /= batch_size as f32;
        for g in total_grads.iter_mut() { *g /= batch_size as f32; }
        monitor.record("reduce", t_reduce);

        // NaN skip with post-mortem: discard batch, diagnose cause
        if total_loss.is_nan() || total_loss.is_infinite() {
            nan_skip_count += 1;
            // Post-mortem: which batch elements caused it?
            let nan_elements: Vec<usize> = batch_results.iter().enumerate()
                .filter(|(_, (loss, _))| loss.is_nan() || loss.is_infinite())
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

        let t_optim = monitor.start();
        clip_grad_norm(&mut total_grads, 1.0);
        let mut params = flatten_params_ex(&model, config.tied);
        optimizer.step(&mut params, &total_grads);
        unflatten_params_ex(&mut model, &params, config.tied);
        monitor.record("optimizer", t_optim);

        let grad_norm: f32 = total_grads.iter().map(|g| g * g).sum::<f32>().sqrt();

        // First-10 health check: per-component gradient norms + weight growth
        if iter < 10 {
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

        // JSONL telemetry — every iteration, with gradient diagnostics every 100
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
            writeln!(log_writer,
                r#"{{"iter":{},"loss":{:.4},"lr":{:.6},"time_ms":{},"nan_skips":{},"model_gn":{:.4},"head_gn":{:.4},"head_pct":{:.1},"layer_gn":[{}],"ode_clamps":{},"ode_max_mag":{:.2},"agc_threshold":{:.3},"agc_mean":{:.3},"agc_std":{:.3}}}"#,
                iter, total_loss, current_lr, iter_start.elapsed().as_millis(), nan_skip_count,
                model_gn, head_gn, head_pct, layer_str, clamp_count, max_mag,
                agc.threshold, agc.ema_mean, agc.ema_std
            ).ok();
        } else {
            writeln!(log_writer,
                r#"{{"iter":{},"loss":{:.4},"lr":{:.6},"time_ms":{},"nan_skips":{}}}"#,
                iter, total_loss, current_lr, iter_start.elapsed().as_millis(), nan_skip_count
            ).ok();
        }
        log_writer.flush().ok();

        if iter % 10 == 0 || iter == total_iters - 1 {
            println!("{:>6} {:>10.4} {:>10.1?}  lr={:.6}  gnorm={:.2}", iter, total_loss, iter_start.elapsed(), current_lr, grad_norm);
            if monitor.enabled() {
                monitor.report(if iter == 0 { 1 } else { 50.min(iter) });
                monitor.reset();
            }
        }

        // Periodic checkpoint: save every 500 iters (all tiers, always untied format)
        if (iter + 1) % 500 == 0 {
            let save_p = flatten_params(&model);
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
    {
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
        use std::io::Write;
        writeln!(log_writer, "{}", summary).ok();
        log_writer.flush().ok();
    }

    // Final checkpoint (always saves full untied params for compatibility)
    let save_params = flatten_params(&model); // untied — includes lm_head
    let groups = model.blocks[0].ffn.out_proj.n_groups();
    if config.tied {
        // Optimizer is tied-size; checkpoint needs untied-size. Save dummy optimizer.
        let dummy_opt = Adam::new(config.lr, save_params.len());
        wave_checkpoint::save_checkpoint(&save_params, vocab_size, model.blocks.len(), groups, total_iters, lr, &dummy_opt, rng.state(), &config.checkpoint_name, dims);
    } else {
        wave_checkpoint::save_checkpoint(&save_params, vocab_size, model.blocks.len(), groups, total_iters, lr, &optimizer, rng.state(), &config.checkpoint_name, dims);
    }
    println!("Checkpoint saved to {}", config.checkpoint_name);
}
