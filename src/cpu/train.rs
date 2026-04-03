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
    pub lm_rank: usize,
    pub wave_decode: bool,
    pub unfreeze_phases: bool,
    pub health_interval: usize, // 0 = disabled
    pub freeze_ode: bool,
    pub head_lr_floor: f32, // 0.0 = disabled, e.g. 0.00003 = 30% of 1e-4
    pub no_corrector: bool, // --no-corrector: disable corrector plate (A/B testing)
    pub layer_scale: DynParam, // --layer-scale dyn | --layer-scale 1.0,0.8,1.0,1.0
    pub lr_scale: DynParam,    // --lr-scale dyn | --lr-scale 1.0,1.5,1.5,0.5,1.0
    pub phase_native: bool,
    pub pythagorean: bool,    // --phase-native: use phase coherence loss, no lm_head
    pub phase_temp: f32,       // temperature for phase-native softmax (default 1.0)
    pub spring_k: f32, // spring constant for dynamic params (0.0 = no spring, 0.1 = moderate)
    pub active_layers: Option<usize>, // --active-layers N: first N layers at eq=1.0, rest at eq=0.0
    pub rk4_weights: DynParam, // --rk4-weights dyn | --rk4-weights standard
    pub wd: DynParam,          // --wd dyn | --wd 0.01 | --wd 0.01,0.02,0.01,0.005,0.01
}

/// A parameter that can be fixed (manual value) or dynamic (model learns it).
#[derive(Clone)]
pub enum DynParam {
    Off,                    // not used
    Dynamic,                // model decides (with spring)
    Fixed(Vec<f32>),        // human prescribes per-group values
}

impl DynParam {
    pub fn is_active(&self) -> bool { !matches!(self, DynParam::Off) }
    pub fn is_dynamic(&self) -> bool { matches!(self, DynParam::Dynamic) }
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
        let mut m = init_model(vocab_size, 42, config.n_layers, config.out_proj_groups, crate::Dims::from_cli(config.n_bands, config.n_head, crate::MAESTRO_DIM, crate::BLOCK_SIZE, crate::RK4_STEPS).with_moduli(config.m1, config.m2).with_tied(config.tied).with_lm_rank(config.lm_rank).with_wave_decode(config.wave_decode).with_unfreeze_phases(config.unfreeze_phases).with_learnable_ode(!config.freeze_ode).with_corrector(!config.no_corrector && !config.freeze_ode).with_layer_scale(config.layer_scale.is_active()).with_lr_scale(config.lr_scale.is_active()).with_pythagorean(config.pythagorean).with_rk4_weights(config.rk4_weights.is_active()), config.alpha, config.beta);
        m.phase_native = config.phase_native; // Must set before count_trainable for correct param count
        let ext_count = count_trainable_ex(&m, config.tied);
        if params.len() == ext_count {
            unflatten_params_ex(&mut m, &params, config.tied);
            println!("  Loaded {} params (with ODE/corrector)", params.len());
        } else {
            // Old checkpoint without ODE/corrector — load base params, ODE starts fresh
            let base_dims = crate::Dims::from_cli(config.n_bands, config.n_head, crate::MAESTRO_DIM, crate::BLOCK_SIZE, crate::RK4_STEPS).with_moduli(config.m1, config.m2).with_tied(config.tied).with_lm_rank(config.lm_rank).with_wave_decode(config.wave_decode).with_unfreeze_phases(config.unfreeze_phases).with_learnable_ode(false).with_corrector(false);
            m = init_model(vocab_size, 42, config.n_layers, config.out_proj_groups, base_dims, config.alpha, config.beta);
            unflatten_params(&mut m, &params);
            // Re-enable learnable ODE on the loaded model
            m.learnable_ode = !config.freeze_ode;
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
        model = init_model(vocab_size, 42, config.n_layers, config.out_proj_groups, crate::Dims::from_cli(config.n_bands, config.n_head, crate::MAESTRO_DIM, crate::BLOCK_SIZE, crate::RK4_STEPS).with_moduli(config.m1, config.m2).with_tied(config.tied).with_lm_rank(config.lm_rank).with_wave_decode(config.wave_decode).with_unfreeze_phases(config.unfreeze_phases).with_learnable_ode(!config.freeze_ode).with_corrector(!config.no_corrector && !config.freeze_ode).with_layer_scale(config.layer_scale.is_active()).with_lr_scale(config.lr_scale.is_active()).with_pythagorean(config.pythagorean).with_rk4_weights(config.rk4_weights.is_active()), config.alpha, config.beta);
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
    let mut dims = crate::Dims::from_cli(config.n_bands, config.n_head, crate::MAESTRO_DIM, crate::BLOCK_SIZE, crate::RK4_STEPS).with_moduli(config.m1, config.m2).with_tied(config.tied).with_lm_rank(config.lm_rank).with_wave_decode(config.wave_decode).with_unfreeze_phases(config.unfreeze_phases).with_learnable_ode(!config.freeze_ode).with_corrector(!config.no_corrector && !config.freeze_ode).with_layer_scale(config.layer_scale.is_active()).with_lr_scale(config.lr_scale.is_active()).with_rk4_weights(config.rk4_weights.is_active());
    dims.phase_temp = config.phase_temp;
    dims.pythagorean = config.pythagorean;

    // Phase-native mode: ODE learns to output in embedding space, no lm_head
    model.phase_native = config.phase_native;

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

    // Monitor state: dynamic parameter evolution tracking
    let mut prev_dyn_snap: Option<crate::common::dyn_param_monitor::DynParamSnapshot> = None;

    // Monitor state: curriculum transition tracking (#8)
    let mut curriculum_tracker = crate::common::curriculum_monitor::CurriculumTracker::new();

    // Monitor state: checkpoint drift tracking (#9)
    let mut checkpoint_tracker = crate::common::checkpoint_monitor::CheckpointTracker::new(model.blocks.len());

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
        // Health data extracted from first batch element at health intervals
        struct BatchHealthData {
            distortion: Option<Vec<crate::common::ode_distortion::LayerDistortionSummary>>,
            grad_flow: Option<Vec<crate::common::gradient_monitor::GradientFlowStats>>,
            attn_stats: Option<Vec<crate::common::attn_monitor::AttentionHeadStats>>,
            flow_stats: Option<Vec<crate::common::layer_flow_monitor::LayerFlowStats>>,
            output_stats: Option<crate::common::output_monitor::OutputDistStats>,
            ode_dynamics: Option<Vec<crate::common::ode_dynamics_monitor::OdeDynamicsStats>>,
        }
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
                s.spawn(move || {
                    let cache = forward_with_cache(model_ref, input, dims, gpu_ref, pp_ref, fg_ref, st_ref, gk_ref, None);

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
                        Some(crate::common::attn_monitor::analyze_attention(model_ref, &cache))
                    } else {
                        None
                    };

                    // Layer signal flow (#2)
                    let flow_stats = if is_health {
                        Some(crate::common::layer_flow_monitor::analyze_flow(&cache, dims))
                    } else {
                        None
                    };

                    // Output distribution (#5)
                    let output_stats = if is_health {
                        let targets_vec: Vec<usize> = target.to_vec();
                        Some(crate::common::output_monitor::analyze_output(&cache.logits, &targets_vec))
                    } else {
                        None
                    };

                    // ODE dynamics deep (#6)
                    let ode_dynamics = if is_health {
                        Some(crate::common::ode_dynamics_monitor::analyze_ode_dynamics(&cache, dims))
                    } else {
                        None
                    };

                    let (loss, grads) = backward(model_ref, &cache, target, dims, gpu_ref, pp_ref, fg_ref);
                    // Gradient flow analysis on first batch element at health intervals
                    let grad_flow = if is_health {
                        Some(crate::common::gradient_monitor::analyze_gradients(&grads, dims))
                    } else {
                        None
                    };
                    (loss, flatten_grads_ex(&grads, dims.tied), BatchHealthData {
                        distortion, grad_flow, attn_stats, flow_stats, output_stats, ode_dynamics,
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
        let mut batch_grad_flow: Option<Vec<crate::common::gradient_monitor::GradientFlowStats>> = None;
        let mut batch_attn_stats: Option<Vec<crate::common::attn_monitor::AttentionHeadStats>> = None;
        let mut batch_flow_stats: Option<Vec<crate::common::layer_flow_monitor::LayerFlowStats>> = None;
        let mut batch_output_stats: Option<crate::common::output_monitor::OutputDistStats> = None;
        let mut batch_ode_dynamics: Option<Vec<crate::common::ode_dynamics_monitor::OdeDynamicsStats>> = None;
        for (loss, fg, health) in &batch_results {
            total_loss += loss;
            for (a, g) in total_grads.iter_mut().zip(fg.iter()) { *a += g; }
            if health.distortion.is_some() && batch_distortion_data.is_none() {
                batch_distortion_data = health.distortion.clone();
            }
            if batch_grad_flow.is_none() {
                if let Some(ref gf) = health.grad_flow {
                    batch_grad_flow = Some(gf.iter().map(|s| crate::common::gradient_monitor::GradientFlowStats {
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
                    batch_attn_stats = Some(stats.iter().map(|s| crate::common::attn_monitor::AttentionHeadStats {
                        layer: s.layer, head: s.head, harmonic: s.harmonic,
                        entropy: s.entropy, max_weight: s.max_weight,
                        top_position: s.top_position, self_attn_frac: s.self_attn_frac,
                    }).collect());
                }
            }
            if batch_flow_stats.is_none() {
                if let Some(ref stats) = health.flow_stats {
                    batch_flow_stats = Some(stats.iter().map(|s| crate::common::layer_flow_monitor::LayerFlowStats {
                        layer: s.layer, input_norm: s.input_norm,
                        attn_output_norm: s.attn_output_norm, ffn_output_norm: s.ffn_output_norm,
                        output_norm: s.output_norm, attn_ratio: s.attn_ratio,
                        ffn_ratio: s.ffn_ratio, residual_ratio: s.residual_ratio,
                        cosine_in_out: s.cosine_in_out,
                    }).collect());
                }
            }
            if batch_output_stats.is_none() {
                if let Some(ref stats) = health.output_stats {
                    batch_output_stats = Some(crate::common::output_monitor::OutputDistStats {
                        avg_entropy: stats.avg_entropy, avg_margin: stats.avg_margin,
                        avg_correct_rank: stats.avg_correct_rank, worst_margin: stats.worst_margin,
                        worst_prompt_pos: stats.worst_prompt_pos, mode_collapse: stats.mode_collapse,
                    });
                }
            }
            if batch_ode_dynamics.is_none() {
                if let Some(ref stats) = health.ode_dynamics {
                    batch_ode_dynamics = Some(stats.iter().map(|s| crate::common::ode_dynamics_monitor::OdeDynamicsStats {
                        layer: s.layer, phase_velocity: s.phase_velocity,
                        energy_in: s.energy_in, energy_out: s.energy_out,
                        energy_ratio: s.energy_ratio, band_energy_std: s.band_energy_std,
                        damping_effective: s.damping_effective,
                    }).collect());
                }
            }
        }
        total_loss /= batch_size as f32;
        for g in total_grads.iter_mut() { *g /= batch_size as f32; }
        monitor.record("reduce", t_reduce);

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
        // The lm_head gradient starves during sustained training (71% → 9%).
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
        // Each group's gradients are multiplied by its lr_scale.
        // The lr_scale evolves via spring toward 1.0.
        if config.lr_scale.is_active() {
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

        let mut params = flatten_params_ex(&model, config.tied);
        if config.wd.is_active() {
            // Per-group weight decay: apply WD manually, then Adam without WD
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
            optimizer.step_wd(&mut params, &total_grads, 0.0); // Adam without WD (already applied)
        } else {
            optimizer.step(&mut params, &total_grads);
        }
        unflatten_params_ex(&mut model, &params, config.tied);

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
        let optim_elapsed = t_optim.elapsed();
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
                    let a_gn = total_grads.get(l * per_layer + per_layer - 2).copied().unwrap_or(0.0).abs();
                    let b_gn = total_grads.get(l * per_layer + per_layer - 1).copied().unwrap_or(0.0).abs();
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
            writeln!(log_writer,
                r#"{{"iter":{},"loss":{:.4},"lr":{:.6},"time_ms":{},"nan_skips":{},"model_gn":{:.4},"head_gn":{:.4},"head_pct":{:.1},"layer_gn":[{}],"ode_clamps":{},"ode_max_mag":{:.2},"agc_threshold":{:.3},"agc_mean":{:.3},"agc_std":{:.3}{}{}{}{}{}}}"#,
                iter, total_loss, current_lr, iter_start.elapsed().as_millis(), nan_skip_count,
                model_gn, head_gn, head_pct, layer_str, clamp_count, max_mag,
                agc.threshold, agc.ema_mean, agc.ema_std, ode_str, ls_str, lrs_str, rk4w_str, wd_str
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

        // Encoding health sample (opt-in via --health-interval)
        if config.health_interval > 0 && iter % config.health_interval == 0 {
            if let Some(h) = crate::common::encoding_health::sample(
                &model, dims, config.use_bpe, &config.tokenizer_path, &stencil,
                config.alpha, config.beta,
            ) {
                let health_json = crate::common::encoding_health::to_json(&h);
                use std::io::Write;
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
            if let Some(ref layers) = batch_distortion_data {
                use std::io::Write;
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
            if let Some(ref gf_stats) = batch_grad_flow {
                let gf_json = crate::common::gradient_monitor::to_json(gf_stats);
                if !gf_json.is_empty() {
                    use std::io::Write;
                    writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, gf_json).ok();
                    log_writer.flush().ok();
                }
            }

            // Dynamic parameter evolution (#7)
            {
                let snap = crate::common::dyn_param_monitor::snapshot(&model, prev_dyn_snap.as_ref());
                let dp_json = crate::common::dyn_param_monitor::to_json(&snap);
                if !dp_json.is_empty() {
                    use std::io::Write;
                    writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, dp_json).ok();
                    log_writer.flush().ok();
                }
                prev_dyn_snap = Some(snap);
            }

            // Throughput (#10)
            {
                let tp_stats = crate::common::throughput_monitor::compute(
                    batch_size,
                    seq_len,
                    iter_start.elapsed().as_secs_f32() * 1000.0,
                    fwd_bwd_elapsed.as_secs_f32() * 1000.0,
                    optim_elapsed.as_secs_f32() * 1000.0,
                );
                let tp_json = crate::common::throughput_monitor::to_json(&tp_stats);
                use std::io::Write;
                writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, tp_json).ok();
                log_writer.flush().ok();
            }

            // --- Monitor suite (batch 2) ---

            // Attention head activity (#1)
            if let Some(ref attn_stats) = batch_attn_stats {
                let attn_json = crate::common::attn_monitor::to_json(attn_stats);
                if !attn_json.is_empty() {
                    use std::io::Write;
                    writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, attn_json).ok();
                    log_writer.flush().ok();
                }
            }

            // Layer signal flow (#2)
            if let Some(ref flow_stats) = batch_flow_stats {
                let flow_json = crate::common::layer_flow_monitor::to_json(flow_stats);
                if !flow_json.is_empty() {
                    use std::io::Write;
                    writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, flow_json).ok();
                    log_writer.flush().ok();
                }
            }

            // Output distribution (#5)
            if let Some(ref output_stats) = batch_output_stats {
                let out_json = crate::common::output_monitor::to_json(output_stats);
                use std::io::Write;
                writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, out_json).ok();
                log_writer.flush().ok();
            }

            // --- Monitor suite (batch 3) ---

            // Embedding space (#4) — medium cost, sample-based
            {
                let embed_stats = crate::common::embedding_monitor::analyze_embeddings(&model);
                let embed_json = crate::common::embedding_monitor::to_json(&embed_stats);
                use std::io::Write;
                writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, embed_json).ok();
                log_writer.flush().ok();
            }

            // ODE dynamics deep (#6)
            if let Some(ref ode_dyn) = batch_ode_dynamics {
                let ode_json = crate::common::ode_dynamics_monitor::to_json(ode_dyn);
                if !ode_json.is_empty() {
                    use std::io::Write;
                    writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, ode_json).ok();
                    log_writer.flush().ok();
                }
            }
        }

        // Curriculum transition (#8) — emits when transition detected (every iter tracking)
        if let Some(ref event) = curriculum_event {
            let cur_json = crate::common::curriculum_monitor::to_json(event);
            use std::io::Write;
            writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, cur_json).ok();
            log_writer.flush().ok();
            eprintln!("  [curriculum {}] stage {} → {} bands, loss jump {:.4}",
                iter, event.stage, event.active_bands, event.loss_jump);
        }

        // Periodic checkpoint: save every 500 iters (all tiers, always untied format)
        // Uses flatten_params_ex to include ODE params + corrector plate (matches optimizer state)
        if (iter + 1) % 500 == 0 {
            let save_p = flatten_params_ex(&model, false);

            // Checkpoint drift (#9) — measure before saving
            if let Some(drift) = checkpoint_tracker.measure(&save_p) {
                let drift_json = crate::common::checkpoint_monitor::to_json(&drift);
                use std::io::Write;
                writeln!(log_writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, drift_json).ok();
                log_writer.flush().ok();
                eprintln!("  [drift {}] total={:.4} relative={:.6} ode={:.4}",
                    iter, drift.total_drift, drift.relative_drift, drift.ode_drift);
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
}
