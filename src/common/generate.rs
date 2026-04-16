//! Text generation using the training forward pass.
//! Guarantees inference matches training — same ODE, corrector, AGC.

use crate::*;
use crate::cpu::forward::forward_with_cache;
use crate::common::wave_model::{count_trainable_ex, unflatten_params_ex};

pub struct GenerateConfig {
    pub resume_path: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub n_layers: usize,
    pub n_bands: usize,
    pub n_head: usize,
    pub out_proj_groups: usize,
    pub maestro_dim: usize,
    pub use_bpe: bool,
    pub tokenizer_path: String,
    pub alpha: f32,
    pub beta: f32,
    pub temperature: f32,
    pub phase_native: bool,
    pub memory_path: Option<String>,
    pub diagnose: bool,
}

pub fn run_generate(config: GenerateConfig) {
    let n_embd = config.n_bands * 2;

    // Load checkpoint
    let (params, ck_vocab, ck_iter, _lr, _rng, _at, _am, _av, _groups, ck_flags, _chi) =
        wave_checkpoint::load_checkpoint(&config.resume_path);

    // Tokenize prompt
    let (token_ids, vocab_size, detokenize): (Vec<usize>, usize, Box<dyn Fn(usize) -> String>) =
    if config.use_bpe {
        let tok = bpe::BpeTokenizer::from_file(&config.tokenizer_path);
        let ids = tok.encode(&config.prompt);
        let tok2 = bpe::BpeTokenizer::from_file(&config.tokenizer_path);
        (ids, ck_vocab, Box::new(move |id| tok2.decode(&[id])))
    } else {
        // Char-level: build vocab from training data file (same as training tokenizer)
        let data_path = std::env::args().skip_while(|a| a != "--data").nth(1)
            .unwrap_or_else(|| std::env::args().nth(1).unwrap_or("data/input.txt".to_string()));
        let text = crate::common::data_loader::load_text_raw(&data_path);
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let vocab = chars.len().min(ck_vocab);
        let char_map: Vec<char> = chars[..vocab].to_vec();
        eprintln!("  Char vocab: {} chars from {}", vocab, data_path);
        let ids: Vec<usize> = config.prompt.chars().filter_map(|c| char_map.iter().position(|&ch| ch == c)).collect();
        let cm2 = char_map.clone();
        (ids, vocab, Box::new(move |id| if id < cm2.len() { cm2[id].to_string() } else { "?".to_string() }))
    };

    let effective_vocab = vocab_size.max(ck_vocab);

    // Build model with correct dims
    let dims = Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, BLOCK_SIZE, RK4_STEPS)
        .with_learnable_ode(false)
        .with_corrector(true);
    let mut mdl = init_model(effective_vocab, 42, config.n_layers, config.out_proj_groups, dims, config.alpha, config.beta);
    let mut loaded = false;

    // v3 checkpoint: use feature flags directly — no ambiguity
    if ck_flags > 0 {
        let has_ode  = ck_flags & (1 << 0) != 0;
        let has_ls   = ck_flags & (1 << 1) != 0;
        let has_rk4  = ck_flags & (1 << 2) != 0;
        let has_harm = ck_flags & (1 << 3) != 0;
        let d = Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, BLOCK_SIZE, RK4_STEPS)
            .with_corrector(has_ode).with_learnable_ode(has_ode)
            .with_layer_scale(has_ls).with_rk4_weights(has_rk4).with_dyn_harmonics(has_harm);
        // Try phase-native first, then standard
        let mut m = init_model(effective_vocab, 42, config.n_layers, config.out_proj_groups, d, config.alpha, config.beta);
        m.phase_native = true;
        if params.len() == count_trainable_ex(&m, false) {
            unflatten_params_ex(&mut m, &params, false);
            let features: Vec<&str> = [has_ls.then_some("ls"), has_rk4.then_some("rk4"), has_harm.then_some("harm")]
                .into_iter().flatten().collect();
            let feat_str = if features.is_empty() { String::new() } else { format!(" + {}", features.join("+")) };
            eprintln!("  Loaded {} params (phase-native{}, flags=0x{:02x}) from {}", params.len(), feat_str, ck_flags, config.resume_path);
            mdl = m;
            loaded = true;
        } else {
            let mut m2 = init_model(effective_vocab, 42, config.n_layers, config.out_proj_groups, d, config.alpha, config.beta);
            if params.len() == count_trainable_ex(&m2, false) {
                unflatten_params_ex(&mut m2, &params, false);
                eprintln!("  Loaded {} params (flags=0x{:02x}) from {}", params.len(), ck_flags, config.resume_path);
                mdl = m2;
                loaded = true;
            }
        }
    }

    // Helper: try a specific Dims configuration
    let try_load = |d: Dims, phase_native: bool, label: &str| -> Option<WavePacketModel> {
        let mut m = init_model(effective_vocab, 42, config.n_layers, config.out_proj_groups, d, config.alpha, config.beta);
        m.phase_native = phase_native;
        if params.len() == count_trainable_ex(&m, false) {
            unflatten_params_ex(&mut m, &params, false);
            eprintln!("  Loaded {} params ({}) from {}", params.len(), label, config.resume_path);
            Some(m)
        } else {
            None
        }
    };

    // Generate all Dims variants systematically (most features → fewest).
    // Each combination of phase_native × layer_scale × rk4_weights × harmonics.
    let base = || Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, BLOCK_SIZE, RK4_STEPS)
        .with_corrector(true).with_learnable_ode(true);
    let feature_combos: Vec<(bool, bool, bool, &str)> = vec![
        // (layer_scale, rk4, harmonics, label)
        (true,  true,  true,  "ls+rk4+harm"),
        (false, true,  true,  "rk4+harm"),
        (true,  true,  false, "ls+rk4"),
        (false, true,  false, "rk4"),
        (true,  false, true,  "ls+harm"),
        (false, false, true,  "harm"),
        (true,  false, false, "ls"),
        (false, false, false, ""),
    ];

    // Phase-native variants
    for &(ls, rk4, harm, suffix) in &feature_combos {
        if !loaded {
            let d = base().with_layer_scale(ls).with_rk4_weights(rk4).with_dyn_harmonics(harm);
            let label = if suffix.is_empty() { "phase-native".to_string() } else { format!("phase-native + {}", suffix) };
            if let Some(m) = try_load(d, true, &label) {
                mdl = m;
                loaded = true;
            }
        }
    }

    // Non-phase-native variants
    for &(ls, rk4, harm, suffix) in &feature_combos {
        if !loaded {
            let d = base().with_layer_scale(ls).with_rk4_weights(rk4).with_dyn_harmonics(harm);
            let label = if suffix.is_empty() { "ext".to_string() } else { format!("ext + {}", suffix) };
            if let Some(m) = try_load(d, false, &label) {
                mdl = m;
                loaded = true;
            }
        }
    }
    // Ext + layer_scale
    if !loaded {
        let d = Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, BLOCK_SIZE, RK4_STEPS)
            .with_corrector(true).with_layer_scale(true);
        let mut m = init_model(effective_vocab, 42, config.n_layers, config.out_proj_groups, d, config.alpha, config.beta);
        if params.len() == count_trainable_ex(&m, false) {
            unflatten_params_ex(&mut m, &params, false);
            mdl = m;
            eprintln!("  Loaded {} params (with ODE/corrector/layer_scale) from {}", params.len(), config.resume_path);
            loaded = true;
        }
    }
    // Ext (ODE + corrector)
    let dims_ext = Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, BLOCK_SIZE, RK4_STEPS)
        .with_corrector(true);
    let mut model_ext = init_model(effective_vocab, 42, config.n_layers, config.out_proj_groups, dims_ext, config.alpha, config.beta);
    let ext_count = count_trainable_ex(&model_ext, false);
    if !loaded && params.len() == ext_count {
        unflatten_params_ex(&mut model_ext, &params, false);
        mdl = model_ext;
        eprintln!("  Loaded {} params (with ODE/corrector) from {}", params.len(), config.resume_path);
        loaded = true;
    }
    // Base (no ODE params)
    if !loaded {
        let dims_base = Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, BLOCK_SIZE, RK4_STEPS)
            .with_learnable_ode(false).with_corrector(false);
        mdl = init_model(effective_vocab, 42, config.n_layers, config.out_proj_groups, dims_base, config.alpha, config.beta);
        unflatten_params(&mut mdl, &params);
        eprintln!("  Loaded {} params (base) from {}", params.len(), config.resume_path);
    }
    mdl.learnable_ode = false;
    mdl.phase_native = config.phase_native;

    eprintln!("  Model: {}L, {}bands, {}dim, {}vocab, iter {}{}",
        config.n_layers, config.n_bands, n_embd, effective_vocab, ck_iter,
        if config.phase_native { " [phase-native]" } else { "" });

    // FFT stencil for ODE
    let stencil = fft_ode::StencilFft::new(config.n_bands);

    // Initialize AGC from model's coupling
    let alpha = mdl.blocks[0].ffn.kerr.alpha;
    let beta = mdl.blocks[0].ffn.kerr.beta;
    crate::ffn_backend::init_agc(alpha, beta);

    // Wave memory: load or create if --memory was specified
    let n_ode_layers = config.n_layers; // all layers have ODE in wave-engine
    let mut wave_mem = config.memory_path.as_ref().map(|path| {
        crate::common::wave_memory::load_or_create(path, n_ode_layers, config.n_bands)
    });
    let mem_offsets = wave_mem.as_ref().map(|m| crate::common::wave_memory::build_offsets(m));
    let mem_slices: Option<Vec<(&[f32], &[f32])>> = mem_offsets.as_ref().map(|o| o.as_slices());

    // Autoregressive generation
    let mut tokens = token_ids.clone();
    let block_size = dims.block_size;

    // Print prompt
    eprint!("  Prompt: ");
    for &id in &token_ids {
        eprint!("{}", detokenize(id));
    }
    eprintln!();
    eprintln!("  Generating {} tokens...", config.max_tokens);
    eprintln!("---");

    // Print prompt tokens as output start
    for &id in &token_ids {
        print!("{}", detokenize(id));
    }

    for _ in 0..config.max_tokens {
        // Truncate to block_size
        let start = if tokens.len() > block_size { tokens.len() - block_size } else { 0 };
        let input = &tokens[start..];

        // Forward pass — SAME as training, with optional memory injection
        let cache = forward_with_cache(&mdl, input, dims, None, None, None, Some(&stencil), None, None,
            mem_slices.as_deref());

        // Get logits/scores for last position
        // Phase-native decode is now handled in forward.rs (dot product against embeddings)
        let last_pos = cache.logits.len() - 1;
        let last_logits = &cache.logits[last_pos];

        // Sample next token
        let next_token = if config.temperature <= 0.0 {
            // Greedy
            last_logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap()
        } else {
            // Temperature sampling
            let max_logit = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = last_logits.iter().map(|&l| ((l - max_logit) / config.temperature).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let probs: Vec<f32> = exp.iter().map(|e| e / sum).collect();

            // Simple sampling with system RNG
            let r: f32 = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut h = DefaultHasher::new();
                tokens.len().hash(&mut h);
                (h.finish() as f32) / (u64::MAX as f32)
            };
            let mut cumsum = 0.0f32;
            let mut chosen = probs.len() - 1;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum { chosen = i; break; }
            }
            chosen
        };

        tokens.push(next_token);
        print!("{}", detokenize(next_token));
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!();
    eprintln!("---");
    eprintln!("  Generated {} tokens (total {} tokens)", config.max_tokens, tokens.len());
}

/// Wave-space generation: runs full attention+ODE on wave inputs,
/// decodes output via nearest-embedding lookup. For wave-trained models.
pub fn run_wave_generate(config: GenerateConfig) {
    let n_embd = config.n_bands * 2;

    // Load checkpoint
    let (params, ck_vocab, ck_iter, _lr, _rng, _at, _am, _av, _groups, ck_flags, _chi) =
        wave_checkpoint::load_checkpoint(&config.resume_path);

    // Tokenize prompt
    let (token_ids, vocab_size, detokenize): (Vec<usize>, usize, Box<dyn Fn(usize) -> String>) =
    if config.use_bpe {
        let tok = bpe::BpeTokenizer::from_file(&config.tokenizer_path);
        let ids = tok.encode(&config.prompt);
        let tok2 = bpe::BpeTokenizer::from_file(&config.tokenizer_path);
        (ids, ck_vocab, Box::new(move |id| tok2.decode(&[id])))
    } else {
        let data_path = std::env::args().skip_while(|a| a != "--data").nth(1)
            .unwrap_or_else(|| std::env::args().nth(1).unwrap_or("data/input.txt".to_string()));
        let text = crate::common::data_loader::load_text_raw(&data_path);
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort(); chars.dedup();
        let vocab = chars.len().min(ck_vocab);
        let char_map: Vec<char> = chars[..vocab].to_vec();
        eprintln!("  Char vocab: {} chars from {}", vocab, data_path);
        let ids: Vec<usize> = config.prompt.chars().filter_map(|c| char_map.iter().position(|&ch| ch == c)).collect();
        let cm2 = char_map.clone();
        (ids, vocab, Box::new(move |id| if id < cm2.len() { cm2[id].to_string() } else { "?".to_string() }))
    };

    let effective_vocab = vocab_size.max(ck_vocab);

    // Build model — try multiple configurations to match checkpoint param count.
    // Checkpoints may have been saved with learnable_ode=true (ODE params in vector)
    // or learnable_ode=false. Try all 4 combinations: {ode, no-ode} × {corrector, no-corrector}.
    let variants: [(bool, bool); 4] = [
        (false, true),  // standard: no ODE, with corrector
        (false, false), // no ODE, no corrector
        (true, true),   // ODE params + corrector (wave training default)
        (true, false),  // ODE params, no corrector
    ];
    let mut mdl = {
        let dims0 = Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, BLOCK_SIZE, RK4_STEPS)
            .with_learnable_ode(false).with_corrector(true);
        init_model(effective_vocab, 42, config.n_layers, config.out_proj_groups, dims0, config.alpha, config.beta)
    };
    mdl.phase_native = true;
    mdl.output_corrector = vec![0.0; config.n_bands];

    let mut dims = Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, BLOCK_SIZE, RK4_STEPS);
    let mut loaded = false;
    for (use_ode, use_corr) in &variants {
        let dims_try = Dims::from_cli(config.n_bands, config.n_head, config.maestro_dim, BLOCK_SIZE, RK4_STEPS)
            .with_learnable_ode(*use_ode).with_corrector(*use_corr);
        let mut m = init_model(effective_vocab, 42, config.n_layers, config.out_proj_groups, dims_try, config.alpha, config.beta);
        m.phase_native = true;
        m.output_corrector = vec![0.0; config.n_bands];
        if params.len() == count_trainable_ex(&m, false) {
            unflatten_params_ex(&mut m, &params, false);
            eprintln!("  [wave-generate] Loaded: ode={}, corrector={}", use_ode, use_corr);
            mdl = m;
            dims = dims_try;
            loaded = true;
            break;
        }
    }
    if !loaded {
        panic!("Cannot match checkpoint param count {} to any model variant", params.len());
    }

    let stencil = fft_ode::StencilFft::new(config.n_bands);
    let alpha = mdl.blocks[0].ffn.kerr.alpha;
    let beta = mdl.blocks[0].ffn.kerr.beta;
    crate::ffn_backend::init_agc(alpha, beta);

    eprintln!("  Model: {}L, {}bands, {}vocab, iter {} [wave-generate]",
        config.n_layers, config.n_bands, effective_vocab, ck_iter);

    // Convert prompt tokens to wave inputs
    let block_size = dims.block_size;
    let mut tokens = token_ids.clone();

    eprint!("  Prompt: ");
    for &id in &token_ids { eprint!("{}", detokenize(id)); }
    eprintln!();
    eprintln!("  Generating {} tokens (wave-space)...", config.max_tokens);
    eprintln!("---");

    for &id in &token_ids { print!("{}", detokenize(id)); }

    for _ in 0..config.max_tokens {
        let start = if tokens.len() > block_size { tokens.len() - block_size } else { 0 };
        let input_tokens = &tokens[start..];

        // Convert tokens to wave inputs (embedding + positional)
        let wave_inputs: Vec<Vec<f32>> = input_tokens.iter().enumerate().map(|(pos, &tok)| {
            let mut h = vec![0.0f32; n_embd];
            if tok < mdl.wte.len() && pos < mdl.wpe.len() {
                for j in 0..n_embd { h[j] = mdl.wte[tok][j] + mdl.wpe[pos][j]; }
            }
            h
        }).collect();

        // Full forward pass through wave path (with attention across positions)
        let cache = crate::cpu::forward::forward_with_cache_from_waves(
            &mdl, &wave_inputs, dims, Some(&stencil),
        );

        // Get last position's output
        let last_pos = cache.post_ln_f.len() - 1;
        let output = &cache.post_ln_f[last_pos];

        // Decode: nearest-neighbour against PURE token embeddings (no positional)
        let mut best_tok = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for v in 0..mdl.vocab_size {
            let emb = &mdl.wte[v];
            let dot: f32 = (0..n_embd).map(|j| output[j] * emb[j]).sum();
            if dot > best_sim { best_sim = dot; best_tok = v; }
        }

        if config.temperature > 0.0 {
            // Temperature sampling over all tokens
            let scores: Vec<f32> = (0..mdl.vocab_size).map(|v| {
                let emb = &mdl.wte[v];
                (0..n_embd).map(|j| output[j] * emb[j]).sum::<f32>() / config.temperature
            }).collect();
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum: f32 = exp.iter().sum();

            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            tokens.len().hash(&mut hasher);
            let r = (hasher.finish() as f32) / (u64::MAX as f32);

            let mut cumsum = 0.0f32;
            best_tok = exp.len() - 1;
            for (i, &e) in exp.iter().enumerate() {
                cumsum += e / sum;
                if r < cumsum { best_tok = i; break; }
            }
        }

        if config.diagnose {
            print_wave_diagnosis(output, best_tok, &mdl.wte, config.n_bands, &*detokenize, tokens.len() - token_ids.len());
        }

        tokens.push(best_tok);
        print!("{}", detokenize(best_tok));
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!();
    eprintln!("---");
    eprintln!("  Generated {} tokens (wave-space decode)", config.max_tokens);
}

/// Diagnose the output wave vs target embedding at the phase level.
fn print_wave_diagnosis(
    output: &[f32],
    chosen_tok: usize,
    wte: &[Vec<f32>],
    n_bands: usize,
    detokenize: &dyn Fn(usize) -> String,
    step: usize,
) {
    let n_embd = n_bands * 2;
    let target = &wte[chosen_tok];

    // Extract phases and magnitudes
    let out_phases = super::wave_analysis::extract_phases(output, n_bands);
    let tgt_phases = super::wave_analysis::extract_phases(target, n_bands);

    let out_mags: Vec<f32> = (0..n_bands).map(|k| {
        (output[k*2]*output[k*2] + output[k*2+1]*output[k*2+1]).sqrt()
    }).collect();
    let tgt_mags: Vec<f32> = (0..n_bands).map(|k| {
        (target[k*2]*target[k*2] + target[k*2+1]*target[k*2+1]).sqrt()
    }).collect();

    // Per-band phase error (wrapped to [0, PI])
    let phase_errors: Vec<f32> = (0..n_bands).map(|k| {
        let diff = (out_phases[k] - tgt_phases[k]).abs();
        let wrapped = if diff > std::f32::consts::PI { std::f32::consts::TAU - diff } else { diff };
        wrapped
    }).collect();

    let mean_phase_err = phase_errors.iter().sum::<f32>() / n_bands as f32;
    let mut sorted_errs = phase_errors.clone();
    sorted_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_phase_err = sorted_errs[n_bands / 2];
    let close_count = phase_errors.iter().filter(|&&e| e < std::f32::consts::FRAC_PI_4).count();
    let far_count = phase_errors.iter().filter(|&&e| e > std::f32::consts::FRAC_PI_4 * 3.0).count();

    // Worst 5 bands
    let mut indexed_errs: Vec<(usize, f32)> = phase_errors.iter().enumerate().map(|(i, &e)| (i, e)).collect();
    indexed_errs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Phase vs magnitude L2 decomposition
    let mut phase_l2 = 0.0f32;
    let mut mag_l2 = 0.0f32;
    for k in 0..n_bands {
        phase_l2 += tgt_mags[k] * tgt_mags[k] * 2.0 * (1.0 - phase_errors[k].cos());
        mag_l2 += (out_mags[k] - tgt_mags[k]) * (out_mags[k] - tgt_mags[k]);
    }
    let total_l2: f32 = (0..n_embd).map(|j| (output[j] - target[j]).powi(2)).sum();
    let phase_pct = if total_l2 > 0.0 { phase_l2 / total_l2 * 100.0 } else { 0.0 };
    let mag_pct = if total_l2 > 0.0 { mag_l2 / total_l2 * 100.0 } else { 0.0 };

    // Grid breakdown
    let half = n_bands / 2;
    let g1_phase: f32 = phase_errors[..half].iter().sum::<f32>() / half as f32;
    let g2_phase: f32 = phase_errors[half..].iter().sum::<f32>() / half as f32;
    let g1_mag: f32 = (0..half).map(|k| (out_mags[k] - tgt_mags[k]).abs()).sum::<f32>() / half as f32;
    let g2_mag: f32 = (half..n_bands).map(|k| (out_mags[k] - tgt_mags[k]).abs()).sum::<f32>() / half as f32;

    // Top-5 decode candidates
    let mut candidates: Vec<(usize, f32, f32, f32)> = (0..wte.len()).map(|v| {
        let emb = &wte[v];
        let dot: f32 = (0..n_embd).map(|j| output[j] * emb[j]).sum();
        let l2: f32 = (0..n_embd).map(|j| (output[j] - emb[j]).powi(2)).sum();
        let emb_phases = super::wave_analysis::extract_phases(emb, n_bands);
        let mean_ang: f32 = (0..n_bands).map(|k| {
            let d = (out_phases[k] - emb_phases[k]).abs();
            if d > std::f32::consts::PI { std::f32::consts::TAU - d } else { d }
        }).sum::<f32>() / n_bands as f32;
        (v, dot, l2, mean_ang)
    }).collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Print
    eprintln!("\n── Step {} diagnose ──", step);
    eprintln!("  Chosen: '{}' (id={})", detokenize(chosen_tok).replace('\n', "\\n"), chosen_tok);
    eprintln!();
    eprintln!("  Phase error vs chosen embedding:");
    eprintln!("    mean={:.2}rad  median={:.2}rad  close(<π/4)={}  far(>3π/4)={}",
        mean_phase_err, median_phase_err, close_count, far_count);
    eprint!("    Worst bands:");
    for i in 0..5.min(indexed_errs.len()) {
        eprint!(" #{}({:.2})", indexed_errs[i].0, indexed_errs[i].1);
    }
    eprintln!();

    eprintln!();
    eprintln!("  Phase vs magnitude split (of total L2={:.1}):", total_l2);
    eprintln!("    Phase-only: {:.1} ({:.0}%)    Magnitude-only: {:.1} ({:.0}%)",
        phase_l2, phase_pct, mag_l2, mag_pct);

    eprintln!();
    eprintln!("  Grid breakdown:");
    eprintln!("    Grid1 (0-{}):  phase={:.2}rad  mag_err={:.2}", half-1, g1_phase, g1_mag);
    eprintln!("    Grid2 ({}-{}): phase={:.2}rad  mag_err={:.2}", half, n_bands-1, g2_phase, g2_mag);

    eprintln!();
    eprintln!("  Top-5 candidates:");
    eprintln!("    {:>4} {:>8} {:>8} {:>8}  token", "rank", "dot", "L2", "phase");
    for i in 0..5.min(candidates.len()) {
        let (v, dot, l2, ang) = candidates[i];
        eprintln!("    #{:<3} {:>8.1} {:>8.1} {:>8.2}  '{}'",
            i+1, dot, l2, ang, detokenize(v).replace('\n', "\\n"));
    }
    eprintln!();
}

/// Teacher-forced accuracy test: feed correct KWDS waves, check decode accuracy.
pub fn run_teacher_force(
    checkpoint_path: &str,
    kwds_path: &str,
    data_path: &str,
    n_layers: usize,
    n_head: usize,
    vocab: usize,
    alpha: f32,
    beta: f32,
) {
    let mut f = std::fs::File::open(kwds_path).expect("Cannot open KWDS file");
    let header = super::kwds::read_header(&mut f).unwrap();
    let n_bands = header.n_bands as usize;
    let n_embd = n_bands * 2;

    let (params, ck_vocab, ck_iter, _, _, _, _, _, _, _, _) = crate::wave_checkpoint::load_checkpoint(checkpoint_path);
    let variants: [(bool, bool); 4] = [(false, true), (false, false), (true, true), (true, false)];
    let effective_vocab = vocab.max(ck_vocab);
    let mut dims = Dims::from_cli(n_bands, n_head, 16, 128, 16);
    let mut model = crate::init_model(effective_vocab, 42, n_layers, 1, dims, alpha, beta);
    for (use_ode, use_corr) in &variants {
        let d = Dims::from_cli(n_bands, n_head, 16, 128, 16)
            .with_learnable_ode(*use_ode).with_corrector(*use_corr);
        let mut mdl = crate::init_model(effective_vocab, 42, n_layers, 1, d, alpha, beta);
        mdl.phase_native = true; mdl.output_corrector = vec![0.0; n_bands];
        if params.len() == count_trainable_ex(&mdl, false) {
            unflatten_params_ex(&mut mdl, &params, false); model = mdl; dims = d; break;
        }
    }
    crate::ffn_backend::init_agc(alpha, beta);
    let stencil = fft_ode::StencilFft::new(n_bands);

    let text = super::data_loader::load_text_raw(data_path);
    let mut chars: Vec<char> = text.chars().collect();
    chars.sort(); chars.dedup();
    let char_map: Vec<char> = chars[..chars.len().min(effective_vocab)].to_vec();
    let detok = |id: usize| -> String {
        if id < char_map.len() { char_map[id].to_string() } else { format!("?{}", id) }
    };

    let seq_len = 64usize;
    println!("Teacher-forced accuracy test: {} positions, seq_len={}", header.n_positions, seq_len);
    println!("  Checkpoint: {} (iter {})", checkpoint_path, ck_iter);

    let test_len = seq_len.min(header.n_positions as usize - 1);
    let inputs = super::kwds::read_input_window(&mut f, &header, 0, test_len).unwrap();
    let targets = super::kwds::read_target_window(&mut f, &header, 0, test_len).unwrap();
    let cache = crate::cpu::forward::forward_with_cache_from_waves(&model, &inputs, dims, Some(&stencil));

    let mut correct = 0usize;
    let mut total = 0usize;
    for pos in 0..cache.post_ln_f.len().min(targets.len()) {
        let output = &cache.post_ln_f[pos];
        let target = &targets[pos];
        let mut target_tok = 0;
        let mut target_best = f32::NEG_INFINITY;
        for v in 0..effective_vocab {
            let dot: f32 = (0..n_embd).map(|j| target[j] * model.wte[v][j]).sum();
            if dot > target_best { target_best = dot; target_tok = v; }
        }
        let mut scores: Vec<(usize, f32)> = (0..effective_vocab).map(|v| {
            let dot: f32 = (0..n_embd).map(|j| output[j] * model.wte[v][j]).sum();
            (v, dot)
        }).collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let chosen = scores[0].0;
        if chosen == target_tok { correct += 1; }
        total += 1;
        if pos < 20 || chosen == target_tok {
            let target_rank = scores.iter().position(|s| s.0 == target_tok).unwrap_or(999) + 1;
            println!("  pos {:3}: output='{}' target='{}' {}  rank={}",
                pos, detok(chosen).replace('\n', "\\n"), detok(target_tok).replace('\n', "\\n"),
                if chosen == target_tok { "✓" } else { "✗" }, target_rank);
        }
    }
    println!("\n  Teacher-forced accuracy: {}/{} ({:.1}%)", correct, total, correct as f64 / total as f64 * 100.0);
}
