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
}

pub fn run_generate(config: GenerateConfig) {
    let n_embd = config.n_bands * 2;

    // Load checkpoint
    let (params, ck_vocab, ck_iter, _lr, _rng, _at, _am, _av, _groups, ck_flags) =
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

        // Forward pass — SAME as training
        let cache = forward_with_cache(&mdl, input, dims, None, None, None, Some(&stencil), None, None);

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
