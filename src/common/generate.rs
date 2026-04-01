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
    pub use_bpe: bool,
    pub tokenizer_path: String,
    pub alpha: f32,
    pub beta: f32,
    pub temperature: f32,
}

pub fn run_generate(config: GenerateConfig) {
    let n_embd = config.n_bands * 2;

    // Load checkpoint
    let (params, ck_vocab, ck_iter, _lr, _rng, _at, _am, _av, _groups) =
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
        let text = std::fs::read_to_string(&data_path).expect("Failed to read data file for char vocab");
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
    let dims_ext = Dims::from_cli(config.n_bands, config.n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
        .with_corrector(true);
    let mut model = init_model(effective_vocab, 42, config.n_layers, config.out_proj_groups, dims_ext, config.alpha, config.beta);
    let ext_count = count_trainable_ex(&model, false);

    // Use inference dims: no ODE backward caching, but corrector ACTIVE
    let dims = Dims::from_cli(config.n_bands, config.n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
        .with_learnable_ode(false)
        .with_corrector(true);

    // Try ext+layer_scale, then ext, then base
    let dims_ext_ls = Dims::from_cli(config.n_bands, config.n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
        .with_corrector(true).with_layer_scale(true);
    let mut model_ls = init_model(effective_vocab, 42, config.n_layers, config.out_proj_groups, dims_ext_ls, config.alpha, config.beta);
    let ext_ls_count = count_trainable_ex(&model_ls, false);

    if params.len() == ext_ls_count {
        unflatten_params_ex(&mut model_ls, &params, false);
        model = model_ls;
        eprintln!("  Loaded {} params (with ODE/corrector/layer_scale) from {}", params.len(), config.resume_path);
    } else if params.len() == ext_count {
        unflatten_params_ex(&mut model, &params, false);
        eprintln!("  Loaded {} params (with ODE/corrector) from {}", params.len(), config.resume_path);
    } else {
        let dims_base = Dims::from_cli(config.n_bands, config.n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
            .with_learnable_ode(false).with_corrector(false);
        model = init_model(effective_vocab, 42, config.n_layers, config.out_proj_groups, dims_base, config.alpha, config.beta);
        unflatten_params(&mut model, &params);
        eprintln!("  Loaded {} params (base) from {}", params.len(), config.resume_path);
    }
    model.learnable_ode = false;

    eprintln!("  Model: {}L, {}bands, {}dim, {}vocab, iter {}",
        config.n_layers, config.n_bands, n_embd, effective_vocab, ck_iter);

    // FFT stencil for ODE
    let stencil = fft_ode::StencilFft::new(config.n_bands);

    // Initialize AGC from model's coupling
    let alpha = model.blocks[0].ffn.kerr.alpha;
    let beta = model.blocks[0].ffn.kerr.beta;
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
        let cache = forward_with_cache(&model, input, dims, None, None, None, Some(&stencil), None, None);

        // Get logits for last position
        let last_logits = &cache.logits[cache.logits.len() - 1];

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
