//! Wave Packet Engine — proof of concept
//!
//! New architecture: parallel attention + FFN with harmonic coherence scoring.
//! Tests whether wave packet mechanics can serve as the core computation primitive.

// ─── Module tree ─────────────────────────────────────────────
// Physical layout: src/common/, src/cpu/, src/wgpu_tier/, src/candle_tier/
// Re-exports below keep old crate:: paths working (shim layer).

#[allow(dead_code)]
mod common;
#[allow(dead_code)]
mod cpu;
#[allow(dead_code)]
mod wgpu_tier;
#[allow(dead_code)]
mod candle_tier;
#[cfg(feature = "serve")]
mod serve_tier;

// Re-export shim — existing code uses crate::model, crate::backend, etc.
// These map to the new physical locations without changing any imports.
#[allow(unused_imports)]
pub use common::model;
#[allow(unused_imports)]
pub use common::embed as wave_embed;
#[allow(unused_imports)]
pub use common::attn as wave_attn;
#[allow(unused_imports)]
pub use common::block as wave_block;
#[allow(unused_imports)]
pub use common::ffn as ffn_backend;
pub use common::wave_model::{WavePacketModel, init_model, init_linear, count_trainable, count_trainable_ex, flatten_params, flatten_params_ex, unflatten_params, unflatten_params_ex};
pub use common::dims::{Dims, PROFILE, N_BANDS, N_EMBD, N_HEAD, N_LAYERS, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS};
pub use cpu::forward::{BlockCache, ForwardCache, forward_with_cache, dual_maestro_forward};
pub use cpu::model_backward::{Gradients, flatten_grads, flatten_grads_ex};
pub use wgpu_tier::diagnostics::{diagnose_ode_gpu_vs_cpu, validate_gpu_fft};
#[allow(unused_imports)]
pub use common::checkpoint as wave_checkpoint;
#[allow(unused_imports)]
pub use common::rng;
#[allow(unused_imports)]
pub use common::bpe;
#[allow(unused_imports)]
pub use common::token_cache;
#[allow(unused_imports)]
pub use common::monitor;
#[allow(unused_imports)]
pub use common::data;
#[allow(unused_imports)]
pub use common::fft_ode;

#[allow(unused_imports)]
pub use cpu::train;
#[allow(unused_imports)]
pub use cpu::backward;

#[allow(unused_imports)]
pub use wgpu_tier::backend;
#[allow(unused_imports)]
pub use wgpu_tier::device as gpu;
#[allow(unused_imports)]
pub use wgpu_tier::gpu_backend;
#[allow(unused_imports)]
pub use wgpu_tier::buffers as gpu_buffers;
#[allow(unused_imports)]
pub use wgpu_tier::dispatch as gpu_dispatch;
#[allow(unused_imports)]
pub use wgpu_tier::ops_forward as gpu_ops_forward;
#[allow(unused_imports)]
pub use wgpu_tier::ops_backward as gpu_ops_backward;
#[allow(unused_imports)]
pub use wgpu_tier::pipelines as gpu_pipelines;
#[allow(unused_imports)]
pub use wgpu_tier::resident as gpu_resident;
#[allow(unused_imports)]
pub use wgpu_tier::validate as gpu_validate;
#[allow(unused_imports)]
pub use wgpu_tier::ffn_gpu;
#[allow(unused_imports)]
pub use wgpu_tier::ffn_full_gpu;

#[allow(unused_imports)]
pub use candle_tier::engine as candle_engine;
#[allow(unused_imports)]
pub use candle_tier::ode as gpu_ode;
#[allow(unused_imports)]
pub use candle_tier::block_diag as block_diagonal;

// ─── Main ───────────────────────────────────────────────────────

fn print_help() { common::help::print_help(); }

fn main() {
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    // Rayon thread pool — configurable via --threads (default: half available cores)
    fn parse_flag_early<T: std::str::FromStr>(name: &str, default: T) -> T {
        std::env::args().skip_while(|a| a != name).nth(1)
            .and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads: usize = parse_flag_early("--threads", available / 2);
    let n_threads = n_threads.max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok();

    println!("wave-engine v0.1.0  ({n_threads} threads, {available} available)\n");

    // ─── Scale mode ───
    if std::env::args().any(|a| a == "--scale") {
        fn pflag<T: std::str::FromStr>(name: &str, default: T) -> T {
            std::env::args().skip_while(|a| a != name).nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let source = std::env::args().skip_while(|a| a != "--scale").nth(1)
            .expect("--scale requires a checkpoint path");
        let target_bands: usize = pflag("--target-bands", 128);
        let target_head: usize = pflag("--target-head", 8);
        let target_layers: Option<usize> = std::env::args().skip_while(|a| a != "--target-layers").nth(1)
            .and_then(|s| s.parse().ok());
        let output: String = pflag("--output", "scaled_checkpoint.bin".to_string());
        let groups: usize = pflag("--out-proj-groups", 1);

        common::scale::scale_checkpoint(&common::scale::ScaleConfig {
            source_path: source,
            target_bands,
            target_head,
            target_layers,
            output_path: output,
            target_groups: groups,
            seed: 42,
        }).unwrap_or_else(|e| { eprintln!("Scale error: {e}"); std::process::exit(1); });
        return;
    }

    // Check for --candle flag first — routes to entirely different training path
    if std::env::args().any(|a| a == "--candle") {
        fn pflag<T: std::str::FromStr>(name: &str, default: T) -> T {
            std::env::args().skip_while(|a| a != name).nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let data_path = std::env::args().nth(1).unwrap_or("data/input.txt".to_string());
        let n_iters: usize = pflag("--iters", 200);
        let n_bands: usize = pflag("--n-bands", N_BANDS);
        let n_head: usize = pflag("--n-head", N_HEAD);
        let n_layers: usize = pflag("--layers", N_LAYERS);
        let maestro_dim: usize = pflag("--maestro-dim", MAESTRO_DIM);
        let rk4_steps: usize = pflag("--rk4-steps", RK4_STEPS);
        let out_proj_groups: usize = pflag("--out-proj-groups", 6);

        let debug_nan = std::env::args().any(|a| a == "--debug-nan");
        let alpha: f32 = pflag("--alpha", 0.1);
        let beta: f32 = pflag("--beta", alpha);
        match candle_engine::engine::train_candle(
            &data_path, n_iters, n_bands, n_head, n_layers, maestro_dim, rk4_steps, out_proj_groups, debug_nan, alpha, beta,
        ) {
            Ok(()) => return,
            Err(e) => { eprintln!("Candle error: {e:?}"); std::process::exit(1); }
        }
    }

    // ─── CLI flag parser ───
    fn parse_flag<T: std::str::FromStr>(name: &str, default: T) -> T {
        std::env::args().skip_while(|a| a != name).nth(1)
            .and_then(|s| s.parse().ok()).unwrap_or(default)
    }

    // Parse dynamic parameter: --flag dyn | --flag 1.0,0.8,1.0 | absent
    fn parse_dyn_param(name: &str) -> train::DynParam {
        let val = std::env::args().skip_while(|a| a != name).nth(1);
        match val {
            None => train::DynParam::Off,
            Some(s) if s == "dyn" || s == "dynamic" => train::DynParam::Dynamic,
            Some(s) if s == "off" || s == "none" || s == "false" => train::DynParam::Off,
            Some(s) => {
                let vals: Vec<f32> = s.split(',').filter_map(|v| v.parse().ok()).collect();
                if vals.is_empty() { train::DynParam::Dynamic } else { train::DynParam::Fixed(vals) }
            }
        }
    }

    // ─── Analyze mode ───
    if std::env::args().any(|a| a == "--analyze") {
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--analyze requires --resume <checkpoint>");
        let n_layers: usize = parse_flag("--layers", N_LAYERS);
        let out_proj_groups: usize = parse_flag("--out-proj-groups", 6);
        let use_bpe = std::env::args().any(|a| a == "--bpe");
        let tokenizer_path: String = parse_flag("--tokenizer", "data/tokenizer.json".to_string());
        let n_bands: usize = parse_flag("--n-bands", N_BANDS);
        let n_head: usize = parse_flag("--n-head", N_HEAD);
        let alpha: f32 = parse_flag("--alpha", 0.1);
        let beta: f32 = parse_flag("--beta", alpha);
        let sub_harmonic = std::env::args().any(|a| a == "--sub-harmonic");
        common::analyze::run_analyze(
            &resume_path, n_layers, out_proj_groups, use_bpe, &tokenizer_path,
            n_bands, n_head, alpha, beta, sub_harmonic,
        );
        return;
    }

    // ─── ODE monitor ───
    if std::env::args().any(|a| a == "--ode-monitor") {
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--ode-monitor requires --resume");
        let n_layers: usize = parse_flag("--layers", N_LAYERS);
        let n_bands: usize = parse_flag("--n-bands", N_BANDS);
        let n_head: usize = parse_flag("--n-head", N_HEAD);
        let out_proj_groups: usize = parse_flag("--out-proj-groups", 6);
        let alpha: f32 = parse_flag("--alpha", 0.1);
        let beta: f32 = parse_flag("--beta", parse_flag("--alpha", 0.1));
        let data_path = std::env::args().skip_while(|a| a != "--data").nth(1)
            .unwrap_or("data/arithmetic_single.txt".to_string());

        let (params, ck_vocab, _, _, _, _, _, _, _, _) = wave_checkpoint::load_checkpoint(&resume_path);
        // Load model (try ext+ls, ext, base)
        let dims_ext_ls = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS).with_corrector(true).with_layer_scale(true);
        let mut model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_ext_ls, alpha, beta);
        let ext_ls = common::wave_model::count_trainable_ex(&model, false);
        let dims_ext = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS).with_corrector(true);
        let mut model2 = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_ext, alpha, beta);
        let ext_count = common::wave_model::count_trainable_ex(&model2, false);
        if params.len() == ext_ls {
            common::wave_model::unflatten_params_ex(&mut model, &params, false);
        } else if params.len() == ext_count {
            model = model2;
            common::wave_model::unflatten_params_ex(&mut model, &params, false);
        } else {
            let dims_base = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
                .with_learnable_ode(false).with_corrector(false);
            model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_base, alpha, beta);
            unflatten_params(&mut model, &params);
        }
        model.learnable_ode = false;

        let dims = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
            .with_learnable_ode(false).with_corrector(true);
        let stencil = fft_ode::StencilFft::new(n_bands);
        crate::ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);

        // Build char vocab
        let text = std::fs::read_to_string(&data_path).expect("Failed to read data");
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort(); chars.dedup();
        let char_map: Vec<char> = chars[..chars.len().min(ck_vocab)].to_vec();
        let encode = |s: &str| -> Vec<usize> {
            s.chars().filter_map(|c| char_map.iter().position(|&ch| ch == c)).collect()
        };
        let decode = |id: usize| -> String {
            if id < char_map.len() { char_map[id].to_string() } else { "?".to_string() }
        };

        let prompt = std::env::args().skip_while(|a| a != "--prompt").nth(1)
            .unwrap_or("3+4=".to_string());
        let compare = std::env::args().skip_while(|a| a != "--compare").nth(1);

        println!("=== ODE Monitor ===\n");
        let tokens = encode(&prompt);
        common::ode_monitor::print_ode_summary(&model, &tokens, dims, &stencil, &prompt, &decode);

        if let Some(ref cmp) = compare {
            let tokens_b = encode(cmp);
            common::ode_monitor::compare_prompts(&model, &tokens, &tokens_b, dims, &stencil, &prompt, cmp);
        }

        return;
    }

    // ─── Phase decode diagnostic ───
    if std::env::args().any(|a| a == "--phase-decode") {
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--phase-decode requires --resume");
        let n_layers: usize = parse_flag("--layers", N_LAYERS);
        let n_bands: usize = parse_flag("--n-bands", N_BANDS);
        let n_head: usize = parse_flag("--n-head", N_HEAD);
        let out_proj_groups: usize = parse_flag("--out-proj-groups", 6);
        let alpha: f32 = parse_flag("--alpha", 0.1);
        let beta: f32 = parse_flag("--beta", parse_flag("--alpha", 0.1));
        let data_path = std::env::args().skip_while(|a| a != "--data").nth(1)
            .unwrap_or("data/arithmetic_single.txt".to_string());

        let (params, ck_vocab, _, _, _, _, _, _, _, _) = wave_checkpoint::load_checkpoint(&resume_path);
        let dims_ext = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS).with_corrector(true).with_layer_scale(true);
        let mut model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_ext, alpha, beta);
        let ext_ls = common::wave_model::count_trainable_ex(&model, false);
        let dims_ext2 = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS).with_corrector(true);
        let mut model2 = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_ext2, alpha, beta);
        let ext_count = common::wave_model::count_trainable_ex(&model2, false);
        if params.len() == ext_ls {
            common::wave_model::unflatten_params_ex(&mut model, &params, false);
        } else if params.len() == ext_count {
            model = model2;
            common::wave_model::unflatten_params_ex(&mut model, &params, false);
        } else {
            let dims_base = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
                .with_learnable_ode(false).with_corrector(false);
            model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_base, alpha, beta);
            unflatten_params(&mut model, &params);
        }
        model.learnable_ode = false;

        let dims = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
            .with_learnable_ode(false).with_corrector(true);
        let stencil = fft_ode::StencilFft::new(n_bands);
        crate::ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);

        // Build char vocab from data file
        let text = std::fs::read_to_string(&data_path).expect("Failed to read data file");
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort(); chars.dedup();
        let char_map: Vec<char> = chars[..chars.len().min(ck_vocab)].to_vec();
        let encode = |s: &str| -> Vec<usize> {
            s.chars().filter_map(|c| char_map.iter().position(|&ch| ch == c)).collect()
        };
        let decode = |id: usize| -> String {
            if id < char_map.len() { char_map[id].to_string() } else { "?".to_string() }
        };

        println!("Phase decode diagnostic: {} params, {} vocab", params.len(), ck_vocab);
        println!("{:<10} {:<10} {:<10} {:<10}", "Prompt", "Expected", "LM_Head", "Phase");
        println!("{}", "-".repeat(45));

        let prompts = ["9-1=", "3+4=", "5-2=", "7+2=", "1+1=", "8-3=", "6+3=", "4-0=", "0+5=", "9-9="];
        let expected = ["8", "7", "3", "9", "2", "5", "9", "4", "5", "0"];
        let mut lm_correct = 0;
        let mut phase_correct = 0;

        for (prompt, exp) in prompts.iter().zip(expected.iter()) {
            let tokens = encode(prompt);
            let (phase_tok, lm_tok, coherences) = common::phase_decode::phase_decode_compare(
                &model, &tokens, dims, &stencil,
            );
            let lm_ans = decode(lm_tok);
            let ph_ans = decode(phase_tok);
            let lm_ok = lm_ans == *exp;
            let ph_ok = ph_ans == *exp;
            if lm_ok { lm_correct += 1; }
            if ph_ok { phase_correct += 1; }
            println!("{:<10} {:<10} {:<10} {:<10}",
                prompt,
                exp,
                format!("{}{}", lm_ans, if lm_ok { " ✓" } else { " ✗" }),
                format!("{}{}", ph_ans, if ph_ok { " ✓" } else { " ✗" }),
            );
        }
        println!("{}", "-".repeat(45));
        println!("LM Head: {}/10    Phase decode: {}/10", lm_correct, phase_correct);
        return;
    }

    // ─── Generate mode ───
    if std::env::args().any(|a| a == "--generate") {
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--generate requires --resume <checkpoint>");
        let prompt = std::env::args().skip_while(|a| a != "--prompt").nth(1)
            .unwrap_or("The ".to_string());
        common::generate::run_generate(common::generate::GenerateConfig {
            resume_path,
            prompt,
            max_tokens: parse_flag("--max-tokens", 200),
            n_layers: parse_flag("--layers", N_LAYERS),
            n_bands: parse_flag("--n-bands", N_BANDS),
            n_head: parse_flag("--n-head", N_HEAD),
            out_proj_groups: parse_flag("--out-proj-groups", 6),
            use_bpe: std::env::args().any(|a| a == "--bpe"),
            tokenizer_path: parse_flag("--tokenizer", "data/tokenizer.json".to_string()),
            alpha: parse_flag("--alpha", 0.1),
            beta: parse_flag("--beta", parse_flag("--alpha", 0.1)),
            temperature: parse_flag("--temperature", 0.0),
            phase_native: std::env::args().any(|a| a == "--phase-native"),
        });
        return;
    }

    // ─── Serve mode (requires --features serve) ───
    #[cfg(feature = "serve")]
    if std::env::args().any(|a| a == "--serve") {
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--serve requires --resume <checkpoint>");
        let n_layers: usize = parse_flag("--layers", N_LAYERS);
        let n_bands: usize = parse_flag("--n-bands", N_BANDS);
        let n_head: usize = parse_flag("--n-head", N_HEAD);
        let out_proj_groups: usize = parse_flag("--out-proj-groups", 6);
        let alpha: f32 = parse_flag("--alpha", 0.1);
        let beta: f32 = parse_flag("--beta", parse_flag("--alpha", 0.1));
        let use_bpe = std::env::args().any(|a| a == "--bpe");
        let tokenizer_path: String = parse_flag("--tokenizer", "data/tokenizer.json".to_string());

        // Load checkpoint
        println!("Loading checkpoint: {resume_path}");
        let (params, ck_vocab, ck_iter, _lr, _rng, _at, _am, _av, _groups, _flags) =
            wave_checkpoint::load_checkpoint(&resume_path);

        // Build model
        let dims = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
            .with_learnable_ode(false).with_corrector(true);
        let dims_ext = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
            .with_corrector(true);
        let mut model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_ext, alpha, beta);
        let ext_count = common::wave_model::count_trainable_ex(&model, false);
        if params.len() == ext_count {
            common::wave_model::unflatten_params_ex(&mut model, &params, false);
            println!("  Loaded {} params (with ODE/corrector)", params.len());
        } else {
            let dims_base = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
                .with_learnable_ode(false).with_corrector(false);
            model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_base, alpha, beta);
            unflatten_params(&mut model, &params);
            println!("  Loaded {} params (base)", params.len());
        }
        model.learnable_ode = false;
        println!("  Model: {}L, {}bands, {}dim, {}vocab, iter {}",
            n_layers, n_bands, n_bands * 2, ck_vocab, ck_iter);

        // Init AGC from model coupling
        crate::ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);

        // Build vocab
        let vocab = if use_bpe {
            let bpe = bpe::BpeTokenizer::from_file(&tokenizer_path);
            println!("  BPE tokenizer: {} vocab", ck_vocab);
            serve_tier::prompt::Vocab::from_bpe(bpe, ck_vocab)
        } else {
            println!("  Char-level: {} vocab", ck_vocab);
            serve_tier::prompt::Vocab::from_chars(ck_vocab)
        };

        let stencil = fft_ode::StencilFft::new(n_bands);

        let state = std::sync::Arc::new(serve_tier::server::AppState {
            model: std::sync::Arc::new(model),
            vocab: std::sync::Arc::new(vocab),
            dims,
            stencil: std::sync::Arc::new(stencil),
            model_name: parse_flag("--model-name", "wave-engine".to_string()),
            api_key: std::env::args().skip_while(|a| a != "--api-key").nth(1),
            host: parse_flag("--host", "127.0.0.1".to_string()),
            port: parse_flag("--port", 8080),
        });

        serve_tier::server::run_server(state);
        return;
    }

    // ─── Training dispatch ───
    train::run_training(train::TrainConfig {
        data_path: std::env::args().nth(1).unwrap_or("data/input.txt".to_string()),
        n_iters: parse_flag("--iters", 500),
        batch_size: parse_flag("--batch", 4),
        seq_len: parse_flag("--seq", 256),
        n_layers: parse_flag("--layers", N_LAYERS),
        lr: parse_flag("--lr", if N_BANDS > 256 { 1e-4 } else { 3e-4 }),
        use_bpe: std::env::args().any(|a| a == "--bpe"),
        tokenizer_path: parse_flag("--tokenizer", "data/tokenizer.json".to_string()),
        resume_path: std::env::args().skip_while(|a| a != "--resume").nth(1),
        use_curriculum: !std::env::args().any(|a| a == "--no-curriculum"),
        use_gpu: std::env::args().any(|a| a == "--gpu"),
        use_monitor: std::env::args().any(|a| a == "--monitor"),
        out_proj_groups: parse_flag("--out-proj-groups", 6),
        checkpoint_name: parse_flag("--checkpoint-name", "checkpoint.bin".to_string()),
        n_bands: parse_flag("--n-bands", N_BANDS),
        n_head: parse_flag("--n-head", N_HEAD),
        alpha: parse_flag("--alpha", 0.1),
        beta: parse_flag("--beta", parse_flag("--alpha", 0.1)), // default beta = alpha
        agc_ceiling: std::env::args().skip_while(|a| a != "--agc-ceiling").nth(1)
            .and_then(|s| s.parse().ok()),
        log_name: std::env::args().skip_while(|a| a != "--log-name").nth(1),
        m1: std::env::args().skip_while(|a| a != "--m1").nth(1).and_then(|s| s.parse().ok()),
        m2: std::env::args().skip_while(|a| a != "--m2").nth(1).and_then(|s| s.parse().ok()),
        tied: std::env::args().any(|a| a == "--tied-embeddings"),
        lm_rank: parse_flag("--lm-rank", 0),
        wave_decode: std::env::args().any(|a| a == "--wave-decode"),
        unfreeze_phases: std::env::args().any(|a| a == "--unfreeze-phases"),
        health_interval: parse_flag("--health-interval", 0),
        freeze_ode: std::env::args().any(|a| a == "--freeze-ode"),
        head_lr_floor: parse_flag("--head-lr-floor", 0.0),
        no_corrector: false, // legacy — now controlled by --corrector flag
        layer_scale: parse_dyn_param("--layer-scale"),
        lr_scale: parse_dyn_param("--lr-scale"),
        phase_native: std::env::args().any(|a| a == "--phase-native"),
        phase_temp: parse_flag("--phase-temp", 1.0),
        pythagorean: std::env::args().any(|a| a == "--pythagorean"),
        spring_k: parse_flag("--spring", 0.1),
        active_layers: std::env::args().skip_while(|a| a != "--active-layers").nth(1).and_then(|s| s.parse().ok()),
        rk4_weights: parse_dyn_param("--rk4-weights"),
        wd: parse_dyn_param("--wd"),
        harmonics: parse_dyn_param("--harmonics"),
        agc_headroom: parse_dyn_param("--agc-headroom"),
        corrector: {
            // --corrector dyn | --corrector off | --no-corrector (legacy)
            let c = parse_dyn_param("--corrector");
            if matches!(c, train::DynParam::Off) && !std::env::args().any(|a| a == "--no-corrector") {
                train::DynParam::Dynamic // default: corrector ON (learnable)
            } else {
                c
            }
        },
    });
}
