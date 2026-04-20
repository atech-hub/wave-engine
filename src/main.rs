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
#[allow(dead_code)]
mod monitors;
mod cli;
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
pub use monitors::monitor;
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

fn main() {
    use clap::Parser;
    let cli = cli::Cli::parse();

    // Global --threads: init rayon pool before any subcommand runs.
    let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads = cli.threads.unwrap_or(available / 2).max(1);
    rayon::ThreadPoolBuilder::new().num_threads(n_threads).build_global().ok();

    match cli.command {
        cli::Command::Train(args) => cmd_train(args),
        cli::Command::TrainWaves(args) => cmd_train_waves(args),
        cli::Command::Generate(args) => cmd_generate(args),
        cli::Command::WaveGenerate(args) => cmd_wave_generate(args),
        cli::Command::Encode(args) => cmd_encode(args),
        cli::Command::ScanMemory(args) => cmd_scan_memory(args),
        cli::Command::GalaxyScan(args) => cmd_galaxy_scan(args),
        cli::Command::Verify(args) => cmd_verify(args),
        cli::Command::Analyze(args) => cmd_analyze(args),
        cli::Command::OdeMonitor(args) => cmd_ode_monitor(args),
        cli::Command::PhaseDecode(args) => cmd_phase_decode(args),
        cli::Command::ConvertDataset(args) => cmd_convert_dataset(args),
        cli::Command::Recommend(args) => cmd_recommend(args),
        cli::Command::ScaleCheckpoint(args) => cmd_scale_checkpoint(args),
        #[cfg(feature = "serve")]
        cli::Command::Serve(args) => cmd_serve(args),
    }
}

// ─── Runtime init ──────────────────────────────────────────────

fn init_runtime() {
    // Thread pool already set by main() via --threads. This stays as a no-op
    // banner so existing callers keep the startup log line.
    println!("wave-engine v0.1.0\n");
}

// ─── Clap command handlers ─────────────────────────────────────

fn cmd_verify(args: cli::VerifyArgs) {
    // Init thread pool
    let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    rayon::ThreadPoolBuilder::new().num_threads(available / 2).build_global().ok();

    match args.command {
        cli::VerifyCommand::Grad(ga) => {
            // Apply --no- overrides
            let attn_path = ga.attention_pathway && !ga.no_attention_pathway;
            let ode_path = ga.ode_pathway && !ga.no_ode_pathway;
            let check_mode = match ga.scope.as_str() {
                "tiny" | "exhaustive" => monitors::junctions::grad_check::CheckMode::Exhaustive,
                "sampled" => monitors::junctions::grad_check::CheckMode::PerSection { n_per_section: 5 },
                other => { eprintln!("Unknown scope: {}. Use tiny, sampled, or exhaustive.", other); return; }
            };
            let config = monitors::junctions::grad_check::GradCheckConfig {
                eps: ga.eps, rel_tol: ga.tol, mode: check_mode, verbose: ga.verbose,
                section_filter: None,
            };
            let m = &ga.model;
            crate::ffn_backend::init_agc(m.alpha, m.beta);

            match ga.mode.as_str() {
                "phase-native" => {
                    println!("Gradient check: phase-native, {}L, {}bands, {}head  [tier={}]",
                        m.layers, m.n_bands, m.n_head, ga.tier);
                    let tokens: Vec<usize> = (0..8).map(|i| i % m.vocab).collect();
                    let targets: Vec<usize> = (1..9).map(|i| i % m.vocab).collect();

                    if ga.tier.as_str() == "candle" {
                        // Candle J1: autograd analytical grads vs FD. Cross-
                        // entropy loss (matches Candle training). ODE runs on
                        // autograd tensor ops so every param gets a real grad
                        // (use_custom_op defaults to false in CandleWaveModel::new).
                        let label = format!("phase-native-candle");
                        let (fwd, bwd, params, labels) =
                            run_candle_phase_native_check(
                                tokens.clone(), targets.clone(),
                                m.layers, m.n_bands, m.n_head, m.vocab, m.alpha, m.beta,
                            );
                        let result = monitors::junctions::grad_check::check_gradients(
                            &label, fwd, bwd, &params, &labels, config,
                        );
                        monitors::junctions::grad_check::print_result(&result);
                        if !result.passed() { std::process::exit(1); }
                        return;
                    }

                    // ─── CPU tier: component monitors + CPU grad check ───
                    {
                        let dims = Dims::from_cli(m.n_bands, m.n_head, 16, 128, 16)
                            .with_learnable_ode(ga.learnable_ode)
                            .with_corrector(ga.learnable_ode || ode_path)
                            .with_attention_pathway(attn_path)
                            .with_ode_pathway(ode_path)
                            .with_split_band(ga.split_band);
                        let mut model = init_model(m.vocab, 42, m.layers, 1, dims, m.alpha, m.beta);
                        model.phase_native = true;
                        model.output_corrector = vec![0.0; m.n_bands];
                        let stencil = fft_ode::StencilFft::new(m.n_bands);
                        let cache = crate::cpu::forward::forward_with_cache(
                            &model, &tokens, dims, None, None, None, Some(&stencil), None, None, None,
                        );
                        let (_loss, grads) = crate::cpu::model_backward::backward(
                            &model, &cache, &targets, dims, None, None, None,
                        );
                        let flow = monitors::gradient_monitor::analyze_gradients(&grads, dims);
                        eprintln!("\n[gradient_monitor] Per-component gradient norms:");
                        for s in &flow {
                            eprintln!("  layer {}:", s.layer);
                            eprintln!("    ln_grad_norm          {:.2e}", s.ln_grad_norm);
                            eprintln!("    maestro_in_grad_norm  {:.2e}", s.maestro_in_grad_norm);
                            eprintln!("    ode_grad_norm         {:.2e}", s.ode_grad_norm);
                            eprintln!("    maestro_out_grad_norm {:.2e}", s.maestro_out_grad_norm);
                            eprintln!("    out_proj_grad_norm    {:.2e}", s.out_proj_grad_norm);
                            eprintln!("    alpha_grad            {:.2e}", s.alpha_grad);
                            eprintln!("    beta_grad             {:.2e}", s.beta_grad);
                            eprintln!("    corrector_grad_norm   {:.2e}", s.corrector_grad_norm);
                        }
                        eprintln!();
                    }

                    let (fwd, bwd, params, labels) = cpu::grad_check_wrapper::phase_native_check(
                        tokens, targets, m.layers, m.n_bands, m.n_head, m.vocab, m.alpha, m.beta,
                        attn_path, ga.learnable_ode, ode_path, ga.split_band, ga.learnable_attn,
                    );
                    let result = monitors::junctions::grad_check::check_gradients(
                        "phase-native", fwd, bwd, &params, &labels, config,
                    );
                    monitors::junctions::grad_check::print_result(&result);
                    if !result.passed() { std::process::exit(1); }
                }
                "wave-input" => {
                    println!("Gradient check: wave-input, {}L, {}bands, {}head", m.layers, m.n_bands, m.n_head);
                    let n_embd = m.n_bands * 2;
                    let mut rng = crate::rng::Rng::new(42);
                    let inputs: Vec<Vec<f32>> = (0..4).map(|_| (0..n_embd).map(|_| rng.uniform(1.0)).collect()).collect();
                    let targets: Vec<Vec<f32>> = (0..4).map(|_| (0..n_embd).map(|_| rng.uniform(1.0)).collect()).collect();
                    let (fwd, bwd, params, labels) = cpu::grad_check_wrapper::wave_input_check(
                        inputs, targets, m.layers, m.n_bands, m.n_head, m.vocab, m.alpha, m.beta,
                    );
                    let result = monitors::junctions::grad_check::check_gradients(
                        "wave-input", fwd, bwd, &params, &labels, config,
                    );
                    monitors::junctions::grad_check::print_result(&result);
                    if !result.passed() { std::process::exit(1); }
                }
                other => {
                    eprintln!("Unknown grad-check mode: {}. Supported: phase-native, wave-input", other);
                }
            }
        }
        cli::VerifyCommand::TierParity(tp) => cmd_verify_tier_parity(tp),
    }
}

// Thin feature gate for the Candle phase-native grad check. Default build
// errors out with a build hint; candle-backend build routes to the real one.
#[cfg(feature = "candle-backend")]
fn run_candle_phase_native_check(
    tokens: Vec<usize>, targets: Vec<usize>,
    n_layers: usize, n_bands: usize, n_head: usize, vocab_size: usize,
    alpha: f32, beta: f32,
) -> (
    impl Fn(&[f32]) -> f64,
    impl Fn(&[f32]) -> (f32, Vec<f32>),
    Vec<f32>,
    monitors::junctions::grad_check::SectionLabels,
) {
    candle_tier::candle_grad_check::grad_check::phase_native_check_candle(
        tokens, targets, n_layers, n_bands, n_head, vocab_size, alpha, beta,
    )
}

#[cfg(not(feature = "candle-backend"))]
fn run_candle_phase_native_check(
    _tokens: Vec<usize>, _targets: Vec<usize>,
    _n_layers: usize, _n_bands: usize, _n_head: usize, _vocab_size: usize,
    _alpha: f32, _beta: f32,
) -> (
    impl Fn(&[f32]) -> f64,
    impl Fn(&[f32]) -> (f32, Vec<f32>),
    Vec<f32>,
    monitors::junctions::grad_check::SectionLabels,
) {
    eprintln!("verify grad --tier candle requires: cargo build --features candle-backend");
    std::process::exit(2);
    // Unreachable — keep the type signature happy with never-returning fns.
    #[allow(unreachable_code)]
    (
        |_: &[f32]| -> f64 { unreachable!() },
        |_: &[f32]| -> (f32, Vec<f32>) { unreachable!() },
        Vec::new(),
        monitors::junctions::grad_check::SectionLabels::new(Vec::new()),
    )
}

fn cmd_verify_tier_parity(args: cli::VerifyTierParityArgs) {
    use crate::monitors::junctions::tier_parity::print_report;
    use crate::monitors::junctions::tier_parity_runner::{run_cpu_vs_wgpu_parity, run_cpu_vs_candle_parity};

    let m = &args.model;
    ffn_backend::init_agc(m.alpha, m.beta);

    let dims = Dims::from_cli(m.n_bands, m.n_head, m.maestro_dim, 128, crate::RK4_STEPS)
        .with_corrector(true)
        .with_split_band(args.split_band);

    let model = if let Some(ref ckpt) = args.resume {
        println!("[J10] Loading checkpoint: {}", ckpt);
        let (model, _dims_loaded) = common::wave_model::load_checkpoint_auto(
            ckpt, m.n_bands, m.n_head, m.layers, m.out_proj_groups, m.alpha, m.beta,
        );
        model
    } else {
        println!("[J10] Using random-init model (seed=42) — no checkpoint.");
        let mut model = init_model(m.vocab, 42, m.layers, m.out_proj_groups, dims, m.alpha, m.beta);
        model.phase_native = true;
        model.output_corrector = vec![0.0; m.n_bands];
        model
    };

    // Deterministic token sequence: 0..seq wrapped mod vocab.
    let tokens: Vec<usize> = (0..args.seq).map(|i| i % m.vocab).collect();
    let tier = args.tier.as_str();
    println!("[J10] Running CPU vs {} parity: {} tokens, {}L, {} bands, {} vocab",
        tier, tokens.len(), m.layers, m.n_bands, m.vocab);

    let run_once = || -> Result<crate::monitors::junctions::tier_parity::ParityReport, String> {
        match tier {
            "wgpu" => Ok(run_cpu_vs_wgpu_parity(&model, &tokens, dims)),
            "candle" => run_cpu_vs_candle_parity(
                &model, &tokens, dims,
                m.alpha, m.beta, model.phase_native,
                /*use_rk4_dyn*/ false, /*use_layer_scale*/ false, /*use_harmonics*/ false,
            ),
            other => Err(format!("Unknown tier '{}'. Use wgpu or candle.", other)),
        }
    };

    let mut worst_report: Option<crate::monitors::junctions::tier_parity::ParityReport> = None;
    for i in 0..args.iters {
        let report = match run_once() {
            Ok(r) => r,
            Err(e) => { eprintln!("[J10] {}", e); std::process::exit(1); }
        };
        if args.verbose || !report.passed() {
            println!("\n── Run {}/{} ──", i + 1, args.iters);
            print_report(&report, false);
            if args.verbose {
                for sec in &report.sections {
                    println!(
                        "  [{}] {} elem  viol={}  max_abs={:.3e}  max_rel={:.3e}  mean_abs={:.3e}",
                        sec.section, sec.n_elements, sec.n_violations,
                        sec.max_abs_diff, sec.max_rel_diff, sec.mean_abs_diff,
                    );
                }
            }
        }
        if worst_report.as_ref().map_or(true, |r| r.n_violations() < report.n_violations()) {
            worst_report = Some(report);
        }
    }

    let report = worst_report.expect("at least one iteration");
    println!("\n=== J10 CPU vs {} parity summary ===", tier);
    print_report(&report, false);
    if !report.passed() { std::process::exit(1); }
}

fn cmd_recommend(args: cli::RecommendArgs) {
    common::recommend::run_recommend(&args.data);
}

fn cmd_scan_memory(args: cli::ScanMemoryArgs) {
    println!("Scanning memory: {}", args.file);
    let mem = kerr_memory::file::load(&args.file)
        .expect("Failed to load memory file");
    let scans = common::wave_memory::scan_memory(&mem);
    common::wave_memory::print_memory_scan(&mem, &scans);

    if let Some(out_path) = args.output {
        common::wave_memory::write_memory_scan_json(&out_path, &mem, &scans)
            .unwrap_or_else(|e| eprintln!("Error writing {}: {}", out_path, e));
        println!("\n  JSON written to: {}", out_path);
    }
}

fn cmd_galaxy_scan(args: cli::GalaxyScanArgs) {
    init_runtime();
    let m = &args.model;
    common::galaxy_scan::run_galaxy_scan_cli(
        &args.checkpoint.resume, m.n_bands, m.n_head, m.layers,
        m.out_proj_groups, m.alpha, m.beta, args.scan_corpus, args.m1, args.m2,
    );
}

fn cmd_generate(args: cli::GenerateArgs) {
    init_runtime();

    let m = &args.model;
    common::generate::run_generate(common::generate::GenerateConfig {
        resume_path: args.checkpoint.resume,
        prompt: args.prompt,
        max_tokens: args.max_tokens,
        n_layers: m.layers,
        n_bands: m.n_bands,
        n_head: m.n_head,
        out_proj_groups: m.out_proj_groups,
        maestro_dim: m.maestro_dim,
        use_bpe: args.bpe,
        tokenizer_path: args.tokenizer,
        alpha: m.alpha,
        beta: m.beta,
        temperature: args.temperature,
        phase_native: args.phase_native,
        memory_path: args.memory,
        diagnose: false,
    });
}

fn cmd_wave_generate(args: cli::WaveGenerateArgs) {
    init_runtime();

    let m = &args.model;

    // Teacher-force mode
    if let Some(ref kwds_path) = args.teacher_force {
        cmd_teacher_force(kwds_path, &args);
        return;
    }

    common::generate::run_wave_generate(common::generate::GenerateConfig {
        resume_path: args.checkpoint.resume,
        prompt: args.prompt,
        max_tokens: args.max_tokens,
        n_layers: m.layers,
        n_bands: m.n_bands,
        n_head: m.n_head,
        out_proj_groups: m.out_proj_groups,
        maestro_dim: m.maestro_dim,
        use_bpe: args.bpe,
        tokenizer_path: args.tokenizer,
        alpha: m.alpha,
        beta: m.beta,
        temperature: args.temperature,
        phase_native: true,
        memory_path: None,
        diagnose: args.wave_diagnose,
    });
}

fn cmd_teacher_force(kwds_path: &str, args: &cli::WaveGenerateArgs) {
    let m = &args.model;
    common::generate::run_teacher_force(
        &args.checkpoint.resume, kwds_path, &args.data,
        m.layers, m.n_head, m.vocab, m.alpha, m.beta,
    );
}

fn cmd_analyze(args: cli::AnalyzeArgs) {
    init_runtime();

    let m = &args.model;
    common::analyze::run_analyze(&args.checkpoint.resume, m.layers, m.out_proj_groups,
        args.bpe, &args.tokenizer, m.n_bands, m.n_head, m.alpha, m.beta, args.sub_harmonic);
}

fn cmd_convert_dataset(args: cli::ConvertDatasetArgs) {
    let m = &args.model;
    println!("Converting dataset: {}", args.data);

    let tok_path = if args.bpe { Some(args.tokenizer.as_str()) } else { None };
    let (tokens, vs) = common::data_loader::load_data(&args.data, args.bpe, tok_path);
    let vocab_size = vs.min(m.vocab);

    if args.per_position {
        // KWDS per-position wave conversion (embeddings + positional, no ODE)
        let dims = Dims::from_cli(m.n_bands, m.n_head, m.maestro_dim, 128, 16);
        let model = init_model(vocab_size, 42, m.layers, m.out_proj_groups, dims, m.alpha, m.beta);
        common::kwds::convert_tokens_to_kwds(&args.output, &tokens, &model.wte, &model.wpe, m.n_bands)
            .expect("KWDS conversion failed");
        println!("  Written to: {}", args.output);
        return;
    }

    // ─── KWMF aggregate mode ───
    // Run tokens through the model in block-size chunks, average per-layer ODE
    // states across positions, merge into persistent wave memory.
    let dims = Dims::from_cli(m.n_bands, m.n_head, m.maestro_dim, 128, 16);
    let model = if let Some(ref ckpt) = args.resume {
        println!("Converting dataset through trained model: {}", ckpt);
        let (mut model, _dims_loaded) = common::wave_model::load_checkpoint_auto(
            ckpt, m.n_bands, m.n_head, m.layers, m.out_proj_groups, m.alpha, m.beta,
        );
        model.learnable_ode = false;
        model
    } else {
        println!("Converting dataset through UNTRAINED model (random init)");
        let mut model = init_model(vocab_size, 42, m.layers, m.out_proj_groups, dims, m.alpha, m.beta);
        model.phase_native = true;
        model.output_corrector = vec![0.0; m.n_bands];
        model.learnable_ode = false;
        model
    };

    ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);
    let stencil = fft_ode::StencilFft::new(m.n_bands);

    let mut mem = common::wave_memory::load_or_create(&args.output, m.layers, m.n_bands);

    let block_size = 128;
    let n_chunks = (tokens.len().saturating_sub(1)) / block_size;
    println!("  Processing {} tokens in {} chunks of {}...", tokens.len(), n_chunks, block_size);

    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * block_size;
        let end = (start + block_size).min(tokens.len());
        let chunk = &tokens[start..end];

        let cache = cpu::forward::forward_with_cache(
            &model, chunk, dims, None, None, None, Some(&stencil), None, None, None,
        );

        // Per-layer ODE states averaged across positions
        let ode_states: Vec<(Vec<f32>, Vec<f32>)> = cache.block_caches.iter().map(|bc| {
            let t = bc.input.len().max(1);
            let mut avg_r = vec![0.0f32; m.n_bands];
            let mut avg_s = vec![0.0f32; m.n_bands];
            for pos in &bc.input {
                for k in 0..m.n_bands.min(pos.len() / 2) {
                    avg_r[k] += pos[k * 2];
                    avg_s[k] += pos[k * 2 + 1];
                }
            }
            let scale = 1.0 / t as f32;
            for k in 0..m.n_bands { avg_r[k] *= scale; avg_s[k] *= scale; }
            (avg_r, avg_s)
        }).collect();

        common::wave_memory::merge_ode_states(&mut mem, &ode_states);

        if (chunk_idx + 1) % 100 == 0 {
            println!("    {}/{} chunks processed", chunk_idx + 1, n_chunks);
        }
    }

    common::wave_memory::save(&args.output, &mem);
    println!("  Dataset converted: {} chunks → {} conversations in {}", n_chunks, mem.n_convos, args.output);

    let scans = common::wave_memory::scan_memory(&mem);
    common::wave_memory::print_memory_scan(&mem, &scans);
}

fn cmd_train_waves(args: cli::TrainWavesArgs) {
    init_runtime();
    let m = &args.model;
    cpu::wave_training::run(cpu::wave_training::WaveTrainConfig {
        kwds_path: args.kwds,
        n_layers: m.layers, n_head: m.n_head, n_bands: m.n_bands,
        out_proj_groups: m.out_proj_groups, vocab: m.vocab,
        alpha: m.alpha, beta: m.beta,
        iters: args.iters, lr: args.lr, seq: args.seq,
        checkpoint_name: args.checkpoint_name,
        resume: args.resume,
    });
}

fn cmd_train(args: cli::TrainArgs) {
    // Init thread pool (banner only; pool set in main)
    init_runtime();

    let m = &args.model;
    // --no-corrector (legacy) OR --corrector off both disable the corrector.
    let no_corrector = args.no_corrector || matches!(args.corrector, train::DynParam::Off);
    let corrector = if args.no_corrector { train::DynParam::Off } else { args.corrector };

    // Curriculum semantics: default ON (matches legacy). --no-curriculum disables.
    // --curriculum kept as explicit opt-in for scripts that set it.
    let use_curriculum = !args.no_curriculum;

    // Pathway semantics (same convention as `verify grad`): default true,
    // --no-* flips off for A/B training experiments.
    let ode_pathway = if args.no_ode_pathway { false } else { args.ode_pathway };
    let attention_pathway = if args.no_attention_pathway { false } else { args.attention_pathway };

    // Build TrainConfig once. Both the Candle path and the CPU/wgpu path read
    // from the same struct — no env-args scanning, no duplication.
    let config = train::TrainConfig {
        data_path: args.data.clone(),
        n_iters: args.iters,
        batch_size: args.batch,
        seq_len: args.seq,
        n_layers: m.layers,
        lr: args.lr,
        use_bpe: args.bpe,
        tokenizer_path: args.tokenizer.clone(),
        resume_path: args.resume.clone(),
        use_curriculum,
        use_gpu: args.gpu,
        use_monitor: args.monitor,
        out_proj_groups: m.out_proj_groups,
        checkpoint_name: args.checkpoint_name.unwrap_or("checkpoint.bin".to_string()),
        n_bands: m.n_bands,
        n_head: m.n_head,
        maestro_dim: m.maestro_dim,
        alpha: m.alpha,
        beta: m.beta,
        agc_ceiling: args.agc_ceiling,
        log_name: args.log_name,
        m1: args.m1,
        m2: args.m2,
        tied: args.tied_embeddings,
        lm_rank: args.lm_rank,
        wave_decode: args.wave_decode,
        unfreeze_phases: args.unfreeze_phases,
        health_interval: args.health_interval,
        freeze_ode: args.freeze_ode,
        head_lr_floor: args.head_lr_floor,
        no_corrector,
        layer_scale: args.layer_scale,
        lr_scale: args.lr_scale,
        phase_native: args.phase_native,
        fwm_strength: args.chi,
        phase_temp: args.phase_temp,
        pythagorean: args.pythagorean,
        spring_k: args.spring,
        active_layers: args.active_layers,
        rk4_weights: args.rk4_weights,
        wd: args.wd,
        harmonics: args.harmonics,
        agc_headroom: args.agc_headroom,
        corrector,
        split_band: args.split_band,
        ode_pathway,
        attention_pathway,
        learnable_attn: args.learnable_attn,
        candle: args.candle,
        cuda_kernel: args.cuda_kernel,
        custom_op: args.custom_op || args.cuda_kernel, // --cuda-kernel implies --custom-op
        gpu_duty: args.gpu_duty.clamp(1, 100),
        debug_nan: args.debug_nan,
    };

    // Candle path routes through candle_tier. Everything else runs the CPU/wgpu loop.
    if config.candle || config.cuda_kernel {
        let ffn = common::ffn_config::FfnConfig::from_flags(
            config.ode_pathway,
            config.split_band,
            config.freeze_ode,
            !config.no_corrector,
        );
        match candle_engine::engine::train_candle(&config, &ffn) {
            Ok(()) => return,
            Err(e) => { eprintln!("Candle error: {e:?}"); std::process::exit(1); }
        }
    }

    train::run_training(config);
}

fn cmd_encode(args: cli::EncodeArgs) {
    let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    rayon::ThreadPoolBuilder::new().num_threads(available / 2).build_global().ok();

    let m = &args.model;
    let n_bands = m.n_bands;
    let n_embd = n_bands * 2;
    let dims = Dims::from_cli(n_bands, m.n_head, 16, 128, 16);

    // Load model or create blank
    let model = if args.blank {
        println!("Creating blank (untrained) model: {}L, {}bands", m.layers, n_bands);
        init_model(128, 42, m.layers, m.out_proj_groups, dims, m.alpha, m.beta)
    } else {
        let resume = args.resume.as_ref().expect("encode requires --resume <checkpoint> or --blank");
        println!("Loading checkpoint: {}", resume);
        let (params, ck_vocab, _, _, _, _, _, _, _, _, _) = wave_checkpoint::load_checkpoint(resume);
        let mut model = init_model(ck_vocab, 42, m.layers, m.out_proj_groups, dims, m.alpha, m.beta);
        let ext_count = count_trainable_ex(&model, false);
        if params.len() < ext_count {
            model.phase_native = true;
            model.output_corrector = vec![0.0; n_bands];
        }
        unflatten_params_ex(&mut model, &params, false);
        println!("  Model: {}L, {}bands, {}vocab, phase_native={}",
            m.layers, n_bands, ck_vocab, model.phase_native);
        model
    };

    crate::ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);

    // Build char map
    let char_map: Vec<char> = if !args.blank && std::path::Path::new(&args.data).exists() {
        let text = common::data_loader::load_text_raw(&args.data);
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort(); chars.dedup();
        chars.truncate(model.vocab_size);
        println!("  Char map: {} chars from {}", chars.len(), args.data);
        chars
    } else {
        (0..model.vocab_size.min(128)).map(|i| i as u8 as char).collect()
    };
    let encode_char = |c: char| -> Option<usize> { char_map.iter().position(|&ch| ch == c) };
    let decode_tok = |tok: usize| -> String {
        if tok < char_map.len() {
            let c = char_map[tok];
            if c.is_ascii_graphic() || c == ' ' { format!("'{}'", c) } else { format!("t{}", tok) }
        } else { format!("t{}", tok) }
    };

    // ─── Relate-vocab ───
    if args.relate_vocab {
        let (labels, reports, dist, profiles) = common::phase_encode::run_relate_vocab(&model, n_bands, Some(&char_map));
        println!("\n=== Vocabulary Relationship Map ===");
        println!("  {} tokens, {} pairs", labels.len(), reports.len());
        println!("\n  Catalog distribution:");
        let mut entries: Vec<_> = dist.iter().collect();
        entries.sort_by_key(|e| std::cmp::Reverse(*e.1));
        for (name, count) in &entries { println!("    {:20} {:5}", name, count); }
        println!("\n  Energy signatures (top amplifiers / dampeners):");
        let mut sorted_profiles: Vec<&common::phase_encode::EnergyProfile> = profiles.iter().collect();
        sorted_profiles.sort_by(|a, b| b.peak_ratio.partial_cmp(&a.peak_ratio).unwrap_or(std::cmp::Ordering::Equal));
        for p in sorted_profiles.iter().take(5) {
            println!("    {:>4}  energy={:.2}x  peak=band{}({:.1}x)  damp=band{}({:.2}x)",
                p.label, p.total_energy_ratio, p.peak_band, p.peak_ratio, p.damp_band, p.damp_ratio);
        }
        let phase_scores: std::collections::HashMap<String, f32> = labels.iter().enumerate().map(|(_i, label)| {
            let total = reports.iter().filter(|r| r.label_a == *label || r.label_b == *label).count();
            let non_conj = reports.iter().filter(|r| {
                (r.label_a == *label || r.label_b == *label)
                && r.catalog_match.as_ref().map(|(n, _)| &**n != "conjunction").unwrap_or(false)
            }).count();
            (label.clone(), if total > 0 { non_conj as f32 / total as f32 } else { 0.0 })
        }).collect();
        let axis_scores = common::catalog_axes::compute_all_axes(&model, n_bands, &char_map, &phase_scores);
        let corr = common::catalog_axes::correlation_matrix(&axis_scores);
        common::catalog_axes::print_axes_summary(&axis_scores, &corr);
        let destruction_profile = common::catalog_axes::compute_destruction_profile(&model, n_bands, args.m1, args.m2);
        println!("\n  Targeted destruction profile:");
        for l in &destruction_profile.per_layer {
            println!("    L{}: on={:.3} off={:.3} ratio={:.2}x", l.layer, l.on_grid_cos, l.off_grid_cos, l.ratio);
        }
        if let Err(e) = common::phase_encode::write_vocab_relations_json(&args.output, &labels, &reports, &dist, Some(&profiles)) {
            eprintln!("Error writing {}: {}", args.output, e);
        } else { println!("\n  Written to: {}", args.output); }
        return;
    }

    // ─── Relate pairwise ───
    if !args.relate.is_empty() || !args.relate_number.is_empty() || !args.relate_catalog.is_empty() {
        let mut items: Vec<(String, Vec<f32>)> = Vec::new();
        for text in &args.relate {
            let tokens: Vec<usize> = text.chars().filter_map(|c| encode_char(c)).collect();
            let mut h = vec![0.0f32; n_embd];
            if let Some(&tok) = tokens.last() {
                let pos = (tokens.len() - 1).min(model.wpe.len() - 1);
                for j in 0..n_embd { h[j] = model.wte[tok][j] + model.wpe[pos][j]; }
            }
            items.push((text.clone(), h));
        }
        for n in &args.relate_number {
            let state = common::phase_encode::encode_number(*n, n_bands, args.m1, args.m2);
            items.push((format!("{}", n), state));
        }
        for spec in &args.relate_catalog {
            let configs = common::phase_encode::parse_catalog_spec(spec);
            let state = common::phase_encode::encode_catalog_state(&configs, n_bands);
            items.push((spec.clone(), state));
        }
        if items.len() < 2 {
            eprintln!("Relate requires at least 2 items.");
            std::process::exit(1);
        }
        let reports = common::phase_encode::run_relate(&model, &items, n_bands);
        for r in &reports { common::phase_encode::print_relate_report(r); }
        if items.len() > 2 {
            let labels: Vec<String> = items.iter().map(|(l, _)| l.clone()).collect();
            common::phase_encode::print_relate_matrix(&labels, &reports);
        }
        return;
    }

    // ─── Single encode ───
    let (_label, encoded, configs) = if let Some(ref text) = args.encode {
        let tokens: Vec<usize> = text.chars().filter_map(|c| encode_char(c)).collect();
        println!("  Tokens: {:?} (from \"{}\")", tokens, text);
        let mut states: Vec<Vec<f32>> = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; n_embd];
            if pos < model.wpe.len() { for j in 0..n_embd { h[j] = model.wte[tok][j] + model.wpe[pos][j]; } }
            states.push(h);
        }
        (format!("Text: \"{}\"", text), states.last().cloned().unwrap_or(vec![0.0; n_embd]), vec![])
    } else if let Some(n) = args.encode_number {
        let state = common::phase_encode::encode_number(n, n_bands, args.m1, args.m2);
        (format!("Number: {}", n), state, vec![])
    } else if let Some(ref spec) = args.encode_catalog {
        let configs = common::phase_encode::parse_catalog_spec(spec);
        let state = common::phase_encode::encode_catalog_state(&configs, n_bands);
        (format!("Catalog: {}", spec), state, configs)
    } else if let Some(ref spec) = args.encode_phases {
        let phases = common::phase_encode::parse_raw_phases(spec);
        let state = common::phase_encode::encode_raw_phases(&phases, n_bands);
        (format!("Raw phases: {}", spec), state, vec![])
    } else {
        eprintln!("No encoding specified. Use --encode, --encode-number, --encode-catalog, --encode-phases, --relate, or --relate-vocab.");
        std::process::exit(1);
    };

    let highlight: Vec<usize> = configs.iter().flat_map(|c| c.bands.iter().copied()).collect();
    common::phase_encode::print_encoded_state(&format!("Encoded state (input to layer {})", args.inject_layer), &encoded, &highlight);

    let (final_out, per_layer) = common::phase_encode::run_encode(&model, &encoded, args.inject_layer, n_bands);
    common::phase_encode::print_layer_cosines(&encoded, &per_layer);
    common::phase_encode::print_encoded_state(&format!("After ODE evolution (output of layer {})", per_layer.len().saturating_sub(1)), &final_out, &highlight);
    common::phase_encode::print_comparison(&encoded, &final_out, &configs);

    if !args.blank {
        println!("\n=== Decoder readout ===");
        let mut scores: Vec<(usize, f32)> = (0..model.vocab_size).map(|tok| {
            let dot: f32 = (0..n_embd).map(|i| model.lm_head[tok][i] * final_out[i]).sum();
            (tok, dot)
        }).collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        print!("  lm_head top 5:");
        for (tok, score) in scores.iter().take(5) { print!("  {} ({:.3})", decode_tok(*tok), score); }
        println!();
        let mut phase_scores: Vec<(usize, f32)> = (0..model.vocab_size).map(|tok| {
            let dot: f32 = (0..n_embd).map(|i| model.wte[tok][i] * final_out[i]).sum();
            (tok, dot)
        }).collect();
        phase_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        print!("  phase-native top 5:");
        for (tok, score) in phase_scores.iter().take(5) { print!("  {} ({:.3})", decode_tok(*tok), score); }
        println!();
    }

    if args.scan {
        println!("\n  Running galaxy scan on output...");
        let all_hidden: Vec<Vec<Vec<f32>>> = per_layer.iter().map(|h| vec![h.clone()]).collect();
        let post_ln_f = vec![final_out.clone()];
        let per_layer_ceilings: Vec<f32> = model.blocks.iter()
            .map(|b| (std::f32::consts::FRAC_PI_2 / (b.ffn.kerr.alpha + 4.0 * b.ffn.kerr.beta)).sqrt().max(0.5))
            .collect();
        let scan_dir = std::path::PathBuf::from("encode_output_galaxy");
        match common::galaxy_scan::run_and_write_full_scan(
            &all_hidden, &post_ln_f, n_bands, &per_layer_ceilings, args.m1, args.m2, &scan_dir,
        ) {
            Ok(scan) => { println!("Galaxy map written to: {}", scan_dir.display()); common::galaxy_scan::print_summary(&scan); }
            Err(e) => { eprintln!("Error: {}", e); }
        }
    }
}

fn cmd_ode_monitor(args: cli::OdeMonitorArgs) {
    init_runtime();

    let m = &args.model;
    let (mut model, dims) = common::wave_model::load_checkpoint_auto(
        &args.checkpoint.resume, m.n_bands, m.n_head, m.layers, m.out_proj_groups, m.alpha, m.beta,
    );
    model.learnable_ode = false;
    let ck_vocab = model.vocab_size;
    let stencil = fft_ode::StencilFft::new(m.n_bands);
    crate::ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);

    let text = common::data_loader::load_text_raw(&args.data);
    let mut chars: Vec<char> = text.chars().collect();
    chars.sort(); chars.dedup();
    let char_map: Vec<char> = chars[..chars.len().min(ck_vocab)].to_vec();
    let encode = |s: &str| -> Vec<usize> {
        s.chars().filter_map(|c| char_map.iter().position(|&ch| ch == c)).collect()
    };
    let decode = |id: usize| -> String {
        if id < char_map.len() { char_map[id].to_string() } else { "?".to_string() }
    };

    let prompt = args.prompt.unwrap_or("3+4=".to_string());
    println!("=== ODE Monitor ===\n");
    let tokens = encode(&prompt);
    monitors::ode_monitor::print_ode_summary(&model, &tokens, dims, &stencil, &prompt, &decode);

    if args.compare.len() >= 1 {
        let tokens_b = encode(&args.compare[0]);
        monitors::ode_monitor::compare_prompts(&model, &tokens, &tokens_b, dims, &stencil, &prompt, &args.compare[0]);
    }
}

fn cmd_phase_decode(args: cli::PhaseDecodeArgs) {
    init_runtime();

    let m = &args.model;
    let (mut model, dims) = common::wave_model::load_checkpoint_auto(
        &args.checkpoint.resume, m.n_bands, m.n_head, m.layers, m.out_proj_groups, m.alpha, m.beta,
    );
    model.learnable_ode = false;
    let ck_vocab = model.vocab_size;
    let stencil = fft_ode::StencilFft::new(m.n_bands);
    crate::ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);

    let text = common::data_loader::load_text_raw(&args.data);
    let mut chars: Vec<char> = text.chars().collect();
    chars.sort(); chars.dedup();
    let char_map: Vec<char> = chars[..chars.len().min(ck_vocab)].to_vec();
    let encode = |s: &str| -> Vec<usize> {
        s.chars().filter_map(|c| char_map.iter().position(|&ch| ch == c)).collect()
    };
    let decode = |id: usize| -> String {
        if id < char_map.len() { char_map[id].to_string() } else { "?".to_string() }
    };

    common::phase_decode::run_diagnostic(&model, dims, &stencil, &encode, &decode, count_trainable_ex(&model, false), ck_vocab);
}

#[cfg(feature = "serve")]
fn cmd_serve(args: cli::ServeArgs) {
    init_runtime();

    let m = &args.model;
    let resume = args.checkpoint.resume.clone();
    println!("Loading checkpoint: {resume}");

    // Use the shared 4-variant auto-loader — handles phase-native × ODE × corrector.
    let (model, dims_serve) = common::wave_model::load_checkpoint_auto(
        &resume, m.n_bands, m.n_head, m.layers, m.out_proj_groups, m.alpha, m.beta,
    );
    let ck_vocab = model.vocab_size;
    println!("  Model: {}L, {}bands, {}dim, {}vocab",
        m.layers, m.n_bands, m.n_bands * 2, ck_vocab);

    ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);

    let vocab = if args.bpe {
        let bpe = common::bpe::BpeTokenizer::from_file(&args.tokenizer);
        println!("  BPE tokenizer: {} vocab from {}", ck_vocab, args.tokenizer);
        serve_tier::prompt::Vocab::from_bpe(bpe, ck_vocab)
    } else {
        // Char-level vocab must mirror the training tokenizer: sorted unique chars
        // from the training data file. Without this, ASCII-default chars (e.g.
        // control bytes at vocab=15) don't match the model's embedding table and
        // decoding produces garbage.
        let data_path = args.data.clone().unwrap_or_else(|| {
            eprintln!("ERROR: serve without --bpe requires --data <path-to-training-file>");
            std::process::exit(1);
        });
        let text = common::data_loader::load_text_raw(&data_path);
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let n = chars.len().min(ck_vocab);
        let char_map: Vec<char> = chars[..n].to_vec();
        println!("  Char-level vocab: {} chars from {}", n, data_path);
        serve_tier::prompt::Vocab {
            vocab_size: ck_vocab,
            bpe: None,
            char_map,
        }
    };

    let stencil = fft_ode::StencilFft::new(m.n_bands);

    let wave_mem = args.memory.as_ref().map(|path| {
        std::sync::Mutex::new(common::wave_memory::load_or_create(path, m.layers, m.n_bands))
    });

    // phase_native flag currently advisory — auto-loader detects from checkpoint shape.
    // Exposed for explicit control when the detection heuristic is ambiguous.
    let _ = args.phase_native;

    let state = std::sync::Arc::new(serve_tier::server::AppState {
        model: std::sync::Arc::new(model),
        vocab: std::sync::Arc::new(vocab),
        dims: dims_serve,
        stencil: std::sync::Arc::new(stencil),
        model_name: args.model_name.clone(),
        api_key: args.token.clone(),
        host: args.host.clone(),
        port: args.port,
        memory: wave_mem,
        memory_path: args.memory.clone(),
    });

    serve_tier::server::run_server(state);
}

fn cmd_scale_checkpoint(args: cli::ScaleCheckpointArgs) {
    common::scale::scale_checkpoint(&common::scale::ScaleConfig {
        source_path: args.checkpoint.resume,
        target_bands: args.tgt_bands,
        target_head: args.target_head,
        target_layers: args.target_layers,
        output_path: args.output,
        target_groups: args.out_proj_groups,
        seed: 42,
    }).unwrap_or_else(|e| { eprintln!("Scale error: {e}"); std::process::exit(1); });
}
