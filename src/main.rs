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
    });
}
