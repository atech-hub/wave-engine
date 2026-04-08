//! Wave Engine library — shared module tree for the main binary and tools.
//!
//! This lib.rs mirrors the module declarations from main.rs so that
//! src/bin/ binaries (wave-probe, etc.) can `use wave_engine::...`
//! to access the canonical functions. No path hacks, no duplicates.

#[allow(dead_code)]
pub mod common;
#[allow(dead_code)]
pub mod cpu;
#[allow(dead_code)]
pub mod wgpu_tier;
#[allow(dead_code)]
pub mod candle_tier;

// Re-export shim — matches main.rs re-exports
pub use common::model;
pub use common::embed as wave_embed;
pub use common::attn as wave_attn;
pub use common::block as wave_block;
pub use common::ffn as ffn_backend;
pub use common::wave_model::{WavePacketModel, init_model, init_linear, count_trainable, count_trainable_ex, flatten_params, flatten_params_ex, unflatten_params, unflatten_params_ex};
pub use common::dims::{Dims, PROFILE, N_BANDS, N_EMBD, N_HEAD, N_LAYERS, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS};
pub use cpu::forward::{BlockCache, ForwardCache, forward_with_cache, dual_maestro_forward};
pub use cpu::model_backward::{Gradients, flatten_grads, flatten_grads_ex};
pub use wgpu_tier::diagnostics::{diagnose_ode_gpu_vs_cpu, validate_gpu_fft};
pub use common::checkpoint as wave_checkpoint;
pub use common::rng;
pub use common::bpe;
pub use common::token_cache;
pub use common::monitor;
pub use common::data;
pub use common::data_loader;
pub use common::fft_ode;
pub use cpu::train;
pub use cpu::backward;
pub use wgpu_tier::backend;
pub use wgpu_tier::device as gpu;
pub use wgpu_tier::gpu_backend;
pub use wgpu_tier::buffers as gpu_buffers;
pub use wgpu_tier::dispatch as gpu_dispatch;
pub use wgpu_tier::ops_forward as gpu_ops_forward;
pub use wgpu_tier::ops_backward as gpu_ops_backward;
pub use wgpu_tier::pipelines as gpu_pipelines;
pub use wgpu_tier::resident as gpu_resident;
pub use wgpu_tier::validate as gpu_validate;
pub use wgpu_tier::ffn_gpu;
pub use wgpu_tier::ffn_full_gpu;
pub use candle_tier::engine as candle_engine;
pub use candle_tier::ode as gpu_ode;
pub use candle_tier::block_diag as block_diagonal;
