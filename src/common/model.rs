//! Full Kerr-ODE model — CPU reference implementation.
//!
//! Matches phaseC_integrated.py exactly for inference.
//! Architecture: 4 blocks, each with CausalSelfAttention + FFN.
//!   Block 0: Attention + PerBandLinear
//!   Blocks 1-3: Attention + KerrMaestroAdd (Kerr-ODE + Maestro)
//!
//! Split into submodules:
//!   model.rs          — weight structs, config, constants, linear algebra helpers
//!   out_proj.rs       — OutProjWeights enum + Dense/BlockDiagonal impls
//!   model_forward.rs  — forward pass (impl ModelWeights)
//!   ode_deriv.rs      — Kerr derivative, RK4 step

use std::f32::consts::PI;

// Re-export submodule contents so `use crate::model::*` keeps working.
pub use super::out_proj::{OutProjWeights, BlockDiagonalWeights};
pub use super::ode_deriv::rk4_step_public;
// model_forward.rs adds methods to ModelWeights via `impl` — no re-export needed.

// ─── Model configuration ──────────────────────────────────────

/// Runtime-configurable architecture dimensions.
/// Stored in ModelWeights so every function can derive dims from data or config.
#[derive(Clone, Copy, Debug)]
pub struct ModelConfig {
    pub n_bands: usize,
    pub n_head: usize,
    pub n_layers: usize,
    pub maestro_dim: usize,
    pub block_size: usize,
    pub rk4_n_steps: usize,
}

impl ModelConfig {
    /// Default config matching the original 128-dim architecture.
    pub fn default_128() -> Self {
        Self {
            n_bands: 64,
            n_head: 4,
            n_layers: 4,
            maestro_dim: 16,
            block_size: 256,
            rk4_n_steps: 8,
        }
    }

    pub fn n_embd(&self) -> usize { self.n_bands * 2 }
    #[allow(dead_code)]
    pub fn head_dim(&self) -> usize { self.n_embd() / self.n_head }
    #[allow(dead_code)]
    pub fn rk4_dt(&self) -> f32 { 1.0 / self.rk4_n_steps as f32 }

    pub fn validate(&self) {
        assert!(self.n_bands > 0, "n_bands must be > 0");
        assert!(self.n_head > 0, "n_head must be > 0");
        assert_eq!(self.n_embd() % self.n_head, 0, "n_embd must be divisible by n_head");
        assert!(self.rk4_n_steps > 0, "rk4_n_steps must be > 0");
    }
}

// Legacy compile-time constants — kept for init.rs defaults and backward compatibility.
// New code should use ModelConfig or derive from data dimensions.
pub const N_BANDS: usize = 64;
pub const N_EMBD: usize = 128;
pub const N_HEAD: usize = 4;
#[allow(dead_code)]
pub const HEAD_DIM: usize = N_EMBD / N_HEAD;
pub const BLOCK_SIZE: usize = 256;
pub const MAESTRO_DIM: usize = 16;
pub const RK4_N_STEPS: usize = 8;
#[allow(dead_code)]
pub const RK4_DT: f32 = 1.0 / RK4_N_STEPS as f32;
pub const N_LAYERS: usize = 4;

/// GELU activation (approximate version matching PyTorch default)
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// Build frozen harmonic embedding table.
/// Returns [vocab_size][n_embd] array.
#[allow(dead_code)]
pub fn build_harmonic_table(vocab_size: usize) -> Vec<Vec<f32>> {
    // n_embd derived from vocab table: caller sizes it via config, but the formula
    // only depends on vocab_size and n_bands. We use N_EMBD for the legacy path
    // and data-derived for the config path. Since this is called from init_model
    // which knows config, the table size is correct.
    let n_embd = N_EMBD; // Legacy — will be parameterized in Phase 2
    let nh = n_embd / 2;
    let scale = 1.0 / (nh as f32).sqrt();
    let mut table = vec![vec![0.0f32; n_embd]; vocab_size];

    for c in 0..vocab_size {
        let theta = c as f32 * 2.0 * PI / vocab_size as f32;
        for h in 0..nh {
            let angle = (h + 1) as f32 * theta;
            table[c][h * 2] = angle.cos() * scale;
            table[c][h * 2 + 1] = angle.sin() * scale;
        }
    }
    table
}

/// Build frozen harmonic embedding table with explicit n_embd.
pub fn build_harmonic_table_sized(vocab_size: usize, n_embd: usize) -> Vec<Vec<f32>> {
    let nh = n_embd / 2;
    let scale = 1.0 / (nh as f32).sqrt();
    let mut table = vec![vec![0.0f32; n_embd]; vocab_size];

    for c in 0..vocab_size {
        let theta = c as f32 * 2.0 * PI / vocab_size as f32;
        for h in 0..nh {
            let angle = (h + 1) as f32 * theta;
            table[c][h * 2] = angle.cos() * scale;
            table[c][h * 2 + 1] = angle.sin() * scale;
        }
    }
    table
}

/// Build positional encoding table.
/// Returns [block_size][n_embd] array.
pub fn build_positional_table(block_size: usize, n_embd: usize) -> Vec<Vec<f32>> {
    let nh = n_embd / 2;
    let scale = 1.0 / (nh as f32).sqrt();
    let mut table = vec![vec![0.0f32; n_embd]; block_size];

    for pos in 0..block_size {
        for h in 0..nh {
            let freq = 1.0 / 10000.0_f32.powf(2.0 * h as f32 / n_embd as f32);
            table[pos][h * 2] = (pos as f32 * freq).cos() * scale;
            table[pos][h * 2 + 1] = (pos as f32 * freq).sin() * scale;
        }
    }
    table
}

// ─── Linear algebra helpers ──────────────────────────────────────

/// Matrix-vector multiply: y = W @ x + b
/// W is [out_dim][in_dim], x is [in_dim], b is [out_dim]
#[inline]
fn linear(w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
    w.iter()
        .zip(b.iter())
        .map(|(row, &bias)| {
            bias + row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum::<f32>()
        })
        .collect()
}

/// Layer normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let std = (var + 1e-5).sqrt();

    let mut y = vec![0.0f32; n];
    for i in 0..n {
        y[i] = (x[i] - mean) / std * weight[i] + bias[i];
    }
    y
}

/// Public wrapper for linear (needed by backward.rs, compute.rs, model_forward.rs).
#[inline]
pub fn linear_fn(w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
    linear(w, b, x)
}

// ─── Weight structures ──────────────────────────────────────────

/// Weights for a Linear layer.
#[derive(Clone)]
pub struct LinearWeights {
    pub w: Vec<Vec<f32>>,  // [out_dim][in_dim]
    pub b: Vec<f32>,       // [out_dim]
}

/// Weights for LayerNorm.
#[derive(Clone)]
pub struct LayerNormWeights {
    pub weight: Vec<f32>,  // [dim]
    pub bias: Vec<f32>,    // [dim]
}

/// Weights for CausalSelfAttention.
#[derive(Clone)]
pub struct AttentionWeights {
    pub c_attn: LinearWeights,  // [3*N_EMBD, N_EMBD]
    pub c_proj: LinearWeights,  // [N_EMBD, N_EMBD]
    pub n_head: usize,          // Number of attention heads (for head_dim derivation)
}

/// Weights for PerBandLinear (Block 0 FFN).
#[derive(Clone)]
pub struct PerBandLinearWeights {
    pub band_w: Vec<[[f32; 2]; 2]>,  // [N_BANDS][2][2]
    pub band_b: Vec<[f32; 2]>,       // [N_BANDS][2]
    pub out_proj: LinearWeights,
}

/// Weights for Kerr-ODE layer.
#[derive(Clone)]
pub struct KerrWeights {
    pub gamma_raw: Vec<f32>,  // [N_BANDS] (before softplus)
    pub omega: Vec<f32>,      // [N_BANDS]
    pub alpha: f32,
    pub beta: f32,
    pub rk4_n_steps: usize,   // ODE integration steps (default 8)
    pub phase_correction: Vec<f32>,  // [N_BANDS] corrector plate — per-band phase offset (init 0.0)
    pub rk4_weights: [f32; 4],  // RK4 combination weights [w1,w2,w3,w4] (standard: 1/6, 1/3, 1/3, 1/6)
    pub coherent_matrix: Vec<Vec<f32>>,  // [n_bands][n_bands] FROZEN antisymmetric (unused, kept for compat)
    pub mix_strength: f32,               // unused, kept for compat
    pub chi: f32,                         // four-wave mixing strength (0.0 = off)
}

/// Weights for Maestro.
#[derive(Clone)]
pub struct MaestroWeights {
    pub squeeze: LinearWeights,   // [MAESTRO_DIM, N_EMBD]
    pub process_1: LinearWeights, // [N_EMBD, MAESTRO_DIM]
}

/// Weights for KerrMaestroAdd block (Blocks 1-3 FFN).
#[derive(Clone)]
pub struct KerrMaestroAddWeights {
    pub kerr: KerrWeights,
    pub maestro: MaestroWeights,
    pub out_proj: LinearWeights,
}

/// Weights for dual-maestro variant — pre-ODE regulator + post-ODE regulator.
#[derive(Clone)]
pub struct KerrDualMaestroWeights {
    pub kerr: KerrWeights,
    pub maestro_in: MaestroWeights,
    pub maestro_out: MaestroWeights,
    pub out_proj: OutProjWeights,  // Dense or BlockDiagonal — consumers use methods
}

/// Weights for one Block.
#[derive(Clone)]
pub struct BlockWeights {
    pub ln_1: LayerNormWeights,
    pub attn: AttentionWeights,
    pub ln_2: LayerNormWeights,
    pub ffn: FfnWeights,
}

/// FFN can be PerBandLinear (block 0), KerrMaestroAdd (blocks 1-3),
/// or KerrDualMaestro (blocks 1-3, high-dim stability variant).
#[derive(Clone)]
pub enum FfnWeights {
    PerBand(PerBandLinearWeights),
    KerrMaestro(KerrMaestroAddWeights),
    KerrDualMaestro(KerrDualMaestroWeights),
}

/// Full model weights.
pub struct ModelWeights {
    pub config: ModelConfig,
    pub vocab_size: usize,
    pub wte_phase: Vec<Vec<f32>>,  // [vocab_size][N_EMBD] (frozen)
    pub wpe: Vec<Vec<f32>>,        // [BLOCK_SIZE][N_EMBD] (frozen)
    pub blocks: Vec<BlockWeights>, // n_layers blocks
    pub ln_f: LayerNormWeights,
    pub lm_head: Vec<Vec<f32>>,    // [vocab_size][N_EMBD] (no bias)
}
