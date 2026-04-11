//! Parallel attention + FFN block — the wave packet dispatch architecture.
//!
//! GPT-J formulation: x = x + attn(LN(x)) + FFN(LN(x))
//! Both attention and FFN take the SAME normalised input.
//! They share no data during computation → GPU runs FFN while CPU runs attention.

use crate::wave_attn::*;

// Re-export model types so callers can use wave_block::KerrWeights etc.
pub use crate::model::{
    LayerNormWeights, LinearWeights, MaestroWeights, KerrWeights,
    KerrDualMaestroWeights, OutProjWeights, BlockDiagonalWeights, layer_norm, gelu,
};

/// Weights for one parallel wave block.
#[derive(Clone)]
pub struct WaveBlockWeights {
    /// Single layer norm (shared by attention and FFN)
    pub ln: LayerNormWeights,
    /// Separate FFN layer norm — conditions ODE input energy at high dimensions
    pub ln_ffn: LayerNormWeights,
    /// Wave coherence attention
    pub attn: WaveAttnWeights,
    /// Dual-maestro FFN (kerr-engine validated)
    pub ffn: KerrDualMaestroWeights,
}

/// Forward pass for a parallel wave block.
/// Returns (output, attn_weights_for_cache).
///
/// The key: `attn_out` and `ffn_out` are computed from the SAME `normed` input.
/// When phase-locked dispatch is enabled, FFN runs on GPU while attention runs on CPU.
pub fn wave_block_forward(
    weights: &WaveBlockWeights,
    hidden: &[Vec<f32>],
    n_bands: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
    let t = hidden.len();
    let n_embd = n_bands * 2;

    // Single shared layer norm
    let normed: Vec<Vec<f32>> = hidden.iter()
        .map(|h| layer_norm(h, &weights.ln.weight, &weights.ln.bias))
        .collect();

    // Attention path (CPU) — harmonic coherence scoring
    let (attn_out, att_weights) = wave_attention_forward(&weights.attn, &normed, n_bands, None);

    // FFN path — dual-maestro Kerr-ODE (same normed input)
    let (ffn_out, _cache) = dual_maestro_forward_cached(&weights.ffn, &normed, None, None);

    // Combine: x = x + attn_out + ffn_out (parallel residual)
    let output: Vec<Vec<f32>> = (0..t).map(|i| {
        let mut v = vec![0.0f32; n_embd];
        for j in 0..n_embd {
            v[j] = hidden[i][j] + attn_out[i][j] + ffn_out[i][j];
        }
        v
    }).collect();

    (output, att_weights)
}

/// FFN intermediates cached during forward for backward reuse.
pub struct FfnForwardCache {
    pub mae_in_sq: Vec<Vec<f32>>,
    pub mae_in_act: Vec<Vec<f32>>,
    pub precond: Vec<Vec<f32>>,
    pub kerr_out: Vec<Vec<f32>>,
    pub mae_out_sq: Vec<Vec<f32>>,
    pub mae_out_act: Vec<Vec<f32>>,
    pub regulated: Vec<Vec<f32>>,
}

/// FFN forward with cached intermediates + optional GPU out_proj.
pub fn dual_maestro_forward_cached(
    weights: &KerrDualMaestroWeights,
    x: &[Vec<f32>],
    gpu: Option<&(dyn crate::backend::ComputeBackend + Send + Sync)>,
    ping_pong: Option<(&crate::ffn_gpu::FfnGpuBuffers, &crate::gpu_pipelines::GpuBackend)>,
) -> (Vec<Vec<f32>>, FfnForwardCache) {
    let t = x.len();
    let n_embd = x[0].len();
    let maestro_dim = weights.maestro_in.squeeze.w.len();

    let mut mae_in_sq_all = Vec::with_capacity(t);
    let mut mae_in_act_all = Vec::with_capacity(t);
    let mut precond_all = Vec::with_capacity(t);
    let mut kerr_out_all: Vec<Vec<f32>> = Vec::new(); // filled by batched ODE
    let mut mae_out_sq_all = Vec::with_capacity(t);
    let mut mae_out_act_all = Vec::with_capacity(t);
    let mut regulated_all = Vec::with_capacity(t);

    for pos in 0..t {
        // Maestro_in: squeeze → GELU → process
        let mut sq = vec![0.0f32; maestro_dim];
        for i in 0..maestro_dim {
            let mut sum = 0.0f32;
            for j in 0..n_embd { sum += weights.maestro_in.squeeze.w[i][j] * x[pos][j]; }
            sq[i] = sum + weights.maestro_in.squeeze.b[i];
        }
        let act: Vec<f32> = sq.iter().map(|&v| gelu(v)).collect();
        let mut mae_in_out = vec![0.0f32; n_embd];
        for i in 0..n_embd {
            let mut sum = 0.0f32;
            for j in 0..maestro_dim { sum += weights.maestro_in.process_1.w[i][j] * act[j]; }
            mae_in_out[i] = sum + weights.maestro_in.process_1.b[i];
        }
        let mut precond = vec![0.0f32; n_embd];
        for i in 0..n_embd { precond[i] = x[pos][i] + mae_in_out[i]; }

        mae_in_sq_all.push(sq);
        mae_in_act_all.push(act);
        precond_all.push(precond);
    }

    // AGC: clamp preconditioned input magnitudes before ODE to prevent NaN.
    // Uses the global static AGC (initialized by init_agc in calling code).
    let n_bands = n_embd / 2;
    {
        let mut agc = super::ffn::AGC.get_or_init(|| std::sync::Mutex::new(super::agc::OdeAgc::new()))
            .lock().unwrap();
        agc.process(&mut precond_all, n_bands);
    }

    // ODE: CPU always. GPU ODE produces different FP values → maestro backward gets
    // wrong gradients → 0.4 loss gap. The ODE is frozen but maestro_out backward
    // uses kerr_out_all to compute maestro squeeze/process gradients on CPU.
    // Those CPU gradient computations need CPU-precision kerr_out_all.
    let kerr_out_all: Vec<Vec<f32>> = precond_all.iter()
        .map(|p| kerr_ode_forward_cpu(&weights.kerr, p)).collect();

    // Maestro_out: per position
    for pos in 0..t {
        let kerr_out = &kerr_out_all[pos];

        // Maestro_out: squeeze → GELU → process
        let mut sq2 = vec![0.0f32; maestro_dim];
        for i in 0..maestro_dim {
            let mut sum = 0.0f32;
            for j in 0..n_embd { sum += weights.maestro_out.squeeze.w[i][j] * kerr_out[j]; }
            sq2[i] = sum + weights.maestro_out.squeeze.b[i];
        }
        let act2: Vec<f32> = sq2.iter().map(|&v| gelu(v)).collect();
        let mut mae_out_out = vec![0.0f32; n_embd];
        for i in 0..n_embd {
            let mut sum = 0.0f32;
            for j in 0..maestro_dim { sum += weights.maestro_out.process_1.w[i][j] * act2[j]; }
            mae_out_out[i] = sum + weights.maestro_out.process_1.b[i];
        }
        let mut regulated = vec![0.0f32; n_embd];
        for i in 0..n_embd { regulated[i] = kerr_out[i] + mae_out_out[i]; }

        mae_out_sq_all.push(sq2);
        mae_out_act_all.push(act2);
        regulated_all.push(regulated);
    }

    // Out_proj: ping-pong GPU when available (Buffer A holds regulated_all for backward)
    let _ = gpu;
    // Out projection via OutProjWeights enum (GPU ping-pong disabled for block-diagonal)
    let output = weights.out_proj.forward_batch(&regulated_all);

    let cache = FfnForwardCache {
        mae_in_sq: mae_in_sq_all,
        mae_in_act: mae_in_act_all,
        precond: precond_all,
        kerr_out: kerr_out_all,
        mae_out_sq: mae_out_sq_all,
        mae_out_act: mae_out_act_all,
        regulated: regulated_all,
    };

    (output, cache)
}

fn maestro_forward_cpu(weights: &MaestroWeights, x: &[f32]) -> Vec<f32> {
    let maestro_dim = weights.squeeze.w.len();
    let n_embd = x.len();

    // Squeeze: [maestro_dim, n_embd] @ x + b
    let mut squeezed = vec![0.0f32; maestro_dim];
    for i in 0..maestro_dim {
        let mut sum = 0.0f32;
        for j in 0..n_embd { sum += weights.squeeze.w[i][j] * x[j]; }
        squeezed[i] = gelu(sum + weights.squeeze.b[i]);
    }

    // Process: [n_embd, maestro_dim] @ activated + b
    let mut processed = vec![0.0f32; n_embd];
    for i in 0..n_embd {
        let mut sum = 0.0f32;
        for j in 0..maestro_dim { sum += weights.process_1.w[i][j] * squeezed[j]; }
        processed[i] = sum + weights.process_1.b[i];
    }

    processed
}

/// ODE forward — uses canonical rk4_step_public from ode_deriv.rs.
fn kerr_ode_forward_cpu(weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
    use super::ode_deriv::rk4_step_public;

    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;
    let n_steps = weights.rk4_n_steps;
    let dt = 1.0 / n_steps as f32;

    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| super::math::softplus(g)).collect();

    let mut r: Vec<f32> = (0..n_bands).map(|k| x[k * 2]).collect();
    let mut s: Vec<f32> = (0..n_bands).map(|k| x[k * 2 + 1]).collect();

    let w = &weights.rk4_weights;
    for _ in 0..n_steps {
        let (r_new, s_new) = rk4_step_public(&r, &s, dt, &gamma, &weights.omega, weights.alpha, weights.beta, weights.chi, w);
        r = r_new;
        s = s_new;
    }

    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands { out[k * 2] = r[k]; out[k * 2 + 1] = s[k]; }
    out
}
