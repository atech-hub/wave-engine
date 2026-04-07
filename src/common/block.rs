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

/// ODE forward — RK4-8 for CPU tiers (perturbative caused NaN in CPU backward path).
/// The Candle tier uses GPU-native perturbative with true autograd backward.
fn kerr_ode_forward_cpu(weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;
    let n_steps = weights.rk4_n_steps;
    let dt = 1.0 / n_steps as f32;

    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| super::math::softplus(g)).collect();

    let mut r: Vec<f32> = (0..n_bands).map(|k| x[k * 2]).collect();
    let mut s: Vec<f32> = (0..n_bands).map(|k| x[k * 2 + 1]).collect();

    let w = &weights.rk4_weights;
    let chi = weights.chi;
    for _ in 0..n_steps {
        let (r_new, s_new) = rk4_step(&r, &s, dt, &gamma, &weights.omega, weights.alpha, weights.beta, chi, w);
        r = r_new;
        s = s_new;
    }

    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands { out[k * 2] = r[k]; out[k * 2 + 1] = s[k]; }
    out
}

fn rk4_step(r: &[f32], s: &[f32], dt: f32, gamma: &[f32], omega: &[f32], alpha: f32, beta: f32, chi: f32, w: &[f32; 4]) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();

    let mut r_tmp = vec![0.0f32; n];
    let mut s_tmp = vec![0.0f32; n];
    let mut k1r = vec![0.0f32; n];
    let mut k1s = vec![0.0f32; n];
    let mut k2r = vec![0.0f32; n];
    let mut k2s = vec![0.0f32; n];
    let mut k3r = vec![0.0f32; n];
    let mut k3s = vec![0.0f32; n];
    let mut k4r = vec![0.0f32; n];
    let mut k4s = vec![0.0f32; n];

    deriv_into(r, s, gamma, omega, alpha, beta, chi, &mut k1r, &mut k1s);

    for i in 0..n { r_tmp[i] = r[i] + 0.5*dt*k1r[i]; }
    for i in 0..n { s_tmp[i] = s[i] + 0.5*dt*k1s[i]; }
    deriv_into(&r_tmp, &s_tmp, gamma, omega, alpha, beta, chi, &mut k2r, &mut k2s);

    for i in 0..n { r_tmp[i] = r[i] + 0.5*dt*k2r[i]; }
    for i in 0..n { s_tmp[i] = s[i] + 0.5*dt*k2s[i]; }
    deriv_into(&r_tmp, &s_tmp, gamma, omega, alpha, beta, chi, &mut k3r, &mut k3s);

    for i in 0..n { r_tmp[i] = r[i] + dt*k3r[i]; }
    for i in 0..n { s_tmp[i] = s[i] + dt*k3s[i]; }
    deriv_into(&r_tmp, &s_tmp, gamma, omega, alpha, beta, chi, &mut k4r, &mut k4s);

    // RK4 combination: r_new = r + dt * (w0*k1 + w1*k2 + w2*k3 + w3*k4)
    // Standard: w = [1/6, 1/3, 1/3, 1/6]. Learnable: model decides.
    let mut r_new = vec![0.0f32; n];
    let mut s_new = vec![0.0f32; n];
    for i in 0..n {
        r_new[i] = r[i] + dt * (w[0]*k1r[i] + w[1]*k2r[i] + w[2]*k3r[i] + w[3]*k4r[i]);
        s_new[i] = s[i] + dt * (w[0]*k1s[i] + w[1]*k2s[i] + w[2]*k3s[i] + w[3]*k4s[i]);
    }
    (r_new, s_new)
}

/// In-place sequential derivative for block.rs RK4 — writes into pre-allocated output buffers.
fn deriv_into(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32, chi: f32,
    dr: &mut [f32], ds: &mut [f32],
) {
    let n = r.len();
    for k in 0..n {
        let mag_sq = r[k] * r[k] + s[k] * s[k];
        let mut ns = 0.0f32;
        if k >= 2 { ns += r[k-2]*r[k-2] + s[k-2]*s[k-2]; }
        if k >= 1 { ns += r[k-1]*r[k-1] + s[k-1]*s[k-1]; }
        if k+1 < n { ns += r[k+1]*r[k+1] + s[k+1]*s[k+1]; }
        if k+2 < n { ns += r[k+2]*r[k+2] + s[k+2]*s[k+2]; }
        let phi = omega[k] + alpha * mag_sq + beta * ns;
        dr[k] = -gamma[k] * r[k] - phi * s[k];
        ds[k] = -gamma[k] * s[k] + phi * r[k];
    }
    // Four-wave mixing: Hamiltonian energy-conserving cubic coupling
    if chi != 0.0 && n > 4 {
        #[inline(always)]
        fn apply_quartet(dr: &mut [f32], ds: &mut [f32], r: &[f32], s: &[f32],
                         chi: f32, a: usize, b: usize, c: usize, d: usize) {
            let (ra, sa) = (r[a], s[a]); let (rb, sb) = (r[b], s[b]);
            let (rc, sc) = (r[c], s[c]); let (rd, sd) = (r[d], s[d]);
            let pab_re = ra*rb - sa*sb; let pab_im = ra*sb + sa*rb;
            let pcd_re = rc*rd - sc*sd; let pcd_im = rc*sd + sc*rd;
            dr[a] += chi * (rb*pcd_im - sb*pcd_re); ds[a] -= chi * (rb*pcd_re + sb*pcd_im);
            dr[b] += chi * (ra*pcd_im - sa*pcd_re); ds[b] -= chi * (ra*pcd_re + sa*pcd_im);
            dr[c] += chi * (pab_im*rd - pab_re*sd); ds[c] -= chi * (pab_re*rd + pab_im*sd);
            dr[d] += chi * (pab_im*rc - pab_re*sc); ds[d] -= chi * (pab_re*rc + pab_im*sc);
        }
        for k in 2..(n - 1) {
            apply_quartet(dr, ds, r, s, chi, k - 2, k + 1, k - 1, k);
        }
        for k in 1..(n - 2) {
            apply_quartet(dr, ds, r, s, chi, k - 1, k + 2, k, k + 1);
        }
    }
}
