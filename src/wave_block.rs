//! Parallel attention + FFN block — the wave packet dispatch architecture.
//!
//! GPT-J formulation: x = x + attn(LN(x)) + FFN(LN(x))
//! Both attention and FFN take the SAME normalised input.
//! They share no data during computation → GPU runs FFN while CPU runs attention.

use crate::wave_attn::*;

// Re-export model types so callers can use wave_block::KerrWeights etc.
pub use crate::model::{
    LayerNormWeights, LinearWeights, MaestroWeights, KerrWeights,
    KerrDualMaestroWeights, layer_norm, gelu,
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
    let output: Vec<Vec<f32>> = if let Some((bufs, gpu_be)) = ping_pong {
        // Flatten regulated_all and weights
        let reg_flat: Vec<f32> = regulated_all.iter().flat_map(|v| v.iter().copied()).collect();
        let mut w_flat = Vec::with_capacity(n_embd * n_embd);
        for row in &weights.out_proj.w { w_flat.extend_from_slice(row); }
        let result_flat = bufs.forward_out_proj(gpu_be, &reg_flat, &w_flat, &weights.out_proj.b, t, n_embd);
        // Buffer A now holds regulated_all in VRAM for backward
        result_flat.chunks(n_embd).map(|c| c.to_vec()).collect()
    } else {
        regulated_all.iter().map(|regulated| {
            let mut projected = vec![0.0f32; n_embd];
            for i in 0..n_embd {
                let mut sum = 0.0f32;
                for j in 0..n_embd { sum += weights.out_proj.w[i][j] * regulated[j]; }
                projected[i] = sum + weights.out_proj.b[i];
            }
            projected
        }).collect()
    };

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

/// Perturbative ODE — single-pass analytical Kerr computation (all CPU tiers).
/// Lab-validated: MSE 0.000005 vs RK4-16, trains better (2.97 vs 3.07).
fn kerr_ode_forward_cpu(weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;

    fn softplus(v: f32) -> f32 { if v > 20.0 { v } else { (1.0 + v.exp()).ln() } }
    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

    // Step 1: Linear solution (damping + base rotation)
    let mut r_lin = vec![0.0f32; n_bands];
    let mut s_lin = vec![0.0f32; n_bands];
    for k in 0..n_bands {
        let r = x[k * 2];
        let s = x[k * 2 + 1];
        let decay = (-gamma[k]).exp();
        let cos_w = weights.omega[k].cos();
        let sin_w = weights.omega[k].sin();
        r_lin[k] = decay * (r * cos_w - s * sin_w);
        s_lin[k] = decay * (r * sin_w + s * cos_w);
    }

    // Step 2: First-order nonlinear correction (SPM + XPM)
    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands {
        let mag_sq = r_lin[k] * r_lin[k] + s_lin[k] * s_lin[k];
        let mut ns = 0.0f32;
        if k >= 2 { ns += r_lin[k-2]*r_lin[k-2] + s_lin[k-2]*s_lin[k-2]; }
        if k >= 1 { ns += r_lin[k-1]*r_lin[k-1] + s_lin[k-1]*s_lin[k-1]; }
        if k+1 < n_bands { ns += r_lin[k+1]*r_lin[k+1] + s_lin[k+1]*s_lin[k+1]; }
        if k+2 < n_bands { ns += r_lin[k+2]*r_lin[k+2] + s_lin[k+2]*s_lin[k+2]; }
        let delta_phi = weights.alpha * mag_sq + weights.beta * ns;
        out[k * 2]     = r_lin[k] - delta_phi * s_lin[k];
        out[k * 2 + 1] = s_lin[k] + delta_phi * r_lin[k];
    }
    out
}

fn rk4_step(r: &[f32], s: &[f32], dt: f32, gamma: &[f32], omega: &[f32], alpha: f32, beta: f32) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();
    let deriv = |r: &[f32], s: &[f32]| -> (Vec<f32>, Vec<f32>) {
        let mut dr = vec![0.0f32; n];
        let mut ds = vec![0.0f32; n];
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
        (dr, ds)
    };

    let (k1r, k1s) = deriv(r, s);
    let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&a, &b)| a + 0.5*dt*b).collect();
    let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&a, &b)| a + 0.5*dt*b).collect();
    let (k2r, k2s) = deriv(&r2, &s2);
    let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&a, &b)| a + 0.5*dt*b).collect();
    let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&a, &b)| a + 0.5*dt*b).collect();
    let (k3r, k3s) = deriv(&r3, &s3);
    let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&a, &b)| a + dt*b).collect();
    let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&a, &b)| a + dt*b).collect();
    let (k4r, k4s) = deriv(&r4, &s4);

    let r_new: Vec<f32> = (0..n).map(|i| r[i] + dt/6.0 * (k1r[i] + 2.0*k2r[i] + 2.0*k3r[i] + k4r[i])).collect();
    let s_new: Vec<f32> = (0..n).map(|i| s[i] + dt/6.0 * (k1s[i] + 2.0*k2s[i] + 2.0*k3s[i] + k4s[i])).collect();
    (r_new, s_new)
}
