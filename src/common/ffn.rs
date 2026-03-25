//! FFN routed through ComputeBackend — same device for forward AND backward.
//!
//! This is the kerr-engine pattern: every operation goes through the backend trait.
//! When backend is CPU → exact precision, same as the original hand-written code.
//! When backend is GPU → all ops on GPU, self-consistent, no ping-pong needed.

use crate::backend::ComputeBackend;
use crate::wave_block::{KerrDualMaestroWeights, gelu};

/// ODE input clamp monitoring — read from training loop for diagnostics
pub static ODE_CLAMP_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static ODE_MAX_MAG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// FFN forward through backend. Hybrid routing: big ops on `backend` (GPU when available),
/// small ops (maestro dim=16) always on CPU for precision.
/// When ping_pong is Some, out_proj routes through GPU ping-pong buffers (13ms→~1ms).
/// When gpu_kernel is Some, ODE neighbour sums computed via GPU FFT shader.
pub fn ffn_forward_via_backend(
    weights: &KerrDualMaestroWeights,
    x: &[Vec<f32>],
    backend: &dyn ComputeBackend,
    stencil: Option<&crate::fft_ode::StencilFft>,
    ping_pong: Option<(&crate::ffn_gpu::FfnGpuBuffers, &crate::gpu_pipelines::GpuBackend)>,
    gpu_kernel: Option<(&crate::fft_ode::GpuKernelFft, &crate::gpu_pipelines::GpuBackend)>,
) -> (Vec<Vec<f32>>, FfnCache) {
    let t = x.len();
    let n_embd = x[0].len();
    let cpu = &crate::backend::CpuBackend;
    let profiling = crate::PROFILE.load(std::sync::atomic::Ordering::Relaxed);

    // 1. Maestro_in: CPU (dim=16 too small for GPU — dispatch overhead > compute)
    let _t_mae_in = std::time::Instant::now();
    let mae_in_sq = cpu.linear_batch(&weights.maestro_in.squeeze.w, &weights.maestro_in.squeeze.b, x);
    let mae_in_act = cpu.gelu_batch(&mae_in_sq);
    let mae_in_out = cpu.linear_batch(&weights.maestro_in.process_1.w, &weights.maestro_in.process_1.b, &mae_in_act);

    // 2. Residual: precond = x + mae_in_out (CPU — element-wise, trivial)
    let mut precond = cpu.vec_add_batch(x, &mae_in_out);
    let _mae_in_dur = _t_mae_in.elapsed();

    // Per-band magnitude clamp before ODE — prevents phase wrapping.
    let n_bands = n_embd / 2;
    let max_band_mag = 2.5f32;
    let mut clamp_count = 0usize;
    let mut max_pre_clamp_mag = 0.0f32;
    for pos_vec in precond.iter_mut() {
        for k in 0..n_bands {
            let r = pos_vec[k * 2];
            let s = pos_vec[k * 2 + 1];
            let mag_sq = r * r + s * s;
            if mag_sq > max_pre_clamp_mag * max_pre_clamp_mag {
                max_pre_clamp_mag = mag_sq.sqrt();
            }
            if mag_sq > max_band_mag * max_band_mag {
                let scale = max_band_mag / mag_sq.sqrt();
                pos_vec[k * 2] *= scale;
                pos_vec[k * 2 + 1] *= scale;
                clamp_count += 1;
            }
        }
    }
    // Store clamp stats for monitoring (thread-safe atomic)
    ODE_CLAMP_COUNT.store(clamp_count, std::sync::atomic::Ordering::Relaxed);
    ODE_MAX_MAG.store(max_pre_clamp_mag.to_bits(), std::sync::atomic::Ordering::Relaxed);

    // 3. ODE: GPU fused when available (one submit, zero readbacks between steps),
    //    CPU FFT fallback, sequential last resort
    let _t_ode = std::time::Instant::now();
    let (kerr_out, ode_device): (Vec<Vec<f32>>, &str) = if let Some((_bufs, gpu_be)) = ping_pong {
        // GPU: perturbative (single dispatch) or fused RK4 based on rk4_n_steps
        let out = if weights.kerr.rk4_n_steps <= 1 {
            gpu_be.gpu_kerr_ode_perturbative_batch(&weights.kerr, &precond)
        } else {
            gpu_be.gpu_kerr_ode_batch_fused(&weights.kerr, &precond)
        };
        (out, if weights.kerr.rk4_n_steps <= 1 { "GPU-perturbative" } else { "GPU-fused" })
    } else if let Some(st) = stencil {
        let out = precond.iter().map(|p| {
            crate::fft_ode::kerr_ode_fft(p, &weights.kerr.gamma_raw, &weights.kerr.omega,
                weights.kerr.alpha, weights.kerr.beta, weights.kerr.rk4_n_steps, st)
        }).collect();
        (out, "CPU-FFT")
    } else {
        (cpu.kerr_ode_batch(&weights.kerr, &precond), "CPU-seq")
    };
    let _ode_dur = _t_ode.elapsed();

    // 4. Maestro_out: CPU (same reason as maestro_in)
    let _t_mae_out = std::time::Instant::now();
    let mae_out_sq = cpu.linear_batch(&weights.maestro_out.squeeze.w, &weights.maestro_out.squeeze.b, &kerr_out);
    let mae_out_act = cpu.gelu_batch(&mae_out_sq);
    let mae_out_out = cpu.linear_batch(&weights.maestro_out.process_1.w, &weights.maestro_out.process_1.b, &mae_out_act);

    // 5. Residual: regulated = kerr_out + mae_out_out (CPU)
    let regulated = cpu.vec_add_batch(&kerr_out, &mae_out_out);
    let _mae_out_dur = _t_mae_out.elapsed();

    // 6. Out projection via OutProjWeights enum (Dense or BlockDiagonal)
    let _t_proj = std::time::Instant::now();
    let output = weights.out_proj.forward_batch(&regulated);
    let proj_device = if weights.out_proj.n_groups() > 1 { "CPU-BD" } else { "CPU" };
    let _proj_dur = _t_proj.elapsed();

    if profiling {
        eprintln!("    [FFN fwd] mae_in: {:.3?}  ODE({}): {:.3?}  mae_out: {:.3?}  out_proj({}): {:.3?}  ({} elem)",
            _mae_in_dur, ode_device, _ode_dur, _mae_out_dur, proj_device, _proj_dur, t);
    }

    let cache = FfnCache {
        input: x.to_vec(),
        mae_in_sq, mae_in_act, precond,
        kerr_out, mae_out_sq, mae_out_act,
        regulated,
    };

    (output, cache)
}

/// FFN backward. Hybrid routing: big ops through `backend`, small ops through CPU.
/// Reads cached intermediates from forward — same values, self-consistent.
/// When ping_pong is Some, out_proj backward routes through GPU (reads Buffer A from forward).
pub fn ffn_backward_via_backend(
    weights: &KerrDualMaestroWeights,
    d_ffn_out: &[Vec<f32>],
    cache: &FfnCache,
    backend: &dyn ComputeBackend,
    ping_pong: Option<(&crate::ffn_gpu::FfnGpuBuffers, &crate::gpu_pipelines::GpuBackend)>,
) -> (Vec<Vec<f32>>, FfnGrads) {
    let t = d_ffn_out.len();
    let n_embd = d_ffn_out[0].len();
    let n_bands = n_embd / 2;
    let maestro_dim = weights.maestro_in.squeeze.w.len();
    let cpu = &crate::backend::CpuBackend;
    let profiling = crate::PROFILE.load(std::sync::atomic::Ordering::Relaxed);

    // ─── Out_proj backward via OutProjWeights enum ───
    let _t_bwd_proj = std::time::Instant::now();
    let d_regulated: Vec<Vec<f32>> = d_ffn_out.iter()
        .map(|dy| weights.out_proj.backward_dx(dy)).collect();
    let (d_out_proj_w, d_out_proj_b) = weights.out_proj.backward_dw_db(d_ffn_out, &cache.regulated);
    let proj_device = if weights.out_proj.n_groups() > 1 { "CPU-BD" } else { "CPU" };
    let _bwd_proj_dur = _t_bwd_proj.elapsed();
    if profiling {
        eprintln!("    [FFN bwd] out_proj({}): {:.3?}  ({} elem)", proj_device, _bwd_proj_dur, t);
    }

    // ─── Maestro_out: CPU (dim=16 — too small for GPU) ───
    let d_mae_out_act = cpu.linear_backward_dx_batch(&d_regulated, &weights.maestro_out.process_1.w);
    let (d_mae_out_pr_w, d_mae_out_pr_b) = cpu.outer_product_accum(&d_regulated, &cache.mae_out_act, true);
    let d_mae_out_sq = cpu.gelu_backward_batch(&d_mae_out_act, &cache.mae_out_sq);
    let d_kerr_from_mae = cpu.linear_backward_dx_batch(&d_mae_out_sq, &weights.maestro_out.squeeze.w);
    let (d_mae_out_sq_w, d_mae_out_sq_b) = cpu.outer_product_accum(&d_mae_out_sq, &cache.kerr_out, true);

    // ─── d_kerr_out = d_regulated (residual) + d_from_mae_out_squeeze ───
    let d_kerr_out = cpu.vec_add_batch(&d_regulated, &d_kerr_from_mae);

    // ─── ODE backward: identity (frozen) ───
    let d_precond = d_kerr_out;

    // ─── Maestro_in: CPU (dim=16) ───
    let d_mae_in_act = cpu.linear_backward_dx_batch(&d_precond, &weights.maestro_in.process_1.w);
    let (d_mae_in_pr_w, d_mae_in_pr_b) = cpu.outer_product_accum(&d_precond, &cache.mae_in_act, true);
    let d_mae_in_sq = cpu.gelu_backward_batch(&d_mae_in_act, &cache.mae_in_sq);
    let d_input_from_mae = cpu.linear_backward_dx_batch(&d_mae_in_sq, &weights.maestro_in.squeeze.w);
    let (d_mae_in_sq_w, d_mae_in_sq_b) = cpu.outer_product_accum(&d_mae_in_sq, &cache.input, true);

    // ─── d_input = d_precond (residual) + d_from_mae_in_squeeze ───
    let d_input = cpu.vec_add_batch(&d_precond, &d_input_from_mae);

    let grads = FfnGrads {
        d_out_proj_w, d_out_proj_b,
        d_mae_out_pr_w, d_mae_out_pr_b,
        d_mae_out_sq_w, d_mae_out_sq_b,
        d_mae_in_pr_w, d_mae_in_pr_b,
        d_mae_in_sq_w, d_mae_in_sq_b,
    };

    (d_input, grads)
}

/// Cached forward intermediates for backward.
pub struct FfnCache {
    pub input: Vec<Vec<f32>>,
    pub mae_in_sq: Vec<Vec<f32>>,
    pub mae_in_act: Vec<Vec<f32>>,
    pub precond: Vec<Vec<f32>>,
    pub kerr_out: Vec<Vec<f32>>,
    pub mae_out_sq: Vec<Vec<f32>>,
    pub mae_out_act: Vec<Vec<f32>>,
    pub regulated: Vec<Vec<f32>>,
}

/// FFN weight gradients.
pub struct FfnGrads {
    pub d_out_proj_w: Vec<Vec<f32>>,
    pub d_out_proj_b: Vec<f32>,
    pub d_mae_out_pr_w: Vec<Vec<f32>>,
    pub d_mae_out_pr_b: Vec<f32>,
    pub d_mae_out_sq_w: Vec<Vec<f32>>,
    pub d_mae_out_sq_b: Vec<f32>,
    pub d_mae_in_pr_w: Vec<Vec<f32>>,
    pub d_mae_in_pr_b: Vec<f32>,
    pub d_mae_in_sq_w: Vec<Vec<f32>>,
    pub d_mae_in_sq_b: Vec<f32>,
}
