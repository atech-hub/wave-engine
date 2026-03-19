//! FFN routed through ComputeBackend — same device for forward AND backward.
//!
//! This is the kerr-engine pattern: every operation goes through the backend trait.
//! When backend is CPU → exact precision, same as the original hand-written code.
//! When backend is GPU → all ops on GPU, self-consistent, no ping-pong needed.

use crate::backend::ComputeBackend;
use crate::wave_block::{KerrDualMaestroWeights, gelu};

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
    let precond = cpu.vec_add_batch(x, &mae_in_out);
    let _mae_in_dur = _t_mae_in.elapsed();

    // 3. ODE: CPU FFT (fastest at 384 bands — no dispatch overhead), sequential fallback
    // GPU FFT shader validated but per-dispatch overhead (64 round-trips) makes it slower
    // than CPU FFT at this scale. GPU FFT wins when entire RK4 is fused on GPU.
    let _t_ode = std::time::Instant::now();
    let (kerr_out, ode_device): (Vec<Vec<f32>>, &str) = if let Some(st) = stencil {
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

    // 6. Out projection: GPU ping-pong when available (13ms→~1ms), else CPU
    // Ping-pong stores regulated in VRAM — backward reads same bits. No mismatch.
    let _t_proj = std::time::Instant::now();
    let (output, proj_device) = if let Some((bufs, gpu_be)) = ping_pong {
        // Flatten regulated for GPU upload
        let reg_flat: Vec<f32> = regulated.iter().flat_map(|v| v.iter().copied()).collect();
        let mut w_flat = Vec::with_capacity(n_embd * n_embd);
        for row in &weights.out_proj.w { w_flat.extend_from_slice(row); }
        let out_flat = bufs.forward_out_proj(gpu_be, &reg_flat, &w_flat, &weights.out_proj.b, t, n_embd);
        // Unflatten output
        let out: Vec<Vec<f32>> = out_flat.chunks(n_embd).map(|c| c.to_vec()).collect();
        (out, "GPU")
    } else {
        let out = cpu.linear_batch(&weights.out_proj.w, &weights.out_proj.b, &regulated);
        (out, "CPU")
    };
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

    // ─── Out_proj backward: GPU ping-pong when available, else CPU ───
    let _t_bwd_proj = std::time::Instant::now();
    let (d_regulated, d_out_proj_w, d_out_proj_b, proj_device) = if let Some((bufs, gpu_be)) = ping_pong {
        // GPU backward: d_x and d_W computed on GPU, reading regulated from Buffer A (forward's bits)
        let d_flat: Vec<f32> = d_ffn_out.iter().flat_map(|v| v.iter().copied()).collect();
        let mut w_flat = Vec::with_capacity(n_embd * n_embd);
        for row in &weights.out_proj.w { w_flat.extend_from_slice(row); }
        let (d_reg_flat, d_w_flat, d_b) = bufs.backward_out_proj(gpu_be, &d_flat, &w_flat, t, n_embd);
        // Unflatten d_regulated
        let d_reg: Vec<Vec<f32>> = d_reg_flat.chunks(n_embd).map(|c| c.to_vec()).collect();
        // Unflatten d_w
        let d_w: Vec<Vec<f32>> = d_w_flat.chunks(n_embd).map(|c| c.to_vec()).collect();
        (d_reg, d_w, d_b, "GPU")
    } else {
        let d_reg = cpu.linear_backward_dx_batch(d_ffn_out, &weights.out_proj.w);
        let (d_w, d_b) = cpu.outer_product_accum(d_ffn_out, &cache.regulated, true);
        (d_reg, d_w, d_b, "CPU")
    };
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
