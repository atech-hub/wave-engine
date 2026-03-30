//! FFN routed through ComputeBackend — same device for forward AND backward.
//!
//! This is the kerr-engine pattern: every operation goes through the backend trait.
//! When backend is CPU → exact precision, same as the original hand-written code.
//! When backend is GPU → all ops on GPU, self-consistent, no ping-pong needed.

use crate::backend::ComputeBackend;
use crate::wave_block::{KerrDualMaestroWeights, gelu};

/// ODE input monitoring — read from training loop for diagnostics
pub static ODE_CLAMP_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static ODE_MAX_MAG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Global AGC — initialized at startup with coupling-derived ceiling.
/// Call init_agc() before training. Falls back to α=0.1 defaults if not initialized.
pub static AGC: std::sync::OnceLock<std::sync::Mutex<crate::common::agc::OdeAgc>> = std::sync::OnceLock::new();

/// Initialize the global AGC with coupling constants.
/// Must be called before training starts. Safe to call multiple times (only first wins).
pub fn init_agc(alpha: f32, beta: f32) {
    AGC.get_or_init(|| std::sync::Mutex::new(crate::common::agc::OdeAgc::from_coupling(alpha, beta)));
}

/// Initialize with explicit ceiling override.
pub fn init_agc_ceiling(ceiling: f32) {
    AGC.get_or_init(|| std::sync::Mutex::new(crate::common::agc::OdeAgc::with_ceiling(ceiling)));
}

/// Get current AGC stats for JSONL logging.
pub fn agc_stats() -> crate::common::agc::AgcStats {
    AGC.get_or_init(|| std::sync::Mutex::new(crate::common::agc::OdeAgc::new()))
        .lock().unwrap().stats()
}

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
    freeze_ode: bool,
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

    // AGC (Automatic Gain Control) — adaptive knee compression before ODE.
    // The model finds its own operating range via EMA of observed magnitudes.
    // Below threshold: signal passes UNCHANGED. Above: smooth compression on excess only.
    // Threshold = EMA_mean + 3σ (adapts over ~200 iters).
    let n_bands = n_embd / 2;
    let (clamp_count, max_pre_clamp_mag) = {
        let mut agc = AGC.get_or_init(|| std::sync::Mutex::new(crate::common::agc::OdeAgc::new()))
            .lock().unwrap();
        agc.process(&mut precond, n_bands)
    };
    ODE_CLAMP_COUNT.store(clamp_count, std::sync::atomic::Ordering::Relaxed);
    ODE_MAX_MAG.store(max_pre_clamp_mag.to_bits(), std::sync::atomic::Ordering::Relaxed);

    // 3. ODE: when !freeze_ode, use caching forward for backward pass.
    //    When freeze_ode, use the fast path (GPU/FFT/sequential, no cache).
    let _t_ode = std::time::Instant::now();
    let (kerr_out, ode_caches, ode_device): (Vec<Vec<f32>>, Option<Vec<crate::cpu::ode_backward::OdeForwardCache>>, &str) =
    if !freeze_ode {
        // Caching forward — stores intermediates for backward
        let mut outs = Vec::with_capacity(t);
        let mut caches = Vec::with_capacity(t);
        for p in &precond {
            let (out, cache) = crate::cpu::ode_backward::ode_forward_with_cache(p, &weights.kerr);
            outs.push(out);
            caches.push(cache);
        }
        (outs, Some(caches), "CPU-cached")
    } else if let Some((_bufs, gpu_be)) = ping_pong {
        // GPU: perturbative (single dispatch) or fused RK4 based on rk4_n_steps
        let out = if weights.kerr.rk4_n_steps <= 1 {
            gpu_be.gpu_kerr_ode_perturbative_batch(&weights.kerr, &precond)
        } else {
            gpu_be.gpu_kerr_ode_batch_fused(&weights.kerr, &precond)
        };
        (out, None, if weights.kerr.rk4_n_steps <= 1 { "GPU-perturbative" } else { "GPU-fused" })
    } else if let Some(st) = stencil {
        let out = precond.iter().map(|p| {
            crate::fft_ode::kerr_ode_fft(p, &weights.kerr.gamma_raw, &weights.kerr.omega,
                weights.kerr.alpha, weights.kerr.beta, weights.kerr.rk4_n_steps, st)
        }).collect();
        (out, None, "CPU-FFT")
    } else {
        (cpu.kerr_ode_batch(&weights.kerr, &precond), None, "CPU-seq")
    };
    let _ode_dur = _t_ode.elapsed();

    // No energy conservation — AGC handles magnitude regulation.
    // Coupling at α=0.1 with AGC ceiling=2.0 matches kerr-engine recipe.

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
        ode_caches,
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

    // ─── ODE backward ───
    let (d_precond, ode_param_grads): (Vec<Vec<f32>>, Option<Vec<crate::cpu::ode_backward::OdeParamGrads>>) =
    if let Some(ref ode_caches) = cache.ode_caches {
        // Full backward through RK4
        let mut d_preconds = Vec::with_capacity(t);
        let mut param_grads = Vec::with_capacity(t);
        for (pos, d_ko) in d_kerr_out.iter().enumerate() {
            let (d_p, pg) = crate::cpu::ode_backward::ode_backward(d_ko, &ode_caches[pos], &weights.kerr);
            d_preconds.push(d_p);
            param_grads.push(pg);
        }
        (d_preconds, Some(param_grads))
    } else {
        // Identity (frozen ODE — legacy behaviour)
        (d_kerr_out, None)
    };

    // ─── Maestro_in: CPU (dim=16) ───
    let d_mae_in_act = cpu.linear_backward_dx_batch(&d_precond, &weights.maestro_in.process_1.w);
    let (d_mae_in_pr_w, d_mae_in_pr_b) = cpu.outer_product_accum(&d_precond, &cache.mae_in_act, true);
    let d_mae_in_sq = cpu.gelu_backward_batch(&d_mae_in_act, &cache.mae_in_sq);
    let d_input_from_mae = cpu.linear_backward_dx_batch(&d_mae_in_sq, &weights.maestro_in.squeeze.w);
    let (d_mae_in_sq_w, d_mae_in_sq_b) = cpu.outer_product_accum(&d_mae_in_sq, &cache.input, true);

    // ─── d_input = d_precond (residual) + d_from_mae_in_squeeze ───
    let d_input = cpu.vec_add_batch(&d_precond, &d_input_from_mae);

    // Accumulate ODE param gradients across positions
    let (d_kerr_gamma_raw, d_kerr_alpha, d_kerr_beta) = if let Some(ref pg_vec) = ode_param_grads {
        let nb = pg_vec[0].d_gamma_raw.len();
        let mut d_gr = vec![0.0f32; nb];
        let mut d_a = 0.0f32;
        let mut d_b = 0.0f32;
        for pg in pg_vec {
            for k in 0..nb { d_gr[k] += pg.d_gamma_raw[k]; }
            d_a += pg.d_alpha;
            d_b += pg.d_beta;
        }
        (Some(d_gr), Some(d_a), Some(d_b))
    } else {
        (None, None, None)
    };

    let grads = FfnGrads {
        d_out_proj_w, d_out_proj_b,
        d_mae_out_pr_w, d_mae_out_pr_b,
        d_mae_out_sq_w, d_mae_out_sq_b,
        d_mae_in_pr_w, d_mae_in_pr_b,
        d_mae_in_sq_w, d_mae_in_sq_b,
        d_kerr_gamma_raw, d_kerr_alpha, d_kerr_beta,
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
    /// ODE forward caches for backward pass — None when --freeze-ode is active
    pub ode_caches: Option<Vec<crate::cpu::ode_backward::OdeForwardCache>>,
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
    // ODE parameter gradients (None when --freeze-ode)
    pub d_kerr_gamma_raw: Option<Vec<f32>>,  // [n_bands]
    pub d_kerr_alpha: Option<f32>,
    pub d_kerr_beta: Option<f32>,
}
