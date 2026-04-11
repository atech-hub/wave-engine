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
    use_corrector: bool,
    layer_agc: Option<&mut crate::common::agc::OdeAgc>,
    memory: Option<(&[f32], &[f32])>,
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
    // Per-layer AGC when provided (dynamic ceiling tracks learned α/β).
    // Falls back to global static for inference/analyze paths.
    let n_bands = n_embd / 2;
    let (clamp_count, max_pre_clamp_mag) = if let Some(agc) = layer_agc {
        agc.process(&mut precond, n_bands)
    } else {
        let mut agc = AGC.get_or_init(|| std::sync::Mutex::new(crate::common::agc::OdeAgc::new()))
            .lock().unwrap();
        agc.process(&mut precond, n_bands)
    };
    ODE_CLAMP_COUNT.store(clamp_count, std::sync::atomic::Ordering::Relaxed);
    ODE_MAX_MAG.store(max_pre_clamp_mag.to_bits(), std::sync::atomic::Ordering::Relaxed);

    // 2b. Wave memory injection: add per-band offsets to ODE initial conditions.
    //     When None, code path is identical (bit-identical baseline).
    if let Some((r_mem, s_mem)) = memory {
        for p in &mut precond {
            for k in 0..n_bands.min(r_mem.len()) {
                p[k * 2] += r_mem[k];
                p[k * 2 + 1] += s_mem[k];
            }
        }
    }

    // 3. ODE: when !freeze_ode, use caching forward for backward pass.
    //    When freeze_ode, use the fast path (GPU/FFT/sequential, no cache).
    //    When !freeze_ode AND GPU available, use GPU forward (backward recomputes internally).
    let _t_ode = std::time::Instant::now();
    let (kerr_out, ode_caches, ode_device): (Vec<Vec<f32>>, Option<Vec<crate::common::ode_backward::OdeForwardCache>>, &str) =
    if !freeze_ode && ping_pong.is_some() {
        // GPU forward for learnable ODE — backward will recompute via gpu_kerr_ode_backward_batch
        let gpu_be = ping_pong.unwrap().1;
        let out = gpu_be.gpu_kerr_ode_batch_fused(&weights.kerr, &precond);
        (out, None, "GPU-learnable")
    } else if !freeze_ode {
        // CPU caching forward — stores intermediates for backward
        let mut outs = Vec::with_capacity(t);
        let mut caches = Vec::with_capacity(t);
        for p in &precond {
            let (out, cache) = crate::common::ode_backward::ode_forward_with_cache(p, &weights.kerr);
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
                weights.kerr.alpha, weights.kerr.beta, weights.kerr.rk4_n_steps, st, &weights.kerr.rk4_weights)
        }).collect();
        (out, None, "CPU-FFT")
    } else {
        (cpu.kerr_ode_batch(&weights.kerr, &precond), None, "CPU-seq")
    };
    let _ode_dur = _t_ode.elapsed();

    // No energy conservation — AGC handles magnitude regulation.
    // Coupling at α=0.1 with AGC ceiling=2.0 matches kerr-engine recipe.

    // 3b. Corrector plate: per-band phase rotation after ODE (Schmidt corrector).
    //     Magnitude preserved (rotation is orthogonal). Only phase changes.
    //     Zero-init = identity rotation = no-op until the model learns corrections.
    //     Precompute sin/cos once per layer — angles don't change during a step.
    let mut kerr_out = kerr_out;
    let corr_sincos: Vec<(f32, f32)> = if use_corrector {
        (0..n_bands).map(|k| weights.kerr.phase_correction[k].sin_cos()).collect()
    } else {
        vec![]
    };
    if use_corrector {
        for pos in &mut kerr_out {
            for k in 0..n_bands {
                let (sin_c, cos_c) = corr_sincos[k];
                let r = pos[2 * k];
                let s = pos[2 * k + 1];
                pos[2 * k]     = r * cos_c - s * sin_c;
                pos[2 * k + 1] = r * sin_c + s * cos_c;
            }
        }
    }

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

    let gpu_ode = ode_device == "GPU-learnable";
    let cache = FfnCache {
        input: x.to_vec(),
        mae_in_sq, mae_in_act, precond,
        kerr_out, mae_out_sq, mae_out_act,
        regulated,
        ode_caches,
        corrector_active: use_corrector,
        gpu_ode_backward: gpu_ode,
        corr_sincos,
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
    //     This is the gradient w.r.t. the CORRECTED kerr_out (post-corrector).
    let d_corrected = cpu.vec_add_batch(&d_regulated, &d_kerr_from_mae);

    // ─── Corrector backward: inverse rotation + d_correction ───
    let (d_kerr_out, d_phase_correction): (Vec<Vec<f32>>, Option<Vec<f32>>) =
    if cache.corrector_active {
        // Corrector is active — rotate gradients back and accumulate d_correction
        // Use precomputed sin/cos from forward (same angles, no recomputation)
        let mut d_raw = Vec::with_capacity(t);
        let mut d_corr = vec![0.0f32; n_bands];
        for pos in 0..t {
            let mut d_pos = vec![0.0f32; n_embd];
            for k in 0..n_bands {
                let (sin_c, cos_c) = cache.corr_sincos[k];
                let d_cr = d_corrected[pos][2 * k];
                let d_cs = d_corrected[pos][2 * k + 1];

                // Inverse rotation for input gradient
                d_pos[2 * k]     =  cos_c * d_cr + sin_c * d_cs;
                d_pos[2 * k + 1] = -sin_c * d_cr + cos_c * d_cs;

                // Recover raw (pre-correction) r, s from corrected cache by inverse rotation
                let cr = cache.kerr_out[pos][2 * k];
                let cs = cache.kerr_out[pos][2 * k + 1];
                let raw_r =  cos_c * cr + sin_c * cs;
                let raw_s = -sin_c * cr + cos_c * cs;

                // d_correction[k] accumulated across positions
                d_corr[k] += d_cr * (-raw_r * sin_c - raw_s * cos_c)
                           + d_cs * ( raw_r * cos_c - raw_s * sin_c);
            }
            d_raw.push(d_pos);
        }
        (d_raw, Some(d_corr))
    } else {
        (d_corrected, None)
    };

    // ─── ODE backward ───
    // Three paths: GPU (recomputes forward), CPU (uses cached intermediates), identity (frozen)
    let (d_precond, ode_param_grads): (Vec<Vec<f32>>, Option<Vec<crate::common::ode_backward::OdeParamGrads>>) =
    if cache.gpu_ode_backward {
        // GPU ODE backward — recomputes forward internally, returns reduced gradients
        let gpu_be = ping_pong.unwrap().1;
        let (d_inputs, d_gamma_raw, _d_omega, d_alpha, d_beta, d_chi) =
            gpu_be.gpu_kerr_ode_backward_batch(&d_kerr_out, &cache.precond, &weights.kerr);
        // Convert to per-position OdeParamGrads format (already summed across positions by GPU)
        // Create one synthetic OdeParamGrads with the full sum, rest zero
        let mut param_grads = Vec::with_capacity(t);
        for pos in 0..t {
            param_grads.push(crate::common::ode_backward::OdeParamGrads {
                d_gamma_raw: if pos == 0 { d_gamma_raw.clone() } else { vec![0.0; n_bands] },
                d_alpha: if pos == 0 { d_alpha } else { 0.0 },
                d_beta: if pos == 0 { d_beta } else { 0.0 },
                d_chi: if pos == 0 { d_chi } else { 0.0 },
                d_rk4_weights: [0.0; 4], // GPU backward doesn't compute RK4 weight grads yet
            });
        }
        (d_inputs, Some(param_grads))
    } else if let Some(ref ode_caches) = cache.ode_caches {
        // CPU ODE backward — full backward through cached RK4
        let mut d_preconds = Vec::with_capacity(t);
        let mut param_grads = Vec::with_capacity(t);
        for (pos, d_ko) in d_kerr_out.iter().enumerate() {
            let (d_p, pg) = crate::common::ode_backward::ode_backward(d_ko, &ode_caches[pos], &weights.kerr);
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
    let (d_kerr_gamma_raw, d_kerr_alpha, d_kerr_beta, d_rk4_weights) = if let Some(ref pg_vec) = ode_param_grads {
        let nb = pg_vec[0].d_gamma_raw.len();
        let mut d_gr = vec![0.0f32; nb];
        let mut d_a = 0.0f32;
        let mut d_b = 0.0f32;
        let mut d_rw = [0.0f32; 4];
        for pg in pg_vec {
            for k in 0..nb { d_gr[k] += pg.d_gamma_raw[k]; }
            d_a += pg.d_alpha;
            d_b += pg.d_beta;
            for w in 0..4 { d_rw[w] += pg.d_rk4_weights[w]; }
        }
        (Some(d_gr), Some(d_a), Some(d_b), Some(d_rw))
    } else {
        (None, None, None, None)
    };

    let grads = FfnGrads {
        d_out_proj_w, d_out_proj_b,
        d_mae_out_pr_w, d_mae_out_pr_b,
        d_mae_out_sq_w, d_mae_out_sq_b,
        d_mae_in_pr_w, d_mae_in_pr_b,
        d_mae_in_sq_w, d_mae_in_sq_b,
        d_kerr_gamma_raw, d_kerr_alpha, d_kerr_beta,
        d_phase_correction,
        d_rk4_weights,
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
    /// ODE forward caches for backward pass — None when GPU or --freeze-ode
    pub ode_caches: Option<Vec<crate::common::ode_backward::OdeForwardCache>>,
    /// Whether corrector plate was applied in forward (gates backward)
    pub corrector_active: bool,
    /// GPU ODE backward path: precond stored, backward recomputes via GPU
    pub gpu_ode_backward: bool,
    /// Precomputed corrector sin/cos — avoids recomputing in backward.
    /// Empty when corrector is inactive.
    pub corr_sincos: Vec<(f32, f32)>,
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
    pub d_phase_correction: Option<Vec<f32>>,  // [n_bands] corrector plate gradient
    pub d_rk4_weights: Option<[f32; 4]>,  // RK4 combination weights gradient
}
