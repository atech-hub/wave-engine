//! J10 runner: run full forward on two tiers (CPU vs wgpu) and diff section-by-section.
//!
//! The framework in `tier_parity.rs` compares two slices and reports violations.
//! This module is the driver that actually produces the two slices — it runs the
//! same model against the same tokens on CPU and on wgpu, then diffs every cached
//! section of the forward pass (per-block normed / attn_out / ffn_out plus FFN
//! sub-sections mae_in / precond / kerr_out / mae_out / regulated, plus final
//! pre/post layer-norm and logits).
//!
//! Tolerances are per-section: LINEAR for matmul/LN/projection outputs, ODE for
//! kerr_out (the RK4 integration accumulates FP noise across 16 sub-steps).

use crate::cpu::forward::{forward_with_cache, ForwardCache};
use crate::monitors::junctions::tier_parity::{
    check_outputs_2d, check_outputs_1d, ParityDiff, ParityReport, Tolerance,
};
use crate::{Dims, WavePacketModel};

/// Run the same forward pass on CPU and wgpu, diff every cached section.
///
/// Returns a ParityReport. Expect the wgpu tier to be initialised before calling
/// (uses `gpu_pipelines::GpuBackend::new()`, which panics if no GPU adapter is
/// available).
pub fn run_cpu_vs_wgpu_parity(
    model: &WavePacketModel,
    tokens: &[usize],
    dims: Dims,
) -> ParityReport {
    let n_bands = dims.n_bands;
    let seq_len = tokens.len();
    let n_embd = dims.n_embd;

    // Both tiers use the same stencil.
    let stencil = crate::fft_ode::StencilFft::new(n_bands);

    // CPU forward — all GPU args None.
    let cpu_cache: ForwardCache = forward_with_cache(
        model, tokens, dims,
        None,           // gpu backend
        None,           // ping_pong buffers
        None,           // full_gpu buffers
        Some(&stencil), // stencil (shared)
        None,           // gpu_kernel
        None,           // layer_agcs
        None,           // memory
    );

    // wgpu forward — init backend + per-forward buffers.
    let gpu_be = crate::gpu_pipelines::GpuBackend::new();
    let ffn_bufs = crate::wgpu_tier::ffn_gpu::FfnGpuBuffers::new(&gpu_be.device, seq_len, n_embd);
    let gpu_kernel = crate::fft_ode::GpuKernelFft::new(n_bands);

    let gpu_cache: ForwardCache = forward_with_cache(
        model, tokens, dims,
        Some(&gpu_be),
        Some((&ffn_bufs, &gpu_be)),
        None,
        Some(&stencil),
        Some((&gpu_kernel, &gpu_be)),
        None,
        None,
    );

    // ── Diff every section ──
    let mut sections: Vec<ParityDiff> = Vec::new();

    // Final output: logits and the LN surrounding it.
    sections.push(check_outputs_2d(
        "logits",
        &cpu_cache.logits,
        &gpu_cache.logits,
        Tolerance::LINEAR,
    ));
    sections.push(check_outputs_2d(
        "pre_ln_f",
        &cpu_cache.pre_ln_f,
        &gpu_cache.pre_ln_f,
        Tolerance::LINEAR,
    ));
    sections.push(check_outputs_2d(
        "post_ln_f",
        &cpu_cache.post_ln_f,
        &gpu_cache.post_ln_f,
        Tolerance::LINEAR,
    ));

    // Per-block sections.
    let n_blocks = cpu_cache.block_caches.len().min(gpu_cache.block_caches.len());
    for b in 0..n_blocks {
        let ca = &cpu_cache.block_caches[b];
        let cb = &gpu_cache.block_caches[b];
        sections.push(check_outputs_2d(
            &format!("b{b}.normed"),
            &ca.normed, &cb.normed, Tolerance::LINEAR,
        ));
        sections.push(check_outputs_2d(
            &format!("b{b}.attn_out"),
            &ca.attn_out, &cb.attn_out, Tolerance::LINEAR,
        ));
        sections.push(check_outputs_2d(
            &format!("b{b}.ffn_out"),
            &ca.ffn_out, &cb.ffn_out, Tolerance::ODE,
        ));

        // FFN sub-section diff when backend cache is populated on both sides.
        if let (Some(fa), Some(fb)) = (ca.ffn_backend_cache.as_ref(), cb.ffn_backend_cache.as_ref()) {
            sections.push(check_outputs_2d(
                &format!("b{b}.ffn.mae_in_sq"),
                &fa.mae_in_sq, &fb.mae_in_sq, Tolerance::LINEAR,
            ));
            sections.push(check_outputs_2d(
                &format!("b{b}.ffn.mae_in_act"),
                &fa.mae_in_act, &fb.mae_in_act, Tolerance::LINEAR,
            ));
            sections.push(check_outputs_2d(
                &format!("b{b}.ffn.precond"),
                &fa.precond, &fb.precond, Tolerance::LINEAR,
            ));
            sections.push(check_outputs_2d(
                &format!("b{b}.ffn.kerr_out"),
                &fa.kerr_out, &fb.kerr_out, Tolerance::ODE,
            ));
            sections.push(check_outputs_2d(
                &format!("b{b}.ffn.mae_out_sq"),
                &fa.mae_out_sq, &fb.mae_out_sq, Tolerance::LINEAR,
            ));
            sections.push(check_outputs_2d(
                &format!("b{b}.ffn.mae_out_act"),
                &fa.mae_out_act, &fb.mae_out_act, Tolerance::LINEAR,
            ));
            sections.push(check_outputs_2d(
                &format!("b{b}.ffn.regulated"),
                &fa.regulated, &fb.regulated, Tolerance::LINEAR,
            ));
        }
    }

    // Guard: unequal block count is itself a violation.
    if cpu_cache.block_caches.len() != gpu_cache.block_caches.len() {
        sections.push(check_outputs_1d(
            "block_count_mismatch",
            &[cpu_cache.block_caches.len() as f32],
            &[gpu_cache.block_caches.len() as f32],
            Tolerance::TIGHT,
        ));
    }

    ParityReport {
        tier_a: "cpu".to_string(),
        tier_b: "wgpu".to_string(),
        sections,
    }
}

/// Run the same forward pass on CPU and Candle, diff the final logits.
///
/// This is the *minimum viable* Candle parity check. Unlike the wgpu runner
/// (which diffs 13 sub-sections per block), Candle's block forward emits only
/// a single output tensor — not the intermediate FfnCache that the shared
/// `common/ffn.rs` pipeline produces. End-to-end logits parity is the honest
/// measurement available today; per-section instrumentation of the Candle
/// forward would be a separate task.
///
/// Weight bridge: we flatten the CPU model's params and load them into the
/// Candle VarMap via `load_wchk_params_into_varmap`, so both models run with
/// bit-identical weights. Forward is deterministic (no dropout), so any
/// divergence is pure FP-order effect from the tensor-ops vs hand-written CPU
/// math.
#[cfg(feature = "candle-backend")]
pub fn run_cpu_vs_candle_parity(
    model: &WavePacketModel,
    tokens: &[usize],
    dims: Dims,
    alpha: f32,
    beta: f32,
    phase_native: bool,
    use_rk4_dyn: bool,
    use_layer_scale: bool,
    use_harmonics: bool,
) -> Result<ParityReport, String> {
    use candle_core::Device;
    use candle_nn::VarMap;
    use crate::candle_tier::candle_model::model::CandleWaveModel;
    use crate::candle_tier::candle_checkpoint::checkpoint::load_wchk_params_into_varmap;

    let n_layers = model.blocks.len();
    let n_bands = dims.n_bands;
    let n_embd = dims.n_embd;
    let maestro_dim = dims.maestro_dim;
    let vocab_size = model.vocab_size;
    let out_proj_groups = if model.blocks[0].ffn.out_proj.n_groups() >= 1 {
        model.blocks[0].ffn.out_proj.n_groups()
    } else {
        1
    };

    // CPU forward — same call the wgpu runner uses so the two are head-to-head.
    let stencil = crate::fft_ode::StencilFft::new(n_bands);
    let cpu_cache: ForwardCache = forward_with_cache(
        model, tokens, dims,
        None, None, None, Some(&stencil), None, None, None,
    );

    // Candle: build a fresh VarMap + model, then bridge weights in. `Device::Cpu`
    // keeps the comparison honest — any CUDA/GPU non-determinism stays out of
    // the numbers until we explicitly test against a GPU device in a follow-up.
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let mut candle_model = CandleWaveModel::new(
        &varmap, vocab_size, &device,
        n_bands, dims.n_head, n_layers, maestro_dim, crate::RK4_STEPS, out_proj_groups,
        alpha, beta, dims.fwm_strength, phase_native,
    ).map_err(|e| format!("CandleWaveModel::new failed: {e:?}"))?;

    // Flatten CPU params and load into the Candle VarMap using the existing bridge.
    let cpu_params = crate::common::wave_model::flatten_params_ex(model, dims.tied);
    load_wchk_params_into_varmap(
        &varmap, &cpu_params,
        n_layers, n_embd, maestro_dim,
        vocab_size, out_proj_groups, n_bands,
        dims.learnable_ode, use_layer_scale, use_rk4_dyn, phase_native,
        &device,
    ).map_err(|e| format!("load_wchk_params_into_varmap failed: {e:?}"))?;

    // Candle requires attn_param_grads initialised (WaveAttentionCustomOp bwd
    // writes to it even when we don't read it back).
    candle_model.attn_param_grads = Some(
        crate::candle_tier::custom_attn::custom_attn::create_attn_grad_storage(n_layers),
    );

    // Route Candle's ODE through the CustomOp that calls into the canonical
    // CPU `ode_forward_with_cache` (or `split_band_forward_with_cache` when
    // dims.split_band is true), so the ODE step matches CPU bit-exactly.
    // Without this, Candle uses its autograd tensor-ops RK4 (ode.rs) which is
    // a separate implementation of the same math and will diverge in f32.
    candle_model.use_custom_op = true;
    candle_model.split_band = dims.split_band;
    candle_model.ode_param_grads = Some(
        crate::candle_tier::custom_ode::custom_ode::create_param_grad_storage(n_layers),
    );

    // Attention weights now match CPU by construction: both tiers call the
    // shared `init_block_attn` from seed 42 at the same RNG position, and
    // Candle advances its RNG past CPU's per-block FFN + post-block lm_head
    // draws to stay aligned for the next block. No runner-side injection.

    let logits_tensor = candle_model.forward(tokens)
        .map_err(|e| format!("CandleWaveModel::forward failed: {e:?}"))?;
    let candle_logits: Vec<Vec<f32>> = logits_tensor.to_vec2::<f32>()
        .map_err(|e| format!("logits.to_vec2 failed: {e:?}"))?;

    let mut sections: Vec<ParityDiff> = Vec::new();
    sections.push(check_outputs_2d(
        "logits",
        &cpu_cache.logits,
        &candle_logits,
        Tolerance::LINEAR,
    ));

    // Catch shape mismatch cheaply as its own section.
    let cpu_shape = (cpu_cache.logits.len(),
        cpu_cache.logits.first().map(|r| r.len()).unwrap_or(0));
    let candle_shape = (candle_logits.len(),
        candle_logits.first().map(|r| r.len()).unwrap_or(0));
    if cpu_shape != candle_shape {
        sections.push(check_outputs_1d(
            "logits_shape",
            &[cpu_shape.0 as f32, cpu_shape.1 as f32],
            &[candle_shape.0 as f32, candle_shape.1 as f32],
            Tolerance::TIGHT,
        ));
    }

    // Silence unused-var warnings on `use_harmonics` — currently it's only
    // consumed by load_wchk_params_into_varmap indirectly (flag compat), kept
    // on the signature so the caller can make the intent explicit.
    let _ = use_harmonics;

    Ok(ParityReport {
        tier_a: "cpu".to_string(),
        tier_b: "candle".to_string(),
        sections,
    })
}

/// Stub when candle-backend feature is off.
#[cfg(not(feature = "candle-backend"))]
pub fn run_cpu_vs_candle_parity(
    _model: &WavePacketModel,
    _tokens: &[usize],
    _dims: Dims,
    _alpha: f32,
    _beta: f32,
    _phase_native: bool,
    _use_rk4_dyn: bool,
    _use_layer_scale: bool,
    _use_harmonics: bool,
) -> Result<ParityReport, String> {
    Err("Candle parity requires: cargo build --features candle-backend".to_string())
}
