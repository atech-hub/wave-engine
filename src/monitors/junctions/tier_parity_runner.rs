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
