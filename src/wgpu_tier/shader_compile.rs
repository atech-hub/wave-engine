//! Shader compilation — all compute pipeline creation for GpuBackend.
//!
//! Each shader gets its own compile_* function. `compile_all` orchestrates
//! them and returns a `CompiledShaders` struct with all pipelines + layouts.

use super::bind_helpers::{storage_ro, storage_rw, uniform_entry};

/// All compiled compute pipelines and their bind group layouts.
pub(crate) struct CompiledShaders {
    pub matvec_pipeline: wgpu::ComputePipeline,
    pub matvec_layout: wgpu::BindGroupLayout,
    pub layer_norm_pipeline: wgpu::ComputePipeline,
    pub layer_norm_layout: wgpu::BindGroupLayout,
    pub kerr_deriv_pipeline: wgpu::ComputePipeline,
    pub kerr_deriv_layout: wgpu::BindGroupLayout,
    pub matvec_bwd_pipeline: wgpu::ComputePipeline,
    pub matvec_bwd_layout: wgpu::BindGroupLayout,
    pub layer_norm_bwd_pipeline: wgpu::ComputePipeline,
    pub layer_norm_bwd_layout: wgpu::BindGroupLayout,
    pub gelu_bwd_pipeline: wgpu::ComputePipeline,
    pub gelu_bwd_layout: wgpu::BindGroupLayout,
    pub attn_bwd_scores_pipeline: wgpu::ComputePipeline,
    pub attn_bwd_scores_layout: wgpu::BindGroupLayout,
    pub attn_bwd_dkv_pipeline: wgpu::ComputePipeline,
    pub attn_bwd_dkv_layout: wgpu::BindGroupLayout,
    pub outer_product_pipeline: wgpu::ComputePipeline,
    pub outer_product_layout: wgpu::BindGroupLayout,
    pub matvec_bwd_batch_pipeline: wgpu::ComputePipeline,
    pub matvec_bwd_batch_layout: wgpu::BindGroupLayout,
    pub matvec_batch_pipeline: wgpu::ComputePipeline,
    pub matvec_batch_layout: wgpu::BindGroupLayout,
    pub layer_norm_batch_pipeline: wgpu::ComputePipeline,
    pub layer_norm_batch_layout: wgpu::BindGroupLayout,
    pub kerr_deriv_batch_pipeline: wgpu::ComputePipeline,
    pub kerr_deriv_batch_layout: wgpu::BindGroupLayout,
    pub kerr_bwd_batch_pipeline: wgpu::ComputePipeline,
    pub kerr_bwd_batch_layout: wgpu::BindGroupLayout,
    pub vec_scale_add_pipeline: wgpu::ComputePipeline,
    pub vec_scale_add_layout: wgpu::BindGroupLayout,
    pub rk4_combine_pipeline: wgpu::ComputePipeline,
    pub rk4_combine_layout: wgpu::BindGroupLayout,
    pub deinterleave_pipeline: wgpu::ComputePipeline,
    pub deinterleave_layout: wgpu::BindGroupLayout,
    pub reinterleave_pipeline: wgpu::ComputePipeline,
    pub reinterleave_layout: wgpu::BindGroupLayout,
    pub gelu_pipeline: wgpu::ComputePipeline,
    pub gelu_layout: wgpu::BindGroupLayout,
    pub vec_add_pipeline: wgpu::ComputePipeline,
    pub vec_add_layout: wgpu::BindGroupLayout,
    pub vec_accumulate_pipeline: wgpu::ComputePipeline,
    pub vec_accumulate_layout: wgpu::BindGroupLayout,
    pub fft_convolve_pipeline: wgpu::ComputePipeline,
    pub fft_convolve_layout: wgpu::BindGroupLayout,
    pub kerr_perturbative_pipeline: wgpu::ComputePipeline,
    pub kerr_perturbative_layout: wgpu::BindGroupLayout,
    pub matvec_block_diag_pipeline: wgpu::ComputePipeline,
    pub matvec_block_diag_layout: wgpu::BindGroupLayout,
}

/// Helper: compile one shader module + pipeline with a given layout.
fn compile_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    entry_point: &str,
    layout_entries: &[wgpu::BindGroupLayoutEntry],
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}_layout")),
        entries: layout_entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_pl")),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}_pipeline")),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, bind_layout)
}

/// Compile all compute shaders and return the pipeline collection.
pub(crate) fn compile_all(device: &wgpu::Device) -> CompiledShaders {
    // ─── Forward shaders ───────────────────────────────────────────

    // matvec: tiled workgroup reduction
    let (matvec_pipeline, matvec_layout) = compile_pipeline(
        device, "matvec",
        include_str!("../../shaders/matvec_tiled.wgsl"), "matvec",
        &[storage_ro(0), storage_ro(1), storage_ro(2), storage_rw(3), uniform_entry(4)],
    );

    // layer_norm
    let (layer_norm_pipeline, layer_norm_layout) = compile_pipeline(
        device, "layer_norm",
        include_str!("../../shaders/layer_norm.wgsl"), "layer_norm",
        &[storage_ro(0), storage_ro(1), storage_ro(2), storage_rw(3), uniform_entry(4)],
    );

    // Kerr derivative
    let (kerr_deriv_pipeline, kerr_deriv_layout) = compile_pipeline(
        device, "kerr_derivative",
        include_str!("../../shaders/kerr_step.wgsl"), "kerr_derivative",
        &[storage_ro(0), storage_ro(1), storage_rw(2), storage_rw(3),
          storage_ro(4), storage_ro(5), uniform_entry(6), storage_ro(7)],
    );

    // ─── Backward shaders ──────────────────────────────────────────

    // matvec_backward: d_x = W^T @ d_y
    let (matvec_bwd_pipeline, matvec_bwd_layout) = compile_pipeline(
        device, "matvec_backward",
        include_str!("../../shaders/matvec_backward_tiled.wgsl"), "matvec_backward",
        &[storage_ro(0), storage_ro(1), storage_rw(2), uniform_entry(3)],
    );

    // layer_norm_backward
    let (layer_norm_bwd_pipeline, layer_norm_bwd_layout) = compile_pipeline(
        device, "layer_norm_backward",
        include_str!("../../shaders/layer_norm_backward.wgsl"), "layer_norm_backward",
        &[storage_ro(0), storage_ro(1), storage_ro(2), storage_rw(3), uniform_entry(4)],
    );

    // gelu_backward
    let (gelu_bwd_pipeline, gelu_bwd_layout) = compile_pipeline(
        device, "gelu_backward",
        include_str!("../../shaders/gelu_backward.wgsl"), "gelu_backward",
        &[storage_ro(0), storage_ro(1), storage_rw(2), uniform_entry(3)],
    );

    // ─── Attention backward shaders ────────────────────────────────

    // attn_backward_scores
    let (attn_bwd_scores_pipeline, attn_bwd_scores_layout) = compile_pipeline(
        device, "attn_backward_scores",
        include_str!("../../shaders/attn_backward_scores.wgsl"), "attn_backward_scores",
        &[storage_ro(0), storage_ro(1), storage_ro(2), storage_ro(3), storage_ro(4),
          storage_rw(5), storage_rw(6), uniform_entry(7)],
    );

    // attn_backward_dkv
    let (attn_bwd_dkv_pipeline, attn_bwd_dkv_layout) = compile_pipeline(
        device, "attn_backward_dkv",
        include_str!("../../shaders/attn_backward_dkv.wgsl"), "attn_backward_dkv",
        &[storage_ro(0), storage_ro(1), storage_ro(2), storage_ro(3),
          storage_rw(4), storage_rw(5), uniform_entry(6)],
    );

    // ─── Outer product ─────────────────────────────────────────────

    let (outer_product_pipeline, outer_product_layout) = compile_pipeline(
        device, "outer_product",
        include_str!("../../shaders/outer_product.wgsl"), "outer_product",
        &[storage_ro(0), storage_ro(1), storage_rw(2), storage_rw(3), uniform_entry(4)],
    );

    // ─── Batched shaders ───────────────────────────────────────────

    // batched matvec backward (tiled + Kahan)
    let (matvec_bwd_batch_pipeline, matvec_bwd_batch_layout) = compile_pipeline(
        device, "matvec_backward_batch",
        include_str!("../../shaders/matvec_backward_batch_tiled_kahan.wgsl"), "matvec_backward_batch",
        &[storage_ro(0), storage_ro(1), storage_rw(2), uniform_entry(3)],
    );

    // batched matvec forward (tiled + Kahan)
    println!("  Precision: tiled + Kahan (ping-pong handles correctness, Kahan handles utilisation)");
    let (matvec_batch_pipeline, matvec_batch_layout) = compile_pipeline(
        device, "matvec_batch",
        include_str!("../../shaders/matvec_batch_tiled_kahan.wgsl"), "matvec_batch",
        &[storage_ro(0), storage_ro(1), storage_ro(2), storage_rw(3), uniform_entry(4)],
    );

    // batched layer norm
    let (layer_norm_batch_pipeline, layer_norm_batch_layout) = compile_pipeline(
        device, "layer_norm_batch",
        include_str!("../../shaders/layer_norm_batch.wgsl"), "layer_norm_batch",
        &[storage_ro(0), storage_ro(1), storage_ro(2), storage_rw(3), uniform_entry(4)],
    );

    // batched Kerr derivative
    let (kerr_deriv_batch_pipeline, kerr_deriv_batch_layout) = compile_pipeline(
        device, "kerr_derivative_batch",
        include_str!("../../shaders/kerr_step_batch.wgsl"), "kerr_derivative_batch",
        &[storage_ro(0), storage_ro(1), storage_rw(2), storage_rw(3),
          storage_ro(4), storage_ro(5), uniform_entry(6), storage_ro(7)],
    );

    // batched Kerr backward (13 bindings — merged alpha+beta to stay within 12 storage limit)
    let kerr_bwd_batch_layout_entries = [
        storage_ro(0), storage_ro(1), storage_ro(2), storage_ro(3),  // r, s, gamma, omega
        storage_ro(4), storage_ro(5),                                 // d_dr, d_ds
        storage_rw(6), storage_rw(7),                                 // d_r, d_s
        storage_rw(8), storage_rw(9),                                 // d_gamma, d_omega
        storage_rw(10),                                               // d_ab_partial [2*n_pos*n_bands]: alpha+beta packed
        storage_rw(11),                                               // d_chi_partial
        uniform_entry(12),                                            // params
    ];
    let (kerr_bwd_batch_pipeline, kerr_bwd_batch_layout) = compile_pipeline(
        device, "kerr_backward_batch",
        include_str!("../../shaders/kerr_backward_batch.wgsl"), "kerr_backward_batch",
        &kerr_bwd_batch_layout_entries,
    );

    // ─── RK4 utility shaders ───────────────────────────────────────

    // vec_scale_add: y = a + scale * b
    let (vec_scale_add_pipeline, vec_scale_add_layout) = compile_pipeline(
        device, "vec_scale_add",
        include_str!("../../shaders/vec_scale_add.wgsl"), "vec_scale_add",
        &[storage_ro(0), storage_ro(1), storage_rw(2), uniform_entry(3)],
    );

    // rk4_combine: y = base + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    let (rk4_combine_pipeline, rk4_combine_layout) = compile_pipeline(
        device, "rk4_combine",
        include_str!("../../shaders/rk4_combine.wgsl"), "rk4_combine",
        &[storage_ro(0), storage_ro(1), storage_ro(2), storage_ro(3),
          storage_ro(4), storage_rw(5), uniform_entry(6)],
    );

    // ─── Fused FFN chain shaders ───────────────────────────────────

    // deinterleave
    let (deinterleave_pipeline, deinterleave_layout) = compile_pipeline(
        device, "deinterleave",
        include_str!("../../shaders/deinterleave.wgsl"), "deinterleave",
        &[storage_ro(0), storage_rw(1), storage_rw(2), uniform_entry(3)],
    );

    // reinterleave
    let (reinterleave_pipeline, reinterleave_layout) = compile_pipeline(
        device, "reinterleave",
        include_str!("../../shaders/reinterleave.wgsl"), "reinterleave",
        &[storage_ro(0), storage_ro(1), storage_rw(2), uniform_entry(3)],
    );

    // GELU (element-wise)
    let (gelu_pipeline, gelu_layout) = compile_pipeline(
        device, "gelu",
        include_str!("../../shaders/gelu.wgsl"), "gelu",
        &[storage_ro(0), storage_rw(1), uniform_entry(2)],
    );

    // vec_add: y = a + b
    let (vec_add_pipeline, vec_add_layout) = compile_pipeline(
        device, "vec_add",
        include_str!("../../shaders/vec_add.wgsl"), "vec_add",
        &[storage_ro(0), storage_ro(1), storage_rw(2), uniform_entry(3)],
    );

    // vec_accumulate: a[i] += b[i]
    let (vec_accumulate_pipeline, vec_accumulate_layout) = compile_pipeline(
        device, "vec_accumulate",
        include_str!("../../shaders/vec_accumulate.wgsl"), "vec_accumulate",
        &[storage_rw(0), storage_ro(1), uniform_entry(2)],
    );

    // ─── Specialty shaders ─────────────────────────────────────────

    // FFT 512-point convolution (OFDM-inspired ODE)
    let fft_convolve_layout_entries = [
        wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
    ];
    let (fft_convolve_pipeline, fft_convolve_layout) = compile_pipeline(
        device, "fft_convolve",
        include_str!("../../shaders/fft_512.wgsl"), "fft_convolve",
        &fft_convolve_layout_entries,
    );

    // perturbative Kerr-ODE (single-dispatch analytical)
    let (kerr_perturbative_pipeline, kerr_perturbative_layout) = compile_pipeline(
        device, "kerr_perturbative_batch",
        include_str!("../../shaders/kerr_perturbative_batch.wgsl"), "kerr_perturbative_batch",
        &[storage_ro(0), storage_ro(1), storage_rw(2), storage_rw(3),
          storage_ro(4), storage_ro(5), storage_ro(6), uniform_entry(7)],
    );

    // block-diagonal batched matvec
    let (matvec_block_diag_pipeline, matvec_block_diag_layout) = compile_pipeline(
        device, "matvec_block_diagonal_batch",
        include_str!("../../shaders/matvec_block_diagonal_batch.wgsl"), "matvec_block_diagonal_batch",
        &[storage_ro(0), storage_ro(1), storage_ro(2), storage_rw(3), uniform_entry(4)],
    );

    CompiledShaders {
        matvec_pipeline,
        matvec_layout,
        layer_norm_pipeline,
        layer_norm_layout,
        kerr_deriv_pipeline,
        kerr_deriv_layout,
        matvec_bwd_pipeline,
        matvec_bwd_layout,
        layer_norm_bwd_pipeline,
        layer_norm_bwd_layout,
        gelu_bwd_pipeline,
        gelu_bwd_layout,
        attn_bwd_scores_pipeline,
        attn_bwd_scores_layout,
        attn_bwd_dkv_pipeline,
        attn_bwd_dkv_layout,
        outer_product_pipeline,
        outer_product_layout,
        matvec_bwd_batch_pipeline,
        matvec_bwd_batch_layout,
        matvec_batch_pipeline,
        matvec_batch_layout,
        layer_norm_batch_pipeline,
        layer_norm_batch_layout,
        kerr_deriv_batch_pipeline,
        kerr_deriv_batch_layout,
        kerr_bwd_batch_pipeline,
        kerr_bwd_batch_layout,
        vec_scale_add_pipeline,
        vec_scale_add_layout,
        rk4_combine_pipeline,
        rk4_combine_layout,
        deinterleave_pipeline,
        deinterleave_layout,
        reinterleave_pipeline,
        reinterleave_layout,
        gelu_pipeline,
        gelu_layout,
        vec_add_pipeline,
        vec_add_layout,
        vec_accumulate_pipeline,
        vec_accumulate_layout,
        fft_convolve_pipeline,
        fft_convolve_layout,
        kerr_perturbative_pipeline,
        kerr_perturbative_layout,
        matvec_block_diag_pipeline,
        matvec_block_diag_layout,
    }
}
