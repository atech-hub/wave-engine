//! GPU pipeline setup — struct definition, initialization, and cache management.
//!
//! This module contains GpuBackend's struct and constructors (new, with_device_index,
//! from_adapter). Uniform param structs live in gpu_uniforms.rs, bind group helpers
//! in bind_helpers.rs, and shader compilation in shader_compile.rs.

use std::sync::Mutex;

use crate::gpu_buffers::GpuBufferPool;

// Re-export uniform structs and bind helpers so existing `use crate::gpu_pipelines::*` still works.
pub(crate) use super::gpu_uniforms::*;
pub(crate) use super::bind_helpers::*;

/// GPU backend — dispatches to WGSL compute shaders.
pub struct GpuBackend {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// Buffer pool for reusable GPU buffers (eliminates per-dispatch allocation).
    pub(crate) pool: Mutex<GpuBufferPool>,
    /// Resident weight buffers — uploaded once, updated after Adam step.
    pub(crate) resident: Mutex<Option<crate::gpu_resident::ResidentWeightBuffers>>,
    /// Block counter for resident dispatch — tracks which FFN block is being processed.
    /// Reset at start of each forward pass (update_weights call).
    pub(crate) ffn_block_counter: std::sync::atomic::AtomicUsize,
    pub(crate) matvec_pipeline: wgpu::ComputePipeline,
    pub(crate) matvec_layout: wgpu::BindGroupLayout,
    pub(crate) layer_norm_pipeline: wgpu::ComputePipeline,
    pub(crate) layer_norm_layout: wgpu::BindGroupLayout,
    pub(crate) kerr_deriv_pipeline: wgpu::ComputePipeline,
    pub(crate) kerr_deriv_layout: wgpu::BindGroupLayout,
    // Backward shaders
    pub(crate) matvec_bwd_pipeline: wgpu::ComputePipeline,
    pub(crate) matvec_bwd_layout: wgpu::BindGroupLayout,
    pub(crate) layer_norm_bwd_pipeline: wgpu::ComputePipeline,
    pub(crate) layer_norm_bwd_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)] // Ready for FFN backward restructuring
    pub(crate) gelu_bwd_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    pub(crate) gelu_bwd_layout: wgpu::BindGroupLayout,
    // Attention backward (two dispatches)
    pub(crate) attn_bwd_scores_pipeline: wgpu::ComputePipeline,
    pub(crate) attn_bwd_scores_layout: wgpu::BindGroupLayout,
    pub(crate) attn_bwd_dkv_pipeline: wgpu::ComputePipeline,
    pub(crate) attn_bwd_dkv_layout: wgpu::BindGroupLayout,
    // Batched outer product: d_w = D_Y^T @ X
    pub(crate) outer_product_pipeline: wgpu::ComputePipeline,
    pub(crate) outer_product_layout: wgpu::BindGroupLayout,
    // Batched matvec backward: d_x[pos] = W^T @ d_y[pos] for all positions
    pub(crate) matvec_bwd_batch_pipeline: wgpu::ComputePipeline,
    pub(crate) matvec_bwd_batch_layout: wgpu::BindGroupLayout,
    // Batched matvec forward: y[pos] = W @ x[pos] + b for all positions
    pub(crate) matvec_batch_pipeline: wgpu::ComputePipeline,
    pub(crate) matvec_batch_layout: wgpu::BindGroupLayout,
    // Batched layer norm: one workgroup per position
    pub(crate) layer_norm_batch_pipeline: wgpu::ComputePipeline,
    pub(crate) layer_norm_batch_layout: wgpu::BindGroupLayout,
    // Batched Kerr derivative: all positions in one dispatch
    pub(crate) kerr_deriv_batch_pipeline: wgpu::ComputePipeline,
    pub(crate) kerr_deriv_batch_layout: wgpu::BindGroupLayout,
    // Batched Kerr derivative backward: all positions in one dispatch
    pub(crate) kerr_bwd_batch_pipeline: wgpu::ComputePipeline,
    pub(crate) kerr_bwd_batch_layout: wgpu::BindGroupLayout,
    // Fused RK4 utilities: vec_scale_add and rk4_combine for chained dispatch
    pub(crate) vec_scale_add_pipeline: wgpu::ComputePipeline,
    pub(crate) vec_scale_add_layout: wgpu::BindGroupLayout,
    pub(crate) rk4_combine_pipeline: wgpu::ComputePipeline,
    pub(crate) rk4_combine_layout: wgpu::BindGroupLayout,
    // Deinterleave/reinterleave for fused FFN chain
    pub(crate) deinterleave_pipeline: wgpu::ComputePipeline,
    pub(crate) deinterleave_layout: wgpu::BindGroupLayout,
    pub(crate) reinterleave_pipeline: wgpu::ComputePipeline,
    pub(crate) reinterleave_layout: wgpu::BindGroupLayout,
    // GELU (element-wise, for fused chain)
    pub(crate) gelu_pipeline: wgpu::ComputePipeline,
    pub(crate) gelu_layout: wgpu::BindGroupLayout,
    // Vec add (element-wise y = a + b, for fused chain)
    pub(crate) vec_add_pipeline: wgpu::ComputePipeline,
    pub(crate) vec_add_layout: wgpu::BindGroupLayout,
    pub(crate) vec_accumulate_pipeline: wgpu::ComputePipeline,
    pub(crate) vec_accumulate_layout: wgpu::BindGroupLayout,
    // FFT 512-point convolution (OFDM-inspired ODE acceleration)
    pub(crate) fft_convolve_pipeline: wgpu::ComputePipeline,
    pub(crate) fft_convolve_layout: wgpu::BindGroupLayout,
    // Perturbative Kerr-ODE: single-dispatch analytical approximation (replaces 192-dispatch RK4)
    pub(crate) kerr_perturbative_pipeline: wgpu::ComputePipeline,
    pub(crate) kerr_perturbative_layout: wgpu::BindGroupLayout,
    // Block-diagonal batched matvec: N groups of group_size dims (replaces dense matvec for out_proj)
    pub(crate) matvec_block_diag_pipeline: wgpu::ComputePipeline,
    pub(crate) matvec_block_diag_layout: wgpu::BindGroupLayout,
}

impl GpuBackend {
    /// Initialize GPU device and compile all compute shaders.
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("Failed to find GPU adapter");

        Self::from_adapter(adapter)
    }

    /// Initialize GPU with a specific adapter selected by index.
    pub fn with_device_index(idx: usize) -> Self {
        let instance = wgpu::Instance::default();
        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).into_iter().collect();
        assert!(
            idx < adapters.len(),
            "GPU device index {idx} out of range ({} adapters available)",
            adapters.len()
        );
        // enumerate_adapters returns owned adapters — take the one we want
        let mut adapters = adapters;
        let adapter = adapters.swap_remove(idx);

        Self::from_adapter(adapter)
    }

    /// Shared constructor: compile shaders and build pipelines from a chosen adapter.
    fn from_adapter(adapter: wgpu::Adapter) -> Self {
        println!("  GPU adapter: {}", adapter.get_info().name);
        println!("  Backend:     {:?}", adapter.get_info().backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("kerr-engine-backend"),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 12,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            },
            None,
        ))
        .expect("Failed to get GPU device");

        let s = super::shader_compile::compile_all(&device);

        Self {
            device,
            queue,
            matvec_pipeline: s.matvec_pipeline,
            matvec_layout: s.matvec_layout,
            layer_norm_pipeline: s.layer_norm_pipeline,
            layer_norm_layout: s.layer_norm_layout,
            kerr_deriv_pipeline: s.kerr_deriv_pipeline,
            kerr_deriv_layout: s.kerr_deriv_layout,
            matvec_bwd_pipeline: s.matvec_bwd_pipeline,
            matvec_bwd_layout: s.matvec_bwd_layout,
            layer_norm_bwd_pipeline: s.layer_norm_bwd_pipeline,
            layer_norm_bwd_layout: s.layer_norm_bwd_layout,
            gelu_bwd_pipeline: s.gelu_bwd_pipeline,
            gelu_bwd_layout: s.gelu_bwd_layout,
            attn_bwd_scores_pipeline: s.attn_bwd_scores_pipeline,
            attn_bwd_scores_layout: s.attn_bwd_scores_layout,
            attn_bwd_dkv_pipeline: s.attn_bwd_dkv_pipeline,
            attn_bwd_dkv_layout: s.attn_bwd_dkv_layout,
            outer_product_pipeline: s.outer_product_pipeline,
            outer_product_layout: s.outer_product_layout,
            matvec_bwd_batch_pipeline: s.matvec_bwd_batch_pipeline,
            matvec_bwd_batch_layout: s.matvec_bwd_batch_layout,
            matvec_batch_pipeline: s.matvec_batch_pipeline,
            matvec_batch_layout: s.matvec_batch_layout,
            layer_norm_batch_pipeline: s.layer_norm_batch_pipeline,
            layer_norm_batch_layout: s.layer_norm_batch_layout,
            kerr_deriv_batch_pipeline: s.kerr_deriv_batch_pipeline,
            kerr_deriv_batch_layout: s.kerr_deriv_batch_layout,
            kerr_bwd_batch_pipeline: s.kerr_bwd_batch_pipeline,
            kerr_bwd_batch_layout: s.kerr_bwd_batch_layout,
            vec_scale_add_pipeline: s.vec_scale_add_pipeline,
            vec_scale_add_layout: s.vec_scale_add_layout,
            rk4_combine_pipeline: s.rk4_combine_pipeline,
            rk4_combine_layout: s.rk4_combine_layout,
            deinterleave_pipeline: s.deinterleave_pipeline,
            deinterleave_layout: s.deinterleave_layout,
            reinterleave_pipeline: s.reinterleave_pipeline,
            reinterleave_layout: s.reinterleave_layout,
            gelu_pipeline: s.gelu_pipeline,
            gelu_layout: s.gelu_layout,
            vec_add_pipeline: s.vec_add_pipeline,
            vec_add_layout: s.vec_add_layout,
            vec_accumulate_pipeline: s.vec_accumulate_pipeline,
            vec_accumulate_layout: s.vec_accumulate_layout,
            fft_convolve_pipeline: s.fft_convolve_pipeline,
            fft_convolve_layout: s.fft_convolve_layout,
            kerr_perturbative_pipeline: s.kerr_perturbative_pipeline,
            kerr_perturbative_layout: s.kerr_perturbative_layout,
            matvec_block_diag_pipeline: s.matvec_block_diag_pipeline,
            matvec_block_diag_layout: s.matvec_block_diag_layout,
            pool: Mutex::new(GpuBufferPool::new()),
            resident: Mutex::new(None),
            ffn_block_counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Invalidate cached weight buffers. Call after optimizer step.
    pub fn invalidate_weight_cache(&self) {
        self.pool.lock().unwrap().invalidate_weights();
    }
}
