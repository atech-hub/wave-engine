//! FFN GPU pipeline — ping-pong buffer management for out_proj.
//!
//! Forward writes regulated_all to Buffer A, computes y = W @ x in Buffer B.
//! Backward reads Buffer A (same bits as forward) for d_W, computes d_x in Buffer D.
//! Single encoder, single submit. No CPU involvement between forward and backward.

use crate::gpu_pipelines::*;
use wgpu::util::DeviceExt;

/// Persistent GPU buffers for the FFN out_proj pipeline.
/// Allocated once at model init, reused every block.
pub struct FfnGpuBuffers {
    pub regulated: wgpu::Buffer,    // Buffer A: regulated_all from forward
    pub ffn_out: wgpu::Buffer,      // Buffer B: forward output y = W @ x
    pub d_ffn_out: wgpu::Buffer,    // Buffer C: backward gradient input
    pub d_regulated: wgpu::Buffer,  // Buffer D: backward d_x output
    pub d_w: wgpu::Buffer,          // Weight gradient accumulator
    pub d_b: wgpu::Buffer,          // Bias gradient accumulator
    pub max_elements: usize,        // max n_pos * n_embd
    pub n_embd: usize,
}

impl FfnGpuBuffers {
    pub fn new(device: &wgpu::Device, max_pos: usize, n_embd: usize) -> Self {
        let elem = max_pos * n_embd;
        let make = |size: usize| device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            regulated: make(elem),
            ffn_out: make(elem),
            d_ffn_out: make(elem),
            d_regulated: make(elem),
            d_w: make(n_embd * n_embd),
            d_b: make(n_embd),
            max_elements: elem,
            n_embd,
        }
    }

    /// Forward: upload regulated_all to Buffer A, compute y = W @ x on GPU.
    /// Buffer A stays in VRAM for backward to read.
    pub fn forward_out_proj(
        &self,
        gpu: &GpuBackend,
        regulated_flat: &[f32],  // [n_pos * n_embd]
        w_flat: &[f32],          // [n_embd * n_embd] row-major
        b: &[f32],               // [n_embd]
        n_pos: usize,
        n_embd: usize,
    ) -> Vec<f32> {
        // Upload regulated_all to Buffer A (stays in VRAM!)
        gpu.queue.write_buffer(&self.regulated, 0, bytemuck::cast_slice(regulated_flat));

        // Upload weights
        let w_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(w_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = MatvecBatchParams {
            out_dim: n_embd as u32, in_dim: n_embd as u32,
            n_pos: n_pos as u32, use_bias: 1,
        };
        let params_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &gpu.matvec_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.regulated.as_entire_binding() }, // Buffer A
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.ffn_out.as_entire_binding() },   // Buffer B
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&gpu.matvec_batch_pipeline);
          pass.set_bind_group(0, &bind_group, &[]);
          pass.dispatch_workgroups(n_embd as u32, n_pos as u32, 1); }
        gpu.queue.submit(Some(encoder.finish()));

        // Readback Buffer B for residual addition on CPU
        gpu.readback(&self.ffn_out, n_pos * n_embd)
    }

    /// Backward: compute d_x and d_W on GPU, reading regulated_all from Buffer A.
    /// Buffer A has the EXACT bits from forward — no recomputation, no mismatch.
    pub fn backward_out_proj(
        &self,
        gpu: &GpuBackend,
        d_ffn_out_flat: &[f32],  // [n_pos * n_embd]
        w_flat: &[f32],          // [n_embd * n_embd] row-major
        n_pos: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {  // (d_regulated, d_w, d_b)
        // Upload d_ffn_out to Buffer C
        gpu.queue.write_buffer(&self.d_ffn_out, 0, bytemuck::cast_slice(d_ffn_out_flat));

        // Zero d_w and d_b buffers
        let zeros_w = vec![0.0f32; n_embd * n_embd];
        let zeros_b = vec![0.0f32; n_embd];
        gpu.queue.write_buffer(&self.d_w, 0, bytemuck::cast_slice(&zeros_w));
        gpu.queue.write_buffer(&self.d_b, 0, bytemuck::cast_slice(&zeros_b));

        // Weights for backward d_x (transpose on CPU, use SAME forward shader)
        let mut wt_flat = vec![0.0f32; n_embd * n_embd];
        for i in 0..n_embd {
            for j in 0..n_embd {
                wt_flat[j * n_embd + i] = w_flat[i * n_embd + j];
            }
        }
        let wt_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&wt_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let w_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(w_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy_bias = [0.0f32];
        let db_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&dummy_bias),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // d_x params (transposed: in=n_embd, out=n_embd)
        let dx_params = MatvecBatchParams {
            out_dim: n_embd as u32, in_dim: n_embd as u32,
            n_pos: n_pos as u32, use_bias: 0,
        };
        let dx_params_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&dx_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // d_W params (outer product)
        let dw_params = OuterProductParams {
            out_dim: n_embd as u32, in_dim: n_embd as u32,
            n_pos: n_pos as u32, compute_bias: 1,
        };
        let dw_params_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&dw_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Bind groups
        let dx_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &gpu.matvec_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wt_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.d_ffn_out.as_entire_binding() },   // Buffer C
                wgpu::BindGroupEntry { binding: 2, resource: db_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.d_regulated.as_entire_binding() },  // Buffer D
                wgpu::BindGroupEntry { binding: 4, resource: dx_params_buf.as_entire_binding() },
            ],
        });

        let dw_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &gpu.outer_product_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.d_ffn_out.as_entire_binding() },  // d_y: Buffer C
                wgpu::BindGroupEntry { binding: 1, resource: self.regulated.as_entire_binding() },  // x: Buffer A (FROM FORWARD!)
                wgpu::BindGroupEntry { binding: 2, resource: self.d_w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.d_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dw_params_buf.as_entire_binding() },
            ],
        });

        // Single encoder: d_x and d_W in one submit
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        // d_x = W^T @ d_y (same forward shader, transposed weights)
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&gpu.matvec_batch_pipeline);
          pass.set_bind_group(0, &dx_bg, &[]);
          pass.dispatch_workgroups(n_embd as u32, n_pos as u32, 1); }
        // d_W = d_y @ x^T (outer product — reads Buffer A from forward!)
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&gpu.outer_product_pipeline);
          pass.set_bind_group(0, &dw_bg, &[]);
          pass.dispatch_workgroups(n_embd as u32, 1, 1); }
        gpu.queue.submit(Some(encoder.finish()));

        // Readback results
        let d_reg = gpu.readback(&self.d_regulated, n_pos * n_embd);
        let d_w = gpu.readback(&self.d_w, n_embd * n_embd);
        let d_b = gpu.readback(&self.d_b, n_embd);

        (d_reg, d_w, d_b)
    }
}
