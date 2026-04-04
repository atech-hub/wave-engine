//! Batched forward dispatch helpers for GpuBackend.
//!
//! Contains the implementation bodies for linear_batch, linear_no_bias_batch,
//! and layer_norm_batch. Called from the ComputeBackend trait impl in dispatch.rs.

use crate::gpu_pipelines::*;
use wgpu::util::DeviceExt;

/// Batched matrix-vector multiply: y[pos] = W @ x[pos] + b for all positions.
pub(crate) fn linear_batch(
    backend: &GpuBackend,
    w: &[Vec<f32>],
    b: &[f32],
    xs: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    if xs.is_empty() { return vec![]; }
    let out_dim = w.len();
    let in_dim = if out_dim > 0 { w[0].len() } else { return vec![vec![]; xs.len()] };
    let n_pos = xs.len();

    // Cache weight buffer by pointer — avoid re-upload within same iteration
    let w_ptr = w.as_ptr() as usize;
    let b_ptr = b.as_ptr() as usize;
    let x_flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
    let out_total = n_pos * out_dim;
    let use_bias = 1u32;

    let mut pool = backend.pool.lock().unwrap();
    if !pool.has_weight(w_ptr) {
        let mut w_flat = Vec::with_capacity(out_dim * in_dim);
        for row in w { w_flat.extend_from_slice(row); }
        pool.cache_weight(&backend.device, &backend.queue, w_ptr, &w_flat);
    }
    if !pool.has_weight(b_ptr) {
        pool.cache_weight(&backend.device, &backend.queue, b_ptr, b);
    }
    // Dispatch with cached weight buffers — hold pool lock for buffer refs
    let x_buf = backend.storage_buf("x", &x_flat);
    pool.ensure_scratch(&backend.device, 0, (out_total * 4) as u64);
    let params = MatvecBatchParams { out_dim: out_dim as u32, in_dim: in_dim as u32, n_pos: n_pos as u32, use_bias };
    pool.write_uniform(&backend.device, &backend.queue, &params);
    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &backend.matvec_batch_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pool.weight_ref(w_ptr).as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pool.weight_ref(b_ptr).as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
        ],
    });
    let mut encoder = backend.device.create_command_encoder(&Default::default());
    { let mut pass = encoder.begin_compute_pass(&Default::default());
      pass.set_pipeline(&backend.matvec_batch_pipeline); pass.set_bind_group(0, &bind_group, &[]);
      pass.dispatch_workgroups(out_dim as u32, n_pos as u32, 1); }
    backend.queue.submit(Some(encoder.finish()));
    let y_flat = pool.readback_scratch(&backend.device, &backend.queue, 0, out_total);
    drop(pool);
    y_flat.chunks(out_dim).map(|c| c.to_vec()).collect()
}

/// Batched matrix-vector multiply without bias: y[pos] = W @ x[pos] for all positions.
pub(crate) fn linear_no_bias_batch(
    backend: &GpuBackend,
    w: &[Vec<f32>],
    xs: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    if xs.is_empty() { return vec![]; }
    let out_dim = w.len();
    let in_dim = if out_dim > 0 { w[0].len() } else { return vec![vec![]; xs.len()] };
    let n_pos = xs.len();

    let w_ptr = w.as_ptr() as usize;
    let x_flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
    let out_total = n_pos * out_dim;

    let mut pool = backend.pool.lock().unwrap();
    if !pool.has_weight(w_ptr) {
        let mut w_flat = Vec::with_capacity(out_dim * in_dim);
        for row in w { w_flat.extend_from_slice(row); }
        pool.cache_weight(&backend.device, &backend.queue, w_ptr, &w_flat);
    }
    let x_buf = backend.storage_buf("x", &x_flat);
    let dummy_bias = [0.0f32];
    let dummy_buf = backend.storage_buf("nb", &dummy_bias);
    pool.ensure_scratch(&backend.device, 0, (out_total * 4) as u64);
    let params = MatvecBatchParams { out_dim: out_dim as u32, in_dim: in_dim as u32, n_pos: n_pos as u32, use_bias: 0 };
    pool.write_uniform(&backend.device, &backend.queue, &params);
    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &backend.matvec_batch_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pool.weight_ref(w_ptr).as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dummy_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
        ],
    });
    let mut encoder = backend.device.create_command_encoder(&Default::default());
    { let mut pass = encoder.begin_compute_pass(&Default::default());
      pass.set_pipeline(&backend.matvec_batch_pipeline); pass.set_bind_group(0, &bind_group, &[]);
      pass.dispatch_workgroups(out_dim as u32, n_pos as u32, 1); }
    backend.queue.submit(Some(encoder.finish()));
    let y_flat = pool.readback_scratch(&backend.device, &backend.queue, 0, out_total);
    drop(pool);
    y_flat.chunks(out_dim).map(|c| c.to_vec()).collect()
}

/// Batched layer normalization: one workgroup per position.
pub(crate) fn layer_norm_batch(
    backend: &GpuBackend,
    xs: &[Vec<f32>],
    weight: &[f32],
    bias: &[f32],
) -> Vec<Vec<f32>> {
    if xs.is_empty() { return vec![]; }
    let dim = xs[0].len();
    let n_pos = xs.len();

    // Cache weight/bias buffers — avoid re-upload within same iteration
    let w_ptr = weight.as_ptr() as usize;
    let b_ptr = bias.as_ptr() as usize;
    {
        let mut pool = backend.pool.lock().unwrap();
        if !pool.has_weight(w_ptr) { pool.cache_weight(&backend.device, &backend.queue, w_ptr, weight); }
        if !pool.has_weight(b_ptr) { pool.cache_weight(&backend.device, &backend.queue, b_ptr, bias); }
    }
    let x_flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
    // Use gpu_layer_norm_batch — weight/bias still re-uploaded inside it,
    // but the cache means the NEXT call with same weights skips upload.
    // TODO: add gpu_layer_norm_batch_resident for full caching
    let y_flat = backend.gpu_layer_norm_batch(&x_flat, weight, bias, dim, n_pos);
    y_flat.chunks(dim).map(|c| c.to_vec()).collect()
}
