//! Backward dispatch helpers for GpuBackend.
//!
//! Contains the implementation bodies for all backward pass operations.
//! Called from the ComputeBackend trait impl in dispatch.rs.

use crate::gpu_pipelines::*;
use crate::model::*;
use wgpu::util::DeviceExt;

/// Kerr-ODE backward pass (batched).
pub(crate) fn kerr_ode_backward_batch(
    backend: &GpuBackend,
    d_outputs: &[Vec<f32>],
    inputs: &[Vec<f32>],
    weights: &KerrWeights,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, f32, f32) {
    backend.gpu_kerr_ode_backward_batch(d_outputs, inputs, weights)
}

/// Single-position linear backward: d_x = W^T @ d_y.
pub(crate) fn linear_backward_dx(
    backend: &GpuBackend,
    d_y: &[f32],
    w: &[Vec<f32>],
) -> Vec<f32> {
    let out_dim = w.len();
    let in_dim = if out_dim > 0 { w[0].len() } else { return vec![] };
    let mut w_flat = Vec::with_capacity(out_dim * in_dim);
    for row in w { w_flat.extend_from_slice(row); }

    let w_buf = backend.storage_buf("bwd_w", &w_flat);
    let dy_buf = backend.storage_buf("bwd_dy", d_y);
    let dx_buf = backend.output_buf("bwd_dx", in_dim);
    let params = MatvecBwdParams {
        out_dim: out_dim as u32, in_dim: in_dim as u32, _pad1: 0, _pad2: 0,
    };
    let params_buf = backend.uniform_buf("bwd_params", &params);

    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matvec_bwd_bg"),
        layout: &backend.matvec_bwd_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dy_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dx_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut encoder = backend.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&backend.matvec_bwd_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(in_dim as u32, 1, 1); // tiled: one workgroup per input element
    }
    backend.queue.submit(Some(encoder.finish()));

    backend.readback(&dx_buf, in_dim)
}

/// Layer norm backward: returns (d_x, d_weight, d_bias).
pub(crate) fn layer_norm_backward(
    backend: &GpuBackend,
    d_y: &[f32],
    x: &[f32],
    weight: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = x.len();
    let dy_buf = backend.storage_buf("lnb_dy", d_y);
    let x_buf = backend.storage_buf("lnb_x", x);
    let w_buf = backend.storage_buf("lnb_w", weight);
    let out_buf = backend.output_buf("lnb_out", dim * 3); // d_x, d_weight, d_bias concatenated
    let params = LayerNormParams { dim: dim as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
    let params_buf = backend.uniform_buf("lnb_params", &params);

    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("layer_norm_bwd_bg"),
        layout: &backend.layer_norm_bwd_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dy_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut encoder = backend.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&backend.layer_norm_bwd_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1); // single workgroup handles all
    }
    backend.queue.submit(Some(encoder.finish()));

    let result = backend.readback(&out_buf, dim * 3);
    let d_x = result[..dim].to_vec();
    let d_weight = result[dim..dim * 2].to_vec();
    let d_bias = result[dim * 2..].to_vec();
    (d_x, d_weight, d_bias)
}

/// GELU backward: d_x = d_y * gelu'(x).
pub(crate) fn gelu_backward(
    backend: &GpuBackend,
    d_y: &[f32],
    x: &[f32],
) -> Vec<f32> {
    let n = x.len();
    let dy_buf = backend.storage_buf("gb_dy", d_y);
    let x_buf = backend.storage_buf("gb_x", x);
    let dx_buf = backend.output_buf("gb_dx", n);
    let params = GeluBwdParams { len: n as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
    let params_buf = backend.uniform_buf("gb_params", &params);

    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gelu_bwd_bg"),
        layout: &backend.gelu_bwd_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dy_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dx_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut encoder = backend.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&backend.gelu_bwd_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (n as u32 + 63) / 64;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    backend.queue.submit(Some(encoder.finish()));

    backend.readback(&dx_buf, n)
}

/// Batched GELU backward.
pub(crate) fn gelu_backward_batch(
    backend: &GpuBackend,
    d_ys: &[Vec<f32>],
    xs: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    if d_ys.is_empty() { return vec![]; }
    let dim = d_ys[0].len();
    let n_pos = d_ys.len();
    let total = n_pos * dim;

    let dy_flat: Vec<f32> = d_ys.iter().flat_map(|v| v.iter().copied()).collect();
    let x_flat: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
    let dy_buf = backend.storage_buf("gbb_dy", &dy_flat);
    let x_buf = backend.storage_buf("gbb_x", &x_flat);
    let dx_buf = backend.output_buf("gbb_dx", total);
    let params = GeluBwdParams { len: total as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
    let params_buf = backend.uniform_buf("gbb_params", &params);

    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &backend.gelu_bwd_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dy_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dx_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
    });
    let mut encoder = backend.device.create_command_encoder(&Default::default());
    { let mut pass = encoder.begin_compute_pass(&Default::default());
      pass.set_pipeline(&backend.gelu_bwd_pipeline); pass.set_bind_group(0, &bind_group, &[]);
      pass.dispatch_workgroups((total as u32 + 63) / 64, 1, 1); }
    backend.queue.submit(Some(encoder.finish()));
    let result = backend.readback(&dx_buf, total);
    result.chunks(dim).map(|c| c.to_vec()).collect()
}

/// Batched layer norm backward. Dispatches per-position on GPU, accumulates d_weight/d_bias.
pub(crate) fn layer_norm_backward_batch(
    backend: &GpuBackend,
    d_ys: &[Vec<f32>],
    xs: &[Vec<f32>],
    weight: &[f32],
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    if d_ys.is_empty() {
        let dim = weight.len();
        return (vec![], vec![0.0; dim], vec![0.0; dim]);
    }
    let dim = weight.len();
    let n_pos = d_ys.len();
    let mut total_dw = vec![0.0f32; dim];
    let mut total_db = vec![0.0f32; dim];
    let mut d_xs = Vec::with_capacity(n_pos);
    // For now, dispatch per-position on GPU (each is fast, ~1us at 896-dim)
    // TODO: write a batched LN backward shader for single-dispatch
    for pos in 0..n_pos {
        let (dx, dw, db) = layer_norm_backward(backend, &d_ys[pos], &xs[pos], weight);
        for i in 0..dim { total_dw[i] += dw[i]; total_db[i] += db[i]; }
        d_xs.push(dx);
    }
    (d_xs, total_dw, total_db)
}

/// Batched linear backward: d_x[pos] = W^T @ d_y[pos] using forward shader with transposed W.
pub(crate) fn linear_backward_dx_batch(
    backend: &GpuBackend,
    d_y: &[Vec<f32>],
    w: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let n_pos = d_y.len();
    let out_dim = w.len();
    let in_dim = if out_dim > 0 { w[0].len() } else { return vec![vec![]; n_pos] };

    // d_x = W^T @ d_y. Transpose W on CPU, then use the SAME forward shader.
    // This ensures forward and backward use identical accumulation patterns.
    // PyTorch does the same thing via cuBLAS — one kernel, transposed input.
    let mut wt_flat = vec![0.0f32; in_dim * out_dim];
    for i in 0..out_dim {
        for j in 0..in_dim {
            wt_flat[j * out_dim + i] = w[i][j];
        }
    }

    let dy_flat: Vec<f32> = d_y.iter().flat_map(|v| v.iter().copied()).collect();
    let wt_buf = backend.storage_buf("mvbb_wt", &wt_flat);
    let dy_buf = backend.storage_buf("mvbb_dy", &dy_flat);
    let dummy_bias = [0.0f32];
    let db_buf = backend.storage_buf("mvbb_db", &dummy_bias);

    let out_total = n_pos * in_dim;
    {
        let mut pool = backend.pool.lock().unwrap();
        pool.ensure_scratch(&backend.device, 0, (out_total * 4) as u64);
        let params = MatvecBatchParams {
            out_dim: in_dim as u32,  // transposed: output is in_dim
            in_dim: out_dim as u32,  // transposed: input is out_dim
            n_pos: n_pos as u32,
            use_bias: 0,
        };
        pool.write_uniform(&backend.device, &backend.queue, &params);
    }
    let mut pool = backend.pool.lock().unwrap();
    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &backend.matvec_batch_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wt_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dy_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: db_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pool.scratch_ref(0).as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pool.uniform_ref().as_entire_binding() },
        ],
    });
    let mut encoder = backend.device.create_command_encoder(&Default::default());
    { let mut pass = encoder.begin_compute_pass(&Default::default());
      pass.set_pipeline(&backend.matvec_batch_pipeline); // SAME shader as forward!
      pass.set_bind_group(0, &bind_group, &[]);
      pass.dispatch_workgroups(in_dim as u32, n_pos as u32, 1); }
    backend.queue.submit(Some(encoder.finish()));

    let dx_flat = pool.readback_scratch(&backend.device, &backend.queue, 0, out_total);
    drop(pool);
    dx_flat.chunks(in_dim).map(|c| c.to_vec()).collect()
}

/// Batched outer product accumulation: d_w = D_Y^T @ X, d_b = sum(d_y).
pub(crate) fn outer_product_accum(
    backend: &GpuBackend,
    d_y: &[Vec<f32>],
    x: &[Vec<f32>],
    compute_bias: bool,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let n_pos = d_y.len();
    let out_dim = d_y[0].len();
    let in_dim = x[0].len();

    // Flatten inputs: d_y[pos][i] -> d_y_flat[pos * out_dim + i]
    let d_y_flat: Vec<f32> = d_y.iter().flat_map(|v| v.iter().copied()).collect();
    let x_flat: Vec<f32> = x.iter().flat_map(|v| v.iter().copied()).collect();

    let dy_buf = backend.storage_buf("op_dy", &d_y_flat);
    let x_buf = backend.storage_buf("op_x", &x_flat);
    let dw_buf = backend.output_buf("op_dw", out_dim * in_dim);
    // d_b buffer — always create it (even if not used) for binding
    let db_buf = backend.output_buf("op_db", out_dim);
    let params = OuterProductParams {
        out_dim: out_dim as u32,
        in_dim: in_dim as u32,
        n_pos: n_pos as u32,
        compute_bias: if compute_bias { 1 } else { 0 },
    };
    let params_buf = backend.uniform_buf("op_params", &params);

    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("outer_product_bg"),
        layout: &backend.outer_product_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dy_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dw_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: db_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut encoder = backend.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&backend.outer_product_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per output row
        pass.dispatch_workgroups(out_dim as u32, 1, 1);
    }
    backend.queue.submit(Some(encoder.finish()));

    // Readback and unflatten
    let dw_flat = backend.readback(&dw_buf, out_dim * in_dim);
    let d_w: Vec<Vec<f32>> = dw_flat.chunks(in_dim).map(|c| c.to_vec()).collect();
    let d_b = if compute_bias {
        backend.readback(&db_buf, out_dim)
    } else {
        vec![0.0f32; out_dim]
    };

    (d_w, d_b)
}

/// Attention backward: two-dispatch GPU implementation.
pub(crate) fn attention_backward(
    backend: &GpuBackend,
    d_pre_proj: &[Vec<f32>],
    q_all: &[Vec<f32>],
    k_all: &[Vec<f32>],
    v_all: &[Vec<f32>],
    att_weights: &[Vec<Vec<f32>>],
    n_head: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let t = d_pre_proj.len();
    let n_embd = d_pre_proj[0].len();
    let head_dim = n_embd / n_head;

    // Flatten inputs to contiguous f32 arrays
    let d_out_flat: Vec<f32> = d_pre_proj.iter().flat_map(|v| v.iter().copied()).collect();
    let q_flat: Vec<f32> = q_all.iter().flat_map(|v| v.iter().copied()).collect();
    let k_flat: Vec<f32> = k_all.iter().flat_map(|v| v.iter().copied()).collect();
    let v_flat: Vec<f32> = v_all.iter().flat_map(|v| v.iter().copied()).collect();

    // att_weights: [n_head][T][T] -> flatten
    let mut att_flat = vec![0.0f32; n_head * t * t];
    for (head, head_weights) in att_weights.iter().enumerate() {
        for (pos, pos_weights) in head_weights.iter().enumerate() {
            for (ki, &w) in pos_weights.iter().enumerate() {
                att_flat[head * t * t + pos * t + ki] = w;
            }
        }
    }

    // Create GPU buffers
    let d_out_buf = backend.storage_buf("ab_d_out", &d_out_flat);
    let q_buf = backend.storage_buf("ab_q", &q_flat);
    let k_buf = backend.storage_buf("ab_k", &k_flat);
    let v_buf = backend.storage_buf("ab_v", &v_flat);
    let att_buf = backend.storage_buf("ab_att", &att_flat);
    let dq_buf = backend.output_buf("ab_dq", t * n_embd);
    let dk_buf = backend.output_buf("ab_dk", t * n_embd);
    let dv_buf = backend.output_buf("ab_dv", t * n_embd);
    let dscore_buf = backend.output_buf("ab_dscore", t * n_head * t);

    let params = AttnBwdParams {
        seq_len: t as u32,
        n_head: n_head as u32,
        head_dim: head_dim as u32,
        n_embd: n_embd as u32,
    };
    let params_buf = backend.uniform_buf("ab_params", &params);
    // Dispatch 2 needs its own uniform buffer (same data, different bind group)
    let params_buf2 = backend.uniform_buf("ab_params2", &params);

    // --- Dispatch 1: attn_backward_scores ---
    let bg1 = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn_bwd_scores_bg"),
        layout: &backend.attn_bwd_scores_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: d_out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: att_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: dq_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: dscore_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: params_buf.as_entire_binding() },
        ],
    });

    // --- Dispatch 2: attn_backward_dkv ---
    let bg2 = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn_bwd_dkv_bg"),
        layout: &backend.attn_bwd_dkv_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: d_out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: att_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dscore_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: dk_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: dv_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: params_buf2.as_entire_binding() },
        ],
    });

    let mut encoder = backend.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&backend.attn_bwd_scores_pipeline);
        pass.set_bind_group(0, &bg1, &[]);
        pass.dispatch_workgroups(t as u32, n_head as u32, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&backend.attn_bwd_dkv_pipeline);
        pass.set_bind_group(0, &bg2, &[]);
        let total_threads = (t * n_embd) as u32;
        pass.dispatch_workgroups((total_threads + 63) / 64, 1, 1);
    }
    backend.queue.submit(Some(encoder.finish()));

    // Readback d_q, d_k, d_v and unflatten
    let dq_flat = backend.readback(&dq_buf, t * n_embd);
    let dk_flat = backend.readback(&dk_buf, t * n_embd);
    let dv_flat = backend.readback(&dv_buf, t * n_embd);

    let unflatten = |flat: Vec<f32>| -> Vec<Vec<f32>> {
        flat.chunks(n_embd).map(|c| c.to_vec()).collect()
    };

    (unflatten(dq_flat), unflatten(dk_flat), unflatten(dv_flat))
}
