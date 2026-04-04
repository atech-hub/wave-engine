//! Full FFN GPU pipeline — entire forward and backward on GPU.
//!
//! All intermediates stay in VRAM. Backward reads the exact bits forward produced.
//! No CPU involvement in the FFN. Self-consistent by construction.
//!
//! Forward: 10 dispatches in one encoder
//!   maestro_in (squeeze → GELU → process) → add → ODE → maestro_out → add → out_proj
//!
//! Backward: 16 dispatches in one encoder (reads forward buffers)
//!   out_proj bwd → maestro_out bwd → ODE identity → maestro_in bwd

use crate::gpu_pipelines::*;
use crate::wave_block::{KerrDualMaestroWeights, LinearWeights};
use wgpu::util::DeviceExt;

/// All GPU buffers for the full FFN pipeline.
pub struct FfnFullBuffers {
    // Forward intermediates (written during forward, read during backward)
    pub input: wgpu::Buffer,         // x (normed)
    pub sq_in: wgpu::Buffer,         // maestro_in squeeze output [T × maestro_dim]
    pub act_in: wgpu::Buffer,        // maestro_in GELU output [T × maestro_dim]
    pub mae_in_out: wgpu::Buffer,    // maestro_in process output [T × n_embd]
    pub precond: wgpu::Buffer,       // x + mae_in_out [T × n_embd]
    pub kerr_out: wgpu::Buffer,      // ODE output [T × n_embd]
    pub sq_out: wgpu::Buffer,        // maestro_out squeeze [T × maestro_dim]
    pub act_out: wgpu::Buffer,       // maestro_out GELU [T × maestro_dim]
    pub mae_out_out: wgpu::Buffer,   // maestro_out process [T × n_embd]
    pub regulated: wgpu::Buffer,     // kerr_out + mae_out_out [T × n_embd]
    pub output: wgpu::Buffer,        // final out_proj result [T × n_embd]

    // Gradient buffers (written during backward)
    pub d_output: wgpu::Buffer,      // d_ffn_out input [T × n_embd]
    pub d_regulated: wgpu::Buffer,   // [T × n_embd]
    pub d_mae_out_out: wgpu::Buffer, // [T × n_embd]
    pub d_act_out: wgpu::Buffer,     // [T × maestro_dim]
    pub d_sq_out: wgpu::Buffer,      // [T × maestro_dim]
    pub d_kerr_out: wgpu::Buffer,    // [T × n_embd]
    pub d_precond: wgpu::Buffer,     // [T × n_embd]
    pub d_mae_in_out: wgpu::Buffer,  // [T × n_embd]
    pub d_act_in: wgpu::Buffer,      // [T × maestro_dim]
    pub d_sq_in: wgpu::Buffer,       // [T × maestro_dim]
    pub d_input: wgpu::Buffer,       // d_normed output [T × n_embd]

    // Weight gradient accumulators
    pub d_out_proj_w: wgpu::Buffer,  // [n_embd × n_embd]
    pub d_out_proj_b: wgpu::Buffer,  // [n_embd]
    pub d_mae_out_pr_w: wgpu::Buffer, // [n_embd × maestro_dim]
    pub d_mae_out_pr_b: wgpu::Buffer, // [n_embd]
    pub d_mae_out_sq_w: wgpu::Buffer, // [maestro_dim × n_embd]
    pub d_mae_out_sq_b: wgpu::Buffer, // [maestro_dim]
    pub d_mae_in_pr_w: wgpu::Buffer,  // [n_embd × maestro_dim]
    pub d_mae_in_pr_b: wgpu::Buffer,  // [n_embd]
    pub d_mae_in_sq_w: wgpu::Buffer,  // [maestro_dim × n_embd]
    pub d_mae_in_sq_b: wgpu::Buffer,  // [maestro_dim]

    // ODE scratch (for fused RK4)
    pub ode_r: wgpu::Buffer,
    pub ode_s: wgpu::Buffer,
    pub ode_r_tmp: wgpu::Buffer,
    pub ode_s_tmp: wgpu::Buffer,
    pub ode_k1r: wgpu::Buffer, pub ode_k1s: wgpu::Buffer,
    pub ode_k2r: wgpu::Buffer, pub ode_k2s: wgpu::Buffer,
    pub ode_k3r: wgpu::Buffer, pub ode_k3s: wgpu::Buffer,
    pub ode_k4r: wgpu::Buffer, pub ode_k4s: wgpu::Buffer,
    pub ode_r_mid: wgpu::Buffer, pub ode_s_mid: wgpu::Buffer,
    pub ode_r_new: wgpu::Buffer, pub ode_s_new: wgpu::Buffer,
}

impl FfnFullBuffers {
    pub fn new(device: &wgpu::Device, max_pos: usize, n_embd: usize, maestro_dim: usize) -> Self {
        let n_bands = n_embd / 2;
        let total_bands = max_pos * n_bands;
        let make = |n: usize| device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let e = max_pos * n_embd;
        let m = max_pos * maestro_dim;
        Self {
            input: make(e), sq_in: make(m), act_in: make(m), mae_in_out: make(e),
            precond: make(e), kerr_out: make(e),
            sq_out: make(m), act_out: make(m), mae_out_out: make(e),
            regulated: make(e), output: make(e),
            d_output: make(e), d_regulated: make(e), d_mae_out_out: make(e),
            d_act_out: make(m), d_sq_out: make(m), d_kerr_out: make(e),
            d_precond: make(e), d_mae_in_out: make(e),
            d_act_in: make(m), d_sq_in: make(m), d_input: make(e),
            d_out_proj_w: make(n_embd * n_embd), d_out_proj_b: make(n_embd),
            d_mae_out_pr_w: make(n_embd * maestro_dim), d_mae_out_pr_b: make(n_embd),
            d_mae_out_sq_w: make(maestro_dim * n_embd), d_mae_out_sq_b: make(maestro_dim),
            d_mae_in_pr_w: make(n_embd * maestro_dim), d_mae_in_pr_b: make(n_embd),
            d_mae_in_sq_w: make(maestro_dim * n_embd), d_mae_in_sq_b: make(maestro_dim),
            ode_r: make(total_bands), ode_s: make(total_bands),
            ode_r_tmp: make(total_bands), ode_s_tmp: make(total_bands),
            ode_k1r: make(total_bands), ode_k1s: make(total_bands),
            ode_k2r: make(total_bands), ode_k2s: make(total_bands),
            ode_k3r: make(total_bands), ode_k3s: make(total_bands),
            ode_k4r: make(total_bands), ode_k4s: make(total_bands),
            ode_r_mid: make(total_bands), ode_s_mid: make(total_bands),
            ode_r_new: make(total_bands), ode_s_new: make(total_bands),
        }
    }
}

/// Run entire FFN forward on GPU. All intermediates stay in VRAM.
/// Returns the output (readback to CPU for residual addition).
pub fn ffn_forward_gpu(
    gpu: &GpuBackend,
    bufs: &FfnFullBuffers,
    weights: &KerrDualMaestroWeights,
    x_flat: &[f32],       // [n_pos × n_embd] normed input
    n_pos: usize,
    n_embd: usize,
    maestro_dim: usize,
) -> Vec<f32> {
    let n_bands = n_embd / 2;
    let total_bands = n_pos * n_bands;

    // Upload input
    gpu.queue.write_buffer(&bufs.input, 0, bytemuck::cast_slice(x_flat));

    // Flatten all weights
    let flatten_w = |w: &[Vec<f32>]| -> Vec<f32> { w.iter().flat_map(|r| r.iter().copied()).collect() };
    let in_sq_w = flatten_w(&weights.maestro_in.squeeze.w);
    let in_pr_w = flatten_w(&weights.maestro_in.process_1.w);
    let out_sq_w = flatten_w(&weights.maestro_out.squeeze.w);
    let out_pr_w = flatten_w(&weights.maestro_out.process_1.w);
    let op_w = flatten_w(&weights.out_proj.as_linear().w);

    // Upload all weight buffers
    let upload = |data: &[f32]| -> wgpu::Buffer {
        gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let w_in_sq = upload(&in_sq_w);
    let b_in_sq = upload(&weights.maestro_in.squeeze.b);
    let w_in_pr = upload(&in_pr_w);
    let b_in_pr = upload(&weights.maestro_in.process_1.b);
    let w_out_sq = upload(&out_sq_w);
    let b_out_sq = upload(&weights.maestro_out.squeeze.b);
    let w_out_pr = upload(&out_pr_w);
    let b_out_pr = upload(&weights.maestro_out.process_1.b);
    let w_op = upload(&op_w);
    let b_op = upload(&weights.out_proj.as_linear().b);

    // ODE weights
    let gamma: Vec<f32> = weights.kerr.gamma_raw.iter().map(|&g| crate::common::math::softplus(g)).collect();
    let w_gamma = upload(&gamma);
    let w_omega = upload(&weights.kerr.omega);
    let ab = [weights.kerr.alpha, weights.kerr.beta];
    let w_ab = upload(&ab);

    // Uniform params
    let make_uniform = |data: &[u8]| -> wgpu::Buffer {
        gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: data, usage: wgpu::BufferUsages::UNIFORM,
        })
    };

    let sq_params = MatvecBatchParams { out_dim: maestro_dim as u32, in_dim: n_embd as u32, n_pos: n_pos as u32, use_bias: 1 };
    let pr_params = MatvecBatchParams { out_dim: n_embd as u32, in_dim: maestro_dim as u32, n_pos: n_pos as u32, use_bias: 1 };
    let op_params = MatvecBatchParams { out_dim: n_embd as u32, in_dim: n_embd as u32, n_pos: n_pos as u32, use_bias: 1 };
    let gelu_params_m = GeluParams { len: (n_pos * maestro_dim) as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
    let va_params = VecAddParams { len: (n_pos * n_embd) as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
    let di_params = DeinterleaveParams { n_bands: n_bands as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0 };
    let kd_params = KerrDerivBatchParams { n_bands: n_bands as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0 };
    let dt = 1.0 / weights.kerr.rk4_n_steps as f32;
    let vsa_half = VecScaleAddParams { len: total_bands as u32, scale: 0.5 * dt, _pad1: 0, _pad2: 0 };
    let vsa_full = VecScaleAddParams { len: total_bands as u32, scale: dt, _pad1: 0, _pad2: 0 };
    let rc_params = Rk4CombineParams { len: total_bands as u32, dt_over_6: dt / 6.0, _pad1: 0, _pad2: 0 };

    let u_sq = make_uniform(bytemuck::bytes_of(&sq_params));
    let u_pr = make_uniform(bytemuck::bytes_of(&pr_params));
    let u_op = make_uniform(bytemuck::bytes_of(&op_params));
    let u_gelu_m = make_uniform(bytemuck::bytes_of(&gelu_params_m));
    let u_va = make_uniform(bytemuck::bytes_of(&va_params));
    let u_di = make_uniform(bytemuck::bytes_of(&di_params));
    let u_kd = make_uniform(bytemuck::bytes_of(&kd_params));
    let u_vsa_half = make_uniform(bytemuck::bytes_of(&vsa_half));
    let u_vsa_full = make_uniform(bytemuck::bytes_of(&vsa_full));
    let u_rc = make_uniform(bytemuck::bytes_of(&rc_params));

    let wg_m = ((n_pos * maestro_dim) as u32 + 63) / 64;
    let wg_e = ((n_pos * n_embd) as u32 + 63) / 64;
    let wg_b = (total_bands as u32 + 63) / 64;
    let buf_size = (total_bands * 4) as u64;

    // ═══ ONE ENCODER FOR ENTIRE FORWARD ═══
    let mut enc = gpu.device.create_command_encoder(&Default::default());

    // 1. maestro_in squeeze: input → sq_in
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.matvec_batch_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: w_in_sq.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.input.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: b_in_sq.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.sq_in.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_sq.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(maestro_dim as u32, n_pos as u32, 1); }

    // 2. GELU: sq_in → act_in
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.gelu_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.sq_in.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.act_in.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: u_gelu_m.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.gelu_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(wg_m, 1, 1); }

    // 3. maestro_in process: act_in → mae_in_out
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.matvec_batch_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: w_in_pr.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.act_in.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: b_in_pr.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.mae_in_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_pr.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(n_embd as u32, n_pos as u32, 1); }

    // 4. vec_add: input + mae_in_out → precond
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.vec_add_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.input.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.mae_in_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bufs.precond.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: u_va.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.vec_add_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(wg_e, 1, 1); }

    // 5. Deinterleave: precond → ode_r, ode_s
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.deinterleave_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.precond.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.ode_r.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bufs.ode_s.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: u_di.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.deinterleave_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(wg_b, 1, 1); }

    // 6. Fused RK4 ODE (16 steps, all in this encoder)
    let deriv_bg = |r_in: &wgpu::Buffer, s_in: &wgpu::Buffer, dr: &wgpu::Buffer, ds: &wgpu::Buffer| {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.kerr_deriv_batch_layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: r_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: s_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dr.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: ds.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: w_gamma.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: w_omega.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: u_kd.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: w_ab.as_entire_binding() },
        ]})
    };
    let vsa_bg = |a: &wgpu::Buffer, b: &wgpu::Buffer, y: &wgpu::Buffer, u: &wgpu::Buffer| {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.vec_scale_add_layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: u.as_entire_binding() },
        ]})
    };

    let bg_k1 = deriv_bg(&bufs.ode_r, &bufs.ode_s, &bufs.ode_k1r, &bufs.ode_k1s);
    let bg_mid_r2 = vsa_bg(&bufs.ode_r, &bufs.ode_k1r, &bufs.ode_r_mid, &u_vsa_half);
    let bg_mid_s2 = vsa_bg(&bufs.ode_s, &bufs.ode_k1s, &bufs.ode_s_mid, &u_vsa_half);
    let bg_k2 = deriv_bg(&bufs.ode_r_mid, &bufs.ode_s_mid, &bufs.ode_k2r, &bufs.ode_k2s);
    let bg_mid_r3 = vsa_bg(&bufs.ode_r, &bufs.ode_k2r, &bufs.ode_r_mid, &u_vsa_half);
    let bg_mid_s3 = vsa_bg(&bufs.ode_s, &bufs.ode_k2s, &bufs.ode_s_mid, &u_vsa_half);
    let bg_k3 = deriv_bg(&bufs.ode_r_mid, &bufs.ode_s_mid, &bufs.ode_k3r, &bufs.ode_k3s);
    let bg_mid_r4 = vsa_bg(&bufs.ode_r, &bufs.ode_k3r, &bufs.ode_r_mid, &u_vsa_full);
    let bg_mid_s4 = vsa_bg(&bufs.ode_s, &bufs.ode_k3s, &bufs.ode_s_mid, &u_vsa_full);
    let bg_k4 = deriv_bg(&bufs.ode_r_mid, &bufs.ode_s_mid, &bufs.ode_k4r, &bufs.ode_k4s);
    let bg_comb_r = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.rk4_combine_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.ode_r.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.ode_k1r.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bufs.ode_k2r.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.ode_k3r.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: bufs.ode_k4r.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: bufs.ode_r_new.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: u_rc.as_entire_binding() },
    ]});
    let bg_comb_s = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.rk4_combine_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.ode_s.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.ode_k1s.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bufs.ode_k2s.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.ode_k3s.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: bufs.ode_k4s.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: bufs.ode_s_new.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: u_rc.as_entire_binding() },
    ]});

    for _ in 0..weights.kerr.rk4_n_steps {
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&gpu.kerr_deriv_batch_pipeline); p.set_bind_group(0, &bg_k1, &[]); p.dispatch_workgroups(wg_b, 1, 1); }
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&gpu.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_r2, &[]); p.dispatch_workgroups(wg_b, 1, 1); }
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&gpu.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_s2, &[]); p.dispatch_workgroups(wg_b, 1, 1); }
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&gpu.kerr_deriv_batch_pipeline); p.set_bind_group(0, &bg_k2, &[]); p.dispatch_workgroups(wg_b, 1, 1); }
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&gpu.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_r3, &[]); p.dispatch_workgroups(wg_b, 1, 1); }
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&gpu.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_s3, &[]); p.dispatch_workgroups(wg_b, 1, 1); }
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&gpu.kerr_deriv_batch_pipeline); p.set_bind_group(0, &bg_k3, &[]); p.dispatch_workgroups(wg_b, 1, 1); }
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&gpu.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_r4, &[]); p.dispatch_workgroups(wg_b, 1, 1); }
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&gpu.vec_scale_add_pipeline); p.set_bind_group(0, &bg_mid_s4, &[]); p.dispatch_workgroups(wg_b, 1, 1); }
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&gpu.kerr_deriv_batch_pipeline); p.set_bind_group(0, &bg_k4, &[]); p.dispatch_workgroups(wg_b, 1, 1); }
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&gpu.rk4_combine_pipeline); p.set_bind_group(0, &bg_comb_r, &[]); p.dispatch_workgroups(wg_b, 1, 1); }
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&gpu.rk4_combine_pipeline); p.set_bind_group(0, &bg_comb_s, &[]); p.dispatch_workgroups(wg_b, 1, 1); }
        enc.copy_buffer_to_buffer(&bufs.ode_r_new, 0, &bufs.ode_r, 0, buf_size);
        enc.copy_buffer_to_buffer(&bufs.ode_s_new, 0, &bufs.ode_s, 0, buf_size);
    }

    // 7. Reinterleave: ode_r, ode_s → kerr_out
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.reinterleave_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.ode_r.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.ode_s.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bufs.kerr_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: u_di.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.reinterleave_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(wg_b, 1, 1); }

    // 8. maestro_out squeeze: kerr_out → sq_out
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.matvec_batch_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: w_out_sq.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.kerr_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: b_out_sq.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.sq_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_sq.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(maestro_dim as u32, n_pos as u32, 1); }

    // 9. GELU: sq_out → act_out
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.gelu_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.sq_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.act_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: u_gelu_m.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.gelu_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(wg_m, 1, 1); }

    // 10. maestro_out process: act_out → mae_out_out
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.matvec_batch_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: w_out_pr.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.act_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: b_out_pr.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.mae_out_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_pr.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(n_embd as u32, n_pos as u32, 1); }

    // 11. vec_add: kerr_out + mae_out_out → regulated
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.vec_add_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.kerr_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.mae_out_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bufs.regulated.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: u_va.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.vec_add_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(wg_e, 1, 1); }

    // 12. out_proj: regulated → output
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.matvec_batch_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: w_op.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.regulated.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: b_op.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.output.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_op.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(n_embd as u32, n_pos as u32, 1); }

    // ONE SUBMIT for entire forward
    gpu.queue.submit(Some(enc.finish()));

    // ONE READBACK — the output for residual addition
    gpu.readback(&bufs.output, n_pos * n_embd)
}

/// Full FFN backward on GPU. Reads forward intermediates from the SAME buffers.
/// Returns (d_input_flat, weight_grads) where weight_grads are all flattened.
pub fn ffn_backward_gpu(
    gpu: &GpuBackend,
    bufs: &FfnFullBuffers,
    weights: &KerrDualMaestroWeights,
    d_ffn_out_flat: &[f32],  // [n_pos × n_embd]
    n_pos: usize,
    n_embd: usize,
    maestro_dim: usize,
) -> FfnGradients {
    // Upload d_ffn_out
    gpu.queue.write_buffer(&bufs.d_output, 0, bytemuck::cast_slice(d_ffn_out_flat));

    // Zero all weight gradient accumulators
    let zero_buf = |buf: &wgpu::Buffer, n: usize| {
        let zeros = vec![0.0f32; n];
        gpu.queue.write_buffer(buf, 0, bytemuck::cast_slice(&zeros));
    };
    zero_buf(&bufs.d_out_proj_w, n_embd * n_embd);
    zero_buf(&bufs.d_out_proj_b, n_embd);
    zero_buf(&bufs.d_mae_out_pr_w, n_embd * maestro_dim);
    zero_buf(&bufs.d_mae_out_pr_b, n_embd);
    zero_buf(&bufs.d_mae_out_sq_w, maestro_dim * n_embd);
    zero_buf(&bufs.d_mae_out_sq_b, maestro_dim);
    zero_buf(&bufs.d_mae_in_pr_w, n_embd * maestro_dim);
    zero_buf(&bufs.d_mae_in_pr_b, n_embd);
    zero_buf(&bufs.d_mae_in_sq_w, maestro_dim * n_embd);
    zero_buf(&bufs.d_mae_in_sq_b, maestro_dim);

    // Flatten weights
    let flatten_w = |w: &[Vec<f32>]| -> Vec<f32> { w.iter().flat_map(|r| r.iter().copied()).collect() };
    let transpose_w = |w: &[Vec<f32>]| -> Vec<f32> {
        let rows = w.len();
        let cols = w[0].len();
        let mut t = vec![0.0f32; rows * cols];
        for i in 0..rows { for j in 0..cols { t[j * rows + i] = w[i][j]; } }
        t
    };

    let upload = |data: &[f32]| -> wgpu::Buffer {
        gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let make_uniform = |data: &[u8]| -> wgpu::Buffer {
        gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: data, usage: wgpu::BufferUsages::UNIFORM,
        })
    };

    // Transposed weights for backward d_x (use SAME shader as forward)
    let wt_op = upload(&transpose_w(&weights.out_proj.as_linear().w));         // n_embd×n_embd
    let wt_out_pr = upload(&transpose_w(&weights.maestro_out.process_1.w)); // maestro×n_embd
    let wt_out_sq = upload(&transpose_w(&weights.maestro_out.squeeze.w));   // n_embd×maestro
    let wt_in_pr = upload(&transpose_w(&weights.maestro_in.process_1.w));   // maestro×n_embd
    let wt_in_sq = upload(&transpose_w(&weights.maestro_in.squeeze.w));     // n_embd×maestro

    // Original weights for outer product d_W
    let w_op = upload(&flatten_w(&weights.out_proj.as_linear().w));
    let w_out_pr = upload(&flatten_w(&weights.maestro_out.process_1.w));
    let w_out_sq = upload(&flatten_w(&weights.maestro_out.squeeze.w));
    let w_in_pr = upload(&flatten_w(&weights.maestro_in.process_1.w));
    let w_in_sq = upload(&flatten_w(&weights.maestro_in.squeeze.w));

    // Uniform params
    let dummy_bias = upload(&[0.0f32]);
    // d_x through n_embd→n_embd (out_proj)
    let u_dx_ee = make_uniform(bytemuck::bytes_of(&MatvecBatchParams { out_dim: n_embd as u32, in_dim: n_embd as u32, n_pos: n_pos as u32, use_bias: 0 }));
    // d_x through n_embd→maestro (process backward: W^T is maestro×n_embd, output is maestro)
    let u_dx_me = make_uniform(bytemuck::bytes_of(&MatvecBatchParams { out_dim: maestro_dim as u32, in_dim: n_embd as u32, n_pos: n_pos as u32, use_bias: 0 }));
    // d_x through maestro→n_embd (squeeze backward: W^T is n_embd×maestro, output is n_embd)
    let u_dx_em = make_uniform(bytemuck::bytes_of(&MatvecBatchParams { out_dim: n_embd as u32, in_dim: maestro_dim as u32, n_pos: n_pos as u32, use_bias: 0 }));
    // outer product params
    let u_op_ee = make_uniform(bytemuck::bytes_of(&OuterProductParams { out_dim: n_embd as u32, in_dim: n_embd as u32, n_pos: n_pos as u32, compute_bias: 1 }));
    let u_op_em = make_uniform(bytemuck::bytes_of(&OuterProductParams { out_dim: n_embd as u32, in_dim: maestro_dim as u32, n_pos: n_pos as u32, compute_bias: 1 }));
    let u_op_me = make_uniform(bytemuck::bytes_of(&OuterProductParams { out_dim: maestro_dim as u32, in_dim: n_embd as u32, n_pos: n_pos as u32, compute_bias: 1 }));
    // GELU backward
    let u_gelu_m = make_uniform(bytemuck::bytes_of(&GeluParams { len: (n_pos * maestro_dim) as u32, _pad1: 0, _pad2: 0, _pad3: 0 }));
    // vec_add for residuals
    let u_va_e = make_uniform(bytemuck::bytes_of(&VecAddParams { len: (n_pos * n_embd) as u32, _pad1: 0, _pad2: 0, _pad3: 0 }));

    let wg_m = ((n_pos * maestro_dim) as u32 + 63) / 64;

    // ═══ ONE ENCODER FOR ENTIRE BACKWARD ═══
    let mut enc = gpu.device.create_command_encoder(&Default::default());

    // 1. out_proj d_x: d_regulated = W_op^T @ d_output
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.matvec_batch_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: wt_op.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.d_output.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: dummy_bias.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.d_regulated.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_dx_ee.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(n_embd as u32, n_pos as u32, 1); }

    // 2. out_proj d_W: d_W = d_output @ regulated^T (READS FORWARD BUFFER!)
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.outer_product_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.d_output.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.regulated.as_entire_binding() }, // ← PING-PONG: forward's buffer!
        wgpu::BindGroupEntry { binding: 2, resource: bufs.d_out_proj_w.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.d_out_proj_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_op_ee.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.outer_product_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(n_embd as u32, 1, 1); }

    // d_regulated splits to d_kerr_out (residual) and d_mae_out
    // Both paths get d_regulated. Copy for both uses.

    // 3. maestro_out process d_x: d_act_out = W_out_pr^T @ d_regulated
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.matvec_batch_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: wt_out_pr.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.d_regulated.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: dummy_bias.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.d_act_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_dx_me.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(maestro_dim as u32, n_pos as u32, 1); }

    // 4. maestro_out process d_W (reads act_out from forward)
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.outer_product_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.d_regulated.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.act_out.as_entire_binding() }, // ← forward buffer!
        wgpu::BindGroupEntry { binding: 2, resource: bufs.d_mae_out_pr_w.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.d_mae_out_pr_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_op_em.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.outer_product_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(n_embd as u32, 1, 1); }

    // 5. GELU backward: d_sq_out = gelu_bwd(d_act_out, sq_out)  (reads sq_out from forward)
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.gelu_bwd_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.d_act_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.sq_out.as_entire_binding() },  // ← forward buffer!
        wgpu::BindGroupEntry { binding: 2, resource: bufs.d_sq_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: u_gelu_m.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.gelu_bwd_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(wg_m, 1, 1); }

    // 6. maestro_out squeeze d_x: d_kerr_from_mae = W_out_sq^T @ d_sq_out
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.matvec_batch_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: wt_out_sq.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.d_sq_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: dummy_bias.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.d_mae_out_out.as_entire_binding() }, // reuse as scratch
        wgpu::BindGroupEntry { binding: 4, resource: u_dx_em.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(n_embd as u32, n_pos as u32, 1); }

    // 7. maestro_out squeeze d_W (reads kerr_out from forward)
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.outer_product_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.d_sq_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.kerr_out.as_entire_binding() }, // ← forward buffer!
        wgpu::BindGroupEntry { binding: 2, resource: bufs.d_mae_out_sq_w.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.d_mae_out_sq_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_op_me.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.outer_product_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(maestro_dim as u32, 1, 1); }

    // 8. d_kerr_out = d_regulated (residual) + d_from_mae_out_squeeze
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.vec_add_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.d_regulated.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.d_mae_out_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bufs.d_kerr_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: u_va_e.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.vec_add_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(((n_pos * n_embd) as u32 + 63) / 64, 1, 1); }

    // 9. ODE backward: identity. d_precond = d_kerr_out (just copy)
    enc.copy_buffer_to_buffer(&bufs.d_kerr_out, 0, &bufs.d_precond, 0, (n_pos * n_embd * 4) as u64);

    // 10. maestro_in process d_x: d_act_in = W_in_pr^T @ d_precond
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.matvec_batch_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: wt_in_pr.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.d_precond.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: dummy_bias.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.d_act_in.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_dx_me.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(maestro_dim as u32, n_pos as u32, 1); }

    // 11. maestro_in process d_W (reads act_in from forward)
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.outer_product_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.d_precond.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.act_in.as_entire_binding() },  // ← forward buffer!
        wgpu::BindGroupEntry { binding: 2, resource: bufs.d_mae_in_pr_w.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.d_mae_in_pr_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_op_em.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.outer_product_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(n_embd as u32, 1, 1); }

    // 12. GELU backward: d_sq_in = gelu_bwd(d_act_in, sq_in)
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.gelu_bwd_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.d_act_in.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.sq_in.as_entire_binding() },  // ← forward buffer!
        wgpu::BindGroupEntry { binding: 2, resource: bufs.d_sq_in.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: u_gelu_m.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.gelu_bwd_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(wg_m, 1, 1); }

    // 13. maestro_in squeeze d_x: d_from_mae_in = W_in_sq^T @ d_sq_in
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.matvec_batch_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: wt_in_sq.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.d_sq_in.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: dummy_bias.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.d_mae_in_out.as_entire_binding() }, // reuse as scratch
        wgpu::BindGroupEntry { binding: 4, resource: u_dx_em.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.matvec_batch_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(n_embd as u32, n_pos as u32, 1); }

    // 14. maestro_in squeeze d_W (reads input from forward)
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.outer_product_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.d_sq_in.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.input.as_entire_binding() },  // ← forward buffer!
        wgpu::BindGroupEntry { binding: 2, resource: bufs.d_mae_in_sq_w.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bufs.d_mae_in_sq_b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: u_op_me.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.outer_product_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(maestro_dim as u32, 1, 1); }

    // 15. d_input = d_precond (residual) + d_from_mae_in_squeeze
    { let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &gpu.vec_add_layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: bufs.d_precond.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: bufs.d_mae_in_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bufs.d_input.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: u_va_e.as_entire_binding() },
    ]});
    let mut p = enc.begin_compute_pass(&Default::default());
    p.set_pipeline(&gpu.vec_add_pipeline); p.set_bind_group(0, &bg, &[]);
    p.dispatch_workgroups(((n_pos * n_embd) as u32 + 63) / 64, 1, 1); }

    // ONE SUBMIT for entire backward
    gpu.queue.submit(Some(enc.finish()));

    // READBACK all results
    let d_input = gpu.readback(&bufs.d_input, n_pos * n_embd);
    let d_out_proj_w = gpu.readback(&bufs.d_out_proj_w, n_embd * n_embd);
    let d_out_proj_b = gpu.readback(&bufs.d_out_proj_b, n_embd);
    let d_mae_out_pr_w = gpu.readback(&bufs.d_mae_out_pr_w, n_embd * maestro_dim);
    let d_mae_out_pr_b = gpu.readback(&bufs.d_mae_out_pr_b, n_embd);
    let d_mae_out_sq_w = gpu.readback(&bufs.d_mae_out_sq_w, maestro_dim * n_embd);
    let d_mae_out_sq_b = gpu.readback(&bufs.d_mae_out_sq_b, maestro_dim);
    let d_mae_in_pr_w = gpu.readback(&bufs.d_mae_in_pr_w, n_embd * maestro_dim);
    let d_mae_in_pr_b = gpu.readback(&bufs.d_mae_in_pr_b, n_embd);
    let d_mae_in_sq_w = gpu.readback(&bufs.d_mae_in_sq_w, maestro_dim * n_embd);
    let d_mae_in_sq_b = gpu.readback(&bufs.d_mae_in_sq_b, maestro_dim);

    FfnGradients {
        d_input,
        d_out_proj_w, d_out_proj_b,
        d_mae_out_pr_w, d_mae_out_pr_b,
        d_mae_out_sq_w, d_mae_out_sq_b,
        d_mae_in_pr_w, d_mae_in_pr_b,
        d_mae_in_sq_w, d_mae_in_sq_b,
    }
}

/// All gradients from the FFN backward pass.
pub struct FfnGradients {
    pub d_input: Vec<f32>,          // [n_pos × n_embd]
    pub d_out_proj_w: Vec<f32>,     // [n_embd × n_embd]
    pub d_out_proj_b: Vec<f32>,     // [n_embd]
    pub d_mae_out_pr_w: Vec<f32>,   // [n_embd × maestro_dim]
    pub d_mae_out_pr_b: Vec<f32>,   // [n_embd]
    pub d_mae_out_sq_w: Vec<f32>,   // [maestro_dim × n_embd]
    pub d_mae_out_sq_b: Vec<f32>,   // [maestro_dim]
    pub d_mae_in_pr_w: Vec<f32>,    // [n_embd × maestro_dim]
    pub d_mae_in_pr_b: Vec<f32>,    // [n_embd]
    pub d_mae_in_sq_w: Vec<f32>,    // [maestro_dim × n_embd]
    pub d_mae_in_sq_b: Vec<f32>,    // [maestro_dim]
}
