//! GPU backward dispatch operations.
//!
//! Kerr derivative backward, Kerr-ODE backward, and RK4 step backward.
//! All batched across positions. Uses legacy buffer helpers (non-pooled)
//! since backward methods have many output buffers.

use crate::gpu_pipelines::*;
use crate::model::*;
use wgpu::util::DeviceExt;

impl GpuBackend {
    pub(crate) fn gpu_kerr_derivative_backward_batch(
        &self,
        d_dr_flat: &[f32], d_ds_flat: &[f32],  // upstream gradients [n_pos * n_bands]
        r_flat: &[f32], s_flat: &[f32],          // cached forward state [n_pos * n_bands]
        gamma: &[f32], omega: &[f32],            // shared params [n_bands]
        alpha: f32, beta: f32, chi: f32,
        n_bands: usize, n_pos: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32, f32, f32) {
        let total = n_pos * n_bands;

        let r_buf = self.storage_buf("kbb_r", r_flat);
        let s_buf = self.storage_buf("kbb_s", s_flat);
        let gamma_buf = self.storage_buf("kbb_gamma", gamma);
        let omega_buf = self.storage_buf("kbb_omega", omega);
        let ddr_buf = self.storage_buf("kbb_ddr", d_dr_flat);
        let dds_buf = self.storage_buf("kbb_dds", d_ds_flat);
        let dr_buf = self.output_buf("kbb_dr", total);
        let ds_buf = self.output_buf("kbb_ds", total);
        let dg_buf = self.output_buf("kbb_dg", total);
        let dom_buf = self.output_buf("kbb_dom", total);
        let dab_buf = self.output_buf("kbb_dab", total * 2); // packed alpha+beta
        let dchi_buf = self.output_buf("kbb_dchi", total);

        let params = KerrBwdBatchParams {
            n_bands: n_bands as u32, n_pos: n_pos as u32, alpha, beta,
            chi, _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
        };
        let params_buf = self.uniform_buf("kbb_params", &params);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kerr_bwd_batch_bg"),
            layout: &self.kerr_bwd_batch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gamma_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: omega_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ddr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: ds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: dg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: dom_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: dab_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: dchi_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kerr_bwd_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((total as u32 + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        let d_r = self.readback(&dr_buf, total);
        let d_s = self.readback(&ds_buf, total);
        let d_gamma = self.readback(&dg_buf, total);
        let d_omega = self.readback(&dom_buf, total);
        let dab_partials = self.readback(&dab_buf, total * 2);
        let da_partials = &dab_partials[..total];
        let db_partials = &dab_partials[total..];
        let dchi_partials = self.readback(&dchi_buf, total);

        // CPU reduction for d_alpha, d_beta, and d_chi
        let d_alpha: f32 = da_partials.iter().sum();
        let d_beta: f32 = db_partials.iter().sum();
        let d_chi: f32 = dchi_partials.iter().sum();

        (d_r, d_s, d_gamma, d_omega, d_alpha, d_beta, d_chi)
    }

    /// Fused Kerr-ODE backward: full RK4 forward+backward in ONE command encoder.
    /// Eliminates 128 GPU round-trips. All dispatches chained, one submit, one readback.
    /// Returns (d_inputs, d_gamma_raw, d_omega, d_alpha, d_beta).
    pub(crate) fn gpu_kerr_ode_backward_batch(
        &self,
        d_outputs: &[Vec<f32>],
        inputs: &[Vec<f32>],
        weights: &KerrWeights,
    ) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, f32, f32, f32) {
        let n_pos = d_outputs.len();
        let n_bands = weights.gamma_raw.len();
        let n_embd = n_bands * 2;
        let n_steps = weights.rk4_n_steps;
        let dt = 1.0 / n_steps as f32;
        let total = n_pos * n_bands;
        let buf_size = (total * 4) as u64;

        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| crate::common::math::softplus(g)).collect();

        // Deinterleave inputs and d_outputs
        let mut r_flat = vec![0.0f32; total];
        let mut s_flat = vec![0.0f32; total];
        let mut dr_flat = vec![0.0f32; total];
        let mut ds_flat = vec![0.0f32; total];
        for pos in 0..n_pos {
            for k in 0..n_bands {
                r_flat[pos * n_bands + k] = inputs[pos][k * 2];
                s_flat[pos * n_bands + k] = inputs[pos][k * 2 + 1];
                dr_flat[pos * n_bands + k] = d_outputs[pos][k * 2];
                ds_flat[pos * n_bands + k] = d_outputs[pos][k * 2 + 1];
            }
        }

        let ab = [weights.alpha, weights.beta];
        let wg = (total as u32 + 63) / 64;

        // ─── Pre-allocate ALL buffers ───
        let make_rw = |_label: &str| self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let make_init = |_label: &str, data: &[f32]| self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // Forward state + scratch
        let r_buf = make_init("bwd_r", &r_flat);
        let s_buf = make_init("bwd_s", &s_flat);
        let k1r = make_rw("bk1r"); let k1s = make_rw("bk1s");
        let k2r = make_rw("bk2r"); let k2s = make_rw("bk2s");
        let k3r = make_rw("bk3r"); let k3s = make_rw("bk3s");
        let k4r = make_rw("bk4r"); let k4s = make_rw("bk4s");
        let r_mid = make_rw("br_mid"); let s_mid = make_rw("bs_mid");
        let r_new = make_rw("br_new"); let s_new = make_rw("bs_new");

        // Per-step state cache
        let r_cache: Vec<wgpu::Buffer> = (0..n_steps).map(|i| {
            let label = format!("rc{i}");
            make_rw(&label)
        }).collect();
        let s_cache: Vec<wgpu::Buffer> = (0..n_steps).map(|i| {
            let label = format!("sc{i}");
            make_rw(&label)
        }).collect();

        // Backward gradient buffers
        let d_r = make_init("d_r", &dr_flat);
        let d_s = make_init("d_s", &ds_flat);
        let d_r_step = make_rw("d_r_step"); let d_s_step = make_rw("d_s_step");
        let d_k_r = make_rw("d_k_r"); let d_k_s = make_rw("d_k_s");
        let d_eval_r = make_rw("d_eval_r"); let d_eval_s = make_rw("d_eval_s");
        let d_extra_r = make_rw("d_extra_r"); let d_extra_s = make_rw("d_extra_s");

        // Parameter gradient outputs (per dispatch) and accumulators
        let dg_step = make_rw("dg_step"); let dom_step = make_rw("dom_step");
        let dab_step = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dab_step"), size: (total * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dchi_step = make_rw("dchi_step");
        let dg_acc = make_init("dg_acc", &vec![0.0f32; total]);
        let dab_acc = make_init("dab_acc", &vec![0.0f32; total * 2]);
        let dchi_acc = make_init("dchi_acc", &vec![0.0f32; total]);
        let zero_buf = make_init("zero", &vec![0.0f32; total]);

        // Shared weight buffers
        let gamma_buf = self.storage_buf("bg", &gamma);
        let omega_buf = self.storage_buf("bo", &weights.omega);
        let ab_buf = self.storage_buf("bab", &ab);

        // ─── Uniform buffers for various scale values ───
        let deriv_params = KerrDerivBatchParams { n_bands: n_bands as u32, n_pos: n_pos as u32, _pad1: 0, _pad2: 0 };
        let deriv_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bwd_deriv_u"), contents: bytemuck::bytes_of(&deriv_params), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bwd_params = KerrBwdBatchParams { n_bands: n_bands as u32, n_pos: n_pos as u32, alpha: weights.alpha, beta: weights.beta, chi: weights.chi, _pad0: 0.0, _pad1: 0.0, _pad2: 0.0 };
        let bwd_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bwd_bwd_u"), contents: bytemuck::bytes_of(&bwd_params), usage: wgpu::BufferUsages::UNIFORM,
        });
        let acc_params = VecAccumulateParams { len: total as u32, _p1: 0, _p2: 0, _p3: 0 };
        let acc_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bwd_acc_u"), contents: bytemuck::bytes_of(&acc_params), usage: wgpu::BufferUsages::UNIFORM,
        });

        // Scale uniforms: dt/6, dt/3, 0.5*dt, dt, 1.0
        let make_vsa_u = |label, scale: f32| {
            let p = VecScaleAddParams { len: total as u32, scale, _pad1: 0, _pad2: 0 };
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label), contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let u_half_dt = make_vsa_u("u_hdt", 0.5 * dt);
        let u_full_dt = make_vsa_u("u_fdt", dt);
        let u_dt6 = make_vsa_u("u_dt6", dt / 6.0);
        let u_dt3 = make_vsa_u("u_dt3", dt / 3.0);
        let rc_params = Rk4CombineParams { len: total as u32, dt_over_6: dt / 6.0, _pad1: 0, _pad2: 0 };
        let rc_u = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bwd_rc_u"), contents: bytemuck::bytes_of(&rc_params), usage: wgpu::BufferUsages::UNIFORM,
        });

        // ─── Bind group helpers ───
        let deriv_bg = |r_in: &wgpu::Buffer, s_in: &wgpu::Buffer, dr: &wgpu::Buffer, ds: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.kerr_deriv_batch_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: r_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: s_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: dr.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ds.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: gamma_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: omega_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: deriv_u.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: ab_buf.as_entire_binding() },
                ],
            })
        };
        let vsa_bg = |a: &wgpu::Buffer, b: &wgpu::Buffer, y: &wgpu::Buffer, u: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.vec_scale_add_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: u.as_entire_binding() },
                ],
            })
        };
        let acc_bg = |a: &wgpu::Buffer, b: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.vec_accumulate_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: acc_u.as_entire_binding() },
                ],
            })
        };
        let bwd_bg = |r_in: &wgpu::Buffer, s_in: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.kerr_bwd_batch_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: r_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: s_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: gamma_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: omega_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: d_k_r.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: d_k_s.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: d_eval_r.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: d_eval_s.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: dg_step.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: dom_step.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: dab_step.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: dchi_step.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 12, resource: bwd_u.as_entire_binding() },
                ],
            })
        };

        // ─── Pre-create all bind groups ───
        // Forward derivative bind groups (reused for cache + recompute)
        let bg_fwd_k1 = deriv_bg(&r_buf, &s_buf, &k1r, &k1s);
        let bg_fwd_k2 = deriv_bg(&r_mid, &s_mid, &k2r, &k2s);
        let bg_fwd_k3 = deriv_bg(&r_mid, &s_mid, &k3r, &k3s);
        let bg_fwd_k4 = deriv_bg(&r_mid, &s_mid, &k4r, &k4s);
        let bg_mid_r2 = vsa_bg(&r_buf, &k1r, &r_mid, &u_half_dt);
        let bg_mid_s2 = vsa_bg(&s_buf, &k1s, &s_mid, &u_half_dt);
        let bg_mid_r3 = vsa_bg(&r_buf, &k2r, &r_mid, &u_half_dt);
        let bg_mid_s3 = vsa_bg(&s_buf, &k2s, &s_mid, &u_half_dt);
        let bg_mid_r4 = vsa_bg(&r_buf, &k3r, &r_mid, &u_full_dt);
        let bg_mid_s4 = vsa_bg(&s_buf, &k3s, &s_mid, &u_full_dt);
        let bg_combine_r = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.rk4_combine_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k1r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k2r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: k3r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: k4r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: r_new.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: rc_u.as_entire_binding() },
            ],
        });
        let bg_combine_s = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.rk4_combine_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k1s.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k2s.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: k3s.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: k4s.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: s_new.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: rc_u.as_entire_binding() },
            ],
        });

        // Backward-specific bind groups
        let bg_bwd_k4 = bwd_bg(&r_mid, &s_mid);  // k4 eval point = r+dt*k3 (in r_mid after reconstruction)
        let bg_bwd_k3 = bwd_bg(&r_mid, &s_mid);  // k3 eval point = r+0.5dt*k2 (same buffers, different data)
        let bg_bwd_k2 = bwd_bg(&r_mid, &s_mid);  // k2 eval point = r+0.5dt*k1
        let bg_bwd_k1 = bwd_bg(&r_buf, &s_buf);  // k1 eval point = cached r, s

        // Scale bind groups: d_k = zero + scale * d_step
        let bg_scale_dt6_r = vsa_bg(&zero_buf, &d_r_step, &d_k_r, &u_dt6);
        let bg_scale_dt6_s = vsa_bg(&zero_buf, &d_s_step, &d_k_s, &u_dt6);
        let bg_scale_dt3_r = vsa_bg(&d_extra_r, &d_r_step, &d_k_r, &u_dt3);
        let bg_scale_dt3_s = vsa_bg(&d_extra_s, &d_s_step, &d_k_s, &u_dt3);
        let bg_scale_k1_r = vsa_bg(&d_extra_r, &d_r_step, &d_k_r, &u_dt6);
        let bg_scale_k1_s = vsa_bg(&d_extra_s, &d_s_step, &d_k_s, &u_dt6);

        // Chain extra: d_extra = zero + scale * d_eval
        let bg_extra_dt_r = vsa_bg(&zero_buf, &d_eval_r, &d_extra_r, &u_full_dt);
        let bg_extra_dt_s = vsa_bg(&zero_buf, &d_eval_s, &d_extra_s, &u_full_dt);
        let bg_extra_hdt_r = vsa_bg(&zero_buf, &d_eval_r, &d_extra_r, &u_half_dt);
        let bg_extra_hdt_s = vsa_bg(&zero_buf, &d_eval_s, &d_extra_s, &u_half_dt);

        // Accumulate bind groups
        let bg_acc_dr = acc_bg(&d_r, &d_eval_r);
        let bg_acc_ds = acc_bg(&d_s, &d_eval_s);
        let bg_acc_dg = acc_bg(&dg_acc, &dg_step);
        let bg_acc_dab = acc_bg(&dab_acc, &dab_step);
        let bg_acc_dchi = acc_bg(&dchi_acc, &dchi_step);

        // Midpoint reconstruction for backward (reuses r_mid/s_mid)
        let bg_bwd_mid_k3_r = vsa_bg(&r_buf, &k2r, &r_mid, &u_half_dt);
        let bg_bwd_mid_k3_s = vsa_bg(&s_buf, &k2s, &s_mid, &u_half_dt);
        let bg_bwd_mid_k2_r = vsa_bg(&r_buf, &k1r, &r_mid, &u_half_dt);
        let bg_bwd_mid_k2_s = vsa_bg(&s_buf, &k1s, &s_mid, &u_half_dt);

        let wg2 = (total as u32 * 2 + 63) / 64; // for packed alpha+beta buffers

        // ─── Dispatch macros ───
        macro_rules! dispatch {
            ($enc:expr, $pipeline:expr, $bg:expr) => {{
                let mut p = $enc.begin_compute_pass(&Default::default());
                p.set_pipeline($pipeline);
                p.set_bind_group(0, $bg, &[]);
                p.dispatch_workgroups(wg, 1, 1);
            }};
        }
        macro_rules! dispatch2x {
            ($enc:expr, $pipeline:expr, $bg:expr) => {{
                let mut p = $enc.begin_compute_pass(&Default::default());
                p.set_pipeline($pipeline);
                p.set_bind_group(0, $bg, &[]);
                p.dispatch_workgroups(wg2, 1, 1);
            }};
        }

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // ═══ PHASE 1: Forward with state caching ═══
        for step in 0..n_steps {
            // Save state before this step
            encoder.copy_buffer_to_buffer(&r_buf, 0, &r_cache[step], 0, buf_size);
            encoder.copy_buffer_to_buffer(&s_buf, 0, &s_cache[step], 0, buf_size);
            // k1
            dispatch!(encoder, &self.kerr_deriv_batch_pipeline, &bg_fwd_k1);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_mid_r2);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_mid_s2);
            // k2
            dispatch!(encoder, &self.kerr_deriv_batch_pipeline, &bg_fwd_k2);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_mid_r3);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_mid_s3);
            // k3
            dispatch!(encoder, &self.kerr_deriv_batch_pipeline, &bg_fwd_k3);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_mid_r4);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_mid_s4);
            // k4
            dispatch!(encoder, &self.kerr_deriv_batch_pipeline, &bg_fwd_k4);
            // combine + copy
            dispatch!(encoder, &self.rk4_combine_pipeline, &bg_combine_r);
            dispatch!(encoder, &self.rk4_combine_pipeline, &bg_combine_s);
            encoder.copy_buffer_to_buffer(&r_new, 0, &r_buf, 0, buf_size);
            encoder.copy_buffer_to_buffer(&s_new, 0, &s_buf, 0, buf_size);
        }

        // ═══ PHASE 2: Backward chain ═══
        for step in (0..n_steps).rev() {
            // Load cached state for this step
            encoder.copy_buffer_to_buffer(&r_cache[step], 0, &r_buf, 0, buf_size);
            encoder.copy_buffer_to_buffer(&s_cache[step], 0, &s_buf, 0, buf_size);
            // Save incoming gradient (Desktop's key insight)
            encoder.copy_buffer_to_buffer(&d_r, 0, &d_r_step, 0, buf_size);
            encoder.copy_buffer_to_buffer(&d_s, 0, &d_s_step, 0, buf_size);

            // Recompute k1-k4 from cached state
            dispatch!(encoder, &self.kerr_deriv_batch_pipeline, &bg_fwd_k1);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_mid_r2);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_mid_s2);
            dispatch!(encoder, &self.kerr_deriv_batch_pipeline, &bg_fwd_k2);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_mid_r3);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_mid_s3);
            dispatch!(encoder, &self.kerr_deriv_batch_pipeline, &bg_fwd_k3);
            // k4 eval point: r_mid = r + dt*k3
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_mid_r4);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_mid_s4);

            // ── k4 backward ──
            // d_k = 0 + (dt/6)*d_step
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_scale_dt6_r);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_scale_dt6_s);
            // kerr_bwd on k4 eval point (r_mid, s_mid)
            dispatch!(encoder, &self.kerr_bwd_batch_pipeline, &bg_bwd_k4);
            // accumulate d_r += d_eval, param grads
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dr);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_ds);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dg);
            dispatch2x!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dab);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dchi);
            // d_extra = 0 + dt * d_eval (chain: k4 depends on k3)
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_extra_dt_r);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_extra_dt_s);

            // ── k3 backward ──
            // d_k = d_extra + (dt/3)*d_step
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_scale_dt3_r);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_scale_dt3_s);
            // Reconstruct k3 eval point: r_mid = r + 0.5*dt*k2
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_bwd_mid_k3_r);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_bwd_mid_k3_s);
            dispatch!(encoder, &self.kerr_bwd_batch_pipeline, &bg_bwd_k3);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dr);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_ds);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dg);
            dispatch2x!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dab);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dchi);
            // d_extra = 0 + 0.5*dt * d_eval
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_extra_hdt_r);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_extra_hdt_s);

            // ── k2 backward ──
            // d_k = d_extra + (dt/3)*d_step
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_scale_dt3_r);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_scale_dt3_s);
            // Reconstruct k2 eval point: r_mid = r + 0.5*dt*k1
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_bwd_mid_k2_r);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_bwd_mid_k2_s);
            dispatch!(encoder, &self.kerr_bwd_batch_pipeline, &bg_bwd_k2);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dr);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_ds);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dg);
            dispatch2x!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dab);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dchi);
            // d_extra = 0 + 0.5*dt * d_eval
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_extra_hdt_r);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_extra_hdt_s);

            // ── k1 backward ──
            // d_k = d_extra + (dt/6)*d_step
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_scale_k1_r);
            dispatch!(encoder, &self.vec_scale_add_pipeline, &bg_scale_k1_s);
            // k1 eval point is the cached state (r_buf, s_buf)
            dispatch!(encoder, &self.kerr_bwd_batch_pipeline, &bg_bwd_k1);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dr);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_ds);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dg);
            dispatch2x!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dab);
            dispatch!(encoder, &self.vec_accumulate_pipeline, &bg_acc_dchi);
        }

        // ═══ ONE SUBMIT ═══
        self.queue.submit(Some(encoder.finish()));

        // ═══ ONE READBACK ═══
        let d_r_out = self.readback(&d_r, total);
        let d_s_out = self.readback(&d_s, total);
        let dg_out = self.readback(&dg_acc, total);
        let dab_out = self.readback(&dab_acc, total * 2);
        let da_out = &dab_out[..total];
        let db_out = &dab_out[total..];
        let dchi_out = self.readback(&dchi_acc, total);

        // CPU reduction: sum across positions for d_gamma, d_alpha, d_beta, d_chi
        let mut d_gamma_acc = vec![0.0f32; n_bands];
        for pos in 0..n_pos {
            for k in 0..n_bands {
                d_gamma_acc[k] += dg_out[pos * n_bands + k];
            }
        }
        let d_alpha: f32 = da_out.iter().sum();
        let d_beta: f32 = db_out.iter().sum();
        let d_chi: f32 = dchi_out.iter().sum();

        // Softplus chain rule for gamma_raw
        let d_gamma_raw: Vec<f32> = (0..n_bands)
            .map(|k| {
                let s = 1.0 / (1.0 + (-weights.gamma_raw[k]).exp());
                d_gamma_acc[k] * s
            }).collect();

        // Reinterleave d_r, d_s → d_inputs
        let d_inputs: Vec<Vec<f32>> = (0..n_pos).map(|pos| {
            let mut d_input = vec![0.0f32; n_embd];
            for k in 0..n_bands {
                d_input[k * 2] = d_r_out[pos * n_bands + k];
                d_input[k * 2 + 1] = d_s_out[pos * n_bands + k];
            }
            d_input
        }).collect();

        (d_inputs, d_gamma_raw, d_gamma_acc, d_alpha, d_beta, d_chi)
    }
}
