//! Candle backend — autograd-based training for wave-engine.
//!
//! cuBLAS handles all matmul (forward AND backward) with automatic consistency.
//! No manual backward wiring. No ping-pong buffers. No shader precision issues.
//! The ODE runs on CPU as a custom op with identity backward.

#![allow(unused_imports)]

#[cfg(feature = "candle-backend")]
pub mod engine {
    use candle_core::{DType, Device, Module, Result, Tensor, D, CpuStorage, Layout, Shape};
    use candle_nn::{Linear, VarBuilder, VarMap};
    use std::time::Instant;

    /// Linear layer with uniform init matching wgpu engine: uniform(-1/sqrt(in_dim), 1/sqrt(in_dim))
    fn linear_uniform(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
        let bound = 1.0 / (in_dim as f64).sqrt();
        let init = candle_nn::Init::Uniform { lo: -bound, up: bound };
        let ws = vb.get_with_hints((out_dim, in_dim), "weight", init)?;
        let bs = vb.get_with_hints(out_dim, "bias", candle_nn::Init::Const(0.0))?;
        Ok(Linear::new(ws, Some(bs)))
    }

    // Import shared config and utilities
    use crate::{N_BANDS, N_EMBD, N_HEAD, N_LAYERS, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS};
    use crate::wave_embed::{build_harmonic_table, build_positional_table};

    // ─── Layer Norm (simple, no candle_nn LayerNorm to avoid version issues) ───

    fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let diff = x.broadcast_sub(&mean)?;
        let var = (&diff * &diff)?.mean_keepdim(D::Minus1)?;
        let inv_std = (var + 1e-5)?.sqrt()?.recip()?;
        let normed = diff.broadcast_mul(&inv_std)?;
        normed.broadcast_mul(weight)?.broadcast_add(bias)
    }

    // ─── Kerr-ODE (CPU, frozen, identity backward) ───

    /// Perturbative ODE — single-pass analytical Kerr computation.
    /// Based on first-order perturbation theory from telecom DSP.
    /// Lab-validated: MSE 0.000005 vs RK4-16, trains BETTER (2.97 vs 3.07).
    /// Replaces 16 iterative RK4 steps with: damping + rotation + correction.
    fn kerr_ode_cpu(x: &[f32], params: &OdeParams) -> Vec<f32> {
        let n_bands = params.gamma_raw.len();
        let n_embd = n_bands * 2;

        // Per-band magnitude clamp — must match CPU tier (src/common/ffn.rs)
        // Knee compressor — derive ceiling from coupling (matches CPU tier AGC physics)
        let threshold = (std::f32::consts::FRAC_PI_2 / (params.alpha + 4.0 * params.beta)).sqrt();
        let mut clamped = x.to_vec();
        for k in 0..n_bands {
            let r = clamped[k * 2];
            let s = clamped[k * 2 + 1];
            let mag = (r * r + s * s).sqrt();
            if mag > threshold && mag > 0.001 {
                let excess = mag - threshold;
                let compressed = threshold + threshold * (excess / threshold).tanh();
                let scale = compressed / mag;
                clamped[k * 2] *= scale;
                clamped[k * 2 + 1] *= scale;
            }
        }
        let x = &clamped;

        fn softplus(v: f32) -> f32 { if v > 20.0 { v } else { (1.0 + v.exp()).ln() } }
        let gamma: Vec<f32> = params.gamma_raw.iter().map(|&g| softplus(g)).collect();

        // Step 1: Linear solution (damping + base rotation)
        let mut r_lin = vec![0.0f32; n_bands];
        let mut s_lin = vec![0.0f32; n_bands];
        for k in 0..n_bands {
            let r = x[k * 2];
            let s = x[k * 2 + 1];
            let decay = (-gamma[k]).exp();
            let cos_w = params.omega[k].cos();
            let sin_w = params.omega[k].sin();
            r_lin[k] = decay * (r * cos_w - s * sin_w);
            s_lin[k] = decay * (r * sin_w + s * cos_w);
        }

        // Step 2: First-order nonlinear correction (SPM + XPM)
        let mut out = vec![0.0f32; n_embd];
        for k in 0..n_bands {
            let mag_sq = r_lin[k] * r_lin[k] + s_lin[k] * s_lin[k];
            let mut ns = 0.0f32;
            if k >= 2 { ns += r_lin[k-2]*r_lin[k-2] + s_lin[k-2]*s_lin[k-2]; }
            if k >= 1 { ns += r_lin[k-1]*r_lin[k-1] + s_lin[k-1]*s_lin[k-1]; }
            if k+1 < n_bands { ns += r_lin[k+1]*r_lin[k+1] + s_lin[k+1]*s_lin[k+1]; }
            if k+2 < n_bands { ns += r_lin[k+2]*r_lin[k+2] + s_lin[k+2]*s_lin[k+2]; }
            let delta_phi = params.alpha * mag_sq + params.beta * ns;
            out[k * 2]     = r_lin[k] - delta_phi * s_lin[k];
            out[k * 2 + 1] = s_lin[k] + delta_phi * r_lin[k];
        }
        out
    }

    // ─── ODE as CustomOp1 — forward runs RK4, backward is identity ───

    struct KerrOdeCustomOp {
        params: OdeParams,
        n_pos: usize,
        n_embd: usize,
    }

    impl candle_core::CustomOp1 for KerrOdeCustomOp {
        fn name(&self) -> &'static str { "kerr_ode" }

        fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
            let data = match storage {
                CpuStorage::F32(d) => d,
                _ => return Err(candle_core::Error::Msg("Expected f32 storage".to_string())),
            };
            // Handle both contiguous and strided layouts
            let mut results = Vec::with_capacity(self.n_pos * self.n_embd);
            eprintln!("    [ODE CustomOp] n_pos={} n_embd={} data.len={} offset={} contiguous={}",
                self.n_pos, self.n_embd, data.len(), layout.start_offset(), layout.is_contiguous());
            if layout.is_contiguous() {
                let offset = layout.start_offset();
                for pos in 0..self.n_pos {
                    let start = offset + pos * self.n_embd;
                    if start + self.n_embd > data.len() {
                        eprintln!("    [ODE] OOB at pos={pos}: start={start} + n_embd={} > len={}", self.n_embd, data.len());
                        results.extend_from_slice(&vec![0.0f32; self.n_embd]);
                        continue;
                    }
                    let input = &data[start..start + self.n_embd];
                    let max_val = input.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
                    if pos == 0 && max_val > 5.0 {
                        eprintln!("    [ODE] pos=0 max_input={max_val:.2}");
                    }
                    if input.iter().any(|v| v.is_nan() || v.is_infinite()) {
                        results.extend_from_slice(&vec![0.0f32; self.n_embd]);
                    } else {
                        // No clamping — match wgpu CPU exactly
                        results.extend_from_slice(&kerr_ode_cpu(input, &self.params));
                    }
                }
            } else {
                // Non-contiguous: gather elements via strides
                let strides = layout.stride();
                let offset = layout.start_offset();
                for pos in 0..self.n_pos {
                    let mut input = vec![0.0f32; self.n_embd];
                    for i in 0..self.n_embd {
                        let idx = offset + pos * strides[0] + i * strides[1];
                        input[i] = if idx < data.len() { data[idx] } else { 0.0 };
                    }
                    if input.iter().any(|v| v.is_nan() || v.is_infinite()) {
                        results.extend_from_slice(&vec![0.0f32; self.n_embd]);
                    } else {
                        results.extend_from_slice(&kerr_ode_cpu(&input, &self.params));
                    }
                }
            }
            let shape = Shape::from_dims(&[self.n_pos, self.n_embd]);
            Ok((CpuStorage::F32(results), shape))
        }

        fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
            // Identity backward: gradient passes through unchanged
            // This is correct because ODE params are frozen — d_precond = d_kerr_out
            Ok(Some(grad_res.clone()))
        }
    }

    /// Run ODE on a batch using CustomOp1 — proper gradient flow.
    fn kerr_ode_batch(x: &Tensor, params: &OdeParams) -> Result<Tensor> {
        let (n_pos, n_embd) = x.dims2()?;
        // Move to CPU for ODE computation (CustomOp1 cpu_fwd)
        let x_cpu = x.to_device(&Device::Cpu)?;
        let op = KerrOdeCustomOp {
            params: OdeParams {
                gamma_raw: params.gamma_raw.clone(),
                omega: params.omega.clone(),
                alpha: params.alpha,
                beta: params.beta,
                rk4_n_steps: params.rk4_n_steps,
            },
            n_pos,
            n_embd,
        };
        let result = x_cpu.apply_op1(op)?;
        // Move back to original device
        result.to_device(x.device())
    }

    pub struct OdeParams {
        pub gamma_raw: Vec<f32>,
        pub omega: Vec<f32>,
        pub alpha: f32,
        pub beta: f32,
        pub rk4_n_steps: usize,
    }

    // ─── Harmonic Coherence Attention (CPU, frozen) ───

    fn wave_attention(
        x: &Tensor,
        pp_ws_cpu: &[Vec<Vec<f32>>],  // pre-cached on CPU
        pp_bs_cpu: &[Vec<f32>],
        vw_cpu: &[Vec<Vec<f32>>],
        vb_cpu: &[Vec<f32>],
        harmonic_ns: &[f32],
        out_proj_w: &Tensor,
        out_proj_b: &Tensor,
        store_attn_weights: bool,     // true when --harmonics dyn
    ) -> Result<(Tensor, Option<Vec<Vec<Vec<f32>>>>)> {
        let (n_pos, n_embd) = x.dims2()?;
        let n_head = harmonic_ns.len();
        let head_dim = n_embd / n_head;

        // Only ONE GPU→CPU transfer: the input activations (these change every call)
        let x_data = x.to_vec2::<f32>()?;
        let mut out_data = vec![0.0f32; n_pos * n_embd];

        // Optional attention weight storage for harmonic backward
        let mut all_att_weights: Option<Vec<Vec<Vec<f32>>>> = if store_attn_weights {
            Some(vec![vec![vec![0.0; n_pos]; n_pos]; n_head])
        } else {
            None
        };

        for head in 0..n_head {
            let offset = head * head_dim;
            fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }
            let harmonic_n = softplus(harmonic_ns[head]);

            // Phase projection — from CPU cache, zero GPU transfers
            let pp_w = &pp_ws_cpu[head];
            let pp_b = &pp_bs_cpu[head];
            let phases: Vec<f32> = (0..n_pos).map(|pos| {
                let mut r = pp_b[0];
                let mut s = pp_b[1];
                for j in 0..n_embd { r += pp_w[0][j] * x_data[pos][j]; s += pp_w[1][j] * x_data[pos][j]; }
                s.atan2(r)
            }).collect();

            // Value projection — from CPU cache, zero GPU transfers
            let vw = &vw_cpu[head];
            let vb = &vb_cpu[head];
            let v_all: Vec<Vec<f32>> = (0..n_pos).map(|pos| {
                let mut v = vec![0.0f32; head_dim];
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..head_dim { sum += vw[d][j] * x_data[pos][offset + j]; }
                    v[d] = sum + vb[d];
                }
                v
            }).collect();

            // Scoring + weighted sum
            for qi in 0..n_pos {
                let mut scores = vec![f32::NEG_INFINITY; n_pos];
                for ki in 0..=qi {
                    let delta = phases[qi] - phases[ki];
                    scores[ki] = (harmonic_n * delta).cos();
                }
                let max_s = scores[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for ki in 0..=qi { scores[ki] = (scores[ki] - max_s).exp(); exp_sum += scores[ki]; }
                if exp_sum > 0.0 { for ki in 0..=qi { scores[ki] /= exp_sum; } }

                // Store the softmax weights for backward
                if let Some(ref mut aw) = all_att_weights {
                    for ki in 0..=qi {
                        aw[head][qi][ki] = scores[ki];
                    }
                }

                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for ki in 0..=qi { sum += scores[ki] * v_all[ki][d]; }
                    out_data[qi * n_embd + offset + d] = sum;
                }
            }
        }

        // Back to tensor, then out_proj through Candle (GPU, frozen but on grad graph for residual)
        let out_tensor = Tensor::from_vec(out_data, (n_pos, n_embd), x.device())?;
        let projected = out_tensor.matmul(&out_proj_w.t()?)?.broadcast_add(out_proj_b)?;
        Ok((projected, all_att_weights))
    }

    // ─── Harmonic backward (manual chain rule — attention runs on CPU, outside autograd) ───

    /// Compute harmonic gradients for one block.
    /// Called AFTER candle backward, using d_out from the grad accumulator.
    /// d_out is [t][n_embd] — the gradient of the block's contribution tensor.
    /// This equals d_attn_out because contribution = attn_out + ffn_out (sum splits gradient equally).
    fn harmonic_backward(
        block: &CandleBlock,
        d_out: &[Vec<f32>],        // [t][n_embd] — gradient of contribution (= d_attn_out)
        n_embd: usize,
    ) -> Vec<f32> {                 // [n_head] — d_loss/d_harmonic_raw per head
        let t = d_out.len();
        let n_head = block.harmonic_ns.len();
        let head_dim = n_embd / n_head;

        fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }
        fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

        let att_w = block.cached_att_weights.as_ref()
            .expect("harmonic backward requires cached attention weights");
        let input = block.cached_normed_cpu.as_ref()
            .expect("harmonic backward requires cached normed input");

        let mut d_harmonic_raws = vec![0.0f32; n_head];

        for h in 0..n_head {
            let harmonic_n = softplus(block.harmonic_ns[h]);
            let offset = h * head_dim;

            // Recompute phases (same as forward)
            let pp_w = &block.phase_proj_ws_cpu[h];
            let pp_b = &block.phase_proj_bs_cpu[h];
            let phases: Vec<f32> = (0..t).map(|pos| {
                let mut r = pp_b[0];
                let mut s = pp_b[1];
                for j in 0..n_embd { r += pp_w[0][j] * input[pos][j]; s += pp_w[1][j] * input[pos][j]; }
                s.atan2(r)
            }).collect();

            // Recompute value projections (same as forward)
            let vw = &block.v_proj_ws_cpu[h];
            let vb = &block.v_proj_bs_cpu[h];
            let v_all: Vec<Vec<f32>> = (0..t).map(|pos| {
                let mut v = vec![0.0f32; head_dim];
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..head_dim { sum += vw[d][j] * input[pos][offset + j]; }
                    v[d] = sum + vb[d];
                }
                v
            }).collect();

            // Accumulate d_h across all query positions
            let mut d_h = 0.0f32;
            for qi in 0..t {
                // d_weight from d_output: d_w[qi][ki] = sum_d d_out[qi][offset+d] * v_all[ki][d]
                let mut d_w_qi = vec![0.0f32; t];
                for ki in 0..=qi {
                    if att_w[h][qi][ki] > 0.0 {
                        let mut dw = 0.0f32;
                        for d in 0..head_dim {
                            dw += d_out[qi][offset + d] * v_all[ki][d];
                        }
                        d_w_qi[ki] = dw;
                    }
                }

                // Softmax backward
                let weighted_sum: f32 = (0..=qi)
                    .map(|ki| att_w[h][qi][ki] * d_w_qi[ki])
                    .sum();

                // Accumulate through cosine derivative
                for ki in 0..=qi {
                    if att_w[h][qi][ki] > 0.0 {
                        let d_score = att_w[h][qi][ki] * (d_w_qi[ki] - weighted_sum);
                        let delta = phases[qi] - phases[ki];
                        let d_score_d_h = -(harmonic_n * delta).sin() * delta;
                        d_h += d_score * d_score_d_h;
                    }
                }
            }

            // Chain through softplus: d_loss/d_harmonic_raw = d_h * sigmoid(harmonic_raw)
            d_harmonic_raws[h] = d_h * sigmoid(block.harmonic_ns[h]);
        }

        d_harmonic_raws
    }

    // ─── Monitor data collected during forward ───

    /// Per-layer flow statistics (norms, ratios, cosine similarity).
    struct CandleLayerFlow {
        layer: usize,
        input_norm: f32,
        attn_out_norm: f32,
        ffn_out_norm: f32,
        output_norm: f32,
        attn_ratio: f32,
        ffn_ratio: f32,
        residual_ratio: f32,
        cosine_in_out: f32,
    }

    /// Per-head attention statistics.
    struct CandleAttnHead {
        layer: usize,
        head: usize,
        harmonic: f32,
        entropy: f32,
        max_weight: f32,
    }

    /// Per-layer ODE dynamics statistics.
    struct CandleOdeDynamics {
        layer: usize,
        energy_in: f32,
        energy_out: f32,
        energy_ratio: f32,
        phase_velocity: f32,
        damping: f32,
        band_energy_std: f32,
    }

    /// Monitor data collected during one forward pass.
    #[derive(Default)]
    struct CandleMonitorData {
        layer_flow: Vec<CandleLayerFlow>,
        attn_heads: Vec<CandleAttnHead>,
        ode_dynamics: Vec<CandleOdeDynamics>,
    }

    impl CandleMonitorData {
        fn layer_flow_json(&self) -> String {
            if self.layer_flow.is_empty() { return String::new(); }
            let entries: Vec<String> = self.layer_flow.iter().map(|s| {
                format!(
                    r#"{{"layer":{},"in_norm":{:.3},"attn_norm":{:.3},"ffn_norm":{:.3},"out_norm":{:.3},"attn_ratio":{:.3},"ffn_ratio":{:.3},"resid_ratio":{:.3},"cos_in_out":{:.4}}}"#,
                    s.layer, s.input_norm, s.attn_out_norm, s.ffn_out_norm, s.output_norm,
                    s.attn_ratio, s.ffn_ratio, s.residual_ratio, s.cosine_in_out,
                )
            }).collect();
            format!(r#""layer_flow":[{}]"#, entries.join(","))
        }

        fn attn_heads_json(&self) -> String {
            if self.attn_heads.is_empty() { return String::new(); }
            let entries: Vec<String> = self.attn_heads.iter().map(|s| {
                format!(
                    r#"{{"layer":{},"head":{},"harmonic":{:.3},"entropy":{:.3},"max_w":{:.4}}}"#,
                    s.layer, s.head, s.harmonic, s.entropy, s.max_weight,
                )
            }).collect();
            format!(r#""attn_heads":[{}]"#, entries.join(","))
        }

        fn ode_dynamics_json(&self) -> String {
            if self.ode_dynamics.is_empty() { return String::new(); }
            let entries: Vec<String> = self.ode_dynamics.iter().map(|s| {
                format!(
                    r#"{{"layer":{},"phase_vel":{:.4},"energy_in":{:.2},"energy_out":{:.2},"energy_ratio":{:.4},"band_std":{:.4},"damping":{:.4}}}"#,
                    s.layer, s.phase_velocity, s.energy_in, s.energy_out,
                    s.energy_ratio, s.band_energy_std, s.damping,
                )
            }).collect();
            format!(r#""ode_dynamics":[{}]"#, entries.join(","))
        }
    }

    /// Output distribution statistics (computed from logits + targets).
    struct CandleOutputDist {
        avg_entropy: f32,
        avg_margin: f32,
        avg_correct_rank: f32,
        worst_margin: f32,
        worst_pos: usize,
        mode_collapse: bool,
    }

    fn compute_output_dist(logits: &Tensor, targets: &[usize]) -> CandleOutputDist {
        let logits_cpu = match logits.to_vec2::<f32>() {
            Ok(v) => v,
            Err(_) => return CandleOutputDist {
                avg_entropy: 0.0, avg_margin: 0.0, avg_correct_rank: 0.0,
                worst_margin: 0.0, worst_pos: 0, mode_collapse: false,
            },
        };
        let stats = crate::common::output_monitor::analyze_output(&logits_cpu, targets);
        CandleOutputDist {
            avg_entropy: stats.avg_entropy,
            avg_margin: stats.avg_margin,
            avg_correct_rank: stats.avg_correct_rank,
            worst_margin: stats.worst_margin,
            worst_pos: stats.worst_prompt_pos,
            mode_collapse: stats.mode_collapse,
        }
    }

    fn output_dist_json(s: &CandleOutputDist) -> String {
        format!(
            r#""output_dist":{{"avg_entropy":{:.3},"avg_margin":{:.4},"avg_correct_rank":{:.1},"worst_margin":{:.4},"worst_pos":{},"mode_collapse":{}}}"#,
            s.avg_entropy, s.avg_margin, s.avg_correct_rank,
            s.worst_margin, s.worst_pos, s.mode_collapse,
        )
    }

    /// Per-layer gradient flow statistics.
    struct CandleGradientFlow {
        layer: usize,
        ln_norm: f32,
        maestro_in_norm: f32,
        ode_norm: f32,
        maestro_out_norm: f32,
        out_proj_norm: f32,
    }

    fn compute_gradient_flow(
        grads: &candle_core::backprop::GradStore,
        varmap: &VarMap,
        n_layers: usize,
    ) -> Vec<CandleGradientFlow> {
        let data = varmap.data().lock().unwrap();
        let mut stats = Vec::with_capacity(n_layers);

        for layer in 0..n_layers {
            let prefix = format!("block.{layer}.");

            let grad_norm_for = |suffix: &str| -> f32 {
                let key = format!("{prefix}{suffix}");
                if let Some(var) = data.get(&key) {
                    if let Some(g) = grads.get(var) {
                        let flat: Vec<f32> = g.flatten_all().unwrap().to_vec1::<f32>().unwrap_or_default();
                        return flat.iter().map(|x| x * x).sum::<f32>().sqrt();
                    }
                }
                0.0
            };

            // LN: combine attn LN weight + bias
            let ln_w = grad_norm_for("ln_w");
            let ln_b = grad_norm_for("ln_b");
            let ln_norm = (ln_w * ln_w + ln_b * ln_b).sqrt();

            // Maestro in: squeeze + process
            let mi_sw = grad_norm_for("mae_in_sq.weight");
            let mi_sb = grad_norm_for("mae_in_sq.bias");
            let mi_pw = grad_norm_for("mae_in_pr.weight");
            let mi_pb = grad_norm_for("mae_in_pr.bias");
            let maestro_in_norm = (mi_sw*mi_sw + mi_sb*mi_sb + mi_pw*mi_pw + mi_pb*mi_pb).sqrt();

            // Maestro out
            let mo_sw = grad_norm_for("mae_out_sq.weight");
            let mo_sb = grad_norm_for("mae_out_sq.bias");
            let mo_pw = grad_norm_for("mae_out_pr.weight");
            let mo_pb = grad_norm_for("mae_out_pr.bias");
            let maestro_out_norm = (mo_sw*mo_sw + mo_sb*mo_sb + mo_pw*mo_pw + mo_pb*mo_pb).sqrt();

            // ODE params: alpha, beta, gamma_raw, phase_correction
            let ode_a = grad_norm_for("ode.alpha");
            let ode_b = grad_norm_for("ode.beta");
            let ode_g = grad_norm_for("ode.gamma_raw");
            let ode_pc = grad_norm_for("phase_correction");
            let ode_rk4 = grad_norm_for("ode.rk4_weights");
            let ode_norm = (ode_a*ode_a + ode_b*ode_b + ode_g*ode_g + ode_pc*ode_pc + ode_rk4*ode_rk4).sqrt();

            // Out proj (block-diagonal groups)
            let mut op_sq = 0.0f32;
            for g in 0..16 { // enough groups for any config
                let w = grad_norm_for(&format!("out_proj.g{g}.weight"));
                let b = grad_norm_for(&format!("out_proj.g{g}.bias"));
                op_sq += w * w + b * b;
            }
            let out_proj_norm = op_sq.sqrt();

            stats.push(CandleGradientFlow {
                layer,
                ln_norm,
                maestro_in_norm,
                ode_norm,
                maestro_out_norm,
                out_proj_norm,
            });
        }

        stats
    }

    fn gradient_flow_json(stats: &[CandleGradientFlow]) -> String {
        if stats.is_empty() { return String::new(); }
        let entries: Vec<String> = stats.iter().map(|s| {
            format!(
                r#"{{"layer":{},"ln":{:.4},"maestro_in":{:.4},"ode":{:.4},"maestro_out":{:.4},"out_proj":{:.4}}}"#,
                s.layer, s.ln_norm, s.maestro_in_norm, s.ode_norm,
                s.maestro_out_norm, s.out_proj_norm,
            )
        }).collect();
        format!(r#""grad_flow":[{}]"#, entries.join(","))
    }

    // ─── Model ───

    pub struct CandleWaveModel {
        // Frozen embeddings
        wte: Tensor,
        wpe: Tensor,

        // Per-block
        blocks: Vec<CandleBlock>,

        // Final
        ln_f_w: Tensor,
        ln_f_b: Tensor,
        lm_head: Tensor,
        output_corrector: Option<Tensor>,  // [1, n_bands] phase-native output corrector
        phase_native: bool,
        layer_agcs: Option<Vec<crate::common::agc::OdeAgc>>,  // per-layer AGC (when --agc-headroom dyn)

        device: Device,

        // Runtime config
        n_bands: usize,
        n_embd: usize,
        n_head: usize,
        block_size: usize,
        debug_nan: bool,
    }

    struct CandleBlock {
        // LN (trained)
        ln_w: Tensor,
        ln_b: Tensor,

        // Attention (frozen) — GPU tensors for out_proj gradient graph
        phase_proj_ws: Vec<Tensor>,
        phase_proj_bs: Vec<Tensor>,
        v_proj_ws: Vec<Tensor>,
        v_proj_bs: Vec<Tensor>,
        harmonic_ns: Vec<f32>,   // harmonic_raw values (softplus before use in scoring)
        harmonic_init: Vec<f32>, // initial values (for spring equilibrium)
        attn_out_proj_w: Tensor,
        attn_out_proj_b: Tensor,

        // Attention (frozen) — CPU-cached copies, eliminates GPU→CPU transfers per forward call
        phase_proj_ws_cpu: Vec<Vec<Vec<f32>>>,  // [n_head][2][n_embd]
        phase_proj_bs_cpu: Vec<Vec<f32>>,        // [n_head][2]
        v_proj_ws_cpu: Vec<Vec<Vec<f32>>>,       // [n_head][head_dim][head_dim]
        v_proj_bs_cpu: Vec<Vec<f32>>,            // [n_head][head_dim]

        // Harmonics dyn — attention cache for manual backward
        harmonic_dyn: bool,                                    // --harmonics dyn active
        cached_att_weights: Option<Vec<Vec<Vec<f32>>>>,        // [n_head][t][t] post-softmax
        cached_normed_cpu: Option<Vec<Vec<f32>>>,              // [t][n_embd] block input (normed)
        cached_layer_output: Option<Tensor>,                   // layer output tensor (on grad graph)

        // FFN (trained via VarMap)
        mae_in_sq: Linear,
        mae_in_pr: Linear,
        ode_params: OdeParams,
        gpu_ode_params: crate::gpu_ode::gpu_ode::GpuOdeParams,
        phase_correction: Tensor,  // [1, n_bands] — corrector plate phase angles (learnable)
        mae_out_sq: Linear,
        mae_out_pr: Linear,
        out_proj: crate::block_diagonal::block_diag::BlockDiagonalLinear,
        layer_scale: Option<Tensor>,  // [1] scalar — residual contribution multiplier
    }

    impl CandleWaveModel {
        pub fn new(varmap: &VarMap, vocab_size: usize, device: &Device,
                   n_bands: usize, n_head: usize, n_layers: usize, maestro_dim: usize,
                   rk4_steps: usize, out_proj_groups: usize, alpha: f32, beta: f32) -> Result<Self> {
            let n_embd = n_bands * 2;
            let block_size = 256; // positional table size
            // Save config for methods
            let n_bands_cfg = n_bands;
            let n_embd_cfg = n_embd;
            let n_head_cfg = n_head;
            let block_size_cfg = block_size;
            let mut rng = crate::rng::Rng::new(42);

            // Frozen embeddings
            let wte_data = build_harmonic_table(vocab_size, n_bands);
            let wte_flat: Vec<f32> = wte_data.iter().flat_map(|r| r.iter().copied()).collect();
            let wte = Tensor::from_vec(wte_flat, (vocab_size, n_embd), device)?;

            let wpe_data = build_positional_table(block_size, n_bands);
            let wpe_flat: Vec<f32> = wpe_data.iter().flat_map(|r| r.iter().copied()).collect();
            let wpe = Tensor::from_vec(wpe_flat, (block_size, n_embd), device)?;

            let vs = VarBuilder::from_varmap(varmap, DType::F32, device);

            let mut blocks = Vec::new();
            for layer in 0..n_layers {
                let prefix = format!("block.{layer}");
                let vs_block = vs.pp(&prefix);

                // LN (trained)
                let ln_w = vs_block.get_with_hints((n_embd,), "ln_w", candle_nn::Init::Const(1.0))?;
                let ln_b = vs_block.get_with_hints((n_embd,), "ln_b", candle_nn::Init::Const(0.0))?;

                // Attention heads (frozen)
                let head_dim = n_embd / n_head;
                let mut phase_proj_ws = Vec::new();
                let mut phase_proj_bs = Vec::new();
                let mut v_proj_ws = Vec::new();
                let mut v_proj_bs = Vec::new();
                let mut harmonic_ns = Vec::new();

                for h in 0..n_head {
                    let limit = 1.0 / (n_embd as f32).sqrt();
                    let pw: Vec<f32> = (0..2*n_embd).map(|_| rng.uniform(limit)).collect();
                    let pb = vec![0.0f32; 2];
                    phase_proj_ws.push(Tensor::from_vec(pw, (2, n_embd), device)?);
                    phase_proj_bs.push(Tensor::from_vec(pb, (2,), device)?);

                    let vlimit = 1.0 / (head_dim as f32).sqrt();
                    let vw: Vec<f32> = (0..head_dim*head_dim).map(|_| rng.uniform(vlimit)).collect();
                    let vb = vec![0.0f32; head_dim];
                    v_proj_ws.push(Tensor::from_vec(vw, (head_dim, head_dim), device)?);
                    v_proj_bs.push(Tensor::from_vec(vb, (head_dim,), device)?);

                    harmonic_ns.push(((h + 1) as f32 * 0.5f32).ln());
                }

                let olimit = 1.0 / (n_embd as f32).sqrt();
                let ow: Vec<f32> = (0..n_embd*n_embd).map(|_| rng.uniform(olimit)).collect();
                let ob = vec![0.0f32; n_embd];
                let attn_out_proj_w = Tensor::from_vec(ow, (n_embd, n_embd), device)?;
                let attn_out_proj_b = Tensor::from_vec(ob, (n_embd,), device)?;

                // FFN (trained)
                let mae_in_sq = linear_uniform(n_embd, maestro_dim, vs_block.pp("mae_in_sq"))?;
                let mae_in_pr = linear_uniform(maestro_dim, n_embd, vs_block.pp("mae_in_pr"))?;
                let mae_out_sq = linear_uniform(n_embd, maestro_dim, vs_block.pp("mae_out_sq"))?;
                let mae_out_pr = linear_uniform(maestro_dim, n_embd, vs_block.pp("mae_out_pr"))?;
                let out_proj = crate::block_diagonal::block_diag::BlockDiagonalLinear::new(
                    n_embd, out_proj_groups, vs_block.pp("out_proj"),  // configurable groups
                )?;

                // ODE params (learnable via VarMap — autograd computes gradients)
                let gamma_raw_val = ((0.1f32).exp() - 1.0).ln();
                let ode_params = OdeParams {
                    gamma_raw: vec![gamma_raw_val; n_bands],
                    omega: (0..n_bands).map(|k| (k + 1) as f32 / n_bands as f32).collect(),
                    alpha,
                    beta,
                    rk4_n_steps: rk4_steps,
                };
                let gpu_ode_params = crate::gpu_ode::gpu_ode::GpuOdeParams::learnable(
                    n_bands, alpha, beta, rk4_steps, vs_block.pp("ode"),
                )?;

                // Cache frozen attention weights on CPU — eliminates 48 GPU→CPU transfers per layer per forward
                let phase_proj_ws_cpu: Vec<Vec<Vec<f32>>> = phase_proj_ws.iter()
                    .map(|t| t.to_vec2::<f32>().unwrap()).collect();
                let phase_proj_bs_cpu: Vec<Vec<f32>> = phase_proj_bs.iter()
                    .map(|t| t.to_vec1::<f32>().unwrap()).collect();
                let v_proj_ws_cpu: Vec<Vec<Vec<f32>>> = v_proj_ws.iter()
                    .map(|t| t.to_vec2::<f32>().unwrap()).collect();
                let v_proj_bs_cpu: Vec<Vec<f32>> = v_proj_bs.iter()
                    .map(|t| t.to_vec1::<f32>().unwrap()).collect();

                // Corrector plate: per-band phase rotation (zero-init = transparent, learnable)
                let phase_correction = vs_block.get_with_hints(
                    (1, n_bands), "phase_correction",
                    candle_nn::Init::Const(0.0),
                )?;

                blocks.push(CandleBlock {
                    ln_w, ln_b,
                    phase_proj_ws, phase_proj_bs, v_proj_ws, v_proj_bs,
                    phase_proj_ws_cpu, phase_proj_bs_cpu, v_proj_ws_cpu, v_proj_bs_cpu,
                    harmonic_init: harmonic_ns.clone(),
                    harmonic_ns, attn_out_proj_w, attn_out_proj_b,
                    harmonic_dyn: false,  // set by train_candle when --harmonics dyn
                    cached_att_weights: None,
                    cached_normed_cpu: None,
                    cached_layer_output: None,
                    mae_in_sq, mae_in_pr, ode_params, gpu_ode_params, phase_correction,
                    mae_out_sq, mae_out_pr, out_proj,
                    layer_scale: None, // set by --layer-scale dyn
                });
            }

            // Final LN + LM head (trained) or output corrector (phase-native)
            let ln_f_w = vs.get_with_hints((n_embd,), "ln_f_w", candle_nn::Init::Const(1.0))?;
            let ln_f_b = vs.get_with_hints((n_embd,), "ln_f_b", candle_nn::Init::Const(0.0))?;
            // TODO: make phase_native configurable via CLI flag
            let phase_native = false; // will be set by caller
            let lm_head = if !phase_native {
                vs.get_with_hints((vocab_size, n_embd), "lm_head",
                    candle_nn::Init::Randn { mean: 0.0, stdev: 1.0 / (n_embd as f64).sqrt() })?
            } else {
                // Dummy — not used in phase-native mode
                Tensor::zeros((1, 1), DType::F32, device)?
            };
            let output_corrector = if phase_native {
                Some(vs.get_with_hints((1, n_bands), "output_corrector", candle_nn::Init::Const(0.0))?)
            } else {
                None
            };

            Ok(Self { wte, wpe, blocks, ln_f_w, ln_f_b, lm_head, output_corrector, phase_native,
                layer_agcs: None, device: device.clone(),
                n_bands: n_bands_cfg, n_embd: n_embd_cfg, n_head: n_head_cfg, block_size: block_size_cfg, debug_nan: false })
        }

        pub fn forward(&mut self, token_ids: &[usize]) -> Result<Tensor> {
            self.forward_with_curriculum(token_ids, &vec![1.0f32; self.n_bands])
        }

        /// Forward with curriculum: soft-mask inactive bands on FFN path only.
        /// `band_masks[k]` is the mask value for band k (0.01 for suppressed, 1.0 for active,
        /// intermediate values during ramp transitions).
        /// Attention sees full vector (frozen). FFN sees masked vector (trains on active bands).
        pub fn forward_with_curriculum(&mut self, token_ids: &[usize], band_masks: &[f32]) -> Result<Tensor> {
            let n_bands = self.n_bands;
            let n_embd = self.n_embd;
            let n_head = self.n_head;
            let block_size = self.block_size;
            let n_pos = token_ids.len();

            // Build GPU-resident mask from per-band values
            let ffn_mask = if band_masks.iter().any(|&v| v < 1.0) {
                let mut mask_data = vec![0.0f32; n_embd];
                for k in 0..n_bands {
                    mask_data[k * 2] = band_masks[k];
                    mask_data[k * 2 + 1] = band_masks[k];
                }
                Some(Tensor::from_vec(mask_data, (1, n_embd), &self.device)?)
            } else {
                None
            };

            // Embedding: lookup + positional (NO masking — LN needs full vector)
            let mut hidden_vecs = vec![0.0f32; n_pos * n_embd];
            let wte_data = self.wte.to_vec2::<f32>()?;
            let wpe_data = self.wpe.to_vec2::<f32>()?;
            for (pos, &tok) in token_ids.iter().enumerate() {
                for i in 0..n_embd {
                    hidden_vecs[pos * n_embd + i] = wte_data[tok][i] + wpe_data[pos][i];
                }
            }
            let mut hidden = Tensor::from_vec(hidden_vecs, (n_pos, n_embd), &self.device)?;

            for (block_idx, block) in self.blocks.iter_mut().enumerate() {
                let normed = layer_norm(&hidden, &block.ln_w, &block.ln_b)?;
                if self.debug_nan {
                    if normed.to_vec2::<f32>()?.iter().any(|r| r.iter().any(|v| v.is_nan())) {
                        eprintln!("  [NaN] block {block_idx} after LN");
                    }
                }

                // Cache normed input on CPU when harmonics dyn (for manual backward)
                if block.harmonic_dyn {
                    block.cached_normed_cpu = Some(normed.to_vec2::<f32>()?);
                }

                // Attention (frozen, CPU scoring, GPU out_proj)
                let (attn_out, att_weights) = wave_attention(
                    &normed,
                    &block.phase_proj_ws_cpu, &block.phase_proj_bs_cpu,
                    &block.v_proj_ws_cpu, &block.v_proj_bs_cpu,
                    &block.harmonic_ns,
                    &block.attn_out_proj_w, &block.attn_out_proj_b,
                    block.harmonic_dyn,
                )?;
                block.cached_att_weights = att_weights;

                // FFN (trained) — soft-mask inactive bands (curriculum)
                // Attention sees full normed (frozen, routes only).
                // FFN sees masked normed (trains on active bands).
                let ffn_input = match &ffn_mask {
                    Some(mask) => normed.broadcast_mul(mask)?,
                    None => normed.clone(),
                };

                // Maestro in
                let mae_in = block.mae_in_sq.forward(&ffn_input)?;
                let mae_in = mae_in.gelu()?;
                let mae_in = block.mae_in_pr.forward(&mae_in)?;
                let precond = (&ffn_input + &mae_in)?;

                // AGC knee compression — per-layer or global
                let precond = {
                    let mut pv: Vec<Vec<f32>> = precond.to_vec2()?;
                    let nb = pv[0].len() / 2;
                    if let Some(ref mut agcs) = self.layer_agcs {
                        agcs[block_idx].process(&mut pv, nb);
                    } else {
                        let mut agc = crate::ffn_backend::AGC.get().unwrap().lock().unwrap();
                        agc.process(&mut pv, nb);
                    }
                    candle_core::Tensor::new(pv, precond.device())?
                };

                if self.debug_nan {
                    let precond_vals = precond.to_vec2::<f32>()?;
                    let precond_max = precond_vals.iter()
                        .flat_map(|r| r.iter()).cloned().fold(0.0f32, |a, b| a.max(b.abs()));
                    if precond_max > 10.0 || precond_max.is_nan() || precond_max.is_infinite() {
                        eprintln!("  [PRECOND] block {block_idx} max={precond_max:.2}");
                    }
                }

                // ODE — GPU-native RK4, autograd-compatible
                let ode_out = crate::gpu_ode::gpu_ode::kerr_ode_gpu(&precond, &block.gpu_ode_params)?;

                // Corrector plate: per-band phase rotation (autograd through sin/cos)
                let effective_ode_out = {
                    let n_b = self.n_bands;
                    let reshaped = ode_out.reshape((n_pos, n_b, 2))?;
                    let r = reshaped.narrow(2, 0, 1)?.squeeze(2)?;
                    let s = reshaped.narrow(2, 1, 1)?.squeeze(2)?;
                    let cos_c = block.phase_correction.cos()?;
                    let sin_c = block.phase_correction.sin()?;
                    let r_rot = (r.broadcast_mul(&cos_c)? - s.broadcast_mul(&sin_c)?)?;
                    let s_rot = (r.broadcast_mul(&sin_c)? + s.broadcast_mul(&cos_c)?)?;
                    let r_exp = r_rot.unsqueeze(2)?;
                    let s_exp = s_rot.unsqueeze(2)?;
                    Tensor::cat(&[&r_exp, &s_exp], 2)?.reshape((n_pos, n_embd))?
                };

                // Maestro out (operates on corrected ODE output — gradients flow through corrector)
                let mae_out = block.mae_out_sq.forward(&effective_ode_out)?;
                let mae_out = mae_out.gelu()?;
                let mae_out = block.mae_out_pr.forward(&mae_out)?;
                let regulated = (&effective_ode_out + &mae_out)?;

                if self.debug_nan {
                    if effective_ode_out.to_vec2::<f32>()?.iter().any(|r| r.iter().any(|v| v.is_nan())) {
                        eprintln!("  [NaN] block {block_idx} ODE output");
                    }
                    if regulated.to_vec2::<f32>()?.iter().any(|r| r.iter().any(|v| v.is_nan())) {
                        eprintln!("  [NaN] block {block_idx} regulated (before out_proj)");
                    }
                }

                // Out proj
                let ffn_out = block.out_proj.forward(&regulated)?;

                if self.debug_nan {
                    if attn_out.to_vec2::<f32>()?.iter().any(|r| r.iter().any(|v| v.is_nan())) {
                        eprintln!("  [NaN] block {block_idx} after attention");
                    }
                    if ffn_out.to_vec2::<f32>()?.iter().any(|r| r.iter().any(|v| v.is_nan())) {
                        eprintln!("  [NaN] block {block_idx} after FFN");
                    }
                }

                // Parallel residual: hidden + scale * (attn_out + ffn_out)
                let contribution = (&attn_out + &ffn_out)?;
                // Cache contribution tensor for harmonic backward (gradient extraction)
                if block.harmonic_dyn {
                    block.cached_layer_output = Some(contribution.clone());
                }
                hidden = if let Some(ref scale) = block.layer_scale {
                    (&hidden + contribution.broadcast_mul(scale)?)?
                } else {
                    (&hidden + &contribution)?
                };
            }

            // Final LN
            let normed = layer_norm(&hidden, &self.ln_f_w, &self.ln_f_b)?;

            let logits = if self.phase_native {
                // Phase-native: output corrector + dot product against frozen embeddings
                if let Some(ref oc) = self.output_corrector {
                    let n_b = self.n_bands;
                    let reshaped = normed.reshape((n_pos, n_b, 2))?;
                    let r = reshaped.narrow(2, 0, 1)?.squeeze(2)?;
                    let s = reshaped.narrow(2, 1, 1)?.squeeze(2)?;
                    let cos_c = oc.cos()?;
                    let sin_c = oc.sin()?;
                    let r_rot = (r.broadcast_mul(&cos_c)? - s.broadcast_mul(&sin_c)?)?;
                    let s_rot = (r.broadcast_mul(&sin_c)? + s.broadcast_mul(&cos_c)?)?;
                    let r_exp = r_rot.unsqueeze(2)?;
                    let s_exp = s_rot.unsqueeze(2)?;
                    let corrected = Tensor::cat(&[&r_exp, &s_exp], 2)?.reshape((n_pos, self.n_embd))?;
                    // Dot product against frozen embeddings (wte): [n_pos, n_embd] × [n_embd, vocab] = [n_pos, vocab]
                    corrected.matmul(&self.wte.t()?)?
                } else {
                    // No output corrector — direct dot product
                    normed.matmul(&self.wte.t()?)?
                }
            } else {
                // Standard: lm_head projection
                normed.matmul(&self.lm_head.t()?)?
            };

            Ok(logits)
        }

        /// Forward pass with monitor data collection.
        /// Same logic as forward_with_curriculum, but captures per-layer norms,
        /// attention stats, and ODE dynamics. Only called at health intervals.
        fn forward_with_monitors(
            &mut self, token_ids: &[usize], band_masks: &[f32],
        ) -> Result<(Tensor, CandleMonitorData)> {
            let n_bands = self.n_bands;
            let n_embd = self.n_embd;
            let n_head = self.n_head;
            let _block_size = self.block_size;
            let n_pos = token_ids.len();
            let mut monitor = CandleMonitorData::default();

            // Build GPU-resident mask from per-band values
            let ffn_mask = if band_masks.iter().any(|&v| v < 1.0) {
                let mut mask_data = vec![0.0f32; n_embd];
                for k in 0..n_bands {
                    mask_data[k * 2] = band_masks[k];
                    mask_data[k * 2 + 1] = band_masks[k];
                }
                Some(Tensor::from_vec(mask_data, (1, n_embd), &self.device)?)
            } else {
                None
            };

            // Embedding
            let mut hidden_vecs = vec![0.0f32; n_pos * n_embd];
            let wte_data = self.wte.to_vec2::<f32>()?;
            let wpe_data = self.wpe.to_vec2::<f32>()?;
            for (pos, &tok) in token_ids.iter().enumerate() {
                for i in 0..n_embd {
                    hidden_vecs[pos * n_embd + i] = wte_data[tok][i] + wpe_data[pos][i];
                }
            }
            let mut hidden = Tensor::from_vec(hidden_vecs, (n_pos, n_embd), &self.device)?;

            for (block_idx, block) in self.blocks.iter().enumerate() {
                let normed = layer_norm(&hidden, &block.ln_w, &block.ln_b)?;

                // ── Attention (with per-head monitoring) ──
                // Run wave_attention on CPU, also extract per-head entropy/max_weight
                let x_data = normed.to_vec2::<f32>()?;
                let head_dim = n_embd / n_head;
                let mut attn_out_data = vec![0.0f32; n_pos * n_embd];

                for head in 0..n_head {
                    let offset = head * head_dim;
                    fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }
                    let harmonic_n = softplus(block.harmonic_ns[head]);

                    let pp_w = &block.phase_proj_ws_cpu[head];
                    let pp_b = &block.phase_proj_bs_cpu[head];
                    let phases: Vec<f32> = (0..n_pos).map(|pos| {
                        let mut r = pp_b[0];
                        let mut s = pp_b[1];
                        for j in 0..n_embd { r += pp_w[0][j] * x_data[pos][j]; s += pp_w[1][j] * x_data[pos][j]; }
                        s.atan2(r)
                    }).collect();

                    let vw = &block.v_proj_ws_cpu[head];
                    let vb = &block.v_proj_bs_cpu[head];
                    let v_all: Vec<Vec<f32>> = (0..n_pos).map(|pos| {
                        let mut v = vec![0.0f32; head_dim];
                        for d in 0..head_dim {
                            let mut sum = 0.0f32;
                            for j in 0..head_dim { sum += vw[d][j] * x_data[pos][offset + j]; }
                            v[d] = sum + vb[d];
                        }
                        v
                    }).collect();

                    // Track attention stats from the last query position
                    let last_qi = n_pos.saturating_sub(1);
                    let mut head_entropy = 0.0f32;
                    let mut head_max_weight = 0.0f32;

                    for qi in 0..n_pos {
                        let mut scores = vec![f32::NEG_INFINITY; n_pos];
                        for ki in 0..=qi {
                            let delta = phases[qi] - phases[ki];
                            scores[ki] = (harmonic_n * delta).cos();
                        }
                        let max_s = scores[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut exp_sum = 0.0f32;
                        for ki in 0..=qi { scores[ki] = (scores[ki] - max_s).exp(); exp_sum += scores[ki]; }
                        if exp_sum > 0.0 { for ki in 0..=qi { scores[ki] /= exp_sum; } }

                        // Capture stats for last position
                        if qi == last_qi {
                            for ki in 0..=qi {
                                let w = scores[ki];
                                if w > head_max_weight { head_max_weight = w; }
                                if w > 0.0 { head_entropy -= w * w.ln(); }
                            }
                        }

                        for d in 0..head_dim {
                            let mut sum = 0.0f32;
                            for ki in 0..=qi { sum += scores[ki] * v_all[ki][d]; }
                            attn_out_data[qi * n_embd + offset + d] = sum;
                        }
                    }

                    monitor.attn_heads.push(CandleAttnHead {
                        layer: block_idx,
                        head,
                        harmonic: harmonic_n,
                        entropy: head_entropy,
                        max_weight: head_max_weight,
                    });
                }

                let attn_out_tensor = Tensor::from_vec(attn_out_data, (n_pos, n_embd), &self.device)?;
                let attn_out = attn_out_tensor.matmul(&block.attn_out_proj_w.t()?)?.broadcast_add(&block.attn_out_proj_b)?;

                // ── FFN path ──
                let ffn_input = match &ffn_mask {
                    Some(mask) => normed.broadcast_mul(mask)?,
                    None => normed.clone(),
                };

                let mae_in = block.mae_in_sq.forward(&ffn_input)?;
                let mae_in = mae_in.gelu()?;
                let mae_in = block.mae_in_pr.forward(&mae_in)?;
                let precond = (&ffn_input + &mae_in)?;

                // AGC
                let precond = {
                    let mut pv: Vec<Vec<f32>> = precond.to_vec2()?;
                    let nb = pv[0].len() / 2;
                    if let Some(ref mut agcs) = self.layer_agcs {
                        agcs[block_idx].process(&mut pv, nb);
                    } else {
                        let mut agc = crate::ffn_backend::AGC.get().unwrap().lock().unwrap();
                        agc.process(&mut pv, nb);
                    }
                    candle_core::Tensor::new(pv, precond.device())?
                };

                // ── ODE with monitoring ──
                // Capture precond norms before ODE
                let precond_cpu = precond.to_vec2::<f32>()?;

                let ode_out = crate::gpu_ode::gpu_ode::kerr_ode_gpu(&precond, &block.gpu_ode_params)?;

                // Capture ode_out norms after ODE
                let ode_out_cpu = ode_out.to_vec2::<f32>()?;

                // ODE dynamics: use first position
                if !precond_cpu.is_empty() && !ode_out_cpu.is_empty() {
                    let pre = &precond_cpu[0];
                    let out = &ode_out_cpu[0];
                    let nb = n_bands.min(pre.len() / 2).min(out.len() / 2);

                    let mut phase_vel_sum = 0.0f32;
                    let mut energy_in = 0.0f32;
                    let mut energy_out = 0.0f32;
                    let mut band_energies = Vec::with_capacity(nb);

                    for k in 0..nb {
                        let r_in = pre[2 * k];
                        let s_in = pre[2 * k + 1];
                        let r_out = out[2 * k];
                        let s_out = out[2 * k + 1];

                        let phase_in = s_in.atan2(r_in);
                        let phase_out = s_out.atan2(r_out);
                        let mut d_phase = phase_out - phase_in;
                        if d_phase > std::f32::consts::PI { d_phase -= 2.0 * std::f32::consts::PI; }
                        if d_phase < -std::f32::consts::PI { d_phase += 2.0 * std::f32::consts::PI; }
                        phase_vel_sum += d_phase.abs();

                        energy_in += r_in * r_in + s_in * s_in;
                        let e_out = r_out * r_out + s_out * s_out;
                        energy_out += e_out;
                        band_energies.push(e_out);
                    }

                    let phase_velocity = if nb > 0 { phase_vel_sum / nb as f32 } else { 0.0 };
                    let energy_ratio = if energy_in > 1e-12 { energy_out / energy_in } else { 1.0 };
                    let damping = 1.0 - energy_ratio;
                    let band_energy_std = if nb > 1 {
                        let mean_e = energy_out / nb as f32;
                        let var: f32 = band_energies.iter().map(|&e| (e - mean_e) * (e - mean_e)).sum::<f32>() / nb as f32;
                        var.sqrt()
                    } else { 0.0 };

                    monitor.ode_dynamics.push(CandleOdeDynamics {
                        layer: block_idx, energy_in, energy_out, energy_ratio,
                        phase_velocity, damping, band_energy_std,
                    });
                }

                // Corrector plate
                let effective_ode_out = {
                    let n_b = self.n_bands;
                    let reshaped = ode_out.reshape((n_pos, n_b, 2))?;
                    let r = reshaped.narrow(2, 0, 1)?.squeeze(2)?;
                    let s = reshaped.narrow(2, 1, 1)?.squeeze(2)?;
                    let cos_c = block.phase_correction.cos()?;
                    let sin_c = block.phase_correction.sin()?;
                    let r_rot = (r.broadcast_mul(&cos_c)? - s.broadcast_mul(&sin_c)?)?;
                    let s_rot = (r.broadcast_mul(&sin_c)? + s.broadcast_mul(&cos_c)?)?;
                    let r_exp = r_rot.unsqueeze(2)?;
                    let s_exp = s_rot.unsqueeze(2)?;
                    Tensor::cat(&[&r_exp, &s_exp], 2)?.reshape((n_pos, n_embd))?
                };

                let mae_out = block.mae_out_sq.forward(&effective_ode_out)?;
                let mae_out = mae_out.gelu()?;
                let mae_out = block.mae_out_pr.forward(&mae_out)?;
                let regulated = (&effective_ode_out + &mae_out)?;
                let ffn_out = block.out_proj.forward(&regulated)?;

                // ── Layer flow monitoring ──
                // Use last position norms on CPU
                let hidden_cpu = hidden.to_vec2::<f32>()?;
                let attn_out_cpu = attn_out.to_vec2::<f32>()?;
                let ffn_out_cpu = ffn_out.to_vec2::<f32>()?;
                let last = n_pos.saturating_sub(1);

                let l2 = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };
                let dot_f = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f32>() };

                let in_vec = &hidden_cpu[last];
                let attn_vec = &attn_out_cpu[last];
                let ffn_vec = &ffn_out_cpu[last];

                let input_norm = l2(in_vec);
                let attn_out_norm = l2(attn_vec);
                let ffn_out_norm = l2(ffn_vec);

                // Reconstruct output: input + attn + ffn (before layer_scale)
                let out_vec: Vec<f32> = (0..n_embd).map(|j| in_vec[j] + attn_vec[j] + ffn_vec[j]).collect();
                let output_norm = l2(&out_vec);
                let inv_out = if output_norm > 1e-12 { 1.0 / output_norm } else { 0.0 };
                let cosine_in_out = if input_norm > 1e-12 && output_norm > 1e-12 {
                    dot_f(in_vec, &out_vec) / (input_norm * output_norm)
                } else { 0.0 };

                monitor.layer_flow.push(CandleLayerFlow {
                    layer: block_idx,
                    input_norm,
                    attn_out_norm,
                    ffn_out_norm,
                    output_norm,
                    attn_ratio: attn_out_norm * inv_out,
                    ffn_ratio: ffn_out_norm * inv_out,
                    residual_ratio: input_norm * inv_out,
                    cosine_in_out,
                });

                // Residual connection
                let contribution = (&attn_out + &ffn_out)?;
                hidden = if let Some(ref scale) = block.layer_scale {
                    (&hidden + contribution.broadcast_mul(scale)?)?
                } else {
                    (&hidden + &contribution)?
                };
            }

            // Final LN + logits (same as forward_with_curriculum)
            let normed = layer_norm(&hidden, &self.ln_f_w, &self.ln_f_b)?;
            let logits = if self.phase_native {
                if let Some(ref oc) = self.output_corrector {
                    let n_b = self.n_bands;
                    let reshaped = normed.reshape((n_pos, n_b, 2))?;
                    let r = reshaped.narrow(2, 0, 1)?.squeeze(2)?;
                    let s = reshaped.narrow(2, 1, 1)?.squeeze(2)?;
                    let cos_c = oc.cos()?;
                    let sin_c = oc.sin()?;
                    let r_rot = (r.broadcast_mul(&cos_c)? - s.broadcast_mul(&sin_c)?)?;
                    let s_rot = (r.broadcast_mul(&sin_c)? + s.broadcast_mul(&cos_c)?)?;
                    let r_exp = r_rot.unsqueeze(2)?;
                    let s_exp = s_rot.unsqueeze(2)?;
                    let corrected = Tensor::cat(&[&r_exp, &s_exp], 2)?.reshape((n_pos, self.n_embd))?;
                    corrected.matmul(&self.wte.t()?)?
                } else {
                    normed.matmul(&self.wte.t()?)?
                }
            } else {
                normed.matmul(&self.lm_head.t()?)?
            };

            Ok((logits, monitor))
        }
    }

    // ─── Training loop ───

    pub fn train_candle(
        data_path: &str, n_iters: usize,
        n_bands: usize, n_head: usize, n_layers: usize,
        maestro_dim: usize, _rk4_steps: usize, out_proj_groups: usize,
        debug_nan: bool, alpha: f32, beta: f32, phase_native: bool,
    ) -> Result<()> {
        // Runtime config — lowercase variables used throughout
        let n_embd = n_bands * 2;
        let block_size = 256usize; // positional table size

        println!("Candle backend — wave-engine\n");
        println!("  Config: {n_bands} bands, {n_head} heads, {n_layers} layers, {maestro_dim} maestro, {out_proj_groups} out_proj groups");

        // Device
        let device = Device::cuda_if_available(0)?;
        println!("  Device: {:?}", device);

        // Load data + tokenize (with token cache — 3min encode → instant reload)
        let use_bpe = std::env::args().any(|a| a == "--bpe");
        let tokenizer_path = std::env::args().skip_while(|a| a != "--tokenizer").nth(1)
            .unwrap_or("data/tokenizer.json".to_string());

        let tok_path_opt = if use_bpe { Some(tokenizer_path.as_str()) } else { None };
        let (tokens, vocab_size) = crate::common::data_loader::load_data(data_path, use_bpe, tok_path_opt);
        let split = (tokens.len() as f32 * 0.9) as usize;
        let train_data = &tokens[..split];
        println!("  Train tokens: {}", train_data.len());

        // Parse dynamic param flags early (needed for model construction + optimizer config)
        let use_rk4_dyn = std::env::args().any(|a| a == "--rk4-weights") &&
            std::env::args().skip_while(|a| a != "--rk4-weights").nth(1).map_or(false, |s| s == "dyn");
        let use_layer_scale_dyn = std::env::args().any(|a| a == "--layer-scale") &&
            std::env::args().skip_while(|a| a != "--layer-scale").nth(1).map_or(false, |s| s == "dyn");
        let use_wd_dyn = std::env::args().any(|a| a == "--wd") &&
            std::env::args().skip_while(|a| a != "--wd").nth(1).map_or(false, |s| s == "dyn");
        let use_agc_headroom_dyn = std::env::args().any(|a| a == "--agc-headroom") &&
            std::env::args().skip_while(|a| a != "--agc-headroom").nth(1).map_or(false, |s| s == "dyn");

        // Model
        let mut varmap = VarMap::new();
        let mut model = CandleWaveModel::new(&varmap, vocab_size, &device,
            n_bands, n_head, n_layers, maestro_dim, _rk4_steps, out_proj_groups, alpha, beta)?;
        model.debug_nan = debug_nan;
        model.phase_native = phase_native;
        // Wire dynamic params
        if use_rk4_dyn {
            for i in 0..model.blocks.len() {
                let key = format!("block.{i}.ode");
                model.blocks[i].gpu_ode_params.set_rk4_learnable(&varmap, &key, &device)?;
            }
        }
        if use_layer_scale_dyn {
            for i in 0..model.blocks.len() {
                let key = format!("block.{i}.layer_scale");
                // Init at 1.0 (no scaling = default behavior)
                let _t = varmap.get((1,), &key, candle_nn::Init::Const(1.0), DType::F32, &device)?;
                // Re-get for the model to use in forward
                let t = varmap.get((1,), &key, candle_nn::Init::Const(1.0), DType::F32, &device)?;
                model.blocks[i].layer_scale = Some(t);
            }
        }
        if phase_native {
            // Create output corrector in VarMap
            let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
            model.output_corrector = Some(vs.get_with_hints(
                (1, n_bands), "output_corrector", candle_nn::Init::Const(0.0),
            )?);
            println!("  Phase-native: dot product against embeddings (zero decoder params)");
        }
        if debug_nan { println!("  [debug-nan] Per-layer NaN detection ENABLED (~6x slower)"); }
        let n_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
        println!("  Trainable params: {n_params}");
        println!("  Architecture: {n_layers} layers, {n_head} heads, {n_bands} bands");

        // Resume from checkpoint if --resume flag
        let resume_path: Option<String> = std::env::args().skip_while(|a| a != "--resume").nth(1);
        let mut start_iter = 0usize;
        if let Some(ref ckpt) = resume_path {
            println!("  Resuming from: {ckpt}");
            if ckpt.ends_with(".safetensors") {
                // Native candle checkpoint
                varmap.load(ckpt)?;
            } else if ckpt.ends_with(".bin") {
                // CPU/wgpu WCHK checkpoint — load and populate VarMap
                let (params, _ck_vocab, ck_iter, _lr, _rng, _at, _am, _av, _groups, ck_flags) =
                    crate::wave_checkpoint::load_checkpoint(ckpt);
                start_iter = ck_iter;
                // Map flat params into VarMap keys (reverse of extract_wchk_params)
                let has_ode = ck_flags & 1 != 0 || ck_flags == 0; // v2 checkpoints have no flags
                let has_ls = ck_flags & 2 != 0;
                let has_rk4 = ck_flags & 4 != 0;
                load_wchk_params_into_varmap(&varmap, &params, n_layers, n_embd, maestro_dim,
                    vocab_size, out_proj_groups, n_bands, has_ode, has_ls, has_rk4, phase_native, &device)?;
                println!("  Loaded {} WCHK params (flags=0x{:02x}) into candle VarMap", params.len(), ck_flags);
            } else {
                eprintln!("  WARNING: unknown checkpoint format: {ckpt}");
            }
            // Read iter from .meta file (try exact match, then strip loss suffix)
            let meta_path = ckpt.replace(".safetensors", ".meta");
            let meta_content = std::fs::read_to_string(&meta_path)
                .or_else(|_| {
                    // Try stripping _lossN.NN from filename
                    let stripped = meta_path.split("_loss").next().unwrap_or(&meta_path);
                    std::fs::read_to_string(format!("{stripped}.meta"))
                })
                .or_else(|_| std::fs::read_to_string("candle_checkpoint_latest.meta"));
            if let Ok(meta) = meta_content {
                for line in meta.lines() {
                    if let Some(v) = line.strip_prefix("iter=") {
                        start_iter = v.parse().unwrap_or(0);
                    }
                }
            }
            println!("  Resumed at iter {start_iter}");
        }

        // CLI flag parsing for Candle path
        fn parse_flag_c<T: std::str::FromStr>(name: &str, default: T) -> T {
            std::env::args().skip_while(|a| a != name).nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let batch_size: usize = parse_flag_c("--batch", 4);
        let seq_len: usize = parse_flag_c("--seq", 256);
        let lr: f64 = parse_flag_c("--lr", if n_bands > 256 { 1e-4 } else { 3e-4 });
        let spring_k: f64 = parse_flag_c("--spring", 0.1);
        let use_rk4_dyn = std::env::args().any(|a| a == "--rk4-weights") &&
            std::env::args().skip_while(|a| a != "--rk4-weights").nth(1).map_or(false, |s| s == "dyn");
        let use_harmonics_dyn = std::env::args().any(|a| a == "--harmonics") &&
            std::env::args().skip_while(|a| a != "--harmonics").nth(1).map_or(false, |s| s == "dyn");
        let use_wd_dyn = std::env::args().any(|a| a == "--wd") &&
            std::env::args().skip_while(|a| a != "--wd").nth(1).map_or(false, |s| s == "dyn");
        let use_layer_scale_dyn = std::env::args().any(|a| a == "--layer-scale") &&
            std::env::args().skip_while(|a| a != "--layer-scale").nth(1).map_or(false, |s| s == "dyn");
        // Wire harmonic_dyn flag on blocks
        if use_harmonics_dyn {
            for block in &mut model.blocks {
                block.harmonic_dyn = true;
            }
        }
        if spring_k > 0.0 {
            let mut dyn_flags = Vec::new();
            if use_rk4_dyn { dyn_flags.push("rk4-weights"); }
            if use_harmonics_dyn { dyn_flags.push("harmonics"); }
            if use_wd_dyn { dyn_flags.push("wd"); }
            if use_layer_scale_dyn { dyn_flags.push("layer-scale"); }
            if use_agc_headroom_dyn { dyn_flags.push("agc-headroom"); }
            if !dyn_flags.is_empty() {
                println!("  Dynamic params: {} (spring k={:.2})", dyn_flags.join(", "), spring_k);
            }
        }

        // Optimizer — when WD is dynamic, disable built-in WD (we apply per-group manually)
        use candle_nn::Optimizer;
        let wd_builtin = if use_wd_dyn { 0.0 } else { 0.01 };
        let mut optimizer = candle_nn::AdamW::new(
            varmap.all_vars(),
            candle_nn::ParamsAdamW { lr, weight_decay: wd_builtin, ..Default::default() },
        )?;
        // Per-group WD scale: [n_layers + 1] (layers + lm_head), init at 1.0 (uniform)
        let mut wd_scale: Vec<f32> = vec![1.0; n_layers + 1];
        // Per-layer AGC headroom: init at 3.0 (3-sigma default)
        let mut agc_headroom: Vec<f32> = vec![3.0; n_layers];
        // Per-layer AGC instances when --agc-headroom dyn (stored on model)
        if use_agc_headroom_dyn {
            let ceiling = (std::f32::consts::FRAC_PI_2 / (alpha + 4.0 * beta)).sqrt().max(0.5);
            model.layer_agcs = Some((0..n_layers).map(|_| crate::common::agc::OdeAgc::with_ceiling_headroom(ceiling, 3.0)).collect());
        }
        let mut rng = crate::rng::Rng::new(1337);

        // Curriculum: soft-mask inactive bands (0.01 scale, not zero)
        let use_curriculum = !std::env::args().any(|a| a == "--no-curriculum");
        let curriculum = if use_curriculum {
            crate::train::CurriculumSchedule::default_4stage(n_bands)
        } else {
            crate::train::CurriculumSchedule::none(n_bands)
        };

        // ─── Pre-flight diagnostics (must match CPU tier) ───────────
        {
            // Check 1: Embedding separation (rebuild table to check)
            let pf_wte = crate::wave_embed::build_harmonic_table(vocab_size, n_bands);
            let self_dot: f32 = pf_wte[0].iter().map(|v| v * v).sum();
            let adj_dot: f32 = if pf_wte.len() > 1 {
                pf_wte[0].iter().zip(&pf_wte[1]).map(|(a, b)| a * b).sum()
            } else { self_dot };
            let separation = self_dot - adj_dot;
            if separation < 0.01 {
                eprintln!("  [preflight] WARNING: Embedding separation {:.6} — geometrically degenerate", separation);
            } else {
                println!("  [preflight] Embedding separation: {:.4} OK", separation);
            }

            // Check 2: Parameter balance
            let lm_head_params = vocab_size * n_embd;
            let total_params = n_params;
            let lm_pct = lm_head_params as f32 / total_params.max(1) as f32 * 100.0;
            if lm_pct > 95.0 {
                eprintln!("  [preflight] WARNING: lm_head is {:.1}% of params — ODE gets <{:.1}% gradient", lm_pct, 100.0 - lm_pct);
            } else {
                println!("  [preflight] Parameter balance: {:.1}% model, {:.1}% lm_head — OK", 100.0 - lm_pct, lm_pct);
            }

            // Check 3: ODE stability
            let alpha = if n_bands <= 128 { 0.01f32 } else { 0.1 };
            let degrees = (alpha + 4.0 * alpha) * 4.0 * 180.0 / std::f32::consts::PI;
            if degrees > 90.0 {
                eprintln!("  [preflight] WARNING: ODE phase shift {:.0}° at M=2.0", degrees);
            } else {
                println!("  [preflight] ODE stability: {:.0}° at M=2.0, alpha={:.4} — OK", degrees, alpha);
            }
        }

        // Initialize AGC with coupling-derived ceiling (matches CPU tier)
        crate::ffn_backend::init_agc(alpha, beta);
        let derived_ceiling = (std::f32::consts::FRAC_PI_2 / (alpha + 4.0 * beta)).sqrt();
        println!("  [preflight] AGC ceiling: {:.2} (derived from α={:.2})", derived_ceiling, alpha);

        let total_iters = start_iter + n_iters;
        println!("\nTraining for {n_iters} iters (batch={batch_size}, seq={seq_len}, lr={lr})");
        if start_iter > 0 { println!("  Resuming from iter {start_iter}, target {total_iters}"); }
        curriculum.describe(total_iters);
        println!("{:>6} {:>10} {:>10}", "Iter", "Loss", "Time");
        println!("{}", "-".repeat(35));

        // JSONL telemetry — tier-specific filename to prevent overwrites
        let log_name = "training_log_candle.jsonl";
        let log_file = std::fs::File::create(log_name).ok();
        let mut log_writer = log_file.map(|f| std::io::BufWriter::new(f));
        println!("  Telemetry: {log_name}");
        let mut nan_skip_count = 0usize;

        // Cosine LR schedule with warmup
        let warmup_iters = 100usize;
        let min_lr_ratio = 0.1;
        let cosine_lr = |iter: usize| -> f64 {
            if iter < warmup_iters {
                lr * (iter + 1) as f64 / warmup_iters as f64
            } else {
                let progress = (iter - warmup_iters) as f64 / (total_iters - warmup_iters).max(1) as f64;
                let decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                lr * (min_lr_ratio + (1.0 - min_lr_ratio) * decay)
            }
        };

        let train_start = Instant::now();
        let health_interval: usize = parse_flag_c("--health-interval", 0);

        for iter in start_iter..total_iters {
            let band_masks = curriculum.band_masks(iter, total_iters, n_bands);
            let iter_start = Instant::now();
            let mut total_loss = 0.0f32;
            let measure_monitors = health_interval > 0 && iter % health_interval == 0;

            let current_lr = cosine_lr(iter);
            optimizer.set_learning_rate(current_lr);

            // Monitor data: captured on last batch of health-interval iterations
            let mut fwd_monitor: Option<CandleMonitorData> = None;
            let mut output_dist: Option<CandleOutputDist> = None;
            let mut grad_flow: Option<Vec<CandleGradientFlow>> = None;

            for _b in 0..batch_size {
                let start = (rng.next_u64() as usize) % (train_data.len() - seq_len - 1);
                let input = &train_data[start..start + seq_len];
                let target = &train_data[start + 1..start + seq_len + 1];
                let is_monitor_batch = measure_monitors && _b == batch_size - 1;

                // Use monitor-instrumented forward on last batch of health intervals
                let (logits, monitor_opt) = if is_monitor_batch {
                    let (l, m) = model.forward_with_monitors(input, &band_masks)?;
                    (l, Some(m))
                } else {
                    (model.forward_with_curriculum(input, &band_masks)?, None)
                };

                // Output distribution monitor (from logits + targets, before loss)
                if is_monitor_batch {
                    output_dist = Some(compute_output_dist(&logits, target));
                }

                let target_tensor = Tensor::from_vec(
                    target.to_vec().iter().map(|&t| t as u32).collect::<Vec<u32>>(),
                    (seq_len,), &device,
                )?;
                let loss = candle_nn::loss::cross_entropy(&logits, &target_tensor)?;
                let loss_val = loss.to_scalar::<f32>()?;

                if loss_val.is_nan() || loss_val.is_infinite() {
                    nan_skip_count += 1;
                    eprintln!("  [NaN skip] iter {iter} batch {_b} (total skips: {nan_skip_count})");
                } else {
                    let grads = loss.backward()?;

                    // Gradient flow monitor (from grads, before optimizer step)
                    if is_monitor_batch {
                        grad_flow = Some(compute_gradient_flow(&grads, &varmap, n_layers));
                    }

                    // ── Harmonic backward (manual, outside autograd) ──
                    // Extract d_contribution from grad graph, compute d_harmonic_raw per head,
                    // apply gradient + spring to harmonic_raws, sync harmonic_ns.
                    if use_harmonics_dyn {
                        let eq_fn = |h: usize| -> f32 { ((h + 1) as f32 * 0.5f32).ln() };
                        let spring_k_harm = 2.0f32; // very stiff — integer harmonics theoretically motivated

                        for block in model.blocks.iter_mut() {
                            if !block.harmonic_dyn { continue; }

                            // Extract gradient of contribution tensor from GradStore
                            let d_out_cpu = if let Some(ref layer_out) = block.cached_layer_output {
                                grads.get(layer_out).map(|g| g.to_vec2::<f32>().ok()).flatten()
                            } else {
                                None
                            };

                            if let Some(d_out) = d_out_cpu {
                                let d_hr = harmonic_backward(block, &d_out, n_embd);

                                for h in 0..block.harmonic_ns.len() {
                                    // Gradient step
                                    block.harmonic_ns[h] -= (current_lr as f32) * d_hr[h];
                                    // Spring pull toward equilibrium
                                    let eq = eq_fn(h);
                                    block.harmonic_ns[h] -= (current_lr as f32) * spring_k_harm * (block.harmonic_ns[h] - eq);
                                }

                                // Sync: harmonic_ns = softplus(harmonic_raws)
                                // Since harmonic_ns stores the raw values (confusing name, but matches CPU tier),
                                // and softplus is applied at use-time in wave_attention, no sync needed here.
                                // The update above directly modifies the raw values.
                            }

                            // Clear caches to free memory
                            block.cached_att_weights = None;
                            block.cached_normed_cpu = None;
                            block.cached_layer_output = None;
                        }
                    }

                    // Gradient clipping via LR scaling.
                    let mut gnorm_sq = 0.0f64;
                    for var in &varmap.all_vars() {
                        if let Some(grad) = grads.get(var) {
                            let g: Vec<f32> = grad.flatten_all()?.to_vec1::<f32>()?;
                            for &v in &g { gnorm_sq += (v as f64) * (v as f64); }
                        }
                    }
                    let gnorm = gnorm_sq.sqrt();
                    if gnorm > 1.0 {
                        optimizer.set_learning_rate(current_lr / gnorm);
                        optimizer.step(&grads)?;
                        optimizer.set_learning_rate(current_lr);
                    } else {
                        optimizer.step(&grads)?;
                    }
                    drop(grads);
                    device.synchronize()?;
                }

                if !loss_val.is_nan() {
                    total_loss += loss_val;
                }

                // Stash forward monitor data
                if let Some(m) = monitor_opt {
                    fwd_monitor = Some(m);
                }
            }

            total_loss /= batch_size as f32;

            // Spring regulation on dynamic params (after optimizer step, like CPU tier)
            // param -= lr * k * (param - equilibrium)
            if spring_k > 0.0 {
                let clr = current_lr;
                let data = varmap.data().lock().unwrap();

                // ODE alpha/beta springs (free, k=0 — self-regulating via AGC)
                // No spring needed — matches CPU tier behavior

                // Corrector plate spring: very loose (k=0.01), eq=0.0 (transparent)
                if phase_native {
                    let k_corr = clr * spring_k * 0.01;
                    for layer in 0..n_layers {
                        let key = format!("block.{layer}.phase_correction");
                        if let Some(var) = data.get(&key) {
                            let vals: Vec<f32> = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                            let new_vals: Vec<f32> = vals.iter().map(|&v| v - (k_corr as f32) * v).collect();
                            let new_tensor = Tensor::from_vec(new_vals, var.shape(), var.device()).unwrap();
                            var.set(&new_tensor).unwrap();
                        }
                    }
                }

                // RK4 weights spring: eq=[1/6,1/3,1/3,1/6], k=2.0 (very stiff)
                if use_rk4_dyn {
                    let k_rk4 = clr * spring_k * 2.0;
                    let eq = [1.0f32/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0];
                    for layer in 0..n_layers {
                        let key = format!("block.{layer}.ode.rk4_weights");
                        if let Some(var) = data.get(&key) {
                            let vals: Vec<f32> = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                            let new_vals: Vec<f32> = vals.iter().enumerate()
                                .map(|(i, &v)| v - (k_rk4 as f32) * (v - eq[i]))
                                .collect();
                            let new_tensor = Tensor::from_vec(new_vals, var.shape(), var.device()).unwrap();
                            var.set(&new_tensor).unwrap();
                        }
                    }
                }

                // Harmonics: gradient + spring already applied in the backward section above
                // (after loss.backward(), before optimizer.step()).
                // The spring here is redundant — harmonic spring is applied per-batch in the backward block.

                // Per-group weight decay (when --wd dyn)
                if use_wd_dyn {
                    let base_wd = 0.01f32;
                    // Apply WD per layer group: param -= lr * base_wd * wd_scale * param
                    for layer in 0..n_layers {
                        let wd_eff = base_wd * wd_scale[layer];
                        let prefix = format!("block.{layer}.");
                        for (key, var) in data.iter() {
                            if key.starts_with(&prefix) {
                                let vals: Vec<f32> = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                                let new_vals: Vec<f32> = vals.iter()
                                    .map(|&v| v - (clr as f32) * wd_eff * v)
                                    .collect();
                                let new_t = Tensor::from_vec(new_vals, var.shape(), var.device()).unwrap();
                                var.set(&new_t).unwrap();
                            }
                        }
                    }
                    // lm_head group (last wd_scale entry)
                    let wd_head = base_wd * wd_scale[n_layers];
                    for (key, var) in data.iter() {
                        if key == "lm_head" || key == "output_corrector" || key.starts_with("ln_f") {
                            let vals: Vec<f32> = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                            let new_vals: Vec<f32> = vals.iter()
                                .map(|&v| v - (clr as f32) * wd_head * v)
                                .collect();
                            let new_t = Tensor::from_vec(new_vals, var.shape(), var.device()).unwrap();
                            var.set(&new_t).unwrap();
                        }
                    }
                }

                // WD spring: eq=1.0, k=1.0 (stiff — uniform regularisation well-motivated)
                if use_wd_dyn {
                    let k_wd = clr * spring_k * 1.0;
                    for s in &mut wd_scale {
                        *s -= (k_wd as f32) * (*s - 1.0);
                        *s = s.clamp(0.01, 10.0);
                    }
                }

                // AGC headroom spring: eq=3.0, k=1.0 (stiff — safety motivated)
                if use_agc_headroom_dyn {
                    let k_agc = clr * spring_k * 1.0;
                    for hr in &mut agc_headroom {
                        *hr -= (k_agc as f32) * (*hr - 3.0);
                        *hr = hr.clamp(1.0, 6.0);
                    }
                    // Update per-layer AGC instances with new headroom
                    if let Some(ref mut agcs) = model.layer_agcs {
                        let ceiling = (std::f32::consts::FRAC_PI_2 / (alpha + 4.0 * beta)).sqrt().max(0.5);
                        for (i, agc) in agcs.iter_mut().enumerate() {
                            *agc = crate::common::agc::OdeAgc::with_ceiling_headroom(ceiling, agc_headroom[i]);
                        }
                    }
                }

                // Layer scale spring: eq=1.0, k=1.0 (moderate)
                if use_layer_scale_dyn {
                    let k_ls = clr * spring_k * 1.0;
                    for layer in 0..n_layers {
                        let key = format!("block.{layer}.layer_scale");
                        if let Some(var) = data.get(&key) {
                            let vals: Vec<f32> = var.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                            let new_vals: Vec<f32> = vals.iter()
                                .map(|&v| (v - (k_ls as f32) * (v - 1.0)).max(0.0)) // soft floor at 0
                                .collect();
                            let new_tensor = Tensor::from_vec(new_vals, var.shape(), var.device()).unwrap();
                            var.set(&new_tensor).unwrap();
                        }
                    }
                }

                drop(data);
            }
            let iter_time = iter_start.elapsed();

            // VRAM monitoring via cudarc (direct CUDA query — shows ALL GPU memory)
            let vram_used_mb = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                .map(|(free, total)| (total - free) / (1024 * 1024))
                .unwrap_or(0);

            // JSONL telemetry — with AGC diagnostics every 100 iters + monitors at health interval
            if let Some(ref mut writer) = log_writer {
                use std::io::Write;
                if iter % 100 == 0 {
                    // AGC + ODE stats
                    let clamp_count = crate::ffn_backend::ODE_CLAMP_COUNT.load(std::sync::atomic::Ordering::Relaxed);
                    let max_mag = f32::from_bits(crate::ffn_backend::ODE_MAX_MAG.load(std::sync::atomic::Ordering::Relaxed));
                    let agc = crate::ffn_backend::agc_stats();

                    // ODE coupling values from VarMap
                    let ode_str = {
                        let data = varmap.data().lock().unwrap();
                        let mut parts = Vec::new();
                        for l in 0..n_layers {
                            let a = data.get(&format!("block.{l}.ode.alpha")).map(|v| v.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]).unwrap_or(alpha);
                            let b = data.get(&format!("block.{l}.ode.beta")).map(|v| v.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]).unwrap_or(beta);
                            let g = data.get(&format!("block.{l}.ode.gamma_raw")).map(|v| {
                                let vals = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                                let sp = |x: f32| -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } };
                                vals.iter().map(|&x| sp(x)).sum::<f32>() / vals.len() as f32
                            }).unwrap_or(0.1);
                            parts.push(format!(r#"{{"a":{:.4},"b":{:.4},"g":{:.4}}}"#, a, b, g));
                        }
                        format!(r#","ode_params":[{}]"#, parts.join(","))
                    };

                    // Dynamic param values
                    let dyn_str = {
                        let mut s = String::new();
                        if use_layer_scale_dyn {
                            let data = varmap.data().lock().unwrap();
                            let vals: Vec<String> = (0..n_layers).map(|l| {
                                data.get(&format!("block.{l}.layer_scale")).map(|v| format!("{:.4}", v.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0])).unwrap_or("1.0000".to_string())
                            }).collect();
                            s += &format!(r#","layer_scale":[{}]"#, vals.join(","));
                        }
                        if use_rk4_dyn {
                            let data = varmap.data().lock().unwrap();
                            let mut parts = Vec::new();
                            for l in 0..n_layers {
                                if let Some(v) = data.get(&format!("block.{l}.ode.rk4_weights")) {
                                    let w = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                                    parts.push(format!(r#"{{"L{}": [{:.4},{:.4},{:.4},{:.4}]}}"#, l, w[0], w[1], w[2], w[3]));
                                }
                            }
                            if !parts.is_empty() { s += &format!(r#","rk4_weights":[{}]"#, parts.join(",")); }
                        }
                        if use_wd_dyn {
                            let vals: Vec<String> = wd_scale.iter().map(|v| format!("{:.4}", v)).collect();
                            s += &format!(r#","wd_scale":[{}]"#, vals.join(","));
                        }
                        if use_agc_headroom_dyn {
                            let vals: Vec<String> = agc_headroom.iter().map(|v| format!("{:.2}", v)).collect();
                            s += &format!(r#","agc_headroom":[{}]"#, vals.join(","));
                        }
                        if use_harmonics_dyn {
                            fn softplus_t(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }
                            let mut parts = Vec::new();
                            for (l, block) in model.blocks.iter().enumerate() {
                                let vals: Vec<String> = block.harmonic_ns.iter().map(|&h| format!("{:.4}", softplus_t(h))).collect();
                                parts.push(format!(r#"{{"L{}": [{}]}}"#, l, vals.join(",")));
                            }
                            s += &format!(r#","harmonics":[{}]"#, parts.join(","));
                        }
                        s
                    };

                    let _ = writeln!(writer,
                        "{{\"iter\":{},\"loss\":{:.4},\"lr\":{:.6},\"time_ms\":{},\"vram_mb\":{},\"nan_skips\":{},\"ode_clamps\":{},\"ode_max_mag\":{:.2},\"agc_threshold\":{:.3},\"agc_mean\":{:.3},\"agc_std\":{:.3}{}{}}}",
                        iter, total_loss, current_lr, iter_time.as_millis(), vram_used_mb, nan_skip_count,
                        clamp_count, max_mag, agc.threshold, agc.ema_mean, agc.ema_std, ode_str, dyn_str
                    );
                } else {
                    let _ = writeln!(writer,
                        "{{\"iter\":{},\"loss\":{:.4},\"lr\":{:.6},\"time_ms\":{},\"vram_mb\":{},\"nan_skips\":{}}}",
                        iter, total_loss, current_lr, iter_time.as_millis(), vram_used_mb, nan_skip_count
                    );
                }

                // Monitor suite at health intervals
                if measure_monitors {
                    // Throughput
                    let tok_s = (batch_size * seq_len) as f32 / iter_time.as_secs_f32().max(0.001);
                    let iter_s = 1.0 / iter_time.as_secs_f32().max(0.001);
                    let _ = writeln!(writer,
                        r#"{{"iter":{},"type":"monitor","throughput":{{"tok_s":{:.0},"iter_s":{:.1},"fwd_ms":{},"vram_mb":{}}}}}"#,
                        iter, tok_s, iter_s, iter_time.as_millis(), vram_used_mb
                    );

                    // Embedding space (static — same analysis as CPU)
                    let embed_stats = crate::common::embedding_monitor::analyze_embeddings(&crate::WavePacketModel {
                        wte: model.wte.to_vec2::<f32>().unwrap_or_default(),
                        wpe: vec![], blocks: vec![], ln_f: crate::model::LayerNormWeights { weight: vec![], bias: vec![] },
                        lm_head: vec![], lm_down: vec![], lm_up: vec![], lm_rank: 0, vocab_size,
                        tied_temperature: 1.0, wd_state: None, learnable_ode: false,
                        use_rk4_weights: false, use_dyn_harmonics: false, layer_scale: vec![], use_layer_scale: false,
                        lr_scale: vec![], use_lr_scale: false, wd_scale: vec![], agc_headroom: vec![],
                        phase_native: false, output_corrector: vec![],
                    });
                    let _ = writeln!(writer, r#"{{"iter":{},"type":"monitor",{}}}"#,
                        iter, crate::common::embedding_monitor::to_json(&embed_stats));

                    // Output distribution (#5)
                    if let Some(ref od) = output_dist {
                        let _ = writeln!(writer, r#"{{"iter":{},"type":"monitor",{}}}"#,
                            iter, output_dist_json(od));
                    }

                    // Layer flow (#2)
                    if let Some(ref fm) = fwd_monitor {
                        let lf_json = fm.layer_flow_json();
                        if !lf_json.is_empty() {
                            let _ = writeln!(writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, lf_json);
                        }

                        // Attention heads (#1)
                        let ah_json = fm.attn_heads_json();
                        if !ah_json.is_empty() {
                            let _ = writeln!(writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, ah_json);
                        }

                        // ODE dynamics (#6)
                        let od_json = fm.ode_dynamics_json();
                        if !od_json.is_empty() {
                            let _ = writeln!(writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, od_json);
                        }
                    }

                    // Gradient flow (#3)
                    if let Some(ref gf) = grad_flow {
                        let gf_json = gradient_flow_json(gf);
                        if !gf_json.is_empty() {
                            let _ = writeln!(writer, r#"{{"iter":{},"type":"monitor",{}}}"#, iter, gf_json);
                        }
                    }
                }

                let _ = writer.flush();
            }

            if iter % 50 == 0 || iter == total_iters - 1 {
                println!("{:>6} {:>10.4} {:>10.1?}  lr={:.6}  vram={}MB", iter, total_loss, iter_time, current_lr, vram_used_mb);
            }

            // Periodic checkpoint: save every 500 iters (leak is fixed, 100 was debug)
            if (iter + 1) % 500 == 0 || iter == total_iters - 1 {
                // NaN guard: never overwrite good checkpoints with corrupted weights
                if total_loss.is_nan() || total_loss == 0.0 || total_loss.is_infinite() {
                    eprintln!("  WARNING: loss={total_loss} — skipping checkpoint (corrupted)");
                } else {
                let st_path = format!("candle_checkpoint_iter{}_loss{:.2}.safetensors", iter + 1, total_loss);
                let meta = format!("iter={}\nloss={}\nlr={}\nvocab_size={}\n", iter + 1, total_loss, current_lr, vocab_size);
                if varmap.save(&st_path).is_ok() {
                    std::fs::write(format!("candle_checkpoint_iter{}_loss{:.2}.meta", iter + 1, total_loss), &meta).ok();
                    println!("  Checkpoint: {st_path}");
                }
                let _ = varmap.save("candle_checkpoint_latest.safetensors");
                std::fs::write("candle_checkpoint_latest.meta", &meta).ok();

                let params = extract_wchk_params(&varmap, &model, n_layers, n_embd, maestro_dim,
                    vocab_size, out_proj_groups, n_bands, phase_native,
                    use_rk4_dyn, use_layer_scale_dyn, use_harmonics_dyn);
                let dummy_adam = crate::train::Adam::new(lr as f32, params.len());
                let mut ck_dims = crate::Dims::from_cli(n_bands, n_head, maestro_dim, 256, _rk4_steps)
                    .with_learnable_ode(true).with_corrector(true)
                    .with_rk4_weights(use_rk4_dyn).with_layer_scale(use_layer_scale_dyn);
                ck_dims.use_dyn_harmonics = use_harmonics_dyn;
                crate::wave_checkpoint::save_checkpoint(
                    &params, vocab_size, n_layers, out_proj_groups, iter + 1, lr as f32,
                    &dummy_adam, 0, "checkpoint.bin", ck_dims,
                );
                } // end NaN guard else
            }
        }

        if nan_skip_count > 0 {
            println!("  Warning: {nan_skip_count} NaN steps skipped during training");
        }
        println!("\nTraining complete. Total: {:.1?}", train_start.elapsed());
        Ok(())
    }

    /// Extract params from VarMap in WCHK flatten_params order.
    /// Order: per block (ln_w, ln_b, ln_ffn_w, ln_ffn_b, mae_in_sq, mae_in_pr,
    ///   mae_out_sq, mae_out_pr, out_proj), then ln_f_w, ln_f_b, lm_head.
    /// Load WCHK flat params into candle VarMap.
    /// Reverse of extract_wchk_params — maps flat param vector to named variables.
    fn load_wchk_params_into_varmap(
        varmap: &VarMap, params: &[f32],
        n_layers: usize, n_embd: usize, maestro_dim: usize,
        vocab_size: usize, out_proj_groups: usize, n_bands: usize,
        has_ode: bool, has_ls: bool, has_rk4: bool, phase_native: bool,
        device: &Device,
    ) -> Result<()> {
        let mut idx = 0;
        let set_var = |varmap: &VarMap, key: &str, vals: &[f32], shape: &[usize], device: &Device| -> Result<()> {
            let data = varmap.data().lock().unwrap();
            if let Some(var) = data.get(key) {
                let t = Tensor::from_slice(vals, shape, device)?;
                var.set(&t)?;
            }
            Ok(())
        };

        for i in 0..n_layers {
            let p = format!("block.{i}");
            // LN
            set_var(varmap, &format!("{p}.ln_w"), &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
            set_var(varmap, &format!("{p}.ln_b"), &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
            // LN FFN (skip — candle uses shared LN)
            idx += n_embd * 2;
            // Maestro in squeeze
            set_var(varmap, &format!("{p}.mae_in_sq.weight"), &params[idx..idx+maestro_dim*n_embd], &[maestro_dim, n_embd], device)?; idx += maestro_dim * n_embd;
            set_var(varmap, &format!("{p}.mae_in_sq.bias"), &params[idx..idx+maestro_dim], &[maestro_dim], device)?; idx += maestro_dim;
            // Maestro in process
            set_var(varmap, &format!("{p}.mae_in_pr.weight"), &params[idx..idx+n_embd*maestro_dim], &[n_embd, maestro_dim], device)?; idx += n_embd * maestro_dim;
            set_var(varmap, &format!("{p}.mae_in_pr.bias"), &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
            // Maestro out squeeze
            set_var(varmap, &format!("{p}.mae_out_sq.weight"), &params[idx..idx+maestro_dim*n_embd], &[maestro_dim, n_embd], device)?; idx += maestro_dim * n_embd;
            set_var(varmap, &format!("{p}.mae_out_sq.bias"), &params[idx..idx+maestro_dim], &[maestro_dim], device)?; idx += maestro_dim;
            // Maestro out process
            set_var(varmap, &format!("{p}.mae_out_pr.weight"), &params[idx..idx+n_embd*maestro_dim], &[n_embd, maestro_dim], device)?; idx += n_embd * maestro_dim;
            set_var(varmap, &format!("{p}.mae_out_pr.bias"), &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
            // Out proj
            if out_proj_groups <= 1 {
                set_var(varmap, &format!("{p}.out_proj.weight"), &params[idx..idx+n_embd*n_embd], &[n_embd, n_embd], device)?; idx += n_embd * n_embd;
                set_var(varmap, &format!("{p}.out_proj.bias"), &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
            } else {
                let gs = n_embd / out_proj_groups;
                for g in 0..out_proj_groups {
                    set_var(varmap, &format!("{p}.out_proj.g{g}.weight"), &params[idx..idx+gs*gs], &[gs, gs], device)?; idx += gs * gs;
                    set_var(varmap, &format!("{p}.out_proj.g{g}.bias"), &params[idx..idx+gs], &[gs], device)?; idx += gs;
                }
            }
            // ODE params
            if has_ode {
                set_var(varmap, &format!("{p}.ode.gamma_raw"), &params[idx..idx+n_bands], &[1, n_bands], device)?; idx += n_bands;
                set_var(varmap, &format!("{p}.ode.alpha"), &params[idx..idx+1], &[1, 1], device)?; idx += 1;
                set_var(varmap, &format!("{p}.ode.beta"), &params[idx..idx+1], &[1, 1], device)?; idx += 1;
                set_var(varmap, &format!("{p}.phase_correction"), &params[idx..idx+n_bands], &[1, n_bands], device)?; idx += n_bands;
                if has_rk4 {
                    set_var(varmap, &format!("{p}.ode.rk4_weights"), &params[idx..idx+4], &[4], device)?; idx += 4;
                }
            }
        }
        // Layer scale
        if has_ls {
            for i in 0..n_layers {
                set_var(varmap, &format!("block.{i}.layer_scale"), &params[idx..idx+1], &[1], device)?; idx += 1;
            }
        }
        // ln_f
        set_var(varmap, "ln_f_w", &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
        set_var(varmap, "ln_f_b", &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
        // Phase-native: output corrector. Standard: lm_head.
        if phase_native {
            set_var(varmap, "output_corrector", &params[idx..idx+n_bands], &[1, n_bands], device)?; idx += n_bands;
        } else {
            set_var(varmap, "lm_head", &params[idx..idx+vocab_size*n_embd], &[vocab_size, n_embd], device)?; idx += vocab_size * n_embd;
        }

        if idx != params.len() {
            eprintln!("  WARNING: WCHK param count mismatch: read {} of {}", idx, params.len());
        }
        Ok(())
    }

    fn extract_wchk_params(varmap: &VarMap, model: &CandleWaveModel, n_layers: usize, n_embd: usize, maestro_dim: usize,
                            vocab_size: usize, out_proj_groups: usize, n_bands: usize,
                            phase_native: bool, use_rk4_dyn: bool, use_layer_scale: bool,
                            use_harmonics: bool) -> Vec<f32> {
        let mut params = Vec::new();

        let get_flat = |name: &str| -> Vec<f32> {
            let data = varmap.data().lock().unwrap();
            data.get(name).map(|t| t.flatten_all().unwrap().to_vec1::<f32>().unwrap()).unwrap_or_default()
        };

        for i in 0..n_layers {
            let p = format!("block.{i}");
            // LN weights
            params.extend(get_flat(&format!("{p}.ln_w")));
            params.extend(get_flat(&format!("{p}.ln_b")));
            // LN FFN — placeholder (candle uses shared LN)
            params.extend(vec![1.0f32; n_embd]);
            params.extend(vec![0.0f32; n_embd]);
            // Maestro in
            params.extend(get_flat(&format!("{p}.mae_in_sq.weight")));
            params.extend(get_flat(&format!("{p}.mae_in_sq.bias")));
            params.extend(get_flat(&format!("{p}.mae_in_pr.weight")));
            params.extend(get_flat(&format!("{p}.mae_in_pr.bias")));
            // Maestro out
            params.extend(get_flat(&format!("{p}.mae_out_sq.weight")));
            params.extend(get_flat(&format!("{p}.mae_out_sq.bias")));
            params.extend(get_flat(&format!("{p}.mae_out_pr.weight")));
            params.extend(get_flat(&format!("{p}.mae_out_pr.bias")));
            // Out proj — dense (groups=1) or block-diagonal (groups>1)
            if out_proj_groups <= 1 {
                params.extend(get_flat(&format!("{p}.out_proj.weight")));
                params.extend(get_flat(&format!("{p}.out_proj.bias")));
            } else {
                for g in 0..out_proj_groups {
                    params.extend(get_flat(&format!("{p}.out_proj.g{g}.weight")));
                    params.extend(get_flat(&format!("{p}.out_proj.g{g}.bias")));
                }
            }
            // ODE params (learnable)
            let gamma = get_flat(&format!("{p}.ode.gamma_raw"));
            if !gamma.is_empty() {
                params.extend(&gamma);
                params.extend(get_flat(&format!("{p}.ode.alpha")));
                params.extend(get_flat(&format!("{p}.ode.beta")));
                params.extend(get_flat(&format!("{p}.phase_correction")));
                if use_rk4_dyn {
                    params.extend(get_flat(&format!("{p}.ode.rk4_weights")));
                }
            }
            // Harmonics (if dynamic) — stored on CandleBlock, not in VarMap
            if use_harmonics {
                params.extend_from_slice(&model.blocks[i].harmonic_ns);
            }
        }
        // Layer scale
        if use_layer_scale {
            for i in 0..n_layers {
                let ls = get_flat(&format!("block.{i}.layer_scale"));
                if !ls.is_empty() { params.extend(&ls); } else { params.push(1.0); }
            }
        }
        // ln_f
        params.extend(get_flat("ln_f_w"));
        params.extend(get_flat("ln_f_b"));
        // Phase-native: output corrector. Standard: lm_head.
        if phase_native {
            let oc = get_flat("output_corrector");
            if !oc.is_empty() {
                params.extend(&oc);
            } else {
                params.extend(vec![0.0f32; n_bands]);
            }
        } else {
            params.extend(get_flat("lm_head"));
        }

        params
    }
}

// Stub when candle feature is not enabled
#[cfg(not(feature = "candle-backend"))]
pub mod engine {
    pub fn train_candle(_data_path: &str, _n_iters: usize, _n_bands: usize, _n_head: usize, _n_layers: usize, _maestro_dim: usize, _rk4_steps: usize, _out_proj_groups: usize, _debug_nan: bool, _alpha: f32, _beta: f32, _phase_native: bool) -> std::result::Result<(), String> {
        Err("Candle backend not enabled. Build with: cargo run --release --features candle-backend".to_string())
    }
}
