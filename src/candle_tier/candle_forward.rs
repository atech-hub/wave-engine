//! Forward pass methods on CandleWaveModel.

#[cfg(feature = "candle-backend")]
pub mod forward {
    use candle_core::{Module, Result, Tensor};

    use crate::candle_tier::candle_model::model::{CandleWaveModel, layer_norm};
    use crate::candle_tier::candle_attention::attention::wave_attention;
    use crate::candle_tier::candle_monitors::monitors::{
        CandleMonitorData, CandleLayerFlow, CandleAttnHead, CandleOdeDynamics,
    };

    impl CandleWaveModel {
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

                // AGC knee compression — differentiable (preserves autograd chain)
                // Extract magnitude for EMA update (detached from graph)
                // but apply clamping through tensor ops (on the graph)
                let precond = {
                    let n_b = self.n_bands;
                    let n_e = self.n_embd;

                    // Update AGC EMA state from detached magnitudes (no grad needed for EMA)
                    let pv_detach: Vec<Vec<f32>> = precond.detach().to_vec2()?;
                    let mags: Vec<f32> = pv_detach.iter().flat_map(|pos| {
                        (0..n_b).map(move |k| (pos[k*2]*pos[k*2] + pos[k*2+1]*pos[k*2+1]).sqrt())
                    }).collect();
                    let threshold = if let Some(ref mut agcs) = self.layer_agcs {
                        agcs[block_idx].observe(&mags);
                        agcs[block_idx].stats().threshold
                    } else {
                        let mut agc = crate::ffn_backend::AGC.get().unwrap().lock().unwrap();
                        agc.observe(&mags);
                        agc.stats().threshold
                    };

                    // Apply clamping through differentiable tensor ops (on the autograd graph)
                    let reshaped = precond.reshape((n_pos, n_b, 2))?;
                    let r = reshaped.narrow(2, 0, 1)?.squeeze(2)?;
                    let s = reshaped.narrow(2, 1, 1)?.squeeze(2)?;
                    let mag_sq = (&r * &r)?.add(&(&s * &s)?)?;
                    let mag = (mag_sq + 1e-12 as f64)?.sqrt()?;
                    // scale = min(1.0, threshold / mag) — knee compression as differentiable min
                    let thresh_tensor = (mag.zeros_like()? + threshold as f64)?;
                    let raw_scale = (thresh_tensor / &mag)?;
                    let ones = raw_scale.ones_like()?;
                    let scale = raw_scale.minimum(&ones)?;
                    // Apply scale to r, s (gradient flows through scale computation)
                    let r_scaled = (r * &scale)?;
                    let s_scaled = (s * &scale)?;
                    let r_exp = r_scaled.unsqueeze(2)?;
                    let s_exp = s_scaled.unsqueeze(2)?;
                    Tensor::cat(&[&r_exp, &s_exp], 2)?.reshape((n_pos, n_e))?
                };

                if self.debug_nan {
                    let precond_vals = precond.to_vec2::<f32>()?;
                    let precond_max = precond_vals.iter()
                        .flat_map(|r| r.iter()).cloned().fold(0.0f32, |a, b| a.max(b.abs()));
                    if precond_max > 10.0 || precond_max.is_nan() || precond_max.is_infinite() {
                        eprintln!("  [PRECOND] block {block_idx} max={precond_max:.2}");
                    }
                }

                // ODE — CustomOp (no autograd graph) or autograd (full graph)
                let ode_out = if self.use_custom_op {
                    let gamma_raw = block.gpu_ode_params.gamma_raw.flatten_all()?.to_vec1::<f32>()?;
                    let omega = block.gpu_ode_params.omega.flatten_all()?.to_vec1::<f32>()?;
                    let alpha_v = block.gpu_ode_params.alpha.flatten_all()?.to_vec1::<f32>()?[0];
                    let beta_v = block.gpu_ode_params.beta.flatten_all()?.to_vec1::<f32>()?[0];
                    let rk4_w = if let Some(ref w) = block.gpu_ode_params.rk4_w {
                        let v = w.to_vec1::<f32>()?;
                        [v[0], v[1], v[2], v[3]]
                    } else {
                        [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]
                    };
                    let param_grads = self.ode_param_grads.as_ref()
                        .expect("CustomOp requires ode_param_grads on model").clone();
                    let op = crate::candle_tier::custom_ode::custom_ode::KerrOdeCustomOp::new(
                        gamma_raw, omega, alpha_v, beta_v, rk4_w,
                        block.gpu_ode_params.rk4_steps, block.gpu_ode_params.n_bands, block_idx,
                        param_grads,
                    );
                    // CustomOp runs on CPU — move tensor CPU→op→GPU
                    let precond_cpu = precond.to_device(&candle_core::Device::Cpu)?;
                    let ode_cpu = precond_cpu.apply_op1(op)?;
                    ode_cpu.to_device(&self.device)?
                } else {
                    crate::gpu_ode::gpu_ode::kerr_ode_gpu(&precond, &block.gpu_ode_params)?
                };

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
        pub fn forward_with_monitors(
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
                    let harmonic_n = crate::common::math::softplus(block.harmonic_ns[head]);

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
                // AGC — differentiable (same as main forward)
                let precond = {
                    let n_b = self.n_bands;
                    let n_e = self.n_embd;
                    let pv_detach: Vec<Vec<f32>> = precond.detach().to_vec2()?;
                    let mags: Vec<f32> = pv_detach.iter().flat_map(|pos| {
                        (0..n_b).map(move |k| (pos[k*2]*pos[k*2] + pos[k*2+1]*pos[k*2+1]).sqrt())
                    }).collect();
                    let threshold = if let Some(ref mut agcs) = self.layer_agcs {
                        agcs[block_idx].observe(&mags);
                        agcs[block_idx].stats().threshold
                    } else {
                        let mut agc = crate::ffn_backend::AGC.get().unwrap().lock().unwrap();
                        agc.observe(&mags);
                        agc.stats().threshold
                    };
                    let reshaped = precond.reshape((n_pos, n_b, 2))?;
                    let r = reshaped.narrow(2, 0, 1)?.squeeze(2)?;
                    let s = reshaped.narrow(2, 1, 1)?.squeeze(2)?;
                    let mag_sq = (&r * &r)?.add(&(&s * &s)?)?;
                    let mag = (mag_sq + 1e-12 as f64)?.sqrt()?;
                    let thresh_tensor = (mag.zeros_like()? + threshold as f64)?;
                    let raw_scale = (thresh_tensor / &mag)?;
                    let ones = raw_scale.ones_like()?;
                    let scale = raw_scale.minimum(&ones)?;
                    let r_scaled = (r * &scale)?;
                    let s_scaled = (s * &scale)?;
                    let r_exp = r_scaled.unsqueeze(2)?;
                    let s_exp = s_scaled.unsqueeze(2)?;
                    Tensor::cat(&[&r_exp, &s_exp], 2)?.reshape((n_pos, n_e))?
                };

                // ── ODE with monitoring ──
                // Capture precond norms before ODE
                let precond_cpu = precond.to_vec2::<f32>()?;

                let ode_out = if self.use_custom_op {
                    let gamma_raw = block.gpu_ode_params.gamma_raw.flatten_all()?.to_vec1::<f32>()?;
                    let omega = block.gpu_ode_params.omega.flatten_all()?.to_vec1::<f32>()?;
                    let alpha_v = block.gpu_ode_params.alpha.flatten_all()?.to_vec1::<f32>()?[0];
                    let beta_v = block.gpu_ode_params.beta.flatten_all()?.to_vec1::<f32>()?[0];
                    let rk4_w = if let Some(ref w) = block.gpu_ode_params.rk4_w {
                        let v = w.to_vec1::<f32>()?;
                        [v[0], v[1], v[2], v[3]]
                    } else {
                        [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]
                    };
                    let param_grads = self.ode_param_grads.as_ref()
                        .expect("CustomOp requires ode_param_grads on model").clone();
                    let op = crate::candle_tier::custom_ode::custom_ode::KerrOdeCustomOp::new(
                        gamma_raw, omega, alpha_v, beta_v, rk4_w,
                        block.gpu_ode_params.rk4_steps, block.gpu_ode_params.n_bands, block_idx,
                        param_grads,
                    );
                    // CustomOp runs on CPU — move tensor CPU→op→GPU
                    let precond_cpu = precond.to_device(&candle_core::Device::Cpu)?;
                    let ode_cpu = precond_cpu.apply_op1(op)?;
                    ode_cpu.to_device(&self.device)?
                } else {
                    crate::gpu_ode::gpu_ode::kerr_ode_gpu(&precond, &block.gpu_ode_params)?
                };

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
}
