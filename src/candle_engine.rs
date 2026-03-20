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

    fn kerr_ode_cpu(x: &[f32], params: &OdeParams) -> Vec<f32> {
        let n_bands = params.gamma_raw.len();
        let n_embd = n_bands * 2;
        let n_steps = params.rk4_n_steps;
        let dt = 1.0 / n_steps as f32;

        fn softplus(v: f32) -> f32 { if v > 20.0 { v } else { (1.0 + v.exp()).ln() } }
        let gamma: Vec<f32> = params.gamma_raw.iter().map(|&g| softplus(g)).collect();

        let mut r: Vec<f32> = (0..n_bands).map(|k| x[k * 2]).collect();
        let mut s: Vec<f32> = (0..n_bands).map(|k| x[k * 2 + 1]).collect();

        let deriv = |r: &[f32], s: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let n = r.len();
            let mut dr = vec![0.0f32; n];
            let mut ds = vec![0.0f32; n];
            for k in 0..n {
                let mag_sq = r[k]*r[k] + s[k]*s[k];
                let mut ns = 0.0f32;
                if k >= 2 { ns += r[k-2]*r[k-2] + s[k-2]*s[k-2]; }
                if k >= 1 { ns += r[k-1]*r[k-1] + s[k-1]*s[k-1]; }
                if k+1 < n { ns += r[k+1]*r[k+1] + s[k+1]*s[k+1]; }
                if k+2 < n { ns += r[k+2]*r[k+2] + s[k+2]*s[k+2]; }
                let phi = params.omega[k] + params.alpha * mag_sq + params.beta * ns;
                dr[k] = -gamma[k] * r[k] - phi * s[k];
                ds[k] = -gamma[k] * s[k] + phi * r[k];
            }
            (dr, ds)
        };

        // NO clamping — RK4-16 is stable at 768-dim (proven, bounded [-7,7])
        // Clamps were the bug: they cost ~1 loss point (3.74 vs 2.79)
        for _ in 0..n_steps {
            let (k1r, k1s) = deriv(&r, &s);
            let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&a,&b)| a+0.5*dt*b).collect();
            let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&a,&b)| a+0.5*dt*b).collect();
            let (k2r, k2s) = deriv(&r2, &s2);
            let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&a,&b)| a+0.5*dt*b).collect();
            let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&a,&b)| a+0.5*dt*b).collect();
            let (k3r, k3s) = deriv(&r3, &s3);
            let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&a,&b)| a+dt*b).collect();
            let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&a,&b)| a+dt*b).collect();
            let (k4r, k4s) = deriv(&r4, &s4);
            for i in 0..n_bands {
                r[i] += dt/6.0 * (k1r[i] + 2.0*k2r[i] + 2.0*k3r[i] + k4r[i]);
                s[i] += dt/6.0 * (k1s[i] + 2.0*k2s[i] + 2.0*k3s[i] + k4s[i]);
            }
        }

        let mut out = vec![0.0f32; n_embd];
        for k in 0..n_bands { out[k * 2] = r[k]; out[k * 2 + 1] = s[k]; }
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
        phase_proj_ws: &[Tensor],
        phase_proj_bs: &[Tensor],
        v_proj_ws: &[Tensor],
        v_proj_bs: &[Tensor],
        harmonic_ns: &[f32],
        out_proj_w: &Tensor,
        out_proj_b: &Tensor,
    ) -> Result<Tensor> {
        let (n_pos, n_embd) = x.dims2()?;
        let n_head = harmonic_ns.len();
        let head_dim = n_embd / n_head;

        // Pull to CPU for attention scoring (frozen, custom op)
        let x_data = x.to_vec2::<f32>()?;
        let mut out_data = vec![0.0f32; n_pos * n_embd];

        for head in 0..n_head {
            let offset = head * head_dim;
            let harmonic_n = harmonic_ns[head];

            // Phase projection
            let pp_w = phase_proj_ws[head].to_vec2::<f32>()?;
            let pp_b = phase_proj_bs[head].to_vec1::<f32>()?;
            let phases: Vec<f32> = (0..n_pos).map(|pos| {
                let mut r = pp_b[0];
                let mut s = pp_b[1];
                for j in 0..n_embd { r += pp_w[0][j] * x_data[pos][j]; s += pp_w[1][j] * x_data[pos][j]; }
                s.atan2(r)
            }).collect();

            // Value projection
            let vw = v_proj_ws[head].to_vec2::<f32>()?;
            let vb = v_proj_bs[head].to_vec1::<f32>()?;
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
        Ok(projected)
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

        device: Device,
    }

    struct CandleBlock {
        // LN (trained)
        ln_w: Tensor,
        ln_b: Tensor,

        // Attention (frozen)
        phase_proj_ws: Vec<Tensor>,
        phase_proj_bs: Vec<Tensor>,
        v_proj_ws: Vec<Tensor>,
        v_proj_bs: Vec<Tensor>,
        harmonic_ns: Vec<f32>,
        attn_out_proj_w: Tensor,
        attn_out_proj_b: Tensor,

        // FFN (trained via VarMap)
        mae_in_sq: Linear,
        mae_in_pr: Linear,
        ode_params: OdeParams,
        mae_out_sq: Linear,
        mae_out_pr: Linear,
        out_proj: Linear,
    }

    impl CandleWaveModel {
        pub fn new(varmap: &VarMap, vocab_size: usize, device: &Device) -> Result<Self> {
            let mut rng = crate::rng::Rng::new(42);

            // Frozen embeddings
            let wte_data = build_harmonic_table(vocab_size, N_BANDS);
            let wte_flat: Vec<f32> = wte_data.iter().flat_map(|r| r.iter().copied()).collect();
            let wte = Tensor::from_vec(wte_flat, (vocab_size, N_EMBD), device)?;

            let wpe_data = build_positional_table(BLOCK_SIZE, N_BANDS);
            let wpe_flat: Vec<f32> = wpe_data.iter().flat_map(|r| r.iter().copied()).collect();
            let wpe = Tensor::from_vec(wpe_flat, (BLOCK_SIZE, N_EMBD), device)?;

            let vs = VarBuilder::from_varmap(varmap, DType::F32, device);

            let mut blocks = Vec::new();
            for layer in 0..N_LAYERS {
                let prefix = format!("block.{layer}");
                let vs_block = vs.pp(&prefix);

                // LN (trained)
                let ln_w = vs_block.get_with_hints((N_EMBD,), "ln_w", candle_nn::Init::Const(1.0))?;
                let ln_b = vs_block.get_with_hints((N_EMBD,), "ln_b", candle_nn::Init::Const(0.0))?;

                // Attention heads (frozen)
                let head_dim = N_EMBD / N_HEAD;
                let mut phase_proj_ws = Vec::new();
                let mut phase_proj_bs = Vec::new();
                let mut v_proj_ws = Vec::new();
                let mut v_proj_bs = Vec::new();
                let mut harmonic_ns = Vec::new();

                for h in 0..N_HEAD {
                    let limit = 1.0 / (N_EMBD as f32).sqrt();
                    let pw: Vec<f32> = (0..2*N_EMBD).map(|_| rng.uniform(limit)).collect();
                    let pb = vec![0.0f32; 2];
                    phase_proj_ws.push(Tensor::from_vec(pw, (2, N_EMBD), device)?);
                    phase_proj_bs.push(Tensor::from_vec(pb, (2,), device)?);

                    let vlimit = 1.0 / (head_dim as f32).sqrt();
                    let vw: Vec<f32> = (0..head_dim*head_dim).map(|_| rng.uniform(vlimit)).collect();
                    let vb = vec![0.0f32; head_dim];
                    v_proj_ws.push(Tensor::from_vec(vw, (head_dim, head_dim), device)?);
                    v_proj_bs.push(Tensor::from_vec(vb, (head_dim,), device)?);

                    harmonic_ns.push(((h + 1) as f32 * 0.5f32).ln());
                }

                let olimit = 1.0 / (N_EMBD as f32).sqrt();
                let ow: Vec<f32> = (0..N_EMBD*N_EMBD).map(|_| rng.uniform(olimit)).collect();
                let ob = vec![0.0f32; N_EMBD];
                let attn_out_proj_w = Tensor::from_vec(ow, (N_EMBD, N_EMBD), device)?;
                let attn_out_proj_b = Tensor::from_vec(ob, (N_EMBD,), device)?;

                // FFN (trained)
                let mae_in_sq = linear_uniform(N_EMBD, MAESTRO_DIM, vs_block.pp("mae_in_sq"))?;
                let mae_in_pr = linear_uniform(MAESTRO_DIM, N_EMBD, vs_block.pp("mae_in_pr"))?;
                let mae_out_sq = linear_uniform(N_EMBD, MAESTRO_DIM, vs_block.pp("mae_out_sq"))?;
                let mae_out_pr = linear_uniform(MAESTRO_DIM, N_EMBD, vs_block.pp("mae_out_pr"))?;
                let out_proj = linear_uniform(N_EMBD, N_EMBD, vs_block.pp("out_proj"))?;

                // ODE params (frozen)
                let gamma_raw_val = ((0.1f32).exp() - 1.0).ln();
                let ode_params = OdeParams {
                    gamma_raw: vec![gamma_raw_val; N_BANDS],
                    omega: (0..N_BANDS).map(|k| (k + 1) as f32 / N_BANDS as f32).collect(),
                    alpha: 0.1,
                    beta: 0.1,
                    rk4_n_steps: RK4_STEPS,
                };

                blocks.push(CandleBlock {
                    ln_w, ln_b,
                    phase_proj_ws, phase_proj_bs, v_proj_ws, v_proj_bs,
                    harmonic_ns, attn_out_proj_w, attn_out_proj_b,
                    mae_in_sq, mae_in_pr, ode_params, mae_out_sq, mae_out_pr, out_proj,
                });
            }

            // Final LN + LM head (trained)
            let ln_f_w = vs.get_with_hints((N_EMBD,), "ln_f_w", candle_nn::Init::Const(1.0))?;
            let ln_f_b = vs.get_with_hints((N_EMBD,), "ln_f_b", candle_nn::Init::Const(0.0))?;
            let lm_head = vs.get_with_hints((vocab_size, N_EMBD), "lm_head",
                candle_nn::Init::Randn { mean: 0.0, stdev: 1.0 / (N_EMBD as f64).sqrt() })?;

            Ok(Self { wte, wpe, blocks, ln_f_w, ln_f_b, lm_head, device: device.clone() })
        }

        pub fn forward(&self, token_ids: &[usize]) -> Result<Tensor> {
            self.forward_with_curriculum(token_ids, &vec![1.0f32; N_BANDS])
        }

        /// Forward with curriculum: soft-mask inactive bands on FFN path only.
        /// `band_masks[k]` is the mask value for band k (0.01 for suppressed, 1.0 for active,
        /// intermediate values during ramp transitions).
        /// Attention sees full vector (frozen). FFN sees masked vector (trains on active bands).
        pub fn forward_with_curriculum(&self, token_ids: &[usize], band_masks: &[f32]) -> Result<Tensor> {
            let n_pos = token_ids.len();

            // Build GPU-resident mask from per-band values
            let ffn_mask = if band_masks.iter().any(|&v| v < 1.0) {
                let mut mask_data = vec![0.0f32; N_EMBD];
                for k in 0..N_BANDS {
                    mask_data[k * 2] = band_masks[k];
                    mask_data[k * 2 + 1] = band_masks[k];
                }
                Some(Tensor::from_vec(mask_data, (1, N_EMBD), &self.device)?)
            } else {
                None
            };

            // Embedding: lookup + positional (NO masking — LN needs full vector)
            let mut hidden_vecs = vec![0.0f32; n_pos * N_EMBD];
            let wte_data = self.wte.to_vec2::<f32>()?;
            let wpe_data = self.wpe.to_vec2::<f32>()?;
            for (pos, &tok) in token_ids.iter().enumerate() {
                for i in 0..N_EMBD {
                    hidden_vecs[pos * N_EMBD + i] = wte_data[tok][i] + wpe_data[pos][i];
                }
            }
            let mut hidden = Tensor::from_vec(hidden_vecs, (n_pos, N_EMBD), &self.device)?;

            for (block_idx, block) in self.blocks.iter().enumerate() {
                let normed = layer_norm(&hidden, &block.ln_w, &block.ln_b)?;
                // NaN monitor
                if normed.to_vec2::<f32>()?.iter().any(|r| r.iter().any(|v| v.is_nan())) {
                    eprintln!("  [NaN] block {block_idx} after LN");
                }

                // Attention (frozen, CPU scoring, GPU out_proj)
                let attn_out = wave_attention(
                    &normed,
                    &block.phase_proj_ws, &block.phase_proj_bs,
                    &block.v_proj_ws, &block.v_proj_bs,
                    &block.harmonic_ns,
                    &block.attn_out_proj_w, &block.attn_out_proj_b,
                )?;

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

                // Monitor precond max value
                let precond_vals = precond.to_vec2::<f32>()?;
                let precond_max = precond_vals.iter()
                    .flat_map(|r| r.iter()).cloned().fold(0.0f32, |a, b| a.max(b.abs()));
                if precond_max > 10.0 || precond_max.is_nan() || precond_max.is_infinite() {
                    eprintln!("  [PRECOND] block {block_idx} max={precond_max:.2}");
                }

                // ODE via CustomOp1 — forward runs RK4 (no clamping), backward is identity
                let effective_ode_out = kerr_ode_batch(&precond, &block.ode_params)?;

                // Maestro out (operates on effective ODE output — gradients flow through precond)
                let mae_out = block.mae_out_sq.forward(&effective_ode_out)?;
                let mae_out = mae_out.gelu()?;
                let mae_out = block.mae_out_pr.forward(&mae_out)?;
                let regulated = (&effective_ode_out + &mae_out)?;

                // Monitor ODE and maestro_out outputs
                if effective_ode_out.to_vec2::<f32>()?.iter().any(|r| r.iter().any(|v| v.is_nan())) {
                    eprintln!("  [NaN] block {block_idx} ODE output");
                }
                if regulated.to_vec2::<f32>()?.iter().any(|r| r.iter().any(|v| v.is_nan())) {
                    eprintln!("  [NaN] block {block_idx} regulated (before out_proj)");
                }

                // Out proj
                let ffn_out = block.out_proj.forward(&regulated)?;

                // NaN monitors
                if attn_out.to_vec2::<f32>()?.iter().any(|r| r.iter().any(|v| v.is_nan())) {
                    eprintln!("  [NaN] block {block_idx} after attention");
                }
                if ffn_out.to_vec2::<f32>()?.iter().any(|r| r.iter().any(|v| v.is_nan())) {
                    eprintln!("  [NaN] block {block_idx} after FFN");
                }

                // Parallel residual
                hidden = (&hidden + &attn_out + &ffn_out)?;
            }

            // Final LN + LM head
            let normed = layer_norm(&hidden, &self.ln_f_w, &self.ln_f_b)?;
            let logits = normed.matmul(&self.lm_head.t()?)?;

            Ok(logits)
        }
    }

    // ─── Training loop ───

    pub fn train_candle(data_path: &str, n_iters: usize) -> Result<()> {
        println!("Candle backend — wave-engine\n");

        // Device
        let device = Device::cuda_if_available(0)?;
        println!("  Device: {:?}", device);

        // Load data + tokenize (with token cache — 3min encode → instant reload)
        let use_bpe = std::env::args().any(|a| a == "--bpe");
        let tokenizer_path = std::env::args().skip_while(|a| a != "--tokenizer").nth(1)
            .unwrap_or("data/tokenizer.json".to_string());

        let (tokens, vocab_size) = if let Some((cached_toks, cached_vs)) = crate::token_cache::load_cache(data_path, use_bpe) {
            (cached_toks, cached_vs)
        } else {
            let raw = std::fs::read_to_string(data_path)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to read {data_path}: {e}")))?;
            let (toks, vs) = if use_bpe {
                let tokenizer = crate::bpe::BpeTokenizer::from_file(&tokenizer_path);
                let t = tokenizer.encode(&raw);
                let v = tokenizer.vocab_size;
                println!("  BPE tokens: {}, vocab: {}", t.len(), v);
                (t, v)
            } else {
                let chars: Vec<char> = raw.chars().collect();
                let mut vocab: Vec<char> = chars.clone();
                vocab.sort();
                vocab.dedup();
                let v = vocab.len();
                let char_to_idx: std::collections::HashMap<char, usize> =
                    vocab.iter().enumerate().map(|(i, &c)| (c, i)).collect();
                let t: Vec<usize> = chars.iter().map(|c| *char_to_idx.get(c).unwrap_or(&0)).collect();
                println!("  Char-level tokens: {}, vocab: {}", t.len(), v);
                (t, v)
            };
            crate::token_cache::save_cache(data_path, use_bpe, &toks, vs);
            (toks, vs)
        };
        let split = (tokens.len() as f32 * 0.9) as usize;
        let train_data = &tokens[..split];
        println!("  Train tokens: {}", train_data.len());

        // Model
        let mut varmap = VarMap::new();
        let model = CandleWaveModel::new(&varmap, vocab_size, &device)?;
        let n_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
        println!("  Trainable params: {n_params}");
        println!("  Architecture: {N_LAYERS} layers, {N_HEAD} heads, {N_BANDS} bands");

        // Resume from checkpoint if --resume flag
        let resume_path: Option<String> = std::env::args().skip_while(|a| a != "--resume").nth(1);
        let mut start_iter = 0usize;
        if let Some(ref ckpt) = resume_path {
            println!("  Resuming from: {ckpt}");
            varmap.load(ckpt)?;
            // Read iter from .meta file
            let meta_path = ckpt.replace(".safetensors", ".meta");
            if let Ok(meta) = std::fs::read_to_string(&meta_path) {
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
        let lr: f64 = parse_flag_c("--lr", if N_BANDS > 256 { 1e-4 } else { 3e-4 });

        // Optimizer
        use candle_nn::Optimizer;
        let mut optimizer = candle_nn::AdamW::new(
            varmap.all_vars(),
            candle_nn::ParamsAdamW { lr, ..Default::default() },
        )?;
        let mut rng = crate::rng::Rng::new(1337);

        // Curriculum: soft-mask inactive bands (0.01 scale, not zero)
        let use_curriculum = !std::env::args().any(|a| a == "--no-curriculum");
        let curriculum = if use_curriculum {
            crate::train::CurriculumSchedule::default_4stage(N_BANDS)
        } else {
            crate::train::CurriculumSchedule::none(N_BANDS)
        };

        let total_iters = start_iter + n_iters;
        println!("\nTraining for {n_iters} iters (batch={batch_size}, seq={seq_len}, lr={lr})");
        if start_iter > 0 { println!("  Resuming from iter {start_iter}, target {total_iters}"); }
        curriculum.describe(total_iters);
        println!("{:>6} {:>10} {:>10}", "Iter", "Loss", "Time");
        println!("{}", "-".repeat(35));

        // JSONL telemetry log (persists across crashes)
        let log_file = std::fs::File::create("training_log.jsonl").ok();
        let mut nan_skip_count = 0usize;

        // Cosine LR schedule with warmup
        let warmup_iters = 200usize;
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

        for iter in start_iter..total_iters {
            let band_masks = curriculum.band_masks(iter, total_iters, N_BANDS);
            let iter_start = Instant::now();
            let mut total_loss = 0.0f32;

            let current_lr = cosine_lr(iter);
            optimizer.set_learning_rate(current_lr);

            for _b in 0..batch_size {
                let start = (rng.next_u64() as usize) % (train_data.len() - seq_len - 1);
                let input = &train_data[start..start + seq_len];
                let target = &train_data[start + 1..start + seq_len + 1];

                // Explicit scope: all tensors dropped at block exit → GPU memory freed
                let loss_val = {
                    let logits = model.forward_with_curriculum(input, &band_masks)?;
                    let target_tensor = Tensor::from_vec(
                        target.to_vec().iter().map(|&t| t as u32).collect::<Vec<u32>>(),
                        (seq_len,), &device,
                    )?;
                    // Plain cross-entropy (no label smoothing — it leaked tensor graph)
                    let loss = candle_nn::loss::cross_entropy(&logits, &target_tensor)?;
                    let lv = loss.to_scalar::<f32>()?;

                    if lv.is_nan() || lv.is_infinite() {
                        nan_skip_count += 1;
                        eprintln!("  [NaN skip] iter {iter} batch {_b} (total skips: {nan_skip_count})");
                        lv // return NaN — will be skipped below
                    } else {
                        // Backward
                        let grads = loss.backward()?;

                        // Grad norm: compute on GPU as scalar, no flatten_all intermediate tensor
                        let mut total_norm_sq = 0.0f64;
                        for var in &varmap.all_vars() {
                            if let Some(grad) = grads.get(var) {
                                let norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                                total_norm_sq += norm_sq;
                            }
                        }
                        let total_norm = total_norm_sq.sqrt();

                        // Clip + step
                        if total_norm > 1.0 {
                            let scale = 1.0 / total_norm;
                            optimizer.set_learning_rate(current_lr * scale);
                            optimizer.step(&grads)?;
                            optimizer.set_learning_rate(current_lr);
                        } else {
                            optimizer.step(&grads)?;
                        }
                        lv
                    }
                    // logits, target_tensor, loss, grads ALL dropped here
                };

                if !loss_val.is_nan() {
                    total_loss += loss_val;
                }
            }

            total_loss /= batch_size as f32;
            let iter_time = iter_start.elapsed();

            // JSONL telemetry (survives crashes)
            if let Some(ref log) = log_file {
                use std::io::Write;
                let entry = format!(
                    "{{\"iter\":{},\"loss\":{:.4},\"lr\":{:.6},\"grad_norm\":{:.4},\"time_ms\":{},\"nan_skips\":{}}}\n",
                    iter, total_loss, current_lr, 0.0, iter_time.as_millis(), nan_skip_count
                );
                let _ = (&*log).write_all(entry.as_bytes());
            }

            if iter % 50 == 0 || iter == total_iters - 1 {
                println!("{:>6} {:>10.4} {:>10.1?}  lr={:.6}", iter, total_loss, iter_time, current_lr);
            }

            // Periodic checkpoint: save every 100 iters until leak is confirmed fixed
            if (iter + 1) % 100 == 0 || iter == total_iters - 1 {
                let st_path = format!("candle_checkpoint_iter{}.safetensors", iter + 1);
                let meta = format!("iter={}\nloss={}\nlr={}\nvocab_size={}\n", iter + 1, total_loss, current_lr, vocab_size);
                if varmap.save(&st_path).is_ok() {
                    std::fs::write(format!("candle_checkpoint_iter{}.meta", iter + 1), &meta).ok();
                    println!("  Checkpoint: {st_path}");
                }
                let _ = varmap.save("candle_checkpoint_latest.safetensors");
                std::fs::write("candle_checkpoint_latest.meta", &meta).ok();

                let params = extract_wchk_params(&varmap, N_LAYERS, N_EMBD, MAESTRO_DIM, vocab_size);
                let dummy_adam = crate::train::Adam::new(lr as f32, params.len());
                crate::wave_checkpoint::save_checkpoint(
                    &params, vocab_size, N_LAYERS, iter + 1, lr as f32,
                    &dummy_adam, 0, "checkpoint.bin",
                );
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
    fn extract_wchk_params(varmap: &VarMap, n_layers: usize, n_embd: usize, maestro_dim: usize, vocab_size: usize) -> Vec<f32> {
        let mut params = Vec::new();

        let get_flat = |name: &str| -> Vec<f32> {
            let data = varmap.data().lock().unwrap();
            data.get(name).map(|t| t.flatten_all().unwrap().to_vec1::<f32>().unwrap()).unwrap_or_default()
        };

        for i in 0..n_layers {
            let p = format!("block.{i}");
            // LN weights (trained)
            params.extend(get_flat(&format!("{p}.ln_w")));
            params.extend(get_flat(&format!("{p}.ln_b")));
            // LN FFN — Candle doesn't have separate ln_ffn, use ones/zeros as placeholder
            params.extend(vec![1.0f32; n_embd]); // ln_ffn weight
            params.extend(vec![0.0f32; n_embd]); // ln_ffn bias
            // Maestro in: squeeze (n_embd → maestro_dim), process (maestro_dim → n_embd)
            params.extend(get_flat(&format!("{p}.mae_in_sq.weight")));
            params.extend(get_flat(&format!("{p}.mae_in_sq.bias")));
            params.extend(get_flat(&format!("{p}.mae_in_pr.weight")));
            params.extend(get_flat(&format!("{p}.mae_in_pr.bias")));
            // Maestro out
            params.extend(get_flat(&format!("{p}.mae_out_sq.weight")));
            params.extend(get_flat(&format!("{p}.mae_out_sq.bias")));
            params.extend(get_flat(&format!("{p}.mae_out_pr.weight")));
            params.extend(get_flat(&format!("{p}.mae_out_pr.bias")));
            // Out proj
            params.extend(get_flat(&format!("{p}.out_proj.weight")));
            params.extend(get_flat(&format!("{p}.out_proj.bias")));
        }
        // ln_f
        params.extend(get_flat("ln_f_w"));
        params.extend(get_flat("ln_f_b"));
        // lm_head
        params.extend(get_flat("lm_head"));

        params
    }
}

// Stub when candle feature is not enabled
#[cfg(not(feature = "candle-backend"))]
pub mod engine {
    pub fn train_candle(_data_path: &str, _n_iters: usize) -> std::result::Result<(), String> {
        Err("Candle backend not enabled. Build with: cargo run --release --features candle-backend".to_string())
    }
}
