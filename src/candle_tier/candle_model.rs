//! CandleWaveModel struct + CandleBlock struct + new() constructor.

#[cfg(feature = "candle-backend")]
pub mod model {
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::{Linear, VarBuilder, VarMap};

    /// Linear layer with uniform init matching wgpu engine: uniform(-1/sqrt(in_dim), 1/sqrt(in_dim))
    pub fn linear_uniform(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
        let bound = 1.0 / (in_dim as f64).sqrt();
        let init = candle_nn::Init::Uniform { lo: -bound, up: bound };
        let ws = vb.get_with_hints((out_dim, in_dim), "weight", init)?;
        let bs = vb.get_with_hints(out_dim, "bias", candle_nn::Init::Const(0.0))?;
        Ok(Linear::new(ws, Some(bs)))
    }

    pub struct OdeParams {
        pub gamma_raw: Vec<f32>,
        pub omega: Vec<f32>,
        pub alpha: f32,
        pub beta: f32,
        pub rk4_n_steps: usize,
    }

    // ─── Model ───

    pub struct CandleWaveModel {
        // Frozen embeddings
        pub wte: Tensor,
        pub wpe: Tensor,

        // Per-block
        pub blocks: Vec<CandleBlock>,

        // Final
        pub ln_f_w: Tensor,
        pub ln_f_b: Tensor,
        pub lm_head: Tensor,
        pub output_corrector: Option<Tensor>,  // [1, n_bands] phase-native output corrector
        pub phase_native: bool,
        pub layer_agcs: Option<Vec<crate::common::agc::OdeAgc>>,  // per-layer AGC (when --agc-headroom dyn)
        pub use_custom_op: bool,  // true = CustomOp ODE (no autograd graph, CPU backward)
        pub use_cuda_kernel: bool, // true = CUDA native kernel (GPU forward, CPU backward)
        pub ode_param_grads: Option<crate::candle_tier::custom_ode::custom_ode::SharedParamGrads>,
        /// Shared attention-param grads populated by WaveAttentionCustomOp::bwd.
        /// One slot per block; optimizer reads it after loss.backward().
        pub attn_param_grads: Option<crate::candle_tier::custom_attn::custom_attn::SharedAttnGrads>,

        pub device: Device,

        // Runtime config
        pub n_bands: usize,
        pub n_embd: usize,
        pub n_head: usize,
        pub block_size: usize,
        pub debug_nan: bool,
    }

    pub struct CandleBlock {
        // LN (trained)
        pub ln_w: Tensor,
        pub ln_b: Tensor,

        // Attention (frozen) — GPU tensors for out_proj gradient graph
        pub phase_proj_ws: Vec<Tensor>,
        pub phase_proj_bs: Vec<Tensor>,
        pub v_proj_ws: Vec<Tensor>,
        pub v_proj_bs: Vec<Tensor>,
        pub harmonic_ns: Vec<f32>,   // harmonic_raw values (softplus before use in scoring)
        pub harmonic_init: Vec<f32>, // initial values (for spring equilibrium)
        pub attn_out_proj_w: Tensor,
        pub attn_out_proj_b: Tensor,

        // Attention (frozen) — CPU-cached copies, eliminates GPU→CPU transfers per forward call
        pub phase_proj_ws_cpu: Vec<Vec<Vec<f32>>>,  // [n_head][2][n_embd]
        pub phase_proj_bs_cpu: Vec<Vec<f32>>,        // [n_head][2]
        pub v_proj_ws_cpu: Vec<Vec<Vec<f32>>>,       // [n_head][head_dim][head_dim]
        pub v_proj_bs_cpu: Vec<Vec<f32>>,            // [n_head][head_dim]
        // Frozen content projection (symmetry-breaker) — matches CPU's
        // `WaveAttnHeadWeights.content_proj_w/b`. Empty `Vec`s mean "no content
        // bias" (pure harmonic coherence); non-empty adds a content-dependent
        // attention bias. Populated in parity-runner via copy from CPU model;
        // initialised empty in CandleWaveModel::new until Candle training
        // owns the init stream alignment.
        pub content_proj_ws_cpu: Vec<Vec<Vec<f32>>>, // [n_head][CONTENT_DIM][n_embd]
        pub content_proj_bs_cpu: Vec<Vec<f32>>,      // [n_head][CONTENT_DIM]
        pub attn_out_proj_w_cpu: Vec<Vec<f32>>,      // [n_embd][n_embd] — cached for shared backward
        pub attn_out_proj_b_cpu: Vec<f32>,           // [n_embd]

        // Harmonics dyn — gate on whether the custom_attn harmonic grad is applied
        // in the optimizer step. The forward path no longer branches on this; the
        // CustomOp always computes d_harmonic_raws, we just choose whether to use them.
        pub harmonic_dyn: bool,

        // FFN (trained via VarMap)
        pub mae_in_sq: Linear,
        pub mae_in_pr: Linear,
        pub ode_params: OdeParams,
        pub gpu_ode_params: crate::gpu_ode::gpu_ode::GpuOdeParams,
        pub phase_correction: Tensor,  // [1, n_bands] — corrector plate phase angles (learnable)
        pub mae_out_sq: Linear,
        pub mae_out_pr: Linear,
        pub out_proj: crate::block_diagonal::block_diag::BlockDiagonalLinear,
        pub layer_scale: Option<Tensor>,  // [1] scalar — residual contribution multiplier
    }

    impl CandleWaveModel {
        pub fn new(varmap: &VarMap, vocab_size: usize, device: &Device,
                   n_bands: usize, n_head: usize, n_layers: usize, maestro_dim: usize,
                   rk4_steps: usize, out_proj_groups: usize, alpha: f32, beta: f32,
                   chi: f32, phase_native: bool) -> Result<Self> {
            let n_embd = n_bands * 2;
            let block_size = 256; // positional table size
            // Save config for methods
            let n_bands_cfg = n_bands;
            let n_embd_cfg = n_embd;
            let n_head_cfg = n_head;
            let block_size_cfg = block_size;
            let mut rng = crate::rng::Rng::new(42);

            // Frozen embeddings
            let wte_data = crate::wave_embed::build_harmonic_table(vocab_size, n_bands);
            let wte_flat: Vec<f32> = wte_data.iter().flat_map(|r| r.iter().copied()).collect();
            let wte = Tensor::from_vec(wte_flat, (vocab_size, n_embd), device)?;

            let wpe_data = crate::wave_embed::build_positional_table(block_size, n_bands);
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
                    n_bands, alpha, beta, chi, rk4_steps, vs_block.pp("ode"),
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

                // Pre-flatten attn_out_proj to CPU Vec for the shared backward
                // path (common::attn::wave_attention_backward_pathway reads
                // out_proj_w as Vec<Vec<f32>>).
                let attn_op_w_flat = attn_out_proj_w.flatten_all()?.to_vec1::<f32>()?;
                let attn_out_proj_w_cpu: Vec<Vec<f32>> = (0..n_embd)
                    .map(|i| attn_op_w_flat[i * n_embd..(i + 1) * n_embd].to_vec())
                    .collect();
                let attn_out_proj_b_cpu: Vec<f32> = attn_out_proj_b.flatten_all()?.to_vec1::<f32>()?;

                blocks.push(CandleBlock {
                    ln_w, ln_b,
                    phase_proj_ws, phase_proj_bs, v_proj_ws, v_proj_bs,
                    phase_proj_ws_cpu, phase_proj_bs_cpu, v_proj_ws_cpu, v_proj_bs_cpu,
                    content_proj_ws_cpu: vec![vec![]; n_head],
                    content_proj_bs_cpu: vec![vec![]; n_head],
                    attn_out_proj_w_cpu,
                    attn_out_proj_b_cpu,
                    harmonic_init: harmonic_ns.clone(),
                    harmonic_ns, attn_out_proj_w, attn_out_proj_b,
                    harmonic_dyn: false,  // set by train_candle when --harmonics dyn
                    mae_in_sq, mae_in_pr, ode_params, gpu_ode_params, phase_correction,
                    mae_out_sq, mae_out_pr, out_proj,
                    layer_scale: None, // set by --layer-scale dyn
                });
            }

            // Final LN + LM head (trained) or output corrector (phase-native)
            let ln_f_w = vs.get_with_hints((n_embd,), "ln_f_w", candle_nn::Init::Const(1.0))?;
            let ln_f_b = vs.get_with_hints((n_embd,), "ln_f_b", candle_nn::Init::Const(0.0))?;
            // phase_native passed from caller — controls lm_head vs output_corrector
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
                layer_agcs: None, use_custom_op: false, use_cuda_kernel: false, ode_param_grads: None, attn_param_grads: None, device: device.clone(),
                n_bands: n_bands_cfg, n_embd: n_embd_cfg, n_head: n_head_cfg, block_size: block_size_cfg, debug_nan: false })
        }
    }

    // ─── Layer Norm (simple, no candle_nn LayerNorm to avoid version issues) ───

    pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor) -> candle_core::Result<Tensor> {
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let diff = x.broadcast_sub(&mean)?;
        let var = (&diff * &diff)?.mean_keepdim(candle_core::D::Minus1)?;
        let inv_std = (var + 1e-5)?.sqrt()?.recip()?;
        let normed = diff.broadcast_mul(&inv_std)?;
        normed.broadcast_mul(weight)?.broadcast_add(bias)
    }
}
