//! Candle-side wrapper for J1 gradient correctness check.
//!
//! Mirrors the CPU grad_check_wrapper structure — given a set of tokens/targets
//! and model dims, returns four pieces that `monitors::junctions::grad_check`
//! consumes directly:
//!   * forward closure: `&[f32] -> f64` (loss given flat params)
//!   * forward+backward closure: `&[f32] -> (f32, Vec<f32>)` (loss + flat grad)
//!   * initial flat params (in the same layout CPU uses — see
//!     `candle_checkpoint::extract_wchk_params`)
//!   * SectionLabels (reuses CPU's `build_section_labels` since the layout
//!     matches)
//!
//! The closures construct a fresh `VarMap` + `CandleWaveModel` per call,
//! load the flat params in via `load_wchk_params_into_varmap`, run forward
//! through the autograd graph (use_custom_op=false so every param sees a
//! real autograd grad), and compute cross-entropy loss. The backward closure
//! additionally calls `loss.backward()` and walks the VarMap to pull out
//! grads in the canonical order.
//!
//! Cross-entropy vs CPU's phase-native loss: J1 only requires
//! analytical==FD per tier — the two don't need to agree across tiers on the
//! loss value itself. Using Candle's native cross-entropy keeps the test
//! path identical to Candle training, which is what we actually want to
//! verify.

#[cfg(feature = "candle-backend")]
pub mod grad_check {
    use candle_core::{Device, Tensor};
    use candle_nn::VarMap;

    use crate::candle_tier::candle_checkpoint::checkpoint::{
        extract_wchk_params, load_wchk_params_into_varmap,
    };
    use crate::candle_tier::candle_model::model::CandleWaveModel;
    use crate::candle_tier::custom_attn::custom_attn::create_attn_grad_storage;
    use crate::common::wave_model::init_model;
    use crate::monitors::junctions::grad_check::SectionLabels;
    use crate::Dims;

    pub fn phase_native_check_candle(
        tokens: Vec<usize>,
        targets: Vec<usize>,
        n_layers: usize,
        n_bands: usize,
        n_head: usize,
        vocab_size: usize,
        alpha: f32,
        beta: f32,
    ) -> (
        impl Fn(&[f32]) -> f64,
        impl Fn(&[f32]) -> (f32, Vec<f32>),
        Vec<f32>,
        SectionLabels,
    ) {
        let n_embd = n_bands * 2;
        let maestro_dim = 16usize;
        let out_proj_groups = 1usize;
        let phase_native = true;
        let chi = 0.0f32;
        let device = Device::Cpu;

        // Build a once-off model to grab the initial flat params + labels.
        // Labels come from the CPU helper (same layout, verified aligned).
        let varmap0 = VarMap::new();
        let _model0 = CandleWaveModel::new(
            &varmap0, vocab_size, &device,
            n_bands, n_head, n_layers, maestro_dim, crate::RK4_STEPS, out_proj_groups,
            alpha, beta, chi, phase_native,
        ).expect("CandleWaveModel::new failed");
        // extract_wchk_params needs a &CandleWaveModel; the one from the varmap
        // above will do since we only read its shape-like fields.
        let initial_params = extract_wchk_params(
            &varmap0, &_model0, n_layers, n_embd, maestro_dim,
            vocab_size, out_proj_groups, n_bands,
            phase_native, false, false, false,
        );

        // Section labels from the matching CPU model layout. Must set
        // phase_native=true before label generation so the last section is
        // labelled `output_corrector` (matching our flatten_candle_grads
        // output) instead of `lm_head`.
        let dims = Dims::from_cli(n_bands, n_head, maestro_dim, 128, crate::RK4_STEPS)
            .with_learnable_ode(true)
            .with_corrector(true);
        let mut cpu_model = init_model(vocab_size, 42, n_layers, out_proj_groups, dims, alpha, beta);
        cpu_model.phase_native = true;
        cpu_model.output_corrector = vec![0.0; n_bands];
        let labels = crate::cpu::grad_check_wrapper::build_section_labels(&cpu_model);

        let tokens = std::sync::Arc::new(tokens);
        let targets = std::sync::Arc::new(targets);
        let fwd_t = tokens.clone();
        let fwd_g = targets.clone();
        let bwd_t = tokens.clone();
        let bwd_g = targets.clone();

        // Forward-only closure. Fresh model per call so params are fully reset.
        let forward_fn = move |params: &[f32]| -> f64 {
            let varmap = VarMap::new();
            let mut model = CandleWaveModel::new(
                &varmap, vocab_size, &Device::Cpu,
                n_bands, n_head, n_layers, maestro_dim, crate::RK4_STEPS, out_proj_groups,
                alpha, beta, chi, phase_native,
            ).expect("CandleWaveModel::new failed");
            load_wchk_params_into_varmap(
                &varmap, params,
                n_layers, n_embd, maestro_dim,
                vocab_size, out_proj_groups, n_bands,
                /*has_ode=*/ true, /*has_ls=*/ false, /*has_rk4=*/ false, phase_native,
                &Device::Cpu,
            ).expect("load_wchk_params_into_varmap failed");
            model.attn_param_grads = Some(create_attn_grad_storage(n_layers));

            let logits = model.forward(&fwd_t).expect("candle forward failed");
            let target_tensor = Tensor::from_vec(
                fwd_g.iter().map(|&t| t as u32).collect::<Vec<u32>>(),
                (fwd_g.len(),), &Device::Cpu,
            ).expect("target tensor");
            let loss = candle_nn::loss::cross_entropy(&logits, &target_tensor)
                .expect("cross_entropy failed");
            loss.to_scalar::<f32>().expect("loss scalar") as f64
        };

        // Forward+backward closure. Extracts grads via extract_wchk_params-
        // style walk over the GradStore (same ordering as initial_params).
        let backward_fn = move |params: &[f32]| -> (f32, Vec<f32>) {
            let varmap = VarMap::new();
            let mut model = CandleWaveModel::new(
                &varmap, vocab_size, &Device::Cpu,
                n_bands, n_head, n_layers, maestro_dim, crate::RK4_STEPS, out_proj_groups,
                alpha, beta, chi, phase_native,
            ).expect("CandleWaveModel::new failed");
            load_wchk_params_into_varmap(
                &varmap, params,
                n_layers, n_embd, maestro_dim,
                vocab_size, out_proj_groups, n_bands,
                true, false, false, phase_native,
                &Device::Cpu,
            ).expect("load_wchk_params_into_varmap failed");
            model.attn_param_grads = Some(create_attn_grad_storage(n_layers));

            let logits = model.forward(&bwd_t).expect("candle forward failed");
            let target_tensor = Tensor::from_vec(
                bwd_g.iter().map(|&t| t as u32).collect::<Vec<u32>>(),
                (bwd_g.len(),), &Device::Cpu,
            ).expect("target tensor");
            let loss = candle_nn::loss::cross_entropy(&logits, &target_tensor)
                .expect("cross_entropy failed");
            let loss_val = loss.to_scalar::<f32>().expect("loss scalar");

            let grads = loss.backward().expect("loss.backward failed");
            let grad_vec = flatten_candle_grads(
                &grads, &varmap,
                n_layers, n_embd, maestro_dim,
                vocab_size, out_proj_groups, n_bands,
                phase_native,
            );
            (loss_val, grad_vec)
        };

        (forward_fn, backward_fn, initial_params, labels)
    }

    /// Walk the grad store in the same order as `extract_wchk_params` writes
    /// params. Any Var not present in the grad store contributes zeros (the
    /// autograd graph didn't touch it — treated as zero gradient for that
    /// param, which is the correct analytical value).
    fn flatten_candle_grads(
        grads: &candle_core::backprop::GradStore,
        varmap: &VarMap,
        n_layers: usize, n_embd: usize, maestro_dim: usize,
        vocab_size: usize, out_proj_groups: usize, n_bands: usize,
        phase_native: bool,
    ) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::new();

        let grab = |name: &str, expected_len: usize| -> Vec<f32> {
            let data = varmap.data().lock().unwrap();
            if let Some(var) = data.get(name) {
                if let Some(g) = grads.get(var) {
                    if let Ok(flat) = g.flatten_all() {
                        if let Ok(v) = flat.to_vec1::<f32>() {
                            if v.len() == expected_len {
                                return v;
                            }
                        }
                    }
                }
            }
            vec![0.0f32; expected_len]
        };

        for i in 0..n_layers {
            let p = format!("block.{i}");
            out.extend(grab(&format!("{p}.ln_w"), n_embd));
            out.extend(grab(&format!("{p}.ln_b"), n_embd));
            // ln_ffn slot in the flat layout is zero — Candle doesn't create
            // the Var at all; CPU keeps it as dead-code constant. Emit zeros.
            out.extend(std::iter::repeat(0.0f32).take(n_embd * 2));
            out.extend(grab(&format!("{p}.mae_in_sq.weight"), maestro_dim * n_embd));
            out.extend(grab(&format!("{p}.mae_in_sq.bias"), maestro_dim));
            out.extend(grab(&format!("{p}.mae_in_pr.weight"), n_embd * maestro_dim));
            out.extend(grab(&format!("{p}.mae_in_pr.bias"), n_embd));
            out.extend(grab(&format!("{p}.mae_out_sq.weight"), maestro_dim * n_embd));
            out.extend(grab(&format!("{p}.mae_out_sq.bias"), maestro_dim));
            out.extend(grab(&format!("{p}.mae_out_pr.weight"), n_embd * maestro_dim));
            out.extend(grab(&format!("{p}.mae_out_pr.bias"), n_embd));
            // out_proj
            if out_proj_groups <= 1 {
                out.extend(grab(&format!("{p}.out_proj.weight"), n_embd * n_embd));
                out.extend(grab(&format!("{p}.out_proj.bias"), n_embd));
            } else {
                let gs = n_embd / out_proj_groups;
                for g in 0..out_proj_groups {
                    out.extend(grab(&format!("{p}.out_proj.g{g}.weight"), gs * gs));
                    out.extend(grab(&format!("{p}.out_proj.g{g}.bias"), gs));
                }
            }
            // ODE params
            out.extend(grab(&format!("{p}.ode.gamma_raw"), n_bands));
            out.extend(grab(&format!("{p}.ode.alpha"), 1));
            out.extend(grab(&format!("{p}.ode.beta"), 1));
            out.extend(grab(&format!("{p}.phase_correction"), n_bands));
        }
        // ln_f
        out.extend(grab("ln_f_w", n_embd));
        out.extend(grab("ln_f_b", n_embd));
        // phase_native → output_corrector, else lm_head
        if phase_native {
            out.extend(grab("output_corrector", n_bands));
        } else {
            out.extend(grab("lm_head", vocab_size * n_embd));
        }
        out
    }
}
