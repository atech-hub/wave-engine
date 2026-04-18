//! Gradient check wrappers for each training mode.
//!
//! Each function builds the closures and section labels that grad_check.rs needs.
//! The monitor stays mode-agnostic; mode-specific wrapping lives here.

use crate::common::wave_model::*;
use crate::common::dims::Dims;
use crate::common::fft_ode;
use crate::monitors::junctions::grad_check::SectionLabels;

/// Build section labels from a model's parameter layout.
/// Mirrors the ordering in flatten_params_ex / count_trainable_ex.
pub fn build_section_labels(model: &WavePacketModel) -> SectionLabels {
    let n_embd = model.ln_f.weight.len();
    let n_bands = n_embd / 2;
    let maestro_dim = model.blocks[0].ffn.maestro_in.squeeze.w.len();
    let mut ranges: Vec<(usize, String)> = Vec::new();
    let mut idx = 0usize;

    for (b, block) in model.blocks.iter().enumerate() {
        ranges.push((idx, format!("block_{}_ln_w", b)));
        idx += n_embd;
        ranges.push((idx, format!("block_{}_ln_b", b)));
        idx += n_embd;
        ranges.push((idx, format!("block_{}_ln_ffn_w", b)));
        idx += n_embd;
        ranges.push((idx, format!("block_{}_ln_ffn_b", b)));
        idx += n_embd;
        ranges.push((idx, format!("block_{}_mae_in_sq_w", b)));
        idx += maestro_dim * n_embd;
        ranges.push((idx, format!("block_{}_mae_in_sq_b", b)));
        idx += maestro_dim;
        ranges.push((idx, format!("block_{}_mae_in_pr_w", b)));
        idx += n_embd * maestro_dim;
        ranges.push((idx, format!("block_{}_mae_in_pr_b", b)));
        idx += n_embd;
        ranges.push((idx, format!("block_{}_mae_out_sq_w", b)));
        idx += maestro_dim * n_embd;
        ranges.push((idx, format!("block_{}_mae_out_sq_b", b)));
        idx += maestro_dim;
        ranges.push((idx, format!("block_{}_mae_out_pr_w", b)));
        idx += n_embd * maestro_dim;
        ranges.push((idx, format!("block_{}_mae_out_pr_b", b)));
        idx += n_embd;
        ranges.push((idx, format!("block_{}_ffn_out_proj", b)));
        idx += block.ffn.out_proj.param_count();
        if model.learnable_ode {
            ranges.push((idx, format!("block_{}_kerr_gamma", b)));
            idx += block.ffn.kerr.gamma_raw.len();
            ranges.push((idx, format!("block_{}_kerr_alpha", b)));
            idx += 1;
            ranges.push((idx, format!("block_{}_kerr_beta", b)));
            idx += 1;
            ranges.push((idx, format!("block_{}_phase_correction", b)));
            idx += block.ffn.kerr.phase_correction.len();
            if model.use_rk4_weights {
                ranges.push((idx, format!("block_{}_rk4_weights", b)));
                idx += 4;
            }
        }
        if model.use_dyn_harmonics {
            ranges.push((idx, format!("block_{}_harmonic_raw", b)));
            idx += block.attn.heads.len();
        }
    }
    if model.use_layer_scale {
        ranges.push((idx, "layer_scale".to_string()));
        idx += model.layer_scale.len();
    }
    ranges.push((idx, "ln_f_w".to_string()));
    idx += n_embd;
    ranges.push((idx, "ln_f_b".to_string()));
    idx += n_embd;
    if model.phase_native {
        ranges.push((idx, "output_corrector".to_string()));
    } else if model.wd_state.is_some() {
        ranges.push((idx, "wave_decode".to_string()));
    } else if model.lm_rank > 0 {
        ranges.push((idx, "lm_down".to_string()));
        idx += model.lm_rank * n_embd;
        ranges.push((idx, "lm_up".to_string()));
    } else {
        ranges.push((idx, "lm_head".to_string()));
    }

    SectionLabels::new(ranges)
}

/// Phase-native training mode: token-based forward, phase-native loss.
pub fn phase_native_check(
    tokens: Vec<usize>,
    targets: Vec<usize>,
    n_layers: usize,
    n_bands: usize,
    n_head: usize,
    vocab_size: usize,
    alpha: f32,
    beta: f32,
    attention_pathway: bool,
    learnable_ode: bool,
    ode_pathway: bool,
) -> (
    impl Fn(&[f32]) -> f64,
    impl Fn(&[f32]) -> (f32, Vec<f32>),
    Vec<f32>,
    SectionLabels,
) {
    let dims = Dims::from_cli(n_bands, n_head, 16, 128, 16)
        .with_learnable_ode(learnable_ode)
        .with_corrector(learnable_ode || ode_pathway) // corrector on when ODE backward is active
        .with_attention_pathway(attention_pathway)
        .with_ode_pathway(ode_pathway);
    let mut model = init_model(vocab_size, 42, n_layers, 1, dims, alpha, beta);
    model.phase_native = true;
    model.output_corrector = vec![0.0; n_bands];

    crate::ffn_backend::init_agc(alpha, beta);
    let stencil = fft_ode::StencilFft::new(n_bands);

    let initial_params = flatten_params_ex(&model, false);
    let labels = build_section_labels(&model);

    let tokens_fwd = std::sync::Arc::new(tokens);
    let targets_fwd = std::sync::Arc::new(targets);
    let tokens_bwd = tokens_fwd.clone();
    let targets_bwd = targets_fwd.clone();

    // J1 verification: accumulate loss at f64 so the FD subtraction in
    // grad_check doesn't cancel against f32 quantization. Model weights stay
    // f32; only the loss arithmetic is lifted.
    let forward_fn = move |params: &[f32]| -> f64 {
        let mut m = init_model(vocab_size, 42, n_layers, 1, dims, alpha, beta);
        m.phase_native = true;
        m.output_corrector = vec![0.0; n_bands];
        unflatten_params_ex(&mut m, params, false);
        let stencil = fft_ode::StencilFft::new(n_bands);
        let cache = crate::cpu::forward::forward_with_cache(
            &m, &tokens_fwd, dims, None, None, None, Some(&stencil), None, None, None,
        );
        let t = tokens_fwd.len();
        let mut total_loss = 0.0_f64;
        for pos in 0..t {
            total_loss += crate::common::phase_loss::phase_native_loss_value_f64(
                &cache.post_ln_f[pos], &m.wte, targets_fwd[pos], n_bands, 1.0, &m.output_corrector,
            );
        }
        total_loss / t.max(1) as f64
    };

    let forward_backward_fn = move |params: &[f32]| -> (f32, Vec<f32>) {
        let mut m = init_model(vocab_size, 42, n_layers, 1, dims, alpha, beta);
        m.phase_native = true;
        m.output_corrector = vec![0.0; n_bands];
        unflatten_params_ex(&mut m, params, false);
        let stencil = fft_ode::StencilFft::new(n_bands);
        let cache = crate::cpu::forward::forward_with_cache(
            &m, &tokens_bwd, dims, None, None, None, Some(&stencil), None, None, None,
        );
        let (loss, grads) = crate::cpu::model_backward::backward(
            &m, &cache, &targets_bwd, dims, None, None, None,
        );
        let flat_grads = crate::cpu::model_backward::flatten_grads_ex(&grads, false);
        (loss, flat_grads)
    };

    (forward_fn, forward_backward_fn, initial_params, labels)
}

/// Wave-input training mode: KWDS wave inputs, L2 loss in wave space.
pub fn wave_input_check(
    inputs: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
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
    let dims = Dims::from_cli(n_bands, n_head, 16, 128, 16);
    let mut model = init_model(vocab_size, 42, n_layers, 1, dims, alpha, beta);
    model.phase_native = true;
    model.output_corrector = vec![0.0; n_bands];
    model.learnable_ode = true;

    crate::ffn_backend::init_agc(alpha, beta);

    let initial_params = flatten_params_ex(&model, false);
    let labels = build_section_labels(&model);

    let inputs_fwd = std::sync::Arc::new(inputs);
    let targets_fwd = std::sync::Arc::new(targets);
    let inputs_bwd = inputs_fwd.clone();
    let targets_bwd = targets_fwd.clone();

    // J1 verification: accumulate L2 loss at f64. See phase_native_check for rationale.
    let forward_fn = move |params: &[f32]| -> f64 {
        let mut m = init_model(vocab_size, 42, n_layers, 1, dims, alpha, beta);
        m.phase_native = true;
        m.output_corrector = vec![0.0; n_bands];
        m.learnable_ode = true;
        unflatten_params_ex(&mut m, params, false);
        let stencil = fft_ode::StencilFft::new(n_bands);
        let cache = crate::cpu::forward::forward_with_cache_from_waves(
            &m, &inputs_fwd, dims, Some(&stencil),
        );
        let t = cache.post_ln_f.len().min(targets_fwd.len());
        let n_embd = n_bands * 2;
        let mut total = 0.0_f64;
        for pos in 0..t {
            let mut pos_loss = 0.0_f64;
            for i in 0..n_embd {
                let diff = (cache.post_ln_f[pos][i] as f64) - (targets_fwd[pos][i] as f64);
                pos_loss += diff * diff;
            }
            total += pos_loss / n_embd as f64;
        }
        total / t.max(1) as f64
    };

    let forward_backward_fn = move |params: &[f32]| -> (f32, Vec<f32>) {
        let mut m = init_model(vocab_size, 42, n_layers, 1, dims, alpha, beta);
        m.phase_native = true;
        m.output_corrector = vec![0.0; n_bands];
        m.learnable_ode = true;
        unflatten_params_ex(&mut m, params, false);
        let stencil = fft_ode::StencilFft::new(n_bands);
        let cache = crate::cpu::forward::forward_with_cache_from_waves(
            &m, &inputs_bwd, dims, Some(&stencil),
        );
        let (loss, grads) = crate::cpu::model_backward::backward_wave(
            &m, &cache, &targets_bwd, dims,
        );
        let flat_grads = crate::cpu::model_backward::flatten_grads_ex(&grads, false);
        (loss, flat_grads)
    };

    (forward_fn, forward_backward_fn, initial_params, labels)
}
