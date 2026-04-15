//! Backward Decomposition Monitor — per-layer gradient flow through physics terms.
//!
//! Measures what fraction of the backward gradient signal flows through
//! damping, phase (SPM+XPM+omega), and FWM at each layer. Complements
//! the forward ode_decomposition monitor: forward shows what the ODE *did*,
//! backward shows what the optimizer *cares about*.
//!
//! Uses a separate instrumented backward pass on a sample input at health
//! intervals. Does not touch the training-path deriv_backward.

use crate::model::KerrWeights;
use crate::common::ode_backward::{ode_forward_with_cache, ode_backward, OdeParamGrads};

/// Per-layer backward decomposition stats.
pub struct BackwardDecompStats {
    pub layer: usize,
    pub total_grad_norm: f32,     // L2 norm of d_input
    pub damping_frac: f32,        // fraction from damping terms
    pub phase_frac: f32,          // fraction from rotation + SPM + XPM
    pub fwm_frac: f32,            // fraction from FWM quartets
    pub d_chi_norm: f32,          // |d_chi| magnitude
    pub d_alpha_norm: f32,        // |d_alpha|
    pub d_beta_norm: f32,         // |d_beta|
    pub d_gamma_norm: f32,        // L2 norm of d_gamma_raw
}

/// Measure backward decomposition for one layer's ODE.
/// Runs the full forward+backward with unit d_output, then reruns with
/// chi=0 and subtracts to isolate FWM's gradient contribution.
/// Also runs with gamma=0 to isolate damping's contribution.
pub fn measure_layer_backward(
    precond: &[f32],
    weights: &KerrWeights,
    layer: usize,
) -> BackwardDecompStats {
    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;
    let d_output: Vec<f32> = vec![1.0f32; n_embd];

    // Full backward (all physics terms active)
    let (out_full, cache_full) = ode_forward_with_cache(precond, weights);
    let (d_input_full, grads_full) = ode_backward(&d_output, &cache_full, weights);
    let total_norm = d_input_full.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Backward without FWM (chi=0)
    let mut weights_no_fwm = weights.clone();
    weights_no_fwm.chi = 0.0;
    let (_out_nf, cache_nf) = ode_forward_with_cache(precond, &weights_no_fwm);
    let (d_input_no_fwm, _) = ode_backward(&d_output, &cache_nf, &weights_no_fwm);

    // Backward damping only (alpha=0, beta=0, chi=0)
    let mut weights_damp = weights.clone();
    weights_damp.alpha = 0.0;
    weights_damp.beta = 0.0;
    weights_damp.chi = 0.0;
    let (_out_d, cache_d) = ode_forward_with_cache(precond, &weights_damp);
    let (d_input_damp, _) = ode_backward(&d_output, &cache_d, &weights_damp);

    // FWM contribution = full - no_fwm (L1 norm of difference)
    let fwm_l1: f32 = d_input_full.iter().zip(&d_input_no_fwm)
        .map(|(&a, &b)| (a - b).abs()).sum();

    // Damping contribution (L1)
    let damping_l1: f32 = d_input_damp.iter().map(|x| x.abs()).sum();

    // Phase contribution = no_fwm - damping (L1)
    let phase_l1: f32 = d_input_no_fwm.iter().zip(&d_input_damp)
        .map(|(&a, &b)| (a - b).abs()).sum();

    let total_l1 = damping_l1 + phase_l1 + fwm_l1;
    let inv = if total_l1 > 1e-12 { 1.0 / total_l1 } else { 0.0 };

    BackwardDecompStats {
        layer,
        total_grad_norm: total_norm,
        damping_frac: damping_l1 * inv,
        phase_frac: phase_l1 * inv,
        fwm_frac: fwm_l1 * inv,
        d_chi_norm: grads_full.d_chi.abs(),
        d_alpha_norm: grads_full.d_alpha.abs(),
        d_beta_norm: grads_full.d_beta.abs(),
        d_gamma_norm: grads_full.d_gamma_raw.iter().map(|x| x * x).sum::<f32>().sqrt(),
    }
}

/// Serialize backward decomposition stats to JSONL fragment.
pub fn to_json(stats: &[BackwardDecompStats], tier: &str) -> String {
    if stats.is_empty() { return String::new(); }
    let entries: Vec<String> = stats.iter().map(|s| {
        format!(
            r#"{{"layer":{},"tier":"{}","total_grad":{:.4},"damping_frac":{:.3},"phase_frac":{:.3},"fwm_frac":{:.3},"d_chi":{:.6},"d_alpha":{:.6},"d_beta":{:.6},"d_gamma":{:.6}}}"#,
            s.layer, tier, s.total_grad_norm,
            s.damping_frac, s.phase_frac, s.fwm_frac,
            s.d_chi_norm, s.d_alpha_norm, s.d_beta_norm, s.d_gamma_norm,
        )
    }).collect();
    format!(r#""ode_backward_decomposition":[{}]"#, entries.join(","))
}
