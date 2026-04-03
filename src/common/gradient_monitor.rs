//! Gradient Flow Per Component Monitor (#3).
//!
//! Breaks down gradient norms by component per layer:
//! LN, maestro_in, ODE (gamma + alpha + beta + corrector + rk4),
//! maestro_out, out_proj, plus individual alpha/beta/corrector/rk4 norms.

use crate::cpu::model_backward::Gradients;
use crate::Dims;

/// Per-layer gradient flow breakdown.
pub struct GradientFlowStats {
    pub layer: usize,
    pub ln_grad_norm: f32,
    pub maestro_in_grad_norm: f32,
    pub ode_grad_norm: f32,
    pub maestro_out_grad_norm: f32,
    pub out_proj_grad_norm: f32,
    pub alpha_grad: f32,
    pub beta_grad: f32,
    pub corrector_grad_norm: f32,
    pub rk4_grad_norm: f32,
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn l2_norm_2d(v: &[Vec<f32>]) -> f32 {
    v.iter().flat_map(|row| row.iter()).map(|x| x * x).sum::<f32>().sqrt()
}

/// Analyze per-component gradient norms for each layer.
pub fn analyze_gradients(grads: &Gradients, _dims: Dims) -> Vec<GradientFlowStats> {
    let n_blocks = grads.block_ln_w.len();
    let mut stats = Vec::with_capacity(n_blocks);

    for b in 0..n_blocks {
        // LayerNorm gradients: attn LN + FFN LN combined
        let ln_attn_norm = (l2_norm(&grads.block_ln_w[b]).powi(2)
            + l2_norm(&grads.block_ln_b[b]).powi(2)).sqrt();
        let ln_ffn_norm = (l2_norm(&grads.block_ln_ffn_w[b]).powi(2)
            + l2_norm(&grads.block_ln_ffn_b[b]).powi(2)).sqrt();
        let ln_grad_norm = (ln_attn_norm.powi(2) + ln_ffn_norm.powi(2)).sqrt();

        // Maestro-in gradients: squeeze + process weights and biases
        let mae_in_sq_w = l2_norm_2d(&grads.block_ffn_mae_in_sq_w[b]);
        let mae_in_sq_b = l2_norm(&grads.block_ffn_mae_in_sq_b[b]);
        let mae_in_pr_w = l2_norm_2d(&grads.block_ffn_mae_in_pr_w[b]);
        let mae_in_pr_b = l2_norm(&grads.block_ffn_mae_in_pr_b[b]);
        let maestro_in_grad_norm = (mae_in_sq_w.powi(2) + mae_in_sq_b.powi(2)
            + mae_in_pr_w.powi(2) + mae_in_pr_b.powi(2)).sqrt();

        // Maestro-out gradients
        let mae_out_sq_w = l2_norm_2d(&grads.block_ffn_mae_out_sq_w[b]);
        let mae_out_sq_b = l2_norm(&grads.block_ffn_mae_out_sq_b[b]);
        let mae_out_pr_w = l2_norm_2d(&grads.block_ffn_mae_out_pr_w[b]);
        let mae_out_pr_b = l2_norm(&grads.block_ffn_mae_out_pr_b[b]);
        let maestro_out_grad_norm = (mae_out_sq_w.powi(2) + mae_out_sq_b.powi(2)
            + mae_out_pr_w.powi(2) + mae_out_pr_b.powi(2)).sqrt();

        // Out projection gradients
        let out_proj_w = l2_norm_2d(&grads.block_ffn_out_proj_w[b]);
        let out_proj_b = l2_norm(&grads.block_ffn_out_proj_b[b]);
        let out_proj_grad_norm = (out_proj_w.powi(2) + out_proj_b.powi(2)).sqrt();

        // ODE param gradients (individual components)
        let alpha_grad = grads.block_ffn_kerr_alpha[b].abs();
        let beta_grad = grads.block_ffn_kerr_beta[b].abs();
        let corrector_grad_norm = l2_norm(&grads.block_ffn_phase_correction[b]);
        let gamma_norm = l2_norm(&grads.block_ffn_kerr_gamma_raw[b]);
        let rk4_grad_norm = l2_norm(&grads.block_ffn_rk4_weights[b]);

        // Combined ODE norm: gamma + alpha + beta + corrector + rk4
        let ode_grad_norm = (gamma_norm.powi(2) + alpha_grad.powi(2) + beta_grad.powi(2)
            + corrector_grad_norm.powi(2) + rk4_grad_norm.powi(2)).sqrt();

        stats.push(GradientFlowStats {
            layer: b,
            ln_grad_norm,
            maestro_in_grad_norm,
            ode_grad_norm,
            maestro_out_grad_norm,
            out_proj_grad_norm,
            alpha_grad,
            beta_grad,
            corrector_grad_norm,
            rk4_grad_norm,
        });
    }

    stats
}

/// Serialize gradient flow stats to JSONL fragment.
/// Format: "grad_flow":[{...}, ...]
pub fn to_json(stats: &[GradientFlowStats]) -> String {
    if stats.is_empty() {
        return String::new();
    }

    let entries: Vec<String> = stats.iter().map(|s| {
        format!(
            r#"{{"layer":{},"ln":{:.4},"maestro_in":{:.4},"ode":{:.4},"maestro_out":{:.4},"out_proj":{:.4},"alpha_grad":{:.6},"beta_grad":{:.6},"corrector":{:.4},"rk4":{:.6}}}"#,
            s.layer, s.ln_grad_norm, s.maestro_in_grad_norm, s.ode_grad_norm,
            s.maestro_out_grad_norm, s.out_proj_grad_norm,
            s.alpha_grad, s.beta_grad, s.corrector_grad_norm, s.rk4_grad_norm,
        )
    }).collect();

    format!(r#""grad_flow":[{}]"#, entries.join(","))
}
