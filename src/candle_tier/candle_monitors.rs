//! Monitor data collection and JSON serialization for candle backend.

#[cfg(feature = "candle-backend")]
pub mod monitors {
    use candle_core::{Tensor, Result};
    use candle_nn::VarMap;

    /// Per-layer flow statistics (norms, ratios, cosine similarity).
    pub struct CandleLayerFlow {
        pub layer: usize,
        pub input_norm: f32,
        pub attn_out_norm: f32,
        pub ffn_out_norm: f32,
        pub output_norm: f32,
        pub attn_ratio: f32,
        pub ffn_ratio: f32,
        pub residual_ratio: f32,
        pub cosine_in_out: f32,
    }

    /// Per-head attention statistics.
    pub struct CandleAttnHead {
        pub layer: usize,
        pub head: usize,
        pub harmonic: f32,
        pub entropy: f32,
        pub max_weight: f32,
    }

    /// Per-layer ODE dynamics statistics.
    pub struct CandleOdeDynamics {
        pub layer: usize,
        pub energy_in: f32,
        pub energy_out: f32,
        pub energy_ratio: f32,
        pub phase_velocity: f32,
        pub damping: f32,
        pub band_energy_std: f32,
    }

    /// Monitor data collected during one forward pass.
    #[derive(Default)]
    pub struct CandleMonitorData {
        pub layer_flow: Vec<CandleLayerFlow>,
        pub attn_heads: Vec<CandleAttnHead>,
        pub ode_dynamics: Vec<CandleOdeDynamics>,
    }

    impl CandleMonitorData {
        pub fn layer_flow_json(&self) -> String {
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

        pub fn attn_heads_json(&self) -> String {
            if self.attn_heads.is_empty() { return String::new(); }
            let entries: Vec<String> = self.attn_heads.iter().map(|s| {
                format!(
                    r#"{{"layer":{},"head":{},"harmonic":{:.3},"entropy":{:.3},"max_w":{:.4}}}"#,
                    s.layer, s.head, s.harmonic, s.entropy, s.max_weight,
                )
            }).collect();
            format!(r#""attn_heads":[{}]"#, entries.join(","))
        }

        pub fn ode_dynamics_json(&self) -> String {
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
    pub struct CandleOutputDist {
        pub avg_entropy: f32,
        pub avg_margin: f32,
        pub avg_correct_rank: f32,
        pub worst_margin: f32,
        pub worst_pos: usize,
        pub mode_collapse: bool,
    }

    pub fn compute_output_dist(logits: &Tensor, targets: &[usize]) -> CandleOutputDist {
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

    pub fn output_dist_json(s: &CandleOutputDist) -> String {
        format!(
            r#""output_dist":{{"avg_entropy":{:.3},"avg_margin":{:.4},"avg_correct_rank":{:.1},"worst_margin":{:.4},"worst_pos":{},"mode_collapse":{}}}"#,
            s.avg_entropy, s.avg_margin, s.avg_correct_rank,
            s.worst_margin, s.worst_pos, s.mode_collapse,
        )
    }

    /// Per-layer gradient flow statistics.
    pub struct CandleGradientFlow {
        pub layer: usize,
        pub ln_norm: f32,
        pub maestro_in_norm: f32,
        pub ode_norm: f32,
        pub maestro_out_norm: f32,
        pub out_proj_norm: f32,
    }

    pub fn compute_gradient_flow(
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

            // Out proj — dense (out_proj.weight) or block-diagonal (out_proj.g0.weight, ...)
            let mut op_sq = 0.0f32;
            // Try dense key first
            let w_dense = grad_norm_for("out_proj.weight");
            let b_dense = grad_norm_for("out_proj.bias");
            op_sq += w_dense * w_dense + b_dense * b_dense;
            // Then block-diagonal groups
            for g in 0..16 {
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

    pub fn gradient_flow_json(stats: &[CandleGradientFlow]) -> String {
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
}
