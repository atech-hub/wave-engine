//! Block-diagonal linear layer — groups of bands processed independently.
//!
//! Replaces dense 768×768 out_proj (589K params) with 12 groups of 64×64 (49K total).
//! 12x parameter reduction in the single biggest compute bottleneck (35% of iter time).
//! Respects ODE's local band structure — ±2 stencil couples 5 bands, well within 32-band groups.

#[cfg(feature = "candle-backend")]
pub mod block_diag {
    use candle_core::{Tensor, Result};
    use candle_nn::{Linear, Module, VarBuilder};

    /// Block-diagonal linear: N_EMBD split into n_groups independent projections.
    /// Each group processes group_size dims → group_size dims.
    /// Forward: split input into groups, project each independently, concatenate.
    pub struct BlockDiagonalLinear {
        groups: Vec<Linear>,
        n_groups: usize,
        group_size: usize,
    }

    impl BlockDiagonalLinear {
        pub fn new(n_embd: usize, n_groups: usize, vb: VarBuilder) -> Result<Self> {
            assert_eq!(n_embd % n_groups, 0, "n_embd must be divisible by n_groups");
            let group_size = n_embd / n_groups;

            let mut groups = Vec::with_capacity(n_groups);
            for g in 0..n_groups {
                let gvb = vb.pp(format!("g{g}"));
                // Uniform init matching the engine's init pattern
                let limit = 1.0 / (group_size as f64).sqrt();
                let w = gvb.get_with_hints(
                    (group_size, group_size), "weight",
                    candle_nn::Init::Uniform { lo: -limit, up: limit },
                )?;
                let b = gvb.get_with_hints(
                    (group_size,), "bias",
                    candle_nn::Init::Const(0.0),
                )?;
                groups.push(Linear::new(w, Some(b)));
            }

            Ok(Self { groups, n_groups, group_size })
        }

        pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let mut outputs = Vec::with_capacity(self.n_groups);
            for g in 0..self.n_groups {
                let start = g * self.group_size;
                let group_input = x.narrow(candle_core::D::Minus1, start, self.group_size)?.contiguous()?;
                let group_output = self.groups[g].forward(&group_input)?;
                outputs.push(group_output);
            }
            Tensor::cat(&outputs, candle_core::D::Minus1)
        }
    }
}
