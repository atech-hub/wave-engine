//! Output projection — dense or block-diagonal.
//!
//! groups=1: standard dense Linear (n_embd × n_embd). Full coupling.
//! groups>1: block-diagonal — N independent group_size×group_size projections.
//!   12x parameter reduction when groups=12 (49K vs 589K at 768-dim).

#[cfg(feature = "candle-backend")]
pub mod block_diag {
    use candle_core::{Tensor, Result};
    use candle_nn::{Linear, Module, VarBuilder};

    /// Output projection: dense (groups=1) or block-diagonal (groups>1).
    pub enum OutProj {
        Dense(Linear),
        BlockDiagonal {
            groups: Vec<Linear>,
            n_groups: usize,
            group_size: usize,
        },
    }

    impl OutProj {
        pub fn new(n_embd: usize, n_groups: usize, vb: VarBuilder) -> Result<Self> {
            if n_groups <= 1 {
                // Dense: single n_embd × n_embd linear
                let limit = 1.0 / (n_embd as f64).sqrt();
                let w = vb.get_with_hints(
                    (n_embd, n_embd), "weight",
                    candle_nn::Init::Uniform { lo: -limit, up: limit },
                )?;
                let b = vb.get_with_hints(
                    (n_embd,), "bias",
                    candle_nn::Init::Const(0.0),
                )?;
                Ok(Self::Dense(Linear::new(w, Some(b))))
            } else {
                // Block-diagonal: n_groups independent projections
                assert_eq!(n_embd % n_groups, 0, "n_embd must be divisible by n_groups");
                let group_size = n_embd / n_groups;
                let mut groups = Vec::with_capacity(n_groups);
                for g in 0..n_groups {
                    let gvb = vb.pp(format!("g{g}"));
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
                Ok(Self::BlockDiagonal { groups, n_groups, group_size })
            }
        }

        pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
            match self {
                Self::Dense(linear) => linear.forward(x),
                Self::BlockDiagonal { groups, n_groups, group_size } => {
                    let mut outputs = Vec::with_capacity(*n_groups);
                    for g in 0..*n_groups {
                        let start = g * group_size;
                        let group_input = x.narrow(candle_core::D::Minus1, start, *group_size)?.contiguous()?;
                        let group_output = groups[g].forward(&group_input)?;
                        outputs.push(group_output);
                    }
                    Tensor::cat(&outputs, candle_core::D::Minus1)
                }
            }
        }
    }

    // Backward compatibility alias
    pub type BlockDiagonalLinear = OutProj;
}
