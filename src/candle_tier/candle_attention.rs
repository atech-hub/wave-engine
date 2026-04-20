//! Harmonic coherence attention — CustomOp wrapper that routes through the
//! canonical `common::attn::wave_attention_forward` + `wave_attention_backward_pathway`.
//! Parity with CPU is by construction: same code path, same weights (via the
//! block's CPU caches), same phase-hashed sparse scoring + content projection.

#[cfg(feature = "candle-backend")]
pub mod attention {
    use candle_core::{Device, Result, Tensor};

    use crate::candle_tier::candle_model::model::CandleBlock;
    use crate::candle_tier::custom_attn::custom_attn::{
        SharedAttnGrads, WaveAttentionCustomOp,
    };
    use crate::common::attn::{WaveAttnHeadWeights, WaveAttnWeights};

    /// Run wave attention for one block on the Candle autograd graph.
    ///
    /// The CustomOp consumes a full `WaveAttnWeights` (same struct as CPU) and
    /// runs the shared `wave_attention_forward` internally. No out_proj matmul
    /// outside the op: the shared forward already applies it, and the shared
    /// backward handles its gradient within the op's `bwd`.
    pub fn wave_attention(
        x: &Tensor,
        block: &CandleBlock,
        n_bands: usize,
        attn_param_grads: SharedAttnGrads,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let n_head = block.harmonic_ns.len();
        let heads: Vec<WaveAttnHeadWeights> = (0..n_head).map(|h| WaveAttnHeadWeights {
            harmonic_raw: block.harmonic_ns[h],
            phase_proj_w: block.phase_proj_ws_cpu[h].clone(),
            phase_proj_b: block.phase_proj_bs_cpu[h].clone(),
            v_proj_w: block.v_proj_ws_cpu[h].clone(),
            v_proj_b: block.v_proj_bs_cpu[h].clone(),
            content_proj_w: block.content_proj_ws_cpu.get(h).cloned().unwrap_or_default(),
            content_proj_b: block.content_proj_bs_cpu.get(h).cloned().unwrap_or_default(),
        }).collect();
        let weights = WaveAttnWeights {
            heads,
            out_proj_w: block.attn_out_proj_w_cpu.clone(),
            out_proj_b: block.attn_out_proj_b_cpu.clone(),
        };

        let op = WaveAttentionCustomOp::new(weights, n_bands, layer_idx, attn_param_grads);

        // CustomOp runs on CPU; round-trip the tensor if it lives elsewhere.
        // apply_op1 preserves the autograd graph — the op's bwd is the
        // registered backward for the x_cpu → out_cpu edge, so `d_normed`
        // enters the grad store before LN backward consumes it.
        let device = x.device().clone();
        let x_cpu = if matches!(device, Device::Cpu) { x.clone() } else { x.to_device(&Device::Cpu)? };
        let out_cpu = x_cpu.apply_op1(op)?;
        let attn_out = if matches!(device, Device::Cpu) { out_cpu } else { out_cpu.to_device(&device)? };
        Ok(attn_out)
    }
}
