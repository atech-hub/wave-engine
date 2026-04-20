//! Harmonic coherence attention — CPU scoring wrapped in a Candle CustomOp1
//! so the autograd graph stays intact through the CPU boundary. Bug-#6
//! equivalent on Candle (attention's `d_normed` contribution used to be lost
//! because `x.to_vec2` severed the graph) is closed by this wrapper.

#[cfg(feature = "candle-backend")]
pub mod attention {
    use candle_core::{Device, Result, Tensor};

    use crate::candle_tier::custom_attn::custom_attn::{
        SharedAttnGrads, WaveAttentionCustomOp,
    };

    // ─── Harmonic Coherence Attention ───
    // Forward is `normed → out_tensor` via CustomOp (CPU, autograd-connected),
    // then `out_proj` as a regular Candle matmul (autograd-tracked).

    pub fn wave_attention(
        x: &Tensor,
        pp_ws_cpu: &[Vec<Vec<f32>>],
        pp_bs_cpu: &[Vec<f32>],
        vw_cpu: &[Vec<Vec<f32>>],
        vb_cpu: &[Vec<f32>],
        harmonic_ns: &[f32],
        out_proj_w: &Tensor,
        out_proj_b: &Tensor,
        attn_param_grads: SharedAttnGrads,
        layer_idx: usize,
        store_attn_weights: bool, // kept for signature compat; attn weights taken off the op when needed
    ) -> Result<(Tensor, Option<Vec<Vec<Vec<f32>>>>)> {
        let (_n_pos, n_embd) = x.dims2()?;

        let op = WaveAttentionCustomOp::new(
            pp_ws_cpu.to_vec(),
            pp_bs_cpu.to_vec(),
            vw_cpu.to_vec(),
            vb_cpu.to_vec(),
            harmonic_ns.to_vec(),
            n_embd,
            layer_idx,
            attn_param_grads,
        );

        // CustomOp runs on CPU — round-trip the input if it lives on a GPU.
        let device = x.device().clone();
        let x_cpu = if matches!(device, Device::Cpu) {
            x.clone()
        } else {
            x.to_device(&Device::Cpu)?
        };

        // The autograd graph lives here: apply_op1 connects x_cpu → out_cpu with
        // WaveAttentionCustomOp::bwd as the registered backward. Moving back to
        // the original device preserves the graph (to_device is autograd-tracked).
        let out_cpu = x_cpu.apply_op1(op)?;
        // Taking att weights off the op requires holding a reference to it, which
        // we cannot do because apply_op1 consumes the op. If a caller needs the
        // softmax weights for monitoring, we would need a separate scoring pass
        // or to expose them via a second shared Arc. None of the current monitors
        // read them post-fix, so we return None for now.
        let _ = store_attn_weights;
        let att_weights: Option<Vec<Vec<Vec<f32>>>> = None;

        let out_tensor = if matches!(device, Device::Cpu) {
            out_cpu
        } else {
            out_cpu.to_device(&device)?
        };

        // out_proj: regular Candle matmul. Autograd handles the backward to
        // out_proj_w / out_proj_b and to out_tensor (which flows back through
        // the CustomOp's bwd into d_normed).
        let projected = out_tensor.matmul(&out_proj_w.t()?)?.broadcast_add(out_proj_b)?;
        Ok((projected, att_weights))
    }

}
