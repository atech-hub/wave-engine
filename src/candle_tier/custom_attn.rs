//! Wave-attention CustomOp — thin wrapper that routes Candle's attention
//! forward/backward through the canonical `common::attn` implementation.
//!
//! History: the first version of this file reimplemented the attention math
//! inside the op (dense causal scoring, no content projection). J10 on
//! seed=42 found max_abs 1.49e1 vs CPU because CPU uses phase-hashed sparse
//! scoring + frozen content projection. Rather than maintain two copies of
//! the math, the op now constructs a `WaveAttnWeights` from the block's CPU
//! caches and delegates to `common::attn::wave_attention_forward` /
//! `wave_attention_backward_pathway`. Same code as CPU; parity by construction.
//!
//! The op's bwd returns `d_normed` (via the shared pathway backward) so
//! Candle's autograd can thread attention's contribution into LN backward
//! alongside the FFN path's contribution — closing bug #6 as before.
//!
//! For `--harmonics dyn`, the op also computes `d_harmonic_raws` per head
//! using the cached softmax weights + phases from the forward, and writes
//! them into the shared `SharedAttnGrads` so the training loop can apply
//! them after `loss.backward()`.

#[cfg(feature = "candle-backend")]
pub mod custom_attn {
    use candle_core::{CpuStorage, CustomOp1, Error, Layout, Result, Shape, Tensor};
    use std::sync::{Arc, Mutex};

    use crate::common::attn::{WaveAttnCache, WaveAttnHeadWeights, WaveAttnWeights};

    /// Per-layer gradients for the trainable attention params. Harmonic `n_h`
    /// is the only one exposed today; phase/value projections are frozen from
    /// init and not autograd-tracked.
    pub struct AttnParamGrads {
        pub d_harmonic_raws: Vec<f32>, // [n_head], gradient of the softplus-raw
    }

    /// Thread-safe storage passed between the CustomOp and the training loop.
    pub type SharedAttnGrads = Arc<Mutex<Vec<Option<AttnParamGrads>>>>;

    /// Create shared storage for `n_layers` blocks.
    pub fn create_attn_grad_storage(n_layers: usize) -> SharedAttnGrads {
        Arc::new(Mutex::new((0..n_layers).map(|_| None).collect()))
    }

    /// Take and clear the stored gradient for one layer.
    pub fn take_attn_param_grads(storage: &SharedAttnGrads, layer: usize) -> Option<AttnParamGrads> {
        let mut v = storage.lock().unwrap();
        if layer < v.len() { v[layer].take() } else { None }
    }

    /// Cached intermediates from the forward — the full `WaveAttnCache`
    /// produced by `wave_attention_forward` plus the weights snapshot so the
    /// shared backward can be invoked verbatim.
    struct AttnOpCache {
        weights: WaveAttnWeights,
        attn_cache: WaveAttnCache,
    }

    /// The CustomOp. Holds a pre-built `WaveAttnWeights` (shared struct used
    /// by CPU attention) plus gradient sinks.
    pub struct WaveAttentionCustomOp {
        weights: WaveAttnWeights,
        n_bands: usize,
        harmonic_ns_raw: Vec<f32>, // [n_head] pre-softplus (mirrors weights.heads[h].harmonic_raw)
        layer_idx: usize,
        cache: Arc<Mutex<Option<AttnOpCache>>>,
        param_grads: SharedAttnGrads,
    }

    impl WaveAttentionCustomOp {
        pub fn new(
            weights: WaveAttnWeights,
            n_bands: usize,
            layer_idx: usize,
            param_grads: SharedAttnGrads,
        ) -> Self {
            let harmonic_ns_raw = weights.heads.iter().map(|h| h.harmonic_raw).collect();
            Self {
                weights, n_bands, harmonic_ns_raw, layer_idx,
                cache: Arc::new(Mutex::new(None)),
                param_grads,
            }
        }
    }

    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    impl CustomOp1 for WaveAttentionCustomOp {
        fn name(&self) -> &'static str { "wave_attention" }

        fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
            let data = match storage {
                CpuStorage::F32(d) => d,
                _ => return Err(Error::Msg("WaveAttentionCustomOp expects F32".into())),
            };
            let dims = layout.dims();
            if dims.len() != 2 {
                return Err(Error::Msg(format!("WaveAttentionCustomOp expects rank-2 input, got {:?}", dims)));
            }
            let n_pos = dims[0];
            let n_embd = dims[1];
            let expected_embd = self.n_bands * 2;
            if n_embd != expected_embd {
                return Err(Error::Msg(format!(
                    "n_embd mismatch: op expects {}, input is {}", expected_embd, n_embd
                )));
            }

            // Reconstruct [n_pos][n_embd] view respecting layout.start_offset
            let start = layout.start_offset();
            let input: Vec<Vec<f32>> = (0..n_pos)
                .map(|p| data[start + p * n_embd..start + (p + 1) * n_embd].to_vec())
                .collect();

            // Shared CPU attention forward. Produces post-out_proj output plus
            // the full intermediate cache needed for backward.
            let (attn_out, _att_w_for_monitor, cache_opt) =
                crate::common::attn::wave_attention_forward(
                    &self.weights, &input, self.n_bands, None, /*return_pathway_cache=*/ true,
                );
            let attn_cache = cache_opt.expect("wave_attention_forward must return cache when requested");

            // Stash for backward.
            *self.cache.lock().unwrap() = Some(AttnOpCache {
                weights: self.weights.clone(),
                attn_cache,
            });

            // Flatten [n_pos][n_embd] → [n_pos * n_embd]
            let flat: Vec<f32> = attn_out.into_iter().flatten().collect();
            Ok((CpuStorage::F32(flat), Shape::from_dims(&[n_pos, n_embd])))
        }

        fn bwd(&self, _arg: &Tensor, _node: &Tensor, output_grad: &Tensor) -> Result<Option<Tensor>> {
            let d_out_flat = output_grad.flatten_all()?.to_vec1::<f32>()?;
            let dims = output_grad.dims();
            if dims.len() != 2 {
                return Err(Error::Msg(format!("attn bwd: rank-2 expected, got {:?}", dims)));
            }
            let n_pos = dims[0];
            let n_embd = dims[1];

            let cache_lock = self.cache.lock().unwrap();
            let op_cache = cache_lock.as_ref()
                .ok_or_else(|| Error::Msg("attn backward called without forward cache".into()))?;

            // Reshape d_out to [n_pos][n_embd] for the shared backward.
            let d_attn_out: Vec<Vec<f32>> = (0..n_pos)
                .map(|p| d_out_flat[p * n_embd..(p + 1) * n_embd].to_vec())
                .collect();

            // Shared CPU backward — handles out_proj, softmax, cos chain,
            // atan2, phase-proj, v-proj, content-proj. Returns d_normed.
            let d_normed = crate::common::attn::wave_attention_backward_pathway(
                &op_cache.weights, &op_cache.attn_cache, &d_attn_out,
            );

            // Harmonic grads for --harmonics dyn. Shared backward doesn't
            // return these (attention is normally frozen), so we compute
            // them here from the cached softmax weights + phases. Only the
            // cos chain's d/dhn term is needed — all other pieces of d_score
            // contribute to d_phases / d_v_all which are already handled.
            let n_head = op_cache.weights.heads.len();
            let mut d_harmonic_raws = vec![0.0f32; n_head];
            let head_dim = n_embd / n_head;

            // Pull d_scores_raw from: d_attn_out → out_proj backward → per-head d_heads
            //                         → value-agg backward → d_att_w_softmax
            //                         → softmax backward → d_scores_raw
            // The shared backward already did this internally but doesn't
            // expose d_scores. Re-derive the minimal piece needed for harmonic
            // grads: it's the same math, scoped to cos-chain d/dhn.
            //
            // We short-cut by reconstructing d_att_w_softmax[h][qi][ki] from
            // the cached v_all and d_heads (which equals d_attn_out after
            // out_proj backward — cheaper to compute here than to plumb).
            let out_proj_w = &op_cache.weights.out_proj_w;
            // d_out_merged = d_attn_out @ out_proj_w.T
            let mut d_out_merged = vec![vec![0.0f32; n_embd]; n_pos];
            for qi in 0..n_pos {
                for k in 0..n_embd {
                    let mut acc = 0.0f32;
                    for j in 0..n_embd {
                        acc += d_attn_out[qi][j] * out_proj_w[j][k];
                    }
                    d_out_merged[qi][k] = acc;
                }
            }

            for h in 0..n_head {
                let hn = crate::common::math::softplus(op_cache.weights.heads[h].harmonic_raw);
                let offset = h * head_dim;
                let att_w = &op_cache.attn_cache.att_w[h];
                let phases = &op_cache.attn_cache.phases[h];
                let v_all = &op_cache.attn_cache.v_all[h];

                let mut d_hn_sum = 0.0f32;
                for qi in 0..n_pos {
                    // d_att_w_softmax[ki] = sum_d d_out_merged[qi][offset+d] * v_all[ki][d]
                    let mut d_sm = vec![0.0f32; qi + 1];
                    for ki in 0..=qi {
                        if att_w[qi][ki] == 0.0 { continue; }
                        let mut acc = 0.0f32;
                        for d in 0..head_dim {
                            acc += d_out_merged[qi][offset + d] * v_all[ki][d];
                        }
                        d_sm[ki] = acc;
                    }
                    // Softmax backward → d_scores_raw
                    let weighted: f32 = (0..=qi).map(|k| att_w[qi][k] * d_sm[k]).sum();
                    for ki in 0..=qi {
                        let w = att_w[qi][ki];
                        if w == 0.0 { continue; }
                        let d_score_raw = w * (d_sm[ki] - weighted);
                        // scores_raw[ki] = cos(hn * delta) + content_bias — content bias is
                        // independent of hn, so d/dhn is only the cos term.
                        let delta = phases[qi] - phases[ki];
                        let sin_term = (hn * delta).sin();
                        d_hn_sum += d_score_raw * (-sin_term) * delta;
                    }
                }
                d_harmonic_raws[h] = d_hn_sum * sigmoid(self.harmonic_ns_raw[h]);
            }

            // Write harmonic grads to shared storage.
            {
                let mut v = self.param_grads.lock().unwrap();
                if self.layer_idx < v.len() {
                    v[self.layer_idx] = Some(AttnParamGrads { d_harmonic_raws });
                }
            }

            let flat: Vec<f32> = d_normed.into_iter().flatten().collect();
            let d_input_tensor = Tensor::from_vec(flat, output_grad.shape(), output_grad.device())?;
            Ok(Some(d_input_tensor))
        }
    }
}
