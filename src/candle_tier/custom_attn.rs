//! Wave-attention CustomOp — CPU scoring + weighted sum wrapped as a Candle
//! `CustomOp1` so the autograd graph doesn't break at the CPU boundary.
//!
//! Forward (`cpu_fwd`): same CPU math that used to live in `wave_attention` —
//! phase projection, harmonic coherence scores, causal softmax, value-weighted
//! sum — but without the `out_proj` step. The `out_proj` matmul stays in the
//! autograd graph as a regular Candle matmul, outside this op.
//!
//! Backward (`bwd`): receives `d_out_tensor` (gradient w.r.t. the weighted-sum
//! output, after autograd has backed through out_proj). Computes the full
//! manual backward chain — softmax backward, cos chain, atan2 backward,
//! phase-proj backward, v-proj backward — and returns `d_normed` as a Tensor
//! so Candle's autograd can propagate it through LN backward into upstream
//! parameters (embeddings, LN weights, everything above this block).
//!
//! This closes the Candle equivalent of CPU bug #6: before this op existed,
//! `x.to_vec2::<f32>()` severed the grad graph and attention's contribution
//! to `d_normed` was lost. Upstream params received FFN-only gradient.
//!
//! Harmonic `n_h` gradients (per head) are accumulated into the shared
//! `SharedAttnGrads` storage (same `Arc<Mutex>` pattern the ODE CustomOp
//! uses for `d_alpha`/`d_beta`/etc.). The training loop reads them out via
//! `take_attn_param_grads` after `loss.backward()`.

#[cfg(feature = "candle-backend")]
pub mod custom_attn {
    use candle_core::{CpuStorage, CustomOp1, Error, Layout, Result, Shape, Tensor};
    use std::sync::{Arc, Mutex};

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

    /// Cached intermediates from the forward — everything the backward needs
    /// so we don't recompute the full chain.
    struct AttnCache {
        att_w: Vec<Vec<Vec<f32>>>, // [n_head][n_pos][n_pos] post-softmax
        phases: Vec<Vec<f32>>,     // [n_head][n_pos] atan2 result
        v_all: Vec<Vec<Vec<f32>>>, // [n_head][n_pos][head_dim]
        r_vals: Vec<Vec<f32>>,     // [n_head][n_pos] — for atan2 backward
        s_vals: Vec<Vec<f32>>,     // [n_head][n_pos]
        input: Vec<Vec<f32>>,      // [n_pos][n_embd] normed (for pp/v backward)
    }

    /// The CustomOp. Holds CPU-resident weights + caches + gradient sinks.
    pub struct WaveAttentionCustomOp {
        pp_ws: Vec<Vec<Vec<f32>>>, // [n_head][2][n_embd]
        pp_bs: Vec<Vec<f32>>,      // [n_head][2]
        vw: Vec<Vec<Vec<f32>>>,    // [n_head][head_dim][head_dim]
        vb: Vec<Vec<f32>>,         // [n_head][head_dim]
        harmonic_ns_raw: Vec<f32>, // [n_head] pre-softplus
        n_head: usize,
        n_embd: usize,
        head_dim: usize,
        layer_idx: usize,
        cache: Arc<Mutex<Option<AttnCache>>>,
        param_grads: SharedAttnGrads,
    }

    impl WaveAttentionCustomOp {
        pub fn new(
            pp_ws: Vec<Vec<Vec<f32>>>,
            pp_bs: Vec<Vec<f32>>,
            vw: Vec<Vec<Vec<f32>>>,
            vb: Vec<Vec<f32>>,
            harmonic_ns_raw: Vec<f32>,
            n_embd: usize,
            layer_idx: usize,
            param_grads: SharedAttnGrads,
        ) -> Self {
            let n_head = harmonic_ns_raw.len();
            let head_dim = n_embd / n_head;
            Self {
                pp_ws, pp_bs, vw, vb, harmonic_ns_raw,
                n_head, n_embd, head_dim, layer_idx,
                cache: Arc::new(Mutex::new(None)),
                param_grads,
            }
        }

        /// Public accessor: take the cached attention weights after a forward.
        /// Used by monitors that want to inspect post-softmax scores.
        pub fn take_att_weights(&self) -> Option<Vec<Vec<Vec<f32>>>> {
            self.cache.lock().unwrap().as_ref().map(|c| c.att_w.clone())
        }
    }

    fn softplus(x: f32) -> f32 { crate::common::math::softplus(x) }
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
            if n_embd != self.n_embd {
                return Err(Error::Msg(format!("n_embd mismatch: op={}, input={}", self.n_embd, n_embd)));
            }
            let head_dim = self.head_dim;

            // Reconstruct [n_pos][n_embd] view respecting layout.start_offset
            let start = layout.start_offset();
            let input: Vec<Vec<f32>> = (0..n_pos)
                .map(|p| data[start + p * n_embd..start + (p + 1) * n_embd].to_vec())
                .collect();

            // Per-head intermediates
            let mut out_data = vec![0.0f32; n_pos * n_embd];
            let mut att_w: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; n_pos]; n_pos]; self.n_head];
            let mut phases_all: Vec<Vec<f32>> = vec![vec![0.0; n_pos]; self.n_head];
            let mut v_all_h: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; head_dim]; n_pos]; self.n_head];
            let mut r_all: Vec<Vec<f32>> = vec![vec![0.0; n_pos]; self.n_head];
            let mut s_all: Vec<Vec<f32>> = vec![vec![0.0; n_pos]; self.n_head];

            for h in 0..self.n_head {
                let offset = h * head_dim;
                let hn = softplus(self.harmonic_ns_raw[h]);
                let pp_w = &self.pp_ws[h];
                let pp_b = &self.pp_bs[h];
                let vw = &self.vw[h];
                let vb = &self.vb[h];

                // Phase projection — full-embd dot product
                for pos in 0..n_pos {
                    let mut r = pp_b[0];
                    let mut s = pp_b[1];
                    for j in 0..n_embd {
                        r += pp_w[0][j] * input[pos][j];
                        s += pp_w[1][j] * input[pos][j];
                    }
                    r_all[h][pos] = r;
                    s_all[h][pos] = s;
                    phases_all[h][pos] = s.atan2(r);
                }

                // Value projection — head-dim slice
                for pos in 0..n_pos {
                    for d in 0..head_dim {
                        let mut sum = vb[d];
                        for j in 0..head_dim { sum += vw[d][j] * input[pos][offset + j]; }
                        v_all_h[h][pos][d] = sum;
                    }
                }

                // Causal softmax + weighted sum
                for qi in 0..n_pos {
                    let mut scores = vec![f32::NEG_INFINITY; n_pos];
                    for ki in 0..=qi {
                        let delta = phases_all[h][qi] - phases_all[h][ki];
                        scores[ki] = (hn * delta).cos();
                    }
                    let max_s = scores[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    for ki in 0..=qi {
                        scores[ki] = (scores[ki] - max_s).exp();
                        exp_sum += scores[ki];
                    }
                    if exp_sum > 0.0 {
                        for ki in 0..=qi { scores[ki] /= exp_sum; }
                    }
                    for ki in 0..=qi { att_w[h][qi][ki] = scores[ki]; }

                    for d in 0..head_dim {
                        let mut sum = 0.0f32;
                        for ki in 0..=qi { sum += scores[ki] * v_all_h[h][ki][d]; }
                        out_data[qi * n_embd + offset + d] = sum;
                    }
                }
            }

            // Cache for backward
            *self.cache.lock().unwrap() = Some(AttnCache {
                att_w,
                phases: phases_all,
                v_all: v_all_h,
                r_vals: r_all,
                s_vals: s_all,
                input,
            });

            Ok((CpuStorage::F32(out_data), Shape::from_dims(&[n_pos, n_embd])))
        }

        fn bwd(&self, _arg: &Tensor, _node: &Tensor, output_grad: &Tensor) -> Result<Option<Tensor>> {
            let d_out_flat = output_grad.flatten_all()?.to_vec1::<f32>()?;
            let dims = output_grad.dims();
            if dims.len() != 2 {
                return Err(Error::Msg(format!("attn bwd: rank-2 expected, got {:?}", dims)));
            }
            let n_pos = dims[0];
            let n_embd = dims[1];
            let head_dim = self.head_dim;

            let cache_lock = self.cache.lock().unwrap();
            let cache = cache_lock.as_ref()
                .ok_or_else(|| Error::Msg("attn backward called without forward cache".into()))?;

            // Shape d_out as [n_pos][n_embd] for convenience.
            let d_out: Vec<Vec<f32>> = (0..n_pos)
                .map(|p| d_out_flat[p * n_embd..(p + 1) * n_embd].to_vec())
                .collect();

            let mut d_normed = vec![vec![0.0f32; n_embd]; n_pos];
            let mut d_harmonic_raws = vec![0.0f32; self.n_head];

            for h in 0..self.n_head {
                let offset = h * head_dim;
                let hn = softplus(self.harmonic_ns_raw[h]);
                let pp_w = &self.pp_ws[h];
                let vw = &self.vw[h];

                let att_w = &cache.att_w[h];
                let phases = &cache.phases[h];
                let v_all = &cache.v_all[h];
                let r_vals = &cache.r_vals[h];
                let s_vals = &cache.s_vals[h];

                // Accumulators for this head
                let mut d_v = vec![vec![0.0f32; head_dim]; n_pos];
                let mut d_phases = vec![0.0f32; n_pos];
                let mut d_hn_sum = 0.0f32;

                for qi in 0..n_pos {
                    // out[qi][offset+d] = sum_{ki<=qi} att_w[qi][ki] * v_all[ki][d]
                    // d_att_w_softmax[ki] = sum_d d_out[qi][offset+d] * v_all[ki][d]
                    // d_v[ki][d] += att_w[qi][ki] * d_out[qi][offset+d]
                    let mut d_att_sm = vec![0.0f32; qi + 1];
                    for ki in 0..=qi {
                        let w = att_w[qi][ki];
                        if w != 0.0 {
                            for d in 0..head_dim {
                                d_att_sm[ki] += d_out[qi][offset + d] * v_all[ki][d];
                                d_v[ki][d] += w * d_out[qi][offset + d];
                            }
                        }
                    }

                    // Softmax backward: d_score_raw[ki] = w[ki] * (d_att_sm[ki] - sum_kj w[kj] * d_att_sm[kj])
                    let weighted_sum: f32 = (0..=qi).map(|k| att_w[qi][k] * d_att_sm[k]).sum();
                    for ki in 0..=qi {
                        let w = att_w[qi][ki];
                        if w == 0.0 { continue; }
                        let d_score_raw = w * (d_att_sm[ki] - weighted_sum);

                        // Cos chain: score_raw[ki] = cos(hn * (phases[qi] - phases[ki]))
                        let delta = phases[qi] - phases[ki];
                        let sin_term = (hn * delta).sin();
                        // d(cos(hn*delta))/d(hn) = -sin(hn*delta) * delta
                        // d(cos(hn*delta))/d(delta) = -sin(hn*delta) * hn
                        d_hn_sum += d_score_raw * (-sin_term) * delta;
                        let d_delta = d_score_raw * (-sin_term) * hn;
                        // delta = phases[qi] - phases[ki]
                        d_phases[qi] += d_delta;
                        d_phases[ki] -= d_delta;
                    }
                }

                // atan2 backward: phases[pos] = atan2(s, r)
                // d phases / d r = -s / (r² + s²);  d phases / d s = r / (r² + s²)
                // Then: d_normed[pos][j] += d_r[pos] * pp_w[0][j] + d_s[pos] * pp_w[1][j]
                for pos in 0..n_pos {
                    let r = r_vals[pos];
                    let s = s_vals[pos];
                    let denom = r * r + s * s;
                    if denom > 1e-20 {
                        let inv = 1.0 / denom;
                        let d_r = d_phases[pos] * (-s) * inv;
                        let d_s = d_phases[pos] *   r  * inv;
                        for j in 0..n_embd {
                            d_normed[pos][j] += d_r * pp_w[0][j] + d_s * pp_w[1][j];
                        }
                    }
                }

                // v-proj backward: v_all[pos][d] = vb[d] + sum_j vw[d][j] * input[pos][offset+j]
                // d_normed[pos][offset+j] += sum_d d_v[pos][d] * vw[d][j]
                for pos in 0..n_pos {
                    for j in 0..head_dim {
                        let mut acc = 0.0f32;
                        for d in 0..head_dim { acc += d_v[pos][d] * vw[d][j]; }
                        d_normed[pos][offset + j] += acc;
                    }
                }

                // Harmonic_n chain: d(softplus(raw))/d(raw) = sigmoid(raw)
                d_harmonic_raws[h] = d_hn_sum * sigmoid(self.harmonic_ns_raw[h]);
            }

            // Write harmonic grads to shared storage for the optimizer to consume.
            {
                let mut v = self.param_grads.lock().unwrap();
                if self.layer_idx < v.len() {
                    v[self.layer_idx] = Some(AttnParamGrads { d_harmonic_raws });
                }
            }

            // Flatten d_normed back to CPU storage and build the return tensor.
            let flat: Vec<f32> = d_normed.into_iter().flatten().collect();
            let d_input_tensor = Tensor::from_vec(flat, output_grad.shape(), output_grad.device())?;
            Ok(Some(d_input_tensor))
        }
    }
}
