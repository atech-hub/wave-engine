//! Forward pass for ModelWeights — CPU reference implementation.
//!
//! Extracted from model.rs. Contains: forward, forward_with_memory,
//! forward_block, causal_self_attention, per_band_linear,
//! kerr_maestro_add, kerr_dual_maestro, kerr_ode_forward,
//! extract_ode_states, maestro_forward.

use super::model::{
    ModelWeights, BlockWeights, FfnWeights, KerrMaestroAddWeights,
    KerrDualMaestroWeights, KerrWeights, MaestroWeights, PerBandLinearWeights,
    AttentionWeights, layer_norm, linear_fn, gelu,
};
use super::ode_deriv::rk4_step_public as rk4_step;
use super::math::softplus;

/// Linear without bias: y[i] = sum_j w[i][j] * x[j]
#[inline]
fn linear_no_bias(w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum::<f32>())
        .collect()
}

impl ModelWeights {
    /// Full forward pass: token indices → logits.
    pub fn forward(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        self.forward_with_memory(tokens, None)
    }

    /// Forward pass with optional wave memory injection.
    ///
    /// `memory_offsets` is a slice of per-layer (r_offset, s_offset) pairs.
    /// Each offset is added to the Kerr-ODE initial conditions before RK4.
    /// When None, the code path is identical to `forward()` (bit-identical).
    pub fn forward_with_memory(
        &self,
        tokens: &[usize],
        memory_offsets: Option<&[(&[f32], &[f32])]>,
    ) -> Vec<Vec<f32>> {
        let t = tokens.len();
        let n_embd = self.config.n_embd();
        assert!(t <= self.config.block_size);

        // Embedding + positional encoding
        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; n_embd];
            for i in 0..n_embd {
                h[i] = self.wte_phase[tok][i] + self.wpe[pos][i];
            }
            hidden.push(h);
        }

        // Process through blocks — track ODE layer index for memory injection
        let mut ode_layer = 0usize;
        for block in &self.blocks {
            let mem = match (&block.ffn, memory_offsets) {
                (FfnWeights::KerrMaestro(_), Some(offsets)) if ode_layer < offsets.len() => {
                    let m = Some(offsets[ode_layer]);
                    ode_layer += 1;
                    m
                }
                (FfnWeights::KerrMaestro(_), _) => { ode_layer += 1; None }
                (FfnWeights::KerrDualMaestro(_), Some(offsets)) if ode_layer < offsets.len() => {
                    let m = Some(offsets[ode_layer]);
                    ode_layer += 1;
                    m
                }
                (FfnWeights::KerrDualMaestro(_), _) => { ode_layer += 1; None }
                _ => None, // PerBandLinear — no ODE, no memory
            };
            hidden = self.forward_block(block, &hidden, mem);
        }

        // Final layer norm + LM head
        let mut logits = Vec::with_capacity(t);
        for h in &hidden {
            let normed = layer_norm(h, &self.ln_f.weight, &self.ln_f.bias);
            let l = linear_no_bias(&self.lm_head, &normed);
            logits.push(l);
        }

        logits
    }

    fn forward_block(
        &self,
        block: &BlockWeights,
        hidden: &[Vec<f32>],
        memory: Option<(&[f32], &[f32])>,
    ) -> Vec<Vec<f32>> {
        let t = hidden.len();
        let n_embd = self.config.n_embd();

        // x = x + attn(ln_1(x))
        let normed_1: Vec<Vec<f32>> = hidden.iter()
            .map(|h| layer_norm(h, &block.ln_1.weight, &block.ln_1.bias))
            .collect();
        let attn_out = self.causal_self_attention(&block.attn, &normed_1);
        let mut h: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let mut v = vec![0.0f32; n_embd];
                for j in 0..n_embd { v[j] = hidden[i][j] + attn_out[i][j]; }
                v
            })
            .collect();

        // x = x + ffn(ln_2(x))
        let normed_2: Vec<Vec<f32>> = h.iter()
            .map(|x| layer_norm(x, &block.ln_2.weight, &block.ln_2.bias))
            .collect();
        let ffn_out = match &block.ffn {
            FfnWeights::PerBand(w) => self.per_band_linear(w, &normed_2),
            FfnWeights::KerrMaestro(w) => self.kerr_maestro_add_with_memory(w, &normed_2, memory),
            FfnWeights::KerrDualMaestro(w) => self.kerr_dual_maestro_forward_with_memory(w, &normed_2, memory),
        };
        for i in 0..t {
            for j in 0..n_embd { h[i][j] += ffn_out[i][j]; }
        }

        h
    }

    fn causal_self_attention(&self, weights: &AttentionWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = x.len();
        let n_embd = self.config.n_embd();
        let n_head = weights.n_head;
        let head_dim = n_embd / n_head;

        // Compute Q, K, V for all positions
        let mut q_all = vec![vec![0.0f32; n_embd]; t];
        let mut k_all = vec![vec![0.0f32; n_embd]; t];
        let mut v_all = vec![vec![0.0f32; n_embd]; t];

        for pos in 0..t {
            let qkv = linear_fn(&weights.c_attn.w, &weights.c_attn.b, &x[pos]);
            for i in 0..n_embd {
                q_all[pos][i] = qkv[i];
                k_all[pos][i] = qkv[n_embd + i];
                v_all[pos][i] = qkv[2 * n_embd + i];
            }
        }

        // Multi-head attention
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![vec![0.0f32; n_embd]; t];

        for head in 0..n_head {
            let offset = head * head_dim;

            // Compute attention scores for this head
            for qi in 0..t {
                // Compute attention weights
                let mut att = vec![f32::NEG_INFINITY; t];
                for ki in 0..=qi {  // causal: only attend to past
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_all[qi][offset + d] * k_all[ki][offset + d];
                    }
                    att[ki] = dot * scale;
                }

                // Softmax
                let max_att = att[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for ki in 0..=qi {
                    att[ki] = (att[ki] - max_att).exp();
                    exp_sum += att[ki];
                }
                for ki in 0..=qi {
                    att[ki] /= exp_sum;
                }

                // Weighted sum of values
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for ki in 0..=qi {
                        sum += att[ki] * v_all[ki][offset + d];
                    }
                    out[qi][offset + d] = sum;
                }
            }
        }

        // Output projection
        let result: Vec<Vec<f32>> = out.iter()
            .map(|o| linear_fn(&weights.c_proj.w, &weights.c_proj.b, o))
            .collect();

        result
    }

    pub fn per_band_linear(&self, weights: &PerBandLinearWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = x.len();
        let n_bands = weights.band_w.len();
        let n_embd = n_bands * 2;
        let mut result = Vec::with_capacity(t);

        for pos in 0..t {
            let mut bands_out = vec![0.0f32; n_embd];

            for band in 0..n_bands {
                let r_in = x[pos][band * 2];
                let s_in = x[pos][band * 2 + 1];
                let w = &weights.band_w[band];
                let b = &weights.band_b[band];

                // y = W @ [r, s] + b  (2x2 matrix)
                bands_out[band * 2] = w[0][0] * r_in + w[1][0] * s_in + b[0];
                bands_out[band * 2 + 1] = w[0][1] * r_in + w[1][1] * s_in + b[1];
            }

            let projected = linear_fn(&weights.out_proj.w, &weights.out_proj.b, &bands_out);
            result.push(projected);
        }

        result
    }

    pub fn kerr_maestro_add(&self, weights: &KerrMaestroAddWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.kerr_maestro_add_with_memory(weights, x, None)
    }

    pub fn kerr_maestro_add_with_memory(
        &self,
        weights: &KerrMaestroAddWeights,
        x: &[Vec<f32>],
        memory: Option<(&[f32], &[f32])>,
    ) -> Vec<Vec<f32>> {
        let t = x.len();
        let mut result = Vec::with_capacity(t);

        for pos in 0..t {
            // Kerr path (with optional memory injection)
            let kerr_out = self.kerr_ode_forward_with_memory(&weights.kerr, &x[pos], memory);

            // Maestro path (no memory injection — global coordination only)
            let maestro_out = self.maestro_forward(&weights.maestro, &x[pos]);

            // Combine + project
            let n_embd = kerr_out.len();
            let mut combined = vec![0.0f32; n_embd];
            for i in 0..n_embd {
                combined[i] = kerr_out[i] + maestro_out[i];
            }

            let projected = linear_fn(&weights.out_proj.w, &weights.out_proj.b, &combined);
            result.push(projected);
        }

        result
    }

    /// Dual-maestro forward: maestro_in → ODE → maestro_out → out_proj.
    /// Pre-ODE maestro normalises input energy. Post-ODE maestro re-synchronises.
    pub fn kerr_dual_maestro_forward(
        &self,
        weights: &KerrDualMaestroWeights,
        x: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        self.kerr_dual_maestro_forward_with_memory(weights, x, None)
    }

    /// Dual-maestro forward with optional wave memory injection.
    pub fn kerr_dual_maestro_forward_with_memory(
        &self,
        weights: &KerrDualMaestroWeights,
        x: &[Vec<f32>],
        memory: Option<(&[f32], &[f32])>,
    ) -> Vec<Vec<f32>> {
        let t = x.len();
        let mut result = Vec::with_capacity(t);

        for pos in 0..t {
            let n_embd = x[pos].len();

            // 1. Input maestro — regulates energy BEFORE ODE
            let mae_in_out = self.maestro_forward(&weights.maestro_in, &x[pos]);
            let mut precond = vec![0.0f32; n_embd];
            for i in 0..n_embd { precond[i] = x[pos][i] + mae_in_out[i]; }

            // 2. ODE runs on pre-conditioned input (with optional memory)
            let kerr_out = self.kerr_ode_forward_with_memory(&weights.kerr, &precond, memory);

            // 3. Output maestro — re-synchronises AFTER ODE
            let mae_out_out = self.maestro_forward(&weights.maestro_out, &kerr_out);
            let mut regulated = vec![0.0f32; n_embd];
            for i in 0..n_embd { regulated[i] = kerr_out[i] + mae_out_out[i]; }

            // 4. Output projection
            let projected = weights.out_proj.forward(&regulated);
            result.push(projected);
        }

        result
    }

    pub fn kerr_ode_forward(&self, weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
        self.kerr_ode_forward_with_memory(weights, x, None)
    }

    /// Kerr-ODE forward pass with optional wave memory injection.
    ///
    /// When `memory` is Some((r_offsets, s_offsets)), the offsets are added
    /// to the initial conditions before RK4 integration. When None, the
    /// code path is identical to the original (bit-identical baseline).
    pub fn kerr_ode_forward_with_memory(
        &self,
        weights: &KerrWeights,
        x: &[f32],
        memory: Option<(&[f32], &[f32])>,
    ) -> Vec<f32> {
        let n_bands = weights.gamma_raw.len();
        let n_embd = n_bands * 2;
        let n_steps = weights.rk4_n_steps;
        let dt = 1.0 / n_steps as f32;

        // Split into real and imaginary parts
        let mut r = vec![0.0f32; n_bands];
        let mut s = vec![0.0f32; n_bands];
        for k in 0..n_bands {
            r[k] = x[k * 2];
            s[k] = x[k * 2 + 1];
        }

        // Wave memory injection: add offsets to initial conditions
        if let Some((r_mem, s_mem)) = memory {
            for k in 0..n_bands.min(r_mem.len()) {
                r[k] += r_mem[k];
                s[k] += s_mem[k];
            }
        }

        // Compute gamma (softplus of raw)
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

        // RK4 integration steps
        for _ in 0..n_steps {
            let (r_new, s_new) = rk4_step(&r, &s, dt, &gamma,
                                           &weights.omega, weights.alpha, weights.beta, weights.chi, &weights.rk4_weights);
            r = r_new;
            s = s_new;
        }

        // Reinterleave
        let mut out = vec![0.0f32; n_embd];
        for k in 0..n_bands {
            out[k * 2] = r[k];
            out[k * 2 + 1] = s[k];
        }
        out
    }

    /// Extract final ODE states from all layers and positions.
    /// Returns ode_states[ode_layer] = (r_avg, s_avg) averaged across positions.
    /// Used for wave memory accumulation — run once per conversation.
    pub fn extract_ode_states(
        &self,
        tokens: &[usize],
        memory_offsets: Option<&[(&[f32], &[f32])]>,
    ) -> Vec<(Vec<f32>, Vec<f32>)> {
        let t = tokens.len();
        let n_embd = self.config.n_embd();
        let n_bands = self.config.n_bands;
        assert!(t <= self.config.block_size);

        // Embedding + positional
        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; n_embd];
            for i in 0..n_embd { h[i] = self.wte_phase[tok][i] + self.wpe[pos][i]; }
            hidden.push(h);
        }

        let mut ode_states: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        let mut ode_layer = 0usize;

        for block in &self.blocks {
            // Attention + residual
            let normed_1: Vec<Vec<f32>> = hidden.iter()
                .map(|h| layer_norm(h, &block.ln_1.weight, &block.ln_1.bias))
                .collect();
            let attn_out = self.causal_self_attention(&block.attn, &normed_1);
            let mut h: Vec<Vec<f32>> = (0..t).map(|i| {
                let mut v = vec![0.0f32; n_embd];
                for j in 0..n_embd { v[j] = hidden[i][j] + attn_out[i][j]; }
                v
            }).collect();

            // FFN
            let normed_2: Vec<Vec<f32>> = h.iter()
                .map(|x| layer_norm(x, &block.ln_2.weight, &block.ln_2.bias))
                .collect();

            match &block.ffn {
                FfnWeights::PerBand(w) => {
                    let ffn_out = self.per_band_linear(w, &normed_2);
                    for i in 0..t {
                        for j in 0..n_embd { h[i][j] += ffn_out[i][j]; }
                    }
                }
                FfnWeights::KerrMaestro(w) => {
                    let mem = match memory_offsets {
                        Some(offsets) if ode_layer < offsets.len() => Some(offsets[ode_layer]),
                        _ => None,
                    };

                    // Extract ODE states from ALL positions, average them
                    let mut avg_r = vec![0.0f32; n_bands];
                    let mut avg_s = vec![0.0f32; n_bands];

                    for pos in 0..t {
                        // Run Kerr-ODE and capture final (r, s)
                        let x = &normed_2[pos];
                        let mut r = vec![0.0f32; n_bands];
                        let mut s = vec![0.0f32; n_bands];
                        for k in 0..n_bands {
                            r[k] = x[k * 2];
                            s[k] = x[k * 2 + 1];
                        }
                        if let Some((r_mem, s_mem)) = mem {
                            for k in 0..n_bands.min(r_mem.len()) {
                                r[k] += r_mem[k];
                                s[k] += s_mem[k];
                            }
                        }
                        let gamma: Vec<f32> = w.kerr.gamma_raw.iter()
                            .map(|&g| softplus(g)).collect();
                        let n_steps = w.kerr.rk4_n_steps;
                        let dt = 1.0 / n_steps as f32;
                        for _ in 0..n_steps {
                            let (r_new, s_new) = rk4_step(&r, &s, dt, &gamma,
                                &w.kerr.omega, w.kerr.alpha, w.kerr.beta, 0.0, &w.kerr.rk4_weights);
                            r = r_new;
                            s = s_new;
                        }
                        for k in 0..n_bands {
                            avg_r[k] += r[k];
                            avg_s[k] += s[k];
                        }
                    }

                    // Average across positions
                    let scale = 1.0 / t as f32;
                    for k in 0..n_bands {
                        avg_r[k] *= scale;
                        avg_s[k] *= scale;
                    }
                    ode_states.push((avg_r, avg_s));

                    // Normal forward for hidden state propagation
                    let ffn_out = self.kerr_maestro_add_with_memory(w, &normed_2, mem);
                    for i in 0..t {
                        for j in 0..n_embd { h[i][j] += ffn_out[i][j]; }
                    }

                    ode_layer += 1;
                }
                FfnWeights::KerrDualMaestro(w) => {
                    let mem = match memory_offsets {
                        Some(offsets) if ode_layer < offsets.len() => Some(offsets[ode_layer]),
                        _ => None,
                    };

                    // Extract ODE states: first apply maestro_in, then run ODE
                    let mut avg_r = vec![0.0f32; n_bands];
                    let mut avg_s = vec![0.0f32; n_bands];

                    for pos in 0..t {
                        // Apply maestro_in pre-conditioning
                        let mae_in_out = self.maestro_forward(&w.maestro_in, &normed_2[pos]);
                        let mut precond = vec![0.0f32; n_embd];
                        for i in 0..n_embd { precond[i] = normed_2[pos][i] + mae_in_out[i]; }

                        let mut r = vec![0.0f32; n_bands];
                        let mut s = vec![0.0f32; n_bands];
                        for k in 0..n_bands {
                            r[k] = precond[k * 2];
                            s[k] = precond[k * 2 + 1];
                        }
                        if let Some((r_mem, s_mem)) = mem {
                            for k in 0..n_bands.min(r_mem.len()) {
                                r[k] += r_mem[k];
                                s[k] += s_mem[k];
                            }
                        }
                        let gamma: Vec<f32> = w.kerr.gamma_raw.iter()
                            .map(|&g| softplus(g)).collect();
                        let n_steps = w.kerr.rk4_n_steps;
                        let dt = 1.0 / n_steps as f32;
                        for _ in 0..n_steps {
                            let (r_new, s_new) = rk4_step(&r, &s, dt, &gamma,
                                &w.kerr.omega, w.kerr.alpha, w.kerr.beta, 0.0, &w.kerr.rk4_weights);
                            r = r_new;
                            s = s_new;
                        }
                        for k in 0..n_bands {
                            avg_r[k] += r[k];
                            avg_s[k] += s[k];
                        }
                    }

                    let scale = 1.0 / t as f32;
                    for k in 0..n_bands {
                        avg_r[k] *= scale;
                        avg_s[k] *= scale;
                    }
                    ode_states.push((avg_r, avg_s));

                    let ffn_out = self.kerr_dual_maestro_forward_with_memory(w, &normed_2, mem);
                    for i in 0..t {
                        for j in 0..n_embd { h[i][j] += ffn_out[i][j]; }
                    }

                    ode_layer += 1;
                }
            }
            hidden = h;
        }

        ode_states
    }

    pub fn maestro_forward(&self, weights: &MaestroWeights, x: &[f32]) -> Vec<f32> {
        // Squeeze: 128 → 16
        let squeezed = linear_fn(&weights.squeeze.w, &weights.squeeze.b, x);

        // GELU activation
        let activated: Vec<f32> = squeezed.iter().map(|&v| gelu(v)).collect();

        // Process: 16 → 128
        linear_fn(&weights.process_1.w, &weights.process_1.b, &activated)
    }
}
