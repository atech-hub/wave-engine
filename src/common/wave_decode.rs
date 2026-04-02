//! Wave transduction decoder — phase coherence output scoring.
//!
//! Replaces lm_head with cos(Δθ) coherence against reference phases.
//! Frozen: reference from embedding table (85 params).
//! Unfrozen: learned reference phases (86K params).
//! All logic self-contained. Forward.rs and model_backward.rs call one-liners.

use std::f32::consts::PI;

/// All state for the wave decode layer.
pub struct WaveDecodeState {
    pub band_weights: Vec<f32>,           // [n_bands]
    pub temperature: f32,
    pub token_cos: Vec<Vec<f32>>,         // [vocab][n_bands] cached cos(ref_phase)
    pub token_sin: Vec<Vec<f32>>,         // [vocab][n_bands] cached sin(ref_phase)
    pub ref_phases: Option<Vec<Vec<f32>>>,// Some([vocab][n_bands]) if unfrozen
    pub n_bands: usize,
    pub vocab_size: usize,
}

/// Gradients for wave decode params.
pub struct WaveDecodeGrads {
    pub d_band_weights: Vec<f32>,
    pub d_temperature: f32,
    pub d_ref_phases: Option<Vec<Vec<f32>>>,
}

// ── Init ──

pub fn init_frozen(wte: &[Vec<f32>], n_bands: usize) -> WaveDecodeState {
    let vocab_size = wte.len();
    let mut token_cos = Vec::with_capacity(vocab_size);
    let mut token_sin = Vec::with_capacity(vocab_size);
    for emb in wte {
        let mut tc = Vec::with_capacity(n_bands);
        let mut ts = Vec::with_capacity(n_bands);
        for k in 0..n_bands {
            let theta = emb[k * 2 + 1].atan2(emb[k * 2]);
            tc.push(theta.cos());
            ts.push(theta.sin());
        }
        token_cos.push(tc);
        token_sin.push(ts);
    }
    WaveDecodeState {
        band_weights: vec![1.0; n_bands],
        temperature: 0.02,
        token_cos,
        token_sin,
        ref_phases: None,
        n_bands,
        vocab_size,
    }
}

pub fn init_unfrozen(wte: &[Vec<f32>], n_bands: usize) -> WaveDecodeState {
    let vocab_size = wte.len();
    let ref_phases: Vec<Vec<f32>> = wte.iter().map(|emb| {
        (0..n_bands).map(|k| emb[k * 2 + 1].atan2(emb[k * 2])).collect()
    }).collect();
    let token_cos: Vec<Vec<f32>> = ref_phases.iter()
        .map(|tp| tp.iter().map(|&p| p.cos()).collect()).collect();
    let token_sin: Vec<Vec<f32>> = ref_phases.iter()
        .map(|tp| tp.iter().map(|&p| p.sin()).collect()).collect();
    WaveDecodeState {
        band_weights: vec![1.0; n_bands],
        temperature: 0.02,
        token_cos,
        token_sin,
        ref_phases: Some(ref_phases),
        n_bands,
        vocab_size,
    }
}

// ── Cache refresh (after optimizer step, unfrozen only) ──

pub fn refresh_cos_sin_cache(state: &mut WaveDecodeState) {
    if let Some(ref phases) = state.ref_phases {
        for v in 0..state.vocab_size {
            for k in 0..state.n_bands {
                state.token_cos[v][k] = phases[v][k].cos();
                state.token_sin[v][k] = phases[v][k].sin();
            }
        }
    }
}

// ── Forward ──

pub fn forward(
    post_ln_f: &[Vec<f32>],
    state: &WaveDecodeState,
) -> Vec<Vec<f32>> {
    let n_bands = state.n_bands;
    let temp = state.temperature;
    let vocab = state.vocab_size;

    post_ln_f.iter().map(|normed| {
        // Extract hidden cos/sin and magnitudes
        let mut h_cos = vec![0.0f32; n_bands];
        let mut h_sin = vec![0.0f32; n_bands];
        let mut h_mags = vec![0.0f32; n_bands];
        for k in 0..n_bands {
            let r = normed[k * 2];
            let s = normed[k * 2 + 1];
            let theta = s.atan2(r);
            h_cos[k] = theta.cos();
            h_sin[k] = theta.sin();
            h_mags[k] = (r * r + s * s).sqrt();
        }
        // Precompute w[k] * mag[k] * cos/sin
        let mut wm_cos = vec![0.0f32; n_bands];
        let mut wm_sin = vec![0.0f32; n_bands];
        for k in 0..n_bands {
            let wm = state.band_weights[k] * h_mags[k];
            wm_cos[k] = wm * h_cos[k];
            wm_sin[k] = wm * h_sin[k];
        }
        // Score: cos expansion
        let mut logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            let mut score = 0.0f32;
            for k in 0..n_bands {
                score += wm_cos[k] * state.token_cos[v][k]
                       + wm_sin[k] * state.token_sin[v][k];
            }
            logits[v] = score * temp;
        }
        logits
    }).collect()
}

// ── Backward ──

pub fn backward(
    d_logits: &[Vec<f32>],
    post_ln_f: &[Vec<f32>],
    state: &WaveDecodeState,
) -> (Vec<Vec<f32>>, WaveDecodeGrads) {
    let n_bands = state.n_bands;
    let vocab = state.vocab_size;
    let temp = state.temperature;
    let t = d_logits.len();

    let mut d_hidden: Vec<Vec<f32>> = vec![vec![0.0f32; n_bands * 2]; t];
    let mut d_band_weights = vec![0.0f32; n_bands];
    let mut d_temperature = 0.0f32;
    let mut d_ref_phases: Option<Vec<Vec<f32>>> = if state.ref_phases.is_some() {
        Some(vec![vec![0.0f32; n_bands]; vocab])
    } else { None };

    for pos in 0..t {
        let normed = &post_ln_f[pos];
        // Extract hidden cos/sin/mag
        let mut h_cos = vec![0.0f32; n_bands];
        let mut h_sin = vec![0.0f32; n_bands];
        let mut h_mags = vec![0.0f32; n_bands];
        for k in 0..n_bands {
            let r = normed[k * 2];
            let s = normed[k * 2 + 1];
            let theta = s.atan2(r);
            h_cos[k] = theta.cos();
            h_sin[k] = theta.sin();
            h_mags[k] = (r * r + s * s).sqrt().max(1e-6);
        }

        for v in 0..vocab {
            let dl = d_logits[pos][v];
            if dl.abs() < 1e-10 { continue; }
            let dl_temp = dl * temp;

            for k in 0..n_bands {
                let cos_d = h_cos[k] * state.token_cos[v][k]
                          + h_sin[k] * state.token_sin[v][k];
                let sin_d = h_sin[k] * state.token_cos[v][k]
                          - h_cos[k] * state.token_sin[v][k];
                let w = state.band_weights[k];
                let mag = h_mags[k];
                let r = normed[k * 2];
                let s = normed[k * 2 + 1];
                let mag_sq = r * r + s * s;

                // d_band_weights
                d_band_weights[k] += dl_temp * mag * cos_d;

                // d_hidden through phase path
                let d_score_d_theta = -w * mag * sin_d;
                let d_theta_dr = -s / mag_sq.max(1e-12);
                let d_theta_ds = r / mag_sq.max(1e-12);
                // d_hidden through magnitude path
                let d_score_d_mag = w * cos_d;
                let d_mag_dr = r / mag;
                let d_mag_ds = s / mag;

                d_hidden[pos][k * 2] += dl_temp * (d_score_d_theta * d_theta_dr + d_score_d_mag * d_mag_dr);
                d_hidden[pos][k * 2 + 1] += dl_temp * (d_score_d_theta * d_theta_ds + d_score_d_mag * d_mag_ds);

                // d_ref_phases (unfrozen only)
                // d_score/d_θ_ref = w * mag * sin(θ_h - θ_ref)
                if let Some(ref mut drp) = d_ref_phases {
                    drp[v][k] += dl_temp * w * mag * sin_d;
                }
            }
        }

        // d_temperature
        let mut raw_score = 0.0f32;
        for v in 0..vocab {
            let mut score = 0.0f32;
            for k in 0..n_bands {
                score += state.band_weights[k] * h_mags[k]
                    * (h_cos[k] * state.token_cos[v][k] + h_sin[k] * state.token_sin[v][k]);
            }
            raw_score += score * d_logits[pos][v];
        }
        d_temperature += raw_score;
    }

    let grads = WaveDecodeGrads { d_band_weights, d_temperature, d_ref_phases };
    (d_hidden, grads)
}

// ── Flatten/Unflatten ──

pub fn param_count(state: &WaveDecodeState) -> usize {
    let mut n = state.band_weights.len() + 1; // band weights + temperature
    if let Some(ref rp) = state.ref_phases {
        n += rp.len() * rp[0].len(); // vocab × n_bands
    }
    n
}

pub fn flatten_params(state: &WaveDecodeState) -> Vec<f32> {
    let mut p = Vec::new();
    p.extend_from_slice(&state.band_weights);
    p.push(state.temperature);
    if let Some(ref rp) = state.ref_phases {
        for row in rp { p.extend_from_slice(row); }
    }
    p
}

pub fn unflatten_params(state: &mut WaveDecodeState, params: &[f32]) {
    let nb = state.n_bands;
    let mut idx = 0;
    state.band_weights.copy_from_slice(&params[idx..idx + nb]); idx += nb;
    state.temperature = params[idx]; idx += 1;
    if let Some(ref mut rp) = state.ref_phases {
        for row in rp.iter_mut() {
            row.copy_from_slice(&params[idx..idx + nb]); idx += nb;
        }
    }
    assert_eq!(idx, params.len());
}

pub fn flatten_grads(grads: &WaveDecodeGrads) -> Vec<f32> {
    let mut g = Vec::new();
    g.extend_from_slice(&grads.d_band_weights);
    g.push(grads.d_temperature);
    if let Some(ref drp) = grads.d_ref_phases {
        for row in drp { g.extend_from_slice(row); }
    }
    g
}
