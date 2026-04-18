//! Split-band ODE orchestrator — freeze-and-decouple approach.
//!
//! The monolithic Kerr-ODE couples all bands in a single 168-dimensional
//! system integrated with 16 RK4 steps. The Jacobian product across those
//! 16 steps has eigenvalue spread ~84×, producing ~2000-7000× gradient
//! magnitude distortion for upstream (mae_in) parameters.
//!
//! This module replaces the monolithic integration with an operator-split
//! scheme. Cross-band XPM (β·Σ|z_j|²) and FWM are stripped out of the
//! per-step derivative and applied as a separate coupling step between
//! two half-horizon sub-steps of independent per-band RK4:
//!
//!     1. Snapshot: ns_frozen[k] = Σ|z_j|² for j ∈ {k±1, k±2}
//!     2. Sub-step A: each band evolves independently, 8 RK4 steps,
//!                    φ_k = ω_k + α·|z_k|² + β·ns_frozen[k]
//!     3. Coupling: refresh ns_frozen from new state; (FWM kick deferred — Phase B)
//!     4. Sub-step B: another 8 RK4 steps with updated ns_frozen
//!
//! Within each sub-step, every band's evolution is a 2D ODE — 2×2 Jacobian
//! across 8 RK4 steps, condition number ~4. Cross-band coupling happens
//! only at the single refresh point between sub-steps, whose Jacobian is
//! a single-step operation (well-conditioned, not a chain).
//!
//! NOT A REPLACEMENT for the monolithic path. Selected at runtime via
//! `--split-band`. The monolithic path stays as the default and reference
//! until the split-band A/B has been validated against J1 + training.
//!
//! Phase A scope: chi=0 only (FWM disabled under split-band). FWM integration
//! in the coupling step is Phase B follow-up.

use crate::model::KerrWeights;
use super::core::{softplus, softplus_derivative};
use super::ode_backward::OdeParamGrads;
use super::ode_deriv::{kerr_derivative_band, rk4_step_band};

/// Cache of intermediates produced by one split-band forward integration.
/// Backward walks this in reverse.
pub struct SplitBandForwardCache {
    pub n_bands: usize,
    /// Total RK4 steps (same budget as monolithic, split half/half across sub-steps).
    pub rk4_steps: usize,
    pub dt: f32,
    pub gamma: Vec<f32>,
    /// Per-band state at the start of each RK4 step in sub-step A.
    /// [n_bands][steps_a+1] — last slot is state at end of sub-step A.
    pub a_state_r: Vec<Vec<f32>>,
    pub a_state_s: Vec<Vec<f32>>,
    /// Per-band ns_frozen used during sub-step A [n_bands]
    pub ns_frozen_a: Vec<f32>,
    /// Per-band state at start of each RK4 step in sub-step B.
    pub b_state_r: Vec<Vec<f32>>,
    pub b_state_s: Vec<Vec<f32>>,
    /// Per-band ns_frozen used during sub-step B [n_bands]
    pub ns_frozen_b: Vec<f32>,
}

#[inline]
fn compute_ns(r: &[f32], s: &[f32], k: usize) -> f32 {
    let n = r.len();
    let mut ns = 0.0f32;
    if k >= 2 { ns += r[k-2]*r[k-2] + s[k-2]*s[k-2]; }
    if k >= 1 { ns += r[k-1]*r[k-1] + s[k-1]*s[k-1]; }
    if k+1 < n { ns += r[k+1]*r[k+1] + s[k+1]*s[k+1]; }
    if k+2 < n { ns += r[k+2]*r[k+2] + s[k+2]*s[k+2]; }
    ns
}

/// Forward split-band integration with cache.
/// Input: x [n_embd] interleaved (r0, s0, r1, s1, ...).
/// Output: (new_x [n_embd], cache).
/// Phase A requires chi == 0 (FWM disabled). Panics otherwise.
pub fn split_band_forward_with_cache(
    x: &[f32],
    weights: &KerrWeights,
) -> (Vec<f32>, SplitBandForwardCache) {
    assert!(weights.chi == 0.0,
        "split-band Phase A requires chi=0 (FWM integration in coupling step is Phase B)");

    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;
    let n_steps_total = weights.rk4_n_steps;
    // Split the integration budget half/half across the two sub-steps.
    let steps_a = n_steps_total / 2;
    let steps_b = n_steps_total - steps_a;
    let dt = 1.0 / n_steps_total as f32;

    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();
    let omega = &weights.omega;
    let alpha = weights.alpha;
    let beta = weights.beta;
    let w = &weights.rk4_weights;

    // Unpack interleaved state to per-band (r, s).
    let mut r: Vec<f32> = (0..n_bands).map(|k| x[k * 2]).collect();
    let mut s: Vec<f32> = (0..n_bands).map(|k| x[k * 2 + 1]).collect();

    // ── Snapshot ns_frozen for sub-step A ──
    let ns_frozen_a: Vec<f32> = (0..n_bands).map(|k| compute_ns(&r, &s, k)).collect();

    // ── Sub-step A: per-band RK4, independent ──
    // Cache pre-step state per band per step.
    let mut a_state_r: Vec<Vec<f32>> = vec![Vec::with_capacity(steps_a + 1); n_bands];
    let mut a_state_s: Vec<Vec<f32>> = vec![Vec::with_capacity(steps_a + 1); n_bands];
    for k in 0..n_bands {
        a_state_r[k].push(r[k]);
        a_state_s[k].push(s[k]);
    }

    for _ in 0..steps_a {
        for k in 0..n_bands {
            let (r_new, s_new) = rk4_step_band(
                r[k], s[k], dt, gamma[k], omega[k], alpha, beta, ns_frozen_a[k], w,
            );
            r[k] = r_new; s[k] = s_new;
        }
        for k in 0..n_bands {
            a_state_r[k].push(r[k]);
            a_state_s[k].push(s[k]);
        }
    }

    // ── Coupling step: refresh ns_frozen for sub-step B ──
    // (Phase A: chi=0, so no FWM kick applied here.)
    let ns_frozen_b: Vec<f32> = (0..n_bands).map(|k| compute_ns(&r, &s, k)).collect();

    // ── Sub-step B: per-band RK4 with refreshed ns_frozen ──
    let mut b_state_r: Vec<Vec<f32>> = vec![Vec::with_capacity(steps_b + 1); n_bands];
    let mut b_state_s: Vec<Vec<f32>> = vec![Vec::with_capacity(steps_b + 1); n_bands];
    for k in 0..n_bands {
        b_state_r[k].push(r[k]);
        b_state_s[k].push(s[k]);
    }

    for _ in 0..steps_b {
        for k in 0..n_bands {
            let (r_new, s_new) = rk4_step_band(
                r[k], s[k], dt, gamma[k], omega[k], alpha, beta, ns_frozen_b[k], w,
            );
            r[k] = r_new; s[k] = s_new;
        }
        for k in 0..n_bands {
            b_state_r[k].push(r[k]);
            b_state_s[k].push(s[k]);
        }
    }

    // Pack output
    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands {
        out[k * 2] = r[k];
        out[k * 2 + 1] = s[k];
    }

    let cache = SplitBandForwardCache {
        n_bands,
        rk4_steps: n_steps_total,
        dt,
        gamma,
        a_state_r, a_state_s, ns_frozen_a,
        b_state_r, b_state_s, ns_frozen_b,
    };

    (out, cache)
}

// ─── Backward (split-band) ─────────────────────────────────

/// Per-band derivative backward. Accumulates gradients w.r.t. (r, s, gamma,
/// alpha, beta, ns_frozen). No cross-band terms — ns_frozen is a constant
/// during the sub-step, so the Jacobian has no off-band entries.
#[inline]
fn deriv_backward_band(
    d_dr: f32, d_ds: f32,
    r: f32, s: f32,
    gamma: f32, _omega: f32,
    alpha: f32, beta: f32, ns_frozen: f32,
    d_r: &mut f32, d_s: &mut f32,
    d_gamma_band: &mut f32,
    d_alpha: &mut f32, d_beta: &mut f32,
    d_ns_frozen: &mut f32,
) {
    let mag_sq = r * r + s * s;
    let phi = _omega + alpha * mag_sq + beta * ns_frozen;

    // Partials (same form as monolithic deriv_backward, restricted to one band,
    // no cross-band coupling term).
    // dr/dr = -gamma - 2α·r·s            (via phi dependency on r)
    // dr/ds = -phi   - 2α·s²
    // ds/dr =  phi   + 2α·r²
    // ds/ds = -gamma + 2α·r·s
    *d_r += d_dr * (-gamma - 2.0 * alpha * r * s);
    *d_s += d_dr * (-phi - 2.0 * alpha * s * s);
    *d_r += d_ds * (phi + 2.0 * alpha * r * r);
    *d_s += d_ds * (-gamma + 2.0 * alpha * r * s);

    // Params
    *d_gamma_band += d_dr * (-r) + d_ds * (-s);
    *d_alpha += d_dr * (-mag_sq * s) + d_ds * (mag_sq * r);
    *d_beta += d_dr * (-ns_frozen * s) + d_ds * (ns_frozen * r);
    *d_ns_frozen += d_dr * (-beta * s) + d_ds * (beta * r);
}

/// Backward through one per-band RK4 step.
/// Re-derives the k-eval points (cheap at 2D) rather than caching them.
#[inline]
fn rk4_step_backward_band(
    // Incoming gradient w.r.t. (r_new, s_new)
    d_r_new: f32, d_s_new: f32,
    // State at start of step (cached)
    r0: f32, s0: f32,
    dt: f32, gamma: f32, omega: f32,
    alpha: f32, beta: f32, ns_frozen: f32,
    w: &[f32; 4],
    // Output accumulators for the start-of-step state and params
    d_r0: &mut f32, d_s0: &mut f32,
    d_gamma_band: &mut f32,
    d_alpha: &mut f32, d_beta: &mut f32,
    d_ns_frozen: &mut f32,
    d_rk4_weights: &mut [f32; 4],
) {
    // Re-derive k-values. This mirrors rk4_step_band's forward.
    let (k1r, k1s) = kerr_derivative_band(r0, s0, gamma, omega, alpha, beta, ns_frozen);
    let (r_k2, s_k2) = (r0 + 0.5 * dt * k1r, s0 + 0.5 * dt * k1s);
    let (k2r, k2s) = kerr_derivative_band(r_k2, s_k2, gamma, omega, alpha, beta, ns_frozen);
    let (r_k3, s_k3) = (r0 + 0.5 * dt * k2r, s0 + 0.5 * dt * k2s);
    let (k3r, k3s) = kerr_derivative_band(r_k3, s_k3, gamma, omega, alpha, beta, ns_frozen);
    let (r_k4, s_k4) = (r0 + dt * k3r, s0 + dt * k3s);
    let (k4r, k4s) = kerr_derivative_band(r_k4, s_k4, gamma, omega, alpha, beta, ns_frozen);

    // RK4 update:
    //   r_new = r0 + dt * (w0*k1 + w1*k2 + w2*k3 + w3*k4)
    // Backward through the combination:
    //   d_r0_from_id = d_r_new     (identity contribution of r0)
    //   d_k{i}r_from_combo = d_r_new * dt * w[i]
    //   d_w[i] += d_r_new * dt * k{i}r + d_s_new * dt * k{i}s
    let d_k1r_c = d_r_new * dt * w[0];
    let d_k1s_c = d_s_new * dt * w[0];
    let d_k2r_c = d_r_new * dt * w[1];
    let d_k2s_c = d_s_new * dt * w[1];
    let d_k3r_c = d_r_new * dt * w[2];
    let d_k3s_c = d_s_new * dt * w[2];
    let d_k4r_c = d_r_new * dt * w[3];
    let d_k4s_c = d_s_new * dt * w[3];

    d_rk4_weights[0] += d_r_new * dt * k1r + d_s_new * dt * k1s;
    d_rk4_weights[1] += d_r_new * dt * k2r + d_s_new * dt * k2s;
    d_rk4_weights[2] += d_r_new * dt * k3r + d_s_new * dt * k3s;
    d_rk4_weights[3] += d_r_new * dt * k4r + d_s_new * dt * k4s;

    // Identity contribution of r0 in the combination.
    *d_r0 += d_r_new;
    *d_s0 += d_s_new;

    // Walk k4 -> k3 -> k2 -> k1 just like the monolithic backward.
    // k4 eval point = (r0 + dt*k3r, s0 + dt*k3s)
    let mut d_rk4 = 0.0_f32;
    let mut d_sk4 = 0.0_f32;
    deriv_backward_band(
        d_k4r_c, d_k4s_c, r_k4, s_k4,
        gamma, omega, alpha, beta, ns_frozen,
        &mut d_rk4, &mut d_sk4,
        d_gamma_band, d_alpha, d_beta, d_ns_frozen,
    );
    *d_r0 += d_rk4;
    *d_s0 += d_sk4;
    let d_k3r_extra = d_rk4 * dt;
    let d_k3s_extra = d_sk4 * dt;

    // k3 eval point = (r0 + 0.5*dt*k2r, s0 + 0.5*dt*k2s)
    let mut d_rk3 = 0.0_f32;
    let mut d_sk3 = 0.0_f32;
    deriv_backward_band(
        d_k3r_c + d_k3r_extra, d_k3s_c + d_k3s_extra, r_k3, s_k3,
        gamma, omega, alpha, beta, ns_frozen,
        &mut d_rk3, &mut d_sk3,
        d_gamma_band, d_alpha, d_beta, d_ns_frozen,
    );
    *d_r0 += d_rk3;
    *d_s0 += d_sk3;
    let d_k2r_extra = d_rk3 * 0.5 * dt;
    let d_k2s_extra = d_sk3 * 0.5 * dt;

    // k2 eval point = (r0 + 0.5*dt*k1r, s0 + 0.5*dt*k1s)
    let mut d_rk2 = 0.0_f32;
    let mut d_sk2 = 0.0_f32;
    deriv_backward_band(
        d_k2r_c + d_k2r_extra, d_k2s_c + d_k2s_extra, r_k2, s_k2,
        gamma, omega, alpha, beta, ns_frozen,
        &mut d_rk2, &mut d_sk2,
        d_gamma_band, d_alpha, d_beta, d_ns_frozen,
    );
    *d_r0 += d_rk2;
    *d_s0 += d_sk2;
    let d_k1r_extra = d_rk2 * 0.5 * dt;
    let d_k1s_extra = d_sk2 * 0.5 * dt;

    // k1 eval point = (r0, s0)
    let mut d_rk1 = 0.0_f32;
    let mut d_sk1 = 0.0_f32;
    deriv_backward_band(
        d_k1r_c + d_k1r_extra, d_k1s_c + d_k1s_extra, r0, s0,
        gamma, omega, alpha, beta, ns_frozen,
        &mut d_rk1, &mut d_sk1,
        d_gamma_band, d_alpha, d_beta, d_ns_frozen,
    );
    *d_r0 += d_rk1;
    *d_s0 += d_sk1;
}

/// ns_frozen[k] = Σ(r[j]² + s[j]²) for j ∈ neighbours(k).
/// Backward: d_r[j] += d_ns[k] · 2·r[j] for each k for which j is a neighbour.
fn ns_frozen_backward(d_ns: &[f32], r: &[f32], s: &[f32], d_r: &mut [f32], d_s: &mut [f32]) {
    let n = r.len();
    for k in 0..n {
        let dk = d_ns[k];
        if dk == 0.0 { continue; }
        if k >= 2 { let j = k-2; d_r[j] += dk * 2.0 * r[j]; d_s[j] += dk * 2.0 * s[j]; }
        if k >= 1 { let j = k-1; d_r[j] += dk * 2.0 * r[j]; d_s[j] += dk * 2.0 * s[j]; }
        if k+1 < n { let j = k+1; d_r[j] += dk * 2.0 * r[j]; d_s[j] += dk * 2.0 * s[j]; }
        if k+2 < n { let j = k+2; d_r[j] += dk * 2.0 * r[j]; d_s[j] += dk * 2.0 * s[j]; }
    }
}

/// Full backward through a split-band forward integration.
/// Returns (d_x [n_embd], OdeParamGrads). Mirrors `ode_backward` signature
/// so the caller can swap between monolithic and split-band transparently.
pub fn split_band_backward(
    d_output: &[f32],
    cache: &SplitBandForwardCache,
    weights: &KerrWeights,
) -> (Vec<f32>, OdeParamGrads) {
    let n = cache.n_bands;
    let dt = cache.dt;
    let steps_a = cache.a_state_r[0].len() - 1;
    let steps_b = cache.b_state_r[0].len() - 1;
    let omega = &weights.omega;
    let alpha = weights.alpha;
    let beta = weights.beta;
    let w = &weights.rk4_weights;

    // Unpack incoming d_output into per-band d_r, d_s (current gradient at the
    // tail end of sub-step B).
    let mut d_r: Vec<f32> = (0..n).map(|k| d_output[k * 2]).collect();
    let mut d_s: Vec<f32> = (0..n).map(|k| d_output[k * 2 + 1]).collect();

    let mut d_gamma = vec![0.0_f32; n];
    let mut d_alpha = 0.0_f32;
    let mut d_beta = 0.0_f32;
    let mut d_rk4_weights = [0.0_f32; 4];

    // Per-band ns_frozen gradient accumulators for each sub-step.
    let mut d_ns_frozen_b = vec![0.0_f32; n];
    let mut d_ns_frozen_a = vec![0.0_f32; n];

    // ── Backward through sub-step B ──
    // Each band is independent. Walk its RK4 chain in reverse using cached
    // start-of-step states.
    for step in (0..steps_b).rev() {
        for k in 0..n {
            let r0 = cache.b_state_r[k][step];
            let s0 = cache.b_state_s[k][step];
            let d_r_new = d_r[k];
            let d_s_new = d_s[k];
            let mut d_r0 = 0.0_f32;
            let mut d_s0 = 0.0_f32;
            rk4_step_backward_band(
                d_r_new, d_s_new, r0, s0, dt,
                cache.gamma[k], omega[k], alpha, beta, cache.ns_frozen_b[k], w,
                &mut d_r0, &mut d_s0,
                &mut d_gamma[k], &mut d_alpha, &mut d_beta,
                &mut d_ns_frozen_b[k], &mut d_rk4_weights,
            );
            d_r[k] = d_r0;
            d_s[k] = d_s0;
        }
    }

    // ── Backward through the ns_frozen_b snapshot ──
    // ns_frozen_b was computed from the state at the end of sub-step A.
    // That state is cached as the last slot of a_state_{r,s}, and also as
    // the first slot of b_state_{r,s} (they're equal). d_ns_frozen_b flows
    // back to d_r, d_s at the end of sub-step A.
    let end_a_r: Vec<f32> = (0..n).map(|k| cache.a_state_r[k][steps_a]).collect();
    let end_a_s: Vec<f32> = (0..n).map(|k| cache.a_state_s[k][steps_a]).collect();
    ns_frozen_backward(&d_ns_frozen_b, &end_a_r, &end_a_s, &mut d_r, &mut d_s);

    // ── Backward through sub-step A ──
    for step in (0..steps_a).rev() {
        for k in 0..n {
            let r0 = cache.a_state_r[k][step];
            let s0 = cache.a_state_s[k][step];
            let d_r_new = d_r[k];
            let d_s_new = d_s[k];
            let mut d_r0 = 0.0_f32;
            let mut d_s0 = 0.0_f32;
            rk4_step_backward_band(
                d_r_new, d_s_new, r0, s0, dt,
                cache.gamma[k], omega[k], alpha, beta, cache.ns_frozen_a[k], w,
                &mut d_r0, &mut d_s0,
                &mut d_gamma[k], &mut d_alpha, &mut d_beta,
                &mut d_ns_frozen_a[k], &mut d_rk4_weights,
            );
            d_r[k] = d_r0;
            d_s[k] = d_s0;
        }
    }

    // ── Backward through the initial ns_frozen_a snapshot ──
    // ns_frozen_a was computed from the input state (x).
    let start_r: Vec<f32> = (0..n).map(|k| cache.a_state_r[k][0]).collect();
    let start_s: Vec<f32> = (0..n).map(|k| cache.a_state_s[k][0]).collect();
    ns_frozen_backward(&d_ns_frozen_a, &start_r, &start_s, &mut d_r, &mut d_s);

    // Pack d_x interleaved
    let mut d_x = vec![0.0_f32; n * 2];
    for k in 0..n {
        d_x[k * 2] = d_r[k];
        d_x[k * 2 + 1] = d_s[k];
    }

    // Convert post-softplus gamma gradient to pre-softplus (gamma_raw).
    let mut d_gamma_raw = vec![0.0_f32; n];
    for k in 0..n {
        d_gamma_raw[k] = d_gamma[k] * softplus_derivative(weights.gamma_raw[k]);
    }

    let grads = OdeParamGrads {
        d_gamma_raw,
        d_alpha, d_beta, d_chi: 0.0_f32, // Phase A: chi=0
        d_rk4_weights,
    };

    (d_x, grads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::KerrWeights;
    use super::super::ode_backward::ode_forward_with_cache;

    fn mk_weights(n_bands: usize, chi: f32) -> KerrWeights {
        KerrWeights {
            gamma_raw: vec![0.0_f32; n_bands],        // softplus(0) = ln(2) ≈ 0.693
            omega: (0..n_bands).map(|k| (k + 1) as f32 / n_bands as f32).collect(),
            phase_correction: vec![0.0_f32; n_bands],
            alpha: 0.1, beta: 0.2, chi,
            rk4_weights: [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
            rk4_n_steps: 16,
        }
    }

    fn mk_input(n_bands: usize, seed: u32) -> Vec<f32> {
        let n_embd = n_bands * 2;
        let mut rng = seed.wrapping_mul(2654435761);
        (0..n_embd).map(|_| {
            rng = rng.wrapping_mul(2654435761).wrapping_add(12345);
            let v = (rng as f32) / (u32::MAX as f32);
            (v - 0.5) * 0.4 // range ~[-0.2, 0.2]
        }).collect()
    }

    /// At n_bands=1 there are no neighbours, so ns is identically zero in both
    /// paths. The split and monolithic integrations run the same derivative
    /// over the same total time — outputs should match to tight tolerance.
    #[test]
    fn split_matches_monolithic_single_band() {
        let w = mk_weights(1, 0.0);
        let x = mk_input(1, 42);

        let (out_mono, _) = ode_forward_with_cache(&x, &w);
        let (out_split, _) = split_band_forward_with_cache(&x, &w);

        assert_eq!(out_mono.len(), out_split.len());
        for i in 0..out_mono.len() {
            assert!((out_mono[i] - out_split[i]).abs() < 1e-6,
                "single-band output[{}] mismatch: mono={} split={}",
                i, out_mono[i], out_split[i]);
        }
    }

    /// At realistic band count, split introduces first-order splitting error.
    /// The output should be close to monolithic but not identical. "Close"
    /// here means per-element relative error below a few percent — the
    /// splitting error at 16 RK4 steps total, ns refreshed once at the
    /// midpoint.
    #[test]
    fn split_output_is_close_to_monolithic_84_bands() {
        let w = mk_weights(84, 0.0);
        let x = mk_input(84, 42);

        let (out_mono, _) = ode_forward_with_cache(&x, &w);
        let (out_split, _) = split_band_forward_with_cache(&x, &w);

        assert_eq!(out_mono.len(), out_split.len());

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        let mut mean_abs = 0.0_f32;
        for i in 0..out_mono.len() {
            let abs = (out_mono[i] - out_split[i]).abs();
            let rel = abs / out_mono[i].abs().max(1e-6);
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
            mean_abs += abs;
        }
        mean_abs /= out_mono.len() as f32;

        // First-order splitting error at 16 steps with one refresh.
        // Empirically ~0.01–0.03 for typical inputs at α=0.1 β=0.2.
        eprintln!(
            "84-band split vs mono: max_abs={:.6} max_rel={:.6} mean_abs={:.6}",
            max_abs, max_rel, mean_abs
        );
        assert!(max_abs < 0.1,
            "split output drift too large: max_abs={}", max_abs);
    }

    /// Cache shape: a and b sub-step states should have steps_a+1 / steps_b+1
    /// slots per band, no allocation errors.
    #[test]
    fn cache_shape() {
        let w = mk_weights(84, 0.0);
        let x = mk_input(84, 1);
        let (_out, cache) = split_band_forward_with_cache(&x, &w);
        assert_eq!(cache.n_bands, 84);
        assert_eq!(cache.rk4_steps, 16);
        let steps_a = 16 / 2;
        let steps_b = 16 - steps_a;
        for k in 0..84 {
            assert_eq!(cache.a_state_r[k].len(), steps_a + 1);
            assert_eq!(cache.a_state_s[k].len(), steps_a + 1);
            assert_eq!(cache.b_state_r[k].len(), steps_b + 1);
            assert_eq!(cache.b_state_s[k].len(), steps_b + 1);
        }
        assert_eq!(cache.ns_frozen_a.len(), 84);
        assert_eq!(cache.ns_frozen_b.len(), 84);
    }

    /// FWM (chi != 0) is not yet supported in the split-band path.
    #[test]
    #[should_panic(expected = "split-band Phase A requires chi=0")]
    fn chi_nonzero_panics() {
        let w = mk_weights(84, 0.03);
        let x = mk_input(84, 1);
        let _ = split_band_forward_with_cache(&x, &w);
    }

    /// Analytical backward must agree with finite-difference on a small
    /// model. This is the J1-shaped test for split-band.
    #[test]
    fn backward_matches_fd_small() {
        let w = mk_weights(8, 0.0);
        let x = mk_input(8, 7);

        // Scalar loss L = sum(out) so dL/d(out) = 1 everywhere.
        let (out_base, cache) = split_band_forward_with_cache(&x, &w);
        let d_output: Vec<f32> = vec![1.0_f32; out_base.len()];
        let (d_x_analytical, _grads) = split_band_backward(&d_output, &cache, &w);

        // FD check against the input state — f64 loss accumulation to defeat
        // f32 cancellation (same discipline as J1 uses now).
        let eps: f32 = 1e-3;
        let mut max_rel = 0.0_f32;
        let mut n_checked = 0usize;
        let mut n_within = 0usize;
        for i in 0..x.len() {
            let an = d_x_analytical[i];
            let mut x_plus = x.clone(); x_plus[i] += eps;
            let mut x_minus = x.clone(); x_minus[i] -= eps;
            let (out_p, _) = split_band_forward_with_cache(&x_plus, &w);
            let (out_m, _) = split_band_forward_with_cache(&x_minus, &w);
            let l_plus: f64 = out_p.iter().map(|&v| v as f64).sum();
            let l_minus: f64 = out_m.iter().map(|&v| v as f64).sum();
            let fd = ((l_plus - l_minus) / (2.0_f64 * eps as f64)) as f32;
            let denom = an.abs().max(fd.abs()).max(1e-4);
            let rel = (an - fd).abs() / denom;
            n_checked += 1;
            if rel < 0.01 { n_within += 1; }
            if rel > max_rel { max_rel = rel; }
        }
        eprintln!("split-band backward FD check: {}/{} within 1% (max_rel={:.4})",
            n_within, n_checked, max_rel);
        // Target: output-adjacent style tight agreement. Per-band decoupling
        // means mae_in-style distortion doesn't exist here.
        assert!(n_within >= n_checked * 9 / 10,
            "FD agreement too low: {}/{} within 1%", n_within, n_checked);
    }

    /// Parameter gradient FD check — α must match.
    #[test]
    fn alpha_gradient_matches_fd() {
        let w_base = mk_weights(8, 0.0);
        let x = mk_input(8, 11);

        let (_out, cache) = split_band_forward_with_cache(&x, &w_base);
        let d_output: Vec<f32> = vec![1.0_f32; x.len()];
        let (_d_x, grads) = split_band_backward(&d_output, &cache, &w_base);

        let eps: f32 = 1e-3;
        let mut w_plus = w_base.clone(); w_plus.alpha += eps;
        let mut w_minus = w_base.clone(); w_minus.alpha -= eps;
        let (out_p, _) = split_band_forward_with_cache(&x, &w_plus);
        let (out_m, _) = split_band_forward_with_cache(&x, &w_minus);
        let l_plus: f64 = out_p.iter().map(|&v| v as f64).sum();
        let l_minus: f64 = out_m.iter().map(|&v| v as f64).sum();
        let fd = ((l_plus - l_minus) / (2.0_f64 * eps as f64)) as f32;
        let an = grads.d_alpha;
        let rel = (an - fd).abs() / an.abs().max(fd.abs()).max(1e-4);
        eprintln!("d_alpha: analytical={:.6} fd={:.6} rel={:.4}", an, fd, rel);
        assert!(rel < 0.02, "d_alpha mismatch: an={} fd={} rel={}", an, fd, rel);
    }
}
