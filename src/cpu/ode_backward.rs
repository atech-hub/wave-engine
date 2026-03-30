//! ODE backward pass — gradient flow through Kerr-ODE RK4 integration.
//!
//! Replaces the identity pass-through (d_precond = d_kerr_out) with proper
//! backpropagation through 16 RK4 steps. Enables learnable α, β, γ per layer.
//!
//! Architecture: stores forward intermediates in OdeForwardCache during the
//! forward pass, then walks backward through the RK4 steps computing gradients
//! via the chain rule through the Kerr nonlinearity Jacobian.

use crate::model::KerrWeights;

// ─── Forward cache ─────────────────────────────────────────

/// All intermediate states from one forward ODE integration.
/// Stored per-position so the backward can reconstruct each RK4 step.
pub struct OdeForwardCache {
    pub n_bands: usize,
    pub rk4_steps: usize,
    pub dt: f32,
    pub gamma: Vec<f32>,              // softplus(gamma_raw), precomputed [n_bands]
    /// State at the start of each RK4 step [rk4_steps][n_bands]
    pub r_at_step: Vec<Vec<f32>>,
    pub s_at_step: Vec<Vec<f32>>,
    /// k-values for each RK4 step [rk4_steps][4][n_bands]
    pub kr: Vec<[Vec<f32>; 4]>,
    pub ks: Vec<[Vec<f32>; 4]>,
}

/// Gradients for ODE parameters from one position.
pub struct OdeParamGrads {
    pub d_gamma_raw: Vec<f32>,  // [n_bands]
    pub d_alpha: f32,
    pub d_beta: f32,
}

// ─── Helpers ───────────────────────────────────────────────

fn softplus(v: f32) -> f32 {
    if v > 20.0 { v } else { (1.0 + v.exp()).ln() }
}

fn softplus_derivative(v: f32) -> f32 {
    // d/dx softplus(x) = sigmoid(x) = 1 / (1 + exp(-x))
    if v > 20.0 { 1.0 } else { 1.0 / (1.0 + (-v).exp()) }
}

/// Compute neighbour sum of mag_sq for band k (stencil ±2).
fn neighbour_sum(r: &[f32], s: &[f32], k: usize) -> f32 {
    let n = r.len();
    let mut ns = 0.0f32;
    if k >= 2 { ns += r[k-2]*r[k-2] + s[k-2]*s[k-2]; }
    if k >= 1 { ns += r[k-1]*r[k-1] + s[k-1]*s[k-1]; }
    if k+1 < n { ns += r[k+1]*r[k+1] + s[k+1]*s[k+1]; }
    if k+2 < n { ns += r[k+2]*r[k+2] + s[k+2]*s[k+2]; }
    ns
}

/// Compute the ODE derivative for all bands.
/// Returns (dr, ds) where dr[k] = -gamma[k]*r[k] - phi[k]*s[k], etc.
fn deriv(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();
    let mut dr = vec![0.0f32; n];
    let mut ds = vec![0.0f32; n];
    for k in 0..n {
        let mag_sq = r[k]*r[k] + s[k]*s[k];
        let ns = neighbour_sum(r, s, k);
        let phi = omega[k] + alpha * mag_sq + beta * ns;
        dr[k] = -gamma[k] * r[k] - phi * s[k];
        ds[k] = -gamma[k] * s[k] + phi * r[k];
    }
    (dr, ds)
}

// ─── Forward with cache ────────────────────────────────────

/// RK4 forward pass that stores all intermediates for backward.
/// Input: x [n_embd] interleaved (r0,s0,r1,s1,...).
/// Output: (out [n_embd], cache).
pub fn ode_forward_with_cache(
    x: &[f32],
    weights: &KerrWeights,
) -> (Vec<f32>, OdeForwardCache) {
    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;
    let n_steps = weights.rk4_n_steps;
    let dt = 1.0 / n_steps as f32;

    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

    let mut r: Vec<f32> = (0..n_bands).map(|k| x[k * 2]).collect();
    let mut s: Vec<f32> = (0..n_bands).map(|k| x[k * 2 + 1]).collect();

    let mut r_at_step = Vec::with_capacity(n_steps);
    let mut s_at_step = Vec::with_capacity(n_steps);
    let mut all_kr = Vec::with_capacity(n_steps);
    let mut all_ks = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        // Save state at start of this step
        r_at_step.push(r.clone());
        s_at_step.push(s.clone());

        // k1 at (r, s)
        let (k1r, k1s) = deriv(&r, &s, &gamma, &weights.omega, weights.alpha, weights.beta);

        // k2 at (r + 0.5*dt*k1, s + 0.5*dt*k1)
        let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&a, &b)| a + 0.5*dt*b).collect();
        let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&a, &b)| a + 0.5*dt*b).collect();
        let (k2r, k2s) = deriv(&r2, &s2, &gamma, &weights.omega, weights.alpha, weights.beta);

        // k3 at (r + 0.5*dt*k2, s + 0.5*dt*k2)
        let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&a, &b)| a + 0.5*dt*b).collect();
        let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&a, &b)| a + 0.5*dt*b).collect();
        let (k3r, k3s) = deriv(&r3, &s3, &gamma, &weights.omega, weights.alpha, weights.beta);

        // k4 at (r + dt*k3, s + dt*k3)
        let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&a, &b)| a + dt*b).collect();
        let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&a, &b)| a + dt*b).collect();
        let (k4r, k4s) = deriv(&r4, &s4, &gamma, &weights.omega, weights.alpha, weights.beta);

        // Store k-values
        all_kr.push([k1r.clone(), k2r.clone(), k3r.clone(), k4r.clone()]);
        all_ks.push([k1s.clone(), k2s.clone(), k3s.clone(), k4s.clone()]);

        // Update state
        let r_new: Vec<f32> = (0..n_bands).map(|i| r[i] + dt/6.0 * (k1r[i] + 2.0*k2r[i] + 2.0*k3r[i] + k4r[i])).collect();
        let s_new: Vec<f32> = (0..n_bands).map(|i| s[i] + dt/6.0 * (k1s[i] + 2.0*k2s[i] + 2.0*k3s[i] + k4s[i])).collect();
        r = r_new;
        s = s_new;
    }

    // Pack output
    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands {
        out[k * 2] = r[k];
        out[k * 2 + 1] = s[k];
    }

    let cache = OdeForwardCache {
        n_bands,
        rk4_steps: n_steps,
        dt,
        gamma,
        r_at_step,
        s_at_step,
        kr: all_kr,
        ks: all_ks,
    };

    (out, cache)
}

// ─── Backward through derivative ───────────────────────────

/// Backward through one evaluation of deriv() at state (r, s).
/// Given d_dr[k] and d_ds[k] (gradients of loss w.r.t. dr[k] and ds[k]),
/// computes gradients w.r.t. r, s, gamma, alpha, beta.
///
/// This is the core Jacobian transpose multiplication.
fn deriv_backward(
    d_dr: &[f32], d_ds: &[f32],     // incoming gradients [n_bands]
    r: &[f32], s: &[f32],           // state at which deriv was evaluated
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
    // Outputs (accumulated into):
    d_r: &mut [f32], d_s: &mut [f32],
    d_gamma: &mut [f32],
    d_alpha: &mut f32, d_beta: &mut f32,
) {
    let n = r.len();

    for k in 0..n {
        let rk = r[k];
        let sk = s[k];
        let mag_sq = rk*rk + sk*sk;
        let ns = neighbour_sum(r, s, k);
        let phi = omega[k] + alpha * mag_sq + beta * ns;

        // Gradient from dr[k] = -gamma[k]*r[k] - phi*s[k]
        // d_r[k] += d_dr[k] * d(dr_k)/d(r_k) = d_dr[k] * (-gamma[k] - 2*alpha*r_k*s_k)
        // d_s[k] += d_dr[k] * d(dr_k)/d(s_k) = d_dr[k] * (-phi - 2*alpha*s_k^2)
        let ddr = d_dr[k];
        let dds = d_ds[k];

        // ── Self-band gradients ──

        // From dr[k]:
        //   d(dr_k)/d(r_k) = -gamma_k - 2*alpha*r_k*s_k
        //   d(dr_k)/d(s_k) = -phi - 2*alpha*s_k*s_k
        d_r[k] += ddr * (-gamma[k] - 2.0 * alpha * rk * sk);
        d_s[k] += ddr * (-phi - 2.0 * alpha * sk * sk);

        // From ds[k]:
        //   d(ds_k)/d(r_k) = phi + 2*alpha*r_k*r_k
        //   d(ds_k)/d(s_k) = -gamma_k + 2*alpha*r_k*s_k
        d_r[k] += dds * (phi + 2.0 * alpha * rk * rk);
        d_s[k] += dds * (-gamma[k] + 2.0 * alpha * rk * sk);

        // ── Cross-band gradients (neighbour coupling) ──
        // d(dr_k)/d(r_j) = -2*beta*r_j*s_k  for j in neighbours(k)
        // d(dr_k)/d(s_j) = -2*beta*s_j*s_k  for j in neighbours(k)
        // d(ds_k)/d(r_j) = 2*beta*r_j*r_k   for j in neighbours(k)
        // d(ds_k)/d(s_j) = 2*beta*s_j*r_k   for j in neighbours(k)
        let neighbours: &[usize] = &[
            k.wrapping_sub(2), k.wrapping_sub(1), k + 1, k + 2
        ];
        for &j in neighbours {
            if j < n {
                d_r[j] += ddr * (-2.0 * beta * r[j] * sk);
                d_s[j] += ddr * (-2.0 * beta * s[j] * sk);
                d_r[j] += dds * (2.0 * beta * r[j] * rk);
                d_s[j] += dds * (2.0 * beta * s[j] * rk);
            }
        }

        // ── Parameter gradients ──

        // d(dr_k)/d(gamma_k) = -r_k
        // d(ds_k)/d(gamma_k) = -s_k
        d_gamma[k] += ddr * (-rk) + dds * (-sk);

        // d(dr_k)/d(alpha) = -(r_k^2 + s_k^2) * s_k = -mag_sq * s_k
        // d(ds_k)/d(alpha) = (r_k^2 + s_k^2) * r_k = mag_sq * r_k
        *d_alpha += ddr * (-mag_sq * sk) + dds * (mag_sq * rk);

        // d(dr_k)/d(beta) = -ns * s_k
        // d(ds_k)/d(beta) = ns * r_k
        *d_beta += ddr * (-ns * sk) + dds * (ns * rk);
    }
}

// ─── Backward through full RK4 integration ─────────────────

/// Compute gradients through the full RK4 integration.
///
/// d_output: gradient of loss w.r.t. ODE output [n_embd] interleaved (r0,s0,r1,s1,...)
/// cache: intermediates from ode_forward_with_cache
/// weights: ODE parameters (for omega, gamma_raw, alpha, beta)
///
/// Returns: (d_input [n_embd], OdeParamGrads)
pub fn ode_backward(
    d_output: &[f32],
    cache: &OdeForwardCache,
    weights: &KerrWeights,
) -> (Vec<f32>, OdeParamGrads) {
    let n = cache.n_bands;
    let dt = cache.dt;

    // Unpack d_output into d_r, d_s
    let mut d_r: Vec<f32> = (0..n).map(|k| d_output[k * 2]).collect();
    let mut d_s: Vec<f32> = (0..n).map(|k| d_output[k * 2 + 1]).collect();

    // Accumulate parameter gradients
    let mut d_gamma = vec![0.0f32; n];   // gradient w.r.t. gamma (post-softplus)
    let mut d_alpha = 0.0f32;
    let mut d_beta = 0.0f32;

    // Walk backward through RK4 steps
    for step in (0..cache.rk4_steps).rev() {
        let r0 = &cache.r_at_step[step];
        let s0 = &cache.s_at_step[step];
        let [ref k1r, ref k2r, ref k3r, ref k4r] = cache.kr[step];
        let [ref k1s, ref k2s, ref k3s, ref k4s] = cache.ks[step];

        // The RK4 update: r_new = r0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        // Backward:
        //   d_r0 += d_r_new  (identity from r0 in the sum)
        //   d_k1r = d_r_new * dt/6
        //   d_k2r = d_r_new * dt/3  (factor of 2)
        //   d_k3r = d_r_new * dt/3
        //   d_k4r = d_r_new * dt/6

        let d_k1r: Vec<f32> = d_r.iter().map(|&v| v * dt / 6.0).collect();
        let d_k1s: Vec<f32> = d_s.iter().map(|&v| v * dt / 6.0).collect();
        let d_k2r: Vec<f32> = d_r.iter().map(|&v| v * dt / 3.0).collect();
        let d_k2s: Vec<f32> = d_s.iter().map(|&v| v * dt / 3.0).collect();
        let d_k3r: Vec<f32> = d_r.iter().map(|&v| v * dt / 3.0).collect();
        let d_k3s: Vec<f32> = d_s.iter().map(|&v| v * dt / 3.0).collect();
        let d_k4r: Vec<f32> = d_r.iter().map(|&v| v * dt / 6.0).collect();
        let d_k4s: Vec<f32> = d_s.iter().map(|&v| v * dt / 6.0).collect();

        // d_r0 already gets d_r_new (identity contribution)
        // (d_r already holds d_r_new, which is the right starting point for d_r0)

        // ── k4 backward ──
        // k4 was evaluated at (r0 + dt*k3r, s0 + dt*k3s)
        let r_k4: Vec<f32> = r0.iter().zip(k3r).map(|(&a, &b)| a + dt * b).collect();
        let s_k4: Vec<f32> = s0.iter().zip(k3s).map(|(&a, &b)| a + dt * b).collect();
        let mut d_r_k4 = vec![0.0f32; n];
        let mut d_s_k4 = vec![0.0f32; n];
        deriv_backward(
            &d_k4r, &d_k4s, &r_k4, &s_k4,
            &cache.gamma, &weights.omega, weights.alpha, weights.beta,
            &mut d_r_k4, &mut d_s_k4, &mut d_gamma, &mut d_alpha, &mut d_beta,
        );
        // d_r_k4, d_s_k4 are gradients w.r.t. the k4 evaluation point
        // k4 eval point = r0 + dt*k3r, so:
        //   d_r0 += d_r_k4
        //   d_k3r += d_r_k4 * dt
        for i in 0..n {
            d_r[i] += d_r_k4[i];
            d_s[i] += d_s_k4[i];
        }
        let mut d_k3r_extra: Vec<f32> = d_r_k4.iter().map(|&v| v * dt).collect();
        let mut d_k3s_extra: Vec<f32> = d_s_k4.iter().map(|&v| v * dt).collect();

        // ── k3 backward ──
        // k3 was evaluated at (r0 + 0.5*dt*k2r, s0 + 0.5*dt*k2s)
        let r_k3: Vec<f32> = r0.iter().zip(k2r).map(|(&a, &b)| a + 0.5 * dt * b).collect();
        let s_k3: Vec<f32> = s0.iter().zip(k2s).map(|(&a, &b)| a + 0.5 * dt * b).collect();
        // Total d_k3 = d_k3r from RK4 weights + d_k3r_extra from k4's dependence on k3
        let total_d_k3r: Vec<f32> = d_k3r.iter().zip(&d_k3r_extra).map(|(&a, &b)| a + b).collect();
        let total_d_k3s: Vec<f32> = d_k3s.iter().zip(&d_k3s_extra).map(|(&a, &b)| a + b).collect();
        let mut d_r_k3 = vec![0.0f32; n];
        let mut d_s_k3 = vec![0.0f32; n];
        deriv_backward(
            &total_d_k3r, &total_d_k3s, &r_k3, &s_k3,
            &cache.gamma, &weights.omega, weights.alpha, weights.beta,
            &mut d_r_k3, &mut d_s_k3, &mut d_gamma, &mut d_alpha, &mut d_beta,
        );
        // k3 eval point = r0 + 0.5*dt*k2r, so:
        //   d_r0 += d_r_k3
        //   d_k2r += d_r_k3 * 0.5*dt
        for i in 0..n {
            d_r[i] += d_r_k3[i];
            d_s[i] += d_s_k3[i];
        }
        let d_k2r_extra: Vec<f32> = d_r_k3.iter().map(|&v| v * 0.5 * dt).collect();
        let d_k2s_extra: Vec<f32> = d_s_k3.iter().map(|&v| v * 0.5 * dt).collect();

        // ── k2 backward ──
        // k2 was evaluated at (r0 + 0.5*dt*k1r, s0 + 0.5*dt*k1s)
        let r_k2: Vec<f32> = r0.iter().zip(k1r).map(|(&a, &b)| a + 0.5 * dt * b).collect();
        let s_k2: Vec<f32> = s0.iter().zip(k1s).map(|(&a, &b)| a + 0.5 * dt * b).collect();
        let total_d_k2r: Vec<f32> = d_k2r.iter().zip(&d_k2r_extra).map(|(&a, &b)| a + b).collect();
        let total_d_k2s: Vec<f32> = d_k2s.iter().zip(&d_k2s_extra).map(|(&a, &b)| a + b).collect();
        let mut d_r_k2 = vec![0.0f32; n];
        let mut d_s_k2 = vec![0.0f32; n];
        deriv_backward(
            &total_d_k2r, &total_d_k2s, &r_k2, &s_k2,
            &cache.gamma, &weights.omega, weights.alpha, weights.beta,
            &mut d_r_k2, &mut d_s_k2, &mut d_gamma, &mut d_alpha, &mut d_beta,
        );
        // k2 eval point = r0 + 0.5*dt*k1r, so:
        //   d_r0 += d_r_k2
        //   d_k1r += d_r_k2 * 0.5*dt
        for i in 0..n {
            d_r[i] += d_r_k2[i];
            d_s[i] += d_s_k2[i];
        }
        let d_k1r_extra: Vec<f32> = d_r_k2.iter().map(|&v| v * 0.5 * dt).collect();
        let d_k1s_extra: Vec<f32> = d_s_k2.iter().map(|&v| v * 0.5 * dt).collect();

        // ── k1 backward ──
        // k1 was evaluated at (r0, s0)
        let total_d_k1r: Vec<f32> = d_k1r.iter().zip(&d_k1r_extra).map(|(&a, &b)| a + b).collect();
        let total_d_k1s: Vec<f32> = d_k1s.iter().zip(&d_k1s_extra).map(|(&a, &b)| a + b).collect();
        let mut d_r_k1 = vec![0.0f32; n];
        let mut d_s_k1 = vec![0.0f32; n];
        deriv_backward(
            &total_d_k1r, &total_d_k1s, r0, s0,
            &cache.gamma, &weights.omega, weights.alpha, weights.beta,
            &mut d_r_k1, &mut d_s_k1, &mut d_gamma, &mut d_alpha, &mut d_beta,
        );
        // k1 eval point = r0, so d_r0 += d_r_k1
        for i in 0..n {
            d_r[i] += d_r_k1[i];
            d_s[i] += d_s_k1[i];
        }
    }

    // Convert d_gamma (post-softplus) to d_gamma_raw via chain rule
    let d_gamma_raw: Vec<f32> = d_gamma.iter().zip(&weights.gamma_raw)
        .map(|(&dg, &gr)| dg * softplus_derivative(gr))
        .collect();

    // Pack d_input
    let n_embd = n * 2;
    let mut d_input = vec![0.0f32; n_embd];
    for k in 0..n {
        d_input[k * 2] = d_r[k];
        d_input[k * 2 + 1] = d_s[k];
    }

    (d_input, OdeParamGrads { d_gamma_raw, d_alpha, d_beta })
}

// ─── Finite difference validation ──────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_weights(n_bands: usize) -> KerrWeights {
        let gamma_raw_val = ((0.1f32).exp() - 1.0).ln();
        KerrWeights {
            gamma_raw: vec![gamma_raw_val; n_bands],
            omega: (0..n_bands).map(|k| (k + 1) as f32 / n_bands as f32).collect(),
            alpha: 0.1,
            beta: 0.2,
            rk4_n_steps: 4, // fewer steps for faster test
        }
    }

    #[test]
    fn test_forward_matches_original() {
        let n_bands = 8;
        let weights = make_test_weights(n_bands);
        let x: Vec<f32> = (0..n_bands * 2).map(|i| (i as f32 * 0.1).sin()).collect();

        let (out_cached, _cache) = ode_forward_with_cache(&x, &weights);

        // Compare with the standalone forward from block.rs (reproduced here)
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();
        let mut r: Vec<f32> = (0..n_bands).map(|k| x[k * 2]).collect();
        let mut s: Vec<f32> = (0..n_bands).map(|k| x[k * 2 + 1]).collect();
        let dt = 1.0 / weights.rk4_n_steps as f32;
        for _ in 0..weights.rk4_n_steps {
            let (dr, ds) = deriv(&r, &s, &gamma, &weights.omega, weights.alpha, weights.beta);
            let r2: Vec<f32> = r.iter().zip(&dr).map(|(&a, &b)| a + 0.5*dt*b).collect();
            let s2: Vec<f32> = s.iter().zip(&ds).map(|(&a, &b)| a + 0.5*dt*b).collect();
            let (k2r, k2s) = deriv(&r2, &s2, &gamma, &weights.omega, weights.alpha, weights.beta);
            let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&a, &b)| a + 0.5*dt*b).collect();
            let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&a, &b)| a + 0.5*dt*b).collect();
            let (k3r, k3s) = deriv(&r3, &s3, &gamma, &weights.omega, weights.alpha, weights.beta);
            let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&a, &b)| a + dt*b).collect();
            let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&a, &b)| a + dt*b).collect();
            let (k4r, k4s) = deriv(&r4, &s4, &gamma, &weights.omega, weights.alpha, weights.beta);
            r = (0..n_bands).map(|i| r[i] + dt/6.0*(dr[i]+2.0*k2r[i]+2.0*k3r[i]+k4r[i])).collect();
            s = (0..n_bands).map(|i| s[i] + dt/6.0*(ds[i]+2.0*k2s[i]+2.0*k3s[i]+k4s[i])).collect();
        }

        for k in 0..n_bands {
            assert!((out_cached[k*2] - r[k]).abs() < 1e-6, "r[{}] mismatch: {} vs {}", k, out_cached[k*2], r[k]);
            assert!((out_cached[k*2+1] - s[k]).abs() < 1e-6, "s[{}] mismatch", k);
        }
    }

    #[test]
    fn test_backward_finite_differences() {
        let n_bands = 8;
        let weights = make_test_weights(n_bands);
        let x: Vec<f32> = (0..n_bands * 2).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let eps = 1e-3;

        // Forward + backward
        let (out, cache) = ode_forward_with_cache(&x, &weights);
        // Use simple sum-of-outputs as loss for gradient checking
        let d_output: Vec<f32> = vec![1.0f32; n_bands * 2];
        let (d_input, param_grads) = ode_backward(&d_output, &cache, &weights);

        // Finite difference check for d_input
        for i in 0..n_bands * 2 {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            let (out_plus, _) = ode_forward_with_cache(&x_plus, &weights);
            let (out_minus, _) = ode_forward_with_cache(&x_minus, &weights);
            let fd: f32 = out_plus.iter().zip(&out_minus).map(|(a, b)| a - b).sum::<f32>() / (2.0 * eps);
            let rel_err = if fd.abs() > 1e-6 { (d_input[i] - fd).abs() / fd.abs() } else { (d_input[i] - fd).abs() };
            assert!(rel_err < 0.05, "d_input[{}]: analytical={:.6}, fd={:.6}, rel_err={:.4}", i, d_input[i], fd, rel_err);
        }

        // Finite difference check for d_alpha
        {
            let mut w_plus = weights.clone();
            let mut w_minus = weights.clone();
            w_plus.alpha += eps;
            w_minus.alpha -= eps;
            let (out_plus, _) = ode_forward_with_cache(&x, &w_plus);
            let (out_minus, _) = ode_forward_with_cache(&x, &w_minus);
            let fd: f32 = out_plus.iter().zip(&out_minus).map(|(a, b)| a - b).sum::<f32>() / (2.0 * eps);
            let rel_err = if fd.abs() > 1e-6 { (param_grads.d_alpha - fd).abs() / fd.abs() } else { (param_grads.d_alpha - fd).abs() };
            assert!(rel_err < 0.05, "d_alpha: analytical={:.6}, fd={:.6}, rel_err={:.4}", param_grads.d_alpha, fd, rel_err);
        }

        // Finite difference check for d_beta
        {
            let mut w_plus = weights.clone();
            let mut w_minus = weights.clone();
            w_plus.beta += eps;
            w_minus.beta -= eps;
            let (out_plus, _) = ode_forward_with_cache(&x, &w_plus);
            let (out_minus, _) = ode_forward_with_cache(&x, &w_minus);
            let fd: f32 = out_plus.iter().zip(&out_minus).map(|(a, b)| a - b).sum::<f32>() / (2.0 * eps);
            let rel_err = if fd.abs() > 1e-6 { (param_grads.d_beta - fd).abs() / fd.abs() } else { (param_grads.d_beta - fd).abs() };
            assert!(rel_err < 0.05, "d_beta: analytical={:.6}, fd={:.6}, rel_err={:.4}", param_grads.d_beta, fd, rel_err);
        }

        // Finite difference check for d_gamma_raw[0]
        {
            let mut w_plus = weights.clone();
            let mut w_minus = weights.clone();
            w_plus.gamma_raw[0] += eps;
            w_minus.gamma_raw[0] -= eps;
            let (out_plus, _) = ode_forward_with_cache(&x, &w_plus);
            let (out_minus, _) = ode_forward_with_cache(&x, &w_minus);
            let fd: f32 = out_plus.iter().zip(&out_minus).map(|(a, b)| a - b).sum::<f32>() / (2.0 * eps);
            let rel_err = if fd.abs() > 1e-6 { (param_grads.d_gamma_raw[0] - fd).abs() / fd.abs() } else { (param_grads.d_gamma_raw[0] - fd).abs() };
            assert!(rel_err < 0.05, "d_gamma_raw[0]: analytical={:.6}, fd={:.6}, rel_err={:.4}", param_grads.d_gamma_raw[0], fd, rel_err);
        }
    }
}
