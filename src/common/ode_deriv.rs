//! Kerr-ODE derivative and RK4 integration.
//!
//! Extracted from model.rs — the ODE numerics live here.
//! Public API: `rk4_step_public()` for backward.rs and compute.rs.

/// Kerr nonlinear derivative: coupled oscillator ODE right-hand side.
///
/// Conv1d neighbour sum with kernel [1, 1, 0, 1, 1], padding=2.
pub(crate) fn kerr_derivative(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();
    let mut dr = vec![0.0f32; n];
    let mut ds = vec![0.0f32; n];

    // Compute mag_sq for all bands
    let mag_sq: Vec<f32> = (0..n).map(|k| r[k] * r[k] + s[k] * s[k]).collect();

    // Conv1d with kernel [1, 1, 0, 1, 1], padding=2
    let mut ns = vec![0.0f32; n];
    for k in 0..n {
        if k >= 2 { ns[k] += mag_sq[k - 2]; }
        if k >= 1 { ns[k] += mag_sq[k - 1]; }
        if k + 1 < n { ns[k] += mag_sq[k + 1]; }
        if k + 2 < n { ns[k] += mag_sq[k + 2]; }
    }

    for k in 0..n {
        let phi = omega[k] + alpha * mag_sq[k] + beta * ns[k];
        dr[k] = -gamma[k] * r[k] - phi * s[k];
        ds[k] = -gamma[k] * s[k] + phi * r[k];
    }

    (dr, ds)
}

/// Single RK4 integration step with weighted combination.
pub(crate) fn rk4_step(
    r: &[f32], s: &[f32], dt: f32,
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32, w: &[f32; 4],
) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();

    // k1
    let (dr1, ds1) = kerr_derivative(r, s, gamma, omega, alpha, beta);

    // k2
    let r2: Vec<f32> = (0..n).map(|k| r[k] + 0.5 * dt * dr1[k]).collect();
    let s2: Vec<f32> = (0..n).map(|k| s[k] + 0.5 * dt * ds1[k]).collect();
    let (dr2, ds2) = kerr_derivative(&r2, &s2, gamma, omega, alpha, beta);

    // k3
    let r3: Vec<f32> = (0..n).map(|k| r[k] + 0.5 * dt * dr2[k]).collect();
    let s3: Vec<f32> = (0..n).map(|k| s[k] + 0.5 * dt * ds2[k]).collect();
    let (dr3, ds3) = kerr_derivative(&r3, &s3, gamma, omega, alpha, beta);

    // k4
    let r4: Vec<f32> = (0..n).map(|k| r[k] + dt * dr3[k]).collect();
    let s4: Vec<f32> = (0..n).map(|k| s[k] + dt * ds3[k]).collect();
    let (dr4, ds4) = kerr_derivative(&r4, &s4, gamma, omega, alpha, beta);

    // Combine: y_new = y + dt * (w0*k1 + w1*k2 + w2*k3 + w3*k4)
    let r_new: Vec<f32> = (0..n)
        .map(|k| r[k] + dt * (w[0] * dr1[k] + w[1] * dr2[k] + w[2] * dr3[k] + w[3] * dr4[k]))
        .collect();
    let s_new: Vec<f32> = (0..n)
        .map(|k| s[k] + dt * (w[0] * ds1[k] + w[1] * ds2[k] + w[2] * ds3[k] + w[3] * ds4[k]))
        .collect();

    (r_new, s_new)
}

/// Public wrapper for rk4_step (needed by backward.rs, compute.rs).
pub fn rk4_step_public(
    r: &[f32], s: &[f32], dt: f32,
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32, w: &[f32; 4],
) -> (Vec<f32>, Vec<f32>) {
    rk4_step(r, s, dt, gamma, omega, alpha, beta, w)
}
