//! Kerr-ODE derivative and RK4 integration.
//!
//! THE SINGLE SOURCE OF TRUTH for all ODE derivative computations.
//! Every caller — training forward, backward cache, monitors, probes,
//! diagnostics — MUST use these functions. No local copies.
//!
//! Public API:
//!   - `kerr_derivative_into()` — in-place derivative (training, monitors)
//!   - `kerr_derivative()` — allocating wrapper (probe, tests)
//!   - `rk4_step_into()` — in-place RK4 step (training forward)
//!   - `rk4_step_public()` — allocating RK4 step (backward, compute, diagnostics)

/// Optional capture buffers for decomposing the derivative into physical terms.
/// When provided, each buffer pair receives the contribution from one effect.
/// All buffers must be pre-zeroed by the caller (or accumulated across RK4 substeps).
pub struct DerivativeCapture<'a> {
    pub damping_dr: &'a mut [f32],
    pub damping_ds: &'a mut [f32],
    pub phase_dr: &'a mut [f32],   // SPM + XPM combined (the φ·z rotation)
    pub phase_ds: &'a mut [f32],
    pub fwm_dr: &'a mut [f32],
    pub fwm_ds: &'a mut [f32],
}

/// Kerr nonlinear derivative — in-place, zero-allocation.
///
/// Writes the ODE right-hand side into pre-allocated `dr`, `ds` buffers.
/// Conv1d neighbour sum with kernel [1, 1, 0, 1, 1], padding=2.
/// chi: four-wave mixing strength (0.0 = disabled).
///
/// When `capture` is Some, also writes per-effect contributions to the
/// provided buffers. Zero overhead when None (single branch check).
pub fn kerr_derivative_into(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32, chi: f32,
    dr: &mut [f32], ds: &mut [f32],
    mut capture: Option<&mut DerivativeCapture<'_>>,
) {
    let n = r.len();

    // Compute mag_sq and neighbour sums
    for k in 0..n {
        let mag_sq = r[k] * r[k] + s[k] * s[k];
        let mut ns = 0.0f32;
        if k >= 2 { ns += r[k-2]*r[k-2] + s[k-2]*s[k-2]; }
        if k >= 1 { ns += r[k-1]*r[k-1] + s[k-1]*s[k-1]; }
        if k+1 < n { ns += r[k+1]*r[k+1] + s[k+1]*s[k+1]; }
        if k+2 < n { ns += r[k+2]*r[k+2] + s[k+2]*s[k+2]; }
        let phi = omega[k] + alpha * mag_sq + beta * ns;
        dr[k] = -gamma[k] * r[k] - phi * s[k];
        ds[k] = -gamma[k] * s[k] + phi * r[k];
    }

    // Four-wave mixing: energy-conserving Hamiltonian flow.
    // For each quartet (a,b,c,d) with a+b=c+d, ALL FOUR bands get a derivative term.
    // Derived from H = chi * Re(z_a * z_b * z_c* * z_d*).
    if chi != 0.0 && n > 4 {
        if let Some(ref mut cap) = capture {
            // Capture path: write FWM terms to both main and capture buffers
            for k in 2..(n - 1) {
                apply_quartet_capture(dr, ds, cap.fwm_dr, cap.fwm_ds, r, s, chi, k - 2, k + 1, k - 1, k);
            }
            for k in 1..(n - 2) {
                apply_quartet_capture(dr, ds, cap.fwm_dr, cap.fwm_ds, r, s, chi, k - 1, k + 2, k, k + 1);
            }
        } else {
            // Fast path: no capture overhead
            for k in 2..(n - 1) {
                apply_quartet(dr, ds, r, s, chi, k - 2, k + 1, k - 1, k);
            }
            for k in 1..(n - 2) {
                apply_quartet(dr, ds, r, s, chi, k - 1, k + 2, k, k + 1);
            }
        }
    }

    // Capture damping and phase contributions if requested
    if let Some(cap) = capture {
        for k in 0..n {
            let mag_sq = r[k] * r[k] + s[k] * s[k];
            let mut ns = 0.0f32;
            if k >= 2 { ns += r[k-2]*r[k-2] + s[k-2]*s[k-2]; }
            if k >= 1 { ns += r[k-1]*r[k-1] + s[k-1]*s[k-1]; }
            if k+1 < n { ns += r[k+1]*r[k+1] + s[k+1]*s[k+1]; }
            if k+2 < n { ns += r[k+2]*r[k+2] + s[k+2]*s[k+2]; }
            let phi = omega[k] + alpha * mag_sq + beta * ns;

            // Damping: -γ·z
            cap.damping_dr[k] += -gamma[k] * r[k];
            cap.damping_ds[k] += -gamma[k] * s[k];
            // Phase rotation: -φ·s, +φ·r (SPM + XPM combined)
            cap.phase_dr[k] += -phi * s[k];
            cap.phase_ds[k] += phi * r[k];
        }
    }
}

/// Kerr nonlinear derivative — allocating version for probe and tests.
pub fn kerr_derivative(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32, chi: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();
    let mut dr = vec![0.0f32; n];
    let mut ds = vec![0.0f32; n];
    kerr_derivative_into(r, s, gamma, omega, alpha, beta, chi, &mut dr, &mut ds, None);
    (dr, ds)
}

/// Single RK4 step — in-place, uses pre-allocated scratch buffers.
/// Caller provides scratch space to avoid per-step allocation.
pub fn rk4_step_into(
    r: &[f32], s: &[f32], dt: f32,
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32, chi: f32, w: &[f32; 4],
    r_out: &mut [f32], s_out: &mut [f32],
    // Scratch: r_tmp, s_tmp, k1r, k1s, k2r, k2s, k3r, k3s, k4r, k4s
    r_tmp: &mut [f32], s_tmp: &mut [f32],
    k1r: &mut [f32], k1s: &mut [f32],
    k2r: &mut [f32], k2s: &mut [f32],
    k3r: &mut [f32], k3s: &mut [f32],
    k4r: &mut [f32], k4s: &mut [f32],
) {
    let n = r.len();

    // k1 at (r, s)
    kerr_derivative_into(r, s, gamma, omega, alpha, beta, chi, k1r, k1s, None);

    // k2 at (r + 0.5*dt*k1, s + 0.5*dt*k1)
    for i in 0..n { r_tmp[i] = r[i] + 0.5 * dt * k1r[i]; }
    for i in 0..n { s_tmp[i] = s[i] + 0.5 * dt * k1s[i]; }
    kerr_derivative_into(r_tmp, s_tmp, gamma, omega, alpha, beta, chi, k2r, k2s, None);

    // k3 at (r + 0.5*dt*k2, s + 0.5*dt*k2)
    for i in 0..n { r_tmp[i] = r[i] + 0.5 * dt * k2r[i]; }
    for i in 0..n { s_tmp[i] = s[i] + 0.5 * dt * k2s[i]; }
    kerr_derivative_into(r_tmp, s_tmp, gamma, omega, alpha, beta, chi, k3r, k3s, None);

    // k4 at (r + dt*k3, s + dt*k3)
    for i in 0..n { r_tmp[i] = r[i] + dt * k3r[i]; }
    for i in 0..n { s_tmp[i] = s[i] + dt * k3s[i]; }
    kerr_derivative_into(r_tmp, s_tmp, gamma, omega, alpha, beta, chi, k4r, k4s, None);

    // Combine: y_new = y + dt * (w0*k1 + w1*k2 + w2*k3 + w3*k4)
    for i in 0..n {
        r_out[i] = r[i] + dt * (w[0] * k1r[i] + w[1] * k2r[i] + w[2] * k3r[i] + w[3] * k4r[i]);
        s_out[i] = s[i] + dt * (w[0] * k1s[i] + w[1] * k2s[i] + w[2] * k3s[i] + w[3] * k4s[i]);
    }
}

/// Single RK4 step — allocating version (backward, compute, diagnostics, probe).
pub fn rk4_step_public(
    r: &[f32], s: &[f32], dt: f32,
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32, chi: f32, w: &[f32; 4],
) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();
    let mut r_out = vec![0.0f32; n];
    let mut s_out = vec![0.0f32; n];
    let mut r_tmp = vec![0.0f32; n];
    let mut s_tmp = vec![0.0f32; n];
    let mut k1r = vec![0.0f32; n];
    let mut k1s = vec![0.0f32; n];
    let mut k2r = vec![0.0f32; n];
    let mut k2s = vec![0.0f32; n];
    let mut k3r = vec![0.0f32; n];
    let mut k3s = vec![0.0f32; n];
    let mut k4r = vec![0.0f32; n];
    let mut k4s = vec![0.0f32; n];
    rk4_step_into(
        r, s, dt, gamma, omega, alpha, beta, chi, w,
        &mut r_out, &mut s_out,
        &mut r_tmp, &mut s_tmp,
        &mut k1r, &mut k1s, &mut k2r, &mut k2s,
        &mut k3r, &mut k3s, &mut k4r, &mut k4s,
    );
    (r_out, s_out)
}

// ─── FWM quartet helpers ──────────────────────────────────

#[inline(always)]
fn apply_quartet(
    dr: &mut [f32], ds: &mut [f32], r: &[f32], s: &[f32],
    chi: f32, a: usize, b: usize, c: usize, d: usize,
) {
    let (ra, sa) = (r[a], s[a]);
    let (rb, sb) = (r[b], s[b]);
    let (rc, sc) = (r[c], s[c]);
    let (rd, sd) = (r[d], s[d]);

    let pab_re = ra*rb - sa*sb;
    let pab_im = ra*sb + sa*rb;
    let pcd_re = rc*rd - sc*sd;
    let pcd_im = rc*sd + sc*rd;

    dr[a] += chi * (rb*pcd_im - sb*pcd_re);
    ds[a] -= chi * (rb*pcd_re + sb*pcd_im);
    dr[b] += chi * (ra*pcd_im - sa*pcd_re);
    ds[b] -= chi * (ra*pcd_re + sa*pcd_im);
    dr[c] += chi * (pab_im*rd - pab_re*sd);
    ds[c] -= chi * (pab_re*rd + pab_im*sd);
    dr[d] += chi * (pab_im*rc - pab_re*sc);
    ds[d] -= chi * (pab_re*rc + pab_im*sc);
}

#[inline(always)]
fn apply_quartet_capture(
    dr: &mut [f32], ds: &mut [f32],
    fwm_dr: &mut [f32], fwm_ds: &mut [f32],
    r: &[f32], s: &[f32],
    chi: f32, a: usize, b: usize, c: usize, d: usize,
) {
    let (ra, sa) = (r[a], s[a]);
    let (rb, sb) = (r[b], s[b]);
    let (rc, sc) = (r[c], s[c]);
    let (rd, sd) = (r[d], s[d]);

    let pab_re = ra*rb - sa*sb;
    let pab_im = ra*sb + sa*rb;
    let pcd_re = rc*rd - sc*sd;
    let pcd_im = rc*sd + sc*rd;

    let da = chi * (rb*pcd_im - sb*pcd_re);
    let sa_v = -(chi * (rb*pcd_re + sb*pcd_im));
    dr[a] += da; ds[a] += sa_v; fwm_dr[a] += da; fwm_ds[a] += sa_v;

    let db = chi * (ra*pcd_im - sa*pcd_re);
    let sb_v = -(chi * (ra*pcd_re + sa*pcd_im));
    dr[b] += db; ds[b] += sb_v; fwm_dr[b] += db; fwm_ds[b] += sb_v;

    let dc = chi * (pab_im*rd - pab_re*sd);
    let sc_v = -(chi * (pab_re*rd + pab_im*sd));
    dr[c] += dc; ds[c] += sc_v; fwm_dr[c] += dc; fwm_ds[c] += sc_v;

    let dd_v = chi * (pab_im*rc - pab_re*sc);
    let sd_v = -(chi * (pab_re*rc + pab_im*sc));
    dr[d] += dd_v; ds[d] += sd_v; fwm_dr[d] += dd_v; fwm_ds[d] += sd_v;
}
