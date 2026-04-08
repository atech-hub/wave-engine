//! FWM Monitor — direct measurement of four-wave mixing during training.
//! Uses the canonical kerr_derivative_into from ode_deriv.rs (single source of truth).
//! Runs a separate monitoring-only ODE forward with DerivativeCapture.
//! Zero cost to training path — only runs at health intervals.

use crate::model::KerrWeights;
use super::ode_deriv::{kerr_derivative_into, DerivativeCapture};

pub struct FwmDiagnostics {
    pub layer: usize,
    pub fwm_ratio: f32,       // ||fwm|| / ||total_deriv||
    pub fwm_vs_phase: f32,    // ||fwm|| / ||phi_contribution||
    pub flux_max: f32,        // max per-band FWM energy flux
    pub flux_mean: f32,       // mean per-band FWM energy flux
    pub top_3_bands: [usize; 3],
    pub mean_band_amp: f32,   // mean band amplitude at ODE input
    pub max_band_amp: f32,    // max band amplitude at ODE input
    pub triple_ratio: f32,    // fraction of quartets with all 4 bands active
    pub rk4_step_ratio: f32,  // step_N fwm magnitude / step_1 fwm magnitude
    // Decomposition fields (populated via DerivativeCapture)
    pub damping_ratio: f32,   // ||damping|| / ||total_deriv||
    pub phase_ratio: f32,     // ||phase_rotation|| / ||total_deriv||
}

/// Measure FWM contribution on a single position's ODE input.
/// Uses canonical RK4 integration (not Euler) matching the training path.
pub fn measure_fwm(precond: &[f32], weights: &KerrWeights, n_bands: usize, layer: usize) -> FwmDiagnostics {
    let n = n_bands;
    let n_steps = weights.rk4_n_steps;
    let dt = 1.0 / n_steps as f32;
    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| crate::common::math::softplus(g)).collect();
    let chi = weights.chi;
    let w = &weights.rk4_weights;

    let mut r: Vec<f32> = (0..n).map(|k| precond[k * 2]).collect();
    let mut s: Vec<f32> = (0..n).map(|k| precond[k * 2 + 1]).collect();

    // Apply AGC-like clamping to match what training actually sees
    let ceiling = (std::f32::consts::FRAC_PI_2 / (weights.alpha + 4.0 * weights.beta)).sqrt().max(0.5);
    for k in 0..n {
        let mag = (r[k]*r[k] + s[k]*s[k]).sqrt();
        if mag > ceiling {
            let scale = ceiling / mag;
            r[k] *= scale;
            s[k] *= scale;
        }
    }

    // Input amplitude stats (after AGC)
    let band_amps: Vec<f32> = (0..n).map(|k| (r[k]*r[k] + s[k]*s[k]).sqrt()).collect();
    let mean_band_amp = band_amps.iter().sum::<f32>() / n as f32;
    let max_band_amp = band_amps.iter().cloned().fold(0.0f32, f32::max);

    // Triple coverage
    let threshold = 0.01f32;
    let mut active_quartets = 0usize;
    let mut total_quartets = 0usize;
    for k in 2..(n-1) {
        total_quartets += 1;
        if band_amps[k-2] > threshold && band_amps[k+1] > threshold && band_amps[k-1] > threshold && band_amps[k] > threshold {
            active_quartets += 1;
        }
    }
    for k in 1..(n-2) {
        total_quartets += 1;
        if band_amps[k-1] > threshold && band_amps[k+2] > threshold && band_amps[k] > threshold && band_amps[k+1] > threshold {
            active_quartets += 1;
        }
    }
    let triple_ratio = if total_quartets > 0 { active_quartets as f32 / total_quartets as f32 } else { 0.0 };

    // Run RK4 steps using canonical derivative with DerivativeCapture
    let mut fwm_mag_step1 = 0.0f32;
    let mut fwm_mag_last = 0.0f32;
    let mut total_fwm_flux = vec![0.0f32; n];
    let mut total_deriv_norm_sq = 0.0f32;
    let mut total_fwm_norm_sq = 0.0f32;
    let mut total_phase_norm_sq = 0.0f32;
    let mut total_damping_norm_sq = 0.0f32;

    // Scratch buffers for RK4
    let mut r_tmp = vec![0.0f32; n];
    let mut s_tmp = vec![0.0f32; n];
    let mut k1r = vec![0.0f32; n]; let mut k1s = vec![0.0f32; n];
    let mut k2r = vec![0.0f32; n]; let mut k2s = vec![0.0f32; n];
    let mut k3r = vec![0.0f32; n]; let mut k3s = vec![0.0f32; n];
    let mut k4r = vec![0.0f32; n]; let mut k4s = vec![0.0f32; n];

    // Capture buffers (accumulated per step, then reset)
    let mut damp_dr = vec![0.0f32; n]; let mut damp_ds = vec![0.0f32; n];
    let mut phase_dr = vec![0.0f32; n]; let mut phase_ds = vec![0.0f32; n];
    let mut fwm_dr = vec![0.0f32; n]; let mut fwm_ds = vec![0.0f32; n];

    for step in 0..n_steps {
        // Zero capture buffers for this step's k1 evaluation
        for i in 0..n { damp_dr[i] = 0.0; damp_ds[i] = 0.0; }
        for i in 0..n { phase_dr[i] = 0.0; phase_ds[i] = 0.0; }
        for i in 0..n { fwm_dr[i] = 0.0; fwm_ds[i] = 0.0; }

        // k1 with capture (captures decomposition at current state)
        {
            let mut cap = DerivativeCapture {
                damping_dr: &mut damp_dr, damping_ds: &mut damp_ds,
                phase_dr: &mut phase_dr, phase_ds: &mut phase_ds,
                fwm_dr: &mut fwm_dr, fwm_ds: &mut fwm_ds,
            };
            kerr_derivative_into(&r, &s, &gamma, &weights.omega, weights.alpha, weights.beta, chi, &mut k1r, &mut k1s, Some(&mut cap));
        }

        // Accumulate stats from k1 capture (representative of this step)
        let fwm_norm: f32 = fwm_dr.iter().map(|x| x*x).sum::<f32>() + fwm_ds.iter().map(|x| x*x).sum::<f32>();
        let deriv_norm: f32 = k1r.iter().map(|x| x*x).sum::<f32>() + k1s.iter().map(|x| x*x).sum::<f32>();
        let phase_norm: f32 = phase_dr.iter().map(|x| x*x).sum::<f32>() + phase_ds.iter().map(|x| x*x).sum::<f32>();
        let damping_norm: f32 = damp_dr.iter().map(|x| x*x).sum::<f32>() + damp_ds.iter().map(|x| x*x).sum::<f32>();
        total_fwm_norm_sq += fwm_norm;
        total_deriv_norm_sq += deriv_norm;
        total_phase_norm_sq += phase_norm;
        total_damping_norm_sq += damping_norm;

        if step == 0 { fwm_mag_step1 = fwm_norm.sqrt(); }
        fwm_mag_last = fwm_norm.sqrt();

        for k in 0..n {
            total_fwm_flux[k] += fwm_dr[k]*fwm_dr[k] + fwm_ds[k]*fwm_ds[k];
        }

        // k2, k3, k4 without capture (we only capture at k1 for stats efficiency)
        for i in 0..n { r_tmp[i] = r[i] + 0.5 * dt * k1r[i]; }
        for i in 0..n { s_tmp[i] = s[i] + 0.5 * dt * k1s[i]; }
        kerr_derivative_into(&r_tmp, &s_tmp, &gamma, &weights.omega, weights.alpha, weights.beta, chi, &mut k2r, &mut k2s, None);

        for i in 0..n { r_tmp[i] = r[i] + 0.5 * dt * k2r[i]; }
        for i in 0..n { s_tmp[i] = s[i] + 0.5 * dt * k2s[i]; }
        kerr_derivative_into(&r_tmp, &s_tmp, &gamma, &weights.omega, weights.alpha, weights.beta, chi, &mut k3r, &mut k3s, None);

        for i in 0..n { r_tmp[i] = r[i] + dt * k3r[i]; }
        for i in 0..n { s_tmp[i] = s[i] + dt * k3s[i]; }
        kerr_derivative_into(&r_tmp, &s_tmp, &gamma, &weights.omega, weights.alpha, weights.beta, chi, &mut k4r, &mut k4s, None);

        // RK4 state update
        for i in 0..n {
            r[i] += dt * (w[0] * k1r[i] + w[1] * k2r[i] + w[2] * k3r[i] + w[3] * k4r[i]);
            s[i] += dt * (w[0] * k1s[i] + w[1] * k2s[i] + w[2] * k3s[i] + w[3] * k4s[i]);
        }
    }

    // Compute summary stats
    let fwm_ratio = if total_deriv_norm_sq > 1e-20 { total_fwm_norm_sq.sqrt() / total_deriv_norm_sq.sqrt() } else { 0.0 };
    let fwm_vs_phase = if total_phase_norm_sq > 1e-20 { total_fwm_norm_sq.sqrt() / total_phase_norm_sq.sqrt() } else { 0.0 };
    let damping_ratio = if total_deriv_norm_sq > 1e-20 { total_damping_norm_sq.sqrt() / total_deriv_norm_sq.sqrt() } else { 0.0 };
    let phase_ratio = if total_deriv_norm_sq > 1e-20 { total_phase_norm_sq.sqrt() / total_deriv_norm_sq.sqrt() } else { 0.0 };
    let flux_max = total_fwm_flux.iter().cloned().fold(0.0f32, f32::max);
    let flux_mean = total_fwm_flux.iter().sum::<f32>() / n as f32;
    let rk4_step_ratio = if fwm_mag_step1 > 1e-10 { fwm_mag_last / fwm_mag_step1 } else { 1.0 };

    // Top 3 bands by flux
    let mut indexed: Vec<(usize, f32)> = total_fwm_flux.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_3 = [
        indexed.get(0).map_or(0, |x| x.0),
        indexed.get(1).map_or(0, |x| x.0),
        indexed.get(2).map_or(0, |x| x.0),
    ];

    FwmDiagnostics {
        layer,
        fwm_ratio, fwm_vs_phase,
        flux_max, flux_mean,
        top_3_bands: top_3,
        mean_band_amp, max_band_amp,
        triple_ratio, rk4_step_ratio,
        damping_ratio, phase_ratio,
    }
}

/// Stability scan: test multiple chi values on one hidden state.
pub fn fwm_stability_scan(
    precond: &[f32], weights: &KerrWeights, n_bands: usize,
    chis: &[f32],
) -> Vec<(f32, FwmDiagnostics, bool)> {
    let mut results = Vec::new();
    for &chi in chis {
        let mut w = weights.clone();
        w.chi = chi;
        let diag = measure_fwm(precond, &w, n_bands, 0);
        let stable = !diag.fwm_ratio.is_nan() && diag.fwm_ratio < 0.5 && diag.rk4_step_ratio < 10.0;
        results.push((chi, diag, stable));
    }
    results
}
