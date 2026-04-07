//! FWM Monitor — direct measurement of four-wave mixing during training.
//! Runs a separate monitoring-only ODE forward with FWM capture.
//! Zero cost to training path — only runs at health intervals.

use crate::model::KerrWeights;

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
    pub rk4_step_ratio: f32,  // step_16 fwm magnitude / step_1 fwm magnitude
}

/// Measure FWM contribution on a single position's ODE input.
pub fn measure_fwm(precond: &[f32], weights: &KerrWeights, n_bands: usize, layer: usize) -> FwmDiagnostics {
    let n = n_bands;
    let n_steps = weights.rk4_n_steps;
    let dt = 1.0 / n_steps as f32;
    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| crate::common::math::softplus(g)).collect();
    let chi = weights.chi;

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

    // Run RK4 steps, capturing FWM contribution at step 1 and step N
    let w = &weights.rk4_weights;
    let mut fwm_mag_step1 = 0.0f32;
    let mut fwm_mag_last = 0.0f32;
    let mut total_fwm_flux = vec![0.0f32; n];
    let mut total_deriv_norm_sq = 0.0f32;
    let mut total_fwm_norm_sq = 0.0f32;
    let mut total_phi_norm_sq = 0.0f32;

    for step in 0..n_steps {
        // Compute full derivative
        let mut dr = vec![0.0f32; n];
        let mut ds = vec![0.0f32; n];
        // SPM/XPM/damping part
        for k in 0..n {
            let mag_sq = r[k]*r[k] + s[k]*s[k];
            let mut ns = 0.0f32;
            if k >= 2 { ns += r[k-2]*r[k-2] + s[k-2]*s[k-2]; }
            if k >= 1 { ns += r[k-1]*r[k-1] + s[k-1]*s[k-1]; }
            if k+1 < n { ns += r[k+1]*r[k+1] + s[k+1]*s[k+1]; }
            if k+2 < n { ns += r[k+2]*r[k+2] + s[k+2]*s[k+2]; }
            let phi = weights.omega[k] + weights.alpha * mag_sq + weights.beta * ns;
            dr[k] = -gamma[k] * r[k] - phi * s[k];
            ds[k] = -gamma[k] * s[k] + phi * r[k];
            total_phi_norm_sq += (phi * s[k]) * (phi * s[k]) + (phi * r[k]) * (phi * r[k]);
        }

        // FWM part (separate accumulation)
        let mut fwm_dr = vec![0.0f32; n];
        let mut fwm_ds = vec![0.0f32; n];
        if chi != 0.0 && n > 4 {
            for k in 2..(n-1) {
                apply_quartet_capture(&mut dr, &mut ds, &mut fwm_dr, &mut fwm_ds, &r, &s, chi, k-2, k+1, k-1, k);
            }
            for k in 1..(n-2) {
                apply_quartet_capture(&mut dr, &mut ds, &mut fwm_dr, &mut fwm_ds, &r, &s, chi, k-1, k+2, k, k+1);
            }
        }

        // Accumulate stats
        let fwm_norm: f32 = fwm_dr.iter().map(|x| x*x).sum::<f32>() + fwm_ds.iter().map(|x| x*x).sum::<f32>();
        let deriv_norm: f32 = dr.iter().map(|x| x*x).sum::<f32>() + ds.iter().map(|x| x*x).sum::<f32>();
        total_fwm_norm_sq += fwm_norm;
        total_deriv_norm_sq += deriv_norm;

        if step == 0 { fwm_mag_step1 = fwm_norm.sqrt(); }
        fwm_mag_last = fwm_norm.sqrt();

        for k in 0..n {
            total_fwm_flux[k] += fwm_dr[k]*fwm_dr[k] + fwm_ds[k]*fwm_ds[k];
        }

        // RK4 step (simplified — just Euler for monitoring, not training)
        for k in 0..n { r[k] += dt * dr[k]; s[k] += dt * ds[k]; }
    }

    // Compute summary stats
    let fwm_ratio = if total_deriv_norm_sq > 1e-20 { total_fwm_norm_sq.sqrt() / total_deriv_norm_sq.sqrt() } else { 0.0 };
    let fwm_vs_phase = if total_phi_norm_sq > 1e-20 { total_fwm_norm_sq.sqrt() / total_phi_norm_sq.sqrt() } else { 0.0 };
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

/// Helper: apply_quartet that writes to BOTH main dr/ds and capture fwm_dr/fwm_ds
#[inline(always)]
fn apply_quartet_capture(
    dr: &mut [f32], ds: &mut [f32],
    fwm_dr: &mut [f32], fwm_ds: &mut [f32],
    r: &[f32], s: &[f32],
    chi: f32, a: usize, b: usize, c: usize, d: usize,
) {
    let (ra, sa) = (r[a], s[a]); let (rb, sb) = (r[b], s[b]);
    let (rc, sc) = (r[c], s[c]); let (rd, sd) = (r[d], s[d]);
    let pab_re = ra*rb - sa*sb; let pab_im = ra*sb + sa*rb;
    let pcd_re = rc*rd - sc*sd; let pcd_im = rc*sd + sc*rd;

    let da = chi * (rb*pcd_im - sb*pcd_re);
    let sa_v = -chi * (rb*pcd_re + sb*pcd_im);
    dr[a] += da; ds[a] += sa_v; fwm_dr[a] += da; fwm_ds[a] += sa_v;

    let db = chi * (ra*pcd_im - sa*pcd_re);
    let sb_v = -chi * (ra*pcd_re + sa*pcd_im);
    dr[b] += db; ds[b] += sb_v; fwm_dr[b] += db; fwm_ds[b] += sb_v;

    let dc = chi * (pab_im*rd - pab_re*sd);
    let sc_v = -chi * (pab_re*rd + pab_im*sd);
    dr[c] += dc; ds[c] += sc_v; fwm_dr[c] += dc; fwm_ds[c] += sc_v;

    let dd_v = chi * (pab_im*rc - pab_re*sc);
    let sd_v = -chi * (pab_re*rc + pab_im*sc);
    dr[d] += dd_v; ds[d] += sd_v; fwm_dr[d] += dd_v; fwm_ds[d] += sd_v;
}
