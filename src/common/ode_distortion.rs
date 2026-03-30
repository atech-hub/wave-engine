//! ODE distortion monitor — measure harmonic distortion through the Kerr ODE.
//!
//! Each band is a known frequency. Distortion products from the nonlinearity
//! land at predictable harmonic positions (3k, 5k, 7k mod n_bands).
//! This is THD (Total Harmonic Distortion) — a standard RF measurement
//! applied to an architecture where it actually makes physical sense.

use std::f32::consts::PI;

/// ODE distortion measurements from one forward pass.
pub struct OdeDistortion {
    // Gain
    pub gain_mean: f32,
    pub gain_max: f32,
    pub n_compressed: usize,  // bands with gain < 0.8
    // Phase distortion
    pub excess_phase_mean: f32,
    pub excess_phase_max: f32,
    pub n_phase_distorted: usize,  // bands with excess > 0.3 rad
    // THD
    pub thd_total: f32,
    pub thd_max_band: usize,
    pub thd_max_value: f32,
    pub n_thd_over_10pct: usize,
    // Intermodulation
    pub intermod_ratio: f32,
}

/// Measure distortion between ODE input and output.
/// precond: ODE input [n_embd], kerr_out: ODE output [n_embd].
pub fn measure_distortion(
    precond: &[f32],
    kerr_out: &[f32],
    omega: &[f32],
    n_bands: usize,
    rk4_steps: usize,
) -> OdeDistortion {
    let dt_total = 1.0; // RK4 integrates over dt=1/steps for `steps` steps = total 1.0

    // Extract magnitudes and phases
    let mut mag_in = vec![0.0f32; n_bands];
    let mut mag_out = vec![0.0f32; n_bands];
    let mut phase_in = vec![0.0f32; n_bands];
    let mut phase_out = vec![0.0f32; n_bands];

    for k in 0..n_bands {
        let ri = precond[k * 2];
        let si = precond[k * 2 + 1];
        let ro = kerr_out[k * 2];
        let so = kerr_out[k * 2 + 1];
        mag_in[k] = (ri * ri + si * si).sqrt();
        mag_out[k] = (ro * ro + so * so).sqrt();
        phase_in[k] = si.atan2(ri);
        phase_out[k] = so.atan2(ro);
    }

    // 1. Per-band gain
    let gains: Vec<f32> = (0..n_bands).map(|k| {
        mag_out[k] / mag_in[k].max(1e-8)
    }).collect();
    let gain_mean = gains.iter().sum::<f32>() / n_bands as f32;
    let gain_max = gains.iter().cloned().fold(0.0f32, f32::max);
    let n_compressed = gains.iter().filter(|&&g| g < 0.8).count();

    // 2. Phase distortion
    let excess_shifts: Vec<f32> = (0..n_bands).map(|k| {
        let mut shift = phase_out[k] - phase_in[k];
        while shift > PI { shift -= 2.0 * PI; }
        while shift < -PI { shift += 2.0 * PI; }
        let expected = omega[k] * dt_total;
        let mut excess = shift - expected;
        while excess > PI { excess -= 2.0 * PI; }
        while excess < -PI { excess += 2.0 * PI; }
        excess.abs()
    }).collect();
    let excess_phase_mean = excess_shifts.iter().sum::<f32>() / n_bands as f32;
    let excess_phase_max = excess_shifts.iter().cloned().fold(0.0f32, f32::max);
    let n_phase_distorted = excess_shifts.iter().filter(|&&e| e > 0.3).count();

    // 3. THD — harmonic energy detection
    let median_mag_in = {
        let mut sorted = mag_in.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[n_bands / 2]
    };

    let mut thd_per_band = vec![0.0f32; n_bands];
    for k in 0..n_bands {
        if mag_in[k] < median_mag_in { continue; } // only measure driven bands

        let fund_energy = mag_out[k] * mag_out[k];
        let mut harmonic_energy = 0.0f32;

        for &mult in &[3usize, 5, 7] {
            let h = (mult * (k + 1)) % n_bands; // harmonic position
            if h == k { continue; } // skip self
            // Energy gain at harmonic position
            let e_in = mag_in[h] * mag_in[h];
            let e_out = mag_out[h] * mag_out[h];
            let gained = (e_out - e_in).max(0.0);
            harmonic_energy += gained;
        }

        thd_per_band[k] = (harmonic_energy).sqrt() / fund_energy.sqrt().max(1e-8);
    }

    let thd_total = {
        let sum_sq: f32 = thd_per_band.iter().map(|t| t * t).sum();
        (sum_sq / n_bands as f32).sqrt()
    };
    let (thd_max_band, thd_max_value) = thd_per_band.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i, v)).unwrap_or((0, 0.0));
    let n_thd_over_10pct = thd_per_band.iter().filter(|&&t| t > 0.10).count();

    // 4. Intermodulation — energy appearing at non-input frequencies
    let total_out_energy: f32 = mag_out.iter().map(|m| m * m).sum();
    let mut intermod_energy = 0.0f32;
    for k in 0..n_bands {
        if mag_in[k] < median_mag_in {
            // Weakly-driven band — energy here is likely intermod
            let gained = (mag_out[k] * mag_out[k] - mag_in[k] * mag_in[k]).max(0.0);
            intermod_energy += gained;
        }
    }
    let intermod_ratio = intermod_energy / total_out_energy.max(1e-8);

    OdeDistortion {
        gain_mean, gain_max, n_compressed,
        excess_phase_mean, excess_phase_max, n_phase_distorted,
        thd_total, thd_max_band, thd_max_value, n_thd_over_10pct,
        intermod_ratio,
    }
}

/// Format as JSON fragment for JSONL.
pub fn to_json(d: &OdeDistortion) -> String {
    format!(
        r#""ode_distortion":{{"thd":{:.4},"thd_max_band":{},"thd_max_val":{:.3},"n_thd_10pct":{},"gain_mean":{:.3},"gain_max":{:.2},"n_compressed":{},"excess_phase":{:.3},"n_distorted":{},"intermod":{:.4}}}"#,
        d.thd_total, d.thd_max_band, d.thd_max_value, d.n_thd_over_10pct,
        d.gain_mean, d.gain_max, d.n_compressed,
        d.excess_phase_mean, d.n_phase_distorted, d.intermod_ratio
    )
}
