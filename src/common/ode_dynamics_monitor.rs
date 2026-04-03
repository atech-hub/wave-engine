//! ODE Dynamics Deep Monitor (#6).
//!
//! Analyzes ODE behavior beyond THD: phase velocity, energy conservation,
//! band energy concentration, and effective damping.
//! Reads precond (ODE input) and kerr_out (ODE output) from the FFN backend cache.

use crate::cpu::forward::ForwardCache;
use crate::Dims;

/// Per-layer ODE dynamics statistics.
pub struct OdeDynamicsStats {
    pub layer: usize,
    pub phase_velocity: f32,
    pub energy_in: f32,
    pub energy_out: f32,
    pub energy_ratio: f32,
    pub band_energy_std: f32,
    pub damping_effective: f32,
}

/// Analyze ODE dynamics from the forward cache.
///
/// For each layer, extracts precond and kerr_out from the FFN backend cache
/// (first sequence position). Computes:
/// - Phase velocity: avg |atan2(s_out,r_out) - atan2(s_in,r_in)| across bands
/// - Energy in/out: sum(r² + s²) for input/output
/// - Energy ratio: out/in (should be ~1.0 with AGC)
/// - Band energy std: standard deviation of per-band energy (concentration)
/// - Damping effective: 1.0 - energy_ratio (positive = dissipation)
pub fn analyze_ode_dynamics(cache: &ForwardCache, dims: Dims) -> Vec<OdeDynamicsStats> {
    let n_bands = dims.n_bands;
    let mut stats = Vec::new();

    for (layer_idx, bc) in cache.block_caches.iter().enumerate() {
        let fc = match bc.ffn_backend_cache {
            Some(ref fc) => fc,
            None => continue,
        };

        // Use first sequence position for analysis
        if fc.precond.is_empty() || fc.kerr_out.is_empty() { continue; }
        let precond = &fc.precond[0];
        let kerr_out = &fc.kerr_out[0];
        if precond.len() < n_bands * 2 || kerr_out.len() < n_bands * 2 { continue; }

        // Phase velocity: |phase_out - phase_in| averaged across bands
        let mut phase_vel_sum = 0.0f32;
        let mut energy_in = 0.0f32;
        let mut energy_out = 0.0f32;
        let mut band_energies_out = Vec::with_capacity(n_bands);

        for k in 0..n_bands {
            let r_in = precond[2 * k];
            let s_in = precond[2 * k + 1];
            let r_out = kerr_out[2 * k];
            let s_out = kerr_out[2 * k + 1];

            // Phase difference (wrapped to [-pi, pi])
            let phase_in = s_in.atan2(r_in);
            let phase_out = s_out.atan2(r_out);
            let mut d_phase = phase_out - phase_in;
            // Wrap to [-pi, pi]
            if d_phase > std::f32::consts::PI { d_phase -= 2.0 * std::f32::consts::PI; }
            if d_phase < -std::f32::consts::PI { d_phase += 2.0 * std::f32::consts::PI; }
            phase_vel_sum += d_phase.abs();

            // Energy
            energy_in += r_in * r_in + s_in * s_in;
            energy_out += r_out * r_out + s_out * s_out;
            band_energies_out.push(r_out * r_out + s_out * s_out);
        }

        let phase_velocity = if n_bands > 0 { phase_vel_sum / n_bands as f32 } else { 0.0 };
        let energy_ratio = if energy_in > 1e-12 { energy_out / energy_in } else { 1.0 };
        let damping_effective = 1.0 - energy_ratio;

        // Band energy std
        let band_energy_std = if n_bands > 1 {
            let mean_e = energy_out / n_bands as f32;
            let var: f32 = band_energies_out.iter()
                .map(|&e| (e - mean_e) * (e - mean_e))
                .sum::<f32>() / n_bands as f32;
            var.sqrt()
        } else {
            0.0
        };

        stats.push(OdeDynamicsStats {
            layer: layer_idx,
            phase_velocity,
            energy_in,
            energy_out,
            energy_ratio,
            band_energy_std,
            damping_effective,
        });
    }

    stats
}

/// Serialize ODE dynamics stats to JSONL fragment.
/// Format: "ode_dynamics":[{...}, ...]
pub fn to_json(stats: &[OdeDynamicsStats]) -> String {
    if stats.is_empty() {
        return String::new();
    }

    let entries: Vec<String> = stats.iter().map(|s| {
        format!(
            r#"{{"layer":{},"phase_vel":{:.4},"energy_in":{:.2},"energy_out":{:.2},"energy_ratio":{:.4},"band_std":{:.4},"damping":{:.4}}}"#,
            s.layer, s.phase_velocity, s.energy_in, s.energy_out,
            s.energy_ratio, s.band_energy_std, s.damping_effective,
        )
    }).collect();

    format!(r#""ode_dynamics":[{}]"#, entries.join(","))
}
