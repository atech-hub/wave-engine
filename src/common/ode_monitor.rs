//! Raw ODE monitor: extract per-band magnitude and phase from the forward cache.
//! Shows exactly what the ODE does to each band at each layer.

use crate::common::wave_model::WavePacketModel;
use crate::common::dims::Dims;
use crate::cpu::forward::{forward_with_cache, ForwardCache};

pub struct BandState {
    pub mag: f32,
    pub phase: f32,
}

pub struct OdeSnapshot {
    pub layer: usize,
    pub position: usize,
    pub precond: Vec<BandState>,  // ODE input per band
    pub kerr_out: Vec<BandState>, // ODE output per band
}

fn to_band_states(data: &[f32], n_bands: usize) -> Vec<BandState> {
    (0..n_bands).map(|k| {
        let r = data[k * 2];
        let s = data[k * 2 + 1];
        BandState {
            mag: (r * r + s * s).sqrt(),
            phase: s.atan2(r),
        }
    }).collect()
}

/// Run forward pass and extract per-band ODE data for all layers at a specific position.
pub fn extract_ode_data(
    model: &WavePacketModel,
    tokens: &[usize],
    dims: Dims,
    stencil: &crate::fft_ode::StencilFft,
    position: usize,
) -> (Vec<OdeSnapshot>, ForwardCache) {
    let cache = forward_with_cache(model, tokens, dims, None, None, None, Some(stencil), None, None);
    let n_bands = dims.n_bands;

    let mut snapshots = Vec::new();
    for (layer_idx, bc) in cache.block_caches.iter().enumerate() {
        if let Some(ref fc) = bc.ffn_backend_cache {
            if position < fc.precond.len() {
                snapshots.push(OdeSnapshot {
                    layer: layer_idx,
                    position,
                    precond: to_band_states(&fc.precond[position], n_bands),
                    kerr_out: to_band_states(&fc.kerr_out[position], n_bands),
                });
            }
        }
    }
    (snapshots, cache)
}

/// Print a compact summary of the ODE transformation for a prompt.
pub fn print_ode_summary(
    model: &WavePacketModel,
    tokens: &[usize],
    dims: Dims,
    stencil: &crate::fft_ode::StencilFft,
    prompt_str: &str,
    decode_fn: &dyn Fn(usize) -> String,
) {
    let n_bands = dims.n_bands;
    let last_pos = tokens.len() - 1; // Position where the model predicts the answer

    let (snapshots, cache) = extract_ode_data(model, tokens, dims, stencil, last_pos);

    // Get the lm_head prediction
    let logits = &cache.logits[last_pos];
    let pred_token = logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i).unwrap();

    println!("Prompt: {}  →  predicted: {}", prompt_str, decode_fn(pred_token));
    println!("Position {} (last token of prompt)", last_pos);

    for snap in &snapshots {
        println!("\n  Layer {} — top 5 bands by magnitude change:", snap.layer);
        let mut band_changes: Vec<(usize, f32, f32, f32, f32, f32)> = (0..n_bands).map(|k| {
            let mag_in = snap.precond[k].mag;
            let mag_out = snap.kerr_out[k].mag;
            let phase_in = snap.precond[k].phase;
            let phase_out = snap.kerr_out[k].phase;
            let d_phase = phase_out - phase_in;
            // Normalize to [-π, π]
            let d_phase = if d_phase > std::f32::consts::PI { d_phase - 2.0 * std::f32::consts::PI }
                          else if d_phase < -std::f32::consts::PI { d_phase + 2.0 * std::f32::consts::PI }
                          else { d_phase };
            (k, mag_in, mag_out, phase_in, phase_out, d_phase)
        }).collect();

        // Sort by absolute phase change (most active bands first)
        band_changes.sort_by(|a, b| b.5.abs().partial_cmp(&a.5.abs()).unwrap());

        println!("    {:>4}  {:>8} {:>8}  {:>8} {:>8} {:>8}",
            "band", "mag_in", "mag_out", "φ_in", "φ_out", "Δφ");
        for &(k, mi, mo, pi, po, dp) in band_changes.iter().take(5) {
            println!("    {:>4}  {:>8.4} {:>8.4}  {:>8.4} {:>8.4} {:>+8.4}",
                k, mi, mo, pi, po, dp);
        }

        // Summary stats
        let avg_mag_in: f32 = snap.precond.iter().map(|b| b.mag).sum::<f32>() / n_bands as f32;
        let avg_mag_out: f32 = snap.kerr_out.iter().map(|b| b.mag).sum::<f32>() / n_bands as f32;
        let avg_dphase: f32 = band_changes.iter().map(|b| b.5.abs()).sum::<f32>() / n_bands as f32;
        println!("    avg_mag: {:.4} → {:.4}  avg|Δφ|: {:.4}", avg_mag_in, avg_mag_out, avg_dphase);
    }

    // Final hidden state summary
    let hidden = &cache.post_ln_f[last_pos];
    let total_mag: f32 = (0..n_bands).map(|k| {
        let r = hidden[k * 2];
        let s = hidden[k * 2 + 1];
        (r * r + s * s).sqrt()
    }).sum::<f32>();
    println!("\n  Final hidden: total_mag={:.4}, avg_mag={:.4}", total_mag, total_mag / n_bands as f32);
}

/// Compare ODE behavior between two prompts — shows where computation diverges.
pub fn compare_prompts(
    model: &WavePacketModel,
    tokens_a: &[usize],
    tokens_b: &[usize],
    dims: Dims,
    stencil: &crate::fft_ode::StencilFft,
    label_a: &str,
    label_b: &str,
) {
    let n_bands = dims.n_bands;
    let pos_a = tokens_a.len() - 1;
    let pos_b = tokens_b.len() - 1;

    let (snaps_a, _) = extract_ode_data(model, tokens_a, dims, stencil, pos_a);
    let (snaps_b, _) = extract_ode_data(model, tokens_b, dims, stencil, pos_b);

    println!("\nComparing: {} vs {}", label_a, label_b);

    for (sa, sb) in snaps_a.iter().zip(snaps_b.iter()) {
        println!("\n  Layer {} — top 5 bands by output phase DIVERGENCE:", sa.layer);
        let mut divergences: Vec<(usize, f32, f32, f32)> = (0..n_bands).map(|k| {
            let phase_a = sa.kerr_out[k].phase;
            let phase_b = sb.kerr_out[k].phase;
            let mut diff = phase_a - phase_b;
            if diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
            if diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }
            (k, phase_a, phase_b, diff)
        }).collect();

        divergences.sort_by(|a, b| b.3.abs().partial_cmp(&a.3.abs()).unwrap());

        println!("    {:>4}  {:>8} {:>8} {:>8}", "band", label_a, label_b, "Δ");
        for &(k, pa, pb, diff) in divergences.iter().take(5) {
            println!("    {:>4}  {:>8.4} {:>8.4} {:>+8.4}", k, pa, pb, diff);
        }

        let avg_div: f32 = divergences.iter().map(|d| d.3.abs()).sum::<f32>() / n_bands as f32;
        println!("    avg divergence: {:.4} rad", avg_div);
    }
}
