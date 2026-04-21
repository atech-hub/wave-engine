//! Galaxy Map Scan — end-of-training pure-band geometric inventory.
//!
//! Five scan layers: per-band profiles, pairwise angular geometry,
//! harmonic coherence matrix, constellation detection (triads + FWM quartets),
//! multi-grid decomposition. 3D coordinates for visualizer output.
//!
//! Output: galaxy_map.json + galaxy_matrix.bin + phases.bin + scan_metadata.json

use super::wave_analysis as wa;
use crate::common::wave_model::*;
use std::path::Path;

// ─── Relationship catalog ───

struct CatalogEntry {
    name: &'static str,
    angle_deg: f32,
    orb_deg: f32,
    harmonic: usize,
}

const CATALOG: &[CatalogEntry] = &[
    CatalogEntry { name: "conjunction", angle_deg: 0.0, orb_deg: 8.0, harmonic: 1 },
    CatalogEntry { name: "opposition", angle_deg: 180.0, orb_deg: 8.0, harmonic: 2 },
    CatalogEntry { name: "trine", angle_deg: 120.0, orb_deg: 8.0, harmonic: 3 },
    CatalogEntry { name: "square", angle_deg: 90.0, orb_deg: 7.0, harmonic: 4 },
    CatalogEntry { name: "quintile", angle_deg: 72.0, orb_deg: 2.0, harmonic: 5 },
    CatalogEntry { name: "sextile", angle_deg: 60.0, orb_deg: 6.0, harmonic: 6 },
    CatalogEntry { name: "semi-square", angle_deg: 45.0, orb_deg: 2.0, harmonic: 8 },
    CatalogEntry { name: "semi-sextile", angle_deg: 30.0, orb_deg: 2.0, harmonic: 12 },
    CatalogEntry { name: "quincunx", angle_deg: 150.0, orb_deg: 2.0, harmonic: 12 },
    CatalogEntry { name: "sesquiquadrate", angle_deg: 135.0, orb_deg: 2.0, harmonic: 8 },
    CatalogEntry { name: "bi-quintile", angle_deg: 144.0, orb_deg: 2.0, harmonic: 5 },
];

fn match_catalog(angle_deg: f32) -> Option<(&'static str, f32)> {
    for entry in CATALOG {
        let diff = (angle_deg - entry.angle_deg).abs();
        let diff = diff.min(360.0 - diff);
        if diff <= entry.orb_deg {
            return Some((entry.name, diff / entry.orb_deg));
        }
    }
    None
}

fn classify_grid(angle_deg: f32, m1: usize, m2: usize) -> &'static str {
    let grid1_step = 360.0 / m1 as f32;
    let grid2_step = 360.0 / m2 as f32;
    let threshold = 5.0;
    let g1_dist = (0..=m1).map(|k| {
        let d = (angle_deg - k as f32 * grid1_step).abs();
        d.min(360.0 - d)
    }).fold(f32::INFINITY, f32::min);
    let g2_dist = (0..=m2).map(|k| {
        let d = (angle_deg - k as f32 * grid2_step).abs();
        d.min(360.0 - d)
    }).fold(f32::INFINITY, f32::min);
    if g1_dist < threshold && g2_dist < threshold { "composite" }
    else if g1_dist < threshold { "grid1" }
    else if g2_dist < threshold { "grid2" }
    else { "approximate" }
}

// ─── Data structures ───

pub struct BandProfile {
    pub index: usize,
    pub position_3d: [f32; 3],
    pub mean_phase: f32,
    pub mean_magnitude: f32,
    pub circular_variance: f32,
    pub boundary_distance: f32,
    pub grid_assignment: &'static str,
    pub mag_min: f32,
    pub mag_max: f32,
    pub mag_std: f32,
}

pub struct PairRelation {
    pub band_a: usize,
    pub band_b: usize,
    pub mean_angular_distance_deg: f32,
    pub stability: f32,
    pub peak_n: usize,
    pub peak_strength: f32,
    pub shifted_coherence: f32,   // best MRL across harmonics {1,2,3,4,6}
    pub shifted_offset: f32,      // optimal phase offset at best harmonic
    pub shifted_harmonic: usize,  // which harmonic has the best MRL
    pub catalog_match: Option<&'static str>,
    pub catalog_orb_fit: f32,
    pub grid_native: &'static str,
}

pub struct TriadConstellation {
    pub bands: [usize; 3],
    pub mean_coherence_n3: f32,
}

pub struct QuartetConstellation {
    pub bands: [usize; 4],
    pub fwm_index_sum: usize,
    pub mean_coherence_n4: f32,
    pub phase_sum_mrl: f32,     // MRL of phase-sum trajectory
    pub category: u8,           // 0=random, 1=oscillating, 2=locked
}

pub struct LayerSummary {
    pub total_pairs: usize,
    pub significant_by_type: Vec<(String, usize)>,
    pub triadic_count: usize,
    pub fwm_quartet_total: usize,
    pub fwm_preserved: usize,    // was-high, still-high (genuine scaffolding kept)
    pub fwm_destroyed: usize,    // was-high, now-low (structure actively removed)
    pub fwm_created: usize,      // was-low, now-high (novel structure training built)
    pub fwm_noise: usize,        // was-low, still-low (never coherent)
    pub fwm_partial: usize,      // middle cases
    pub fwm_significant: usize,  // stored quartets
    pub fwm_mean_deviation: f32,
    pub fwm_hist_2d: [[u32; 10]; 10], // [baseline_bin][trained_bin]
    pub shifted_pair_count: usize,  // pairs with hidden coherence (MRL > 0.5 and MRL > 3*|peak|)
    pub quartet_locked: usize,      // quartets with phase-sum MRL > 0.7
    pub quartet_oscillating: usize, // quartets with phase-sum MRL 0.3-0.7
    pub sphere_fill_fraction: f32,
    pub sphere_center_fraction: f32,
    pub grid1_frac: f32,
    pub grid2_frac: f32,
    pub composite_frac: f32,
    pub approximate_frac: f32,
}

pub struct LayerScan {
    pub layer: usize,
    pub bands: Vec<BandProfile>,
    pub top_pairs: Vec<PairRelation>,
    pub triads: Vec<TriadConstellation>,
    pub fwm_quartets: Vec<QuartetConstellation>,
    pub summary: LayerSummary,
    pub pair_spectra: Vec<Vec<f32>>,  // [n_pairs][12] for binary output
}

pub struct GalaxyScan {
    pub layers: Vec<LayerScan>,
    pub n_bands: usize,
    pub n_positions: usize,
    pub agc_ceiling: f32,
    pub m1: usize,
    pub m2: usize,
}

// ─── Core scan ───

pub fn run_galaxy_scan(
    hidden_states: &[Vec<Vec<f32>>],  // per-layer [n_positions][n_embd]
    post_ln_f: &[Vec<f32>],
    n_bands: usize,
    per_layer_ceilings: &[f32],  // one per layer + one for post_ln_f (use last layer's)
    m1: usize,
    m2: usize,
) -> GalaxyScan {
    let n_positions = post_ln_f.len();
    let mut all_phases: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut all_hidden: Vec<&[Vec<f32>]> = Vec::new();
    for layer_hidden in hidden_states {
        all_phases.push(wa::extract_all_phases(layer_hidden, n_bands));
        all_hidden.push(layer_hidden);
    }
    all_phases.push(wa::extract_all_phases(post_ln_f, n_bands));
    all_hidden.push(post_ln_f);

    let mut layers = Vec::new();
    for (li, phases) in all_phases.iter().enumerate() {
        // Use per-layer ceiling, fallback to last available for post_ln_f layer
        let ceiling = per_layer_ceilings.get(li)
            .or_else(|| per_layer_ceilings.last())
            .copied().unwrap_or(1.0);
        layers.push(scan_layer(li, phases, all_hidden[li], n_bands, n_positions, ceiling, m1, m2));
    }

    let global_ceiling = per_layer_ceilings.first().copied().unwrap_or(1.0);
    GalaxyScan { layers, n_bands, n_positions, agc_ceiling: global_ceiling, m1, m2 }
}

fn scan_layer(
    layer: usize, phases: &[Vec<f32>], hidden: &[Vec<f32>],
    n_bands: usize, n_positions: usize,
    ceiling: f32, m1: usize, m2: usize,
) -> LayerScan {
    // Layer 0: Per-band profiles with real magnitudes
    let mut bands = Vec::with_capacity(n_bands);
    for k in 0..n_bands {
        let band_phases: Vec<f32> = phases.iter().map(|p| p[k]).collect();
        let cv = wa::circular_variance(&band_phases);
        let mean_phase = band_phases.iter().map(|&p| p.sin()).sum::<f32>()
            .atan2(band_phases.iter().map(|&p| p.cos()).sum::<f32>());
        // Real magnitude from hidden states: sqrt(r^2 + s^2) per position
        let band_mags: Vec<f32> = hidden.iter().map(|h| {
            if k * 2 + 1 < h.len() {
                (h[k * 2] * h[k * 2] + h[k * 2 + 1] * h[k * 2 + 1]).sqrt()
            } else { 0.0 }
        }).collect();
        let mean_mag = band_mags.iter().sum::<f32>() / band_mags.len().max(1) as f32;
        let mag_min = band_mags.iter().cloned().fold(f32::INFINITY, f32::min);
        let mag_max = band_mags.iter().cloned().fold(0.0f32, f32::max);
        let mag_std = {
            let var = band_mags.iter().map(|&m| (m - mean_mag) * (m - mean_mag)).sum::<f32>() / band_mags.len().max(1) as f32;
            var.sqrt()
        };
        let boundary_dist = if ceiling > 0.0 { (ceiling - mean_mag).max(0.0) / ceiling } else { 1.0 };
        let grid = if k < n_bands / 2 { "grid1" } else { "grid2" };
        let x = mean_mag * mean_phase.cos();
        let y = mean_mag * mean_phase.sin();
        let z = boundary_dist;
        bands.push(BandProfile {
            index: k, position_3d: [x, y, z], mean_phase, mean_magnitude: mean_mag,
            circular_variance: cv, boundary_distance: boundary_dist,
            grid_assignment: grid, mag_min, mag_max, mag_std,
        });
    }

    // Layer 1+2: Pairwise angular geometry + harmonic coherence
    let n_pairs = n_bands * (n_bands - 1) / 2;
    let mut all_pairs = Vec::with_capacity(n_pairs);
    let mut pair_spectra = Vec::with_capacity(n_pairs);
    let mut sig_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut grid_counts = [0usize; 4]; // grid1, grid2, composite, approximate

    for i in 0..n_bands {
        for j in (i+1)..n_bands {
            let phases_i: Vec<f32> = phases.iter().map(|p| p[i]).collect();
            let phases_j: Vec<f32> = phases.iter().map(|p| p[j]).collect();
            let spectrum = wa::harmonic_spectrum(&phases_i, &phases_j, 12);
            let (peak_str, peak_n) = wa::best_harmonic_coherence(&phases_i, &phases_j, 12);

            // Shifted coherence (MRL) — best across harmonics 1,2,3,4,6
            // MRL captures coherence at ANY fixed offset, not just zero
            let mut best_mrl = 0.0f32;
            let mut best_offset = 0.0f32;
            let mut best_mrl_n = 1usize;
            for &nh in &[1u32, 2, 3, 4, 6] {
                let n_f = nh as f32;
                let mut ss = 0.0f32;
                let mut sc = 0.0f32;
                for pos in 0..n_positions {
                    let delta = n_f * (phases[pos][i] - phases[pos][j]);
                    ss += delta.sin();
                    sc += delta.cos();
                }
                let this_mrl = (ss * ss + sc * sc).sqrt() / n_positions as f32;
                if this_mrl > best_mrl {
                    best_mrl = this_mrl;
                    best_offset = ss.atan2(sc) / n_f;
                    best_mrl_n = nh as usize;
                }
            }
            let mrl = best_mrl;
            let shifted_offset = best_offset;

            // Mean angular distance
            let diffs: Vec<f32> = phases_i.iter().zip(&phases_j)
                .map(|(&a, &b)| {
                    let d = (a - b).abs() % (2.0 * std::f32::consts::PI);
                    if d > std::f32::consts::PI { 2.0 * std::f32::consts::PI - d } else { d }
                }).collect();
            let mean_dist = diffs.iter().sum::<f32>() / n_positions as f32;
            let mean_dist_deg = mean_dist.to_degrees();

            // Stability
            let diff_cv = wa::circular_variance(&diffs.iter().map(|&d| d).collect::<Vec<_>>());
            let stability = 1.0 - diff_cv;

            // Catalog match
            let cat = match_catalog(mean_dist_deg);
            let cat_name = cat.map(|(n, _)| n);
            let cat_fit = cat.map(|(_, f)| f).unwrap_or(1.0);
            if let Some(name) = cat_name {
                *sig_counts.entry(name.to_string()).or_insert(0) += 1;
            }

            // Grid nativity
            let grid = classify_grid(mean_dist_deg, m1, m2);
            match grid {
                "grid1" => grid_counts[0] += 1,
                "grid2" => grid_counts[1] += 1,
                "composite" => grid_counts[2] += 1,
                _ => grid_counts[3] += 1,
            }

            all_pairs.push(PairRelation {
                band_a: i, band_b: j, mean_angular_distance_deg: mean_dist_deg,
                stability, peak_n, peak_strength: peak_str,
                shifted_coherence: mrl, shifted_offset, shifted_harmonic: best_mrl_n,
                catalog_match: cat_name, catalog_orb_fit: cat_fit, grid_native: grid,
            });
            pair_spectra.push(spectrum);
        }
    }

    // Top pairs: merge top-100 by peak_strength with top-100 by shifted_coherence, cap at 200
    let mut top_by_peak: Vec<usize> = (0..all_pairs.len()).collect();
    top_by_peak.sort_by(|&a, &b| all_pairs[b].peak_strength.partial_cmp(&all_pairs[a].peak_strength).unwrap_or(std::cmp::Ordering::Equal));
    top_by_peak.truncate(100);
    let mut top_by_mrl: Vec<usize> = (0..all_pairs.len()).collect();
    top_by_mrl.sort_by(|&a, &b| all_pairs[b].shifted_coherence.partial_cmp(&all_pairs[a].shifted_coherence).unwrap_or(std::cmp::Ordering::Equal));
    top_by_mrl.truncate(100);
    // Merge and deduplicate
    let mut seen = std::collections::HashSet::new();
    let mut top_indices = Vec::new();
    for idx in top_by_peak.into_iter().chain(top_by_mrl.into_iter()) {
        if seen.insert(idx) { top_indices.push(idx); }
    }
    top_indices.truncate(200);
    let top_pairs: Vec<PairRelation> = top_indices.into_iter().map(|idx| {
        let p = &all_pairs[idx];
        PairRelation {
            band_a: p.band_a, band_b: p.band_b,
            mean_angular_distance_deg: p.mean_angular_distance_deg,
            stability: p.stability, peak_n: p.peak_n, peak_strength: p.peak_strength,
            shifted_coherence: p.shifted_coherence, shifted_offset: p.shifted_offset, shifted_harmonic: p.shifted_harmonic,
            catalog_match: p.catalog_match, catalog_orb_fit: p.catalog_orb_fit,
            grid_native: p.grid_native,
        }
    }).collect();

    // Count shifted pairs (hidden coherence)
    let shifted_pair_count = all_pairs.iter()
        .filter(|p| p.shifted_coherence > 0.5 && p.shifted_coherence > 3.0 * p.peak_strength.abs().max(0.01))
        .count();

    // Layer 3: Triads (brute-force enumerate, filter by trine orb)
    let trine_orb = 15.0f32; // permissive for first build
    let mut triads = Vec::new();
    for i in 0..n_bands {
        for j in (i+1)..n_bands {
            for k_idx in (j+1)..n_bands {
                // Check all three pairwise distances near 120deg
                let get_pair_dist = |a: usize, b: usize| -> f32 {
                    let pa: Vec<f32> = phases.iter().map(|p| p[a]).collect();
                    let pb: Vec<f32> = phases.iter().map(|p| p[b]).collect();
                    let diffs: Vec<f32> = pa.iter().zip(&pb).map(|(&x, &y)| {
                        let d = (x - y).abs() % (2.0 * std::f32::consts::PI);
                        if d > std::f32::consts::PI { 2.0 * std::f32::consts::PI - d } else { d }
                    }).collect();
                    diffs.iter().sum::<f32>() / n_positions as f32
                };
                let d_ij = get_pair_dist(i, j).to_degrees();
                let d_jk = get_pair_dist(j, k_idx).to_degrees();
                let d_ik = get_pair_dist(i, k_idx).to_degrees();
                if (d_ij - 120.0).abs() < trine_orb
                    && (d_jk - 120.0).abs() < trine_orb
                    && (d_ik - 120.0).abs() < trine_orb
                {
                    let pi: Vec<f32> = phases.iter().map(|p| p[i]).collect();
                    let pj: Vec<f32> = phases.iter().map(|p| p[j]).collect();
                    let pk: Vec<f32> = phases.iter().map(|p| p[k_idx]).collect();
                    let c_ij = wa::best_harmonic_coherence(&pi, &pj, 12).0;
                    let c_jk = wa::best_harmonic_coherence(&pj, &pk, 12).0;
                    let c_ik = wa::best_harmonic_coherence(&pi, &pk, 12).0;
                    triads.push(TriadConstellation {
                        bands: [i, j, k_idx],
                        mean_coherence_n3: (c_ij + c_jk + c_ik) / 3.0,
                    });
                }
            }
        }
    }

    // Layer 3: FWM quartets (a+b=c+d constraint) with deviation from embedding baseline
    //
    // The multi-grid embedding provides structural coherence for some quartets
    // (37% at default grids m1=5,m2=7). Training can increase or decrease
    // coherence from this baseline. We measure SIGNED deviation:
    //   positive = training created coherence (moved toward matching)
    //   negative = training broke coherence (moved away from matching)
    //
    // Compute embedding baseline: phase_sum_diff for each quartet from the
    // multi-grid embedding (deterministic, depends only on n_bands, m1, m2).
    let half = n_bands / 2;
    let embed_phases: Vec<f32> = (0..n_bands).map(|k| {
        if k < half {
            2.0 * std::f32::consts::PI * ((k + 1) % m1) as f32 / m1 as f32
        } else {
            2.0 * std::f32::consts::PI * (((k - half) + 1) % m2) as f32 / m2 as f32
        }
    }).collect();

    let mut fwm_quartets = Vec::new();
    let mut fwm_total = 0usize;
    // 4-category classification based on (baseline, trained) coherence
    // Thresholds: high >= 0.7, low < 0.3, middle = 0.3..0.7
    let coh_high = 0.7f32;
    let coh_low = 0.3f32;
    let mut fwm_preserved = 0usize;  // was-high, still-high
    let mut fwm_destroyed = 0usize;  // was-high, now-low
    let mut fwm_created = 0usize;    // was-low, now-high
    let mut fwm_noise = 0usize;      // was-low, still-low
    let mut fwm_partial = 0usize;    // middle cases
    let mut fwm_dev_sum = 0.0f32;
    // 2D histogram: (baseline_coh, trained_coh) in 10x10 bins [0,0.1),[0.1,0.2),...,[0.9,1.0]
    let mut fwm_hist_2d = [[0u32; 10]; 10];

    for a in 0..n_bands {
        for b in (a+1)..n_bands {
            let sum = a + b;
            for c in 0..n_bands {
                let d = sum.wrapping_sub(c);
                if d < n_bands && d > c && c != a && c != b && d != a && d != b {
                    // Embedding baseline coherence for this quartet
                    let base_diff = (embed_phases[a] + embed_phases[b]) - (embed_phases[c] + embed_phases[d]);
                    let base_coh = base_diff.cos().abs();

                    // Trained coherence: mean phase difference across positions
                    let trained_diffs: Vec<f32> = phases.iter().map(|p| {
                        ((p[a] + p[b]) - (p[c] + p[d])).cos()
                    }).collect();
                    let trained_coh = (trained_diffs.iter().sum::<f32>() / n_positions as f32).abs();

                    let deviation = trained_coh - base_coh;
                    fwm_total += 1;
                    fwm_dev_sum += deviation;

                    // 2D histogram bin
                    let bi = (base_coh * 10.0).min(9.0) as usize;
                    let ti = (trained_coh * 10.0).min(9.0) as usize;
                    fwm_hist_2d[bi][ti] += 1;

                    // 4-category: (baseline high/low) x (trained high/low)
                    if base_coh >= coh_high && trained_coh >= coh_high { fwm_preserved += 1; }
                    else if base_coh >= coh_high && trained_coh < coh_low { fwm_destroyed += 1; }
                    else if base_coh < coh_low && trained_coh >= coh_high { fwm_created += 1; }
                    else if base_coh < coh_low && trained_coh < coh_low { fwm_noise += 1; }
                    else { fwm_partial += 1; }

                    // Phase-sum trajectory MRL for quartet classification
                    let (ps_ss, ps_sc) = (0..n_positions).fold((0.0f32, 0.0f32), |(ss, sc), pos| {
                        let ps = phases[pos][a] + phases[pos][b] - phases[pos][c] - phases[pos][d];
                        (ss + ps.sin(), sc + ps.cos())
                    });
                    let ps_mrl = (ps_ss * ps_ss + ps_sc * ps_sc).sqrt() / n_positions as f32;
                    let cat = if ps_mrl > 0.7 { 2u8 } // locked
                              else if ps_mrl > 0.3 { 1u8 } // oscillating
                              else { 0u8 }; // random

                    // Store quartets with significant deviation, novel creation, or non-random trajectory
                    if deviation.abs() > 0.15 || (base_coh < coh_low && trained_coh >= coh_high) || cat > 0 {
                        fwm_quartets.push(QuartetConstellation {
                            bands: [a, b, c, d],
                            fwm_index_sum: sum,
                            mean_coherence_n4: trained_coh,
                            phase_sum_mrl: ps_mrl,
                            category: cat,
                        });
                    }
                }
            }
        }
    }

    // Layer 5: Summary
    let sig_by_type: Vec<(String, usize)> = sig_counts.into_iter().collect();
    let fill_frac = bands.iter().filter(|b| b.boundary_distance < 0.1).count() as f32 / n_bands as f32;
    let center_frac = bands.iter().filter(|b| b.mean_magnitude < 0.1 * ceiling).count() as f32 / n_bands as f32;
    let total = grid_counts.iter().sum::<usize>().max(1) as f32;

    let fwm_mean_dev = if fwm_total > 0 { fwm_dev_sum / fwm_total as f32 } else { 0.0 };

    let summary = LayerSummary {
        total_pairs: n_pairs,
        significant_by_type: sig_by_type,
        triadic_count: triads.len(),
        fwm_quartet_total: fwm_total,
        fwm_preserved,
        fwm_destroyed,
        fwm_created,
        fwm_noise,
        fwm_partial,
        fwm_significant: fwm_quartets.len(),
        fwm_hist_2d,
        fwm_mean_deviation: fwm_mean_dev,
        shifted_pair_count,
        quartet_locked: fwm_quartets.iter().filter(|q| q.category == 2).count(),
        quartet_oscillating: fwm_quartets.iter().filter(|q| q.category == 1).count(),
        sphere_fill_fraction: fill_frac,
        sphere_center_fraction: center_frac,
        grid1_frac: grid_counts[0] as f32 / total,
        grid2_frac: grid_counts[1] as f32 / total,
        composite_frac: grid_counts[2] as f32 / total,
        approximate_frac: grid_counts[3] as f32 / total,
    };

    LayerScan { layer, bands, top_pairs, triads, fwm_quartets, summary, pair_spectra }
}

// ─── Output writers ───

pub fn write_galaxy_map_json(scan: &GalaxyScan, path: &Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;

    write!(f, r#"{{"schema_version":"1.0","n_layers":{},"n_bands":{},"n_positions":{},"agc_ceiling":{:.4},"grids":{{"m1":{},"m2":{}}},"layers":["#,
        scan.layers.len(), scan.n_bands, scan.n_positions, scan.agc_ceiling, scan.m1, scan.m2)?;

    for (li, layer) in scan.layers.iter().enumerate() {
        if li > 0 { write!(f, ",")?; }
        write!(f, r#"{{"layer":{},"#, layer.layer)?;

        // Bands
        write!(f, r#""bands":["#)?;
        for (bi, b) in layer.bands.iter().enumerate() {
            if bi > 0 { write!(f, ",")?; }
            write!(f, r#"{{"i":{},"pos":[{:.3},{:.3},{:.3}],"phase":{:.3},"mag":{:.3},"cv":{:.3},"grid":"{}"}}"#,
                b.index, b.position_3d[0], b.position_3d[1], b.position_3d[2],
                b.mean_phase, b.mean_magnitude, b.circular_variance, b.grid_assignment)?;
        }
        write!(f, "],")?;

        // Top pairs
        write!(f, r#""top_pairs":["#)?;
        for (pi, p) in layer.top_pairs.iter().enumerate() {
            if pi > 0 { write!(f, ",")?; }
            let cat = p.catalog_match.unwrap_or("none");
            write!(f, r#"{{"a":{},"b":{},"dist_deg":{:.1},"stability":{:.3},"peak_n":{},"peak_str":{:.3},"mrl":{:.3},"mrl_n":{},"offset":{:.3},"cat":"{}","grid":"{}"}}"#,
                p.band_a, p.band_b, p.mean_angular_distance_deg, p.stability,
                p.peak_n, p.peak_strength, p.shifted_coherence, p.shifted_harmonic, p.shifted_offset,
                cat, p.grid_native)?;
        }
        write!(f, "],")?;

        // Triads
        write!(f, r#""triads":["#)?;
        for (ti, t) in layer.triads.iter().enumerate() {
            if ti > 0 { write!(f, ",")?; }
            write!(f, r#"{{"bands":[{},{},{}],"coh":{:.3}}}"#,
                t.bands[0], t.bands[1], t.bands[2], t.mean_coherence_n3)?;
        }
        write!(f, "],")?;

        // FWM quartets
        write!(f, r#""fwm_quartets":["#)?;
        for (qi, q) in layer.fwm_quartets.iter().enumerate() {
            if qi > 0 { write!(f, ",")?; }
            let cat_str = match q.category { 2 => "locked", 1 => "oscillating", _ => "random" };
            write!(f, r#"{{"bands":[{},{},{},{}],"sum":{},"coh":{:.3},"ps_mrl":{:.3},"cat":"{}"}}"#,
                q.bands[0], q.bands[1], q.bands[2], q.bands[3],
                q.fwm_index_sum, q.mean_coherence_n4, q.phase_sum_mrl, cat_str)?;
        }
        write!(f, "],")?;

        // Summary (includes full-matrix catalog counts + FWM deviation stats)
        let sig: Vec<String> = layer.summary.significant_by_type.iter()
            .map(|(k, v)| format!("\"{}\":{}", k, v)).collect();
        write!(f, r#""summary":{{"pairs":{},"triads":{},"shifted_pairs":{},"fwm":{{"total":{},"preserved":{},"destroyed":{},"created":{},"noise":{},"partial":{},"significant":{},"mean_dev":{:.4}}},"fill":{:.3},"center":{:.3},"catalog":{{{}}}, "grid":{{"g1":{:.3},"g2":{:.3},"comp":{:.3},"approx":{:.3}}}}}"#,
            layer.summary.total_pairs, layer.summary.triadic_count,
            layer.summary.shifted_pair_count,
            layer.summary.fwm_quartet_total, layer.summary.fwm_preserved,
            layer.summary.fwm_destroyed, layer.summary.fwm_created,
            layer.summary.fwm_noise, layer.summary.fwm_partial,
            layer.summary.fwm_significant, layer.summary.fwm_mean_deviation,
            layer.summary.sphere_fill_fraction, layer.summary.sphere_center_fraction,
            sig.join(","),
            layer.summary.grid1_frac, layer.summary.grid2_frac,
            layer.summary.composite_frac, layer.summary.approximate_frac)?;

        // 2D histogram: [baseline_bin][trained_bin], 10x10
        write!(f, r#","fwm_hist_2d":["#)?;
        for (ri, row) in layer.summary.fwm_hist_2d.iter().enumerate() {
            if ri > 0 { write!(f, ",")?; }
            write!(f, "[")?;
            for (ci, &val) in row.iter().enumerate() {
                if ci > 0 { write!(f, ",")?; }
                write!(f, "{}", val)?;
            }
            write!(f, "]")?;
        }
        write!(f, "]")?;

        write!(f, "}}")?;
    }
    write!(f, "]}}")?;
    Ok(())
}

pub fn write_galaxy_matrix_bin(scan: &GalaxyScan, path: &Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    // Header: GALX magic + version + n_layers + n_bands
    f.write_all(&0x47414C58u32.to_le_bytes())?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(scan.layers.len() as u32).to_le_bytes())?;
    f.write_all(&(scan.n_bands as u32).to_le_bytes())?;
    // Per-layer pair spectra
    for layer in &scan.layers {
        for spectrum in &layer.pair_spectra {
            for &v in spectrum {
                f.write_all(&v.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

pub fn write_phases_bin(all_phases: &[Vec<Vec<f32>>], n_bands: usize, path: &Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    let n_layers = all_phases.len();
    let n_positions = if !all_phases.is_empty() && !all_phases[0].is_empty() { all_phases[0].len() } else { 0 };
    // Header: PHAS magic + version + dims
    f.write_all(&0x50484153u32.to_le_bytes())?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(n_layers as u32).to_le_bytes())?;
    f.write_all(&(n_bands as u32).to_le_bytes())?;
    f.write_all(&(n_positions as u32).to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?; // pad
    for layer_phases in all_phases {
        for pos_phases in layer_phases {
            for &v in pos_phases {
                f.write_all(&v.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

/// Run the full scan and write all output files.
pub fn run_and_write_full_scan(
    hidden_states: &[Vec<Vec<f32>>],
    post_ln_f: &[Vec<f32>],
    n_bands: usize,
    per_layer_ceilings: &[f32],
    m1: usize,
    m2: usize,
    output_dir: &Path,
) -> std::io::Result<GalaxyScan> {
    let scan = run_galaxy_scan(hidden_states, post_ln_f, n_bands, per_layer_ceilings, m1, m2);
    std::fs::create_dir_all(output_dir)?;
    write_galaxy_map_json(&scan, &output_dir.join("galaxy_map.json"))?;
    write_galaxy_matrix_bin(&scan, &output_dir.join("galaxy_matrix.bin"))?;

    // Write phases
    let mut all_phases = Vec::new();
    for layer_hidden in hidden_states {
        all_phases.push(wa::extract_all_phases(layer_hidden, n_bands));
    }
    all_phases.push(wa::extract_all_phases(post_ln_f, n_bands));
    write_phases_bin(&all_phases, n_bands, &output_dir.join("phases.bin"))?;

    Ok(scan)
}

/// Print a console summary of the scan.
pub fn print_summary(scan: &GalaxyScan) {
    if let Some(final_layer) = scan.layers.last() {
        let total_sig: usize = final_layer.summary.significant_by_type.iter().map(|(_, v)| v).sum();
        let s = &final_layer.summary;
        eprintln!("  Galaxy: {} sig pairs, {} shifted pairs, {} triads", total_sig, s.shifted_pair_count, s.triadic_count);
        eprintln!("  FWM quartets ({} total): preserved={}, destroyed={}, created={}, noise={}, partial={}",
            s.fwm_quartet_total, s.fwm_preserved, s.fwm_destroyed,
            s.fwm_created, s.fwm_noise, s.fwm_partial);
        eprintln!("    mean_dev={:.4}, locked={}, oscillating={}",
            s.fwm_mean_deviation, s.quartet_locked, s.quartet_oscillating);
        eprintln!("  Grid: g1={:.1}% g2={:.1}% comp={:.1}% approx={:.1}%",
            s.grid1_frac * 100.0, s.grid2_frac * 100.0,
            s.composite_frac * 100.0, s.approximate_frac * 100.0);
    }
}

/// CLI entry point for galaxy-scan subcommand.
pub fn run_galaxy_scan_cli(
    checkpoint_path: &str,
    n_bands: usize, n_head: usize, n_layers: usize,
    out_proj_groups: usize, alpha: f32, beta: f32,
    data_path: String, scan_corpus: Option<String>, m1: usize, m2: usize,
) {
    // Shared loader — tries learnable_ode / corrector / learnable_attn variants
    // and picks the one whose param count matches the checkpoint.
    let (model, dims) = crate::common::wave_model::load_checkpoint_auto(
        checkpoint_path, n_bands, n_head, n_layers, out_proj_groups, alpha, beta,
    );

    let scan_path = scan_corpus.unwrap_or(data_path);
    let (tokens, _) = super::data_loader::load_data(&scan_path, false, None);
    let scan_len = tokens.len().min(200).min(128);
    let stencil = crate::fft_ode::StencilFft::new(n_bands * 2);
    let cache = crate::cpu::forward::forward_with_cache(
        &model, &tokens[..scan_len], dims, None, None, None, Some(&stencil), None, None, None,
    );
    let all_hidden: Vec<Vec<Vec<f32>>> = cache.block_caches.iter().map(|bc| bc.input.clone()).collect();
    let per_layer_ceilings: Vec<f32> = model.blocks.iter()
        .map(|b| (std::f32::consts::FRAC_PI_2 / (b.ffn.kerr.alpha + 4.0 * b.ffn.kerr.beta)).sqrt().max(0.5))
        .collect();
    let galaxy_dir = std::path::PathBuf::from(checkpoint_path.replace(".bin", "_galaxy"));
    match run_and_write_full_scan(
        &all_hidden, &cache.post_ln_f, n_bands, &per_layer_ceilings, m1, m2, &galaxy_dir,
    ) {
        Ok(scan) => { println!("Galaxy map written to: {}", galaxy_dir.display()); print_summary(&scan); }
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    }
}
