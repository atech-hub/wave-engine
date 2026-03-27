//! Sub-harmonic diagnostics — detect cross-band information channels.
//!
//! The Kerr-ODE couples adjacent bands via β·Σ|ψ_neighbour|². This coupling
//! can create inter-modulation products — information encoded in the
//! *relationships* between bands, not just in individual bands.
//!
//! Five diagnostics:
//! 1. Differential phase clustering (phase differences between adjacent bands)
//! 2. Magnitude coupling (correlation decay with distance)
//! 3. Inter-modulation spectrum (FFT of magnitude profile)
//! 4. Cross-band semantic discrimination (Δθ and |ψ| based)
//! 5. Coupling energy budget (self vs cross modulation)

use std::f32::consts::PI;

// ─── Helpers ───────────────────────────────────────────────

fn extract_phases(hidden: &[f32], n_bands: usize) -> Vec<f32> {
    (0..n_bands).map(|k| {
        hidden[k * 2 + 1].atan2(hidden[k * 2])
    }).collect()
}

fn mean_resultant_length(angles: &[f32]) -> f32 {
    let n = angles.len() as f32;
    if n < 1.0 { return 0.0; }
    let sum_cos: f32 = angles.iter().map(|a| a.cos()).sum();
    let sum_sin: f32 = angles.iter().map(|a| a.sin()).sum();
    ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
}

fn pearson_corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    if n < 2.0 { return 0.0; }
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a < 1e-10 || var_b < 1e-10 { return 0.0; }
    cov / (var_a.sqrt() * var_b.sqrt())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 { return 0.0; }
    dot / (norm_a * norm_b)
}

// ─── Diagnostic 1: Differential Phase Clustering ───────────

/// Phase differences between adjacent bands for one hidden state.
fn differential_phases(hidden: &[f32], n_bands: usize) -> Vec<f32> {
    let phases = extract_phases(hidden, n_bands);
    (0..n_bands - 1).map(|k| {
        let mut diff = phases[k + 1] - phases[k];
        while diff > PI { diff -= 2.0 * PI; }
        while diff < -PI { diff += 2.0 * PI; }
        diff
    }).collect()
}

/// Mean resultant length of differential phases across positions.
pub fn differential_phase_clustering(hidden_states: &[Vec<f32>], n_bands: usize) -> f32 {
    let mut total_mrl = 0.0f32;
    for k in 0..n_bands - 1 {
        let diffs: Vec<f32> = hidden_states.iter()
            .map(|h| {
                let theta_k = h[k * 2 + 1].atan2(h[k * 2]);
                let theta_k1 = h[(k + 1) * 2 + 1].atan2(h[(k + 1) * 2]);
                let mut d = theta_k1 - theta_k;
                while d > PI { d -= 2.0 * PI; }
                while d < -PI { d += 2.0 * PI; }
                d
            })
            .collect();
        total_mrl += mean_resultant_length(&diffs);
    }
    total_mrl / (n_bands - 1) as f32
}

// ─── Diagnostic 2: Magnitude Coupling ──────────────────────

pub struct MagnitudeCoupling {
    pub coupling_decay: Vec<f32>, // correlation at d=1..5
    pub distant_corr: f32,       // d=10 control
    pub max_pair: (usize, usize, f32),
}

pub fn magnitude_coupling(hidden_states: &[Vec<f32>], n_bands: usize) -> MagnitudeCoupling {
    // Extract magnitudes per band across positions
    let mags: Vec<Vec<f32>> = (0..n_bands).map(|k| {
        hidden_states.iter().map(|h| {
            (h[k * 2] * h[k * 2] + h[k * 2 + 1] * h[k * 2 + 1]).sqrt()
        }).collect()
    }).collect();

    let corr_at_distance = |d: usize| -> f32 {
        let mut total = 0.0f32;
        let mut count = 0;
        for k in 0..n_bands.saturating_sub(d) {
            total += pearson_corr(&mags[k], &mags[k + d]).abs();
            count += 1;
        }
        if count > 0 { total / count as f32 } else { 0.0 }
    };

    let coupling_decay: Vec<f32> = (1..=5).map(|d| corr_at_distance(d)).collect();
    let distant_corr = corr_at_distance(10);

    // Find most correlated adjacent pair
    let mut max_pair = (0, 1, 0.0f32);
    for k in 0..n_bands.saturating_sub(1) {
        let c = pearson_corr(&mags[k], &mags[k + 1]).abs();
        if c > max_pair.2 { max_pair = (k, k + 1, c); }
    }

    MagnitudeCoupling { coupling_decay, distant_corr, max_pair }
}

// ─── Diagnostic 3: Inter-Modulation Spectrum ───────────────

pub struct InterModSpectrum {
    pub spectral_peaks: Vec<(usize, f32)>, // (period, normalised power)
    pub spectral_entropy: f32,
    pub dominant_period: usize,
}

pub fn intermod_spectrum(hidden_states: &[Vec<f32>], n_bands: usize) -> InterModSpectrum {
    use rustfft::{FftPlanner, num_complex::Complex};

    // Average magnitude profile across all positions
    let mut avg_mag = vec![0.0f32; n_bands];
    for h in hidden_states {
        for k in 0..n_bands {
            avg_mag[k] += (h[k * 2] * h[k * 2] + h[k * 2 + 1] * h[k * 2 + 1]).sqrt();
        }
    }
    for v in &mut avg_mag { *v /= hidden_states.len() as f32; }

    // Remove DC
    let mean: f32 = avg_mag.iter().sum::<f32>() / n_bands as f32;
    let centered: Vec<f32> = avg_mag.iter().map(|v| v - mean).collect();

    // FFT
    let fft_len = n_bands.next_power_of_two();
    let mut input: Vec<Complex<f32>> = centered.iter()
        .map(|&v| Complex::new(v, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_len)
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    fft.process(&mut input);

    // Power spectrum (skip DC)
    let power: Vec<f32> = input[1..fft_len / 2].iter()
        .map(|c| c.norm_sqr())
        .collect();

    let total_power: f32 = power.iter().sum::<f32>().max(1e-10);
    let mut peaks: Vec<(usize, f32)> = power.iter().enumerate()
        .map(|(i, &p)| (i + 1, p / total_power))
        .collect();
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Spectral entropy
    let probs: Vec<f32> = power.iter().map(|&p| p / total_power).collect();
    let entropy: f32 = -probs.iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f32>();
    let max_entropy = (power.len() as f32).ln().max(1e-10);

    let dominant = peaks.first().map(|&(f, _)| f).unwrap_or(1);

    InterModSpectrum {
        spectral_peaks: peaks.into_iter().take(5).collect(),
        spectral_entropy: entropy / max_entropy,
        dominant_period: dominant,
    }
}

// ─── Diagnostic 4: Cross-Band Semantic Discrimination ──────

pub struct CrossBandDiscrimination {
    pub per_band_ratio: f32,
    pub diff_phase_ratio: f32,
    pub magnitude_ratio: f32,
}

fn discrimination_ratio(
    representations: &[Vec<f32>],
    related: &[(usize, usize)],
    random: &[(usize, usize)],
) -> f32 {
    if related.is_empty() || random.is_empty() { return 1.0; }
    let rel_mean: f32 = related.iter()
        .map(|&(a, b)| cosine_similarity(&representations[a], &representations[b]).abs())
        .sum::<f32>() / related.len() as f32;
    let rand_mean: f32 = random.iter()
        .map(|&(a, b)| cosine_similarity(&representations[a], &representations[b]).abs())
        .sum::<f32>() / random.len() as f32;
    if rand_mean < 1e-10 { return rel_mean * 100.0; }
    rel_mean / rand_mean
}

pub fn cross_band_discrimination(
    hidden_states: &[Vec<f32>],
    related_pairs: &[(usize, usize)],
    random_pairs: &[(usize, usize)],
    n_bands: usize,
) -> CrossBandDiscrimination {
    // Per-band phase
    let phases: Vec<Vec<f32>> = hidden_states.iter()
        .map(|h| extract_phases(h, n_bands))
        .collect();
    let per_band = discrimination_ratio(&phases, related_pairs, random_pairs);

    // Differential phase
    let diff_phases: Vec<Vec<f32>> = hidden_states.iter()
        .map(|h| differential_phases(h, n_bands))
        .collect();
    let diff_ratio = discrimination_ratio(&diff_phases, related_pairs, random_pairs);

    // Magnitude pattern
    let magnitudes: Vec<Vec<f32>> = hidden_states.iter()
        .map(|h| {
            (0..n_bands).map(|k| {
                (h[k * 2] * h[k * 2] + h[k * 2 + 1] * h[k * 2 + 1]).sqrt()
            }).collect()
        })
        .collect();
    let mag_ratio = discrimination_ratio(&magnitudes, related_pairs, random_pairs);

    CrossBandDiscrimination { per_band_ratio: per_band, diff_phase_ratio: diff_ratio, magnitude_ratio: mag_ratio }
}

// ─── Diagnostic 5: Coupling Energy Budget ──────────────────

pub struct CouplingBudget {
    pub self_mod_energy: f32,
    pub cross_mod_energy: f32,
    pub cross_self_ratio: f32,
    pub most_coupled_band: (usize, f32),
    pub least_coupled_band: (usize, f32),
}

pub fn coupling_budget(
    hidden_states: &[Vec<f32>],
    n_bands: usize,
    alpha: f32,
    beta: f32,
) -> CouplingBudget {
    let n_pos = hidden_states.len();
    let mut per_band_self = vec![0.0f32; n_bands];
    let mut per_band_cross = vec![0.0f32; n_bands];

    for h in hidden_states {
        let mag_sq: Vec<f32> = (0..n_bands).map(|k| {
            h[k * 2] * h[k * 2] + h[k * 2 + 1] * h[k * 2 + 1]
        }).collect();

        for k in 0..n_bands {
            let self_mod = alpha * mag_sq[k];
            let mut cross_mod = 0.0f32;
            if k >= 2 { cross_mod += mag_sq[k - 2]; }
            if k >= 1 { cross_mod += mag_sq[k - 1]; }
            if k + 1 < n_bands { cross_mod += mag_sq[k + 1]; }
            if k + 2 < n_bands { cross_mod += mag_sq[k + 2]; }
            cross_mod *= beta;

            per_band_self[k] += self_mod;
            per_band_cross[k] += cross_mod;
        }
    }

    for v in &mut per_band_self { *v /= n_pos as f32; }
    for v in &mut per_band_cross { *v /= n_pos as f32; }

    let self_total: f32 = per_band_self.iter().sum::<f32>() / n_bands as f32;
    let cross_total: f32 = per_band_cross.iter().sum::<f32>() / n_bands as f32;

    let per_band_ratio: Vec<f32> = per_band_self.iter().zip(&per_band_cross)
        .map(|(&s, &c)| if s > 1e-10 { c / s } else { 0.0 })
        .collect();

    let most = per_band_ratio.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i, v)).unwrap_or((0, 0.0));
    let least = per_band_ratio.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i, v)).unwrap_or((0, 0.0));

    CouplingBudget {
        self_mod_energy: self_total,
        cross_mod_energy: cross_total,
        cross_self_ratio: if self_total > 1e-10 { cross_total / self_total } else { 0.0 },
        most_coupled_band: most,
        least_coupled_band: least,
    }
}

// ─── Report ────────────────────────────────────────────────

pub fn print_report(
    hidden_states: &[Vec<f32>],
    related_pairs: &[(usize, usize)],
    random_pairs: &[(usize, usize)],
    n_bands: usize,
    alpha: f32,
    beta: f32,
    per_band_clustering: f32,
) {
    println!("\n=== Sub-Harmonic Diagnostics ===");

    // 1. Differential Phase Clustering
    let diff_clust = differential_phase_clustering(hidden_states, n_bands);
    println!("\n1. Differential Phase Clustering");
    println!("   Per-band clustering:    {:.3}  (existing)", per_band_clustering);
    println!("   Differential clustering: {:.3}  (phase differences between adjacent bands)", diff_clust);
    let ratio = if per_band_clustering > 1e-10 { diff_clust / per_band_clustering } else { 0.0 };
    println!("   Ratio: {:.2}x", ratio);
    if diff_clust > per_band_clustering {
        println!("   → Cross-band phase encoding STRONGER than within-band");
    } else {
        println!("   → Information primarily in individual bands");
    }

    // 2. Magnitude Coupling
    let mc = magnitude_coupling(hidden_states, n_bands);
    println!("\n2. Magnitude Coupling");
    for (i, &c) in mc.coupling_decay.iter().enumerate() {
        let label = if i < 2 { " ← ODE coupled" } else { "" };
        println!("   d={}: {:.3}{}", i + 1, c, label);
    }
    println!("   d=10: {:.3}  (distant control)", mc.distant_corr);
    println!("   Most correlated pair: band {} ↔ band {} (r={:.3})", mc.max_pair.0, mc.max_pair.1, mc.max_pair.2);
    if mc.coupling_decay.len() >= 3 && mc.coupling_decay[0] > mc.coupling_decay[2] * 1.5 {
        println!("   → Coupling decay matches ODE stencil signature");
    }

    // 3. Inter-Modulation Spectrum
    let ims = intermod_spectrum(hidden_states, n_bands);
    println!("\n3. Inter-Modulation Spectrum");
    println!("   Magnitude profile FFT — energy distribution pattern:");
    for (i, &(period, power)) in ims.spectral_peaks.iter().take(3).enumerate() {
        println!("     Peak {}: period={} ({:.1}% of energy)", i + 1, period, power * 100.0);
    }
    println!("   Spectral entropy: {:.3}  (0=concentrated, 1=uniform)", ims.spectral_entropy);
    if ims.spectral_entropy < 0.7 {
        println!("   → Concentrated energy pattern — inter-modulation structure present");
    } else {
        println!("   → Diffuse energy — limited inter-modulation structure");
    }

    // 4. Cross-Band Semantic Discrimination
    let cbd = cross_band_discrimination(hidden_states, related_pairs, random_pairs, n_bands);
    println!("\n4. Cross-Band Semantic Discrimination");
    println!("   Per-band phase (θ):      ratio = {:.2}x", cbd.per_band_ratio);
    println!("   Differential phase (Δθ): ratio = {:.2}x", cbd.diff_phase_ratio);
    println!("   Magnitude pattern (|ψ|): ratio = {:.2}x", cbd.magnitude_ratio);
    if cbd.diff_phase_ratio > 1.2 || cbd.magnitude_ratio > 1.2 {
        println!("   → CROSS-BAND SEMANTICS DETECTED — information in band relationships!");
    } else if cbd.diff_phase_ratio > cbd.per_band_ratio * 1.1 {
        println!("   → Emerging cross-band signal");
    }

    // 5. Coupling Energy Budget
    let cb = coupling_budget(hidden_states, n_bands, alpha, beta);
    println!("\n5. Coupling Energy Budget");
    println!("   Self-modulation (α·|ψ|²):     {:.4}", cb.self_mod_energy);
    println!("   Cross-modulation (β·Σ|ψ_n|²): {:.4}", cb.cross_mod_energy);
    println!("   Cross/Self ratio:              {:.2}x", cb.cross_self_ratio);
    println!("   Most coupled:  band {} ({:.2}x)", cb.most_coupled_band.0, cb.most_coupled_band.1);
    println!("   Least coupled: band {} ({:.2}x)", cb.least_coupled_band.0, cb.least_coupled_band.1);
    if cb.cross_self_ratio > 1.0 {
        println!("   → Neighbours DOMINATE — sub-channels physically plausible");
    } else if cb.cross_self_ratio > 0.5 {
        println!("   → Balanced — both mechanisms contributing");
    } else {
        println!("   → Self-dominated — bands mostly independent");
    }
}
