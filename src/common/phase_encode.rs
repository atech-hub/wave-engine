//! Phase Encode Tool — direct phase encoding into the model.
//!
//! Bypasses the normal token→embedding→ODE pipeline. Encodes anything
//! directly into phase angles and observes what the ODE dynamics do.
//!
//! Five encoding modes: text, number, catalog, raw phases, compound.
//! --relate mode computes per-harmonic coherence profiles between pairs.

use std::f32::consts::PI;

// ─── Encoding types ───

/// A parsed encoding specification.
pub enum Encoding {
    Text(String),
    Number(u64),
    Catalog(Vec<CatalogConfig>),
    RawPhases(Vec<(usize, f32)>),
}

/// A single catalog relationship configuration.
pub struct CatalogConfig {
    pub rel_type: RelationType,
    pub bands: Vec<usize>,
}

/// Geometric relationship types from the catalog.
#[derive(Clone, Copy, Debug)]
pub enum RelationType {
    Conjunction,
    Opposition,
    Trine,
    Square,
    Quintile,
    Sextile,
    SemiSquare,
    SemiSextile,
    Quincunx,
    Sesquiquadrate,
    BiQuintile,
    Quartet,
}

impl RelationType {
    /// Target angle in radians for this relationship type.
    pub fn angle_rad(self) -> f32 {
        match self {
            Self::Conjunction     => 0.0,
            Self::Opposition      => PI,
            Self::Trine           => 2.0 * PI / 3.0,
            Self::Square          => PI / 2.0,
            Self::Quintile        => 2.0 * PI / 5.0,
            Self::Sextile         => PI / 3.0,
            Self::SemiSquare      => PI / 4.0,
            Self::SemiSextile     => PI / 6.0,
            Self::Quincunx        => 5.0 * PI / 6.0,
            Self::Sesquiquadrate  => 3.0 * PI / 4.0,
            Self::BiQuintile      => 4.0 * PI / 5.0,
            Self::Quartet         => 0.0, // phase-matching, not angular
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Conjunction     => "conjunction",
            Self::Opposition      => "opposition",
            Self::Trine           => "trine",
            Self::Square          => "square",
            Self::Quintile        => "quintile",
            Self::Sextile         => "sextile",
            Self::SemiSquare      => "semi-square",
            Self::SemiSextile     => "semi-sextile",
            Self::Quincunx        => "quincunx",
            Self::Sesquiquadrate  => "sesquiquadrate",
            Self::BiQuintile      => "bi-quintile",
            Self::Quartet         => "quartet",
        }
    }
}

fn parse_rel_type(s: &str) -> Option<RelationType> {
    match s.to_lowercase().as_str() {
        "conjunction" | "conj"        => Some(RelationType::Conjunction),
        "opposition" | "opp"         => Some(RelationType::Opposition),
        "trine"                       => Some(RelationType::Trine),
        "square"                      => Some(RelationType::Square),
        "quintile"                    => Some(RelationType::Quintile),
        "sextile"                     => Some(RelationType::Sextile),
        "semi-square" | "semisquare" => Some(RelationType::SemiSquare),
        "semi-sextile" | "semisextile" => Some(RelationType::SemiSextile),
        "quincunx"                    => Some(RelationType::Quincunx),
        "sesquiquadrate"              => Some(RelationType::Sesquiquadrate),
        "bi-quintile" | "biquintile" => Some(RelationType::BiQuintile),
        "quartet"                     => Some(RelationType::Quartet),
        _ => None,
    }
}

// ─── Catalog table (for output matching) ───

struct CatalogEntry {
    name: &'static str,
    angle_deg: f32,
    orb_deg: f32,
}

const CATALOG: &[CatalogEntry] = &[
    CatalogEntry { name: "conjunction",     angle_deg:   0.0, orb_deg: 8.0 },
    CatalogEntry { name: "opposition",      angle_deg: 180.0, orb_deg: 8.0 },
    CatalogEntry { name: "trine",           angle_deg: 120.0, orb_deg: 8.0 },
    CatalogEntry { name: "square",          angle_deg:  90.0, orb_deg: 7.0 },
    CatalogEntry { name: "quintile",        angle_deg:  72.0, orb_deg: 2.0 },
    CatalogEntry { name: "sextile",         angle_deg:  60.0, orb_deg: 6.0 },
    CatalogEntry { name: "semi-square",     angle_deg:  45.0, orb_deg: 2.0 },
    CatalogEntry { name: "semi-sextile",    angle_deg:  30.0, orb_deg: 2.0 },
    CatalogEntry { name: "quincunx",        angle_deg: 150.0, orb_deg: 2.0 },
    CatalogEntry { name: "sesquiquadrate",  angle_deg: 135.0, orb_deg: 2.0 },
    CatalogEntry { name: "bi-quintile",     angle_deg: 144.0, orb_deg: 2.0 },
];

fn match_catalog(angle_deg: f32) -> Option<(&'static str, f32)> {
    for entry in CATALOG {
        let diff = (angle_deg - entry.angle_deg).abs();
        let diff = diff.min(360.0 - diff);
        if diff <= entry.orb_deg {
            return Some((entry.name, diff));
        }
    }
    None
}

// ─── Encoding functions ───

/// Encode a catalog configuration into a hidden state vector [n_embd].
/// All bands start at phase 0, magnitude 1. Specified bands are rotated.
pub fn encode_catalog_state(configs: &[CatalogConfig], n_bands: usize) -> Vec<f32> {
    let n_embd = n_bands * 2;
    let mut state = vec![0.0f32; n_embd];
    // All bands at phase 0, magnitude 1: r=1, s=0
    for k in 0..n_bands {
        state[k * 2] = 1.0;
    }

    for cfg in configs {
        apply_relationship(&mut state, cfg);
    }
    state
}

fn apply_relationship(state: &mut [f32], cfg: &CatalogConfig) {
    let n_bands = state.len() / 2;
    match cfg.rel_type {
        RelationType::Quartet => {
            // Phase-matched: θ_A + θ_B = θ_C + θ_D
            // Set A,B to phase 0, C to pi/4, D to -pi/4
            if cfg.bands.len() >= 4 {
                let a = cfg.bands[0].min(n_bands - 1);
                let b = cfg.bands[1].min(n_bands - 1);
                let c = cfg.bands[2].min(n_bands - 1);
                let d = cfg.bands[3].min(n_bands - 1);
                // A at 0, B at 0
                set_band_phase(state, a, 0.0);
                set_band_phase(state, b, 0.0);
                // C at π/4, D at -π/4 (sum = 0 = A+B)
                set_band_phase(state, c, PI / 4.0);
                set_band_phase(state, d, -PI / 4.0);
            }
        }
        _ => {
            // Angular relationships: rotate subsequent bands relative to first
            let angle = cfg.rel_type.angle_rad();
            if cfg.bands.is_empty() { return; }
            let first = cfg.bands[0].min(n_bands - 1);
            // First band stays at current phase (reference)
            let ref_phase = state[first * 2 + 1].atan2(state[first * 2]);
            for (i, &band) in cfg.bands.iter().enumerate().skip(1) {
                let band = band.min(n_bands - 1);
                let target = ref_phase + i as f32 * angle;
                set_band_phase(state, band, target);
            }
        }
    }
}

fn set_band_phase(state: &mut [f32], band: usize, phase: f32) {
    // Preserve magnitude, set phase
    let r = state[band * 2];
    let s = state[band * 2 + 1];
    let mag = (r * r + s * s).sqrt().max(1e-8);
    state[band * 2] = mag * phase.cos();
    state[band * 2 + 1] = mag * phase.sin();
}

/// Encode raw phase specifications into a hidden state.
pub fn encode_raw_phases(phases: &[(usize, f32)], n_bands: usize) -> Vec<f32> {
    let n_embd = n_bands * 2;
    let mut state = vec![0.0f32; n_embd];
    // All bands at phase 0, magnitude 1
    for k in 0..n_bands {
        state[k * 2] = 1.0;
    }
    for &(band, phase) in phases {
        if band < n_bands {
            state[band * 2] = phase.cos();
            state[band * 2 + 1] = phase.sin();
        }
    }
    state
}

/// Encode a number via multi-grid phases (same as embed.rs but for single value).
pub fn encode_number(n: u64, n_bands: usize, m1: usize, m2: usize) -> Vec<f32> {
    let n_embd = n_bands * 2;
    let half = n_bands / 2;
    let mut state = vec![0.0f32; n_embd];

    let theta1 = (n % m1 as u64) as f32 * 2.0 * PI / m1 as f32;
    for h in 0..half {
        let phase = (h + 1) as f32 * theta1;
        state[h * 2] = phase.cos();
        state[h * 2 + 1] = phase.sin();
    }
    let theta2 = (n % m2 as u64) as f32 * 2.0 * PI / m2 as f32;
    for h in 0..half {
        let idx = half + h;
        let phase = (h + 1) as f32 * theta2;
        state[idx * 2] = phase.cos();
        state[idx * 2 + 1] = phase.sin();
    }
    state
}

// ─── Parsing ───

/// Parse a catalog specification string like "trine:35,63+opposition:12,54"
pub fn parse_catalog_spec(s: &str) -> Vec<CatalogConfig> {
    s.split('+').filter_map(|part| {
        let part = part.trim();
        let colon = part.find(':')?;
        let rel_name = &part[..colon];
        let band_str = &part[colon + 1..];
        let rel_type = parse_rel_type(rel_name)?;
        let bands: Vec<usize> = band_str.split(',')
            .filter_map(|b| b.trim().parse().ok())
            .collect();
        if bands.is_empty() { return None; }
        Some(CatalogConfig { rel_type, bands })
    }).collect()
}

/// Parse raw phase specification like "10:1.047,20:2.094"
pub fn parse_raw_phases(s: &str) -> Vec<(usize, f32)> {
    s.split(',').filter_map(|part| {
        let colon = part.find(':')?;
        let band: usize = part[..colon].trim().parse().ok()?;
        let phase: f32 = part[colon + 1..].trim().parse().ok()?;
        Some((band, phase))
    }).collect()
}

// ─── State analysis ───

/// Per-band info from a hidden state.
pub struct BandInfo {
    pub index: usize,
    pub phase: f32,    // radians
    pub magnitude: f32,
}

/// Extract per-band phase and magnitude from interleaved r/s state.
pub fn extract_bands(state: &[f32]) -> Vec<BandInfo> {
    let n_bands = state.len() / 2;
    (0..n_bands).map(|k| {
        let r = state[k * 2];
        let s = state[k * 2 + 1];
        BandInfo {
            index: k,
            phase: s.atan2(r),
            magnitude: (r * r + s * s).sqrt(),
        }
    }).collect()
}

/// Cosine similarity between two states.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 { return 0.0; }
    dot / (na * nb)
}

// ─── Comparison report ───

pub struct CatalogMatch {
    pub band_a: usize,
    pub band_b: usize,
    pub angle_deg: f32,
    pub name: &'static str,
    pub drift_deg: f32,
}

/// Find all catalog-matching pairs in a state.
pub fn find_catalog_matches(state: &[f32]) -> Vec<CatalogMatch> {
    let bands = extract_bands(state);
    let n = bands.len();
    let mut matches = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = bands[j].phase - bands[i].phase;
            let diff = diff.rem_euclid(2.0 * PI);
            let deg = diff.to_degrees();
            if let Some((name, drift)) = match_catalog(deg) {
                matches.push(CatalogMatch {
                    band_a: i,
                    band_b: j,
                    angle_deg: deg,
                    name,
                    drift_deg: drift,
                });
            }
        }
    }
    matches
}

/// Compare input and output states, report what changed.
pub struct ComparisonReport {
    pub per_band: Vec<BandDelta>,
    pub cos_similarity: f32,
    pub injected_matches: Vec<CatalogMatch>,
    pub output_matches: Vec<CatalogMatch>,
}

pub struct BandDelta {
    pub index: usize,
    pub input_phase: f32,
    pub output_phase: f32,
    pub phase_drift_deg: f32,
    pub input_mag: f32,
    pub output_mag: f32,
    pub mag_ratio: f32,
}

pub fn compare_states(input: &[f32], output: &[f32]) -> ComparisonReport {
    let n_bands = input.len() / 2;
    let in_bands = extract_bands(input);
    let out_bands = extract_bands(output);
    let per_band: Vec<BandDelta> = (0..n_bands).map(|k| {
        let drift = (out_bands[k].phase - in_bands[k].phase).rem_euclid(2.0 * PI);
        let drift = if drift > PI { drift - 2.0 * PI } else { drift };
        BandDelta {
            index: k,
            input_phase: in_bands[k].phase,
            output_phase: out_bands[k].phase,
            phase_drift_deg: drift.to_degrees(),
            input_mag: in_bands[k].magnitude,
            output_mag: out_bands[k].magnitude,
            mag_ratio: out_bands[k].magnitude / in_bands[k].magnitude.max(1e-8),
        }
    }).collect();

    ComparisonReport {
        cos_similarity: cosine_similarity(input, output),
        injected_matches: find_catalog_matches(input),
        output_matches: find_catalog_matches(output),
        per_band,
    }
}

// ─── Energy deformation signatures ───

/// Per-band magnitude ratio (output/input) — the spectral fingerprint of how the ODE
/// processes this token. Phase tells WHERE. Energy deformation tells HOW MUCH.
pub fn deformation_vector(input: &[f32], output: &[f32]) -> Vec<f32> {
    let n_bands = input.len() / 2;
    (0..n_bands).map(|k| {
        let in_r = input[k * 2];
        let in_s = input[k * 2 + 1];
        let out_r = output[k * 2];
        let out_s = output[k * 2 + 1];
        let mag_in = (in_r * in_r + in_s * in_s).sqrt().max(1e-8);
        let mag_out = (out_r * out_r + out_s * out_s).sqrt();
        mag_out / mag_in
    }).collect()
}

/// Cosine similarity between two deformation vectors.
pub fn deformation_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 { return 0.0; }
    dot / (na * nb)
}

/// Per-token energy profile summary.
pub struct EnergyProfile {
    pub label: String,
    pub deformation: Vec<f32>,   // per-band mag_out/mag_in
    pub total_energy_ratio: f32, // sum(mag_out²) / sum(mag_in²)
    pub peak_band: usize,        // band with highest amplification
    pub peak_ratio: f32,         // that band's ratio
    pub damp_band: usize,        // band with most damping
    pub damp_ratio: f32,         // that band's ratio
}

pub fn compute_energy_profile(label: &str, input: &[f32], output: &[f32]) -> EnergyProfile {
    let deformation = deformation_vector(input, output);
    let n_bands = deformation.len();

    let energy_in: f32 = (0..n_bands).map(|k| {
        input[k*2]*input[k*2] + input[k*2+1]*input[k*2+1]
    }).sum();
    let energy_out: f32 = (0..n_bands).map(|k| {
        output[k*2]*output[k*2] + output[k*2+1]*output[k*2+1]
    }).sum();

    let mut peak_band = 0;
    let mut peak_ratio = 0.0f32;
    let mut damp_band = 0;
    let mut damp_ratio = f32::MAX;
    for (k, &r) in deformation.iter().enumerate() {
        if r > peak_ratio { peak_ratio = r; peak_band = k; }
        if r < damp_ratio { damp_ratio = r; damp_band = k; }
    }

    EnergyProfile {
        label: label.to_string(),
        deformation,
        total_energy_ratio: energy_out / energy_in.max(1e-8),
        peak_band,
        peak_ratio,
        damp_band,
        damp_ratio,
    }
}

// ─── Relate mode: per-harmonic coherence profile ───

pub struct HarmonicProfile {
    pub n: usize,
    pub mean_coherence: f32,
    pub bands_above_threshold: usize,
    pub per_band: Vec<f32>,
}

pub struct RelateReport {
    pub label_a: String,
    pub label_b: String,
    pub mean_angular_distance_deg: f32,
    pub catalog_match: Option<(&'static str, f32)>,
    pub harmonics: Vec<HarmonicProfile>,
    pub shifted_mrl: f32,
    pub shifted_offset: f32,
    pub shifted_harmonic: usize,
    pub deformation_sim: f32,  // cosine similarity of energy deformation vectors
}

/// Compute full harmonic coherence profile between two output states.
/// `deform_a`/`deform_b` are optional per-token deformation vectors.
pub fn relate_states(
    state_a: &[f32],
    state_b: &[f32],
    label_a: &str,
    label_b: &str,
) -> RelateReport {
    relate_states_with_deformation(state_a, state_b, label_a, label_b, None, None)
}

pub fn relate_states_with_deformation(
    state_a: &[f32],
    state_b: &[f32],
    label_a: &str,
    label_b: &str,
    deform_a: Option<&[f32]>,
    deform_b: Option<&[f32]>,
) -> RelateReport {
    let n_bands = state_a.len() / 2;
    let bands_a = extract_bands(state_a);
    let bands_b = extract_bands(state_b);

    // Mean angular distance
    let mut sum_cos = 0.0f32;
    let mut sum_sin = 0.0f32;
    for k in 0..n_bands {
        let diff = bands_a[k].phase - bands_b[k].phase;
        sum_cos += diff.cos();
        sum_sin += diff.sin();
    }
    let mean_angle = sum_sin.atan2(sum_cos);
    let mean_deg = mean_angle.to_degrees().rem_euclid(360.0);

    // Per-harmonic coherence
    let harm_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 27, 36, 60];
    let threshold = 0.8;
    let harmonics: Vec<HarmonicProfile> = harm_numbers.iter().map(|&n| {
        let per_band: Vec<f32> = (0..n_bands).map(|k| {
            let diff = bands_a[k].phase - bands_b[k].phase;
            (n as f32 * diff).cos()
        }).collect();
        let mean = per_band.iter().sum::<f32>() / n_bands as f32;
        let above = per_band.iter().filter(|&&c| c > threshold).count();
        HarmonicProfile { n, mean_coherence: mean, bands_above_threshold: above, per_band }
    }).collect();

    // Shifted coherence (MRL) — find best harmonic
    let search_harmonics = [1, 2, 3, 4, 6];
    let mut best_mrl = 0.0f32;
    let mut best_offset = 0.0f32;
    let mut best_n = 1;
    for &n in &search_harmonics {
        let mut sc = 0.0f32;
        let mut ss = 0.0f32;
        for k in 0..n_bands {
            let diff = bands_a[k].phase - bands_b[k].phase;
            let angle = n as f32 * diff;
            sc += angle.cos();
            ss += angle.sin();
        }
        let mrl = (sc * sc + ss * ss).sqrt() / n_bands as f32;
        if mrl > best_mrl {
            best_mrl = mrl;
            best_offset = ss.atan2(sc) / n as f32;
            best_n = n;
        }
    }

    let catalog_match = match_catalog(mean_deg);

    let deform_sim = match (deform_a, deform_b) {
        (Some(a), Some(b)) => deformation_similarity(a, b),
        _ => 0.0,
    };

    RelateReport {
        label_a: label_a.to_string(),
        label_b: label_b.to_string(),
        mean_angular_distance_deg: mean_deg,
        catalog_match,
        harmonics,
        shifted_mrl: best_mrl,
        shifted_offset: best_offset,
        shifted_harmonic: best_n,
        deformation_sim: deform_sim,
    }
}

// ─── Printing ───

/// Print the encoded state (input bands).
pub fn print_encoded_state(label: &str, state: &[f32], highlight_bands: &[usize]) {
    println!("=== {} ===", label);
    let bands = extract_bands(state);
    for bi in highlight_bands {
        if *bi < bands.len() {
            let b = &bands[*bi];
            println!("  Band {:3}: θ={:7.3} rad ({:6.1}°)  mag={:.3}",
                b.index, b.phase, b.phase.to_degrees(), b.magnitude);
        }
    }
    if highlight_bands.is_empty() {
        // Show bands with non-trivial phase (not near 0)
        let active: Vec<_> = bands.iter()
            .filter(|b| b.phase.abs() > 0.01)
            .collect();
        if active.is_empty() {
            println!("  All bands at phase 0.0, magnitude 1.0");
        } else {
            for b in active.iter().take(10) {
                println!("  Band {:3}: θ={:7.3} rad ({:6.1}°)  mag={:.3}",
                    b.index, b.phase, b.phase.to_degrees(), b.magnitude);
            }
            if active.len() > 10 {
                println!("  ... and {} more active bands", active.len() - 10);
            }
        }
    }
}

/// Print per-layer cosine similarity.
pub fn print_layer_cosines(input: &[f32], per_layer_outputs: &[Vec<f32>]) {
    print!("  cos(input, output) per layer:");
    for (i, output) in per_layer_outputs.iter().enumerate() {
        print!("  L{}: {:.2}", i, cosine_similarity(input, output));
    }
    println!();
}

/// Print the comparison report.
pub fn print_comparison(input: &[f32], output: &[f32], injected_configs: &[CatalogConfig]) {
    let report = compare_states(input, output);
    println!("\n=== Comparison (input → output) ===");
    println!("  Cosine similarity: {:.4}", report.cos_similarity);

    // Show injected relationships
    if !injected_configs.is_empty() {
        println!("\n  Injected:");
        for cfg in injected_configs {
            println!("    {} at bands {:?}", cfg.rel_type.name(), cfg.bands);
        }
    }

    // Check if injected relationships survived
    println!("\n  Catalog matches at output ({} total):", report.output_matches.len());
    let show = report.output_matches.len().min(15);
    for m in report.output_matches.iter().take(show) {
        println!("    {} at bands ({}, {}): {:.1}° (drift {:.1}°)",
            m.name, m.band_a, m.band_b, m.angle_deg, m.drift_deg);
    }
    if report.output_matches.len() > show {
        println!("    ... and {} more", report.output_matches.len() - show);
    }

    // Top phase drifts
    let mut drifts: Vec<&BandDelta> = report.per_band.iter()
        .filter(|d| d.phase_drift_deg.abs() > 1.0)
        .collect();
    drifts.sort_by(|a, b| b.phase_drift_deg.abs().partial_cmp(&a.phase_drift_deg.abs()).unwrap());
    if !drifts.is_empty() {
        println!("\n  Top phase drifts:");
        for d in drifts.iter().take(10) {
            println!("    Band {:3}: {:+.1}° (mag {:.3} → {:.3}, ratio {:.2})",
                d.index, d.phase_drift_deg, d.input_mag, d.output_mag, d.mag_ratio);
        }
    }
}

/// Print a relate report.
pub fn print_relate_report(report: &RelateReport) {
    println!("\n=== Relationship: \"{}\" ↔ \"{}\" ===", report.label_a, report.label_b);
    let cat_str = match report.catalog_match {
        Some((name, drift)) => format!("{} (drift {:.1}°)", name, drift),
        None => "none".to_string(),
    };
    println!("  Mean angular distance: {:.1}° (nearest catalog: {})", report.mean_angular_distance_deg, cat_str);

    println!("\n  Harmonic profile:");
    for h in &report.harmonics {
        let label = match h.n {
            1 => "identity",
            2 => "opposition",
            3 => "trine",
            4 => "square",
            5 => "quintile",
            6 => "sextile",
            8 => "semi-square",
            12 => "semi-sextile",
            _ => "other",
        };
        let strength = if h.mean_coherence.abs() > 0.7 { "STRONG" }
            else if h.mean_coherence.abs() > 0.3 { "moderate" }
            else { "weak" };
        println!("    n={:2}: mean_coh={:6.3}  bands>{:.1}: {:3}  ({} — {})",
            h.n, h.mean_coherence, 0.8, h.bands_above_threshold, label, strength);
    }

    println!("\n  Shifted coherence: MRL={:.3} at n={}, offset={:.3} rad",
        report.shifted_mrl, report.shifted_harmonic, report.shifted_offset);
}

/// Print pairwise relationship matrix for multiple items.
pub fn print_relate_matrix(labels: &[String], reports: &[RelateReport]) {
    let n = labels.len();
    // Header
    print!("\n{:>12}", "");
    for l in labels { print!("{:>12}", &l[..l.len().min(10)]); }
    println!();

    let mut idx = 0;
    for i in 0..n {
        print!("{:>12}", &labels[i][..labels[i].len().min(10)]);
        for j in 0..n {
            if i == j {
                print!("{:>12}", "—");
            } else if i < j {
                let r = &reports[idx];
                let name = match r.catalog_match {
                    Some((n, _)) => n,
                    None => "none",
                };
                print!("{:>12}", &name[..name.len().min(10)]);
                idx += 1;
            } else {
                // Find the report for (j, i)
                let target_idx = j * (2 * n - j - 1) / 2 + (i - j - 1);
                if target_idx < reports.len() {
                    let r = &reports[target_idx];
                    let name = match r.catalog_match {
                        Some((n, _)) => n,
                        None => "none",
                    };
                    print!("{:>12}", &name[..name.len().min(10)]);
                } else {
                    print!("{:>12}", "?");
                }
            }
        }
        println!();
    }
}

/// Forward an injected hidden state from a specific layer through a WavePacketModel.
/// Returns (post_ln_f_output, per_layer_outputs).
/// CPU-only — fine for single-position encode tool.
pub fn forward_from_layer(
    model: &crate::WavePacketModel,
    encoded_state: &[f32],
    inject_layer: usize,
    n_bands: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    use super::block::{layer_norm, wave_block_forward};
    let mut hidden = vec![encoded_state.to_vec()]; // single position
    let mut per_layer: Vec<Vec<f32>> = Vec::new();

    for (i, block) in model.blocks.iter().enumerate() {
        if i < inject_layer {
            per_layer.push(hidden[0].clone());
            continue;
        }
        let (output, _attn_w) = wave_block_forward(block, &hidden, n_bands);
        let scale = if model.use_layer_scale { model.layer_scale[i] } else { 1.0 };
        if (scale - 1.0).abs() > 1e-6 {
            // Apply layer scale if not 1.0
            let n_embd = n_bands * 2;
            let rescaled: Vec<Vec<f32>> = output.iter().map(|h| {
                let mut v = vec![0.0f32; n_embd];
                for j in 0..n_embd {
                    v[j] = hidden[0][j] + scale * (h[j] - hidden[0][j]);
                }
                v
            }).collect();
            hidden = rescaled;
        } else {
            hidden = output;
        }
        per_layer.push(hidden[0].clone());
    }

    // Final layer norm
    let normed = layer_norm(&hidden[0], &model.ln_f.weight, &model.ln_f.bias);
    (normed, per_layer)
}

/// Run the full encode pipeline. Returns (post_ln_f, per_layer_outputs).
pub fn run_encode(
    model: &crate::WavePacketModel,
    encoded_state: &[f32],
    inject_layer: usize,
    n_bands: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    forward_from_layer(model, encoded_state, inject_layer, n_bands)
}

/// Run relate mode: encode multiple items, forward each, compute pairwise profiles.
pub fn run_relate(
    model: &crate::WavePacketModel,
    items: &[(String, Vec<f32>)],
    n_bands: usize,
) -> Vec<RelateReport> {
    let outputs: Vec<Vec<f32>> = items.iter()
        .map(|(_, state)| {
            let (final_out, _) = forward_from_layer(model, state, 0, n_bands);
            final_out
        })
        .collect();

    let mut reports = Vec::new();
    for i in 0..items.len() {
        for j in (i + 1)..items.len() {
            reports.push(relate_states(
                &outputs[i], &outputs[j],
                &items[i].0, &items[j].0,
            ));
        }
    }
    reports
}

/// Run relate-vocab: encode all tokens, forward, compute full pairwise matrix.
/// `char_map` provides display labels for tokens (from data file vocab).
/// Returns (labels, reports, catalog_dist, energy_profiles).
pub fn run_relate_vocab(
    model: &crate::WavePacketModel,
    n_bands: usize,
    char_map: Option<&[char]>,
) -> (Vec<String>, Vec<RelateReport>, std::collections::HashMap<String, usize>, Vec<EnergyProfile>) {
    let vocab = model.vocab_size;
    let n_embd = n_bands * 2;

    println!("  Encoding {} tokens through ODE...", vocab);
    let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(vocab);
    let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(vocab);
    for tok in 0..vocab {
        let mut h = vec![0.0f32; n_embd];
        for i in 0..n_embd {
            h[i] = model.wte[tok][i] + model.wpe[0][i];
        }
        let (final_out, _) = forward_from_layer(model, &h, 0, n_bands);
        inputs.push(h);
        outputs.push(final_out);
    }

    let labels: Vec<String> = match char_map {
        Some(cm) => (0..vocab).map(|t| {
            if t < cm.len() {
                let c = cm[t];
                if c.is_ascii_graphic() || c == ' ' { format!("{}", c) } else { format!("t{}", t) }
            } else { format!("t{}", t) }
        }).collect(),
        None => (0..vocab).map(|t| format!("t{}", t)).collect(),
    };

    // Compute per-token energy deformation profiles
    println!("  Computing energy deformation profiles...");
    let deformations: Vec<Vec<f32>> = (0..vocab)
        .map(|tok| deformation_vector(&inputs[tok], &outputs[tok]))
        .collect();
    let profiles: Vec<EnergyProfile> = (0..vocab)
        .map(|tok| compute_energy_profile(&labels[tok], &inputs[tok], &outputs[tok]))
        .collect();

    let total = vocab * (vocab - 1) / 2;
    println!("  Computing {} pairwise relationships (phase + energy)...", total);
    let mut reports = Vec::with_capacity(total);
    let mut dist: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for i in 0..vocab {
        for j in (i + 1)..vocab {
            let r = relate_states_with_deformation(
                &outputs[i], &outputs[j],
                &labels[i], &labels[j],
                Some(&deformations[i]), Some(&deformations[j]),
            );
            if let Some((name, _)) = r.catalog_match {
                *dist.entry(name.to_string()).or_insert(0) += 1;
            }
            reports.push(r);
        }
    }

    (labels, reports, dist, profiles)
}

/// Write relate-vocab results to JSON (with energy profiles).
pub fn write_vocab_relations_json(
    path: &str,
    labels: &[String],
    reports: &[RelateReport],
    dist: &std::collections::HashMap<String, usize>,
    profiles: Option<&[EnergyProfile]>,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    write!(f, "{{\n  \"vocab_size\": {},\n  \"total_pairs\": {},\n", labels.len(), reports.len())?;
    write!(f, "  \"catalog_distribution\": {{\n")?;
    let mut entries: Vec<_> = dist.iter().collect();
    entries.sort_by_key(|e| std::cmp::Reverse(*e.1));
    for (i, (name, count)) in entries.iter().enumerate() {
        write!(f, "    \"{}\": {}{}\n", name, count, if i + 1 < entries.len() { "," } else { "" })?;
    }
    write!(f, "  }},\n  \"pairs\": [\n")?;
    // Top pairs by shifted MRL
    let mut sorted: Vec<(usize, &RelateReport)> = reports.iter().enumerate().collect();
    sorted.sort_by(|a, b| b.1.shifted_mrl.partial_cmp(&a.1.shifted_mrl).unwrap_or(std::cmp::Ordering::Equal));
    let limit = sorted.len(); // write all pairs
    for (i, (_, r)) in sorted.iter().take(limit).enumerate() {
        let cat = match r.catalog_match {
            Some((n, _)) => format!("\"{}\"", n),
            None => "null".to_string(),
        };
        // Escape label strings for JSON safety
        let esc = |s: &str| -> String {
            s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t")
        };
        write!(f, "    {{\"a\":\"{}\",\"b\":\"{}\",\"angle\":{:.1},\"catalog\":{},\"mrl\":{:.3},\"harm\":{},\"deform_sim\":{:.3}}}{}\n",
            esc(&r.label_a), esc(&r.label_b), r.mean_angular_distance_deg, cat,
            r.shifted_mrl, r.shifted_harmonic, r.deformation_sim,
            if i + 1 < limit { "," } else { "" })?;
    }
    write!(f, "  ]")?;

    // Write energy profiles if provided
    if let Some(profs) = profiles {
        write!(f, ",\n  \"energy_profiles\": [\n")?;
        for (i, p) in profs.iter().enumerate() {
            let esc = |s: &str| -> String {
                s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
            };
            write!(f, "    {{\"token\":\"{}\",\"total_energy_ratio\":{:.3},\"peak_band\":{},\"peak_ratio\":{:.3},\"damp_band\":{},\"damp_ratio\":{:.3}}}{}\n",
                esc(&p.label), p.total_energy_ratio, p.peak_band, p.peak_ratio, p.damp_band, p.damp_ratio,
                if i + 1 < profs.len() { "," } else { "" })?;
        }
        write!(f, "  ]")?;
    }

    write!(f, "\n}}\n")?;
    Ok(())
}
