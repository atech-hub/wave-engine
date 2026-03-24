//! Wave structure analysis — harmonic coherence diagnostics for trained models.
//!
//! Uses cos(n × Δθ) from the research framework (wave.rs) to measure
//! what a model has learned: semantic discrimination, grammar coherence,
//! phase clustering, band census, depth curves, harmonic spectra.
//!
//! This is NOT cosine similarity. This is per-harmonic coherence sweep —
//! the same math that detects relationships cosine similarity collapses.

/// Harmonic coherence: cos(n × (θ_a - θ_b)).
/// n=1: identity. n=2: opposition. n=3: trine/family. etc.
pub fn harmonic_coherence(theta_a: f32, theta_b: f32, n: f32) -> f32 {
    (n * (theta_a - theta_b)).cos()
}

/// Extract phase angles from interleaved (r, s) hidden state.
/// Input: [n_embd] with [r0, s0, r1, s1, ...]. Output: [n_bands] phases.
pub fn extract_phases(hidden: &[f32], n_bands: usize) -> Vec<f32> {
    (0..n_bands).map(|k| {
        let r = hidden[k * 2];
        let s = hidden[k * 2 + 1];
        s.atan2(r)
    }).collect()
}

/// Extract phases for all positions. Input: [n_pos][n_embd]. Output: [n_pos][n_bands].
pub fn extract_all_phases(hidden: &[Vec<f32>], n_bands: usize) -> Vec<Vec<f32>> {
    hidden.iter().map(|h| extract_phases(h, n_bands)).collect()
}

/// Circular variance: 1.0 - |mean(exp(iθ))|.
/// 0.0 = all same angle (clustered). 1.0 = uniform (dispersed).
pub fn circular_variance(phases: &[f32]) -> f32 {
    let n = phases.len() as f32;
    if n == 0.0 { return 1.0; }
    let sum_cos: f32 = phases.iter().map(|p| p.cos()).sum();
    let sum_sin: f32 = phases.iter().map(|p| p.sin()).sum();
    let mean_r = ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt();
    1.0 - mean_r
}

/// Mean resultant length: |mean(exp(iθ))|. Inverse of circular variance.
/// 1.0 = all same angle. 0.0 = uniform.
pub fn mean_resultant_length(phases: &[f32]) -> f32 {
    1.0 - circular_variance(phases)
}

/// Mean harmonic coherence between two phase vectors across all bands.
/// Sweeps n=1..max_n, returns the best (highest absolute).
pub fn best_harmonic_coherence(phases_a: &[f32], phases_b: &[f32], max_n: usize) -> (f32, usize) {
    let n_bands = phases_a.len();
    let mut best_coh = 0.0f32;
    let mut best_n = 1;
    for n in 1..=max_n {
        let coh: f32 = phases_a.iter().zip(phases_b)
            .map(|(&pa, &pb)| harmonic_coherence(pa, pb, n as f32).abs())
            .sum::<f32>() / n_bands as f32;
        if coh > best_coh {
            best_coh = coh;
            best_n = n;
        }
    }
    (best_coh, best_n)
}

/// Full harmonic spectrum: coherence at each harmonic n=1..max_n.
pub fn harmonic_spectrum(phases_a: &[f32], phases_b: &[f32], max_n: usize) -> Vec<f32> {
    let n_bands = phases_a.len();
    (1..=max_n).map(|n| {
        phases_a.iter().zip(phases_b)
            .map(|(&pa, &pb)| harmonic_coherence(pa, pb, n as f32))
            .sum::<f32>() / n_bands as f32
    }).collect()
}

/// Average phases across a span of token positions (for multi-token words).
/// Uses circular mean: atan2(mean(sin), mean(cos)) per band.
pub fn average_phases_over_span(phases: &[Vec<f32>], positions: &[usize]) -> Vec<f32> {
    if positions.len() == 1 {
        return phases[positions[0]].clone();
    }
    let n_bands = phases[0].len();
    (0..n_bands).map(|band| {
        let sum_cos: f32 = positions.iter().map(|&p| phases[p][band].cos()).sum();
        let sum_sin: f32 = positions.iter().map(|&p| phases[p][band].sin()).sum();
        sum_sin.atan2(sum_cos) // circular mean
    }).collect()
}

// ─── Diagnostic 1: Semantic Discrimination ──────────────────────

pub struct DiscriminationResult {
    pub related_mean: f32,
    pub random_mean: f32,
    pub ratio: f32,
}

pub fn semantic_discrimination(
    phases: &[Vec<f32>],
    related_pairs: &[(usize, usize)],
    random_pairs: &[(usize, usize)],
    max_harmonic: usize,
) -> DiscriminationResult {
    let related_mean = if related_pairs.is_empty() { 0.0 } else {
        related_pairs.iter().map(|&(a, b)| {
            best_harmonic_coherence(&phases[a], &phases[b], max_harmonic).0
        }).sum::<f32>() / related_pairs.len() as f32
    };

    let random_mean = if random_pairs.is_empty() { 0.0 } else {
        random_pairs.iter().map(|&(a, b)| {
            best_harmonic_coherence(&phases[a], &phases[b], max_harmonic).0
        }).sum::<f32>() / random_pairs.len() as f32
    };

    let ratio = related_mean / random_mean.max(0.001);
    DiscriminationResult { related_mean, random_mean, ratio }
}

/// Span-based discrimination: words may span multiple BPE tokens.
/// Each word is represented by its span of positions; phases are averaged via circular mean.
pub fn semantic_discrimination_spans(
    phases: &[Vec<f32>],
    related_pairs: &[(Vec<usize>, Vec<usize>)],  // (word_a_positions, word_b_positions)
    random_pairs: &[(Vec<usize>, Vec<usize>)],
    max_harmonic: usize,
) -> DiscriminationResult {
    let related_mean = if related_pairs.is_empty() { 0.0 } else {
        related_pairs.iter().map(|(span_a, span_b)| {
            let avg_a = average_phases_over_span(phases, span_a);
            let avg_b = average_phases_over_span(phases, span_b);
            best_harmonic_coherence(&avg_a, &avg_b, max_harmonic).0
        }).sum::<f32>() / related_pairs.len() as f32
    };

    let random_mean = if random_pairs.is_empty() { 0.0 } else {
        random_pairs.iter().map(|(span_a, span_b)| {
            let avg_a = average_phases_over_span(phases, span_a);
            let avg_b = average_phases_over_span(phases, span_b);
            best_harmonic_coherence(&avg_a, &avg_b, max_harmonic).0
        }).sum::<f32>() / random_pairs.len() as f32
    };

    let ratio = related_mean / random_mean.max(0.001);
    DiscriminationResult { related_mean, random_mean, ratio }
}

// ─── Diagnostic 2: Grammar Coherence ────────────────────────────

pub fn grammar_coherence(
    phases_a: &[Vec<f32>],  // sentence A phases [n_pos][n_bands]
    phases_b: &[Vec<f32>],  // sentence B phases
    pairs_a: &[(usize, usize)],  // (subject_pos, verb_pos) in A
    pairs_b: &[(usize, usize)],  // same roles in B
    max_harmonic: usize,
) -> f32 {
    if pairs_a.is_empty() { return 0.0; }
    let mut consistency = 0.0f32;
    for (&(a1, a2), &(b1, b2)) in pairs_a.iter().zip(pairs_b) {
        let coh_a = best_harmonic_coherence(&phases_a[a1], &phases_a[a2], max_harmonic).0;
        let coh_b = best_harmonic_coherence(&phases_b[b1], &phases_b[b2], max_harmonic).0;
        consistency += 1.0 - (coh_a - coh_b).abs();
    }
    consistency / pairs_a.len() as f32
}

// ─── Diagnostic 3: Depth Curve ──────────────────────────────────

pub fn depth_curve(
    per_layer_phases: &[Vec<Vec<f32>>],
    related_pairs: &[(usize, usize)],
    random_pairs: &[(usize, usize)],
    max_harmonic: usize,
) -> Vec<f32> {
    per_layer_phases.iter().map(|layer_phases| {
        let result = semantic_discrimination(layer_phases, related_pairs, random_pairs, max_harmonic);
        result.ratio
    }).collect()
}

// ─── Diagnostic 4: Band Census ──────────────────────────────────

pub struct BandCensus {
    pub universal: usize,
    pub word_specific: usize,
    pub mean_circular_variance: f32,
    /// Circular variance per band — for histogram analysis
    pub per_band_cv: Vec<f32>,
    /// Is the distribution bimodal (clear split) or continuous?
    pub bimodal: bool,
}

pub fn band_census(phases: &[Vec<f32>], n_bands: usize) -> BandCensus {
    let n_pos = phases.len();
    let mut cvs = Vec::with_capacity(n_bands);

    for band in 0..n_bands {
        let band_phases: Vec<f32> = (0..n_pos).map(|p| phases[p][band]).collect();
        cvs.push(circular_variance(&band_phases));
    }

    let total_cv: f32 = cvs.iter().sum();

    // Threshold at median
    let mut sorted = cvs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[n_bands / 2];

    let universal = cvs.iter().filter(|&&cv| cv < median).count();
    let word_specific = n_bands - universal;

    // Bimodality test: check if there's a gap in the middle of the distribution
    // If the density dips between two peaks, the split is natural, not arbitrary
    let n_bins = 10;
    let min_cv = sorted[0];
    let max_cv = sorted[n_bands - 1];
    let bin_width = (max_cv - min_cv) / n_bins as f32;
    let mut bins = vec![0usize; n_bins];
    for &cv in &cvs {
        let bin = ((cv - min_cv) / bin_width.max(0.001)) as usize;
        bins[bin.min(n_bins - 1)] += 1;
    }
    // Bimodal if the minimum bin count in the middle third is < 30% of max bin
    let middle_min = bins[n_bins / 3..2 * n_bins / 3].iter().min().copied().unwrap_or(0);
    let overall_max = bins.iter().max().copied().unwrap_or(1);
    let bimodal = middle_min < overall_max * 30 / 100;

    BandCensus {
        universal,
        word_specific,
        mean_circular_variance: total_cv / n_bands as f32,
        per_band_cv: cvs,
        bimodal,
    }
}

/// Print a text histogram of circular variance distribution
pub fn print_cv_histogram(cvs: &[f32], n_bins: usize) {
    let mut sorted = cvs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_cv = sorted[0];
    let max_cv = sorted[sorted.len() - 1];
    let bin_width = (max_cv - min_cv) / n_bins as f32;

    let mut bins = vec![0usize; n_bins];
    for &cv in cvs {
        let bin = ((cv - min_cv) / bin_width.max(0.001)) as usize;
        bins[bin.min(n_bins - 1)] += 1;
    }

    let max_count = bins.iter().max().copied().unwrap_or(1);
    let bar_width = 40;

    println!("   CV Distribution ({} bands):", cvs.len());
    for (i, &count) in bins.iter().enumerate() {
        let lo = min_cv + i as f32 * bin_width;
        let hi = lo + bin_width;
        let bar_len = count * bar_width / max_count.max(1);
        let bar: String = (0..bar_len).map(|_| '#').collect();
        let label = if lo < 0.5 { "U" } else { "W" }; // Universal vs Word-specific
        println!("   {:.2}-{:.2} [{label}] |{bar:<width$}| {count}", lo, hi, width = bar_width);
    }
}

// ─── Diagnostic 5: Phase Clustering ─────────────────────────────

pub fn phase_clustering(phases: &[Vec<f32>], n_bands: usize) -> f32 {
    let n_pos = phases.len();
    let mut total_r = 0.0f32;

    for band in 0..n_bands {
        let band_phases: Vec<f32> = (0..n_pos).map(|p| phases[p][band]).collect();
        total_r += mean_resultant_length(&band_phases);
    }
    total_r / n_bands as f32
}

// ─── Full Report ────────────────────────────────────────────────

pub fn print_report(
    checkpoint: &str,
    n_layers: usize,
    n_bands: usize,
    n_tokens: usize,
    per_layer_phases: &[Vec<Vec<f32>>],
    related_pairs: &[(usize, usize)],
    random_pairs: &[(usize, usize)],
    token_labels: &[String],
) {
    let max_h = 12;

    println!("\n=== Wave Structure Report ===");
    println!("Checkpoint: {checkpoint}");
    println!("Layers: {n_layers}, Bands: {n_bands}, Tokens: {n_tokens}");

    // Use deepest layer for main diagnostics
    let deep = per_layer_phases.last().unwrap();

    // 1. Semantic Discrimination
    let disc = semantic_discrimination(deep, related_pairs, random_pairs, max_h);
    let verdict1 = if disc.ratio > 2.0 { "STRONG SEMANTIC STRUCTURE" }
        else if disc.ratio > 1.5 { "EMERGING STRUCTURE" }
        else { "NOT YET" };
    println!("\n1. Semantic Discrimination");
    println!("   Related: {:.3}    Random: {:.3}    Ratio: {:.1}x    {verdict1}",
        disc.related_mean, disc.random_mean, disc.ratio);

    // 2. Depth Curve
    let curve = depth_curve(per_layer_phases, related_pairs, random_pairs, max_h);
    let peak_layer = curve.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i).unwrap_or(0);
    let peak_ratio = curve[peak_layer];
    let verdict3 = if peak_layer > n_layers * 3 / 4 { "USING MOST LAYERS" }
        else if peak_layer > n_layers / 2 { "USING HALF DEPTH" }
        else { "SHALLOW PEAK" };
    println!("\n2. Depth Curve");
    println!("   Peak: layer {} (ratio {:.1}x)    {verdict3}", peak_layer, peak_ratio);
    print!("   Curve: ");
    for (i, &r) in curve.iter().enumerate() {
        if i > 0 && i % 12 == 0 { print!("\n          "); }
        print!("{:.1} ", r);
    }
    println!();

    // 3. Band Census + Histogram
    let census = band_census(deep, n_bands);
    let pct_universal = census.universal as f32 / n_bands as f32 * 100.0;
    let bimodal_str = if census.bimodal { "BIMODAL (natural split)" } else { "CONTINUOUS (threshold arbitrary)" };
    println!("\n3. Band Census");
    println!("   Universal: {} ({:.0}%)    Word-specific: {} ({:.0}%)",
        census.universal, pct_universal,
        census.word_specific, 100.0 - pct_universal);
    println!("   Distribution: {bimodal_str}");
    println!("   Reference: 67/33 at 64 bands (research repo)");
    print_cv_histogram(&census.per_band_cv, 10);

    // 4. Phase Clustering
    let clustering = phase_clustering(deep, n_bands);
    let verdict5 = if clustering > 0.3 { "STRUCTURED" }
        else if clustering > 0.15 { "PARTIALLY STRUCTURED" }
        else { "RANDOM" };
    println!("\n4. Phase Clustering");
    println!("   Mean resultant: {:.3}    {verdict5}", clustering);

    // 5. Harmonic Spectra for labelled pairs
    if related_pairs.len() >= 1 {
        println!("\n5. Harmonic Spectra");
        for &(a, b) in related_pairs.iter().take(3) {
            let label_a = if a < token_labels.len() { &token_labels[a] } else { "?" };
            let label_b = if b < token_labels.len() { &token_labels[b] } else { "?" };
            let spectrum = harmonic_spectrum(&deep[a], &deep[b], max_h);
            let (best_coh, best_n) = best_harmonic_coherence(&deep[a], &deep[b], max_h);
            print!("   ({label_a}, {label_b}): ");
            for (i, &s) in spectrum.iter().enumerate() {
                print!("n{}={:.2} ", i + 1, s);
            }
            println!("  peak: n={best_n} ({best_coh:.2})");
        }
    }

    // Summary
    let mut positive = 0;
    if disc.ratio > 1.5 { positive += 1; }
    if peak_layer > n_layers / 2 { positive += 1; }
    if pct_universal > 50.0 && pct_universal < 90.0 { positive += 1; }
    if clustering > 0.15 { positive += 1; }
    println!("\nOverall: {positive}/4 diagnostics positive.");
}
