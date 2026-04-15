//! Framework Monitor — live harmonic coherence diagnostics at health intervals.
//!
//! Measures semantic discrimination, band census, phase clustering, depth curve,
//! and harmonic spectra for canonical pairs during training. Reports per-layer
//! to JSONL alongside ode_decomposition and other monitors.
//!
//! Reuses wave_analysis.rs primitives. Cost: ~5-10ms per health interval.

use crate::common::wave_analysis as wa;

/// Per-layer framework statistics.
pub struct FrameworkStats {
    pub layer: usize,
    pub discrimination_ratio: f32,
    pub related_mean: f32,
    pub random_mean: f32,
    pub band_census_universal: usize,
    pub band_census_word_specific: usize,
    pub band_census_mean_cv: f32,
    pub phase_clustering: f32,
    pub pair_peaks: Vec<PairPeak>,
}

pub struct PairPeak {
    pub label: String,
    pub peak_n: usize,
    pub peak_strength: f32,
}

/// Full framework report across all layers.
pub struct FrameworkReport {
    pub per_layer: Vec<FrameworkStats>,
    pub depth_curve: Vec<f32>,
    pub dominant_depth_peak: usize,
}

/// Canonical test text for framework monitoring.
/// Same as analyze.rs — covers semantics, grammar, and registers.
pub const FRAMEWORK_TEST_TEXT: &str = concat!(
    "The cat sat on the mat. ",
    "The dog sat on the rug. ",
    "A noun is the name of something. ",
    "A verb is a word for action. ",
    "The boy kicked the ball. ",
    "The ball was kicked by the boy. ",
    "To be or not to be that is the question. ",
    "The contract shall be binding upon execution. ",
    "The rate of change increases with temperature. ",
    "How are you doing today my friend. ",
    "Love is patient and kind. ",
    "War brings destruction and death. ",
);

/// Canonical semantic pairs for framework monitoring.
const SEMANTIC_PAIRS: &[(&str, &str)] = &[
    ("cat", "dog"),
    ("boy", "ball"),
    ("noun", "verb"),
    ("love", "war"),
    ("mat", "rug"),
];

const RANDOM_PAIRS: &[(&str, &str)] = &[
    ("cat", "contract"),
    ("verb", "temperature"),
    ("boy", "question"),
];

/// Tokenize the framework test text using char-level tokenizer.
/// Returns (token_ids, vocab_size). Clamps to max_vocab if provided.
pub fn tokenize_test_text(max_vocab: usize) -> (Vec<usize>, usize) {
    let chars: Vec<char> = FRAMEWORK_TEST_TEXT.chars().collect();
    let mut vc: Vec<char> = chars.iter().cloned()
        .collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    vc.sort();
    let c2i: std::collections::HashMap<char, usize> = vc.iter()
        .enumerate().map(|(i, &c)| (c, i)).collect();
    let ids: Vec<usize> = chars.iter()
        .map(|c| c2i[c].min(max_vocab.saturating_sub(1)))
        .collect();
    (ids, vc.len().min(max_vocab))
}

/// Build canonical pair spans from the test text tokens.
pub fn build_canonical_pairs(token_strings: &[String]) -> (
    Vec<(Vec<usize>, Vec<usize>)>,   // related span pairs
    Vec<(Vec<usize>, Vec<usize>)>,   // random span pairs
    Vec<String>,                      // related labels
) {
    let find_word_span = |word: &str| -> Option<Vec<usize>> {
        let word_lower = word.to_lowercase();
        for start in 0..token_strings.len() {
            let mut concat = String::new();
            for end in start..token_strings.len().min(start + 5) {
                concat.push_str(&token_strings[end]);
                let clean = concat.to_lowercase();
                let clean = clean.trim();
                if clean == word_lower || clean == format!(" {word_lower}") {
                    return Some((start..=end).collect());
                }
            }
        }
        None
    };

    let mut related = Vec::new();
    let mut labels = Vec::new();
    for (w1, w2) in SEMANTIC_PAIRS {
        if let (Some(a), Some(b)) = (find_word_span(w1), find_word_span(w2)) {
            labels.push(format!("{w1}/{w2}"));
            related.push((a, b));
        }
    }

    let mut random = Vec::new();
    for (w1, w2) in RANDOM_PAIRS {
        if let (Some(a), Some(b)) = (find_word_span(w1), find_word_span(w2)) {
            random.push((a, b));
        }
    }

    (related, random, labels)
}

/// Run the framework scan on a forward cache's hidden states.
/// `all_layer_hidden` is per-layer hidden states [n_layers][n_positions][n_embd].
/// `post_ln_f` is the final layer-norm output [n_positions][n_embd].
pub fn run_framework_scan(
    all_layer_hidden: &[Vec<Vec<f32>>],
    post_ln_f: &[Vec<f32>],
    n_bands: usize,
    related_spans: &[(Vec<usize>, Vec<usize>)],
    random_spans: &[(Vec<usize>, Vec<usize>)],
    labels: &[String],
) -> FrameworkReport {
    let n_layers = all_layer_hidden.len();
    let mut per_layer = Vec::new();
    let mut depth_curve = Vec::new();

    // Extract phases per layer + final
    let mut all_phases: Vec<Vec<Vec<f32>>> = Vec::new();
    for layer_hidden in all_layer_hidden {
        all_phases.push(wa::extract_all_phases(layer_hidden, n_bands));
    }
    let final_phases = wa::extract_all_phases(post_ln_f, n_bands);
    all_phases.push(final_phases);

    for (li, phases) in all_phases.iter().enumerate() {
        // Semantic discrimination (span-averaged)
        let disc = wa::semantic_discrimination_spans(
            phases, related_spans, random_spans, 12,
        );
        let disc_ratio = disc.ratio;
        let related_mean = disc.related_mean;
        let random_mean = disc.random_mean;

        // Band census
        let census = wa::band_census(phases, n_bands);
        let universal_count = census.universal;
        let word_specific_count = census.word_specific;
        let mean_cv = census.mean_circular_variance;

        // Phase clustering
        let clustering = wa::phase_clustering(phases, n_bands);

        // Top pair peaks
        let mut pair_peaks = Vec::new();
        for (idx, (span_a, span_b)) in related_spans.iter().enumerate() {
            let avg_a = wa::average_phases_over_span(phases, span_a);
            let avg_b = wa::average_phases_over_span(phases, span_b);
            let (best_coh, best_n) = wa::best_harmonic_coherence(&avg_a, &avg_b, 12);
            let label = labels.get(idx).cloned().unwrap_or_default();
            pair_peaks.push(PairPeak { label, peak_n: best_n, peak_strength: best_coh });
        }

        depth_curve.push(disc_ratio);

        per_layer.push(FrameworkStats {
            layer: li,
            discrimination_ratio: disc_ratio,
            related_mean,
            random_mean,
            band_census_universal: universal_count,
            band_census_word_specific: word_specific_count,
            band_census_mean_cv: mean_cv,
            phase_clustering: clustering,
            pair_peaks,
        });
    }

    let dominant_depth_peak = depth_curve.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);

    FrameworkReport {
        per_layer,
        depth_curve,
        dominant_depth_peak,
    }
}

/// Serialize framework report to JSONL fragment.
pub fn to_json(report: &FrameworkReport) -> String {
    let layers: Vec<String> = report.per_layer.iter().map(|s| {
        let peaks: Vec<String> = s.pair_peaks.iter().map(|p| {
            format!(r#"{{"pair":"{}","n":{},"str":{:.3}}}"#, p.label, p.peak_n, p.peak_strength)
        }).collect();
        format!(
            r#"{{"layer":{},"disc_ratio":{:.1},"related":{:.3},"random":{:.4},"universal":{},"word_specific":{},"mean_cv":{:.3},"clustering":{:.3},"peaks":[{}]}}"#,
            s.layer, s.discrimination_ratio, s.related_mean, s.random_mean,
            s.band_census_universal, s.band_census_word_specific,
            s.band_census_mean_cv, s.phase_clustering,
            peaks.join(","),
        )
    }).collect();

    let dc: Vec<String> = report.depth_curve.iter().map(|v| format!("{:.1}", v)).collect();

    format!(
        r#""framework":{{"depth_curve":[{}],"peak_layer":{},"layers":[{}]}}"#,
        dc.join(","), report.dominant_depth_peak, layers.join(","),
    )
}
