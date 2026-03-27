//! Analyze mode — extracted from main.rs.
//! Forward pass + wave structure diagnostics (--analyze flag).

use crate::common::wave_analysis as wa;
use crate::common::dims::{N_BANDS, N_HEAD, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS};
use crate::Dims;
use crate::{init_model, unflatten_params};
use crate::wave_checkpoint;
use crate::bpe;
use crate::fft_ode;
use crate::cpu::forward::forward_with_cache;

pub fn run_analyze(
    resume_path: &str,
    n_layers: usize,
    out_proj_groups: usize,
    use_bpe: bool,
    tokenizer_path: &str,
    n_bands: usize,
    n_head: usize,
    alpha: f32,
    beta: f32,
    sub_harmonic: bool,
) {
    println!("Analyze mode: harmonic coherence diagnostics\n");

    // Curated test sentences — covering semantics, grammar, and registers
    let test_text = concat!(
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

    // Tokenize — BPE or char-level
    let (token_ids, vocab_size, token_strings) = if use_bpe {
        let bpe = bpe::BpeTokenizer::from_file(tokenizer_path);
        let ids: Vec<usize> = bpe.encode(test_text);
        let strings: Vec<String> = ids.iter().map(|&id| bpe.decode(&[id])).collect();
        let vs: usize = ids.iter().max().copied().unwrap_or(0) + 1; // conservative vocab bound
        (ids, vs, strings)
    } else {
        let chars: Vec<char> = test_text.chars().collect();
        let mut vc: Vec<char> = chars.iter().cloned()
            .collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        vc.sort();
        let c2i: std::collections::HashMap<char, usize> = vc.iter()
            .enumerate().map(|(i, &c)| (c, i)).collect();
        let ids: Vec<usize> = chars.iter().map(|c| c2i[c]).collect();
        let strings: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        (ids, vc.len(), strings)
    };

    // Find word spans — a word may be 1 token (char-level) or multiple (BPE sub-tokens).
    // Concatenate adjacent token strings and find where the target word appears.
    // Find the token span for a word. BPE tokens may have leading space (Ġ → " ").
    // "cat" could be [" c", "at"] or [" cat"] depending on vocab size.
    let find_word_span = |word: &str| -> Option<Vec<usize>> {
        let word_lower = word.to_lowercase();
        for start in 0..token_strings.len() {
            let mut concat = String::new();
            for end in start..token_strings.len().min(start + 5) {
                concat.push_str(&token_strings[end]);
                // Clean: strip leading BPE space marker, lowercase, trim
                let clean = concat.replace('\u{0120}', " ").to_lowercase();
                let clean = clean.trim();
                // Exact match: the concatenated tokens form exactly this word
                if clean == word_lower || clean == format!(" {word_lower}") {
                    return Some((start..=end).collect());
                }
            }
        }
        None
    };

    println!("  Test text: {} tokens", token_ids.len());
    println!("  Tokenizer: {}", if use_bpe { "BPE" } else { "char-level" });

    // Build semantic pairs — works with both single-token and multi-token words
    let mut related_span_pairs: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    let mut related_labels: Vec<(String, String)> = Vec::new();
    let mut random_span_pairs: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();

    let semantic_pairs = [
        ("cat", "dog"),         // same category (animals)
        ("sat", "kicked"),      // same category (verbs)
        ("boy", "ball"),        // subject-object in same sentence
        ("noun", "verb"),       // same category (grammar terms)
        ("love", "war"),        // semantic opposites
        ("mat", "rug"),         // synonyms
    ];
    let random_pair_words = [
        ("cat", "contract"),
        ("verb", "temperature"),
        ("boy", "question"),
        ("dog", "execution"),
        ("sat", "binding"),
        ("mat", "change"),
    ];

    for (w1, w2) in &semantic_pairs {
        if let (Some(span_a), Some(span_b)) = (find_word_span(w1), find_word_span(w2)) {
            println!("    Related: ({w1}@{:?}, {w2}@{:?})", span_a, span_b);
            related_labels.push((w1.to_string(), w2.to_string()));
            related_span_pairs.push((span_a, span_b));
        }
    }
    for (w1, w2) in &random_pair_words {
        if let (Some(span_a), Some(span_b)) = (find_word_span(w1), find_word_span(w2)) {
            random_span_pairs.push((span_a, span_b));
        }
    }

    // Also build single-position pairs for backward compatibility (band census, clustering)
    let mut related_pairs: Vec<(usize, usize)> = related_span_pairs.iter()
        .map(|(a, b)| (a[0], b[0])).collect();
    let mut random_pairs: Vec<(usize, usize)> = random_span_pairs.iter()
        .map(|(a, b)| (a[0], b[0])).collect();

    if related_span_pairs.is_empty() {
        println!("  WARNING: No semantic pairs found in tokens. Using positional fallback.");
        let t = token_ids.len();
        for i in (0..t.min(20)).step_by(2) {
            if i + 1 < t {
                related_pairs.push((i, i + 1));
                related_span_pairs.push((vec![i], vec![i + 1]));
            }
        }
        for i in 0..t.min(10).min(t / 2) {
            random_pairs.push((i, (i + t / 2) % t));
            random_span_pairs.push((vec![i], vec![(i + t / 2) % t]));
        }
    }

    println!("  Pairs: {} related, {} random", related_pairs.len(), random_pairs.len());

    // Load model from checkpoint
    let (params, ck_vocab, _ck_iter, _ck_lr, _ck_rng, _adam_t, _adam_m, _adam_v, _ck_groups) =
        wave_checkpoint::load_checkpoint(resume_path);
    let effective_vocab = vocab_size.max(ck_vocab);
    let dims = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS);
    let mut model = init_model(effective_vocab, 42, n_layers, out_proj_groups, dims, alpha, beta);
    unflatten_params(&mut model, &params);
    println!("  Loaded {} params from {}", params.len(), resume_path);

    // Truncate to block_size if needed (char-level can exceed positional table)
    let max_tokens = dims.block_size.min(token_ids.len());
    let token_ids = token_ids[..max_tokens].to_vec();
    let token_strings: Vec<String> = token_strings[..max_tokens].to_vec();
    // Filter all pairs that reference positions beyond truncation
    related_span_pairs.retain(|(a, b)| a.iter().all(|&i| i < max_tokens) && b.iter().all(|&i| i < max_tokens));
    random_span_pairs.retain(|(a, b)| a.iter().all(|&i| i < max_tokens) && b.iter().all(|&i| i < max_tokens));
    related_labels.truncate(related_span_pairs.len());
    // Rebuild single-position pairs from filtered spans
    related_pairs = related_span_pairs.iter().map(|(a, b)| (a[0], b[0])).collect();
    random_pairs = random_span_pairs.iter().map(|(a, b)| (a[0], b[0])).collect();

    // Forward pass
    let stencil = fft_ode::StencilFft::new(dims.n_bands);
    let cache = forward_with_cache(&model, &token_ids, dims, None, None, None, Some(&stencil), None);

    // Extract per-layer phases
    let mut per_layer_phases: Vec<Vec<Vec<f32>>> = Vec::new();
    for bc in &cache.block_caches {
        per_layer_phases.push(wa::extract_all_phases(&bc.input, dims.n_bands));
    }
    per_layer_phases.push(wa::extract_all_phases(&cache.post_ln_f, dims.n_bands));

    // Build token labels for display (use related pair words where available)
    let display_labels: Vec<String> = token_strings.iter()
        .map(|s| s.trim().replace('\n', "\\n").chars().take(12).collect())
        .collect();

    // Run full report (uses span-based discrimination for proper multi-token words)
    {
        let deep = per_layer_phases.last().unwrap();
        let disc = wa::semantic_discrimination_spans(deep, &related_span_pairs, &random_span_pairs, 12);
        let verdict = if disc.ratio > 2.0 { "STRONG SEMANTIC STRUCTURE" }
            else if disc.ratio > 1.5 { "EMERGING STRUCTURE" }
            else { "NOT YET" };
        println!("\n=== Wave Structure Report ===");
        println!("Checkpoint: {resume_path}");
        println!("Layers: {n_layers}, Bands: {}, Tokens: {}", dims.n_bands, token_ids.len());
        println!("\n1. Semantic Discrimination (span-averaged for multi-token words)");
        println!("   Related: {:.3}    Random: {:.3}    Ratio: {:.1}x    {verdict}",
            disc.related_mean, disc.random_mean, disc.ratio);
        // Print matched pairs
        for (label, (span_a, span_b)) in related_labels.iter().zip(&related_span_pairs) {
            let avg_a = wa::average_phases_over_span(deep, span_a);
            let avg_b = wa::average_phases_over_span(deep, span_b);
            let spectrum = wa::harmonic_spectrum(&avg_a, &avg_b, 12);
            let (best_coh, best_n) = wa::best_harmonic_coherence(&avg_a, &avg_b, 12);
            println!("   ({}, {}): peak n={best_n} ({best_coh:.2})  spans {:?} {:?}",
                label.0, label.1, span_a, span_b);
        }
    }
    // Rest of report (band census, clustering, depth curve use positional pairs)
    wa::print_report(
        resume_path, n_layers, dims.n_bands, token_ids.len(),
        &per_layer_phases, &related_pairs, &random_pairs, &display_labels,
    );

    // Sub-harmonic diagnostics (--sub-harmonic flag)
    if sub_harmonic {
        let deep_hidden = &cache.post_ln_f; // use deepest hidden states (raw, not phases)
        let clustering = wa::phase_clustering(per_layer_phases.last().unwrap(), dims.n_bands);
        crate::common::sub_harmonic::print_report(
            deep_hidden, &related_pairs, &random_pairs,
            dims.n_bands, alpha, beta, clustering,
        );
    }

    // Save JSON report
    std::fs::create_dir_all("analysis").ok();
    let deep = per_layer_phases.last().unwrap();
    let disc = wa::semantic_discrimination_spans(deep, &related_span_pairs, &random_span_pairs, 12);
    let census = wa::band_census(deep, dims.n_bands);
    let clustering = wa::phase_clustering(deep, dims.n_bands);
    let curve = wa::depth_curve(&per_layer_phases, &related_pairs, &random_pairs, 12);

    let report = serde_json::json!({
        "checkpoint": resume_path,
        "n_layers": n_layers,
        "n_bands": dims.n_bands,
        "n_tokens": token_ids.len(),
        "semantic_discrimination": {
            "related_mean": disc.related_mean,
            "random_mean": disc.random_mean,
            "ratio": disc.ratio,
        },
        "band_census": {
            "universal": census.universal,
            "word_specific": census.word_specific,
            "bimodal": census.bimodal,
            "mean_cv": census.mean_circular_variance,
        },
        "phase_clustering": clustering,
        "depth_curve": curve,
        "related_pairs": related_labels.iter().map(|(a, b)| format!("{a}/{b}")).collect::<Vec<_>>(),
    });
    std::fs::write("analysis/wave_report.json",
        serde_json::to_string_pretty(&report).unwrap()).unwrap();
    println!("\nSaved: analysis/wave_report.json");
}
