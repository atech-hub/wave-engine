//! Catalog-analog measurement axes — four independently-confirmed properties
//! of how the model treats tokens, plus correlation tracking.
//!
//! Phase: WHERE tokens sit relative to each other (from relate-vocab)
//! Dignity: context sensitivity — how much processing changes with context
//! Direction: order sensitivity — how much processing changes with token order
//! Destruction: how aggressively the model processes each token (solo encoding)
//!
//! Design constraint: expose raw per-token values, NOT composite scores.
//! The axes appear independent at 168-dim 80K but may align at convergence.
//! Track the correlation matrix across scans to detect alignment emergence.

use std::f32::consts::PI;

/// Fixed context set for dignity measurement — reproducible across scans.
const DIGNITY_CONTEXTS: &[char] = &['e', 't', '.', ' ', 'a', 'n', 's'];

/// Number of partner tokens for directional measurement.
const DIRECTION_PARTNERS: usize = 8;

/// Per-token scores across all four axes.
pub struct TokenAxisScores {
    pub token: String,
    pub phase_distinctiveness: f32,  // fraction of non-conjunction pairs (0..1)
    pub dignity: f32,                // max |cos shift| across contexts (0..1, high = context-dependent)
    pub direction: f32,              // mean |asymmetry| across partners (0..1)
    pub destruction: f32,            // 1 - solo L3 cos (0..1, high = heavily processed)
}

/// Per-layer targeted destruction profile (on-grid vs off-grid).
pub struct DestructionProfile {
    pub per_layer: Vec<DestructionLayer>,
}

pub struct DestructionLayer {
    pub layer: usize,
    pub on_grid_cos: f32,
    pub off_grid_cos: f32,
    pub ratio: f32,  // off/on — higher = more targeted
}

/// Pairwise correlation matrix across the four axes.
pub struct CorrelationMatrix {
    pub phase_dignity: f32,
    pub phase_direction: f32,
    pub phase_destruction: f32,
    pub dignity_direction: f32,
    pub dignity_destruction: f32,
    pub direction_destruction: f32,
}

/// Compute dignity score for a single token.
/// Returns max |cos_context - cos_solo| across standard contexts.
pub fn compute_dignity(
    model: &crate::WavePacketModel,
    token_id: usize,
    vocab: &[char],
    n_bands: usize,
) -> f32 {
    let n_embd = n_bands * 2;

    // Solo encoding: just the token at position 0
    let mut solo_state = vec![0.0f32; n_embd];
    for i in 0..n_embd {
        solo_state[i] = model.wte[token_id][i] + model.wpe[0][i];
    }
    let (solo_out, solo_layers) = super::phase_encode::forward_from_layer(model, &solo_state, 0, n_bands);
    let solo_l3 = if solo_layers.len() >= 4 {
        super::phase_encode::cosine_similarity(&solo_state, &solo_layers[3])
    } else {
        super::phase_encode::cosine_similarity(&solo_state, &solo_out)
    };

    let tok_char = if token_id < vocab.len() { vocab[token_id] } else { return 0.0 };

    let mut max_shift = 0.0f32;
    for &ctx_char in DIGNITY_CONTEXTS {
        if ctx_char == tok_char { continue; }
        let ctx_id = vocab.iter().position(|&c| c == ctx_char);
        let ctx_id = match ctx_id { Some(id) => id, None => continue };

        // Token followed by context char (token at pos 0, context at pos 1)
        // We measure the LAST position's cos, which is the context char
        // But dignity is about the FOCUS token — use position 0
        // For a 2-token sequence, attention from pos 1 sees pos 0
        // Encode as: focus token at last position (pos 1) with context at pos 0
        let mut state = vec![0.0f32; n_embd];
        for i in 0..n_embd {
            state[i] = model.wte[token_id][i] + model.wpe[1][i]; // focus at pos 1
        }
        let (ctx_out, ctx_layers) = super::phase_encode::forward_from_layer(model, &state, 0, n_bands);
        let ctx_l3 = if ctx_layers.len() >= 4 {
            super::phase_encode::cosine_similarity(&state, &ctx_layers[3])
        } else {
            super::phase_encode::cosine_similarity(&state, &ctx_out)
        };

        let shift = (ctx_l3 - solo_l3).abs();
        if shift > max_shift { max_shift = shift; }
    }
    max_shift
}

/// Compute directional asymmetry for a single token.
/// Returns mean |asymmetry| across sampled partner tokens.
pub fn compute_direction(
    model: &crate::WavePacketModel,
    token_id: usize,
    vocab_size: usize,
    n_bands: usize,
) -> f32 {
    let n_embd = n_bands * 2;
    let n_partners = DIRECTION_PARTNERS.min(vocab_size - 1);
    let mut asym_sum = 0.0f32;
    let mut count = 0;

    // Sample partners evenly across vocab
    for p in 0..n_partners {
        let partner_id = (p * vocab_size / n_partners) % vocab_size;
        if partner_id == token_id { continue; }

        // AB: token at pos 0, partner at pos 1 — measure last position
        let mut ab_state = vec![0.0f32; n_embd];
        for i in 0..n_embd {
            ab_state[i] = model.wte[partner_id][i] + model.wpe[1][i];
        }
        let (ab_out, ab_layers) = super::phase_encode::forward_from_layer(model, &ab_state, 0, n_bands);
        let ab_l3 = if ab_layers.len() >= 4 {
            super::phase_encode::cosine_similarity(&ab_state, &ab_layers[3])
        } else {
            super::phase_encode::cosine_similarity(&ab_state, &ab_out)
        };

        // BA: partner at pos 0, token at pos 1
        let mut ba_state = vec![0.0f32; n_embd];
        for i in 0..n_embd {
            ba_state[i] = model.wte[token_id][i] + model.wpe[1][i];
        }
        let (ba_out, ba_layers) = super::phase_encode::forward_from_layer(model, &ba_state, 0, n_bands);
        let ba_l3 = if ba_layers.len() >= 4 {
            super::phase_encode::cosine_similarity(&ba_state, &ba_layers[3])
        } else {
            super::phase_encode::cosine_similarity(&ba_state, &ba_out)
        };

        asym_sum += (ab_l3 - ba_l3).abs();
        count += 1;
    }

    if count > 0 { asym_sum / count as f32 } else { 0.0 }
}

/// Compute destruction score for a single token (1 - solo L3 cos).
pub fn compute_destruction(
    model: &crate::WavePacketModel,
    token_id: usize,
    n_bands: usize,
) -> f32 {
    let n_embd = n_bands * 2;
    let mut state = vec![0.0f32; n_embd];
    for i in 0..n_embd {
        state[i] = model.wte[token_id][i] + model.wpe[0][i];
    }
    let (_out, layers) = super::phase_encode::forward_from_layer(model, &state, 0, n_bands);
    let l3_cos = if layers.len() >= 4 {
        super::phase_encode::cosine_similarity(&state, &layers[3])
    } else {
        0.0
    };
    1.0 - l3_cos
}

/// Compute targeted destruction profile (on-grid vs off-grid per layer).
pub fn compute_destruction_profile(
    model: &crate::WavePacketModel,
    n_bands: usize,
    m1: usize,
    m2: usize,
) -> DestructionProfile {
    let n_embd = n_bands * 2;
    let half = n_bands / 2;
    let n_layers = model.blocks.len();
    let n_on = 10.min(m1 * m2);
    let n_off = 10;

    // On-grid positions
    let mut on_per_layer: Vec<Vec<f32>> = vec![vec![]; n_layers];
    for pos in 0..n_on {
        let theta1 = (pos % m1) as f32 * 2.0 * PI / m1 as f32;
        let theta2 = (pos % m2) as f32 * 2.0 * PI / m2 as f32;
        let mut state = vec![0.0f32; n_embd];
        for n in 0..half {
            let phase = (n + 1) as f32 * theta1;
            state[n * 2] = phase.cos();
            state[n * 2 + 1] = phase.sin();
        }
        for n in 0..half {
            let phase = (n + 1) as f32 * theta2;
            state[(half + n) * 2] = phase.cos();
            state[(half + n) * 2 + 1] = phase.sin();
        }
        let (_out, layers) = super::phase_encode::forward_from_layer(model, &state, 0, n_bands);
        for (l, layer_out) in layers.iter().enumerate() {
            on_per_layer[l].push(super::phase_encode::cosine_similarity(&state, layer_out));
        }
    }

    // Off-grid (fractional positions)
    let fracs = [0.5, 1.5, 2.5, 3.5, 4.5, 0.25, 0.75, 1.33, 2.67, 3.14];
    let mut off_per_layer: Vec<Vec<f32>> = vec![vec![]; n_layers];
    for &frac in fracs.iter().take(n_off) {
        let theta1 = (frac % m1 as f32) * 2.0 * PI / m1 as f32;
        let theta2 = (frac % m2 as f32) * 2.0 * PI / m2 as f32;
        let mut state = vec![0.0f32; n_embd];
        for n in 0..half {
            let phase = (n + 1) as f32 * theta1;
            state[n * 2] = phase.cos();
            state[n * 2 + 1] = phase.sin();
        }
        for n in 0..half {
            let phase = (n + 1) as f32 * theta2;
            state[(half + n) * 2] = phase.cos();
            state[(half + n) * 2 + 1] = phase.sin();
        }
        let (_out, layers) = super::phase_encode::forward_from_layer(model, &state, 0, n_bands);
        for (l, layer_out) in layers.iter().enumerate() {
            off_per_layer[l].push(super::phase_encode::cosine_similarity(&state, layer_out));
        }
    }

    let per_layer: Vec<DestructionLayer> = (0..n_layers).map(|l| {
        let on_avg = if on_per_layer[l].is_empty() { 0.0 } else {
            on_per_layer[l].iter().sum::<f32>() / on_per_layer[l].len() as f32
        };
        let off_avg = if off_per_layer[l].is_empty() { 0.0 } else {
            off_per_layer[l].iter().sum::<f32>() / off_per_layer[l].len() as f32
        };
        DestructionLayer {
            layer: l,
            on_grid_cos: on_avg,
            off_grid_cos: off_avg,
            ratio: off_avg / on_avg.max(0.001),
        }
    }).collect();

    DestructionProfile { per_layer }
}

/// Compute Pearson correlation between two float slices.
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    if n < 3.0 { return 0.0; }
    let ma: f32 = a.iter().sum::<f32>() / n;
    let mb: f32 = b.iter().sum::<f32>() / n;
    let cov: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - ma) * (y - mb)).sum::<f32>() / n;
    let va: f32 = a.iter().map(|&x| (x - ma) * (x - ma)).sum::<f32>() / n;
    let vb: f32 = b.iter().map(|&x| (x - mb) * (x - mb)).sum::<f32>() / n;
    let denom = (va * vb).sqrt();
    if denom < 1e-10 { 0.0 } else { cov / denom }
}

/// Compute correlation matrix across all four axes.
pub fn correlation_matrix(scores: &[TokenAxisScores]) -> CorrelationMatrix {
    let phase: Vec<f32> = scores.iter().map(|s| s.phase_distinctiveness).collect();
    // Invert dignity: high dignity_inv = context-independent = structurally important
    let dignity_inv: Vec<f32> = scores.iter().map(|s| 1.0 - s.dignity).collect();
    let direction: Vec<f32> = scores.iter().map(|s| s.direction).collect();
    let destruction: Vec<f32> = scores.iter().map(|s| s.destruction).collect();

    CorrelationMatrix {
        phase_dignity: pearson(&phase, &dignity_inv),
        phase_direction: pearson(&phase, &direction),
        phase_destruction: pearson(&phase, &destruction),
        dignity_direction: pearson(&dignity_inv, &direction),
        dignity_destruction: pearson(&dignity_inv, &destruction),
        direction_destruction: pearson(&direction, &destruction),
    }
}

/// Compute all four axes for every token in the vocabulary.
pub fn compute_all_axes(
    model: &crate::WavePacketModel,
    n_bands: usize,
    char_map: &[char],
    phase_scores: &std::collections::HashMap<String, f32>,
) -> Vec<TokenAxisScores> {
    let vocab_size = model.vocab_size;
    println!("  Computing catalog axes for {} tokens...", vocab_size.min(char_map.len()));

    (0..vocab_size.min(char_map.len())).map(|tok| {
        let label = if char_map[tok].is_ascii_graphic() || char_map[tok] == ' ' {
            format!("{}", char_map[tok])
        } else {
            format!("t{}", tok)
        };

        let phase = phase_scores.get(&label).copied().unwrap_or(0.0);
        let dignity = compute_dignity(model, tok, char_map, n_bands);
        let direction = compute_direction(model, tok, vocab_size, n_bands);
        let destruction = compute_destruction(model, tok, n_bands);

        if (tok + 1) % 20 == 0 {
            println!("    {}/{} tokens...", tok + 1, vocab_size.min(char_map.len()));
        }

        TokenAxisScores {
            token: label,
            phase_distinctiveness: phase,
            dignity,
            direction,
            destruction,
        }
    }).collect()
}

/// Print axis scores summary.
pub fn print_axes_summary(scores: &[TokenAxisScores], corr: &CorrelationMatrix) {
    println!("\n=== Catalog Axes ===");
    println!("  {:>5} {:>7} {:>7} {:>7} {:>9}", "token", "phase", "dignity", "direct", "destruct");
    let mut sorted: Vec<&TokenAxisScores> = scores.iter().collect();
    sorted.sort_by(|a, b| b.phase_distinctiveness.partial_cmp(&a.phase_distinctiveness).unwrap_or(std::cmp::Ordering::Equal));
    for s in sorted.iter().take(10) {
        println!("  {:>5} {:>7.3} {:>7.3} {:>7.3} {:>9.3}",
            s.token, s.phase_distinctiveness, s.dignity, s.direction, s.destruction);
    }

    println!("\n  Correlation matrix:");
    println!("    phase ↔ dignity_inv:    {:+.4}", corr.phase_dignity);
    println!("    phase ↔ direction:      {:+.4}", corr.phase_direction);
    println!("    phase ↔ destruction:    {:+.4}", corr.phase_destruction);
    println!("    dignity ↔ direction:    {:+.4}", corr.dignity_direction);
    println!("    dignity ↔ destruction:  {:+.4}", corr.dignity_destruction);
    println!("    direction ↔ destruction:{:+.4}", corr.direction_destruction);
}

/// Write axis data to JSON.
pub fn write_axes_json(
    f: &mut dyn std::io::Write,
    scores: &[TokenAxisScores],
    corr: &CorrelationMatrix,
    profile: Option<&DestructionProfile>,
) -> std::io::Result<()> {
    write!(f, "  \"catalog_axes\": {{\n")?;
    write!(f, "    \"per_token\": [\n")?;
    for (i, s) in scores.iter().enumerate() {
        let esc = s.token.replace('\\', "\\\\").replace('"', "\\\"");
        write!(f, "      {{\"token\":\"{}\",\"phase\":{:.4},\"dignity\":{:.4},\"direction\":{:.4},\"destruction\":{:.4}}}{}",
            esc, s.phase_distinctiveness, s.dignity, s.direction, s.destruction,
            if i + 1 < scores.len() { ",\n" } else { "\n" })?;
    }
    write!(f, "    ],\n")?;
    write!(f, "    \"correlation_matrix\": {{\n")?;
    write!(f, "      \"phase_dignity\":{:.4},\"phase_direction\":{:.4},\"phase_destruction\":{:.4},\n",
        corr.phase_dignity, corr.phase_direction, corr.phase_destruction)?;
    write!(f, "      \"dignity_direction\":{:.4},\"dignity_destruction\":{:.4},\"direction_destruction\":{:.4}\n",
        corr.dignity_direction, corr.dignity_destruction, corr.direction_destruction)?;
    write!(f, "    }}")?;
    if let Some(prof) = profile {
        write!(f, ",\n    \"destruction_profile\": [\n")?;
        for (i, l) in prof.per_layer.iter().enumerate() {
            write!(f, "      {{\"layer\":{},\"on_grid\":{:.4},\"off_grid\":{:.4},\"ratio\":{:.3}}}{}",
                l.layer, l.on_grid_cos, l.off_grid_cos, l.ratio,
                if i + 1 < prof.per_layer.len() { ",\n" } else { "\n" })?;
        }
        write!(f, "    ]")?;
    }
    write!(f, "\n  }}")?;
    Ok(())
}
