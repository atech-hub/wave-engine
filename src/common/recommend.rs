//! Architecture calculator — analyzes data and recommends optimal configuration.
//!
//! Two-bottleneck model: both bands (discrimination) AND attention (position)
//! must be satisfied. Fixing one without the other gives zero improvement.
//!
//! Phase-native is the default recommendation (no lm_head, vocab-independent params).

use std::collections::BTreeSet;

// ── Practical band values (validated or interpolated) ──────────────
const PRACTICAL_BANDS: &[usize] = &[42, 64, 84, 96, 128, 192, 256, 384, 512];

// ── Task types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskType {
    Arithmetic,
    Words,
    Grammar,
    Language,
}

impl TaskType {
    pub fn default_seq(&self) -> usize {
        match self {
            TaskType::Arithmetic | TaskType::Words => 16,
            TaskType::Grammar | TaskType::Language => 256,
        }
    }

    pub fn base_layers(&self) -> usize {
        match self {
            TaskType::Arithmetic | TaskType::Words => 4,
            TaskType::Grammar => 6,
            TaskType::Language => 8,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            TaskType::Arithmetic => "arithmetic",
            TaskType::Words => "words",
            TaskType::Grammar => "grammar",
            TaskType::Language => "language",
        }
    }

    pub fn from_str(s: &str) -> Option<TaskType> {
        match s {
            "arithmetic" | "arith" => Some(TaskType::Arithmetic),
            "words" | "word" => Some(TaskType::Words),
            "grammar" | "gram" => Some(TaskType::Grammar),
            "language" | "lang" => Some(TaskType::Language),
            _ => None,
        }
    }
}

// ── Data analysis ──────────────────────────────────────────────────

struct DataProfile {
    file_path: String,
    corpus_chars: usize,
    vocab_size: usize,
    task: TaskType,
    task_overridden: bool,
}

fn analyze_data(path: &str, task_override: Option<TaskType>) -> DataProfile {
    // Use data_loader for format detection (handles .txt, .jsonl, directories)
    let text = crate::common::data_loader::load_text_raw(path);

    let corpus_chars = text.len();

    // Character-level vocab
    let chars: BTreeSet<char> = text.chars().collect();
    let vocab_size = chars.len();

    let task_overridden = task_override.is_some();
    let task = task_override.unwrap_or_else(|| detect_task_type(&text, vocab_size));

    DataProfile { file_path: path.to_string(), corpus_chars, vocab_size, task, task_overridden }
}

fn detect_task_type(data: &str, vocab_size: usize) -> TaskType {
    let lines: Vec<&str> = data.lines().take(100).collect();

    // Arithmetic: small vocab, has + and =
    if vocab_size <= 20 && data.contains('+') && data.contains('=') {
        return TaskType::Arithmetic;
    }

    // Words: small vocab, short lines with =
    if vocab_size <= 40 && lines.iter().all(|l| l.len() < 20 && l.contains('=')) {
        return TaskType::Words;
    }

    // Grammar vs Language by corpus size
    if data.len() < 5_000_000 {
        TaskType::Grammar
    } else {
        TaskType::Language
    }
}

// ── Coprime moduli (duplicated from embed.rs for independence) ─────

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

fn find_coprime_moduli(vocab_size: usize) -> (usize, usize) {
    let root = (vocab_size as f64).sqrt().ceil() as usize;
    let mut m1 = root;
    if m1 % 2 == 0 { m1 += 1; }
    let mut m2 = m1 + 2;
    while gcd(m1, m2) != 1 || m1 * m2 < vocab_size {
        m2 += 1;
    }
    (m1, m2)
}

/// Count active bands (bands that are NOT dead on either grid).
/// A band is dead if it maps to the same phase for all tokens on that grid.
/// Dead ratio approximation: 1/m1 + 1/m2 of half-bands each.
fn compute_active_bands(n_bands: usize, m1: usize, m2: usize) -> usize {
    let half = n_bands / 2;
    // For grid 1: bands at harmonic n where n is a multiple of m1 are dead
    let dead1 = half / m1;
    // For grid 2: bands at harmonic n where n is a multiple of m2 are dead
    let dead2 = half / m2;
    let active = n_bands - dead1 - dead2;
    active
}

// ── Next practical band value ──────────────────────────────────────

fn next_practical(min_bands: usize) -> usize {
    for &b in PRACTICAL_BANDS {
        if b >= min_bands { return b; }
    }
    // Beyond our table — round up to next multiple of 64
    ((min_bands + 63) / 64) * 64
}

// ── Recommendation ─────────────────────────────────────────────────

struct Recommendation {
    // Dataset
    file_path: String,
    vocab_size: usize,
    corpus_chars: usize,
    m1: usize,
    m2: usize,
    task: TaskType,
    task_overridden: bool,

    // Bands
    n_bands: usize,
    active_bands: usize,
    tokens_per_dim: f32,
    band_utilisation: f32,

    // Attention
    seq_len: usize,
    n_head: usize,
    head_dim: usize,
    pos_per_head: f32,

    // Depth
    n_layers: usize,
    max_useful_layers: usize,

    // Training
    phase1_iters: usize,
    phase2_iters: usize,

    // Compute
    compute_tier: &'static str,
    gpu_flag: bool,

    // Model
    params: usize,

    // Warnings
    warnings: Vec<String>,
}

fn compute_recommendation(profile: &DataProfile, seq_override: Option<usize>) -> Recommendation {
    let (m1, m2) = find_coprime_moduli(profile.vocab_size);

    // ── Bottleneck 1: Bands ──
    // tokens_per_dim < 0.50 threshold
    // effective_dim = active_bands * 2 (real + imag per band)
    // We need: vocab / effective_dim < 0.50
    // So: effective_dim > vocab / 0.50 = vocab * 2
    // active_bands > vocab (since effective_dim = active_bands * 2)
    // But active_bands = n_bands * (1 - dead_ratio) * utilisation_cap
    let min_active = ((profile.vocab_size as f64) / (2.0 * 0.50)).ceil() as usize;
    let dead_ratio = 1.0 / m1 as f64 + 1.0 / m2 as f64;
    let min_bands_raw = ((min_active as f64) / (1.0 - dead_ratio) / 0.85).ceil() as usize;
    let n_bands = next_practical(min_bands_raw.max(42)); // minimum 42 bands

    let active_bands = compute_active_bands(n_bands, m1, m2);
    let effective_dim = active_bands * 2;
    let tokens_per_dim = profile.vocab_size as f32 / effective_dim as f32;
    let band_utilisation = active_bands as f32 / n_bands as f32;

    // ── Bottleneck 2: Attention ──
    let seq_len = seq_override.unwrap_or_else(|| profile.task.default_seq());
    let min_heads = 4usize.max(((seq_len as f64) / 32.0).ceil() as usize); // positions_per_head < 32 (stricter than spec's 40)
    let dim = n_bands * 2;

    // Find valid (n_head, head_dim) where head_dim >= 16 and dim divisible by n_head
    let (n_head, head_dim) = find_head_config(dim, min_heads);
    let pos_per_head = seq_len as f32 / n_head as f32;

    // ── Layers ──
    let max_useful = 2 + active_bands / 20;
    let n_layers = profile.task.base_layers().min(max_useful);

    // ── Training iterations ──
    let windows = if seq_len > 0 { profile.corpus_chars / seq_len } else { 1 };
    let phase1_iters = 10_000;
    let phase2_iters = (windows * 10).max(40_000).min(500_000);

    // ── Compute tier ──
    let (compute_tier, gpu_flag) = match n_bands {
        0..=127 => ("CPU", false),
        128..=255 => ("CPU or GPU (similar performance)", false),
        _ => ("GPU (--gpu)", true),
    };

    // ── Parameters (phase-native) ──
    let params = estimate_params(n_bands, n_head, n_layers);

    // ── Warnings ──
    let mut warnings = Vec::new();

    if tokens_per_dim >= 0.50 {
        warnings.push(format!(
            "tokens_per_dim = {:.2} (>= 0.50) — discrimination may be insufficient",
            tokens_per_dim
        ));
    }
    if band_utilisation >= 0.85 {
        warnings.push(format!(
            "band_utilisation = {:.0}% (>= 85%) — consider more bands",
            band_utilisation * 100.0
        ));
    }
    if pos_per_head >= 40.0 {
        warnings.push(format!(
            "positions_per_head = {:.0} (>= 40) — attention may be too diffuse",
            pos_per_head
        ));
    }
    if head_dim < 16 {
        warnings.push(format!(
            "head_dim = {} (< 16) — attention heads may lack capacity",
            head_dim
        ));
    }
    if n_layers < profile.task.base_layers() {
        warnings.push(format!(
            "layers capped at {} (max useful) vs {} (task default) — need more bands for deeper model",
            n_layers, profile.task.base_layers()
        ));
    }
    let exposures = if seq_len > 0 { windows / seq_len.max(1) } else { 0 };
    if exposures < 10 && profile.corpus_chars > 0 {
        warnings.push(format!(
            "exposures_per_window ~{} (< 10) — corpus may be too small for seq={}",
            windows.max(1), seq_len
        ));
    }

    Recommendation {
        file_path: profile.file_path.clone(),
        vocab_size: profile.vocab_size,
        corpus_chars: profile.corpus_chars,
        m1, m2,
        task: profile.task,
        task_overridden: profile.task_overridden,
        n_bands, active_bands,
        tokens_per_dim, band_utilisation,
        seq_len, n_head, head_dim, pos_per_head,
        n_layers, max_useful_layers: max_useful,
        phase1_iters, phase2_iters,
        compute_tier, gpu_flag,
        params,
        warnings,
    }
}

fn find_head_config(dim: usize, min_heads: usize) -> (usize, usize) {
    // Try from min_heads upward, find first that divides dim with head_dim >= 16
    let mut h = min_heads;
    loop {
        if dim % h == 0 {
            let hd = dim / h;
            if hd >= 16 {
                return (h, hd);
            }
        }
        h += 1;
        if h > dim {
            // Fallback: just use min_heads even if not perfectly divisible
            return (min_heads, dim / min_heads);
        }
    }
}

/// Phase-native parameter estimation (no lm_head, vocab-independent).
///
/// Per block:
///   - Attention: Q,K,V projections = 3 * dim * dim, output proj = dim * dim → 4 * dim^2
///   - FFN (Kerr-ODE): alpha, beta per layer (2 params), plus stencil weights = ~n_bands
///   - Maestro: maestro_dim * dim (default maestro_dim = 4)
///   - Layer norm: 2 * dim (scale + bias) × 2 (pre-attn + pre-ffn) = 4 * dim
///   - Output projection: dim * dim (grouped, so dim * dim / groups)
///
/// Global:
///   - Positional encoding: frozen (0 trainable)
///   - Token embedding: frozen harmonic (0 trainable)
///   - Output corrector: n_bands params (phase-native)
fn estimate_params(n_bands: usize, n_head: usize, n_layers: usize) -> usize {
    let dim = n_bands * 2;
    let maestro_dim = 4; // default

    let mut per_block = 0usize;

    // Attention: Wq, Wk, Wv, Wo = 4 * dim * dim
    per_block += 4 * dim * dim;

    // Maestro mixing: maestro_dim * dim * 2 (in + out)
    per_block += maestro_dim * dim * 2;

    // Kerr-ODE params per layer: alpha + beta + stencil weights
    // Stencil is [1,1,0,1,1] pattern — 5 fixed weights, but learnable version uses n_bands
    per_block += 2 + n_bands;

    // Layer norm: scale + bias, two per block (pre-attn + pre-ffn)
    per_block += 4 * dim;

    let total_blocks = per_block * n_layers;

    // Output corrector: n_bands (phase-native, no lm_head)
    let output = n_bands;

    // Total
    total_blocks + output
}

/// Estimate time per iteration in milliseconds (CPU, rough heuristic).
fn estimate_time_per_iter_ms(n_bands: usize, n_layers: usize, seq_len: usize) -> f64 {
    // Rough calibration: 42 bands, 4 layers, seq=16 ~ 5ms
    // Scales roughly as dim^2 * layers * seq
    let base_ms = 5.0;
    let dim_factor = (n_bands as f64 / 42.0).powi(2);
    let layer_factor = n_layers as f64 / 4.0;
    let seq_factor = seq_len as f64 / 16.0;
    base_ms * dim_factor * layer_factor * seq_factor
}

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { result.push(','); }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn format_time(ms: f64) -> String {
    let seconds = ms / 1000.0;
    if seconds < 60.0 {
        format!("{:.0} seconds", seconds)
    } else if seconds < 3600.0 {
        format!("{:.0} minutes", seconds / 60.0)
    } else {
        format!("{:.1} hours", seconds / 3600.0)
    }
}

fn print_recommendation(rec: &Recommendation) {
    let dim = rec.n_bands * 2;

    println!("Wave-Engine Architecture Recommendation");
    println!("{}", "=".repeat(54));
    println!();

    // ── Dataset ──
    println!("Dataset:");
    println!("  File:            {}", rec.file_path);
    println!("  Vocabulary:      {} tokens (character-level)", rec.vocab_size);
    println!("  Corpus:          {} characters", format_number(rec.corpus_chars));
    println!("  Moduli:          m1={}, m2={} (coverage={})", rec.m1, rec.m2, rec.m1 * rec.m2);
    let task_source = if rec.task_overridden { "override" } else { "auto-detected" };
    println!("  Task:            {} ({})", rec.task.name(), task_source);
    println!();

    // ── Bottleneck 1: Bands ──
    println!("Bottleneck 1 -- Bands:");
    println!("  Recommended:     {} bands ({}-dim)", rec.n_bands, dim);
    println!("  Active:          {}/{} ({:.0}%)", rec.active_bands, rec.n_bands,
        rec.band_utilisation * 100.0);
    println!("  Tokens/eff dim:  {:.2}              {}",
        rec.tokens_per_dim,
        if rec.tokens_per_dim < 0.50 { "PASS (< 0.50)" } else { "FAIL (>= 0.50)" });
    println!();

    // ── Bottleneck 2: Attention ──
    println!("Bottleneck 2 -- Attention:");
    println!("  Sequence length: {}", rec.seq_len);
    println!("  Heads:           {}", rec.n_head);
    println!("  Head dimension:  {}", rec.head_dim);
    println!("  Pos per head:    {:.0}                {}",
        rec.pos_per_head,
        if rec.pos_per_head < 40.0 { "PASS (< 40)" } else { "FAIL (>= 40)" });
    println!();

    // ── Depth ──
    println!("Depth:");
    println!("  Layers:          {}", rec.n_layers);
    println!("  Max useful:      {} (at {} active bands)", rec.max_useful_layers, rec.active_bands);
    println!("  Reason:          {} (task default: {})", rec.task.name(), rec.task.base_layers());
    println!();

    // ── Compute ──
    println!("Compute:");
    println!("  Recommended:     {}", rec.compute_tier);
    println!();

    // ── Training ──
    let ms_per_iter = estimate_time_per_iter_ms(rec.n_bands, rec.n_layers, rec.seq_len);
    let phase1_time = ms_per_iter * rec.phase1_iters as f64;
    let phase2_time = ms_per_iter * rec.phase2_iters as f64;

    println!("Training:");
    println!("  Phase 1:         {} iters (quick check -- stop if loss plateaus)",
        format_number(rec.phase1_iters));
    println!("  Phase 2:         {} iters (full run if loss still dropping)",
        format_number(rec.phase2_iters));
    println!("  Est. time/iter:  ~{:.0}ms", ms_per_iter);
    println!("  Est. Phase 1:    ~{}", format_time(phase1_time));
    println!("  Est. Phase 2:    ~{}", format_time(phase2_time));
    println!();

    // ── Model ──
    let model_size_mb = rec.params as f64 * 4.0 / (1024.0 * 1024.0);
    println!("Model:");
    println!("  Parameters:      ~{} (phase-native, vocab-independent)", format_number(rec.params));
    println!("  Model file:      ~{:.1} MB", model_size_mb);
    println!();

    // ── Warnings ──
    println!("Warnings:");
    if rec.warnings.is_empty() {
        println!("  (none)");
    } else {
        for w in &rec.warnings {
            println!("  WARNING: {}", w);
        }
    }
    println!();

    // ── Command ──
    let gpu_str = if rec.gpu_flag { " --gpu" } else { "" };
    println!("Run:");
    println!("  # Phase 1: quick check");
    println!("  wave-engine {} \\", rec.file_path);
    println!("    --n-bands {} --n-head {} --layers {} --seq {} \\",
        rec.n_bands, rec.n_head, rec.n_layers, rec.seq_len);
    println!("    --alpha 0.1 --beta 0.2 --out-proj-groups 1 \\");
    println!("    --iters {} --lr 3e-4 --phase-native{} \\",
        rec.phase1_iters, gpu_str);
    println!("    --rk4-weights dyn --harmonics dyn \\");
    println!("    --health-interval 1000");
    println!();
    println!("  # Phase 2: if loss still dropping at 10K");
    println!("  wave-engine {} \\", rec.file_path);
    println!("    --resume checkpoint.bin \\");
    println!("    --n-bands {} --n-head {} --layers {} --seq {} \\",
        rec.n_bands, rec.n_head, rec.n_layers, rec.seq_len);
    println!("    --alpha 0.1 --beta 0.2 --out-proj-groups 1 \\");
    println!("    --iters {} --lr 3e-4 --phase-native{} \\",
        rec.phase2_iters, gpu_str);
    println!("    --rk4-weights dyn --harmonics dyn \\");
    println!("    --health-interval 2000");
}

// ── Public entry point ─────────────────────────────────────────────

pub fn run_recommend(data_path: &str) {
    // Parse optional flags
    let task_override = std::env::args()
        .skip_while(|a| a != "--task")
        .nth(1)
        .and_then(|s| TaskType::from_str(&s));

    let seq_override: Option<usize> = std::env::args()
        .skip_while(|a| a != "--seq")
        .nth(1)
        .and_then(|s| s.parse().ok());

    let profile = analyze_data(data_path, task_override);
    let rec = compute_recommendation(&profile, seq_override);
    print_recommendation(&rec);
}
