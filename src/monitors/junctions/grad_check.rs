//! Gradient correctness monitor — finite-difference verification of analytical gradients.
//!
//! Mode-agnostic: takes forward and forward+backward closures from any training mode.
//! Per-section reporting highlights systematic failures (e.g., "all of block_0_phase_proj_w fails").
//!
//! See GRAD-CHECK-MONITOR-SPEC.md for design rationale.

use crate::common::rng::Rng;

// ─── Configuration ───

pub struct GradCheckConfig {
    pub eps: f32,
    pub rel_tol: f32,
    pub mode: CheckMode,
    pub verbose: bool,
    pub section_filter: Option<Vec<String>>,
}

pub enum CheckMode {
    Exhaustive,
    Sampled { n: usize },
    PerSection { n_per_section: usize },
}

impl Default for GradCheckConfig {
    fn default() -> Self {
        Self {
            eps: 1e-4,
            rel_tol: 1e-3,
            mode: CheckMode::PerSection { n_per_section: 5 },
            verbose: false,
            section_filter: None,
        }
    }
}

// ─── Section labels ───

pub struct SectionLabels {
    /// Sorted list of (start_index, section_name).
    /// A parameter at index i belongs to the section whose start_index is the largest value <= i.
    pub ranges: Vec<(usize, String)>,
}

impl SectionLabels {
    pub fn new(ranges: Vec<(usize, String)>) -> Self {
        Self { ranges }
    }

    pub fn label_for(&self, param_index: usize) -> &str {
        // Binary search for the largest start_index <= param_index
        match self.ranges.binary_search_by_key(&param_index, |(start, _)| *start) {
            Ok(i) => &self.ranges[i].1,
            Err(i) if i > 0 => &self.ranges[i - 1].1,
            _ => "unknown",
        }
    }

    /// Get all unique section names in order.
    pub fn sections(&self) -> Vec<&str> {
        self.ranges.iter().map(|(_, name)| name.as_str()).collect()
    }

    /// Get parameter indices belonging to a section.
    /// Returns (start, end) range where end is the next section's start (or total params).
    pub fn range_for(&self, section: &str, total_params: usize) -> Option<(usize, usize)> {
        for (i, (start, name)) in self.ranges.iter().enumerate() {
            if name == section {
                let end = if i + 1 < self.ranges.len() { self.ranges[i + 1].0 } else { total_params };
                return Some((*start, end));
            }
        }
        None
    }
}

// ─── Result types ───

pub struct GradCheckResult {
    pub mode_name: String,
    pub n_params_total: usize,
    pub n_params_checked: usize,
    pub n_passed: usize,
    pub max_rel_err: f32,
    pub median_rel_err: f32,
    pub failures: Vec<GradCheckFailure>,
    pub per_section_summary: Vec<SectionSummary>,
}

pub struct GradCheckFailure {
    pub param_index: usize,
    pub section: String,
    pub fd_value: f32,
    pub analytical_value: f32,
    pub rel_err: f32,
    pub abs_err: f32,
}

pub struct SectionSummary {
    pub section: String,
    pub n_checked: usize,
    pub n_passed: usize,
    pub max_rel_err: f32,
    pub median_rel_err: f32,
    pub all_zero_gradients: bool,
}

impl GradCheckResult {
    pub fn passed(&self) -> bool {
        self.n_passed == self.n_params_checked
    }
}

// ─── Core logic ───

pub fn check_gradients<F1, F2>(
    mode_name: &str,
    forward_fn: F1,
    forward_backward_fn: F2,
    initial_params: &[f32],
    section_labels: &SectionLabels,
    config: GradCheckConfig,
) -> GradCheckResult
where
    F1: Fn(&[f32]) -> f64,
    F2: Fn(&[f32]) -> (f32, Vec<f32>),
{
    let n_total = initial_params.len();

    // Step 1: Get analytical gradients at the base point
    let (_base_loss, analytical) = forward_backward_fn(initial_params);
    assert_eq!(analytical.len(), n_total, "Gradient length {} != param length {}", analytical.len(), n_total);

    // Step 2: Select parameters to check
    let indices = select_indices(&config.mode, n_total, section_labels, &config.section_filter);

    // Step 3: Run FD checks
    let mut params_buf = initial_params.to_vec();
    let mut all_errors: Vec<(usize, f32)> = Vec::new(); // (index, rel_err)
    let mut failures: Vec<GradCheckFailure> = Vec::new();
    let mut n_passed = 0usize;

    // Per-section tracking
    let mut section_results: std::collections::HashMap<String, Vec<(f32, f32)>> = std::collections::HashMap::new(); // section -> [(rel_err, analytical)]

    for &idx in &indices {
        // +eps
        params_buf[idx] = initial_params[idx] + config.eps;
        let loss_plus = forward_fn(&params_buf);
        // -eps
        params_buf[idx] = initial_params[idx] - config.eps;
        let loss_minus = forward_fn(&params_buf);
        // restore
        params_buf[idx] = initial_params[idx];

        // forward_fn now returns f64 — the loss is accumulated at f64 inside
        // the forward closure so the FD subtraction doesn't cancel against f32
        // quantization. Analytical gradient is promoted to f64 for comparison.
        let fd64 = (loss_plus - loss_minus) / (2.0_f64 * config.eps as f64);
        let an64 = analytical[idx] as f64;
        let denom64 = fd64.abs().max(an64.abs()).max(1e-8_f64);
        let rel_err = ((fd64 - an64).abs() / denom64) as f32;
        let abs_err = (fd64 - an64).abs() as f32;
        let fd = fd64 as f32;
        let an = analytical[idx];

        let section = section_labels.label_for(idx).to_string();

        section_results.entry(section.clone()).or_default().push((rel_err, an));

        if config.verbose {
            let status = if rel_err < config.rel_tol { "OK" } else { "FAIL" };
            eprintln!("  [{}] param[{}] {}: fd={:.6} an={:.6} rel={:.6} {}",
                mode_name, idx, section, fd, an, rel_err, status);
        }

        if rel_err < config.rel_tol {
            n_passed += 1;
        } else {
            failures.push(GradCheckFailure {
                param_index: idx, section, fd_value: fd,
                analytical_value: an, rel_err, abs_err,
            });
        }
        all_errors.push((idx, rel_err));
    }

    // Step 4: Compute per-section summaries
    let mut per_section_summary: Vec<SectionSummary> = Vec::new();
    // Iterate sections in order
    for (_, sec_name) in &section_labels.ranges {
        if let Some(results) = section_results.get(sec_name) {
            let n_checked = results.len();
            let n_sec_passed = results.iter().filter(|(re, _)| *re < config.rel_tol).count();
            let mut errs: Vec<f32> = results.iter().map(|(re, _)| *re).collect();
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let max_err = errs.last().copied().unwrap_or(0.0);
            let median_err = if errs.is_empty() { 0.0 } else { errs[errs.len() / 2] };
            let all_zero = results.iter().all(|(_, an)| an.abs() < 1e-12);

            // Avoid duplicate section entries
            if !per_section_summary.iter().any(|s| s.section == *sec_name) {
                per_section_summary.push(SectionSummary {
                    section: sec_name.clone(),
                    n_checked, n_passed: n_sec_passed,
                    max_rel_err: max_err, median_rel_err: median_err,
                    all_zero_gradients: all_zero,
                });
            }
        }
    }

    // Overall stats
    let mut all_rel: Vec<f32> = all_errors.iter().map(|(_, re)| *re).collect();
    all_rel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let max_rel_err = all_rel.last().copied().unwrap_or(0.0);
    let median_rel_err = if all_rel.is_empty() { 0.0 } else { all_rel[all_rel.len() / 2] };

    // Sort failures by rel_err descending
    failures.sort_by(|a, b| b.rel_err.partial_cmp(&a.rel_err).unwrap());

    GradCheckResult {
        mode_name: mode_name.to_string(),
        n_params_total: n_total,
        n_params_checked: indices.len(),
        n_passed,
        max_rel_err,
        median_rel_err,
        failures,
        per_section_summary,
    }
}

// ─── Parameter selection ───

fn select_indices(
    mode: &CheckMode,
    n_total: usize,
    labels: &SectionLabels,
    filter: &Option<Vec<String>>,
) -> Vec<usize> {
    match mode {
        CheckMode::Exhaustive => {
            let all: Vec<usize> = (0..n_total).collect();
            apply_filter(all, labels, filter)
        }
        CheckMode::Sampled { n } => {
            let mut rng = Rng::new(42);
            let mut indices: Vec<usize> = (0..(*n).min(n_total))
                .map(|_| rng.next_u64() as usize % n_total)
                .collect();
            indices.sort();
            indices.dedup();
            apply_filter(indices, labels, filter)
        }
        CheckMode::PerSection { n_per_section } => {
            let mut indices = Vec::new();
            let mut rng = Rng::new(42);
            let sections = labels.sections();
            let mut seen = std::collections::HashSet::new();
            for sec in sections {
                if seen.contains(sec) { continue; }
                seen.insert(sec);
                if let Some((start, end)) = labels.range_for(sec, n_total) {
                    let section_size = end - start;
                    if section_size == 0 { continue; }
                    let n = (*n_per_section).min(section_size);
                    for _ in 0..n {
                        indices.push(start + (rng.next_u64() as usize % section_size));
                    }
                }
            }
            indices.sort();
            indices.dedup();
            apply_filter(indices, labels, filter)
        }
    }
}

fn apply_filter(indices: Vec<usize>, labels: &SectionLabels, filter: &Option<Vec<String>>) -> Vec<usize> {
    match filter {
        None => indices,
        Some(allowed) => indices.into_iter()
            .filter(|&i| allowed.iter().any(|a| labels.label_for(i) == a.as_str()))
            .collect(),
    }
}

// ─── Pretty printer ───

pub fn print_result(result: &GradCheckResult) {
    eprintln!("\n=== Gradient Check: {} ===", result.mode_name);
    eprintln!("Total params: {}", result.n_params_total);
    eprintln!("Checked: {}", result.n_params_checked);
    eprintln!("Passed: {} / {}", result.n_passed, result.n_params_checked);
    eprintln!();

    // Per-section summary
    eprintln!("Per-section summary:");
    for sec in &result.per_section_summary {
        let status = if sec.n_passed == sec.n_checked {
            format!("(max_err {:.1e})", sec.max_rel_err)
        } else if sec.all_zero_gradients {
            "← DEAD GRADIENT".to_string()
        } else if sec.n_passed == 0 {
            format!("(max_err {:.2}, median {:.2})  ← LIKELY BUG", sec.max_rel_err, sec.median_rel_err)
        } else {
            format!("(max_err {:.2})  ← MIXED", sec.max_rel_err)
        };
        eprintln!("  {:<35} {}/{} passed  {}",
            sec.section, sec.n_passed, sec.n_checked, status);
    }

    // Top failures
    if !result.failures.is_empty() {
        eprintln!();
        let n_show = 10.min(result.failures.len());
        eprintln!("Failures (top {} by rel_err):", n_show);
        for f in result.failures.iter().take(n_show) {
            eprintln!("  param[{}] {}: fd={:.6} an={:.6} rel={:.4}",
                f.param_index, f.section, f.fd_value, f.analytical_value, f.rel_err);
        }
    }

    eprintln!();
    let verdict = if result.passed() { "PASS" } else {
        let n_bad = result.per_section_summary.iter().filter(|s| s.n_passed < s.n_checked).count();
        if n_bad > 0 {
            "FAIL"
        } else { "PASS" }
    };
    if result.passed() {
        eprintln!("VERDICT: PASS");
    } else {
        let n_bad = result.per_section_summary.iter().filter(|s| s.n_passed < s.n_checked).count();
        eprintln!("VERDICT: FAIL ({} sections with errors)", n_bad);
    }
    eprintln!();
}
