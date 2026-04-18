//! J10: Tier parity — CPU and GPU forward passes must agree to numerical tolerance.
//!
//! Extends the existing `ode_parity.rs` (which covers the ODE derivative) to the
//! full forward pass. Provides a diff/tolerance framework that compares two
//! output tensors element-wise and reports max/mean absolute & relative error.
//! Per-section tolerance is supported so the ODE section can be looser than
//! linear sections.
//!
//! The monitor is tier-agnostic: the caller supplies two outputs (from whichever
//! backends they want to compare) and the monitor reports. `check_outputs_1d`
//! and `check_outputs_2d` are the primary entry points. Wiring a specific
//! CPU-vs-GPU forward runner lives alongside the forward code — not here.

/// Tolerance for a single parity comparison. Absolute and relative bounds; an
/// element passes if `|a - b| <= tol_abs + tol_rel * max(|a|, |b|)`.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub tol_abs: f32,
    pub tol_rel: f32,
}

impl Tolerance {
    pub const TIGHT: Self = Self { tol_abs: 1e-5, tol_rel: 1e-5 };
    pub const LINEAR: Self = Self { tol_abs: 1e-4, tol_rel: 1e-4 };
    /// Looser tolerance for the ODE: RK4 accumulates FP error across 16 steps.
    pub const ODE: Self = Self { tol_abs: 1e-3, tol_rel: 1e-3 };

    pub fn new(tol_abs: f32, tol_rel: f32) -> Self { Self { tol_abs, tol_rel } }

    #[inline]
    pub fn accepts(&self, a: f32, b: f32) -> bool {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs());
        diff <= self.tol_abs + self.tol_rel * scale
    }
}

/// One element-wise comparison summary.
#[derive(Debug, Clone)]
pub struct ParityDiff {
    pub section: String,
    pub n_elements: usize,
    pub n_violations: usize,
    pub max_abs_diff: f32,
    pub max_rel_diff: f32,
    pub mean_abs_diff: f32,
    /// Up to 10 worst-offending (flat_index, a, b, abs_diff) tuples.
    pub worst: Vec<(usize, f32, f32, f32)>,
}

impl ParityDiff {
    pub fn passed(&self) -> bool { self.n_violations == 0 }
}

fn rel_diff(a: f32, b: f32) -> f32 {
    let scale = a.abs().max(b.abs());
    if scale < 1e-12 { 0.0 } else { (a - b).abs() / scale }
}

/// Compare two flat slices. Any length mismatch is reported as a single
/// synthetic violation at index 0.
pub fn check_outputs_1d(section: &str, a: &[f32], b: &[f32], tol: Tolerance) -> ParityDiff {
    if a.len() != b.len() {
        return ParityDiff {
            section: section.to_string(),
            n_elements: a.len().max(b.len()),
            n_violations: a.len().max(b.len()),
            max_abs_diff: f32::INFINITY,
            max_rel_diff: f32::INFINITY,
            mean_abs_diff: f32::INFINITY,
            worst: vec![(0, a.len() as f32, b.len() as f32, f32::INFINITY)],
        };
    }
    let n = a.len();
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut n_viol = 0usize;
    let mut offenders: Vec<(usize, f32, f32, f32)> = Vec::new();

    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        let r = rel_diff(a[i], b[i]);
        sum_abs += d as f64;
        if d > max_abs { max_abs = d; }
        if r > max_rel { max_rel = r; }
        if !tol.accepts(a[i], b[i]) {
            n_viol += 1;
            offenders.push((i, a[i], b[i], d));
        }
    }
    offenders.sort_by(|x, y| y.3.partial_cmp(&x.3).unwrap_or(std::cmp::Ordering::Equal));
    offenders.truncate(10);

    ParityDiff {
        section: section.to_string(),
        n_elements: n,
        n_violations: n_viol,
        max_abs_diff: max_abs,
        max_rel_diff: max_rel,
        mean_abs_diff: if n == 0 { 0.0 } else { (sum_abs / n as f64) as f32 },
        worst: offenders,
    }
}

/// Compare two `[rows][cols]` tensors. Rows of unequal length produce a
/// violation for the whole mismatched row.
pub fn check_outputs_2d(section: &str, a: &[Vec<f32>], b: &[Vec<f32>], tol: Tolerance) -> ParityDiff {
    // Flatten both and delegate to 1D check if shapes match.
    if a.len() != b.len() || a.iter().zip(b.iter()).any(|(x, y)| x.len() != y.len()) {
        return ParityDiff {
            section: section.to_string(),
            n_elements: 0,
            n_violations: 1,
            max_abs_diff: f32::INFINITY,
            max_rel_diff: f32::INFINITY,
            mean_abs_diff: f32::INFINITY,
            worst: vec![(0, 0.0, 0.0, f32::INFINITY)],
        };
    }
    let mut flat_a: Vec<f32> = Vec::with_capacity(a.iter().map(|r| r.len()).sum());
    let mut flat_b: Vec<f32> = Vec::with_capacity(flat_a.capacity());
    for (ra, rb) in a.iter().zip(b.iter()) {
        flat_a.extend_from_slice(ra);
        flat_b.extend_from_slice(rb);
    }
    check_outputs_1d(section, &flat_a, &flat_b, tol)
}

/// Aggregate result of a multi-section parity run.
pub struct ParityReport {
    pub tier_a: String,
    pub tier_b: String,
    pub sections: Vec<ParityDiff>,
}

impl ParityReport {
    pub fn passed(&self) -> bool { self.sections.iter().all(|s| s.passed()) }
    pub fn n_violations(&self) -> usize { self.sections.iter().map(|s| s.n_violations).sum() }
}

/// Print a report to stderr.
pub fn print_report(report: &ParityReport, assert_mode: bool) {
    if report.passed() {
        let n = report.sections.iter().map(|s| s.n_elements).sum::<usize>();
        eprintln!("[J10] Parity {} vs {}: {} sections, {} elements, 0 violations — PASS",
            report.tier_a, report.tier_b, report.sections.len(), n);
        return;
    }
    eprintln!("[J10] PARITY FAILURE: {} vs {} — {} violations",
        report.tier_a, report.tier_b, report.n_violations());
    for sec in &report.sections {
        if sec.passed() { continue; }
        eprintln!("  [{}] {} / {} violations  max_abs={:.3e}  max_rel={:.3e}  mean_abs={:.3e}",
            sec.section, sec.n_violations, sec.n_elements,
            sec.max_abs_diff, sec.max_rel_diff, sec.mean_abs_diff);
        for (idx, va, vb, d) in sec.worst.iter().take(5) {
            eprintln!("    [{}] a={:.6} b={:.6} |d|={:.3e}", idx, va, vb, d);
        }
    }
    if assert_mode {
        panic!("J10: Tier parity failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_pass() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = a.clone();
        let d = check_outputs_1d("x", &a, &b, Tolerance::TIGHT);
        assert!(d.passed(), "{:?}", d);
        assert_eq!(d.n_violations, 0);
        assert_eq!(d.max_abs_diff, 0.0);
    }

    #[test]
    fn small_noise_within_tolerance_passes() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0 + 1e-6, 2.0 - 5e-7, 3.0 + 2e-6];
        let d = check_outputs_1d("x", &a, &b, Tolerance::TIGHT);
        assert!(d.passed(), "tight tolerance should accept ~1e-6 noise: {:?}", d);
    }

    #[test]
    fn large_diff_caught() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.5, 3.0]; // +0.5 in middle
        let d = check_outputs_1d("x", &a, &b, Tolerance::LINEAR);
        assert!(!d.passed());
        assert_eq!(d.n_violations, 1);
        assert!((d.max_abs_diff - 0.5).abs() < 1e-6);
        assert_eq!(d.worst[0].0, 1);
    }

    #[test]
    fn ode_tolerance_accepts_where_tight_rejects() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0 + 5e-4, 2.0 - 3e-4];
        // TIGHT rejects
        let tight = check_outputs_1d("x", &a, &b, Tolerance::TIGHT);
        assert!(!tight.passed());
        // ODE accepts
        let ode = check_outputs_1d("x", &a, &b, Tolerance::ODE);
        assert!(ode.passed(), "ODE tolerance should accept ~1e-3 drift: {:?}", ode);
    }

    #[test]
    fn length_mismatch_is_violation() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let d = check_outputs_1d("x", &a, &b, Tolerance::TIGHT);
        assert!(!d.passed());
        assert_eq!(d.max_abs_diff, f32::INFINITY);
    }

    #[test]
    fn twod_flatten_and_check() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![1.0, 2.0], vec![3.0, 4.0 + 1e-3]];
        let tight = check_outputs_2d("y", &a, &b, Tolerance::TIGHT);
        assert!(!tight.passed());
        assert_eq!(tight.n_violations, 1);
        let loose = check_outputs_2d("y", &a, &b, Tolerance::ODE);
        assert!(loose.passed());
    }

    #[test]
    fn twod_shape_mismatch_violates() {
        let a = vec![vec![1.0, 2.0]];
        let b = vec![vec![1.0, 2.0, 3.0]];
        let d = check_outputs_2d("y", &a, &b, Tolerance::TIGHT);
        assert!(!d.passed());
    }

    #[test]
    fn per_section_tolerance_report() {
        // LN must be tight; ODE can be loose.
        let ln_a = vec![0.1, -0.1, 0.05];
        let ln_b = vec![0.1 + 2e-3, -0.1, 0.05]; // fails TIGHT
        let ode_a = vec![0.5, -0.5];
        let ode_b = vec![0.5 + 2e-4, -0.5 - 1e-4]; // within ODE tolerance
        let ln = check_outputs_1d("ln", &ln_a, &ln_b, Tolerance::TIGHT);
        let ode = check_outputs_1d("ode", &ode_a, &ode_b, Tolerance::ODE);
        let report = ParityReport {
            tier_a: "cpu".into(), tier_b: "wgpu".into(),
            sections: vec![ln, ode],
        };
        assert!(!report.passed()); // LN section fails
        assert_eq!(report.n_violations(), 1);
        // ODE section passed
        assert!(report.sections[1].passed());
    }

    #[test]
    fn accepts_mixes_abs_and_rel() {
        let t = Tolerance::new(1e-5, 1e-3);
        // Small values -> tol dominated by tol_abs
        assert!(t.accepts(0.0, 5e-6));
        assert!(!t.accepts(0.0, 1e-4));
        // Large values -> tol dominated by tol_rel
        assert!(t.accepts(1000.0, 1000.5));
        assert!(!t.accepts(1000.0, 1010.0));
    }
}
