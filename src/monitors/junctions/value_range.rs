//! J7: Value range monitoring — tensors must stay within declared bounds.
//!
//! Each component declares a range contract for a named tensor: min/max element
//! bounds, expected mean/std range, and optional row-sum-to-one for softmax rows.
//! `audit_model_state` walks a map of (label → tensor) and reports every
//! contract violation.
//!
//! This monitor provides framework + standard wave-engine contracts. Wiring into
//! forward hot paths is deliberately NOT done here — audits should run at
//! training startup and periodically via train_health, not every forward call.
//! The AGC bypass bug in the encode/relate path would have been caught by the
//! `precond_post_agc` contract.

use std::collections::HashMap;

/// Contract for one named tensor. All fields are optional — a contract may
/// constrain just a min/max, or just a norm range, or be a softmax-row check.
#[derive(Clone, Debug)]
pub struct RangeContract {
    pub name: String,
    /// Inclusive lower bound on every element (if set).
    pub elem_min: Option<f32>,
    /// Inclusive upper bound on every element (if set).
    pub elem_max: Option<f32>,
    /// Allowed range for the mean of the tensor (if set).
    pub mean_range: Option<(f32, f32)>,
    /// Allowed range for the std (population) of the tensor (if set).
    pub std_range: Option<(f32, f32)>,
    /// When true, the tensor is treated as [rows][cols] and each row must sum to 1.0 ± row_sum_tol.
    pub row_sum_to_one: bool,
    /// Tolerance for row-sum-to-one check (and for mean/std range slack).
    pub tol: f32,
}

impl RangeContract {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            elem_min: None,
            elem_max: None,
            mean_range: None,
            std_range: None,
            row_sum_to_one: false,
            tol: 1e-4,
        }
    }
    pub fn with_elem_bounds(mut self, min: f32, max: f32) -> Self {
        self.elem_min = Some(min); self.elem_max = Some(max); self
    }
    pub fn with_elem_max(mut self, max: f32) -> Self { self.elem_max = Some(max); self }
    pub fn with_mean_range(mut self, lo: f32, hi: f32) -> Self { self.mean_range = Some((lo, hi)); self }
    pub fn with_std_range(mut self, lo: f32, hi: f32) -> Self { self.std_range = Some((lo, hi)); self }
    pub fn with_softmax_rows(mut self) -> Self { self.row_sum_to_one = true; self }
    pub fn with_tol(mut self, tol: f32) -> Self { self.tol = tol; self }
}

/// One violation record.
#[derive(Debug, Clone)]
pub struct RangeViolation {
    pub name: String,
    pub kind: ViolationKind,
    pub actual: f32,
    pub bound: f32,
    pub index: Option<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub enum ViolationKind {
    ElemBelowMin,
    ElemAboveMax,
    MeanOutOfRange,
    StdOutOfRange,
    RowSumNotOne,
}

/// Registry of named contracts. Cloneable so callers can build a base set
/// once and amend per run.
#[derive(Clone, Default)]
pub struct RangeRegistry {
    contracts: HashMap<String, RangeContract>,
}

impl RangeRegistry {
    pub fn new() -> Self { Self::default() }
    pub fn register(&mut self, c: RangeContract) { self.contracts.insert(c.name.clone(), c); }
    pub fn get(&self, name: &str) -> Option<&RangeContract> { self.contracts.get(name) }
    pub fn len(&self) -> usize { self.contracts.len() }
    pub fn is_empty(&self) -> bool { self.contracts.is_empty() }
}

/// AGC ceiling formula from the engine: sqrt((pi/2) / (alpha + 4*beta)).
/// Used for precond-post-AGC and ODE-output magnitude contracts.
#[inline]
pub fn agc_ceiling(alpha: f32, beta: f32) -> f32 {
    ((std::f32::consts::PI / 2.0) / (alpha + 4.0 * beta)).sqrt()
}

/// Build the standard wave-engine registry given model hyperparameters.
/// Contracts:
///   ln_output           — zero-mean, unit-variance (post LayerNorm).
///   softmax_rows        — elements in [0, 1], rows sum to 1.
///   phase_angles        — elements in [-pi, pi].
///   ode_output_mag      — per-band magnitude ≤ AGC ceiling (elem_max on |Z|).
///   precond_post_agc    — per-band magnitude ≤ AGC ceiling. Violated by the
///                         encode/relate AGC bypass bug; audit catches it.
pub fn standard_wave_engine_registry(alpha: f32, beta: f32) -> RangeRegistry {
    let mut reg = RangeRegistry::new();
    let ceiling = agc_ceiling(alpha, beta);

    reg.register(
        RangeContract::new("ln_output")
            .with_mean_range(-0.1, 0.1)
            .with_std_range(0.5, 2.0)
            .with_tol(1e-3),
    );
    reg.register(
        RangeContract::new("softmax_rows")
            .with_elem_bounds(0.0, 1.0 + 1e-5)
            .with_softmax_rows()
            .with_tol(1e-4),
    );
    reg.register(
        RangeContract::new("phase_angles")
            .with_elem_bounds(-std::f32::consts::PI, std::f32::consts::PI),
    );
    // |Z| bound: values here are band magnitudes. ceiling with a small headroom
    // (AGC clamps asymptotically, not hard-cuts).
    reg.register(
        RangeContract::new("ode_output_mag")
            .with_elem_bounds(0.0, ceiling * 1.01),
    );
    reg.register(
        RangeContract::new("precond_post_agc")
            .with_elem_bounds(0.0, ceiling * 1.01),
    );
    reg
}

/// A tensor presented for audit. Either a flat 1D slice or a 2D [rows][cols]
/// view — softmax-row contracts only apply to 2D.
pub enum TensorView<'a> {
    OneD(&'a [f32]),
    TwoD(&'a [Vec<f32>]),
}

/// Audit a single tensor against a contract. Returns all violations found.
pub fn check_tensor(contract: &RangeContract, view: &TensorView) -> Vec<RangeViolation> {
    let mut out = Vec::new();

    let (mut n, mut sum, mut sum_sq) = (0usize, 0.0f64, 0.0f64);
    let emit_elem = |val: f32, idx: (usize, usize), out: &mut Vec<RangeViolation>| {
        if let Some(m) = contract.elem_min {
            if val < m - contract.tol {
                out.push(RangeViolation {
                    name: contract.name.clone(),
                    kind: ViolationKind::ElemBelowMin,
                    actual: val, bound: m, index: Some(idx),
                });
            }
        }
        if let Some(m) = contract.elem_max {
            if val > m + contract.tol {
                out.push(RangeViolation {
                    name: contract.name.clone(),
                    kind: ViolationKind::ElemAboveMax,
                    actual: val, bound: m, index: Some(idx),
                });
            }
        }
    };

    match view {
        TensorView::OneD(xs) => {
            for (i, &v) in xs.iter().enumerate() {
                emit_elem(v, (0, i), &mut out);
                n += 1; sum += v as f64; sum_sq += (v as f64) * (v as f64);
            }
        }
        TensorView::TwoD(rows) => {
            for (r, row) in rows.iter().enumerate() {
                for (c, &v) in row.iter().enumerate() {
                    emit_elem(v, (r, c), &mut out);
                    n += 1; sum += v as f64; sum_sq += (v as f64) * (v as f64);
                }
                if contract.row_sum_to_one {
                    let s: f32 = row.iter().sum();
                    if (s - 1.0).abs() > contract.tol {
                        out.push(RangeViolation {
                            name: contract.name.clone(),
                            kind: ViolationKind::RowSumNotOne,
                            actual: s, bound: 1.0, index: Some((r, 0)),
                        });
                    }
                }
            }
        }
    }

    if n > 0 && (contract.mean_range.is_some() || contract.std_range.is_some()) {
        let mean = (sum / n as f64) as f32;
        let var = (sum_sq / n as f64 - (sum / n as f64).powi(2)).max(0.0) as f32;
        let std = var.sqrt();
        if let Some((lo, hi)) = contract.mean_range {
            if mean < lo - contract.tol || mean > hi + contract.tol {
                out.push(RangeViolation {
                    name: contract.name.clone(),
                    kind: ViolationKind::MeanOutOfRange,
                    actual: mean, bound: if mean < lo { lo } else { hi }, index: None,
                });
            }
        }
        if let Some((lo, hi)) = contract.std_range {
            if std < lo - contract.tol || std > hi + contract.tol {
                out.push(RangeViolation {
                    name: contract.name.clone(),
                    kind: ViolationKind::StdOutOfRange,
                    actual: std, bound: if std < lo { lo } else { hi }, index: None,
                });
            }
        }
    }

    out
}

/// Full audit result.
pub struct AuditResult {
    pub n_tensors_audited: usize,
    pub violations: Vec<RangeViolation>,
}

impl AuditResult {
    pub fn passed(&self) -> bool { self.violations.is_empty() }
}

/// Audit a set of named tensors against a registry. Tensors without a matching
/// contract are skipped silently. Tensors with a contract produce zero or more
/// violations.
pub fn audit_model_state<'a>(
    registry: &RangeRegistry,
    tensors: &[(&str, TensorView<'a>)],
) -> AuditResult {
    let mut violations = Vec::new();
    let mut n_audited = 0;
    for (name, view) in tensors {
        if let Some(c) = registry.get(name) {
            n_audited += 1;
            violations.extend(check_tensor(c, view));
        }
    }
    AuditResult { n_tensors_audited: n_audited, violations }
}

/// Print audit result.
pub fn print_result(result: &AuditResult, assert_mode: bool) {
    if result.passed() {
        eprintln!("[J7] Value range audit: {} tensors, 0 violations — PASS", result.n_tensors_audited);
    } else {
        eprintln!("[J7] RANGE VIOLATIONS: {} violations in {} audited tensors",
            result.violations.len(), result.n_tensors_audited);
        let n_show = 20.min(result.violations.len());
        for v in result.violations.iter().take(n_show) {
            let idx = v.index.map(|(r, c)| format!("[{},{}]", r, c)).unwrap_or_default();
            eprintln!("  {}{}: {:?} actual={:.4e} bound={:.4e}",
                v.name, idx, v.kind, v.actual, v.bound);
        }
        if result.violations.len() > n_show {
            eprintln!("  ... and {} more", result.violations.len() - n_show);
        }
        if assert_mode {
            panic!("J7: Value range audit failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elem_bounds_in_range() {
        let c = RangeContract::new("x").with_elem_bounds(-1.0, 1.0);
        let data = vec![0.0, 0.5, -0.9, 1.0, -1.0];
        let v = check_tensor(&c, &TensorView::OneD(&data));
        assert!(v.is_empty(), "in-range values should pass, got {:?}", v);
    }

    #[test]
    fn elem_above_max_caught() {
        let c = RangeContract::new("x").with_elem_bounds(-1.0, 1.0);
        let data = vec![0.0, 0.5, 1.5];
        let v = check_tensor(&c, &TensorView::OneD(&data));
        assert_eq!(v.len(), 1);
        assert!(matches!(v[0].kind, ViolationKind::ElemAboveMax));
        assert_eq!(v[0].index, Some((0, 2)));
    }

    #[test]
    fn elem_below_min_caught() {
        let c = RangeContract::new("x").with_elem_bounds(0.0, 1.0);
        let data = vec![0.5, -0.1];
        let v = check_tensor(&c, &TensorView::OneD(&data));
        assert_eq!(v.len(), 1);
        assert!(matches!(v[0].kind, ViolationKind::ElemBelowMin));
    }

    #[test]
    fn softmax_rows_pass_when_valid() {
        let c = RangeContract::new("attn")
            .with_elem_bounds(0.0, 1.0 + 1e-5)
            .with_softmax_rows()
            .with_tol(1e-4);
        let rows = vec![
            vec![0.2, 0.3, 0.5],
            vec![0.8, 0.1, 0.1],
        ];
        let v = check_tensor(&c, &TensorView::TwoD(&rows));
        assert!(v.is_empty(), "valid softmax rows should pass, got {:?}", v);
    }

    #[test]
    fn softmax_rows_catch_bad_sum() {
        let c = RangeContract::new("attn")
            .with_elem_bounds(0.0, 1.0 + 1e-5)
            .with_softmax_rows()
            .with_tol(1e-4);
        let rows = vec![
            vec![0.2, 0.3, 0.3], // sums to 0.8 — violation
        ];
        let v = check_tensor(&c, &TensorView::TwoD(&rows));
        assert!(v.iter().any(|x| matches!(x.kind, ViolationKind::RowSumNotOne)),
            "bad row sum should be caught, got {:?}", v);
    }

    #[test]
    fn mean_std_range_pass() {
        // zero-mean unit-std-ish
        let c = RangeContract::new("ln")
            .with_mean_range(-0.1, 0.1)
            .with_std_range(0.5, 2.0)
            .with_tol(1e-3);
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let v = check_tensor(&c, &TensorView::OneD(&data));
        assert!(v.is_empty(), "zero-mean moderate-std data should pass, got {:?}", v);
    }

    #[test]
    fn mean_out_of_range_caught() {
        let c = RangeContract::new("ln").with_mean_range(-0.1, 0.1);
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let v = check_tensor(&c, &TensorView::OneD(&data));
        assert!(v.iter().any(|x| matches!(x.kind, ViolationKind::MeanOutOfRange)));
    }

    #[test]
    fn agc_ceiling_formula() {
        // With alpha=0.1, beta=0.2: sqrt(pi/2 / (0.1 + 0.8)) ≈ 1.3213
        let ceiling = agc_ceiling(0.1, 0.2);
        assert!((ceiling - 1.3213).abs() < 1e-3, "ceiling {} not matching expected", ceiling);
    }

    #[test]
    fn precond_post_agc_catches_bypass() {
        // AGC bypass => magnitudes blow past the ceiling.
        let reg = standard_wave_engine_registry(0.1, 0.2);
        let ceiling = agc_ceiling(0.1, 0.2);
        // A band that is 3x the ceiling — classic AGC-skipped signal.
        let bad = vec![ceiling * 3.0, ceiling * 0.5, ceiling * 0.9];
        let result = audit_model_state(
            &reg,
            &[("precond_post_agc", TensorView::OneD(&bad))],
        );
        print_result(&result, false);
        assert!(!result.passed(), "bypass should be caught");
        assert!(result.violations.iter().any(|v| v.name == "precond_post_agc"
            && matches!(v.kind, ViolationKind::ElemAboveMax)));
    }

    #[test]
    fn standard_registry_passes_healthy_state() {
        let reg = standard_wave_engine_registry(0.1, 0.2);
        // Synthetic healthy tensors
        let ln_row = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let softmax = vec![vec![0.25, 0.25, 0.25, 0.25]];
        let phases = vec![-1.0, 0.0, 1.0, 2.5];
        let mag = vec![0.1, 0.5, 1.0, 1.2]; // under ceiling
        let result = audit_model_state(&reg, &[
            ("ln_output", TensorView::OneD(&ln_row)),
            ("softmax_rows", TensorView::TwoD(&softmax)),
            ("phase_angles", TensorView::OneD(&phases)),
            ("precond_post_agc", TensorView::OneD(&mag)),
        ]);
        print_result(&result, false);
        assert!(result.passed(), "healthy state should pass, got {:?}", result.violations);
        assert_eq!(result.n_tensors_audited, 4);
    }

    #[test]
    fn unregistered_tensor_skipped() {
        let reg = standard_wave_engine_registry(0.1, 0.2);
        let data = vec![1e9; 4]; // would violate anything
        let result = audit_model_state(&reg, &[
            ("made_up_name", TensorView::OneD(&data)),
        ]);
        assert!(result.passed());
        assert_eq!(result.n_tensors_audited, 0);
    }
}
