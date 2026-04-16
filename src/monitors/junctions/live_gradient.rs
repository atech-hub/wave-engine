//! J6: Live gradient detection — every trainable parameter must see nonzero gradient.
//!
//! Tracks per-section max absolute gradient over a rolling window during training.
//! After warmup, any section with max_abs == 0.0 is structurally dead — a bug.
//! Would have caught content_proj_w within 100 iterations if it had been in the param vector.
//!
//! Cheap: one max comparison per parameter per iteration. Always on.

use std::collections::HashMap;
use super::grad_check::SectionLabels;

/// Persistent state — lives across training iterations.
pub struct GradientLiveness {
    /// Per section: max absolute gradient seen in the current window.
    section_max_abs: HashMap<String, f32>,
    /// Total iterations observed since creation or last reset.
    pub iterations_observed: usize,
    /// Window size for rolling max (resets after this many iters).
    window_size: usize,
}

/// Report from a liveness check.
pub struct LivenessReport {
    pub iterations_observed: usize,
    pub sections_checked: usize,
    pub dead_sections: Vec<String>,
    pub weakest_section: Option<(String, f32)>,
}

impl LivenessReport {
    pub fn passed(&self) -> bool { self.dead_sections.is_empty() }
}

impl GradientLiveness {
    /// Create a new liveness tracker.
    pub fn new(window_size: usize) -> Self {
        Self {
            section_max_abs: HashMap::new(),
            iterations_observed: 0,
            window_size,
        }
    }

    /// Update with gradients from one training iteration.
    /// Called every iter — must be cheap.
    pub fn update(&mut self, flat_grads: &[f32], labels: &SectionLabels) {
        self.iterations_observed += 1;

        // Reset window periodically so stale maxes don't mask dead sections
        if self.iterations_observed % self.window_size == 0 {
            self.section_max_abs.clear();
        }

        // Track max abs per section
        for (start, name) in &labels.ranges {
            let end = labels.range_for(name, flat_grads.len())
                .map(|(_, e)| e)
                .unwrap_or(flat_grads.len());
            let section_max = flat_grads[*start..end].iter()
                .map(|g| g.abs())
                .fold(0.0f32, f32::max);

            let entry = self.section_max_abs.entry(name.clone()).or_insert(0.0);
            if section_max > *entry { *entry = section_max; }
        }
    }

    /// Check for dead sections after warmup.
    pub fn report(&self, warmup_iters: usize) -> LivenessReport {
        if self.iterations_observed < warmup_iters {
            return LivenessReport {
                iterations_observed: self.iterations_observed,
                sections_checked: 0,
                dead_sections: vec![],
                weakest_section: None,
            };
        }

        let mut dead = Vec::new();
        let mut weakest: Option<(String, f32)> = None;

        for (name, &max_abs) in &self.section_max_abs {
            if max_abs == 0.0 {
                dead.push(name.clone());
            }
            match &weakest {
                None => weakest = Some((name.clone(), max_abs)),
                Some((_, w)) if max_abs < *w => weakest = Some((name.clone(), max_abs)),
                _ => {}
            }
        }

        dead.sort();

        LivenessReport {
            iterations_observed: self.iterations_observed,
            sections_checked: self.section_max_abs.len(),
            dead_sections: dead,
            weakest_section: weakest,
        }
    }
}

/// Print liveness report.
pub fn print_report(report: &LivenessReport) {
    if report.sections_checked == 0 {
        eprintln!("[J6] Liveness: {} iters observed (still in warmup)", report.iterations_observed);
        return;
    }

    let alive = report.sections_checked - report.dead_sections.len();
    if report.passed() {
        eprintln!("[J6] Liveness OK: {}/{} sections alive after {} iters",
            alive, report.sections_checked, report.iterations_observed);
        if let Some((ref name, val)) = report.weakest_section {
            eprintln!("  Weakest: {} (max_abs={:.2e})", name, val);
        }
    } else {
        eprintln!("[J6] DEAD GRADIENT DETECTED: {} dead sections after {} iters",
            report.dead_sections.len(), report.iterations_observed);
        for name in &report.dead_sections {
            eprintln!("  DEAD: {} — zero gradient across entire window", name);
        }
        if let Some((ref name, val)) = report.weakest_section {
            eprintln!("  Weakest alive: {} (max_abs={:.2e})", name, val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_labels() -> SectionLabels {
        SectionLabels::new(vec![
            (0, "section_a".to_string()),
            (4, "section_b".to_string()),
            (8, "section_c".to_string()),
        ])
    }

    #[test]
    fn test_all_alive() {
        let labels = test_labels();
        let mut liveness = GradientLiveness::new(100);

        // 10 iterations with nonzero gradients in all sections
        for _ in 0..10 {
            let grads = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
            liveness.update(&grads, &labels);
        }

        let report = liveness.report(5);
        print_report(&report);
        assert!(report.passed(), "All sections should be alive");
        assert_eq!(report.dead_sections.len(), 0);
    }

    #[test]
    fn test_dead_section() {
        let labels = test_labels();
        let mut liveness = GradientLiveness::new(100);

        // 10 iterations where section_b has zero gradient
        for _ in 0..10 {
            let grads = vec![0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.8];
            liveness.update(&grads, &labels);
        }

        let report = liveness.report(5);
        print_report(&report);
        assert!(!report.passed(), "section_b should be dead");
        assert_eq!(report.dead_sections, vec!["section_b"]);
    }

    #[test]
    fn test_warmup_skips() {
        let labels = test_labels();
        let mut liveness = GradientLiveness::new(100);

        // Only 3 iterations — below warmup of 5
        for _ in 0..3 {
            let grads = vec![0.0; 12]; // all zero
            liveness.update(&grads, &labels);
        }

        let report = liveness.report(5);
        assert!(report.passed(), "Should pass during warmup even with zero grads");
        assert_eq!(report.sections_checked, 0);
    }

    #[test]
    fn test_window_reset() {
        let labels = test_labels();
        let mut liveness = GradientLiveness::new(5);

        // 5 iters with nonzero grads
        for _ in 0..5 {
            let grads = vec![0.1; 12];
            liveness.update(&grads, &labels);
        }
        // Window resets at iter 5. Next 5 iters with section_b dead.
        for _ in 0..5 {
            let grads = vec![0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.8];
            liveness.update(&grads, &labels);
        }

        let report = liveness.report(1);
        print_report(&report);
        assert!(!report.passed(), "section_b should be dead after window reset");
    }
}
