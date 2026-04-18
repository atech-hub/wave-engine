//! Self-tests for the gradient check monitor.
//! Validates the monitor catches known bugs and passes correct implementations.

#[cfg(test)]
mod tests {
    use crate::monitors::junctions::grad_check::*;

    fn simple_labels(n: usize) -> SectionLabels {
        SectionLabels::new(vec![(0, "all_params".to_string())])
    }

    /// A correct backward for f(x) = sum(x_i^2) → grad_i = 2*x_i
    #[test]
    fn test_passes_on_correct_backward() {
        let n = 10;
        let params: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let forward = |p: &[f32]| -> f64 { p.iter().map(|x| (*x as f64) * (*x as f64)).sum() };
        let backward = |p: &[f32]| -> (f32, Vec<f32>) {
            let loss: f32 = p.iter().map(|x| x * x).sum();
            let grads: Vec<f32> = p.iter().map(|x| 2.0 * x).collect();
            (loss, grads)
        };

        let result = check_gradients(
            "test_correct", forward, backward, &params,
            &simple_labels(n), GradCheckConfig { mode: CheckMode::Exhaustive, rel_tol: 0.02, ..Default::default() },
        );

        print_result(&result);
        assert!(result.passed(), "Correct backward should pass. Max err: {}", result.max_rel_err);
        assert_eq!(result.n_passed, n);
    }

    /// A buggy backward: grad_i = x_i (missing factor of 2)
    /// Monitor should catch this with ~0.5 relative error on every param.
    #[test]
    fn test_catches_known_bug() {
        let n = 10;
        let params: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let forward = |p: &[f32]| -> f64 { p.iter().map(|x| (*x as f64) * (*x as f64)).sum() };
        let buggy_backward = |p: &[f32]| -> (f32, Vec<f32>) {
            let loss: f32 = p.iter().map(|x| x * x).sum();
            let grads: Vec<f32> = p.iter().map(|x| *x).collect(); // BUG: missing 2×
            (loss, grads)
        };

        let result = check_gradients(
            "test_buggy", forward, buggy_backward, &params,
            &simple_labels(n), GradCheckConfig { mode: CheckMode::Exhaustive, ..Default::default() },
        );

        print_result(&result);
        assert!(!result.passed(), "Buggy backward should FAIL");
        assert_eq!(result.failures.len(), n, "Every param should fail");
        // Each failure should have rel_err ≈ 0.5 (analytical is half of FD)
        for f in &result.failures {
            assert!(f.rel_err > 0.3 && f.rel_err < 0.7,
                "Expected ~0.5 rel_err, got {}", f.rel_err);
        }
    }

    /// Dead gradient: backward returns all zeros.
    /// Monitor should flag all_zero_gradients in the section summary.
    #[test]
    fn test_catches_dead_gradient() {
        let n = 10;
        let params: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let forward = |p: &[f32]| -> f64 { p.iter().map(|x| (*x as f64) * (*x as f64)).sum() };
        let dead_backward = |p: &[f32]| -> (f32, Vec<f32>) {
            let loss: f32 = p.iter().map(|x| x * x).sum();
            (loss, vec![0.0; p.len()]) // All dead
        };

        let result = check_gradients(
            "test_dead", forward, dead_backward, &params,
            &simple_labels(n), GradCheckConfig { mode: CheckMode::Exhaustive, ..Default::default() },
        );

        print_result(&result);
        assert!(!result.passed(), "Dead gradient should FAIL");
        assert!(result.per_section_summary[0].all_zero_gradients,
            "Section should be flagged as all_zero_gradients");
    }

    /// Per-section reporting: two sections, one correct and one buggy.
    /// Monitor should pass the correct section and fail the buggy one.
    #[test]
    fn test_per_section_isolation() {
        // 8 params: first 4 correct, last 4 buggy
        let n = 8;
        let params: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let forward = |p: &[f32]| -> f64 { p.iter().map(|x| (*x as f64) * (*x as f64)).sum() };
        let mixed_backward = |p: &[f32]| -> (f32, Vec<f32>) {
            let loss: f32 = p.iter().map(|x| x * x).sum();
            let mut grads: Vec<f32> = p.iter().map(|x| 2.0 * x).collect();
            // Bug in second half: zero out gradients
            for i in 4..8 { grads[i] = 0.0; }
            (loss, grads)
        };

        let labels = SectionLabels::new(vec![
            (0, "correct_section".to_string()),
            (4, "buggy_section".to_string()),
        ]);

        let result = check_gradients(
            "test_mixed", forward, mixed_backward, &params,
            &labels, GradCheckConfig { mode: CheckMode::Exhaustive, rel_tol: 0.01, ..Default::default() },
        );

        print_result(&result);

        // Find section summaries
        let correct = result.per_section_summary.iter().find(|s| s.section == "correct_section").unwrap();
        let buggy = result.per_section_summary.iter().find(|s| s.section == "buggy_section").unwrap();

        assert_eq!(correct.n_passed, correct.n_checked, "Correct section should all pass");
        assert_eq!(buggy.n_passed, 0, "Buggy section should all fail");
        assert!(buggy.all_zero_gradients, "Buggy section should be flagged as dead gradient");
    }
}
