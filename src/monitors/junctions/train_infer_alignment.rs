//! J8: Training/inference alignment — training loss should correlate with a
//! downstream inference metric over a rolling window.
//!
//! The monitor is a pure accumulator. The caller supplies (iter, train_loss,
//! infer_metric) tuples under a label; J8 maintains a per-label rolling window
//! and reports the Pearson correlation between loss and metric.
//!
//! Semantics by convention:
//!   - `infer_metric` should be oriented so that UP = better (accuracy, hit@k).
//!   - `train_loss` decreases as training improves.
//!   - Therefore the expected correlation is NEGATIVE (loss down ↔ metric up).
//!   - `correlation_ok(...)` returns Alignment::Ok/Weak/Inverted based on the
//!     rolling correlation vs the caller's thresholds.
//!
//! Multiple labels are supported simultaneously so a run can track e.g.
//! (L2_loss, decode_accuracy) and (L2_loss, phase_coherence) with one monitor.
//!
//! Wiring (holding out a dataset, choosing an inference mode, calling every N
//! iterations) is a separate follow-up — not part of this monitor.

use std::collections::HashMap;
use crate::common::math::pearson_correlation;

/// One datapoint recorded under a label.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub iter: usize,
    pub train_loss: f32,
    pub infer_metric: f32,
}

/// Per-label rolling sample buffer.
#[derive(Clone)]
struct Buffer {
    samples: Vec<Sample>,
    window_size: usize,
}

impl Buffer {
    fn push(&mut self, s: Sample) {
        self.samples.push(s);
        let excess = self.samples.len().saturating_sub(self.window_size);
        if excess > 0 { self.samples.drain(0..excess); }
    }
}

/// Multi-label accumulator. One monitor, many labels, each with its own window.
pub struct AlignmentTracker {
    default_window: usize,
    buffers: HashMap<String, Buffer>,
}

/// Result of a correlation check.
#[derive(Debug, Clone)]
pub struct AlignmentReport {
    pub label: String,
    pub window_len: usize,
    pub latest_train_loss: Option<f32>,
    pub latest_infer_metric: Option<f32>,
    /// Pearson r over the window. None if too few samples or zero variance.
    pub correlation: Option<f32>,
    pub verdict: Alignment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    /// Not enough samples yet or variance too low to judge.
    Undetermined,
    /// Correlation is negative and strong enough — loss and metric move together.
    Ok,
    /// Correlation is weak: monitor can no longer confirm alignment.
    Weak,
    /// Correlation has inverted sign (positive): training loss improves but
    /// inference metric stagnates or degrades. Red flag.
    Inverted,
}

impl AlignmentTracker {
    pub fn new(default_window: usize) -> Self {
        assert!(default_window >= 2, "window must be >= 2");
        Self { default_window, buffers: HashMap::new() }
    }

    /// Record a sample under `label`. The buffer for the label is created on
    /// first use with the default window size.
    pub fn record(&mut self, label: &str, sample: Sample) {
        let buf = self.buffers.entry(label.to_string()).or_insert_with(|| Buffer {
            samples: Vec::new(),
            window_size: self.default_window,
        });
        buf.push(sample);
    }

    /// Set (or override) the window size for a specific label. Existing excess
    /// samples are trimmed.
    pub fn set_window(&mut self, label: &str, window: usize) {
        assert!(window >= 2, "window must be >= 2");
        let buf = self.buffers.entry(label.to_string()).or_insert_with(|| Buffer {
            samples: Vec::new(),
            window_size: window,
        });
        buf.window_size = window;
        let excess = buf.samples.len().saturating_sub(window);
        if excess > 0 { buf.samples.drain(0..excess); }
    }

    /// Evaluate one label against the given thresholds.
    /// `weak_at`: if |r| < weak_at, verdict is Weak.
    /// `inverted_at`: if r > inverted_at (positive correlation), verdict is Inverted.
    /// Expected sign is negative (loss down ↔ metric up), so a significantly
    /// positive r is the red flag.
    pub fn report(&self, label: &str, weak_at: f32, inverted_at: f32) -> AlignmentReport {
        let buf = match self.buffers.get(label) {
            Some(b) => b,
            None => return AlignmentReport {
                label: label.to_string(),
                window_len: 0,
                latest_train_loss: None,
                latest_infer_metric: None,
                correlation: None,
                verdict: Alignment::Undetermined,
            },
        };

        let xs: Vec<f32> = buf.samples.iter().map(|s| s.train_loss).collect();
        let ys: Vec<f32> = buf.samples.iter().map(|s| s.infer_metric).collect();
        let latest = buf.samples.last();

        let correlation = pearson_correlation(&xs, &ys);
        let verdict = match correlation {
            None => Alignment::Undetermined,
            Some(r) if r > inverted_at => Alignment::Inverted,
            Some(r) if r.abs() < weak_at => Alignment::Weak,
            Some(_) => Alignment::Ok,
        };

        AlignmentReport {
            label: label.to_string(),
            window_len: buf.samples.len(),
            latest_train_loss: latest.map(|s| s.train_loss),
            latest_infer_metric: latest.map(|s| s.infer_metric),
            correlation,
            verdict,
        }
    }

    /// Report every label currently being tracked.
    pub fn report_all(&self, weak_at: f32, inverted_at: f32) -> Vec<AlignmentReport> {
        let mut labels: Vec<&String> = self.buffers.keys().collect();
        labels.sort();
        labels.into_iter().map(|l| self.report(l, weak_at, inverted_at)).collect()
    }
}

/// Print one report.
pub fn print_report(r: &AlignmentReport) {
    let corr = r.correlation.map(|v| format!("{:+.3}", v)).unwrap_or_else(|| "n/a".into());
    let tag = match r.verdict {
        Alignment::Ok => "OK",
        Alignment::Weak => "WEAK",
        Alignment::Inverted => "INVERTED",
        Alignment::Undetermined => "UNDET",
    };
    eprintln!("[J8] {:<24} window={} r={} → {}", r.label, r.window_len, corr, tag);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_when_loss_and_metric_anti_correlate() {
        let mut t = AlignmentTracker::new(10);
        // Loss goes 1.0 → 0.1, metric goes 0.1 → 1.0 (clean anti-correlation).
        for i in 0..10 {
            let loss = 1.0 - 0.09 * i as f32;
            let metric = 0.1 + 0.09 * i as f32;
            t.record("decode_acc", Sample { iter: i, train_loss: loss, infer_metric: metric });
        }
        let r = t.report("decode_acc", 0.3, 0.3);
        print_report(&r);
        assert_eq!(r.verdict, Alignment::Ok);
        assert!(r.correlation.unwrap() < -0.9);
    }

    #[test]
    fn inverted_when_loss_down_metric_also_down() {
        let mut t = AlignmentTracker::new(10);
        for i in 0..10 {
            let loss = 1.0 - 0.09 * i as f32;
            let metric = 1.0 - 0.09 * i as f32; // metric degrades — BAD
            t.record("decode_acc", Sample { iter: i, train_loss: loss, infer_metric: metric });
        }
        let r = t.report("decode_acc", 0.3, 0.3);
        print_report(&r);
        assert_eq!(r.verdict, Alignment::Inverted);
        assert!(r.correlation.unwrap() > 0.9);
    }

    #[test]
    fn weak_when_metric_is_flat_with_noise() {
        let mut t = AlignmentTracker::new(20);
        // Loss descends; metric wanders near a constant with small noise => low |r|.
        let noise = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.02, -0.01, 0.03,
                     -0.02, 0.01, -0.01, 0.02, -0.03, 0.01, 0.02, -0.02, 0.03, -0.01];
        for i in 0..20 {
            let loss = 1.0 - 0.045 * i as f32;
            let metric = 0.5 + noise[i];
            t.record("decode_acc", Sample { iter: i, train_loss: loss, infer_metric: metric });
        }
        let r = t.report("decode_acc", 0.5, 0.3);
        print_report(&r);
        assert_eq!(r.verdict, Alignment::Weak,
            "expected Weak, got {:?} (r={:?})", r.verdict, r.correlation);
    }

    #[test]
    fn undetermined_below_min_samples() {
        let mut t = AlignmentTracker::new(10);
        t.record("x", Sample { iter: 0, train_loss: 1.0, infer_metric: 0.1 });
        let r = t.report("x", 0.3, 0.3);
        assert_eq!(r.verdict, Alignment::Undetermined);
        assert!(r.correlation.is_none());
    }

    #[test]
    fn multi_label_independent_tracking() {
        let mut t = AlignmentTracker::new(10);
        for i in 0..10 {
            let loss = 1.0 - 0.09 * i as f32;
            t.record("decode_acc", Sample { iter: i, train_loss: loss, infer_metric: 0.1 + 0.09 * i as f32 });
            t.record("phase_coh", Sample { iter: i, train_loss: loss, infer_metric: 0.5 + 0.02 * i as f32 });
        }
        let r_acc = t.report("decode_acc", 0.3, 0.3);
        let r_phc = t.report("phase_coh", 0.3, 0.3);
        assert_eq!(r_acc.verdict, Alignment::Ok);
        assert_eq!(r_phc.verdict, Alignment::Ok);
        // Both anti-correlate with loss, but they don't share samples.
        let all = t.report_all(0.3, 0.3);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn rolling_window_evicts_stale_samples() {
        let mut t = AlignmentTracker::new(5);
        // Push 10 — only last 5 retained.
        for i in 0..10 {
            t.record("x", Sample { iter: i, train_loss: 1.0 - 0.1 * i as f32, infer_metric: i as f32 });
        }
        let r = t.report("x", 0.3, 0.3);
        assert_eq!(r.window_len, 5);
        // Latest iter should be 9.
        assert_eq!(r.latest_infer_metric, Some(9.0));
    }
}
