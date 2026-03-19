//! Pipeline monitor — per-section timing for GPU/CPU visibility.
//! Zero overhead when disabled. Enable with --monitor flag.

use std::time::{Duration, Instant};
use std::collections::HashMap;

pub struct PipelineMonitor {
    timers: HashMap<&'static str, (Duration, usize)>,
    gpu_dispatches: usize,
    enabled: bool,
}

impl PipelineMonitor {
    pub fn new(enabled: bool) -> Self {
        Self { timers: HashMap::new(), gpu_dispatches: 0, enabled }
    }

    #[inline(always)]
    pub fn enabled(&self) -> bool { self.enabled }

    #[inline(always)]
    pub fn start(&self) -> Instant {
        Instant::now()
    }

    #[inline(always)]
    pub fn record(&mut self, name: &'static str, start: Instant) {
        if !self.enabled { return; }
        let entry = self.timers.entry(name).or_insert((Duration::ZERO, 0));
        entry.0 += start.elapsed();
        entry.1 += 1;
    }

    pub fn gpu_dispatch(&mut self) {
        if !self.enabled { return; }
        self.gpu_dispatches += 1;
    }

    pub fn report(&self, n_iters: usize) {
        if !self.enabled { return; }

        // Sort by total time descending
        let mut entries: Vec<(&str, Duration, usize)> = self.timers.iter()
            .map(|(&name, &(dur, count))| (name, dur, count))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));

        let total: Duration = entries.iter().map(|(_, d, _)| *d).sum();
        let total_ms = total.as_secs_f64() * 1000.0;

        println!("  Pipeline breakdown (total over {n_iters} iters, {total_ms:.0}ms total):");
        for (name, dur, count) in &entries {
            let ms = dur.as_secs_f64() * 1000.0;
            let pct = if total_ms > 0.0 { ms / total_ms * 100.0 } else { 0.0 };
            let per_call = if *count > 0 { ms / *count as f64 } else { 0.0 };
            println!("    {:<25} {:>8.1}ms ({:>5.1}%)  [{:>5} calls, {:.2}ms/call]",
                name, ms, pct, count, per_call);
        }
        if self.gpu_dispatches > 0 {
            println!("  GPU dispatches: {} ({:.1}/iter)",
                self.gpu_dispatches, self.gpu_dispatches as f64 / n_iters as f64);
        }
    }

    pub fn reset(&mut self) {
        self.timers.clear();
        self.gpu_dispatches = 0;
    }
}
