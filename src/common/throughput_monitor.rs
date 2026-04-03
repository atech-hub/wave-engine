//! Throughput Monitor (#10).
//!
//! Tracks tokens/sec, iters/sec, and per-phase timing
//! (forward+backward, optimizer).

/// Throughput statistics for one iteration.
pub struct ThroughputStats {
    pub tokens_per_sec: f32,
    pub iters_per_sec: f32,
    pub forward_ms: f32,
    pub backward_ms: f32,
    pub optimizer_ms: f32,
}

/// Compute throughput stats from timing measurements.
///
/// - `batch_size`: number of sequences per batch
/// - `seq_len`: tokens per sequence
/// - `iter_elapsed_ms`: total wall time for this iteration (ms)
/// - `fwd_bwd_ms`: time for forward+backward pass (ms) — combined because batched
/// - `optimizer_ms`: time for optimizer step (ms)
pub fn compute(
    batch_size: usize,
    seq_len: usize,
    iter_elapsed_ms: f32,
    fwd_bwd_ms: f32,
    optimizer_ms: f32,
) -> ThroughputStats {
    let total_tokens = (batch_size * seq_len) as f32;
    let iter_secs = iter_elapsed_ms / 1000.0;

    // Tokens/sec: total tokens processed per wall-clock second
    let tokens_per_sec = if iter_secs > 0.0 { total_tokens / iter_secs } else { 0.0 };
    let iters_per_sec = if iter_secs > 0.0 { 1.0 / iter_secs } else { 0.0 };

    // Split fwd+bwd time: we don't have separate forward/backward timers in the
    // batched path, so report the combined time as forward_ms and leave backward_ms
    // as the remainder (iter - fwd_bwd - optimizer - reduce overhead).
    // This gives the user the actionable breakdown.
    let forward_ms = fwd_bwd_ms;
    let backward_ms = (iter_elapsed_ms - fwd_bwd_ms - optimizer_ms).max(0.0);

    ThroughputStats {
        tokens_per_sec,
        iters_per_sec,
        forward_ms,
        backward_ms,
        optimizer_ms,
    }
}

/// Serialize throughput stats to JSONL fragment.
/// Format: "throughput":{...}
pub fn to_json(stats: &ThroughputStats) -> String {
    format!(
        r#""throughput":{{"tok_s":{:.0},"iter_s":{:.1},"fwd_ms":{:.1},"bwd_ms":{:.1},"opt_ms":{:.1}}}"#,
        stats.tokens_per_sec, stats.iters_per_sec,
        stats.forward_ms, stats.backward_ms, stats.optimizer_ms,
    )
}
