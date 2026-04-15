//! Curriculum Transition Monitor (#8).
//!
//! Tracks loss jumps at band transitions. Maintains a ring buffer of recent
//! losses and detects when active_bands changes, emitting a CurriculumStats
//! at each transition.

/// Statistics for one curriculum stage transition.
pub struct CurriculumStats {
    pub stage: usize,
    pub active_bands: usize,
    pub loss_before: f32,
    pub loss_after: f32,
    pub loss_jump: f32,
}

/// Stateful tracker for curriculum transitions.
///
/// Call `update()` every iteration with the current loss and active band count.
/// Returns `Some(CurriculumStats)` when a transition is detected (i.e. when
/// active_bands changes) and enough post-transition data has been collected.
pub struct CurriculumTracker {
    /// Ring buffer of recent losses (last N iterations)
    recent_losses: Vec<f32>,
    /// Maximum size of the ring buffer
    buffer_size: usize,
    /// Current write position in ring buffer
    write_pos: usize,
    /// Number of losses stored so far
    count: usize,
    /// Previous active_bands value (to detect transitions)
    prev_active_bands: usize,
    /// Current stage number (increments at each transition)
    stage: usize,
    /// Average loss from the 10 iterations before transition (stored at transition)
    loss_before: Option<f32>,
    /// Losses collected after transition (up to 10)
    post_transition_losses: Vec<f32>,
    /// Whether we are currently collecting post-transition losses
    collecting_post: bool,
}

impl CurriculumTracker {
    pub fn new() -> Self {
        Self {
            recent_losses: Vec::with_capacity(10),
            buffer_size: 10,
            write_pos: 0,
            count: 0,
            prev_active_bands: 0,
            stage: 0,
            loss_before: None,
            post_transition_losses: Vec::new(),
            collecting_post: false,
        }
    }

    /// Update with current iteration data. Returns Some when a transition
    /// has been detected AND enough post-transition data is collected.
    pub fn update(&mut self, _iter: usize, loss: f32, active_bands: usize) -> Option<CurriculumStats> {
        // First call: initialize prev_active_bands
        if self.prev_active_bands == 0 {
            self.prev_active_bands = active_bands;
        }

        // Detect transition: active_bands changed
        if active_bands != self.prev_active_bands && !self.collecting_post {
            // Compute loss_before from ring buffer
            let n = self.count.min(self.buffer_size);
            if n > 0 {
                let sum: f32 = if self.count >= self.buffer_size {
                    self.recent_losses.iter().sum()
                } else {
                    self.recent_losses[..n].iter().sum()
                };
                self.loss_before = Some(sum / n as f32);
            }
            self.stage += 1;
            self.prev_active_bands = active_bands;
            self.post_transition_losses.clear();
            self.collecting_post = true;
        }

        // Collect post-transition losses
        if self.collecting_post {
            self.post_transition_losses.push(loss);
            if self.post_transition_losses.len() >= 10 {
                // Enough data — emit the transition stats
                let loss_after = self.post_transition_losses.iter().sum::<f32>()
                    / self.post_transition_losses.len() as f32;
                let loss_before = self.loss_before.unwrap_or(loss_after);
                let loss_jump = loss_after - loss_before;

                self.collecting_post = false;
                self.post_transition_losses.clear();

                // Also record these post-transition losses into the ring buffer
                // (they are valid recent losses for the next stage)
                // Push current loss into ring buffer before returning
                self.push_loss(loss);

                return Some(CurriculumStats {
                    stage: self.stage,
                    active_bands,
                    loss_before,
                    loss_after,
                    loss_jump,
                });
            }
            // Don't push to ring buffer while collecting post-transition
            return None;
        }

        // Normal: push to ring buffer
        self.push_loss(loss);
        None
    }

    fn push_loss(&mut self, loss: f32) {
        if self.recent_losses.len() < self.buffer_size {
            self.recent_losses.push(loss);
        } else {
            self.recent_losses[self.write_pos] = loss;
        }
        self.write_pos = (self.write_pos + 1) % self.buffer_size;
        self.count += 1;
    }
}

/// Serialize curriculum stats to JSONL fragment.
/// Format: "curriculum":{...}
pub fn to_json(stats: &CurriculumStats) -> String {
    format!(
        r#""curriculum":{{"stage":{},"bands":{},"loss_before":{:.4},"loss_after":{:.4},"loss_jump":{:.4}}}"#,
        stats.stage, stats.active_bands, stats.loss_before, stats.loss_after, stats.loss_jump,
    )
}
