//! Checkpoint Drift Monitor (#9).
//!
//! Tracks parameter change between checkpoint saves. Stores the previous
//! checkpoint's flattened params and computes L2 distance on each new save.

/// Drift statistics between two consecutive checkpoints.
pub struct CheckpointDrift {
    pub total_drift: f32,
    pub relative_drift: f32,
    pub per_layer_drift: Vec<f32>,
    pub ode_drift: f32,
}

/// Stateful tracker that stores previous checkpoint params.
///
/// Call `measure()` at each checkpoint save with the current flattened params.
/// Returns `None` on the first call (no previous to compare against).
pub struct CheckpointTracker {
    prev_params: Option<Vec<f32>>,
    /// Number of layers — used to estimate per-layer param boundaries
    n_layers: usize,
}

impl CheckpointTracker {
    pub fn new(n_layers: usize) -> Self {
        Self {
            prev_params: None,
            n_layers,
        }
    }

    /// Measure drift from the previous checkpoint.
    ///
    /// `current_params`: flattened parameter vector (from flatten_params_ex).
    ///
    /// Returns None on the first call, Some(drift) on subsequent calls.
    /// Stores a clone of current_params for the next comparison.
    pub fn measure(&mut self, current_params: &[f32]) -> Option<CheckpointDrift> {
        let result = if let Some(ref prev) = self.prev_params {
            if prev.len() != current_params.len() {
                // Param count changed (architecture change) — reset
                None
            } else {
                Some(compute_drift(prev, current_params, self.n_layers))
            }
        } else {
            None
        };

        self.prev_params = Some(current_params.to_vec());
        result
    }
}

/// Compute drift between two parameter vectors.
fn compute_drift(prev: &[f32], current: &[f32], n_layers: usize) -> CheckpointDrift {
    let n = prev.len();

    // Total L2 drift
    let total_drift: f32 = prev.iter().zip(current.iter())
        .map(|(&p, &c)| (c - p) * (c - p))
        .sum::<f32>()
        .sqrt();

    // Previous norm for relative drift
    let prev_norm: f32 = prev.iter().map(|&p| p * p).sum::<f32>().sqrt();
    let relative_drift = if prev_norm > 1e-12 { total_drift / prev_norm } else { 0.0 };

    // Per-layer drift: split params evenly across layers (approximate)
    // The actual layout is per-block, but equal splitting gives a useful signal.
    let per_layer_drift = if n_layers > 0 {
        let chunk_size = n / n_layers;
        (0..n_layers).map(|l| {
            let start = l * chunk_size;
            let end = if l == n_layers - 1 { n } else { (l + 1) * chunk_size };
            prev[start..end].iter().zip(current[start..end].iter())
                .map(|(&p, &c)| (c - p) * (c - p))
                .sum::<f32>()
                .sqrt()
        }).collect()
    } else {
        vec![total_drift]
    };

    // ODE drift: alpha/beta/gamma are small scalars near the end of each layer's block.
    // Rather than parsing exact offsets (fragile), we estimate ODE drift as the drift
    // in the last 10% of each layer's chunk (where ODE params typically live).
    let ode_drift = if n_layers > 0 {
        let chunk_size = n / n_layers;
        let ode_frac = chunk_size / 10; // last 10% of each layer
        let mut ode_sq = 0.0f32;
        for l in 0..n_layers {
            let ode_start = if l == n_layers - 1 { n - ode_frac } else { (l + 1) * chunk_size - ode_frac };
            let ode_end = if l == n_layers - 1 { n } else { (l + 1) * chunk_size };
            for i in ode_start..ode_end {
                let d = current[i] - prev[i];
                ode_sq += d * d;
            }
        }
        ode_sq.sqrt()
    } else {
        0.0
    };

    CheckpointDrift {
        total_drift,
        relative_drift,
        per_layer_drift,
        ode_drift,
    }
}

/// Serialize checkpoint drift to JSONL fragment.
/// Format: "drift":{...}
pub fn to_json(drift: &CheckpointDrift) -> String {
    let layer_strs: Vec<String> = drift.per_layer_drift.iter()
        .map(|d| format!("{:.4}", d)).collect();

    format!(
        r#""drift":{{"total":{:.4},"relative":{:.6},"per_layer":[{}],"ode":{:.4}}}"#,
        drift.total_drift, drift.relative_drift,
        layer_strs.join(","), drift.ode_drift,
    )
}
