//! Output Distribution Monitor (#5).
//!
//! Analyzes logit distributions: entropy, margin between top-1 and top-2,
//! rank of correct token, and mode collapse detection.

/// Aggregate output distribution statistics across all positions.
pub struct OutputDistStats {
    pub avg_entropy: f32,
    pub avg_margin: f32,
    pub avg_correct_rank: f32,
    pub worst_margin: f32,
    pub worst_prompt_pos: usize,
    pub mode_collapse: bool,
}

/// Analyze output logits against target tokens.
///
/// - `logits`: logits per position, logits[pos][vocab]
/// - `targets`: correct token id for each position
///
/// Positions are matched: logits[i] is scored against targets[i].
/// Both slices must have the same length.
pub fn analyze_output(logits: &[Vec<f32>], targets: &[usize]) -> OutputDistStats {
    let n = logits.len().min(targets.len());
    if n == 0 {
        return OutputDistStats {
            avg_entropy: 0.0, avg_margin: 0.0, avg_correct_rank: 0.0,
            worst_margin: 0.0, worst_prompt_pos: 0, mode_collapse: false,
        };
    }

    let mut total_entropy = 0.0f32;
    let mut total_margin = 0.0f32;
    let mut total_correct_rank = 0.0f32;
    let mut worst_margin = f32::MAX;
    let mut worst_pos = 0usize;
    let mut first_winner: Option<usize> = None;
    let mut all_same_winner = true;

    for pos in 0..n {
        let l = &logits[pos];
        if l.is_empty() { continue; }

        // Numerically stable softmax
        let max_l = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = l.iter().map(|&x| (x - max_l).exp()).collect();
        let sum_exp = exps.iter().sum::<f32>();
        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

        // Entropy: -sum(p * ln(p))
        let entropy: f32 = exps.iter()
            .map(|&e| {
                let p = e * inv_sum;
                if p > 1e-12 { -p * p.ln() } else { 0.0 }
            })
            .sum();
        total_entropy += entropy;

        // Find top-1 and top-2 probabilities
        let mut top1_idx = 0usize;
        let mut top1_val = exps[0];
        let mut top2_val = f32::NEG_INFINITY;
        for i in 1..exps.len() {
            if exps[i] > top1_val {
                top2_val = top1_val;
                top1_val = exps[i];
                top1_idx = i;
            } else if exps[i] > top2_val {
                top2_val = exps[i];
            }
        }
        let top1_prob = top1_val * inv_sum;
        let top2_prob = if top2_val > f32::NEG_INFINITY { top2_val * inv_sum } else { 0.0 };
        let margin = top1_prob - top2_prob;

        total_margin += margin;
        if margin < worst_margin {
            worst_margin = margin;
            worst_pos = pos;
        }

        // Mode collapse: check if same token always wins
        match first_winner {
            None => { first_winner = Some(top1_idx); }
            Some(fw) => { if top1_idx != fw { all_same_winner = false; } }
        }

        // Rank of correct token (1-indexed)
        let target = targets[pos];
        if target < exps.len() {
            let target_val = exps[target];
            let rank = exps.iter().filter(|&&e| e > target_val).count() + 1;
            total_correct_rank += rank as f32;
        } else {
            total_correct_rank += exps.len() as f32; // worst rank if target OOV
        }
    }

    let n_f = n as f32;
    OutputDistStats {
        avg_entropy: total_entropy / n_f,
        avg_margin: total_margin / n_f,
        avg_correct_rank: total_correct_rank / n_f,
        worst_margin: if worst_margin == f32::MAX { 0.0 } else { worst_margin },
        worst_prompt_pos: worst_pos,
        mode_collapse: all_same_winner && n > 1,
    }
}

/// Serialize output distribution stats to JSONL fragment.
/// Format: "output_dist":{...}
pub fn to_json(stats: &OutputDistStats) -> String {
    format!(
        r#""output_dist":{{"avg_entropy":{:.3},"avg_margin":{:.4},"avg_correct_rank":{:.1},"worst_margin":{:.4},"worst_pos":{},"mode_collapse":{}}}"#,
        stats.avg_entropy, stats.avg_margin, stats.avg_correct_rank,
        stats.worst_margin, stats.worst_prompt_pos, stats.mode_collapse,
    )
}
