//! Embedding Space Monitor (#4).
//!
//! Analyzes token separation in embedding space: pairwise distances,
//! per-band utilization (phase variance), and effective dimensionality.
//! Medium cost — samples 50 random pairs for distance, but scans all
//! vocab tokens for band utilization.

use crate::WavePacketModel;

/// Embedding space statistics.
pub struct EmbeddingSpaceStats {
    pub avg_inter_token_distance: f32,
    pub min_token_distance: f32,
    pub min_pair: (usize, usize),
    pub band_utilization_mean: f32,
    pub band_utilization_min: f32,
    pub worst_band: usize,
    pub effective_dimensionality: usize,
}

/// Analyze the embedding table for token separation and band utilization.
///
/// Distance metric: dot-product distance = ||a||² + ||b||² - 2*dot(a,b).
/// Band utilization: variance of (r,s) values across all vocab tokens per band.
/// Effective dimensionality: bands with utilization > 0.5 * mean.
///
/// Samples 50 random pairs for distance (deterministic seed from vocab_size).
pub fn analyze_embeddings(model: &WavePacketModel) -> EmbeddingSpaceStats {
    let wte = &model.wte;
    let vocab_size = wte.len();
    if vocab_size < 2 {
        return EmbeddingSpaceStats {
            avg_inter_token_distance: 0.0,
            min_token_distance: 0.0,
            min_pair: (0, 0),
            band_utilization_mean: 0.0,
            band_utilization_min: 0.0,
            worst_band: 0,
            effective_dimensionality: 0,
        };
    }
    let n_embd = wte[0].len();
    let n_bands = n_embd / 2;

    // --- Pairwise distance (sample 50 pairs) ---
    let n_pairs = 50usize.min(vocab_size * (vocab_size - 1) / 2);

    // Precompute self-dots for sampled tokens
    // Simple deterministic pair selection: stride through all pairs
    let total_pairs = vocab_size * (vocab_size - 1) / 2;
    let stride = if total_pairs > n_pairs { total_pairs / n_pairs } else { 1 };

    let mut sum_dist = 0.0f32;
    let mut min_dist = f32::MAX;
    let mut min_pair = (0usize, 0usize);
    let mut count = 0usize;

    // Iterate through pairs with stride
    let mut pair_idx = 0usize;
    'outer: for i in 0..vocab_size {
        for j in (i + 1)..vocab_size {
            if pair_idx % stride == 0 {
                let self_i: f32 = wte[i].iter().map(|x| x * x).sum();
                let self_j: f32 = wte[j].iter().map(|x| x * x).sum();
                let cross: f32 = wte[i].iter().zip(wte[j].iter()).map(|(&a, &b)| a * b).sum();
                let dist = (self_i + self_j - 2.0 * cross).max(0.0).sqrt();

                sum_dist += dist;
                if dist < min_dist {
                    min_dist = dist;
                    min_pair = (i, j);
                }
                count += 1;
                if count >= n_pairs { break 'outer; }
            }
            pair_idx += 1;
        }
    }

    let avg_inter_token_distance = if count > 0 { sum_dist / count as f32 } else { 0.0 };
    if min_dist == f32::MAX { min_dist = 0.0; }

    // --- Band utilization: variance of (r,s) across all vocab tokens per band ---
    let mut band_util = vec![0.0f32; n_bands];

    for k in 0..n_bands {
        // Compute mean of r and s across all tokens
        let mut mean_r = 0.0f32;
        let mut mean_s = 0.0f32;
        for tok in 0..vocab_size {
            mean_r += wte[tok][2 * k];
            mean_s += wte[tok][2 * k + 1];
        }
        mean_r /= vocab_size as f32;
        mean_s /= vocab_size as f32;

        // Compute variance (sum of variances of r and s components)
        let mut var = 0.0f32;
        for tok in 0..vocab_size {
            let dr = wte[tok][2 * k] - mean_r;
            let ds = wte[tok][2 * k + 1] - mean_s;
            var += dr * dr + ds * ds;
        }
        band_util[k] = var / vocab_size as f32;
    }

    let band_utilization_mean = if n_bands > 0 {
        band_util.iter().sum::<f32>() / n_bands as f32
    } else { 0.0 };

    let mut band_utilization_min = f32::MAX;
    let mut worst_band = 0usize;
    for (k, &u) in band_util.iter().enumerate() {
        if u < band_utilization_min {
            band_utilization_min = u;
            worst_band = k;
        }
    }
    if band_utilization_min == f32::MAX { band_utilization_min = 0.0; }

    // Effective dimensionality: bands with utilization > 0.5 * mean
    let threshold = 0.5 * band_utilization_mean;
    let effective_dimensionality = band_util.iter().filter(|&&u| u > threshold).count();

    EmbeddingSpaceStats {
        avg_inter_token_distance,
        min_token_distance: min_dist,
        min_pair,
        band_utilization_mean,
        band_utilization_min,
        worst_band,
        effective_dimensionality,
    }
}

/// Serialize embedding space stats to JSONL fragment.
/// Format: "embedding_space":{...}
pub fn to_json(stats: &EmbeddingSpaceStats) -> String {
    format!(
        r#""embedding_space":{{"avg_dist":{:.3},"min_dist":{:.3},"min_pair":[{},{}],"band_util_mean":{:.4},"band_util_min":{:.4},"worst_band":{},"effective_dim":{}}}"#,
        stats.avg_inter_token_distance, stats.min_token_distance,
        stats.min_pair.0, stats.min_pair.1,
        stats.band_utilization_mean, stats.band_utilization_min,
        stats.worst_band, stats.effective_dimensionality,
    )
}
