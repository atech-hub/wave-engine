//! Attention Head Activity Monitor (#1).
//!
//! Computes per-head statistics from the cached attention weights:
//! entropy, max weight, top position, harmonic number, self-attention fraction.

use crate::WavePacketModel;
use crate::cpu::forward::ForwardCache;

/// Per-head attention statistics at one position (last token).
pub struct AttentionHeadStats {
    pub layer: usize,
    pub head: usize,
    pub harmonic: f32,
    pub entropy: f32,
    pub max_weight: f32,
    pub top_position: usize,
    pub self_attn_frac: f32,
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Analyze attention heads from a forward cache.
///
/// Extracts attention weights from the last query position for each head
/// in each layer. The forward cache stores att_weights[head][qi][ki].
pub fn analyze_attention(model: &WavePacketModel, cache: &ForwardCache) -> Vec<AttentionHeadStats> {
    let mut stats = Vec::new();

    for (layer_idx, bc) in cache.block_caches.iter().enumerate() {
        let n_head = bc.att_weights.len();
        if n_head == 0 { continue; }

        // Sequence length from the input (or attn weights shape)
        let t = bc.att_weights[0].len();
        if t == 0 { continue; }

        let last_pos = t - 1;

        for head in 0..n_head {
            let weights = &bc.att_weights[head][last_pos];

            // Only positions 0..=last_pos have valid weights (causal mask)
            let valid = &weights[..=last_pos];

            // Entropy: -sum(p * ln(p)) for p > 0
            let entropy = valid.iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.ln())
                .sum::<f32>();

            // Max weight and argmax
            let mut max_weight = 0.0f32;
            let mut top_position = 0usize;
            for (i, &w) in valid.iter().enumerate() {
                if w > max_weight {
                    max_weight = w;
                    top_position = i;
                }
            }

            // Self-attention fraction: weight on own position
            let self_attn_frac = valid[last_pos];

            // Harmonic number from learned parameter
            let harmonic = softplus(model.blocks[layer_idx].attn.heads[head].harmonic_raw);

            stats.push(AttentionHeadStats {
                layer: layer_idx,
                head,
                harmonic,
                entropy,
                max_weight,
                top_position,
                self_attn_frac,
            });
        }
    }

    stats
}

/// Serialize attention head stats to JSONL fragment.
/// Format: "attn_heads":[{...}, ...]
pub fn to_json(stats: &[AttentionHeadStats]) -> String {
    if stats.is_empty() {
        return String::new();
    }

    let entries: Vec<String> = stats.iter().map(|s| {
        format!(
            r#"{{"layer":{},"head":{},"harmonic":{:.3},"entropy":{:.3},"max_w":{:.4},"top_pos":{},"self_frac":{:.4}}}"#,
            s.layer, s.head, s.harmonic, s.entropy, s.max_weight, s.top_position, s.self_attn_frac,
        )
    }).collect();

    format!(r#""attn_heads":[{}]"#, entries.join(","))
}
