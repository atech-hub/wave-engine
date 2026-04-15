//! Layer Signal Flow Monitor (#2).
//!
//! Measures per-layer contribution: input/attn/ffn/output norms, ratios,
//! and cosine similarity between input and output (direction change).

use crate::cpu::forward::ForwardCache;
use crate::Dims;

/// Per-layer signal flow statistics.
pub struct LayerFlowStats {
    pub layer: usize,
    pub input_norm: f32,
    pub attn_output_norm: f32,
    pub ffn_output_norm: f32,
    pub output_norm: f32,
    pub attn_ratio: f32,
    pub ffn_ratio: f32,
    pub residual_ratio: f32,
    pub cosine_in_out: f32,
    // Band amplitude stats (from ODE precond)
    pub band_amp_min: f32,
    pub band_amp_max: f32,
    pub band_amp_mean: f32,
    pub band_amp_std: f32,
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>()
}

/// Analyze layer signal flow from a forward cache.
///
/// For each block, uses the last sequence position to compute norms.
/// The residual connection is: output = input + scale * (attn + ffn).
/// We measure norms of each component and their ratios.
pub fn analyze_flow(cache: &ForwardCache, _dims: Dims) -> Vec<LayerFlowStats> {
    let n_layers = cache.block_caches.len();
    let mut stats = Vec::with_capacity(n_layers);

    for (layer_idx, bc) in cache.block_caches.iter().enumerate() {
        let t = bc.input.len();
        if t == 0 { continue; }
        let last = t - 1;

        let input = &bc.input[last];
        let attn_out = &bc.attn_out[last];
        let ffn_out = &bc.ffn_out[last];

        let input_norm = l2_norm(input);
        let attn_output_norm = l2_norm(attn_out);
        let ffn_output_norm = l2_norm(ffn_out);

        // Reconstruct output: input + attn + ffn (layer_scale handled at block level,
        // but we measure the raw contribution norms here).
        let n_embd = input.len();
        let output: Vec<f32> = (0..n_embd).map(|j| {
            input[j] + attn_out[j] + ffn_out[j]
        }).collect();
        let output_norm = l2_norm(&output);

        // Ratios (guard against zero output)
        let inv_out = if output_norm > 1e-12 { 1.0 / output_norm } else { 0.0 };
        let attn_ratio = attn_output_norm * inv_out;
        let ffn_ratio = ffn_output_norm * inv_out;
        let residual_ratio = input_norm * inv_out;

        // Cosine similarity between input and output direction
        let cosine_in_out = if input_norm > 1e-12 && output_norm > 1e-12 {
            dot(input, &output) / (input_norm * output_norm)
        } else {
            0.0
        };

        // Band amplitude stats from ODE precond (if available)
        let precond = if let Some(ref fc) = bc.ffn_backend_cache {
            if !fc.precond.is_empty() { Some(&fc.precond[last]) } else { None }
        } else if !bc.ffn_precond.is_empty() {
            Some(&bc.ffn_precond[last])
        } else { None };

        let (band_amp_min, band_amp_max, band_amp_mean, band_amp_std) = if let Some(p) = precond {
            let n_bands = p.len() / 2;
            let amps: Vec<f32> = (0..n_bands).map(|k| {
                (p[k*2]*p[k*2] + p[k*2+1]*p[k*2+1]).sqrt()
            }).collect();
            let min = amps.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = amps.iter().cloned().fold(0.0f32, f32::max);
            let mean = amps.iter().sum::<f32>() / n_bands as f32;
            let std = (amps.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / n_bands as f32).sqrt();
            (min, max, mean, std)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        stats.push(LayerFlowStats {
            layer: layer_idx,
            input_norm,
            attn_output_norm,
            ffn_output_norm,
            output_norm,
            attn_ratio,
            ffn_ratio,
            residual_ratio,
            cosine_in_out,
            band_amp_min, band_amp_max, band_amp_mean, band_amp_std,
        });
    }

    stats
}

/// Serialize layer flow stats to JSONL fragment.
/// Format: "layer_flow":[{...}, ...]
pub fn to_json(stats: &[LayerFlowStats]) -> String {
    if stats.is_empty() {
        return String::new();
    }

    let entries: Vec<String> = stats.iter().map(|s| {
        format!(
            r#"{{"layer":{},"in_norm":{:.3},"attn_norm":{:.3},"ffn_norm":{:.3},"out_norm":{:.3},"attn_ratio":{:.3},"ffn_ratio":{:.3},"resid_ratio":{:.3},"cos_in_out":{:.4},"band_amp_min":{:.4},"band_amp_max":{:.4},"band_amp_mean":{:.4},"band_amp_std":{:.4}}}"#,
            s.layer, s.input_norm, s.attn_output_norm, s.ffn_output_norm, s.output_norm,
            s.attn_ratio, s.ffn_ratio, s.residual_ratio, s.cosine_in_out,
            s.band_amp_min, s.band_amp_max, s.band_amp_mean, s.band_amp_std,
        )
    }).collect();

    format!(r#""layer_flow":[{}]"#, entries.join(","))
}
