//! Encoding health monitor — real-time ODE encoding strategy sampling.
//!
//! Opt-in via --health-interval N. Computes four metrics on fixed reference
//! tokens during training: channel balance (θ/Δθ), spectral entropy,
//! coupling concentration, and band distribution shape.

use crate::common::sub_harmonic;
use crate::common::wave_analysis as wa;
use crate::cpu::forward::forward_with_cache;
use crate::WavePacketModel;
use crate::Dims;

/// Encoding health snapshot.
pub struct HealthSample {
    pub theta_disc: f32,
    pub delta_theta_disc: f32,
    pub entropy: f32,
    pub top_band: usize,
    pub concentration: f32,
    pub bimodal_score: f32,
}

/// Fixed reference tokens — hard-coded for determinism.
/// "The cat sat on the mat. A noun is a word."
const REF_TEXT: &str = "The cat sat on the mat. A noun is a word that names a person.";

/// Run one encoding health sample. Returns None if reference can't be tokenized.
pub fn sample(
    model: &WavePacketModel,
    dims: Dims,
    use_bpe: bool,
    tokenizer_path: &str,
    stencil: &crate::fft_ode::StencilFft,
    alpha: f32,
    beta: f32,
) -> Option<HealthSample> {
    // Tokenize fixed reference
    let token_ids = if use_bpe {
        let bpe = crate::bpe::BpeTokenizer::from_file(tokenizer_path);
        bpe.encode(REF_TEXT)
    } else {
        // Char-level fallback
        let chars: Vec<char> = REF_TEXT.chars().collect();
        let mut vocab: Vec<char> = chars.clone();
        vocab.sort(); vocab.dedup();
        let c2i: std::collections::HashMap<char, usize> = vocab.iter()
            .enumerate().map(|(i, &c)| (c, i)).collect();
        chars.iter().filter_map(|c| c2i.get(c).copied()).collect()
    };

    if token_ids.len() < 4 { return None; }

    // Truncate to block_size
    let max_t = dims.block_size.min(token_ids.len());
    let token_ids = &token_ids[..max_t];

    // Forward pass (CPU, no GPU — this is a diagnostic)
    let cache = forward_with_cache(model, token_ids, dims, None, None, None, Some(stencil), None);
    let hidden = &cache.post_ln_f;

    let n_bands = dims.n_bands;

    // Build simple related/random pairs from fixed positions
    // cat=1-2, mat=7-8, noun=12 (positions depend on tokenizer but approximate)
    let t = hidden.len();
    let mut related = Vec::new();
    let mut random = Vec::new();
    if t >= 10 {
        related.push((1, 7));  // cat/mat region
        related.push((3, 5));  // sat/on region
        random.push((1, t / 2));
        random.push((3, t - 2));
    } else {
        // Fallback: adjacent vs distant
        for i in (0..t.min(6)).step_by(2) {
            if i + 1 < t { related.push((i, i + 1)); }
        }
        for i in 0..t.min(3) {
            random.push((i, (i + t / 2) % t));
        }
    }

    if related.is_empty() || random.is_empty() { return None; }

    // 1. Channel balance (θ and Δθ discrimination)
    let cbd = sub_harmonic::cross_band_discrimination(hidden, &related, &random, n_bands);

    // 2. Spectral entropy
    let ims = sub_harmonic::intermod_spectrum(hidden, n_bands);

    // 3. Coupling concentration
    let cb = sub_harmonic::coupling_budget(hidden, n_bands, alpha, beta);

    // 4. Bimodal score (CV of circular variances)
    let bimodal = compute_bimodal_score(hidden, n_bands);

    Some(HealthSample {
        theta_disc: cbd.per_band_ratio,
        delta_theta_disc: cbd.diff_phase_ratio,
        entropy: ims.spectral_entropy,
        top_band: cb.most_coupled_band.0,
        concentration: cb.most_coupled_band.1,
        bimodal_score: bimodal,
    })
}

/// Bimodal score: CV of the per-band circular variances.
/// Low = continuous distribution. High = bimodal split.
fn compute_bimodal_score(hidden: &[Vec<f32>], n_bands: usize) -> f32 {
    let cvs: Vec<f32> = (0..n_bands).map(|k| {
        wa::circular_variance(
            &hidden.iter().map(|h| h[k * 2 + 1].atan2(h[k * 2])).collect::<Vec<_>>()
        )
    }).collect();

    let mean: f32 = cvs.iter().sum::<f32>() / cvs.len() as f32;
    if mean < 1e-10 { return 0.0; }
    let var: f32 = cvs.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cvs.len() as f32;
    var.sqrt() / mean // coefficient of variation
}

/// Format health sample as JSON fragment for JSONL embedding.
pub fn to_json(h: &HealthSample) -> String {
    format!(
        r#""enc_health":{{"θ_disc":{:.2},"Δθ_disc":{:.2},"entropy":{:.3},"top_band":{},"concentration":{:.1},"bimodal":{:.2}}}"#,
        h.theta_disc, h.delta_theta_disc, h.entropy, h.top_band, h.concentration, h.bimodal_score
    )
}
