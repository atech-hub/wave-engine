//! Generation loop using the engine's training forward pass.
//! NO reimplemented forward — calls forward_with_cache from cpu/forward.rs.
//! This guarantees inference matches training: same ODE, corrector, AGC.

use crate::common::wave_model::WavePacketModel;
use crate::common::dims::Dims;
use crate::cpu::forward::forward_with_cache;
use crate::common::rng::Rng;
use super::prompt::Vocab;

pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
}

pub struct GenerationResult {
    pub tokens: Vec<usize>,
    pub text: String,
}

pub struct TokenEvent {
    pub text: String,
    pub done: bool,
}

/// Generate all tokens (non-streaming) using engine's forward_with_cache.
pub fn generate(
    model: &WavePacketModel,
    prompt_tokens: &[usize],
    config: &GenerationConfig,
    vocab: &Vocab,
    dims: Dims,
    stencil: &crate::fft_ode::StencilFft,
) -> GenerationResult {
    let mut rng = make_rng();
    let block_size = dims.block_size;
    let mut tokens = prompt_tokens.to_vec();
    let mut generated = Vec::new();

    for _ in 0..config.max_tokens {
        let start = if tokens.len() > block_size { tokens.len() - block_size } else { 0 };
        let input = &tokens[start..];

        // SAME forward pass as training — zero reimplementation
        let cache = forward_with_cache(model, input, dims, None, None, None, Some(stencil), None, None);
        let logits = cache.logits.last().unwrap();

        let mut logits = logits.clone();
        if let Some(penalty) = config.repetition_penalty {
            if penalty != 1.0 {
                apply_repetition_penalty(&mut logits, &tokens, penalty);
            }
        }

        let token = sample_token(&logits, config, &mut rng);
        tokens.push(token);
        generated.push(token);
    }

    let text = vocab.decode(&generated);
    GenerationResult { tokens: generated, text }
}

/// Generate tokens one at a time, calling on_token for each (streaming).
pub fn generate_streaming<F>(
    model: &WavePacketModel,
    prompt_tokens: &[usize],
    config: &GenerationConfig,
    vocab: &Vocab,
    dims: Dims,
    stencil: &crate::fft_ode::StencilFft,
    mut on_token: F,
) where
    F: FnMut(TokenEvent) -> bool,
{
    let mut rng = make_rng();
    let block_size = dims.block_size;
    let mut tokens = prompt_tokens.to_vec();

    for i in 0..config.max_tokens {
        let start = if tokens.len() > block_size { tokens.len() - block_size } else { 0 };
        let input = &tokens[start..];

        let cache = forward_with_cache(model, input, dims, None, None, None, Some(stencil), None, None);
        let mut logits = cache.logits.last().unwrap().clone();

        if let Some(penalty) = config.repetition_penalty {
            if penalty != 1.0 {
                apply_repetition_penalty(&mut logits, &tokens, penalty);
            }
        }

        let token = sample_token(&logits, config, &mut rng);
        tokens.push(token);

        let text = vocab.decode(&[token]);
        let done = i + 1 >= config.max_tokens;

        if !on_token(TokenEvent { text, done }) {
            break;
        }
    }
}

fn sample_token(logits: &[f32], config: &GenerationConfig, rng: &mut Rng) -> usize {
    let temp = config.temperature.max(1e-8);
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();

    let mut candidates: Vec<usize> = (0..scaled.len()).collect();
    if let Some(k) = config.top_k {
        if k < candidates.len() {
            candidates.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
            candidates.truncate(k);
        }
    }

    let max_l = candidates.iter().map(|&i| scaled[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut exp_vals: Vec<(usize, f32)> = candidates.iter()
        .map(|&i| (i, (scaled[i] - max_l).exp()))
        .collect();

    if config.top_p < 1.0 {
        exp_vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let sum: f32 = exp_vals.iter().map(|(_, v)| v).sum();
        let threshold = config.top_p * sum;
        let mut cumsum = 0.0;
        let mut cutoff = exp_vals.len();
        for (i, (_, v)) in exp_vals.iter().enumerate() {
            cumsum += v;
            if cumsum >= threshold { cutoff = i + 1; break; }
        }
        exp_vals.truncate(cutoff);
    }

    let sum: f32 = exp_vals.iter().map(|(_, v)| v).sum();
    let mut r = rng.next_f32() * sum;
    for &(idx, val) in &exp_vals {
        r -= val;
        if r <= 0.0 { return idx; }
    }
    exp_vals.last().unwrap().0
}

fn apply_repetition_penalty(logits: &mut [f32], tokens: &[usize], penalty: f32) {
    for &tok in tokens {
        if tok < logits.len() {
            if logits[tok] > 0.0 { logits[tok] /= penalty; }
            else { logits[tok] *= penalty; }
        }
    }
}

fn make_rng() -> Rng {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64;
    Rng::new(nanos | 1)
}
