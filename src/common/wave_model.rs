//! Wave-engine model: struct definition, initialization, flatten/unflatten.
//! Extracted from main.rs to keep it focused on CLI dispatch.

use crate::model::*;
use crate::wave_attn::{WaveAttnWeights, WaveAttnHeadWeights};
use crate::wave_block::WaveBlockWeights;
use crate::wave_embed::{build_harmonic_table_with_moduli, build_harmonic_table_pythagorean, build_positional_table};
use crate::common::rng::Rng;
use crate::Dims;
use crate::common::wave_decode;

pub struct WavePacketModel {
    pub wte: Vec<Vec<f32>>,
    pub wpe: Vec<Vec<f32>>,
    pub blocks: Vec<WaveBlockWeights>,
    pub ln_f: LayerNormWeights,
    pub lm_head: Vec<Vec<f32>>,
    pub lm_down: Vec<Vec<f32>>,
    pub lm_up: Vec<Vec<f32>>,
    pub lm_rank: usize,
    pub vocab_size: usize,
    pub tied_temperature: f32,
    // Wave transduction decoder (self-contained module)
    pub wd_state: Option<crate::common::wave_decode::WaveDecodeState>,
    pub learnable_ode: bool, // true = ODE params in flatten/unflatten, false = frozen
    pub layer_scale: Vec<f32>, // per-layer residual scaling (1.0 = default, learnable when dynamic)
    pub use_layer_scale: bool, // true = layer_scale is learnable parameter
    pub lr_scale: Vec<f32>, // per-group LR multiplier [n_layers + 1 for lm_head] (training only)
    pub use_lr_scale: bool,
    pub phase_native: bool, // true = use phase coherence loss instead of lm_head
    pub output_corrector: Vec<f32>, // [n_bands] per-band phase rotation before phase comparison
}

pub fn init_linear(rng: &mut Rng, out_dim: usize, in_dim: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let limit = 1.0 / (in_dim as f32).sqrt();
    let w: Vec<Vec<f32>> = (0..out_dim)
        .map(|_| (0..in_dim).map(|_| rng.uniform(limit)).collect())
        .collect();
    let b = vec![0.0f32; out_dim];
    (w, b)
}

pub fn init_model(vocab_size: usize, seed: u64, n_layers: usize, out_proj_groups: usize, d: Dims, alpha: f32, beta: f32) -> WavePacketModel {
    let mut rng = Rng::new(seed);

    let wte = if d.pythagorean {
        build_harmonic_table_pythagorean(vocab_size, d.n_bands)
    } else {
        build_harmonic_table_with_moduli(vocab_size, d.n_bands, d.m1, d.m2)
    };
    let wpe = build_positional_table(d.block_size, d.n_bands);

    let mut blocks = Vec::new();
    for _ in 0..n_layers {
        let ln = LayerNormWeights { weight: vec![1.0f32; d.n_embd], bias: vec![0.0f32; d.n_embd] };

        let head_dim = d.n_embd / d.n_head;
        let heads: Vec<WaveAttnHeadWeights> = (0..d.n_head).map(|h| {
            let (phase_w, phase_b) = init_linear(&mut rng, 2, d.n_embd);
            let (v_w, v_b) = init_linear(&mut rng, head_dim, head_dim);
            WaveAttnHeadWeights {
                harmonic_raw: ((h + 1) as f32 * 0.5f32).ln(),
                phase_proj_w: phase_w,
                phase_proj_b: phase_b,
                v_proj_w: v_w,
                v_proj_b: v_b,
            }
        }).collect();
        let (out_w, out_b) = init_linear(&mut rng, d.n_embd, d.n_embd);
        let attn = WaveAttnWeights { heads, out_proj_w: out_w, out_proj_b: out_b };

        let gamma_raw_val = ((0.1f32).exp() - 1.0).ln();
        let kerr = KerrWeights {
            gamma_raw: vec![gamma_raw_val; d.n_bands],
            omega: (0..d.n_bands).map(|k| (k + 1) as f32 / d.n_bands as f32).collect(),
            alpha,
            beta,
            rk4_n_steps: d.rk4_steps,
            phase_correction: vec![0.0; d.n_bands],
        };
        let (sq_w, sq_b) = init_linear(&mut rng, d.maestro_dim, d.n_embd);
        let (pr_w, pr_b) = init_linear(&mut rng, d.n_embd, d.maestro_dim);
        let maestro_in = MaestroWeights {
            squeeze: LinearWeights { w: sq_w, b: sq_b },
            process_1: LinearWeights { w: pr_w, b: pr_b },
        };
        let (sq_w2, sq_b2) = init_linear(&mut rng, d.maestro_dim, d.n_embd);
        let (pr_w2, pr_b2) = init_linear(&mut rng, d.n_embd, d.maestro_dim);
        let maestro_out = MaestroWeights {
            squeeze: LinearWeights { w: sq_w2, b: sq_b2 },
            process_1: LinearWeights { w: pr_w2, b: pr_b2 },
        };
        let out_proj = if out_proj_groups <= 1 {
            let (op_w, op_b) = init_linear(&mut rng, d.n_embd, d.n_embd);
            OutProjWeights::dense(op_w, op_b)
        } else {
            let group_size = d.n_embd / out_proj_groups;
            let groups: Vec<LinearWeights> = (0..out_proj_groups).map(|_| {
                let (w, b) = init_linear(&mut rng, group_size, group_size);
                LinearWeights { w, b }
            }).collect();
            OutProjWeights::BlockDiagonal(BlockDiagonalWeights {
                groups, n_groups: out_proj_groups, group_size,
            })
        };
        let ffn = KerrDualMaestroWeights { kerr, maestro_in, maestro_out, out_proj };

        let ln_ffn = LayerNormWeights { weight: vec![1.0f32; d.n_embd], bias: vec![0.0f32; d.n_embd] };
        blocks.push(WaveBlockWeights { ln, ln_ffn, attn, ffn });
    }

    let ln_f = LayerNormWeights { weight: vec![1.0f32; d.n_embd], bias: vec![0.0f32; d.n_embd] };
    let lm_rank = d.lm_rank;
    let (lm_head, lm_down, lm_up) = if lm_rank > 0 && lm_rank < d.n_embd {
        // Low-rank factored lm_head
        let limit_down = 1.0 / (d.n_embd as f32).sqrt();
        let down: Vec<Vec<f32>> = (0..lm_rank)
            .map(|_| (0..d.n_embd).map(|_| rng.uniform(limit_down)).collect())
            .collect();
        let limit_up = 1.0 / (lm_rank as f32).sqrt();
        let up: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| (0..lm_rank).map(|_| rng.uniform(limit_up)).collect())
            .collect();
        (vec![], down, up) // lm_head empty when low-rank
    } else if d.tied {
        (wte.clone(), vec![], vec![])
    } else {
        let limit = 1.0 / (d.n_embd as f32).sqrt();
        let head: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| (0..d.n_embd).map(|_| rng.uniform(limit)).collect())
            .collect();
        (head, vec![], vec![])
    };

    // Wave transduction decoder (self-contained module)
    let wd_state = if d.wave_decode {
        if d.unfreeze_phases {
            Some(wave_decode::init_unfrozen(&wte, d.n_bands))
        } else {
            Some(wave_decode::init_frozen(&wte, d.n_bands))
        }
    } else { None };

    WavePacketModel {
        wte, wpe, blocks, ln_f, lm_head, lm_down, lm_up, lm_rank, vocab_size,
        tied_temperature: 1.0, wd_state, learnable_ode: d.learnable_ode,
        layer_scale: vec![1.0; n_layers],
        use_layer_scale: d.use_layer_scale,
        lr_scale: vec![1.0; n_layers + 1], // +1 for lm_head group
        use_lr_scale: d.use_lr_scale,
        phase_native: false,
        output_corrector: vec![0.0; d.n_bands], // 84 phase rotations, zero = transparent
    }
}

pub fn count_trainable(model: &WavePacketModel) -> usize {
    count_trainable_ex(model, false)
}

pub fn count_trainable_ex(model: &WavePacketModel, tied: bool) -> usize {
    let n_embd = model.ln_f.weight.len();
    let maestro_dim = model.blocks[0].ffn.maestro_in.squeeze.w.len();
    let mut n = 0;
    for block in &model.blocks {
        n += n_embd * 2;
        n += n_embd * 2;
        n += maestro_dim * n_embd + maestro_dim;
        n += n_embd * maestro_dim + n_embd;
        n += maestro_dim * n_embd + maestro_dim;
        n += n_embd * maestro_dim + n_embd;
        n += block.ffn.out_proj.param_count();
        if model.learnable_ode {
            n += block.ffn.kerr.gamma_raw.len(); // gamma_raw per band
            n += 1; // alpha
            n += 1; // beta
            n += block.ffn.kerr.phase_correction.len(); // corrector plate
        }
    }
    if model.use_layer_scale {
        n += model.layer_scale.len(); // 1 per layer
    }
    n += n_embd * 2;
    if model.phase_native {
        n += model.output_corrector.len(); // output corrector for phase-native decode
        // No lm_head params — phase coherence replaces the decoder
    } else if let Some(ref wds) = model.wd_state {
        n += wave_decode::param_count(wds);
    } else if model.lm_rank > 0 {
        n += model.lm_rank * n_embd;
        n += model.vocab_size * model.lm_rank;
    } else if !tied {
        n += model.vocab_size * n_embd;
    } else {
        n += 1; // tied_temperature
    }
    n
}

pub fn flatten_params(model: &WavePacketModel) -> Vec<f32> {
    flatten_params_ex(model, false)
}

pub fn flatten_params_ex(model: &WavePacketModel, tied: bool) -> Vec<f32> {
    let mut p = Vec::new();
    for block in &model.blocks {
        p.extend_from_slice(&block.ln.weight);
        p.extend_from_slice(&block.ln.bias);
        p.extend_from_slice(&block.ln_ffn.weight);
        p.extend_from_slice(&block.ln_ffn.bias);
        for row in &block.ffn.maestro_in.squeeze.w { p.extend_from_slice(row); }
        p.extend_from_slice(&block.ffn.maestro_in.squeeze.b);
        for row in &block.ffn.maestro_in.process_1.w { p.extend_from_slice(row); }
        p.extend_from_slice(&block.ffn.maestro_in.process_1.b);
        for row in &block.ffn.maestro_out.squeeze.w { p.extend_from_slice(row); }
        p.extend_from_slice(&block.ffn.maestro_out.squeeze.b);
        for row in &block.ffn.maestro_out.process_1.w { p.extend_from_slice(row); }
        p.extend_from_slice(&block.ffn.maestro_out.process_1.b);
        block.ffn.out_proj.flatten_into(&mut p);
        if model.learnable_ode {
            p.extend_from_slice(&block.ffn.kerr.gamma_raw);
            p.push(block.ffn.kerr.alpha);
            p.push(block.ffn.kerr.beta);
            p.extend_from_slice(&block.ffn.kerr.phase_correction);
        }
    }
    if model.use_layer_scale {
        p.extend_from_slice(&model.layer_scale);
    }
    p.extend_from_slice(&model.ln_f.weight);
    p.extend_from_slice(&model.ln_f.bias);
    if model.phase_native {
        p.extend_from_slice(&model.output_corrector);
        // No lm_head in param vector — phase coherence replaces the decoder
    } else if let Some(ref wds) = model.wd_state {
        p.extend_from_slice(&wave_decode::flatten_params(wds));
    } else if model.lm_rank > 0 {
        for row in &model.lm_down { p.extend_from_slice(row); }
        for row in &model.lm_up { p.extend_from_slice(row); }
    } else if !tied {
        for row in &model.lm_head { p.extend_from_slice(row); }
    } else {
        p.push(model.tied_temperature);
    }
    p
}

pub fn unflatten_params(model: &mut WavePacketModel, params: &[f32]) {
    unflatten_params_ex(model, params, false);
}

pub fn unflatten_params_ex(model: &mut WavePacketModel, params: &[f32], tied: bool) {
    let n_embd = model.ln_f.weight.len();
    let maestro_dim = model.blocks[0].ffn.maestro_in.squeeze.w.len();
    let mut idx = 0;
    for block in &mut model.blocks {
        block.ln.weight.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        block.ln.bias.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        block.ln_ffn.weight.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        block.ln_ffn.bias.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        for row in &mut block.ffn.maestro_in.squeeze.w { row.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd; }
        block.ffn.maestro_in.squeeze.b.copy_from_slice(&params[idx..idx+maestro_dim]); idx += maestro_dim;
        for row in &mut block.ffn.maestro_in.process_1.w { row.copy_from_slice(&params[idx..idx+maestro_dim]); idx += maestro_dim; }
        block.ffn.maestro_in.process_1.b.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        for row in &mut block.ffn.maestro_out.squeeze.w { row.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd; }
        block.ffn.maestro_out.squeeze.b.copy_from_slice(&params[idx..idx+maestro_dim]); idx += maestro_dim;
        for row in &mut block.ffn.maestro_out.process_1.w { row.copy_from_slice(&params[idx..idx+maestro_dim]); idx += maestro_dim; }
        block.ffn.maestro_out.process_1.b.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        block.ffn.out_proj.unflatten_from(params, &mut idx);
        if model.learnable_ode {
            let nb = block.ffn.kerr.gamma_raw.len();
            block.ffn.kerr.gamma_raw.copy_from_slice(&params[idx..idx+nb]); idx += nb;
            block.ffn.kerr.alpha = params[idx]; idx += 1;
            block.ffn.kerr.beta = params[idx]; idx += 1;
            // Clamp alpha/beta: α stays bounded, β wider now that dynamic AGC tracks safety
            block.ffn.kerr.alpha = block.ffn.kerr.alpha.clamp(0.01, 0.5);
            block.ffn.kerr.beta = block.ffn.kerr.beta.clamp(0.01, 1.0);
            // Corrector plate phase corrections
            let nc = block.ffn.kerr.phase_correction.len();
            block.ffn.kerr.phase_correction.copy_from_slice(&params[idx..idx+nc]); idx += nc;
        }
    }
    if model.use_layer_scale {
        let nl = model.layer_scale.len();
        model.layer_scale.copy_from_slice(&params[idx..idx+nl]); idx += nl;
        // Soft floor only — spring handles regulation, no gate clamp
        for s in &mut model.layer_scale { if *s < 0.0 { *s = 0.0; } }
    }
    model.ln_f.weight.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
    model.ln_f.bias.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
    if model.phase_native {
        let nc = model.output_corrector.len();
        model.output_corrector.copy_from_slice(&params[idx..idx+nc]); idx += nc;
        // No lm_head to unflatten
    } else if let Some(ref mut wds) = model.wd_state {
        let wn = wave_decode::param_count(wds);
        wave_decode::unflatten_params(wds, &params[idx..idx + wn]);
        idx += wn;
        wave_decode::refresh_cos_sin_cache(wds);
    } else if model.lm_rank > 0 {
        let rank = model.lm_rank;
        for row in &mut model.lm_down { row.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd; }
        for row in &mut model.lm_up { row.copy_from_slice(&params[idx..idx+rank]); idx += rank; }
    } else if !tied {
        for row in &mut model.lm_head { row.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd; }
    } else {
        model.tied_temperature = params[idx]; idx += 1;
    }
    assert_eq!(idx, params.len());
}
