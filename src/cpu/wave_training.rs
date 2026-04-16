//! Wave-space training from KWDS datasets (L2 loss).
//! Extracted from main.rs cmd_train_waves handler.

use crate::common::wave_model::*;
use crate::common::dims::Dims;
use crate::common::fft_ode;
use crate::common::kwds;
use crate::cpu::train;

pub struct WaveTrainConfig {
    pub kwds_path: String,
    pub n_layers: usize,
    pub n_head: usize,
    pub n_bands: usize, // from KWDS header, overrides CLI
    pub out_proj_groups: usize,
    pub vocab: usize,
    pub alpha: f32,
    pub beta: f32,
    pub iters: usize,
    pub lr: f32,
    pub seq: usize,
    pub checkpoint_name: String,
    pub resume: Option<String>,
}

pub fn run(config: WaveTrainConfig) {
    let mut f = std::fs::File::open(&config.kwds_path).expect("Cannot open KWDS file");
    let header = kwds::read_header(&mut f).unwrap();
    let n_bands = header.n_bands as usize;
    let n_positions = header.n_positions as usize;
    println!("Training from KWDS: {} positions, {} bands, {:.1} MB",
        n_positions, n_bands, header.file_size() as f64 / (1024.0 * 1024.0));

    let dims = Dims::from_cli(n_bands, config.n_head, 16, 128, 16);
    let mut start_iter = 0usize;
    let mut model = init_model(config.vocab, 42, config.n_layers, config.out_proj_groups, dims, config.alpha, config.beta);
    model.phase_native = true;
    model.output_corrector = vec![0.0; n_bands];
    model.learnable_ode = true;

    if let Some(ref ckpt) = config.resume {
        let (params, _ck_vocab, ck_iter, _, _, _, _, _, _, _, _) = crate::wave_checkpoint::load_checkpoint(ckpt);
        let ext_count = count_trainable_ex(&model, false);
        if params.len() == ext_count {
            unflatten_params_ex(&mut model, &params, false);
            start_iter = ck_iter;
            println!("  Resumed from {} at iter {}", ckpt, ck_iter);
        } else {
            eprintln!("  WARNING: param count mismatch ({} vs {}), starting fresh", params.len(), ext_count);
        }
    }

    crate::ffn_backend::init_agc(config.alpha, config.beta);
    let stencil = fft_ode::StencilFft::new(n_bands);
    let mut rng = crate::rng::Rng::new(1337);
    let n_trainable = count_trainable_ex(&model, false);
    println!("  Model: {}L, {}bands, {} trainable params", config.n_layers, n_bands, n_trainable);

    let mut adam_m = vec![0.0f32; n_trainable];
    let mut adam_v = vec![0.0f32; n_trainable];
    let mut adam_t = 0u64;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let adam_eps = 1e-8f32;

    let total_iters = start_iter + config.iters;
    println!("  Training for {} iters ({}→{}), seq_len={}, lr={}", config.iters, start_iter, total_iters, config.seq, config.lr);
    let mut best_loss = f32::MAX;
    let t0 = std::time::Instant::now();

    for iter in start_iter..total_iters {
        let max_start = n_positions.saturating_sub(config.seq + 1);
        let start = (rng.next_u64() as usize) % max_start.max(1);
        let window_len = config.seq.min(n_positions - start - 1);
        let inputs = kwds::read_input_window(&mut f, &header, start as u64, window_len).unwrap();
        let targets = kwds::read_target_window(&mut f, &header, start as u64, window_len).unwrap();
        let cache = crate::cpu::forward::forward_with_cache_from_waves(&model, &inputs, dims, Some(&stencil));
        let (loss, grads) = crate::cpu::model_backward::backward_wave(&model, &cache, &targets, dims);

        let cos_sim = {
            let mut sum = 0.0f32;
            let ct = cache.post_ln_f.len().min(targets.len());
            for pos in 0..ct {
                let pred = &cache.post_ln_f[pos];
                let tgt = &targets[pos];
                let dot: f32 = pred.iter().zip(tgt.iter()).map(|(&a, &b)| a * b).sum();
                let np: f32 = pred.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nt: f32 = tgt.iter().map(|x| x * x).sum::<f32>().sqrt();
                if np > 1e-8 && nt > 1e-8 { sum += dot / (np * nt); }
            }
            sum / ct.max(1) as f32
        };

        if loss < best_loss { best_loss = loss; }
        let flat_grads = crate::cpu::model_backward::flatten_grads_ex(&grads, false);
        adam_t += 1;
        let mut params = flatten_params_ex(&model, false);
        for i in 0..params.len() {
            adam_m[i] = beta1 * adam_m[i] + (1.0 - beta1) * flat_grads[i];
            adam_v[i] = beta2 * adam_v[i] + (1.0 - beta2) * flat_grads[i] * flat_grads[i];
            let m_hat = adam_m[i] / (1.0 - beta1.powi(adam_t as i32));
            let v_hat = adam_v[i] / (1.0 - beta2.powi(adam_t as i32));
            params[i] -= config.lr * m_hat / (v_hat.sqrt() + adam_eps);
        }
        unflatten_params_ex(&mut model, &params, false);

        if iter % 100 == 0 || iter == total_iters - 1 {
            let elapsed = t0.elapsed().as_millis();
            let ms_per = if iter > start_iter { elapsed / (iter - start_iter) as u128 } else { 0 };
            println!("  iter {:6}  l2_loss {:.6}  cos_sim {:.4}  best_l2 {:.6}  {}ms/iter",
                iter, loss, cos_sim, best_loss, ms_per);
        }
    }

    println!("\n=== Wave Training Complete ===");
    println!("  Best L2 loss: {:.6}", best_loss);

    let final_params = flatten_params_ex(&model, false);
    let n_params = final_params.len();
    let dummy_adam = train::Adam::new(config.lr, n_params);
    crate::wave_checkpoint::save_checkpoint(
        &final_params, config.vocab, config.n_layers, config.out_proj_groups,
        total_iters, config.lr, &dummy_adam, rng.state(), &config.checkpoint_name, dims,
    );
    println!("  Saved to: {}", config.checkpoint_name);
}
