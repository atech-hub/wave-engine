//! Wave-space training from KWDS datasets.
//!
//! Consumes the same `TrainConfig` as token training and reuses every shared
//! subsystem — Adam (with checkpoint state), Dims builder (split-band,
//! pathway flags, learnable attention), NaN guard, stall detector, JSONL
//! telemetry, periodic checkpoint saves. The only differences from the
//! token-training loop are:
//!
//! * Input batches come from a KWDS file (per-position wave states) instead
//!   of a tokenised text file.
//! * Forward routes through `forward_with_cache_from_waves` (same FfnConfig
//!   plumbing as token forward — split-band, ODE pathway, attention pathway
//!   all honoured).
//! * Backward routes through `backward_wave`, which computes L2 loss on the
//!   ODE output states against the KWDS target slices.
//!
//! Historical note: the pre-convergence version of this file (126 lines)
//! had a hand-rolled Adam that lost state on checkpoint save, no NaN guard,
//! no logging, and built Dims without the split-band / pathway / learnable
//! flags. The 12/109 arithmetic result was trained against that reduced
//! pipeline. This version closes that gap.
//!
//! Not yet shared with the token loop (follow-ups):
//! * GPU backends (`--gpu`, `--candle`) — wave-side wgpu/Candle forward
//!   functions are pending.
//! * Full 17-monitor suite — several monitors read token targets.
//! * Curriculum band masking — loss is continuous, not categorical; masking
//!   semantics need a separate decision.
//! * Galaxy scan on best checkpoint at end — independent of training path.

use std::fs::File;
use std::io::BufWriter;

use crate::common::wave_model::*;
use crate::common::dims::Dims;
use crate::common::fft_ode;
use crate::common::kwds;
use crate::cpu::train::{Adam, TrainConfig, clip_grad_norm};

pub fn run(config: &TrainConfig) {
    // ── Open KWDS, read header ───────────────────────────────────
    let mut kwds_file = File::open(&config.data_path)
        .expect("Cannot open KWDS file");
    let header = kwds::read_header(&mut kwds_file).expect("malformed KWDS header");
    let n_bands_file = header.n_bands as usize;
    let n_positions = header.n_positions as usize;

    if n_bands_file != config.n_bands {
        eprintln!(
            "  WARNING: KWDS n_bands={} differs from --n-bands={}; using KWDS value",
            n_bands_file, config.n_bands,
        );
    }
    let n_bands = n_bands_file;
    let n_embd = n_bands * 2;

    println!("wave-engine v0.1.0\n");
    println!("Wave-space training from KWDS: {} positions, {} bands, {:.1} MB",
        n_positions, n_bands, header.file_size() as f64 / (1024.0 * 1024.0));

    // ── Build Dims from TrainConfig ──────────────────────────────
    // Every flag token training honours, wave training honours too.
    let freeze_ode = config.freeze_ode;
    let use_corrector = config.corrector.is_active() && !freeze_ode;
    let dims = Dims::from_cli(
        n_bands, config.n_head, config.maestro_dim,
        crate::BLOCK_SIZE, crate::RK4_STEPS,
    )
        .with_moduli(config.m1, config.m2)
        .with_tied(config.tied)
        .with_lm_rank(config.lm_rank)
        .with_wave_decode(config.wave_decode)
        .with_unfreeze_phases(config.unfreeze_phases)
        .with_learnable_ode(!freeze_ode)
        .with_corrector(use_corrector)
        .with_layer_scale(config.layer_scale.is_active())
        .with_lr_scale(config.lr_scale.is_active())
        .with_pythagorean(config.pythagorean)
        .with_rk4_weights(config.rk4_weights.is_active())
        .with_dyn_harmonics(config.harmonics.is_active())
        .with_split_band(config.split_band)
        .with_ode_pathway(config.ode_pathway)
        .with_attention_pathway(config.attention_pathway)
        .with_learnable_attn(config.learnable_attn);

    // ── Model init / resume ──────────────────────────────────────
    let vocab_size = config.n_bands.max(1); // wave training doesn't use vocab for decoding; any >0 is fine
    let mut model;
    let mut start_iter = 0usize;
    let mut optimizer;
    let mut rng = crate::rng::Rng::new(1337);

    model = init_model(vocab_size, 42, config.n_layers, config.out_proj_groups, dims, config.alpha, config.beta);
    model.phase_native = config.phase_native;
    if config.phase_native {
        model.output_corrector = vec![0.0; n_bands];
    }

    if let Some(ref ckpt) = config.resume_path {
        println!("Resuming from checkpoint: {ckpt}");
        let (params, _ck_vocab, ck_iter, _ck_lr, ck_rng, adam_t, adam_m, adam_v, _ck_groups, _ck_flags, _ck_chi)
            = crate::wave_checkpoint::load_checkpoint(ckpt);
        let ext_count = count_trainable_ex(&model, config.tied);
        if params.len() == ext_count {
            unflatten_params_ex(&mut model, &params, config.tied);
            println!("  Loaded {} params", params.len());
        } else {
            eprintln!("  WARNING: param count {} != expected {} — starting fresh", params.len(), ext_count);
        }
        start_iter = ck_iter;
        if adam_m.len() == count_trainable_ex(&model, config.tied) {
            optimizer = Adam::from_checkpoint(config.lr, adam_t, adam_m, adam_v);
            println!("  Resumed Adam state (t={})", adam_t);
        } else {
            optimizer = Adam::new(config.lr, count_trainable_ex(&model, config.tied));
            println!("  Adam state size mismatch — starting fresh optimizer");
        }
        rng = crate::rng::Rng::from_state(ck_rng);
    } else {
        optimizer = Adam::new(config.lr, count_trainable_ex(&model, config.tied));
    }

    let n_trainable = count_trainable_ex(&model, config.tied);
    println!("  Model: {}L, {}bands, {} vocab (unused for wave), {} trainable params",
        config.n_layers, n_bands, vocab_size, n_trainable);
    println!("  Dims: split_band={} ode_pathway={} attention_pathway={} learnable_attn={} corrector={}",
        dims.split_band, dims.ode_pathway, dims.attention_pathway, dims.learnable_attn, dims.use_corrector);

    // ── Runtime resources ────────────────────────────────────────
    crate::ffn_backend::init_agc(config.alpha, config.beta);
    let stencil = fft_ode::StencilFft::new(n_bands);
    let seq_len = config.seq_len;
    let total_iters = start_iter + config.n_iters;
    let lr = config.lr;
    let warmup_iters = 100usize;
    let min_lr_ratio = 0.1f32;

    // ── JSONL telemetry (same format as token loop) ──────────────
    let log_name = config.log_name.clone().unwrap_or_else(|| {
        let stem = std::path::Path::new(&config.checkpoint_name)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "wave".to_string());
        let parent = std::path::Path::new(&config.checkpoint_name).parent()
            .filter(|p| !p.as_os_str().is_empty());
        let name = format!("training_log_{}.jsonl", stem);
        match parent {
            Some(p) => p.join(name).to_string_lossy().into_owned(),
            None => name,
        }
    });
    let log_file = File::create(&log_name).expect("cannot create log");
    println!("  Telemetry: {log_name}");
    let mut log_writer = BufWriter::new(log_file);

    // ── Stall detector state ─────────────────────────────────────
    let mut best_loss = f32::MAX;
    let mut best_iter = start_iter;
    let mut loss_at_500 = f32::MAX;
    let mut loss_at_2000 = f32::MAX;
    let mut nan_skip_count = 0usize;

    println!("\nTraining for {} iterations (seq={}, lr={})",
        config.n_iters, seq_len, lr);
    println!("{:>6} {:>10} {:>10}  lr       gnorm    cos_sim  best_l2", "Iter", "L2_loss", "Time");
    println!("{}", "-".repeat(72));

    let train_start = std::time::Instant::now();

    // ── Main loop ────────────────────────────────────────────────
    for iter in start_iter..total_iters {
        let iter_start = std::time::Instant::now();

        // Cosine LR schedule with warmup (same as token loop).
        let current_lr = if iter < warmup_iters {
            lr * (iter as f32 + 1.0) / warmup_iters as f32
        } else {
            let progress = (iter - warmup_iters) as f32 / (total_iters - warmup_iters).max(1) as f32;
            let min_lr = lr * min_lr_ratio;
            min_lr + 0.5 * (lr - min_lr) * (1.0 + (progress * std::f32::consts::PI).cos())
        };
        optimizer.lr = current_lr;

        // Random window into KWDS.
        let max_start = n_positions.saturating_sub(seq_len + 1).max(1);
        let start = (rng.next_u64() as usize) % max_start;
        let window_len = seq_len.min(n_positions - start - 1);

        let inputs = kwds::read_input_window(&mut kwds_file, &header, start as u64, window_len)
            .expect("KWDS input read failed");
        let targets = kwds::read_target_window(&mut kwds_file, &header, start as u64, window_len)
            .expect("KWDS target read failed");

        let cache = crate::cpu::forward::forward_with_cache_from_waves(
            &model, &inputs, dims, Some(&stencil),
        );
        let (loss, grads) = crate::cpu::model_backward::backward_wave(
            &model, &cache, &targets, dims,
        );

        // NaN guard — same policy as token loop: skip optimizer step on NaN.
        if !loss.is_finite() {
            nan_skip_count += 1;
            eprintln!("  [NaN skip] iter {iter} (total skips: {nan_skip_count})");
            continue;
        }

        // Post-forward cosine similarity (wave-specific diagnostic).
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

        if loss < best_loss {
            best_loss = loss;
            best_iter = iter;
        }

        // Optimizer step via shared Adam (same code path as token loop).
        let mut flat_grads = crate::cpu::model_backward::flatten_grads_ex(&grads, config.tied);
        let grad_norm: f32 = flat_grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        clip_grad_norm(&mut flat_grads, 1.0);
        let mut params = flatten_params_ex(&model, config.tied);
        optimizer.step(&mut params, &flat_grads);
        unflatten_params_ex(&mut model, &params, config.tied);

        // Console + JSONL every 100 iters (plus first + last).
        let is_log_iter = iter % 100 == 0 || iter == total_iters - 1 || iter == start_iter;
        if is_log_iter {
            let elapsed = iter_start.elapsed();
            println!(
                "{:>6} {:>10.4} {:>10.1?}  lr={:.6}  gnorm={:.2}  cos={:.3}  best={:.4}",
                iter, loss, elapsed, current_lr, grad_norm, cos_sim, best_loss,
            );
            use std::io::Write;
            let _ = writeln!(
                log_writer,
                r#"{{"iter":{},"loss":{:.4},"lr":{:.6},"gnorm":{:.4},"cos_sim":{:.4},"nan_skips":{},"loss_mode":"wave_l2"}}"#,
                iter, loss, current_lr, grad_norm, cos_sim, nan_skip_count,
            );
        }

        // Stall detector — same thresholds as token loop.
        if iter == start_iter + 500 { loss_at_500 = loss; }
        if iter == start_iter + 2000 {
            loss_at_2000 = loss;
            if loss_at_2000 / loss_at_500.max(1e-6) > 0.97 {
                eprintln!("  [stall] iter 2000 loss {:.4} vs iter 500 {:.4} — no meaningful decrease", loss_at_2000, loss_at_500);
            }
        }
        if iter == start_iter + 5000 {
            let ratio = loss / loss_at_2000.max(1e-6);
            if ratio > 0.98 {
                eprintln!("  [stall abort] iter 5000 loss {:.4} vs iter 2000 {:.4} — aborting run", loss, loss_at_2000);
                break;
            }
        }

        // Periodic checkpoint: every 500 iters + best-loss snapshot.
        if iter > start_iter && iter % 500 == 0 {
            save_iter_checkpoint(&model, &config.checkpoint_name, iter + 1, current_lr, &optimizer, rng.state(), dims, config.tied);
        }
    }

    let elapsed = train_start.elapsed();
    println!("\n=== Wave Training Complete ===");
    println!("  Best L2 loss: {:.6} at iter {}", best_loss, best_iter);
    println!("  NaN skips: {}", nan_skip_count);
    println!("  Total time: {:.1?}", elapsed);

    // Final checkpoint.
    let final_params = flatten_params_ex(&model, config.tied);
    crate::wave_checkpoint::save_checkpoint(
        &final_params, vocab_size, config.n_layers, config.out_proj_groups,
        total_iters, optimizer.lr, &optimizer, rng.state(),
        &config.checkpoint_name, dims,
    );
    println!("  Saved to: {}", config.checkpoint_name);
}

fn save_iter_checkpoint(
    model: &WavePacketModel,
    base_name: &str,
    iter: usize,
    lr: f32,
    optimizer: &Adam,
    rng_state: u64,
    dims: Dims,
    tied: bool,
) {
    let params = flatten_params_ex(model, tied);
    let path = {
        let ck = std::path::Path::new(base_name);
        let stem = ck.file_stem().map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "checkpoint".to_string());
        let parent = ck.parent().filter(|p| !p.as_os_str().is_empty());
        let name = format!("{}_iter{}.bin", stem, iter);
        match parent {
            Some(p) => p.join(name).to_string_lossy().into_owned(),
            None => name,
        }
    };
    crate::wave_checkpoint::save_checkpoint(
        &params, model.vocab_size, model.blocks.len(), 1,
        iter, lr, optimizer, rng_state, &path, dims,
    );
}
