//! Wave Packet Engine — proof of concept
//!
//! New architecture: parallel attention + FFN with harmonic coherence scoring.
//! Tests whether wave packet mechanics can serve as the core computation primitive.

// ─── Module tree ─────────────────────────────────────────────
// Physical layout: src/common/, src/cpu/, src/wgpu_tier/, src/candle_tier/
// Re-exports below keep old crate:: paths working (shim layer).

#[allow(dead_code)]
mod common;
#[allow(dead_code)]
mod cpu;
#[allow(dead_code)]
mod wgpu_tier;
#[allow(dead_code)]
mod candle_tier;
#[allow(dead_code)]
mod monitors;
mod cli;
#[cfg(feature = "serve")]
mod serve_tier;

// Re-export shim — existing code uses crate::model, crate::backend, etc.
// These map to the new physical locations without changing any imports.
#[allow(unused_imports)]
pub use common::model;
#[allow(unused_imports)]
pub use common::embed as wave_embed;
#[allow(unused_imports)]
pub use common::attn as wave_attn;
#[allow(unused_imports)]
pub use common::block as wave_block;
#[allow(unused_imports)]
pub use common::ffn as ffn_backend;
pub use common::wave_model::{WavePacketModel, init_model, init_linear, count_trainable, count_trainable_ex, flatten_params, flatten_params_ex, unflatten_params, unflatten_params_ex};
pub use common::dims::{Dims, PROFILE, N_BANDS, N_EMBD, N_HEAD, N_LAYERS, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS};
pub use cpu::forward::{BlockCache, ForwardCache, forward_with_cache, dual_maestro_forward};
pub use cpu::model_backward::{Gradients, flatten_grads, flatten_grads_ex};
pub use wgpu_tier::diagnostics::{diagnose_ode_gpu_vs_cpu, validate_gpu_fft};
#[allow(unused_imports)]
pub use common::checkpoint as wave_checkpoint;
#[allow(unused_imports)]
pub use common::rng;
#[allow(unused_imports)]
pub use common::bpe;
#[allow(unused_imports)]
pub use common::token_cache;
#[allow(unused_imports)]
pub use monitors::monitor;
#[allow(unused_imports)]
pub use common::data;
#[allow(unused_imports)]
pub use common::fft_ode;

#[allow(unused_imports)]
pub use cpu::train;
#[allow(unused_imports)]
pub use cpu::backward;

#[allow(unused_imports)]
pub use wgpu_tier::backend;
#[allow(unused_imports)]
pub use wgpu_tier::device as gpu;
#[allow(unused_imports)]
pub use wgpu_tier::gpu_backend;
#[allow(unused_imports)]
pub use wgpu_tier::buffers as gpu_buffers;
#[allow(unused_imports)]
pub use wgpu_tier::dispatch as gpu_dispatch;
#[allow(unused_imports)]
pub use wgpu_tier::ops_forward as gpu_ops_forward;
#[allow(unused_imports)]
pub use wgpu_tier::ops_backward as gpu_ops_backward;
#[allow(unused_imports)]
pub use wgpu_tier::pipelines as gpu_pipelines;
#[allow(unused_imports)]
pub use wgpu_tier::resident as gpu_resident;
#[allow(unused_imports)]
pub use wgpu_tier::validate as gpu_validate;
#[allow(unused_imports)]
pub use wgpu_tier::ffn_gpu;
#[allow(unused_imports)]
pub use wgpu_tier::ffn_full_gpu;

#[allow(unused_imports)]
pub use candle_tier::engine as candle_engine;
#[allow(unused_imports)]
pub use candle_tier::ode as gpu_ode;
#[allow(unused_imports)]
pub use candle_tier::block_diag as block_diagonal;

// ─── Main ───────────────────────────────────────────────────────

fn print_help() { common::help::print_help(); }

fn main() {
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    // ─── Convert dataset to wave memory ───
    if std::env::args().any(|a| a == "--convert-dataset") {
        fn pflag_cd<T: std::str::FromStr>(name: &str, default: T) -> T {
            std::env::args().skip_while(|a| a != name).nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let data_path = std::env::args().skip_while(|a| a != "--convert-dataset").nth(1)
            .expect("--convert-dataset requires a data file path");
        let output = std::env::args().skip_while(|a| a != "--output").nth(1)
            .unwrap_or("dataset_waves.kwmf".to_string());
        let n_bands: usize = pflag_cd("--n-bands", 84);
        let n_head: usize = pflag_cd("--n-head", 4);
        let n_layers: usize = pflag_cd("--layers", 4);
        let alpha: f32 = pflag_cd("--alpha", 0.1);
        let beta: f32 = pflag_cd("--beta", 0.2);
        let out_proj_groups: usize = pflag_cd("--out-proj-groups", 1);
        let use_bpe = std::env::args().any(|a| a == "--bpe");
        let tokenizer_path: String = pflag_cd("--tokenizer", "data/tokenizer.json".to_string());
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1);

        let (tokens, vocab_size) = common::data_loader::load_data(&data_path, use_bpe,
            if use_bpe { Some(&tokenizer_path) } else { None });

        // Per-position mode: write KWDS file (embedding + positional, no ODE)
        if std::env::args().any(|a| a == "--per-position") {
            let dims = Dims::from_cli(n_bands, n_head, 16, 128, 16);
            let model = init_model(vocab_size, 42, n_layers, out_proj_groups, dims, alpha, beta);
            println!("Converting to KWDS (per-position): {} tokens, {} bands", tokens.len(), n_bands);
            common::kwds::convert_tokens_to_kwds(&output, &tokens, &model.wte, &model.wpe, n_bands)
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });
            return;
        }

        // Aggregate mode: accumulate into KWMF
        let dims = Dims::from_cli(n_bands, n_head, 16, 128, 16);

        let model = if let Some(ref ckpt) = resume_path {
            println!("Converting dataset through trained model: {}", ckpt);
            let (params, ck_vocab, _, _, _, _, _, _, _, _, _) = wave_checkpoint::load_checkpoint(ckpt);
            let mut m = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims, alpha, beta);
            m.phase_native = true;
            m.output_corrector = vec![0.0; n_bands];
            if params.len() == count_trainable_ex(&m, false) {
                unflatten_params_ex(&mut m, &params, false);
            } else {
                let dims_nc = Dims::from_cli(n_bands, n_head, 16, 128, 16).with_corrector(false);
                m = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_nc, alpha, beta);
                m.phase_native = true;
                m.output_corrector = vec![0.0; n_bands];
                unflatten_params_ex(&mut m, &params, false);
            }
            m.learnable_ode = false;
            m
        } else {
            println!("Converting dataset through UNTRAINED model (random init)");
            let mut m = init_model(vocab_size, 42, n_layers, out_proj_groups, dims, alpha, beta);
            m.phase_native = true;
            m.output_corrector = vec![0.0; n_bands];
            m.learnable_ode = false;
            m
        };

        // Init AGC
        crate::ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);
        let stencil = fft_ode::StencilFft::new(n_bands);

        // Create fresh memory
        let mut mem = common::wave_memory::load_or_create(&output, n_layers, n_bands);

        // Process dataset in chunks (block_size windows)
        let block_size = 128;
        let n_chunks = (tokens.len().saturating_sub(1)) / block_size;
        println!("  Processing {} tokens in {} chunks of {}...", tokens.len(), n_chunks, block_size);

        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * block_size;
            let end = (start + block_size).min(tokens.len());
            let chunk = &tokens[start..end];

            // Forward pass through model
            let cache = crate::cpu::forward::forward_with_cache(
                &model, chunk, dims, None, None, None, Some(&stencil), None, None, None,
            );

            // Extract per-layer hidden states, average across positions
            let ode_states: Vec<(Vec<f32>, Vec<f32>)> = cache.block_caches.iter().map(|bc| {
                let t = bc.input.len().max(1);
                let mut avg_r = vec![0.0f32; n_bands];
                let mut avg_s = vec![0.0f32; n_bands];
                for pos in &bc.input {
                    for k in 0..n_bands.min(pos.len() / 2) {
                        avg_r[k] += pos[k * 2];
                        avg_s[k] += pos[k * 2 + 1];
                    }
                }
                let scale = 1.0 / t as f32;
                for k in 0..n_bands { avg_r[k] *= scale; avg_s[k] *= scale; }
                (avg_r, avg_s)
            }).collect();

            common::wave_memory::merge_ode_states(&mut mem, &ode_states);

            if (chunk_idx + 1) % 100 == 0 {
                println!("    {}/{} chunks processed", chunk_idx + 1, n_chunks);
            }
        }

        // Save
        common::wave_memory::save(&output, &mem);
        println!("  Dataset converted: {} chunks → {} conversations in {}", n_chunks, mem.n_convos, output);

        // Auto-scan
        let scans = common::wave_memory::scan_memory(&mem);
        common::wave_memory::print_memory_scan(&mem, &scans);

        return;
    }

    // ─── Phase encode mode ───
    if std::env::args().any(|a| a == "--encode" || a == "--encode-number"
        || a == "--encode-catalog" || a == "--encode-phases"
        || a == "--relate" || a == "--relate-vocab")
    {
        fn pflag_enc<T: std::str::FromStr>(name: &str, default: T) -> T {
            std::env::args().skip_while(|a| a != name).nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let n_bands: usize = pflag_enc("--n-bands", 84);
        let n_head: usize = pflag_enc("--n-head", 4);
        let n_layers: usize = pflag_enc("--layers", 4);
        let alpha: f32 = pflag_enc("--alpha", 0.1);
        let beta: f32 = pflag_enc("--beta", 0.2);
        let out_proj_groups: usize = pflag_enc("--out-proj-groups", 1);
        let m1: usize = pflag_enc("--m1", 5);
        let m2: usize = pflag_enc("--m2", 7);
        let inject_layer: usize = pflag_enc("--inject-layer", 0);
        let blank = std::env::args().any(|a| a == "--blank");
        let do_scan = std::env::args().any(|a| a == "--scan");

        let dims = Dims::from_cli(n_bands, n_head, 16, 128, 16);

        // Load model or create blank (uses WavePacketModel — same as all other CLI modes)
        let model = if blank {
            println!("Creating blank (untrained) model: {}L, {}bands", n_layers, n_bands);
            init_model(128, 42, n_layers, out_proj_groups, dims, alpha, beta)
        } else {
            let resume = std::env::args().skip_while(|a| a != "--resume").nth(1)
                .expect("--encode/--relate requires --resume <checkpoint> or --blank");
            println!("Loading checkpoint: {}", resume);
            let (params, ck_vocab, _, _, _, _, _, _, _, _, _ck_chi) = wave_checkpoint::load_checkpoint(&resume);
            let mut model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims, alpha, beta);
            // Phase-native detection
            let ext_count = count_trainable_ex(&model, false);
            if params.len() < ext_count {
                model.phase_native = true;
                model.output_corrector = vec![0.0; n_bands];
            }
            unflatten_params_ex(&mut model, &params, false);
            println!("  Model: {}L, {}bands, {}vocab, phase_native={}",
                n_layers, n_bands, ck_vocab, model.phase_native);
            model
        };

        // Initialize AGC — required for ODE stability in encode/relate paths
        crate::ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);

        let n_embd = n_bands * 2;

        // Build char map from data file for text encoding
        let data_path: String = pflag_enc("--data", "data/input.txt".to_string());
        let char_map: Vec<char> = if !blank && std::path::Path::new(&data_path).exists() {
            let text = common::data_loader::load_text_raw(&data_path);
            let mut chars: Vec<char> = text.chars().collect();
            chars.sort(); chars.dedup();
            chars.truncate(model.vocab_size);
            println!("  Char map: {} chars from {}", chars.len(), data_path);
            chars
        } else {
            // Fallback: ASCII identity (works for models with ASCII-range vocab)
            (0..model.vocab_size.min(128)).map(|i| i as u8 as char).collect()
        };
        let encode_char = |c: char| -> Option<usize> {
            char_map.iter().position(|&ch| ch == c)
        };
        let decode_tok = |tok: usize| -> String {
            if tok < char_map.len() {
                let c = char_map[tok];
                if c.is_ascii_graphic() || c == ' ' { format!("'{}'", c) }
                else { format!("t{}", tok) }
            } else {
                format!("t{}", tok)
            }
        };

        // ─── Relate-vocab mode ───
        if std::env::args().any(|a| a == "--relate-vocab") {
            let output: String = pflag_enc("--output", "vocab_relations.json".to_string());
            let (labels, reports, dist, profiles) = common::phase_encode::run_relate_vocab(&model, n_bands, Some(&char_map));
            println!("\n=== Vocabulary Relationship Map ===");
            println!("  {} tokens, {} pairs", labels.len(), reports.len());
            println!("\n  Catalog distribution:");
            let mut entries: Vec<_> = dist.iter().collect();
            entries.sort_by_key(|e| std::cmp::Reverse(*e.1));
            for (name, count) in &entries {
                println!("    {:20} {:5}", name, count);
            }
            // Energy profile summary
            println!("\n  Energy signatures (top amplifiers / dampeners):");
            let mut sorted_profiles: Vec<&common::phase_encode::EnergyProfile> = profiles.iter().collect();
            sorted_profiles.sort_by(|a, b| b.peak_ratio.partial_cmp(&a.peak_ratio).unwrap_or(std::cmp::Ordering::Equal));
            for p in sorted_profiles.iter().take(5) {
                println!("    {:>4}  energy={:.2}x  peak=band{}({:.1}x)  damp=band{}({:.2}x)",
                    p.label, p.total_energy_ratio, p.peak_band, p.peak_ratio, p.damp_band, p.damp_ratio);
            }
            // Compute catalog axes (phase, dignity, direction, destruction)
            let phase_scores: std::collections::HashMap<String, f32> = labels.iter().enumerate().map(|(i, label)| {
                let total = reports.iter().filter(|r| r.label_a == *label || r.label_b == *label).count();
                let non_conj = reports.iter().filter(|r| {
                    (r.label_a == *label || r.label_b == *label)
                    && r.catalog_match.map(|(n, _)| n != "conjunction").unwrap_or(false)
                }).count();
                (label.clone(), if total > 0 { non_conj as f32 / total as f32 } else { 0.0 })
            }).collect();
            let axis_scores = common::catalog_axes::compute_all_axes(&model, n_bands, &char_map, &phase_scores);
            let corr = common::catalog_axes::correlation_matrix(&axis_scores);
            common::catalog_axes::print_axes_summary(&axis_scores, &corr);

            // Targeted destruction profile
            let destruction_profile = common::catalog_axes::compute_destruction_profile(&model, n_bands, m1, m2);
            println!("\n  Targeted destruction profile:");
            for l in &destruction_profile.per_layer {
                println!("    L{}: on={:.3} off={:.3} ratio={:.2}x", l.layer, l.on_grid_cos, l.off_grid_cos, l.ratio);
            }

            if let Err(e) = common::phase_encode::write_vocab_relations_json(&output, &labels, &reports, &dist, Some(&profiles)) {
                eprintln!("Error writing {}: {}", output, e);
            } else {
                println!("\n  Written to: {}", output);
            }
            return;
        }

        // ─── Relate mode (pairwise) ───
        if std::env::args().any(|a| a == "--relate") {
            let mut items: Vec<(String, Vec<f32>)> = Vec::new();
            let args: Vec<String> = std::env::args().collect();
            let mut i = 1;
            while i < args.len() {
                if args[i] == "--relate" {
                    if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                        let text = &args[i + 1];
                        // Encode text: build state for last position (multi-char sequences)
                        let tokens: Vec<usize> = text.chars()
                            .filter_map(|c| encode_char(c))
                            .collect();
                        let mut h = vec![0.0f32; n_embd];
                        if let Some(&tok) = tokens.last() {
                            let pos = (tokens.len() - 1).min(model.wpe.len() - 1);
                            for j in 0..n_embd {
                                h[j] = model.wte[tok][j] + model.wpe[pos][j];
                            }
                        }
                        items.push((text.clone(), h));
                        i += 2;
                        continue;
                    }
                } else if args[i] == "--relate-number" {
                    if i + 1 < args.len() {
                        let n: u64 = args[i + 1].parse().unwrap_or(0);
                        let state = common::phase_encode::encode_number(n, n_bands, m1, m2);
                        items.push((format!("{}", n), state));
                        i += 2;
                        continue;
                    }
                } else if args[i] == "--relate-catalog" {
                    if i + 1 < args.len() {
                        let spec = &args[i + 1];
                        let configs = common::phase_encode::parse_catalog_spec(spec);
                        let state = common::phase_encode::encode_catalog_state(&configs, n_bands);
                        items.push((spec.clone(), state));
                        i += 2;
                        continue;
                    }
                }
                i += 1;
            }

            if items.len() < 2 {
                eprintln!("--relate requires at least 2 items. Use --relate \"a\" --relate \"b\"");
                std::process::exit(1);
            }

            let reports = common::phase_encode::run_relate(&model, &items, n_bands);
            for r in &reports {
                common::phase_encode::print_relate_report(r);
            }
            if items.len() > 2 {
                let labels: Vec<String> = items.iter().map(|(l, _)| l.clone()).collect();
                common::phase_encode::print_relate_matrix(&labels, &reports);
            }
            return;
        }

        // ─── Single encode mode ───
        let (_label, encoded, configs) = if let Some(text) = std::env::args().skip_while(|a| a != "--encode").nth(1) {
            let tokens: Vec<usize> = text.chars()
                .filter_map(|c| encode_char(c))
                .collect();
            println!("  Tokens: {:?} (from \"{}\")", tokens, text);
            let mut states: Vec<Vec<f32>> = Vec::new();
            for (pos, &tok) in tokens.iter().enumerate() {
                let mut h = vec![0.0f32; n_embd];
                if pos < model.wpe.len() {
                    for j in 0..n_embd {
                        h[j] = model.wte[tok][j] + model.wpe[pos][j];
                    }
                }
                states.push(h);
            }
            let state = states.last().cloned().unwrap_or(vec![0.0; n_embd]);
            (format!("Text: \"{}\"", text), state, vec![])
        } else if let Some(n_str) = std::env::args().skip_while(|a| a != "--encode-number").nth(1) {
            let n: u64 = n_str.parse().expect("--encode-number requires an integer");
            let state = common::phase_encode::encode_number(n, n_bands, m1, m2);
            (format!("Number: {}", n), state, vec![])
        } else if let Some(spec) = std::env::args().skip_while(|a| a != "--encode-catalog").nth(1) {
            let configs = common::phase_encode::parse_catalog_spec(&spec);
            let state = common::phase_encode::encode_catalog_state(&configs, n_bands);
            let label = format!("Catalog: {}", spec);
            (label, state, configs)
        } else if let Some(spec) = std::env::args().skip_while(|a| a != "--encode-phases").nth(1) {
            let phases = common::phase_encode::parse_raw_phases(&spec);
            let state = common::phase_encode::encode_raw_phases(&phases, n_bands);
            let label = format!("Raw phases: {}", spec);
            (label, state, vec![])
        } else {
            eprintln!("No encoding specified");
            std::process::exit(1);
        };

        // Show input
        let highlight: Vec<usize> = configs.iter().flat_map(|c| c.bands.iter().copied()).collect();
        common::phase_encode::print_encoded_state(&format!("Encoded state (input to layer {})", inject_layer), &encoded, &highlight);

        // Forward through model
        let (final_out, per_layer) = common::phase_encode::run_encode(&model, &encoded, inject_layer, n_bands);

        // Per-layer cosines
        common::phase_encode::print_layer_cosines(&encoded, &per_layer);

        // Show output bands
        common::phase_encode::print_encoded_state(&format!("After ODE evolution (output of layer {})", per_layer.len().saturating_sub(1)), &final_out, &highlight);

        // Comparison
        common::phase_encode::print_comparison(&encoded, &final_out, &configs);

        // Decoder readout
        if !blank {
            println!("\n=== Decoder readout ===");
            // lm_head readout
            let mut scores: Vec<(usize, f32)> = (0..model.vocab_size).map(|tok| {
                let dot: f32 = (0..n_embd).map(|i| model.lm_head[tok][i] * final_out[i]).sum();
                (tok, dot)
            }).collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            print!("  lm_head top 5:");
            for (tok, score) in scores.iter().take(5) {
                print!("  {} ({:.3})", decode_tok(*tok), score);
            }
            println!();

            // Phase-native readout (dot product against embedding table)
            let mut phase_scores: Vec<(usize, f32)> = (0..model.vocab_size).map(|tok| {
                let dot: f32 = (0..n_embd).map(|i| model.wte[tok][i] * final_out[i]).sum();
                (tok, dot)
            }).collect();
            phase_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            print!("  phase-native top 5:");
            for (tok, score) in phase_scores.iter().take(5) {
                print!("  {} ({:.3})", decode_tok(*tok), score);
            }
            println!();
        }

        // Optional galaxy scan
        if do_scan {
            println!("\n  Running galaxy scan on output...");
            let all_hidden: Vec<Vec<Vec<f32>>> = per_layer.iter()
                .map(|h| vec![h.clone()])
                .collect();
            let post_ln_f = vec![final_out.clone()];
            let per_layer_ceilings: Vec<f32> = model.blocks.iter()
                .map(|b| (std::f32::consts::FRAC_PI_2 / (b.ffn.kerr.alpha + 4.0 * b.ffn.kerr.beta)).sqrt().max(0.5))
                .collect();
            let scan_dir = std::path::PathBuf::from("encode_output_galaxy");
            match common::galaxy_scan::run_and_write_full_scan(
                &all_hidden, &post_ln_f, n_bands, &per_layer_ceilings, m1, m2, &scan_dir,
            ) {
                Ok(scan) => {
                    println!("  Galaxy map written to: {}", scan_dir.display());
                    common::galaxy_scan::print_summary(&scan);
                }
                Err(e) => eprintln!("  Galaxy scan error: {}", e),
            }
        }

        return;
    }

    // ─── Scan memory mode ───
    if std::env::args().any(|a| a == "--scan-memory") {
        let memory_path = std::env::args().skip_while(|a| a != "--scan-memory").nth(1)
            .expect("--scan-memory requires a .kwmf file path");
        let output = std::env::args().skip_while(|a| a != "--output").nth(1);

        println!("Scanning memory: {}", memory_path);
        let mem = kerr_memory::file::load(&memory_path)
            .expect("Failed to load memory file");
        let scans = common::wave_memory::scan_memory(&mem);
        common::wave_memory::print_memory_scan(&mem, &scans);

        if let Some(out_path) = output {
            common::wave_memory::write_memory_scan_json(&out_path, &mem, &scans)
                .unwrap_or_else(|e| eprintln!("Error writing {}: {}", out_path, e));
            println!("\n  JSON written to: {}", out_path);
        }
        return;
    }

    // ─── Galaxy scan mode (retrospective on existing checkpoint) ───
    if std::env::args().any(|a| a == "--galaxy-scan") {
        fn pflag_gs<T: std::str::FromStr>(name: &str, default: T) -> T {
            std::env::args().skip_while(|a| a != name).nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let resume = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--galaxy-scan requires --resume <checkpoint>");
        let data_path = std::env::args().skip_while(|a| a != "--scan-corpus").nth(1)
            .or_else(|| {
                // Try first positional arg that isn't a flag
                std::env::args().skip(1).find(|a| !a.starts_with("--") && !a.starts_with('-'))
            })
            .unwrap_or_else(|| "data/input.txt".to_string());
        let n_bands: usize = pflag_gs("--n-bands", 84);
        let n_head: usize = pflag_gs("--n-head", 4);
        let n_layers: usize = pflag_gs("--layers", 4);
        let alpha: f32 = pflag_gs("--alpha", 0.1);
        let beta: f32 = pflag_gs("--beta", 0.2);
        let out_proj_groups: usize = pflag_gs("--out-proj-groups", 1);
        let m1: usize = pflag_gs("--m1", 5);
        let m2: usize = pflag_gs("--m2", 7);

        let (params, ck_vocab, _, _, _, _, _, _, _, _, _ck_chi) = wave_checkpoint::load_checkpoint(&resume);
        // Try all 4 layout variants: {ode, no-ode} × {corrector, no-corrector}
        let variants: [(bool, bool); 4] = [
            (false, true), (false, false), (true, true), (true, false),
        ];
        let mut dims = Dims::from_cli(n_bands, n_head, 16, 128, 16);
        let mut model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims, alpha, beta);
        let mut loaded = false;
        for (use_ode, use_corr) in &variants {
            let d = Dims::from_cli(n_bands, n_head, 16, 128, 16)
                .with_learnable_ode(*use_ode).with_corrector(*use_corr);
            let mut m = init_model(ck_vocab, 42, n_layers, out_proj_groups, d, alpha, beta);
            m.phase_native = true;
            m.output_corrector = vec![0.0; n_bands];
            if params.len() == count_trainable_ex(&m, false) {
                unflatten_params_ex(&mut m, &params, false);
                eprintln!("  [galaxy-scan] Loaded: ode={}, corrector={}", use_ode, use_corr);
                model = m;
                dims = d;
                loaded = true;
                break;
            }
            // Also try non-phase-native
            let mut m2 = init_model(ck_vocab, 42, n_layers, out_proj_groups, d, alpha, beta);
            if params.len() == count_trainable_ex(&m2, false) {
                unflatten_params_ex(&mut m2, &params, false);
                eprintln!("  [galaxy-scan] Loaded (non-PN): ode={}, corrector={}", use_ode, use_corr);
                model = m2;
                dims = d;
                loaded = true;
                break;
            }
        }
        if !loaded {
            panic!("Cannot match checkpoint param count {} to any model variant", params.len());
        }

        // Load test corpus (first 200 tokens of data file)
        let (tokens, _vs) = common::data_loader::load_data(&data_path, false, None);
        let scan_len = tokens.len().min(200).min(128); // clamp to block_size
        let stencil = fft_ode::StencilFft::new(n_bands * 2);
        let cache = crate::cpu::forward::forward_with_cache(
            &model, &tokens[..scan_len], dims, None, None, None, Some(&stencil), None, None, None,
        );
        let all_hidden: Vec<Vec<Vec<f32>>> = cache.block_caches.iter()
            .map(|bc| bc.input.clone()).collect();
        // Per-layer AGC ceilings from learned alpha/beta
        let per_layer_ceilings: Vec<f32> = model.blocks.iter()
            .map(|b| (std::f32::consts::FRAC_PI_2 / (b.ffn.kerr.alpha + 4.0 * b.ffn.kerr.beta)).sqrt().max(0.5))
            .collect();
        let galaxy_dir = std::path::PathBuf::from(resume.replace(".bin", "_galaxy"));
        match common::galaxy_scan::run_and_write_full_scan(
            &all_hidden, &cache.post_ln_f, n_bands, &per_layer_ceilings, m1, m2, &galaxy_dir,
        ) {
            Ok(scan) => {
                println!("Galaxy map written to: {}", galaxy_dir.display());
                common::galaxy_scan::print_summary(&scan);
            }
            Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
        }
        return;
    }

    // ─── Check gradients mode ───
    fn parse_flag_gradients<T: std::str::FromStr>(name: &str, default: T) -> T {
        std::env::args().skip_while(|a| a != name).nth(1)
            .and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    if std::env::args().any(|a| a == "--check-gradients") {
        let n_bands: usize = parse_flag_gradients("--n-bands", 84);
        let alpha: f32 = parse_flag_gradients("--alpha", 0.1);
        let beta: f32 = parse_flag_gradients("--beta", 0.2);
        let chi: f32 = parse_flag_gradients("--fwm-strength", 0.0);
        let rk4_steps: usize = parse_flag_gradients("--rk4-steps", 16);
        let gamma_raw_val = ((0.1f32).exp() - 1.0).ln();
        let weights = crate::model::KerrWeights {
            gamma_raw: vec![gamma_raw_val; n_bands],
            omega: (0..n_bands).map(|k| (k + 1) as f32 / n_bands as f32).collect(),
            alpha, beta,
            rk4_n_steps: rk4_steps,
            phase_correction: vec![0.0; n_bands],
            rk4_weights: [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0],
            chi,
        };
        if chi > 0.0 {
            eprintln!("Testing with FWM chi={} (FWM Jacobian included)", chi);
        }
        let (passed, total, max_err, details) = common::ode_backward::check_gradients(&weights);
        println!("Gradient check: {}/{} passed, max_rel_err={:.6}", total - details.len(), total, max_err);
        for d in &details { println!("  FAIL: {}", d); }
        if passed { println!("PASS"); } else { println!("FAIL"); std::process::exit(1); }
        return;
    }

    // ─── Recommend mode ───
    if std::env::args().any(|a| a == "--recommend") {
        let data_path = std::env::args().skip_while(|a| a != "--recommend").nth(1)
            .expect("--recommend requires a data file path");
        common::recommend::run_recommend(&data_path);
        return;
    }

    // Rayon thread pool — configurable via --threads (default: half available cores)
    fn parse_flag_early<T: std::str::FromStr>(name: &str, default: T) -> T {
        std::env::args().skip_while(|a| a != name).nth(1)
            .and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_threads: usize = parse_flag_early("--threads", available / 2);
    let n_threads = n_threads.max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok();

    println!("wave-engine v0.1.0  ({n_threads} threads, {available} available)\n");

    // ─── Scale mode ───
    if std::env::args().any(|a| a == "--scale") {
        fn pflag<T: std::str::FromStr>(name: &str, default: T) -> T {
            std::env::args().skip_while(|a| a != name).nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let source = std::env::args().skip_while(|a| a != "--scale").nth(1)
            .expect("--scale requires a checkpoint path");
        let target_bands: usize = pflag("--target-bands", 128);
        let target_head: usize = pflag("--target-head", 8);
        let target_layers: Option<usize> = std::env::args().skip_while(|a| a != "--target-layers").nth(1)
            .and_then(|s| s.parse().ok());
        let output: String = pflag("--output", "scaled_checkpoint.bin".to_string());
        let groups: usize = pflag("--out-proj-groups", 1);

        common::scale::scale_checkpoint(&common::scale::ScaleConfig {
            source_path: source,
            target_bands,
            target_head,
            target_layers,
            output_path: output,
            target_groups: groups,
            seed: 42,
        }).unwrap_or_else(|e| { eprintln!("Scale error: {e}"); std::process::exit(1); });
        return;
    }

    // Check for --candle flag first — routes to entirely different training path
    if std::env::args().any(|a| a == "--candle") {
        fn pflag<T: std::str::FromStr>(name: &str, default: T) -> T {
            std::env::args().skip_while(|a| a != name).nth(1)
                .and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let data_path = std::env::args().nth(1).unwrap_or("data/input.txt".to_string());
        let n_iters: usize = pflag("--iters", 200);
        let n_bands: usize = pflag("--n-bands", N_BANDS);
        let n_head: usize = pflag("--n-head", N_HEAD);
        let n_layers: usize = pflag("--layers", N_LAYERS);
        let maestro_dim: usize = pflag("--maestro-dim", MAESTRO_DIM);
        let rk4_steps: usize = pflag("--rk4-steps", RK4_STEPS);
        let out_proj_groups: usize = pflag("--out-proj-groups", 6);

        let debug_nan = std::env::args().any(|a| a == "--debug-nan");
        let alpha: f32 = pflag("--alpha", 0.1);
        let beta: f32 = pflag("--beta", alpha);
        let chi: f32 = pflag("--fwm-strength", 0.0);
        let phase_native = std::env::args().any(|a| a == "--phase-native");
        match candle_engine::engine::train_candle(
            &data_path, n_iters, n_bands, n_head, n_layers, maestro_dim, rk4_steps, out_proj_groups, debug_nan, alpha, beta, chi, phase_native,
        ) {
            Ok(()) => return,
            Err(e) => { eprintln!("Candle error: {e:?}"); std::process::exit(1); }
        }
    }

    // ─── CLI flag parser ───
    fn parse_flag<T: std::str::FromStr>(name: &str, default: T) -> T {
        std::env::args().skip_while(|a| a != name).nth(1)
            .and_then(|s| s.parse().ok()).unwrap_or(default)
    }

    // Parse dynamic parameter: --flag dyn | --flag 1.0,0.8,1.0 | absent
    fn parse_dyn_param(name: &str) -> train::DynParam {
        let val = std::env::args().skip_while(|a| a != name).nth(1);
        match val {
            None => train::DynParam::Off,
            Some(s) if s == "dyn" || s == "dynamic" => train::DynParam::Dynamic,
            Some(s) if s == "off" || s == "none" || s == "false" => train::DynParam::Off,
            Some(s) => {
                let vals: Vec<f32> = s.split(',').filter_map(|v| v.parse().ok()).collect();
                if vals.is_empty() { train::DynParam::Dynamic } else { train::DynParam::Fixed(vals) }
            }
        }
    }

    // ─── Analyze mode ───
    if std::env::args().any(|a| a == "--analyze") {
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--analyze requires --resume <checkpoint>");
        let n_layers: usize = parse_flag("--layers", N_LAYERS);
        let out_proj_groups: usize = parse_flag("--out-proj-groups", 6);
        let use_bpe = std::env::args().any(|a| a == "--bpe");
        let tokenizer_path: String = parse_flag("--tokenizer", "data/tokenizer.json".to_string());
        let n_bands: usize = parse_flag("--n-bands", N_BANDS);
        let n_head: usize = parse_flag("--n-head", N_HEAD);
        let alpha: f32 = parse_flag("--alpha", 0.1);
        let beta: f32 = parse_flag("--beta", alpha);
        let sub_harmonic = std::env::args().any(|a| a == "--sub-harmonic");
        common::analyze::run_analyze(
            &resume_path, n_layers, out_proj_groups, use_bpe, &tokenizer_path,
            n_bands, n_head, alpha, beta, sub_harmonic,
        );
        return;
    }

    // ─── ODE monitor ───
    if std::env::args().any(|a| a == "--ode-monitor") {
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--ode-monitor requires --resume");
        let n_layers: usize = parse_flag("--layers", N_LAYERS);
        let n_bands: usize = parse_flag("--n-bands", N_BANDS);
        let n_head: usize = parse_flag("--n-head", N_HEAD);
        let out_proj_groups: usize = parse_flag("--out-proj-groups", 6);
        let alpha: f32 = parse_flag("--alpha", 0.1);
        let beta: f32 = parse_flag("--beta", parse_flag("--alpha", 0.1));
        let data_path = std::env::args().skip_while(|a| a != "--data").nth(1)
            .unwrap_or("data/arithmetic_single.txt".to_string());

        let (params, ck_vocab, _, _, _, _, _, _, _, _, _) = wave_checkpoint::load_checkpoint(&resume_path);
        // Load model (try ext+ls, ext, base)
        let dims_ext_ls = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS).with_corrector(true).with_layer_scale(true);
        let mut model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_ext_ls, alpha, beta);
        let ext_ls = common::wave_model::count_trainable_ex(&model, false);
        let dims_ext = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS).with_corrector(true);
        let mut model2 = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_ext, alpha, beta);
        let ext_count = common::wave_model::count_trainable_ex(&model2, false);
        if params.len() == ext_ls {
            common::wave_model::unflatten_params_ex(&mut model, &params, false);
        } else if params.len() == ext_count {
            model = model2;
            common::wave_model::unflatten_params_ex(&mut model, &params, false);
        } else {
            let dims_base = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
                .with_learnable_ode(false).with_corrector(false);
            model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_base, alpha, beta);
            unflatten_params(&mut model, &params);
        }
        model.learnable_ode = false;

        let dims = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
            .with_learnable_ode(false).with_corrector(true);
        let stencil = fft_ode::StencilFft::new(n_bands);
        crate::ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);

        // Build char vocab (supports .txt, .jsonl, directories)
        let text = common::data_loader::load_text_raw(&data_path);
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort(); chars.dedup();
        let char_map: Vec<char> = chars[..chars.len().min(ck_vocab)].to_vec();
        let encode = |s: &str| -> Vec<usize> {
            s.chars().filter_map(|c| char_map.iter().position(|&ch| ch == c)).collect()
        };
        let decode = |id: usize| -> String {
            if id < char_map.len() { char_map[id].to_string() } else { "?".to_string() }
        };

        let prompt = std::env::args().skip_while(|a| a != "--prompt").nth(1)
            .unwrap_or("3+4=".to_string());
        let compare = std::env::args().skip_while(|a| a != "--compare").nth(1);

        println!("=== ODE Monitor ===\n");
        let tokens = encode(&prompt);
        monitors::ode_monitor::print_ode_summary(&model, &tokens, dims, &stencil, &prompt, &decode);

        if let Some(ref cmp) = compare {
            let tokens_b = encode(cmp);
            monitors::ode_monitor::compare_prompts(&model, &tokens, &tokens_b, dims, &stencil, &prompt, cmp);
        }

        return;
    }

    // ─── Phase decode diagnostic ───
    if std::env::args().any(|a| a == "--phase-decode") {
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--phase-decode requires --resume");
        let n_layers: usize = parse_flag("--layers", N_LAYERS);
        let n_bands: usize = parse_flag("--n-bands", N_BANDS);
        let n_head: usize = parse_flag("--n-head", N_HEAD);
        let out_proj_groups: usize = parse_flag("--out-proj-groups", 6);
        let alpha: f32 = parse_flag("--alpha", 0.1);
        let beta: f32 = parse_flag("--beta", parse_flag("--alpha", 0.1));
        let data_path = std::env::args().skip_while(|a| a != "--data").nth(1)
            .unwrap_or("data/arithmetic_single.txt".to_string());

        let (params, ck_vocab, _, _, _, _, _, _, _, _, _) = wave_checkpoint::load_checkpoint(&resume_path);
        let dims_ext = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS).with_corrector(true).with_layer_scale(true);
        let mut model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_ext, alpha, beta);
        let ext_ls = common::wave_model::count_trainable_ex(&model, false);
        let dims_ext2 = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS).with_corrector(true);
        let mut model2 = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_ext2, alpha, beta);
        let ext_count = common::wave_model::count_trainable_ex(&model2, false);
        if params.len() == ext_ls {
            common::wave_model::unflatten_params_ex(&mut model, &params, false);
        } else if params.len() == ext_count {
            model = model2;
            common::wave_model::unflatten_params_ex(&mut model, &params, false);
        } else {
            let dims_base = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
                .with_learnable_ode(false).with_corrector(false);
            model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_base, alpha, beta);
            unflatten_params(&mut model, &params);
        }
        model.learnable_ode = false;

        let dims = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
            .with_learnable_ode(false).with_corrector(true);
        let stencil = fft_ode::StencilFft::new(n_bands);
        crate::ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);

        // Build char vocab from data file (supports .txt, .jsonl, directories)
        let text = common::data_loader::load_text_raw(&data_path);
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort(); chars.dedup();
        let char_map: Vec<char> = chars[..chars.len().min(ck_vocab)].to_vec();
        let encode = |s: &str| -> Vec<usize> {
            s.chars().filter_map(|c| char_map.iter().position(|&ch| ch == c)).collect()
        };
        let decode = |id: usize| -> String {
            if id < char_map.len() { char_map[id].to_string() } else { "?".to_string() }
        };

        println!("Phase decode diagnostic: {} params, {} vocab", params.len(), ck_vocab);
        println!("{:<10} {:<10} {:<10} {:<10}", "Prompt", "Expected", "LM_Head", "Phase");
        println!("{}", "-".repeat(45));

        let prompts = ["9-1=", "3+4=", "5-2=", "7+2=", "1+1=", "8-3=", "6+3=", "4-0=", "0+5=", "9-9="];
        let expected = ["8", "7", "3", "9", "2", "5", "9", "4", "5", "0"];
        let mut lm_correct = 0;
        let mut phase_correct = 0;

        for (prompt, exp) in prompts.iter().zip(expected.iter()) {
            let tokens = encode(prompt);
            let (phase_tok, lm_tok, coherences) = common::phase_decode::phase_decode_compare(
                &model, &tokens, dims, &stencil,
            );
            let lm_ans = decode(lm_tok);
            let ph_ans = decode(phase_tok);
            let lm_ok = lm_ans == *exp;
            let ph_ok = ph_ans == *exp;
            if lm_ok { lm_correct += 1; }
            if ph_ok { phase_correct += 1; }
            println!("{:<10} {:<10} {:<10} {:<10}",
                prompt,
                exp,
                format!("{}{}", lm_ans, if lm_ok { " ✓" } else { " ✗" }),
                format!("{}{}", ph_ans, if ph_ok { " ✓" } else { " ✗" }),
            );
        }
        println!("{}", "-".repeat(45));
        println!("LM Head: {}/10    Phase decode: {}/10", lm_correct, phase_correct);
        return;
    }

    // ─── Gradient check mode ───
    if std::env::args().any(|a| a == "--grad-check") {
        let mode = std::env::args().skip_while(|a| a != "--grad-check").nth(1)
            .unwrap_or("phase-native".to_string());
        let scope = std::env::args().skip_while(|a| a != "--grad-check-scope").nth(1)
            .unwrap_or("sampled".to_string());
        let check_eps: f32 = parse_flag("--grad-check-eps", 1e-4);
        let check_tol: f32 = parse_flag("--grad-check-tol", 0.01);
        let verbose = std::env::args().any(|a| a == "--grad-check-verbose");
        let n_layers: usize = parse_flag("--layers", 1);
        let n_bands: usize = parse_flag("--n-bands", 4);
        let n_head: usize = parse_flag("--n-head", 2);
        let alpha: f32 = parse_flag("--alpha", 0.1);
        let beta: f32 = parse_flag("--beta", 0.2);

        let check_mode = match scope.as_str() {
            "tiny" | "exhaustive" => monitors::junctions::grad_check::CheckMode::Exhaustive,
            "sampled" => monitors::junctions::grad_check::CheckMode::PerSection { n_per_section: 5 },
            other => { eprintln!("Unknown scope: {}. Use tiny, sampled, or exhaustive.", other); return; }
        };
        let config = monitors::junctions::grad_check::GradCheckConfig {
            eps: check_eps, rel_tol: check_tol, mode: check_mode, verbose,
            section_filter: None,
        };

        crate::ffn_backend::init_agc(alpha, beta);

        match mode.as_str() {
            "phase-native" => {
                println!("Gradient check: phase-native, {}L, {}bands, {}head", n_layers, n_bands, n_head);
                let vocab = 15usize;
                let n_embd = n_bands * 2;
                // Generate tiny token sequence
                let tokens: Vec<usize> = (0..8).map(|i| i % vocab).collect();
                let targets: Vec<usize> = (1..9).map(|i| i % vocab).collect();
                let (fwd, fwd_bwd, params, labels) = cpu::grad_check_wrapper::phase_native_check(
                    tokens, targets, n_layers, n_bands, n_head, vocab, alpha, beta,
                );
                let result = monitors::junctions::grad_check::check_gradients(
                    "phase-native", fwd, fwd_bwd, &params, &labels, config,
                );
                monitors::junctions::grad_check::print_result(&result);
                if !result.passed() { std::process::exit(1); }
            }
            "wave-input" => {
                println!("Gradient check: wave-input, {}L, {}bands, {}head", n_layers, n_bands, n_head);
                let vocab = 15usize;
                let n_embd = n_bands * 2;
                // Generate tiny wave inputs/targets
                let mut rng = crate::rng::Rng::new(42);
                let inputs: Vec<Vec<f32>> = (0..4).map(|_| (0..n_embd).map(|_| rng.uniform(1.0)).collect()).collect();
                let targets: Vec<Vec<f32>> = (0..4).map(|_| (0..n_embd).map(|_| rng.uniform(1.0)).collect()).collect();
                let (fwd, fwd_bwd, params, labels) = cpu::grad_check_wrapper::wave_input_check(
                    inputs, targets, n_layers, n_bands, n_head, vocab, alpha, beta,
                );
                let result = monitors::junctions::grad_check::check_gradients(
                    "wave-input", fwd, fwd_bwd, &params, &labels, config,
                );
                monitors::junctions::grad_check::print_result(&result);
                if !result.passed() { std::process::exit(1); }
            }
            other => {
                eprintln!("Unknown grad-check mode: {}. Supported: phase-native, wave-input", other);
                return;
            }
        }
        return;
    }

    // ─── Wave-generate mode (for wave-trained models) ───
    if std::env::args().any(|a| a == "--wave-generate") {
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--wave-generate requires --resume <checkpoint>");
        let prompt = std::env::args().skip_while(|a| a != "--prompt").nth(1)
            .unwrap_or("3+4=".to_string());
        common::generate::run_wave_generate(common::generate::GenerateConfig {
            resume_path,
            prompt,
            max_tokens: parse_flag("--max-tokens", 10),
            n_layers: parse_flag("--layers", N_LAYERS),
            n_bands: parse_flag("--n-bands", N_BANDS),
            n_head: parse_flag("--n-head", N_HEAD),
            out_proj_groups: parse_flag("--out-proj-groups", 1),
            maestro_dim: parse_flag("--maestro-dim", MAESTRO_DIM),
            use_bpe: std::env::args().any(|a| a == "--bpe"),
            tokenizer_path: parse_flag("--tokenizer", "data/tokenizer.json".to_string()),
            alpha: parse_flag("--alpha", 0.1),
            beta: parse_flag("--beta", parse_flag("--alpha", 0.1)),
            temperature: parse_flag("--temperature", 0.0),
            phase_native: true,
            memory_path: None,
            diagnose: std::env::args().any(|a| a == "--wave-diagnose"),
        });
        return;
    }

    // ─── Generate mode ───
    if std::env::args().any(|a| a == "--generate") {
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--generate requires --resume <checkpoint>");
        let prompt = std::env::args().skip_while(|a| a != "--prompt").nth(1)
            .unwrap_or("The ".to_string());
        common::generate::run_generate(common::generate::GenerateConfig {
            resume_path,
            prompt,
            max_tokens: parse_flag("--max-tokens", 200),
            n_layers: parse_flag("--layers", N_LAYERS),
            n_bands: parse_flag("--n-bands", N_BANDS),
            n_head: parse_flag("--n-head", N_HEAD),
            out_proj_groups: parse_flag("--out-proj-groups", 6),
            maestro_dim: parse_flag("--maestro-dim", MAESTRO_DIM),
            use_bpe: std::env::args().any(|a| a == "--bpe"),
            tokenizer_path: parse_flag("--tokenizer", "data/tokenizer.json".to_string()),
            alpha: parse_flag("--alpha", 0.1),
            beta: parse_flag("--beta", parse_flag("--alpha", 0.1)),
            temperature: parse_flag("--temperature", 0.0),
            phase_native: std::env::args().any(|a| a == "--phase-native"),
            memory_path: std::env::args().skip_while(|a| a != "--memory").nth(1),
            diagnose: false,
        });
        return;
    }

    // ─── Serve mode (requires --features serve) ───
    #[cfg(feature = "serve")]
    if std::env::args().any(|a| a == "--serve") {
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1)
            .expect("--serve requires --resume <checkpoint>");
        let n_layers: usize = parse_flag("--layers", N_LAYERS);
        let n_bands: usize = parse_flag("--n-bands", N_BANDS);
        let n_head: usize = parse_flag("--n-head", N_HEAD);
        let out_proj_groups: usize = parse_flag("--out-proj-groups", 6);
        let alpha: f32 = parse_flag("--alpha", 0.1);
        let beta: f32 = parse_flag("--beta", parse_flag("--alpha", 0.1));
        let use_bpe = std::env::args().any(|a| a == "--bpe");
        let tokenizer_path: String = parse_flag("--tokenizer", "data/tokenizer.json".to_string());

        // Load checkpoint
        println!("Loading checkpoint: {resume_path}");
        let (params, ck_vocab, ck_iter, _lr, _rng, _at, _am, _av, _groups, _flags, _chi) =
            wave_checkpoint::load_checkpoint(&resume_path);

        // Build model
        let dims = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
            .with_learnable_ode(false).with_corrector(true);
        let dims_ext = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
            .with_corrector(true);
        let mut model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_ext, alpha, beta);
        let ext_count = common::wave_model::count_trainable_ex(&model, false);
        if params.len() == ext_count {
            common::wave_model::unflatten_params_ex(&mut model, &params, false);
            println!("  Loaded {} params (with ODE/corrector)", params.len());
        } else {
            let dims_base = Dims::from_cli(n_bands, n_head, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
                .with_learnable_ode(false).with_corrector(false);
            model = init_model(ck_vocab, 42, n_layers, out_proj_groups, dims_base, alpha, beta);
            unflatten_params(&mut model, &params);
            println!("  Loaded {} params (base)", params.len());
        }
        model.learnable_ode = false;
        println!("  Model: {}L, {}bands, {}dim, {}vocab, iter {}",
            n_layers, n_bands, n_bands * 2, ck_vocab, ck_iter);

        // Init AGC from model coupling
        crate::ffn_backend::init_agc(model.blocks[0].ffn.kerr.alpha, model.blocks[0].ffn.kerr.beta);

        // Build vocab
        let vocab = if use_bpe {
            let bpe = bpe::BpeTokenizer::from_file(&tokenizer_path);
            println!("  BPE tokenizer: {} vocab", ck_vocab);
            serve_tier::prompt::Vocab::from_bpe(bpe, ck_vocab)
        } else {
            println!("  Char-level: {} vocab", ck_vocab);
            serve_tier::prompt::Vocab::from_chars(ck_vocab)
        };

        let stencil = fft_ode::StencilFft::new(n_bands);

        // Wave memory
        let memory_path: Option<String> = std::env::args().skip_while(|a| a != "--memory").nth(1);
        let wave_mem = memory_path.as_ref().map(|path| {
            std::sync::Mutex::new(
                common::wave_memory::load_or_create(path, n_layers, n_bands)
            )
        });

        let state = std::sync::Arc::new(serve_tier::server::AppState {
            model: std::sync::Arc::new(model),
            vocab: std::sync::Arc::new(vocab),
            dims,
            stencil: std::sync::Arc::new(stencil),
            model_name: parse_flag("--model-name", "wave-engine".to_string()),
            api_key: std::env::args().skip_while(|a| a != "--api-key").nth(1),
            host: parse_flag("--host", "127.0.0.1".to_string()),
            port: parse_flag("--port", 8080),
            memory: wave_mem,
            memory_path,
        });

        serve_tier::server::run_server(state);
        return;
    }

    // ─── Train from KWDS (wave dataset) ───
    if std::env::args().any(|a| a == "--train-from-kwds") {
        let kwds_path = std::env::args().skip_while(|a| a != "--train-from-kwds").nth(1)
            .expect("--train-from-kwds requires a .kwds file path");
        let n_iters: usize = parse_flag("--iters", 10000);
        let n_layers: usize = parse_flag("--layers", N_LAYERS);
        let n_head: usize = parse_flag("--n-head", N_HEAD);
        let out_proj_groups: usize = parse_flag("--out-proj-groups", 1);
        let alpha: f32 = parse_flag("--alpha", 0.1);
        let beta: f32 = parse_flag("--beta", parse_flag("--alpha", 0.1));
        let lr: f32 = parse_flag("--lr", 3e-4);
        let seq_len: usize = parse_flag("--seq", 128);
        let checkpoint_name: String = parse_flag("--checkpoint-name", "wave_trained.bin".to_string());

        // Read KWDS header
        let mut f = std::fs::File::open(&kwds_path).expect("Cannot open KWDS file");
        let header = common::kwds::read_header(&mut f).unwrap();
        let n_bands = header.n_bands as usize;
        let n_embd = header.n_embd as usize;
        let n_positions = header.n_positions as usize;
        println!("Training from KWDS: {} positions, {} bands, {:.1} MB",
            n_positions, n_bands, header.file_size() as f64 / (1024.0 * 1024.0));

        // Create model — vocab_size must match the data's actual vocabulary
        let dims = Dims::from_cli(n_bands, n_head, 16, 128, 16);
        let vocab_size: usize = parse_flag("--vocab", 15);
        let resume_path = std::env::args().skip_while(|a| a != "--resume").nth(1);
        let mut start_iter = 0usize;

        let mut model = init_model(vocab_size, 42, n_layers, out_proj_groups, dims, alpha, beta);
        model.phase_native = true;
        model.output_corrector = vec![0.0; n_bands];
        model.learnable_ode = true;

        // Resume from checkpoint if provided
        if let Some(ref ckpt) = resume_path {
            let (params, _ck_vocab, ck_iter, _, _, _, _, _, _, _, _) = wave_checkpoint::load_checkpoint(ckpt);
            let ext_count = count_trainable_ex(&model, false);
            if params.len() == ext_count {
                unflatten_params_ex(&mut model, &params, false);
                start_iter = ck_iter;
                println!("  Resumed from {} at iter {}", ckpt, ck_iter);
            } else {
                eprintln!("  WARNING: param count mismatch ({} vs {}), starting fresh", params.len(), ext_count);
            }
        }

        crate::ffn_backend::init_agc(alpha, beta);
        let stencil = fft_ode::StencilFft::new(n_bands);

        let mut rng = crate::rng::Rng::new(1337);
        let n_trainable = count_trainable_ex(&model, false);
        println!("  Model: {}L, {}bands, {} trainable params", n_layers, n_bands, n_trainable);

        // Finite-difference gradient check on one batch
        {
            println!("  Gradient check (cosine loss)...");
            let inputs = common::kwds::read_input_window(&mut f, &header, 0, 8).unwrap();
            let targets = common::kwds::read_target_window(&mut f, &header, 0, 8).unwrap();
            // Check the wave_loss module's cosine gradient
            let (passed, max_err) = common::wave_loss::check_cosine_gradient(&inputs[0], &targets[0]);
            println!("    Cosine grad check: {} (max_rel_err={:.6})", if passed { "PASS" } else { "FAIL" }, max_err);
        }

        // Training loop with real backward pass
        // Adam state
        let mut adam_m = vec![0.0f32; n_trainable];
        let mut adam_v = vec![0.0f32; n_trainable];
        let mut adam_t = 0u64;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let adam_eps = 1e-8f32;

        let total_iters = start_iter + n_iters;
        println!("  Training for {} iters ({}→{}), seq_len={}, lr={}", n_iters, start_iter, total_iters, seq_len, lr);
        let mut best_loss = f32::MAX;
        let t0 = std::time::Instant::now();

        for iter in start_iter..total_iters {
            let max_start = n_positions.saturating_sub(seq_len + 1);
            let start = (rng.next_u64() as usize) % max_start.max(1);
            let window_len = seq_len.min(n_positions - start - 1);

            let inputs = common::kwds::read_input_window(&mut f, &header, start as u64, window_len).unwrap();
            let targets = common::kwds::read_target_window(&mut f, &header, start as u64, window_len).unwrap();

            // Forward
            let cache = crate::cpu::forward::forward_with_cache_from_waves(
                &model, &inputs, dims, Some(&stencil),
            );

            // Backward with wave targets (L2 loss — gradients stay strong near optimum)
            let (loss, grads) = crate::cpu::model_backward::backward_wave(&model, &cache, &targets, dims);

            // Also compute cosine similarity for monitoring decode readiness
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

            // Flatten grads and apply Adam
            let flat_grads = crate::cpu::model_backward::flatten_grads_ex(&grads, false);
            adam_t += 1;
            let mut params = flatten_params_ex(&model, false);
            for i in 0..params.len() {
                adam_m[i] = beta1 * adam_m[i] + (1.0 - beta1) * flat_grads[i];
                adam_v[i] = beta2 * adam_v[i] + (1.0 - beta2) * flat_grads[i] * flat_grads[i];
                let m_hat = adam_m[i] / (1.0 - beta1.powi(adam_t as i32));
                let v_hat = adam_v[i] / (1.0 - beta2.powi(adam_t as i32));
                params[i] -= lr * m_hat / (v_hat.sqrt() + adam_eps);
            }
            unflatten_params_ex(&mut model, &params, false);

            if iter % 100 == 0 || iter == n_iters - 1 {
                let elapsed = t0.elapsed().as_millis();
                let ms_per = if iter > 0 { elapsed / iter as u128 } else { 0 };
                println!("  iter {:6}  l2_loss {:.6}  cos_sim {:.4}  best_l2 {:.6}  {}ms/iter",
                    iter, loss, cos_sim, best_loss, ms_per);
            }
        }

        println!("\n=== Wave Training Complete ===");
        println!("  Iters: {} ({}→{})", n_iters, start_iter, total_iters);
        println!("  Best L2 loss: {:.6}", best_loss);
        println!("  Target cosine for decode: ~0.95");

        // Save checkpoint
        let final_params = flatten_params_ex(&model, false);
        let n_params = final_params.len();
        let dummy_adam = train::Adam::new(lr, n_params);
        wave_checkpoint::save_checkpoint(
            &final_params, vocab_size, n_layers, out_proj_groups,
            total_iters, lr, &dummy_adam, rng.state(), &checkpoint_name, dims,
        );
        println!("  Saved to: {}", checkpoint_name);

        // Galaxy scan on the wave-trained model
        println!("\n  Running galaxy scan...");
        // Need to process some data through the model for the scan
        // Use the KWDS input waves as the scan data
        let scan_len = 128.min(n_positions);
        let scan_inputs = common::kwds::read_input_window(&mut f, &header, 0, scan_len).unwrap();
        let scan_cache = crate::cpu::forward::forward_with_cache_from_waves(
            &model, &scan_inputs, dims, Some(&stencil),
        );
        let all_hidden: Vec<Vec<Vec<f32>>> = scan_cache.block_caches.iter()
            .map(|bc| bc.input.clone()).collect();
        let per_layer_ceilings: Vec<f32> = model.blocks.iter()
            .map(|b| (std::f32::consts::FRAC_PI_2 / (b.ffn.kerr.alpha + 4.0 * b.ffn.kerr.beta)).sqrt().max(0.5))
            .collect();
        let m1 = 5; let m2 = 7; // default grids
        let galaxy_dir = std::path::PathBuf::from(checkpoint_name.replace(".bin", "_galaxy"));
        match common::galaxy_scan::run_and_write_full_scan(
            &all_hidden, &scan_cache.post_ln_f, n_bands, &per_layer_ceilings, m1, m2, &galaxy_dir,
        ) {
            Ok(scan) => {
                println!("  Galaxy map written to: {}", galaxy_dir.display());
                common::galaxy_scan::print_summary(&scan);
            }
            Err(e) => eprintln!("  Galaxy scan error: {}", e),
        }

        return;
    }

    // ─── Training dispatch ───
    train::run_training(train::TrainConfig {
        data_path: std::env::args().nth(1).unwrap_or("data/input.txt".to_string()),
        n_iters: parse_flag("--iters", 500),
        batch_size: parse_flag("--batch", 4),
        seq_len: parse_flag("--seq", 256),
        n_layers: parse_flag("--layers", N_LAYERS),
        lr: parse_flag("--lr", if N_BANDS > 256 { 1e-4 } else { 3e-4 }),
        use_bpe: std::env::args().any(|a| a == "--bpe"),
        tokenizer_path: parse_flag("--tokenizer", "data/tokenizer.json".to_string()),
        resume_path: std::env::args().skip_while(|a| a != "--resume").nth(1),
        use_curriculum: !std::env::args().any(|a| a == "--no-curriculum"),
        use_gpu: std::env::args().any(|a| a == "--gpu"),
        use_monitor: std::env::args().any(|a| a == "--monitor"),
        out_proj_groups: parse_flag("--out-proj-groups", 6),
        checkpoint_name: parse_flag("--checkpoint-name", "checkpoint.bin".to_string()),
        n_bands: parse_flag("--n-bands", N_BANDS),
        n_head: parse_flag("--n-head", N_HEAD),
        maestro_dim: parse_flag("--maestro-dim", MAESTRO_DIM),
        alpha: parse_flag("--alpha", 0.1),
        beta: parse_flag("--beta", parse_flag("--alpha", 0.1)), // default beta = alpha
        agc_ceiling: std::env::args().skip_while(|a| a != "--agc-ceiling").nth(1)
            .and_then(|s| s.parse().ok()),
        log_name: std::env::args().skip_while(|a| a != "--log-name").nth(1),
        m1: std::env::args().skip_while(|a| a != "--m1").nth(1).and_then(|s| s.parse().ok()),
        m2: std::env::args().skip_while(|a| a != "--m2").nth(1).and_then(|s| s.parse().ok()),
        tied: std::env::args().any(|a| a == "--tied-embeddings"),
        lm_rank: parse_flag("--lm-rank", 0),
        wave_decode: std::env::args().any(|a| a == "--wave-decode"),
        unfreeze_phases: std::env::args().any(|a| a == "--unfreeze-phases"),
        health_interval: parse_flag("--health-interval", 0),
        freeze_ode: std::env::args().any(|a| a == "--freeze-ode"),
        head_lr_floor: parse_flag("--head-lr-floor", 0.0),
        no_corrector: false, // legacy — now controlled by --corrector flag
        layer_scale: parse_dyn_param("--layer-scale"),
        lr_scale: parse_dyn_param("--lr-scale"),
        phase_native: std::env::args().any(|a| a == "--phase-native"),
        fwm_strength: parse_flag("--fwm-strength", 0.0),
        phase_temp: parse_flag("--phase-temp", 1.0),
        pythagorean: std::env::args().any(|a| a == "--pythagorean"),
        spring_k: parse_flag("--spring", 0.1),
        active_layers: std::env::args().skip_while(|a| a != "--active-layers").nth(1).and_then(|s| s.parse().ok()),
        rk4_weights: parse_dyn_param("--rk4-weights"),
        wd: parse_dyn_param("--wd"),
        harmonics: parse_dyn_param("--harmonics"),
        agc_headroom: parse_dyn_param("--agc-headroom"),
        corrector: {
            // --corrector dyn | --corrector off | --no-corrector (legacy)
            let c = parse_dyn_param("--corrector");
            if matches!(c, train::DynParam::Off) && !std::env::args().any(|a| a == "--no-corrector") {
                train::DynParam::Dynamic // default: corrector ON (learnable)
            } else {
                c
            }
        },
    });
}
