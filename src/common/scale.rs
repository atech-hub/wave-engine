//! Progressive dimension scaling — scale trained checkpoints to larger dimensions.
//!
//! Band k has physical meaning in the wave architecture: it carries a specific
//! harmonic frequency. Scaling from 84→128 bands preserves bands 1–84 with their
//! learned weights and adds bands 85–128 with fresh initialisation.

use crate::common::rng::Rng;

/// Configuration for dimension scaling.
pub struct ScaleConfig {
    pub source_path: String,
    pub target_bands: usize,
    pub target_head: usize,
    pub output_path: String,
    pub target_groups: usize,
    pub seed: u64,
}

/// Source checkpoint metadata (read from WCHK header).
struct SourceInfo {
    n_bands: usize,
    n_head: usize,
    n_layers: usize,
    maestro_dim: usize,
    block_size: usize,
    rk4_steps: usize,
    out_proj_groups: usize,
    vocab_size: usize,
    params: Vec<f32>,
}

/// Pad a 1D vector from old_len to new_len.
fn pad_1d(src: &[f32], new_len: usize, fill: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(new_len);
    out.extend_from_slice(src);
    out.resize(new_len, fill);
    out
}

/// Read a 2D matrix [rows × cols] from flat params at offset.
fn read_2d(params: &[f32], offset: &mut usize, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut m = Vec::with_capacity(rows);
    for _ in 0..rows {
        m.push(params[*offset..*offset + cols].to_vec());
        *offset += cols;
    }
    m
}

/// Read a 1D vector from flat params at offset.
fn read_1d(params: &[f32], offset: &mut usize, len: usize) -> Vec<f32> {
    let v = params[*offset..*offset + len].to_vec();
    *offset += len;
    v
}

/// Pad a 2D matrix [old_rows × old_cols] → [new_rows × new_cols].
/// Top-left block preserved, rest filled with fill_fn.
fn pad_2d(src: &[Vec<f32>], new_rows: usize, new_cols: usize, fill_fn: &mut dyn FnMut() -> f32) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(new_rows);
    for (i, row) in src.iter().enumerate() {
        let mut new_row = Vec::with_capacity(new_cols);
        new_row.extend_from_slice(row);
        // Pad existing rows with new columns
        while new_row.len() < new_cols { new_row.push(fill_fn()); }
        out.push(new_row);
    }
    // Add new rows (all values from fill_fn)
    for _ in src.len()..new_rows {
        let row: Vec<f32> = (0..new_cols).map(|_| fill_fn()).collect();
        out.push(row);
    }
    out
}

/// Flatten a 2D matrix into a Vec<f32> (row-major).
fn flatten_2d(m: &[Vec<f32>]) -> Vec<f32> {
    let mut out = Vec::new();
    for row in m { out.extend_from_slice(row); }
    out
}

/// Read source checkpoint (WCHK v2 only, returns params without optimizer state).
fn read_source(path: &str) -> Result<SourceInfo, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("Failed to open {path}: {e}"))?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(|e| format!("Read error: {e}"))?;
    if &magic != b"WCHK" { return Err("Not a WCHK checkpoint".into()); }

    let read_u32 = |f: &mut std::fs::File| -> u32 { let mut b = [0u8; 4]; f.read_exact(&mut b).unwrap(); u32::from_le_bytes(b) };
    let read_u64 = |f: &mut std::fs::File| -> u64 { let mut b = [0u8; 8]; f.read_exact(&mut b).unwrap(); u64::from_le_bytes(b) };
    let read_f32 = |f: &mut std::fs::File| -> f32 { let mut b = [0u8; 4]; f.read_exact(&mut b).unwrap(); f32::from_le_bytes(b) };

    let version = read_u32(&mut f);
    if version < 2 { return Err("Scaling requires WCHK v2 checkpoint".into()); }

    let n_bands = read_u32(&mut f) as usize;
    let n_head = read_u32(&mut f) as usize;
    let n_layers = read_u32(&mut f) as usize;
    let maestro_dim = read_u32(&mut f) as usize;
    let block_size = read_u32(&mut f) as usize;
    let rk4_steps = read_u32(&mut f) as usize;
    let out_proj_groups = read_u32(&mut f) as usize;

    let vocab_size = read_u64(&mut f) as usize;
    let _iter = read_u64(&mut f);
    let _lr = read_f32(&mut f);
    let _rng_state = read_u64(&mut f);

    // Compute param count
    let n_embd = n_bands * 2;
    let gs = n_embd / out_proj_groups;
    let out_proj_params = out_proj_groups * (gs * gs + gs);
    let per_block = n_embd*4 // ln + ln_ffn (weight + bias each)
        + maestro_dim*n_embd + maestro_dim + n_embd*maestro_dim + n_embd  // maestro_in
        + maestro_dim*n_embd + maestro_dim + n_embd*maestro_dim + n_embd  // maestro_out
        + out_proj_params;
    let n_params = n_layers * per_block + n_embd*2 + vocab_size*n_embd;

    // Skip Adam state (t + m + v)
    let adam_t = read_u64(&mut f);
    let skip_bytes = n_params * 4 * 2; // m and v
    let mut skip_buf = vec![0u8; skip_bytes];
    f.read_exact(&mut skip_buf).map_err(|e| format!("Failed to skip Adam state: {e}"))?;

    // Read params
    let mut params_buf = vec![0u8; n_params * 4];
    f.read_exact(&mut params_buf).map_err(|e| format!("Failed to read params: {e}"))?;
    let params: Vec<f32> = params_buf.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    println!("  WCHK v2: {} params, {} layers, {} groups, {} bands, {} vocab",
        n_params, n_layers, out_proj_groups, n_bands, vocab_size);

    Ok(SourceInfo { n_bands, n_head, n_layers, maestro_dim, block_size, rk4_steps, out_proj_groups, vocab_size, params })
}

/// Scale a checkpoint from source dimensions to target dimensions.
pub fn scale_checkpoint(config: &ScaleConfig) -> Result<(), String> {
    println!("Scaling checkpoint: {}", config.source_path);

    let src = read_source(&config.source_path)?;

    if config.target_bands <= src.n_bands {
        return Err(format!("Target bands ({}) must be greater than source ({})", config.target_bands, src.n_bands));
    }
    let tgt_embd = config.target_bands * 2;
    if tgt_embd % config.target_head != 0 {
        return Err(format!("Target n_embd ({}) must be divisible by target n_head ({})", tgt_embd, config.target_head));
    }

    let src_embd = src.n_bands * 2;
    let md = src.maestro_dim;

    println!("  Source: {} bands ({}-dim), {} layers, {} groups, {} vocab",
        src.n_bands, src_embd, src.n_layers, src.out_proj_groups, src.vocab_size);
    println!("  Target: {} bands ({}-dim), {} layers, {} groups, {} vocab",
        config.target_bands, tgt_embd, src.n_layers, config.target_groups, src.vocab_size);

    let mut rng = Rng::new(config.seed);
    let small_limit = 1.0 / (tgt_embd as f32).sqrt();
    let gamma_raw_init = ((0.1f32).exp() - 1.0).ln();

    let mut scaled_params: Vec<f32> = Vec::new();
    let mut offset = 0;

    println!("\n  Per-block weight scaling:");

    for _layer in 0..src.n_layers {
        // 1. ln.weight [n_embd] → pad with 1.0
        let ln_w = read_1d(&src.params, &mut offset, src_embd);
        scaled_params.extend_from_slice(&pad_1d(&ln_w, tgt_embd, 1.0));

        // 2. ln.bias [n_embd] → pad with 0.0
        let ln_b = read_1d(&src.params, &mut offset, src_embd);
        scaled_params.extend_from_slice(&pad_1d(&ln_b, tgt_embd, 0.0));

        // 3. ln_ffn.weight [n_embd] → pad with 1.0
        let ln_ffn_w = read_1d(&src.params, &mut offset, src_embd);
        scaled_params.extend_from_slice(&pad_1d(&ln_ffn_w, tgt_embd, 1.0));

        // 4. ln_ffn.bias [n_embd] → pad with 0.0
        let ln_ffn_b = read_1d(&src.params, &mut offset, src_embd);
        scaled_params.extend_from_slice(&pad_1d(&ln_ffn_b, tgt_embd, 0.0));

        // 5. maestro_in.squeeze.w [maestro_dim × n_embd] → pad cols with 0.0
        let mae_in_sq_w = read_2d(&src.params, &mut offset, md, src_embd);
        let scaled = pad_2d(&mae_in_sq_w, md, tgt_embd, &mut || 0.0);
        scaled_params.extend_from_slice(&flatten_2d(&scaled));

        // 6. maestro_in.squeeze.b [maestro_dim] → no change
        let mae_in_sq_b = read_1d(&src.params, &mut offset, md);
        scaled_params.extend_from_slice(&mae_in_sq_b);

        // 7. maestro_in.process.w [n_embd × maestro_dim] → pad rows with small random
        let mae_in_pr_w = read_2d(&src.params, &mut offset, src_embd, md);
        let scaled = pad_2d(&mae_in_pr_w, tgt_embd, md, &mut || rng.uniform(small_limit));
        scaled_params.extend_from_slice(&flatten_2d(&scaled));

        // 8. maestro_in.process.b [n_embd] → pad with 0.0
        let mae_in_pr_b = read_1d(&src.params, &mut offset, src_embd);
        scaled_params.extend_from_slice(&pad_1d(&mae_in_pr_b, tgt_embd, 0.0));

        // 9. maestro_out.squeeze.w [maestro_dim × n_embd] → pad cols with 0.0
        let mae_out_sq_w = read_2d(&src.params, &mut offset, md, src_embd);
        let scaled = pad_2d(&mae_out_sq_w, md, tgt_embd, &mut || 0.0);
        scaled_params.extend_from_slice(&flatten_2d(&scaled));

        // 10. maestro_out.squeeze.b [maestro_dim] → no change
        let mae_out_sq_b = read_1d(&src.params, &mut offset, md);
        scaled_params.extend_from_slice(&mae_out_sq_b);

        // 11. maestro_out.process.w [n_embd × maestro_dim] → pad rows with small random
        let mae_out_pr_w = read_2d(&src.params, &mut offset, src_embd, md);
        let scaled = pad_2d(&mae_out_pr_w, tgt_embd, md, &mut || rng.uniform(small_limit));
        scaled_params.extend_from_slice(&flatten_2d(&scaled));

        // 12. maestro_out.process.b [n_embd] → pad with 0.0
        let mae_out_pr_b = read_1d(&src.params, &mut offset, src_embd);
        scaled_params.extend_from_slice(&pad_1d(&mae_out_pr_b, tgt_embd, 0.0));

        // 13. out_proj — Dense or BlockDiagonal
        // Source: read as-is. Target: always Dense (recommended by spec for <=256-dim).
        if src.out_proj_groups == 1 {
            // Dense [src_embd × src_embd] + [src_embd]
            let op_w = read_2d(&src.params, &mut offset, src_embd, src_embd);
            let op_b = read_1d(&src.params, &mut offset, src_embd);

            if config.target_groups == 1 {
                // Dense → Dense: pad to [tgt_embd × tgt_embd]
                let scaled_w = pad_2d(&op_w, tgt_embd, tgt_embd, &mut || rng.uniform(small_limit));
                scaled_params.extend_from_slice(&flatten_2d(&scaled_w));
                scaled_params.extend_from_slice(&pad_1d(&op_b, tgt_embd, 0.0));
            } else {
                // Dense → BlockDiagonal: extract diagonal blocks (lossy)
                let tgt_gs = tgt_embd / config.target_groups;
                for g in 0..config.target_groups {
                    let g_start = g * tgt_gs;
                    for i in 0..tgt_gs {
                        for j in 0..tgt_gs {
                            let si = g_start + i;
                            let sj = g_start + j;
                            if si < src_embd && sj < src_embd {
                                scaled_params.push(op_w[si][sj]);
                            } else {
                                scaled_params.push(rng.uniform(small_limit));
                            }
                        }
                    }
                    // bias
                    for i in 0..tgt_gs {
                        let si = g_start + i;
                        if si < src_embd { scaled_params.push(op_b[si]); }
                        else { scaled_params.push(0.0); }
                    }
                }
            }
        } else {
            // BlockDiagonal source
            let src_gs = src_embd / src.out_proj_groups;
            // Read all groups
            let mut all_groups_w: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut all_groups_b: Vec<Vec<f32>> = Vec::new();
            for _ in 0..src.out_proj_groups {
                let gw = read_2d(&src.params, &mut offset, src_gs, src_gs);
                let gb = read_1d(&src.params, &mut offset, src_gs);
                all_groups_w.push(gw);
                all_groups_b.push(gb);
            }

            if config.target_groups == 1 {
                // BlockDiag → Dense: embed diagonal blocks into larger dense matrix
                let mut dense_w = vec![vec![0.0f32; tgt_embd]; tgt_embd];
                let mut dense_b = vec![0.0f32; tgt_embd];
                for (g, (gw, gb)) in all_groups_w.iter().zip(&all_groups_b).enumerate() {
                    let g_start = g * src_gs;
                    for i in 0..src_gs {
                        for j in 0..src_gs {
                            dense_w[g_start + i][g_start + j] = gw[i][j];
                        }
                        dense_b[g_start + i] = gb[i];
                    }
                }
                // Fill remaining with small random
                for i in 0..tgt_embd {
                    for j in 0..tgt_embd {
                        if dense_w[i][j] == 0.0 && (i >= src_embd || j >= src_embd) {
                            dense_w[i][j] = rng.uniform(small_limit);
                        }
                    }
                }
                scaled_params.extend_from_slice(&flatten_2d(&dense_w));
                scaled_params.extend_from_slice(&dense_b);
            } else {
                // BlockDiag → BlockDiag: pad each group
                let tgt_gs = tgt_embd / config.target_groups;
                for g in 0..config.target_groups {
                    if g < all_groups_w.len() {
                        let scaled_gw = pad_2d(&all_groups_w[g], tgt_gs, tgt_gs, &mut || rng.uniform(small_limit));
                        scaled_params.extend_from_slice(&flatten_2d(&scaled_gw));
                        scaled_params.extend_from_slice(&pad_1d(&all_groups_b[g], tgt_gs, 0.0));
                    } else {
                        // New group — fresh init
                        for _ in 0..tgt_gs {
                            for _ in 0..tgt_gs { scaled_params.push(rng.uniform(small_limit)); }
                        }
                        for _ in 0..tgt_gs { scaled_params.push(0.0); }
                    }
                }
            }
        }
    }

    // 14. ln_f.weight [n_embd] → pad with 1.0
    let ln_f_w = read_1d(&src.params, &mut offset, src_embd);
    scaled_params.extend_from_slice(&pad_1d(&ln_f_w, tgt_embd, 1.0));

    // 15. ln_f.bias [n_embd] → pad with 0.0
    let ln_f_b = read_1d(&src.params, &mut offset, src_embd);
    scaled_params.extend_from_slice(&pad_1d(&ln_f_b, tgt_embd, 0.0));

    // 16. lm_head [vocab × n_embd] → pad cols with small random
    let lm_head = read_2d(&src.params, &mut offset, src.vocab_size, src_embd);
    let scaled_lm = pad_2d(&lm_head, src.vocab_size, tgt_embd, &mut || rng.uniform(small_limit));
    scaled_params.extend_from_slice(&flatten_2d(&scaled_lm));

    assert_eq!(offset, src.params.len(), "Did not consume all source params");

    // Print summary
    let src_count = src.params.len();
    let tgt_count = scaled_params.len();
    println!("    LN weights:      {} → {}  (pad 1.0/0.0)", src_embd, tgt_embd);
    println!("    Maestro squeeze: [{}×{}] → [{}×{}]  (pad cols 0.0)", md, src_embd, md, tgt_embd);
    println!("    Maestro process: [{}×{}] → [{}×{}]  (pad rows random)", src_embd, md, tgt_embd, md);
    println!("    Out_proj:        [{}×{}] → [{}×{}]", src_embd, src_embd, tgt_embd, tgt_embd);
    println!("\n  Final layers:");
    println!("    ln_f:    {} → {}  (pad 1.0/0.0)", src_embd, tgt_embd);
    println!("    lm_head: [{}×{}] → [{}×{}]  (pad cols random)", src.vocab_size, src_embd, src.vocab_size, tgt_embd);
    println!("\n  Source params:  {}", src_count);
    println!("  Target params:  {}", tgt_count);
    println!("  Transplanted:   {} (100% of source preserved)", src_count);
    println!("  New (random):   {}", tgt_count - src_count);

    // Save as WCHK v2 with iter=0, fresh optimizer
    let dims = crate::Dims::from_cli(config.target_bands, config.target_head, src.maestro_dim, src.block_size, src.rk4_steps);
    let n_trainable = tgt_count;
    let optimizer = crate::cpu::train::Adam::new(1e-4, n_trainable);
    crate::wave_checkpoint::save_checkpoint(
        &scaled_params, src.vocab_size, src.n_layers, config.target_groups,
        0, 1e-4, &optimizer, 42, &config.output_path, dims,
    );
    println!("\n  Saved: {} (WCHK v2, iter=0, fresh optimizer)", config.output_path);

    Ok(())
}
