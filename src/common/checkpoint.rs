//! WCHK checkpoint format — save/load trained model weights + optimizer state.
//!
//! v1: 6 config fields (n_bands, n_head, n_layers, maestro_dim, block_size, rk4_steps)
//! v2: adds out_proj_groups (7th config field) — enables Dense or BlockDiagonal
//!
//! Param layout for out_proj depends on groups:
//!   groups=1 (Dense): n_embd×n_embd weight + n_embd bias
//!   groups=N (BlockDiagonal): N × (group_size×group_size weight + group_size bias)

use crate::train::Adam;

/// Save model checkpoint in WCHK v2 format.
/// Config fields use runtime values (Dims), not compile-time constants.
pub fn save_checkpoint(
    params: &[f32],
    vocab_size: usize,
    n_layers: usize,
    out_proj_groups: usize,
    iter: usize,
    lr: f32,
    optimizer: &Adam,
    rng_state: u64,
    path: &str,
    dims: crate::Dims,
) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Failed to create checkpoint file");

    // Magic + version (v2 = has out_proj_groups)
    f.write_all(b"WCHK").unwrap();
    f.write_all(&2u32.to_le_bytes()).unwrap();

    // Config (7 fields in v2) — uses runtime Dims, not compile-time constants
    f.write_all(&(dims.n_bands as u32).to_le_bytes()).unwrap();
    f.write_all(&(dims.n_head as u32).to_le_bytes()).unwrap();
    f.write_all(&(n_layers as u32).to_le_bytes()).unwrap();
    f.write_all(&(dims.maestro_dim as u32).to_le_bytes()).unwrap();
    f.write_all(&(dims.block_size as u32).to_le_bytes()).unwrap();
    f.write_all(&(dims.rk4_steps as u32).to_le_bytes()).unwrap();
    f.write_all(&(out_proj_groups as u32).to_le_bytes()).unwrap();

    // Metadata
    f.write_all(&(vocab_size as u64).to_le_bytes()).unwrap();
    f.write_all(&(iter as u64).to_le_bytes()).unwrap();
    f.write_all(&lr.to_le_bytes()).unwrap();
    f.write_all(&rng_state.to_le_bytes()).unwrap();

    // Optimizer state
    let (adam_t, adam_m, adam_v) = optimizer.checkpoint_state();
    f.write_all(&(adam_t as u64).to_le_bytes()).unwrap();
    for &v in adam_m { f.write_all(&v.to_le_bytes()).unwrap(); }
    for &v in adam_v { f.write_all(&v.to_le_bytes()).unwrap(); }

    // Parameters
    let n = params.len();
    for &v in params { f.write_all(&v.to_le_bytes()).unwrap(); }

    println!("  WCHK v2: {n} params, {n_layers} layers, {out_proj_groups} groups, {:.1}MB",
        (4+4+7*4+8+8+4+8+8+n*4*2+n*4) as f64 / 1e6);
}

/// Load checkpoint for resume.
/// Returns (params, vocab_size, iter, lr, rng_state, adam_t, adam_m, adam_v, out_proj_groups).
pub fn load_checkpoint(path: &str) -> (Vec<f32>, usize, usize, f32, u64, usize, Vec<f32>, Vec<f32>, usize) {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("Failed to open {path}: {e}"));

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"WCHK", "Not a WCHK checkpoint");

    let read_u32 = |f: &mut std::fs::File| -> u32 { let mut b = [0u8; 4]; f.read_exact(&mut b).unwrap(); u32::from_le_bytes(b) };
    let read_u64 = |f: &mut std::fs::File| -> u64 { let mut b = [0u8; 8]; f.read_exact(&mut b).unwrap(); u64::from_le_bytes(b) };
    let read_f32_single = |f: &mut std::fs::File| -> f32 { let mut b = [0u8; 4]; f.read_exact(&mut b).unwrap(); f32::from_le_bytes(b) };

    let version = read_u32(&mut f);
    assert!(version == 1 || version == 2, "Unknown WCHK version {version}");

    // Config
    let ck_bands = read_u32(&mut f) as usize;
    let ck_head = read_u32(&mut f) as usize;
    let ck_layers = read_u32(&mut f) as usize;
    let ck_maestro = read_u32(&mut f) as usize;
    let _ck_block_size = read_u32(&mut f) as usize;
    let _ck_rk4 = read_u32(&mut f) as usize;

    // v2 adds out_proj_groups; v1 defaults to dense (groups=1)
    let out_proj_groups = if version >= 2 {
        read_u32(&mut f) as usize
    } else {
        1 // v1 = dense out_proj
    };

    // Use config from checkpoint header — no compile-time constant assertions
    let n_embd = ck_bands * 2;

    let vocab_size = read_u64(&mut f) as usize;
    let iter = read_u64(&mut f) as usize;
    let lr = read_f32_single(&mut f);
    let rng_state = read_u64(&mut f);

    // Compute param count from checkpoint config (not compile-time constants)
    let gs = n_embd / out_proj_groups;
    let out_proj_params = out_proj_groups * (gs * gs + gs);
    let per_block = n_embd*2 + n_embd*2
        + ck_maestro*n_embd + ck_maestro + n_embd*ck_maestro + n_embd
        + ck_maestro*n_embd + ck_maestro + n_embd*ck_maestro + n_embd
        + out_proj_params;
    let lm_head_params = vocab_size * n_embd;
    let n_base = ck_layers * per_block + n_embd*2 + lm_head_params;
    // Extended: +gamma_raw(n_bands) + alpha(1) + beta(1) + phase_correction(n_bands) per layer
    let n_ext = n_base + ck_layers * (ck_bands + 1 + 1 + ck_bands);
    // Extended + layer_scale: +1 per layer for layer contribution scaling
    let n_ext_ls = n_ext + ck_layers;
    // Phase-native variants: no lm_head, add output_corrector (n_bands phase rotations)
    let n_phase = n_ext - lm_head_params + ck_bands; // with 84-param output corrector
    let n_phase_no_oc = n_ext - lm_head_params; // without output corrector
    let n_phase_ls = n_ext_ls - lm_head_params + ck_bands;
    // RK4 weights variants: +4 per layer for learnable [w0,w1,w2,w3]
    let rk4_extra = ck_layers * 4;
    // Harmonics variants: +n_head per layer for learnable harmonic_raw
    let harm_extra = ck_layers * ck_head;
    // Build all variant sizes (most features → fewest)
    let variants: Vec<usize> = vec![
        // phase-native + layer_scale + rk4 + harmonics
        n_ext_ls + rk4_extra + harm_extra - lm_head_params + ck_bands,
        // phase-native + rk4 + harmonics
        n_ext + rk4_extra + harm_extra - lm_head_params + ck_bands,
        // phase-native + layer_scale + rk4
        n_ext_ls + rk4_extra - lm_head_params + ck_bands,
        // ext + layer_scale + rk4 + harmonics
        n_ext_ls + rk4_extra + harm_extra,
        // phase-native + rk4
        n_ext + rk4_extra - lm_head_params + ck_bands,
        // ext + rk4 + harmonics
        n_ext + rk4_extra + harm_extra,
        // phase-native + harmonics
        n_ext + harm_extra - lm_head_params + ck_bands,
        // ext + layer_scale + rk4
        n_ext_ls + rk4_extra,
        // ext + rk4
        n_ext + rk4_extra,
        // ext + layer_scale
        n_ext_ls,
        // phase-native + layer_scale
        n_ext_ls - lm_head_params + ck_bands,
        // phase-native
        n_ext - lm_head_params + ck_bands,
        // ext
        n_ext,
        // phase-native no output corrector
        n_ext - lm_head_params,
        // base
        n_base,
    ];

    // Determine actual param count from file size
    let file_len = f.metadata().map(|m| m.len()).unwrap_or(0) as usize;
    let header_size = 4 + 4 + 7*4 + 8 + 8 + 4 + 8 + 8; // magic + version + config + metadata + adam_t
    // file = header + adam_m(n*4) + adam_v(n*4) + params(n*4) = header + n*12
    let data_bytes = file_len.saturating_sub(header_size);
    let n_params = variants.iter().find(|&&v| data_bytes == v * 12).copied().unwrap_or(n_base);

    let adam_t = read_u64(&mut f) as usize;
    let read_f32_vec = |f: &mut std::fs::File, n: usize| -> Vec<f32> {
        let mut buf = vec![0u8; n * 4];
        f.read_exact(&mut buf).unwrap();
        buf.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let adam_m = read_f32_vec(&mut f, n_params);
    let adam_v = read_f32_vec(&mut f, n_params);
    let params = read_f32_vec(&mut f, n_params);

    println!("  WCHK v{version}: iter {iter}, lr {lr:.6}, {n_params} params, {ck_layers} layers, {out_proj_groups} groups");
    (params, vocab_size, iter, lr, rng_state, adam_t, adam_m, adam_v, out_proj_groups)
}
