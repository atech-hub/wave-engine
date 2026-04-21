//! WCHK checkpoint format — save/load trained model weights + optimizer state.
//!
//! v1: 6 config fields (n_bands, n_head, n_layers, maestro_dim, block_size, rk4_steps)
//! v2: adds out_proj_groups (7th config field) — enables Dense or BlockDiagonal
//! v3: adds feature_flags u32 — disambiguates param layouts that have same size
//! v4: adds chi f32 — FWM strength persisted in checkpoint (first-class feature)
//!
//! Feature flags (v3+):
//!   bit 0: learnable_ode (gamma/alpha/beta/corrector in param vector)
//!   bit 1: layer_scale (per-layer residual scaling)
//!   bit 2: rk4_weights (per-layer RK4 combination weights)
//!   bit 3: dyn_harmonics (per-head harmonic numbers)
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

    // Magic + version (v4 = has chi field)
    f.write_all(b"WCHK").unwrap();
    f.write_all(&4u32.to_le_bytes()).unwrap();

    // Config (7 fields) — uses runtime Dims, not compile-time constants
    f.write_all(&(dims.n_bands as u32).to_le_bytes()).unwrap();
    f.write_all(&(dims.n_head as u32).to_le_bytes()).unwrap();
    f.write_all(&(n_layers as u32).to_le_bytes()).unwrap();
    f.write_all(&(dims.maestro_dim as u32).to_le_bytes()).unwrap();
    f.write_all(&(dims.block_size as u32).to_le_bytes()).unwrap();
    f.write_all(&(dims.rk4_steps as u32).to_le_bytes()).unwrap();
    f.write_all(&(out_proj_groups as u32).to_le_bytes()).unwrap();

    // Feature flags (v3) — disambiguates param layouts
    let mut flags: u32 = 0;
    if dims.learnable_ode  { flags |= 1 << 0; }
    if dims.use_layer_scale { flags |= 1 << 1; }
    if dims.use_rk4_weights { flags |= 1 << 2; }
    if dims.use_dyn_harmonics { flags |= 1 << 3; }
    if dims.learnable_attn  { flags |= 1 << 4; }
    // phase_native: detect from param count (no lm_head = phase_native)
    // We check if the save was phase_native by seeing if params count is smaller than non-phase
    f.write_all(&flags.to_le_bytes()).unwrap();

    // FWM strength (v4) — persisted so resume doesn't lose chi
    f.write_all(&dims.fwm_strength.to_le_bytes()).unwrap();

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

    println!("  WCHK v4: {n} params, {n_layers} layers, {out_proj_groups} groups, flags=0x{flags:02x}, chi={:.3}, {:.1}MB",
        dims.fwm_strength,
        (4+4+8*4+8+8+4+8+8+n*4*2+n*4) as f64 / 1e6);
}

/// Load checkpoint for resume.
/// Returns (params, vocab_size, iter, lr, rng_state, adam_t, adam_m, adam_v, out_proj_groups, feature_flags, chi).
/// Feature flags (v3+): bit0=learnable_ode, bit1=layer_scale, bit2=rk4_weights, bit3=dyn_harmonics.
/// v1/v2 checkpoints return flags=0, chi=0.0.
pub fn load_checkpoint(path: &str) -> (Vec<f32>, usize, usize, f32, u64, usize, Vec<f32>, Vec<f32>, usize, u32, f32) {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("Failed to open {path}: {e}"));

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"WCHK", "Not a WCHK checkpoint");

    let read_u32 = |f: &mut std::fs::File| -> u32 { let mut b = [0u8; 4]; f.read_exact(&mut b).unwrap(); u32::from_le_bytes(b) };
    let read_u64 = |f: &mut std::fs::File| -> u64 { let mut b = [0u8; 8]; f.read_exact(&mut b).unwrap(); u64::from_le_bytes(b) };
    let read_f32_single = |f: &mut std::fs::File| -> f32 { let mut b = [0u8; 4]; f.read_exact(&mut b).unwrap(); f32::from_le_bytes(b) };

    let version = read_u32(&mut f);
    assert!(version >= 1 && version <= 4, "Unknown WCHK version {version}");

    // Config
    let ck_bands = read_u32(&mut f) as usize;
    let ck_head = read_u32(&mut f) as usize;
    let ck_layers = read_u32(&mut f) as usize;
    let ck_maestro = read_u32(&mut f) as usize;
    let _ck_block_size = read_u32(&mut f) as usize;
    let _ck_rk4 = read_u32(&mut f) as usize;

    // v2+ adds out_proj_groups; v1 defaults to dense (groups=1)
    let out_proj_groups = if version >= 2 {
        read_u32(&mut f) as usize
    } else {
        1 // v1 = dense out_proj
    };

    // v3 adds feature flags — disambiguates param layouts
    let feature_flags = if version >= 3 {
        read_u32(&mut f)
    } else {
        0 // v1/v2: no flags, fall back to file-size detection
    };
    let has_learnable_ode = feature_flags & (1 << 0) != 0;
    let has_layer_scale   = feature_flags & (1 << 1) != 0;
    let has_rk4_weights   = feature_flags & (1 << 2) != 0;
    let has_dyn_harmonics = feature_flags & (1 << 3) != 0;
    let _has_learnable_attn = feature_flags & (1 << 4) != 0;

    // v4: chi (FWM strength) persisted in checkpoint
    let chi = if version >= 4 { read_f32_single(&mut f) } else { 0.0 };
    if version < 4 && chi == 0.0 {
        // v3 and earlier: FWM was not persisted. chi defaults to 0.0.
        // If the user passes --fwm-strength on the CLI, that takes effect on resume.
    }

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

    // Determine param count from file size (authoritative).
    //
    // Data layout after header: adam_m[N] + adam_v[N] + params[N], so data_bytes = N × 12.
    // This is robust to feature-flag gaps — e.g. legacy checkpoints saved with
    // learnable_attn=true but without the corresponding flag bit (fixed in the same
    // commit that added bit 1<<4) still load correctly because we trust file size.
    let n_params = if version >= 3 {
        // magic(4) + ver(4) + config(7×4=28) + flags(4) + chi_if_v4(4) + vocab(8) + iter(8) + lr(4) + rng(8) + adam_t(8)
        let header_size_v3 = 4 + 4 + 8*4 + 8 + 8 + 4 + 8 + 8 + if version >= 4 { 4 } else { 0 };
        let file_len = f.metadata().map(|m| m.len()).unwrap_or(0) as usize;
        let data_bytes = file_len.saturating_sub(header_size_v3);
        let n_from_size = data_bytes / 12;

        // Sanity: flag-based estimate (excluding learnable_attn, which adds per-head
        // attention projections whose size depends on head layout). If the flag-based
        // estimate matches file size exactly, use it verbatim; otherwise trust file size.
        let mut n_flag = ck_layers * per_block + n_embd * 2;
        if has_learnable_ode { n_flag += ck_layers * (ck_bands + 1 + 1 + ck_bands); }
        if has_layer_scale   { n_flag += ck_layers; }
        if has_rk4_weights   { n_flag += ck_layers * 4; }
        if has_dyn_harmonics { n_flag += ck_layers * ck_head; }
        let n_with_lm = n_flag + lm_head_params;
        let n_with_oc = n_flag + ck_bands;

        if data_bytes == n_with_oc * 12 { n_with_oc }
        else if data_bytes == n_with_lm * 12 { n_with_lm }
        else { n_from_size }
    } else {
        // v1/v2: fall back to file-size detection across all variants
        let header_size = 4 + 4 + 7*4 + 8 + 8 + 4 + 8 + 8;
        let file_len = f.metadata().map(|m| m.len()).unwrap_or(0) as usize;
        let data_bytes = file_len.saturating_sub(header_size);
        variants.iter().find(|&&v| data_bytes == v * 12).copied().unwrap_or(n_base)
    };

    let adam_t = read_u64(&mut f) as usize;
    let read_f32_vec = |f: &mut std::fs::File, n: usize| -> Vec<f32> {
        let mut buf = vec![0u8; n * 4];
        f.read_exact(&mut buf).unwrap();
        buf.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let adam_m = read_f32_vec(&mut f, n_params);
    let adam_v = read_f32_vec(&mut f, n_params);
    let params = read_f32_vec(&mut f, n_params);

    if version >= 4 {
        println!("  WCHK v{version}: iter {iter}, lr {lr:.6}, {n_params} params, {ck_layers} layers, {out_proj_groups} groups, flags=0x{feature_flags:02x}, chi={chi:.3}");
    } else if version >= 3 {
        println!("  WCHK v{version}: iter {iter}, lr {lr:.6}, {n_params} params, {ck_layers} layers, {out_proj_groups} groups, flags=0x{feature_flags:02x}");
    } else {
        println!("  WCHK v{version}: iter {iter}, lr {lr:.6}, {n_params} params, {ck_layers} layers, {out_proj_groups} groups");
    }
    (params, vocab_size, iter, lr, rng_state, adam_t, adam_m, adam_v, out_proj_groups, feature_flags, chi)
}
