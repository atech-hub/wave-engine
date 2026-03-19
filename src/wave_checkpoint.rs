//! WCHK checkpoint format — save/load trained model weights + optimizer state.
//!
//! Format: magic "WCHK" + version + config + metadata + optimizer + params.
//! Supports bit-perfect resume (Adam m/v/t + RNG state preserved).

use crate::{N_BANDS, N_EMBD, N_HEAD, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS};
use crate::train::Adam;

/// Save model checkpoint in WCHK format.
pub fn save_checkpoint(
    params: &[f32],
    vocab_size: usize,
    n_layers: usize,
    iter: usize,
    lr: f32,
    optimizer: &Adam,
    rng_state: u64,
    path: &str,
) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Failed to create checkpoint file");

    // Magic + version
    f.write_all(b"WCHK").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();

    // Config (self-describing)
    f.write_all(&(N_BANDS as u32).to_le_bytes()).unwrap();
    f.write_all(&(N_HEAD as u32).to_le_bytes()).unwrap();
    f.write_all(&(n_layers as u32).to_le_bytes()).unwrap();
    f.write_all(&(MAESTRO_DIM as u32).to_le_bytes()).unwrap();
    f.write_all(&(BLOCK_SIZE as u32).to_le_bytes()).unwrap();
    f.write_all(&(RK4_STEPS as u32).to_le_bytes()).unwrap();

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

    let file_size = 4 + 4 + 6*4 + 8 + 8 + 4 + 8 + 8 + n*4*2 + n*4;
    println!("  WCHK: {n} params, {n_layers} layers, {:.1}MB", file_size as f64 / 1e6);
}

/// Load checkpoint for resume.
/// Returns (params, vocab_size, iter, lr, rng_state, adam_t, adam_m, adam_v).
pub fn load_checkpoint(path: &str) -> (Vec<f32>, usize, usize, f32, u64, usize, Vec<f32>, Vec<f32>) {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("Failed to open {path}: {e}"));

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"WCHK", "Not a WCHK checkpoint");

    let read_u32 = |f: &mut std::fs::File| -> u32 { let mut b = [0u8; 4]; f.read_exact(&mut b).unwrap(); u32::from_le_bytes(b) };
    let read_u64 = |f: &mut std::fs::File| -> u64 { let mut b = [0u8; 8]; f.read_exact(&mut b).unwrap(); u64::from_le_bytes(b) };
    let read_f32_single = |f: &mut std::fs::File| -> f32 { let mut b = [0u8; 4]; f.read_exact(&mut b).unwrap(); f32::from_le_bytes(b) };

    let version = read_u32(&mut f);
    assert_eq!(version, 1, "Unknown WCHK version");

    let ck_bands = read_u32(&mut f) as usize;
    let ck_head = read_u32(&mut f) as usize;
    let ck_layers = read_u32(&mut f) as usize;
    let ck_maestro = read_u32(&mut f) as usize;
    let _ck_block_size = read_u32(&mut f) as usize;
    let _ck_rk4 = read_u32(&mut f) as usize;

    assert_eq!(ck_bands, N_BANDS, "Checkpoint bands mismatch");
    assert_eq!(ck_head, N_HEAD, "Checkpoint heads mismatch");
    assert_eq!(ck_maestro, MAESTRO_DIM, "Checkpoint maestro_dim mismatch");

    let vocab_size = read_u64(&mut f) as usize;
    let iter = read_u64(&mut f) as usize;
    let lr = read_f32_single(&mut f);
    let rng_state = read_u64(&mut f);

    // Compute param count
    let per_block = N_EMBD*2 + N_EMBD*2 + MAESTRO_DIM*N_EMBD + MAESTRO_DIM + N_EMBD*MAESTRO_DIM + N_EMBD
        + MAESTRO_DIM*N_EMBD + MAESTRO_DIM + N_EMBD*MAESTRO_DIM + N_EMBD + N_EMBD*N_EMBD + N_EMBD;
    let n_params = ck_layers * per_block + N_EMBD*2 + vocab_size*N_EMBD;

    let adam_t = read_u64(&mut f) as usize;
    let read_f32_vec = |f: &mut std::fs::File, n: usize| -> Vec<f32> {
        let mut buf = vec![0u8; n * 4];
        f.read_exact(&mut buf).unwrap();
        buf.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let adam_m = read_f32_vec(&mut f, n_params);
    let adam_v = read_f32_vec(&mut f, n_params);
    let params = read_f32_vec(&mut f, n_params);

    println!("  Resumed: iter {iter}, lr {lr:.6}, {n_params} params, {ck_layers} layers");
    (params, vocab_size, iter, lr, rng_state, adam_t, adam_m, adam_v)
}
