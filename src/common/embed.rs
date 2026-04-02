//! Wave packet embedding — tokens as phase positions on harmonic circles.
//!
//! Multi-grid: two coprime moduli, each grid gets half the bands.
//! Tokens that collide on grid 1 are scattered on grid 2.
//! Same principle as Chinese Sexagenary (10 stems × 12 branches = 60 unique).
//!
//! Validated: 96x separation improvement at 2K vocab, 11,800x at 50K vocab.
//! See investigations/multi-grid/ in the research repo.

use std::f32::consts::PI;

/// Find two coprime moduli near sqrt(vocab_size) whose product ≥ vocab_size.
/// Uses the Sexagenary principle: small incommensurate grids cover more space
/// than one large grid. Adjacent tokens on grid 1 are scattered on grid 2.
fn find_coprime_moduli(vocab_size: usize) -> (usize, usize) {
    let root = (vocab_size as f64).sqrt().ceil() as usize;
    // Find two coprimes near root whose product >= vocab_size
    let mut m1 = root;
    // Make m1 odd to help coprimality
    if m1 % 2 == 0 { m1 += 1; }
    let mut m2 = m1 + 2; // start slightly above m1
    // Ensure coprime (gcd = 1)
    while gcd(m1, m2) != 1 || m1 * m2 < vocab_size {
        m2 += 1;
    }
    (m1, m2)
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

/// Build multi-grid harmonic embedding table.
/// Grid 1 (half bands): token_id mod m1 → phase on m1-circle
/// Grid 2 (half bands): token_id mod m2 → phase on m2-circle
/// where m1, m2 are coprime and m1 × m2 ≥ vocab_size.
pub fn build_harmonic_table(vocab_size: usize, n_bands: usize) -> Vec<Vec<f32>> {
    build_harmonic_table_with_moduli(vocab_size, n_bands, None, None)
}

/// Pythagorean sphere encoding: magnitude decays as 1/sqrt(n+1).
/// Fundamental band loudest, harmonics decay like physical waves.
/// Same total energy per token (normalised after construction).
pub fn build_harmonic_table_pythagorean(vocab_size: usize, n_bands: usize) -> Vec<Vec<f32>> {
    let mut table = build_harmonic_table_with_moduli(vocab_size, n_bands, None, None);
    let half = n_bands / 2;
    for emb in &mut table {
        // Apply Pythagorean magnitude profile to both grids
        for n in 0..half {
            let mag = 1.0 / ((n + 1) as f32).sqrt();
            emb[n * 2] *= mag;
            emb[n * 2 + 1] *= mag;
        }
        for n in 0..half {
            let idx = half + n;
            let mag = 1.0 / ((n + 1) as f32).sqrt();
            emb[idx * 2] *= mag;
            emb[idx * 2 + 1] *= mag;
        }
        // Normalise to same total energy as flat (sum of mag² = n_bands for flat)
        let energy: f32 = (0..n_bands).map(|k| emb[k*2]*emb[k*2] + emb[k*2+1]*emb[k*2+1]).sum();
        let scale = (n_bands as f32 / energy.max(1e-8)).sqrt();
        for j in 0..n_bands*2 { emb[j] *= scale; }
    }
    table
}

pub fn build_harmonic_table_with_moduli(vocab_size: usize, n_bands: usize, m1_override: Option<usize>, m2_override: Option<usize>) -> Vec<Vec<f32>> {
    let n_embd = n_bands * 2;
    let half = n_bands / 2;
    let (m1, m2) = match (m1_override, m2_override) {
        (Some(a), Some(b)) => {
            assert!(gcd(a, b) == 1, "m1={a} and m2={b} must be coprime (gcd=1)");
            eprintln!("  [embed] Multi-grid (manual): m1={a}, m2={b}, lcm_coverage={}, vocab={vocab_size}", a * b);
            (a, b)
        }
        (None, None) => find_coprime_moduli(vocab_size),
        _ => panic!("--m1 and --m2 must both be provided or both omitted"),
    };

    // Log the grid configuration
    eprintln!("  [embed] Multi-grid: m1={}, m2={}, lcm_coverage={}, vocab={}", m1, m2, m1 * m2, vocab_size);

    (0..vocab_size).map(|tok| {
        let mut emb = vec![0.0f32; n_embd];
        // Grid 1: tok mod m1 on m1-circle
        let theta1 = (tok % m1) as f32 * 2.0 * PI / m1 as f32;
        for n in 0..half {
            let phase = (n + 1) as f32 * theta1;
            emb[n * 2] = phase.cos();
            emb[n * 2 + 1] = phase.sin();
        }
        // Grid 2: tok mod m2 on m2-circle
        let theta2 = (tok % m2) as f32 * 2.0 * PI / m2 as f32;
        for n in 0..half {
            let idx = half + n;
            let phase = (n + 1) as f32 * theta2;
            emb[idx * 2] = phase.cos();
            emb[idx * 2 + 1] = phase.sin();
        }
        emb
    }).collect()
}

/// Build positional encoding table: position → phase offset.
/// Uses sinusoidal encoding (standard transformer) on the harmonic circle.
pub fn build_positional_table(block_size: usize, n_bands: usize) -> Vec<Vec<f32>> {
    let n_embd = n_bands * 2;
    (0..block_size).map(|pos| {
        let mut pe = vec![0.0f32; n_embd];
        for n in 0..n_bands {
            let freq = 1.0 / (10000.0f32).powf(2.0 * n as f32 / n_embd as f32);
            pe[n * 2] = (pos as f32 * freq).sin();
            pe[n * 2 + 1] = (pos as f32 * freq).cos();
        }
        pe
    }).collect()
}

/// Embed tokens: look up harmonic table + add positional encoding.
pub fn embed_tokens(
    tokens: &[usize],
    wte: &[Vec<f32>],
    wpe: &[Vec<f32>],
    n_embd: usize,
) -> Vec<Vec<f32>> {
    tokens.iter().enumerate().map(|(pos, &tok)| {
        let mut h = vec![0.0f32; n_embd];
        for i in 0..n_embd {
            h[i] = wte[tok][i] + wpe[pos][i];
        }
        h
    }).collect()
}
