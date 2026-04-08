//! Wave Probe — characterise the ODE as a pure wave transformation.
//!
//! Run: cargo run --release --bin wave-probe -- --mode all
//!      cargo run --release --bin wave-probe -- --mode spectral --fwm-strength 0.03 --input-magnitude 1.3
//!      cargo run --release --bin wave-probe -- --mode spectral --load-checkpoint checkpoint.bin
//!
//! Probes the ODE with controlled inputs (single-band, two-band, noise)
//! and measures what comes out. Reports physics decomposition (damping/phase/FWM)
//! for every mode. No loss, no decoder, no training.
//! Pure forward-pass scattering analysis.

// Canonical imports from the wave-engine crate (single source of truth)
use wave_engine::common::ode_deriv::{kerr_derivative_into, rk4_step_public, DerivativeCapture};
use wave_engine::common::checkpoint::load_checkpoint;
use wave_engine::common::math::softplus;
use wave_engine::{WavePacketModel, init_model, unflatten_params_ex, Dims};

/// Simple LCG random number generator (deterministic, no deps)
struct SimpleRng { state: u64 }
impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: seed | 1 } }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

struct ProbeCfg {
    n_bands: usize,
    alpha: f32,
    beta: f32,
    chi: f32,
    rk4_steps: usize,
    seed: u64,
    input_magnitude: f32,
    // Per-band arrays (either uniform from CLI or learned from checkpoint)
    gamma_raw: Vec<f32>,     // raw values before softplus
    omega: Vec<f32>,
    rk4_weights: [f32; 4],
    phase_correction: Vec<f32>,
}

impl ProbeCfg {
    /// Create from CLI flags (uniform weights)
    fn from_cli(n_bands: usize, alpha: f32, beta: f32, gamma_val: f32, chi: f32,
                rk4_steps: usize, seed: u64, input_magnitude: f32) -> Self {
        let raw = ((gamma_val).exp() - 1.0).ln();
        Self {
            n_bands, alpha, beta, chi, rk4_steps, seed, input_magnitude,
            gamma_raw: vec![raw; n_bands],
            omega: (0..n_bands).map(|k| (k + 1) as f32 * std::f32::consts::PI / n_bands as f32).collect(),
            rk4_weights: [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0],
            phase_correction: vec![0.0; n_bands],
        }
    }

    fn gamma(&self) -> Vec<f32> {
        self.gamma_raw.iter().map(|&g| softplus(g)).collect()
    }
    fn rk4_w(&self) -> [f32; 4] { self.rk4_weights }
    fn dt(&self) -> f32 { 1.0 / self.rk4_steps as f32 }
}

// ─── Checkpoint loading (uses canonical load_checkpoint + init_model + unflatten) ───

/// Load checkpoint and extract per-layer KerrWeights using canonical functions.
/// Requires --n-bands, --n-head, --layers to match the checkpoint architecture.
/// Uses load_checkpoint + init_model + unflatten_params_ex — no duplicate parser.
fn load_checkpoint_to_kerr(
    path: &str, n_bands: usize, n_head: usize, n_layers: usize, chi: f32,
) -> Result<Vec<ProbeCfg>, String> {
    let (params, vocab_size, _iter, _lr, _rng, _adam_t, _adam_m, _adam_v, out_proj_groups, flags)
        = load_checkpoint(path);

    let dims = Dims::from_cli(n_bands, n_head, 16, 128, 16);

    let mut model = init_model(vocab_size, 42, n_layers, out_proj_groups, dims, 0.1, 0.2);

    // Apply feature flags from checkpoint so param layout matches
    let has_learnable_ode = flags & (1 << 0) != 0;
    let has_layer_scale   = flags & (1 << 1) != 0;
    let has_rk4_weights   = flags & (1 << 2) != 0;
    let has_dyn_harmonics = flags & (1 << 3) != 0;
    model.learnable_ode = has_learnable_ode;
    if has_layer_scale {
        model.use_layer_scale = true;
        model.layer_scale = vec![1.0; n_layers];
    }
    if has_rk4_weights {
        model.use_rk4_weights = true;
    }
    if has_dyn_harmonics {
        model.use_dyn_harmonics = true;
    }

    // Detect phase-native from param count: phase-native has no lm_head params
    let expected_with_lm = wave_engine::count_trainable_ex(&model, false);
    if params.len() < expected_with_lm {
        model.phase_native = true;
        model.output_corrector = vec![0.0; n_bands];
    }
    unflatten_params_ex(&mut model, &params, false);

    let mut cfgs = Vec::new();
    for (layer_idx, block) in model.blocks.iter().enumerate() {
        let k = &block.ffn.kerr;
        let gamma_mean: f32 = k.gamma_raw.iter().map(|&g| softplus(g)).sum::<f32>() / n_bands as f32;
        println!("  Layer {}: alpha={:.4} beta={:.4} gamma_mean={:.4} chi={:.4}",
            layer_idx, k.alpha, k.beta, gamma_mean, chi);
        cfgs.push(ProbeCfg {
            n_bands,
            alpha: k.alpha,
            beta: k.beta,
            chi,
            rk4_steps: k.rk4_n_steps,
            seed: 42,
            input_magnitude: 1.0, // overridden by caller
            gamma_raw: k.gamma_raw.clone(),
            omega: k.omega.clone(),
            rk4_weights: k.rk4_weights,
            phase_correction: k.phase_correction.clone(),
        });
    }
    Ok(cfgs)
}

// ─── Physics decomposition ───

/// Physics decomposition from the ODE derivative.
/// `phase_frac` combines SPM + XPM + omega rotation (DerivativeCapture groups them
/// as a single phase_dr/phase_ds pair). Splitting SPM from XPM would require
/// separate capture buffers in ode_deriv.rs — deferred until we need the distinction.
struct PhysicsDecomposition {
    damping_frac: f32,
    phase_frac: f32,   // SPM + XPM + omega combined
    fwm_frac: f32,
    total_deriv_norm: f32,
}

impl PhysicsDecomposition {
    fn print(&self, indent: &str) {
        println!("{}physics: damping={:.3} phase={:.3} fwm={:.3} (||dz/dt||={:.4})",
            indent, self.damping_frac, self.phase_frac, self.fwm_frac, self.total_deriv_norm);
    }
}

/// Run the full ODE integration (rk4_steps iterations of rk4_step)
fn run_ode(r_in: &[f32], s_in: &[f32], cfg: &ProbeCfg) -> (Vec<f32>, Vec<f32>) {
    let gamma = cfg.gamma();
    let w = cfg.rk4_w();
    let dt = cfg.dt();
    let mut r = r_in.to_vec();
    let mut s = s_in.to_vec();
    for _ in 0..cfg.rk4_steps {
        let (r_new, s_new) = rk4_step_public(&r, &s, dt, &gamma, &cfg.omega, cfg.alpha, cfg.beta, cfg.chi, &w);
        r = r_new;
        s = s_new;
    }
    (r, s)
}

/// Run the ODE with physics decomposition capture.
/// Returns (r_out, s_out, decomposition).
///
/// Decomposition is sampled from the k1 derivative at each RK4 step.
/// This is a close approximation to the full weighted sum — k1 is evaluated
/// at the current state and dominates the RK4 combination. Capturing all
/// four substeps would require threading DerivativeCapture through rk4_step_public,
/// deferred until we need that precision.
fn run_ode_with_decomposition(r_in: &[f32], s_in: &[f32], cfg: &ProbeCfg) -> (Vec<f32>, Vec<f32>, PhysicsDecomposition) {
    let gamma = cfg.gamma();
    let w = cfg.rk4_w();
    let dt = cfg.dt();
    let n = cfg.n_bands;

    let mut r = r_in.to_vec();
    let mut s = s_in.to_vec();

    // Accumulators for decomposition (sum of squares across all steps)
    let mut total_damping_sq = 0.0f32;
    let mut total_phase_sq = 0.0f32;
    let mut total_fwm_sq = 0.0f32;
    let mut total_deriv_sq = 0.0f32;

    // Capture buffers
    let mut damp_dr = vec![0.0f32; n]; let mut damp_ds = vec![0.0f32; n];
    let mut phase_dr = vec![0.0f32; n]; let mut phase_ds = vec![0.0f32; n];
    let mut fwm_dr = vec![0.0f32; n]; let mut fwm_ds = vec![0.0f32; n];
    let mut k1r = vec![0.0f32; n]; let mut k1s = vec![0.0f32; n];

    for _ in 0..cfg.rk4_steps {
        // Zero capture buffers for this step
        for i in 0..n { damp_dr[i] = 0.0; damp_ds[i] = 0.0; }
        for i in 0..n { phase_dr[i] = 0.0; phase_ds[i] = 0.0; }
        for i in 0..n { fwm_dr[i] = 0.0; fwm_ds[i] = 0.0; }

        // Capture k1 derivative with decomposition
        {
            let mut cap = DerivativeCapture {
                damping_dr: &mut damp_dr, damping_ds: &mut damp_ds,
                phase_dr: &mut phase_dr, phase_ds: &mut phase_ds,
                fwm_dr: &mut fwm_dr, fwm_ds: &mut fwm_ds,
            };
            kerr_derivative_into(&r, &s, &gamma, &cfg.omega, cfg.alpha, cfg.beta, cfg.chi, &mut k1r, &mut k1s, Some(&mut cap));
        }

        // Accumulate norms from k1 (representative of this step)
        total_damping_sq += damp_dr.iter().map(|x| x*x).sum::<f32>() + damp_ds.iter().map(|x| x*x).sum::<f32>();
        total_phase_sq += phase_dr.iter().map(|x| x*x).sum::<f32>() + phase_ds.iter().map(|x| x*x).sum::<f32>();
        total_fwm_sq += fwm_dr.iter().map(|x| x*x).sum::<f32>() + fwm_ds.iter().map(|x| x*x).sum::<f32>();
        total_deriv_sq += k1r.iter().map(|x| x*x).sum::<f32>() + k1s.iter().map(|x| x*x).sum::<f32>();

        // Full RK4 step for state update
        let (r_new, s_new) = rk4_step_public(&r, &s, dt, &gamma, &cfg.omega, cfg.alpha, cfg.beta, cfg.chi, &w);
        r = r_new;
        s = s_new;
    }

    let total_norm = total_deriv_sq.sqrt();
    let inv = if total_norm > 1e-20 { 1.0 / total_norm } else { 0.0 };

    let decomp = PhysicsDecomposition {
        damping_frac: total_damping_sq.sqrt() * inv,
        phase_frac: total_phase_sq.sqrt() * inv,
        fwm_frac: total_fwm_sq.sqrt() * inv,
        total_deriv_norm: total_norm,
    };

    (r, s, decomp)
}

fn energy(r: &[f32], s: &[f32]) -> Vec<f32> {
    r.iter().zip(s).map(|(&ri, &si)| ri * ri + si * si).collect()
}

fn total_energy(r: &[f32], s: &[f32]) -> f32 {
    energy(r, s).iter().sum()
}

fn cosine(r1: &[f32], s1: &[f32], r2: &[f32], s2: &[f32]) -> f32 {
    let dot: f32 = r1.iter().zip(r2).map(|(&a, &b)| a * b).sum::<f32>()
        + s1.iter().zip(s2).map(|(&a, &b)| a * b).sum::<f32>();
    let n1 = total_energy(r1, s1).sqrt();
    let n2 = total_energy(r2, s2).sqrt();
    if n1 > 1e-10 && n2 > 1e-10 { dot / (n1 * n2) } else { 0.0 }
}

fn print_header(mode: &str, cfg: &ProbeCfg) {
    let gamma_mean: f32 = cfg.gamma_raw.iter().map(|&g| softplus(g)).sum::<f32>() / cfg.n_bands as f32;
    println!("\n=== Mode: {} ===", mode);
    println!("  n_bands={}  alpha={:.3}  beta={:.3}  gamma_mean={:.3}  chi={:.3}  rk4_steps={}  input_mag={:.2}",
        cfg.n_bands, cfg.alpha, cfg.beta, gamma_mean, cfg.chi, cfg.rk4_steps, cfg.input_magnitude);
}

/// Apply input_magnitude scaling to r,s vectors
fn scale_input(r: &mut [f32], s: &mut [f32], mag: f32) {
    if (mag - 1.0).abs() > 1e-6 {
        for v in r.iter_mut() { *v *= mag; }
        for v in s.iter_mut() { *v *= mag; }
    }
}

// ─── Mode 1: Single-band excitation ───

fn run_single_band(cfg: &ProbeCfg) {
    print_header("single-band", cfg);
    for &k in &[5, 20, 40, 60, 78] {
        if k >= cfg.n_bands { continue; }
        let mut r = vec![0.0f32; cfg.n_bands];
        let mut s = vec![0.0f32; cfg.n_bands];
        r[k] = 1.0;
        scale_input(&mut r, &mut s, cfg.input_magnitude);
        let (r_out, s_out, decomp) = run_ode_with_decomposition(&r, &s, cfg);
        let e = energy(&r_out, &s_out);
        let excited = e[k];
        let mut neighbours = Vec::new();
        for &j in &[k.wrapping_sub(2), k.wrapping_sub(1), k + 1, k + 2] {
            if j < cfg.n_bands { neighbours.push((j, e[j])); }
        }
        let neigh_str: Vec<String> = neighbours.iter().map(|(j, v)| format!("k{}={:.6}", j, v)).collect();
        let mut max_distant = 0.0f32;
        let mut max_distant_k = 0;
        for j in 0..cfg.n_bands {
            if j != k && !neighbours.iter().any(|(nk, _)| *nk == j) {
                if e[j] > max_distant { max_distant = e[j]; max_distant_k = j; }
            }
        }
        let cos = cosine(&r, &s, &r_out, &s_out);
        let total = total_energy(&r_out, &s_out);
        println!("  [k={}] input_energy={:.4}", k, total_energy(&r, &s));
        println!("    excited_band_energy={:.6}", excited);
        println!("    neighbour_energy: {}", neigh_str.join(", "));
        println!("    max_distant_energy={:.6} (on band {})", max_distant, max_distant_k);
        println!("    total_output_energy={:.6}", total);
        println!("    cos(input, output)={:.6}", cos);
        decomp.print("    ");
    }
}

// ─── Mode 2: Two-band constructive ───

fn run_two_band_constructive(cfg: &ProbeCfg) {
    print_header("two-band-constructive", cfg);
    let cases = vec![
        (20, 40, "distant (20 apart)"),
        (20, 21, "adjacent"),
        (10, 70, "far apart (60 apart)"),
    ];
    let amp = 1.0 / 2.0f32.sqrt();
    for (a, b, label) in cases {
        if a >= cfg.n_bands || b >= cfg.n_bands { continue; }
        let mut r = vec![0.0f32; cfg.n_bands];
        let mut s = vec![0.0f32; cfg.n_bands];
        r[a] = amp;
        r[b] = amp;
        scale_input(&mut r, &mut s, cfg.input_magnitude);
        let (r_out, s_out, decomp) = run_ode_with_decomposition(&r, &s, cfg);
        let e = energy(&r_out, &s_out);
        let cos = cosine(&r, &s, &r_out, &s_out);
        let total = total_energy(&r_out, &s_out);
        let non_excited: f32 = e.iter().enumerate()
            .filter(|&(k, _)| k != a && k != b)
            .map(|(_, &v)| v).sum();
        println!("  [bands={},{} ({})] in-phase", a, b, label);
        println!("    band_{}_energy={:.6}  band_{}_energy={:.6}", a, e[a], b, e[b]);
        println!("    non_excited_energy={:.6} ({:.2}% of total)", non_excited, 100.0 * non_excited / total);
        println!("    total_output_energy={:.6}", total);
        println!("    cos(input, output)={:.6}", cos);
        decomp.print("    ");
    }
}

// ─── Mode 3: Two-band destructive ───

fn run_two_band_destructive(cfg: &ProbeCfg) {
    print_header("two-band-destructive", cfg);
    let amp = 1.0 / 2.0f32.sqrt();
    for &(a, b) in &[(20usize, 21usize), (20, 40)] {
        if a >= cfg.n_bands || b >= cfg.n_bands { continue; }
        let mut r_c = vec![0.0f32; cfg.n_bands];
        let mut s_c = vec![0.0f32; cfg.n_bands];
        r_c[a] = amp; r_c[b] = amp;
        scale_input(&mut r_c, &mut s_c, cfg.input_magnitude);
        let (rc_out, sc_out, decomp_c) = run_ode_with_decomposition(&r_c, &s_c, cfg);
        let mut r_d = vec![0.0f32; cfg.n_bands];
        let mut s_d = vec![0.0f32; cfg.n_bands];
        r_d[a] = amp; r_d[b] = -amp;
        scale_input(&mut r_d, &mut s_d, cfg.input_magnitude);
        let (rd_out, sd_out, decomp_d) = run_ode_with_decomposition(&r_d, &s_d, cfg);
        let cos_c = cosine(&r_c, &s_c, &rc_out, &sc_out);
        let cos_d = cosine(&r_d, &s_d, &rd_out, &sd_out);
        let diff = (total_energy(&rc_out, &sc_out) - total_energy(&rd_out, &sd_out)).abs();
        println!("  [bands={},{} constructive vs destructive]", a, b);
        println!("    constructive: total_energy={:.6} cos={:.6}", total_energy(&rc_out, &sc_out), cos_c);
        decomp_c.print("      ");
        println!("    destructive:  total_energy={:.6} cos={:.6}", total_energy(&rd_out, &sd_out), cos_d);
        decomp_d.print("      ");
        println!("    energy_difference={:.6} (0=sign-blind, >0=sign-sensitive)", diff);
    }
}

// ─── Mode 4: Linearity check ───

fn run_linearity_check(cfg: &ProbeCfg) {
    print_header("linearity-check", cfg);
    let mut rng = SimpleRng::new(cfg.seed);
    let mut ratios = Vec::new();
    for trial in 0..10 {
        let n = cfg.n_bands;
        let mut ra = vec![0.0f32; n]; let mut sa = vec![0.0f32; n];
        let mut rb = vec![0.0f32; n]; let mut sb = vec![0.0f32; n];
        for k in 0..n { ra[k] = rng.next_f32(); sa[k] = rng.next_f32(); }
        for k in 0..n { rb[k] = rng.next_f32(); sb[k] = rng.next_f32(); }
        let na = total_energy(&ra, &sa).sqrt();
        let nb = total_energy(&rb, &sb).sqrt();
        for k in 0..n { ra[k] /= na; sa[k] /= na; rb[k] /= nb; sb[k] /= nb; }
        scale_input(&mut ra, &mut sa, cfg.input_magnitude);
        scale_input(&mut rb, &mut sb, cfg.input_magnitude);
        let (oa_r, oa_s) = run_ode(&ra, &sa, cfg);
        let (ob_r, ob_s) = run_ode(&rb, &sb, cfg);
        let rab_r: Vec<f32> = ra.iter().zip(&rb).map(|(&a, &b)| a + b).collect();
        let rab_s: Vec<f32> = sa.iter().zip(&sb).map(|(&a, &b)| a + b).collect();
        let (oab_r, oab_s) = run_ode(&rab_r, &rab_s, cfg);
        let sum_r: Vec<f32> = oa_r.iter().zip(&ob_r).map(|(&a, &b)| a + b).collect();
        let sum_s: Vec<f32> = oa_s.iter().zip(&ob_s).map(|(&a, &b)| a + b).collect();
        let diff_norm: f32 = oab_r.iter().zip(&sum_r).map(|(&a, &b)| (a - b) * (a - b)).sum::<f32>()
            + oab_s.iter().zip(&sum_s).map(|(&a, &b)| (a - b) * (a - b)).sum::<f32>();
        let sum_norm: f32 = total_energy(&sum_r, &sum_s);
        let ratio = diff_norm.sqrt() / sum_norm.sqrt().max(1e-10);
        ratios.push(ratio);
        if trial < 3 {
            println!("  trial {}: nonlinearity_ratio={:.6} (0=linear, 1=fully nonlinear)", trial, ratio);
        }
    }
    let avg: f32 = ratios.iter().sum::<f32>() / ratios.len() as f32;
    let max: f32 = ratios.iter().cloned().fold(0.0f32, f32::max);
    println!("  avg_nonlinearity={:.6}  max={:.6} (over 10 trials)", avg, max);
}

// ─── Mode 5: Magnitude sweep ───

fn run_magnitude_sweep(cfg: &ProbeCfg) {
    print_header("magnitude-sweep", cfg);
    let k = 40.min(cfg.n_bands - 1);
    println!("  Exciting band {} at different magnitudes (x input_mag={:.2}):", k, cfg.input_magnitude);
    for &mag in &[0.1f32, 0.3, 1.0, 3.0, 10.0] {
        let effective_mag = mag * cfg.input_magnitude;
        let mut r = vec![0.0f32; cfg.n_bands];
        let s = vec![0.0f32; cfg.n_bands];
        r[k] = effective_mag;
        let (r_out, s_out, decomp) = run_ode_with_decomposition(&r, &s, cfg);
        let e = energy(&r_out, &s_out);
        let cos = cosine(&r, &s, &r_out, &s_out);
        let total = total_energy(&r_out, &s_out);
        let self_frac = e[k] / total.max(1e-10);
        let neigh: f32 = [k.wrapping_sub(1), k.wrapping_sub(2), k+1, k+2].iter()
            .filter(|&&j| j < cfg.n_bands).map(|&j| e[j]).sum();
        println!("  mag={:5.1} (eff={:.2}): total_E={:.4} self={:.4} neigh={:.4} cos={:.6} fwm={:.3}",
            mag, effective_mag, total, self_frac, neigh / total.max(1e-10), cos, decomp.fwm_frac);
    }
}

// ─── Mode 6: Spectral response ───

fn run_spectral(cfg: &ProbeCfg) {
    print_header("spectral", cfg);
    let mut rng = SimpleRng::new(cfg.seed);
    let n = cfg.n_bands;
    let mut r = vec![0.0f32; n];
    let mut s = vec![0.0f32; n];
    for k in 0..n { r[k] = rng.next_f32(); s[k] = rng.next_f32(); }
    let norm = total_energy(&r, &s).sqrt();
    for k in 0..n { r[k] /= norm; s[k] /= norm; }
    scale_input(&mut r, &mut s, cfg.input_magnitude);
    let e_in = energy(&r, &s);
    let (r_out, s_out, decomp) = run_ode_with_decomposition(&r, &s, cfg);
    let e_out = energy(&r_out, &s_out);
    let cos = cosine(&r, &s, &r_out, &s_out);
    let total_in = total_energy(&r, &s);
    let total_out = total_energy(&r_out, &s_out);
    let gains: Vec<f32> = e_in.iter().zip(&e_out).map(|(&i, &o)| o / i.max(1e-10)).collect();
    let avg_gain = gains.iter().sum::<f32>() / n as f32;
    let min_gain = gains.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_gain = gains.iter().cloned().fold(0.0f32, f32::max);
    let mean_in = e_in.iter().sum::<f32>() / n as f32;
    let mean_out = e_out.iter().sum::<f32>() / n as f32;
    let cov: f32 = e_in.iter().zip(&e_out).map(|(&i, &o)| (i - mean_in) * (o - mean_out)).sum::<f32>();
    let var_in: f32 = e_in.iter().map(|&i| (i - mean_in) * (i - mean_in)).sum::<f32>();
    let var_out: f32 = e_out.iter().map(|&o| (o - mean_out) * (o - mean_out)).sum::<f32>();
    let corr = cov / (var_in.sqrt() * var_out.sqrt()).max(1e-10);
    println!("  total_energy: in={:.6} out={:.6} ratio={:.6}", total_in, total_out, total_out / total_in);
    println!("  per_band_gain: avg={:.4} min={:.4} max={:.4}", avg_gain, min_gain, max_gain);
    println!("  spectrum_correlation={:.6} (1=identity, 0=reshuffled)", corr);
    println!("  cos(input, output)={:.6}", cos);
    decomp.print("  ");
    println!("  first_10_bands (in_energy -> out_energy [gain]):");
    for k in 0..10.min(n) {
        println!("    band {}: {:.6} -> {:.6} [{:.2}x]", k, e_in[k], e_out[k], gains[k]);
    }
}

// ─── Mode 7: Determinism check ───

fn run_determinism(cfg: &ProbeCfg) {
    print_header("determinism", cfg);
    let mut rng = SimpleRng::new(cfg.seed);
    let n = cfg.n_bands;
    let mut r = vec![0.0f32; n]; let mut s = vec![0.0f32; n];
    for k in 0..n { r[k] = rng.next_f32(); s[k] = rng.next_f32(); }
    scale_input(&mut r, &mut s, cfg.input_magnitude);
    let (r1, s1) = run_ode(&r, &s, cfg);
    let (r2, s2) = run_ode(&r, &s, cfg);
    let max_diff: f32 = r1.iter().zip(&r2).map(|(&a, &b)| (a - b).abs())
        .chain(s1.iter().zip(&s2).map(|(&a, &b)| (a - b).abs()))
        .fold(0.0f32, f32::max);
    println!("  max_abs_difference={:.2e}", max_diff);
    if max_diff < 1e-6 { println!("  PASS (deterministic)"); }
    else { println!("  FAIL (non-deterministic!)"); }
}

// ─── Mode 8: Energy conservation test ───

/// Energy conservation threshold: 1e-3 (0.1%)
///
/// RK4 truncation error on this ODE varies with amplitude from ~1e-6
/// at low magnitudes to ~4e-3 at mag=3.0 (non-uniform power law due to mixed
/// linear+cubic contributions to the 5th derivative). Empirically measured
/// at 84 bands, rk4_steps=16:
///   mag=0.5 → 2.3e-6,  mag=1.0 → 5.8e-6,  mag=1.3 → 1.3e-5,
///   mag=2.0 → 1.0e-4,  mag=3.0 → 3.8e-3
/// Doubling rk4_steps drives error below f32 precision (O(dt^5) confirmed).
///
/// A flat 1e-3 threshold:
///   - PASSES at magnitudes 0.5-2.0 with 10-400x margin
///   - CATCHES real physics bugs: a missing quartet term (1.09% drift, 2026-04-07)
///     is >10x above this threshold
///   - If this fails at default settings (mag≤1.3), investigate the ODE, not the threshold
const ENERGY_THRESHOLD: f32 = 1e-3;

fn run_energy_conservation(cfg: &ProbeCfg) {
    print_header("energy-conservation", cfg);
    let mut rng = SimpleRng::new(cfg.seed);
    let n = cfg.n_bands;

    // Test 1: gamma=0, chi=0 — pure Hamiltonian, should conserve exactly
    println!("  [test 1] gamma=0, chi=0 (pure Hamiltonian)");
    let mut r = vec![0.0f32; n]; let mut s = vec![0.0f32; n];
    for k in 0..n { r[k] = rng.next_f32(); s[k] = rng.next_f32(); }
    scale_input(&mut r, &mut s, cfg.input_magnitude);
    let e_in = total_energy(&r, &s);
    let gamma_zero: Vec<f32> = vec![0.0f32; n];
    let omega = &cfg.omega;
    let w = cfg.rk4_w();
    let dt = cfg.dt();
    let mut r1 = r.clone(); let mut s1 = s.clone();
    for _ in 0..cfg.rk4_steps {
        let (rn, sn) = rk4_step_public(&r1, &s1, dt, &gamma_zero, &omega, cfg.alpha, cfg.beta, 0.0, &w);
        r1 = rn; s1 = sn;
    }
    let e_out1 = total_energy(&r1, &s1);
    let err1 = (e_out1 - e_in).abs() / e_in;
    let margin1 = if err1 > 1e-20 { ENERGY_THRESHOLD / err1 } else { f32::INFINITY };
    println!("    energy: in={:.8} out={:.8} err={:.2e} {} (margin: {:.0}x)",
        e_in, e_out1, err1, if err1 < ENERGY_THRESHOLD { "PASS" } else { "FAIL" }, margin1);

    // Test 2: gamma=0, chi>0 — FWM should also conserve (Hamiltonian)
    println!("  [test 2] gamma=0, chi={:.3} (FWM Hamiltonian)", cfg.chi);
    let mut r2 = r.clone(); let mut s2 = s.clone();
    for _ in 0..cfg.rk4_steps {
        let (rn, sn) = rk4_step_public(&r2, &s2, dt, &gamma_zero, &omega, cfg.alpha, cfg.beta, cfg.chi, &w);
        r2 = rn; s2 = sn;
    }
    let e_out2 = total_energy(&r2, &s2);
    let err2 = (e_out2 - e_in).abs() / e_in;
    let margin2 = if err2 > 1e-20 { ENERGY_THRESHOLD / err2 } else { f32::INFINITY };
    println!("    energy: in={:.8} out={:.8} err={:.2e} {} (margin: {:.0}x)",
        e_in, e_out2, err2, if err2 < ENERGY_THRESHOLD { "PASS" } else { "FAIL" }, margin2);

    // Test 3: full physics — energy should decrease (damping)
    println!("  [test 3] full physics (damping active)");
    let (r3, s3) = run_ode(&r, &s, cfg);
    let e_out3 = total_energy(&r3, &s3);
    let ratio = e_out3 / e_in;
    println!("    energy: in={:.8} out={:.8} ratio={:.6} {}", e_in, e_out3, ratio,
        if ratio < 1.0 { "PASS (damping reduces energy)" } else { "UNEXPECTED (energy grew)" });
}

// ─── Run modes on a config ───

fn run_mode(mode: &str, cfg: &ProbeCfg) {
    match mode {
        "single-band" => run_single_band(cfg),
        "two-band-constructive" => run_two_band_constructive(cfg),
        "two-band-destructive" => run_two_band_destructive(cfg),
        "linearity-check" => run_linearity_check(cfg),
        "magnitude-sweep" => run_magnitude_sweep(cfg),
        "spectral" => run_spectral(cfg),
        "determinism" => run_determinism(cfg),
        "energy-conservation" => run_energy_conservation(cfg),
        "all" => {
            run_determinism(cfg);
            run_single_band(cfg);
            run_two_band_constructive(cfg);
            run_two_band_destructive(cfg);
            run_linearity_check(cfg);
            run_magnitude_sweep(cfg);
            run_spectral(cfg);
            run_energy_conservation(cfg);
        }
        _ => eprintln!("Unknown mode: {}. Use: single-band, two-band-constructive, two-band-destructive, linearity-check, magnitude-sweep, spectral, determinism, energy-conservation, all", mode),
    }
}

// ─── Main ───

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.iter().position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "all".to_string());

    let parse = |name: &str, default: f32| -> f32 {
        args.iter().position(|a| a == name)
            .and_then(|i| args.get(i + 1)?.parse().ok())
            .unwrap_or(default)
    };
    let parse_usize = |name: &str, default: usize| -> usize {
        args.iter().position(|a| a == name)
            .and_then(|i| args.get(i + 1)?.parse().ok())
            .unwrap_or(default)
    };
    let parse_str = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };

    let input_magnitude = parse("--input-magnitude", 1.0);
    let seed = parse_usize("--seed", 42) as u64;
    let chi = parse("--fwm-strength", 0.0);

    println!("wave-probe: ODE scattering analysis");

    // Checkpoint mode: load learned weights per layer from a trained model
    if let Some(ckpt_path) = parse_str("--load-checkpoint") {
        let n_bands = parse_usize("--n-bands", 84);
        let n_head = parse_usize("--n-head", 4);
        let n_layers = parse_usize("--layers", 4);

        match load_checkpoint_to_kerr(&ckpt_path, n_bands, n_head, n_layers, chi) {
            Ok(layer_cfgs) => {
                for (layer_idx, mut cfg) in layer_cfgs.into_iter().enumerate() {
                    cfg.input_magnitude = input_magnitude;
                    cfg.seed = seed;
                    println!("\n========== Layer {} — learned weights ==========", layer_idx);
                    run_mode(&mode, &cfg);
                }
            }
            Err(e) => {
                eprintln!("Error loading checkpoint: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    // CLI uniform weights mode (default)
    let cfg = ProbeCfg::from_cli(
        parse_usize("--n-bands", 84),
        parse("--alpha", 0.1),
        parse("--beta", 0.2),
        parse("--gamma", 0.1),
        chi,
        parse_usize("--rk4-steps", 16),
        seed,
        input_magnitude,
    );

    run_mode(&mode, &cfg);
}
