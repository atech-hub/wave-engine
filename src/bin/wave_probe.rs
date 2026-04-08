//! Wave Probe — characterise the ODE as a pure wave transformation.
//!
//! Run: cargo run --release --bin wave-probe -- --mode all
//!
//! Probes the ODE with controlled inputs (single-band, two-band, noise)
//! and measures what comes out. No loss, no decoder, no training.
//! Pure forward-pass scattering analysis.

// Import the canonical ODE code from the engine (single source of truth)
#[path = "../common/ode_deriv.rs"]
mod ode_deriv;
use ode_deriv::rk4_step_public;

/// Simple LCG random number generator (deterministic, no deps)
struct SimpleRng { state: u64 }
impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: seed | 1 } }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

fn softplus(x: f32) -> f32 { (1.0 + x.exp()).ln() }

struct ProbeCfg {
    n_bands: usize,
    alpha: f32,
    beta: f32,
    gamma_val: f32,
    chi: f32,
    rk4_steps: usize,
    seed: u64,
}

impl ProbeCfg {
    fn gamma(&self) -> Vec<f32> {
        let raw = ((self.gamma_val).exp() - 1.0).ln();
        vec![softplus(raw); self.n_bands]
    }
    fn omega(&self) -> Vec<f32> {
        (0..self.n_bands).map(|k| (k + 1) as f32 * std::f32::consts::PI / self.n_bands as f32).collect()
    }
    fn rk4_w(&self) -> [f32; 4] { [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0] }
    fn dt(&self) -> f32 { 1.0 / self.rk4_steps as f32 }
}

/// Run the full ODE integration (rk4_steps iterations of rk4_step)
fn run_ode(r_in: &[f32], s_in: &[f32], cfg: &ProbeCfg) -> (Vec<f32>, Vec<f32>) {
    let gamma = cfg.gamma();
    let omega = cfg.omega();
    let w = cfg.rk4_w();
    let dt = cfg.dt();
    let mut r = r_in.to_vec();
    let mut s = s_in.to_vec();
    for _ in 0..cfg.rk4_steps {
        let (r_new, s_new) = rk4_step_public(&r, &s, dt, &gamma, &omega, cfg.alpha, cfg.beta, cfg.chi, &w);
        r = r_new;
        s = s_new;
    }
    (r, s)
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
    println!("\n=== Mode: {} ===", mode);
    println!("  n_bands={}  alpha={:.3}  beta={:.3}  gamma={:.3}  chi={:.3}  rk4_steps={}",
        cfg.n_bands, cfg.alpha, cfg.beta, cfg.gamma_val, cfg.chi, cfg.rk4_steps);
}

// ─── Mode 1: Single-band excitation ───

fn run_single_band(cfg: &ProbeCfg) {
    print_header("single-band", cfg);
    for &k in &[5, 20, 40, 60, 78] {
        if k >= cfg.n_bands { continue; }
        let mut r = vec![0.0f32; cfg.n_bands];
        let mut s = vec![0.0f32; cfg.n_bands];
        r[k] = 1.0;
        let (r_out, s_out) = run_ode(&r, &s, cfg);
        let e = energy(&r_out, &s_out);
        let excited = e[k];
        let mut neighbours = Vec::new();
        for &j in &[k.wrapping_sub(2), k.wrapping_sub(1), k + 1, k + 2] {
            if j < cfg.n_bands { neighbours.push((j, e[j])); }
        }
        let neigh_str: Vec<String> = neighbours.iter().map(|(j, v)| format!("k{}={:.6}", j, v)).collect();
        // Max energy on any non-neighbour band
        let mut max_distant = 0.0f32;
        let mut max_distant_k = 0;
        for j in 0..cfg.n_bands {
            if j != k && !neighbours.iter().any(|(nk, _)| *nk == j) {
                if e[j] > max_distant { max_distant = e[j]; max_distant_k = j; }
            }
        }
        let cos = cosine(&r, &s, &r_out, &s_out);
        let total = total_energy(&r_out, &s_out);
        println!("  [k={}] input_energy=1.0000", k);
        println!("    excited_band_energy={:.6}", excited);
        println!("    neighbour_energy: {}", neigh_str.join(", "));
        println!("    max_distant_energy={:.6} (on band {})", max_distant, max_distant_k);
        println!("    total_output_energy={:.6}", total);
        println!("    cos(input, output)={:.6}", cos);
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
        let s = vec![0.0f32; cfg.n_bands];
        r[a] = amp;
        r[b] = amp;
        let (r_out, s_out) = run_ode(&r, &s, cfg);
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
    }
}

// ─── Mode 3: Two-band destructive ───

fn run_two_band_destructive(cfg: &ProbeCfg) {
    print_header("two-band-destructive", cfg);
    let amp = 1.0 / 2.0f32.sqrt();
    for &(a, b) in &[(20usize, 21usize), (20, 40)] {
        if a >= cfg.n_bands || b >= cfg.n_bands { continue; }
        // Constructive
        let mut r_c = vec![0.0f32; cfg.n_bands];
        let s_c = vec![0.0f32; cfg.n_bands];
        r_c[a] = amp; r_c[b] = amp;
        let (rc_out, sc_out) = run_ode(&r_c, &s_c, cfg);
        // Destructive
        let mut r_d = vec![0.0f32; cfg.n_bands];
        let s_d = vec![0.0f32; cfg.n_bands];
        r_d[a] = amp; r_d[b] = -amp;
        let (rd_out, sd_out) = run_ode(&r_d, &s_d, cfg);
        let cos_c = cosine(&r_c, &s_c, &rc_out, &sc_out);
        let cos_d = cosine(&r_d, &s_d, &rd_out, &sd_out);
        let diff = (total_energy(&rc_out, &sc_out) - total_energy(&rd_out, &sd_out)).abs();
        println!("  [bands={},{} constructive vs destructive]", a, b);
        println!("    constructive: total_energy={:.6} cos={:.6}", total_energy(&rc_out, &sc_out), cos_c);
        println!("    destructive:  total_energy={:.6} cos={:.6}", total_energy(&rd_out, &sd_out), cos_d);
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
        // Normalise to unit energy
        let na = total_energy(&ra, &sa).sqrt();
        let nb = total_energy(&rb, &sb).sqrt();
        for k in 0..n { ra[k] /= na; sa[k] /= na; rb[k] /= nb; sb[k] /= nb; }
        // ODE(A), ODE(B)
        let (oa_r, oa_s) = run_ode(&ra, &sa, cfg);
        let (ob_r, ob_s) = run_ode(&rb, &sb, cfg);
        // ODE(A+B)
        let rab_r: Vec<f32> = ra.iter().zip(&rb).map(|(&a, &b)| a + b).collect();
        let rab_s: Vec<f32> = sa.iter().zip(&sb).map(|(&a, &b)| a + b).collect();
        let (oab_r, oab_s) = run_ode(&rab_r, &rab_s, cfg);
        // ODE(A) + ODE(B)
        let sum_r: Vec<f32> = oa_r.iter().zip(&ob_r).map(|(&a, &b)| a + b).collect();
        let sum_s: Vec<f32> = oa_s.iter().zip(&ob_s).map(|(&a, &b)| a + b).collect();
        // ||ODE(A+B) - (ODE(A)+ODE(B))||
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
    println!("  Exciting band {} at different magnitudes:", k);
    for &mag in &[0.1f32, 0.3, 1.0, 3.0, 10.0] {
        let mut r = vec![0.0f32; cfg.n_bands];
        let s = vec![0.0f32; cfg.n_bands];
        r[k] = mag;
        let (r_out, s_out) = run_ode(&r, &s, cfg);
        let e = energy(&r_out, &s_out);
        let cos = cosine(&r, &s, &r_out, &s_out);
        let total = total_energy(&r_out, &s_out);
        let self_frac = e[k] / total.max(1e-10);
        let neigh: f32 = [k.wrapping_sub(1), k.wrapping_sub(2), k+1, k+2].iter()
            .filter(|&&j| j < cfg.n_bands).map(|&j| e[j]).sum();
        println!("  mag={:5.1}: total_E={:.4} self_frac={:.4} neigh_frac={:.4} cos={:.6}",
            mag, total, self_frac, neigh / total.max(1e-10), cos);
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
    // Normalise to unit total energy
    let norm = total_energy(&r, &s).sqrt();
    for k in 0..n { r[k] /= norm; s[k] /= norm; }
    let e_in = energy(&r, &s);
    let (r_out, s_out) = run_ode(&r, &s, cfg);
    let e_out = energy(&r_out, &s_out);
    let cos = cosine(&r, &s, &r_out, &s_out);
    let total_in = total_energy(&r, &s);
    let total_out = total_energy(&r_out, &s_out);
    // Gain ratio per band
    let gains: Vec<f32> = e_in.iter().zip(&e_out).map(|(&i, &o)| o / i.max(1e-10)).collect();
    let avg_gain = gains.iter().sum::<f32>() / n as f32;
    let min_gain = gains.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_gain = gains.iter().cloned().fold(0.0f32, f32::max);
    // Correlation between input and output spectra
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
    // Show first 10 bands
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
    let (r1, s1) = run_ode(&r, &s, cfg);
    let (r2, s2) = run_ode(&r, &s, cfg);
    let max_diff: f32 = r1.iter().zip(&r2).map(|(&a, &b)| (a - b).abs())
        .chain(s1.iter().zip(&s2).map(|(&a, &b)| (a - b).abs()))
        .fold(0.0f32, f32::max);
    println!("  max_abs_difference={:.2e}", max_diff);
    if max_diff < 1e-6 { println!("  PASS (deterministic)"); }
    else { println!("  FAIL (non-deterministic!)"); }
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

    let cfg = ProbeCfg {
        n_bands: parse_usize("--n-bands", 84),
        alpha: parse("--alpha", 0.1),
        beta: parse("--beta", 0.2),
        gamma_val: parse("--gamma", 0.1),
        chi: parse("--fwm-strength", 0.0),
        rk4_steps: parse_usize("--rk4-steps", 16),
        seed: parse_usize("--seed", 42) as u64,
    };

    println!("wave-probe: ODE scattering analysis");

    match mode.as_str() {
        "single-band" => run_single_band(&cfg),
        "two-band-constructive" => run_two_band_constructive(&cfg),
        "two-band-destructive" => run_two_band_destructive(&cfg),
        "linearity-check" => run_linearity_check(&cfg),
        "magnitude-sweep" => run_magnitude_sweep(&cfg),
        "spectral" => run_spectral(&cfg),
        "determinism" => run_determinism(&cfg),
        "all" => {
            run_determinism(&cfg);
            run_single_band(&cfg);
            run_two_band_constructive(&cfg);
            run_two_band_destructive(&cfg);
            run_linearity_check(&cfg);
            run_magnitude_sweep(&cfg);
            run_spectral(&cfg);
        }
        _ => eprintln!("Unknown mode: {}. Use: single-band, two-band-constructive, two-band-destructive, linearity-check, magnitude-sweep, spectral, determinism, all", mode),
    }
}
