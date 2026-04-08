//! ODE Parity Test Battery — shared correctness definition for all tiers.
//!
//! The CPU `kerr_derivative_into` is the specification. Every tier's implementation
//! is measured against this battery. When adding a new physics term, add test cases
//! here and all tiers get tested automatically.

use super::ode_deriv::kerr_derivative_into;
use super::math::softplus;

/// A single parity test case: inputs + expected outputs from the CPU canonical.
pub struct ParityCase {
    pub name: &'static str,
    pub n_bands: usize,
    pub r_in: Vec<f32>,
    pub s_in: Vec<f32>,
    pub gamma: Vec<f32>,
    pub omega: Vec<f32>,
    pub alpha: f32,
    pub beta: f32,
    pub chi: f32,
    pub expected_dr: Vec<f32>,
    pub expected_ds: Vec<f32>,
}

pub struct ParityReport {
    pub tier: String,
    pub total_cases: usize,
    pub passed: usize,
    pub failed: Vec<ParityFailure>,
}

pub struct ParityFailure {
    pub case_name: String,
    pub max_abs_error: f32,
    pub worst_band: usize,
}

/// Run a parity check: given a derivative function, run it against every case
/// and return a report of failures (cases where max abs error > tolerance).
pub fn check_parity<F>(
    tier_name: &str,
    battery: &[ParityCase],
    derivative_fn: F,
    tolerance: f32,
) -> ParityReport
where
    F: Fn(&ParityCase) -> (Vec<f32>, Vec<f32>),
{
    let mut passed = 0;
    let mut failed = Vec::new();

    for case in battery {
        let (actual_dr, actual_ds) = derivative_fn(case);
        let mut max_abs = 0.0f32;
        let mut worst_band = 0;

        for k in 0..case.n_bands {
            let err_r = (actual_dr[k] - case.expected_dr[k]).abs();
            let err_s = (actual_ds[k] - case.expected_ds[k]).abs();
            let err = err_r.max(err_s);
            if err > max_abs {
                max_abs = err;
                worst_band = k;
            }
        }

        if max_abs <= tolerance {
            passed += 1;
        } else {
            failed.push(ParityFailure {
                case_name: case.name.to_string(),
                max_abs_error: max_abs,
                worst_band,
            });
        }
    }

    ParityReport {
        tier: tier_name.to_string(),
        total_cases: battery.len(),
        passed,
        failed,
    }
}

impl ParityReport {
    pub fn print(&self) {
        println!("  Parity [{}]: {}/{} passed", self.tier, self.passed, self.total_cases);
        for f in &self.failed {
            println!("    FAIL: {} — max_abs_err={:.2e} at band {}", f.case_name, f.max_abs_error, f.worst_band);
        }
    }
    pub fn all_passed(&self) -> bool { self.failed.is_empty() }
}

// ─── Battery construction ───

/// Simple deterministic RNG for test vector generation (no deps)
struct TestRng { state: u64 }
impl TestRng {
    fn new(seed: u64) -> Self { Self { state: seed | 1 } }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

fn uniform_gamma(n: usize, val: f32) -> Vec<f32> {
    let raw = ((val).exp() - 1.0).ln();
    (0..n).map(|_| softplus(raw)).collect()
}

fn standard_omega(n: usize) -> Vec<f32> {
    (0..n).map(|k| (k + 1) as f32 * std::f32::consts::PI / n as f32).collect()
}

fn make_case(
    name: &'static str, n: usize,
    r_in: Vec<f32>, s_in: Vec<f32>,
    gamma: Vec<f32>, omega: Vec<f32>,
    alpha: f32, beta: f32, chi: f32,
) -> ParityCase {
    let mut dr = vec![0.0f32; n];
    let mut ds = vec![0.0f32; n];
    kerr_derivative_into(&r_in, &s_in, &gamma, &omega, alpha, beta, chi, &mut dr, &mut ds, None);
    ParityCase { name, n_bands: n, r_in, s_in, gamma, omega, alpha, beta, chi, expected_dr: dr, expected_ds: ds }
}

/// Generate the full parity test battery. Each case computes its expected output
/// by calling the CPU canonical kerr_derivative_into. This is the ONLY place
/// that defines "correct."
pub fn generate_parity_battery() -> Vec<ParityCase> {
    let n = 84;
    let alpha = 0.1f32;
    let beta = 0.2f32;
    let g = uniform_gamma(n, 0.1);
    let o = standard_omega(n);
    let mut battery = Vec::new();

    // 1. Zero input — expected: all zeros
    battery.push(make_case("zero_input", n,
        vec![0.0; n], vec![0.0; n], g.clone(), o.clone(), alpha, beta, 0.0));

    // 2. Single band mid, chi=0
    let mut r2 = vec![0.0f32; n]; r2[40] = 1.0;
    battery.push(make_case("single_band_mid_chi0", n,
        r2.clone(), vec![0.0; n], g.clone(), o.clone(), alpha, beta, 0.0));

    // 3. Single band mid, chi=0.03
    battery.push(make_case("single_band_mid_chi003", n,
        r2, vec![0.0; n], g.clone(), o.clone(), alpha, beta, 0.03));

    // 4. Two band adjacent, chi=0.03
    let mut r4 = vec![0.0f32; n]; r4[20] = 0.7; r4[21] = 0.7;
    battery.push(make_case("two_band_adjacent_chi003", n,
        r4, vec![0.0; n], g.clone(), o.clone(), alpha, beta, 0.03));

    // 5. Two band distant, chi=0.03
    let mut r5 = vec![0.0f32; n]; r5[20] = 0.7; r5[40] = 0.7;
    battery.push(make_case("two_band_distant_chi003", n,
        r5, vec![0.0; n], g.clone(), o.clone(), alpha, beta, 0.03));

    // 6. Broadband unit, chi=0 (regression gate)
    let mut rng = TestRng::new(42);
    let r6: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let s6: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    // Normalize to unit total energy
    let norm6 = (r6.iter().map(|x| x*x).sum::<f32>() + s6.iter().map(|x| x*x).sum::<f32>()).sqrt();
    let r6n: Vec<f32> = r6.iter().map(|x| x / norm6).collect();
    let s6n: Vec<f32> = s6.iter().map(|x| x / norm6).collect();
    battery.push(make_case("broadband_unit_chi0", n,
        r6n.clone(), s6n.clone(), g.clone(), o.clone(), alpha, beta, 0.0));

    // 7. Broadband unit, chi=0.03
    battery.push(make_case("broadband_unit_chi003", n,
        r6n.clone(), s6n.clone(), g.clone(), o.clone(), alpha, beta, 0.03));

    // 8. Broadband training amplitude (~1.3 per band), chi=0.03
    let r8: Vec<f32> = r6n.iter().map(|x| x * 1.3 * (n as f32).sqrt()).collect();
    let s8: Vec<f32> = s6n.iter().map(|x| x * 1.3 * (n as f32).sqrt()).collect();
    battery.push(make_case("broadband_training_chi003", n,
        r8.clone(), s8.clone(), g.clone(), o.clone(), alpha, beta, 0.03));

    // 9. Broadband training amplitude, chi=0.5 (stress test)
    battery.push(make_case("broadband_training_chi05", n,
        r8.clone(), s8.clone(), g.clone(), o.clone(), alpha, beta, 0.5));

    // 10. Quartet matched: bands 10,30,15,25 (10+30=15+25=40)
    let mut r10 = vec![0.0f32; n];
    r10[10] = 1.0; r10[30] = 1.0; r10[15] = 1.0; r10[25] = 1.0;
    battery.push(make_case("quartet_matched", n,
        r10, vec![0.0; n], g.clone(), o.clone(), alpha, beta, 0.03));

    // 11. Quartet unmatched: bands 10,30,12,25 (10+30≠12+25)
    let mut r11 = vec![0.0f32; n];
    r11[10] = 1.0; r11[30] = 1.0; r11[12] = 1.0; r11[25] = 1.0;
    battery.push(make_case("quartet_unmatched", n,
        r11, vec![0.0; n], g.clone(), o.clone(), alpha, beta, 0.03));

    // 12. Learned gamma profile (non-uniform)
    let mut rng12 = TestRng::new(123);
    let gamma12: Vec<f32> = (0..n).map(|_| softplus(rng12.next_f32() * 0.5)).collect();
    battery.push(make_case("learned_gamma_profile", n,
        r8.clone(), s8.clone(), gamma12, o.clone(), alpha, beta, 0.03));

    // 13. Edge bands (0 and n-1)
    let mut r13 = vec![0.0f32; n]; r13[0] = 1.0; r13[n-1] = 1.0;
    battery.push(make_case("edge_bands", n,
        r13, vec![0.0; n], g.clone(), o.clone(), alpha, beta, 0.03));

    // 14. Negative chi
    battery.push(make_case("negative_chi", n,
        r6n.clone(), s6n.clone(), g.clone(), o.clone(), alpha, beta, -0.03));

    // 15. Large n_bands (168)
    let n2 = 168;
    let g2 = uniform_gamma(n2, 0.1);
    let o2 = standard_omega(n2);
    let mut rng15 = TestRng::new(77);
    let r15: Vec<f32> = (0..n2).map(|_| rng15.next_f32() * 0.5).collect();
    let s15: Vec<f32> = (0..n2).map(|_| rng15.next_f32() * 0.5).collect();
    battery.push(make_case("large_n_bands_168", n2,
        r15, s15, g2, o2, alpha, beta, 0.03));

    battery
}

// ─── Self-consistency test ───

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_battery_self_consistency() {
        let battery = generate_parity_battery();
        let report = check_parity("cpu_self", &battery, |case| {
            let mut dr = vec![0.0f32; case.n_bands];
            let mut ds = vec![0.0f32; case.n_bands];
            kerr_derivative_into(
                &case.r_in, &case.s_in, &case.gamma, &case.omega,
                case.alpha, case.beta, case.chi, &mut dr, &mut ds, None,
            );
            (dr, ds)
        }, 0.0); // exact match — same function, same inputs

        report.print();
        assert!(report.all_passed(), "CPU self-consistency failed — battery generator is broken");
    }
}
