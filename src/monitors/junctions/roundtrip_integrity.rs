//! J4: Roundtrip integrity — flatten and unflatten must be exact inverses.
//!
//! Invariant: flatten(unflatten(flatten(model))) == flatten(model) bitwise.
//! If they disagree, checkpoint save/load silently corrupts weights.
//! Runs as unit test and at training startup. Mandatory.

use crate::common::wave_model::*;
use crate::common::dims::Dims;

/// Result of the roundtrip check.
pub struct RoundtripResult {
    pub n_params: usize,
    pub bitwise_equal: bool,
    pub n_mismatched: usize,
    pub mismatches: Vec<(usize, f32, f32)>, // (index, original, after_roundtrip)
}

impl RoundtripResult {
    pub fn passed(&self) -> bool { self.bitwise_equal }
}

/// Check flatten/unflatten roundtrip for a given model configuration.
/// Builds a model with random weights, flattens, unflattens into a second model,
/// flattens again, and compares bitwise.
pub fn check_roundtrip(
    vocab_size: usize,
    n_layers: usize,
    n_bands: usize,
    n_head: usize,
    dims: Dims,
    alpha: f32,
    beta: f32,
) -> RoundtripResult {
    // Build model with random weights (seed 99 for distinct from training seed 42)
    let mut model = init_model(vocab_size, 99, n_layers, 1, dims, alpha, beta);
    model.phase_native = true;
    model.output_corrector = vec![0.0; n_bands];
    model.learnable_ode = dims.learnable_ode;

    // First flatten
    let flat1 = flatten_params_ex(&model, false);
    let n_params = flat1.len();

    // Unflatten into a fresh model with same structure
    let mut model2 = init_model(vocab_size, 0, n_layers, 1, dims, alpha, beta);
    model2.phase_native = true;
    model2.output_corrector = vec![0.0; n_bands];
    model2.learnable_ode = dims.learnable_ode;
    unflatten_params_ex(&mut model2, &flat1, false);

    // Second flatten
    let flat2 = flatten_params_ex(&model2, false);

    // Compare bitwise
    let mut mismatches = Vec::new();
    for i in 0..n_params.min(flat2.len()) {
        if flat1[i].to_bits() != flat2[i].to_bits() {
            mismatches.push((i, flat1[i], flat2[i]));
        }
    }
    // Length mismatch counts as all-mismatched
    if flat1.len() != flat2.len() {
        mismatches.push((n_params, flat1.len() as f32, flat2.len() as f32));
    }

    let bitwise_equal = mismatches.is_empty();
    let n_mismatched = mismatches.len();

    RoundtripResult { n_params, bitwise_equal, n_mismatched, mismatches }
}

/// Print the result.
pub fn print_result(result: &RoundtripResult, assert_mode: bool) {
    if result.bitwise_equal {
        eprintln!("[J4] Roundtrip OK: {} params, flatten→unflatten→flatten bitwise identical", result.n_params);
    } else {
        eprintln!("[J4] ROUNDTRIP MISMATCH: {} params, {} mismatched", result.n_params, result.n_mismatched);
        let n_show = 10.min(result.mismatches.len());
        for &(idx, orig, after) in result.mismatches.iter().take(n_show) {
            eprintln!("  param[{}]: original={:.8} after_roundtrip={:.8} diff={:.2e}",
                idx, orig, after, (orig - after).abs());
        }
        if result.n_mismatched > n_show {
            eprintln!("  ... and {} more", result.n_mismatched - n_show);
        }
        if assert_mode {
            panic!("J4: Roundtrip integrity failed — flatten/unflatten are not inverses");
        }
    }
}

/// Run roundtrip check for multiple common configurations.
/// Returns true if all pass.
pub fn check_all_configs(alpha: f32, beta: f32) -> bool {
    let configs: Vec<(&str, usize, usize, usize, usize, Dims)> = vec![
        ("phase-native (frozen ODE)", 15, 2, 4, 2,
            Dims::from_cli(4, 2, 16, 128, 16).with_learnable_ode(false).with_corrector(false)),
        ("phase-native (learnable ODE)", 15, 2, 4, 2,
            Dims::from_cli(4, 2, 16, 128, 16).with_learnable_ode(true).with_corrector(true)),
        ("standard lm_head", 15, 2, 4, 2, {
            let mut d = Dims::from_cli(4, 2, 16, 128, 16);
            d.learnable_ode = false;
            d
        }),
        ("dyn harmonics", 15, 2, 4, 2,
            Dims::from_cli(4, 2, 16, 128, 16).with_dyn_harmonics(true)),
        ("4-layer 128-band (realistic)", 77, 4, 128, 4,
            Dims::from_cli(128, 4, 16, 128, 16).with_learnable_ode(true).with_corrector(true)),
    ];

    let mut all_pass = true;
    for (name, vocab, n_layers, n_bands, n_head, dims) in configs {
        let result = check_roundtrip(vocab, n_layers, n_bands, n_head, dims, alpha, beta);
        eprint!("[J4] {:<40} ", name);
        if result.passed() {
            eprintln!("{} params — PASS", result.n_params);
        } else {
            eprintln!("{} params — FAIL ({} mismatched)", result.n_params, result.n_mismatched);
            all_pass = false;
        }
    }
    all_pass
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_phase_native() {
        let dims = Dims::from_cli(4, 2, 16, 128, 16)
            .with_learnable_ode(false).with_corrector(false);
        let result = check_roundtrip(15, 1, 4, 2, dims, 0.1, 0.2);
        print_result(&result, false);
        assert!(result.passed(), "Phase-native roundtrip failed: {} mismatches", result.n_mismatched);
    }

    #[test]
    fn test_roundtrip_learnable_ode() {
        let dims = Dims::from_cli(4, 2, 16, 128, 16)
            .with_learnable_ode(true).with_corrector(true);
        let result = check_roundtrip(15, 1, 4, 2, dims, 0.1, 0.2);
        print_result(&result, false);
        assert!(result.passed(), "Learnable ODE roundtrip failed: {} mismatches", result.n_mismatched);
    }

    #[test]
    fn test_roundtrip_dyn_harmonics() {
        let dims = Dims::from_cli(4, 2, 16, 128, 16).with_dyn_harmonics(true);
        let result = check_roundtrip(15, 1, 4, 2, dims, 0.1, 0.2);
        print_result(&result, false);
        assert!(result.passed(), "Dyn harmonics roundtrip failed: {} mismatches", result.n_mismatched);
    }

    #[test]
    fn test_roundtrip_realistic_scale() {
        let dims = Dims::from_cli(128, 4, 16, 128, 16)
            .with_learnable_ode(true).with_corrector(true);
        let result = check_roundtrip(77, 4, 128, 4, dims, 0.1, 0.2);
        print_result(&result, false);
        assert!(result.passed(), "Realistic scale roundtrip failed: {} mismatches", result.n_mismatched);
    }

    #[test]
    fn test_roundtrip_learnable_attn() {
        // With learnable_ode=true, content_proj is populated in CPU attn init,
        // so the flatten path includes it. Verifies that the full set of
        // attention params survives save → load intact.
        let dims = Dims::from_cli(4, 2, 16, 128, 16)
            .with_learnable_ode(true).with_corrector(true)
            .with_learnable_attn(true);
        let result = check_roundtrip(15, 1, 4, 2, dims, 0.1, 0.2);
        print_result(&result, false);
        assert!(result.passed(), "Learnable-attn roundtrip failed: {} mismatches", result.n_mismatched);
    }

    #[test]
    fn test_roundtrip_learnable_attn_no_content_proj() {
        // learnable_ode=false → content_proj vectors are empty, so the flatten
        // path skips their bytes automatically. Must still round-trip.
        let dims = Dims::from_cli(4, 2, 16, 128, 16)
            .with_learnable_ode(false).with_corrector(false)
            .with_learnable_attn(true);
        let result = check_roundtrip(15, 1, 4, 2, dims, 0.1, 0.2);
        print_result(&result, false);
        assert!(result.passed(), "Learnable-attn-no-content roundtrip failed: {} mismatches", result.n_mismatched);
    }
}
