//! J9: Tensor shape consistency — catch dimension mismatches at boundaries.
//!
//! Provides assertion helpers for function boundaries and a model-wide
//! shape audit that runs at training startup. Catches init bugs, refactor
//! bugs, and config mismatches before they become deep-loop panics.

use crate::common::wave_model::WavePacketModel;

/// Result of a model-wide shape audit.
pub struct ShapeAuditResult {
    pub n_checks: usize,
    pub failures: Vec<ShapeFailure>,
}

#[derive(Debug)]
pub struct ShapeFailure {
    pub location: String,
    pub expected: String,
    pub actual: String,
}

impl ShapeAuditResult {
    pub fn passed(&self) -> bool { self.failures.is_empty() }
}

/// Audit all tensor shapes in a model for internal consistency.
/// Runs once at startup — checks every weight tensor matches the model's dimensions.
pub fn audit_model_shapes(model: &WavePacketModel) -> ShapeAuditResult {
    let mut failures = Vec::new();
    let mut n_checks = 0;
    let n_embd = model.ln_f.weight.len();
    let n_bands = n_embd / 2;

    // ln_f
    check(&mut failures, &mut n_checks, "ln_f.weight", model.ln_f.weight.len(), n_embd);
    check(&mut failures, &mut n_checks, "ln_f.bias", model.ln_f.bias.len(), n_embd);

    // wte: [vocab_size][n_embd]
    check(&mut failures, &mut n_checks, "wte.len", model.wte.len(), model.vocab_size);
    if !model.wte.is_empty() {
        check(&mut failures, &mut n_checks, "wte[0].len", model.wte[0].len(), n_embd);
    }

    // wpe: [block_size][n_embd]
    if !model.wpe.is_empty() {
        check(&mut failures, &mut n_checks, "wpe[0].len", model.wpe[0].len(), n_embd);
    }

    // output_corrector
    if model.phase_native {
        check(&mut failures, &mut n_checks, "output_corrector.len", model.output_corrector.len(), n_bands);
    }

    // Per-block checks
    let n_head = if !model.blocks.is_empty() { model.blocks[0].attn.heads.len() } else { 0 };
    let head_dim = if n_head > 0 { n_embd / n_head } else { 0 };

    for (b, block) in model.blocks.iter().enumerate() {
        let prefix = format!("blocks[{}]", b);

        // Layer norms
        check(&mut failures, &mut n_checks, &format!("{}.ln.weight", prefix), block.ln.weight.len(), n_embd);
        check(&mut failures, &mut n_checks, &format!("{}.ln.bias", prefix), block.ln.bias.len(), n_embd);
        check(&mut failures, &mut n_checks, &format!("{}.ln_ffn.weight", prefix), block.ln_ffn.weight.len(), n_embd);
        check(&mut failures, &mut n_checks, &format!("{}.ln_ffn.bias", prefix), block.ln_ffn.bias.len(), n_embd);

        // Kerr weights
        check(&mut failures, &mut n_checks, &format!("{}.kerr.gamma_raw", prefix), block.ffn.kerr.gamma_raw.len(), n_bands);
        check(&mut failures, &mut n_checks, &format!("{}.kerr.omega", prefix), block.ffn.kerr.omega.len(), n_bands);
        check(&mut failures, &mut n_checks, &format!("{}.kerr.phase_correction", prefix), block.ffn.kerr.phase_correction.len(), n_bands);

        // Attention heads
        check(&mut failures, &mut n_checks, &format!("{}.attn.heads.len", prefix), block.attn.heads.len(), n_head);
        for (h, head) in block.attn.heads.iter().enumerate() {
            let hp = format!("{}.attn.heads[{}]", prefix, h);
            check(&mut failures, &mut n_checks, &format!("{}.phase_proj_w.len", hp), head.phase_proj_w.len(), 2);
            if !head.phase_proj_w.is_empty() {
                check(&mut failures, &mut n_checks, &format!("{}.phase_proj_w[0].len", hp), head.phase_proj_w[0].len(), n_embd);
            }
            check(&mut failures, &mut n_checks, &format!("{}.phase_proj_b.len", hp), head.phase_proj_b.len(), 2);
            check(&mut failures, &mut n_checks, &format!("{}.v_proj_w.len", hp), head.v_proj_w.len(), head_dim);
            if !head.v_proj_w.is_empty() {
                check(&mut failures, &mut n_checks, &format!("{}.v_proj_w[0].len", hp), head.v_proj_w[0].len(), head_dim);
            }
            check(&mut failures, &mut n_checks, &format!("{}.v_proj_b.len", hp), head.v_proj_b.len(), head_dim);
        }

        // Attention out_proj
        check(&mut failures, &mut n_checks, &format!("{}.attn.out_proj_w.len", prefix), block.attn.out_proj_w.len(), n_embd);
        if !block.attn.out_proj_w.is_empty() {
            check(&mut failures, &mut n_checks, &format!("{}.attn.out_proj_w[0].len", prefix), block.attn.out_proj_w[0].len(), n_embd);
        }
        check(&mut failures, &mut n_checks, &format!("{}.attn.out_proj_b.len", prefix), block.attn.out_proj_b.len(), n_embd);
    }

    ShapeAuditResult { n_checks, failures }
}

fn check(failures: &mut Vec<ShapeFailure>, n_checks: &mut usize, location: &str, actual: usize, expected: usize) {
    *n_checks += 1;
    if actual != expected {
        failures.push(ShapeFailure {
            location: location.to_string(),
            expected: format!("{}", expected),
            actual: format!("{}", actual),
        });
    }
}

/// Print audit result.
pub fn print_result(result: &ShapeAuditResult, assert_mode: bool) {
    if result.passed() {
        eprintln!("[J9] Shape audit: {}/{} checks passed — PASS", result.n_checks, result.n_checks);
    } else {
        eprintln!("[J9] SHAPE MISMATCH: {} failures in {} checks", result.failures.len(), result.n_checks);
        for f in &result.failures {
            eprintln!("  {}: expected {} got {}", f.location, f.expected, f.actual);
        }
        if assert_mode {
            panic!("J9: Shape audit failed — model has inconsistent tensor dimensions");
        }
    }
}

/// Assertion helpers for function boundaries (debug builds only).
/// Call these at the start of major functions.
#[inline]
pub fn assert_2d_shape(name: &str, tensor: &[Vec<f32>], expected_rows: usize, expected_cols: usize) {
    debug_assert_eq!(tensor.len(), expected_rows,
        "{}: expected {} rows, got {}", name, expected_rows, tensor.len());
    debug_assert!(tensor.iter().all(|r| r.len() == expected_cols),
        "{}: expected all rows len={}, got {:?}", name, expected_cols,
        tensor.iter().map(|r| r.len()).collect::<Vec<_>>());
}

#[inline]
pub fn assert_1d_shape(name: &str, tensor: &[f32], expected_len: usize) {
    debug_assert_eq!(tensor.len(), expected_len,
        "{}: expected len={}, got {}", name, expected_len, tensor.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::wave_model::init_model;
    use crate::common::dims::Dims;

    #[test]
    fn test_healthy_model_passes() {
        let dims = Dims::from_cli(4, 2, 16, 128, 16);
        let model = init_model(15, 42, 2, 1, dims, 0.1, 0.2);
        let result = audit_model_shapes(&model);
        print_result(&result, false);
        assert!(result.passed(), "Healthy model should pass shape audit");
        assert!(result.n_checks > 20, "Should check at least 20 shapes, got {}", result.n_checks);
    }

    #[test]
    fn test_phase_native_model_passes() {
        let dims = Dims::from_cli(4, 2, 16, 128, 16);
        let mut model = init_model(15, 42, 1, 1, dims, 0.1, 0.2);
        model.phase_native = true;
        model.output_corrector = vec![0.0; 4];
        let result = audit_model_shapes(&model);
        print_result(&result, false);
        assert!(result.passed());
    }

    #[test]
    fn test_realistic_model_passes() {
        let dims = Dims::from_cli(128, 4, 16, 128, 16);
        let model = init_model(77, 42, 4, 1, dims, 0.1, 0.2);
        let result = audit_model_shapes(&model);
        print_result(&result, false);
        assert!(result.passed(), "Realistic 128-band 4-layer model should pass");
    }

    #[test]
    fn test_corrupted_model_fails() {
        let dims = Dims::from_cli(4, 2, 16, 128, 16);
        let mut model = init_model(15, 42, 1, 1, dims, 0.1, 0.2);
        // Corrupt a tensor — wrong length
        model.blocks[0].attn.heads[0].phase_proj_b = vec![0.0; 5]; // should be 2
        let result = audit_model_shapes(&model);
        print_result(&result, false);
        assert!(!result.passed(), "Corrupted model should fail");
        assert!(result.failures.iter().any(|f| f.location.contains("phase_proj_b")));
    }
}
