//! J5: Vector length consistency — params, count, and grads must agree.
//!
//! Invariant: flatten_params_ex.len() == count_trainable_ex == flatten_grads_ex.len()
//! If these disagree, Adam writes updates to wrong parameter offsets.
//! Runs at training startup before the first Adam step. Mandatory.

/// Result of the vector length check.
pub struct VectorLengthResult {
    pub params_len: usize,
    pub count_trainable: usize,
    pub grads_len: usize,
    pub all_equal: bool,
}

impl VectorLengthResult {
    pub fn passed(&self) -> bool { self.all_equal }
}

/// Check that all three vectors agree on length.
pub fn check_vector_lengths(
    params_len: usize,
    count_trainable: usize,
    grads_len: usize,
) -> VectorLengthResult {
    let all_equal = params_len == count_trainable && count_trainable == grads_len;
    VectorLengthResult { params_len, count_trainable, grads_len, all_equal }
}

/// Print the result. Panics in assert mode if lengths mismatch.
pub fn print_result(result: &VectorLengthResult, assert_mode: bool) {
    if result.all_equal {
        eprintln!("[J5] Vector lengths OK: {} params = {} count = {} grads",
            result.params_len, result.count_trainable, result.grads_len);
    } else {
        eprintln!("[J5] VECTOR LENGTH MISMATCH:");
        eprintln!("  flatten_params_ex.len() = {}", result.params_len);
        eprintln!("  count_trainable_ex()    = {}", result.count_trainable);
        eprintln!("  flatten_grads_ex.len()  = {}", result.grads_len);
        if result.params_len != result.count_trainable {
            eprintln!("  !! params vs count differ by {}",
                (result.params_len as isize - result.count_trainable as isize).abs());
        }
        if result.params_len != result.grads_len {
            eprintln!("  !! params vs grads differ by {}",
                (result.params_len as isize - result.grads_len as isize).abs());
        }
        if assert_mode {
            panic!("J5: Vector length mismatch — training cannot proceed safely");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passes_when_equal() {
        let r = check_vector_lengths(1000, 1000, 1000);
        assert!(r.passed());
    }

    #[test]
    fn test_fails_when_params_differ() {
        let r = check_vector_lengths(1000, 999, 1000);
        assert!(!r.passed());
        assert_eq!(r.params_len, 1000);
        assert_eq!(r.count_trainable, 999);
    }

    #[test]
    fn test_fails_when_grads_differ() {
        let r = check_vector_lengths(1000, 1000, 1001);
        assert!(!r.passed());
    }

    #[test]
    fn test_all_three_different() {
        let r = check_vector_lengths(100, 200, 300);
        assert!(!r.passed());
    }
}
