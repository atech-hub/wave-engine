//! J3: Gradient pathway completeness — forward fan-out must equal backward fan-in.
//!
//! Every tensor consumed by multiple paths in the forward pass must have gradient
//! contributions from ALL those paths in the backward pass. The d_normed bug was
//! exactly this: normed was read by both attention and FFN in forward, but only
//! FFN contributed to d_normed in backward.
//!
//! This monitor uses a registration-based approach: forward and backward code
//! register their fan-out consumers and fan-in contributors into a PathwayRegistry.
//! After backward completes, check_balance() verifies every fan-out is matched.

use std::collections::{HashMap, HashSet};

/// A fan-out point: one tensor read by multiple consumers.
#[derive(Debug, Clone)]
pub struct FanOut {
    pub tensor: String,
    pub consumers: Vec<String>,
}

/// A fan-in point: multiple gradient contributors merging into one gradient tensor.
#[derive(Debug, Clone)]
pub struct FanIn {
    pub tensor: String,
    pub contributors: HashSet<String>,
}

/// Registry that tracks fan-out and fan-in during a forward/backward pass.
pub struct PathwayRegistry {
    fan_outs: HashMap<String, Vec<String>>,  // tensor → consumers
    fan_ins: HashMap<String, HashSet<String>>, // d_tensor → contributors
}

/// Result of a pathway balance check.
pub struct PathwayResult {
    pub n_fan_outs: usize,
    pub n_balanced: usize,
    pub imbalances: Vec<PathwayImbalance>,
}

/// One imbalanced fan-out point.
#[derive(Debug)]
pub struct PathwayImbalance {
    pub tensor: String,
    pub expected_contributors: Vec<String>,
    pub actual_contributors: Vec<String>,
    pub missing: Vec<String>,
}

impl PathwayResult {
    pub fn passed(&self) -> bool { self.imbalances.is_empty() }
}

impl PathwayRegistry {
    pub fn new() -> Self {
        Self {
            fan_outs: HashMap::new(),
            fan_ins: HashMap::new(),
        }
    }

    /// Register a fan-out: tensor is consumed by these paths in forward.
    pub fn register_fan_out(&mut self, tensor: &str, consumers: &[&str]) {
        self.fan_outs.insert(tensor.to_string(), consumers.iter().map(|s| s.to_string()).collect());
    }

    /// Register a fan-in contribution: this path contributed gradient to d_tensor in backward.
    pub fn register_fan_in(&mut self, tensor: &str, contributor: &str) {
        self.fan_ins.entry(tensor.to_string()).or_default().insert(contributor.to_string());
    }

    /// Check that every fan-out has matching fan-in contributors.
    pub fn check_balance(&self) -> PathwayResult {
        let mut imbalances = Vec::new();
        let mut n_balanced = 0;

        for (tensor, consumers) in &self.fan_outs {
            let actual = self.fan_ins.get(tensor);
            let actual_set: HashSet<&str> = actual
                .map(|s| s.iter().map(|x| x.as_str()).collect())
                .unwrap_or_default();

            let missing: Vec<String> = consumers.iter()
                .filter(|c| !actual_set.contains(c.as_str()))
                .cloned()
                .collect();

            if missing.is_empty() {
                n_balanced += 1;
            } else {
                imbalances.push(PathwayImbalance {
                    tensor: tensor.clone(),
                    expected_contributors: consumers.clone(),
                    actual_contributors: actual_set.iter().map(|s| s.to_string()).collect(),
                    missing,
                });
            }
        }

        PathwayResult {
            n_fan_outs: self.fan_outs.len(),
            n_balanced,
            imbalances,
        }
    }

    /// Reset for next iteration.
    pub fn reset_fan_ins(&mut self) {
        self.fan_ins.clear();
    }
}

/// Build the static pathway map for the wave-engine architecture.
/// This declares what the forward fan-out points ARE — the contract.
pub fn build_engine_pathways(n_layers: usize, learnable_attn: bool) -> PathwayRegistry {
    let mut reg = PathwayRegistry::new();

    for b in 0..n_layers {
        // normed is read by both attention and FFN
        reg.register_fan_out(
            &format!("block_{}_normed", b),
            &["attention", "ffn"],
        );

        // block input is consumed by layer_norm and residual passthrough
        reg.register_fan_out(
            &format!("block_{}_input", b),
            &["layer_norm", "residual"],
        );
    }

    reg
}

/// Simulate backward fan-in registration for the CURRENT engine state.
/// This is what the backward code ACTUALLY does (or should do).
pub fn register_current_backward(reg: &mut PathwayRegistry, n_layers: usize, learnable_attn: bool) {
    for b in 0..n_layers {
        // FFN always contributes to d_normed
        reg.register_fan_in(&format!("block_{}_normed", b), "ffn");

        // Attention contributes to d_normed ONLY when learnable_attn is true
        // (or when dyn_harmonics is true, but that's a separate path)
        // When frozen: attention doesn't contribute → known imbalance
        if learnable_attn {
            reg.register_fan_in(&format!("block_{}_normed", b), "attention");
        }

        // Both layer_norm and residual always contribute to d_input
        reg.register_fan_in(&format!("block_{}_input", b), "layer_norm");
        reg.register_fan_in(&format!("block_{}_input", b), "residual");
    }
}

/// Print pathway check result.
pub fn print_result(result: &PathwayResult) {
    if result.passed() {
        eprintln!("[J3] Pathway completeness: {}/{} fan-outs balanced — PASS",
            result.n_balanced, result.n_fan_outs);
    } else {
        eprintln!("[J3] PATHWAY IMBALANCE: {}/{} fan-outs balanced",
            result.n_balanced, result.n_fan_outs);
        for imb in &result.imbalances {
            eprintln!("  {} — expected {:?}, got {:?}, missing {:?}",
                imb.tensor, imb.expected_contributors, imb.actual_contributors, imb.missing);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balanced_with_learnable_attn() {
        let mut reg = build_engine_pathways(2, true);
        register_current_backward(&mut reg, 2, true);
        let result = reg.check_balance();
        print_result(&result);
        assert!(result.passed(), "Should be balanced when attention contributes to d_normed");
    }

    #[test]
    fn test_imbalanced_frozen_attention() {
        // This is the CURRENT state of the engine: frozen attention doesn't contribute
        let mut reg = build_engine_pathways(2, false);
        register_current_backward(&mut reg, 2, false);
        let result = reg.check_balance();
        print_result(&result);
        // Should fail — attention reads normed but doesn't contribute to d_normed
        assert!(!result.passed(), "Frozen attention should show imbalance on d_normed");
        assert_eq!(result.imbalances.len(), 2, "One imbalance per block");
        for imb in &result.imbalances {
            assert!(imb.tensor.contains("normed"), "Imbalance should be on normed tensor");
            assert_eq!(imb.missing, vec!["attention"], "Missing contributor should be attention");
        }
    }

    #[test]
    fn test_input_always_balanced() {
        let mut reg = build_engine_pathways(1, false);
        register_current_backward(&mut reg, 1, false);
        let result = reg.check_balance();
        // block_input should be balanced (both layer_norm and residual contribute)
        let input_imbalances: Vec<_> = result.imbalances.iter()
            .filter(|i| i.tensor.contains("input"))
            .collect();
        assert!(input_imbalances.is_empty(), "block_input should always be balanced");
    }

    #[test]
    fn test_missing_ffn_detected() {
        // Simulate a bug where FFN backward is removed
        let mut reg = build_engine_pathways(1, false);
        // Only register residual and layer_norm for input, nothing for normed
        reg.register_fan_in("block_0_input", "layer_norm");
        reg.register_fan_in("block_0_input", "residual");
        let result = reg.check_balance();
        print_result(&result);
        assert!(!result.passed());
        // normed should show both attention AND ffn missing
        let normed_imb = result.imbalances.iter().find(|i| i.tensor == "block_0_normed").unwrap();
        assert!(normed_imb.missing.contains(&"attention".to_string()));
        assert!(normed_imb.missing.contains(&"ffn".to_string()));
    }
}
