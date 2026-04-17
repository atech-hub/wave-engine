//! Shared math primitives — one copy, used everywhere.

/// Softplus activation: log(1 + exp(x)), numerically stable.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Derivative of softplus = sigmoid.
#[inline]
pub fn softplus_derivative(x: f32) -> f32 {
    if x > 20.0 { 1.0 } else { 1.0 / (1.0 + (-x).exp()) }
}

/// Cross-entropy backward for a single position.
pub fn cross_entropy_backward(logits: &[f32], target: usize) -> Vec<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_l: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum_exp: f32 = exp_l.iter().sum();
    let mut d = exp_l.iter().map(|&e| e / sum_exp).collect::<Vec<f32>>();
    d[target] -= 1.0;
    d
}
