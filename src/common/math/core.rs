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

/// Pearson correlation coefficient between two equal-length series.
/// Returns None when either series has zero variance (undefined) or length < 2.
pub fn pearson_correlation(xs: &[f32], ys: &[f32]) -> Option<f32> {
    if xs.len() != ys.len() || xs.len() < 2 { return None; }
    let n = xs.len() as f32;
    let mean_x: f32 = xs.iter().sum::<f32>() / n;
    let mean_y: f32 = ys.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;
    for i in 0..xs.len() {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 0.0 || var_y <= 0.0 { return None; }
    Some(cov / (var_x.sqrt() * var_y.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_perfect_positive() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&xs, &ys).unwrap();
        assert!((r - 1.0).abs() < 1e-5, "expected +1, got {}", r);
    }

    #[test]
    fn pearson_perfect_negative() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![8.0, 6.0, 4.0, 2.0];
        let r = pearson_correlation(&xs, &ys).unwrap();
        assert!((r + 1.0).abs() < 1e-5, "expected -1, got {}", r);
    }

    #[test]
    fn pearson_constant_series_is_none() {
        let xs = vec![1.0, 1.0, 1.0];
        let ys = vec![1.0, 2.0, 3.0];
        assert!(pearson_correlation(&xs, &ys).is_none());
    }

    #[test]
    fn pearson_length_mismatch_is_none() {
        let xs = vec![1.0, 2.0];
        let ys = vec![1.0, 2.0, 3.0];
        assert!(pearson_correlation(&xs, &ys).is_none());
    }
}
