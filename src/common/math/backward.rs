//! Shared backward primitives — pure math, no tier-specific code.
//!
//! All functions operate on &[f32] slices. Used by CPU, wgpu, and candle tiers.
//! Notation: d_x means d(loss)/d(x).

use std::f32::consts::PI;

// ─── Linear backward ───────────────────────────────────────────

/// Backward through y = W @ x + b. Returns (d_x, d_w, d_b).
pub fn linear_backward(
    d_y: &[f32],      // [out_dim]
    x: &[f32],        // [in_dim]
    w: &[Vec<f32>],   // [out_dim][in_dim]
) -> (Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    let out_dim = w.len();
    let in_dim = x.len();

    let mut d_x = vec![0.0f32; in_dim];
    for j in 0..in_dim {
        let mut sum = 0.0f32;
        for i in 0..out_dim { sum += d_y[i] * w[i][j]; }
        d_x[j] = sum;
    }

    let mut d_w = vec![vec![0.0f32; in_dim]; out_dim];
    for i in 0..out_dim {
        for j in 0..in_dim { d_w[i][j] = d_y[i] * x[j]; }
    }

    let d_b = d_y.to_vec();
    (d_x, d_w, d_b)
}

/// Backward through linear: compute only d_x = W^T @ d_y.
pub fn linear_backward_dx_only(d_y: &[f32], w: &[Vec<f32>]) -> Vec<f32> {
    let out_dim = w.len();
    let in_dim = w[0].len();
    let mut d_x = vec![0.0f32; in_dim];
    for j in 0..in_dim {
        let mut sum = 0.0f32;
        for i in 0..out_dim { sum += d_y[i] * w[i][j]; }
        d_x[j] = sum;
    }
    d_x
}

/// Backward through y = W @ x (no bias). Returns (d_x, d_w).
#[allow(dead_code)]
pub fn linear_no_bias_backward(
    d_y: &[f32], x: &[f32], w: &[Vec<f32>],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let out_dim = w.len();
    let in_dim = x.len();
    let mut d_x = vec![0.0f32; in_dim];
    for j in 0..in_dim {
        for i in 0..out_dim { d_x[j] += d_y[i] * w[i][j]; }
    }
    let mut d_w = vec![vec![0.0f32; in_dim]; out_dim];
    for i in 0..out_dim {
        for j in 0..in_dim { d_w[i][j] = d_y[i] * x[j]; }
    }
    (d_x, d_w)
}

// ─── Layer norm backward ────────────────────────────────────────

/// Backward through layer norm: y = (x - mean) / std * weight + bias.
pub fn layer_norm_backward(
    d_y: &[f32], x: &[f32], weight: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = x.len();
    let nf = n as f32;
    let eps = 1e-5f32;

    let mean: f32 = x.iter().sum::<f32>() / nf;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / nf;
    let std = (var + eps).sqrt();
    let inv_std = 1.0 / std;

    let x_hat: Vec<f32> = x.iter().map(|v| (v - mean) * inv_std).collect();
    let d_weight: Vec<f32> = (0..n).map(|i| d_y[i] * x_hat[i]).collect();
    let d_bias = d_y.to_vec();
    let d_x_hat: Vec<f32> = (0..n).map(|i| d_y[i] * weight[i]).collect();

    let d_x_hat_sum: f32 = d_x_hat.iter().sum();
    let d_x_hat_x_hat_sum: f32 = (0..n).map(|i| d_x_hat[i] * x_hat[i]).sum();

    let mut d_x = vec![0.0f32; n];
    for i in 0..n {
        d_x[i] = inv_std / nf * (nf * d_x_hat[i] - d_x_hat_sum - x_hat[i] * d_x_hat_x_hat_sum);
    }

    (d_x, d_weight, d_bias)
}

// ─── Activation backward ────────────────────────────────────────

/// Backward through GELU: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu_backward(d_y: &[f32], x: &[f32]) -> Vec<f32> {
    let sqrt_2_pi = (2.0f32 / PI).sqrt();
    x.iter().zip(d_y.iter()).map(|(&xi, &dy)| {
        let x3 = xi * xi * xi;
        let inner = sqrt_2_pi * (xi + 0.044715 * x3);
        let tanh_inner = inner.tanh();
        let sech2 = 1.0 - tanh_inner * tanh_inner;
        let d_inner = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * xi * xi);
        let grad = 0.5 * (1.0 + tanh_inner) + 0.5 * xi * sech2 * d_inner;
        dy * grad
    }).collect()
}

/// Backward through softplus: y = log(1 + exp(x)), dy/dx = sigmoid(x).
pub fn softplus_backward(d_y: f32, x: f32) -> f32 {
    let sig = if x > 20.0 { 1.0 } else { x.exp() / (1.0 + x.exp()) };
    d_y * sig
}
