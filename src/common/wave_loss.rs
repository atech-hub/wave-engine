//! Wave-space loss functions for train-from-waves.
//!
//! Computes cosine distance loss and its gradient between predicted
//! hidden states and target wave patterns from a KWDS file.

/// Cosine distance loss between predicted and target waves.
/// Returns (loss, d_pred) where d_pred is the gradient w.r.t. predicted.
///
/// loss = 1 - cos(pred, target) = 1 - (pred·target) / (|pred|·|target|)
///
/// d_loss/d_pred = -(target / (|pred|·|target|)) + pred * (pred·target) / (|pred|³·|target|)
pub fn cosine_loss(pred: &[f32], target: &[f32]) -> (f32, Vec<f32>) {
    let n = pred.len();
    let dot: f32 = pred.iter().zip(target.iter()).map(|(&a, &b)| a * b).sum();
    let np: f32 = pred.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nt: f32 = target.iter().map(|x| x * x).sum::<f32>().sqrt();

    if np < 1e-8 || nt < 1e-8 {
        return (1.0, vec![0.0; n]);
    }

    let cos = dot / (np * nt);
    let loss = 1.0 - cos;

    // Gradient: d(1 - cos)/d_pred = -d(cos)/d_pred
    // d(cos)/d_pred_i = target_i / (np * nt) - pred_i * dot / (np³ * nt)
    let np3 = np * np * np;
    let mut grad = vec![0.0f32; n];
    for i in 0..n {
        let d_cos = target[i] / (np * nt) - pred[i] * dot / (np3 * nt);
        grad[i] = -d_cos; // negative because loss = 1 - cos
    }

    (loss, grad)
}

/// Batch cosine loss: average over all positions.
/// Returns (mean_loss, per_position_gradients).
pub fn batch_cosine_loss(
    predictions: &[Vec<f32>],
    targets: &[Vec<f32>],
) -> (f32, Vec<Vec<f32>>) {
    let t = predictions.len().min(targets.len());
    let mut total_loss = 0.0f32;
    let mut grads = Vec::with_capacity(t);

    for pos in 0..t {
        let (loss, grad) = cosine_loss(&predictions[pos], &targets[pos]);
        total_loss += loss;
        // Scale gradient by 1/t for mean loss
        let scaled: Vec<f32> = grad.iter().map(|&g| g / t as f32).collect();
        grads.push(scaled);
    }

    (total_loss / t as f32, grads)
}

/// Finite-difference gradient check for cosine loss.
/// Returns (passed, max_rel_error).
pub fn check_cosine_gradient(pred: &[f32], target: &[f32]) -> (bool, f32) {
    let eps = 1e-4f32;
    let (_, analytical) = cosine_loss(pred, target);
    let mut max_err = 0.0f32;

    for i in 0..pred.len().min(20) { // check first 20 dims
        let mut pred_plus = pred.to_vec();
        let mut pred_minus = pred.to_vec();
        pred_plus[i] += eps;
        pred_minus[i] -= eps;
        let (loss_plus, _) = cosine_loss(&pred_plus, target);
        let (loss_minus, _) = cosine_loss(&pred_minus, target);
        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let rel_err = if analytical[i].abs() > 1e-6 {
            ((numerical - analytical[i]) / analytical[i]).abs()
        } else {
            (numerical - analytical[i]).abs()
        };
        if rel_err > max_err { max_err = rel_err; }
    }

    (max_err < 0.01, max_err)
}
