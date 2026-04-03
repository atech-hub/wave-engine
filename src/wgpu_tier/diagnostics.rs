//! GPU diagnostics — extracted from main.rs.
//! diagnose_ode_gpu_vs_cpu, validate_gpu_fft.

use crate::backend;
use crate::gpu_pipelines;
use crate::fft_ode;
use crate::wave_block::*;
use crate::common::dims::{N_BANDS, N_EMBD, RK4_STEPS};

#[allow(dead_code)]
pub fn diagnose_ode_gpu_vs_cpu(gpu_be: &gpu_pipelines::GpuBackend) {
    let gpu: &dyn backend::ComputeBackend = gpu_be;
    let n_bands = N_BANDS;
    let n_embd = N_EMBD;

    // Create test input
    let x: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.01).sin()).collect();

    let gamma_raw_val = ((0.1f32).exp() - 1.0).ln();
    let weights = KerrWeights {
        gamma_raw: vec![gamma_raw_val; n_bands],
        omega: (0..n_bands).map(|k| (k + 1) as f32 / n_bands as f32).collect(),
        alpha: 0.1, beta: 0.1, rk4_n_steps: RK4_STEPS,
        phase_correction: vec![0.0; n_bands],
        rk4_weights: [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0],
    };

    // CPU ODE
    let cpu_out = kerr_ode_forward_cpu_standalone(&weights, &x);

    // GPU ODE (batched, single position)
    let gpu_out = gpu.kerr_ode_batch(&weights, &[x.clone()]);
    let gpu_out = &gpu_out[0];

    let max_diff = cpu_out.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_diff: f32 = cpu_out.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs()).sum::<f32>() / n_embd as f32;

    eprintln!("ODE diagnostic: max_diff={:.2e}, mean_diff={:.2e}", max_diff, mean_diff);
    eprintln!("  CPU[0..5]: {:?}", &cpu_out[..5]);
    eprintln!("  GPU[0..5]: {:?}", &gpu_out[..5]);

    // Also test linear_batch (out_proj equivalent)
    let w: Vec<Vec<f32>> = (0..n_embd).map(|i| {
        (0..n_embd).map(|j| ((i * n_embd + j) as f32 * 0.001).cos()).collect()
    }).collect();
    let b: Vec<f32> = (0..n_embd).map(|i| i as f32 * 0.01).collect();
    let inputs = vec![x.clone(); 64];
    let gpu_linear = gpu.linear_batch(&w, &b, &inputs);
    // CPU reference
    let cpu_linear: Vec<Vec<f32>> = inputs.iter().map(|xi| {
        let mut y = vec![0.0f32; n_embd];
        for i in 0..n_embd {
            let mut sum = 0.0f32;
            for j in 0..n_embd { sum += w[i][j] * xi[j]; }
            y[i] = sum + b[i];
        }
        y
    }).collect();
    let linear_max = gpu_linear[0].iter().zip(cpu_linear[0].iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let linear_mean = gpu_linear[0].iter().zip(cpu_linear[0].iter())
        .map(|(a, b)| (a - b).abs()).sum::<f32>() / n_embd as f32;
    let max_mag = cpu_linear[0].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rel_err = if max_mag > 0.0 { linear_max / max_mag } else { 0.0 };
    eprintln!("Linear diagnostic: max_diff={:.2e}, mean_diff={:.2e}, max_mag={:.1}, rel_err={:.2e}",
        linear_max, linear_mean, max_mag, rel_err);

    // ─── GPU FFT convolution validation ───
    validate_gpu_fft(gpu_be);
}

pub fn validate_gpu_fft(gpu: &gpu_pipelines::GpuBackend) {
    use rustfft::num_complex::Complex;

    let n_bands = N_BANDS;
    let n_positions = 4; // test with 4 positions

    // Create test mag_sq data
    let mag_sq: Vec<f32> = (0..n_positions * n_bands)
        .map(|i| ((i as f32 * 0.1).sin()).abs())
        .collect();

    // CPU reference: use StencilFft from fft_ode
    let stencil = fft_ode::StencilFft::new(n_bands);
    let cpu_results: Vec<f32> = (0..n_positions).flat_map(|pos| {
        let slice = &mag_sq[pos * n_bands..(pos + 1) * n_bands];
        stencil.convolve(slice)
    }).collect();

    // Precompute kernel FFT for GPU (same kernel as StencilFft)
    let fft_len = n_bands.next_power_of_two(); // 512
    let mut kernel = vec![Complex::new(0.0f32, 0.0); fft_len];
    kernel[1] = Complex::new(1.0, 0.0);
    kernel[2] = Complex::new(1.0, 0.0);
    if fft_len >= 2 {
        kernel[fft_len - 1] = Complex::new(1.0, 0.0);
        kernel[fft_len - 2] = Complex::new(1.0, 0.0);
    }
    let mut planner = rustfft::FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(fft_len);
    fft_fwd.process(&mut kernel);

    let kernel_re: Vec<f32> = kernel.iter().map(|c| c.re).collect();
    let kernel_im: Vec<f32> = kernel.iter().map(|c| c.im).collect();

    // GPU FFT convolution
    let gpu_results = gpu.gpu_fft_convolve(&mag_sq, &kernel_re, &kernel_im, n_positions, n_bands);

    // Compare
    let max_diff = cpu_results.iter().zip(gpu_results.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_diff = cpu_results.iter().zip(gpu_results.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / cpu_results.len() as f32;

    eprintln!("GPU FFT validation ({n_bands} bands, {n_positions} positions): max_diff={:.2e}, mean_diff={:.2e}",
        max_diff, mean_diff);
    if max_diff < 1e-3 {
        eprintln!("  GPU FFT: PASS");
    } else {
        eprintln!("  GPU FFT: FAIL — check shader");
        // Print first few for debugging
        eprintln!("  CPU[0..5]: {:?}", &cpu_results[..5]);
        eprintln!("  GPU[0..5]: {:?}", &gpu_results[..5]);
    }
}

fn kerr_ode_forward_cpu_standalone(weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;
    let n_steps = weights.rk4_n_steps;
    let dt = 1.0 / n_steps as f32;
    fn softplus(v: f32) -> f32 { if v > 20.0 { v } else { (1.0 + v.exp()).ln() } }
    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();
    let mut r: Vec<f32> = (0..n_bands).map(|k| x[k * 2]).collect();
    let mut s: Vec<f32> = (0..n_bands).map(|k| x[k * 2 + 1]).collect();
    let w = &weights.rk4_weights;
    for _ in 0..n_steps {
        let (r_new, s_new) = rk4_step_standalone(&r, &s, dt, &gamma, &weights.omega, weights.alpha, weights.beta, w);
        r = r_new; s = s_new;
    }
    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands { out[k * 2] = r[k]; out[k * 2 + 1] = s[k]; }
    out
}

fn rk4_step_standalone(r: &[f32], s: &[f32], dt: f32, gamma: &[f32], omega: &[f32], alpha: f32, beta: f32, w: &[f32; 4]) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();
    let deriv = |r: &[f32], s: &[f32]| -> (Vec<f32>, Vec<f32>) {
        let mut dr = vec![0.0f32; n]; let mut ds = vec![0.0f32; n];
        for k in 0..n {
            let mag_sq = r[k]*r[k] + s[k]*s[k];
            let mut ns = 0.0f32;
            if k >= 2 { ns += r[k-2]*r[k-2] + s[k-2]*s[k-2]; }
            if k >= 1 { ns += r[k-1]*r[k-1] + s[k-1]*s[k-1]; }
            if k+1 < n { ns += r[k+1]*r[k+1] + s[k+1]*s[k+1]; }
            if k+2 < n { ns += r[k+2]*r[k+2] + s[k+2]*s[k+2]; }
            let phi = omega[k] + alpha * mag_sq + beta * ns;
            dr[k] = -gamma[k] * r[k] - phi * s[k];
            ds[k] = -gamma[k] * s[k] + phi * r[k];
        }
        (dr, ds)
    };
    let (k1r, k1s) = deriv(r, s);
    let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&a,&b)| a+0.5*dt*b).collect();
    let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&a,&b)| a+0.5*dt*b).collect();
    let (k2r, k2s) = deriv(&r2, &s2);
    let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&a,&b)| a+0.5*dt*b).collect();
    let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&a,&b)| a+0.5*dt*b).collect();
    let (k3r, k3s) = deriv(&r3, &s3);
    let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&a,&b)| a+dt*b).collect();
    let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&a,&b)| a+dt*b).collect();
    let (k4r, k4s) = deriv(&r4, &s4);
    let rn: Vec<f32> = (0..n).map(|i| r[i]+dt*(w[0]*k1r[i]+w[1]*k2r[i]+w[2]*k3r[i]+w[3]*k4r[i])).collect();
    let sn: Vec<f32> = (0..n).map(|i| s[i]+dt*(w[0]*k1s[i]+w[1]*k2s[i]+w[2]*k3s[i]+w[3]*k4s[i])).collect();
    (rn, sn)
}
