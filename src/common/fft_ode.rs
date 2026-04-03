//! FFT-based ODE derivative — OFDM-inspired parallel computation.
//!
//! The stencil coupling ns[k] = mag_sq[k-2] + mag_sq[k-1] + mag_sq[k+1] + mag_sq[k+2]
//! is a convolution with kernel [1, 1, 0, 1, 1]. Convolution in spatial domain =
//! multiplication in frequency domain. FFT makes it parallel.
//!
//! Telecom processes 384+ coupled subcarriers in real-time on phone chips.
//! Same problem, proven solution.

use rustfft::{FftPlanner, num_complex::Complex};

/// Precomputed FFT of the stencil kernel. Created once, reused every derivative call.
pub struct StencilFft {
    kernel_fft: Vec<Complex<f32>>,
    fft_len: usize,  // padded to power of 2
    planner_fwd: std::sync::Arc<dyn rustfft::Fft<f32>>,
    planner_inv: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl StencilFft {
    /// Precompute the stencil kernel FFT for n_bands.
    pub fn new(n_bands: usize) -> Self {
        // Pad to next power of 2 for radix-2 FFT efficiency
        let fft_len = n_bands.next_power_of_two();

        // Build stencil kernel: [0, 1, 1, 0, 0, ..., 0, 1, 1]
        // kernel[1] = k+1, kernel[2] = k+2, kernel[N-1] = k-1, kernel[N-2] = k-2
        let mut kernel = vec![Complex::new(0.0f32, 0.0); fft_len];
        kernel[1] = Complex::new(1.0, 0.0);   // k+1
        kernel[2] = Complex::new(1.0, 0.0);   // k+2
        if fft_len >= 2 {
            kernel[fft_len - 1] = Complex::new(1.0, 0.0);  // k-1
            kernel[fft_len - 2] = Complex::new(1.0, 0.0);  // k-2
        }

        // Precompute FFT of kernel
        let mut planner = FftPlanner::new();
        let fft_fwd = planner.plan_fft_forward(fft_len);
        let fft_inv = planner.plan_fft_inverse(fft_len);
        fft_fwd.process(&mut kernel);

        Self {
            kernel_fft: kernel,
            fft_len,
            planner_fwd: fft_fwd,
            planner_inv: fft_inv,
        }
    }

    /// Compute neighbour sums for all bands simultaneously via FFT convolution.
    /// Input: mag_sq[n_bands]. Output: neighbour_sums[n_bands].
    pub fn convolve(&self, mag_sq: &[f32]) -> Vec<f32> {
        let n_bands = mag_sq.len();

        // Pad mag_sq to fft_len
        let mut input: Vec<Complex<f32>> = mag_sq.iter()
            .map(|&v| Complex::new(v, 0.0))
            .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
            .take(self.fft_len)
            .collect();

        // FFT(mag_sq)
        self.planner_fwd.process(&mut input);

        // Element-wise multiply with kernel FFT
        for i in 0..self.fft_len {
            input[i] *= self.kernel_fft[i];
        }

        // IFFT
        self.planner_inv.process(&mut input);

        // Normalize (rustfft doesn't normalize IFFT)
        let scale = 1.0 / self.fft_len as f32;

        // Extract real parts for the first n_bands elements
        input[..n_bands].iter().map(|c| c.re * scale).collect()
    }
}

/// FFT-based ODE derivative — all neighbour sums computed in parallel.
pub fn deriv_fft(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
    stencil: &StencilFft,
) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();

    // 1. Compute mag_sq for all bands
    let mag_sq: Vec<f32> = (0..n).map(|k| r[k]*r[k] + s[k]*s[k]).collect();

    // 2. Compute ALL neighbour sums via FFT convolution (parallel!)
    let neighbour_sums = stencil.convolve(&mag_sq);

    // 3. Compute derivatives (all bands, no dependencies)
    let mut dr = vec![0.0f32; n];
    let mut ds = vec![0.0f32; n];
    for k in 0..n {
        let phi = omega[k] + alpha * mag_sq[k] + beta * neighbour_sums[k];
        dr[k] = -gamma[k] * r[k] - phi * s[k];
        ds[k] = -gamma[k] * s[k] + phi * r[k];
    }

    (dr, ds)
}

/// Sequential stencil derivative (for comparison/validation).
pub fn deriv_sequential(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();
    let mut dr = vec![0.0f32; n];
    let mut ds = vec![0.0f32; n];
    deriv_sequential_into(r, s, gamma, omega, alpha, beta, &mut dr, &mut ds);
    (dr, ds)
}

/// In-place sequential derivative — writes into pre-allocated output buffers.
/// Same computation as `deriv_sequential` but zero allocation.
pub fn deriv_sequential_into(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
    dr: &mut [f32], ds: &mut [f32],
) {
    let n = r.len();
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
}

/// In-place FFT derivative — writes into pre-allocated output buffers.
/// Same computation as `deriv_fft` but zero allocation for dr/ds.
fn deriv_fft_into(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
    stencil: &StencilFft,
    dr: &mut [f32], ds: &mut [f32],
) {
    let n = r.len();

    // 1. Compute mag_sq for all bands
    let mag_sq: Vec<f32> = (0..n).map(|k| r[k]*r[k] + s[k]*s[k]).collect();

    // 2. Compute ALL neighbour sums via FFT convolution (parallel!)
    let neighbour_sums = stencil.convolve(&mag_sq);

    // 3. Compute derivatives (all bands, no dependencies)
    for k in 0..n {
        let phi = omega[k] + alpha * mag_sq[k] + beta * neighbour_sums[k];
        dr[k] = -gamma[k] * r[k] - phi * s[k];
        ds[k] = -gamma[k] * s[k] + phi * r[k];
    }
}

/// Full RK4 ODE step using FFT-based derivative.
/// Below 256 bands, uses sequential derivative (7x faster than FFT at 84 bands).
/// Pre-allocates all RK4 scratch buffers outside the loop (zero per-step allocation).
pub fn kerr_ode_fft(x: &[f32], gamma_raw: &[f32], omega: &[f32], alpha: f32, beta: f32, rk4_steps: usize, stencil: &StencilFft, rk4_w: &[f32; 4]) -> Vec<f32> {
    let n_bands = gamma_raw.len();
    let n_embd = n_bands * 2;
    let dt = 1.0 / rk4_steps as f32;
    let use_sequential = n_bands < 256;

    fn softplus(v: f32) -> f32 { if v > 20.0 { v } else { (1.0 + v.exp()).ln() } }
    let gamma: Vec<f32> = gamma_raw.iter().map(|&g| softplus(g)).collect();

    let mut r: Vec<f32> = (0..n_bands).map(|k| x[k * 2]).collect();
    let mut s: Vec<f32> = (0..n_bands).map(|k| x[k * 2 + 1]).collect();

    // Pre-allocate ALL scratch buffers
    let mut r_tmp = vec![0.0f32; n_bands];
    let mut s_tmp = vec![0.0f32; n_bands];
    let mut k1r = vec![0.0f32; n_bands];
    let mut k1s = vec![0.0f32; n_bands];
    let mut k2r = vec![0.0f32; n_bands];
    let mut k2s = vec![0.0f32; n_bands];
    let mut k3r = vec![0.0f32; n_bands];
    let mut k3s = vec![0.0f32; n_bands];
    let mut k4r = vec![0.0f32; n_bands];
    let mut k4s = vec![0.0f32; n_bands];

    for _ in 0..rk4_steps {
        // k1 at (r, s)
        if use_sequential {
            deriv_sequential_into(&r, &s, &gamma, omega, alpha, beta, &mut k1r, &mut k1s);
        } else {
            deriv_fft_into(&r, &s, &gamma, omega, alpha, beta, stencil, &mut k1r, &mut k1s);
        }

        // r_tmp = r + 0.5*dt*k1r, s_tmp = s + 0.5*dt*k1s
        for i in 0..n_bands { r_tmp[i] = r[i] + 0.5*dt*k1r[i]; }
        for i in 0..n_bands { s_tmp[i] = s[i] + 0.5*dt*k1s[i]; }

        // k2 at (r_tmp, s_tmp)
        if use_sequential {
            deriv_sequential_into(&r_tmp, &s_tmp, &gamma, omega, alpha, beta, &mut k2r, &mut k2s);
        } else {
            deriv_fft_into(&r_tmp, &s_tmp, &gamma, omega, alpha, beta, stencil, &mut k2r, &mut k2s);
        }

        // r_tmp = r + 0.5*dt*k2r, s_tmp = s + 0.5*dt*k2s
        for i in 0..n_bands { r_tmp[i] = r[i] + 0.5*dt*k2r[i]; }
        for i in 0..n_bands { s_tmp[i] = s[i] + 0.5*dt*k2s[i]; }

        // k3 at (r_tmp, s_tmp)
        if use_sequential {
            deriv_sequential_into(&r_tmp, &s_tmp, &gamma, omega, alpha, beta, &mut k3r, &mut k3s);
        } else {
            deriv_fft_into(&r_tmp, &s_tmp, &gamma, omega, alpha, beta, stencil, &mut k3r, &mut k3s);
        }

        // r_tmp = r + dt*k3r, s_tmp = s + dt*k3s
        for i in 0..n_bands { r_tmp[i] = r[i] + dt*k3r[i]; }
        for i in 0..n_bands { s_tmp[i] = s[i] + dt*k3s[i]; }

        // k4 at (r_tmp, s_tmp)
        if use_sequential {
            deriv_sequential_into(&r_tmp, &s_tmp, &gamma, omega, alpha, beta, &mut k4r, &mut k4s);
        } else {
            deriv_fft_into(&r_tmp, &s_tmp, &gamma, omega, alpha, beta, stencil, &mut k4r, &mut k4s);
        }

        // Final RK4 combination
        for i in 0..n_bands {
            r[i] += dt * (rk4_w[0]*k1r[i] + rk4_w[1]*k2r[i] + rk4_w[2]*k3r[i] + rk4_w[3]*k4r[i]);
            s[i] += dt * (rk4_w[0]*k1s[i] + rk4_w[1]*k2s[i] + rk4_w[2]*k3s[i] + rk4_w[3]*k4s[i]);
        }
    }

    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands { out[k * 2] = r[k]; out[k * 2 + 1] = s[k]; }
    out
}

/// Precomputed GPU kernel FFT — uploaded once, reused every call.
pub struct GpuKernelFft {
    pub kernel_re: Vec<f32>,  // [512]
    pub kernel_im: Vec<f32>,  // [512]
}

impl GpuKernelFft {
    /// Precompute the kernel FFT for GPU dispatch.
    pub fn new(n_bands: usize) -> Self {
        use rustfft::num_complex::Complex;
        let fft_len = n_bands.next_power_of_two();
        let mut kernel = vec![Complex::new(0.0f32, 0.0); fft_len];
        kernel[1] = Complex::new(1.0, 0.0);
        kernel[2] = Complex::new(1.0, 0.0);
        if fft_len >= 2 {
            kernel[fft_len - 1] = Complex::new(1.0, 0.0);
            kernel[fft_len - 2] = Complex::new(1.0, 0.0);
        }
        let mut planner = FftPlanner::new();
        let fft_fwd = planner.plan_fft_forward(fft_len);
        fft_fwd.process(&mut kernel);
        Self {
            kernel_re: kernel.iter().map(|c| c.re).collect(),
            kernel_im: kernel.iter().map(|c| c.im).collect(),
        }
    }
}

/// Batched RK4 ODE with GPU FFT for neighbour sums.
/// Processes all positions in parallel via GPU FFT convolution at each derivative call.
pub fn kerr_ode_batch_gpu_fft(
    xs: &[Vec<f32>],
    gamma_raw: &[f32],
    omega: &[f32],
    alpha: f32,
    beta: f32,
    rk4_steps: usize,
    gpu_kernel: &GpuKernelFft,
    gpu: &crate::gpu_pipelines::GpuBackend,
    rk4_w: &[f32; 4],
) -> Vec<Vec<f32>> {
    let n_pos = xs.len();
    if n_pos == 0 { return vec![]; }
    let n_bands = gamma_raw.len();
    let n_embd = n_bands * 2;
    let dt = 1.0 / rk4_steps as f32;

    fn softplus(v: f32) -> f32 { if v > 20.0 { v } else { (1.0 + v.exp()).ln() } }
    let gamma: Vec<f32> = gamma_raw.iter().map(|&g| softplus(g)).collect();

    // Deinterleave all positions into flat r/s arrays [n_pos * n_bands]
    let mut r = vec![0.0f32; n_pos * n_bands];
    let mut s = vec![0.0f32; n_pos * n_bands];
    for (pos, x) in xs.iter().enumerate() {
        for k in 0..n_bands {
            r[pos * n_bands + k] = x[k * 2];
            s[pos * n_bands + k] = x[k * 2 + 1];
        }
    }

    // Batched derivative using GPU FFT for neighbour sums
    let deriv_gpu = |r: &[f32], s: &[f32]| -> (Vec<f32>, Vec<f32>) {
        // 1. Compute mag_sq for all (pos, band)
        let mag_sq: Vec<f32> = r.iter().zip(s.iter())
            .map(|(&ri, &si)| ri * ri + si * si).collect();

        // 2. GPU FFT convolution: all positions' neighbour sums in one dispatch
        let ns = gpu.gpu_fft_convolve(&mag_sq, &gpu_kernel.kernel_re, &gpu_kernel.kernel_im, n_pos, n_bands);

        // 3. Compute derivatives
        let mut dr = vec![0.0f32; n_pos * n_bands];
        let mut ds = vec![0.0f32; n_pos * n_bands];
        for pos in 0..n_pos {
            let base = pos * n_bands;
            for k in 0..n_bands {
                let idx = base + k;
                let phi = omega[k] + alpha * mag_sq[idx] + beta * ns[idx];
                dr[idx] = -gamma[k] * r[idx] - phi * s[idx];
                ds[idx] = -gamma[k] * s[idx] + phi * r[idx];
            }
        }
        (dr, ds)
    };

    // RK4 integration (sequential across steps, batched across positions)
    for _ in 0..rk4_steps {
        let (k1r, k1s) = deriv_gpu(&r, &s);
        let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&a,&b)| a+0.5*dt*b).collect();
        let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&a,&b)| a+0.5*dt*b).collect();
        let (k2r, k2s) = deriv_gpu(&r2, &s2);
        let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&a,&b)| a+0.5*dt*b).collect();
        let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&a,&b)| a+0.5*dt*b).collect();
        let (k3r, k3s) = deriv_gpu(&r3, &s3);
        let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&a,&b)| a+dt*b).collect();
        let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&a,&b)| a+dt*b).collect();
        let (k4r, k4s) = deriv_gpu(&r4, &s4);
        for i in 0..n_pos * n_bands {
            r[i] += dt * (rk4_w[0]*k1r[i] + rk4_w[1]*k2r[i] + rk4_w[2]*k3r[i] + rk4_w[3]*k4r[i]);
            s[i] += dt * (rk4_w[0]*k1s[i] + rk4_w[1]*k2s[i] + rk4_w[2]*k3s[i] + rk4_w[3]*k4s[i]);
        }
    }

    // Reinterleave
    (0..n_pos).map(|pos| {
        let mut out = vec![0.0f32; n_embd];
        for k in 0..n_bands {
            out[k * 2] = r[pos * n_bands + k];
            out[k * 2 + 1] = s[pos * n_bands + k];
        }
        out
    }).collect()
}

/// Validation: compare FFT derivative against sequential.
pub fn validate_fft_derivative(n_bands: usize) {
    let stencil = StencilFft::new(n_bands);

    // Test input
    let r: Vec<f32> = (0..n_bands).map(|k| (k as f32 * 0.1).sin()).collect();
    let s: Vec<f32> = (0..n_bands).map(|k| (k as f32 * 0.1).cos()).collect();
    let gamma: Vec<f32> = vec![0.1; n_bands];
    let omega: Vec<f32> = (0..n_bands).map(|k| (k+1) as f32 / n_bands as f32).collect();

    let (dr_seq, ds_seq) = deriv_sequential(&r, &s, &gamma, &omega, 0.1, 0.1);
    let (dr_fft, ds_fft) = deriv_fft(&r, &s, &gamma, &omega, 0.1, 0.1, &stencil);

    let max_diff_r = dr_seq.iter().zip(&dr_fft).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    let max_diff_s = ds_seq.iter().zip(&ds_fft).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);

    println!("FFT ODE validation ({n_bands} bands):");
    println!("  dr max_diff: {:.2e}", max_diff_r);
    println!("  ds max_diff: {:.2e}", max_diff_s);
    if max_diff_r < 1e-4 && max_diff_s < 1e-4 {
        println!("  PASS");
    } else {
        println!("  FAIL — check kernel setup");
    }
}
