//! GPU-native Kerr-ODE via Candle tensor ops — RK4-N integration.
//!
//! Full RK4 integration matching CPU/wgpu tiers. All operations are
//! Candle tensor ops → GPU-native, autograd-compatible. Gradient flows
//! through the actual computation (not identity backward).
//!
//! Replaces the perturbative single-step approximation with proper
//! multi-step RK4 integration for accuracy parity with CPU/wgpu.

#[cfg(feature = "candle-backend")]
pub mod gpu_ode {
    use candle_core::{Tensor, DType, Device, Result};

    /// Precomputed ODE parameters as GPU-resident tensors.
    /// Created once at model init, reused every forward pass.
    pub struct GpuOdeParams {
        pub gamma: Tensor,    // [1, n_bands] — softplus(gamma_raw), damping coefficient
        pub omega: Tensor,    // [1, n_bands] — natural frequency per band
        pub alpha: Tensor,    // scalar as [1, 1] — self-coupling
        pub beta: Tensor,     // scalar as [1, 1] — cross-coupling
        pub n_bands: usize,
        pub rk4_steps: usize,
    }

    impl GpuOdeParams {
        /// Create GPU-resident ODE params from raw values.
        pub fn new(
            gamma_raw: &[f32],
            omega: &[f32],
            alpha: f32,
            beta: f32,
            rk4_steps: usize,
            device: &Device,
        ) -> Result<Self> {
            let n_bands = gamma_raw.len();
            fn softplus(v: f32) -> f32 { if v > 20.0 { v } else { (1.0 + v.exp()).ln() } }

            let gamma_vals: Vec<f32> = gamma_raw.iter().map(|&g| softplus(g)).collect();
            let omega_vals: Vec<f32> = omega.to_vec();

            Ok(Self {
                gamma: Tensor::from_vec(gamma_vals, (1, n_bands), device)?,
                omega: Tensor::from_vec(omega_vals, (1, n_bands), device)?,
                alpha: Tensor::from_vec(vec![alpha], (1, 1), device)?,
                beta: Tensor::from_vec(vec![beta], (1, 1), device)?,
                n_bands,
                rk4_steps,
            })
        }

        /// Update from learned coupling values (called after optimizer step).
        pub fn update_coupling(&mut self, alpha: f32, beta: f32, gamma_raw: &[f32]) -> Result<()> {
            fn softplus(v: f32) -> f32 { if v > 20.0 { v } else { (1.0 + v.exp()).ln() } }
            let device = self.gamma.device().clone();
            let n_bands = self.n_bands;
            let gamma_vals: Vec<f32> = gamma_raw.iter().map(|&g| softplus(g)).collect();
            self.gamma = Tensor::from_vec(gamma_vals, (1, n_bands), &device)?;
            self.alpha = Tensor::from_vec(vec![alpha], (1, 1), &device)?;
            self.beta = Tensor::from_vec(vec![beta], (1, 1), &device)?;
            Ok(())
        }
    }

    /// Compute the Kerr-ODE derivative for all bands.
    ///
    /// dr[k] = -gamma[k] * r[k] - phi[k] * s[k]
    /// ds[k] = -gamma[k] * s[k] + phi[k] * r[k]
    /// where phi[k] = omega[k] + alpha * (r[k]² + s[k]²) + beta * neighbour_sum
    ///
    /// All operations are batched across positions: r, s are [n_pos, n_bands].
    fn kerr_derivative(
        r: &Tensor, s: &Tensor,
        params: &GpuOdeParams,
    ) -> Result<(Tensor, Tensor)> {
        let (n_pos, n_bands) = r.dims2()?;

        // mag_sq = r² + s²
        let mag_sq = (r * r)?.add(&(s * s)?)?;

        // Neighbour sum via pad + narrow (GPU-friendly, no loops)
        let zeros = Tensor::zeros((n_pos, 2), DType::F32, r.device())?;
        let padded = Tensor::cat(&[&zeros, &mag_sq, &zeros], 1)?;
        let ns_m2 = padded.narrow(1, 0, n_bands)?;
        let ns_m1 = padded.narrow(1, 1, n_bands)?;
        let ns_p1 = padded.narrow(1, 3, n_bands)?;
        let ns_p2 = padded.narrow(1, 4, n_bands)?;
        let neighbour_sum = ((ns_m2 + ns_m1)? + (ns_p1 + ns_p2)?)?;

        // phi = omega + alpha * mag_sq + beta * neighbour_sum
        let phi = (mag_sq.broadcast_mul(&params.alpha)? + neighbour_sum.broadcast_mul(&params.beta)?)?
            .broadcast_add(&params.omega)?;

        // dr = -gamma * r - phi * s
        // ds = -gamma * s + phi * r
        let dr = (r.broadcast_mul(&params.gamma)?.neg()? - (&phi * s)?)?;
        let ds = (s.broadcast_mul(&params.gamma)?.neg()? + (&phi * r)?)?;

        Ok((dr, ds))
    }

    /// GPU-native RK4 ODE forward pass.
    ///
    /// Input: x of shape [n_pos, n_embd] (interleaved r,s pairs)
    /// Output: transformed x of same shape
    ///
    /// All operations are Candle tensor ops → GPU-native, autograd-compatible.
    /// Matches CPU/wgpu RK4-N integration exactly.
    pub fn kerr_ode_gpu(x: &Tensor, params: &GpuOdeParams) -> Result<Tensor> {
        let (n_pos, n_embd) = x.dims2()?;
        let n_bands = params.n_bands;
        let n_steps = params.rk4_steps;
        let dt = 1.0 / n_steps as f64;

        // Split interleaved [r0,s0,r1,s1,...] into separate r and s
        let x_reshaped = x.reshape((n_pos, n_bands, 2))?;
        let mut r = x_reshaped.narrow(2, 0, 1)?.squeeze(2)?;  // [n_pos, n_bands]
        let mut s = x_reshaped.narrow(2, 1, 1)?.squeeze(2)?;  // [n_pos, n_bands]

        // RK4 integration loop
        for _ in 0..n_steps {
            // k1 at (r, s)
            let (k1r, k1s) = kerr_derivative(&r, &s, params)?;

            // k2 at (r + 0.5*dt*k1, s + 0.5*dt*k1)
            let r2 = (&r + (&k1r * (0.5 * dt))?)?;
            let s2 = (&s + (&k1s * (0.5 * dt))?)?;
            let (k2r, k2s) = kerr_derivative(&r2, &s2, params)?;

            // k3 at (r + 0.5*dt*k2, s + 0.5*dt*k2)
            let r3 = (&r + (&k2r * (0.5 * dt))?)?;
            let s3 = (&s + (&k2s * (0.5 * dt))?)?;
            let (k3r, k3s) = kerr_derivative(&r3, &s3, params)?;

            // k4 at (r + dt*k3, s + dt*k3)
            let r4 = (&r + (&k3r * dt)?)?;
            let s4 = (&s + (&k3s * dt)?)?;
            let (k4r, k4s) = kerr_derivative(&r4, &s4, params)?;

            // Combine: r_new = r + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            let dt6 = dt / 6.0;
            r = (&r + ((&k1r + (&k2r * 2.0)? + (&k3r * 2.0)? + &k4r)? * dt6)?)?;
            s = (&s + ((&k1s + (&k2s * 2.0)? + (&k3s * 2.0)? + &k4s)? * dt6)?)?;
        }

        // Interleave back to [n_pos, n_embd]
        let r_expanded = r.unsqueeze(2)?;  // [n_pos, n_bands, 1]
        let s_expanded = s.unsqueeze(2)?;  // [n_pos, n_bands, 1]
        let interleaved = Tensor::cat(&[&r_expanded, &s_expanded], 2)?;
        interleaved.reshape((n_pos, n_embd))
    }
}
