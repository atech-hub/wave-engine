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

    /// ODE parameters — either frozen (pre-computed) or learnable (in VarMap).
    /// When learnable, alpha/beta/gamma_raw are Tensors tracked by autograd.
    /// When frozen, they're constant tensors (no gradient).
    pub struct GpuOdeParams {
        pub gamma_raw: Tensor,     // [1, n_bands] — raw damping (before softplus)
        pub omega: Tensor,         // [1, n_bands] — natural frequency per band (always frozen)
        pub alpha: Tensor,         // [1, 1] — self-coupling
        pub beta: Tensor,          // [1, 1] — cross-coupling
        pub rk4_w: Option<Tensor>, // [4] — RK4 combination weights (None = standard 1/6,1/3,1/3,1/6)
        pub n_bands: usize,
        pub rk4_steps: usize,
        pub learnable: bool,
    }

    impl GpuOdeParams {
        /// Create frozen ODE params (no gradient flow).
        pub fn new(
            gamma_raw: &[f32],
            omega: &[f32],
            alpha: f32,
            beta: f32,
            rk4_steps: usize,
            device: &Device,
        ) -> Result<Self> {
            let n_bands = gamma_raw.len();
            Ok(Self {
                gamma_raw: Tensor::from_vec(gamma_raw.to_vec(), (1, n_bands), device)?,
                omega: Tensor::from_vec(omega.to_vec(), (1, n_bands), device)?,
                alpha: Tensor::from_vec(vec![alpha], (1, 1), device)?,
                beta: Tensor::from_vec(vec![beta], (1, 1), device)?,
                rk4_w: None, // standard weights
                n_bands,
                rk4_steps,
                learnable: false,
            })
        }

        /// Create learnable ODE params from VarMap (autograd tracks gradients).
        pub fn learnable(
            n_bands: usize,
            alpha: f32,
            beta: f32,
            rk4_steps: usize,
            vb: candle_nn::VarBuilder,
        ) -> Result<Self> {
            let gamma_raw_val = ((0.1f32).exp() - 1.0).ln();
            let omega_vals: Vec<f32> = (0..n_bands).map(|k| (k + 1) as f32 / n_bands as f32).collect();
            let device = vb.device().clone();

            let gamma_raw = vb.get_with_hints(
                (1, n_bands), "gamma_raw",
                candle_nn::Init::Const(gamma_raw_val as f64),
            )?;
            let alpha_t = vb.get_with_hints(
                (1, 1), "alpha",
                candle_nn::Init::Const(alpha as f64),
            )?;
            let beta_t = vb.get_with_hints(
                (1, 1), "beta",
                candle_nn::Init::Const(beta as f64),
            )?;
            // Omega is always frozen (natural frequencies)
            let omega = Tensor::from_vec(omega_vals, (1, n_bands), &device)?;

            Ok(Self {
                gamma_raw,
                omega,
                alpha: alpha_t,
                beta: beta_t,
                rk4_w: None, // set via set_rk4_learnable() if --rk4-weights dyn
                n_bands,
                rk4_steps,
                learnable: true,
            })
        }
    }

    impl GpuOdeParams {
        /// Make RK4 weights learnable (call after construction).
        pub fn set_rk4_learnable(&mut self, varmap: &candle_nn::VarMap, key_prefix: &str, device: &Device) -> Result<()> {
            let key = format!("{key_prefix}.rk4_weights");
            // Step 1: Create var in VarMap (zero placeholder)
            let _tensor = varmap.get((4,), &key, candle_nn::Init::Const(0.0), DType::F32, device)?;
            // Step 2: Overwrite with correct [1/6, 1/3, 1/3, 1/6]
            let correct_init = Tensor::from_slice(
                &[1.0f32/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0], (4,), device,
            )?;
            {
                let data = varmap.data().lock().unwrap();
                if let Some(var) = data.get(&key) {
                    var.set(&correct_init)?;
                }
            }
            // Step 3: Re-get tensor (now correct values, tracked by optimizer)
            let tensor = varmap.get((4,), &key, candle_nn::Init::Const(0.0), DType::F32, device)?;
            self.rk4_w = Some(tensor);
            Ok(())
        }
    }

    /// Softplus as a tensor op (autograd-compatible).
    fn tensor_softplus(x: &Tensor) -> Result<Tensor> {
        // softplus(x) = log(1 + exp(x))
        // For numerical stability: when x > 20, softplus(x) ≈ x
        let ones = x.ones_like()?;
        let exp_x = x.exp()?;
        let result = (ones + exp_x)?.log()?;
        Ok(result)
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
        gamma: &Tensor,  // [1, n_bands] — already softplus'd
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
        let dr = (r.broadcast_mul(gamma)?.neg()? - (&phi * s)?)?;
        let ds = (s.broadcast_mul(gamma)?.neg()? + (&phi * r)?)?;

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

        // Compute gamma from gamma_raw via softplus (autograd traces through)
        let gamma = tensor_softplus(&params.gamma_raw)?;

        // RK4 integration loop
        for _ in 0..n_steps {
            // k1 at (r, s)
            let (k1r, k1s) = kerr_derivative(&r, &s, &gamma, params)?;

            // k2 at (r + 0.5*dt*k1, s + 0.5*dt*k1)
            let r2 = (&r + (&k1r * (0.5 * dt))?)?;
            let s2 = (&s + (&k1s * (0.5 * dt))?)?;
            let (k2r, k2s) = kerr_derivative(&r2, &s2, &gamma, params)?;

            // k3 at (r + 0.5*dt*k2, s + 0.5*dt*k2)
            let r3 = (&r + (&k2r * (0.5 * dt))?)?;
            let s3 = (&s + (&k2s * (0.5 * dt))?)?;
            let (k3r, k3s) = kerr_derivative(&r3, &s3, &gamma, params)?;

            // k4 at (r + dt*k3, s + dt*k3)
            let r4 = (&r + (&k3r * dt)?)?;
            let s4 = (&s + (&k3s * dt)?)?;
            let (k4r, k4s) = kerr_derivative(&r4, &s4, &gamma, params)?;

            // Combine: r_new = r + dt * (w0*k1 + w1*k2 + w2*k3 + w3*k4)
            if let Some(ref w) = params.rk4_w {
                // Learnable weights — extract scalars (autograd traces through)
                let w0 = w.narrow(0, 0, 1)?.reshape(())?;
                let w1 = w.narrow(0, 1, 1)?.reshape(())?;
                let w2 = w.narrow(0, 2, 1)?.reshape(())?;
                let w3 = w.narrow(0, 3, 1)?.reshape(())?;
                r = (&r + ((k1r.broadcast_mul(&w0)? + k2r.broadcast_mul(&w1)? + k3r.broadcast_mul(&w2)? + k4r.broadcast_mul(&w3)?)? * dt)?)?;
                s = (&s + ((k1s.broadcast_mul(&w0)? + k2s.broadcast_mul(&w1)? + k3s.broadcast_mul(&w2)? + k4s.broadcast_mul(&w3)?)? * dt)?)?;
            } else {
                // Standard RK4: dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                let dt6 = dt / 6.0;
                r = (&r + ((&k1r + (&k2r * 2.0)? + (&k3r * 2.0)? + &k4r)? * dt6)?)?;
                s = (&s + ((&k1s + (&k2s * 2.0)? + (&k3s * 2.0)? + &k4s)? * dt6)?)?;
            }
        }

        // Interleave back to [n_pos, n_embd]
        let r_expanded = r.unsqueeze(2)?;  // [n_pos, n_bands, 1]
        let s_expanded = s.unsqueeze(2)?;  // [n_pos, n_bands, 1]
        let interleaved = Tensor::cat(&[&r_expanded, &s_expanded], 2)?;
        interleaved.reshape((n_pos, n_embd))
    }
}
