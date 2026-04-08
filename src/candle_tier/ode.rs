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
        pub chi: f32,              // four-wave mixing strength (0.0 = off, NOT learnable)
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
            chi: f32,
            rk4_steps: usize,
            device: &Device,
        ) -> Result<Self> {
            let n_bands = gamma_raw.len();
            Ok(Self {
                gamma_raw: Tensor::from_vec(gamma_raw.to_vec(), (1, n_bands), device)?,
                omega: Tensor::from_vec(omega.to_vec(), (1, n_bands), device)?,
                alpha: Tensor::from_vec(vec![alpha], (1, 1), device)?,
                beta: Tensor::from_vec(vec![beta], (1, 1), device)?,
                chi,
                rk4_w: None, // standard weights
                n_bands,
                rk4_steps,
                learnable: false,
            })
        }

        /// Create learnable ODE params from VarMap (autograd tracks gradients).
        /// chi is NOT learnable (FWM Jacobian is zero on all tiers).
        pub fn learnable(
            n_bands: usize,
            alpha: f32,
            beta: f32,
            chi: f32,
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
                chi,
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
        let mut dr = (r.broadcast_mul(gamma)?.neg()? - (&phi * s)?)?;
        let mut ds = (s.broadcast_mul(gamma)?.neg()? + (&phi * r)?)?;

        // FWM contribution (only if chi != 0)
        if params.chi != 0.0 && n_bands > 4 {
            let (fwm_dr, fwm_ds) = compute_fwm_contribution(r, s, params.chi, n_bands, n_pos)?;
            dr = (dr + fwm_dr)?;
            ds = (ds + fwm_ds)?;
        }

        Ok((dr, ds))
    }

    /// Compute FWM contribution for all bands via tensor shifts.
    /// Uses the same quartet enumeration as the CPU canonical.
    /// Returns (fwm_dr, fwm_ds) each of shape [n_pos, n_bands].
    fn compute_fwm_contribution(
        r: &Tensor, s: &Tensor,
        chi: f32, n_bands: usize, n_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let device = r.device();
        let z1 = Tensor::zeros((n_pos, 1), DType::F32, device)?;
        let z2 = Tensor::zeros((n_pos, 2), DType::F32, device)?;

        // Shifted versions of r and s
        let r_m2 = Tensor::cat(&[&z2, &r.narrow(1, 0, n_bands - 2)?], 1)?;
        let r_m1 = Tensor::cat(&[&z1, &r.narrow(1, 0, n_bands - 1)?], 1)?;
        let r_p1 = Tensor::cat(&[&r.narrow(1, 1, n_bands - 1)?, &z1], 1)?;
        let r_p2 = Tensor::cat(&[&r.narrow(1, 2, n_bands - 2)?, &z2], 1)?;
        let s_m2 = Tensor::cat(&[&z2, &s.narrow(1, 0, n_bands - 2)?], 1)?;
        let s_m1 = Tensor::cat(&[&z1, &s.narrow(1, 0, n_bands - 1)?], 1)?;
        let s_p1 = Tensor::cat(&[&s.narrow(1, 1, n_bands - 1)?, &z1], 1)?;
        let s_p2 = Tensor::cat(&[&s.narrow(1, 2, n_bands - 2)?, &z2], 1)?;

        let mut fwm_dr = Tensor::zeros((n_pos, n_bands), DType::F32, device)?;
        let mut fwm_ds = Tensor::zeros((n_pos, n_bands), DType::F32, device)?;

        // Family A: quartet (a=k-2, b=k+1, c=k-1, d=k)
        // For band k (role d): a=k-2, b=k+1, c=k-1, d=k
        //   p_ab = z_{k-2} * z_{k+1}
        //   contribution to dr[k] = chi * (p_ab_im * r_{k-1} - p_ab_re * s_{k-1})
        //   contribution to ds[k] = -chi * (p_ab_re * r_{k-1} + p_ab_im * s_{k-1})
        {
            // p_ab = z_{k-2} * z_{k+1}: complex multiply
            let pab_re = (&r_m2 * &r_p1)?.sub(&(&s_m2 * &s_p1)?)?;
            let pab_im = (&r_m2 * &s_p1)?.add(&(&s_m2 * &r_p1)?)?;
            // p_cd = z_{k-1} * z_k: complex multiply
            let pcd_re = (&r_m1 * r)?.sub(&(&s_m1 * s)?)?;
            let pcd_im = (&r_m1 * s)?.add(&(&s_m1 * r)?)?;

            // Role d (band = k): dr[k] += chi * (pab_im*r_c - pab_re*s_c) where c=k-1
            fwm_dr = (fwm_dr + ((&pab_im * &r_m1)?.sub(&(&pab_re * &s_m1)?)? * chi as f64)?)?;
            fwm_ds = (fwm_ds - ((&pab_re * &r_m1)?.add(&(&pab_im * &s_m1)?)? * chi as f64)?)?;

            // Role c (band = k-1): dr[k-1] += chi * (pab_im*r_d - pab_re*s_d) where d=k
            // This writes to k-1, so we shift the result right by 1
            let dc_dr = (&pab_im * r)?.sub(&(&pab_re * s)?)?;
            let dc_ds_neg = (&pab_re * r)?.add(&(&pab_im * s)?)?;
            // Shift: contribution computed at position k goes to position k-1
            // narrow(k=2..n-1) then pad right
            let dc_dr_shifted = Tensor::cat(&[&z1, &dc_dr.narrow(1, 2, n_bands - 3)?, &z2], 1)?;
            let dc_ds_shifted = Tensor::cat(&[&z1, &dc_ds_neg.narrow(1, 2, n_bands - 3)?, &z2], 1)?;
            fwm_dr = (fwm_dr + (dc_dr_shifted * chi as f64)?)?;
            fwm_ds = (fwm_ds - (dc_ds_shifted * chi as f64)?)?;

            // Role a (band = k-2): dr[k-2] += chi * (r_b*pcd_im - s_b*pcd_re) where b=k+1
            let da_dr = (&r_p1 * &pcd_im)?.sub(&(&s_p1 * &pcd_re)?)?;
            let da_ds_neg = (&r_p1 * &pcd_re)?.add(&(&s_p1 * &pcd_im)?)?;
            // Contribution at k goes to k-2: shift right by 2
            let da_dr_shifted = Tensor::cat(&[&z2, &da_dr.narrow(1, 2, n_bands - 3)?, &z1], 1)?;
            let da_ds_shifted = Tensor::cat(&[&z2, &da_ds_neg.narrow(1, 2, n_bands - 3)?, &z1], 1)?;
            fwm_dr = (fwm_dr + (da_dr_shifted * chi as f64)?)?;
            fwm_ds = (fwm_ds - (da_ds_shifted * chi as f64)?)?;

            // Role b (band = k+1): dr[k+1] += chi * (r_a*pcd_im - s_a*pcd_re) where a=k-2
            let db_dr = (&r_m2 * &pcd_im)?.sub(&(&s_m2 * &pcd_re)?)?;
            let db_ds_neg = (&r_m2 * &pcd_re)?.add(&(&s_m2 * &pcd_im)?)?;
            // Contribution at k goes to k+1: shift left by 1
            let db_dr_shifted = Tensor::cat(&[&db_dr.narrow(1, 2, n_bands - 3)?, &z1, &z2], 1)?;
            let db_ds_shifted = Tensor::cat(&[&db_ds_neg.narrow(1, 2, n_bands - 3)?, &z1, &z2], 1)?;
            fwm_dr = (fwm_dr + (db_dr_shifted * chi as f64)?)?;
            fwm_ds = (fwm_ds - (db_ds_shifted * chi as f64)?)?;
        }

        // Family B: quartet (a=k-1, b=k+2, c=k, d=k+1)
        {
            let pab_re = (&r_m1 * &r_p2)?.sub(&(&s_m1 * &s_p2)?)?;
            let pab_im = (&r_m1 * &s_p2)?.add(&(&s_m1 * &r_p2)?)?;
            let pcd_re = (r * &r_p1)?.sub(&(s * &s_p1)?)?;
            let pcd_im = (r * &s_p1)?.add(&(s * &r_p1)?)?;

            // Role d (band = k+1): dr[k+1] += chi * (pab_im*r_c - pab_re*s_c) where c=k
            let dd_dr = (&pab_im * r)?.sub(&(&pab_re * s)?)?;
            let dd_ds_neg = (&pab_re * r)?.add(&(&pab_im * s)?)?;
            // Shift left by 1: contribution at k goes to k+1
            let dd_dr_s = Tensor::cat(&[&dd_dr.narrow(1, 1, n_bands - 3)?, &z1, &z2], 1)?;
            let dd_ds_s = Tensor::cat(&[&dd_ds_neg.narrow(1, 1, n_bands - 3)?, &z1, &z2], 1)?;
            fwm_dr = (fwm_dr + (dd_dr_s * chi as f64)?)?;
            fwm_ds = (fwm_ds - (dd_ds_s * chi as f64)?)?;

            // Role c (band = k): dr[k] += chi * (pab_im*r_d - pab_re*s_d) where d=k+1
            let dc_dr = (&pab_im * &r_p1)?.sub(&(&pab_re * &s_p1)?)?;
            let dc_ds_neg = (&pab_re * &r_p1)?.add(&(&pab_im * &s_p1)?)?;
            let dc_dr_s = Tensor::cat(&[&z1, &dc_dr.narrow(1, 1, n_bands - 3)?, &z2], 1)?;
            let dc_ds_s = Tensor::cat(&[&z1, &dc_ds_neg.narrow(1, 1, n_bands - 3)?, &z2], 1)?;
            fwm_dr = (fwm_dr + (dc_dr_s * chi as f64)?)?;
            fwm_ds = (fwm_ds - (dc_ds_s * chi as f64)?)?;

            // Role a (band = k-1): dr[k-1] += chi * (r_b*pcd_im - s_b*pcd_re) where b=k+2
            let da_dr = (&r_p2 * &pcd_im)?.sub(&(&s_p2 * &pcd_re)?)?;
            let da_ds_neg = (&r_p2 * &pcd_re)?.add(&(&s_p2 * &pcd_im)?)?;
            let da_dr_s = Tensor::cat(&[&z1, &da_dr.narrow(1, 1, n_bands - 3)?, &z2], 1)?;
            let da_ds_s = Tensor::cat(&[&z1, &da_ds_neg.narrow(1, 1, n_bands - 3)?, &z2], 1)?;
            fwm_dr = (fwm_dr + (da_dr_s * chi as f64)?)?;
            fwm_ds = (fwm_ds - (da_ds_s * chi as f64)?)?;

            // Role b (band = k+2): dr[k+2] += chi * (r_a*pcd_im - s_a*pcd_re) where a=k-1
            let db_dr = (&r_m1 * &pcd_im)?.sub(&(&s_m1 * &pcd_re)?)?;
            let db_ds_neg = (&r_m1 * &pcd_re)?.add(&(&s_m1 * &pcd_im)?)?;
            let db_dr_s = Tensor::cat(&[&db_dr.narrow(1, 1, n_bands - 3)?, &z2, &z1], 1)?;
            let db_ds_s = Tensor::cat(&[&db_ds_neg.narrow(1, 1, n_bands - 3)?, &z2, &z1], 1)?;
            fwm_dr = (fwm_dr + (db_dr_s * chi as f64)?)?;
            fwm_ds = (fwm_ds - (db_ds_s * chi as f64)?)?;
        }

        Ok((fwm_dr, fwm_ds))
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
