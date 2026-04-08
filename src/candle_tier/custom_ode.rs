//! CustomOp ODE — RK4 forward without autograd, CPU backward with cached states.
//!
//! Replaces candle autograd RK4 (983K graph nodes, 2,550ms/iter) with:
//!   Forward:  CPU RK4 via common::ode_backward::ode_forward_with_cache (proven, tested)
//!   Backward: CPU chain rule via common::ode_backward::ode_backward (proven, tested)
//!
//! Zero new math. The backward code is shared with the CPU/wgpu tiers.

#[cfg(feature = "candle-backend")]
pub mod custom_ode {
    use candle_core::{CpuStorage, CustomOp1, Layout, Shape, Result, Tensor, DType, Error};
    use std::sync::{Arc, Mutex};

    /// Accumulated ODE parameter gradients from backward — per layer.
    pub struct OdeParamGradsAccum {
        pub d_gamma_raw: Vec<f32>,
        pub d_alpha: f32,
        pub d_beta: f32,
        pub d_chi: f32,
        pub d_rk4_weights: [f32; 4],
    }

    /// Shared gradient storage — passed between CustomOp and training loop via Arc<Mutex>.
    /// Replaces thread_local! which broke when backward ran on a different thread.
    pub type SharedParamGrads = Arc<Mutex<Vec<Option<OdeParamGradsAccum>>>>;

    /// Create shared gradient storage for N layers.
    pub fn create_param_grad_storage(n_layers: usize) -> SharedParamGrads {
        Arc::new(Mutex::new((0..n_layers).map(|_| None).collect()))
    }

    /// Take the accumulated gradients for a layer (clears the slot).
    pub fn take_param_grads(storage: &SharedParamGrads, layer: usize) -> Option<OdeParamGradsAccum> {
        let mut v = storage.lock().unwrap();
        if layer < v.len() { v[layer].take() } else { None }
    }

    /// Cached forward intermediates — shared between forward and backward via Arc<Mutex>.
    struct OdeCache {
        caches: Vec<crate::common::ode_backward::OdeForwardCache>,
        weights: crate::model::KerrWeights,
    }

    /// The CustomOp — holds ODE params, runs forward without autograd graph.
    pub struct KerrOdeCustomOp {
        gamma_raw: Vec<f32>,
        omega: Vec<f32>,
        alpha: f32,
        beta: f32,
        rk4_weights: [f32; 4],
        rk4_steps: usize,
        n_bands: usize,
        layer_idx: usize,
        cache: Arc<Mutex<Option<OdeCache>>>,
        param_grads: SharedParamGrads,
    }

    impl KerrOdeCustomOp {
        pub fn new(
            gamma_raw: Vec<f32>, omega: Vec<f32>,
            alpha: f32, beta: f32, rk4_weights: [f32; 4],
            rk4_steps: usize, n_bands: usize, layer_idx: usize,
            param_grads: SharedParamGrads,
        ) -> Self {
            Self {
                gamma_raw, omega, alpha, beta, rk4_weights,
                rk4_steps, n_bands, layer_idx,
                cache: Arc::new(Mutex::new(None)),
                param_grads,
            }
        }

        fn make_weights(&self) -> crate::model::KerrWeights {
            crate::model::KerrWeights {
                gamma_raw: self.gamma_raw.clone(),
                omega: self.omega.clone(),
                alpha: self.alpha,
                beta: self.beta,
                rk4_n_steps: self.rk4_steps,
                phase_correction: vec![0.0; self.n_bands], // corrector applied separately
                rk4_weights: self.rk4_weights,
                chi: 0.0,
            }
        }
    }

    impl CustomOp1 for KerrOdeCustomOp {
        fn name(&self) -> &'static str { "kerr_ode_rk4" }

        fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
            let input = match storage {
                CpuStorage::F32(data) => data,
                _ => return Err(Error::Msg("KerrOdeCustomOp expects F32".to_string())),
            };
            let dims = layout.dims();
            let n_pos = dims[0];
            let n_embd = dims[1];

            let weights = self.make_weights();

            // Run the EXACT same forward as CPU tier — with cache for backward
            let mut outputs = vec![0.0f32; n_pos * n_embd];
            let mut caches = Vec::with_capacity(n_pos);

            for pos in 0..n_pos {
                let start = layout.start_offset() + pos * n_embd;
                let x = &input[start..start + n_embd];
                let (out, cache) = crate::common::ode_backward::ode_forward_with_cache(x, &weights);
                outputs[pos * n_embd..(pos + 1) * n_embd].copy_from_slice(&out);
                caches.push(cache);
            }

            // Store cache for backward
            *self.cache.lock().unwrap() = Some(OdeCache {
                caches,
                weights: self.make_weights(),
            });

            Ok((CpuStorage::F32(outputs), Shape::from_dims(&[n_pos, n_embd])))
        }

        fn bwd(&self, _arg: &Tensor, _node: &Tensor, output_grad: &Tensor) -> Result<Option<Tensor>> {
            // Pull gradient to CPU
            let d_output_flat = output_grad.flatten_all()?.to_vec1::<f32>()?;
            let dims = output_grad.dims();
            let n_pos = dims[0];
            let n_embd = dims[1];

            let cache_lock = self.cache.lock().unwrap();
            let ode_cache = cache_lock.as_ref()
                .ok_or_else(|| Error::Msg("ODE backward called without forward cache".to_string()))?;

            // Run the EXACT same backward as CPU tier
            let mut d_inputs = vec![0.0f32; n_pos * n_embd];
            let mut total_d_gamma_raw = vec![0.0f32; self.n_bands];
            let mut total_d_alpha = 0.0f32;
            let mut total_d_beta = 0.0f32;
            let mut total_d_chi = 0.0f32;
            let mut total_d_rk4_weights = [0.0f32; 4];

            for pos in 0..n_pos {
                let d_out = &d_output_flat[pos * n_embd..(pos + 1) * n_embd];
                let (d_input, pg) = crate::common::ode_backward::ode_backward(
                    d_out, &ode_cache.caches[pos], &ode_cache.weights,
                );
                d_inputs[pos * n_embd..(pos + 1) * n_embd].copy_from_slice(&d_input);

                for k in 0..self.n_bands { total_d_gamma_raw[k] += pg.d_gamma_raw[k]; }
                total_d_alpha += pg.d_alpha;
                total_d_beta += pg.d_beta;
                total_d_chi += pg.d_chi;
                for w in 0..4 { total_d_rk4_weights[w] += pg.d_rk4_weights[w]; }
            }

            // Store param gradients in shared storage (Arc<Mutex>, thread-safe)
            {
                let mut v = self.param_grads.lock().unwrap();
                if self.layer_idx < v.len() {
                    v[self.layer_idx] = Some(OdeParamGradsAccum {
                        d_gamma_raw: total_d_gamma_raw,
                        d_alpha: total_d_alpha,
                        d_beta: total_d_beta,
                        d_chi: total_d_chi,
                        d_rk4_weights: total_d_rk4_weights,
                    });
                }
            }

            let d_input_tensor = Tensor::from_vec(d_inputs, output_grad.shape(), output_grad.device())?;
            Ok(Some(d_input_tensor))
        }
    }
}
