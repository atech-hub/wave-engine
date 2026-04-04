//! Checkpoint save/load for WCHK compatibility.

#[cfg(feature = "candle-backend")]
pub mod checkpoint {
    use candle_core::{Device, Result, Tensor};
    use candle_nn::VarMap;

    use crate::candle_tier::candle_model::model::CandleWaveModel;

    /// Load WCHK flat params into candle VarMap.
    /// Reverse of extract_wchk_params — maps flat param vector to named variables.
    pub fn load_wchk_params_into_varmap(
        varmap: &VarMap, params: &[f32],
        n_layers: usize, n_embd: usize, maestro_dim: usize,
        vocab_size: usize, out_proj_groups: usize, n_bands: usize,
        has_ode: bool, has_ls: bool, has_rk4: bool, phase_native: bool,
        device: &Device,
    ) -> Result<()> {
        let mut idx = 0;
        let set_var = |varmap: &VarMap, key: &str, vals: &[f32], shape: &[usize], device: &Device| -> Result<()> {
            let data = varmap.data().lock().unwrap();
            if let Some(var) = data.get(key) {
                let t = Tensor::from_slice(vals, shape, device)?;
                var.set(&t)?;
            }
            Ok(())
        };

        for i in 0..n_layers {
            let p = format!("block.{i}");
            // LN
            set_var(varmap, &format!("{p}.ln_w"), &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
            set_var(varmap, &format!("{p}.ln_b"), &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
            // LN FFN (skip — candle uses shared LN)
            idx += n_embd * 2;
            // Maestro in squeeze
            set_var(varmap, &format!("{p}.mae_in_sq.weight"), &params[idx..idx+maestro_dim*n_embd], &[maestro_dim, n_embd], device)?; idx += maestro_dim * n_embd;
            set_var(varmap, &format!("{p}.mae_in_sq.bias"), &params[idx..idx+maestro_dim], &[maestro_dim], device)?; idx += maestro_dim;
            // Maestro in process
            set_var(varmap, &format!("{p}.mae_in_pr.weight"), &params[idx..idx+n_embd*maestro_dim], &[n_embd, maestro_dim], device)?; idx += n_embd * maestro_dim;
            set_var(varmap, &format!("{p}.mae_in_pr.bias"), &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
            // Maestro out squeeze
            set_var(varmap, &format!("{p}.mae_out_sq.weight"), &params[idx..idx+maestro_dim*n_embd], &[maestro_dim, n_embd], device)?; idx += maestro_dim * n_embd;
            set_var(varmap, &format!("{p}.mae_out_sq.bias"), &params[idx..idx+maestro_dim], &[maestro_dim], device)?; idx += maestro_dim;
            // Maestro out process
            set_var(varmap, &format!("{p}.mae_out_pr.weight"), &params[idx..idx+n_embd*maestro_dim], &[n_embd, maestro_dim], device)?; idx += n_embd * maestro_dim;
            set_var(varmap, &format!("{p}.mae_out_pr.bias"), &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
            // Out proj
            if out_proj_groups <= 1 {
                set_var(varmap, &format!("{p}.out_proj.weight"), &params[idx..idx+n_embd*n_embd], &[n_embd, n_embd], device)?; idx += n_embd * n_embd;
                set_var(varmap, &format!("{p}.out_proj.bias"), &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
            } else {
                let gs = n_embd / out_proj_groups;
                for g in 0..out_proj_groups {
                    set_var(varmap, &format!("{p}.out_proj.g{g}.weight"), &params[idx..idx+gs*gs], &[gs, gs], device)?; idx += gs * gs;
                    set_var(varmap, &format!("{p}.out_proj.g{g}.bias"), &params[idx..idx+gs], &[gs], device)?; idx += gs;
                }
            }
            // ODE params
            if has_ode {
                set_var(varmap, &format!("{p}.ode.gamma_raw"), &params[idx..idx+n_bands], &[1, n_bands], device)?; idx += n_bands;
                set_var(varmap, &format!("{p}.ode.alpha"), &params[idx..idx+1], &[1, 1], device)?; idx += 1;
                set_var(varmap, &format!("{p}.ode.beta"), &params[idx..idx+1], &[1, 1], device)?; idx += 1;
                set_var(varmap, &format!("{p}.phase_correction"), &params[idx..idx+n_bands], &[1, n_bands], device)?; idx += n_bands;
                if has_rk4 {
                    set_var(varmap, &format!("{p}.ode.rk4_weights"), &params[idx..idx+4], &[4], device)?; idx += 4;
                }
            }
        }
        // Layer scale
        if has_ls {
            for i in 0..n_layers {
                set_var(varmap, &format!("block.{i}.layer_scale"), &params[idx..idx+1], &[1], device)?; idx += 1;
            }
        }
        // ln_f
        set_var(varmap, "ln_f_w", &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
        set_var(varmap, "ln_f_b", &params[idx..idx+n_embd], &[n_embd], device)?; idx += n_embd;
        // Phase-native: output corrector. Standard: lm_head.
        if phase_native {
            set_var(varmap, "output_corrector", &params[idx..idx+n_bands], &[1, n_bands], device)?; idx += n_bands;
        } else {
            set_var(varmap, "lm_head", &params[idx..idx+vocab_size*n_embd], &[vocab_size, n_embd], device)?; idx += vocab_size * n_embd;
        }

        if idx != params.len() {
            eprintln!("  WARNING: WCHK param count mismatch: read {} of {}", idx, params.len());
        }
        Ok(())
    }

    pub fn extract_wchk_params(varmap: &VarMap, model: &CandleWaveModel, n_layers: usize, n_embd: usize, maestro_dim: usize,
                            vocab_size: usize, out_proj_groups: usize, n_bands: usize,
                            phase_native: bool, use_rk4_dyn: bool, use_layer_scale: bool,
                            use_harmonics: bool) -> Vec<f32> {
        let mut params = Vec::new();

        let get_flat = |name: &str| -> Vec<f32> {
            let data = varmap.data().lock().unwrap();
            data.get(name).map(|t| t.flatten_all().unwrap().to_vec1::<f32>().unwrap()).unwrap_or_default()
        };

        for i in 0..n_layers {
            let p = format!("block.{i}");
            // LN weights
            params.extend(get_flat(&format!("{p}.ln_w")));
            params.extend(get_flat(&format!("{p}.ln_b")));
            // LN FFN — placeholder (candle uses shared LN)
            params.extend(vec![1.0f32; n_embd]);
            params.extend(vec![0.0f32; n_embd]);
            // Maestro in
            params.extend(get_flat(&format!("{p}.mae_in_sq.weight")));
            params.extend(get_flat(&format!("{p}.mae_in_sq.bias")));
            params.extend(get_flat(&format!("{p}.mae_in_pr.weight")));
            params.extend(get_flat(&format!("{p}.mae_in_pr.bias")));
            // Maestro out
            params.extend(get_flat(&format!("{p}.mae_out_sq.weight")));
            params.extend(get_flat(&format!("{p}.mae_out_sq.bias")));
            params.extend(get_flat(&format!("{p}.mae_out_pr.weight")));
            params.extend(get_flat(&format!("{p}.mae_out_pr.bias")));
            // Out proj — dense (groups=1) or block-diagonal (groups>1)
            if out_proj_groups <= 1 {
                params.extend(get_flat(&format!("{p}.out_proj.weight")));
                params.extend(get_flat(&format!("{p}.out_proj.bias")));
            } else {
                for g in 0..out_proj_groups {
                    params.extend(get_flat(&format!("{p}.out_proj.g{g}.weight")));
                    params.extend(get_flat(&format!("{p}.out_proj.g{g}.bias")));
                }
            }
            // ODE params (learnable)
            let gamma = get_flat(&format!("{p}.ode.gamma_raw"));
            if !gamma.is_empty() {
                params.extend(&gamma);
                params.extend(get_flat(&format!("{p}.ode.alpha")));
                params.extend(get_flat(&format!("{p}.ode.beta")));
                params.extend(get_flat(&format!("{p}.phase_correction")));
                if use_rk4_dyn {
                    params.extend(get_flat(&format!("{p}.ode.rk4_weights")));
                }
            }
            // Harmonics (if dynamic) — stored on CandleBlock, not in VarMap
            if use_harmonics {
                params.extend_from_slice(&model.blocks[i].harmonic_ns);
            }
        }
        // Layer scale
        if use_layer_scale {
            for i in 0..n_layers {
                let ls = get_flat(&format!("block.{i}.layer_scale"));
                if !ls.is_empty() { params.extend(&ls); } else { params.push(1.0); }
            }
        }
        // ln_f
        params.extend(get_flat("ln_f_w"));
        params.extend(get_flat("ln_f_b"));
        // Phase-native: output corrector. Standard: lm_head.
        if phase_native {
            let oc = get_flat("output_corrector");
            if !oc.is_empty() {
                params.extend(&oc);
            } else {
                params.extend(vec![0.0f32; n_bands]);
            }
        } else {
            params.extend(get_flat("lm_head"));
        }

        params
    }
}
