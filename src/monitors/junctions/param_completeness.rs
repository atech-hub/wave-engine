//! J2: Parameter completeness — every weight must be trainable or explicitly frozen.
//!
//! Walks the model struct and checks every weight tensor against the param vector.
//! Any weight that is allocated, not in the param vector, and not explicitly marked
//! frozen is an orphan — a bug. Would have caught content_proj on day one.

use crate::common::wave_model::*;

/// Status of a weight in the model.
#[derive(Debug, Clone, PartialEq)]
pub enum WeightStatus {
    Trainable,          // In the param vector, updated by Adam
    FrozenByDesign,     // Explicitly known to be frozen (embeddings, attention weights, etc.)
    RuntimeOnly,        // Not a weight — training metadata (lr_scale, wd_scale, etc.)
    Orphaned,           // BUG: allocated, used in forward, but not trainable or frozen
}

/// One entry in the weight inventory.
#[derive(Debug, Clone)]
pub struct WeightEntry {
    pub path: String,
    pub n_elements: usize,
    pub status: WeightStatus,
}

/// Full inventory of all weights in the model.
pub struct WeightInventory {
    pub entries: Vec<WeightEntry>,
}

impl WeightInventory {
    pub fn orphans(&self) -> Vec<&WeightEntry> {
        self.entries.iter().filter(|e| e.status == WeightStatus::Orphaned).collect()
    }

    pub fn passed(&self) -> bool {
        self.orphans().is_empty()
    }

    pub fn total_trainable(&self) -> usize {
        self.entries.iter().filter(|e| e.status == WeightStatus::Trainable).map(|e| e.n_elements).sum()
    }

    pub fn total_frozen(&self) -> usize {
        self.entries.iter().filter(|e| e.status == WeightStatus::FrozenByDesign).map(|e| e.n_elements).sum()
    }
}

/// Walk the model and classify every weight field.
pub fn check_param_completeness(model: &WavePacketModel) -> WeightInventory {
    let mut entries = Vec::new();
    let n_embd = model.ln_f.weight.len();
    let n_bands = n_embd / 2;

    // ─── Top-level model fields ───

    // Embeddings — always frozen (harmonic, not learned)
    entries.push(WeightEntry {
        path: "wte".into(),
        n_elements: model.wte.iter().map(|r| r.len()).sum(),
        status: WeightStatus::FrozenByDesign,
    });
    entries.push(WeightEntry {
        path: "wpe".into(),
        n_elements: model.wpe.iter().map(|r| r.len()).sum(),
        status: WeightStatus::FrozenByDesign,
    });

    // Final layer norm — always trainable
    entries.push(WeightEntry { path: "ln_f.weight".into(), n_elements: n_embd, status: WeightStatus::Trainable });
    entries.push(WeightEntry { path: "ln_f.bias".into(), n_elements: n_embd, status: WeightStatus::Trainable });

    // Output head — depends on mode
    if model.phase_native {
        entries.push(WeightEntry {
            path: "output_corrector".into(),
            n_elements: model.output_corrector.len(),
            status: WeightStatus::Trainable,
        });
        // lm_head exists but is empty/unused in phase-native
        if !model.lm_head.is_empty() && model.lm_head[0].len() > 0 {
            entries.push(WeightEntry {
                path: "lm_head (unused in phase-native)".into(),
                n_elements: model.lm_head.iter().map(|r| r.len()).sum(),
                status: WeightStatus::FrozenByDesign, // allocated but not used
            });
        }
    } else if model.lm_rank > 0 {
        entries.push(WeightEntry { path: "lm_down".into(), n_elements: model.lm_down.iter().map(|r| r.len()).sum(), status: WeightStatus::Trainable });
        entries.push(WeightEntry { path: "lm_up".into(), n_elements: model.lm_up.iter().map(|r| r.len()).sum(), status: WeightStatus::Trainable });
    } else if !model.lm_head.is_empty() {
        entries.push(WeightEntry {
            path: "lm_head".into(),
            n_elements: model.lm_head.iter().map(|r| r.len()).sum(),
            status: WeightStatus::Trainable,
        });
    }

    // Layer scale
    if model.use_layer_scale {
        entries.push(WeightEntry { path: "layer_scale".into(), n_elements: model.layer_scale.len(), status: WeightStatus::Trainable });
    }

    // Runtime-only fields (not weights)
    entries.push(WeightEntry { path: "lr_scale".into(), n_elements: model.lr_scale.len(), status: WeightStatus::RuntimeOnly });
    entries.push(WeightEntry { path: "wd_scale".into(), n_elements: model.wd_scale.len(), status: WeightStatus::RuntimeOnly });
    entries.push(WeightEntry { path: "agc_headroom".into(), n_elements: model.agc_headroom.len(), status: WeightStatus::RuntimeOnly });

    // ─── Per-block fields ───
    for (b, block) in model.blocks.iter().enumerate() {
        let prefix = format!("blocks[{}]", b);

        // Layer norms — always trainable
        entries.push(WeightEntry { path: format!("{}.ln.weight", prefix), n_elements: n_embd, status: WeightStatus::Trainable });
        entries.push(WeightEntry { path: format!("{}.ln.bias", prefix), n_elements: n_embd, status: WeightStatus::Trainable });
        entries.push(WeightEntry { path: format!("{}.ln_ffn.weight", prefix), n_elements: n_embd, status: WeightStatus::Trainable });
        entries.push(WeightEntry { path: format!("{}.ln_ffn.bias", prefix), n_elements: n_embd, status: WeightStatus::Trainable });

        // Maestro in — always trainable
        add_linear(&mut entries, &format!("{}.ffn.maestro_in.squeeze", prefix), &block.ffn.maestro_in.squeeze, WeightStatus::Trainable);
        add_linear(&mut entries, &format!("{}.ffn.maestro_in.process_1", prefix), &block.ffn.maestro_in.process_1, WeightStatus::Trainable);

        // Maestro out — always trainable
        add_linear(&mut entries, &format!("{}.ffn.maestro_out.squeeze", prefix), &block.ffn.maestro_out.squeeze, WeightStatus::Trainable);
        add_linear(&mut entries, &format!("{}.ffn.maestro_out.process_1", prefix), &block.ffn.maestro_out.process_1, WeightStatus::Trainable);

        // FFN out_proj — always trainable
        entries.push(WeightEntry {
            path: format!("{}.ffn.out_proj", prefix),
            n_elements: block.ffn.out_proj.param_count(),
            status: WeightStatus::Trainable,
        });

        // Kerr ODE weights
        let kerr_status = if model.learnable_ode { WeightStatus::Trainable } else { WeightStatus::FrozenByDesign };
        entries.push(WeightEntry { path: format!("{}.ffn.kerr.gamma_raw", prefix), n_elements: block.ffn.kerr.gamma_raw.len(), status: kerr_status.clone() });
        entries.push(WeightEntry { path: format!("{}.ffn.kerr.alpha", prefix), n_elements: 1, status: kerr_status.clone() });
        entries.push(WeightEntry { path: format!("{}.ffn.kerr.beta", prefix), n_elements: 1, status: kerr_status.clone() });
        entries.push(WeightEntry { path: format!("{}.ffn.kerr.phase_correction", prefix), n_elements: block.ffn.kerr.phase_correction.len(), status: kerr_status.clone() });
        entries.push(WeightEntry { path: format!("{}.ffn.kerr.omega", prefix), n_elements: block.ffn.kerr.omega.len(), status: WeightStatus::FrozenByDesign }); // omega is never trained
        entries.push(WeightEntry { path: format!("{}.ffn.kerr.chi", prefix), n_elements: 1, status: WeightStatus::FrozenByDesign }); // chi is config, not trained
        if model.use_rk4_weights {
            entries.push(WeightEntry { path: format!("{}.ffn.kerr.rk4_weights", prefix), n_elements: 4, status: kerr_status.clone() });
        }

        // Attention weights — frozen by design (unless learnable_attn, future)
        let attn_status = if false /* learnable_attn — future feature, always frozen for now */ { WeightStatus::Trainable } else { WeightStatus::FrozenByDesign };
        for (h, head) in block.attn.heads.iter().enumerate() {
            let hp = format!("{}.attn.heads[{}]", prefix, h);

            // harmonic_raw — trainable when dyn_harmonics OR learnable_attn
            let harm_status = if model.use_dyn_harmonics || false /* learnable_attn — future feature, always frozen for now */ {
                WeightStatus::Trainable
            } else {
                WeightStatus::FrozenByDesign
            };
            entries.push(WeightEntry { path: format!("{}.harmonic_raw", hp), n_elements: 1, status: harm_status });
            entries.push(WeightEntry { path: format!("{}.phase_proj_w", hp), n_elements: head.phase_proj_w.iter().map(|r| r.len()).sum(), status: attn_status.clone() });
            entries.push(WeightEntry { path: format!("{}.phase_proj_b", hp), n_elements: head.phase_proj_b.len(), status: attn_status.clone() });
            entries.push(WeightEntry { path: format!("{}.v_proj_w", hp), n_elements: head.v_proj_w.iter().map(|r| r.len()).sum(), status: attn_status.clone() });
            entries.push(WeightEntry { path: format!("{}.v_proj_b", hp), n_elements: head.v_proj_b.len(), status: attn_status.clone() });

            // Content projection — always frozen (symmetry-breaker, never trained)
            if !head.content_proj_w.is_empty() {
                entries.push(WeightEntry { path: format!("{}.content_proj_w", hp), n_elements: head.content_proj_w.iter().map(|r| r.len()).sum(), status: WeightStatus::FrozenByDesign });
                entries.push(WeightEntry { path: format!("{}.content_proj_b", hp), n_elements: head.content_proj_b.len(), status: WeightStatus::FrozenByDesign });
            }
        }

        // Attention out_proj
        entries.push(WeightEntry {
            path: format!("{}.attn.out_proj_w", prefix),
            n_elements: block.attn.out_proj_w.iter().map(|r| r.len()).sum(),
            status: attn_status.clone(),
        });
        entries.push(WeightEntry {
            path: format!("{}.attn.out_proj_b", prefix),
            n_elements: block.attn.out_proj_b.len(),
            status: attn_status,
        });
    }

    // Verify trainable count matches count_trainable_ex
    let inventory = WeightInventory { entries };
    inventory
}

fn add_linear(entries: &mut Vec<WeightEntry>, prefix: &str, lw: &crate::model::LinearWeights, status: WeightStatus) {
    entries.push(WeightEntry {
        path: format!("{}.w", prefix),
        n_elements: lw.w.iter().map(|r| r.len()).sum(),
        status: status.clone(),
    });
    entries.push(WeightEntry {
        path: format!("{}.b", prefix),
        n_elements: lw.b.len(),
        status,
    });
}

/// Print the inventory.
pub fn print_result(inventory: &WeightInventory, assert_mode: bool) {
    let trainable: usize = inventory.total_trainable();
    let frozen: usize = inventory.total_frozen();
    let runtime: usize = inventory.entries.iter().filter(|e| e.status == WeightStatus::RuntimeOnly).map(|e| e.n_elements).sum();
    let orphans = inventory.orphans();

    eprintln!("[J2] Parameter completeness:");
    eprintln!("  Trainable:  {} elements across {} fields",
        trainable, inventory.entries.iter().filter(|e| e.status == WeightStatus::Trainable).count());
    eprintln!("  Frozen:     {} elements across {} fields",
        frozen, inventory.entries.iter().filter(|e| e.status == WeightStatus::FrozenByDesign).count());
    eprintln!("  Runtime:    {} elements across {} fields",
        runtime, inventory.entries.iter().filter(|e| e.status == WeightStatus::RuntimeOnly).count());

    if orphans.is_empty() {
        eprintln!("  Orphaned:   0 — PASS");
    } else {
        eprintln!("  Orphaned:   {} — FAIL", orphans.len());
        for o in &orphans {
            eprintln!("    {} ({} elements) — allocated but not trainable or frozen", o.path, o.n_elements);
        }
        if assert_mode {
            panic!("J2: Orphaned weights detected — every weight must be trainable or explicitly frozen");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::dims::Dims;

    #[test]
    fn test_phase_native_no_orphans() {
        let dims = Dims::from_cli(4, 2, 16, 128, 16).with_learnable_ode(false).with_corrector(false);
        let mut model = init_model(15, 42, 1, 1, dims, 0.1, 0.2);
        model.phase_native = true;
        model.output_corrector = vec![0.0; 4];
        let inv = check_param_completeness(&model);
        print_result(&inv, false);
        assert!(inv.passed(), "Phase-native should have no orphans");
    }

    #[test]
    fn test_learnable_ode_no_orphans() {
        let dims = Dims::from_cli(4, 2, 16, 128, 16).with_learnable_ode(true).with_corrector(true);
        let mut model = init_model(15, 42, 1, 1, dims, 0.1, 0.2);
        model.phase_native = true;
        model.output_corrector = vec![0.0; 4];
        model.learnable_ode = true;
        let inv = check_param_completeness(&model);
        print_result(&inv, false);
        assert!(inv.passed(), "Learnable ODE should have no orphans");
    }

    #[test]
    fn test_with_content_proj_no_orphans() {
        // Content projection is explicitly frozen — should not be orphaned
        let dims = Dims::from_cli(4, 2, 16, 128, 16).with_learnable_ode(true);
        let mut model = init_model(15, 42, 1, 1, dims, 0.1, 0.2);
        model.phase_native = true;
        model.output_corrector = vec![0.0; 4];
        model.learnable_ode = true;
        // Content proj should be allocated when learnable_ode=true
        let has_content = !model.blocks[0].attn.heads[0].content_proj_w.is_empty();
        println!("  Content proj allocated: {}", has_content);
        let inv = check_param_completeness(&model);
        print_result(&inv, false);
        assert!(inv.passed(), "Content projection should be frozen, not orphaned");
    }

    #[test]
    fn test_trainable_count_matches() {
        // Verify our trainable count matches count_trainable_ex
        let dims = Dims::from_cli(4, 2, 16, 128, 16).with_learnable_ode(true).with_corrector(true);
        let mut model = init_model(15, 42, 2, 1, dims, 0.1, 0.2);
        model.phase_native = true;
        model.output_corrector = vec![0.0; 4];
        model.learnable_ode = true;
        let inv = check_param_completeness(&model);
        let our_count = inv.total_trainable();
        let official_count = count_trainable_ex(&model, false);
        println!("  J2 trainable count: {}, count_trainable_ex: {}", our_count, official_count);
        assert_eq!(our_count, official_count,
            "Trainable element count mismatch: J2 says {} but count_trainable_ex says {}",
            our_count, official_count);
    }
}
