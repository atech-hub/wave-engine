//! Wave memory integration — load, inject, accumulate, save.
//!
//! Bridges kerr-memory (the library) with the wave-engine.
//! All memory logic lives here so that serve_tier, generate mode, and
//! future phase-native decoder variants can all use the same code.
//!
//! The accumulated WaveMemory state is per-layer r/s in the model's
//! native coordinate system — also usable as a comparison target
//! for phase-native decoding (the decoder-as-experience path).

use kerr_memory::memory::WaveMemory;

/// Pre-scaled memory offsets ready to inject into ODE initial conditions.
/// Also usable as a comparison target for phase-native decoding.
pub struct MemoryOffsets {
    pub offsets: Vec<(Vec<f32>, Vec<f32>)>,
}

impl MemoryOffsets {
    /// Build offset slices in the format forward_with_memory expects.
    pub fn as_slices(&self) -> Vec<(&[f32], &[f32])> {
        self.offsets.iter()
            .map(|(r, s)| (r.as_slice(), s.as_slice()))
            .collect()
    }
}

/// Build injection offsets from a WaveMemory (the read path).
pub fn build_offsets(memory: &WaveMemory) -> MemoryOffsets {
    let alpha = memory.config.alpha;
    MemoryOffsets {
        offsets: memory.layers.iter()
            .map(|l| l.scaled_offsets(alpha))
            .collect(),
    }
}

/// Merge extracted ODE states into persistent memory (the write path).
/// `ode_states` is per-layer (r_avg, s_avg) from model.extract_ode_states().
pub fn merge_ode_states(
    memory: &mut WaveMemory,
    ode_states: &[(Vec<f32>, Vec<f32>)],
) {
    let beta = memory.config.beta;
    let w = 1.0 - beta;
    for (layer_idx, (r_avg, s_avg)) in ode_states.iter().enumerate() {
        if layer_idx >= memory.layers.len() { break; }
        let n = memory.n_bands.min(r_avg.len());
        for k in 0..n {
            memory.layers[layer_idx].r[k] = beta * memory.layers[layer_idx].r[k] + w * r_avg[k];
            memory.layers[layer_idx].s[k] = beta * memory.layers[layer_idx].s[k] + w * s_avg[k];
        }
    }
    memory.n_convos += 1;
}

/// Load a KWMF file or create a fresh memory.
pub fn load_or_create(path: &str, n_ode_layers: usize, n_bands: usize) -> WaveMemory {
    if std::path::Path::new(path).exists() {
        println!("Loading wave memory from: {path}");
        let mem = kerr_memory::file::load(path)
            .expect("Failed to load memory file");
        println!("  Memory: {} layers, {} bands, {} conversations",
            mem.n_layers(), mem.n_bands, mem.n_convos);
        mem
    } else {
        println!("Creating new wave memory: {path}");
        let mem = WaveMemory::zeros(n_ode_layers, n_bands);
        println!("  Memory: {} layers, {} bands (fresh)",
            n_ode_layers, n_bands);
        mem
    }
}

/// Save memory to disk. Logs errors but doesn't panic.
pub fn save(path: &str, memory: &WaveMemory) {
    if let Err(e) = kerr_memory::file::save(path, memory) {
        eprintln!("  [memory save failed: {e}]");
    } else {
        println!("  Memory saved to {path} ({} conversations)", memory.n_convos);
    }
}
