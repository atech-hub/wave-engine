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

// ─── Memory scanning ───

/// Per-layer scan results for a memory file.
pub struct MemoryLayerScan {
    pub layer: usize,
    pub avg_magnitude: f32,
    pub max_band: usize,
    pub max_magnitude: f32,
    pub min_band: usize,
    pub min_magnitude: f32,
    pub top_bands: Vec<(usize, f32)>, // top 5 by magnitude
    pub catalog_matches: Vec<(String, usize)>, // sorted by count desc
    pub total_energy: f32,
    pub band_phases: Vec<f32>, // per-band phase (radians)
    pub band_magnitudes: Vec<f32>, // per-band magnitude
}

/// Scan a memory file and produce per-layer analysis.
pub fn scan_memory(memory: &WaveMemory) -> Vec<MemoryLayerScan> {
    use std::f32::consts::PI;
    let n_bands = memory.n_bands;

    memory.layers.iter().enumerate().map(|(layer_idx, layer)| {
        // Per-band phase and magnitude
        let phases: Vec<f32> = (0..n_bands).map(|k| layer.s[k].atan2(layer.r[k])).collect();
        let mags: Vec<f32> = (0..n_bands).map(|k| (layer.r[k] * layer.r[k] + layer.s[k] * layer.s[k]).sqrt()).collect();

        let avg_mag = mags.iter().sum::<f32>() / n_bands as f32;
        let total_energy: f32 = mags.iter().map(|m| m * m).sum();
        let max_band = mags.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
        let min_band = mags.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);

        // Top 5 bands
        let mut sorted: Vec<(usize, f32)> = mags.iter().enumerate().map(|(i, &m)| (i, m)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_bands: Vec<(usize, f32)> = sorted.into_iter().take(5).collect();

        // Catalog matching
        let catalog: &[(&str, f32, f32)] = &[
            ("conjunction", 0.0, 8.0), ("opposition", 180.0, 8.0),
            ("trine", 120.0, 8.0), ("square", 90.0, 7.0),
            ("quintile", 72.0, 2.0), ("sextile", 60.0, 6.0),
            ("semi-square", 45.0, 2.0), ("semi-sextile", 30.0, 2.0),
            ("quincunx", 150.0, 2.0), ("sesquiquadrate", 135.0, 2.0),
            ("bi-quintile", 144.0, 2.0),
        ];
        let mut cat_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for i in 0..n_bands {
            for j in (i + 1)..n_bands {
                let diff = (phases[j] - phases[i]).rem_euclid(2.0 * PI);
                let deg = diff.to_degrees();
                for &(name, angle, orb) in catalog {
                    let d = (deg - angle).abs().min((360.0 - (deg - angle).abs()).abs());
                    if d <= orb {
                        *cat_counts.entry(name.to_string()).or_insert(0) += 1;
                        break;
                    }
                }
            }
        }
        let mut catalog_matches: Vec<(String, usize)> = cat_counts.into_iter().collect();
        catalog_matches.sort_by_key(|e| std::cmp::Reverse(e.1));

        MemoryLayerScan {
            layer: layer_idx,
            avg_magnitude: avg_mag,
            max_band, max_magnitude: mags[max_band],
            min_band, min_magnitude: mags[min_band],
            top_bands,
            catalog_matches,
            total_energy,
            band_phases: phases,
            band_magnitudes: mags,
        }
    }).collect()
}

/// Print memory scan results.
pub fn print_memory_scan(memory: &WaveMemory, scans: &[MemoryLayerScan]) {
    println!("=== Wave Memory Scan ===");
    println!("  {} layers, {} bands, {} conversations", memory.n_layers(), memory.n_bands, memory.n_convos);
    println!("  Config: alpha={:.3}, beta={:.3}", memory.config.alpha, memory.config.beta);
    println!();

    for scan in scans {
        println!("--- Layer {} ---", scan.layer);
        println!("  Avg magnitude: {:.4}    Total energy: {:.4}", scan.avg_magnitude, scan.total_energy);
        print!("  Top bands:");
        for (band, mag) in &scan.top_bands {
            print!("  b{}({:.3})", band, mag);
        }
        println!();
        println!("  Weakest: b{}({:.4})", scan.min_band, scan.min_magnitude);
        println!("  Catalog matches:");
        for (name, count) in &scan.catalog_matches {
            println!("    {:20} {:5}", name, count);
        }
        println!();
    }

    // Cross-layer comparison
    println!("--- Cross-layer summary ---");
    println!("  {:>8} {:>10} {:>10} {:>12} {:>12}", "Layer", "Avg mag", "Energy", "Top band", "Conjunctions");
    for scan in scans {
        let conj = scan.catalog_matches.iter().find(|(n, _)| n == "conjunction").map(|(_, c)| *c).unwrap_or(0);
        println!("  {:>8} {:>10.4} {:>10.4} {:>12} {:>12}",
            scan.layer, scan.avg_magnitude, scan.total_energy,
            format!("b{}({:.3})", scan.top_bands[0].0, scan.top_bands[0].1),
            conj);
    }
}

/// Write memory scan to JSON.
pub fn write_memory_scan_json(path: &str, memory: &WaveMemory, scans: &[MemoryLayerScan]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    write!(f, "{{\n  \"n_layers\": {},\n  \"n_bands\": {},\n  \"n_convos\": {},\n",
        memory.n_layers(), memory.n_bands, memory.n_convos)?;
    write!(f, "  \"config\": {{\"alpha\": {:.6}, \"beta\": {:.6}}},\n", memory.config.alpha, memory.config.beta)?;
    write!(f, "  \"layers\": [\n")?;
    for (i, scan) in scans.iter().enumerate() {
        write!(f, "    {{\n")?;
        write!(f, "      \"layer\": {},\n", scan.layer)?;
        write!(f, "      \"avg_magnitude\": {:.6},\n", scan.avg_magnitude)?;
        write!(f, "      \"total_energy\": {:.6},\n", scan.total_energy)?;
        write!(f, "      \"max_band\": {}, \"max_magnitude\": {:.6},\n", scan.max_band, scan.max_magnitude)?;
        write!(f, "      \"min_band\": {}, \"min_magnitude\": {:.6},\n", scan.min_band, scan.min_magnitude)?;
        write!(f, "      \"top_bands\": [")?;
        for (j, (band, mag)) in scan.top_bands.iter().enumerate() {
            write!(f, "[{},{:.4}]{}", band, mag, if j + 1 < scan.top_bands.len() { "," } else { "" })?;
        }
        write!(f, "],\n")?;
        write!(f, "      \"catalog\": {{")?;
        for (j, (name, count)) in scan.catalog_matches.iter().enumerate() {
            write!(f, "\"{}\":{}{}", name, count, if j + 1 < scan.catalog_matches.len() { "," } else { "" })?;
        }
        write!(f, "}}\n")?;
        write!(f, "    }}{}\n", if i + 1 < scans.len() { "," } else { "" })?;
    }
    write!(f, "  ]\n}}\n")?;
    Ok(())
}

/// Save memory to disk. Logs errors but doesn't panic.
pub fn save(path: &str, memory: &WaveMemory) {
    if let Err(e) = kerr_memory::file::save(path, memory) {
        eprintln!("  [memory save failed: {e}]");
    } else {
        println!("  Memory saved to {path} ({} conversations)", memory.n_convos);
    }
}
