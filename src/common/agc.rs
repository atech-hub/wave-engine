/// Automatic Gain Control for ODE input magnitude.
///
/// Like a radio AGC circuit — the model finds its own operating range:
///   - Detector: measures per-band magnitudes each forward pass
///   - Memory: EMA tracks running mean and variance
///   - Gain control: knee compressor with adaptive threshold
///
/// The threshold = EMA_mean + headroom × EMA_std_dev.
/// Below threshold: signal passes through UNCHANGED (no compression tax).
/// Above threshold: smooth compression on the EXCESS only.
/// The threshold adapts as the model learns — no manual tuning.
///
/// Electronics analogy:
///   - Fixed resistor (hard clamp at 2.5): clips, distorts, maestro fights it
///   - Zener diode (tanh): smooth but compresses everything, even normal signal
///   - AGC voltage regulator (this): adapts to actual signal, only clips outliers

/// AGC state — one per ODE call (global across layers, or per-layer if needed).
#[derive(Clone, Debug)]
pub struct OdeAgc {
    /// Running mean of per-band magnitudes (EMA)
    ema_mean: f32,
    /// Running variance (EMA of squared deviations)
    ema_var: f32,
    /// Headroom: threshold = mean + headroom × std_dev
    /// 3.0 = three sigma (99.7% of normal magnitudes pass through untouched)
    headroom: f32,
    /// EMA decay rate. 0.995 = adapts over ~200 iterations.
    decay: f32,
    /// Minimum threshold — prevents collapse (ODE stability floor)
    min_threshold: f32,
    /// Maximum threshold — ODE physics ceiling. δφ < 90° requires mag < sqrt(π/2 / (α+4β)).
    /// At α=β=0.01: sqrt(π/2 / 0.05) ≈ 5.6. Set to 6.0 for margin.
    max_threshold: f32,
    /// Current computed threshold (read by training loop for logging)
    pub threshold: f32,
    /// Iteration count (for warmup)
    count: usize,
}

impl OdeAgc {
    /// Create AGC with physics-derived ceiling from coupling constants.
    /// ceiling = sqrt(pi/2 / (alpha + 4*beta)) — the ODE stability limit.
    pub fn from_coupling(alpha: f32, beta: f32) -> Self {
        let physics_ceiling = (std::f32::consts::FRAC_PI_2 / (alpha + 4.0 * beta)).sqrt();
        Self::with_ceiling(physics_ceiling)
    }

    /// Create AGC with explicit ceiling (for --agc-ceiling override).
    pub fn with_ceiling(ceiling: f32) -> Self {
        let ceiling = ceiling.max(0.5); // absolute minimum
        Self {
            ema_mean: 1.0,
            ema_var: 0.5,
            headroom: 3.0,
            decay: 0.995,
            min_threshold: (ceiling * 0.3).max(0.5), // floor scales with ceiling
            max_threshold: ceiling,
            threshold: (ceiling * 0.75).max(0.5),     // start at 75% of ceiling
            count: 0,
        }
    }

    /// Create AGC with explicit ceiling and custom headroom.
    pub fn with_ceiling_headroom(ceiling: f32, headroom: f32) -> Self {
        let ceiling = ceiling.max(0.5);
        Self {
            ema_mean: 1.0,
            ema_var: 0.5,
            headroom,
            decay: 0.995,
            min_threshold: (ceiling * 0.3).max(0.5),
            max_threshold: ceiling,
            threshold: (ceiling * 0.75).max(0.5),
            count: 0,
        }
    }

    /// Create with defaults (backward compat — uses α=0.1 coupling).
    pub fn new() -> Self {
        Self::from_coupling(0.1, 0.1)
    }

    /// Update ceiling from current coupling constants.
    /// Called after each optimizer step when ODE params are learnable.
    pub fn update_ceiling(&mut self, alpha: f32, beta: f32) {
        let physics = (std::f32::consts::FRAC_PI_2 / (alpha + 4.0 * beta)).sqrt().max(0.5);
        self.max_threshold = physics;
        self.min_threshold = (physics * 0.3).max(0.5);
        // Clamp current threshold to new ceiling if needed
        if self.threshold > self.max_threshold {
            self.threshold = self.max_threshold;
        }
    }

    /// Update ceiling with explicit CLI maximum override.
    pub fn update_ceiling_with_max(&mut self, alpha: f32, beta: f32, cli_max: Option<f32>) {
        let physics = (std::f32::consts::FRAC_PI_2 / (alpha + 4.0 * beta)).sqrt().max(0.5);
        let effective = match cli_max {
            Some(c) => physics.min(c),
            None => physics,
        };
        self.max_threshold = effective;
        self.min_threshold = (effective * 0.3).max(0.5);
        if self.threshold > self.max_threshold {
            self.threshold = self.max_threshold;
        }
    }

    /// Update AGC state with observed magnitudes.
    /// Call BEFORE applying compression — measures the raw maestro output.
    pub fn observe(&mut self, magnitudes: &[f32]) {
        if magnitudes.is_empty() { return; }

        // Guard against NaN and runaway values — either poisons the EMA permanently.
        // Filter: must be finite AND below 2× physics ceiling (if mag > 12, model is broken).
        let ceiling_2x = self.max_threshold * 2.0;
        let clean: Vec<f32> = magnitudes.iter()
            .filter(|m| m.is_finite() && **m < ceiling_2x)
            .copied().collect();
        if clean.is_empty() { return; }

        let n = clean.len() as f32;
        let batch_mean: f32 = clean.iter().sum::<f32>() / n;
        let batch_var: f32 = clean.iter()
            .map(|&m| (m - batch_mean) * (m - batch_mean))
            .sum::<f32>() / n;

        // Double-check the computed stats are finite
        if !batch_mean.is_finite() || !batch_var.is_finite() { return; }

        self.count += 1;

        if self.count < 10 {
            // Warmup: use batch stats directly
            self.ema_mean = batch_mean;
            self.ema_var = batch_var;
        } else {
            // EMA update
            self.ema_mean = self.decay * self.ema_mean + (1.0 - self.decay) * batch_mean;
            self.ema_var = self.decay * self.ema_var + (1.0 - self.decay) * batch_var;
        }

        // Adaptive threshold: mean + headroom * std_dev, bounded by physics
        let std_dev = self.ema_var.sqrt().max(0.01);
        self.threshold = (self.ema_mean + self.headroom * std_dev)
            .max(self.min_threshold)   // floor: prevent collapse
            .min(self.max_threshold);  // ceiling: ODE stability limit
    }

    /// Knee compressor: unchanged below threshold, smooth compression above.
    ///
    /// Unlike tanh which compresses EVERYTHING:
    ///   tanh at mag=4.0, threshold=5.0 → output 3.32 (17% loss on normal signal!)
    ///
    /// Knee compressor at mag=4.0, threshold=5.0 → output 4.0 (zero loss!)
    /// Knee compressor at mag=7.0, threshold=5.0 → output 6.52 (gentle compression on excess)
    ///
    /// Formula: below threshold → pass through. Above → compress only the excess.
    ///   output = threshold + threshold × tanh((mag - threshold) / threshold)
    #[inline]
    fn knee_compress(mag: f32, threshold: f32) -> f32 {
        if mag <= threshold {
            mag // pass through — zero compression in normal range
        } else {
            // Smooth compression on excess only
            let excess = mag - threshold;
            threshold + threshold * (excess / threshold).tanh()
        }
    }

    /// Observe magnitudes and apply knee compression.
    /// Returns (bands_in_compression, max_pre_compress_magnitude).
    pub fn process(&mut self, precond: &mut [Vec<f32>], n_bands: usize) -> (usize, f32) {
        // 1. Collect magnitudes for observation (BEFORE compression)
        let mags: Vec<f32> = precond.iter()
            .flat_map(|pos| (0..n_bands).map(move |k| {
                let r = pos[k * 2];
                let s = pos[k * 2 + 1];
                (r * r + s * s).sqrt()
            }))
            .collect();

        // 2. Update AGC state (threshold adapts)
        self.observe(&mags);

        // 3. Apply knee compression with adaptive threshold
        let threshold = self.threshold;
        let mut compress_count = 0usize;
        let mut max_mag = 0.0f32;

        for pos_vec in precond.iter_mut() {
            for k in 0..n_bands {
                let r = pos_vec[k * 2];
                let s = pos_vec[k * 2 + 1];
                let mag = (r * r + s * s).sqrt();

                if mag > max_mag { max_mag = mag; }

                if mag > threshold && mag > 0.001 {
                    // Knee compress: only the excess above threshold
                    let compressed = Self::knee_compress(mag, threshold);
                    let scale = compressed / mag;
                    pos_vec[k * 2] *= scale;
                    pos_vec[k * 2 + 1] *= scale;
                    compress_count += 1;
                }
                // Below threshold: NOTHING happens. Zero compression tax.
            }
        }

        (compress_count, max_mag)
    }

    /// Get current stats for JSONL logging.
    pub fn stats(&self) -> AgcStats {
        AgcStats {
            threshold: self.threshold,
            ema_mean: self.ema_mean,
            ema_std: self.ema_var.sqrt(),
            count: self.count,
        }
    }
}

/// AGC diagnostic stats for JSONL telemetry.
#[derive(Clone, Debug)]
pub struct AgcStats {
    pub threshold: f32,
    pub ema_mean: f32,
    pub ema_std: f32,
    pub count: usize,
}
