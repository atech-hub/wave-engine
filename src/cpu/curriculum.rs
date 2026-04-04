//! Curriculum schedule — progressive band activation for training.

/// Progressive band curriculum — starts with fewer bands, opens progressively.
pub struct CurriculumSchedule {
    stages: Vec<(usize, f32)>,
}

impl CurriculumSchedule {
    pub fn default_4stage(n_bands: usize) -> Self {
        // Scale stages proportionally: 12.5%, 25%, 50%, 100% of bands
        // At 64 bands: 8, 16, 32, 64. At 384 bands: 48, 96, 192, 384.
        let s1 = (n_bands / 8).max(8);
        let s2 = (n_bands / 4).max(s1);
        let s3 = (n_bands / 2).max(s2);
        Self { stages: vec![(s1, 0.20), (s2, 0.25), (s3, 0.25), (n_bands, 0.30)] }
    }

    pub fn none(n_bands: usize) -> Self {
        Self { stages: vec![(n_bands, 1.0)] }
    }

    pub fn active_bands(&self, iter: usize, n_iters: usize) -> usize {
        let mut cumulative = 0.0f32;
        for &(bands, frac) in &self.stages {
            cumulative += frac;
            if iter < (cumulative * n_iters as f32) as usize { return bands; }
        }
        self.stages.last().unwrap().0
    }

    /// Compute per-band mask values with gradual ramp at stage transitions.
    /// Returns [n_bands] with values in [0.01, 1.0].
    /// Active bands = 1.0, suppressed = 0.01, ramping bands interpolate linearly.
    pub fn band_masks(&self, iter: usize, n_iters: usize, n_bands: usize) -> Vec<f32> {
        let ramp_iters = 200usize; // linear ramp over 200 iterations
        let mut masks = vec![0.01f32; n_bands];

        // Find each stage's start iter and band range
        let mut stage_start = 0usize;
        let mut prev_bands = 0usize;
        for &(bands, frac) in &self.stages {
            let stage_end = stage_start + (frac * n_iters as f32) as usize;

            // Bands from prev_bands..bands are activated at this stage
            for k in prev_bands..bands.min(n_bands) {
                if iter >= stage_start + ramp_iters {
                    // Fully active (past ramp)
                    masks[k] = 1.0;
                } else if iter >= stage_start {
                    // Ramping: linear from 0.01 to 1.0
                    let progress = (iter - stage_start) as f32 / ramp_iters as f32;
                    masks[k] = 0.01 + progress * 0.99;
                }
                // else: still suppressed (0.01)
            }

            prev_bands = bands;
            stage_start = stage_end;
        }

        masks
    }

    pub fn describe(&self, n_iters: usize) {
        let ramp = 200;
        print!("  Curriculum: ");
        let mut start = 0;
        for &(bands, frac) in &self.stages {
            let end = start + (frac * n_iters as f32) as usize;
            print!("{bands} bands (iters {start}-{end}, ramp {ramp})  ");
            start = end;
        }
        println!();
    }
}
