//! Training core — Adam optimizer, gradient clipping, TrainConfig, DynParam.
//!
//! The training loop itself is in train_loop.rs.
//! Curriculum schedule is in curriculum.rs.
//! Health monitoring and spring regulation are in train_health.rs.

// Re-export run_training so callers can still use train::run_training
pub use crate::cpu::train_loop::run_training;
pub use crate::cpu::curriculum::CurriculumSchedule;

// ─── Adam optimizer ─────────────────────────────────────────────

pub struct Adam {
    pub lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Adam {
    pub fn new(lr: f32, n: usize) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, m: vec![0.0; n], v: vec![0.0; n] }
    }
    pub fn from_checkpoint(lr: f32, t: usize, m: Vec<f32>, v: Vec<f32>) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t, m, v }
    }
    pub fn checkpoint_state(&self) -> (usize, &[f32], &[f32]) {
        (self.t, &self.m, &self.v)
    }
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step_wd(params, grads, 0.01);
    }
    /// AdamW: weight decay applied before momentum update.
    pub fn step_wd(&mut self, params: &mut [f32], grads: &[f32], wd: f32) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            if wd > 0.0 { params[i] -= self.lr * wd * params[i]; }
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) {
    let norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() { *g *= scale; }
    }
}

// ─── Training configuration ────────────────────────────────────

pub struct TrainConfig {
    pub data_path: String,
    pub n_iters: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub n_layers: usize,
    pub lr: f32,
    pub use_bpe: bool,
    pub tokenizer_path: String,
    pub resume_path: Option<String>,
    pub use_curriculum: bool,
    pub use_gpu: bool,
    pub use_monitor: bool,
    pub out_proj_groups: usize,
    pub checkpoint_name: String,
    pub n_bands: usize,
    pub n_head: usize,
    pub maestro_dim: usize,
    pub alpha: f32,
    pub beta: f32,
    pub agc_ceiling: Option<f32>, // None = auto-derive from alpha
    pub log_name: Option<String>, // Custom log filename (default: training_log_{tier}.jsonl)
    pub m1: Option<usize>,
    pub m2: Option<usize>,
    pub tied: bool,
    pub lm_rank: usize,
    pub wave_decode: bool,
    pub unfreeze_phases: bool,
    pub health_interval: usize, // 0 = disabled
    pub freeze_ode: bool,
    pub head_lr_floor: f32, // 0.0 = disabled, e.g. 0.00003 = 30% of 1e-4
    pub no_corrector: bool, // --no-corrector: disable corrector plate (A/B testing)
    pub layer_scale: DynParam, // --layer-scale dyn | --layer-scale 1.0,0.8,1.0,1.0
    pub lr_scale: DynParam,    // --lr-scale dyn | --lr-scale 1.0,1.5,1.5,0.5,1.0
    pub phase_native: bool,
    pub mix_strength: f32,  // --mix-strength 0.05 (coherent coupling, 0.0=off)
    pub fwm_strength: f32,  // --fwm-strength 5.0 (four-wave mixing chi, 0.0=off)
    pub pythagorean: bool,    // --phase-native: use phase coherence loss, no lm_head
    pub phase_temp: f32,       // temperature for phase-native softmax (default 1.0)
    pub spring_k: f32, // spring constant for dynamic params (0.0 = no spring, 0.1 = moderate)
    pub active_layers: Option<usize>, // --active-layers N: first N layers at eq=1.0, rest at eq=0.0
    pub rk4_weights: DynParam, // --rk4-weights dyn | --rk4-weights standard
    pub wd: DynParam,          // --wd dyn | --wd 0.01 | --wd 0.01,0.02,0.01,0.005,0.01
    pub harmonics: DynParam,   // --harmonics dyn | --harmonics 0.5,1.0,1.5,2.0
    pub agc_headroom: DynParam, // --agc-headroom dyn | --agc-headroom 2.0,3.0,3.0,4.0
    pub corrector: DynParam,    // --corrector dyn | --corrector off (replaces --no-corrector)
}

/// A parameter that can be fixed (manual value) or dynamic (model learns it).
#[derive(Clone)]
pub enum DynParam {
    Off,                    // not used
    Dynamic,                // model decides (with spring)
    Fixed(Vec<f32>),        // human prescribes per-group values
}

impl DynParam {
    pub fn is_active(&self) -> bool { !matches!(self, DynParam::Off) }
    pub fn is_dynamic(&self) -> bool { matches!(self, DynParam::Dynamic) }
}
