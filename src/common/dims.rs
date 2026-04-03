//! Compile-time defaults and runtime dimensions.
//! Constants are fallbacks when CLI flags not provided.

// Compile-time defaults
pub const N_BANDS: usize = 384;
pub const N_EMBD: usize = N_BANDS * 2;
pub const N_HEAD: usize = 12;
pub const N_LAYERS: usize = 24;
pub const MAESTRO_DIM: usize = 16;
pub const BLOCK_SIZE: usize = 256;
pub const RK4_STEPS: usize = 16;

/// Pipeline profiling flag — set true via --monitor.
pub static PROFILE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Runtime model dimensions — replaces compile-time constants.
/// Passed through init_model, forward, backward, and analyze.
#[derive(Clone, Copy)]
pub struct Dims {
    pub n_bands: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub maestro_dim: usize,
    pub block_size: usize,
    pub rk4_steps: usize,
    pub m1: Option<usize>,
    pub m2: Option<usize>,
    pub tied: bool,
    pub lm_rank: usize, // 0 = full rank (no factoring)
    pub wave_decode: bool,
    pub unfreeze_phases: bool,
    pub learnable_ode: bool, // true = ODE backward active, false = identity (--freeze-ode)
    pub use_corrector: bool, // true = corrector plate active (per-band phase correction after ODE)
    pub use_layer_scale: bool, // true = per-layer residual scaling is learnable
    pub use_lr_scale: bool, // true = per-group LR scaling is learnable
    pub phase_temp: f32, // temperature for phase-native loss (0.0 = use default 1.0)
    pub pythagorean: bool, // true = Pythagorean sphere encoding (1/sqrt(n+1) magnitude decay)
    pub use_rk4_weights: bool, // true = per-layer RK4 combination weights are learnable
    pub use_dyn_harmonics: bool, // true = per-head harmonic numbers are learnable
}

impl Dims {
    pub fn from_cli(n_bands: usize, n_head: usize, maestro_dim: usize, block_size: usize, rk4_steps: usize) -> Self {
        Self { n_bands, n_embd: n_bands * 2, n_head, maestro_dim, block_size, rk4_steps, m1: None, m2: None, tied: false, lm_rank: 0, wave_decode: false, unfreeze_phases: false, learnable_ode: true, use_corrector: true, use_layer_scale: false, use_lr_scale: false, phase_temp: 0.0, pythagorean: false, use_rk4_weights: false, use_dyn_harmonics: false }
    }
    pub fn with_tied(mut self, tied: bool) -> Self {
        self.tied = tied;
        self
    }
    pub fn with_lm_rank(mut self, lm_rank: usize) -> Self {
        self.lm_rank = lm_rank;
        self
    }
    pub fn with_wave_decode(mut self, wd: bool) -> Self {
        self.wave_decode = wd;
        self
    }
    pub fn with_unfreeze_phases(mut self, uf: bool) -> Self {
        self.unfreeze_phases = uf;
        self
    }
    pub fn with_learnable_ode(mut self, lo: bool) -> Self {
        self.learnable_ode = lo;
        self
    }
    pub fn with_corrector(mut self, c: bool) -> Self {
        self.use_corrector = c;
        self
    }
    pub fn with_layer_scale(mut self, ls: bool) -> Self {
        self.use_layer_scale = ls;
        self
    }
    pub fn with_lr_scale(mut self, ls: bool) -> Self {
        self.use_lr_scale = ls;
        self
    }
    pub fn with_pythagorean(mut self, p: bool) -> Self {
        self.pythagorean = p;
        self
    }
    pub fn with_rk4_weights(mut self, rw: bool) -> Self {
        self.use_rk4_weights = rw;
        self
    }
    pub fn with_dyn_harmonics(mut self, dh: bool) -> Self {
        self.use_dyn_harmonics = dh;
        self
    }
    pub fn with_moduli(mut self, m1: Option<usize>, m2: Option<usize>) -> Self {
        self.m1 = m1;
        self.m2 = m2;
        self
    }
    pub fn defaults() -> Self {
        Self::from_cli(N_BANDS, N_HEAD, MAESTRO_DIM, BLOCK_SIZE, RK4_STEPS)
    }
}
