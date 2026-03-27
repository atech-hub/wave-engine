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
}

impl Dims {
    pub fn from_cli(n_bands: usize, n_head: usize, maestro_dim: usize, block_size: usize, rk4_steps: usize) -> Self {
        Self { n_bands, n_embd: n_bands * 2, n_head, maestro_dim, block_size, rk4_steps, m1: None, m2: None, tied: false }
    }
    pub fn with_tied(mut self, tied: bool) -> Self {
        self.tied = tied;
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
