//! Candle backend — autograd-based training for wave-engine.
//!
//! Split into focused modules:
//!   candle_model.rs     — CandleWaveModel struct + CandleBlock struct + new()
//!   candle_forward.rs   — forward(), forward_with_curriculum(), forward_with_monitors()
//!   candle_attention.rs — wave_attention() + harmonic_backward()
//!   candle_train.rs     — train_candle() training loop
//!   candle_checkpoint.rs — extract_wchk_params() + load_wchk_params_into_varmap()
//!   candle_monitors.rs  — Monitor structs + compute functions + JSON serializers

#[cfg(feature = "candle-backend")]
pub mod engine {
    // Re-export all public items from split modules
    pub use crate::candle_tier::candle_model::model::*;
    pub use crate::candle_tier::candle_forward::forward::*;
    pub use crate::candle_tier::candle_attention::attention::*;
    pub use crate::candle_tier::candle_train::train::*;
    pub use crate::candle_tier::candle_checkpoint::checkpoint::*;
    pub use crate::candle_tier::candle_monitors::monitors::*;
}

// Stub when candle feature is not enabled
#[cfg(not(feature = "candle-backend"))]
pub mod engine {
    pub fn train_candle(_data_path: &str, _n_iters: usize, _n_bands: usize, _n_head: usize, _n_layers: usize, _maestro_dim: usize, _rk4_steps: usize, _out_proj_groups: usize, _debug_nan: bool, _alpha: f32, _beta: f32, _phase_native: bool) -> std::result::Result<(), String> {
        Err("Candle backend not enabled. Build with: cargo run --release --features candle-backend".to_string())
    }
}
