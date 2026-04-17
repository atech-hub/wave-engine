//! Pure math primitives for the wave-engine.
//!
//! Everything in this folder satisfies the purity contract:
//! deterministic, no shared state, no I/O, no side effects.
//! Given the same inputs, always returns the same outputs.

pub mod core;           // softplus, cross_entropy_backward (was common/math.rs)
pub mod backward;       // linear, layer_norm, gelu, softplus backward
pub mod ode_deriv;      // Kerr derivative, RK4 step, FWM coupling
pub mod ode_backward;   // ODE forward-with-cache, ODE backward, FWM Jacobian
pub mod attn_backward;  // attention backward ten-step pipe
pub mod phase_loss;     // phase-native loss function

// Re-export core math functions at the math:: level.
// crate::common::math::softplus continues to work without callers
// needing to know about the core submodule.
pub use core::*;
