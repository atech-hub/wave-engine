//! Training and diagnostic monitors — grouped for clarity.
//!
//! Each monitor is a self-contained module that observes one aspect of the
//! training pipeline (gradients, attention, ODE dynamics, etc.) and reports
//! metrics. Monitors are called from train_loop.rs and train_health.rs.

pub mod monitor;
pub mod attn_monitor;
pub mod checkpoint_monitor;
pub mod curriculum_monitor;
pub mod dyn_param_monitor;
pub mod embedding_monitor;
pub mod encoding_health;
pub mod framework_monitor;
pub mod fwm_monitor;
pub mod gradient_monitor;
pub mod iq_monitor;
pub mod layer_flow_monitor;
pub mod ode_backward_monitor;
pub mod ode_dynamics_monitor;
pub mod ode_monitor;
pub mod output_monitor;
pub mod throughput_monitor;
pub mod junctions;  // Junction monitors — contract verification at component boundaries
