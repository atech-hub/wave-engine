//! FfnConfig — shared feature flags read by `ffn_forward_via_backend`.
//!
//! Populated once from CLI args (training path) or via `inference()` for
//! non-training callers. Replaces the four scattered boolean parameters
//! (`freeze_ode`, `use_corrector`, `ode_pathway`, `split_band`) that the FFN
//! function used to accept individually.
//!
//! Why this exists: the wave-engine has three compute tiers (CPU, wgpu, Candle).
//! CPU and wgpu share `common/ffn.rs` so new flags reach both automatically.
//! Candle historically reimplements the FFN and scans `std::env::args()` for
//! flags — which means every new flag has to be added there separately and
//! silently rots if anyone forgets. Consolidating the flag surface into one
//! struct is step 1 of retiring that divergence.
//!
//! Intentionally NOT in FfnConfig (would mix concerns):
//! * `memory` — per-layer runtime data, stays as a function arg alongside
//!   `layer_agc`.
//! * `curriculum_active` — controls band masking in the train loop *before*
//!   the FFN is called; not read inside `ffn_forward_via_backend`.
//! * `monitor` — already exposed via `ffn::PROFILE: AtomicBool`.
//! * `debug_nan` — Candle-only diagnostic; CPU has no equivalent.
//! * `attention_pathway` — read by attention forward, not FFN forward.
//! * `chi` — lives on `weights.kerr.chi`; the canonical source is the weight
//!   struct so FfnConfig doesn't duplicate it.

/// Per-run feature flags that steer `ffn_forward_via_backend`.
///
/// Every field maps 1:1 to a behaviour the FFN forward function currently
/// branches on. Defaults match the established training configuration
/// (pathways on, corrector on, split-band and freeze-ode off).
#[derive(Clone, Copy, Debug)]
pub struct FfnConfig {
    /// Real ODE Jacobian flows gradients to upstream parameters (default: true).
    /// Disabling this reverts to the legacy identity shortcut for A/B diagnostics.
    pub ode_pathway: bool,

    /// Freeze-and-decouple ODE integration for cleaner gradients (default: false).
    /// Phase A currently requires chi=0.
    pub split_band: bool,

    /// Identity shortcut over the ODE (default: false).
    /// Used only for A/B baselines; degrades gradient correctness when on.
    pub freeze_ode: bool,

    /// Enable the per-band corrector plate after the ODE (default: true).
    /// Off when the corrector DynParam is `off` or `--no-corrector` is set.
    pub use_corrector: bool,
}

impl Default for FfnConfig {
    fn default() -> Self {
        Self::inference()
    }
}

impl FfnConfig {
    /// Build from the four scattered flags (matches the old function-arg order
    /// so callsites can do a mechanical substitution). The library stays
    /// CLI-agnostic; `main::cmd_train` resolves clap args (including
    /// `--no-ode-pathway`, `--no-corrector`, etc.) into these four booleans
    /// and passes them in.
    pub fn from_flags(
        ode_pathway: bool,
        split_band: bool,
        freeze_ode: bool,
        use_corrector: bool,
    ) -> Self {
        Self { ode_pathway, split_band, freeze_ode, use_corrector }
    }

    /// Defaults for inference / analysis / generation paths that don't parse
    /// TrainArgs. ODE frozen, corrector on, pathway flags on, split-band off.
    /// Callers can adjust individual fields after construction.
    pub fn inference() -> Self {
        Self {
            ode_pathway: true,
            split_band: false,
            freeze_ode: true,
            use_corrector: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_defaults_match_doc() {
        let c = FfnConfig::inference();
        assert!(c.ode_pathway);
        assert!(!c.split_band);
        assert!(c.freeze_ode);
        assert!(c.use_corrector);
    }

    #[test]
    fn default_is_inference() {
        let d = FfnConfig::default();
        let i = FfnConfig::inference();
        assert_eq!(d.ode_pathway, i.ode_pathway);
        assert_eq!(d.split_band, i.split_band);
        assert_eq!(d.freeze_ode, i.freeze_ode);
        assert_eq!(d.use_corrector, i.use_corrector);
    }
}
