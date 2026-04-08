# Tier Capability Matrix

**Last updated:** 2026-04-08

Which ODE features work on which compute tier. This is the single source of truth for "what works where."

## Feature Matrix

| Feature | CPU | wgpu | candle | Notes |
|---|---|---|---|---|
| Damping (γ) | ✅ | ✅ | ✅ | Per-band, from softplus(gamma_raw) |
| SPM (α) | ✅ | ✅ | ✅ | Self-phase modulation |
| XPM (β) | ✅ | ✅ | ✅ | Cross-phase modulation, [1,1,0,1,1] kernel |
| FWM (χ) | ✅ | ✅ | ✅ | Four-wave mixing, Hamiltonian quartets |
| Learnable α | ✅ | ✅ | ✅ | |
| Learnable β | ✅ | ✅ | ✅ | |
| Learnable γ (per-band) | ✅ | ✅ | ✅ | |
| Learnable χ | ❌ | ❌ | ❌ | FWM Jacobian not derived; chi is constant |
| Learnable rk4_weights | ✅ | ✅ | ✅ | |
| Corrector plate | ✅ | ✅ | ✅ | Per-band phase correction post-ODE |
| AGC ceiling | ✅ | ✅ | ✅ | Automatic gain control |
| DerivativeCapture | ✅ | ❌ | ❌ | CPU-only diagnostic (damping/phase/FWM decomposition) |
| Perturbative fast path | ❌ | ✅ (χ=0 only) | ❌ | wgpu only; auto-falls back to fused RK4 when χ≠0 |

## Parity Testing

All tiers are measured against `ode_deriv::kerr_derivative_into` (CPU canonical) via the shared test battery in `src/common/ode_parity.rs`. 15 test cases covering zero input, sparse, broadband, various chi values, edge bands, and different band counts.

## Checkpoint Format

WCHK v4 persists χ in the header. Checkpoints trained with FWM retain their chi value across save/resume on any tier. v3 and earlier checkpoints default to χ=0.0 on load.

## How to Add a New Physics Term

1. Implement in CPU `ode_deriv.rs::kerr_derivative_into` first
2. Add test cases to `ode_parity.rs::generate_parity_battery`
3. Add to wgpu WGSL shader (`shaders/kerr_step_batch.wgsl`)
4. Add to candle tensor ops (`candle_tier/ode.rs::kerr_derivative`)
5. Update this matrix
6. Verify parity: `cargo test ode_parity`
