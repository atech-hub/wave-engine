# Tier Capability Matrix

**Last updated:** 2026-04-10

Which ODE features work on which compute tier. This is the single source of truth for "what works where."

## Feature Matrix

| Feature | CPU | wgpu | candle | Notes |
|---|---|---|---|---|
| Damping (γ) | ✅ | ✅ | ✅ | Per-band, from softplus(gamma_raw) |
| SPM (α) | ✅ | ✅ | ✅ | Self-phase modulation |
| XPM (β) | ✅ | ✅ | ✅ | Cross-phase modulation, [1,1,0,1,1] kernel |
| FWM forward (χ) | ✅ | ✅ | ✅ | Four-wave mixing, Hamiltonian quartets |
| FWM Jacobian (backward) | ✅ | ✅ (fixed v0.5.0) | ✅ | Analytical per-quartet gradient, all 8 role partials. wgpu had 13-binding bug (max_storage_buffers_per_shader_stage=12), fixed by packing d_alpha+d_beta into d_ab_partial |
| Learnable α | ✅ | ✅ | ✅ | |
| Learnable β | ✅ | ✅ | ✅ | |
| Learnable γ (per-band) | ✅ | ✅ | ✅ | |
| Learnable χ | ❌ | ❌ | ❌ | d_chi computed but not wired to optimizer (Phase 4 future work) |
| Learnable rk4_weights | ✅ | ✅ | ✅ | |
| Corrector plate | ✅ | ✅ | ✅ | Per-band phase correction post-ODE |
| AGC ceiling | ✅ | ✅ | ✅ | Automatic gain control |
| DerivativeCapture (fwd) | ✅ | ❌ | ❌ | CPU-only diagnostic (damping/phase/FWM forward decomposition) |
| BackwardCapture (bwd) | ✅ | ❌ | ❌ | CPU-only diagnostic (gradient flow per physics term) |
| Perturbative fast path | ❌ | ✅ (χ=0 only) | ❌ | wgpu only; auto-falls back to fused RK4 when χ≠0 |
| CUDA fused kernel | ❌ | ❌ | ✅ | Fused AGC+RK4+FWM forward+backward, --cuda-kernel flag |

## Parity Testing

All tiers are measured against `ode_deriv::kerr_derivative_into` (CPU canonical) via the shared test battery in `src/common/ode_parity.rs`. 15 test cases covering zero input, sparse, broadband, various chi values, edge bands, and different band counts.

## Gradient Checking

`wave-engine --check-gradients` validates analytical gradients against finite differences. At chi=0: 171/171 PASS. At chi=0.03: 172/172 PASS (includes d_chi). The FWM Jacobian is complete across all tiers.

## Checkpoint Format

WCHK v4 persists χ in the header. Checkpoints trained with FWM retain their chi value across save/resume on any tier. v3 and earlier checkpoints default to χ=0.0 on load. CPU resume applies checkpoint chi as fallback when CLI doesn't specify --fwm-strength.

## Monitors (13 total at health intervals)

| Monitor | JSONL type | What it measures |
|---|---|---|
| Attention heads | attn_heads | Per-head entropy, max weight, harmonic |
| Layer flow | layer_flow | Norms, ratios, cosine, band amplitudes |
| Gradient breakdown | grad_flow | Per-component gradient norms |
| Embedding space | embedding_space | Token separation, band utilization (iter 0 only) |
| Output distribution | output_dist | Logit entropy, margins, mode collapse |
| ODE dynamics | ode_dynamics | Phase velocity, energy ratio, damping |
| ODE forward decomposition | ode_decomposition | damping/phase/FWM fractions per layer |
| ODE backward decomposition | ode_backward_decomposition | Gradient flow per physics term, d_chi norm |
| I/Q analysis | iq | I/Q discrimination, phase stats |
| Dynamic params | dyn_params | Layer scale, RK4 weights, spring tension |
| Curriculum transitions | curriculum | Stage changes, loss jumps |
| Checkpoint drift | drift | L2 distance between saves |
| Throughput | throughput | Tokens/sec, iter timing |

## Analysis Tools

| Tool | Flag | What it does |
|---|---|---|
| Analyze | --analyze | Wave structure diagnostics, embedding health |
| ODE monitor | --ode-monitor | Per-band ODE data for specific prompts |
| Phase decode | --phase-decode | Compare lm_head vs phase coherence decoding |
| Galaxy scan | --galaxy-scan | End-of-training geometric inventory (5 layers) |
| Phase encode | --encode* | Direct phase injection into ODE layers |
| Relate | --relate | Per-harmonic coherence profiles between encodings |
| Relate vocab | --relate-vocab | Full vocabulary pairwise relationship matrix + energy signatures |
| Generate | --generate | Text generation from checkpoint |
| Scale | --scale | Scale checkpoint to different dimensions |
| Gradient check | --check-gradients | Finite-diff validation of analytical gradients |
| Recommend | --recommend | Architecture recommendations for a dataset |
| Framework monitor | (at health intervals) | Live harmonic coherence during training |

## How to Add a New Physics Term

1. Implement in CPU `ode_deriv.rs::kerr_derivative_into` first
2. Derive analytical Jacobian in `ode_backward.rs::deriv_backward`
3. Add test cases to `ode_parity.rs::generate_parity_battery`
4. Verify gradient checker passes: `wave-engine --check-gradients --fwm-strength X`
5. Add to wgpu forward shader (`shaders/kerr_step_batch.wgsl`)
6. Add to wgpu backward shader (`shaders/kerr_backward_batch.wgsl`)
7. Add to candle tensor ops (`candle_tier/ode.rs::kerr_derivative`)
8. Add to candle CUDA forward+backward kernels (`candle_tier/cuda_ode.rs`)
9. Update this matrix
10. Verify parity: `cargo test ode_parity`
