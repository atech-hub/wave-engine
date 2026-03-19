# Candle Backend — Status Report

## Date: 2026-03-19

## What Works
- Candle backend compiles and runs (feature flag: --features candle-backend)
- 4-layer model trains cleanly: loss 4.44 → 2.79 in 200 iters, zero NaN
- 400ms/iter at 4 layers (CPU), compared to wgpu's 130ms
- Autograd handles all gradients automatically (maestro, GELU, out_proj, LN, lm_head)
- Gradient clipping implemented (lr scaling workaround)
- Pipeline monitors show loss, grad norm, NaN detection

## What Doesn't Work (Yet)
- **ODE integration causes NaN.** The ode_delta workaround (detached ODE output + residual)
  creates numerical instability after 4-50 iterations depending on batch.
- Root cause: the ODE detached output doesn't participate in the grad graph,
  so maestro_in gets no gradient through the ODE path. The workaround
  (precond + ode_delta) re-attaches precond but the large delta values
  amplify gradients.

## Current Config
- ODE **skipped** in Candle backend (identity pass-through)
- Model trains without ODE — loss still descends because maestro + out_proj
  learn useful transformations even without ODE nonlinearity
- Loss 2.79 at 200 iters (4 layers) — comparable to wgpu's 2.50 at 200 iters

## GPU Status
- Device detected as CPU (CUDA not available on this Candle build)
- Need candle-core with CUDA feature for GPU: candle-core = { features = ["cuda"] }
- Metal feature available for Apple Silicon

## Next Steps
1. **Enable CUDA:** Add `features = ["cuda"]` to candle-core dependency
2. **Fix ODE gradient:** Implement Candle CustomOp1 trait for identity backward
3. **24-layer test:** Scale to production config once ODE is resolved
4. **Benchmark:** Compare Candle GPU vs wgpu at 24 layers
