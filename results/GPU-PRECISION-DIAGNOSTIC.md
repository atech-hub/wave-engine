# GPU Precision Diagnostic — Root Cause Found

## Date: 2026-03-18

## Diagnostic Results

| Operation | GPU vs CPU max_diff | Issue? |
|-----------|--------------------| -------|
| ODE (fused RK4, 16 steps, 384 bands) | **3.58e-7** | No — excellent precision |
| Linear (768×768 matvec, tiled shader) | **2.52e-4** | YES — 700x worse than ODE |

## Root Cause

The GPU matvec shader (tiled workgroup reduction) accumulates 768 f32 elements
in a different order than the CPU loop. Each element differs by ~2.5e-4.

Over 24 layers × 200 iterations, this 2.5e-4 per-element error compounds
catastrophically in the weight gradients when forward (GPU) and backward (CPU)
use different precision paths.

## Why the Working Config Works

The 1.1s/iter config uses GPU ONLY for attention out_proj. Attention is frozen —
no backward gradients flow through it. The 2.5e-4 error doesn't accumulate
because the weights never update.

## Why FFN GPU Diverges

The FFN out_proj is trained (weights update every iteration). GPU forward
produces output that differs from what CPU backward expects by 2.5e-4 per element.
The gradient `d_W = d_y × x^T` accumulates this error into weight updates.
After a few iterations, the weights diverge from the correct trajectory.

## Fix Options

1. **All-CPU FFN** (current working baseline): 1.1s/iter, loss correct. Ship this.
2. **Improve matvec shader**: Kahan summation or double-precision accumulator
   in the tiled reduction. Reduces error from 2.5e-4 to ~1e-7. Significant shader work.
3. **All-GPU FFN**: Both forward AND backward on GPU. No mismatch because both
   use the same accumulation order. Requires GPU backward implementation.
4. **Accept mismatch with smaller lr**: Lower lr reduces gradient magnitude,
   making the 2.5e-4 error less significant. Untested — may just slow divergence.

## Recommendation

Ship with 1.1s/iter CPU FFN + GPU frozen attention. That's correct and fast.
The matvec precision fix (Option 2) or GPU backward (Option 3) are separate
engineering projects that don't block the 24-layer model.

## The ODE Was Never the Problem

The GPU ODE is precise to 3.58e-7 — better than most GPU operations.
The fused RK4 shader produces excellent results. The divergence was
always from the matvec shader, not the ODE shader.
