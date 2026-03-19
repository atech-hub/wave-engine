# GPU Working — 2.3x Speedup Confirmed

## Date: 2026-03-18

## The Result

| Path | iter/s | Loss@0 | Loss@49 | Total (50 iters) |
|------|--------|--------|---------|-------------------|
| CPU only | 2.1-2.6s | 4.2928 | 2.8514 | 127.1s |
| GPU hybrid | **1.1s** | 4.2928 | 2.8514 | **55.3s** |

- Loss is **bit-identical** between CPU and GPU paths
- GPU is **2.3x faster** than CPU-only
- GPU% is 2-9% (burst pattern — small work, done fast, overlapped with CPU)

## What's on GPU

Only the **frozen attention out_proj** — a 768×768 × 64-position matvec per block per batch element.
This has no backward pass (attention is frozen) → no forward/backward FP mismatch.

The GPU work happens in parallel with CPU FFN work (parallel block architecture).
Neither waits for the other → wall-clock time = max(GPU, CPU) not sum.

## Why GPU% is Low But Speedup is Large

The GPU finishes its matvec in ~200μs. The CPU takes ~100ms for the FFN/ODE.
The GPU does a tiny fraction of the total compute but does it AT THE SAME TIME
as the CPU, eliminating the sequential bottleneck.

GPU% measures compute volume. Speedup measures parallelism. They're different.

## Config

24 layers, 768-dim, 384 bands, 12 heads, maestro_dim=16, rk4=16, lr=1e-4
batch=4, seq=64, Shakespeare char-level
15.5M trainable parameters (attention frozen)

## What This Means for the 2000-iter Overnight Run

At 1.1s/iter × 2000 iters = ~37 minutes with GPU
vs 2.3s/iter × 2000 iters = ~77 minutes CPU-only

The overnight run should use --gpu.
