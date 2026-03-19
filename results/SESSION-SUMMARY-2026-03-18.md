# Wave Packet Engine — Session Summary 2026-03-18/19

## What was built

New engine from scratch: `C:\claude\wave-packet-engine`
- Parallel attn+FFN blocks (GPT-J formulation)
- Harmonic coherence attention (frozen, no training)
- Projected phase scoring (O(T²) not O(T²×n_bands))
- Dual-maestro Kerr-ODE FFN
- Ping-pong GPU buffers for forward/backward consistency
- Pipeline monitor (always-on timing)
- Full scaling ladder validated

## Key Results

### Architecture proof
- 109K params, loss 2.55, frozen attention at 128-dim — beats kerr-engine's 354K/2.62
- Architecture learns from wave mechanics alone, no attention training needed

### Scaling ladder (all pass, zero NaN)
| Dim | Params | iter/s | Loss@200 |
|-----|--------|--------|----------|
| 128 | 111K | 23ms | 2.66 |
| 768 | 2.6M | 430ms | 2.50 |
| 896 | 3.5M | 500ms | 2.49 |
| 1024 | 4.6M | 650ms | 2.47 |
| 1280 | 7.0M | 1.1s | 2.49 |
| 1536 | 10M | 2.4s | 2.48 |

### Layer scaling (768-dim, GPU hybrid)
| Layers | Params | iter/s | Loss@49 |
|--------|--------|--------|---------|
| 4 | 2.6M | 130ms | 3.03 |
| 8 | 5.2M | 290ms | 3.14 |
| 12 | 7.8M | 440ms | 3.20 |
| 24 | 15.5M | 830ms | 2.85 |
| 32 | 20.7M | 1.1s | 3.18 |
| 48 | 31.0M | 1.7s | 3.16 |

### GPU pipeline
- **Working config:** CPU maestro+ODE + GPU ping-pong out_proj = 0.8s/iter, loss 2.74
- **Ping-pong pattern:** forward writes to Buffer A, backward reads same Buffer A. Correct by construction.
- **Full GPU FFN:** works but maestro dim=16 dispatches add FP noise → loss 2.98 vs 2.74. Not worth it.
- **GPU%:** 22-27% sustained (ping-pong out_proj), 36% peak (full GPU FFN)
- **Matvec precision:** tiled shader 2.52e-4 vs CPU, ODE 3.58e-7 vs CPU

### Findings
- **Maestro ceiling:** dim=16 is universal constant, not a ratio. 48:1 compression at 768-dim works same as 8:1 at 128-dim.
- **GPU matvec precision:** tiled workgroup reduction gives 2.52e-4 error vs CPU. Kahan brings it to 5.34e-5. Neither is enough for mixed CPU/GPU training — need ping-pong pattern.
- **Ping-pong is the fix:** not Kahan, not f64, not matching accumulation order. Store forward values in GPU buffer, backward reads same buffer. Same bits = correct gradients.
- **Router analogy:** CPU = control plane (attention scoring, phase calculations). GPU = data plane (matvecs, ODE). They don't need to sync per-operation, just at block boundaries.

## Working Config for Production

```
24 layers, 768-dim, 384 bands, 12 heads
maestro_dim = 16, rk4_steps = 16, lr = 1e-4
batch = 4, seq = 64
CPU: maestro (dim=16), ODE (RK4), attention scoring, LN, optimizer
GPU: attention out_proj (frozen), FFN out_proj (ping-pong forward+backward)
iter/s: 0.8s | Loss@200: 2.74 | GPU%: 22-27%
```

## Next Steps (from Desktop's plan)
1. **4-layer experiment machine** — 130ms/iter, test everything fast
2. **Wikitext-103** — real English data, BPE tokenizer, 4 layers first
3. **24-layer production run** — 2000 iters, checkpoint, sample text
4. **Full GPU backward** — move maestro to GPU only if maestro_dim grows beyond 16
