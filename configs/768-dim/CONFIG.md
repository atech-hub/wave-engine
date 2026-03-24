# 768-dim Configuration — Production Tier

**Status:** PARTIAL — 24L Candle trained, CPU tier untested at this scale
**Hardware:** GPU recommended (CUDA via Candle tier), CPU possible but slow
**Use case:** Full English, production serving, 50K BPE vocabulary

---

## Proven Configuration (Candle GPU)

```bash
wave-engine data/combined_10mb.txt --candle \
  --layers 24 --n-bands 384 --n-head 12 \
  --iters 3000 --bpe \
  --checkpoint-name model_768_24L_50kbpe.safetensors
```

## Training Results (24L Candle, 50K BPE)

| Run | Corpus | Iters | Best loss | Time | Notes |
|-----|--------|-------|-----------|------|-------|
| 1 | 4.8MB Shakespeare mix | 3000 | 4.76 (best at iter 2000) | 3.9 hrs | Cosine decayed too fast |
| 2 | 12.4MB diverse English | 3000 | 3.12 (best at iter 614) | ~4 hrs | LR starved after peak |

## Architecture

| Parameter | Value |
|-----------|-------|
| Dimension | 768 (384 bands × 2) |
| Layers | 24 |
| Attention heads | 12 |
| Head dim | 64 |
| Maestro dim | 16 |
| Out proj | Block-diagonal, 6 groups |
| ODE solver | Perturbative (Candle) / RK4-16 (CPU) |
| ODE coupling | α=0.1, β=0.1 |
| Embeddings | Single-grid viable (separation 0.148) |

## Notes

- At 768-dim, single-grid embeddings have 0.148 separation for 50K vocab — viable but tight. Multi-grid would improve this to ~18.6 but hasn't been tested at this scale.
- The 24L model trains on Candle (CUDA) at 4.3s/iter. CPU tier at this scale is ~57s/iter (lm_head dominated).
- Per-band ODE clamp not yet ported to Candle tier.

## What Needs Testing

- [ ] Multi-grid embeddings at 384 bands (improvement over single-grid?)
- [ ] Per-band clamp on Candle tier
- [ ] Longer training runs (10K+ iters with proper cosine scheduling)
- [ ] CPU tier viability (speed, memory usage)
- [ ] Wave structure diagnostics at 24L depth
- [ ] Model serving through wave-server

## Pending

Blocked by: Candle tier multi-grid + clamp porting, longer training runs with proper LR scheduling.
