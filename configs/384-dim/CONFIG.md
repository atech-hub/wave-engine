# 384-dim Configuration — Power User Tier

**Status:** PLANNED — findings from 168-dim carry forward
**Hardware:** CPU or GPU
**Use case:** Coherent English, word-level BPE, scaled diagnostics

---

## Predicted Configuration

Based on 168-dim findings and scaling rules:

```bash
wave-engine data/combined_10mb.txt \
  --layers 8 --n-bands 192 --n-head 8 \
  --out-proj-groups 6 \
  --iters 50000 --batch 4 --seq 128 --lr 1e-4 \
  --bpe --tokenizer data/tokenizer_8k.json \
  --checkpoint-name model_384_8L_8kbpe.bin
```

## What Carries Forward from 168-dim

- Multi-grid coprime embeddings (required — single-grid separation only 0.019 at 50K vocab)
- Per-band ODE magnitude clamp at 2.5
- Pre-flight diagnostics
- JSONL telemetry + NaN post-mortem
- Cosine LR scheduling
- AdamW weight decay

## What Needs Testing

- [ ] ODE coupling constant α at 192 bands (currently hardcoded 0.01 for ≤128, 0.1 for 384+)
- [ ] Optimal vocab size (8K predicted — gives ~44% model gradient share at 8L)
- [ ] Layer count for target gradient balance
- [ ] Training speed at this dimension
- [ ] Loss floor and convergence behaviour
- [ ] Wave structure diagnostics comparison with 168-dim

## Pending

Blocked by: completion of 168-dim validation runs and ODE coupling auto-scale issue.
