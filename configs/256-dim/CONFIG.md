# 256-dim Configuration — Mid-Range Tier

**Status:** VALIDATED — AGC regulation proven, stable training through 20K iters
**Hardware:** Any CPU (no GPU required)
**Use case:** Bridge between research (168-dim) and power user (384-dim). Best gradient balance tested with BPE (71%). Site of the ODE magnitude regulation discovery.

---

## Recommended Configuration

```bash
# 512 BPE at 256-dim — 71% model gradient share, AGC self-regulates
./target/release/wave-engine data/combined_10mb.txt \
  --layers 12 --n-bands 128 --n-head 8 \
  --out-proj-groups 8 \
  --iters 20000 --batch 4 --seq 128 --lr 1e-4 \
  --bpe --tokenizer data/tokenizer_512.json \
  --checkpoint-name model_256_12L_512bpe.bin
```

---

## Architecture

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dimension | 256 (128 bands × 2) | 52% more bands than 168-dim |
| Layers | 12 | 71% model gradient share at 512 BPE |
| Attention heads | 8 | Head dim = 32 |
| Maestro dim | 16 | Standard bottleneck |
| Out proj | Block-diagonal, 8 groups | Group size = 32 |
| ODE coupling | α=0.01, β=0.01 | ≤128 bands threshold |
| ODE regulation | AGC with knee compressor | Adaptive threshold, physics ceiling at 6.0 |
| Embeddings | Multi-grid coprime | Required for BPE at this dimension |
| Total params | 448K | 317K model + 131K lm_head |
| Speed | ~180ms/iter | CPU, batch=4, seq=128 |

---

## Tokenizer Comparison (500-iter sweeps)

| Metric | 168-dim 6L (512 BPE) | 256-dim 12L (1K BPE) | 256-dim 12L (512 BPE) |
|--------|---------------------|----------------------|----------------------|
| Params | 186K | 579K | 448K |
| Model % | 54% | 55% | **71%** |
| Speed | 65ms | 210ms | 163ms |
| Start loss | 6.41 | 7.09 | 6.42 |
| End loss (500 iters) | 5.95 | 6.97 | 6.16 |
| Descent | 0.46 | 0.12 | 0.26 |
| Zero NaN | Yes | Yes | Yes |

**Key finding:** 512 BPE at 256-dim gives **71% model gradient share** — the healthiest ratio tested with BPE at any dimension. 1K BPE at 55% is sluggish. 512 BPE is the sweet spot.

---

## ODE Magnitude Regulation (discovered at this dimension)

The 256-dim configuration was the testbed for solving the V-shaped divergence problem — where training loss descends, then rises, then partially recovers. Five controlled tests, each changing only the magnitude regulation, identified the cause and the fix.

### The five tests (all at 256-dim 12L 512 BPE, 20K iters, lr=1e-4)

| Test | Regulation | Best loss | Avg 14-16K | Avg 18-20K | V-shape? | NaN |
|------|-----------|-----------|------------|------------|----------|-----|
| A | Hard clamp 2.5 | 4.16 | 6.27 | 6.31 | YES | 0 |
| B | Hard clamp 2.5, lr=3e-5 | 4.65 | 6.12 | 6.09 | No (slow) | 0 |
| C | Hard clamp 5.0 | 3.75 | 6.28 | rising | Mild | 0 |
| D | Soft tanh 5.0 | 3.83 | 6.02 | 6.15 | Mild late | 0 |
| **E** | **AGC + ceiling 6.0** | **3.76** | **5.86** | **5.88** | **NO** | **0** |

### What caused the V-shape

The maestro pre-conditioner learns to push band magnitudes higher as training progresses — higher magnitudes carry more information through the ODE. A fixed clamp creates a ceiling the maestro fights against, causing gradient distortion and divergence. Gradient monitors proved this: clamp rate escalated from 1.3% to 5.9% (Test A) and 0% to 92% (Test C) over training.

### The fix: AGC with physics-based ceiling

Automatic Gain Control (AGC) tracks the maestro's actual operating range via EMA and sets the clamp threshold adaptively. A knee compressor passes normal magnitudes unchanged and only compresses outliers above the threshold. The physics ceiling (max_threshold=6.0) prevents the threshold from exceeding the ODE's stability limit (δφ < 90° at α=0.01 → mag < 5.6).

The AGC adapted from threshold 3.27 → 6.0 in 4000 iters, then held at the ceiling. The maestro operates freely at mag 5.1-5.6 with zero compression on normal signal.

**Full investigation:** See `investigations/ode-regulation/INVESTIGATION.md` in the research repo.

### The electronics progression (Marco's analogy)

| Stage | Electronics | Wave-engine | Result |
|-------|------------|-------------|--------|
| 1 | Fixed resistor | Hard clamp 2.5 | V-shape — clips signal |
| 2 | Larger resistor | Hard clamp 5.0 | Delayed — eventually 92% clamped |
| 3 | Zener diode | Tanh soft clamp | Stable but over-compresses by 17% |
| 4 | AGC + rail voltage | AGC + physics ceiling | **Model self-regulates within ODE limits** |

---

## Training Results (Test E — AGC, best run)

| Metric | Value |
|--------|-------|
| Best loss | 3.76 at iter 14848 |
| Rolling avg at 14-16K | 5.86 (descending) |
| Rolling avg at 18-20K | 5.88 (stable) |
| V-shape | None — monotonic descent through 20K |
| NaN skips | 0 |
| AGC threshold | Adapted 3.27 → 6.0, held at ceiling |
| Maestro operating range | mag 5.1-5.6 |
| Compression rate | 1.4% at iter 19K (outliers only) |
| Speed | ~180ms/iter |

### Rolling average progression (Test E)

```
iter  0-2K:   6.19  (learning)
iter  2-4K:   6.20  (slight bump)
iter  4-6K:   6.23  (peak — turbulence zone)
iter  6-8K:   6.09  (recovering)
iter  8-10K:  6.07  (descending)
iter 10-12K:  5.95  (descending)
iter 12-14K:  5.91  (descending)
iter 14-16K:  5.86  (new low)
iter 18-20K:  5.88  (stable)
```

Compare: Test A (hard clamp 2.5) was at 6.53 and diverging at iter 8-10K.

---

## Gradient Balance at 256-dim

| Vocab | Layers | Model % | Total params | Speed | Tested |
|-------|--------|---------|-------------|-------|--------|
| **512 BPE** | **12** | **71%** | **448K** | **180ms** | **Yes — best 3.76, stable 20K** |
| 512 BPE | 8 | 60% | 258K | ~140ms | No |
| 768 BPE | 10 | 55% | 395K | ~175ms | No |
| 1K BPE | 12 | 55% | 579K | 210ms | Yes — 0.12 descent (sluggish) |

---

## Ideal Use Cases for 256-dim

Bridges the gap between 168-dim specialist tasks and 384-dim general English:

| Use Case | Vocab | Why 256-dim fits |
|----------|-------|-----------------|
| Structured English (templates, forms) | 512-1K | 128 bands with 71% model share — enough for common word patterns |
| Code generation (Python/JS) | 512-1K | Keywords + common identifiers, deeper nesting than 168-dim |
| Domain-specific text (legal, medical) | 768-1K | Restricted vocabulary domains |
| Music with lyrics | 256-512 | MIDI notes + syllable tokens, harmonic structure + text |
| ODE regulation research | Any | The dimension where the AGC was developed and validated |

---

## Known Limitations

1. **Sub-word tokenization at 512 BPE** — common words are 2-3 sub-tokens, same as 168-dim. For single-token words, need ≥384-dim with 4K+ vocab.

2. **Rolling average plateaus at ~5.86** — the model is still learning at 20K iters but descent has slowed. Longer runs or cycle restarts may push lower.

3. **Serving test not yet done** — the best checkpoint (model_b_256dim_12L_512bpe_agc_best.bin) has not been tested through wave-server.

4. **AGC currently CPU tier only** — Candle and wgpu tiers still have the tanh soft clamp. AGC needs porting after validation.

---

## Findings Discovered at This Dimension

### 1. ODE Magnitude Regulation (new investigation)
The V-shaped divergence pattern is caused by the maestro fighting a fixed magnitude clamp. The fix is physics-bounded AGC — an adaptive threshold that tracks the maestro's operating range and is capped by the ODE stability limit. Five tests, each building on the previous failure, led to the solution. See `investigations/ode-regulation/INVESTIGATION.md`.

### 2. Gradient Balance Threshold (confirmed from 168-dim)
The ≥44% model gradient share threshold holds at 256-dim. At 71% (512 BPE, 12L), the model learns 2× faster than at 55% (1K BPE, 12L).

### 3. Training Window Extended by AGC
At 168-dim with hard clamp 2.5, the training window was ~10K iters. At 256-dim with AGC, the rolling average descends through 20K iters and is still stable. The "capacity ceiling" was partly a clamp artifact — the AGC extends the useful training window.

---

## Pre-flight Expected Output

```
[preflight] Embedding separation: 135.10 OK (4.0 tokens/band)
[preflight] Parameter balance: 70.7% model, 29.3% lm_head — OK
[preflight] ODE stability: 11° at M=2.0, alpha=0.0100 beta=0.0100 — OK
```

---

## Useful Commands

```bash
# Quick 500-iter test (verify configuration works)
./target/release/wave-engine data/combined_10mb.txt --layers 12 --n-bands 128 --n-head 8 \
  --out-proj-groups 8 --iters 500 --batch 4 --seq 128 --lr 1e-4 \
  --bpe --tokenizer data/tokenizer_512.json

# Full 20K training run with AGC
./target/release/wave-engine data/combined_10mb.txt --layers 12 --n-bands 128 --n-head 8 \
  --out-proj-groups 8 --iters 20000 --batch 4 --seq 128 --lr 1e-4 \
  --bpe --tokenizer data/tokenizer_512.json \
  --checkpoint-name model_256_12L_512bpe.bin

# Wave structure analysis
./target/release/wave-engine --analyze --resume model_256_12L_512bpe.bin \
  --layers 12 --n-bands 128 --n-head 8 --out-proj-groups 8 \
  --bpe --tokenizer data/tokenizer_512.json
```
