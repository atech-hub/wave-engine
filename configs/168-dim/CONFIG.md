# 168-dim Configuration — Research Tier

**Status:** VALIDATED — stable training, diagnostic pipeline complete
**Hardware:** Any CPU (no GPU required)
**Use case:** Fast research iteration, wave structure diagnostics, architecture experiments

---

## Recommended Configuration

```bash
# 512 BPE — sweet spot for 168-dim (sub-word tokens, fast, stable)
./target/release/wave-engine data/combined_10mb.txt \
  --layers 6 --n-bands 84 --n-head 4 \
  --out-proj-groups 6 \
  --iters 10000 --batch 4 --seq 128 --lr 1e-4 \
  --bpe --tokenizer data/tokenizer_512.json \
  --checkpoint-name model_168_6L_512bpe.bin

# Char-level — fastest convergence, best loss descent
./target/release/wave-engine data/combined_10mb.txt \
  --layers 4 --n-bands 84 --n-head 4 \
  --out-proj-groups 6 \
  --iters 10000 --batch 4 --seq 64 --lr 3e-4 \
  --checkpoint-name model_168_4L_char.bin
```

---

## Architecture

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dimension | 168 (84 bands × 2) | Smallest viable BPE dimension |
| Layers | 6 (BPE) / 4 (char) | 6L gives 54% model gradient share at 512 BPE |
| Attention heads | 4 | Head dim = 42 |
| Maestro dim | 16 | Standard bottleneck |
| Out proj | Block-diagonal, 6 groups | Group size = 28 |
| ODE solver | RK4-16 (CPU tier) | 16 integration steps |
| ODE coupling | α=0.01, β=0.01 | Scaled for ≤128 bands |
| Embeddings | Multi-grid coprime | Two incommensurate circles (Pattern 53) |
| Per-band clamp | 2.5 max magnitude | Prevents ODE phase wrapping |

---

## Tokenizer Comparison

Tested all tokenizer sizes at 168-dim. Results from 500-iteration sweeps:

| Tokenizer | Vocab | Total params | Speed | Start loss | Best loss | Gradient balance | Verdict |
|-----------|-------|-------------|-------|------------|-----------|-----------------|---------|
| Char-level | 186 | 98K | 65ms | 5.47 | 4.10 | 68% model | Best convergence |
| **512 BPE** | **512** | **153K (4L) / 186K (6L)** | **57-80ms** | **6.47** | **3.95** | **44% (4L) / 54% (6L)** | **Recommended for BPE** |
| 1K BPE | 1024 | 239K | 88ms | 7.07 | 6.66 | 28% model | Marginal — gradient starved |
| 2K BPE | 2048 | 411K | 150ms | 7.76 | 7.28 | 16% model | Too heavy — loss plateau |

**Key finding:** The gradient balance threshold for effective learning is ~44% model share. Below this, the lm_head dominates and the ODE/maestro can't learn effectively. At 168-dim, 512 BPE at 6 layers is the sweet spot.

### Word coverage at 512 BPE

512 BPE captures common function words as single tokens ("the", "and", "of", "to", "in", "is", "it") but splits most content words into 2-3 sub-tokens ("cat" → "c"+"at", "king" → "k"+"ing"). Compression ratio: ~2.3 chars/token. The model learns to compose meaning from sub-tokens.

For single-token whole words, you need ≥4K vocab which requires ≥384-dim. See the [384-dim configuration](../384-dim/CONFIG.md) (when available).

---

## Layer Scaling

| Layers | Model params | Model % (512 BPE) | Best loss | Late avg | Speed |
|--------|-------------|-------------------|-----------|---------|-------|
| 4 | 67K | 44% | 4.31 | 8.34 | 57ms |
| **6** | **100K** | **54%** | **3.95** | **5.94** | **80ms** |
| 8 | 134K | 61% | (untested) | — | ~114ms |

6 layers is the minimum for BPE at 168-dim. The extra capacity stabilises training — 4L bounces wildly while 6L maintains steady rolling average around 5.8-6.0.

---

## Training Results

### 6L, 512 BPE, 83K iterations (1.5 hour run)

| Metric | Value |
|--------|-------|
| Best loss | 3.95 (at iter 6641) |
| Best rolling avg | ~5.85 (iter 4K-7K window) |
| Divergence onset | iter ~25K (loss > 10) |
| NaN skips | 0 (stable throughout, even during divergence) |
| Total time | ~95 minutes |
| Speed | 75ms/iter average |

**Training window:** The model learns effectively for ~5K-10K iterations (0.5-1 corpus pass). After iter 10K, the rolling average drifts upward. By iter 25K it diverges to loss > 10. This is a capacity limitation — 100K model params memorise a subset of patterns in the first pass, then overfit and forget on subsequent passes.

**Recommended approach:** Use `--iters 10000` with cosine LR for 168-dim BPE. The LR decays fast enough to lock in gains from the first pass. Checkpoint at iter 7000 is typically the best model. Longer runs waste power — the model's useful training window is ~10K iters at this dimension.

### Wave Structure Diagnostics (at iter 3500)

| Diagnostic | Value | Interpretation |
|-----------|-------|----------------|
| Phase clustering | 0.988 | Near-perfect structured phase space |
| Band census | BIMODAL (69 universal / 15 word-specific) | Natural split matches theory |
| Semantic discrimination | 1.0x | Not measurable at 512 BPE (sub-tokens) |
| Depth peak | Layer 3 (of 6) | Halfway — consistent with theory |
| Dominant harmonic | n=1 | Adjacent token coherence near 1.0 |

---

## Known Limitations

1. **Training window limited to ~10K iterations** — the model learns effectively for one corpus pass (~5K-10K iters). Beyond that, the rolling average rises and the model diverges by iter 25K. This is not instability (zero NaN throughout) but a capacity limitation: 100K model params memorise patterns from the first pass and overfit on subsequent passes.

2. **High loss variance** — the model learns ~20% of BPE patterns well but guesses on the rest. Easy batches → loss 4.0. Hard batches → loss 7.0+. Best single-batch loss reaches 3.95 but rolling average stays around 5.8-6.0. This is a capacity limitation, not instability.

3. **Semantic discrimination untestable** — at 512 BPE, words like "cat" and "dog" are split into sub-tokens. The `--analyze` tool can't find semantic pairs. Use char-level for semantic pair analysis.

4. **Token cache doesn't key on tokenizer** — switching between tokenizers requires `rm -f data/*.bpe.tokens` to clear the cache. Otherwise the old tokenizer's cached tokens contaminate the new run.

---

## Findings Discovered at This Dimension

These findings were discovered during 168-dim development and apply to all dimensions:

### 1. Multi-Grid Harmonic Embeddings (Pattern 53)
Single-circle harmonic embeddings degenerate at high vocab/bands ratio. Two coprime modular circles provide 101x-11,800x improvement in token separation. Required for any BPE training below 768-dim.

### 2. Per-Band ODE Input Clamping
The maestro pre-conditioner can push individual bands past the ODE's stability threshold (δφ > 90° per step). Clamping per-band input magnitude to 2.5 before the ODE prevents phase wrapping without limiting the maestro's directional learning.

### 3. ODE Coupling Scales with Band Count
α=β=0.1 (calibrated for 64 bands at char-level) causes NaN at 84 bands with BPE. α=β=0.01 is stable. The coupling must be weaker at lower band counts to prevent chaotic phase accumulation.

Sweep data:
| Alpha | Good rate |
|-------|-----------|
| 0.100 | 16% |
| 0.047 | 76% |
| 0.022 | 84% |
| 0.010 | 93% |

### 4. Gradient Balance Threshold
Effective learning requires ≥44% model gradient share. Below this, the lm_head dominates and the ODE/maestro are gradient-starved. The threshold determines the maximum vocab size for each dimension:

| Dimension | Max vocab (4L, ≥44%) | Max vocab (6L, ≥44%) | Max vocab (8L, ≥44%) |
|-----------|---------------------|---------------------|---------------------|
| 168 | 512 | 768 | 1024 |
| 384 | 2048 | 4096 | 8000 |
| 768 | 8000 | 16000 | 32000 |

### 5. Maestro Dim is NOT the Variable
Tested maestro_dim at 4, 16, and 32 — all produced identical NaN rates. The maestro dimension does not affect ODE stability. The coupling constants (α, β) are the controlling variable.

---

## Pre-flight Expected Output

```
[preflight] Embedding separation: 91.79 OK (6.1 tokens/band)
[preflight] Parameter balance: 53.8% model, 46.2% lm_head — OK
[preflight] ODE stability: 11° at M=2.0, alpha=0.0100 beta=0.0100 — OK
```

If any pre-flight check warns or fails, the training configuration needs adjustment before proceeding.

---

## Serving with Wave-Server

```bash
./target/release/wave-server --model model_168_6L_512bpe.bin --bpe --tokenizer data/tokenizer_512.json --port 3000
```

**Note:** The wave-server must have multi-grid embeddings and per-band ODE clamp matching the engine's forward pass. Models trained with multi-grid will produce garbage if served with single-grid embeddings.

---

## Useful Commands

```bash
# Quick 500-iter test (verify configuration works)
./target/release/wave-engine data/combined_10mb.txt --layers 6 --n-bands 84 --n-head 4 \
  --out-proj-groups 6 --iters 500 --batch 4 --seq 128 --lr 1e-4 \
  --bpe --tokenizer data/tokenizer_512.json

# Wave structure analysis on trained checkpoint
./target/release/wave-engine --analyze --resume model_168_6L_512bpe.bin \
  --layers 6 --n-bands 84 --n-head 4 --out-proj-groups 6 \
  --bpe --tokenizer data/tokenizer_512.json

# Char-level baseline (fastest convergence, for diagnostics)
./target/release/wave-engine data/combined_10mb.txt --layers 4 --n-bands 84 --n-head 4 \
  --out-proj-groups 6 --iters 500 --batch 4 --seq 64 --lr 3e-4
```
