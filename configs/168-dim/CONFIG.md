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
| ODE regulation | AGC with knee compressor | Adaptive threshold, physics ceiling at 6.0 |

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

### 6L, 512 BPE, 20K iterations — AGC (current best)

| Metric | Value |
|--------|-------|
| Best loss | **3.76** (at iter 10646) |
| Rolling avg at 8-10K | 5.87 (descending) |
| Rolling avg at 14-16K | 5.91 (stable) |
| Rolling avg at 18-20K | 5.90 (stable) |
| V-shape | **None** |
| Divergence | **None through 20K** |
| NaN skips | 0 |
| AGC threshold | Adapted 3.25 → 6.0, held at ceiling |
| Maestro max magnitude | 7.95 (knee compressor manages outliers) |
| Speed | 83ms/iter average |

### Previous results (hard clamp 2.5 — superseded)

| Metric | Value |
|--------|-------|
| Best loss | 3.95 (at iter 6641) |
| Best rolling avg | ~5.85 (iter 4K-7K window) |
| Divergence onset | iter ~10K (rolling avg rising) |
| NaN skips | 0 |

**Note:** The previous 3.95 best loss and 10K training window were **clamp artifacts**. The hard clamp at 2.5 throttled the maestro, causing V-shape divergence. With AGC, best loss improved to 3.76 and training is stable through 20K+.

**Training window (with AGC):** The model trains stably through 20K+ iterations with no divergence. Rolling averages descend monotonically from 6.10 to 5.87 and hold at 5.90 through 20K. Previous results showing a "10K training window" and divergence at 25K were **clamp artifacts** — the hard clamp at 2.5 caused the maestro to fight the ceiling, producing gradient distortion and eventual V-shape divergence. With AGC, the maestro self-regulates within the ODE's physics limit and no divergence occurs.

**Recommended approach:** Use `--iters 20000` with cosine LR for 168-dim BPE. Checkpoint around iter 10K-15K is typically the best model.

### Wave Structure Diagnostics (at iter 3500)

| Diagnostic | Value | Interpretation |
|-----------|-------|----------------|
| Phase clustering | 0.988 | Near-perfect structured phase space |
| Band census | BIMODAL (69 universal / 15 word-specific) | Natural split matches theory |
| Semantic discrimination | 1.0x | Not measurable at 512 BPE (sub-tokens) |
| Depth peak | Layer 3 (of 6) | Halfway — consistent with theory |
| Dominant harmonic | n=1 | Adjacent token coherence near 1.0 |

---

## Ideal Use Cases for 168-dim

The 168-dim model has 84 harmonic bands, ~100K model params, and trains at 57-80ms/iter. It builds near-perfect wave structure (0.988 phase clustering) but can't fit enough patterns for general English. This makes it ideal for tasks with **small vocabularies and deep structural patterns** — exactly where harmonic coherence shines.

| Use Case | Vocab | Why 168-dim fits | What it demonstrates |
|----------|-------|-----------------|---------------------|
| **MIDI music generation** | ~88 (notes) | 88 notes = 87% model gradient share. Musical harmony IS harmonic coherence — octaves (n=2), fifths (n=3), fourths (n=4) map directly to the ODE's frequency bands | Wave architecture generating music from its native mathematical substrate |
| **Simple arithmetic** | ~15 (0-9 + ops) | Tiny vocab, learnable rules, verifiable output. "2+3=5" is a pattern a 100K model can memorise completely | Proof that wave-engine learns logical rules, not just statistics |
| **DNA sequences** | 4 (A,T,G,C) | 4 tokens on 84 bands = maximally separated. Codon patterns, promoter regions, repeats — deep structure in tiny vocab | Bioinformatics on a laptop with no GPU |
| **Chemical formulas** | ~120 (elements) | Small vocab, strict grammar. Harmonic coherence maps to chemical bonding — n=2 for double bonds, n=3 for resonance structures | Domain-specific language model for chemistry |
| **Code keywords** | ~50-200 | Python has ~35 keywords + operators. Pattern space is small but structure is deep (nesting, scope, indentation) | Structured reasoning at tiny scale |
| **Chess/game moves** | ~200 | Small move vocabulary, deep positional patterns. Phase-based attention could encode board state as angles | Game AI from harmonic dynamics |
| **Morse/signal patterns** | 3-10 | Trivial vocab, rhythmic temporal structure — natural fit for coupled oscillators | Signal processing proof of concept |
| **Wave structure research** | Any small vocab | Fast iteration (57ms/iter), full diagnostic pipeline, bimodal band census confirmed | Architecture experiments before scaling up |

**The principle:** 168-dim excels when vocab is small enough that the model has >50% gradient share AND the task has structural relationships that map to harmonic coherence. The smaller the vocab relative to the dimension, the more of the model's capacity goes to learning structure rather than distinguishing tokens.

**For the 148 cloners:** If your task has <200 tokens and structural patterns, 168-dim trains in minutes on any CPU and might outperform much larger models that waste capacity on vocabulary overhead. The wave architecture's efficiency advantage is strongest at small scale with structured data.

---

## Known Limitations

1. ~~**Training window limited to ~10K iterations**~~ — **CORRECTED:** With AGC, training is stable through 20K+ iterations. The previous "10K limit" was a clamp artifact. The model still has capacity limitations (100K params) but no longer diverges.

2. **High loss variance** — the model learns ~20% of BPE patterns well but guesses on the rest. Easy batches → loss 4.0. Hard batches → loss 7.0+. Best single-batch loss reaches 3.76 but rolling average stays around 5.9. This is a capacity limitation, not instability.

3. **Semantic discrimination untestable** — at 512 BPE, words like "cat" and "dog" are split into sub-tokens. The `--analyze` tool can't find semantic pairs. Use char-level for semantic pair analysis.

4. **Token cache doesn't key on tokenizer** — switching between tokenizers requires `rm -f data/*.bpe.tokens` to clear the cache. Otherwise the old tokenizer's cached tokens contaminate the new run.

---

## Findings Discovered at This Dimension

These findings were discovered during 168-dim development and apply to all dimensions:

### 1. Multi-Grid Harmonic Embeddings (Pattern 53)
Single-circle harmonic embeddings degenerate at high vocab/bands ratio. Two coprime modular circles provide 101x-11,800x improvement in token separation. Required for any BPE training below 768-dim.

### 2. ODE Magnitude Regulation (AGC)
The maestro pre-conditioner pushes band magnitudes higher as training progresses. A fixed clamp (2.5 or 5.0) creates a ceiling the maestro fights, causing V-shape divergence. The fix is Automatic Gain Control (AGC) with a knee compressor: adaptive threshold tracks the maestro's operating range via EMA, physics ceiling at 6.0 prevents exceeding ODE stability. See `investigations/ode-regulation/INVESTIGATION.md` in the research repo.

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

### Serving Test Result (6L, 512 BPE, iter 7000, loss 3.95)

Prompt: "The cat sat on the" → Output: English fragments ("may", "will", "the") but not coherent sentences.

The model recognises common sub-word patterns but cannot compose them into language at this scale. 100K model params is below the threshold for sentence-level generation. This is expected — the model's wave structure diagnostics show it IS building harmonic organisation (0.94 clustering, bimodal bands), it just lacks the capacity to translate that structure into coherent output.

**For coherent English output, use 384-dim (Model B) or larger.**

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
