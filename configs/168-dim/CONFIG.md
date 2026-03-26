# 168-dim Configuration — Research Tier

**Status:** VALIDATED — produces English word fragments at char-level, stable training
**Hardware:** Any CPU (no GPU required)
**Use case:** Fast research iteration, char-level text generation, wave structure diagnostics, architecture experiments. Trains in minutes.

---

## Recommended Configuration

```bash
# Char-level — BEST RESULTS (loss 2.25, produces English words)
./target/release/wave-engine data/input.txt \
  --layers 4 --n-bands 84 --n-head 4 \
  --out-proj-groups 1 \
  --alpha 0.1 --agc-ceiling 1.0 \
  --iters 10000 --batch 4 --seq 64 --lr 3e-4 \
  --checkpoint-name model_168_4L_char.bin

# 512 BPE — stable training, sub-word tokens (needs testing with new settings)
./target/release/wave-engine data/input.txt \
  --layers 6 --n-bands 84 --n-head 4 \
  --out-proj-groups 1 \
  --alpha 0.1 --agc-ceiling 1.0 \
  --iters 20000 --batch 4 --seq 128 --lr 1e-4 \
  --bpe --tokenizer data/tokenizer_512.json \
  --checkpoint-name model_168_6L_512bpe.bin
```

---

## Architecture

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dimension | 168 (84 bands × 2) | Smallest viable dimension |
| Layers | 4 (char) / 6 (BPE) | 4L sufficient for char-level |
| Attention heads | 4 | Head dim = 42, frozen harmonic coherence |
| Maestro dim | 16 | Standard bottleneck |
| **Out proj** | **Dense (1 group)** | **Block-diagonal starves model at this scale** |
| ODE solver | RK4-16 (CPU tier) | 16 integration steps |
| **ODE coupling** | **α=0.1, β=0.1** | **Strong coupling — the Kerr effect must be real, not token** |
| Embeddings | Multi-grid coprime | Two incommensurate circles (Pattern 53) |
| **ODE regulation** | **AGC, ceiling=1.0** | **Sphere boundary — forces information into phase** |

### Critical: Linked Parameters

These three settings are NOT independent — they define the operating regime together:

| Setting | Why this value |
|---------|---------------|
| α=0.1 (strong coupling) | The Kerr nonlinearity needs to do real computation. At α=0.01 the ODE is essentially a damped rotation — no meaningful cross-band interaction. |
| AGC ceiling=1.0 (tight) | Physics: safe mag at α=0.1 is 1.77. But ceiling=1.0 beats 2.0 — the model learns better when constrained to the unit circle. Phase carries semantics, magnitude is just an amplifier. |
| Dense out_proj (1 group) | At 168-dim, block-diagonal (6 groups) drops params from 171K to 77K. That starves the model — the out_proj is where bands mix into predictions. Dense is required at this scale. |

---

## Char-Level Results (CURRENT BEST)

### Ceiling sweep (4L, char-level, dense out_proj, α=0.1, 3K iters)

| AGC Ceiling | Best loss | NaN | Notes |
|-------------|-----------|-----|-------|
| **1.0** | **2.25** | **0** | **Best — sphere boundary** |
| 1.5 | 2.36 | 0 | Slightly worse |
| 2.0 | 2.35 | 0 | Similar to 1.5 |
| 2.5 | 3.10 | 0 | Block-diagonal (77K) — starved |
| 6.0 | 4.22 | 779 | ODE blows up at α=0.1 |

**Key finding:** Tighter ceiling = better loss. The model learns best when magnitudes are constrained near the unit circle. This matches the spherical investigation: phase carries semantics (20x clustering), magnitude just amplifies (383x).

### Serving test (4L, char-level, loss 2.25)

**Prompt:** "hello"

**Output:** Word fragments — "the", "to", "you", "she", "our", "is", "and". Shakespeare character name patterns ("ANLIORD:", "DUKE"). Dialogue format with colons. Stage direction structures. Punctuation placement learned.

At 171K params and loss 2.25, the model is 80% of the way to English. Character names introduce dialogue. Articles precede nouns. Punctuation marks boundaries. Not fluent, but structured. Longer training (10K-20K iters) should push loss below 2.0 and produce recognisable words.

### Comparison with kerr-engine (predecessor)

| | Kerr-engine | Wave-engine (old settings) | Wave-engine (new settings) |
|---|---|---|---|
| Coupling α | 0.1 | 0.01 (too weak) | **0.1** |
| Out proj | Dense | Block-diagonal (6) | **Dense (1)** |
| AGC ceiling | None | 6.0 (too wide) | **1.0** |
| Params | 354K | 77K (starved) | **171K** |
| Best loss | 2.05 | 3.76 | **2.25** |
| Output | Word fragments | Garbage | **Word fragments** |

The wave-engine now matches the kerr-engine's territory with improved stability (AGC prevents NaN that kerr-engine was vulnerable to at scale).

---

## Out_proj: Dense vs Block-Diagonal at 168-dim

This was a critical discovery. Block-diagonal out_proj saves 96% of parameters at 768-dim (where out_proj dominates). At 168-dim it saves 70% — but 70% of a tiny model starves it:

| Out proj | Groups | Group size | Params per layer | Total model params | Best loss |
|----------|--------|-----------|-----------------|-------------------|-----------|
| **Dense** | **1** | **168** | **28,392** | **171K** | **2.25** |
| Block-diag | 6 | 28 | 4,872 | 77K | 3.02 |

The out_proj is where the ODE's per-band representations get mixed into token-level predictions. With block-diagonal, bands can only mix within groups of 28. With dense, all 168 dimensions interact. At small scale, the mixing capacity matters more than the parameter savings.

**Rule:** Dense out_proj at ≤256-dim. Block-diagonal at ≥384-dim where out_proj dominates parameter count.

---

## BPE Results (needs retesting with new settings)

Previous BPE results used α=0.01 and block-diagonal out_proj. These are SUPERSEDED — the model was running with 10% coupling strength and 45% fewer params than needed. All BPE results below should be retested with α=0.1, ceiling=1.0, dense out_proj.

### Previous BPE results (α=0.01, block-diagonal — superseded)

| Config | Regulation | Best loss | Avg 14-16K | V-shape? |
|--------|-----------|-----------|------------|----------|
| Hard clamp 2.5 | Fixed | 4.16 | 6.27 | YES |
| Hard clamp 5.0 | Fixed | 3.75 | 6.28 | Mild |
| Soft tanh 5.0 | Fixed | 3.83 | 6.02 | Mild late |
| AGC ceiling 6.0 | Adaptive | 3.76 | 5.86 | NO |

**Note:** These results proved the AGC concept (V-shape elimination) but used the wrong coupling/ceiling/out_proj settings. The actual learning capacity was throttled. Retesting with corrected settings is pending.

---

## Tokenizer Comparison (needs retesting)

Previous sweep used block-diagonal out_proj (77K-186K params). With dense out_proj, all param counts increase significantly, changing the gradient balance:

| Tokenizer | Vocab | Params (old block-diag) | Params (new dense) | Gradient balance (est.) |
|-----------|-------|------------------------|-------------------|------------------------|
| Char-level | 65-186 | 77-98K | **171K** | **~85%** (proven) |
| 512 BPE | 512 | 153-186K | ~270K | ~70% (estimated) |
| 1K BPE | 1024 | 239K | ~350K | ~55% (estimated) |

---

## Training Results: AGC Regulation Discovery

The ODE magnitude regulation investigation was conducted at 256-dim but applies to all tiers. See `investigations/ode-regulation/INVESTIGATION.md` in the research repo and [256-dim CONFIG.md](../256-dim/CONFIG.md) for the full five-test progression.

The key progression:
1. Hard clamp (fixed resistor) → throttled, V-shape
2. Hard clamp 5.0 (bigger resistor) → delayed throttle
3. Soft tanh (zener diode) → over-compressed normal signal
4. AGC no ceiling (voltage regulator) → ODE blew up
5. **AGC + physics ceiling (AGC + rail voltage) → model self-regulates**

---

## Wave Structure Diagnostics (at iter 3500, BPE)

| Diagnostic | Value | Interpretation |
|-----------|-------|----------------|
| Phase clustering | 0.988 | Near-perfect structured phase space |
| Band census | BIMODAL (69 universal / 15 word-specific) | Natural split matches theory |
| Semantic discrimination | 1.0x | Not measurable at 512 BPE (sub-tokens) |
| Depth peak | Layer 3 (of 6) | Halfway — consistent with theory |
| Dominant harmonic | n=1 | Adjacent token coherence near 1.0 |

---

## Ideal Use Cases for 168-dim

The 168-dim model with dense out_proj has 171K model params and trains at 40-80ms/iter. With strong coupling (α=0.1) it builds real harmonic structure AND produces text at char-level. Ideal for tasks with small vocabularies and structural patterns.

| Use Case | Vocab | Why 168-dim fits |
|----------|-------|-----------------|
| **Char-level English** | 65 | 85% gradient share, loss 2.25, produces word fragments at 171K params |
| **MIDI music generation** | ~88 | Musical harmony IS harmonic coherence — octaves (n=2), fifths (n=3), fourths (n=4) map directly to ODE frequency bands |
| **Simple arithmetic** | ~15 | Tiny vocab, learnable rules, verifiable output |
| **DNA sequences** | 4 | 4 tokens on 84 bands = maximally separated. Codons, promoters, repeats |
| **Chemical formulas** | ~120 | Small vocab, strict grammar, harmonic coherence maps to bonding |
| **Code keywords** | ~50-200 | Python ~35 keywords + operators, deep structure |
| **Wave structure research** | Any small vocab | Fast iteration, full diagnostic pipeline |

**For the cloners:** Use `--out-proj-groups 1 --alpha 0.1 --agc-ceiling 1.0` at 168-dim. Block-diagonal out_proj and weak coupling were the wrong defaults for this scale.

---

## Known Limitations

1. **Char-level only (for text output)** — BPE at 168-dim needs retesting with corrected settings. Previous BPE results used wrong coupling/out_proj.

2. **Loss 2.25 produces fragments, not sentences** — the model generates word fragments and Shakespeare formatting but not fluent English. Longer training or more layers may push loss below 2.0 where coherent words emerge.

3. **Capacity-limited multi-epoch training** — at 171K params, the model diverges after ~1 corpus pass on 12.4MB data. This is genuine capacity (confirmed by 256-dim comparison), not a regulation artifact. Use smaller corpora (1.1MB Shakespeare) or larger dimensions for multi-epoch.

4. **Token cache doesn't key on tokenizer** — switching tokenizers requires `rm -f data/*.bpe.tokens`.

---

## Findings Discovered at This Dimension

### 1. Sphere Boundary (AGC ceiling=1.0)
Tighter AGC ceiling produces better loss. Ceiling=1.0 (unit circle) beats 1.5, 2.0, and all higher values. The ODE should transform PHASE, not amplify MAGNITUDE. Forces information into the dimension where the architecture reads it. Connects to spherical investigation: "the circle was always a sphere."

### 2. Strong Coupling Required (α=0.1)
At α=0.01, the ODE phase shift at mag=2.0 is only 11° — barely nonlinear. The Kerr self-phase and cross-phase modulation are effectively turned off. At α=0.1, phase shift is 115° — real nonlinear band interaction. The ODE does meaningful computation.

### 3. Dense Out_proj at Small Scale
Block-diagonal out_proj saves 96% at 768-dim but starves the model at 168-dim (77K vs 171K params). The out_proj mixes band-level representations into token predictions — reducing mixing capacity at small scale prevents the model from composing predictions. Dense out_proj is required below ~256-dim.

### 4. Coupling and Ceiling are Linked
Higher coupling requires lower ceiling: ceiling = √(π/2 / (α + 4β)). α=0.1 → safe mag 1.77. α=0.01 → safe mag 5.6. Setting one without the other produces either NaN (ceiling too high) or throttling (ceiling too low relative to coupling).

### 5. Multi-Grid Harmonic Embeddings (Pattern 53)
Two coprime modular circles provide 101x-11,800x token separation improvement. Required for BPE. At char-level (65 vocab), single-grid separation is already 74.66 — multi-grid still used for consistency.

### 6. ODE Magnitude Regulation (AGC)
Automatic Gain Control with knee compressor replaces all fixed clamps. Adaptive threshold tracks the maestro's operating range. Physics ceiling prevents ODE chaos. See `investigations/ode-regulation/INVESTIGATION.md`.

### 7. Gradient Balance Threshold (≥44%)
Effective learning requires ≥44% model gradient share. Below this, the lm_head dominates. Dense out_proj at 168-dim char-level gives 85% — far above threshold.

---

## Pre-flight Expected Output (char-level, α=0.1)

```
[preflight] Embedding separation: 91.79 OK (6.1 tokens/band)
[preflight] Parameter balance: ~85% model, ~15% lm_head — OK
[preflight] ODE stability: 115° at M=2.0, alpha=0.1000 beta=0.1000 — WARNING (AGC ceiling protects)
```

The ODE stability check will warn at α=0.1 because the phase shift exceeds 90° at mag=2.0. This is expected and safe — the AGC ceiling at 1.0 keeps magnitudes well below 2.0.

---

## Useful Commands

```bash
# Char-level (best results — recommended starting point)
./target/release/wave-engine data/input.txt --layers 4 --n-bands 84 --n-head 4 \
  --out-proj-groups 1 --alpha 0.1 --agc-ceiling 1.0 \
  --iters 3000 --batch 4 --seq 64 --lr 3e-4

# Longer char-level run (push below loss 2.0)
./target/release/wave-engine data/input.txt --layers 4 --n-bands 84 --n-head 4 \
  --out-proj-groups 1 --alpha 0.1 --agc-ceiling 1.0 \
  --iters 20000 --batch 4 --seq 64 --lr 3e-4 \
  --checkpoint-name model_168_char_best.bin

# BPE (needs testing with new settings)
./target/release/wave-engine data/input.txt --layers 6 --n-bands 84 --n-head 4 \
  --out-proj-groups 1 --alpha 0.1 --agc-ceiling 1.0 \
  --iters 20000 --batch 4 --seq 128 --lr 1e-4 \
  --bpe --tokenizer data/tokenizer_512.json

# Wave structure analysis
./target/release/wave-engine --analyze --resume model_168_char_best.bin \
  --layers 4 --n-bands 84 --n-head 4 --out-proj-groups 1

# Serve through wave-server
./target/release/wave-server model_168_char_best.bin data/input.txt --port 8090
```
