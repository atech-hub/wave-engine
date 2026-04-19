# Wave-Engine Lab Manual

**Version:** 1.0 — April 19, 2026
**Engine:** wave-engine (Rust, Apache 2.0, [github.com/atech-hub/wave-engine](https://github.com/atech-hub/wave-engine))
**Hardware requirement:** Any machine with a Rust toolchain. GPU optional (wgpu for AMD/Intel/NVIDIA; Candle for NVIDIA CUDA).

---

## What This Is

The wave-engine is a research platform for studying coupled-oscillator dynamics in neural networks. It replaces standard MLP layers with Kerr-nonlinear harmonic oscillator ODEs and provides instruments to measure the internal structure the model builds during training.

This manual describes what instruments exist, what each one measures, how to invoke it, and what baselines to compare against. It is a reference card, not a tutorial. For the mathematical framework, see the [Wave Coherence](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) repository.

**The lab has two sides:**

| Side | Question it answers | Instruments |
|------|-------------------|-------------|
| **Engine Lab** (Part 1) | Is the engine running correctly? | 17 component monitors + 10 junction monitors + training controls |
| **Analysis Lab** (Part 2) | What did the model build? | Galaxy scan, phase encoder, relate-vocab, four-axis probes |

A **trained checkpoint** is the interface between the two sides. Part 1 tells you the checkpoint was produced correctly. Part 2 tells you what's inside it.

---

# Part 1: The Engine Lab — Correctness & Controls

## 1.1 Architecture in One Paragraph

Each transformer block contains: layer norm → parallel attention and FFN branches → residual add. The FFN replaces the standard MLP with: input projection (mae_in) → pre-ODE maestro → Kerr-ODE (RK4 integration of coupled harmonic oscillators) → post-ODE maestro → output projection (out_proj). The ODE evolves N bands as coupled oscillators with self-phase modulation (α), cross-phase modulation (β), damping (γ), natural frequency (ω), and optional four-wave mixing (χ). Training learns the standard transformer parameters (embeddings, attention, layer norms, projections) plus the ODE coupling constants.

## 1.2 Subcommands

```
wave-engine train           Standard token-prediction training
wave-engine train-waves     Wave-space training (L2 loss on ODE states)
wave-engine generate        Token-based text generation
wave-engine wave-generate   Generation from wave-trained checkpoints
wave-engine encode          Phase encoding, relate pairs, vocabulary scan
wave-engine galaxy-scan     Geometric inventory of a checkpoint
wave-engine verify grad     Junction monitor: gradient correctness (J1)
wave-engine analyze         Checkpoint analysis and diagnostics
wave-engine ode-monitor     Per-band magnitude/phase inspection
wave-engine phase-decode    Compare phase-native vs lm_head decoding
wave-engine convert-dataset Create KWDS/KWMF format datasets
wave-engine recommend       Architecture sizing recommendations
wave-engine scale-checkpoint Scale checkpoint to different dimensions
wave-engine serve           Inference server (OpenAI-compatible API)
```

## 1.3 Training Controls

These are the experimental variables. Changing one while holding others constant produces a controlled A/B test.

### Architecture parameters

| Flag | Default | What it controls | Typical range |
|------|---------|-----------------|---------------|
| `--n-bands` | 84 | Frequency bands (embedding dim = 2 × n_bands) | 42–192 |
| `--n-head` | 4 | Attention heads | 2–8 |
| `--layers` | 4 | Transformer blocks | 2–8 |
| `--alpha` | 0.1 | Self-phase modulation (SPM) | 0.01–0.2 |
| `--beta` | 0.2 | Cross-phase modulation (XPM, neighbours ±1/±2) | 0.1–0.3 |
| `--chi` | 0.0 | Four-wave mixing strength | 0.0 or 0.03 |
| `--maestro-dim` | 16 | Bottleneck dimension for pre/post-ODE conditioning | 8–32 |
| `--out-proj-groups` | 1 | Output projection groups (1 = dense) | 1 or 6 |

**Known operating constraints:**
- α=0.01 is 10× too weak — Kerr nonlinearity effectively disabled
- β=0.2 is the validated sweet spot (7.82× coupling ratio). β=0.1 under-couples (3.94×). β=0.3 over-couples.
- AGC ceiling is physics-linked: ceiling = √(π/2 / (α + 4β)). At α=0.1, β=0.2 → ceiling ≈ 1.0
- Dense out_proj required at ≤256-dim. Block-diagonal starves the model at small dimensions.

### Training parameters

| Flag | Default | What it controls |
|------|---------|-----------------|
| `--lr` | 3e-4 | Learning rate |
| `--iters` | 10000 | Training iterations |
| `--seq` | 128 | Sequence length |
| `--phase-native` | off | Use phase-native loss instead of cross-entropy through lm_head |
| `--bpe` | off | Use BPE tokenizer instead of character-level |
| `--curriculum` | off | Enable curriculum training schedule |
| `--split-band` | off | Freeze-and-decouple ODE integration (cleaner gradients) |
| `--chi 0.03` | — | Enable four-wave mixing (requires chi=0 for split-band Phase A) |
| `--candle` | off | Use Candle/CUDA backend |
| `--cuda-kernel` | off | Use fused CUDA kernel (implies --candle) |

### Pathway flags

| Flag | Default | What it controls |
|------|---------|-----------------|
| `--ode-pathway` | on | Real ODE Jacobian flows gradients to upstream parameters |
| `--attention-pathway` | on | Attention backward contributes to d_normed |
| `--no-ode-pathway` | — | Disable ODE pathway (identity shortcut, for comparison only) |
| `--no-attention-pathway` | — | Disable attention pathway (for comparison only) |

**These should almost always be left at defaults.** The `--no-*` variants exist only for controlled A/B experiments documenting what the pathways contribute.

### Example: standard arithmetic training

```bash
wave-engine train data/arithmetic.txt \
    --n-bands 84 --n-head 4 --layers 4 \
    --alpha 0.1 --beta 0.2 \
    --lr 0.001 --iters 40000 --seq 16 \
    --phase-native --split-band
```

### Example: grammar training with FWM

```bash
wave-engine train data/grammar_lesson_1.txt \
    --n-bands 84 --n-head 4 --layers 4 \
    --alpha 0.1 --beta 0.2 --chi 0.03 \
    --lr 0.001 --iters 80000 --seq 128 \
    --phase-native --vocab 77
```

## 1.4 Component Monitors (17)

These fire automatically during training and report to the console. They track quantities but do not check correctness — that is the junction monitors' job.

| Monitor | File | What it tracks |
|---------|------|---------------|
| Gradient | gradient_monitor.rs | Per-parameter gradient norms, NaN detection, gnorm |
| Attention | attn_monitor.rs | Attention entropy, max weight, head utilisation |
| ODE dynamics | ode_dynamics_monitor.rs | Per-layer α, β, γ values, coupling ratios |
| ODE detail | ode_monitor.rs | Per-band magnitudes, phase shifts, AGC compression |
| Layer flow | layer_flow_monitor.rs | cos(input, output) per layer, channel balance (θ/Δθ ratio) |
| Embedding | embedding_monitor.rs | Embedding separation, dead token detection |
| Output | output_monitor.rs | Logit distribution, entropy, top-k statistics |
| Throughput | throughput_monitor.rs | ms/iter, tokens/sec |
| FWM | fwm_monitor.rs | Four-wave mixing contribution as % of derivative |
| Framework | framework_monitor.rs | Live harmonic coherence during training |
| Curriculum | curriculum_monitor.rs | Current curriculum stage, stage transitions |
| Dynamic params | dyn_param_monitor.rs | Spring-regulated hyperparameter values |
| Checkpoint | checkpoint_monitor.rs | Save/load integrity, NaN guard |
| Encoding health | encoding_health.rs | Phase separation, grid integrity |
| ODE backward | ode_backward_monitor.rs | Backward gradient decomposition per physics term |
| I/Q | iq_monitor.rs | In-phase/quadrature channel statistics |
| Monitor base | monitor.rs | Shared monitor infrastructure |

**Reading the monitors:** the most important single number during training is `cos(in,out)` per layer from the layer flow monitor. Healthy training shows L0-L2 with moderate preservation (0.3-0.7) and L3 decreasing during grammar (the regime shift). If any layer shows cos > 0.95, the ODE is doing nothing at that layer.

## 1.5 Junction Monitors (10)

These verify correctness at component boundaries. They do not run during normal training — invoke them via `wave-engine verify` or as self-tests.

| ID | Monitor | What it verifies |
|----|---------|-----------------|
| J1 | grad_check | Analytical gradient matches finite-difference (section-aware: output-adjacent must pass; ODE-adjacent documented) |
| J2 | param_completeness | No orphaned weights — every trainable parameter is in flatten/unflatten |
| J3 | pathway_completeness | d_normed receives contributions from both attention and FFN branches |
| J4 | roundtrip_integrity | Checkpoint save → load produces identical parameters |
| J5 | vector_length | params.len() == count_trainable_ex == grads.len() across every module |
| J6 | live_gradient | Gradient norms are finite and non-zero during actual training |
| J7 | value_range | Activations and gradients stay within expected bounds |
| J8 | train_infer_alignment | Training forward and inference forward produce identical output |
| J9 | tensor_shape | Every tensor operation produces correctly shaped output |
| J10 | tier_parity | CPU and GPU tiers produce matching results (within tolerance) |

**Running J1 (gradient check):**

```bash
# Standard check (sampled parameters, section-aware)
wave-engine verify grad --scope sampled --split-band

# Exhaustive check at tiny scale
wave-engine verify grad --scope tiny --eps 1e-3 --verbose

# Check with specific pathways disabled
wave-engine verify grad --no-ode-pathway
```

**J1 section-aware interpretation:** output-adjacent sections (output_corrector, ln_f, ln_w) must pass at the specified tolerance. ODE-adjacent sections (mae_in, mae_out weights) will show higher max_err due to the stiff-ODE Jacobian product — this is mathematically expected (see Pattern 149) and documented, not gated.

## 1.6 Baselines

These checkpoints are the reference points for comparison:

| Checkpoint | Dataset | Config | Result | Use as |
|-----------|---------|--------|--------|--------|
| `model_a_gs_best.bin` | arithmetic (512 BPE) | 84 bands, 4H, 4L | Best BPE arithmetic | BPE baseline |
| `checkpoint.bin` | char-level best | 84 bands, 4H, 4L | Best char-level | Char baseline |
| Post-fix monolithic | arithmetic.txt 86KB | 40K iters, PN, split-band off | 52/991 correct | Monolithic gradient baseline |
| Post-fix split-band | arithmetic.txt 86KB | 40K iters, PN, split-band on | 76/991 correct (+46%) | Split-band gradient baseline |

## 1.7 Stall Detector

Built into training. Fires warnings at iteration 2000 if loss has not decreased meaningfully (loss_2000/loss_500 > 0.97). Aborts at iteration 5000 if still stalled (loss_5000/loss_2000 > 0.98). Pre-flight abort if initial loss exceeds 5× ln(vocab). These thresholds catch misconfigured runs early without wasting compute.

## 1.8 Architecture Calculator

```bash
wave-engine recommend data/your_dataset.txt
```

Reads the dataset, computes vocabulary size, checks two independent bottlenecks, and prints a recommended configuration:

1. **Band bottleneck:** tokens_per_effective_dim < 0.50, with dead band accounting via coprime moduli
2. **Attention bottleneck:** positions_per_head < 40, head_dim ≥ 16

Both must be satisfied simultaneously. Fixing one without the other gives zero accuracy gain.

---

# Part 2: The Analysis Lab — Exploration & Measurement

These instruments examine trained checkpoints. They do not modify the model.

## 2.1 Galaxy Scan — Geometric Inventory

```bash
wave-engine galaxy-scan --resume checkpoint.bin \
    --n-bands 84 --layers 4 --scan-corpus data/input.txt
```

Produces a five-layer structural map of the model's learned geometry:

| Layer | What it maps |
|-------|-------------|
| Per-band profiles | Phase, magnitude, circular variance, boundary distance, grid assignment per band |
| Pairwise geometry | Mean angular distance between every band pair, catalog matching (11 relationship types) |
| Harmonic coherence | cos(n·Δθ) at 12 harmonics for every pair — the full spectral signature of each relationship |
| Constellations | Triads (120° triangles), FWM quartets (phase-matched a+b=c+d), locked/oscillating/random classification |
| Summary statistics | Catalog distribution, sphere fill fraction, grid nativity, locked quartet count |

**Output files:** `galaxy_map.json` (~21MB, human-readable), `galaxy_matrix.bin` (full pair matrix), `phases.bin` (raw per-position per-band phases), `scan_metadata.json`.

**Reading the output:** use `python scripts/summarize_galaxy.py galaxy_map.json` for a compact ~25KB summary. Pairwise diffs between two scans include automatic confound warnings (dataset mismatch, training length mismatch, architecture mismatch).

**Key metrics to watch:**
- **Locked FWM quartets:** how many band-quartets are permanently phase-coherent. Higher = richer relational structure. Grammar produces thousands; arithmetic at small scale produces few or none.
- **Catalog distribution:** which of the 11 relationship types appear. Grammar uses all 11; arithmetic uses 2.
- **Conjunction %:** what fraction of band pairs cluster at 0°. Lower = more geometric diversity.

## 2.2 Phase Encoder — Direct Geometric Injection

```bash
# Encode a token through the trained ODE
wave-engine encode --resume checkpoint.bin --encode "s" --data data/input.txt

# Encode through an untrained (blank) model for comparison
wave-engine encode --blank --encode "s" --data data/input.txt

# Inject at a specific layer
wave-engine encode --resume checkpoint.bin --encode "s" --inject-layer 2
```

Bypasses the token→embedding pipeline and injects phase configurations directly into ODE layers. Five encoding modes: text (`--encode`), number (`--encode-number`), catalog relationship (`--encode-catalog`), raw phases (`--encode-phases`), compound (multi-token sequences).

**Blank vs trained comparison** reveals what training changed about the ODE dynamics. An untrained model applies only the default damping and rotation. A trained model's ODE has learned coupling constants that reshape the input — the difference IS what training taught the ODE.

## 2.3 Relate-Vocab — Full Vocabulary Relationship Scan

```bash
wave-engine encode --resume checkpoint.bin --relate-vocab \
    --data data/input.txt --output vocab_relations.json
```

Encodes every token through the ODE and computes pairwise harmonic coherence profiles. For each pair: angular differences per band, coherence at harmonics n={1,2,3,4,5,6,8,12}, shifted MRL with optimal offset, and catalog matching.

**What to look for:**
- **Structurally distinctive tokens:** low conjunction %, many non-conjunction relationships. In grammar: 's' (8% conjunction), 'q' (1%), '?' (3%).
- **Task fingerprint:** the distribution of catalog types across the vocabulary. Arithmetic = mostly conjunctions with minimal diversity. Grammar = full catalog with rich diversity.
- **Coherence scaffolding:** same-grid pairs have high MRL but cluster in conjunctions. Cross-grid pairs have lower MRL but richer catalog diversity. The interesting structure lives between grids.

**Pairwise relate (two specific items):**

```bash
wave-engine encode --resume checkpoint.bin \
    --relate "s" --relate "t" --data data/input.txt
```

## 2.4 Four-Axis Measurement Framework

Four complementary instruments for characterising what the model has learned. Each captures a different aspect of the learned representation. All operate on the same checkpoint but measure different properties.

### Axis 1: Phase (WHERE)

Geometric position of tokens relative to each other. Measured via the relate-vocab harmonic coherence profiles described in §2.3. The primary metric is MRL (Mean Resultant Length) — higher MRL = more phase-coherent pair.

### Axis 2: Energy (HOW)

Per-token ODE processing signature. Encode each token, compute per-band magnitude ratio mag_out/mag_in (the deformation vector). Tokens with similar linguistic roles produce similar deformation patterns.

**Key metric:** deform_sim — cosine similarity between two tokens' deformation vectors. Grammar mean 0.46 (differentiated processing). Arithmetic mean 0.66 (homogeneous processing).

**Phase-energy correlation:** r = 0.51. Partially related, not redundant. Some tokens are phase-distinctive but energy-generic ('s'). Others are energy-distinctive but phase-generic ('.', ':').

### Axis 3: Dignity (CONTEXT)

How context modifies per-token processing intensity. Encode a focus token alone and in various contexts. Measure cos(input, output) shift.

**Key metric:** maximum context shift. 'e' shifts 0.41 between solo and "e." contexts. 's' shifts at most 0.13. Structurally important tokens are context-stable; common tokens are context-sensitive.

### Axis 4: Direction (ORDER)

How token order affects ODE processing. Encode "ab" and "ba", compare output energy.

**Key metric:** mean |asymmetry| across all token pairs. lm_head models ~0.14, phase-native ~0.04. Three independent levers: decoder type, data augmentation, FWM.

### Inter-axis correlations

At 168-dim, 80K iters: direction and destruction share 77% variance (r = -0.88). Other correlations weak to moderate. Zero tokens in all four top-10s.

**Caveat (documented):** these correlations were measured on an undertrained model near its capacity ceiling. The correlation structure may change at larger dimensions. The relate-vocab tool computes the full correlation matrix automatically on every scan, so alignment emergence can be tracked as models mature.

## 2.5 Hidden Coherence Probe

The galaxy scan's default coherence check tests cos(n·Δθ) — which misses pairs that are coherent at a non-zero phase offset (e.g., consistently 30° apart rather than 0° or 180° apart).

The hidden coherence probe extends the search by computing MRL with shifted offsets, revealing pairs whose coherence is invisible to the standard check.

**Finding that motivated this:** Marco asked "what's the wavelength?" — which exposed that the galaxy scan was measuring at 0° only. Baked into the engine the same day it was discovered.

## 2.6 Wave Memory Inspection

```bash
wave-engine scan-memory path/to/memory.kwmf
```

The wave memory file (1.5KB persistent harmonic band state) stores accumulated experience as r_k/s_k per band per layer. The scan-memory command takes the KWMF file as a positional argument and runs the same harmonic census tools used for model analysis during training, applied to the memory state. This reveals what the model has accumulated — which bands have shifted, which remain at baseline, whether the distribution looks healthy or distorted. Optional flags (`--n-bands`, `--layers`, etc.) are used if the memory's implied model shape differs from the defaults.

---

# Appendix A: Quick Reference Card

## "I just cloned the repo. What do I do first?"

```bash
# 1. Build the engine
cargo build --release

# 2. Check the architecture calculator on your dataset
wave-engine recommend data/your_dataset.txt

# 3. Train with recommended config
wave-engine train data/your_dataset.txt \
    --n-bands 84 --n-head 4 --layers 4 \
    --alpha 0.1 --beta 0.2 \
    --lr 0.001 --iters 10000 --phase-native --split-band

# 4. Verify gradient correctness
wave-engine verify grad --split-band

# 5. Look at what the model built
wave-engine galaxy-scan --resume checkpoint_best.bin --n-bands 84 --layers 4
python scripts/summarize_galaxy.py galaxy_map.json

# 6. Explore vocabulary relationships
wave-engine encode --resume checkpoint_best.bin --relate-vocab --data data/your_dataset.txt

# 7. Generate text
wave-engine generate --resume checkpoint_best.bin --prompt "The " --phase-native
```

## "Something looks wrong. How do I diagnose?"

| Symptom | First check | Likely cause |
|---------|------------|-------------|
| Loss doesn't decrease | Stall detector should fire at iter 2000 | Wrong lr, wrong vocab size, data issue |
| NaN in gradients | Gradient monitor reports NaN skips | α or β too high for dimension, AGC ceiling exceeded |
| All layers show cos > 0.95 | Layer flow monitor | ODE is identity — check --ode-pathway is on |
| One channel dominates (θ >> Δθ) | Layer flow monitor channel ratio | Channel drift — may self-correct with learnable ODE |
| Loss V-shapes after initial descent | Loss trajectory | AGC ceiling too low, or magnitude exceeded stability threshold |
| Generation produces garbage | Generate with --temperature 0 | Normal at loss > 2.0 for char-level. Need loss < 1.0 for coherent output |

## "I want to compare two checkpoints."

```bash
# Galaxy scan both
wave-engine galaxy-scan --resume checkpoint_A.bin --n-bands 84 --layers 4
mv galaxy_map.json galaxy_A.json

wave-engine galaxy-scan --resume checkpoint_B.bin --n-bands 84 --layers 4
mv galaxy_map.json galaxy_B.json

# Diff with automatic confound warnings
python scripts/summarize_galaxy.py galaxy_A.json galaxy_B.json
```

The diff script flags: dataset mismatch, training length mismatch, architecture mismatch. If confounds are present, the comparison is documented but not conclusive.

---

# Appendix B: File Formats

| Extension | Description | Size (typical) |
|-----------|------------|----------------|
| `.bin` | WCHK checkpoint (architecture in header + flat f32 params + Adam moments) | 0.5-5 MB |
| `.kwds` | Wave-space dataset (per-position ODE states) | Varies |
| `.kwmf` | Wave memory file (per-band per-layer r_k/s_k) | 1.5 KB |
| `galaxy_map.json` | Galaxy scan output (full structural map) | ~21 MB |
| `galaxy_matrix.bin` | Pairwise coherence matrix (binary) | ~50 MB |
| `phases.bin` | Raw per-position per-band phases | ~10 MB |
| `vocab_relations.json` | Relate-vocab output | ~1 MB |

---

# Appendix C: The Bridge

A trained checkpoint is the interface between the Engine Lab and the Analysis Lab. Part 1 instruments (monitors, junction checks, stall detector, pathway verification) ensure the checkpoint was produced correctly. Part 2 instruments (galaxy scan, relate-vocab, four-axis probes) reveal what structure the checkpoint contains.

**The discipline:** never analyse a checkpoint that hasn't been verified. Never trust a structural finding from an engine whose junction monitors haven't been run. The monitors exist so the measurements are trustworthy.

This manual describes the instruments. The [ENGINE-PATTERNS](ENGINE-PATTERNS-INDEX.md) document fences the implementation bridge as a commons. The engine source code is the ground truth.

---

**Maintained by:** Marco Da Cunha (Independent Researcher)
**Engine repo:** [github.com/atech-hub/wave-engine](https://github.com/atech-hub/wave-engine)
**Framework repo:** [github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive)
