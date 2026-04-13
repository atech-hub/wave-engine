# wave-engine

A research platform for wave-coherent neural architectures. Started as a training engine, grew into a lab: one binary, one command, and a full instrument suite — galaxy scan, catalog axes, phase encode, vocabulary relationship mapping, wave memory scanning, dataset-to-wave conversion, and wave-native training — alongside three training tiers (CPU, cross-platform GPU via wgpu, NVIDIA CUDA via Candle). Replaces standard MLP layers with coupled harmonic oscillators (Kerr-ODE) governed by a differential equation.

Part of the [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) research project.

## Quick Start

```bash
# Build
cargo build --release

# Train on CPU (best quality, any hardware)
./target/release/wave-engine data/input.txt --iters 200

# Train with GPU acceleration (any GPU — NVIDIA, AMD, Intel, Apple)
./target/release/wave-engine data/input.txt --iters 200 --gpu

# Train with NVIDIA CUDA (requires --features candle-backend)
cargo build --release --features candle-backend
./target/release/wave-engine data/input.txt --iters 200 --candle

# Train with BPE tokenizer
./target/release/wave-engine data/input.txt --iters 200 --bpe --tokenizer data/tokenizer.json

# Let the engine recommend architecture for your data
./target/release/wave-engine --recommend data/your_corpus.txt

# See all options
./target/release/wave-engine --help
```

No Python. No pip. No CUDA toolkit. Build once, run anywhere. Every feature serves observation — the engine trains models, the instruments measure what the model builds while it trains.

## Validated Results

Training capability validated on four tasks. The measurement instruments are what the findings came from.

| Task | Accuracy | Config | Finding |
|------|----------|--------|---------|
| Arithmetic | **55/55 (100%)** | 168-dim, 4H, 4L, phase-native | Data presentation was the bottleneck, not architecture |
| Word classification | **46/51 (90.2%)** | 168-dim, 4H, 4L, phase-native | ODE composes characters into word meaning |
| Grammar (char) | **best 2.34** | 168-dim, 4H, 4L, PN+FWM | L3 regime shift, 4,766 locked quartets |
| Grammar (BPE) | *in progress* | 168-dim, 4H, 4L, PN+FWM, BPE-512 | Training for relate-vocab semantic test |

The architecture self-organises differently per task: arithmetic uses sharp β/α coupling splits with early-binding, words use gradual ramps with late-binding, grammar needs more bands (384-dim test next).

## Model Configurations

| Tier | Dimension | Bands | Best For | Details |
|------|-----------|-------|----------|---------|
| **Research** | 168-dim | 84 | Stress-testing, diagnostics, fast iteration. Trains in minutes on any CPU. | [168-dim config](configs/168-dim/CONFIG.md) |
| **Grammar** | 384-dim | 192 | Calculator-recommended for grammar. 8 heads, 6 layers, ~3.6M params. | [384-dim config](configs/384-dim/CONFIG.md) |
| **Production** | 768-dim | 384 | Full English, 50K BPE vocabulary. 24-layer models. | [768-dim config](configs/768-dim/CONFIG.md) |

See [configs/README.md](configs/README.md) for the complete guide.

## Usage

```
wave-engine <data> [options]

ARGUMENTS:
    DATA              Path to training data (auto-detects format)
                      Supported formats:
                        data/input.txt          Plain text
                        data/corpus.jsonl       HuggingFace JSONL (extracts 'text' field)
                        data/wikitext/          Directory (concatenates .txt + .jsonl)
                      Files > 500MB: auto-tokenized to .wtok binary, memory-mapped

TRAINING:
    --iters N         Training iterations                    [default: 500]
    --batch N         Batch size                             [default: 4]
    --seq N           Sequence length (context window)       [default: 256]
    --lr F            Learning rate                          [default: 1e-4]
    --layers N        Number of transformer blocks           [default: 24]

ARCHITECTURE (all tiers — CPU, wgpu, Candle):
    --n-bands N       Harmonic frequency bands (n_embd = N×2) [default: 384]
    --n-head N        Number of attention heads               [default: 12]
    --maestro-dim N   Maestro bottleneck width                [default: 16]
    --rk4-steps N     ODE integration steps                   [default: 16]
    --out-proj-groups N  Block-diagonal groups (1=dense)      [default: 6]
    --m1 N              Multi-grid modulus 1 (must pair with --m2, coprime)
    --m2 N              Multi-grid modulus 2 (must pair with --m1, coprime)
    --phase-native      Phase coherence loss — dot product against frozen embeddings
                        No lm_head needed. 55/55 arithmetic, 46/51 words.
    --tied-embeddings   Use harmonic wte as output decoder (null at 168-dim)
    --wave-decode       Phase coherence output decoder (85 params, validated 5.84)
    --unfreeze-phases   Learn reference phases (use with --wave-decode, 86K params)
    --lm-rank N         Low-rank lm_head factoring (0=full rank)  [default: 0]

    Common presets:
      168-dim:   --n-bands 84   --n-head 4   (fast diagnostic model)
      768-dim:   --n-bands 384  --n-head 12  (default)
      4096-dim:  --n-bands 2048 --n-head 32  --out-proj-groups 32

ODE COUPLING (linked to AGC ceiling — stronger coupling = tighter ceiling):
    --alpha F         ODE self-phase coupling                [default: 0.1]
    --beta F          ODE cross-phase coupling               [default: same as alpha]
    --agc-ceiling F   AGC max threshold (auto-derived if omitted)
    --freeze-ode      Freeze ODE params (identity backward — legacy behaviour)
                      Default: ODE α/β/γ are learnable per layer

    α and β control the Kerr nonlinearity. β independently controls the
    cross-band coupling ratio — this is a key design parameter:

      α=0.1, β=0.1: cross/self = 3.94x  (one encoding strategy at a time)
      α=0.1, β=0.2: cross/self = 7.82x  (dual encoding, 2x discrimination)
      α=0.1, β=0.3: cross/self = 11.79x (over-coupled, both channels fail)

    AGC ceiling auto-derives from coupling: ceiling = sqrt(π/2 / (α + 4β))

CORRECTOR PLATE (per-band phase correction after ODE):
    --corrector dyn     Learnable phase correction (default, spring k=0.01)
    --corrector off     Disable corrector plate (A/B testing)
    --no-corrector      Legacy alias for --corrector off

DYNAMIC PARAMETERS (--flag dyn = model decides, --flag V,V = human prescribes):
    --layer-scale dyn   Per-layer residual scaling. Spring eq=1.0, k=1.0.
    --rk4-weights dyn   Per-layer RK4 integration weights. Spring eq=[1/6,1/3,1/3,1/6], k=2.0.
    --wd dyn            Per-group weight decay scaling. Spring eq=1.0, k=1.0.
    --harmonics dyn     Per-head learnable harmonic numbers. Spring eq=initial, k=2.0.
    --agc-headroom dyn  Per-layer AGC headroom (sigma). Spring eq=3.0, k=1.0.
    --lr-scale dyn      Per-group LR scaling. Spring eq=1.0, k=0.5.
    --spring F          Global spring constant for all dynamic params [default: 0.1]
    --active-layers N   First N layers at scale=1.0, rest at scale=0.0.

RESUME:
    --resume FILE     Resume training from checkpoint
                      CPU/wgpu: WCHK .bin file (restores weights + Adam + RNG)
                      Candle:   .safetensors file (restores weights only)
    --no-curriculum   Disable progressive band curriculum (all bands from start)
    --checkpoint-name Save checkpoint to this filename       [default: checkpoint.bin]
    --log-name FILE   Custom training log filename (auto-derived from checkpoint name)

TOKENIZER:
    --bpe             Use BPE tokenizer (GPT-2 style)
    --tokenizer FILE  Path to tokenizer.json                 [default: data/tokenizer.json]
    (default)         Character-level tokenization

ACCELERATION:
    --gpu             Enable wgpu GPU (Vulkan/Metal/DX12)
    --candle          Use Candle CUDA backend (requires --features candle-backend)
    --custom-op       Use CustomOp ODE (no autograd graph, faster at 384-dim+)
    --gpu-duty N      GPU duty cycle 1-100% (Candle only)          [default: 100]
                      50 = work one batch, sleep same duration (~50% GPU usage)
                      For hot climates, laptops, shared machines

PERFORMANCE:
    --threads N       Rayon thread pool size (default: half available cores)

MONITORS:
    --monitor         Enable per-section pipeline timing (forward profiling)
    --debug-nan       Enable per-layer NaN detection (Candle only, ~6x slower)
    --health-interval N  Full diagnostic suite every N iters (0=off)    [default: 0]
                        10 monitors: attention heads, layer flow, gradient breakdown,
                        embedding space, output distribution, ODE dynamics,
                        dynamic param trajectories, curriculum transitions,
                        checkpoint drift, throughput. All in JSONL.

FOUR-WAVE MIXING:
    --fwm-strength F    Chi parameter for four-wave mixing (0=off)  [default: 0.0]
                        Hamiltonian quartets: phase-matched a+b=c+d coupling.
                        Recommended: 0.03. Full Jacobian backward on all tiers.

WAVE MEMORY:
    --memory FILE       Wave memory file (.kwmf). Created if not found.
                        Nudges ODE initial conditions based on accumulated
                        conversation experience. Model weights stay frozen.

GALAXY SCAN:
    --galaxy-scan       Generate galaxy map of learned harmonic structure.
                        Requires --resume <checkpoint>. Five-layer scan:
                        per-band profiles, pairwise geometry with catalog matching,
                        triads, FWM quartets, multi-grid decomposition.
                        Output: <checkpoint>_galaxy/ with galaxy_map.json,
                        galaxy_matrix.bin, phases.bin.
                        Auto-runs at end of training.
    --scan-corpus FILE  Data file for galaxy scan tokens (default: positional arg)

PHASE ENCODE (direct phase injection into model):
    --encode TEXT        Encode text through embedding, observe ODE evolution
    --encode-number N    Encode integer via multi-grid phases
    --encode-catalog S   Encode catalog config: "trine:35,63", "opposition:12,54"
                         Compound: "trine:35,63+opposition:12,54"
    --encode-phases S    Raw band:radian pairs: "10:1.047,20:2.094"
    --inject-layer N     Inject at layer N input (default: 0 = full model)
    --blank              Use untrained model (see what physics alone does)
    --scan               Run full galaxy scan on encode output
    --data FILE          Data file for char vocab (needed for text encode)

RELATE (harmonic coherence profiles between encodings):
    --relate A           Pairwise profile. Repeat for multiple items.
    --relate-number N    Add a number to relate comparison
    --relate-catalog S   Add a catalog config to relate comparison
    --relate-vocab       Full vocabulary pairwise relationship matrix
                         Includes energy deformation signatures (mag_out/mag_in).
    --output FILE        Output path for relate-vocab JSON

DIAGNOSTICS:
    --ode-monitor       Show raw per-band ODE data for a prompt
    --phase-decode      Compare lm_head vs phase coherence decoding
    --head-lr-floor F   Minimum effective LR for lm_head (prevents starvation)

GENERATION:
    --generate          Autoregressive text generation from checkpoint
                        Requires --resume <checkpoint>
    --prompt TEXT       Starting text for generation                [default: varies]
    --max-tokens N      Number of tokens to generate               [default: 100]
    --temperature F     Sampling temperature (0=greedy)             [default: 0.0]

SERVING:
    --serve             Start OpenAI-compatible API server (requires --features serve)
                        Requires --resume <checkpoint>. Uses KV-cache.
    --port N            Server port                                [default: 8080]
    --host ADDR         Server bind address                        [default: 127.0.0.1]

ARCHITECTURE CALCULATOR:
    --recommend FILE    Analyze dataset and recommend optimal architecture
                        Two-bottleneck model: bands + attention must both pass.
                        Prints configuration, warnings, and copy-paste CLI commands.
    --task TYPE         Override task detection (arithmetic/words/grammar/language)

ANALYSIS:
    --analyze         Run wave structure diagnostics on a trained model (no training)
                      Requires --resume <checkpoint>. Uses cos(n*dtheta) harmonic coherence.
    --sub-harmonic    Add sub-harmonic diagnostics to --analyze

DIMENSION SCALING:
    --scale FILE      Scale a trained checkpoint to larger dimensions
    --target-bands N  Target number of bands for scaling      [default: 128]
    --target-head N   Target number of attention heads        [default: 8]
    --target-layers N Target number of layers (optional — adds fresh layers)
    --output FILE     Output path for scaled checkpoint       [default: scaled_checkpoint.bin]
```

## Examples

### Your first training run

Download any plain text file (Shakespeare, Wikipedia, a novel — anything works) and save it as `data/input.txt`. Then:

```bash
./target/release/wave-engine data/input.txt --layers 4 --iters 200
```

This trains a 4-layer model on CPU for 200 iterations. You'll see loss descending from ~4.5 to ~2.5. The model saves to `checkpoint.bin` when done.

### 168-dim diagnostic model with BPE

The fastest way to experiment with the wave architecture:

```bash
# 512 BPE (good composition at 168-dim)
./target/release/wave-engine data/input.txt --layers 4 --n-bands 84 --n-head 4 \
    --out-proj-groups 1 --bpe --tokenizer data/tokenizer_512.json --iters 20000

# 1K BPE (deeper vocabulary, longer training)
./target/release/wave-engine data/input.txt --layers 4 --n-bands 84 --n-head 4 \
    --out-proj-groups 1 --bpe --tokenizer data/tokenizer_1k.json --iters 20000
```

### Phase-native training (recommended)

Phase-native mode uses dot product against frozen embeddings instead of a learned lm_head. Zero decoder parameters — the model's parameter count doesn't scale with vocabulary size. Works with any tokenization (character-level, BPE, or any future scheme). You feed it data, the ODE computes, the dot product translates. 55/55 on arithmetic, 46/51 on word classification:

```bash
./target/release/wave-engine data/arithmetic_augmented.txt --layers 4 --n-bands 84 --n-head 4 \
    --out-proj-groups 1 --alpha 0.1 --beta 0.2 --phase-native \
    --lr 3e-4 --seq 16 --no-curriculum --iters 40000
```

### Let the engine recommend architecture

```bash
# Analyze your data and get the optimal configuration
./target/release/wave-engine --recommend data/grammar_corpus.txt

# Example output:
#   Recommendation: --n-bands 192 --n-head 8 --layers 6
#   Bands: 94% utilisation (needs more) → 192 bands
#   Attention: max_weight 0.025 (too diffuse) → 8 heads
```

### Asymmetric coupling (β=0.2 — recommended)

β controls cross-band coupling strength independently of α. At β=0.2, the model sustains dual encoding (both per-band and cross-band semantic channels active):

```bash
./target/release/wave-engine data/input.txt --layers 4 --n-bands 84 --n-head 4 \
    --out-proj-groups 1 --alpha 0.1 --beta 0.2 \
    --bpe --tokenizer data/tokenizer_1k.json --iters 10000
```

### Dynamic parameters — let the model self-configure

```bash
# Let the model learn per-layer RK4 integration weights and harmonics
./target/release/wave-engine data/input.txt --layers 4 --n-bands 84 --n-head 4 \
    --out-proj-groups 1 --alpha 0.1 --beta 0.2 --phase-native \
    --rk4-weights dyn --harmonics dyn --spring 0.1 \
    --health-interval 500 --iters 10000
```

### GPU duty cycle — for hot climates and shared machines

```bash
# Run at 50% GPU utilisation (2x slower, identical results, GPU stays cool)
./target/release/wave-engine data/input.txt --candle --gpu-duty 50 --iters 1000

# Run at 25% for overnight training on a laptop
./target/release/wave-engine data/input.txt --candle --gpu-duty 25 --iters 10000
```

### Resume training

```bash
# Resume from CPU/wgpu checkpoint
./target/release/wave-engine data/input.txt --resume checkpoint.bin --iters 10000

# Resume from Candle checkpoint
./target/release/wave-engine data/input.txt --candle --resume candle_checkpoint_latest.safetensors --iters 5000
```

### Analyse a trained model

```bash
# Basic wave structure analysis
./target/release/wave-engine --analyze --resume checkpoint.bin \
    --layers 4 --n-bands 84 --n-head 4 --out-proj-groups 1

# With sub-harmonic diagnostics
./target/release/wave-engine --analyze --sub-harmonic --resume checkpoint.bin \
    --layers 4 --n-bands 84 --n-head 4 --out-proj-groups 1 \
    --alpha 0.1 --beta 0.2
```

### Scale a checkpoint to larger dimensions

```bash
# Scale 168-dim (84 bands) checkpoint to 256-dim (128 bands)
./target/release/wave-engine --scale checkpoint.bin \
    --target-bands 128 --target-head 8 --out-proj-groups 1 \
    --output model_256_from_168.bin

# Train the scaled model
./target/release/wave-engine data/input.txt --resume model_256_from_168.bin \
    --layers 4 --n-bands 128 --n-head 8 --out-proj-groups 1 \
    --alpha 0.1 --beta 0.2 --phase-native --iters 20000
```

### Phase encode — inject geometry into the model

```bash
# What does a trained model do to a trine?
./target/release/wave-engine --encode-catalog "trine:35,63" --resume checkpoint.bin \
    --layers 4 --n-bands 84 --data data/input.txt

# Compare blank (untrained physics) vs trained model
./target/release/wave-engine --encode-catalog "opposition:20,60" --blank --n-bands 84
./target/release/wave-engine --encode-catalog "opposition:20,60" --resume checkpoint.bin

# Relationship between tokens after ODE processing
./target/release/wave-engine --relate "3" --relate "+" --relate "=" \
    --resume checkpoint.bin --data data/arithmetic.txt

# Full vocabulary relationship matrix with energy signatures
./target/release/wave-engine --relate-vocab --resume checkpoint.bin \
    --data data/input.txt --output vocab_relations.json
```

### Galaxy scan — geometric inventory of learned structure

```bash
# Scan an existing checkpoint
./target/release/wave-engine --galaxy-scan --resume checkpoint.bin \
    --layers 4 --n-bands 84 --scan-corpus data/input.txt

# Summarise the scan (Python, stdlib only)
python scripts/summarize_galaxy.py checkpoint_galaxy/galaxy_map.json

# Compare two scans
python scripts/summarize_galaxy.py scan_a/galaxy_map.json scan_b/galaxy_map.json
```

### Generate text

```bash
./target/release/wave-engine --generate --resume checkpoint.bin \
    --layers 4 --n-bands 84 --n-head 4 --phase-native \
    --prompt "The " --max-tokens 200 --temperature 0.8
```

### Serve your trained model

```bash
# Built-in OpenAI-compatible server (requires --features serve)
cargo build --release --features serve
./target/release/wave-engine --serve --resume checkpoint.bin \
    --layers 4 --n-bands 84 --n-head 4 --port 8080

# Or use wave-server for production deployment
./target/release/wave-server ../wave-engine/checkpoint.bin \
    --bpe ../wave-engine/data/tokenizer.json --port 8080

# Connect any OpenAI-compatible chat UI to http://localhost:8080/v1
```

## Training Tiers

Three tiers, same model, compatible checkpoints. GPU tiers earn their keep at 256+ bands — at small dimensions CPU is fastest.

**168-dim (small model — CPU wins):**

| Tier | Flag | ms/iter | Loss @1K | Notes |
|------|------|---------|----------|-------|
| CPU | *(none)* | **17ms** | **1.49** | Gold standard, any hardware |
| Candle CustomOp | `--candle --custom-op` | 62ms | 1.88 | NVIDIA only |
| wgpu | `--gpu` | 160ms | 1.49 | Any GPU, identical to CPU |
| Candle autograd | `--candle` | 1,300ms | 1.99 | Graph overhead dominates |

**384-dim (grammar scale — GPU wins):**

| Tier | Flag | ms/iter | Speedup | Notes |
|------|------|---------|---------|-------|
| Candle CustomOp | `--candle --custom-op` | **902ms** | **1.54x** | Fastest tier |
| wgpu | `--gpu` | 1,035ms | 1.34x | Any GPU |
| CPU | *(none)* | 1,385ms | 1.0x | Baseline |
| Candle autograd | `--candle` | 2,550ms | 0.54x | Don't use — graph overhead |

*Measured April 2026 on Intel i7-14700K + RTX 4070 Ti.*

**Recommendation:** Use CPU for 168-dim diagnostics. Use `--candle --custom-op` for 384-dim+ training. The architecture calculator (`--recommend`) will suggest the right tier based on your data.

## Built-In Monitors

### Always-on (no flags needed)

| Monitor | Output | Description |
|---------|--------|-------------|
| Loss + time | Console | Per-iteration loss and wall-clock time |
| Gradient norm | Console | Displayed every 10 iterations |
| Pre-flight checks | Console | Embedding separation, param balance, ODE stability at startup |
| First-10 health | Console | Gradient norms, component balance (first 10 iters) |
| NaN recovery | Console | Detects NaN loss, skips step, logs count, continues |
| VRAM tracking | Console (Candle) | Real-time GPU memory via cudarc |
| JSONL telemetry | `training_log_*.jsonl` | Per-iteration: loss, lr, time_ms, vram_mb, ODE/AGC stats |
| Checkpoint guard | — | Refuses to save when loss is NaN/Inf/zero |
| Auto summary | End of run | Best loss, rolling averages, speed, config summary |

### Health suite (`--health-interval N`)

13 diagnostic monitors captured every N iterations into JSONL:

1. **Attention heads** — per-head entropy, max_weight, harmonic values
2. **Layer signal flow** — input/output norms, attention/FFN ratios, cosine, band amplitudes
3. **Gradient breakdown** — per-component gradient norms (ODE, maestro, out_proj, lm_head)
4. **Embedding space** — band utilisation, token separation (iter 0 only for frozen embeddings)
5. **Output distribution** — correct rank, top-k accuracy, entropy
6. **ODE dynamics** — energy ratio, phase velocity, damping, band energy std
7. **ODE forward decomposition** — damping/phase/FWM fractions per layer (what ODE did)
8. **ODE backward decomposition** — gradient flow per physics term, d_chi norm (what optimizer cares about)
9. **I/Q analysis** — I/Q channel discrimination, phase statistics
10. **Dynamic param trajectories** — α/β/γ per layer, RK4 weights, harmonics, layer scale
11. **Curriculum transitions** — band activation schedule, loss jumps
12. **Checkpoint drift** — weight distance from previous checkpoint
13. **Throughput** — tokens/sec, iters/sec, forward time, VRAM

## Architecture Calculator

Built into the binary. Analyzes your dataset and recommends the optimal architecture:

```bash
./target/release/wave-engine --recommend data/your_data.txt
```

Uses a two-bottleneck model — both bands AND attention must be satisfied:
- **Bands bottleneck:** tokens_per_dim < 0.50, band utilisation < 85%
- **Attention bottleneck:** max attention weight > 0.10, positions per head < 40

Proven: fixing attention alone without fixing bands gives zero accuracy gain (validated by 8H8L grammar test).

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Three training tiers | ✓ | CPU, wgpu (any GPU), Candle/CUDA (NVIDIA) |
| Instrument suite | ✓ | Galaxy scan, relate-vocab, phase encode, dataset-to-wave converter, wave memory scanner, catalog axes — every instrument observes the trained model's geometry |
| Four-wave mixing (FWM) | ✓ | Hamiltonian cubic coupling (`--fwm-strength`). All 3 tiers. Analytical Jacobian. |
| FWM CUDA kernel | ✓ | Fused AGC+RK4+FWM forward+backward — zero overhead vs chi=0 |
| Phase-native loss | ✓ | Dot product against embeddings — no lm_head. 55/55 arithmetic. |
| Learnable ODE | ✓ | Per-layer α/β/γ self-organise (loss 3.18, all-time best) |
| Corrector plate | ✓ | Per-band phase correction after ODE (336 params, THD drops 4x) |
| Architecture calculator | ✓ | `--recommend` analyzes data and suggests configuration |
| 7 dynamic parameters | ✓ | Self-configuring with spring regulation |
| 13 diagnostic monitors | ✓ | Full health suite via `--health-interval` (forward + backward decomposition) |
| Wave-probe | ✓ | ODE scattering analysis binary: 10 modes, physics decomposition, checkpoint loading |
| Parity test battery | ✓ | 15 test cases validating CPU/wgpu/candle produce identical physics |
| Gradient checker | ✓ | `--check-gradients` validates analytical Jacobian (172/172 with FWM) |
| CustomOp ODE backward | ✓ | Manual backward through RK4 — fastest at 384-dim (902ms) |
| GPU duty cycle | ✓ | `--gpu-duty 50` for thermal/power management |
| BPE tokenizer | ✓ | HuggingFace tokenizer.json format with disk caching |
| Checkpoint save/load | ✓ | WCHK v4 format — persists chi, optimizer state, resume on any tier |
| Asymmetric coupling | ✓ | Independent `--alpha` and `--beta` for cross/self coupling |
| Progressive dim scaling | ✓ | Scale trained checkpoints to larger dimensions (`--scale`) |
| Curriculum training | ✓ | Soft-mask band unlocking, LN-safe at 24 layers |
| FFT ODE | ✓ | OFDM-inspired stencil convolution, validated at 1.19e-7 precision |
| Text generation | ✓ | `--generate` with temperature sampling |
| API server | ✓ | `--serve` for OpenAI-compatible endpoint (requires --features serve) |
| JSONL data loading | ✓ | HuggingFace dataset format + directory support |

## Architecture

Wave-engine implements a novel neural architecture where standard MLP feed-forward layers are replaced with coupled harmonic oscillators integrated via a fourth-order Runge-Kutta solver.

**Block structure (GPT-J parallel formulation):**
```
x = x + attention(LN(x)) + FFN(LN(x))
```

**FFN: Dual-Maestro Kerr-ODE**
```
input → maestro_in (768→16→768) → [AGC] → ODE (RK4-16) → [corrector] → maestro_out (768→16→768) → out_proj
```

The maestro layers are learned bottleneck coordinators (dim=16, a universal constant validated across 128-dim to 1536-dim). The ODE evolves coupled oscillator bands through nonlinear Kerr dynamics — self-phase modulation (α), cross-phase modulation (β), nearest-neighbour coupling, and four-wave mixing (χ) for Hamiltonian energy-conserving cubic band interactions. The corrector plate applies per-band phase corrections after the ODE (336 params, zero-init, magnitude-preserving).

With learnable ODE, each layer self-organises its own coupling: L0 as per-band specialist (high α), L1-L3 as cross-band specialists (low α, high β). No load balancer needed — the model IS its own load balancer.

**Attention: Frozen Harmonic Coherence**

Standard Q/K dot-product attention is replaced with phase-based scoring: `cos(n × Δφ)` where `n` is a learned harmonic number and `Δφ` is the phase difference between positions. Attention weights are frozen during training — only the FFN and layer norms learn.

**Default configuration:**
| Parameter | Value |
|-----------|-------|
| Layers | 24 (parallel blocks) |
| Embedding dim | 768 (384 bands × 2) |
| Attention heads | 12 |
| Maestro dim | 16 |
| RK4 steps | 16 |
| Block size | 256 |

**Parameter efficiency:** The Kerr-ODE FFN uses 640K parameters per block vs 4.72M for a standard 4x-expansion MLP — 7.4x fewer parameters.

## Project Structure

```
src/
├── main.rs                  CLI dispatch, mode routing
├── common/                  Shared library (all tiers import from here)
│   ├── model.rs             Weight structs, layer_norm, gelu, linear
│   ├── wave_model.rs        Model init, flatten/unflatten params
│   ├── attn.rs              Harmonic coherence attention (frozen)
│   ├── block.rs             Parallel block (GPT-J formulation)
│   ├── ffn.rs               FFN routing via ComputeBackend
│   ├── embed.rs             Frozen harmonic + positional embeddings
│   ├── fft_ode.rs           OFDM-inspired FFT ODE derivative
│   ├── ode_backward.rs      Manual ODE backward (shared by all tiers)
│   ├── ode_deriv.rs         ODE derivative computation
│   ├── agc.rs               Automatic gain control (coupling-derived ceiling)
│   ├── math.rs              Shared math (softplus, etc — deduplicated)
│   ├── recommend.rs         Architecture calculator (--recommend)
│   ├── generate.rs          Text generation (--generate)
│   ├── analyze.rs           Wave structure diagnostics (--analyze)
│   ├── scale.rs             Progressive dimension scaling (--scale)
│   ├── checkpoint.rs        WCHK v4 checkpoint save/load (persists chi)
│   ├── ode_parity.rs        Parity test battery (15 cases, all tiers)
│   ├── data_loader.rs       Text, JSONL, directory loading
│   ├── dims.rs              Dimension constants + Dims struct
│   ├── help.rs              CLI help text
│   └── *_monitor.rs         13 diagnostic monitors
├── cpu/
│   ├── forward.rs           Forward pass with cache
│   ├── model_backward.rs    Backward pass, gradient computation
│   ├── backward.rs          Loss backward
│   ├── train.rs             Training config, Adam optimizer
│   ├── train_loop.rs        CPU/wgpu training loop
│   ├── train_health.rs      Health monitoring + spring regulation
│   └── curriculum.rs        Band curriculum schedule
├── wgpu_tier/               Cross-platform GPU backend (35 WGSL shaders)
│   ├── pipelines.rs         Pipeline + shader compilation
│   ├── dispatch.rs          ComputeBackend trait implementation
│   ├── ops_forward.rs       Forward ops (fused RK4)
│   ├── ops_backward.rs      Backward ops (analytical gradients)
│   ├── diagnostics.rs       GPU diagnostic + validation
│   └── ...                  buffers, resident, ffn_gpu, ffn_full_gpu
├── bin/
│   └── wave_probe.rs        ODE scattering analysis (10 modes, physics decomposition)
├── lib.rs                   Library target (shared module tree for binaries)
├── candle_tier/             NVIDIA CUDA backend
│   ├── engine.rs            Module re-exports
│   ├── candle_model.rs      Model struct + constructor
│   ├── candle_forward.rs    Forward pass (autograd-compatible)
│   ├── candle_train.rs      Training loop
│   ├── candle_attention.rs  Attention (CPU scoring)
│   ├── candle_checkpoint.rs WCHK ↔ safetensors conversion
│   ├── candle_monitors.rs   Monitor data extraction
│   ├── custom_ode.rs        CustomOp1 — manual backward, no graph
│   ├── cuda_ode.rs          Fused CUDA kernel — AGC+RK4+FWM forward+backward
│   ├── ode.rs               GPU-native RK4 ODE (autograd path)
│   └── block_diag.rs        Block-diagonal linear
├── serve_tier/              OpenAI-compatible API server
│   ├── server.rs            HTTP server + routing
│   └── prompt.rs            Vocab encoding/decoding
└── shaders/                 35 WGSL compute shaders
```

## Key Findings

Validated through testing and documented honestly (12 corrections + 1 null):

- **ODE backward is the root cause fix.** Frozen ODE backward was the single cause of channel drift, sustained training degradation, and layer integration issues. Implementing proper gradient flow produced loss 3.18 (all-time best) in 30K sustained iters — beating 70K iters of cycling at 3.91.
- **β is an independent design parameter.** β=0.2 with α=0.1 produces 7.82x cross/self coupling, dual encoding, and rotational learning. β=0.3 over-couples. α=β crystallises.
- **Phase-native decoding works.** Dot product against frozen embeddings with zero decoder params. 55/55 arithmetic, 46/51 word classification.
- **Two-bottleneck model.** Architecture needs BOTH sufficient bands AND sufficient attention heads. Fixing one without the other gives zero improvement (proven by 8H8L grammar test).
- **The model self-organises per task.** Arithmetic: sharp β/α split, early-binding. Words: gradual ramp, late-binding. Grammar: weak coupling, capacity-limited.
- **Dead bands are coupling relays.** Can't remove them — 70 bands → 42/55 accuracy.
- **Four-wave mixing works.** Hamiltonian energy-conserving cubic coupling between band quartets. FWM is 8-10% of the ODE derivative at chi=0.03 and grows during training. Top FWM flux bands migrate as the model learns — goal-directed band mixing, not passive coupling.
- **FWM accelerates alpha-collapse.** Both FWM and non-FWM models converge to the same structural pattern (deep layers suppress alpha, amplify beta), but FWM models differentiate layers more aggressively.
- **Corrector plate.** 336 params (0.1% of model) of per-band phase rotation after ODE reduces THD 4x. Inspired by Schmidt corrector optics.
- **Decoder determines learned geometry.** Phase-native preserves 5,866 FWM quartets and creates 1,404 novel ones from the embedding baseline. lm_head preserves zero and creates zero. The decoder choice — not the physics, not the data — is the dominant lever on what geometric structure survives training. Phase-native builds primary catalog relationships (squares, trines, oppositions); lm_head builds secondary aspects.
- **Training is subtractive against the embedding.** The multi-grid harmonic embedding provides structural FWM quartet coherence at initialisation. Training selectively removes this structure, with the decoder controlling what survives. This is inverted from standard ML where models learn representations from scratch.
- **7 architectural invariants** confirmed across all configurations.

## Galaxy Scan

Every training run automatically generates a **galaxy map** — a pure-band geometric inventory of the model's learned harmonic structure. The scan maps all 3,486 band pairs across 12 harmonics, detects triadic constellations and FWM quartets, classifies relationships against a [catalog of 11 geometric types](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive/blob/main/docs/geometric-relationship-catalog.md) drawn from the framework's mathematical foundations, and places every band in 3D coordinates within the AGC-bounded sphere.

**What it measures:**

The framework's core operator — `cos(n * (θ_a - θ_b))` — detects different relationship types at each harmonic `n`. The galaxy scan applies this systematically to every band pair at every layer, producing a complete map of the learned phase geometry. Relationships that the framework predicts (triads at n=3, squares at n=4, oppositions at n=2) appear organically in trained models without being explicitly encoded.

**What it found:**

Phase-native models build rich geometric structure at the output layer: 984 triads, 42,218 coherent FWM quartets, and catalog-matched relationships including squares (90°), trines (120°), and oppositions (180°). lm_head models on the same data build 9x fewer triads and a qualitatively different geometric vocabulary. The decoder choice shapes the entire model's learned geometry through backward gradient flow — a finding that connects directly to the framework's Proposition 3.5 (Cosine Similarity Blindness), which proves mathematically that aggregating harmonic channels into a single scalar destroys detectable relationships.

**Output files** (in `<checkpoint>_galaxy/`):

| File | Size | Contents |
|------|------|----------|
| `galaxy_map.json` | ~20 MB | Full scan: per-band profiles, top pairs, triads, FWM quartets, catalog matches, 3D coordinates |
| `galaxy_matrix.bin` | ~650 KB | Complete pair coherence matrix at all 12 harmonics (GALX format) |
| `phases.bin` | ~260 KB | Raw per-band per-position phases for retrospective analysis (PHAS format) |

**Usage:**

```bash
# Auto-generates at end of every training run (no extra flags needed)
wave-engine data/input.txt --iters 10000

# Scan an existing checkpoint retrospectively
wave-engine --galaxy-scan --resume checkpoint.bin --n-bands 84 --layers 4

# Summarize a scan (Python, stdlib only)
python scripts/summarize_galaxy.py checkpoint_galaxy/

# Compare two scans
python scripts/summarize_galaxy.py --compare model_a_galaxy/ model_b_galaxy/
```

The galaxy scan is the first instrument that makes the [Wave Coherence framework's](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) harmonic relationships visible inside a trained neural network. The geometric relationship catalog used by the scan draws on angular mathematics formalised across multiple civilisations over thousands of years — the same circle divisions and relationship types independently discovered by traditions spanning continents and millennia, now appearing organically in the learned state of a wave-based neural architecture.

## Requirements

- **Rust** (edition 2024)
- **Any GPU** for `--gpu` mode (Vulkan, Metal, or DX12 support)
- **NVIDIA GPU + CUDA** for `--candle` mode (optional feature)

No Python, no pip, no CUDA toolkit for the default build.

```bash
# Default build (CPU + wgpu)
cargo build --release

# With Candle/CUDA support
cargo build --release --features candle-backend

# With built-in API server
cargo build --release --features serve

# All features
cargo build --release --features "candle-backend serve"
```

## Contributing

The maintainer ([Marco Da Cunha](https://github.com/atech-hub)) is an IT systems administrator, not a software engineer. This engine was built through collaboration with AI — Claude Desktop for architecture and analysis, Claude Code for implementation and testing. This is stated openly and honestly.

**Main branch is protected.** Fork, branch, submit a PR with test results showing training still converges.

**Current targets for contributors:**

| Target | Impact | Difficulty |
|--------|--------|------------|
| Grammar at 384-dim | Calculator recommends 192 bands, 8 heads, 6 layers | Active |
| Wikitext-103 pipeline | Validate architecture on real English | Medium |
| fp16 for linear ops | Halve memory and PCIe transfer size | Medium |
| AMD/Intel GPU testing | Validate wgpu tier on non-NVIDIA | Small |
| Wasm/web tier | Run small models in the browser | Medium |

## Important: Model Compatibility

Models trained by wave-engine use a novel architecture (Kerr-ODE, harmonic coherence attention, phase-native loss) that is **not compatible** with standard inference tools. Trained checkpoints will **only** work with wave-engine's built-in `--generate` and `--serve` modes, or with [wave-server](https://github.com/atech-hub/wave-server). They cannot be loaded by LM Studio, Ollama, llama.cpp, vLLM, or Hugging Face Transformers.

## Related

- [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) — The parent research project
- [wave-server](https://github.com/atech-hub/wave-server) — OpenAI-compatible inference server with KV-cache and wave memory
- [kerr-memory](https://github.com/atech-hub/kerr-memory) — Persistent wave memory state
- [kerr-engine](https://github.com/atech-hub/kerr-engine) — First implementation, historical reference

## References and Acknowledgments

- **Secondini et al. (2015)** — "[Fiber Nonlinearity Mitigation in WDM Systems: Enhanced Split-Step Fourier Method](https://arxiv.org/abs/1507.00921)." The ESSFM single-step approach inspired the perturbative Kerr-ODE.
- **Lin et al. (2022)** — "[Perturbation-Aided Sample-Based Learned Digital Back-Propagation](https://arxiv.org/abs/2110.05563)." Informed the α/β phase correction structure.
- **Pal et al. (2024)** — "[Coupled Lugiato-Lefèvre Equation for Nonlinear Frequency Comb Generation](https://arxiv.org/abs/2404.05646v2)." Physical basis for Kerr coupling terms.
- **Ng (2026)** — "[RYS-XLarge: Repeated Blocks for Parameter Efficiency](https://huggingface.co/blog/rys-xlarge)." Inspiration for repeated-blocks experiment.
- **Listopad (2025)** — "[ResonanceDB: Phase-Aware Vector Database](https://arxiv.org/abs/2509.09691)." Independent validation of phase-aware vector similarity.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Credits

- **Marco Da Cunha** — Architecture, direction, pattern recognition
- **Claude Desktop (Opus)** — Architecture design, analysis, documentation
- **Claude Code** — Implementation, testing, GPU infrastructure
