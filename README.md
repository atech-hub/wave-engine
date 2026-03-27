# wave-engine

Training engine for wave-coherent neural architectures. Replaces standard MLP layers with coupled harmonic oscillators (Kerr-ODE) governed by a differential equation. Three training tiers — CPU, cross-platform GPU (wgpu), and NVIDIA GPU (Candle/CUDA) — all producing the same model from one binary, one command.

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

# See all options
./target/release/wave-engine --help
```

No Python. No pip. No CUDA toolkit. Build once, run anywhere.

## Model Configurations

Proven configurations for each dimension tier, with recommended settings, training results, and ideal use cases. Start with the tier that matches your hardware and goals.

| Tier | Dimension | Params | Speed | Best For | Details |
|------|-----------|--------|-------|----------|--------|
| **Research / Specialist** | 168-dim | 186K–340K | 57-80ms | Small-vocab structured tasks — music, code, chemistry, DNA, arithmetic. Trains in minutes on any CPU. | [168-dim config](configs/168-dim/CONFIG.md) |
| **Mid-Range** | 256-dim | 258-579K | 140-210ms | Domain-specific text, structured English, code generation. Bridge between research and production. | [256-dim config](configs/256-dim/CONFIG.md) |
| **Power User** | 384-dim | ~500K | ~200ms | Coherent English with word-level BPE. Enough capacity for sentence-level generation. | [384-dim config](configs/384-dim/CONFIG.md) |
| **Production** | 768-dim | ~4M | 1.2s CPU / 4.3s GPU | Full English, 50K BPE vocabulary. 24-layer models trained on Candle/CUDA. | [768-dim config](configs/768-dim/CONFIG.md) |

See [configs/README.md](configs/README.md) for the complete guide.

## Usage

```
wave-engine <data> [options]

ARGUMENTS:
    DATA              Path to training data file (e.g. data/input.txt)

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
    --rk4-steps N     ODE integration steps (CPU/wgpu only)   [default: 16]
    --out-proj-groups N  Block-diagonal groups (1=dense)      [default: 6]
    --m1 N              Multi-grid modulus 1 (must pair with --m2, coprime)
    --m2 N              Multi-grid modulus 2 (must pair with --m1, coprime)
    --tied-embeddings   Use harmonic wte as output decoder (experimental)

    Common presets:
      168-dim:   --n-bands 84   --n-head 4   (fast diagnostic model)
      768-dim:   --n-bands 384  --n-head 12  (default)
      4096-dim:  --n-bands 2048 --n-head 32  --out-proj-groups 32

ODE COUPLING (linked to AGC ceiling — stronger coupling = tighter ceiling):
    --alpha F         ODE self-phase coupling                [default: 0.1]
    --beta F          ODE cross-phase coupling               [default: same as alpha]
    --agc-ceiling F   AGC max threshold (auto-derived if omitted)

    α and β control the Kerr nonlinearity. β independently controls the
    cross-band coupling ratio — this is a key design parameter:

      α=0.1, β=0.1: cross/self = 3.94x  (one encoding strategy at a time)
      α=0.1, β=0.2: cross/self = 7.82x  (dual encoding, 2x discrimination)
      α=0.1, β=0.3: cross/self = 11.79x (over-coupled, both channels fail)

    AGC ceiling auto-derives from coupling: ceiling = sqrt(π/2 / (α + 4β))
      β=0.1: ceiling = 1.77
      β=0.2: ceiling = 1.32 (use --agc-ceiling 1.0 for stability)
      β=0.3: ceiling = 1.10 (use --agc-ceiling 0.85)

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

PERFORMANCE:
    --threads N       Rayon thread pool size (default: half available cores)

MONITORS:
    --monitor         Enable per-section pipeline timing (forward profiling)
    --debug-nan       Enable per-layer NaN detection (Candle only, ~6x slower)

ANALYSIS:
    --analyze         Run wave structure diagnostics on a trained model (no training)
                      Requires --resume <checkpoint>. Uses cos(n*dtheta) harmonic coherence.
                      Reports: semantic discrimination, depth curve, band census,
                      phase clustering, harmonic spectra.
    --sub-harmonic    Add sub-harmonic diagnostics to --analyze:
                      per-band/cross-band discrimination, coupling energy budget,
                      inter-modulation spectrum, magnitude correlation.

DIMENSION SCALING:
    --scale FILE      Scale a trained checkpoint to larger dimensions
    --target-bands N  Target number of bands for scaling      [default: 128]
    --target-head N   Target number of attention heads        [default: 8]
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
    --out-proj-groups 1 --bpe --tokenizer data/tokenizer_1k_gs.json --iters 20000
```

### Asymmetric coupling (β=0.2 — recommended)

β controls cross-band coupling strength independently of α. At β=0.2, the model sustains dual encoding (both per-band and cross-band semantic channels active) and learns 4x faster than α=β=0.1:

```bash
# β=0.2 with 1K BPE — strongest discrimination measured (3.21x at C2)
./target/release/wave-engine data/input.txt --layers 4 --n-bands 84 --n-head 4 \
    --out-proj-groups 1 --alpha 0.1 --beta 0.2 --agc-ceiling 1.0 \
    --bpe --tokenizer data/tokenizer_1k_gs.json --iters 10000 \
    --checkpoint-name model_beta02.bin
```

### Resume with custom log name

```bash
# Resume training — log auto-derives from checkpoint name
./target/release/wave-engine data/input.txt --resume model_beta02.bin --iters 10000

# Or specify a custom log
./target/release/wave-engine data/input.txt --resume model_beta02.bin --iters 10000 \
    --log-name training_log_beta02_c2.jsonl
```

### Analyse a trained model

```bash
# Basic wave structure analysis
./target/release/wave-engine --analyze --resume checkpoint.bin \
    --layers 4 --n-bands 84 --n-head 4 --out-proj-groups 1

# With sub-harmonic diagnostics (cross-band coupling, encoding strategies)
./target/release/wave-engine --analyze --sub-harmonic --resume model_beta02.bin \
    --layers 4 --n-bands 84 --n-head 4 --out-proj-groups 1 \
    --alpha 0.1 --beta 0.2 --agc-ceiling 1.0 \
    --bpe --tokenizer data/tokenizer_1k_gs.json
```

Sub-harmonic diagnostics report: per-band (θ) and cross-band (Δθ) discrimination, coupling energy budget, inter-modulation spectrum, and magnitude correlation. These reveal the two encoding strategies the model uses and how they interact.

### Scale a checkpoint to larger dimensions

```bash
# Scale 168-dim (84 bands) checkpoint to 256-dim (128 bands)
./target/release/wave-engine --scale model_beta02.bin \
    --target-bands 128 --target-head 8 --out-proj-groups 1 \
    --output model_256_from_168.bin

# Train the scaled model at 256-dim
./target/release/wave-engine data/input.txt --resume model_256_from_168.bin \
    --layers 4 --n-bands 128 --n-head 8 --out-proj-groups 1 \
    --alpha 0.1 --beta 0.2 --agc-ceiling 1.0 \
    --bpe --tokenizer data/tokenizer_1k_gs.json --iters 20000
```

Scaling preserves the learned weights for existing bands (1–84) and initialises new bands (85–128) with fresh weights. The model inherits the semantic structure from the smaller checkpoint.

### Custom multi-grid moduli

The embedding system uses two coprime moduli for token separation. Normally auto-detected, but can be overridden:

```bash
./target/release/wave-engine data/input.txt --layers 4 --n-bands 84 --n-head 4 \
    --m1 33 --m2 35 --bpe --tokenizer data/tokenizer_1k_gs.json --iters 10000
```

Both `--m1` and `--m2` must be provided together and must be coprime (GCD=1).

### Scale up with GPU

```bash
# wgpu — any GPU (NVIDIA, AMD, Intel, Apple Silicon)
./target/release/wave-engine data/input.txt --layers 4 --iters 200 --gpu

# Candle CUDA — NVIDIA only (fastest)
cargo build --release --features candle-backend
./target/release/wave-engine data/input.txt --candle --layers 4 --iters 200
```

### Production training (24 layers, diverse corpus)

```bash
cat grammar.txt literature.txt > data/training.txt
./target/release/wave-engine data/training.txt --candle \
    --layers 24 --iters 5000 --seq 256 --batch 4 --lr 1e-4 \
    --bpe --tokenizer data/tokenizer.json
```

### Serve your trained model

After training, serve with [wave-server](https://github.com/atech-hub/wave-server):

```bash
./target/release/wave-server ../wave-engine/checkpoint.bin \
    --bpe ../wave-engine/data/tokenizer.json --port 8080

# Connect any OpenAI-compatible chat UI to http://localhost:8080/v1
```

## Training Tiers

The engine provides three tiers to match your hardware. All tiers train the same model architecture and produce compatible checkpoints.

| Tier | Flag | Speed (4L) | Loss @ 200 | Params | Hardware |
|------|------|-----------|-----------|--------|----------|
| CPU | *(none)* | 520ms/iter | **2.52** | 2.63M | Any computer |
| wgpu | `--gpu` | 520ms/iter | **2.52** (identical to CPU) | 2.63M | Any GPU (Vulkan/Metal/DX12) |
| Candle CUDA | `--candle` | 213ms/iter | **2.81** (block-diagonal, 4x fewer FFN params) | 657K | NVIDIA only |

*Measured March 22 2026: 4 layers, seq=64, batch=4, 200 iters, Shakespeare, no curriculum, RTX 4070 Ti.*

**CPU** gives the best training quality on any hardware — a Raspberry Pi, a cloud VM, a 10-year-old laptop. **wgpu** runs on any GPU without CUDA, producing identical results to CPU. **Candle CUDA** is 2.4x faster with block-diagonal output projection and perturbative ODE.

## Built-In Monitors

### Always-on (no flags needed)

| Monitor | Output | Description |
|---------|--------|-------------|
| Loss + time | Console | Per-iteration loss and wall-clock time |
| Gradient norm | Console | `gnorm=` after each iteration |
| Pre-flight checks | Console | Embedding separation, param balance, ODE stability at startup |
| First-10 health | Console | Gradient norms, component balance (first 10 iters) |
| NaN recovery | Console | Detects NaN loss, skips step, logs count, continues |
| VRAM tracking | Console (Candle) | Real-time GPU memory via cudarc |
| JSONL telemetry | `training_log_*.jsonl` | Per-iteration: loss, lr, time_ms, vram_mb, ODE/AGC stats |
| Checkpoint guard | — | Refuses to save when loss is NaN/Inf/zero |
| Auto training summary | End of run | Best loss, rolling averages, speed, config summary + JSONL line |

### Optional

| Flag | What it does |
|------|-------------|
| `--monitor` | Per-section FFN timing: mae_in, ODE, mae_out, out_proj breakdown per block |
| `--debug-nan` | Per-layer NaN detection in Candle tier (~6x slower) |

### Training logs

Log filenames auto-derive from checkpoint name to prevent overwrites:
- `checkpoint.bin` → `training_log.jsonl`
- `model_beta02.bin` → `training_log_beta02.jsonl`
- Or override with `--log-name custom_log.jsonl`

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Three training tiers | ✓ | CPU, wgpu (any GPU), Candle/CUDA (NVIDIA) |
| BPE tokenizer | ✓ | HuggingFace tokenizer.json format |
| Token cache | ✓ | BPE encoding cached to disk — instant reload |
| Checkpoint save/load | ✓ | WCHK format with optimizer state, iteration count, resume support |
| Asymmetric coupling | ✓ | Independent `--alpha` and `--beta` for cross/self coupling control |
| Sub-harmonic diagnostics | ✓ | Per-band (θ) and cross-band (Δθ) discrimination, coupling budget |
| Progressive dim scaling | ✓ | Scale trained checkpoints to larger dimensions (`--scale`) |
| Configurable multi-grid | ✓ | Custom coprime moduli (`--m1`, `--m2`) for embedding separation |
| Custom log names | ✓ | Auto-derived or manual (`--log-name`) — no more overwrites |
| Curriculum training | ✓ | Soft-mask band unlocking (+1.46pp validated), LN-safe at 24 layers |
| GPU fused ODE | ✓ | One submit, 192 dispatches, zero CPU readbacks between RK4 steps |
| FFT ODE | ✓ | OFDM-inspired stencil convolution, validated at 1.19e-7 precision |
| Ping-pong buffers | ✓ | Forward/backward read same GPU bits, eliminates precision mismatch |
| Pipeline monitor | ✓ | Per-section FFN timing (`--monitor`) |

## Architecture

Wave-engine implements a novel neural architecture where standard MLP feed-forward layers are replaced with coupled harmonic oscillators integrated via a fourth-order Runge-Kutta solver.

**Block structure (GPT-J parallel formulation):**
```
x = x + attention(LN(x)) + FFN(LN(x))
```

**FFN: Dual-Maestro Kerr-ODE**
```
input → maestro_in (768→16→768) → ODE (RK4, 16 steps) → maestro_out (768→16→768) → out_proj (768→768)
```

The maestro layers are learned bottleneck coordinators (dim=16, a universal constant validated across 128-dim to 1536-dim). The ODE evolves coupled oscillator bands through nonlinear Kerr dynamics — self-phase modulation (α), cross-phase modulation (β), and nearest-neighbour coupling.

**ODE coupling: α and β**

The `--alpha` and `--beta` flags control the Kerr nonlinearity independently. α governs self-phase modulation (each band's own amplitude affects its phase). β governs cross-phase modulation (neighbouring bands' amplitudes affect phase). The cross/self coupling ratio is determined by the architecture and α/β values, not learned during training.

At α=0.1, β=0.2, the cross-modulation is ~7.82x stronger than self-modulation. This enables the model to use two simultaneous encoding strategies (per-band phase and cross-band phase differences) where α=β only allows one at a time. See the [research repo](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) for the full investigation.

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

## GPU Acceleration

### wgpu (cross-platform)

The `--gpu` flag enables GPU acceleration via wgpu, which works on any GPU supporting Vulkan, Metal, or DX12 — NVIDIA, AMD, Intel, Apple Silicon.

What runs on GPU: frozen attention output projection, trained FFN output projection via ping-pong buffers, fused ODE integration. What stays on CPU: maestro layers (dim=16, too small for GPU dispatch), attention scoring (frozen), layer normalization.

### Candle/CUDA (NVIDIA)

```bash
cargo build --release --features candle-backend
./target/release/wave-engine data/input.txt --candle --layers 4 --iters 200
```

Uses cuBLAS for all matrix operations with autograd. The Kerr-ODE runs as a `CustomOp1` with identity backward.

### A note on GPU utilisation

GPU utilisation oscillates between 20-63% instead of 100%. **This is normal** — the wave-engine replaces dense MLP layers (4.72M params, 589K multiply-adds per block) with a Kerr-ODE (770 parameters, ~6K operations per block). The GPU has less work to do because the architecture is more efficient.

## Key Findings

These are validated through testing and documented honestly:

- **β is an independent design parameter.** β=0.2 with α=0.1 produces 1.9x stronger semantic discrimination in 1/5 of the training time compared to α=β=0.1. The coupling ratio (3.94x→7.82x) enables dual encoding where both per-band and cross-band channels carry semantics simultaneously.
- **Maestro dim=16 is a universal constant.** Tested at dim 16/96/128/160 on 768-dim — all within 0.028 loss.
- **ODE is 28% of FFN time, not 80%.** The output projection (768×768 matmul) dominates at 66%.
- **Frozen attention loses nothing on tested datasets.** Harmonic coherence scoring produces equivalent quality without training attention weights.
- **7.4x parameter efficiency holds across scale.** 640K FFN params per block vs 4.72M standard MLP.

## Project Structure

```
src/
├── main.rs                  CLI dispatch, mode routing
├── common/                  Shared modules (all tiers)
│   ├── model.rs             Weight structs, layer_norm, gelu, linear
│   ├── attn.rs              Harmonic coherence attention (frozen)
│   ├── block.rs             Parallel block (GPT-J formulation)
│   ├── ffn.rs               FFN routing via ComputeBackend
│   ├── embed.rs             Frozen harmonic + positional embeddings
│   ├── wave_model.rs        Model init, flatten/unflatten params
│   ├── dims.rs              Dimension constants + Dims struct
│   ├── analyze.rs           --analyze mode (wave structure diagnostics)
│   ├── sub_harmonic.rs      Sub-harmonic diagnostics (θ/Δθ encoding strategies)
│   ├── scale.rs             Progressive dimension scaling (--scale)
│   ├── checkpoint.rs        WCHK v2 checkpoint save/load
│   ├── fft_ode.rs           OFDM-inspired FFT ODE derivative
│   ├── help.rs              CLI help text
│   └── ...                  bpe, token_cache, data, rng, monitor
├── cpu/
│   ├── forward.rs           Forward pass with cache
│   ├── model_backward.rs    Backward pass, gradient computation
│   ├── backward.rs          Loss backward
│   └── train.rs             CPU/wgpu training loop
├── wgpu_tier/               Cross-platform GPU backend (35 WGSL shaders)
│   ├── pipelines.rs         Pipeline + shader compilation
│   ├── dispatch.rs          ComputeBackend trait implementation
│   ├── ops_forward.rs       Forward ops (fused RK4, perturbative)
│   ├── ops_backward.rs      Backward ops (analytical gradients)
│   ├── diagnostics.rs       GPU diagnostic functions
│   └── ...                  buffers, resident, ffn_gpu, ffn_full_gpu
├── candle_tier/             NVIDIA CUDA backend
│   ├── engine.rs            Candle training loop with autograd
│   ├── ode.rs               GPU-native perturbative ODE
│   └── block_diag.rs        Block-diagonal linear via Candle
└── shaders/                 35 WGSL compute shaders
```

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
```

## Contributing

The maintainer ([Marco Da Cunha](https://github.com/atech-hub)) is an IT systems administrator, not a software engineer. This engine was built through collaboration with AI — Claude Desktop for architecture and analysis, Claude Code for implementation and testing. This is stated openly and honestly.

**Main branch is protected.** Fork, branch, submit a PR with test results showing training still converges.

**Known targets for contributors:**

| Target | Impact | Difficulty |
|--------|--------|------------|
| Wikitext-103 training pipeline | Validate architecture on real English | Medium |
| Low-rank lm_head | Reduce output projection cost at small dims | Medium |
| Stochastic resonance integration | Validated -8.8% perplexity improvement (α=0.05) | Small |
| Batched inference dispatch | Use engine's existing batch patterns in wave-server | Medium |
| fp16 for linear ops | Halve memory and PCIe transfer size | Medium |
| Speculative decoding | Lower per-token latency for serving | Large |

## Important: Model Compatibility

Models trained by wave-engine use a novel architecture (Kerr-ODE, harmonic coherence attention, block-diagonal projections) that is **not compatible** with standard inference tools. Trained checkpoints will **only** work with [wave-server](https://github.com/atech-hub/wave-server). They cannot be loaded by LM Studio, Ollama, llama.cpp, vLLM, or Hugging Face Transformers.

Train with wave-engine → serve with wave-server → connect any OpenAI-compatible chat UI.

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
